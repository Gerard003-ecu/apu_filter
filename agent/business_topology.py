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
        - Aristas: APU → Insumo (relación "contiene").

        Returns:
            nx.DiGraph: Grafo de topología de negocio
        """
        G = nx.DiGraph()

        logger.info("Iniciando construcción del Budget Graph...")

        # 1. Agregar nodos APU desde presupuesto (fuente primaria)
        if presupuesto_df is not None and not presupuesto_df.empty:
            apu_codes_added = set()
            for idx, row in presupuesto_df.iterrows():
                apu_code = str(row.get(ColumnNames.CODIGO_APU, "")).strip()
                if not apu_code or apu_code in apu_codes_added:
                    continue

                G.add_node(
                    apu_code,
                    type="APU",
                    description=row.get(ColumnNames.DESCRIPCION_APU, ""),
                    quantity=row.get(ColumnNames.CANTIDAD_PRESUPUESTO, 0.0),
                    source="presupuesto_df",
                    original_index=idx
                )
                apu_codes_added.add(apu_code)

            logger.debug(f"Agregados {len(apu_codes_added)} nodos APU desde presupuesto_df")

        # 2. Procesar detalle de APUs para insumos y relaciones
        if apus_detail_df is not None and not apus_detail_df.empty:
            # Contadores para métricas
            inferred_apus = 0
            insumos_added = set()
            edges_added = 0

            for idx, row in apus_detail_df.iterrows():
                apu_code = str(row.get(ColumnNames.CODIGO_APU, "")).strip()
                insumo_desc = str(row.get(ColumnNames.DESCRIPCION_INSUMO, "")).strip()

                if not apu_code or not insumo_desc:
                    continue

                # Agregar APU si no existe (inferido)
                if apu_code not in G:
                    G.add_node(
                        apu_code,
                        type="APU",
                        inferred=True,
                        source="apus_detail_df",
                        original_index=idx
                    )
                    inferred_apus += 1

                # ID único para insumo (usando descripción como clave)
                insumo_id = insumo_desc

                # Agregar nodo insumo si no existe
                if insumo_id not in G:
                    G.add_node(
                        insumo_id,
                        type="INSUMO",
                        description=insumo_desc,
                        tipo_insumo=row.get(ColumnNames.TIPO_INSUMO, ""),
                        unit_cost=row.get(ColumnNames.COSTO_INSUMO_EN_APU, 0.0),
                        source="apus_detail_df",
                        original_index=idx
                    )
                    insumos_added.add(insumo_id)

                # Calcular costo total para la arista
                unit_cost = float(row.get(ColumnNames.COSTO_INSUMO_EN_APU, 0.0))
                quantity = float(row.get(ColumnNames.CANTIDAD_APU, 0.0))
                total_cost = unit_cost * quantity if unit_cost and quantity else 0.0

                # Agregar arista APU → Insumo con atributos
                if not G.has_edge(apu_code, insumo_id):
                    G.add_edge(
                        apu_code,
                        insumo_id,
                        unit_cost=unit_cost,
                        quantity=quantity,
                        total_cost=total_cost,
                        original_index=idx
                    )
                    edges_added += 1
                else:
                    # Si ya existe la arista, sumar cantidades y costos
                    edge_data = G[apu_code][insumo_id]
                    existing_qty = edge_data.get('quantity', 0.0)
                    existing_total = edge_data.get('total_cost', 0.0)
                    G[apu_code][insumo_id]['quantity'] = existing_qty + quantity
                    G[apu_code][insumo_id]['total_cost'] = existing_total + total_cost

            logger.debug(
                f"Procesado apus_detail_df: {inferred_apus} APUs inferidos, "
                f"{len(insumos_added)} insumos únicos, {edges_added} aristas agregadas"
            )

        # 3. Validación básica del grafo
        apu_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'APU']
        insumo_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'INSUMO']

        logger.info(
            f"Budget Graph construido: {G.number_of_nodes()} nodos "
            f"({len(apu_nodes)} APUs, {len(insumo_nodes)} insumos), "
            f"{G.number_of_edges()} aristas"
        )

        return G


class BusinessTopologicalAnalyzer:
    """Analizador de la topología de negocio."""

    def __init__(self, telemetry: Optional[TelemetryContext] = None):
        self.telemetry = telemetry

    def calculate_betti_numbers(self, graph: nx.DiGraph) -> Tuple[int, int]:
        """
        Calcula los números de Betti (β₀, β₁) para el grafo.

        β₀ = Número de componentes conexas (componentes débilmente conexas)
        β₁ = Rank del primer grupo de homología = |E| - |V| + β₀

        Para grafos dirigidos, trabajamos con el grafo subyacente no dirigido
        para el cálculo topológico.

        Returns:
            Tuple[int, int]: (β₀, β₁)
        """
        if graph.number_of_nodes() == 0:
            return 0, 0

        # Para grafos dirigidos, usamos el grafo no dirigido subyacente
        # para cálculos topológicos algebraicos
        undirected_graph = graph.to_undirected()

        # β₀: Componentes conexas
        beta_0 = nx.number_connected_components(undirected_graph)

        # β₁: Ciclos independientes (rank del grupo de homología H₁)
        # Fórmula: β₁ = |E| - |V| + β₀
        n_edges = undirected_graph.number_of_edges()
        n_nodes = undirected_graph.number_of_nodes()
        beta_1 = max(0, n_edges - n_nodes + beta_0)

        return beta_0, beta_1

    def analyze_structural_integrity(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Analiza la integridad estructural del grafo de negocio.

        Calcula:
        1. Números de Betti (β₀, β₁)
        2. Ciclos simples (dependencias circulares)
        3. Componentes anómalas
        4. Métricas de conectividad

        Returns:
            Dict con métricas y detalles de anomalías
        """
        metrics = {}

        # === 1. CÁLCULOS TOPOLÓGICOS ALGEBRAICOS ===
        beta_0, beta_1 = self.calculate_betti_numbers(graph)

        # === 2. DETECCIÓN DE CICLOS (Dependencias circulares) ===
        circular_dependencies = []
        try:
            # Limitamos la búsqueda para grafos grandes
            max_cycles = 100
            cycle_iterator = nx.simple_cycles(graph)

            cycles_collected = 0
            for cycle in cycle_iterator:
                if cycles_collected >= max_cycles:
                    logger.warning(f"Límite de {max_cycles} ciclos alcanzado")
                    break
                circular_dependencies.append(" → ".join(cycle))
                cycles_collected += 1
        except Exception as e:
            logger.error(f"Error calculando ciclos: {e}")
            circular_dependencies = []

        # === 3. ANÁLISIS DE CONECTIVIDAD ===
        # Componentes débilmente conexas (para grafo dirigido)
        weakly_connected_components = list(nx.weakly_connected_components(graph))

        # Componentes fuertemente conexas (para detectar clusters de dependencia)
        strongly_connected_components = list(nx.strongly_connected_components(graph))

        # === 4. DETECCIÓN DE NODOS ANÓMALOS ===
        # Nodos completamente aislados (grado 0)
        isolates = list(nx.isolates(graph))

        # Clasificar nodos aislados
        orphan_resources = []
        empty_activities = []

        for node in isolates:
            node_data = graph.nodes[node]
            node_type = node_data.get("type")
            if node_type == "INSUMO":
                orphan_resources.append({
                    "id": node,
                    "description": node_data.get("description", ""),
                    "tipo_insumo": node_data.get("tipo_insumo", "")
                })
            elif node_type == "APU":
                empty_activities.append({
                    "id": node,
                    "description": node_data.get("description", ""),
                    "inferred": node_data.get("inferred", False)
                })

        # Insumos no utilizados (in_degree = 0) pero potencialmente conectados
        unused_resources = []
        for node, in_deg in graph.in_degree():
            node_data = graph.nodes[node]
            if node_data.get("type") == "INSUMO" and in_deg == 0 and node not in isolates:
                # Verificar si tiene out_degree (no debería en este modelo)
                out_deg = graph.out_degree(node)
                if out_deg == 0:
                    unused_resources.append({
                        "id": node,
                        "description": node_data.get("description", ""),
                        "tipo_insumo": node_data.get("tipo_insumo", "")
                    })

        # APUs sin insumos (out_degree = 0)
        empty_apus = []
        for node, out_deg in graph.out_degree():
            node_data = graph.nodes[node]
            if node_data.get("type") == "APU" and out_deg == 0 and node not in isolates:
                empty_apus.append({
                    "id": node,
                    "description": node_data.get("description", ""),
                    "inferred": node_data.get("inferred", False)
                })

        # === 5. MÉTRICAS ADICIONALES ===
        # Distribución de grados
        in_degrees = dict(graph.in_degree())
        out_degrees = dict(graph.out_degree())

        # Insumos más utilizados
        if in_degrees:
            max_in_degree = max(in_degrees.values())
            critical_resources = [
                node for node, deg in in_degrees.items()
                if deg == max_in_degree and graph.nodes[node].get("type") == "INSUMO"
            ][:5]  # Top 5
        else:
            critical_resources = []

        # === 6. COMPILAR RESULTADOS ===
        metrics = {
            # Métricas topológicas algebraicas
            "business.betti_b0": beta_0,
            "business.betti_b1": beta_1,

            # Métricas de ciclos
            "business.simple_cycles_count": len(circular_dependencies),
            "business.strongly_connected_components": len(strongly_connected_components),

            # Métricas de conectividad
            "business.weakly_connected_components": len(weakly_connected_components),
            "business.isolated_nodes_count": len(isolates),

            # Métricas de anomalías
            "business.orphan_resources_count": len(orphan_resources) + len(unused_resources),
            "business.empty_activities_count": len(empty_activities) + len(empty_apus),

            # Detalles para diagnóstico
            "details": {
                "betti_numbers": {
                    "beta_0": beta_0,
                    "beta_1": beta_1,
                    "formula": "β₁ = |E| - |V| + β₀"
                },
                "circular_dependencies": circular_dependencies,
                "weakly_connected_components": [
                    list(comp) for comp in weakly_connected_components[:5]
                ],
                "strongly_connected_components": [
                    list(comp) for comp in strongly_connected_components if len(comp) > 1
                ],
                "orphan_resources": orphan_resources + unused_resources,
                "empty_activities": empty_activities + empty_apus,
                "critical_resources": critical_resources,
                "graph_summary": {
                    "total_nodes": graph.number_of_nodes(),
                    "total_edges": graph.number_of_edges(),
                    "apu_nodes": len([n for n, d in graph.nodes(data=True) if d.get('type') == 'APU']),
                    "insumo_nodes": len([n for n, d in graph.nodes(data=True) if d.get('type') == 'INSUMO']),
                    "density": nx.density(graph),
                    "is_dag": nx.is_directed_acyclic_graph(graph)
                }
            }
        }

        # Inyectar métricas en telemetría
        if self.telemetry:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.telemetry.record_metric("business_topology", key, value)
            # Registrar métricas adicionales
            self.telemetry.record_metric(
                "business_topology",
                "business.graph_density",
                metrics["details"]["graph_summary"]["density"]
            )

        return metrics

    def get_audit_report(self, analysis_result: Dict[str, Any]) -> List[str]:
        """
        Genera un reporte legible para humanos basado en el análisis.

        Args:
            analysis_result: Resultado de analyze_structural_integrity

        Returns:
            Lista de líneas del reporte
        """
        report = []
        details = analysis_result.get("details", {})

        # Encabezado con resumen topológico
        betti_info = details.get("betti_numbers", {})
        report.append("=== ANÁLISIS TOPOLÓGICO DEL PRESUPUESTO ===")
        report.append(f"β₀ (Componentes conexas): {betti_info.get('beta_0', 0)}")
        report.append(f"β₁ (Ciclos independientes): {betti_info.get('beta_1', 0)}")

        # Verificar si es un DAG (Directed Acyclic Graph)
        if details.get("graph_summary", {}).get("is_dag", True):
            report.append("✓ El grafo es acíclico (DAG) - estructura jerárquica válida")
        else:
            report.append("⚠ El grafo contiene ciclos - revisar dependencias")

        # 1. Alertas de dependencias circulares
        cycles = details.get("circular_dependencies", [])
        if cycles:
            report.append(f"\n❌ ALERTA CRÍTICA: {len(cycles)} dependencia(s) circular(es) detectada(s):")
            for i, cycle in enumerate(cycles[:5], 1):
                report.append(f"   {i}. {cycle}")
            if len(cycles) > 5:
                report.append(f"   ... y {len(cycles) - 5} más.")
            report.append("   Esto causará errores de cálculo y debe corregirse.")

        # 2. Componentes fuertemente conexas (clusters de dependencia)
        sccs = details.get("strongly_connected_components", [])
        if sccs:
            report.append(f"\n⚠ Componentes fuertemente conexas detectadas: {len(sccs)}")
            for i, scc in enumerate(sccs[:3], 1):
                if len(scc) > 1:  # Solo mostrar componentes no triviales
                    report.append(f"   Cluster {i}: {len(scc)} elementos conectados")

        # 3. Recursos huérfanos/no utilizados
        orphans = details.get("orphan_resources", [])
        if orphans:
            report.append(f"\n⚠ Recursos no utilizados: {len(orphans)}")
            for resource in orphans[:5]:
                desc = resource.get('description', resource['id'])
                report.append(f"   • {resource['id']} ({desc})")
            if len(orphans) > 5:
                report.append(f"   ... y {len(orphans) - 5} más.")

        # 4. Actividades vacías
        empty_acts = details.get("empty_activities", [])
        if empty_acts:
            report.append(f"\n⚠ Actividades sin insumos: {len(empty_acts)}")
            for act in empty_acts[:5]:
                desc = act.get('description', act['id'])
                inferred = " (inferida)" if act.get('inferred') else ""
                report.append(f"   • {act['id']}{inferred}: {desc}")
            if len(empty_acts) > 5:
                report.append(f"   ... y {len(empty_acts) - 5} más.")

        # 5. Recursos críticos (más utilizados)
        critical = details.get("critical_resources", [])
        if critical and len(critical) > 0:
            report.append(f"\nℹ Recursos más utilizados (dependencias críticas):")
            for resource in critical[:3]:
                report.append(f"   • {resource}")

        # 6. Resumen final
        if not cycles and not orphans and not empty_acts:
            report.append("\n✅ La topología del presupuesto es estructuralmente sólida.")
        else:
            report.append(f"\n📊 Resumen: {len(cycles)} ciclos, "
                         f"{len(orphans)} recursos no usados, "
                         f"{len(empty_acts)} actividades vacías")

        return report
