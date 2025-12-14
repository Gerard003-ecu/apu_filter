import logging
import textwrap
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
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
    """

    beta_0: int
    beta_1: int
    euler_characteristic: int

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
    """

    integrity_score: float
    waste_alerts: List[str]
    circular_risks: List[str]
    complexity_level: str
    details: Dict[str, Any] = field(default_factory=dict)


class BudgetGraphBuilder:
    """Construye el Grafo del Presupuesto (Topología de Negocio) Versión 2."""

    def __init__(self):
        """Inicializa el constructor del grafo."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def _sanitize_code(self, value: Any) -> str:
        """
        Sanitiza el código o identificador asegurando que sea una cadena limpia.

        Args:
            value (Any): Valor a sanitizar.

        Returns:
            str: Cadena sanitizada o vacía si el valor es nulo.
        """
        if pd.isna(value) or value is None:
            return ""
        return str(value).strip()

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """
        Convierte un valor a float de manera segura.

        Args:
            value (Any): Valor a convertir.
            default (float): Valor por defecto si falla la conversión.

        Returns:
            float: Valor numérico convertido.
        """
        try:
            if pd.isna(value):
                return default
            return float(value)
        except (ValueError, TypeError):
            return default

    def _create_apu_attributes(
        self, row: pd.Series, source: str, idx: int, inferred: bool
    ) -> Dict[str, Any]:
        """
        Crea el diccionario de atributos para un nodo APU.

        Args:
            row (pd.Series): Fila de datos del APU.
            source (str): Fuente de los datos ('presupuesto' o 'detail').
            idx (int): Índice original en el DataFrame.
            inferred (bool): Indica si el APU fue inferido desde detalles.

        Returns:
            Dict[str, Any]: Diccionario de atributos del nodo.
        """
        attrs = {
            "type": "APU",
            "source": source,
            "original_index": idx,
            "inferred": inferred,
        }
        if not inferred:
            attrs["description"] = self._sanitize_code(row.get(ColumnNames.DESCRIPCION_APU))
            attrs["quantity"] = self._safe_float(row.get(ColumnNames.CANTIDAD_PRESUPUESTO))
        return attrs

    def _create_insumo_attributes(
        self, row: pd.Series, insumo_desc: str, source: str, idx: int
    ) -> Dict[str, Any]:
        """
        Crea el diccionario de atributos para un nodo Insumo.

        Args:
            row (pd.Series): Fila de datos del insumo.
            insumo_desc (str): Descripción del insumo.
            source (str): Fuente de los datos.
            idx (int): Índice original en el DataFrame.

        Returns:
            Dict[str, Any]: Diccionario de atributos del nodo.
        """
        return {
            "type": "INSUMO",
            "description": insumo_desc,
            "tipo_insumo": self._sanitize_code(row.get(ColumnNames.TIPO_INSUMO)),
            "unit_cost": self._safe_float(row.get(ColumnNames.COSTO_INSUMO_EN_APU)),
            "source": source,
            "original_index": idx,
        }

    def _upsert_edge(
        self, G: nx.DiGraph, u: str, v: str, unit_cost: float, quantity: float, idx: int
    ) -> bool:
        """
        Inserta o actualiza una arista acumulando cantidades y costos.

        Args:
            G (nx.DiGraph): El grafo dirigido.
            u (str): Nodo origen (APU).
            v (str): Nodo destino (Insumo).
            unit_cost (float): Costo unitario.
            quantity (float): Cantidad.
            idx (int): Índice del registro original.

        Returns:
            bool: True si es una nueva arista, False si se actualizó una existente.
        """
        total_cost = unit_cost * quantity
        is_new = False

        if G.has_edge(u, v):
            edge_data = G[u][v]
            # Acumular
            new_qty = edge_data["quantity"] + quantity
            new_total = edge_data["total_cost"] + total_cost

            # Actualizar atributos
            G[u][v]["quantity"] = new_qty
            G[u][v]["total_cost"] = new_total
            G[u][v]["occurrence_count"] += 1
            if "original_indices" in G[u][v]:
                G[u][v]["original_indices"].append(idx)
        else:
            is_new = True
            G.add_edge(
                u,
                v,
                quantity=quantity,
                unit_cost=unit_cost,
                total_cost=total_cost,
                occurrence_count=1,
                original_indices=[idx],
            )
        return is_new

    def _compute_graph_statistics(self, G: nx.DiGraph) -> Dict[str, int]:
        """
        Calcula estadísticas básicas del grafo construido.

        Args:
            G (nx.DiGraph): El grafo construido.

        Returns:
            Dict[str, int]: Estadísticas (conteo de nodos APU, Insumos, etc.).
        """
        apu_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "APU"]
        insumo_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "INSUMO"]
        inferred_apus = [n for n in apu_nodes if G.nodes[n].get("inferred", False)]

        return {
            "apu_count": len(apu_nodes),
            "insumo_count": len(insumo_nodes),
            "inferred_count": len(inferred_apus),
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
        }

    def build(
        self, presupuesto_df: pd.DataFrame, apus_detail_df: pd.DataFrame
    ) -> nx.DiGraph:
        """
        Construye un grafo dirigido optimizado a partir de los DataFrames.

        Args:
            presupuesto_df (pd.DataFrame): DataFrame del presupuesto.
            apus_detail_df (pd.DataFrame): DataFrame con el detalle de los APUs.

        Returns:
            nx.DiGraph: Grafo dirigido representando la topología del presupuesto.
        """
        G = nx.DiGraph(name="BudgetTopology")
        self.logger.info("Iniciando construcción del Grafo de Presupuesto V2...")

        # 1. Agregar nodos APU desde presupuesto
        if presupuesto_df is not None and not presupuesto_df.empty:
            for idx, row in presupuesto_df.iterrows():
                apu_code = self._sanitize_code(row.get(ColumnNames.CODIGO_APU))
                if not apu_code:
                    continue

                attrs = self._create_apu_attributes(
                    row, source="presupuesto", idx=idx, inferred=False
                )
                # Usar add_node actualiza atributos si ya existe
                G.add_node(apu_code, **attrs)

        # 2. Procesar detalle de APUs
        if apus_detail_df is not None and not apus_detail_df.empty:
            for idx, row in apus_detail_df.iterrows():
                apu_code = self._sanitize_code(row.get(ColumnNames.CODIGO_APU))
                insumo_desc = self._sanitize_code(row.get(ColumnNames.DESCRIPCION_INSUMO))

                if not apu_code or not insumo_desc:
                    continue

                # Asegurar que el nodo APU existe (inferencia)
                if apu_code not in G:
                    attrs = self._create_apu_attributes(
                        row, source="detail", idx=idx, inferred=True
                    )
                    G.add_node(apu_code, **attrs)

                # Asegurar que el nodo Insumo existe
                # Usando la descripción como ID por consistencia con V1
                insumo_id = insumo_desc
                if insumo_id not in G:
                    attrs = self._create_insumo_attributes(
                        row, insumo_desc, source="detail", idx=idx
                    )
                    G.add_node(insumo_id, **attrs)

                # Insertar o actualizar arista
                qty = self._safe_float(row.get(ColumnNames.CANTIDAD_APU))
                cost = self._safe_float(row.get(ColumnNames.COSTO_INSUMO_EN_APU))

                self._upsert_edge(G, apu_code, insumo_id, cost, qty, idx)

        stats = self._compute_graph_statistics(G)
        self.logger.info(f"Grafo construido: {stats}")
        return G


class BusinessTopologicalAnalyzer:
    """Analizador de topología de negocio V2 con Telemetría Granular."""

    def __init__(self, telemetry: Optional[TelemetryContext] = None, max_cycles: int = 100):
        """
        Inicializa el analizador topológico.

        Args:
            telemetry (Optional[TelemetryContext]): Contexto para registrar métricas.
            max_cycles (int): Número máximo de ciclos a detectar antes de truncar.
        """
        self.telemetry = telemetry
        self.max_cycles = max_cycles
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_betti_numbers(self, graph: nx.DiGraph) -> TopologicalMetrics:
        """
        Calcula métricas topológicas invariantes (Números de Betti).

        Args:
            graph (nx.DiGraph): El grafo a analizar.

        Returns:
            TopologicalMetrics: Métricas calculadas (beta_0, beta_1, Euler).
        """
        if graph.number_of_nodes() == 0:
            return TopologicalMetrics(0, 0, 0)

        undirected = nx.MultiGraph()
        undirected.add_nodes_from(graph.nodes(data=True))
        undirected.add_edges_from(graph.edges(data=True))

        # Beta 0: Componentes Conexas
        beta_0 = nx.number_connected_components(undirected)

        # Conteos
        n_edges = undirected.number_of_edges()
        n_nodes = undirected.number_of_nodes()

        # Beta 1: Ciclos
        # Característica de Euler chi = V - E = beta_0 - beta_1 (para complejos 1D)
        # Por lo tanto, beta_1 = beta_0 - V + E
        beta_1 = max(0, beta_0 - n_nodes + n_edges)

        chi = beta_0 - beta_1

        return TopologicalMetrics(beta_0=beta_0, beta_1=beta_1, euler_characteristic=chi)

    def calculate_metrics(self, graph: nx.DiGraph) -> TopologicalMetrics:
        """Alias para compatibilidad hacia atrás."""
        return self.calculate_betti_numbers(graph)

    def _detect_cycles(self, graph: nx.DiGraph) -> Tuple[List[str], bool]:
        """
        Detecta ciclos en el grafo.

        Args:
            graph (nx.DiGraph): Grafo a analizar.

        Returns:
            Tuple[List[str], bool]: Lista de ciclos (strings) y flag de truncamiento.
        """
        cycles = []
        truncated = False
        try:
            raw_cycles = list(nx.simple_cycles(graph))
            if len(raw_cycles) > self.max_cycles:
                raw_cycles = raw_cycles[: self.max_cycles]
                truncated = True

            for cycle in raw_cycles:
                # Agregar nodo inicial al final para cerrar el bucle en la representación
                cycle_repr = cycle + [cycle[0]]
                cycles.append(" → ".join(map(str, cycle_repr)))
        except Exception as e:
            self.logger.error(f"Error detectando ciclos: {e}")
        return cycles, truncated

    def _classify_anomalous_nodes(
        self, graph: nx.DiGraph
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Clasifica nodos en categorías anómalas.

        Args:
            graph (nx.DiGraph): Grafo a analizar.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Diccionario con listas de nodos aislados, huérfanos y vacíos.
        """
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
                "out_degree": out_degrees.get(node, 0),
            }

            is_isolated = (in_degrees.get(node, 0) == 0) and (out_degrees.get(node, 0) == 0)

            if is_isolated:
                isolated_nodes.append(node_info)

            if (
                data.get("type") == "INSUMO"
                and in_degrees.get(node, 0) == 0
                and not is_isolated
            ):
                orphan_insumos.append(node_info)
            elif data.get("type") == "INSUMO" and is_isolated:
                # Insumos aislados también son huérfanos
                orphan_insumos.append(node_info)

            if data.get("type") == "APU" and out_degrees.get(node, 0) == 0:
                empty_apus.append(node_info)

        return {
            "isolated_nodes": isolated_nodes,
            "orphan_insumos": orphan_insumos,
            "empty_apus": empty_apus,
        }

    def _identify_critical_resources(
        self, graph: nx.DiGraph, top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Identifica recursos críticos (alto grado de entrada).

        Args:
            graph (nx.DiGraph): Grafo a analizar.
            top_n (int): Número de recursos top a retornar.

        Returns:
            List[Dict[str, Any]]: Lista de recursos críticos con su grado.
        """
        resources = []
        for node, data in graph.nodes(data=True):
            if data.get("type") == "INSUMO":
                degree = graph.in_degree(node)
                if degree > 0:
                    resources.append(
                        {
                            "id": node,
                            "in_degree": degree,
                            "description": data.get("description", ""),
                        }
                    )

        resources.sort(key=lambda x: x["in_degree"], reverse=True)
        return resources[:top_n]

    def _compute_connectivity_analysis(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Calcula métricas de conectividad y componentes fuertemente conexas.

        Args:
            graph (nx.DiGraph): Grafo a analizar.

        Returns:
            Dict[str, Any]: Diccionario con resultados de conectividad.
        """
        return {
            "is_dag": nx.is_directed_acyclic_graph(graph),
            "num_wcc": nx.number_weakly_connected_components(graph),
            "is_weakly_connected": nx.is_weakly_connected(graph),
            "num_non_trivial_scc": len(
                [c for c in nx.strongly_connected_components(graph) if len(c) > 1]
            ),
            "non_trivial_scc": [
                list(c) for c in nx.strongly_connected_components(graph) if len(c) > 1
            ],
        }

    def _interpret_topology(self, metrics: TopologicalMetrics) -> Dict[str, str]:
        """
        Genera interpretaciones legibles por humanos de las métricas.

        Args:
            metrics (TopologicalMetrics): Métricas a interpretar.

        Returns:
            Dict[str, str]: Interpretaciones de beta_0, beta_1 y euler.
        """
        return {
            "beta_0": f"{metrics.beta_0} componente(s) conexa(s). {'Espacio conexo' if metrics.is_connected else 'Espacio fragmentado'}.",
            "beta_1": f"{metrics.beta_1} ciclo(s) independiente(s). {'Estructura acíclica' if metrics.beta_1 == 0 else 'Complejidad cíclica presente'}.",
            "euler": f"Característica de Euler: {metrics.euler_characteristic}",
        }

    def generate_executive_report(self, graph: nx.DiGraph) -> ConstructionRiskReport:
        """
        Genera un reporte de riesgos traducido a lenguaje de construcción.

        Args:
            graph (nx.DiGraph): Grafo a reportar.

        Returns:
            ConstructionRiskReport: Objeto de reporte estructurado.
        """
        metrics = self.calculate_betti_numbers(graph)
        cycles, _ = self._detect_cycles(graph)
        anomalies = self._classify_anomalous_nodes(graph)

        # Calcular puntuaciones y niveles
        integrity_score = 100.0

        # Penalizaciones
        if metrics.beta_1 > 0:
            integrity_score -= 50  # Crítico: Referencias circulares

        isolated_count = len(anomalies["isolated_nodes"])
        if isolated_count > 0:
            integrity_score -= min(30, isolated_count * 2)

        orphan_count = len(anomalies["orphan_insumos"])
        if orphan_count > 0:
            integrity_score -= min(20, orphan_count * 1)

        integrity_score = max(0, integrity_score)

        # Nivel de Complejidad basado en densidad/beta_1
        density = nx.density(graph)
        if metrics.beta_1 > 0:
            complexity = "Alta (Crítica)"
        elif density > 0.1:
            complexity = "Alta"
        elif density > 0.05:
            complexity = "Media"
        else:
            complexity = "Baja"

        # Traducir alertas
        waste_alerts = []
        if isolated_count > 0:
            waste_alerts.append(
                f"Alerta: {isolated_count} Insumos en base de datos no se utilizan en el presupuesto."
            )
        if orphan_count > 0:
            waste_alerts.append(
                f"Alerta: {orphan_count} Recursos definidos sin asignación a APUs."
            )

        circular_risks = []
        if metrics.beta_1 > 0:
            circular_risks.append(
                "CRÍTICO: Se detectaron referencias circulares en los precios unitarios."
            )

        return ConstructionRiskReport(
            integrity_score=integrity_score,
            waste_alerts=waste_alerts,
            circular_risks=circular_risks,
            complexity_level=complexity,
            details={
                "metrics": asdict(metrics),
                "cycles": cycles,
                "anomalies": anomalies,
                "density": density,
            },
        )

    def analyze_structural_integrity(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Ejecuta análisis completo y emite telemetría.

        Args:
            graph (nx.DiGraph): Grafo a analizar.

        Returns:
            Dict[str, Any]: Resultados detallados y métricas planas para telemetría.
        """
        metrics = self.calculate_betti_numbers(graph)
        cycles, cycles_truncated = self._detect_cycles(graph)
        anomalies = self._classify_anomalous_nodes(graph)
        critical = self._identify_critical_resources(graph)
        connectivity = self._compute_connectivity_analysis(graph)
        interpretation = self._interpret_topology(metrics)

        # Reporte ejecutivo para mejora de telemetría
        exec_report = self.generate_executive_report(graph)

        # Estructura para reporte
        details = {
            "topology": {"betti_numbers": asdict(metrics), "interpretation": interpretation},
            "cycles": {"count": len(cycles), "list": cycles, "truncated": cycles_truncated},
            "connectivity": connectivity,
            "anomalies": anomalies,
            "critical_resources": critical,
            "graph_summary": {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "density": nx.density(graph),
            },
            "executive_report": asdict(exec_report),
        }

        # Métricas planas para telemetría/conveniencia
        flat_results = {
            "business.betti_b0": metrics.beta_0,
            "business.betti_b1": metrics.beta_1,
            "business.euler_characteristic": metrics.euler_characteristic,
            "business.cycles_count": len(cycles),
            "business.is_dag": 1 if connectivity["is_dag"] else 0,
            "business.isolated_count": len(anomalies["isolated_nodes"]),
            "business.orphan_insumos_count": len(anomalies["orphan_insumos"]),
            "business.empty_apus_count": len(anomalies["empty_apus"]),
            "business.integrity_score": exec_report.integrity_score,
            "details": details,
        }

        # Telemetría
        if self.telemetry:
            for k, v in flat_results.items():
                if isinstance(v, (int, float)):
                    self.telemetry.record_metric(k, v)

        return flat_results

    def analyze(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Wrapper de compatibilidad para análisis."""
        new_result = self.analyze_structural_integrity(graph)

        metrics = TopologicalMetrics(
            beta_0=new_result["business.betti_b0"],
            beta_1=new_result["business.betti_b1"],
            euler_characteristic=new_result["business.euler_characteristic"],
        )
        # Agregar campos antiguos que estaban en V1 pero no son invariantes topológicos estrictos
        metrics_dict = asdict(metrics)
        metrics_dict["density"] = new_result["details"]["graph_summary"]["density"]
        metrics_dict["is_dag"] = bool(new_result["business.is_dag"])
        # Chi fue alias de euler_characteristic
        metrics_dict["chi"] = metrics.euler_characteristic

        anomalies = {
            "cycles": new_result["details"]["cycles"]["list"],
            "isolates_count": new_result["business.isolated_count"],
            "orphan_insumos_count": new_result["business.orphan_insumos_count"],
            "empty_apus_count": new_result["business.empty_apus_count"],
        }

        return {"metrics": metrics_dict, "anomalies": anomalies}

    def get_audit_report(self, analysis_result_or_graph: Any) -> List[str]:
        """
        Genera un reporte ASCII art profesional enfocado en construcción.

        Args:
            analysis_result_or_graph (Any): Grafo o resultado de análisis.

        Returns:
            List[str]: Líneas del reporte formateado.
        """
        exec_report = None

        if isinstance(analysis_result_or_graph, nx.DiGraph):
            exec_report = self.generate_executive_report(analysis_result_or_graph)
            metrics_dict = exec_report.details.get("metrics", {})
            anomalies = exec_report.details.get("anomalies", {})
            cycles_list = exec_report.details.get("cycles", [])
            density = exec_report.details.get("density", 0.0)
        else:
            if isinstance(analysis_result_or_graph, dict):
                if (
                    "details" in analysis_result_or_graph
                    and "executive_report" in analysis_result_or_graph["details"]
                ):
                    er_dict = analysis_result_or_graph["details"]["executive_report"]
                    exec_report = ConstructionRiskReport(
                        integrity_score=er_dict.get("integrity_score", 0.0),
                        waste_alerts=er_dict.get("waste_alerts", []),
                        circular_risks=er_dict.get("circular_risks", []),
                        complexity_level=er_dict.get("complexity_level", "Desconocida"),
                        details=er_dict.get("details", {}),
                    )
                    metrics_dict = er_dict.get("details", {}).get("metrics", {})
                    anomalies = er_dict.get("details", {}).get("anomalies", {})
                    cycles_list = er_dict.get("details", {}).get("cycles", [])
                    density = er_dict.get("details", {}).get("density", 0.0)
                elif "metrics" in analysis_result_or_graph:
                    # Compatibilidad hacia atrás
                    m = analysis_result_or_graph["metrics"]
                    a = analysis_result_or_graph["anomalies"]
                    cycles_list = a.get("cycles", [])

                    beta_1 = m.get("beta_1", 0)
                    isolated_count = a.get("isolates_count", 0)

                    integrity_score = 100.0
                    if beta_1 > 0:
                        integrity_score -= 50
                    if isolated_count > 0:
                        integrity_score -= min(30, isolated_count * 2)

                    waste_alerts = []
                    if isolated_count > 0:
                        waste_alerts.append(
                            f"Alerta: {isolated_count} Insumos en base de datos no se utilizan en el presupuesto."
                        )

                    circular_risks = []
                    if beta_1 > 0:
                        circular_risks.append(
                            "CRÍTICO: Se detectaron referencias circulares en los precios unitarios."
                        )

                    exec_report = ConstructionRiskReport(
                        integrity_score=integrity_score,
                        waste_alerts=waste_alerts,
                        circular_risks=circular_risks,
                        complexity_level="Desconocida",
                        details={},
                    )
                    metrics_dict = m
                    anomalies = {
                        "isolated_nodes": [{}] * isolated_count,
                        "orphan_insumos": [{}] * a.get("orphan_insumos_count", 0),
                        "empty_apus": [{}] * a.get("empty_apus_count", 0),
                    }
                    density = m.get("density", 0.0)
                else:
                    return ["Error: Formato de análisis no reconocido."]

        if exec_report is None:
            return ["Error: No se pudo generar el reporte ejecutivo."]

        lines = []
        lines.append("┌──────────────────────────────────────────────────┐")
        lines.append("│      AUDITORÍA ESTRUCTURAL DEL PRESUPUESTO       │")
        lines.append("├──────────────────────────────────────────────────┤")
        lines.append(
            f"│ PUNTUACIÓN DE INTEGRIDAD: {exec_report.integrity_score:>6.1f} / 100.0          │"
        )
        lines.append(f"│ Nivel de Complejidad:     {exec_report.complexity_level:<23}│")
        lines.append("├──────────────────────────────────────────────────┤")
        lines.append("│ [MÉTRICAS TÉCNICAS]                              │")
        lines.append(f"│ Ciclos de Costo (Errores): {metrics_dict.get('beta_1', 0):<22}│")
        lines.append(f"│ Componentes Conexas:       {metrics_dict.get('beta_0', 0):<22}│")
        lines.append(f"│ Densidad de Conexiones:    {density:.4f}                │")
        lines.append("├──────────────────────────────────────────────────┤")

        if exec_report.circular_risks:
            lines.append("│ [ERRORES CRÍTICOS]                               │")
            for risk in exec_report.circular_risks:
                wrapped_lines = textwrap.wrap(risk, width=44)
                for line in wrapped_lines:
                    lines.append(f"│ ❌ {line:<44} │")

            for i, cycle in enumerate(cycles_list[:3]):
                c_str = cycle
                if len(c_str) > 38:
                    c_str = c_str[:35] + "..."
                lines.append(f"│    {i + 1}. {c_str:<38} │")
        else:
            lines.append("│ [ESTADO]                                         │")
            lines.append("│ ✅ Estructura de Costos Saludable y Auditable.   │")

        iso_count = len(anomalies.get("isolated_nodes", []))
        orphan_count = len(anomalies.get("orphan_insumos", []))
        empty_count = len(anomalies.get("empty_apus", []))

        if exec_report.waste_alerts or iso_count > 0 or orphan_count > 0 or empty_count > 0:
            lines.append("├──────────────────────────────────────────────────┤")
            lines.append("│ [POSIBLE DESPERDICIO / ALERTAS]                  │")

            # Usar etiquetas explícitas si counts > 0
            if iso_count > 0:
                lines.append(f"│ ⚠ Recursos Fantasma (Sin uso): {iso_count:<18}│")

            if empty_count > 0:
                lines.append(f"│ ⚠ APUs Vacíos:          {empty_count:<25}│")

            # Mostrar alertas detalladas, con wrap
            for alert in exec_report.waste_alerts:
                wrapped_lines = textwrap.wrap(alert, width=44)
                for line in wrapped_lines:
                    lines.append(f"│ ⚠ {line:<44} │")

        lines.append("└──────────────────────────────────────────────────┘")

        return lines
