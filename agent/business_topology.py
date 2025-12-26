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
    """

    integrity_score: float
    waste_alerts: List[str]
    circular_risks: List[str]
    complexity_level: str
    details: Dict[str, Any] = field(default_factory=dict)
    financial_risk_level: Optional[str] = None
    strategic_narrative: Optional[str] = None


class BudgetGraphBuilder:
    """Construye el Grafo del Presupuesto (Topología de Negocio) Versión 2 con estructura Piramidal."""

    def __init__(self):
        """Inicializa el constructor del grafo."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ROOT_NODE = "PROYECTO_TOTAL"

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

    def _create_node_attributes(
        self,
        node_type: str,
        level: int,
        source: str = "generated",
        idx: int = -1,
        inferred: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Helper para crear atributos de nodos consistentemente.
        """
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
        """
        Crea el diccionario de atributos para un nodo APU (Nivel 2).
        """
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
        """
        Crea el diccionario de atributos para un nodo Insumo (Nivel 3).
        """
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
        """
        Inserta o actualiza una arista acumulando cantidades y costos.
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
        """
        apu_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "APU"]
        insumo_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "INSUMO"]
        inferred_apus = [n for n in apu_nodes if G.nodes[n].get("inferred", False)]
        chapter_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "CAPITULO"]

        return {
            "chapter_count": len(chapter_nodes),
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
        Construye un grafo dirigido piramidal a partir de los DataFrames.
        Estructura: Proyecto (L0) -> [Capítulos (L1)] -> APUs (L2) -> Insumos (L3)
        """
        G = nx.DiGraph(name="BudgetTopology")
        self.logger.info("Iniciando construcción del Grafo Piramidal de Presupuesto...")

        # 0. Nodo Raíz: PROYECTO_TOTAL (Nivel 0)
        G.add_node(self.ROOT_NODE, type="ROOT", level=0, description="Proyecto Completo")

        # 1. Procesar Presupuesto (Niveles 1 y 2)
        if presupuesto_df is not None and not presupuesto_df.empty:
            for idx, row in presupuesto_df.iterrows():
                apu_code = self._sanitize_code(row.get(ColumnNames.CODIGO_APU))
                if not apu_code:
                    continue

                # Crear nodo APU (Nivel 2)
                attrs = self._create_apu_attributes(
                    row, source="presupuesto", idx=idx, inferred=False
                )
                G.add_node(apu_code, **attrs)

                # Gestionar Jerarquía de Capítulos (Nivel 1)
                # Intentamos detectar columnas que sugieran agrupación ("CAPITULO", "TITULO", etc.)
                # Por defecto buscamos 'CAPITULO' o 'CATEGORIA' si existen en el DF
                chapter_name = None
                possible_chapter_cols = ["CAPITULO", "CATEGORIA", "TITULO"]
                for col in possible_chapter_cols:
                    if col in presupuesto_df.columns:
                        val = self._sanitize_code(row.get(col))
                        if val:
                            chapter_name = val
                            break

                if chapter_name:
                    # Crear nodo Capítulo si no existe
                    if chapter_name not in G:
                        G.add_node(
                            chapter_name,
                            type="CAPITULO",
                            level=1,
                            description=f"Capítulo: {chapter_name}",
                        )
                        # Conectar Raíz -> Capítulo
                        if not G.has_edge(self.ROOT_NODE, chapter_name):
                            G.add_edge(self.ROOT_NODE, chapter_name, relation="CONTAINS")

                    # Conectar Capítulo -> APU
                    G.add_edge(chapter_name, apu_code, relation="CONTAINS")
                else:
                    # Si no hay capítulo, conectar Raíz -> APU directo (bypass nivel 1)
                    G.add_edge(self.ROOT_NODE, apu_code, relation="CONTAINS")

        # 2. Procesar detalle de APUs (Nivel 3: Insumos)
        if apus_detail_df is not None and not apus_detail_df.empty:
            for idx, row in apus_detail_df.iterrows():
                apu_code = self._sanitize_code(row.get(ColumnNames.CODIGO_APU))
                insumo_desc = self._sanitize_code(row.get(ColumnNames.DESCRIPCION_INSUMO))

                if not apu_code or not insumo_desc:
                    continue

                # Asegurar que el nodo APU existe (inferencia si no estaba en presupuesto)
                if apu_code not in G:
                    attrs = self._create_apu_attributes(
                        row, source="detail", idx=idx, inferred=True
                    )
                    G.add_node(apu_code, **attrs)
                    # Conectar APU inferido al Raíz por defecto (ya que no sabemos capítulo)
                    if not G.has_edge(self.ROOT_NODE, apu_code):
                        G.add_edge(self.ROOT_NODE, apu_code, relation="CONTAINS_INFERRED")

                # Asegurar que el nodo Insumo existe
                insumo_id = insumo_desc
                if insumo_id not in G:
                    attrs = self._create_insumo_attributes(
                        row, insumo_desc, source="detail", idx=idx
                    )
                    G.add_node(insumo_id, **attrs)

                # Insertar o actualizar arista APU -> Insumo
                qty = self._safe_float(row.get(ColumnNames.CANTIDAD_APU))
                cost = self._safe_float(row.get(ColumnNames.COSTO_INSUMO_EN_APU))

                self._upsert_edge(G, apu_code, insumo_id, cost, qty, idx)

        stats = self._compute_graph_statistics(G)
        self.logger.info(f"Grafo Piramidal construido: {stats}")
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

    def calculate_euler_efficiency(self, graph: nx.DiGraph) -> float:
        """
        Calcula la Eficiencia de Euler normalizada.

        La Característica de Euler (χ = V - E) mide la redundancia topológica.
        - En un árbol perfecto (sin ciclos, mínima redundancia): E = V - 1 => χ = 1.
        - En un grafo denso (muchas conexiones): E >> V => χ << 0.

        La eficiencia se define como la proximidad a una estructura arbórea (ideal para jerarquías de costos).

        Formula: Efficiency = 1 / (1 + max(0, Edges - Nodes + 1))

        Returns:
            float: Score entre 0.0 (Caos/Sobrecarga) y 1.0 (Orden/Jerarquía pura).
        """
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()

        if n_nodes == 0:
            return 1.0

        # Penalizamos el exceso de aristas sobre el mínimo necesario (árbol spanning)
        # Un árbol conectado tiene V-1 aristas.
        excess_edges = max(0, n_edges - (n_nodes - 1))

        # Usamos una función de decaimiento para normalizar
        efficiency = 1.0 / (1.0 + (excess_edges / n_nodes))

        return round(efficiency, 4)

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

        efficiency = self.calculate_euler_efficiency(graph)

        return TopologicalMetrics(
            beta_0=beta_0,
            beta_1=beta_1,
            euler_characteristic=chi,
            euler_efficiency=efficiency,
        )

    def calculate_pyramid_stability(self, graph: nx.DiGraph) -> float:
        """
        Calcula el Índice de Estabilidad Piramidal (Psi).

        Fórmula: Psi = (Num_Insumos / Num_APUs) * (1 / Densidad)

        Interpretación:
        - Psi > 10: Estructura Robusta (Base ancha, bajo acoplamiento).
        - Psi < 1: Pirámide Invertida (Base estrecha, alto riesgo de colapso).

        Args:
            graph: Grafo del presupuesto.

        Returns:
            float: Valor de estabilidad (0.0 si no se puede calcular).
        """
        # 1. Contar nodos por tipo
        nodes_data = graph.nodes(data=True)
        num_apus = sum(1 for _, d in nodes_data if d.get("type") == "APU")
        num_insumos = sum(1 for _, d in nodes_data if d.get("type") == "INSUMO")

        # 2. Validaciones para evitar división por cero
        if num_apus == 0:
            return 0.0

        density = nx.density(graph)
        if density == 0:
            return 0.0  # Grafo desconectado/vacío no es estable

        # 3. Cálculo
        base_ratio = num_insumos / num_apus
        stability = base_ratio * (1.0 / density)

        return round(stability, 2)

    def calculate_metrics(self, graph: nx.DiGraph) -> TopologicalMetrics:
        """Alias para compatibilidad hacia atrás."""
        return self.calculate_betti_numbers(graph)

    def _get_raw_cycles(self, graph: nx.DiGraph) -> Tuple[List[List[str]], bool]:
        """
        Obtiene los ciclos crudos (listas de nodos) del grafo.

        Returns:
            Tuple[List[List[str]], bool]: Lista de listas de nodos y flag de truncamiento.
        """
        cycles = []
        truncated = False
        try:
            # nx.simple_cycles retorna un generador
            # No queremos consumirlo todo si es infinito o enorme, pero simple_cycles en DiGraph es finito.
            # Aun así, para grafos densos puede ser costoso. Usamos un límite.
            cycle_gen = nx.simple_cycles(graph)

            # Consumir con límite
            count = 0
            for cycle in cycle_gen:
                cycles.append(cycle)
                count += 1
                if count >= self.max_cycles:
                    truncated = True
                    break
        except Exception as e:
            self.logger.error(f"Error detectando ciclos crudos: {e}")
        return cycles, truncated

    def _detect_cycles(self, graph: nx.DiGraph) -> Tuple[List[str], bool]:
        """
        Detecta ciclos en el grafo y los devuelve como strings formateados.

        Args:
            graph (nx.DiGraph): Grafo a analizar.

        Returns:
            Tuple[List[str], bool]: Lista de ciclos (strings) y flag de truncamiento.
        """
        formatted_cycles = []
        raw_cycles, truncated = self._get_raw_cycles(graph)

        for cycle in raw_cycles:
            # Agregar nodo inicial al final para cerrar el bucle en la representación
            cycle_repr = cycle + [cycle[0]]
            formatted_cycles.append(" → ".join(map(str, cycle_repr)))

        return formatted_cycles, truncated

    def detect_risk_synergy(
        self, graph: nx.DiGraph, raw_cycles: Optional[List[List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Detecta Sinergia de Riesgo (Efecto Dominó / Producto Cup Simulado).

        Busca ciclos que compartan "Nodos Críticos" (Insumos de alta centralidad).
        Si dos ciclos comparten nodos, sus riesgos se multiplican, no se suman.

        Args:
            graph (nx.DiGraph): El grafo.
            raw_cycles (Optional[List[List[str]]]): Ciclos pre-calculados para optimización.

        Returns:
            Dict con 'synergy_detected' (bool), 'shared_nodes' (list) y 'intersecting_cycles_count'.
        """
        if raw_cycles is None:
            raw_cycles, _ = self._get_raw_cycles(graph)

        if len(raw_cycles) < 2:
            return {
                "synergy_detected": False,
                "shared_nodes": [],
                "intersecting_cycles_count": 0,
            }

        # 1. Identificar nodos críticos (top 10% por grado)
        critical_nodes = set()
        degrees = dict(graph.degree())
        if degrees:
            # Umbral: Percentil 90 o simplemente nodos con degree > 2 si el grafo es pequeño
            sorted_degrees = sorted(degrees.values(), reverse=True)
            threshold_idx = max(0, int(len(sorted_degrees) * 0.1))
            # Fallback para grafos pequeños: degree > 1
            degree_threshold = max(2, sorted_degrees[threshold_idx] if sorted_degrees else 0)

            critical_nodes = {n for n, d in degrees.items() if d >= degree_threshold}

        # 2. Buscar intersecciones entre ciclos en nodos críticos
        shared_critical_nodes = set()
        intersecting_count = 0

        # Convertir ciclos a sets para intersección rápida
        cycle_sets = [set(c) for c in raw_cycles]

        # Comparación O(N^2) sobre número de ciclos (usualmente bajo por max_cycles)
        for i in range(len(cycle_sets)):
            for j in range(i + 1, len(cycle_sets)):
                intersection = cycle_sets[i].intersection(cycle_sets[j])

                # Filtrar solo nodos críticos en la intersección
                critical_intersection = intersection.intersection(critical_nodes)

                if critical_intersection:
                    shared_critical_nodes.update(critical_intersection)
                    intersecting_count += 1

        return {
            "synergy_detected": len(shared_critical_nodes) > 0,
            "shared_nodes": list(shared_critical_nodes),
            "intersecting_cycles_count": intersecting_count,
        }

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

            # Ajuste para Root Node: No es aislado, es el origen
            if data.get("type") == "ROOT":
                continue

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
            "efficiency": f"Eficiencia de Euler: {metrics.euler_efficiency:.2%}",
        }

    def generate_executive_report(
        self, graph: nx.DiGraph, financial_metrics: Optional[Dict[str, Any]] = None
    ) -> ConstructionRiskReport:
        """
        Genera un reporte de riesgos combinando análisis topológico y financiero.

        Args:
            graph (nx.DiGraph): Grafo a reportar.
            financial_metrics (Optional[Dict[str, Any]]): Métricas financieras opcionales.

        Returns:
            ConstructionRiskReport: Objeto de reporte estructurado.
        """
        metrics = self.calculate_betti_numbers(graph)
        raw_cycles, truncated = self._get_raw_cycles(graph)
        cycles = []
        for c in raw_cycles:
            cycle_repr = c + [c[0]]
            cycles.append(" → ".join(map(str, cycle_repr)))

        synergy = self.detect_risk_synergy(graph, raw_cycles)

        anomalies = self._classify_anomalous_nodes(graph)
        pyramid_stability = self.calculate_pyramid_stability(graph)

        # 1. Análisis de Integridad Estructural (Topológico)
        integrity_score = 100.0
        if metrics.beta_1 > 0:
            integrity_score -= 50

        # Penalización por Sinergia de Riesgo (Producto Cup)
        if synergy["synergy_detected"]:
            integrity_score -= 20

        isolated_count = len(anomalies["isolated_nodes"])
        integrity_score -= min(30, isolated_count * 2)
        orphan_count = len(anomalies["orphan_insumos"])
        integrity_score -= min(20, orphan_count * 1)
        integrity_score = max(0, integrity_score)

        density = nx.density(graph) if graph else 0.0
        complexity = (
            "Alta (Crítica)"
            if metrics.beta_1 > 0
            else ("Alta" if density > 0.1 else "Media" if density > 0.05 else "Baja")
        )

        waste_alerts = []
        if isolated_count > 0:
            waste_alerts.append(f"Alerta: {isolated_count} Insumos no utilizados.")
        if orphan_count > 0:
            waste_alerts.append(f"Alerta: {orphan_count} Recursos sin asignación a APUs.")

        # Alerta de Eficiencia de Euler
        if metrics.euler_efficiency < 0.5:
            waste_alerts.append(
                f"Alerta de Gestión: Baja Eficiencia de Euler ({metrics.euler_efficiency:.2f}). Sobrecarga de enlaces."
            )

        # Alerta de Estabilidad Piramidal
        if pyramid_stability < 10.0 and graph.number_of_nodes() > 10:
            waste_alerts.append(
                f"Alerta Estructural: Baja estabilidad piramidal ({pyramid_stability:.1f}). Posible estructura invertida o excesivamente compleja."
            )

        circular_risks = []
        if metrics.beta_1 > 0:
            circular_risks.append("CRÍTICO: Referencias circulares detectadas.")

        if synergy["synergy_detected"]:
            circular_risks.append(
                f"RIESGO SISTÉMICO: Sinergia detectada entre {synergy['intersecting_cycles_count']} ciclos. Efecto Dominó probable."
            )

        # 2. Análisis de Riesgo Financiero
        financial_risk = None
        if financial_metrics:
            volatility = financial_metrics.get("volatility", 0.0)
            roi = financial_metrics.get("roi", 0.0)

            if roi < 0:
                financial_risk = "CRÍTICO"
            elif volatility > 0.20:
                financial_risk = "ALTO"
            elif volatility > 0.10:
                financial_risk = "MEDIO"
            else:
                financial_risk = "BAJO"

            # 3. Combinación de Riesgos
            if (
                metrics.beta_1 > 0 or synergy["synergy_detected"]
            ) and financial_risk == "ALTO":
                financial_risk = "CATÁSTROFICO"

        return ConstructionRiskReport(
            integrity_score=integrity_score,
            waste_alerts=waste_alerts,
            circular_risks=circular_risks,
            complexity_level=complexity,
            financial_risk_level=financial_risk,
            details={
                "metrics": asdict(metrics),
                "cycles": cycles,
                "anomalies": anomalies,
                "density": density,
                "pyramid_stability": pyramid_stability,
                "synergy_risk": synergy,
                "financial_metrics_input": financial_metrics or {},
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
        raw_cycles, cycles_truncated = self._get_raw_cycles(graph)
        cycles_str = []
        for c in raw_cycles:
            cycle_repr = c + [c[0]]
            cycles_str.append(" → ".join(map(str, cycle_repr)))

        synergy = self.detect_risk_synergy(graph, raw_cycles)
        anomalies = self._classify_anomalous_nodes(graph)
        critical = self._identify_critical_resources(graph)
        connectivity = self._compute_connectivity_analysis(graph)
        interpretation = self._interpret_topology(metrics)
        pyramid_stability = self.calculate_pyramid_stability(graph)

        # Reporte ejecutivo para mejora de telemetría
        exec_report = self.generate_executive_report(graph)

        # Estructura para reporte
        details = {
            "topology": {"betti_numbers": asdict(metrics), "interpretation": interpretation},
            "cycles": {
                "count": len(cycles_str),
                "list": cycles_str,
                "truncated": cycles_truncated,
            },
            "synergy_risk": synergy,
            "connectivity": connectivity,
            "anomalies": anomalies,
            "critical_resources": critical,
            "graph_summary": {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "density": nx.density(graph),
                "pyramid_stability": pyramid_stability,
            },
            "executive_report": asdict(exec_report),
        }

        # Métricas planas para telemetría/conveniencia
        flat_results = {
            "business.betti_b0": metrics.beta_0,
            "business.betti_b1": metrics.beta_1,
            "business.euler_characteristic": metrics.euler_characteristic,
            "business.euler_efficiency": metrics.euler_efficiency,
            "business.cycles_count": len(cycles_str),
            "business.synergy_detected": 1 if synergy["synergy_detected"] else 0,
            "business.is_dag": 1 if connectivity["is_dag"] else 0,
            "business.isolated_count": len(anomalies["isolated_nodes"]),
            "business.orphan_insumos_count": len(anomalies["orphan_insumos"]),
            "business.empty_apus_count": len(anomalies["empty_apus"]),
            "business.integrity_score": exec_report.integrity_score,
            "business.pyramid_stability": pyramid_stability,
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
            euler_efficiency=new_result["business.euler_efficiency"],
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
        pyramid_stability = 0.0

        if isinstance(analysis_result_or_graph, nx.DiGraph):
            exec_report = self.generate_executive_report(analysis_result_or_graph)
            metrics_dict = exec_report.details.get("metrics", {})
            anomalies = exec_report.details.get("anomalies", {})
            cycles_list = exec_report.details.get("cycles", [])
            density = exec_report.details.get("density", 0.0)
            pyramid_stability = exec_report.details.get("pyramid_stability", 0.0)
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
                    pyramid_stability = er_dict.get("details", {}).get(
                        "pyramid_stability", 0.0
                    )
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
        lines.append(
            f"│ Eficiencia de Euler:       {metrics_dict.get('euler_efficiency', 0.0):<22.2f}│"
        )
        lines.append(f"│ Densidad de Conexiones:    {density:.4f}                │")
        lines.append(
            f"│ Estabilidad Piramidal:     {pyramid_stability:.2f}                │"
        )
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
