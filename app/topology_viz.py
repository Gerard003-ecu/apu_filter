"""
Módulo de visualización de topología para la aplicación Flask.
Este módulo extiende app.py añadiendo endpoints para Cytoscape.js.
Implementa el concepto de "Caja de Cristal" (Glass Box), transformando la visualización
en un Microscopio Estructural para la auditabilidad forense de riesgos lógicos.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from flask import Blueprint, jsonify, session, current_app, Response
import networkx as nx
import pandas as pd
import logging
import sys
from pathlib import Path

# Asegurar que la raíz del proyecto esté en sys.path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from agent.business_topology import BudgetGraphBuilder, BusinessTopologicalAnalyzer

topology_bp = Blueprint('topology', __name__)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTES
# =============================================================================

class NodeType(str, Enum):
    """Tipos de nodos en el grafo de presupuesto."""
    BUDGET = "BUDGET"
    CHAPTER = "CHAPTER"
    ITEM = "ITEM"
    APU = "APU"
    INSUMO = "INSUMO"
    UNKNOWN = "UNKNOWN"


class NodeColor(str, Enum):
    """Colores para visualización de nodos."""
    RED = "#EF4444"     # Ciclos, Islas (Riesgo), Estrés
    BLUE = "#3B82F6"    # APU (Normal)
    ORANGE = "#F97316"  # Insumo (Recurso)
    BLACK = "#1E293B"   # Raíz / Presupuesto / Capítulos
    GRAY = "#9CA3AF"    # Desconocido / Vacío


class NodeClass(str, Enum):
    """Clases CSS para visualización de nodos."""
    NORMAL = "normal"
    CYCLE = "cycle"                   # Legacy mapping
    CIRCULAR = "circular-dependency-node" # New forensic mapping
    ISOLATED = "isolated"
    EMPTY = "empty"
    STRESS = "inverted-pyramid-stress" # High load insumos


class SessionKeys:
    """Claves de sesión utilizadas."""
    PROCESSED_DATA = "processed_data"
    PRESUPUESTO = "presupuesto"
    APUS_DETAIL = "apus_detail"


# Configuración de truncado
LABEL_MAX_LENGTH = 20
LABEL_ELLIPSIS = "..."

# Separador de ciclos (debe coincidir con BusinessTopologicalAnalyzer)
CYCLE_SEPARATOR = " → "


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CytoscapeNode:
    """Representación de un nodo para Cytoscape.js."""
    id: str
    label: str
    node_type: str
    color: str
    level: int = 0
    cost: float = 0.0
    weight: float = 0.0
    is_evidence: bool = False
    classes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a formato esperado por Cytoscape.js."""
        return {
            "data": {
                "id": self.id,
                "label": self.label,
                "type": self.node_type,
                "color": self.color,
                "level": self.level,
                "cost": self.cost,
                "weight": self.weight,        # Forensic attribute
                "is_evidence": self.is_evidence # Forensic attribute
            },
            "classes": " ".join(self.classes)
        }


@dataclass
class CytoscapeEdge:
    """Representación de una arista para Cytoscape.js."""
    source: str
    target: str
    cost: float = 0.0
    is_evidence: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a formato esperado por Cytoscape.js."""
        data = {
            "source": self.source,
            "target": self.target,
            "cost": self.cost,
        }
        if self.is_evidence:
             data["is_evidence"] = True
        return {"data": data}


@dataclass
class AnomalyData:
    """Datos de anomalías extraídos del análisis."""
    isolated_ids: Set[str] = field(default_factory=set)
    orphan_ids: Set[str] = field(default_factory=set)
    empty_ids: Set[str] = field(default_factory=set)
    nodes_in_cycles: Set[str] = field(default_factory=set)
    stressed_ids: Set[str] = field(default_factory=set) # New: Inverted Pyramid Stress


# =============================================================================
# FUNCIONES DE VALIDACIÓN
# =============================================================================

def validate_session_data(data: Any) -> Tuple[bool, Optional[str]]:
    """Valida que los datos de sesión tengan la estructura esperada."""
    if data is None:
        return False, "Datos de sesión son None"

    if not isinstance(data, dict):
        return False, f"Se esperaba dict, se recibió {type(data).__name__}"

    has_presupuesto = SessionKeys.PRESUPUESTO in data
    has_apus = SessionKeys.APUS_DETAIL in data

    if not has_presupuesto and not has_apus:
        return False, "No se encontraron datos de presupuesto ni APUs"

    return True, None


def validate_graph(graph: Any) -> Tuple[bool, Optional[str]]:
    """Valida que el grafo sea un objeto NetworkX válido."""
    if graph is None:
        return False, "El grafo es None"

    if not isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        return False, f"Se esperaba grafo NetworkX, se recibió {type(graph).__name__}"

    return True, None


# =============================================================================
# FUNCIONES DE EXTRACCIÓN Y TRANSFORMACIÓN
# =============================================================================

def extract_dataframes_from_session(data: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extrae y convierte los datos de sesión a DataFrames."""
    presupuesto_list = data.get(SessionKeys.PRESUPUESTO, [])
    apus_list = data.get(SessionKeys.APUS_DETAIL, [])

    df_presupuesto = pd.DataFrame(presupuesto_list) if presupuesto_list else pd.DataFrame()
    df_apus_detail = pd.DataFrame(apus_list) if apus_list else pd.DataFrame()

    return df_presupuesto, df_apus_detail


def extract_anomaly_data(analysis_result: Dict[str, Any]) -> AnomalyData:
    """Extrae datos de anomalías del resultado del análisis de forma segura."""
    anomaly_data = AnomalyData()

    if not isinstance(analysis_result, dict):
        return anomaly_data

    details = analysis_result.get("details", {})

    # Extraer anomalías básicas
    anomalies = details.get("anomalies", {})
    if isinstance(anomalies, dict):
        anomaly_data.isolated_ids = _extract_ids_from_list(anomalies.get("isolated_nodes", []))
        anomaly_data.orphan_ids = _extract_ids_from_list(anomalies.get("orphan_insumos", []))
        anomaly_data.empty_ids = _extract_ids_from_list(anomalies.get("empty_apus", []))

    # Extraer nodos en ciclos
    cycles_data = details.get("cycles", {})
    if isinstance(cycles_data, dict):
        cycles_list = cycles_data.get("list", [])
        anomaly_data.nodes_in_cycles = _extract_nodes_from_cycles(cycles_list)

    return anomaly_data


def _identify_stressed_nodes(graph: nx.DiGraph) -> Set[str]:
    """
    Identifica nodos de Insumo bajo 'Estrés de Pirámide Invertida'.
    Criterio: Out-degree desproporcionadamente alto.

    Threshold:
    - Si APUs totales > 10: Insumo sostiene > 30% de los APUs.
    - Si APUs totales <= 10: Insumo sostiene > 50% de los APUs.
    """
    stressed = set()

    # Contar APUs totales
    apu_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "APU"]
    total_apus = len(apu_nodes)

    if total_apus == 0:
        return stressed

    threshold_ratio = 0.30 if total_apus > 10 else 0.50

    for node, data in graph.nodes(data=True):
        if data.get("type") == "INSUMO":
            out_degree = graph.out_degree(node)
            # Calcular ratio de soporte (cuántos APUs dependen de este insumo)
            # Nota: out_degree cuenta aristas, asumiendo 1 arista por APU dependiente.
            if out_degree / total_apus > threshold_ratio:
                stressed.add(str(node))

    return stressed


def _extract_ids_from_list(items: Any) -> Set[str]:
    """Extrae IDs de una lista de diccionarios de forma segura."""
    if not isinstance(items, list):
        return set()
    ids = set()
    for item in items:
        if isinstance(item, dict) and "id" in item:
            ids.add(str(item["id"]))
        elif isinstance(item, str):
            ids.add(item)
    return ids


def _extract_nodes_from_cycles(cycles_list: Any) -> Set[str]:
    """Extrae nodos únicos de la lista de ciclos."""
    if not isinstance(cycles_list, list):
        return set()
    nodes = set()
    for cycle_str in cycles_list:
        if not isinstance(cycle_str, str):
            continue
        parts = cycle_str.split(CYCLE_SEPARATOR)
        valid_parts = [p.strip() for p in parts if p.strip()]
        if len(valid_parts) > 1:
            if valid_parts[-1] == valid_parts[0]:
                valid_parts = valid_parts[:-1]
        nodes.update(valid_parts)
    return nodes


# =============================================================================
# FUNCIONES DE CONSTRUCCIÓN DE ELEMENTOS
# =============================================================================

def build_node_element(
    node_id: Any,
    attrs: Dict[str, Any],
    anomaly_data: AnomalyData
) -> CytoscapeNode:
    """Construye un elemento de nodo para Cytoscape.js con metadatos forenses."""
    node_id_str = str(node_id)
    node_type = _get_node_type(attrs)
    classes = _determine_node_classes(node_id_str, node_type, anomaly_data)
    color = _determine_node_color(node_id_str, node_type, anomaly_data)
    label = _build_node_label(node_id_str, attrs)
    cost = _get_node_cost(attrs)
    level = _safe_get_int(attrs, "level", 0)

    # Metadatos Forenses
    is_evidence = (
        node_id_str in anomaly_data.nodes_in_cycles or
        node_id_str in anomaly_data.stressed_ids
    )

    return CytoscapeNode(
        id=node_id_str,
        label=label,
        node_type=node_type,
        color=color,
        level=level,
        cost=cost,
        weight=cost, # Map cost to weight for forensic visualization
        is_evidence=is_evidence,
        classes=classes
    )


def build_edge_element(
    source: Any,
    target: Any,
    attrs: Dict[str, Any],
    anomaly_data: AnomalyData
) -> CytoscapeEdge:
    """Construye un elemento de arista con metadatos forenses."""
    cost = _safe_get_float(attrs, "total_cost", 0.0)
    src_str = str(source)
    tgt_str = str(target)

    # Forensic Edge Logic
    # Mark edge as evidence if it connects two nodes in the SAME cycle
    # (Simple heuristic, could be more robust if we tracked cycle paths specifically)
    is_evidence = (
        src_str in anomaly_data.nodes_in_cycles and
        tgt_str in anomaly_data.nodes_in_cycles
    )

    return CytoscapeEdge(
        source=src_str,
        target=tgt_str,
        cost=cost,
        is_evidence=is_evidence
    )


def _get_node_type(attrs: Dict[str, Any]) -> str:
    """Obtiene el tipo de nodo de forma segura."""
    node_type = attrs.get("type", NodeType.UNKNOWN.value)
    return node_type.upper() if isinstance(node_type, str) else NodeType.UNKNOWN.value


def _determine_node_classes(
    node_id: str,
    node_type: str,
    anomaly_data: AnomalyData
) -> List[str]:
    """Determina las clases CSS para un nodo basado en su estado forense."""
    classes = [node_type]

    # Prioridad Forense
    if node_id in anomaly_data.nodes_in_cycles:
        classes.append(NodeClass.CIRCULAR.value)
        classes.append(NodeClass.CYCLE.value) # Legacy compatibility

    if node_id in anomaly_data.stressed_ids:
        classes.append(NodeClass.STRESS.value)

    if node_id in anomaly_data.isolated_ids or node_id in anomaly_data.orphan_ids:
        classes.append(NodeClass.ISOLATED.value)

    if node_id in anomaly_data.empty_ids:
        classes.append(NodeClass.EMPTY.value)

    if len(classes) == 1: # Only type added
        classes.append(NodeClass.NORMAL.value)

    return classes


def _determine_node_color(
    node_id: str,
    node_type: str,
    anomaly_data: AnomalyData
) -> str:
    """Determina el color del nodo basado en riesgo."""
    # 1. Riesgo Crítico (Rojo)
    if node_id in anomaly_data.nodes_in_cycles:
        return NodeColor.RED.value
    if node_id in anomaly_data.stressed_ids:
        return NodeColor.RED.value
    if node_id in anomaly_data.isolated_ids or node_id in anomaly_data.orphan_ids:
        return NodeColor.RED.value

    # 2. Tipos Normales
    if node_type in [NodeType.BUDGET.value, NodeType.CHAPTER.value, NodeType.ITEM.value]:
        return NodeColor.BLACK.value
    elif node_type == NodeType.APU.value:
        return NodeColor.BLUE.value
    elif node_type == NodeType.INSUMO.value:
        return NodeColor.ORANGE.value

    return NodeColor.GRAY.value


def _build_node_label(node_id: str, attrs: Dict[str, Any]) -> str:
    """Construye la etiqueta del nodo con truncado."""
    description = attrs.get("description")
    if not description:
        return node_id
    desc_str = str(description).strip()
    if not desc_str:
        return node_id
    if len(desc_str) > LABEL_MAX_LENGTH:
        truncated = desc_str[:LABEL_MAX_LENGTH].strip() + LABEL_ELLIPSIS
    else:
        truncated = desc_str
    return f"{node_id}\n{truncated}"


def _get_node_cost(attrs: Dict[str, Any]) -> float:
    """Obtiene el costo del nodo."""
    total_cost = _safe_get_float(attrs, "total_cost", None)
    if total_cost is not None:
        return total_cost
    return _safe_get_float(attrs, "unit_cost", 0.0)


def _safe_get_float(data: Dict[str, Any], key: str, default: Optional[float]) -> Optional[float]:
    value = data.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_get_int(data: Dict[str, Any], key: str, default: int) -> int:
    value = data.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


# =============================================================================
# ENDPOINT PRINCIPAL
# =============================================================================

def build_graph_from_session(data: Dict[str, Any]) -> nx.DiGraph:
    """Construye el grafo de presupuesto desde los datos de sesión."""
    df_presupuesto, df_apus_detail = extract_dataframes_from_session(data)
    if df_presupuesto.empty and df_apus_detail.empty:
        raise ValueError("No hay datos suficientes para construir el grafo")
    builder = BudgetGraphBuilder()
    graph = builder.build(df_presupuesto, df_apus_detail)
    is_valid, error_msg = validate_graph(graph)
    if not is_valid:
        raise ValueError(f"Grafo inválido: {error_msg}")
    return graph


def analyze_graph_for_visualization(graph: nx.DiGraph) -> AnomalyData:
    """Analiza el grafo para obtener datos de visualización forense."""
    try:
        analyzer = BusinessTopologicalAnalyzer()
        analysis_result = analyzer.analyze_structural_integrity(graph)

        # 1. Base Analysis
        anomaly_data = extract_anomaly_data(analysis_result)

        # 2. Forensic Enrichment (Caja de Cristal)
        anomaly_data.stressed_ids = _identify_stressed_nodes(graph)

        return anomaly_data
    except Exception as e:
        logger.warning(f"Error en análisis topológico: {e}. Usando datos vacíos.")
        return AnomalyData()


def convert_graph_to_cytoscape_elements(
    graph: nx.DiGraph,
    anomaly_data: AnomalyData
) -> List[Dict[str, Any]]:
    """Convierte el grafo a formato Cytoscape con atributos forenses."""
    elements = []

    # Procesar nodos
    for node_id, attrs in graph.nodes(data=True):
        try:
            node_element = build_node_element(node_id, attrs or {}, anomaly_data)
            elements.append(node_element.to_dict())
        except Exception as e:
            logger.warning(f"Error procesando nodo '{node_id}': {e}")
            elements.append({
                "data": {"id": str(node_id), "label": str(node_id), "type": "UNKNOWN"},
                "classes": "UNKNOWN normal"
            })

    # Procesar aristas
    for source, target, attrs in graph.edges(data=True):
        try:
            edge_element = build_edge_element(source, target, attrs or {}, anomaly_data)
            elements.append(edge_element.to_dict())
        except Exception as e:
            logger.warning(f"Error procesando arista '{source}' -> '{target}': {e}")
            elements.append({
                "data": {"source": str(source), "target": str(target), "cost": 0}
            })

    return elements


def create_error_response(message: str, status_code: int) -> Tuple[Response, int]:
    return jsonify({
        "error": message,
        "elements": [],
        "success": False
    }), status_code


def create_success_response(elements: List[Dict[str, Any]]) -> Response:
    return jsonify({
        "elements": elements,
        "count": len(elements),
        "success": True
    })


@topology_bp.route('/api/visualization/project-graph', methods=['GET'])
@topology_bp.route('/api/visualization/topology', methods=['GET'])
def get_project_graph() -> Tuple[Response, int]:
    """Endpoint para obtener la estructura del grafo forense."""
    if SessionKeys.PROCESSED_DATA not in session:
        return create_error_response("No hay datos de sesión", 404)

    try:
        data = session[SessionKeys.PROCESSED_DATA]
        is_valid, error_msg = validate_session_data(data)
        if not is_valid:
            return create_error_response(f"Datos de sesión inválidos: {error_msg}", 400)

        graph = build_graph_from_session(data)
        if graph.number_of_nodes() == 0:
            return create_success_response([])

        anomaly_data = analyze_graph_for_visualization(graph)
        elements = convert_graph_to_cytoscape_elements(graph, anomaly_data)

        return create_success_response(elements), 200

    except Exception as e:
        logger.error(f"Error inesperado generando visualización: {e}", exc_info=True)
        return create_error_response(f"Error interno: {type(e).__name__}", 500)


@topology_bp.route('/api/visualization/topology/stats', methods=['GET'])
def get_topology_stats() -> Tuple[Response, int]:
    """Endpoint estadísticas del grafo."""
    if SessionKeys.PROCESSED_DATA not in session:
        return create_error_response("No hay datos de sesión", 404)

    try:
        data = session[SessionKeys.PROCESSED_DATA]
        is_valid, error_msg = validate_session_data(data)
        if not is_valid:
            return create_error_response(f"Datos inválidos: {error_msg}", 400)

        graph = build_graph_from_session(data)
        anomaly_data = analyze_graph_for_visualization(graph)

        stats = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "isolated_nodes": len(anomaly_data.isolated_ids),
            "orphan_nodes": len(anomaly_data.orphan_ids),
            "empty_apus": len(anomaly_data.empty_ids),
            "nodes_in_cycles": len(anomaly_data.nodes_in_cycles),
            "stressed_nodes": len(anomaly_data.stressed_ids), # Metric for Inverted Pyramid
            "success": True
        }
        return jsonify(stats), 200

    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}", exc_info=True)
        return create_error_response(str(e), 500)
