"""
Este módulo transforma el grafo topológico abstracto y los diagnósticos matemáticos
en una representación visual interactiva (Cytoscape.js). Su objetivo es hacer
tangible la "Física del Negocio", permitiendo a los usuarios humanos auditar
visualmente las patologías detectadas por el Consejo de Sabios.

Mapeo Semántico-Visual (La Leyenda del Mapa):
---------------------------------------------
1. Nodos de Riesgo (Color ROJO):
   Visualizan puntos de falla crítica identificados por la topología:
   - Ciclos (β1 > 0): "Socavones lógicos" o dependencias circulares.
   - Islas (β0 > 1): Recursos fragmentados o desconectados del proyecto.
   - Estrés (In-Degree alto): Insumos que soportan una carga estructural excesiva ("Pirámide Invertida").

2. Jerarquía Visual:
   Organiza los nodos en estratos (Proyecto -> Capítulo -> APU -> Insumo) para
   revelar la arquitectura de la información y facilitar la navegación "Drill-down".

3. Aristas de Evidencia:
   Resalta las conexiones específicas que forman parte de un ciclo o conflicto,
   diferenciándolas de las relaciones nominales para acelerar el análisis forense.

4. Enriquecimiento de Anomalías:
   Inyecta metadatos de diagnóstico (`AnomalyData`) directamente en los elementos
   visuales, convirtiendo el gráfico en una herramienta de depuración activa.
"""

import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd
from flask import Blueprint, Response, jsonify, session

# Asegurar que la raíz del proyecto esté en sys.path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from agent.business_topology import BudgetGraphBuilder, BusinessTopologicalAnalyzer

topology_bp = Blueprint("topology", __name__)
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

    RED = "#EF4444"  # Ciclos, Islas (Riesgo), Estrés
    BLUE = "#3B82F6"  # APU (Normal)
    ORANGE = "#F97316"  # Insumo (Recurso)
    BLACK = "#1E293B"  # Raíz / Presupuesto / Capítulos
    GRAY = "#9CA3AF"  # Desconocido / Vacío


class NodeClass(str, Enum):
    """Clases CSS para visualización de nodos."""

    NORMAL = "normal"
    CYCLE = "cycle"  # Legacy mapping
    CIRCULAR = "circular-dependency-node"  # New forensic mapping
    ISOLATED = "isolated"
    EMPTY = "empty"
    STRESS = "inverted-pyramid-stress"  # High load insumos


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
                "weight": self.weight,  # Forensic attribute
                "is_evidence": self.is_evidence,  # Forensic attribute
            },
            "classes": " ".join(self.classes),
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
    stressed_ids: Set[str] = field(default_factory=set)  # New: Inverted Pyramid Stress


# =============================================================================
# FUNCIONES DE VALIDACIÓN
# =============================================================================


def validate_session_data(data: Any) -> Tuple[bool, Optional[str]]:
    """
    Valida que los datos de sesión tengan la estructura esperada.

    Validaciones:
    - Tipo de dato principal (dict)
    - Existencia de claves requeridas
    - Tipo de datos internos (listas)
    - Contenido no vacío
    """
    if data is None:
        return False, "Datos de sesión son None"

    if not isinstance(data, dict):
        return False, f"Se esperaba dict, se recibió {type(data).__name__}"

    presupuesto = data.get(SessionKeys.PRESUPUESTO)
    apus = data.get(SessionKeys.APUS_DETAIL)

    has_presupuesto = presupuesto is not None
    has_apus = apus is not None

    if not has_presupuesto and not has_apus:
        return False, "No se encontraron datos de presupuesto ni APUs"

    # Validar que sean listas cuando existen
    if has_presupuesto and not isinstance(presupuesto, list):
        return (
            False,
            f"'presupuesto' debe ser lista, se recibió {type(presupuesto).__name__}",
        )

    if has_apus and not isinstance(apus, list):
        return False, f"'apus_detail' debe ser lista, se recibió {type(apus).__name__}"

    # Validar contenido no completamente vacío
    presupuesto_empty = not presupuesto if has_presupuesto else True
    apus_empty = not apus if has_apus else True

    if presupuesto_empty and apus_empty:
        return False, "Ambas listas de datos están vacías"

    return True, None


def validate_graph(graph: Any) -> Tuple[bool, Optional[str]]:
    """
    Valida que el grafo sea un objeto NetworkX válido y utilizable.

    Validaciones:
    - Tipo de objeto (grafo NetworkX)
    - Integridad estructural básica
    """
    if graph is None:
        return False, "El grafo es None"

    valid_types = (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)
    if not isinstance(graph, valid_types):
        return False, f"Se esperaba grafo NetworkX, se recibió {type(graph).__name__}"

    # Validar que el grafo no esté corrupto
    try:
        _ = graph.number_of_nodes()
        _ = graph.number_of_edges()
    except Exception as e:
        return False, f"Grafo corrupto o inaccesible: {e}"

    return True, None


# =============================================================================
# FUNCIONES DE EXTRACCIÓN Y TRANSFORMACIÓN
# =============================================================================


def extract_dataframes_from_session(
    data: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extrae y convierte los datos de sesión a DataFrames.

    Maneja casos de datos malformados o tipos incorrectos.
    """
    df_presupuesto = pd.DataFrame()
    df_apus_detail = pd.DataFrame()

    presupuesto_list = data.get(SessionKeys.PRESUPUESTO)
    apus_list = data.get(SessionKeys.APUS_DETAIL)

    # Procesar presupuesto con validación de tipo
    if presupuesto_list and isinstance(presupuesto_list, list):
        try:
            # Filtrar elementos que no sean diccionarios
            valid_items = [item for item in presupuesto_list if isinstance(item, dict)]
            if valid_items:
                df_presupuesto = pd.DataFrame(valid_items)
        except Exception as e:
            logger.warning(f"Error creando DataFrame de presupuesto: {e}")

    # Procesar APUs con validación de tipo
    if apus_list and isinstance(apus_list, list):
        try:
            valid_items = [item for item in apus_list if isinstance(item, dict)]
            if valid_items:
                df_apus_detail = pd.DataFrame(valid_items)
        except Exception as e:
            logger.warning(f"Error creando DataFrame de APUs: {e}")

    return df_presupuesto, df_apus_detail


def extract_anomaly_data(analysis_result: Dict[str, Any]) -> AnomalyData:
    """
    Extrae datos de anomalías del resultado del análisis de forma segura.

    Implementa extracción defensiva con logging de problemas.
    """
    anomaly_data = AnomalyData()

    if not isinstance(analysis_result, dict):
        logger.debug(f"analysis_result no es dict: {type(analysis_result).__name__}")
        return anomaly_data

    details = analysis_result.get("details")
    if not isinstance(details, dict):
        logger.debug("'details' no encontrado o no es dict")
        return anomaly_data

    # Extraer anomalías básicas con validación
    anomalies = details.get("anomalies")
    if isinstance(anomalies, dict):
        anomaly_data.isolated_ids = _extract_ids_from_list(
            anomalies.get("isolated_nodes", [])
        )
        anomaly_data.orphan_ids = _extract_ids_from_list(anomalies.get("orphan_insumos", []))
        anomaly_data.empty_ids = _extract_ids_from_list(anomalies.get("empty_apus", []))
    else:
        logger.debug("'anomalies' no es dict, se omiten anomalías básicas")

    # Extraer nodos en ciclos con validación
    cycles_data = details.get("cycles")
    if isinstance(cycles_data, dict):
        cycles_list = cycles_data.get("list", [])
        anomaly_data.nodes_in_cycles = _extract_nodes_from_cycles(cycles_list)
    else:
        logger.debug("'cycles' no es dict, se omiten datos de ciclos")

    return anomaly_data


def _identify_stressed_nodes(graph: nx.DiGraph) -> Set[str]:
    """
    Identifica nodos de Insumo bajo 'Estrés de Pirámide Invertida'.

    Un insumo está estresado cuando soporta una proporción alta de APUs,
    creando un punto de fallo centralizado.

    Threshold dinámico:
    - APUs > 10: Insumo sostiene > 30% de los APUs
    - APUs <= 10: Insumo sostiene > 50% de los APUs
    - Mínimo absoluto: al menos 2 conexiones para ser considerado estresado

    Nota: Usa in_degree asumiendo dirección APU -> INSUMO (dependencia).
          Si la dirección es inversa, ajustar a out_degree.
    """
    stressed: Set[str] = set()

    # Validar tipo de grafo
    if not isinstance(graph, nx.DiGraph):
        logger.warning("_identify_stressed_nodes requiere DiGraph para análisis direccional")
        return stressed

    # Contar APUs totales
    apu_nodes = [
        n
        for n, d in graph.nodes(data=True)
        if d.get("type", "").upper() == NodeType.APU.value
    ]
    total_apus = len(apu_nodes)

    if total_apus == 0:
        return stressed

    # Definir threshold dinámico
    threshold_ratio = 0.30 if total_apus > 10 else 0.50
    min_absolute_connections = 2  # Evitar falsos positivos en grafos pequeños

    for node, data in graph.nodes(data=True):
        node_type = data.get("type", "").upper()
        if node_type != NodeType.INSUMO.value:
            continue

        # Usar in_degree: cuenta cuántos nodos apuntan a este insumo
        # Alternativamente out_degree si la dirección es INSUMO -> APU
        connection_count = graph.in_degree(node)

        # Verificar umbral relativo Y absoluto
        ratio = connection_count / total_apus if total_apus > 0 else 0
        if ratio > threshold_ratio and connection_count >= min_absolute_connections:
            stressed.add(str(node))
            logger.debug(
                f"Nodo estresado: {node} (conexiones: {connection_count}, "
                f"ratio: {ratio:.2%}, threshold: {threshold_ratio:.0%})"
            )

    return stressed


def _extract_ids_from_list(items: Any) -> Set[str]:
    """
    Extrae IDs de una lista de elementos de forma segura.

    Soporta:
    - Lista de diccionarios con clave 'id'
    - Lista de strings directos
    - Lista mixta de ambos tipos
    """
    if not isinstance(items, list):
        return set()

    ids: Set[str] = set()
    for item in items:
        extracted_id = None

        if isinstance(item, dict):
            raw_id = item.get("id")
            if raw_id is not None:
                extracted_id = str(raw_id).strip()
        elif isinstance(item, (str, int)):
            extracted_id = str(item).strip()

        if extracted_id:  # No agregar strings vacíos
            ids.add(extracted_id)

    return ids


def _extract_nodes_from_cycles(cycles_list: Any) -> Set[str]:
    """
    Extrae nodos únicos de la lista de ciclos.

    Formato esperado de ciclo: "A → B → C → A"
    El último nodo (si es igual al primero) se elimina para evitar duplicados.
    """
    if not isinstance(cycles_list, list):
        return set()

    nodes: Set[str] = set()

    for cycle_str in cycles_list:
        if not isinstance(cycle_str, str):
            logger.debug(f"Elemento de ciclo no es string: {type(cycle_str).__name__}")
            continue

        if not cycle_str.strip():
            continue

        # Separar y limpiar partes
        parts = cycle_str.split(CYCLE_SEPARATOR)
        valid_parts = [p.strip() for p in parts if p.strip()]

        if len(valid_parts) < 2:
            logger.debug(f"Ciclo inválido (menos de 2 nodos): {cycle_str}")
            continue

        # Remover duplicado de cierre si existe
        if valid_parts[-1] == valid_parts[0]:
            valid_parts = valid_parts[:-1]

        nodes.update(valid_parts)

    return nodes


# =============================================================================
# FUNCIONES DE CONSTRUCCIÓN DE ELEMENTOS
# =============================================================================


def build_node_element(
    node_id: Any, attrs: Dict[str, Any], anomaly_data: AnomalyData
) -> CytoscapeNode:
    """
    Construye un elemento de nodo para Cytoscape.js con metadatos forenses.

    Garantiza que todos los campos tengan valores válidos incluso con datos incompletos.
    """
    # Normalizar ID a string no vacío
    node_id_str = str(node_id).strip() if node_id is not None else "unknown"
    if not node_id_str:
        node_id_str = "unknown"

    # Garantizar attrs como dict
    safe_attrs = attrs if isinstance(attrs, dict) else {}

    node_type = _get_node_type(safe_attrs)
    classes = _determine_node_classes(node_id_str, node_type, anomaly_data)
    color = _determine_node_color(node_id_str, node_type, anomaly_data)
    label = _build_node_label(node_id_str, safe_attrs)
    cost = _get_node_cost(safe_attrs)
    level = _safe_get_int(safe_attrs, "level", 0)

    # Determinar si es evidencia forense
    is_evidence = (
        node_id_str in anomaly_data.nodes_in_cycles
        or node_id_str in anomaly_data.stressed_ids
        or node_id_str in anomaly_data.isolated_ids
        or node_id_str in anomaly_data.orphan_ids
    )

    return CytoscapeNode(
        id=node_id_str,
        label=label,
        node_type=node_type,
        color=color,
        level=level,
        cost=cost,
        weight=cost,
        is_evidence=is_evidence,
        classes=classes,
    )


def build_edge_element(
    source: Any, target: Any, attrs: Dict[str, Any], anomaly_data: AnomalyData
) -> CytoscapeEdge:
    """
    Construye un elemento de arista con metadatos forenses.

    Marca aristas como evidencia si conectan nodos en el mismo ciclo.
    """
    safe_attrs = attrs if isinstance(attrs, dict) else {}

    # Normalizar IDs
    src_str = str(source).strip() if source is not None else ""
    tgt_str = str(target).strip() if target is not None else ""

    if not src_str or not tgt_str:
        raise ValueError(f"Arista inválida: source='{source}', target='{target}'")

    cost = _safe_get_float(safe_attrs, "total_cost", 0.0)

    # Determinar si es arista de evidencia forense
    # Una arista es evidencia si ambos extremos están en ciclos
    src_in_cycle = src_str in anomaly_data.nodes_in_cycles
    tgt_in_cycle = tgt_str in anomaly_data.nodes_in_cycles
    is_evidence = src_in_cycle and tgt_in_cycle

    return CytoscapeEdge(
        source=src_str,
        target=tgt_str,
        cost=cost if cost is not None else 0.0,
        is_evidence=is_evidence,
    )


def _get_node_type(attrs: Dict[str, Any]) -> str:
    """
    Obtiene el tipo de nodo de forma segura y normalizada.
    """
    node_type = attrs.get("type")

    if node_type is None:
        return NodeType.UNKNOWN.value

    if not isinstance(node_type, str):
        try:
            node_type = str(node_type)
        except Exception:
            return NodeType.UNKNOWN.value

    normalized = node_type.upper().strip()

    # Validar contra tipos conocidos
    valid_types = {nt.value for nt in NodeType}
    return normalized if normalized in valid_types else NodeType.UNKNOWN.value


def _determine_node_classes(
    node_id: str, node_type: str, anomaly_data: AnomalyData
) -> List[str]:
    """
    Determina las clases CSS para un nodo basado en su estado forense.

    Orden de prioridad (acumulativo):
    1. Tipo base
    2. Estado de ciclo (crítico)
    3. Estado de estrés
    4. Estado de aislamiento
    5. Estado vacío
    6. Normal (si no hay anomalías)
    """
    classes: List[str] = [node_type]
    has_anomaly = False

    # Ciclos - Prioridad máxima
    if node_id in anomaly_data.nodes_in_cycles:
        classes.append(NodeClass.CIRCULAR.value)
        classes.append(NodeClass.CYCLE.value)  # Legacy
        has_anomaly = True

    # Estrés de pirámide invertida
    if node_id in anomaly_data.stressed_ids:
        classes.append(NodeClass.STRESS.value)
        has_anomaly = True

    # Aislamiento (islas o huérfanos)
    if node_id in anomaly_data.isolated_ids or node_id in anomaly_data.orphan_ids:
        classes.append(NodeClass.ISOLATED.value)
        has_anomaly = True

    # APUs vacíos
    if node_id in anomaly_data.empty_ids:
        classes.append(NodeClass.EMPTY.value)
        has_anomaly = True

    # Si no hay anomalías, marcar como normal
    if not has_anomaly:
        classes.append(NodeClass.NORMAL.value)

    return classes


def _determine_node_color(node_id: str, node_type: str, anomaly_data: AnomalyData) -> str:
    """
    Determina el color del nodo basado en jerarquía de riesgo.

    Jerarquía (mayor a menor prioridad):
    1. Rojo: Ciclos, Estrés, Aislamiento (riesgo crítico)
    2. Negro: Budget, Chapter, Item (estructura)
    3. Azul: APU (análisis)
    4. Naranja: Insumo (recurso)
    5. Gris: Desconocido
    """
    # 1. Verificar riesgos críticos primero
    is_in_cycle = node_id in anomaly_data.nodes_in_cycles
    is_stressed = node_id in anomaly_data.stressed_ids
    is_isolated = node_id in anomaly_data.isolated_ids
    is_orphan = node_id in anomaly_data.orphan_ids

    if is_in_cycle or is_stressed or is_isolated or is_orphan:
        return NodeColor.RED.value

    # 2. Colores por tipo jerárquico
    structural_types = {NodeType.BUDGET.value, NodeType.CHAPTER.value, NodeType.ITEM.value}
    if node_type in structural_types:
        return NodeColor.BLACK.value

    if node_type == NodeType.APU.value:
        return NodeColor.BLUE.value

    if node_type == NodeType.INSUMO.value:
        return NodeColor.ORANGE.value

    # 3. Tipo desconocido
    return NodeColor.GRAY.value


def _build_node_label(node_id: str, attrs: Dict[str, Any]) -> str:
    """
    Construye la etiqueta del nodo con truncado inteligente.

    Formato: "ID\nDescripción truncada..."
    Si no hay descripción válida, retorna solo el ID.
    """
    description = attrs.get("description")

    # Validar descripción
    if description is None:
        return node_id

    # Convertir a string de forma segura
    try:
        desc_str = str(description).strip()
    except Exception:
        return node_id

    if not desc_str:
        return node_id

    # Aplicar truncado si es necesario
    if len(desc_str) > LABEL_MAX_LENGTH:
        # Truncar en límite de palabra si es posible
        truncated = desc_str[:LABEL_MAX_LENGTH]
        last_space = truncated.rfind(" ")
        if last_space > LABEL_MAX_LENGTH // 2:  # Solo si no perdemos mucho texto
            truncated = truncated[:last_space]
        truncated = truncated.rstrip() + LABEL_ELLIPSIS
    else:
        truncated = desc_str

    return f"{node_id}\n{truncated}"


def _get_node_cost(attrs: Dict[str, Any]) -> float:
    """
    Obtiene el costo del nodo con fallback jerárquico.

    Prioridad: total_cost > unit_cost > 0.0
    """
    total_cost = _safe_get_float(attrs, "total_cost", None)
    if total_cost is not None and total_cost >= 0:
        return total_cost

    unit_cost = _safe_get_float(attrs, "unit_cost", None)
    if unit_cost is not None and unit_cost >= 0:
        return unit_cost

    return 0.0


def _safe_get_float(
    data: Dict[str, Any], key: str, default: Optional[float]
) -> Optional[float]:
    """
    Extrae un valor float de forma segura, manejando múltiples tipos de entrada.
    """
    if not isinstance(data, dict):
        return default

    value = data.get(key)
    if value is None:
        return default

    # Ya es float
    if isinstance(value, float):
        return value if not (value != value) else default  # Check NaN

    # Es int
    if isinstance(value, int):
        return float(value)

    # Intentar conversión
    try:
        result = float(value)
        return result if not (result != result) else default  # Check NaN
    except (ValueError, TypeError):
        return default


def _safe_get_int(data: Dict[str, Any], key: str, default: int) -> int:
    """
    Extrae un valor int de forma segura.
    """
    if not isinstance(data, dict):
        return default

    value = data.get(key)
    if value is None:
        return default

    if isinstance(value, int) and not isinstance(value, bool):
        return value

    if isinstance(value, float):
        return int(value) if not (value != value) else default  # Check NaN

    try:
        return int(float(value))  # Maneja strings como "3.0"
    except (ValueError, TypeError):
        return default


# =============================================================================
# ENDPOINT PRINCIPAL
# =============================================================================


def build_graph_from_session(data: Dict[str, Any]) -> nx.DiGraph:
    """
    Construye el grafo de presupuesto desde los datos de sesión.

    Raises:
        ValueError: Si los datos son insuficientes o el grafo resultante es inválido.
    """
    df_presupuesto, df_apus_detail = extract_dataframes_from_session(data)

    if df_presupuesto.empty and df_apus_detail.empty:
        raise ValueError("No hay datos suficientes para construir el grafo")

    try:
        builder = BudgetGraphBuilder()
        graph = builder.build(df_presupuesto, df_apus_detail)
    except Exception as e:
        logger.error(f"Error en BudgetGraphBuilder.build(): {e}")
        raise ValueError(f"Error construyendo grafo: {e}") from e

    is_valid, error_msg = validate_graph(graph)
    if not is_valid:
        raise ValueError(f"Grafo inválido: {error_msg}")

    return graph


def analyze_graph_for_visualization(graph: nx.DiGraph) -> AnomalyData:
    """
    Analiza el grafo para obtener datos de visualización forense.

    Combina análisis estructural básico con enriquecimiento forense (Caja de Cristal).
    Retorna AnomalyData vacío si hay errores, permitiendo visualización degradada.
    """
    anomaly_data = AnomalyData()

    # 1. Análisis estructural base
    try:
        analyzer = BusinessTopologicalAnalyzer()
        analysis_result = analyzer.analyze_structural_integrity(graph)
        anomaly_data = extract_anomaly_data(analysis_result)
    except Exception as e:
        logger.warning(f"Error en análisis topológico base: {e}")

    # 2. Enriquecimiento forense (independiente del paso anterior)
    try:
        anomaly_data.stressed_ids = _identify_stressed_nodes(graph)
    except Exception as e:
        logger.warning(f"Error identificando nodos estresados: {e}")
        # stressed_ids permanece vacío por defecto

    return anomaly_data


def convert_graph_to_cytoscape_elements(
    graph: nx.DiGraph, anomaly_data: AnomalyData
) -> List[Dict[str, Any]]:
    """
    Convierte el grafo a formato Cytoscape con atributos forenses.

    Garantiza que todos los nodos y aristas se procesen, usando fallbacks
    para elementos problemáticos.
    """
    elements: List[Dict[str, Any]] = []
    node_ids_processed: Set[str] = set()

    # Procesar nodos
    for node_id, attrs in graph.nodes(data=True):
        node_id_str = str(node_id)
        try:
            node_element = build_node_element(node_id, attrs or {}, anomaly_data)
            elements.append(node_element.to_dict())
            node_ids_processed.add(node_id_str)
        except Exception as e:
            logger.warning(f"Error procesando nodo '{node_id}': {e}")
            # Fallback: nodo minimal
            elements.append(
                {
                    "data": {
                        "id": node_id_str,
                        "label": node_id_str,
                        "type": NodeType.UNKNOWN.value,
                        "color": NodeColor.GRAY.value,
                        "level": 0,
                        "cost": 0.0,
                        "weight": 0.0,
                        "is_evidence": False,
                    },
                    "classes": f"{NodeType.UNKNOWN.value} {NodeClass.NORMAL.value}",
                }
            )
            node_ids_processed.add(node_id_str)

    # Procesar aristas (solo si ambos nodos existen)
    for source, target, attrs in graph.edges(data=True):
        src_str = str(source)
        tgt_str = str(target)

        # Validar que ambos nodos fueron procesados
        if src_str not in node_ids_processed or tgt_str not in node_ids_processed:
            logger.warning(f"Arista huérfana ignorada: '{src_str}' -> '{tgt_str}'")
            continue

        try:
            edge_element = build_edge_element(source, target, attrs or {}, anomaly_data)
            elements.append(edge_element.to_dict())
        except Exception as e:
            logger.warning(f"Error procesando arista '{src_str}' -> '{tgt_str}': {e}")
            # Fallback: arista minimal
            elements.append({"data": {"source": src_str, "target": tgt_str, "cost": 0.0}})

    return elements


def create_error_response(message: str, status_code: int) -> Tuple[Response, int]:
    """Crea respuesta de error estandarizada."""
    return jsonify(
        {"error": message, "elements": [], "count": 0, "success": False}
    ), status_code


def create_success_response(
    elements: List[Dict[str, Any]], status_code: int = 200
) -> Tuple[Response, int]:
    """Crea respuesta exitosa estandarizada."""
    return jsonify(
        {"elements": elements, "count": len(elements), "success": True, "error": None}
    ), status_code


@topology_bp.route("/api/visualization/project-graph", methods=["GET"])
@topology_bp.route("/api/visualization/topology", methods=["GET"])
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

        return create_success_response(elements)

    except ValueError as e:
        logger.warning(f"Error de validación: {e}")
        return create_error_response(str(e), 400)

    except Exception as e:
        logger.error(f"Error inesperado generando visualización: {e}", exc_info=True)
        return create_error_response(f"Error interno: {type(e).__name__}", 500)


@topology_bp.route("/api/visualization/topology/stats", methods=["GET"])
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
            "stressed_nodes": len(anomaly_data.stressed_ids),  # Metric for Inverted Pyramid
            "success": True,
        }
        return jsonify(stats), 200

    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}", exc_info=True)
        return create_error_response(str(e), 500)
