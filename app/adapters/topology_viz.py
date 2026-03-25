"""
=========================================================================================
Módulo: Topology Visualization (Operador de Inmersión Isomórfica y Semántica Cromática)
Ubicación: app/adapters/topology_viz.py
=========================================================================================

Naturaleza Ciber-Física y Topológica:
    Este módulo rechaza el rol de un simple "adaptador de frontend". Opera como el
    Operador de Inmersión (Embedding) que proyecta el Complejo Simplicial Abstracto 
    del presupuesto (espacio métrico de alta dimensionalidad) hacia una Variedad de 
    Observabilidad bidimensional (Cytoscape.js). Su mandato axiomático es hacer 
    tangible a la intuición humana el estrés termodinámico y las patologías 
    homológicas sin degradar la precisión del tensor original.

1. Inmersión Topológica y Preservación de Invariantes:
    Mapea el 1-esqueleto del grafo de negocio G = (V, E) a elementos visuales 
    preservando estrictamente la homotopía. Las anomalías matemáticas 
    (ej. Socavones Lógicos donde β₁ > 0, o Islas de Datos donde β₀ > 1) se 
    traducen isomórficamente en tensores cromáticos y espaciales inmutables, 
    permitiendo a la gerencia "ver" la matemática del riesgo.

2. Semántica Cromática y Termodinámica Visual (StressConfig):
    Los nodos no se colorean por heurísticas estéticas, sino por el nivel de "estrés" 
    termodinámico. El módulo detecta asimetrías de inercia (SPOF o "Pirámides Invertidas", 
    donde el índice Ψ < 1.0) e inyecta perturbaciones visuales (como coloración roja 
    pulsante #EF4444) para representar la concentración anómala de energía logística y 
    la inminencia de un colapso estructural.

3. Cirugía Topológica y Degradación Segura (Tolerancia a Defectos):
    Abandona la interrupción binaria (crash) por un modelo de Cirugía Topológica. 
    Si un nodo o arista (σ_i) presenta degeneración numérica durante la serialización 
    (ej. inyección de singularidades NaN), el operador ejecuta un "fallback" aislando 
    el defecto local. Esto preserva la conectividad global del colector (variedad) y 
    previene que un defecto sintáctico colapse la observabilidad del ecosistema completo.

4. Auditoría Forense y Trazabilidad Tensorial (Caja de Cristal):
    Los metadatos presentados al usuario no son meros "tooltips"; son la proyección 
    explícita de la Cadena de Custodia. Cada elemento visual encapsula su derivación 
    directa del Laplaciano Combinatorio, los scores de anomalía y el costo acotado, 
    permitiendo al Estrato Ejecutivo auditar las Actas de Deliberación del Consejo 
    de Sabios directamente sobre la geometría del problema.
=========================================================================================
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd
from flask import Blueprint, Response, jsonify, request, session

from app.tactics.business_topology import (
    BudgetGraphBuilder,
    BusinessTopologicalAnalyzer,
)

topology_bp = Blueprint("topology", __name__)
logger = logging.getLogger(__name__)


# ======================================================================
# CONSTANTES Y ENUMS
# ======================================================================


class NodeType(str, Enum):
    """Tipos semánticos de nodos en el grafo de negocio."""

    BUDGET = "BUDGET"
    CHAPTER = "CHAPTER"
    ITEM = "ITEM"
    APU = "APU"
    INSUMO = "INSUMO"
    UNKNOWN = "UNKNOWN"


class NodeColor(str, Enum):
    """Paleta de colores para codificación visual de nodos."""

    RED = "#EF4444"
    BLUE = "#3B82F6"
    ORANGE = "#F97316"
    BLACK = "#1E293B"
    GRAY = "#9CA3AF"


class NodeClass(str, Enum):
    """Clases CSS para Cytoscape.js que codifican estado semántico."""

    NORMAL = "normal"
    CYCLE = "cycle"
    CIRCULAR = "circular-dependency-node"
    ISOLATED = "isolated"
    EMPTY = "empty"
    STRESS = "inverted-pyramid-stress"
    HOT = "thermal-hotspot"
    ANOMALOUS = "anomalous"


class SessionKeys:
    """Claves canónicas para acceso a datos de sesión."""

    PROCESSED_DATA = "processed_data"
    PRESUPUESTO = "presupuesto"
    APUS_DETAIL = "apus_detail"


LABEL_MAX_LENGTH: int = 20
LABEL_ELLIPSIS: str = "..."
CYCLE_SEPARATOR: str = " -> "
ALLOWED_STRATUM_FILTERS: frozenset[int] = frozenset({0, 1, 2, 3})

# Convención declarativa de visibilidad por estrato.
#
# Cada estrato define qué niveles jerárquicos son visibles:
#   3 (PHYSICS):  solo nivel 3 (insumos)
#   2 (TACTICS):  niveles 2–3 (APUs + insumos)
#   1 (STRATEGY): niveles 1–2 (items + APUs)
#   0 (WISDOM):   niveles 0–1 (presupuesto + capítulos)
#
# La no-acumulatividad es intencional: cada estrato muestra una
# "ventana" de dos niveles adyacentes para mantener contexto
# sin sobrecargar la visualización.
STRATUM_VISIBLE_LEVELS: Dict[int, frozenset[int]] = {
    3: frozenset({3}),
    2: frozenset({2, 3}),
    1: frozenset({1, 2}),
    0: frozenset({0, 1}),
}

# Conjunto canónico de tipos válidos (calculado una vez).
_VALID_NODE_TYPES: frozenset[str] = frozenset(
    member.value for member in NodeType
)


# ======================================================================
# CONFIGURACIÓN DE ESTRÉS TOPOLÓGICO
# ======================================================================


@dataclass(frozen=True)
class StressConfig:
    """
    Configuración para detección de nodos estresados.

    Un insumo se considera estresado si:
        in_degree(insumo) / total_apus > threshold_ratio
        AND in_degree(insumo) >= min_absolute_connections

    El ``threshold_ratio`` se ajusta dinámicamente:
        - Grafos grandes (> ``large_graph_threshold`` APUs): usa ``ratio_large``
        - Grafos pequeños: usa ``ratio_small``
    """

    large_graph_threshold: int = 10
    ratio_large: float = 0.30
    ratio_small: float = 0.50
    min_absolute_connections: int = 2

    def __post_init__(self) -> None:
        if self.large_graph_threshold < 1:
            raise ValueError(
                f"large_graph_threshold debe ser ≥ 1; "
                f"recibido={self.large_graph_threshold}"
            )
        if not (0.0 < self.ratio_large <= 1.0):
            raise ValueError(
                f"ratio_large debe estar en (0, 1]; recibido={self.ratio_large}"
            )
        if not (0.0 < self.ratio_small <= 1.0):
            raise ValueError(
                f"ratio_small debe estar en (0, 1]; recibido={self.ratio_small}"
            )
        if self.min_absolute_connections < 1:
            raise ValueError(
                f"min_absolute_connections debe ser ≥ 1; "
                f"recibido={self.min_absolute_connections}"
            )

    def get_threshold_ratio(self, total_apus: int) -> float:
        """Retorna el umbral de ratio según el tamaño del grafo."""
        return (
            self.ratio_large
            if total_apus > self.large_graph_threshold
            else self.ratio_small
        )


# Instancia por defecto (inmutable).
_DEFAULT_STRESS_CONFIG = StressConfig()


# ======================================================================
# DATA CLASSES
# ======================================================================


@dataclass(frozen=True)
class CytoscapeNode:
    """Representación inmutable de un nodo Cytoscape.js."""

    id: str
    label: str
    node_type: str
    color: str
    level: int = 0
    cost: float = 0.0
    weight: float = 0.0
    score: float = 0.0
    is_evidence: bool = False
    tooltip: str = ""
    classes: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a formato Cytoscape.js."""
        return {
            "data": {
                "id": self.id,
                "label": self.label,
                "type": self.node_type,
                "color": self.color,
                "level": self.level,
                "cost": self.cost,
                "weight": self.weight,
                "score": self.score,
                "is_evidence": self.is_evidence,
                "tooltip": self.tooltip,
            },
            "classes": " ".join(self.classes),
        }


@dataclass(frozen=True)
class CytoscapeEdge:
    """Representación inmutable de una arista Cytoscape.js."""

    source: str
    target: str
    cost: float = 0.0
    score: float = 0.0
    is_evidence: bool = False
    tooltip: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a formato Cytoscape.js."""
        return {
            "data": {
                "source": self.source,
                "target": self.target,
                "cost": self.cost,
                "score": self.score,
                "is_evidence": self.is_evidence,
                "tooltip": self.tooltip,
            }
        }


@dataclass(frozen=True)
class AnomalyData:
    """
    Contenedor inmutable de datos de anomalía extraídos del análisis topológico.

    La inmutabilidad garantiza que los datos de anomalía no se modifiquen
    accidentalmente durante la construcción de elementos Cytoscape.
    """

    isolated_ids: frozenset[str] = field(default_factory=frozenset)
    orphan_ids: frozenset[str] = field(default_factory=frozenset)
    empty_ids: frozenset[str] = field(default_factory=frozenset)
    nodes_in_cycles: frozenset[str] = field(default_factory=frozenset)
    stressed_ids: frozenset[str] = field(default_factory=frozenset)
    hot_ids: frozenset[str] = field(default_factory=frozenset)

    anomalous_nodes: frozenset[str] = field(default_factory=frozenset)
    node_scores: Dict[str, float] = field(default_factory=dict)
    anomalous_edges: frozenset[Tuple[str, str]] = field(
        default_factory=frozenset
    )
    edge_scores: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def is_node_anomalous(self, node_id: str) -> bool:
        """Verifica si un nodo tiene cualquier tipo de anomalía."""
        return (
            node_id in self.nodes_in_cycles
            or node_id in self.stressed_ids
            or node_id in self.isolated_ids
            or node_id in self.orphan_ids
            or node_id in self.empty_ids
            or node_id in self.hot_ids
            or node_id in self.anomalous_nodes
        )


@dataclass(frozen=True)
class ValidationOutcome:
    """Resultado de una operación de validación."""

    ok: bool
    message: Optional[str] = None


# ======================================================================
# HELPERS DE NORMALIZACIÓN
# ======================================================================


def _normalize_identifier(value: Any, default: str = "unknown") -> str:
    """
    Normaliza un valor arbitrario a un identificador string limpio.

    Retorna ``default`` si el valor es ``None``, no convertible, o vacío
    tras strip.
    """
    if value is None:
        return default
    try:
        text = str(value).strip()
    except (TypeError, ValueError, AttributeError):
        return default
    return text if text else default


def _normalize_edge_tuple(source: Any, target: Any) -> Tuple[str, str]:
    """Normaliza un par (source, target) a tupla de identificadores."""
    return (_normalize_identifier(source), _normalize_identifier(target))


def _safe_finite_float(value: Any, default: float = 0.0) -> float:
    """
    Convierte a float finito. Retorna ``default`` si la conversión falla
    o el resultado no es finito.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _safe_nonnegative_finite_float(
    value: Any, default: float = 0.0
) -> float:
    """
    Convierte a float finito no negativo.

    Retorna ``default`` si la conversión falla, el resultado no es finito,
    o es negativo.
    """
    result = _safe_finite_float(value, default=default)
    return result if result >= 0.0 else default


def _safe_int_from_any(value: Any, default: int = 0) -> int:
    """
    Convierte un valor arbitrario a int de forma segura.

    Excluye ``bool`` para evitar que ``True``/``False`` se interpreten
    como ``1``/``0`` silenciosamente.
    """
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _truncate_text(text: str, max_length: int = LABEL_MAX_LENGTH) -> str:
    """
    Trunca texto a ``max_length`` caracteres con ellipsis.

    Intenta cortar en el último espacio para preservar legibilidad,
    siempre que el espacio esté en la segunda mitad del texto truncado.

    Retorna cadena vacía si el texto resultante tras rstrip es vacío.
    """
    if not text or not text.strip():
        return ""

    if len(text) <= max_length:
        return text

    truncated = text[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length // 2:
        truncated = truncated[:last_space]

    result = truncated.rstrip()
    return f"{result}{LABEL_ELLIPSIS}" if result else ""


def _deduplicate_preserve_order(items: Iterable[str]) -> Tuple[str, ...]:
    """
    Deduplica preservando orden de primera aparición.

    Complejidad: O(n) tiempo y espacio.
    """
    seen: Set[str] = set()
    result: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return tuple(result)


def _get_visible_levels(
    stratum_filter: Optional[int],
) -> Optional[frozenset[int]]:
    """
    Retorna los niveles visibles para un filtro de estrato dado.

    Retorna ``None`` si no hay filtro (todos los niveles visibles)
    o si el filtro no está en el dominio permitido.
    """
    if stratum_filter is None:
        return None
    if stratum_filter not in ALLOWED_STRATUM_FILTERS:
        return None
    return STRATUM_VISIBLE_LEVELS.get(stratum_filter)


# ======================================================================
# VALIDACIÓN
# ======================================================================


def validate_session_data(data: Any) -> ValidationOutcome:
    """
    Valida estructura de datos de sesión para construcción de grafo.

    Requiere que ``data`` sea un dict con al menos una de las claves
    ``presupuesto`` o ``apus_detail`` como lista no vacía.
    """
    if data is None:
        return ValidationOutcome(False, "Los datos de sesión son None.")
    if not isinstance(data, dict):
        return ValidationOutcome(
            False,
            f"Se esperaba dict, se recibió {type(data).__name__}.",
        )

    presupuesto = data.get(SessionKeys.PRESUPUESTO)
    apus = data.get(SessionKeys.APUS_DETAIL)

    has_presupuesto = presupuesto is not None
    has_apus = apus is not None

    if not has_presupuesto and not has_apus:
        return ValidationOutcome(
            False,
            "No se encontraron datos de presupuesto ni de APUs.",
        )

    if has_presupuesto and not isinstance(presupuesto, list):
        return ValidationOutcome(
            False,
            f"'presupuesto' debe ser lista, se recibió "
            f"{type(presupuesto).__name__}.",
        )
    if has_apus and not isinstance(apus, list):
        return ValidationOutcome(
            False,
            f"'apus_detail' debe ser lista, se recibió "
            f"{type(apus).__name__}.",
        )

    presupuesto_empty = (not presupuesto) if has_presupuesto else True
    apus_empty = (not apus) if has_apus else True

    if presupuesto_empty and apus_empty:
        return ValidationOutcome(
            False, "Las listas de presupuesto y APUs están vacías."
        )

    return ValidationOutcome(True, None)


def validate_graph(graph: Any) -> ValidationOutcome:
    """
    Valida que el grafo sea una instancia NetworkX accesible y no corrupta.

    Realiza verificación activa de accesibilidad (lectura de nodos y aristas).
    """
    if graph is None:
        return ValidationOutcome(False, "El grafo es None.")

    valid_types = (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)
    if not isinstance(graph, valid_types):
        return ValidationOutcome(
            False,
            f"Se esperaba grafo NetworkX, se recibió "
            f"{type(graph).__name__}.",
        )

    try:
        _ = graph.number_of_nodes()
        _ = graph.number_of_edges()
        _ = list(graph.nodes(data=True))
        _ = list(graph.edges(data=True))
    except Exception as exc:
        return ValidationOutcome(
            False, f"Grafo corrupto o inaccesible: {exc}"
        )

    return ValidationOutcome(True, None)


def validate_stratum_filter(
    stratum_filter: Optional[int],
) -> ValidationOutcome:
    """Valida que el filtro de estrato esté en el dominio permitido."""
    if stratum_filter is None:
        return ValidationOutcome(True, None)
    if not isinstance(stratum_filter, int) or isinstance(
        stratum_filter, bool
    ):
        return ValidationOutcome(
            False,
            f"stratum_filter debe ser int o None, recibido "
            f"{type(stratum_filter).__name__}.",
        )
    if stratum_filter not in ALLOWED_STRATUM_FILTERS:
        return ValidationOutcome(
            False,
            f"stratum_filter fuera de dominio permitido: "
            f"{stratum_filter}.",
        )
    return ValidationOutcome(True, None)


# ======================================================================
# EXTRACCIÓN DE DATOS
# ======================================================================


def _extract_dataframe_from_list(
    items: Any, source_name: str
) -> pd.DataFrame:
    """
    Extrae un DataFrame de una lista de diccionarios con validación.

    Filtra elementos no-dict y registra descartados.
    """
    if not isinstance(items, list):
        return pd.DataFrame()

    valid_items = [item for item in items if isinstance(item, dict)]
    discarded = len(items) - len(valid_items)

    if discarded > 0:
        logger.warning(
            "Se descartaron %d registros no dict en '%s'.",
            discarded,
            source_name,
        )

    if not valid_items:
        return pd.DataFrame()

    try:
        return pd.DataFrame(valid_items)
    except Exception as exc:
        logger.warning(
            "Error creando DataFrame de %s: %s", source_name, exc
        )
        return pd.DataFrame()


def extract_dataframes_from_session(
    data: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extrae DataFrames de presupuesto y APUs desde datos de sesión.

    Retorna tupla ``(df_presupuesto, df_apus_detail)``, cada uno
    puede ser un DataFrame vacío si los datos no están disponibles.
    """
    df_presupuesto = _extract_dataframe_from_list(
        data.get(SessionKeys.PRESUPUESTO), "presupuesto"
    )
    df_apus_detail = _extract_dataframe_from_list(
        data.get(SessionKeys.APUS_DETAIL), "apus_detail"
    )
    return df_presupuesto, df_apus_detail


def extract_anomaly_data(
    analysis_result: Dict[str, Any],
) -> AnomalyData:
    """
    Extrae y normaliza datos de anomalía desde el resultado del análisis
    topológico.

    Maneja defensivamente estructuras parciales o ausentes.
    """
    if not isinstance(analysis_result, dict):
        logger.debug(
            "analysis_result no es dict: %s",
            type(analysis_result).__name__,
        )
        return AnomalyData()

    details = analysis_result.get("details")
    if not isinstance(details, dict):
        logger.debug("'details' no encontrado o no es dict")
        return AnomalyData()

    # --- Anomalías básicas ---
    anomalies = details.get("anomalies")
    if isinstance(anomalies, dict):
        isolated_ids = frozenset(
            _extract_ids_from_list(anomalies.get("isolated_nodes", []))
        )
        orphan_ids = frozenset(
            _extract_ids_from_list(anomalies.get("orphan_insumos", []))
        )
        empty_ids = frozenset(
            _extract_ids_from_list(anomalies.get("empty_apus", []))
        )
    else:
        isolated_ids = frozenset()
        orphan_ids = frozenset()
        empty_ids = frozenset()

    # --- Ciclos ---
    cycles_data = details.get("cycles")
    if isinstance(cycles_data, list):
        nodes_in_cycles = frozenset(
            _extract_nodes_from_cycles(cycles_data)
        )
    elif isinstance(cycles_data, dict):
        nodes_in_cycles = frozenset(
            _extract_nodes_from_cycles(cycles_data.get("list", []))
        )
    else:
        nodes_in_cycles = frozenset()

    # --- Nodos anómalos ---
    anomalous_nodes_raw = details.get("anomalous_nodes")
    anomalous_nodes = (
        frozenset(_extract_ids_from_list(anomalous_nodes_raw))
        if isinstance(anomalous_nodes_raw, list)
        else frozenset()
    )

    # --- Scores de nodos ---
    node_scores_raw = details.get("node_scores")
    node_scores: Dict[str, float] = {}
    if isinstance(node_scores_raw, dict):
        for key, value in node_scores_raw.items():
            node_scores[_normalize_identifier(key)] = _safe_finite_float(
                value, 0.0
            )

    # --- Aristas anómalas ---
    anomalous_edges_raw = details.get("anomalous_edges")
    anomalous_edges: Set[Tuple[str, str]] = set()
    if isinstance(anomalous_edges_raw, list):
        for edge in anomalous_edges_raw:
            normalized = _try_normalize_edge(edge)
            if normalized is not None:
                anomalous_edges.add(normalized)

    # --- Scores de aristas ---
    edge_scores_raw = details.get("edge_scores")
    edge_scores: Dict[Tuple[str, str], float] = {}
    if isinstance(edge_scores_raw, dict):
        for key, value in edge_scores_raw.items():
            if isinstance(key, (tuple, list)) and len(key) == 2:
                edge_scores[
                    _normalize_edge_tuple(key[0], key[1])
                ] = _safe_finite_float(value, 0.0)

    return AnomalyData(
        isolated_ids=isolated_ids,
        orphan_ids=orphan_ids,
        empty_ids=empty_ids,
        nodes_in_cycles=nodes_in_cycles,
        anomalous_nodes=anomalous_nodes,
        node_scores=node_scores,
        anomalous_edges=frozenset(anomalous_edges),
        edge_scores=edge_scores,
    )


def _try_normalize_edge(edge: Any) -> Optional[Tuple[str, str]]:
    """
    Intenta normalizar una arista desde diversos formatos de entrada.

    Soporta:
    - ``(source, target)`` o ``[source, target]``
    - ``{"source": ..., "target": ...}``

    Retorna ``None`` si no se puede normalizar.
    """
    if isinstance(edge, (list, tuple)) and len(edge) == 2:
        return _normalize_edge_tuple(edge[0], edge[1])
    if isinstance(edge, dict):
        src = edge.get("source")
        tgt = edge.get("target")
        if src is not None and tgt is not None:
            return _normalize_edge_tuple(src, tgt)
    return None


def _extract_ids_from_list(items: Any) -> Set[str]:
    """
    Extrae identificadores normalizados de una lista heterogénea.

    Soporta elementos que sean strings, números, o dicts con clave ``"id"``.
    """
    if not isinstance(items, list):
        return set()

    ids: Set[str] = set()
    for item in items:
        if isinstance(item, dict):
            extracted = _normalize_identifier(
                item.get("id"), default=""
            )
        else:
            extracted = _normalize_identifier(item, default="")
        if extracted:
            ids.add(extracted)
    return ids


def _extract_nodes_from_cycles(cycles_list: Any) -> Set[str]:
    """
    Extrae nodos participantes en ciclos.

    Soporta dos formatos de entrada:
    1. Strings con separador ``" -> "`` o ``"→"``:
       ``"A -> B -> C -> A"``
    2. Listas/tuplas de nodos:
       ``["A", "B", "C", "A"]``
    """
    if not isinstance(cycles_list, list):
        return set()

    nodes: Set[str] = set()

    for cycle_entry in cycles_list:
        extracted = _parse_single_cycle(cycle_entry)
        nodes.update(extracted)

    return nodes


def _parse_single_cycle(cycle_entry: Any) -> Set[str]:
    """
    Parsea un único ciclo en formato string o lista/tupla.

    Para strings: separa por ``"->"`` o ``"→"`` y extrae nodos.
    Para listas/tuplas: extrae cada elemento como nodo.

    Elimina el último nodo si es igual al primero (cierre del ciclo).
    Un ciclo válido debe tener al menos 2 nodos distintos.
    """
    nodes: Set[str] = set()

    if isinstance(cycle_entry, str):
        raw = cycle_entry.strip()
        if not raw:
            return nodes

        normalized = raw.replace("→", "->")
        parts = [part.strip() for part in normalized.split("->") if part.strip()]

    elif isinstance(cycle_entry, (list, tuple)):
        parts = [
            str(item).strip()
            for item in cycle_entry
            if item is not None and str(item).strip()
        ]
    else:
        logger.debug(
            "Elemento de ciclo con tipo no soportado: %s",
            type(cycle_entry).__name__,
        )
        return nodes

    if len(parts) < 2:
        logger.debug(
            "Ciclo inválido o degenerado (< 2 nodos): %s", cycle_entry
        )
        return nodes

    # Eliminar cierre redundante: [A, B, C, A] → [A, B, C]
    if parts[-1] == parts[0]:
        parts = parts[:-1]

    for part in parts:
        node_id = _normalize_identifier(part, default="")
        if node_id:
            nodes.add(node_id)

    return nodes


def _identify_stressed_nodes(
    graph: nx.DiGraph,
    config: StressConfig = _DEFAULT_STRESS_CONFIG,
) -> frozenset[str]:
    """
    Identifica nodos INSUMO con concentración anómala de conexiones
    entrantes (patrón de pirámide invertida).

    Un insumo se considera estresado si:
        in_degree(insumo) / total_apus > threshold_ratio
        AND in_degree(insumo) ≥ min_absolute_connections

    El threshold se ajusta según el tamaño del grafo.

    Requiere ``DiGraph`` para análisis direccional.
    """
    if not isinstance(graph, nx.DiGraph):
        logger.warning(
            "_identify_stressed_nodes requiere DiGraph para análisis "
            "direccional; recibido=%s",
            type(graph).__name__,
        )
        return frozenset()

    apu_nodes = [
        n
        for n, d in graph.nodes(data=True)
        if _get_node_type(d if isinstance(d, dict) else {})
        == NodeType.APU.value
    ]
    total_apus = len(apu_nodes)
    if total_apus == 0:
        return frozenset()

    threshold_ratio = config.get_threshold_ratio(total_apus)
    stressed: Set[str] = set()

    for node, data in graph.nodes(data=True):
        node_type = _get_node_type(data if isinstance(data, dict) else {})
        if node_type != NodeType.INSUMO.value:
            continue

        connection_count = int(graph.in_degree(node))
        ratio = connection_count / total_apus
        if (
            ratio > threshold_ratio
            and connection_count >= config.min_absolute_connections
        ):
            stressed.add(_normalize_identifier(node))

    return frozenset(stressed)


# ======================================================================
# LÓGICA SEMÁNTICA VISUAL
# ======================================================================


def _get_node_type(attrs: Dict[str, Any]) -> str:
    """
    Extrae y normaliza el tipo de un nodo desde sus atributos.

    Retorna ``NodeType.UNKNOWN.value`` si el tipo es nulo, no convertible,
    o no pertenece al conjunto canónico.
    """
    raw_type = attrs.get("type")
    if raw_type is None:
        return NodeType.UNKNOWN.value

    try:
        normalized = str(raw_type).strip().upper()
    except (TypeError, ValueError, AttributeError):
        return NodeType.UNKNOWN.value

    return normalized if normalized in _VALID_NODE_TYPES else NodeType.UNKNOWN.value


def _get_node_cost(attrs: Dict[str, Any]) -> float:
    """
    Extrae el costo de un nodo con precedencia definida.

    Precedencia:
    1. ``total_cost`` (si presente y ≥ 0)
    2. ``unit_cost`` (si presente y ≥ 0)
    3. ``0.0`` (fallback)

    Usa sentinel interno ``-1.0`` para distinguir "ausente" de "cero legítimo".
    """
    _SENTINEL = -1.0

    total_cost = _safe_finite_float(attrs.get("total_cost"), default=_SENTINEL)
    if total_cost >= 0.0:
        return total_cost

    unit_cost = _safe_finite_float(attrs.get("unit_cost"), default=_SENTINEL)
    if unit_cost >= 0.0:
        return unit_cost

    return 0.0


def _get_node_score(
    node_id: str, anomaly_data: AnomalyData
) -> float:
    """Obtiene el score de anomalía de un nodo."""
    return _safe_nonnegative_finite_float(
        anomaly_data.node_scores.get(node_id), default=0.0
    )


def _get_edge_score(
    source: str, target: str, anomaly_data: AnomalyData
) -> float:
    """
    Obtiene el score de anomalía de una arista.

    Busca en ambas direcciones (forward y reverse) para grafos
    donde la dirección de la arista almacenada puede no coincidir
    con la del score.
    """
    forward = anomaly_data.edge_scores.get((source, target))
    if forward is not None:
        return _safe_nonnegative_finite_float(forward, default=0.0)

    reverse = anomaly_data.edge_scores.get((target, source))
    if reverse is not None:
        return _safe_nonnegative_finite_float(reverse, default=0.0)

    return 0.0


def _determine_node_classes(
    node_id: str, node_type: str, anomaly_data: AnomalyData
) -> Tuple[str, ...]:
    """
    Determina las clases CSS de un nodo basándose en su tipo y anomalías.

    Siempre incluye ``node_type`` como primera clase.
    Añade ``NodeClass.NORMAL`` solo si no hay ninguna anomalía.
    """
    classes: List[str] = [node_type]

    if node_id in anomaly_data.nodes_in_cycles:
        classes.extend(
            [NodeClass.CIRCULAR.value, NodeClass.CYCLE.value]
        )

    if node_id in anomaly_data.stressed_ids:
        classes.append(NodeClass.STRESS.value)

    if (
        node_id in anomaly_data.isolated_ids
        or node_id in anomaly_data.orphan_ids
    ):
        classes.append(NodeClass.ISOLATED.value)

    if node_id in anomaly_data.empty_ids:
        classes.append(NodeClass.EMPTY.value)

    if node_id in anomaly_data.hot_ids:
        classes.append(NodeClass.HOT.value)

    if node_id in anomaly_data.anomalous_nodes:
        classes.append(NodeClass.ANOMALOUS.value)

    # Si solo tiene node_type como clase, es un nodo normal
    if len(classes) == 1:
        classes.append(NodeClass.NORMAL.value)

    return _deduplicate_preserve_order(classes)


def _determine_node_color(
    node_id: str, node_type: str, anomaly_data: AnomalyData
) -> str:
    """
    Determina el color de un nodo según su estado de anomalía y tipo.

    Prioridad: anomalía (rojo) > tipo semántico > fallback (gris).
    Incluye ``empty_ids`` en la evaluación de anomalía.
    """
    if anomaly_data.is_node_anomalous(node_id):
        return NodeColor.RED.value

    if node_type in {
        NodeType.BUDGET.value,
        NodeType.CHAPTER.value,
        NodeType.ITEM.value,
    }:
        return NodeColor.BLACK.value
    if node_type == NodeType.APU.value:
        return NodeColor.BLUE.value
    if node_type == NodeType.INSUMO.value:
        return NodeColor.ORANGE.value
    return NodeColor.GRAY.value


def _build_node_label(node_id: str, attrs: Dict[str, Any]) -> str:
    """
    Construye la etiqueta visual de un nodo.

    Formato: ``"{node_id}\\n{descripcion_truncada}"`` si hay descripción,
    o solo ``node_id`` en caso contrario.
    """
    description = attrs.get("description")
    if description is None:
        return node_id

    try:
        desc_str = str(description).strip()
    except (TypeError, ValueError, AttributeError):
        return node_id

    if not desc_str:
        return node_id

    return f"{node_id}\n{_truncate_text(desc_str)}"


def _build_node_tooltip(
    node_id: str,
    attrs: Dict[str, Any],
    anomaly_data: AnomalyData,
    node_type: str,
    cost: float,
    score: float,
    level: int,
) -> str:
    """
    Construye tooltip forense para auditoría humana.

    Incluye: ID, descripción, tipo, nivel, costo, score y estado de anomalías.
    """
    description = attrs.get("description")
    desc = ""
    if description is not None:
        try:
            desc = str(description).strip()
        except (TypeError, ValueError, AttributeError):
            pass

    _ANOMALY_LABELS: Tuple[Tuple[frozenset[str], str], ...] = (
        (anomaly_data.nodes_in_cycles, "ciclo"),
        (anomaly_data.stressed_ids, "estrés"),
        (anomaly_data.isolated_ids, "aislado"),
        (anomaly_data.orphan_ids, "huérfano"),
        (anomaly_data.empty_ids, "vacío"),
        (anomaly_data.hot_ids, "hotspot"),
        (anomaly_data.anomalous_nodes, "anómalo"),
    )

    flags = [
        label for id_set, label in _ANOMALY_LABELS if node_id in id_set
    ]

    flags_str = ", ".join(flags) if flags else "sin anomalías"

    parts = [
        f"ID: {node_id}",
        f"Tipo: {node_type}",
        f"Nivel: {level}",
        f"Costo: {cost:.2f}",
        f"Score: {score:.4f}",
        f"Estado: {flags_str}",
    ]
    if desc:
        parts.insert(1, f"Descripción: {desc}")
    return "\n".join(parts)


def _build_edge_tooltip(
    source: str,
    target: str,
    cost: float,
    score: float,
    is_evidence: bool,
) -> str:
    """Construye tooltip forense para aristas."""
    evidence_str = "sí" if is_evidence else "no"
    return "\n".join(
        [
            f"Origen: {source}",
            f"Destino: {target}",
            f"Costo: {cost:.2f}",
            f"Score: {score:.4f}",
            f"Evidencia forense: {evidence_str}",
        ]
    )


# ======================================================================
# CONSTRUCCIÓN DE ELEMENTOS CYTOSCAPE
# ======================================================================


def build_node_element(
    node_id: Any,
    attrs: Dict[str, Any],
    anomaly_data: AnomalyData,
) -> CytoscapeNode:
    """
    Construye un elemento CytoscapeNode completo desde datos crudos.

    Normaliza identificador, extrae métricas, determina semántica visual,
    y construye tooltip forense.
    """
    node_id_str = _normalize_identifier(node_id)
    safe_attrs = attrs if isinstance(attrs, dict) else {}

    node_type = _get_node_type(safe_attrs)
    level = max(
        0, _safe_int_from_any(safe_attrs.get("level"), default=0)
    )
    cost = _get_node_cost(safe_attrs)
    score = _get_node_score(node_id_str, anomaly_data)
    label = _build_node_label(node_id_str, safe_attrs)
    classes = _determine_node_classes(
        node_id_str, node_type, anomaly_data
    )
    color = _determine_node_color(node_id_str, node_type, anomaly_data)

    is_evidence = anomaly_data.is_node_anomalous(node_id_str)

    tooltip = _build_node_tooltip(
        node_id=node_id_str,
        attrs=safe_attrs,
        anomaly_data=anomaly_data,
        node_type=node_type,
        cost=cost,
        score=score,
        level=level,
    )

    return CytoscapeNode(
        id=node_id_str,
        label=label,
        node_type=node_type,
        color=color,
        level=level,
        cost=cost,
        weight=cost,
        score=score,
        is_evidence=is_evidence,
        tooltip=tooltip,
        classes=classes,
    )


def _build_fallback_node(node_id_str: str) -> CytoscapeNode:
    """Construye un nodo fallback para errores de serialización."""
    return CytoscapeNode(
        id=node_id_str,
        label=node_id_str,
        node_type=NodeType.UNKNOWN.value,
        color=NodeColor.GRAY.value,
        level=0,
        cost=0.0,
        weight=0.0,
        score=0.0,
        is_evidence=False,
        tooltip=(
            f"ID: {node_id_str}\n"
            f"Tipo: UNKNOWN\n"
            f"Estado: fallback por error de serialización"
        ),
        classes=(NodeType.UNKNOWN.value, NodeClass.NORMAL.value),
    )


def _build_fallback_edge(
    src_str: str, tgt_str: str
) -> CytoscapeEdge:
    """Construye una arista fallback para errores de serialización."""
    return CytoscapeEdge(
        source=src_str,
        target=tgt_str,
        cost=0.0,
        score=0.0,
        is_evidence=False,
        tooltip=(
            f"Origen: {src_str}\n"
            f"Destino: {tgt_str}\n"
            "Estado: fallback por error de serialización"
        ),
    )


def build_edge_element(
    source: Any,
    target: Any,
    attrs: Dict[str, Any],
    anomaly_data: AnomalyData,
) -> CytoscapeEdge:
    """
    Construye un elemento CytoscapeEdge completo.

    Raises:
        ValueError: Si source o target se normalizan a string vacío.
    """
    src_str = _normalize_identifier(source, default="")
    tgt_str = _normalize_identifier(target, default="")
    if not src_str or not tgt_str:
        raise ValueError(
            f"Arista inválida: source={source!r}, target={target!r}"
        )

    safe_attrs = attrs if isinstance(attrs, dict) else {}
    cost = _safe_nonnegative_finite_float(
        safe_attrs.get("total_cost"), default=0.0
    )
    score = _get_edge_score(src_str, tgt_str, anomaly_data)

    explicit_evidence = (
        (src_str, tgt_str) in anomaly_data.anomalous_edges
        or (tgt_str, src_str) in anomaly_data.anomalous_edges
    )
    heuristic_cycle_evidence = (
        src_str in anomaly_data.nodes_in_cycles
        and tgt_str in anomaly_data.nodes_in_cycles
    )
    is_evidence = explicit_evidence or heuristic_cycle_evidence

    tooltip = _build_edge_tooltip(
        source=src_str,
        target=tgt_str,
        cost=cost,
        score=score,
        is_evidence=is_evidence,
    )

    return CytoscapeEdge(
        source=src_str,
        target=tgt_str,
        cost=cost,
        score=score,
        is_evidence=is_evidence,
        tooltip=tooltip,
    )


# ======================================================================
# PIPELINE PRINCIPAL
# ======================================================================


def build_graph_from_session(data: Dict[str, Any]) -> nx.DiGraph:
    """
    Construye un DiGraph desde datos de sesión.

    Pipeline: extracción → construcción → validación → conversión.

    Raises:
        ValueError: Si no hay datos suficientes o el grafo es inválido.
    """
    df_presupuesto, df_apus_detail = extract_dataframes_from_session(data)

    if df_presupuesto.empty and df_apus_detail.empty:
        raise ValueError(
            "No hay datos suficientes para construir el grafo."
        )

    try:
        builder = BudgetGraphBuilder()
        graph = builder.build(df_presupuesto, df_apus_detail)
    except Exception as exc:
        logger.error(
            "Error en BudgetGraphBuilder.build(): %s",
            exc,
            exc_info=True,
        )
        raise ValueError(f"Error construyendo grafo: {exc}") from exc

    outcome = validate_graph(graph)
    if not outcome.ok:
        raise ValueError(f"Grafo inválido: {outcome.message}")

    if not isinstance(graph, nx.DiGraph):
        try:
            graph = nx.DiGraph(graph)
        except Exception as exc:
            raise ValueError(
                f"No fue posible convertir el grafo a DiGraph: {exc}"
            ) from exc

    return graph


def analyze_graph_for_visualization(
    graph: nx.DiGraph,
    stress_config: StressConfig = _DEFAULT_STRESS_CONFIG,
) -> AnomalyData:
    """
    Ejecuta análisis topológico completo para visualización.

    Combina:
    1. Análisis de integridad estructural (ciclos, aislados, huérfanos).
    2. Detección de estrés topológico (pirámide invertida).
    3. Análisis de flujo térmico (hotspots).

    Cada análisis es independiente y degradable: si uno falla,
    los demás continúan.
    """
    # 1. Análisis de integridad estructural
    base_anomaly: Optional[AnomalyData] = None
    try:
        analyzer = BusinessTopologicalAnalyzer(telemetry=None)
        analysis_result = analyzer.analyze_structural_integrity(graph)
        base_anomaly = extract_anomaly_data(analysis_result)
    except Exception as exc:
        logger.warning("Error en análisis topológico base: %s", exc)

    if base_anomaly is None:
        base_anomaly = AnomalyData()

    # 2. Detección de estrés
    stressed_ids: frozenset[str] = frozenset()
    try:
        stressed_ids = _identify_stressed_nodes(
            graph, config=stress_config
        )
    except Exception as exc:
        logger.warning("Error identificando nodos estresados: %s", exc)

    # 3. Análisis térmico
    hot_ids: frozenset[str] = frozenset()
    try:
        analyzer = BusinessTopologicalAnalyzer(telemetry=None)
        thermal_result = analyzer.analyze_thermal_flow(graph)
        hotspots = (
            thermal_result.get("hotspots", [])
            if isinstance(thermal_result, dict)
            else []
        )
        if isinstance(hotspots, list):
            hot_ids = frozenset(
                _normalize_identifier(h.get("id"))
                for h in hotspots
                if isinstance(h, dict) and h.get("id") is not None
            )
    except Exception as exc:
        logger.warning("Error en análisis térmico visual: %s", exc)

    # Combinar resultados en un nuevo AnomalyData inmutable
    return AnomalyData(
        isolated_ids=base_anomaly.isolated_ids,
        orphan_ids=base_anomaly.orphan_ids,
        empty_ids=base_anomaly.empty_ids,
        nodes_in_cycles=base_anomaly.nodes_in_cycles,
        stressed_ids=stressed_ids,
        hot_ids=hot_ids,
        anomalous_nodes=base_anomaly.anomalous_nodes,
        node_scores=base_anomaly.node_scores,
        anomalous_edges=base_anomaly.anomalous_edges,
        edge_scores=base_anomaly.edge_scores,
    )


def _prepare_session_and_graph(
    session_data: Dict[str, Any],
) -> Tuple[nx.DiGraph, AnomalyData]:
    """
    Lógica compartida entre endpoints: valida sesión, construye grafo
    y ejecuta análisis.

    Raises:
        ValueError: Si la validación o construcción falla.
    """
    outcome = validate_session_data(session_data)
    if not outcome.ok:
        raise ValueError(f"Datos de sesión inválidos: {outcome.message}")

    graph = build_graph_from_session(session_data)
    anomaly_data = analyze_graph_for_visualization(graph)
    return graph, anomaly_data


def convert_graph_to_cytoscape_elements(
    graph: nx.DiGraph,
    anomaly_data: AnomalyData,
    stratum_filter: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Convierte un grafo NetworkX en elementos serializables para Cytoscape.js.

    Pipeline:
    1. Validar grafo y filtro de estrato.
    2. Iterar nodos, filtrar por nivel visible, construir elementos.
    3. Iterar aristas, incluir solo las que conectan nodos visibles.
    4. Elementos con error usan fallback degradado.

    Args:
        graph: DiGraph validado.
        anomaly_data: Datos de anomalía pre-calculados.
        stratum_filter: Filtro de estrato opcional (0–3).

    Returns:
        Lista de dicts serializables para Cytoscape.js.

    Raises:
        ValueError: Si el grafo o el filtro son inválidos.
    """
    outcome = validate_graph(graph)
    if not outcome.ok:
        raise ValueError(outcome.message or "Grafo inválido.")

    filter_outcome = validate_stratum_filter(stratum_filter)
    if not filter_outcome.ok:
        raise ValueError(
            filter_outcome.message or "Filtro de estrato inválido."
        )

    elements: List[Dict[str, Any]] = []
    node_ids_processed: Set[str] = set()
    visible_levels = _get_visible_levels(stratum_filter)

    # --- Nodos ---
    for node_id, attrs in graph.nodes(data=True):
        safe_attrs = attrs if isinstance(attrs, dict) else {}
        node_id_str = _normalize_identifier(node_id)
        level = max(
            0,
            _safe_int_from_any(safe_attrs.get("level"), default=0),
        )

        if visible_levels is not None and level not in visible_levels:
            continue

        try:
            node_element = build_node_element(
                node_id, safe_attrs, anomaly_data
            )
            elements.append(node_element.to_dict())
        except Exception as exc:
            logger.warning(
                "Error procesando nodo '%s': %s", node_id_str, exc
            )
            elements.append(
                _build_fallback_node(node_id_str).to_dict()
            )

        node_ids_processed.add(node_id_str)

    # --- Aristas ---
    for source, target, attrs in graph.edges(data=True):
        src_str = _normalize_identifier(source)
        tgt_str = _normalize_identifier(target)

        if (
            src_str not in node_ids_processed
            or tgt_str not in node_ids_processed
        ):
            logger.debug(
                "Arista fuera del subgrafo visible ignorada: "
                "'%s' -> '%s'",
                src_str,
                tgt_str,
            )
            continue

        try:
            safe_attrs = attrs if isinstance(attrs, dict) else {}
            edge_element = build_edge_element(
                source, target, safe_attrs, anomaly_data
            )
            elements.append(edge_element.to_dict())
        except Exception as exc:
            logger.warning(
                "Error procesando arista '%s' -> '%s': %s",
                src_str,
                tgt_str,
                exc,
            )
            elements.append(
                _build_fallback_edge(src_str, tgt_str).to_dict()
            )

    return elements


# ======================================================================
# RESPUESTAS HTTP
# ======================================================================


def create_error_response(
    message: str, status_code: int
) -> Tuple[Response, int]:
    """Genera respuesta HTTP de error estandarizada."""
    return (
        jsonify(
            {
                "error": str(message),
                "elements": [],
                "count": 0,
                "success": False,
            }
        ),
        status_code,
    )


def create_success_response(
    elements: List[Dict[str, Any]], status_code: int = 200
) -> Tuple[Response, int]:
    """Genera respuesta HTTP de éxito estandarizada."""
    return (
        jsonify(
            {
                "elements": elements,
                "count": len(elements),
                "success": True,
                "error": None,
            }
        ),
        status_code,
    )


# ======================================================================
# ENDPOINTS
# ======================================================================


@topology_bp.route(
    "/api/visualization/project-graph", methods=["GET"]
)
@topology_bp.route("/api/visualization/topology", methods=["GET"])
def get_project_graph() -> Tuple[Response, int]:
    """
    Endpoint GET para obtener el grafo de proyecto como elementos Cytoscape.js.

    Query params:
        stratum (int, opcional): Filtro de estrato (0–3).

    Responses:
        200: Elementos generados exitosamente.
        400: Datos de sesión o parámetros inválidos.
        404: No hay datos de sesión.
        500: Error interno inesperado.
    """
    if SessionKeys.PROCESSED_DATA not in session:
        return create_error_response("No hay datos de sesión.", 404)

    try:
        data = session[SessionKeys.PROCESSED_DATA]
        graph, anomaly_data = _prepare_session_and_graph(data)

        if graph.number_of_nodes() == 0:
            return create_success_response([])

        stratum_filter = _parse_stratum_param(
            request.args.get("stratum")
        )

        elements = convert_graph_to_cytoscape_elements(
            graph=graph,
            anomaly_data=anomaly_data,
            stratum_filter=stratum_filter,
        )
        return create_success_response(elements)

    except ValueError as exc:
        logger.warning("Error de validación en visualización: %s", exc)
        return create_error_response(str(exc), 400)

    except Exception as exc:
        logger.error(
            "Error inesperado generando visualización: %s",
            exc,
            exc_info=True,
        )
        return create_error_response(
            f"Error interno: {type(exc).__name__}", 500
        )


@topology_bp.route(
    "/api/visualization/topology/stats", methods=["GET"]
)
def get_topology_stats() -> Tuple[Response, int]:
    """
    Endpoint GET para obtener estadísticas topológicas del grafo.

    Responses:
        200: Estadísticas calculadas exitosamente.
        400: Datos inválidos.
        404: No hay datos de sesión.
        500: Error interno.
    """
    if SessionKeys.PROCESSED_DATA not in session:
        return create_error_response("No hay datos de sesión.", 404)

    try:
        data = session[SessionKeys.PROCESSED_DATA]
        graph, anomaly_data = _prepare_session_and_graph(data)

        stats = {
            "nodes": int(graph.number_of_nodes()),
            "edges": int(graph.number_of_edges()),
            "isolated_nodes": len(anomaly_data.isolated_ids),
            "orphan_nodes": len(anomaly_data.orphan_ids),
            "empty_apus": len(anomaly_data.empty_ids),
            "nodes_in_cycles": len(anomaly_data.nodes_in_cycles),
            "stressed_nodes": len(anomaly_data.stressed_ids),
            "hot_nodes": len(anomaly_data.hot_ids),
            "anomalous_nodes": len(anomaly_data.anomalous_nodes),
            "anomalous_edges": len(anomaly_data.anomalous_edges),
            "success": True,
        }
        return jsonify(stats), 200

    except ValueError as exc:
        logger.warning(
            "Error de validación en estadísticas: %s", exc
        )
        return create_error_response(str(exc), 400)

    except Exception as exc:
        logger.error(
            "Error obteniendo estadísticas topológicas: %s",
            exc,
            exc_info=True,
        )
        return create_error_response(str(exc), 500)


def _parse_stratum_param(
    raw_param: Optional[str],
) -> Optional[int]:
    """
    Parsea el parámetro de query ``stratum`` de forma segura.

    Retorna ``None`` si el parámetro es ausente, no numérico,
    o fuera del dominio permitido.
    """
    if raw_param is None:
        return None
    try:
        candidate = int(raw_param)
    except (TypeError, ValueError):
        logger.debug(
            "Parámetro 'stratum' inválido ignorado: %r", raw_param
        )
        return None

    if candidate not in ALLOWED_STRATUM_FILTERS:
        logger.debug(
            "Parámetro 'stratum' fuera de dominio: %r", raw_param
        )
        return None

    return candidate