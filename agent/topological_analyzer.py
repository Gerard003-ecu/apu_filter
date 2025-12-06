"""
Topological State Analyzer
==========================
Módulo para el análisis de la topología del sistema y homología persistente.

Fundamentos Matemáticos:
------------------------
1. Números de Betti (β):
   - β₀: Componentes conexas (objetos disjuntos)
   - β₁: Ciclos independientes (agujeros 1-dimensionales)
   - Para grafos: β₁ = |E| - |V| + β₀ (Euler-Poincaré)

2. Homología Persistente:
   - Estudia la evolución de características topológicas a través de una filtración
   - Características de vida larga = señales estructurales
   - Características de vida corta = ruido topológico

3. Característica de Euler:
   - χ = β₀ - β₁ = |V| - |E|
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import networkx as nx

logger = logging.getLogger("TopologicalAnalyzer")


# =============================================================================
# ENUMS Y TIPOS DE DATOS
# =============================================================================

class MetricState(Enum):
    """Estados posibles de una métrica según análisis de persistencia."""
    STABLE = auto()    # Bajo control, sin excursiones
    NOISE = auto()     # Excursión breve (muerte rápida en diagrama)
    FEATURE = auto()   # Patrón estructural (vida larga)
    CRITICAL = auto()  # Estado crítico persistente y activo
    UNKNOWN = auto()   # Datos insuficientes para clasificar


class HealthLevel(Enum):
    """Niveles de salud del sistema."""
    HEALTHY = auto()      # Sistema operando óptimamente
    DEGRADED = auto()     # Degradación parcial
    UNHEALTHY = auto()    # Problemas significativos
    CRITICAL = auto()     # Fallo inminente o activo


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class BettiNumbers:
    """
    Números de Betti del sistema.

    Invariantes topológicos que caracterizan la "forma" del grafo de servicios.
    """
    b0: int  # Componentes conexas
    b1: int  # Ciclos independientes
    num_vertices: int = 0
    num_edges: int = 0

    def __post_init__(self):
        if self.b0 < 0 or self.b1 < 0:
            raise ValueError("Los números de Betti no pueden ser negativos")

    @property
    def is_connected(self) -> bool:
        """Sistema completamente conectado (un solo componente)."""
        return self.b0 == 1

    @property
    def is_acyclic(self) -> bool:
        """Grafo sin ciclos (árbol o bosque)."""
        return self.b1 == 0

    @property
    def is_ideal(self) -> bool:
        """Estado ideal: conectado y acíclico."""
        return self.is_connected and self.is_acyclic

    @property
    def euler_characteristic(self) -> int:
        """Característica de Euler: χ = β₀ - β₁ = |V| - |E|."""
        return self.b0 - self.b1

    def __str__(self) -> str:
        status = "✓" if self.is_ideal else "⚠"
        return f"Betti({status} β₀={self.b0}, β₁={self.b1}, χ={self.euler_characteristic})"


@dataclass(frozen=True)
class PersistenceInterval:
    """
    Intervalo de nacimiento-muerte para homología persistente.

    Representa una característica topológica en el diagrama de persistencia.
    Un punto (birth, death) en el diagrama.
    """
    birth: int       # Índice de nacimiento
    death: int       # Índice de muerte (-1 si aún vive)
    dimension: int   # Dimensión de la característica (0 para componentes)
    amplitude: float = 0.0  # Amplitud máxima durante la vida

    @property
    def lifespan(self) -> float:
        """Duración de vida de la característica."""
        if self.death < 0:
            return float('inf')
        return self.death - self.birth

    @property
    def is_alive(self) -> bool:
        """¿La característica sigue viva?"""
        return self.death < 0

    @property
    def persistence(self) -> float:
        """
        Persistencia normalizada.
        Distancia perpendicular a la diagonal en el diagrama.
        """
        if self.death < 0:
            return float('inf')
        return (self.death - self.birth) / math.sqrt(2)

    def __str__(self) -> str:
        death_str = "∞" if self.death < 0 else str(self.death)
        return f"[{self.birth}, {death_str})"


@dataclass
class RequestLoopInfo:
    """Información sobre bucles de reintentos detectados."""
    request_id: str
    count: int
    first_seen: int  # Índice en el historial
    last_seen: int


@dataclass
class TopologicalHealth:
    """Resumen completo de salud topológica del sistema."""
    betti: BettiNumbers
    disconnected_nodes: FrozenSet[str]
    missing_edges: FrozenSet[Tuple[str, str]]
    request_loops: Tuple[RequestLoopInfo, ...]
    health_score: float  # 0.0 (crítico) a 1.0 (saludable)
    level: HealthLevel
    diagnostics: Dict[str, str] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        return self.level == HealthLevel.HEALTHY


@dataclass
class PersistenceAnalysisResult:
    """Resultado completo del análisis de persistencia."""
    state: MetricState
    intervals: Tuple[PersistenceInterval, ...]
    feature_count: int
    noise_count: int
    active_count: int
    max_lifespan: float
    total_persistence: float  # Suma de persistencias (área bajo barras)
    metadata: Dict[str, Union[int, float, str]]


# =============================================================================
# CLASE PRINCIPAL: SystemTopology
# =============================================================================

class SystemTopology:
    """
    Representa el estado de los servicios como un espacio topológico (Grafo).

    Fundamento Matemático:
    ----------------------
    Modelamos el sistema como un grafo no dirigido G = (V, E) donde:
    - V = conjunto de servicios (nodos)
    - E = conexiones activas entre servicios (aristas)

    Los números de Betti caracterizan la topología:
    - β₀ = número de componentes conexas
    - β₁ = |E| - |V| + β₀ (rango del ciclo del grafo)

    Para un sistema saludable típico:
    - β₀ = 1 (todo conectado)
    - β₁ = 0 (sin redundancias cíclicas)
    """

    REQUIRED_NODES: FrozenSet[str] = frozenset({
        "Agent",
        "Core",
        "Redis",
        "Filesystem"
    })

    # Topología esperada: árbol con Core como hub
    EXPECTED_TOPOLOGY: FrozenSet[Tuple[str, str]] = frozenset({
        ("Agent", "Core"),
        ("Core", "Redis"),
        ("Core", "Filesystem"),
    })

    def __init__(
        self,
        max_history: int = 50,
        custom_nodes: Optional[Set[str]] = None,
        custom_topology: Optional[Set[Tuple[str, str]]] = None
    ):
        """
        Inicializa la topología del sistema.

        Args:
            max_history: Tamaño máximo del historial de requests.
            custom_nodes: Nodos adicionales a monitorear.
            custom_topology: Topología esperada personalizada.

        Raises:
            ValueError: Si max_history < 1.
        """
        if max_history < 1:
            raise ValueError(f"max_history debe ser >= 1, recibido: {max_history}")

        self._graph = nx.Graph()
        self._max_history = max_history

        # Configurar nodos requeridos
        all_nodes = set(self.REQUIRED_NODES)
        if custom_nodes:
            all_nodes.update(custom_nodes)
        self._graph.add_nodes_from(all_nodes)

        # Configurar topología esperada
        self._expected_topology = set(self.EXPECTED_TOPOLOGY)
        if custom_topology:
            self._expected_topology.update(custom_topology)

        # Historial de requests con timestamps relativos
        self._request_history: deque = deque(maxlen=max_history)
        self._request_index: int = 0

        logger.debug(
            f"SystemTopology inicializado: {len(all_nodes)} nodos, "
            f"historial máximo: {max_history}"
        )

    # -------------------------------------------------------------------------
    # Gestión de Nodos
    # -------------------------------------------------------------------------

    @property
    def nodes(self) -> Set[str]:
        """Conjunto de todos los nodos en el grafo."""
        return set(self._graph.nodes())

    @property
    def edges(self) -> Set[Tuple[str, str]]:
        """Conjunto de todas las aristas activas."""
        return set(self._graph.edges())

    def add_node(self, node: str) -> bool:
        """
        Agrega un nodo dinámico al grafo.

        Args:
            node: Nombre del servicio a agregar.

        Returns:
            True si se agregó, False si ya existía o es inválido.
        """
        if not node or not isinstance(node, str) or not node.strip():
            logger.warning(f"Intento de agregar nodo inválido: {repr(node)}")
            return False

        node = node.strip()
        if node in self._graph:
            return False

        self._graph.add_node(node)
        logger.debug(f"Nodo agregado dinámicamente: {node}")
        return True

    def remove_node(self, node: str) -> bool:
        """
        Elimina un nodo del grafo (solo si no es requerido).

        Args:
            node: Nombre del servicio a eliminar.

        Returns:
            True si se eliminó correctamente.
        """
        if node in self.REQUIRED_NODES:
            logger.warning(f"No se puede eliminar nodo requerido: {node}")
            return False

        if node not in self._graph:
            return False

        self._graph.remove_node(node)
        logger.debug(f"Nodo eliminado: {node}")
        return True

    def has_node(self, node: str) -> bool:
        """Verifica si un nodo existe en el grafo."""
        return node in self._graph

    # -------------------------------------------------------------------------
    # Gestión de Conectividad
    # -------------------------------------------------------------------------

    def update_connectivity(
        self,
        active_connections: List[Tuple[str, str]],
        validate_nodes: bool = True,
        auto_add_nodes: bool = False
    ) -> Tuple[int, List[str]]:
        """
        Actualiza las conexiones del grafo basado en la telemetría.

        Args:
            active_connections: Lista de pares (origen, destino) activos.
            validate_nodes: Si True, valida que los nodos existan.
            auto_add_nodes: Si True, agrega nodos faltantes automáticamente.

        Returns:
            Tupla (edges_added, warnings) con número de aristas agregadas
            y lista de advertencias.
        """
        warnings: List[str] = []
        valid_edges: List[Tuple[str, str]] = []

        for item in active_connections:
            # Validar formato de arista
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                warnings.append(f"Formato de arista inválido: {repr(item)}")
                continue

            src, dst = item

            # Validar tipos
            if not isinstance(src, str) or not isinstance(dst, str):
                warnings.append(f"Nodos deben ser strings: ({repr(src)}, {repr(dst)})")
                continue

            src, dst = src.strip(), dst.strip()

            # Evitar auto-loops
            if src == dst:
                warnings.append(f"Auto-loop ignorado: {src}")
                continue

            # Validar existencia de nodos
            if validate_nodes:
                nodes_to_check = [(src, "origen"), (dst, "destino")]
                skip_edge = False

                for node, role in nodes_to_check:
                    if node not in self._graph:
                        if auto_add_nodes:
                            self.add_node(node)
                            warnings.append(f"Nodo {role} agregado automáticamente: {node}")
                        else:
                            warnings.append(f"Nodo {role} no existe: {node}")
                            skip_edge = True
                            break

                if skip_edge:
                    continue

            valid_edges.append((src, dst))

        # Actualizar grafo atómicamente
        self._graph.clear_edges()
        self._graph.add_edges_from(valid_edges)

        for warn in warnings:
            logger.warning(warn)

        logger.debug(f"Conectividad actualizada: {len(valid_edges)} aristas activas")
        return len(valid_edges), warnings

    def add_edge(self, src: str, dst: str) -> bool:
        """Agrega una arista individual si los nodos existen."""
        if src not in self._graph or dst not in self._graph:
            return False
        if src == dst:
            return False
        self._graph.add_edge(src, dst)
        return True

    def remove_edge(self, src: str, dst: str) -> bool:
        """Elimina una arista si existe."""
        if self._graph.has_edge(src, dst):
            self._graph.remove_edge(src, dst)
            return True
        return False

    # -------------------------------------------------------------------------
    # Registro de Requests
    # -------------------------------------------------------------------------

    def record_request(self, request_id: str) -> bool:
        """
        Registra un request_id para análisis de patrones de reintentos.

        Args:
            request_id: Identificador único del request.

        Returns:
            True si se registró correctamente.
        """
        if not request_id or not isinstance(request_id, str):
            return False

        request_id = request_id.strip()
        if not request_id:
            return False

        self._request_history.append((self._request_index, request_id))
        self._request_index += 1
        return True

    def clear_request_history(self) -> None:
        """Limpia el historial de requests."""
        self._request_history.clear()
        self._request_index = 0

    # -------------------------------------------------------------------------
    # Cálculos Topológicos (Números de Betti)
    # -------------------------------------------------------------------------

    def calculate_betti_numbers(
        self,
        include_isolated: bool = True
    ) -> BettiNumbers:
        """
        Calcula los números de Betti del sistema actual.

        Matemáticamente para un grafo G = (V, E):
        - β₀ = número de componentes conexas
        - β₁ = |E| - |V| + β₀ (del teorema de Euler-Poincaré)

        Interpretación del sistema:
        - β₀ = 1: Sistema totalmente conectado (Ideal)
        - β₀ > 1: Partición de red o servicios aislados
        - β₁ = 0: Arquitectura en árbol (Ideal)
        - β₁ > 0: Existen ciclos (dependencias circulares)

        Args:
            include_isolated: Si True, incluye nodos sin conexiones.

        Returns:
            BettiNumbers con β₀ y β₁.
        """
        if include_isolated:
            subgraph = self._graph
        else:
            # Solo nodos con al menos una conexión
            connected_nodes = [n for n in self._graph.nodes() if self._graph.degree(n) > 0]
            subgraph = self._graph.subgraph(connected_nodes)

        num_vertices = subgraph.number_of_nodes()
        num_edges = subgraph.number_of_edges()

        if num_vertices == 0:
            return BettiNumbers(b0=0, b1=0, num_vertices=0, num_edges=0)

        # β₀: Componentes conexas
        b0 = nx.number_connected_components(subgraph)

        # β₁: Usando fórmula de Euler para grafos
        # χ = |V| - |E| = β₀ - β₁
        # Por tanto: β₁ = |E| - |V| + β₀
        b1 = num_edges - num_vertices + b0

        # Por definición, β₁ ≥ 0 para grafos válidos
        b1 = max(0, b1)

        return BettiNumbers(
            b0=b0,
            b1=b1,
            num_vertices=num_vertices,
            num_edges=num_edges
        )

    def calculate_cyclomatic_complexity(self) -> int:
        """
        Calcula la complejidad ciclomática del grafo.

        M = E - N + 2P (fórmula de McCabe generalizada)
        donde P = número de componentes conexas.

        Para nuestro caso con grafos no dirigidos:
        M = β₁ + 1 por componente conexa

        Returns:
            Complejidad ciclomática total.
        """
        betti = self.calculate_betti_numbers()
        # Para cada componente, la complejidad base es 1
        return betti.b1 + betti.b0

    # -------------------------------------------------------------------------
    # Detección de Ciclos y Anomalías
    # -------------------------------------------------------------------------

    def detect_request_loops(
        self,
        threshold: int = 3,
        window: Optional[int] = None
    ) -> List[RequestLoopInfo]:
        """
        Detecta patrones de reintentos en el historial de requests.

        Nota: Esto representa "ciclos" en el espacio de fase del flujo
        de datos, NO ciclos topológicos del grafo (que son β₁).

        Args:
            threshold: Número mínimo de repeticiones para considerar un loop.
            window: Ventana de análisis (None = todo el historial).

        Returns:
            Lista de RequestLoopInfo ordenada por frecuencia descendente.
        """
        if not self._request_history:
            return []

        if threshold < 2:
            threshold = 2

        # Obtener ventana de análisis
        history = list(self._request_history)
        if window and window < len(history):
            history = history[-window:]

        # Agrupar por request_id
        request_info: Dict[str, Dict] = {}
        for idx, req_id in history:
            if req_id not in request_info:
                request_info[req_id] = {
                    'count': 0,
                    'first': idx,
                    'last': idx
                }
            request_info[req_id]['count'] += 1
            request_info[req_id]['last'] = idx

        # Filtrar y ordenar
        loops = [
            RequestLoopInfo(
                request_id=req_id,
                count=info['count'],
                first_seen=info['first'],
                last_seen=info['last']
            )
            for req_id, info in request_info.items()
            if info['count'] >= threshold
        ]

        return sorted(loops, key=lambda x: x.count, reverse=True)

    def find_structural_cycles(self) -> List[List[str]]:
        """
        Encuentra todos los ciclos fundamentales en el grafo.

        Usa la base de ciclos del grafo, que tiene exactamente β₁ ciclos.

        Returns:
            Lista de ciclos, donde cada ciclo es una lista de nodos.
        """
        try:
            return list(nx.cycle_basis(self._graph))
        except nx.NetworkXError:
            return []

    def get_disconnected_nodes(self) -> FrozenSet[str]:
        """
        Identifica nodos requeridos que están desconectados.

        Returns:
            Conjunto inmutable de nodos sin conexiones.
        """
        disconnected = frozenset(
            node for node in self.REQUIRED_NODES
            if node in self._graph and self._graph.degree(node) == 0
        )
        return disconnected

    def get_missing_connections(self) -> FrozenSet[Tuple[str, str]]:
        """
        Identifica conexiones esperadas que faltan.

        Returns:
            Conjunto inmutable de aristas esperadas no presentes.
        """
        missing = frozenset(
            edge for edge in self._expected_topology
            if not self._graph.has_edge(*edge)
        )
        return missing

    def get_unexpected_connections(self) -> FrozenSet[Tuple[str, str]]:
        """
        Identifica conexiones no esperadas en la topología.

        Returns:
            Conjunto de aristas presentes pero no esperadas.
        """
        current_edges = set(self._graph.edges())
        # Normalizar dirección de aristas para comparación
        expected_normalized = set()
        for u, v in self._expected_topology:
            expected_normalized.add((min(u, v), max(u, v)))

        unexpected = set()
        for u, v in current_edges:
            normalized = (min(u, v), max(u, v))
            if normalized not in expected_normalized:
                unexpected.add((u, v))

        return frozenset(unexpected)

    # -------------------------------------------------------------------------
    # Análisis de Salud
    # -------------------------------------------------------------------------

    def get_topological_health(self) -> TopologicalHealth:
        """
        Calcula un resumen completo de la salud topológica del sistema.

        El score de salud se calcula considerando:
        - Conectividad (β₀ = 1 es ideal)
        - Ausencia de ciclos (β₁ = 0 es ideal)
        - Nodos desconectados
        - Conexiones faltantes
        - Bucles de reintentos

        Returns:
            TopologicalHealth con métricas agregadas y diagnósticos.
        """
        betti = self.calculate_betti_numbers()
        disconnected = self.get_disconnected_nodes()
        missing = self.get_missing_connections()
        loops = self.detect_request_loops()

        # Calcular score de salud (0.0 a 1.0)
        score = 1.0
        diagnostics: Dict[str, str] = {}

        # Penalizar por componentes desconectados (muy grave)
        if betti.b0 > 1:
            penalty = 0.25 * (betti.b0 - 1)
            score -= penalty
            diagnostics["connectivity"] = (
                f"Sistema fragmentado en {betti.b0} componentes (-{penalty:.2f})"
            )

        # Penalizar por ciclos estructurales
        if betti.b1 > 0:
            penalty = 0.10 * min(betti.b1, 3)  # Cap en 3 ciclos
            score -= penalty
            diagnostics["cycles"] = (
                f"{betti.b1} ciclo(s) detectado(s) (-{penalty:.2f})"
            )

        # Penalizar por nodos desconectados
        if disconnected:
            penalty = 0.15 * len(disconnected)
            score -= penalty
            diagnostics["disconnected"] = (
                f"Nodos aislados: {', '.join(disconnected)} (-{penalty:.2f})"
            )

        # Penalizar por conexiones esperadas faltantes
        if missing:
            penalty = 0.12 * len(missing)
            score -= penalty
            edges_str = ', '.join(f"{u}-{v}" for u, v in missing)
            diagnostics["missing_edges"] = (
                f"Conexiones faltantes: {edges_str} (-{penalty:.2f})"
            )

        # Penalizar por bucles de reintentos
        if loops:
            penalty = 0.05 * min(len(loops), 5)  # Cap en 5 loops
            score -= penalty
            diagnostics["retry_loops"] = (
                f"{len(loops)} patrón(es) de reintento (-{penalty:.2f})"
            )

        score = max(0.0, min(1.0, score))

        # Determinar nivel de salud
        if score >= 0.9:
            level = HealthLevel.HEALTHY
        elif score >= 0.7:
            level = HealthLevel.DEGRADED
        elif score >= 0.4:
            level = HealthLevel.UNHEALTHY
        else:
            level = HealthLevel.CRITICAL

        if not diagnostics:
            diagnostics["status"] = "Sistema operando óptimamente"

        return TopologicalHealth(
            betti=betti,
            disconnected_nodes=disconnected,
            missing_edges=missing,
            request_loops=tuple(loops),
            health_score=score,
            level=level,
            diagnostics=diagnostics
        )

    # -------------------------------------------------------------------------
    # Utilidades
    # -------------------------------------------------------------------------

    def get_adjacency_matrix(self) -> Dict[str, Dict[str, int]]:
        """Retorna la matriz de adyacencia como diccionario anidado."""
        nodes = sorted(self._graph.nodes())
        matrix = {n: {m: 0 for m in nodes} for n in nodes}
        for u, v in self._graph.edges():
            matrix[u][v] = 1
            matrix[v][u] = 1
        return matrix

    def to_dict(self) -> Dict:
        """Serializa el estado actual a un diccionario."""
        return {
            "nodes": list(self._graph.nodes()),
            "edges": list(self._graph.edges()),
            "betti_numbers": {
                "b0": self.calculate_betti_numbers().b0,
                "b1": self.calculate_betti_numbers().b1,
            },
            "request_history_size": len(self._request_history),
        }

    def __repr__(self) -> str:
        betti = self.calculate_betti_numbers()
        return (
            f"SystemTopology(nodes={len(self._graph.nodes())}, "
            f"edges={len(self._graph.edges())}, {betti})"
        )


# =============================================================================
# CLASE: PersistenceHomology
# =============================================================================

class PersistenceHomology:
    """
    Análisis de Homología Persistente para métricas de series temporales.

    Fundamento Matemático:
    ----------------------
    La homología persistente estudia la evolución de características
    topológicas a través de una filtración (secuencia de subespacios).

    Para series temporales, implementamos una filtración por nivel:
    1. Definimos un umbral θ
    2. Para cada punto temporal t, observamos si f(t) > θ
    3. Las "excursiones" sobre el umbral son características 0-dimensionales
    4. Registramos nacimiento (cruce ascendente) y muerte (cruce descendente)

    Diagrama de Persistencia:
    - Cada característica se representa como punto (birth, death)
    - Persistencia = death - birth (distancia a la diagonal)
    - Alta persistencia = característica estructural
    - Baja persistencia = ruido

    Métricas derivadas:
    - Total persistence: Σ(death - birth) para todas las características
    - Persistence entropy: Medida de complejidad del diagrama
    """

    DEFAULT_WINDOW_SIZE: int = 20
    MIN_WINDOW_SIZE: int = 3

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE):
        """
        Inicializa el analizador de persistencia.

        Args:
            window_size: Tamaño de la ventana deslizante.

        Raises:
            ValueError: Si window_size < MIN_WINDOW_SIZE.
        """
        if window_size < self.MIN_WINDOW_SIZE:
            raise ValueError(
                f"window_size debe ser >= {self.MIN_WINDOW_SIZE}, "
                f"recibido: {window_size}"
            )

        self.window_size = window_size
        self._buffers: Dict[str, deque] = {}

        logger.debug(f"PersistenceHomology inicializado: window_size={window_size}")

    # -------------------------------------------------------------------------
    # Gestión de Datos
    # -------------------------------------------------------------------------

    @property
    def metrics(self) -> Set[str]:
        """Conjunto de métricas registradas."""
        return set(self._buffers.keys())

    def add_reading(self, metric_name: str, value: float) -> bool:
        """
        Agrega una lectura al buffer de una métrica.

        Args:
            metric_name: Nombre de la métrica.
            value: Valor de la lectura.

        Returns:
            True si se agregó correctamente.
        """
        # Validar nombre
        if not metric_name or not isinstance(metric_name, str):
            logger.warning(f"Nombre de métrica inválido: {repr(metric_name)}")
            return False

        metric_name = metric_name.strip()
        if not metric_name:
            return False

        # Validar valor numérico
        if not isinstance(value, (int, float)):
            logger.warning(f"Valor no numérico para {metric_name}: {repr(value)}")
            return False

        if math.isnan(value):
            logger.warning(f"Valor NaN ignorado para {metric_name}")
            return False

        # Manejar infinitos con cap
        if math.isinf(value):
            logger.warning(f"Valor infinito detectado para {metric_name}, aplicando cap")
            value = math.copysign(1e308, value)

        # Agregar al buffer
        if metric_name not in self._buffers:
            self._buffers[metric_name] = deque(maxlen=self.window_size)

        self._buffers[metric_name].append(value)
        return True

    def add_readings_batch(
        self,
        metric_name: str,
        values: List[float]
    ) -> int:
        """
        Agrega múltiples lecturas a una métrica.

        Returns:
            Número de lecturas agregadas exitosamente.
        """
        count = 0
        for value in values:
            if self.add_reading(metric_name, value):
                count += 1
        return count

    def get_buffer(self, metric_name: str) -> Optional[List[float]]:
        """Obtiene copia del buffer de una métrica."""
        buffer = self._buffers.get(metric_name)
        return list(buffer) if buffer else None

    def clear_metric(self, metric_name: str) -> bool:
        """Elimina una métrica específica."""
        if metric_name in self._buffers:
            del self._buffers[metric_name]
            return True
        return False

    def clear_all(self) -> None:
        """Limpia todos los buffers."""
        self._buffers.clear()

    # -------------------------------------------------------------------------
    # Cálculo de Persistencia
    # -------------------------------------------------------------------------

    def _compute_persistence_intervals(
        self,
        data: List[float],
        threshold: float
    ) -> List[PersistenceInterval]:
        """
        Calcula intervalos de persistencia para una serie de datos.

        Identifica todas las excursiones sobre el umbral y registra
        su nacimiento/muerte como características 0-dimensionales.

        Args:
            data: Lista de valores.
            threshold: Umbral de referencia.

        Returns:
            Lista de intervalos ordenados por nacimiento.
        """
        if not data:
            return []

        intervals: List[PersistenceInterval] = []
        above = False
        birth_idx = 0
        max_amplitude = 0.0

        for i, value in enumerate(data):
            is_above = value > threshold

            if is_above and not above:
                # Nacimiento: cruce ascendente
                birth_idx = i
                max_amplitude = value - threshold
                above = True
            elif is_above and above:
                # Actualizar amplitud máxima
                max_amplitude = max(max_amplitude, value - threshold)
            elif not is_above and above:
                # Muerte: cruce descendente
                intervals.append(PersistenceInterval(
                    birth=birth_idx,
                    death=i,
                    dimension=0,
                    amplitude=max_amplitude
                ))
                above = False
                max_amplitude = 0.0

        # Característica aún viva
        if above:
            intervals.append(PersistenceInterval(
                birth=birth_idx,
                death=-1,  # Indica que sigue viva
                dimension=0,
                amplitude=max_amplitude
            ))

        return intervals

    def get_persistence_diagram(
        self,
        metric_name: str,
        threshold: float
    ) -> List[PersistenceInterval]:
        """
        Calcula el diagrama de persistencia para una métrica.

        Args:
            metric_name: Nombre de la métrica.
            threshold: Umbral de referencia.

        Returns:
            Lista de intervalos de persistencia.
        """
        buffer = self._buffers.get(metric_name)
        if not buffer:
            return []

        return self._compute_persistence_intervals(list(buffer), threshold)

    def compute_total_persistence(
        self,
        intervals: List[PersistenceInterval],
        include_active: bool = False,
        active_lifespan_estimate: Optional[int] = None
    ) -> float:
        """
        Calcula la persistencia total (suma de duraciones de vida).

        Esta métrica cuantifica la "cantidad total de estructura"
        en el diagrama de persistencia.

        Args:
            intervals: Lista de intervalos.
            include_active: Si True, incluye características activas.
            active_lifespan_estimate: Estimación de vida para activas.

        Returns:
            Suma de persistencias finitas.
        """
        total = 0.0
        for interval in intervals:
            if interval.is_alive:
                if include_active and active_lifespan_estimate:
                    total += active_lifespan_estimate
            else:
                total += interval.lifespan
        return total

    def compute_persistence_entropy(
        self,
        intervals: List[PersistenceInterval]
    ) -> float:
        """
        Calcula la entropía del diagrama de persistencia.

        H = -Σ (p_i * log(p_i)) donde p_i = persistence_i / total_persistence

        Alta entropía = muchas características de vida similar
        Baja entropía = pocas características dominantes

        Args:
            intervals: Lista de intervalos.

        Returns:
            Entropía normalizada [0, 1].
        """
        finite_intervals = [i for i in intervals if not i.is_alive]
        if len(finite_intervals) < 2:
            return 0.0

        lifespans = [i.lifespan for i in finite_intervals]
        total = sum(lifespans)

        if total == 0:
            return 0.0

        # Calcular probabilidades
        probs = [l / total for l in lifespans]

        # Calcular entropía
        entropy = -sum(p * math.log(p) for p in probs if p > 0)

        # Normalizar por máxima entropía posible
        max_entropy = math.log(len(probs))
        if max_entropy == 0:
            return 0.0

        return entropy / max_entropy

    # -------------------------------------------------------------------------
    # Análisis de Alto Nivel
    # -------------------------------------------------------------------------

    def analyze_persistence(
        self,
        metric_name: str,
        threshold: float,
        noise_ratio: float = 0.2,
        critical_ratio: float = 0.5
    ) -> PersistenceAnalysisResult:
        """
        Análisis completo de persistencia para una métrica.

        Clasificación topológica:
        - STABLE: Sin excursiones sobre el umbral
        - NOISE: Solo excursiones de vida corta (< noise_ratio * window)
        - FEATURE: Excursiones de vida larga (patrón estructural)
        - CRITICAL: Excursión activa y persistente (> critical_ratio * window)

        Args:
            metric_name: Nombre de la métrica.
            threshold: Umbral de referencia.
            noise_ratio: Proporción para considerar ruido (default 20%).
            critical_ratio: Proporción para considerar crítico (default 50%).

        Returns:
            PersistenceAnalysisResult con análisis completo.
        """
        buffer = self._buffers.get(metric_name)

        # Datos insuficientes
        if not buffer or len(buffer) < self.MIN_WINDOW_SIZE:
            return PersistenceAnalysisResult(
                state=MetricState.UNKNOWN,
                intervals=tuple(),
                feature_count=0,
                noise_count=0,
                active_count=0,
                max_lifespan=0,
                total_persistence=0,
                metadata={"reason": "insufficient_data", "samples": len(buffer) if buffer else 0}
            )

        # Validar ratios
        noise_ratio = max(0.05, min(0.5, noise_ratio))
        critical_ratio = max(noise_ratio, min(0.9, critical_ratio))

        data = list(buffer)
        intervals = self._compute_persistence_intervals(data, threshold)

        # Sin excursiones
        if not intervals:
            return PersistenceAnalysisResult(
                state=MetricState.STABLE,
                intervals=tuple(),
                feature_count=0,
                noise_count=0,
                active_count=0,
                max_lifespan=0,
                total_persistence=0,
                metadata={
                    "reason": "below_threshold",
                    "max_value": max(data),
                    "threshold": threshold
                }
            )

        # Calcular límites
        noise_limit = max(1, int(self.window_size * noise_ratio))
        critical_limit = int(self.window_size * critical_ratio)

        # Clasificar intervalos
        noise_intervals: List[PersistenceInterval] = []
        feature_intervals: List[PersistenceInterval] = []
        active_intervals: List[PersistenceInterval] = []

        data_length = len(data)

        for interval in intervals:
            if interval.is_alive:
                # Característica activa
                actual_lifespan = data_length - interval.birth
                active_intervals.append(interval)

                # Verificar si es crítica
                if actual_lifespan >= critical_limit:
                    return PersistenceAnalysisResult(
                        state=MetricState.CRITICAL,
                        intervals=tuple(intervals),
                        feature_count=len(feature_intervals),
                        noise_count=len(noise_intervals),
                        active_count=len(active_intervals),
                        max_lifespan=float('inf'),
                        total_persistence=self.compute_total_persistence(intervals),
                        metadata={
                            "reason": "persistent_active_excursion",
                            "active_duration": actual_lifespan,
                            "critical_threshold": critical_limit,
                            "birth_index": interval.birth
                        }
                    )
            else:
                # Característica finalizada
                if interval.lifespan >= noise_limit:
                    feature_intervals.append(interval)
                else:
                    noise_intervals.append(interval)

        # Calcular métricas
        max_lifespan = 0.0
        if feature_intervals:
            max_lifespan = max(i.lifespan for i in feature_intervals)
        elif noise_intervals:
            max_lifespan = max(i.lifespan for i in noise_intervals)

        total_persistence = self.compute_total_persistence(intervals)

        # Determinar estado final
        if feature_intervals or (active_intervals and any(
            data_length - i.birth >= noise_limit for i in active_intervals
        )):
            state = MetricState.FEATURE
        elif noise_intervals or active_intervals:
            state = MetricState.NOISE
        else:
            state = MetricState.STABLE

        return PersistenceAnalysisResult(
            state=state,
            intervals=tuple(intervals),
            feature_count=len(feature_intervals),
            noise_count=len(noise_intervals),
            active_count=len(active_intervals),
            max_lifespan=max_lifespan,
            total_persistence=total_persistence,
            metadata={
                "noise_limit": noise_limit,
                "critical_limit": critical_limit,
                "window_size": self.window_size,
                "data_length": data_length
            }
        )

    def get_statistics(self, metric_name: str) -> Optional[Dict[str, float]]:
        """
        Estadísticas descriptivas del buffer de una métrica.

        Returns:
            Dict con min, max, mean, std, median o None si no hay datos.
        """
        buffer = self._buffers.get(metric_name)
        if not buffer:
            return None

        data = sorted(buffer)
        n = len(data)

        if n == 0:
            return None

        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / n

        # Mediana
        mid = n // 2
        if n % 2 == 0:
            median = (data[mid - 1] + data[mid]) / 2
        else:
            median = data[mid]

        return {
            "min": data[0],
            "max": data[-1],
            "mean": mean,
            "std": variance ** 0.5,
            "median": median,
            "count": n,
            "range": data[-1] - data[0]
        }

    def compare_diagrams(
        self,
        metric1: str,
        metric2: str,
        threshold: float
    ) -> float:
        """
        Compara dos diagramas de persistencia usando distancia de bottleneck simplificada.

        Returns:
            Distancia entre diagramas (0 = idénticos).
        """
        diagram1 = self.get_persistence_diagram(metric1, threshold)
        diagram2 = self.get_persistence_diagram(metric2, threshold)

        if not diagram1 and not diagram2:
            return 0.0

        # Extraer duraciones finitas
        lifespans1 = sorted([i.lifespan for i in diagram1 if not i.is_alive])
        lifespans2 = sorted([i.lifespan for i in diagram2 if not i.is_alive])

        if not lifespans1 and not lifespans2:
            return 0.0

        # Distancia de Wasserstein-1 aproximada
        max_len = max(len(lifespans1), len(lifespans2))
        lifespans1.extend([0] * (max_len - len(lifespans1)))
        lifespans2.extend([0] * (max_len - len(lifespans2)))

        return sum(abs(l1 - l2) for l1, l2 in zip(lifespans1, lifespans2))

    def __repr__(self) -> str:
        return (
            f"PersistenceHomology(window_size={self.window_size}, "
            f"metrics={len(self._buffers)})"
        )


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def compute_wasserstein_distance(
    intervals1: List[PersistenceInterval],
    intervals2: List[PersistenceInterval],
    p: int = 2
) -> float:
    """
    Calcula la distancia p-Wasserstein entre dos diagramas de persistencia.

    W_p = (Σ |l1_i - l2_i|^p)^(1/p)

    Esta es una aproximación que compara distribuciones de duraciones.

    Args:
        intervals1: Primer diagrama de persistencia.
        intervals2: Segundo diagrama de persistencia.
        p: Orden de la norma (default=2).

    Returns:
        Distancia de Wasserstein aproximada.
    """
    if p < 1:
        raise ValueError(f"p debe ser >= 1, recibido: {p}")

    if not intervals1 and not intervals2:
        return 0.0

    # Extraer duraciones finitas
    def get_lifespans(intervals: List[PersistenceInterval]) -> List[float]:
        return sorted([i.lifespan for i in intervals if not i.is_alive])

    lifespans1 = get_lifespans(intervals1)
    lifespans2 = get_lifespans(intervals2)

    if not lifespans1 and not lifespans2:
        return 0.0

    # Un diagrama vacío: distancia es la suma de duraciones del otro
    if not lifespans1:
        return sum(l ** p for l in lifespans2) ** (1/p)
    if not lifespans2:
        return sum(l ** p for l in lifespans1) ** (1/p)

    # Igualar longitudes con padding
    max_len = max(len(lifespans1), len(lifespans2))
    lifespans1.extend([0.0] * (max_len - len(lifespans1)))
    lifespans2.extend([0.0] * (max_len - len(lifespans2)))

    # Calcular distancia
    distance = sum(
        abs(l1 - l2) ** p
        for l1, l2 in zip(lifespans1, lifespans2)
    ) ** (1/p)

    return distance


def create_simple_topology() -> SystemTopology:
    """
    Crea una topología simple con conexiones default.

    Útil para testing y ejemplos.
    """
    topology = SystemTopology()
    topology.update_connectivity([
        ("Agent", "Core"),
        ("Core", "Redis"),
        ("Core", "Filesystem")
    ])
    return topology


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Ejemplo de SystemTopology
    print("=" * 60)
    print("DEMO: SystemTopology")
    print("=" * 60)

    topology = SystemTopology()

    # Estado inicial (sin conexiones)
    print(f"\nEstado inicial: {topology}")
    health = topology.get_topological_health()
    print(f"Salud: {health.level.name} (score: {health.health_score:.2f})")
    print(f"Diagnósticos: {health.diagnostics}")

    # Agregar conexiones
    topology.update_connectivity([
        ("Agent", "Core"),
        ("Core", "Redis"),
        ("Core", "Filesystem")
    ])

    print(f"\nCon conexiones: {topology}")
    health = topology.get_topological_health()
    print(f"Salud: {health.level.name} (score: {health.health_score:.2f})")
    print(f"Betti: {health.betti}")

    # Simular ciclo
    topology.update_connectivity([
        ("Agent", "Core"),
        ("Core", "Redis"),
        ("Core", "Filesystem"),
        ("Redis", "Agent")  # Crea un ciclo
    ])

    print(f"\nCon ciclo: {topology}")
    betti = topology.calculate_betti_numbers()
    print(f"Betti: {betti}")
    print(f"Ciclos encontrados: {topology.find_structural_cycles()}")

    # Ejemplo de PersistenceHomology
    print("\n" + "=" * 60)
    print("DEMO: PersistenceHomology")
    print("=" * 60)

    ph = PersistenceHomology(window_size=20)

    # Simular datos con excursión
    import random
    random.seed(42)

    for i in range(20):
        # Valor base con ruido
        value = 50 + random.gauss(0, 5)
        # Excursión en el medio
        if 8 <= i <= 14:
            value += 30
        ph.add_reading("cpu_usage", value)

    print(f"\nDatos de CPU simulados")
    stats = ph.get_statistics("cpu_usage")
    print(f"Estadísticas: {stats}")

    result = ph.analyze_persistence("cpu_usage", threshold=70)
    print(f"\nAnálisis (umbral=70):")
    print(f"  Estado: {result.state.name}")
    print(f"  Intervalos: {len(result.intervals)}")
    print(f"  Features: {result.feature_count}, Noise: {result.noise_count}")
    print(f"  Persistencia total: {result.total_persistence}")
    print(f"  Metadata: {result.metadata}")
