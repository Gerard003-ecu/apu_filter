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
        """Validación post-inicialización de invariantes."""
        if self.b0 < 0 or self.b1 < 0:
            raise ValueError("Los números de Betti no pueden ser negativos")

        if self.num_vertices < 0 or self.num_edges < 0:
            raise ValueError("El número de vértices y aristas no puede ser negativo")

        # Invariante: β₀ ≤ |V| (máximo una componente por vértice)
        if self.num_vertices > 0 and self.b0 > self.num_vertices:
            raise ValueError(
                f"β₀={self.b0} no puede exceder el número de vértices={self.num_vertices}"
            )

        # Invariante: para grafos simples, β₁ = |E| - |V| + β₀
        # Si tenemos vértices y aristas, verificar consistencia
        if self.num_vertices > 0 and self.num_edges >= 0:
            expected_b1 = self.num_edges - self.num_vertices + self.b0
            if expected_b1 >= 0 and self.b1 != expected_b1:
                raise ValueError(
                    f"Inconsistencia en Euler-Poincaré: β₁={self.b1} "
                    f"esperado={expected_b1} (|E|={self.num_edges}, |V|={self.num_vertices}, β₀={self.b0})"
                )

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
        # Plano de Datos (La Base)
        ("Core", "Redis"),
        ("Core", "Filesystem"),
        # Plano de Control (La Cúspide - El Agente observa todo)
        ("Agent", "Core"),
        ("Agent", "Redis"),      # Nueva conexión lógica
        ("Agent", "Filesystem")  # Nueva conexión lógica
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
        # Validación de tipo estricta
        if not isinstance(node, str):
            logger.warning(f"Intento de agregar nodo con tipo inválido: {type(node).__name__}")
            return False

        # Validación de contenido
        node_cleaned = node.strip()
        if not node_cleaned:
            logger.warning(f"Intento de agregar nodo vacío o solo espacios: {repr(node)}")
            return False

        # Validación de caracteres problemáticos (opcional pero recomendada)
        if any(c in node_cleaned for c in '\x00\n\r\t'):
            logger.warning(f"Nodo contiene caracteres de control no permitidos: {repr(node)}")
            return False

        if node_cleaned in self._graph:
            logger.debug(f"Nodo ya existe, ignorando: {node_cleaned}")
            return False

        self._graph.add_node(node_cleaned)
        logger.debug(f"Nodo agregado dinámicamente: {node_cleaned}")
        return True

    def remove_node(self, node: str) -> bool:
        """
        Elimina un nodo del grafo (solo si no es requerido).

        Args:
            node: Nombre del servicio a eliminar.

        Returns:
            True si se eliminó correctamente.
        """
        # Validación de tipo
        if not isinstance(node, str):
            logger.warning(f"Tipo inválido para remove_node: {type(node).__name__}")
            return False

        node = node.strip()

        if not node:
            return False

        if node in self.REQUIRED_NODES:
            logger.warning(f"No se puede eliminar nodo requerido: {node}")
            return False

        if node not in self._graph:
            logger.debug(f"Nodo no existe, nada que eliminar: {node}")
            return False

        # Registrar aristas que se perderán (para debugging/auditoría)
        lost_edges = list(self._graph.edges(node))
        if lost_edges:
            logger.debug(f"Eliminando nodo {node} con {len(lost_edges)} aristas asociadas")

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

        La operación es atómica: si hay errores críticos, el estado anterior
        se preserva.

        Args:
            active_connections: Lista de pares (origen, destino) activos.
            validate_nodes: Si True, valida que los nodos existan.
            auto_add_nodes: Si True, agrega nodos faltantes automáticamente.

        Returns:
            Tupla (edges_added, warnings) con número de aristas agregadas
            y lista de advertencias.

        Raises:
            TypeError: Si active_connections no es iterable.
        """
        # Validación de entrada principal
        if active_connections is None:
            logger.warning("active_connections es None, tratando como lista vacía")
            active_connections = []

        if not hasattr(active_connections, '__iter__'):
            raise TypeError(
                f"active_connections debe ser iterable, recibido: {type(active_connections).__name__}"
            )

        warnings: List[str] = []
        valid_edges: List[Tuple[str, str]] = []
        nodes_to_add: Set[str] = set()

        for idx, item in enumerate(active_connections):
            # Validar formato de arista
            if not isinstance(item, (tuple, list)):
                warnings.append(f"[{idx}] Formato de arista inválido (no es tupla/lista): {repr(item)}")
                continue

            if len(item) != 2:
                warnings.append(f"[{idx}] Arista debe tener exactamente 2 elementos: {repr(item)}")
                continue

            src, dst = item

            # Validar tipos de nodos
            if not isinstance(src, str) or not isinstance(dst, str):
                warnings.append(
                    f"[{idx}] Nodos deben ser strings: ({type(src).__name__}, {type(dst).__name__})"
                )
                continue

            src, dst = src.strip(), dst.strip()

            # Validar nodos no vacíos
            if not src or not dst:
                warnings.append(f"[{idx}] Nodos no pueden ser vacíos después de strip")
                continue

            # Evitar auto-loops
            if src == dst:
                warnings.append(f"[{idx}] Auto-loop ignorado: {src}")
                continue

            # Validar existencia de nodos
            if validate_nodes:
                missing_nodes = []

                for node, role in [(src, "origen"), (dst, "destino")]:
                    if node not in self._graph:
                        if auto_add_nodes:
                            nodes_to_add.add(node)
                            warnings.append(f"[{idx}] Nodo {role} será agregado: {node}")
                        else:
                            missing_nodes.append(f"{role}={node}")

                if missing_nodes and not auto_add_nodes:
                    warnings.append(f"[{idx}] Nodos faltantes: {', '.join(missing_nodes)}")
                    continue

            valid_edges.append((src, dst))

        # Fase de commit atómico
        # Guardar estado previo para rollback en caso de error
        previous_edges = list(self._graph.edges())

        try:
            # Agregar nodos nuevos primero
            for node in nodes_to_add:
                self._graph.add_node(node)

            # Actualizar aristas
            self._graph.clear_edges()
            self._graph.add_edges_from(valid_edges)

        except Exception as e:
            # Rollback: restaurar estado anterior
            logger.error(f"Error durante actualización, ejecutando rollback: {e}")
            self._graph.clear_edges()
            self._graph.add_edges_from(previous_edges)
            raise RuntimeError(f"Fallo en update_connectivity, estado restaurado: {e}") from e

        for warn in warnings:
            logger.warning(warn)

        logger.debug(
            f"Conectividad actualizada: {len(valid_edges)} aristas activas, "
            f"{len(nodes_to_add)} nodos agregados, {len(warnings)} advertencias"
        )
        return len(valid_edges), warnings

    def add_edge(self, src: str, dst: str) -> bool:
        """
        Agrega una arista individual si los nodos existen.

        Args:
            src: Nodo origen.
            dst: Nodo destino.

        Returns:
            True si se agregó correctamente.
        """
        # Validación de tipos
        if not isinstance(src, str) or not isinstance(dst, str):
            logger.warning(
                f"add_edge requiere strings: src={type(src).__name__}, dst={type(dst).__name__}"
            )
            return False

        src, dst = src.strip(), dst.strip()

        if not src or not dst:
            logger.warning("add_edge: nodos no pueden ser vacíos")
            return False

        if src not in self._graph:
            logger.debug(f"add_edge: nodo origen no existe: {src}")
            return False

        if dst not in self._graph:
            logger.debug(f"add_edge: nodo destino no existe: {dst}")
            return False

        if src == dst:
            logger.warning(f"add_edge: auto-loop no permitido: {src}")
            return False

        if self._graph.has_edge(src, dst):
            logger.debug(f"add_edge: arista ya existe: ({src}, {dst})")
            return False

        self._graph.add_edge(src, dst)
        return True

    def remove_edge(self, src: str, dst: str) -> bool:
        """
        Elimina una arista si existe.

        Args:
            src: Nodo origen.
            dst: Nodo destino.

        Returns:
            True si se eliminó correctamente.
        """
        # Validación de tipos
        if not isinstance(src, str) or not isinstance(dst, str):
            logger.warning(
                f"remove_edge requiere strings: src={type(src).__name__}, dst={type(dst).__name__}"
            )
            return False

        src, dst = src.strip(), dst.strip()

        if not src or not dst:
            return False

        if self._graph.has_edge(src, dst):
            self._graph.remove_edge(src, dst)
            logger.debug(f"Arista eliminada: ({src}, {dst})")
            return True

        logger.debug(f"remove_edge: arista no existe: ({src}, {dst})")
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

        Args:
            include_isolated: Si True, incluye nodos sin conexiones.
                          Si False, solo considera nodos con grado > 0.

        Returns:
            BettiNumbers con β₀ y β₁.

        Note:
            Para grafos vacíos (sin nodos) retorna β₀=0, β₁=0.
            Para grafos con solo nodos aislados y include_isolated=False,
            retorna β₀=0, β₁=0.
        """
        if include_isolated:
            subgraph = self._graph
        else:
            # Solo nodos con al menos una conexión
            connected_nodes = [
                n for n in self._graph.nodes()
                if self._graph.degree(n) > 0
            ]
            if not connected_nodes:
                # Todos los nodos están aislados
                return BettiNumbers(b0=0, b1=0, num_vertices=0, num_edges=0)
            subgraph = self._graph.subgraph(connected_nodes)

        num_vertices = subgraph.number_of_nodes()
        num_edges = subgraph.number_of_edges()

        # Caso: grafo vacío
        if num_vertices == 0:
            return BettiNumbers(b0=0, b1=0, num_vertices=0, num_edges=0)

        # β₀: Componentes conexas
        # Para un grafo vacío de aristas pero con nodos, cada nodo es una componente
        if num_edges == 0:
            b0 = num_vertices  # Cada nodo aislado es una componente
            b1 = 0  # Sin aristas no hay ciclos
        else:
            b0 = nx.number_connected_components(subgraph)
            # β₁: Usando fórmula de Euler para grafos
            # χ = |V| - |E| = β₀ - β₁
            # Por tanto: β₁ = |E| - |V| + β₀
            b1 = num_edges - num_vertices + b0

        # Invariante: β₁ ≥ 0 para grafos simples no dirigidos
        # Si b1 < 0, indica inconsistencia (no debería ocurrir)
        if b1 < 0:
            logger.warning(
                f"β₁ calculado como negativo ({b1}), ajustando a 0. "
                f"Esto sugiere inconsistencia: |V|={num_vertices}, |E|={num_edges}, β₀={b0}"
            )
            b1 = 0

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

        Args:
            threshold: Número mínimo de repeticiones para considerar un loop.
                       Valores < 2 se ajustan automáticamente a 2.
            window: Ventana de análisis (None = todo el historial).
                    Valores <= 0 se tratan como None.

        Returns:
            Lista de RequestLoopInfo ordenada por frecuencia descendente.
        """
        if not self._request_history:
            return []

        # Validación y normalización de threshold
        if not isinstance(threshold, int):
            try:
                threshold = int(threshold)
            except (TypeError, ValueError):
                logger.warning(f"threshold inválido: {threshold}, usando default=3")
                threshold = 3

        # Mínimo significativo es 2 (una repetición)
        if threshold < 2:
            logger.debug(f"threshold={threshold} ajustado a 2 (mínimo significativo)")
            threshold = 2

        # Validación y normalización de window
        if window is not None:
            if not isinstance(window, int):
                try:
                    window = int(window)
                except (TypeError, ValueError):
                    logger.warning(f"window inválido: {window}, usando todo el historial")
                    window = None

            if window is not None and window <= 0:
                logger.debug(f"window={window} no válido, usando todo el historial")
                window = None

        # Obtener ventana de análisis
        history = list(self._request_history)

        if window and window < len(history):
            history = history[-window:]

        if not history:
            return []

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

        # Ordenamiento estable: primero por count (desc), luego por last_seen (desc)
        return sorted(loops, key=lambda x: (x.count, x.last_seen), reverse=True)

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
    # Visualización
    # -------------------------------------------------------------------------

    def visualize_topology(self, output_path: str = "data/topology_status.png") -> bool:
        """
        Genera una imagen PNG del grafo actual vs esperado.

        Las aristas existentes se dibujan en VERDE.
        Las aristas faltantes se dibujan en ROJO PUNTEADO.

        Args:
            output_path: Ruta donde guardar la imagen.

        Returns:
            True si se generó correctamente.
        """
        try:
            import matplotlib.pyplot as plt
            # Usar backend no interactivo para evitar errores en entornos sin display
            plt.switch_backend('Agg')
        except ImportError:
            logger.error("matplotlib no está instalado, no se puede visualizar")
            return False

        try:
            # Crear figura
            plt.figure(figsize=(10, 8))

            # Crear grafo compuesto (actual + esperado)
            viz_graph = nx.Graph()

            # Agregar todos los nodos (actuales y esperados)
            all_nodes = set(self._graph.nodes())
            for u, v in self._expected_topology:
                all_nodes.add(u)
                all_nodes.add(v)
            viz_graph.add_nodes_from(all_nodes)

            # Clasificar aristas
            existing_edges = []
            missing_edges = []

            # Aristas existentes (Verde)
            for u, v in self._graph.edges():
                viz_graph.add_edge(u, v)
                existing_edges.append((u, v))

            # Aristas faltantes (Rojo Punteado)
            for u, v in self._expected_topology:
                if not self._graph.has_edge(u, v):
                    viz_graph.add_edge(u, v)
                    missing_edges.append((u, v))

            # Calcular layout
            pos = nx.spring_layout(viz_graph, seed=42)  # Seed para consistencia

            # Dibujar Nodos
            nx.draw_networkx_nodes(
                viz_graph, pos,
                node_size=2000,
                node_color='lightblue',
                edgecolors='black'
            )
            nx.draw_networkx_labels(viz_graph, pos)

            # Dibujar Aristas Existentes
            if existing_edges:
                nx.draw_networkx_edges(
                    viz_graph, pos,
                    edgelist=existing_edges,
                    edge_color='green',
                    width=2,
                    style='solid',
                    label='Activa'
                )

            # Dibujar Aristas Faltantes
            if missing_edges:
                nx.draw_networkx_edges(
                    viz_graph, pos,
                    edgelist=missing_edges,
                    edge_color='red',
                    width=2,
                    style='dashed',
                    label='Faltante'
                )

            plt.title(f"Estado Topológico del Sistema (Betti: {self.calculate_betti_numbers()})")
            plt.legend(loc='upper right')
            plt.axis('off')

            # Asegurar directorio
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            plt.savefig(output_path)
            plt.close()

            logger.info(f"Visualización guardada en: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error generando visualización: {e}")
            plt.close()
            return False

    # -------------------------------------------------------------------------
    # Análisis de Salud
    # -------------------------------------------------------------------------

    def get_topological_health(self) -> TopologicalHealth:
        """
        Calcula un resumen completo de la salud topológica del sistema.

        El score de salud se calcula con penalizaciones acotadas para evitar
        scores negativos extremos antes del clamp final.

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

        # Definir pesos y caps para cada categoría de penalización
        # Esto garantiza que ninguna categoría individual domine el score

        PENALTY_CAPS = {
            'connectivity': 0.35,      # Máximo 35% por fragmentación
            'cycles': 0.15,            # Máximo 15% por ciclos
            'disconnected': 0.20,      # Máximo 20% por nodos aislados
            'missing_edges': 0.20,     # Máximo 20% por conexiones faltantes
            'retry_loops': 0.10,       # Máximo 10% por bucles de reintentos
        }

        # Penalizar por componentes desconectados (muy grave)
        if betti.b0 > 1:
            raw_penalty = 0.20 * (betti.b0 - 1)
            penalty = min(raw_penalty, PENALTY_CAPS['connectivity'])
            score -= penalty
            diagnostics["connectivity"] = (
                f"Sistema fragmentado en {betti.b0} componentes (-{penalty:.2f})"
            )

        # Penalizar por ciclos estructurales
        if betti.b1 > 0:
            raw_penalty = 0.08 * betti.b1
            penalty = min(raw_penalty, PENALTY_CAPS['cycles'])
            score -= penalty
            diagnostics["cycles"] = (
                f"{betti.b1} ciclo(s) detectado(s) (-{penalty:.2f})"
            )

        # Penalizar por nodos requeridos desconectados
        if disconnected:
            num_disconnected = len(disconnected)
            num_required = len(self.REQUIRED_NODES)
            # Proporcional a la fracción de nodos requeridos afectados
            if num_required > 0:
                raw_penalty = PENALTY_CAPS['disconnected'] * (num_disconnected / num_required)
            else:
                raw_penalty = 0.05 * num_disconnected
            penalty = min(raw_penalty, PENALTY_CAPS['disconnected'])
            score -= penalty
            diagnostics["disconnected"] = (
                f"Nodos aislados: {', '.join(sorted(disconnected))} (-{penalty:.2f})"
            )

        # Penalizar por conexiones esperadas faltantes
        if missing:
            num_missing = len(missing)
            num_expected = len(self._expected_topology)
            # Proporcional a la fracción de topología faltante
            if num_expected > 0:
                raw_penalty = PENALTY_CAPS['missing_edges'] * (num_missing / num_expected)
            else:
                raw_penalty = 0.10 * num_missing
            penalty = min(raw_penalty, PENALTY_CAPS['missing_edges'])
            score -= penalty
            edges_str = ', '.join(f"{u}-{v}" for u, v in sorted(missing))
            diagnostics["missing_edges"] = (
                f"Conexiones faltantes: {edges_str} (-{penalty:.2f})"
            )

        # Penalizar por bucles de reintentos (indicador de problemas de flujo)
        if loops:
            # Considerar tanto cantidad como severidad (frecuencia)
            total_retries = sum(loop.count for loop in loops)
            raw_penalty = 0.02 * len(loops) + 0.005 * min(total_retries, 20)
            penalty = min(raw_penalty, PENALTY_CAPS['retry_loops'])
            score -= penalty
            diagnostics["retry_loops"] = (
                f"{len(loops)} patrón(es) de reintento, {total_retries} total (-{penalty:.2f})"
            )

        # Clamp final (aunque con caps no debería ser necesario ir a negativo)
        score = max(0.0, min(1.0, score))

        # Determinar nivel de salud con histéresis implícita en los umbrales
        if score >= 0.85:
            level = HealthLevel.HEALTHY
        elif score >= 0.65:
            level = HealthLevel.DEGRADED
        elif score >= 0.35:
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
            health_score=round(score, 4),  # Redondear para evitar ruido flotante
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
            metric_name: Nombre de la métrica (no vacío).
            value: Valor de la lectura (numérico finito, infinitos se capean).

        Returns:
            True si se agregó correctamente, False si se rechazó.
        """
        # Validar nombre de métrica
        if not isinstance(metric_name, str):
            logger.warning(
                f"Nombre de métrica debe ser string: {type(metric_name).__name__}"
            )
            return False

        metric_name = metric_name.strip()
        if not metric_name:
            logger.warning("Nombre de métrica vacío rechazado")
            return False

        # Validar valor numérico
        if not isinstance(value, (int, float)):
            logger.warning(f"Valor no numérico para {metric_name}: {type(value).__name__}")
            return False

        # Convertir int a float para consistencia
        value = float(value)

        # Rechazar NaN estrictamente
        if math.isnan(value):
            logger.warning(f"Valor NaN rechazado para {metric_name}")
            return False

        # Manejar infinitos con cap logarítmico para preservar orden de magnitud
        if math.isinf(value):
            # Usar un valor grande pero representable
            MAX_FINITE = 1e100  # Más conservador que 1e308
            capped_value = math.copysign(MAX_FINITE, value)
            logger.warning(
                f"Valor infinito para {metric_name} capeado a {capped_value:.2e}"
            )
            value = capped_value

        # Detectar valores extremos que podrían causar problemas numéricos
        if abs(value) > 1e100:
            logger.debug(
                f"Valor extremo detectado para {metric_name}: {value:.2e}"
            )

        # Agregar al buffer (crear si no existe)
        if metric_name not in self._buffers:
            self._buffers[metric_name] = deque(maxlen=self.window_size)
            logger.debug(f"Nuevo buffer creado para métrica: {metric_name}")

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

        Identifica todas las excursiones estrictamente por encima del umbral
        y registra su nacimiento/muerte como características 0-dimensionales.

        Comportamiento en casos límite:
        - Valores exactamente iguales al threshold se consideran "por debajo"
        - Múltiples excursiones consecutivas se fusionan en una sola
        - La última excursión puede quedar "viva" (death=-1)

        Args:
            data: Lista de valores numéricos.
            threshold: Umbral de referencia.

        Returns:
            Lista de intervalos ordenados por nacimiento.
        """
        if not data:
            return []

        # Validar que threshold sea numérico y finito
        if not isinstance(threshold, (int, float)) or math.isnan(threshold):
            logger.warning(f"Threshold inválido: {threshold}, usando 0.0")
            threshold = 0.0

        if math.isinf(threshold):
            # Con threshold infinito, ningún valor finito puede excederlo
            if threshold > 0:
                return []  # +inf: nada está por encima
            # -inf: todo está por encima (una sola excursión)
            return [PersistenceInterval(
                birth=0,
                death=-1,
                dimension=0,
                amplitude=max(data) - threshold if data else 0.0
            )]

        intervals: List[PersistenceInterval] = []
        above = False
        birth_idx = 0
        max_amplitude = 0.0
        current_max_value = threshold  # Track del valor máximo durante excursión

        for i, value in enumerate(data):
            # Manejar NaN en datos: tratar como "por debajo" del threshold
            if math.isnan(value):
                is_above = False
            else:
                is_above = value > threshold  # Estrictamente mayor

            if is_above and not above:
                # Nacimiento: primer cruce ascendente
                birth_idx = i
                max_amplitude = value - threshold
                current_max_value = value
                above = True

            elif is_above and above:
                # Continúa por encima: actualizar amplitud máxima
                amplitude = value - threshold
                if amplitude > max_amplitude:
                    max_amplitude = amplitude
                    current_max_value = value

            elif not is_above and above:
                # Muerte: cruce descendente (o NaN)
                intervals.append(PersistenceInterval(
                    birth=birth_idx,
                    death=i,
                    dimension=0,
                    amplitude=max_amplitude
                ))
                above = False
                max_amplitude = 0.0
                current_max_value = threshold

        # Característica aún viva al final de la serie
        if above:
            intervals.append(PersistenceInterval(
                birth=birth_idx,
                death=-1,  # Marca como viva
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
        - UNKNOWN: Datos insuficientes para análisis
        - STABLE: Sin excursiones sobre el umbral
        - NOISE: Solo excursiones de vida corta (< noise_ratio * window)
        - FEATURE: Excursiones de vida larga (patrón estructural)
        - CRITICAL: Excursión activa prolongada (> critical_ratio * window)

        Args:
            metric_name: Nombre de la métrica.
            threshold: Umbral de referencia.
            noise_ratio: Proporción para considerar ruido (default 20%).
                     Se clampea a [0.05, 0.5].
            critical_ratio: Proporción para considerar crítico (default 50%).
                        Se clampea a [noise_ratio, 0.9].

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
                max_lifespan=0.0,
                total_persistence=0.0,
                metadata={
                    "reason": "insufficient_data",
                    "samples": len(buffer) if buffer else 0,
                    "minimum_required": self.MIN_WINDOW_SIZE
                }
            )

        # Validar y normalizar ratios
        noise_ratio = float(max(0.05, min(0.5, noise_ratio)))
        critical_ratio = float(max(noise_ratio + 0.1, min(0.9, critical_ratio)))

        # Garantizar orden lógico
        if critical_ratio <= noise_ratio:
            critical_ratio = min(0.9, noise_ratio + 0.2)

        data = list(buffer)
        data_length = len(data)
        intervals = self._compute_persistence_intervals(data, threshold)

        # Sin excursiones: sistema estable
        if not intervals:
            data_max = max(data) if data else 0.0
            return PersistenceAnalysisResult(
                state=MetricState.STABLE,
                intervals=tuple(),
                feature_count=0,
                noise_count=0,
                active_count=0,
                max_lifespan=0.0,
                total_persistence=0.0,
                metadata={
                    "reason": "below_threshold",
                    "max_value": data_max,
                    "threshold": threshold,
                    "margin": threshold - data_max
                }
            )

        # Calcular límites basados en tamaño efectivo de datos
        noise_limit = max(1, int(data_length * noise_ratio))
        critical_limit = max(noise_limit + 1, int(data_length * critical_ratio))

        # Clasificar intervalos
        noise_intervals: List[PersistenceInterval] = []
        feature_intervals: List[PersistenceInterval] = []
        active_intervals: List[PersistenceInterval] = []

        critical_active: Optional[PersistenceInterval] = None

        for interval in intervals:
            if interval.is_alive:
                actual_lifespan = data_length - interval.birth
                active_intervals.append(interval)

                # Verificar si es crítica (activa y muy persistente)
                if actual_lifespan >= critical_limit:
                    critical_active = interval
            else:
                # Característica finalizada: clasificar por duración
                if interval.lifespan >= noise_limit:
                    feature_intervals.append(interval)
                else:
                    noise_intervals.append(interval)

        # Si hay una excursión crítica activa, retornar inmediatamente
        if critical_active is not None:
            actual_lifespan = data_length - critical_active.birth
            return PersistenceAnalysisResult(
                state=MetricState.CRITICAL,
                intervals=tuple(intervals),
                feature_count=len(feature_intervals),
                noise_count=len(noise_intervals),
                active_count=len(active_intervals),
                max_lifespan=float('inf'),
                total_persistence=self.compute_total_persistence(
                    intervals,
                    include_active=True,
                    active_lifespan_estimate=actual_lifespan
                ),
                metadata={
                    "reason": "persistent_active_excursion",
                    "active_duration": actual_lifespan,
                    "critical_threshold": critical_limit,
                    "birth_index": critical_active.birth,
                    "amplitude": critical_active.amplitude
                }
            )

        # Calcular métricas agregadas
        all_finite_lifespans = [
            i.lifespan for i in intervals if not i.is_alive
        ]
        max_lifespan = max(all_finite_lifespans) if all_finite_lifespans else 0.0
        total_persistence = self.compute_total_persistence(intervals)

        # Determinar estado final con lógica clara
        # Prioridad: features > active significativas > noise > stable
        has_features = len(feature_intervals) > 0
        has_significant_active = any(
            data_length - i.birth >= noise_limit
            for i in active_intervals
        )
        has_noise = len(noise_intervals) > 0 or len(active_intervals) > 0

        if has_features or has_significant_active:
            state = MetricState.FEATURE
            reason = "structural_pattern_detected"
        elif has_noise:
            state = MetricState.NOISE
            reason = "transient_excursions_only"
        else:
            state = MetricState.STABLE
            reason = "no_significant_excursions"

        return PersistenceAnalysisResult(
            state=state,
            intervals=tuple(intervals),
            feature_count=len(feature_intervals),
            noise_count=len(noise_intervals),
            active_count=len(active_intervals),
            max_lifespan=max_lifespan,
            total_persistence=total_persistence,
            metadata={
                "reason": reason,
                "noise_limit": noise_limit,
                "critical_limit": critical_limit,
                "window_size": self.window_size,
                "data_length": data_length,
                "noise_ratio_used": noise_ratio,
                "critical_ratio_used": critical_ratio
            }
        )

    def get_statistics(self, metric_name: str) -> Optional[Dict[str, float]]:
        """
        Estadísticas descriptivas del buffer de una métrica.

        Usa varianza muestral (n-1) para estimación no sesgada.
        Incluye percentiles adicionales para mejor caracterización.

        Returns:
            Dict con estadísticas o None si no hay datos.
        """
        buffer = self._buffers.get(metric_name)
        if not buffer:
            return None

        data = list(buffer)
        n = len(data)

        if n == 0:
            return None

        # Caso especial: un solo dato
        if n == 1:
            value = data[0]
            return {
                "min": value,
                "max": value,
                "mean": value,
                "std": 0.0,
                "variance": 0.0,
                "median": value,
                "count": 1,
                "range": 0.0,
                "p25": value,
                "p75": value,
                "iqr": 0.0
            }

        data_sorted = sorted(data)

        # Media
        mean = sum(data_sorted) / n

        # Varianza muestral (dividir por n-1 para estimador no sesgado)
        variance = sum((x - mean) ** 2 for x in data_sorted) / (n - 1)
        std = variance ** 0.5

        # Función auxiliar para percentiles
        def percentile(sorted_data: List[float], p: float) -> float:
            """Calcula el percentil p (0-100) usando interpolación lineal."""
            if not sorted_data:
                return 0.0
            k = (len(sorted_data) - 1) * (p / 100.0)
            f = int(k)
            c = f + 1 if f + 1 < len(sorted_data) else f
            if f == c:
                return sorted_data[f]
            return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)

        # Calcular percentiles
        p25 = percentile(data_sorted, 25)
        median = percentile(data_sorted, 50)
        p75 = percentile(data_sorted, 75)

        return {
            "min": data_sorted[0],
            "max": data_sorted[-1],
            "mean": mean,
            "std": std,
            "variance": variance,
            "median": median,
            "count": n,
            "range": data_sorted[-1] - data_sorted[0],
            "p25": p25,
            "p75": p75,
            "iqr": p75 - p25  # Rango intercuartílico
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
    Calcula la distancia p-Wasserstein aproximada entre dos diagramas de persistencia.

    Esta es una aproximación que compara distribuciones ordenadas de duraciones.
    La distancia real de Wasserstein requiere resolver un problema de
    matching óptimo, pero esta aproximación es computacionalmente eficiente
    y proporciona una métrica útil para comparar complejidad topológica.

    W_p ≈ (Σ |l1_i - l2_i|^p)^(1/p)

    Args:
        intervals1: Primer diagrama de persistencia.
        intervals2: Segundo diagrama de persistencia.
        p: Orden de la norma (debe ser >= 1, default=2).

    Returns:
        Distancia de Wasserstein aproximada (>= 0).

    Raises:
        ValueError: Si p < 1.
    """
    # Validación de parámetro p
    if not isinstance(p, (int, float)):
        raise TypeError(f"p debe ser numérico, recibido: {type(p).__name__}")

    if p < 1:
        raise ValueError(f"p debe ser >= 1, recibido: {p}")

    # Convertir a float para cálculos
    p = float(p)

    # Caso trivial: ambos vacíos
    if not intervals1 and not intervals2:
        return 0.0

    # Extraer duraciones finitas de cada diagrama
    def get_lifespans(intervals: List[PersistenceInterval]) -> List[float]:
        lifespans = []
        for i in intervals:
            if not i.is_alive:
                lifespan = i.lifespan
                # Validar que lifespan sea finito y no negativo
                if math.isfinite(lifespan) and lifespan >= 0:
                    lifespans.append(lifespan)
        return sorted(lifespans)

    lifespans1 = get_lifespans(intervals1)
    lifespans2 = get_lifespans(intervals2)

    # Ambos sin intervalos finitos
    if not lifespans1 and not lifespans2:
        return 0.0

    # Un diagrama vacío: la distancia es la norma-p del otro
    # (matching con la diagonal, que tiene lifespan 0)
    if not lifespans1:
        return sum(l ** p for l in lifespans2) ** (1.0 / p)
    if not lifespans2:
        return sum(l ** p for l in lifespans1) ** (1.0 / p)

    # Igualar longitudes con padding de ceros (matching con diagonal)
    max_len = max(len(lifespans1), len(lifespans2))

    # Extender con ceros para matching
    padded1 = lifespans1 + [0.0] * (max_len - len(lifespans1))
    padded2 = lifespans2 + [0.0] * (max_len - len(lifespans2))

    # Calcular distancia con manejo de overflow
    try:
        total = sum(
            abs(l1 - l2) ** p
            for l1, l2 in zip(padded1, padded2)
        )
        distance = total ** (1.0 / p)
    except OverflowError:
        logger.warning("Overflow en cálculo de Wasserstein, retornando infinito")
        return float('inf')

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
