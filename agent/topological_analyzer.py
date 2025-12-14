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
        Calcula los números de Betti del sistema actual con validación rigurosa.

        Para un grafo G = (V, E):
        - β₀ = número de componentes conexas
        - β₁ = |E| - |V| + β₀ (Teorema de Euler-Poincaré)

        Casos especiales manejados:
        1. Grafo vacío (sin nodos): β₀=0, β₁=0
        2. Nodos aislados: cada uno es una componente conexa
        3. Grafos no conexos: se calcula correctamente β₁ por componente
        4. Consistencia con fórmula de Euler para grafos

        Args:
            include_isolated: Si True, incluye nodos sin conexiones como componentes.

        Returns:
            BettiNumbers válidos y consistentes.

        Raises:
            RuntimeError: Si hay inconsistencia matemática irreconciliable.
        """
        # Caso especial: grafo vacío
        if self._graph.number_of_nodes() == 0:
            return BettiNumbers(b0=0, b1=0, num_vertices=0, num_edges=0)

        # Determinar subgrafo a analizar
        if include_isolated:
            subgraph = self._graph
        else:
            # Solo nodos con grado > 0
            connected_nodes = [n for n in self._graph.nodes()
                               if self._graph.degree(n) > 0]
            if not connected_nodes:
                return BettiNumbers(b0=0, b1=0, num_vertices=0, num_edges=0)
            subgraph = self._graph.subgraph(connected_nodes)

        num_vertices = subgraph.number_of_nodes()
        num_edges = subgraph.number_of_edges()

        # Caso 1: Sin aristas (todos nodos aislados)
        if num_edges == 0:
            # Cada nodo aislado es una componente conexa
            b0 = num_vertices
            b1 = 0
        else:
            # Calcular componentes conexas usando algoritmo robusto
            try:
                b0 = nx.number_connected_components(subgraph)
            except nx.NetworkXError as e:
                logger.error(f"Error calculando componentes conexas: {e}")
                # Fallback: contar nodos con grado > 0
                b0 = len([n for n in subgraph.nodes() if subgraph.degree(n) > 0]) or 1

            # Calcular β₁ usando fórmula de Euler-Poincaré
            # χ = |V| - |E| = β₀ - β₁  =>  β₁ = |E| - |V| + β₀
            b1 = num_edges - num_vertices + b0

            # Validación: β₁ debe ser ≥ 0 para grafos simples
            if b1 < 0:
                # Esto indica inconsistencia en los datos o cálculo
                logger.warning(
                    f"β₁ negativo calculado ({b1}). Recalculando con enfoque conservador. "
                    f"V={num_vertices}, E={num_edges}, β₀={b0}"
                )

                # Enfoque conservador: β₁ = max(0, número de ciclos fundamentales)
                try:
                    # Contar ciclos fundamentales usando base de ciclos
                    cycles = nx.cycle_basis(subgraph)
                    b1 = len(cycles)
                except nx.NetworkXError:
                    # Si falla, usar fórmula ajustada
                    b1 = max(0, num_edges - num_vertices + b0)

        # Validación final de consistencia
        if b0 < 0 or b1 < 0:
            raise RuntimeError(
                f"Números de Betti inválidos: β₀={b0}, β₁={b1}. "
                f"Verifique la integridad del grafo."
            )

        # Verificar invariante: β₀ ≤ |V|
        if b0 > num_vertices:
            logger.warning(
                f"β₀ ({b0}) > número de vértices ({num_vertices}). "
                f"Ajustando β₀ = {num_vertices}"
            )
            b0 = num_vertices
            # Recalcular β₁ para mantener consistencia
            b1 = max(0, num_edges - num_vertices + b0)

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
        window: Optional[int] = None,
        min_time_between_repeats: Optional[int] = None
    ) -> List[RequestLoopInfo]:
        """
        Detecta patrones de reintentos con análisis temporal robusto.

        Mejoras:
        1. Análisis de frecuencia temporal (no solo conteo)
        2. Detección de patrones periódicos
        3. Filtrado por densidad temporal
        4. Clasificación por severidad

        Args:
            threshold: Mínimo número de repeticiones para considerar loop.
            window: Ventana temporal de análisis (en índices).
            min_time_between_repeats: Mínimo tiempo entre repeticiones para
                                    considerar loop válido (evita falsos positivos).

        Returns:
            Lista de RequestLoopInfo ordenada por severidad.
        """
        if not self._request_history:
            return []

        # Validación de parámetros
        threshold = max(2, int(threshold)) if isinstance(threshold, (int, float)) else 3

        if window is not None:
            try:
                window = int(window)
                if window <= 0:
                    window = None
            except (TypeError, ValueError):
                window = None

        if min_time_between_repeats is not None:
            try:
                min_time_between_repeats = int(min_time_between_repeats)
                if min_time_between_repeats < 0:
                    min_time_between_repeats = None
            except (TypeError, ValueError):
                min_time_between_repeats = None

        # Obtener historial dentro de ventana
        history = list(self._request_history)
        if window and window < len(history):
            history = history[-window:]

        if len(history) < threshold:
            return []

        # Agrupar por request_id con análisis temporal
        request_analysis: Dict[str, Dict] = {}

        for idx, req_id in history:
            if req_id not in request_analysis:
                request_analysis[req_id] = {
                    'count': 0,
                    'indices': [],
                    'first': idx,
                    'last': idx,
                    'intervals': []
                }

            info = request_analysis[req_id]
            info['count'] += 1
            info['indices'].append(idx)
            info['last'] = idx

            # Calcular intervalos entre ocurrencias
            if len(info['indices']) > 1:
                last_idx = info['indices'][-2]
                interval = idx - last_idx
                info['intervals'].append(interval)

        # Filtrar y clasificar loops
        loops = []

        for req_id, info in request_analysis.items():
            if info['count'] < threshold:
                continue

            # Análisis de patrones temporales
            indices = info['indices']
            intervals = info['intervals']

            # Calcular métricas de regularidad
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                if len(intervals) > 1:
                    # Desviación estándar de intervalos (medida de regularidad)
                    interval_variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)
                    interval_std = interval_variance ** 0.5
                    regularity = avg_interval / (interval_std + 1e-10)  # Coeficiente de variación inverso
                else:
                    interval_std = 0.0
                    regularity = 1.0
            else:
                avg_interval = 0.0
                interval_std = 0.0
                regularity = 1.0

            # Calcular densidad temporal
            time_span = info['last'] - info['first'] + 1
            if time_span > 0:
                density = info['count'] / time_span
            else:
                density = 1.0

            # Filtrar por mínimo tiempo entre repeticiones si se especifica
            if min_time_between_repeats is not None and intervals:
                min_observed_interval = min(intervals)
                if min_observed_interval < min_time_between_repeats:
                    # Los reintentos están demasiado cerca, podría no ser un loop significativo
                    logger.debug(
                        f"Request {req_id} tiene intervalo mínimo {min_observed_interval} < "
                        f"{min_time_between_repeats}, filtrando"
                    )
                    continue

            # Calcular severidad compuesta
            # Factores: frecuencia, regularidad, densidad
            frequency_score = min(1.0, info['count'] / 10.0)  # Normalizado a máx 10 repeticiones
            regularity_score = min(1.0, regularity / 5.0)  # Normalizado
            density_score = min(1.0, density * 5.0)  # Normalizado

            severity = 0.5 * frequency_score + 0.3 * regularity_score + 0.2 * density_score

            # Crear objeto de información extendido
            loop_info = RequestLoopInfo(
                request_id=req_id,
                count=info['count'],
                first_seen=info['first'],
                last_seen=info['last']
            )

            # Agregar metadatos adicionales (podría extenderse la clase si fuera necesario)
            # Por ahora usamos un atributo adicional no oficial para análisis
            setattr(loop_info, '_metadata', {
                'avg_interval': avg_interval,
                'interval_std': interval_std,
                'regularity': regularity,
                'density': density,
                'severity': severity
            })

            loops.append(loop_info)

        # Ordenar por severidad (primario) y luego por frecuencia (secundario)
        def get_severity(loop: RequestLoopInfo) -> float:
            metadata = getattr(loop, '_metadata', {})
            return metadata.get('severity', 0.0)

        return sorted(loops, key=lambda x: (get_severity(x), x.count), reverse=True)

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

    def visualize_topology(
        self,
        output_path: str = "data/topology_status.png",
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 100
    ) -> bool:
        """
        Genera una imagen PNG del grafo actual vs esperado.

        Las aristas existentes se dibujan en VERDE.
        Las aristas faltantes se dibujan en ROJO PUNTEADO.
        Las aristas inesperadas se dibujan en NARANJA.

        Args:
            output_path: Ruta donde guardar la imagen.
            figsize: Tamaño de la figura (ancho, alto) en pulgadas.
            dpi: Resolución de la imagen.

        Returns:
            True si se generó correctamente, False en caso contrario.
        """
        # Validación de parámetros
        if not isinstance(output_path, str):
            logger.error(f"output_path debe ser string, recibido: {type(output_path).__name__}")
            return False

        output_path = output_path.strip()
        if not output_path:
            logger.error("output_path no puede ser vacío")
            return False

        # Validar extensión
        valid_extensions = {'.png', '.jpg', '.jpeg', '.pdf', '.svg'}
        import os
        _, ext = os.path.splitext(output_path.lower())
        if ext not in valid_extensions:
            logger.warning(
                f"Extensión '{ext}' no reconocida, agregando '.png'"
            )
            output_path = output_path + '.png'

        # Validar figsize y dpi
        try:
            figsize = (int(figsize[0]), int(figsize[1]))
            if figsize[0] <= 0 or figsize[1] <= 0:
                raise ValueError("Dimensiones deben ser positivas")
        except (TypeError, IndexError, ValueError) as e:
            logger.warning(f"figsize inválido: {e}, usando default (10, 8)")
            figsize = (10, 8)

        try:
            dpi = int(dpi)
            if dpi <= 0:
                raise ValueError("DPI debe ser positivo")
        except (TypeError, ValueError) as e:
            logger.warning(f"dpi inválido: {e}, usando default 100")
            dpi = 100

        # Importar matplotlib con manejo de errores
        try:
            import matplotlib
            matplotlib.use('Agg')  # Backend no interactivo antes de importar pyplot
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error(
                "matplotlib no está instalado. Instalar con: pip install matplotlib"
            )
            return False
        except Exception as e:
            logger.error(f"Error configurando matplotlib: {e}")
            return False

        fig = None
        try:
            # Crear figura
            fig, ax = plt.subplots(figsize=figsize)

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
            unexpected_edges = []

            # Normalizar aristas esperadas para comparación
            expected_normalized = {
                tuple(sorted([u, v])) for u, v in self._expected_topology
            }

            # Aristas existentes
            for u, v in self._graph.edges():
                viz_graph.add_edge(u, v)
                normalized = tuple(sorted([u, v]))
                if normalized in expected_normalized:
                    existing_edges.append((u, v))
                else:
                    unexpected_edges.append((u, v))

            # Aristas faltantes
            for u, v in self._expected_topology:
                if not self._graph.has_edge(u, v):
                    viz_graph.add_edge(u, v)
                    missing_edges.append((u, v))

            # Calcular layout con seed para reproducibilidad
            try:
                pos = nx.spring_layout(viz_graph, seed=42, k=2.0)
            except Exception:
                # Fallback a layout circular si spring falla
                pos = nx.circular_layout(viz_graph)

            # Colorear nodos según estado
            node_colors = []
            for node in viz_graph.nodes():
                if node not in self._graph:
                    node_colors.append('lightgray')  # Nodo no existe
                elif self._graph.degree(node) == 0:
                    node_colors.append('salmon')  # Nodo aislado
                elif node in self.REQUIRED_NODES:
                    node_colors.append('lightblue')  # Nodo requerido conectado
                else:
                    node_colors.append('lightgreen')  # Nodo dinámico

            # Dibujar nodos
            nx.draw_networkx_nodes(
                viz_graph, pos, ax=ax,
                node_size=2000,
                node_color=node_colors,
                edgecolors='black',
                linewidths=2
            )
            nx.draw_networkx_labels(viz_graph, pos, ax=ax, font_size=10)

            # Dibujar aristas existentes (Verde)
            if existing_edges:
                nx.draw_networkx_edges(
                    viz_graph, pos, ax=ax,
                    edgelist=existing_edges,
                    edge_color='green',
                    width=2.5,
                    style='solid',
                    alpha=0.8
                )

            # Dibujar aristas faltantes (Rojo punteado)
            if missing_edges:
                nx.draw_networkx_edges(
                    viz_graph, pos, ax=ax,
                    edgelist=missing_edges,
                    edge_color='red',
                    width=2,
                    style='dashed',
                    alpha=0.7
                )

            # Dibujar aristas inesperadas (Naranja)
            if unexpected_edges:
                nx.draw_networkx_edges(
                    viz_graph, pos, ax=ax,
                    edgelist=unexpected_edges,
                    edge_color='orange',
                    width=2,
                    style='dotted',
                    alpha=0.7
                )

            # Título con información de Betti
            betti = self.calculate_betti_numbers()
            title = (
                f"Estado Topológico del Sistema\n"
                f"β₀={betti.b0} (componentes), β₁={betti.b1} (ciclos), "
                f"χ={betti.euler_characteristic}"
            )
            ax.set_title(title, fontsize=12, fontweight='bold')

            # Leyenda
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='green', linewidth=2, label='Conexión Activa'),
                Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Conexión Faltante'),
                Line2D([0], [0], color='orange', linewidth=2, linestyle=':', label='Conexión Inesperada'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

            ax.axis('off')
            plt.tight_layout()

            # Asegurar directorio existe
            output_dir = os.path.dirname(output_path)
            if output_dir:  # Si hay directorio (no es solo nombre de archivo)
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except OSError as e:
                    logger.error(f"No se pudo crear directorio '{output_dir}': {e}")
                    return False

            # Guardar figura
            try:
                plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
            except PermissionError:
                logger.error(f"Sin permisos para escribir en: {output_path}")
                return False
            except OSError as e:
                logger.error(f"Error de sistema guardando imagen: {e}")
                return False

            logger.info(f"Visualización guardada en: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error generando visualización: {e}", exc_info=True)
            return False

        finally:
            # Limpieza garantizada de recursos de matplotlib
            if fig is not None:
                plt.close(fig)
            else:
                plt.close('all')

    # -------------------------------------------------------------------------
    # Análisis de Salud
    # -------------------------------------------------------------------------

    def get_topological_health(self) -> TopologicalHealth:
        """
        Calcula salud topológica con modelo de penalizaciones normalizado.

        Modelo matemático:
        Score = 1.0 - Σ(penalización_i × peso_i)

        Donde:
        - penalización_i ∈ [0, 1] (normalizada por máximo posible)
        - peso_i ∈ [0, 1] con Σpesos_i = 1.0

        Pesos configurados:
        - Fragmentación (β₀ > 1): 0.35
        - Ciclos estructurales (β₁ > 0): 0.20
        - Nodos requeridos desconectados: 0.25
        - Conexiones esperadas faltantes: 0.15
        - Bucles de reintento: 0.05

        Returns:
            TopologicalHealth con score matemáticamente fundamentado.
        """
        # Calcular métricas base
        betti = self.calculate_betti_numbers()
        disconnected = self.get_disconnected_nodes()
        missing = self.get_missing_connections()
        loops = self.detect_request_loops()

        # Configuración de pesos (deben sumar 1.0)
        WEIGHTS = {
            'fragmentation': 0.35,    # β₀ > 1
            'cycles': 0.20,           # β₁ > 0
            'disconnected': 0.25,      # Nodos requeridos aislados
            'missing_edges': 0.15,     # Topología esperada incompleta
            'retry_loops': 0.05        # Patrones de reintento
        }

        # Verificar que pesos suman 1.0
        weight_sum = sum(WEIGHTS.values())
        if abs(weight_sum - 1.0) > 1e-10:
            logger.warning(f"Pesos no suman 1.0 ({weight_sum}), normalizando")
            WEIGHTS = {k: v/weight_sum for k, v in WEIGHTS.items()}

        diagnostics = {}
        penalty_score = 0.0

        # 1. Penalización por fragmentación (β₀ > 1)
        if betti.b0 > 1:
            # Normalizar: máximo fragmentación = todos nodos aislados
            max_components = betti.num_vertices
            actual_components = betti.b0
            if max_components > 0:
                # penalización = (componentes - 1) / (nodos - 1)
                # Esto da 0 para 1 componente, ~1 para todos nodos aislados
                fragmentation_penalty = min(1.0, (actual_components - 1) / (max_components - 1))
            else:
                fragmentation_penalty = 0.0

            penalty_score += fragmentation_penalty * WEIGHTS['fragmentation']
            diagnostics['fragmentation'] = (
                f"Sistema fragmentado en {betti.b0} componentes "
                f"(penalización: {fragmentation_penalty:.3f})"
            )

        # 2. Penalización por ciclos estructurales
        if betti.b1 > 0:
            # Normalizar por máximo de ciclos posibles en grafo simple
            # Para grafo completo K_n: máximo ciclos = combinaciones
            n = betti.num_vertices
            if n >= 3:
                # Máximo teórico de ciclos fundamentales ≈ C(n,3) para grafo completo
                max_cycles_approx = math.comb(n, 3) if n <= 20 else n * (n-1) * (n-2) // 6
                if max_cycles_approx > 0:
                    cycles_penalty = min(1.0, betti.b1 / max_cycles_approx)
                else:
                    cycles_penalty = 0.0
            else:
                cycles_penalty = 1.0 if betti.b1 > 0 else 0.0

            penalty_score += cycles_penalty * WEIGHTS['cycles']
            diagnostics['cycles'] = (
                f"{betti.b1} ciclo(s) estructural(es) "
                f"(penalización: {cycles_penalty:.3f})"
            )

        # 3. Penalización por nodos requeridos desconectados
        if disconnected:
            num_disconnected = len(disconnected)
            num_required = len(self.REQUIRED_NODES)

            if num_required > 0:
                disconnected_penalty = num_disconnected / num_required
            else:
                disconnected_penalty = 0.0

            penalty_score += disconnected_penalty * WEIGHTS['disconnected']
            nodes_str = ', '.join(sorted(disconnected))
            diagnostics['disconnected'] = (
                f"{num_disconnected}/{num_required} nodos requeridos desconectados: {nodes_str} "
                f"(penalización: {disconnected_penalty:.3f})"
            )

        # 4. Penalización por conexiones esperadas faltantes
        if missing:
            num_missing = len(missing)
            num_expected = len(self._expected_topology)

            if num_expected > 0:
                missing_penalty = num_missing / num_expected
            else:
                missing_penalty = 0.0

            penalty_score += missing_penalty * WEIGHTS['missing_edges']
            edges_str = ', '.join(f"{u}-{v}" for u, v in sorted(missing))
            diagnostics['missing_edges'] = (
                f"{num_missing}/{num_expected} conexiones esperadas faltantes: {edges_str} "
                f"(penalización: {missing_penalty:.3f})"
            )

        # 5. Penalización por bucles de reintento
        if loops:
            total_retries = sum(loop.count for loop in loops)
            unique_loops = len(loops)

            # Modelo combinado: considera tanto frecuencia como cantidad de bucles únicos
            # Normalizar por capacidad del historial
            max_retries = self._max_history
            if max_retries > 0:
                retry_frequency_penalty = min(1.0, total_retries / (2 * max_retries))
            else:
                retry_frequency_penalty = 0.0

            # Penalización por diversidad de bucles
            loop_diversity_penalty = min(1.0, unique_loops / 5.0)  # Máximo 5 bucles únicos

            # Combinar ambas penalizaciones
            retry_penalty = 0.7 * retry_frequency_penalty + 0.3 * loop_diversity_penalty

            penalty_score += retry_penalty * WEIGHTS['retry_loops']
            diagnostics['retry_loops'] = (
                f"{unique_loops} patrón(es) de reintento, {total_retries} intentos totales "
                f"(penalización: {retry_penalty:.3f})"
            )

        # Calcular score final (asegurar [0, 1])
        health_score = max(0.0, min(1.0, 1.0 - penalty_score))

        # Determinar nivel con histéresis y márgenes
        if health_score >= 0.90:
            level = HealthLevel.HEALTHY
        elif health_score >= 0.70:
            level = HealthLevel.DEGRADED
        elif health_score >= 0.40:
            level = HealthLevel.UNHEALTHY
        else:
            level = HealthLevel.CRITICAL

        # Si no hay diagnósticos, sistema está óptimo
        if not diagnostics:
            diagnostics['status'] = "Sistema topológicamente óptimo"

        return TopologicalHealth(
            betti=betti,
            disconnected_nodes=disconnected,
            missing_edges=missing,
            request_loops=tuple(loops),
            health_score=round(health_score, 4),
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
        """
        Serializa el estado actual a un diccionario.

        Incluye toda la información necesaria para reconstruir o debuggear
        el estado topológico del sistema.

        Returns:
            Dict con estado completo serializable a JSON.
        """
        # Calcular Betti una sola vez (optimización)
        betti = self.calculate_betti_numbers()

        # Obtener health de forma lazy solo si es necesario
        disconnected = self.get_disconnected_nodes()
        missing = self.get_missing_connections()
        unexpected = self.get_unexpected_connections()

        return {
            "nodes": sorted(self._graph.nodes()),  # Ordenado para consistencia
            "edges": sorted(
                [tuple(sorted(e)) for e in self._graph.edges()]
            ),  # Normalizado y ordenado
            "betti_numbers": {
                "b0": betti.b0,
                "b1": betti.b1,
                "num_vertices": betti.num_vertices,
                "num_edges": betti.num_edges,
                "euler_characteristic": betti.euler_characteristic,
                "is_connected": betti.is_connected,
                "is_acyclic": betti.is_acyclic,
                "is_ideal": betti.is_ideal,
            },
            "topology_status": {
                "disconnected_nodes": sorted(disconnected),
                "missing_connections": sorted(
                    [tuple(sorted(e)) for e in missing]
                ),
                "unexpected_connections": sorted(
                    [tuple(sorted(e)) for e in unexpected]
                ),
            },
            "request_history": {
                "size": len(self._request_history),
                "max_size": self._max_history,
                "current_index": self._request_index,
            },
            "configuration": {
                "required_nodes": sorted(self.REQUIRED_NODES),
                "expected_topology_size": len(self._expected_topology),
            }
        }

    def __repr__(self) -> str:
        """
        Representación string del objeto para debugging.

        Diseñado para nunca fallar, incluso si el estado interno es inconsistente.
        """
        try:
            node_count = len(self._graph.nodes())
            edge_count = len(self._graph.edges())
        except Exception:
            node_count = "?"
            edge_count = "?"

        try:
            betti = self.calculate_betti_numbers()
            betti_str = str(betti)
        except Exception as e:
            betti_str = f"BettiError({type(e).__name__})"

        return (
            f"SystemTopology(nodes={node_count}, "
            f"edges={edge_count}, {betti_str})"
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
        Calcula intervalos de persistencia con manejo riguroso de casos límite.

        Mejoras implementadas:
        1. Validación exhaustiva de inputs
        2. Manejo de NaN, ±Inf, y valores extremos
        3. Detección robusta de cruces por umbral
        4. Fusión de excursiones adyacentes
        5. Cálculo preciso de amplitud

        Args:
            data: Lista de valores de serie temporal.
            threshold: Umbral para detección de excursiones.

        Returns:
            Lista de intervalos de persistencia válidos y consistentes.
        """
        # Validación exhaustiva de inputs
        if not isinstance(data, list):
            raise TypeError(f"data debe ser lista, recibido: {type(data).__name__}")

        if not data:
            return []

        # Validar y normalizar threshold
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            logger.warning(f"threshold inválido {threshold}, usando 0.0")
            threshold = 0.0

        # Manejar threshold infinito
        if math.isinf(threshold):
            if threshold > 0:  # +inf
                # Ningún valor finito puede exceder +inf
                return []
            else:  # -inf
                # Todos los valores están por encima
                valid_data = [x for x in data if not math.isnan(x)]
                if not valid_data:
                    return []
                max_val = max(valid_data)
                return [PersistenceInterval(
                    birth=0,
                    death=-1,  # Siempre vivo
                    dimension=0,
                    amplitude=max_val - threshold if math.isfinite(max_val) else 0.0
                )]

        # Preprocesar datos: manejar NaN y valores extremos
        processed_data = []
        nan_indices = []

        for i, value in enumerate(data):
            if math.isnan(value):
                nan_indices.append(i)
                # Para análisis de persistencia, NaN se trata como valor ausente
                # Usamos un valor sentinela que indica "no disponible"
                processed_data.append(None)
            else:
                # Convertir a float y capar valores extremos pero preservando signo
                if math.isinf(value):
                    capped = math.copysign(1e100, value)
                    logger.debug(f"Valor infinito en índice {i} capeado a {capped}")
                    processed_data.append(capped)
                else:
                    processed_data.append(float(value))

        # Algoritmo principal de detección de excursiones
        intervals: List[PersistenceInterval] = []
        in_excursion = False
        birth_idx = 0
        current_max = -math.inf
        current_amplitude = 0.0

        for i, value in enumerate(processed_data):
            # Determinar si estamos por encima del umbral
            # None (NaN) siempre cuenta como por debajo
            if value is None:
                is_above = False
            else:
                is_above = value > threshold

            # Transición: inicia excursión
            if is_above and not in_excursion:
                birth_idx = i
                current_max = value
                current_amplitude = value - threshold
                in_excursion = True

            # Dentro de excursión
            elif is_above and in_excursion:
                if value > current_max:
                    current_max = value
                    current_amplitude = value - threshold

            # Transición: termina excursión
            elif not is_above and in_excursion:
                # Verificar que la excursión tiene duración mínima
                duration = i - birth_idx
                if duration > 0:  # Excursión no instantánea
                    intervals.append(PersistenceInterval(
                        birth=birth_idx,
                        death=i,
                        dimension=0,
                        amplitude=current_amplitude
                    ))
                in_excursion = False
                current_max = -math.inf
                current_amplitude = 0.0

        # Excursión activa al final de la serie
        if in_excursion:
            duration = len(data) - birth_idx
            if duration > 0:
                intervals.append(PersistenceInterval(
                    birth=birth_idx,
                    death=-1,  # Indica que sigue activa
                    dimension=0,
                    amplitude=current_amplitude
                ))

        # Fusión de excursiones adyacentes (separadas por NaN o valores en threshold)
        if intervals and nan_indices:
            merged_intervals = []
            current = intervals[0]

            for next_interval in intervals[1:]:
                # Verificar si las excursiones están adyacentes o muy cercanas
                gap = next_interval.birth - current.death
                if 0 <= gap <= 1 and current.death >= 0:  # next_interval no es activo
                    # Fusionar: extender muerte y tomar máxima amplitud
                    merged_amplitude = max(current.amplitude, next_interval.amplitude)
                    merged = PersistenceInterval(
                        birth=current.birth,
                        death=next_interval.death,
                        dimension=0,
                        amplitude=merged_amplitude
                    )
                    current = merged
                else:
                    merged_intervals.append(current)
                    current = next_interval

            merged_intervals.append(current)
            intervals = merged_intervals

        # Validación final: eliminar excursiones con amplitud no positiva
        valid_intervals = [
            interval for interval in intervals
            if interval.amplitude > 0 or interval.is_alive
        ]

        return valid_intervals

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
        Análisis de persistencia con clasificación probabilística robusta.

        Modelo de clasificación:
        1. STABLE: P(excursión) < 0.1 y persistencia_total < umbral_ruido
        2. NOISE: Excursiones de vida corta (< noise_ratio × ventana)
        3. FEATURE: Excursiones de vida larga con alta persistencia
        4. CRITICAL: Excursión activa prolongada (> critical_ratio × ventana)
        5. UNKNOWN: Datos insuficientes o inválidos

        Args:
            metric_name: Nombre de la métrica.
            threshold: Umbral para detección de excursiones.
            noise_ratio: Ratio para clasificar ruido (default 0.2 = 20%).
            critical_ratio: Ratio para clasificar crítico (default 0.5 = 50%).

        Returns:
            PersistenceAnalysisResult con clasificación fundamentada.
        """
        # Validación de parámetros
        if noise_ratio <= 0 or noise_ratio >= 1:
            logger.warning(f"noise_ratio inválido {noise_ratio}, usando 0.2")
            noise_ratio = 0.2

        if critical_ratio <= noise_ratio or critical_ratio >= 1:
            logger.warning(
                f"critical_ratio inválido {critical_ratio} (debe ser > {noise_ratio}), "
                f"usando {noise_ratio + 0.3}"
            )
            critical_ratio = min(0.9, noise_ratio + 0.3)

        # Obtener datos
        buffer = self._buffers.get(metric_name)

        # Caso: datos insuficientes
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
                    "minimum_required": self.MIN_WINDOW_SIZE,
                    "confidence": 0.0
                }
            )

        data = list(buffer)
        data_length = len(data)

        # Calcular estadísticas básicas para contexto
        try:
            stats = self.get_statistics(metric_name) or {}
            data_mean = stats.get('mean', 0.0)
            data_std = stats.get('std', 0.0)
        except Exception:
            data_mean = sum(data) / len(data) if data else 0.0
            data_std = 0.0

        # Calcular intervalos de persistencia
        intervals = self._compute_persistence_intervals(data, threshold)

        # Caso: sin excursiones
        if not intervals:
            # Verificar si el sistema está genuinamente estable
            # o si el threshold es demasiado alto
            max_value = max(data) if data else 0.0
            threshold_margin = threshold - max_value

            return PersistenceAnalysisResult(
                state=MetricState.STABLE,
                intervals=tuple(),
                feature_count=0,
                noise_count=0,
                active_count=0,
                max_lifespan=0.0,
                total_persistence=0.0,
                metadata={
                    "reason": "no_excursions",
                    "max_value": max_value,
                    "threshold": threshold,
                    "margin": threshold_margin,
                    "mean": data_mean,
                    "std": data_std,
                    "confidence": 0.95 if threshold_margin > 2 * data_std else 0.70
                }
            )

        # Calcular límites basados en ratios
        noise_limit = max(1, int(data_length * noise_ratio))
        critical_limit = max(noise_limit + 1, int(data_length * critical_ratio))

        # Clasificar intervalos
        noise_intervals = []
        feature_intervals = []
        active_intervals = []

        for interval in intervals:
            if interval.is_alive:
                active_intervals.append(interval)
            else:
                if interval.lifespan >= noise_limit:
                    feature_intervals.append(interval)
                else:
                    noise_intervals.append(interval)

        # Determinar si hay excursión crítica activa
        critical_active = None
        for interval in active_intervals:
            active_duration = data_length - interval.birth
            if active_duration >= critical_limit:
                critical_active = interval
                break

        # Si hay excursión crítica activa, clasificar como CRITICAL inmediatamente
        if critical_active:
            active_duration = data_length - critical_active.birth
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
                    active_lifespan_estimate=active_duration
                ),
                metadata={
                    "reason": "persistent_active_excursion",
                    "active_duration": active_duration,
                    "critical_limit": critical_limit,
                    "birth_index": critical_active.birth,
                    "amplitude": critical_active.amplitude,
                    "confidence": 0.99
                }
            )

        # Calcular métricas de persistencia
        total_persistence = self.compute_total_persistence(intervals)

        # Calcular persistencia máxima entre intervalos finitos
        finite_intervals = [i for i in intervals if not i.is_alive]
        if finite_intervals:
            max_lifespan = max(i.lifespan for i in finite_intervals)
            avg_lifespan = sum(i.lifespan for i in finite_intervals) / len(finite_intervals)
        else:
            max_lifespan = 0.0
            avg_lifespan = 0.0

        # Calcular métricas de distribución
        if finite_intervals:
            lifespans = [i.lifespan for i in finite_intervals]
            persistence_entropy = self.compute_persistence_entropy(intervals)
        else:
            persistence_entropy = 0.0

        # Lógica de clasificación basada en múltiples factores
        has_features = len(feature_intervals) > 0
        has_active = len(active_intervals) > 0
        has_noise = len(noise_intervals) > 0

        # Calcular score de confianza para cada estado
        scores = {
            'STABLE': 0.0,
            'NOISE': 0.0,
            'FEATURE': 0.0
        }

        # Factor 1: Proporción de datos en excursión
        total_excursion_points = sum(
            (i.death if i.death >= 0 else data_length) - i.birth
            for i in intervals
        )
        excursion_ratio = total_excursion_points / data_length if data_length > 0 else 0.0

        # Factor 2: Persistencia normalizada
        max_possible_persistence = data_length * len(intervals) if intervals else 1
        normalized_persistence = total_persistence / max_possible_persistence if max_possible_persistence > 0 else 0.0

        # Asignar scores
        if excursion_ratio < 0.1 and normalized_persistence < 0.2:
            scores['STABLE'] = 0.8

        if has_noise and not has_features and not has_active:
            scores['NOISE'] = 0.7 + 0.3 * min(1.0, excursion_ratio)

        if has_features or (has_active and avg_lifespan > noise_limit):
            feature_score = 0.6
            feature_score += 0.2 * min(1.0, normalized_persistence)
            feature_score += 0.2 * min(1.0, persistence_entropy)
            scores['FEATURE'] = feature_score

        # Determinar estado final
        if scores['FEATURE'] >= 0.5:
            state = MetricState.FEATURE
            reason = "structural_features_detected"
            confidence = scores['FEATURE']
        elif scores['NOISE'] >= 0.5:
            state = MetricState.NOISE
            reason = "transient_noise_dominant"
            confidence = scores['NOISE']
        elif scores['STABLE'] >= 0.5:
            state = MetricState.STABLE
            reason = "system_stable"
            confidence = scores['STABLE']
        else:
            # Caso ambiguo: usar heurística conservadora
            if has_active:
                state = MetricState.FEATURE
                reason = "active_excursion_with_ambiguity"
            else:
                state = MetricState.NOISE
                reason = "ambiguous_pattern_default_to_noise"
            confidence = 0.4

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
                "confidence": round(confidence, 3),
                "excursion_ratio": round(excursion_ratio, 3),
                "normalized_persistence": round(normalized_persistence, 3),
                "persistence_entropy": round(persistence_entropy, 3),
                "noise_limit": noise_limit,
                "critical_limit": critical_limit,
                "data_length": data_length,
                "statistics": {
                    "mean": round(data_mean, 3),
                    "std": round(data_std, 3)
                }
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
        threshold: float,
        method: str = "wasserstein"
    ) -> float:
        """
        Compara dos diagramas de persistencia usando distancia seleccionada.

        Métodos disponibles:
        - "wasserstein": Distancia de Wasserstein-1 aproximada (default)
        - "bottleneck": Distancia bottleneck simplificada (máximo de diferencias)

        Args:
            metric1: Nombre de la primera métrica.
            metric2: Nombre de la segunda métrica.
            threshold: Umbral para generación de diagramas.
            method: Método de comparación ("wasserstein" o "bottleneck").

        Returns:
            Distancia entre diagramas (0 = idénticos, mayor = más diferentes).
            Retorna -1.0 si hay error en los parámetros.
        """
        # Validar nombres de métricas
        if not isinstance(metric1, str) or not isinstance(metric2, str):
            logger.warning(
                f"Nombres de métricas deben ser strings: "
                f"metric1={type(metric1).__name__}, metric2={type(metric2).__name__}"
            )
            return -1.0

        metric1 = metric1.strip()
        metric2 = metric2.strip()

        if not metric1 or not metric2:
            logger.warning("Nombres de métricas no pueden ser vacíos")
            return -1.0

        # Validar threshold
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            logger.warning(f"threshold debe ser numérico, recibido: {type(threshold).__name__}")
            return -1.0

        if math.isnan(threshold):
            logger.warning("threshold no puede ser NaN")
            return -1.0

        # Validar método
        valid_methods = {"wasserstein", "bottleneck"}
        if method not in valid_methods:
            logger.warning(
                f"Método '{method}' no válido. Usando 'wasserstein'. "
                f"Opciones: {valid_methods}"
            )
            method = "wasserstein"

        # Obtener diagramas
        diagram1 = self.get_persistence_diagram(metric1, threshold)
        diagram2 = self.get_persistence_diagram(metric2, threshold)

        # Casos especiales: ambos vacíos
        if not diagram1 and not diagram2:
            return 0.0

        # Extraer duraciones finitas
        lifespans1 = sorted([i.lifespan for i in diagram1 if not i.is_alive])
        lifespans2 = sorted([i.lifespan for i in diagram2 if not i.is_alive])

        # Ambos sin intervalos finitos
        if not lifespans1 and not lifespans2:
            # Comparar por número de intervalos activos
            active1 = len([i for i in diagram1 if i.is_alive])
            active2 = len([i for i in diagram2 if i.is_alive])
            return float(abs(active1 - active2))

        # Igualar longitudes con padding
        max_len = max(len(lifespans1), len(lifespans2))
        lifespans1.extend([0.0] * (max_len - len(lifespans1)))
        lifespans2.extend([0.0] * (max_len - len(lifespans2)))

        # Calcular distancia según método
        if method == "wasserstein":
            # Distancia de Wasserstein-1 (suma de diferencias absolutas)
            return sum(abs(l1 - l2) for l1, l2 in zip(lifespans1, lifespans2))

        elif method == "bottleneck":
            # Distancia bottleneck (máxima diferencia)
            if not lifespans1 or not lifespans2:
                # Uno vacío: máximo del otro
                return max(lifespans1 + lifespans2) if (lifespans1 or lifespans2) else 0.0
            return max(abs(l1 - l2) for l1, l2 in zip(lifespans1, lifespans2))

        # Fallback (no debería llegar aquí)
        return 0.0

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

    print("\nDatos de CPU simulados")
    stats = ph.get_statistics("cpu_usage")
    print(f"Estadísticas: {stats}")

    result = ph.analyze_persistence("cpu_usage", threshold=70)
    print("\nAnálisis (umbral=70):")
    print(f"  Estado: {result.state.name}")
    print(f"  Intervalos: {len(result.intervals)}")
    print(f"  Features: {result.feature_count}, Noise: {result.noise_count}")
    print(f"  Persistencia total: {result.total_persistence}")
    print(f"  Metadata: {result.metadata}")
