"""
Topological State Analyzer
==========================
Módulo para el análisis de la topología del sistema y homología persistente.
Calcula invariantes topológicos (números de Betti) y persistencia de errores
para una observabilidad más profunda.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from collections import deque
import networkx as nx

logger = logging.getLogger("TopologicalAnalyzer")

class SystemTopology:
    """
    Representa el estado de los servicios como un espacio topológico (Grafo).

    Servicios monitoreados (Nodos):
    - Agent
    - Core
    - Redis
    - Filesystem (implícito en operaciones de disco)
    """

    REQUIRED_NODES = {"Agent", "Core", "Redis", "Filesystem"}

    def __init__(self):
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.REQUIRED_NODES)

        # Historial de request_ids para detectar ciclos (b1)
        self.request_history = deque(maxlen=50)

    def update_connectivity(self, active_connections: List[Tuple[str, str]]):
        """
        Actualiza las conexiones del grafo basado en la telemetría reciente.

        Args:
            active_connections: Lista de pares (origen, destino) que están comunicándose.
                                Ej: [("Agent", "Core"), ("Core", "Redis")]
        """
        self.graph.clear_edges()
        self.graph.add_edges_from(active_connections)

    def record_request(self, request_id: str):
        """Registra un request_id para análisis de ciclos."""
        if request_id:
            self.request_history.append(request_id)

    def calculate_betti_numbers(self) -> Tuple[int, int]:
        """
        Calcula los números de Betti del sistema actual.

        b0: Número de componentes conexas.
            - b0 = 1: Sistema totalmente conectado (Ideal).
            - b0 > 1: Partición de red o servicio caído.

        b1: Número de ciclos (agujeros 1-dimensionales).
            - b1 = 0: Flujo acíclico (Ideal).
            - b1 > 0: Bucles de reintentos o redundancia circular ineficiente.

        Returns:
            Tuple (b0, b1)
        """
        # Cálculo de b0 (Componentes conexas)
        # Filtramos nodos que son parte del grafo base
        subgraph = self.graph.subgraph(self.REQUIRED_NODES)
        b0 = nx.number_connected_components(subgraph)

        # Cálculo de b1 (Ciclos en el flujo de datos)
        # Heurística: Si detectamos el mismo request_id repetido frecuentemente
        # en un periodo corto, lo consideramos un "ciclo de reintento" topológico.
        b1 = self._detect_cycles_in_history()

        return b0, b1

    def _detect_cycles_in_history(self) -> int:
        """
        Detecta ciclos lógicos basados en repetición de requests.
        Simula b1 en el espacio de fase del flujo de datos.
        """
        if not self.request_history:
            return 0

        # Contamos repeticiones
        counts = {}
        for req_id in self.request_history:
            counts[req_id] = counts.get(req_id, 0) + 1

        # Un ciclo se define como un request que aparece > 3 veces en la ventana reciente
        loops = sum(1 for count in counts.values() if count > 3)
        return loops

class PersistenceHomology:
    """
    Análisis de Homología Persistente para métricas de series temporales.
    Distingue entre "Ruido" (vida corta) y "Características Estructurales" (vida larga).
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.buffers: Dict[str, deque] = {}

    def add_reading(self, metric_name: str, value: float):
        """Agrega una lectura al buffer de una métrica."""
        if metric_name not in self.buffers:
            self.buffers[metric_name] = deque(maxlen=self.window_size)
        self.buffers[metric_name].append(value)

    def analyze_persistence(self, metric_name: str, threshold: float) -> str:
        """
        Analiza la persistencia de una métrica por encima de un umbral.

        Retorna:
            "STABLE": Métrica bajo control.
            "NOISE": Excursión breve sobre el umbral (muerte rápida).
            "FEATURE": Excursión persistente (vida larga, problema estructural).
        """
        buffer = self.buffers.get(metric_name)
        if not buffer or len(buffer) < 3:
            return "STABLE"

        data = list(buffer)

        # Identificar segmentos continuos por encima del umbral
        above_threshold = [x > threshold for x in data]

        if not any(above_threshold):
            return "STABLE"

        # Calcular la duración de la excursión más reciente
        # Recorremos hacia atrás
        duration = 0
        for is_above in reversed(above_threshold):
            if is_above:
                duration += 1
            else:
                break

        # Interpretación Topológica
        # Duración corta (< 20% ventana) = Ruido (barra corta)
        # Duración larga (>= 20% ventana) = Característica (barra larga)

        noise_limit = max(1, int(self.window_size * 0.2))

        if duration == 0:
            return "STABLE"
        elif duration < noise_limit:
            return "NOISE"
        else:
            return "FEATURE"
