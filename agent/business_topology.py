import logging
import textwrap
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse.linalg

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
        strategic_narrative (Optional[str]): Narrativa estratégica para decisores (La Voz del Consejo).
    """

    integrity_score: float
    waste_alerts: List[str]
    circular_risks: List[str]
    complexity_level: str
    details: Dict[str, Any] = field(default_factory=dict)
    financial_risk_level: Optional[str] = None
    strategic_narrative: Optional[str] = None


class BudgetGraphBuilder:
    """
    Construye el Grafo del Presupuesto (Topología de Negocio) Versión 2 con estructura Piramidal.
    Adopta la lógica de 'Upsert' y manejo jerárquico de la Propuesta 2.
    """

    def __init__(self):
        """Inicializa el constructor del grafo."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ROOT_NODE = "PROYECTO_TOTAL"

    def _sanitize_code(self, value: Any) -> str:
        """Sanitiza el código o identificador asegurando una cadena limpia y normalizada."""
        if pd.isna(value) or value is None:
            return ""
        sanitized = str(value).strip()
        sanitized = " ".join(sanitized.split())
        return sanitized

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """
        Convierte un valor a float de manera segura con soporte para formatos locales.

        Lógica de detección:
        1. Si hay ambos separadores: el último determina el decimal
        2. Si solo hay coma: se evalúa por posición y longitud de decimales
        """
        if pd.isna(value) or value is None:
            return default

        if isinstance(value, (int, float)):
            return float(value)

        try:
            str_value = str(value).strip().replace("\xa0", "").replace(" ", "")

            if not str_value:
                return default

            has_comma = "," in str_value
            has_dot = "." in str_value

            if has_comma and has_dot:
                # Determinar cuál es el separador decimal por posición
                last_comma = str_value.rfind(",")
                last_dot = str_value.rfind(".")

                if last_dot > last_comma:
                    # Formato: 1,234,567.89 (inglés)
                    str_value = str_value.replace(",", "")
                else:
                    # Formato: 1.234.567,89 (europeo)
                    str_value = str_value.replace(".", "").replace(",", ".")

            elif has_comma and not has_dot:
                parts = str_value.split(",")
                # Heurística: si la última parte tiene 1-3 dígitos, es decimal
                if len(parts) == 2 and 1 <= len(parts[1]) <= 3 and parts[1].isdigit():
                    str_value = str_value.replace(",", ".")
                else:
                    # Es separador de miles
                    str_value = str_value.replace(",", "")

            return float(str_value)

        except (ValueError, TypeError, AttributeError):
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
        """Inserta o actualiza una arista aplicando agregación de cantidades y costos (Upsert)."""
        total_cost = unit_cost * quantity

        if G.has_edge(u, v):
            edge = G[u][v]
            edge["quantity"] += quantity
            edge["total_cost"] += total_cost
            edge["occurrence_count"] += 1
            if "original_indices" not in edge:
                edge["original_indices"] = []
            edge["original_indices"].append(idx)
            return False

        G.add_edge(
            u,
            v,
            quantity=quantity,
            unit_cost=unit_cost,
            total_cost=total_cost,
            occurrence_count=1,
            original_indices=[idx],
        )
        return True

    def _compute_graph_statistics(self, G: nx.DiGraph) -> Dict[str, int]:
        stats = {
            "chapter_count": 0,
            "apu_count": 0,
            "insumo_count": 0,
            "inferred_count": 0,
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
        }

        type_counters = {
            "CAPITULO": "chapter_count",
            "APU": "apu_count",
            "INSUMO": "insumo_count",
        }

        for _, data in G.nodes(data=True):
            node_type = data.get("type")
            if node_type in type_counters:
                stats[type_counters[node_type]] += 1
                if node_type == "APU" and data.get("inferred", False):
                    stats["inferred_count"] += 1

        return stats

    def _process_presupuesto_row(
        self, G: nx.DiGraph, row: pd.Series, idx: int, chapter_cols: List[str]
    ) -> None:
        apu_code = self._sanitize_code(row.get(ColumnNames.CODIGO_APU))
        if not apu_code:
            return

        # Calcular Costo Total del APU para asignar peso a la arista
        # Intentar obtener VALOR_TOTAL_APU directo
        total_cost = self._safe_float(row.get(ColumnNames.VALOR_TOTAL_APU))
        if total_cost == 0.0:
            # Fallback: Cantidad * Precio Unitario
            qty = self._safe_float(row.get(ColumnNames.CANTIDAD_PRESUPUESTO))
            price = self._safe_float(row.get(ColumnNames.PRECIO_UNIT_APU))
            total_cost = qty * price

        # Crear nodo APU (Nivel 2)
        attrs = self._create_apu_attributes(
            row, source="presupuesto", idx=idx, inferred=False
        )
        G.add_node(apu_code, **attrs)

        # Buscar y establecer jerarquía de capítulo
        chapter_name = None
        for col in chapter_cols:
            val = self._sanitize_code(row.get(col))
            if val:
                chapter_name = val
                break

        if chapter_name:
            if chapter_name not in G:
                G.add_node(
                    chapter_name,
                    type="CAPITULO",
                    level=1,
                    description=f"Capítulo: {chapter_name}",
                )
                # Inicializar arista Root -> Capítulo con peso 0 (se acumulará)
                G.add_edge(
                    self.ROOT_NODE,
                    chapter_name,
                    relation="CONTAINS",
                    weight=0.0,
                    total_cost=0.0,
                )

            # Acumular costo en arista Root -> Capítulo
            if G.has_edge(self.ROOT_NODE, chapter_name):
                edge_rc = G[self.ROOT_NODE][chapter_name]
                edge_rc["weight"] = edge_rc.get("weight", 0.0) + total_cost
                edge_rc["total_cost"] = edge_rc.get("total_cost", 0.0) + total_cost

            # Arista Capítulo -> APU
            G.add_edge(
                chapter_name,
                apu_code,
                relation="CONTAINS",
                weight=total_cost,
                total_cost=total_cost,
            )
        else:
            # Arista Root -> APU
            G.add_edge(
                self.ROOT_NODE,
                apu_code,
                relation="CONTAINS",
                weight=total_cost,
                total_cost=total_cost,
            )

    def _process_apu_detail_row(self, G: nx.DiGraph, row: pd.Series, idx: int) -> None:
        apu_code = self._sanitize_code(row.get(ColumnNames.CODIGO_APU))
        insumo_desc = self._sanitize_code(row.get(ColumnNames.DESCRIPCION_INSUMO))

        if not apu_code or not insumo_desc:
            return

        # Inferir APU si no existe
        if apu_code not in G:
            attrs = self._create_apu_attributes(row, source="detail", idx=idx, inferred=True)
            G.add_node(apu_code, **attrs)
            G.add_edge(self.ROOT_NODE, apu_code, relation="CONTAINS_INFERRED")

        # Crear o reutilizar nodo Insumo
        if insumo_desc not in G:
            attrs = self._create_insumo_attributes(
                row, insumo_desc, source="detail", idx=idx
            )
            G.add_node(insumo_desc, **attrs)

        # Establecer relación APU -> Insumo con agregación
        qty = self._safe_float(row.get(ColumnNames.CANTIDAD_APU))
        cost = self._safe_float(row.get(ColumnNames.COSTO_INSUMO_EN_APU))
        self._upsert_edge(G, apu_code, insumo_desc, cost, qty, idx)

    def build(
        self, presupuesto_df: pd.DataFrame, apus_detail_df: pd.DataFrame
    ) -> nx.DiGraph:
        """
        Construye un grafo dirigido piramidal representando la topología del presupuesto.
        """
        G = nx.DiGraph(name="BudgetTopology")
        self.logger.info("Iniciando construcción del Grafo Piramidal de Presupuesto...")

        # Nivel 0: Nodo Raíz
        G.add_node(self.ROOT_NODE, type="ROOT", level=0, description="Proyecto Completo")

        # Columnas candidatas para identificar capítulos
        chapter_cols = ["CAPITULO", "CATEGORIA", "TITULO"]

        # Niveles 1 y 2: Procesar Presupuesto
        if presupuesto_df is not None and not presupuesto_df.empty:
            available_chapter_cols = [c for c in chapter_cols if c in presupuesto_df.columns]
            for idx, row in presupuesto_df.iterrows():
                self._process_presupuesto_row(G, row, idx, available_chapter_cols)

        # Nivel 3: Procesar Detalle de APUs (Insumos)
        if apus_detail_df is not None and not apus_detail_df.empty:
            for idx, row in apus_detail_df.iterrows():
                self._process_apu_detail_row(G, row, idx)

        stats = self._compute_graph_statistics(G)
        self.logger.info(f"Grafo Piramidal construido: {stats}")
        return G


class BusinessTopologicalAnalyzer:
    """
    Analizador de topología de negocio V2 con Telemetría Granular.
    Fusión Estratégica:
    - Motor Matemático: Propuesta 1 (El Cerebro Forense)
    - Narrativa: Propuesta 2 (La Voz del Consejo)
    """

    def __init__(self, telemetry: Optional[TelemetryContext] = None, max_cycles: int = 100):
        self.telemetry = telemetry
        self.max_cycles = max_cycles
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze_spectral_stability(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Calcula la Estabilidad Espectral y métricas derivadas del Laplaciano.

        Fundamentos Matemáticos:
        - Fiedler Value (λ₂): Segundo eigenvalor más pequeño del Laplaciano.
          Mide la conectividad algebraica. λ₂ > 0 ⟺ grafo conexo.
        - Gap Espectral: λ₂ / λ_max. Proxy de sincronizabilidad.
        - Energía Espectral: ||L||_F² = Σλᵢ². Mide dispersión estructural.
        - Longitud de Onda: 1/λ_max. Escala característica de difusión.

        Returns:
            Dict con métricas espectrales y diagnóstico de resonancia.
        """
        n_nodes = graph.number_of_nodes()

        default_result = {
            "fiedler_value": 0.0,
            "spectral_gap": 0.0,
            "spectral_energy": 0.0,
            "wavelength": float("inf"),
            "resonance_risk": False,
            "algebraic_connectivity": 0.0,
            "eigenvalues": [],
            "status": "insufficient_nodes",
        }

        if n_nodes < 2:
            return default_result

        try:
            # Convertir a no dirigido para análisis espectral simétrico
            ud_graph = graph.to_undirected()

            # Remover nodos aislados para estabilidad numérica del Laplaciano normalizado
            isolated = list(nx.isolates(ud_graph))
            if isolated:
                ud_graph = ud_graph.copy()
                ud_graph.remove_nodes_from(isolated)

            n_effective = ud_graph.number_of_nodes()
            if n_effective < 2:
                default_result["status"] = "degenerate_after_isolation_removal"
                return default_result

            # Laplaciano Normalizado: L_norm = I - D^(-1/2) A D^(-1/2)
            L = nx.normalized_laplacian_matrix(ud_graph).astype(np.float64)

            # Energía Espectral vía norma de Frobenius (eficiente, no requiere eigendecomposición)
            if scipy.sparse.issparse(L):
                spectral_energy = float(scipy.sparse.linalg.norm(L, "fro") ** 2)
            else:
                spectral_energy = float(np.linalg.norm(L, "fro") ** 2)

            # Cálculo de Eigenvalores
            if n_effective <= 50:
                # Decomposición completa para grafos pequeños
                eigenvalues = np.linalg.eigvalsh(L.toarray())
                eigenvalues = np.sort(np.real(eigenvalues))
            else:
                # Sparse solver para grafos grandes
                # SM (Smallest Magnitude) para obtener los más pequeños
                k_small = min(n_effective - 1, 6)
                try:
                    eigenvalues_small = scipy.sparse.linalg.eigsh(
                        L,
                        k=k_small,
                        which="SM",
                        return_eigenvectors=False,
                        tol=1e-6,
                        maxiter=n_effective * 100,
                    )
                except scipy.sparse.linalg.ArpackNoConvergence as e:
                    eigenvalues_small = e.eigenvalues

                # LM (Largest Magnitude) para λ_max
                try:
                    eigenvalues_large = scipy.sparse.linalg.eigsh(
                        L, k=1, which="LM", return_eigenvectors=False
                    )
                except scipy.sparse.linalg.ArpackNoConvergence as e:
                    eigenvalues_large = (
                        e.eigenvalues if len(e.eigenvalues) > 0 else np.array([2.0])
                    )

                eigenvalues = np.sort(
                    np.concatenate([eigenvalues_small, eigenvalues_large])
                )

            if len(eigenvalues) < 2:
                default_result["status"] = "insufficient_eigenvalues"
                return default_result

            # Fiedler Value: λ₂ (segundo eigenvalor más pequeño)
            # λ₁ ≈ 0 siempre para Laplaciano.
            # Si grafo es conexo, λ₂ > 0. Si desconectado, λ₂ ≈ 0.
            fiedler_value = float(eigenvalues[1]) if len(eigenvalues) >= 2 else 0.0

            # Limpiar ruido numérico negativo
            if fiedler_value < 1e-9:
                fiedler_value = 0.0

            # λ_max (para Laplaciano normalizado, está acotado por 2)
            lambda_max = float(eigenvalues[-1]) if len(eigenvalues) > 0 else 2.0
            lambda_max = min(lambda_max, 2.0)  # Cota teórica

            # Gap Espectral Normalizado
            spectral_gap = fiedler_value / lambda_max if lambda_max > 1e-10 else 0.0

            # Longitud de Onda (escala de difusión característica)
            wavelength = 1.0 / lambda_max if lambda_max > 1e-10 else float("inf")

            # Diagnóstico de Resonancia: espectro degenerado indica vulnerabilidad
            # Coeficiente de variación bajo indica concentración espectral
            if len(eigenvalues) > 2:
                eigen_std = np.std(eigenvalues)
                eigen_mean = np.mean(eigenvalues)
                cv = eigen_std / eigen_mean if eigen_mean > 1e-10 else 0.0
                resonance_risk = cv < 0.15
            else:
                resonance_risk = True  # Muy pocos eigenvalores = alto riesgo

            return {
                "fiedler_value": round(fiedler_value, 6),
                "spectral_gap": round(spectral_gap, 6),
                "spectral_energy": round(spectral_energy, 4),
                "wavelength": round(wavelength, 6),
                "resonance_risk": bool(resonance_risk),
                "algebraic_connectivity": round(fiedler_value, 6),
                "lambda_max": round(lambda_max, 6),
                "isolated_nodes_removed": len(isolated),
                "eigenvalues": [round(float(e), 6) for e in eigenvalues[:8]],
                "status": "success",
            }

        except Exception as e:
            self.logger.error(f"Error en análisis espectral: {e}", exc_info=True)
            default_result["status"] = f"error: {str(e)}"
            return default_result

    def calculate_euler_efficiency(self, graph: nx.DiGraph) -> float:
        """Calcula la Eficiencia de Euler normalizada mediante decaimiento exponencial (Propuesta 1)."""
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()

        if n_nodes <= 1:
            return 1.0

        min_edges = n_nodes - 1
        excess_edges = max(0, n_edges - min_edges)
        efficiency = np.exp(-excess_edges / n_nodes) if n_nodes > 0 else 1.0
        return round(efficiency, 4)

    def calculate_betti_numbers(self, graph: nx.DiGraph) -> TopologicalMetrics:
        """Calcula métricas topológicas invariantes (Números de Betti)."""
        if graph.number_of_nodes() == 0:
            return TopologicalMetrics(0, 0, 0, 1.0)

        # Usar MultiGraph para preservar todas las aristas y calcular correctamente Betti_1
        undirected = nx.MultiGraph()
        undirected.add_nodes_from(graph.nodes(data=True))
        undirected.add_edges_from(graph.edges(data=True))

        beta_0 = nx.number_connected_components(undirected)
        n_edges = undirected.number_of_edges()
        n_nodes = undirected.number_of_nodes()

        beta_1 = max(0, n_edges - n_nodes + beta_0)
        euler_char = beta_0 - beta_1
        efficiency = self.calculate_euler_efficiency(graph)

        return TopologicalMetrics(
            beta_0=beta_0,
            beta_1=beta_1,
            euler_characteristic=euler_char,
            euler_efficiency=efficiency,
        )

    def calculate_pyramid_stability(self, graph: nx.DiGraph) -> float:
        """
        Calcula el Índice de Estabilidad Piramidal (Ψ) normalizado.

        Modelo Físico:
        - Una pirámide estable tiene base ancha (muchos insumos) y cúspide estrecha (pocos APUs).
        - Ψ ∈ [0, 1] donde 1 = máxima estabilidad.

        Componentes:
        1. Ratio Base/Altura: log₁₀(1 + insumos/APUs)
        2. Factor de Densidad: penaliza grafos muy densos (inestables)
        3. Factor de Aciclicidad: ciclos reducen estabilidad estructural
        4. Factor de Conectividad: fragmentación reduce estabilidad

        Returns:
            float: Índice Ψ normalizado en [0, 1]
        """
        if graph.number_of_nodes() == 0:
            return 0.0

        # Conteo por tipo de nodo
        type_counts = {"APU": 0, "INSUMO": 0, "CAPITULO": 0}
        for _, data in graph.nodes(data=True):
            node_type = data.get("type", "")
            if node_type in type_counts:
                type_counts[node_type] += 1

        num_apus = type_counts["APU"]
        num_insumos = type_counts["INSUMO"]

        # Caso degenerado: sin APUs o sin insumos
        if num_apus == 0:
            return (
                0.0 if num_insumos == 0 else 0.5
            )  # Solo insumos: parcialmente estable

        if num_insumos == 0:
            return 0.3  # Solo APUs sin detalle: baja estabilidad

        # 1. Ratio de Base (normalizado con log para escalar grandes diferencias)
        base_ratio = num_insumos / num_apus
        # Normalizar: ratio ideal ~5-10 insumos por APU
        ratio_score = min(1.0, np.log10(1 + base_ratio) / np.log10(11))  # log10(11) ≈ 1.04

        # 2. Factor de Densidad
        density = nx.density(graph)
        # Densidad óptima para DAG jerárquico: ~0.01-0.1
        # Penalizar densidades muy altas (> 0.3) o muy bajas (< 0.001)
        if density < 0.001:
            density_score = 0.5  # Muy disperso
        elif density > 0.5:
            density_score = 0.3  # Muy denso
        else:
            density_score = 1.0 - min(0.7, density)

        # 3. Factor de Aciclicidad
        is_dag = nx.is_directed_acyclic_graph(graph)
        if is_dag:
            acyclic_score = 1.0
        else:
            # Penalización proporcional al número de SCCs no triviales
            sccs = [c for c in nx.strongly_connected_components(graph) if len(c) > 1]
            cycle_penalty = min(0.5, len(sccs) * 0.1)
            acyclic_score = 0.5 - cycle_penalty

        # 4. Factor de Conectividad
        if nx.is_weakly_connected(graph):
            connectivity_score = 1.0
        else:
            num_components = nx.number_weakly_connected_components(graph)
            connectivity_score = 1.0 / num_components

        # Combinación ponderada
        weights = {
            "ratio": 0.35,
            "density": 0.20,
            "acyclic": 0.30,
            "connectivity": 0.15,
        }

        stability = (
            weights["ratio"] * ratio_score
            + weights["density"] * density_score
            + weights["acyclic"] * acyclic_score
            + weights["connectivity"] * connectivity_score
        )

        return round(max(0.0, min(1.0, stability)), 4)

    def audit_integration_homology(
        self, graph_a: nx.DiGraph, graph_b: nx.DiGraph
    ) -> Dict[str, Any]:
        """
        Ejecuta el Test de Mayer-Vietoris para auditoría de fusión topológica.

        Teorema de Mayer-Vietoris (simplificado para grafos):
        Para X = A ∪ B con intersección A ∩ B:

        β₁(X) ≤ β₁(A) + β₁(B) + β₀(A∩B) - 1  (cota superior)

        La emergencia de nuevos ciclos indica conflicto de integración.

        Returns:
            Dict con diagnóstico de fusión y narrativa.
        """
        # Calcular métricas individuales
        metrics_a = self.calculate_betti_numbers(graph_a)
        metrics_b = self.calculate_betti_numbers(graph_b)

        # Construir unión
        graph_union = nx.compose(graph_a, graph_b)
        metrics_union = self.calculate_betti_numbers(graph_union)

        # Construir intersección (subgrafo inducido por nodos comunes)
        nodes_a = set(graph_a.nodes())
        nodes_b = set(graph_b.nodes())
        common_nodes = nodes_a & nodes_b

        # Intersección: aristas que existen en AMBOS grafos
        graph_intersection = nx.DiGraph()
        if common_nodes:
            graph_intersection.add_nodes_from((n, graph_a.nodes[n]) for n in common_nodes)
            # Solo aristas presentes en ambos grafos
            for u, v in graph_a.edges():
                if u in common_nodes and v in common_nodes and graph_b.has_edge(u, v):
                    graph_intersection.add_edge(u, v)

        metrics_intersection = self.calculate_betti_numbers(graph_intersection)

        # Fórmula de Mayer-Vietoris para β₁
        # β₁(A∪B) = β₁(A) + β₁(B) - β₁(A∩B) + δ
        # donde δ = β₀(A∩B) - β₀(A) - β₀(B) + β₀(A∪B)

        # Cálculo del término de conexión
        delta_connectivity = (
            metrics_intersection.beta_0
            - metrics_a.beta_0
            - metrics_b.beta_0
            + metrics_union.beta_0
        )

        # Valor teórico esperado de β₁ en la unión
        beta1_theoretical = (
            metrics_a.beta_1
            + metrics_b.beta_1
            - metrics_intersection.beta_1
            + max(0, delta_connectivity)
        )

        # Emergencia: diferencia entre observado y suma simple
        emergent_observed = metrics_union.beta_1 - (metrics_a.beta_1 + metrics_b.beta_1)

        # Discrepancia respecto al modelo teórico
        discrepancy = abs(metrics_union.beta_1 - beta1_theoretical)

        # Determinación del veredicto
        if discrepancy <= 1:
            if emergent_observed > 0:
                verdict = "INTEGRATION_CONFLICT"
                severity = "warning"
            elif emergent_observed < 0:
                verdict = "TOPOLOGY_SIMPLIFIED"
                severity = "info"
            else:
                verdict = "CLEAN_MERGE"
                severity = "success"
        else:
            verdict = "INCONSISTENT_TOPOLOGY"
            severity = "error"

        # Análisis de interfaz (nodos de frontera)
        boundary_nodes = []
        for node in common_nodes:
            # Nodo frontera: tiene aristas hacia nodos exclusivos de A o B
            neighbors_a = set(graph_a.successors(node)) | set(
                graph_a.predecessors(node)
            )
            neighbors_b = set(graph_b.successors(node)) | set(
                graph_b.predecessors(node)
            )
            exclusive_neighbors = (neighbors_a - nodes_b) | (neighbors_b - nodes_a)
            if exclusive_neighbors:
                boundary_nodes.append(node)

        narrative = self._generate_mayer_vietoris_narrative(
            emergent_observed, discrepancy, len(boundary_nodes)
        )

        return {
            "status": verdict,
            "severity": severity,
            "delta_beta_1": emergent_observed,
            "beta_1_observed": metrics_union.beta_1,
            "beta_1_theoretical": beta1_theoretical,
            "discrepancy": discrepancy,
            "boundary_nodes": boundary_nodes,
            "details": {
                "beta_1_A": metrics_a.beta_1,
                "beta_1_B": metrics_b.beta_1,
                "beta_1_intersection": metrics_intersection.beta_1,
                "beta_1_union": metrics_union.beta_1,
                "beta_0_A": metrics_a.beta_0,
                "beta_0_B": metrics_b.beta_0,
                "beta_0_intersection": metrics_intersection.beta_0,
                "beta_0_union": metrics_union.beta_0,
                "common_nodes_count": len(common_nodes),
                "boundary_nodes_count": len(boundary_nodes),
                "delta_connectivity": delta_connectivity,
            },
            "narrative": narrative,
        }

    def _generate_mayer_vietoris_narrative(
        self, observed: int, discrepancy: float, boundary_count: int
    ) -> str:
        """Genera narrativa contextualizada del análisis Mayer-Vietoris."""
        parts = []

        if discrepancy > 2:
            parts.append(
                f"⚠️ ANOMALÍA TOPOLÓGICA: Discrepancia significativa (Δ={discrepancy:.1f}). "
                f"La estructura combinada no corresponde al modelo teórico. "
                f"Revisar coherencia de datos en {boundary_count} nodos de frontera."
            )
        elif discrepancy > 1:
            parts.append(
                f"⚠️ Discrepancia menor detectada (Δ={discrepancy:.1f}). "
                f"Posible redundancia en la interfaz de fusión."
            )

        if observed > 0:
            parts.append(
                f"🚨 CONFLICTO DE INTEGRACIÓN: La fusión generó {observed} nuevo(s) ciclo(s) "
                f"de dependencia no presentes en los componentes originales. "
                f"Esto indica incompatibilidad estructural en la interfaz."
            )
        elif observed < 0:
            parts.append(
                f"✅ SIMPLIFICACIÓN TOPOLÓGICA: La fusión eliminó {abs(observed)} ciclo(s) "
                f"redundante(s). La estructura combinada es más eficiente."
            )
        elif discrepancy <= 1:
            parts.append(
                "✅ FUSIÓN LIMPIA: No se detectaron conflictos estructurales. "
                "La integración es topológicamente neutral."
            )

        return " ".join(parts) if parts else "Análisis completado sin observaciones."

    def _get_raw_cycles(self, graph: nx.DiGraph) -> Tuple[List[List[str]], bool]:
        """Obtiene los ciclos crudos con algoritmo Johnson optimizado (Propuesta 1)."""
        cycles = []
        truncated = False
        try:
            cycle_generator = nx.simple_cycles(graph)
            max_cycle_length = 10
            for count, cycle in enumerate(cycle_generator):
                if count >= self.max_cycles:
                    truncated = True
                    self.logger.warning(f"Truncado de ciclos en {self.max_cycles}")
                    break
                if len(cycle) <= max_cycle_length:
                    cycles.append(cycle)
        except Exception as e:
            self.logger.error(f"Error en detección de ciclos: {e}")

        cycles.sort(key=len)
        return cycles, truncated

    def _detect_cycles(self, graph: nx.DiGraph) -> Tuple[List[str], bool]:
        """
        Detecta y formatea ciclos en el grafo (Compatibilidad hacia atrás).
        """
        raw_cycles, truncated = self._get_raw_cycles(graph)
        formatted_cycles = [" → ".join(map(str, c + [c[0]])) for c in raw_cycles]
        return formatted_cycles, truncated

    def detect_risk_synergy(
        self, graph: nx.DiGraph, raw_cycles: Optional[List[List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Detecta Sinergia de Riesgo: ciclos que comparten nodos críticos.

        Concepto:
        Si dos ciclos de dependencia comparten un nodo de alta centralidad,
        un fallo en ese nodo dispara ambos ciclos simultáneamente (efecto dominó).

        Métricas:
        - Nodos Puente: nodos de alta centralidad en múltiples ciclos.
        - Score de Sinergia: probabilidad relativa de fallo en cascada.

        Returns:
            Dict con detección de sinergia, nodos puente y score normalizado.
        """
        if raw_cycles is None:
            raw_cycles, _ = self._get_raw_cycles(graph)

        default_result = {
            "synergy_detected": False,
            "shared_nodes": [],
            "intersecting_cycles_count": 0,
            "bridge_nodes": [],
            "synergy_score": 0.0,
            "risk_level": "NINGUNO",
            "details": {},
        }

        if len(raw_cycles) < 2:
            return default_result

        # Calcular Betweenness Centrality
        try:
            if graph.number_of_nodes() > 1000:
                # Aproximación para grafos grandes
                betweenness = nx.betweenness_centrality(
                    graph, normalized=True, k=min(100, graph.number_of_nodes())
                )
            else:
                betweenness = nx.betweenness_centrality(graph, normalized=True)
        except Exception as e:
            self.logger.warning(f"Error calculando betweenness: {e}")
            betweenness = {n: 0.0 for n in graph.nodes()}

        # Umbral adaptativo para nodos críticos
        if betweenness:
            bc_values = list(betweenness.values())
            if len(bc_values) >= 4:
                threshold = np.percentile(bc_values, 75)
            else:
                threshold = np.mean(bc_values)
        else:
            threshold = 0.0

        critical_nodes = {n for n, c in betweenness.items() if c >= threshold and c > 0}

        # Analizar intersecciones de ciclos
        cycle_sets = [set(c) for c in raw_cycles]
        n_cycles = len(cycle_sets)

        synergy_pairs = []
        bridge_node_occurrences: Dict[str, List[Tuple[int, int]]] = {}

        for i in range(n_cycles):
            for j in range(i + 1, n_cycles):
                intersection = cycle_sets[i] & cycle_sets[j]

                # Sinergia significativa: ≥2 nodos compartidos o 1 nodo crítico
                critical_intersection = intersection & critical_nodes

                if len(intersection) >= 2 or critical_intersection:
                    synergy_pairs.append((i, j, intersection))

                    for node in intersection:
                        if node not in bridge_node_occurrences:
                            bridge_node_occurrences[node] = []
                        bridge_node_occurrences[node].append((i, j))

        if not synergy_pairs:
            return default_result

        # Identificar nodos puente ordenados por impacto
        bridge_nodes = sorted(
            [
                {
                    "id": node,
                    "cycles_connected": len(occurrences),
                    "betweenness": round(betweenness.get(node, 0), 4),
                    "is_critical": node in critical_nodes,
                }
                for node, occurrences in bridge_node_occurrences.items()
            ],
            key=lambda x: (x["cycles_connected"], x["betweenness"]),
            reverse=True,
        )

        # Score de Sinergia normalizado [0, 1]
        # Componentes:
        # 1. Ratio de pares con sinergia vs total posible
        total_pairs = n_cycles * (n_cycles - 1) / 2
        pair_ratio = len(synergy_pairs) / total_pairs if total_pairs > 0 else 0

        # 2. Concentración de puentes críticos
        critical_bridges = sum(1 for b in bridge_nodes if b["is_critical"])
        critical_ratio = critical_bridges / len(bridge_nodes) if bridge_nodes else 0

        # 3. Promedio de conexiones por puente
        avg_connections = (
            np.mean([b["cycles_connected"] for b in bridge_nodes]) if bridge_nodes else 0
        )
        connection_factor = min(1.0, avg_connections / 3)  # Normalizar a 3 conexiones

        synergy_score = (
            0.4 * pair_ratio + 0.35 * critical_ratio + 0.25 * connection_factor
        )
        synergy_score = round(min(1.0, synergy_score), 4)

        # Nivel de riesgo
        if synergy_score > 0.6:
            risk_level = "CRÍTICO"
        elif synergy_score > 0.3:
            risk_level = "ALTO"
        elif synergy_score > 0.1:
            risk_level = "MEDIO"
        else:
            risk_level = "BAJO"

        return {
            "synergy_detected": True,
            "shared_nodes": list(bridge_node_occurrences.keys()),
            "intersecting_cycles_count": len(synergy_pairs),
            "bridge_nodes": bridge_nodes[:10],  # Top 10
            "synergy_score": synergy_score,
            "risk_level": risk_level,
            "details": {
                "total_cycles": n_cycles,
                "synergy_pairs": len(synergy_pairs),
                "critical_bridges": critical_bridges,
                "pair_ratio": round(pair_ratio, 4),
            },
        }

    def analyze_thermal_flow(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Calcula el Flujo Térmico Estructural (Modelo de Difusión de Riesgo).

        Modelo Físico:
        1. Temperatura Base: asignada a insumos según volatilidad histórica.
        2. Conducción Térmica: el calor fluye de hojas hacia la raíz, ponderado por costo.
        3. Temperatura Sistémica: promedio ponderado en el nodo raíz.

        La "temperatura" es un proxy de la sensibilidad del presupuesto a variaciones
        de precio en insumos volátiles (combustibles, acero, etc.).

        Returns:
            Dict con temperatura del sistema, hotspots y gradiente térmico completo.
        """
        # Definición de temperaturas base por tipo de insumo (escala 0-100)
        # Basado en volatilidad histórica de precios en construcción
        BASE_TEMPERATURES = {
            "COMBUSTIBLE": 95.0, "GASOLINA": 95.0, "DIESEL": 95.0, "ACPM": 95.0,
            "ASFALTO": 90.0, "BITUMEN": 90.0,
            "ACERO": 85.0, "HIERRO": 85.0, "VARILLA": 85.0, "ALAMBRE": 80.0,
            "COBRE": 88.0, "ALUMINIO": 82.0,
            "CEMENTO": 60.0, "CONCRETO": 55.0, "AGREGADO": 45.0, "ARENA": 40.0,
            "TRANSPORTE": 75.0, "FLETE": 75.0, "ACARREO": 70.0,
            "MAQUINARIA": 50.0, "EQUIPO": 45.0, "HERRAMIENTA": 35.0,
            "MANO DE OBRA": 25.0, "OFICIAL": 25.0, "AYUDANTE": 20.0,
            "MADERA": 55.0, "FORMALETA": 50.0,
        }
        DEFAULT_TEMP = 30.0

        def get_base_temperature(description: str, tipo: str) -> float:
            """Determina temperatura base por matching de keywords."""
            text = f"{description} {tipo}".upper()
            matched_temp = DEFAULT_TEMP
            for keyword, temp in BASE_TEMPERATURES.items():
                if keyword in text:
                    matched_temp = max(matched_temp, temp)
            return matched_temp

        # Inicializar estructuras
        node_temperatures: Dict[str, float] = {}
        node_costs: Dict[str, float] = {}

        # Paso 1: Asignar temperatura base a nodos INSUMO (hojas)
        for node, data in graph.nodes(data=True):
            if data.get("type") == "INSUMO":
                desc = str(data.get("description", ""))
                tipo = str(data.get("tipo_insumo", ""))
                node_temperatures[node] = get_base_temperature(desc, tipo)

                # Costo total del insumo = suma de aristas entrantes (desde APUs)
                total_cost = sum(
                    graph[pred][node].get("total_cost", 0.0)
                    for pred in graph.predecessors(node)
                )
                node_costs[node] = max(total_cost, data.get("unit_cost", 0.0))
            else:
                node_temperatures[node] = 0.0
                node_costs[node] = 0.0

        # Paso 2: Propagación hacia arriba (Insumos → APUs → Capítulos → Root)
        # Usar BFS desde hojas o Topological Sort inverso
        try:
            # Orden topológico: raíz primero, hojas al final
            # Invertir para procesar hojas primero
            topo_order = list(reversed(list(nx.topological_sort(graph))))
        except nx.NetworkXUnfeasible:
            # Si hay ciclos, usar ordenamiento por nivel (heurístico)
            topo_order = sorted(
                graph.nodes(),
                key=lambda n: graph.nodes[n].get("level", 0),
                reverse=True
            )

        for node in topo_order:
            node_type = graph.nodes[node].get("type", "")

            # Los insumos ya tienen temperatura asignada
            if node_type == "INSUMO":
                continue

            # Calcular temperatura ponderada de hijos (sucesores)
            children = list(graph.successors(node))
            if not children:
                node_temperatures[node] = DEFAULT_TEMP * 0.5
                continue

            weighted_sum = 0.0
            cost_sum = 0.0

            for child in children:
                # El peso es el costo de la arista padre→hijo
                edge_data = graph[node][child]
                edge_cost = edge_data.get("total_cost", 0.0)
                if edge_cost == 0.0:
                    edge_cost = edge_data.get("weight", 0.0)
                if edge_cost == 0.0:
                    # Fallback: usar costo del hijo
                    edge_cost = node_costs.get(child, 1.0)

                child_temp = node_temperatures.get(child, DEFAULT_TEMP)

                weighted_sum += child_temp * edge_cost
                cost_sum += edge_cost

            if cost_sum > 0:
                node_temperatures[node] = weighted_sum / cost_sum
                node_costs[node] = cost_sum
            else:
                node_temperatures[node] = DEFAULT_TEMP * 0.5

        # Paso 3: Temperatura del Sistema (nodo ROOT o promedio global)
        root_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "ROOT"]

        if root_nodes:
            system_temp = node_temperatures.get(root_nodes[0], 0.0)
        else:
            # Promedio ponderado de APUs
            apu_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "APU"]
            if apu_nodes:
                total_cost = sum(node_costs.get(n, 0) for n in apu_nodes)
                if total_cost > 0:
                    system_temp = sum(
                        node_temperatures[n] * node_costs[n]
                        for n in apu_nodes
                    ) / total_cost
                else:
                    system_temp = np.mean([node_temperatures[n] for n in apu_nodes])
            else:
                system_temp = 0.0

        # Paso 4: Identificar Hotspots (Top N por temperatura, con costo significativo)
        # Filtrar nodos con costo > percentil 10 para evitar ruido
        cost_threshold = np.percentile(
            [c for c in node_costs.values() if c > 0], 10
        ) if any(c > 0 for c in node_costs.values()) else 0

        candidate_hotspots = [
            (node, temp) for node, temp in node_temperatures.items()
            if node_costs.get(node, 0) > cost_threshold
        ]
        candidate_hotspots.sort(key=lambda x: x[1], reverse=True)

        hotspots = [
            {
                "id": node,
                "temperature": round(temp, 1),
                "type": graph.nodes[node].get("type", ""),
                "cost": round(node_costs.get(node, 0), 2),
                "description": graph.nodes[node].get("description", "")[:50]
            }
            for node, temp in candidate_hotspots[:10]
        ]

        # Clasificación del riesgo térmico
        if system_temp > 70:
            thermal_risk_level = "CRÍTICO"
        elif system_temp > 50:
            thermal_risk_level = "ALTO"
        elif system_temp > 35:
            thermal_risk_level = "MEDIO"
        else:
            thermal_risk_level = "BAJO"

        return {
            "system_temperature": round(system_temp, 2),
            "thermal_risk_level": thermal_risk_level,
            "hotspots": hotspots,
            "thermal_gradient": {k: round(v, 2) for k, v in node_temperatures.items()},
            "cost_distribution": {k: round(v, 2) for k, v in node_costs.items()},
            "stats": {
                "max_temperature": round(max(node_temperatures.values()), 2) if node_temperatures else 0,
                "min_temperature": round(min(node_temperatures.values()), 2) if node_temperatures else 0,
                "std_temperature": round(float(np.std(list(node_temperatures.values()))), 2) if node_temperatures else 0,
            }
        }

    def analyze_inflationary_convection(
        self, graph: nx.DiGraph, fluid_nodes: List[str]
    ) -> Dict[str, Any]:
        """
        Analiza el contagio de calor (inflación) por convección.

        Args:
            graph: El grafo del presupuesto.
            fluid_nodes: Lista de nodos que actúan como fluido (ej. 'TRANSPORTE', 'COMBUSTIBLE').

        Returns:
            Mapa de calor convectivo.
        """
        # 1. Identificar nodos "bañados" por el fluido
        # (Nodos que tienen una arista entrante desde un nodo de transporte)
        affected_nodes = set()
        for fluid in fluid_nodes:
            if fluid in graph:
                # Sucesores del transporte (a quién afecta el transporte)
                # Ojo: En grafos de dependencia, A depende de B.
                # Si Muro depende de Transporte, Transporte es sucesor de Muro en flujo de costo?
                # Depende de la dirección del grafo. Asumamos APU -> Insumo.
                # Entonces si APU tiene arista a Transporte, el APU es afectado.
                # Pero en topología normal, APU contiene insumo, APU -> Insumo.
                # Si transporte sube precio, afecta al costo de APU.
                # Entonces debemos buscar predecesores (quien contiene al transporte).
                predecessors = list(graph.predecessors(fluid))
                affected_nodes.update(predecessors)

        # 2. Calcular Coeficiente de Transferencia de Calor (h)
        # h = % del costo que corresponde al fluido
        convection_impact = {}
        for node in affected_nodes:
            # Calcular peso del transporte en el nodo
            # Peso total del nodo es suma de costos de sus hijos? No necesariamente.
            # Estimación simple: costo total de aristas salientes del nodo es su costo directo.
            # O usar 'quantity' * 'unit_cost' de las aristas.

            # En graph.edges[u, v], tenemos total_cost.
            # Costo total del APU ≈ suma(total_cost de aristas salientes)
            total_cost_node = sum(
                graph[node][succ].get("total_cost", 0.0)
                for succ in graph.successors(node)
            )

            fluid_cost = 0.0
            for f in fluid_nodes:
                if graph.has_edge(node, f):
                    fluid_cost += graph[node][f].get("total_cost", 0.0)

            h_coefficient = (
                fluid_cost / total_cost_node if total_cost_node > 0 else 0.0
            )
            convection_impact[node] = h_coefficient

        high_risk_nodes = [n for n, h in convection_impact.items() if h > 0.2]

        return {
            "affected_nodes_count": len(affected_nodes),
            "average_convection_coefficient": sum(convection_impact.values())
            / len(affected_nodes)
            if affected_nodes
            else 0,
            "high_risk_nodes": high_risk_nodes,
            "convection_impact": convection_impact,
        }

    def _compute_connectivity_analysis(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Calcula métricas de conectividad avanzadas (Propuesta 1)."""
        if graph.number_of_nodes() == 0:
            return {
                "is_dag": True,
                "num_wcc": 0,
                "is_weakly_connected": True,
                "num_scc": 0,
                "num_non_trivial_scc": 0,
                "scc_sizes": [],
                "non_trivial_scc": [],
                "articulation_points": [],
                "average_clustering": 0.0,
            }

        undirected = graph.to_undirected()
        scc = list(nx.strongly_connected_components(graph))
        non_trivial_scc = [c for c in scc if len(c) > 1]
        articulation_points = list(nx.articulation_points(undirected))

        try:
            avg_clustering = nx.average_clustering(undirected)
        except:
            avg_clustering = 0.0

        return {
            "is_dag": nx.is_directed_acyclic_graph(graph),
            "num_wcc": nx.number_weakly_connected_components(graph),
            "is_weakly_connected": nx.is_weakly_connected(graph),
            "num_scc": len(non_trivial_scc),
            "num_non_trivial_scc": len(non_trivial_scc),  # Alias for compat
            "scc_sizes": [len(c) for c in non_trivial_scc],
            "non_trivial_scc": [list(c) for c in non_trivial_scc],
            "articulation_points": articulation_points,
            "average_clustering": round(avg_clustering, 4),
        }

    def _classify_anomalous_nodes(
        self, graph: nx.DiGraph
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Clasifica nodos anómalos."""
        result = {"isolated_nodes": [], "orphan_insumos": [], "empty_apus": []}
        in_degrees = dict(graph.in_degree())
        out_degrees = dict(graph.out_degree())

        for node, data in graph.nodes(data=True):
            node_type = data.get("type")
            if node_type == "ROOT":
                continue

            in_deg = in_degrees.get(node, 0)
            out_deg = out_degrees.get(node, 0)

            node_info = {
                "id": node,
                "type": node_type,
                "description": data.get("description", ""),
                "inferred": data.get("inferred", False),
                "in_degree": in_deg,
                "out_degree": out_deg,
            }

            is_isolated = in_deg == 0 and out_deg == 0
            if is_isolated:
                result["isolated_nodes"].append(node_info)
                if node_type == "INSUMO":
                    result["orphan_insumos"].append(node_info)
            elif node_type == "INSUMO" and in_deg == 0:
                result["orphan_insumos"].append(node_info)
            elif node_type == "APU" and out_deg == 0:
                result["empty_apus"].append(node_info)
        return result

    def _identify_critical_resources(
        self, graph: nx.DiGraph, top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Identifica recursos (insumos) críticos por centralidad de grado."""
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

    def _interpret_topology(self, metrics: TopologicalMetrics) -> Dict[str, str]:
        """Genera interpretaciones semánticas (Compatibilidad hacia atrás)."""
        connectivity_status = (
            "Espacio conexo" if metrics.is_connected else "Espacio fragmentado"
        )
        cycle_status = (
            "Estructura acíclica" if metrics.beta_1 == 0 else "Complejidad cíclica presente"
        )

        return {
            "beta_0": f"{metrics.beta_0} componente(s) conexa(s). {connectivity_status}.",
            "beta_1": f"{metrics.beta_1} ciclo(s) independiente(s). {cycle_status}.",
            "euler": f"Característica de Euler: χ = {metrics.euler_characteristic}",
            "efficiency": f"Eficiencia de Euler: {metrics.euler_efficiency:.2%}",
        }

    def generate_executive_report(
        self, graph: nx.DiGraph, financial_metrics: Optional[Dict[str, Any]] = None
    ) -> ConstructionRiskReport:
        """
        Genera reporte ejecutivo con modelo de scoring bayesiano multi-factor.

        Factores del Score de Integridad:
        1. Eficiencia de Euler (topología)
        2. Estabilidad Piramidal (estructura)
        3. Factor de Densidad (complejidad)
        4. Factor Espectral (resonancia)

        Penalizaciones:
        - Ciclos de dependencia
        - Sinergia de riesgo
        - Anomalías (nodos aislados, huérfanos)
        - Riesgo de resonancia espectral
        """
        # Cálculos base
        metrics = self.calculate_betti_numbers(graph)
        raw_cycles, truncated = self._get_raw_cycles(graph)
        cycles = [" → ".join(c + [c[0]]) for c in raw_cycles]

        synergy = self.detect_risk_synergy(graph, raw_cycles)
        anomalies = self._classify_anomalous_nodes(graph)
        pyramid_stability = self.calculate_pyramid_stability(graph)
        connectivity = self._compute_connectivity_analysis(graph)
        spectral = self.analyze_spectral_stability(graph)

        # Detección de fluidos convectivos
        fluid_keywords = [
            "TRANSPORTE",
            "COMBUSTIBLE",
            "FLETE",
            "ACARREO",
            "GASOLINA",
            "DIESEL",
            "ACPM",
            "ASFALTO",
        ]
        fluid_nodes = [
            n
            for n, d in graph.nodes(data=True)
            if d.get("type") == "INSUMO"
            and any(k in str(d.get("description", "")).upper() for k in fluid_keywords)
        ]
        convection = self.analyze_inflationary_convection(graph, fluid_nodes)
        thermal = self.analyze_thermal_flow(graph)

        # ═══════════════════════════════════════════════════════════════
        # MODELO DE SCORING BAYESIANO MULTI-FACTOR
        # ═══════════════════════════════════════════════════════════════

        density = nx.density(graph) if graph.number_of_nodes() > 0 else 0.0

        # Factor 1: Eficiencia de Euler (0-1)
        euler_factor = metrics.euler_efficiency

        # Factor 2: Estabilidad Piramidal (ya normalizado 0-1)
        stability_factor = pyramid_stability

        # Factor 3: Densidad Óptima (penaliza extremos)
        if density < 0.001:
            density_factor = 0.7  # Muy disperso
        elif density > 0.5:
            density_factor = 0.5  # Muy denso
        else:
            density_factor = 1.0 - (density * 0.5)  # Escala lineal inversa

        # Factor 4: Espectral (basado en conectividad algebraica y resonancia)
        fiedler = spectral.get("fiedler_value", 0.0)
        if spectral.get("resonance_risk", False):
            spectral_factor = 0.6
        elif fiedler > 0.1:
            spectral_factor = 1.0
        elif fiedler > 0.01:
            spectral_factor = 0.85
        else:
            spectral_factor = 0.7

        # Pesos del modelo (suman 1.0)
        weights = {
            "euler": 0.30,
            "stability": 0.25,
            "density": 0.20,
            "spectral": 0.25,
        }

        base_score = 100.0 * (
            weights["euler"] * euler_factor
            + weights["stability"] * stability_factor
            + weights["density"] * density_factor
            + weights["spectral"] * spectral_factor
        )

        # ═══════════════════════════════════════════════════════════════
        # PENALIZACIONES
        # ═══════════════════════════════════════════════════════════════

        penalty = 0.0
        penalty_details = []

        # Penalización por ciclos (hasta -25 puntos)
        if metrics.beta_1 > 0:
            cycle_penalty = min(25.0, metrics.beta_1 * 5.0)
            penalty += cycle_penalty
            penalty_details.append(f"Ciclos ({metrics.beta_1}): -{cycle_penalty:.1f}")

        # Penalización por sinergia de riesgo (hasta -20 puntos)
        if synergy["synergy_detected"]:
            synergy_penalty = min(20.0, synergy["synergy_score"] * 30.0)
            penalty += synergy_penalty
            penalty_details.append(
                f"Sinergia ({synergy['synergy_score']:.2f}): -{synergy_penalty:.1f}"
            )

        # Penalización por resonancia espectral (hasta -10 puntos)
        if spectral.get("resonance_risk", False):
            penalty += 10.0
            penalty_details.append("Resonancia espectral: -10.0")

        # Penalización por anomalías (hasta -15 puntos)
        iso_count = len(anomalies["isolated_nodes"])
        orphan_count = len(anomalies["orphan_insumos"])
        empty_count = len(anomalies["empty_apus"])

        if iso_count + orphan_count + empty_count > 0:
            anomaly_penalty = min(
                15.0, (iso_count * 2 + orphan_count * 1.5 + empty_count * 1)
            )
            penalty += anomaly_penalty
            penalty_details.append(
                f"Anomalías ({iso_count}+{orphan_count}+{empty_count}): -{anomaly_penalty:.1f}"
            )

        # Penalización por riesgo térmico (hasta -10 puntos)
        if thermal.get("thermal_risk_level") == "CRÍTICO":
            penalty += 10.0
            penalty_details.append("Riesgo térmico crítico: -10.0")
        elif thermal.get("thermal_risk_level") == "ALTO":
            penalty += 5.0
            penalty_details.append("Riesgo térmico alto: -5.0")

        # Score final
        integrity_score = max(0.0, min(100.0, base_score - penalty))
        integrity_score = round(integrity_score, 1)

        # ═══════════════════════════════════════════════════════════════
        # CLASIFICACIÓN DE COMPLEJIDAD
        # ═══════════════════════════════════════════════════════════════

        complexity_score = (
            0.30 * min(1.0, metrics.beta_1 / max(1, graph.number_of_nodes()) * 10)
            + 0.25 * density
            + 0.25 * (1.0 - metrics.euler_efficiency)
            + 0.20 * (1.0 - pyramid_stability)
        )

        if complexity_score > 0.5:
            complexity_level = "CRÍTICA"
        elif complexity_score > 0.3:
            complexity_level = "Alta"
        elif complexity_score > 0.15:
            complexity_level = "Media"
        else:
            complexity_level = "Baja"

        # ═══════════════════════════════════════════════════════════════
        # ALERTAS Y RIESGOS
        # ═══════════════════════════════════════════════════════════════

        waste_alerts = []
        if iso_count > 0:
            waste_alerts.append(
                f"🚨 {iso_count} nodo(s) aislado(s) detectado(s) (recursos sin uso)."
            )
        if orphan_count > 0:
            waste_alerts.append(
                f"⚠️ {orphan_count} insumo(s) huérfano(s) (sin vinculación a APU)."
            )
        if empty_count > 0:
            waste_alerts.append(
                f"⚠️ {empty_count} APU(s) vacío(s) (sin detalle de insumos)."
            )
        if metrics.euler_efficiency < 0.6:
            waste_alerts.append(
                f"⚠️ Eficiencia topológica baja ({metrics.euler_efficiency:.2%})."
            )
        if truncated:
            waste_alerts.append(
                f"ℹ️ Análisis de ciclos truncado a {self.max_cycles} (hay más)."
            )

        circular_risks = []
        if metrics.beta_1 > 0:
            circular_risks.append(
                f"🚨 CRÍTICO: {metrics.beta_1} ciclo(s) de dependencia detectado(s)."
            )
            for cycle in cycles[:3]:
                circular_risks.append(f"   ↳ {cycle[:80]}{'...' if len(cycle) > 80 else ''}")

        if synergy["synergy_detected"]:
            circular_risks.append(
                f"☣️ RIESGO SISTÉMICO: Sinergia de riesgo nivel {synergy['risk_level']} "
                f"(score: {synergy['synergy_score']:.2f})."
            )

        if convection["high_risk_nodes"]:
            circular_risks.append(
                f"🔥 RIESGO CONVECTIVO: {len(convection['high_risk_nodes'])} nodo(s) "
                f"altamente sensible(s) a variación de transporte/combustible."
            )

        if spectral.get("resonance_risk", False):
            circular_risks.append(
                "🔊 RIESGO DE RESONANCIA: Espectro degenerado detectado. "
                "Alta vulnerabilidad a perturbaciones sistémicas."
            )

        if thermal.get("thermal_risk_level") in ["CRÍTICO", "ALTO"]:
            circular_risks.append(
                f"🌡️ TEMPERATURA SISTÉMICA {thermal['thermal_risk_level']}: "
                f"{thermal['system_temperature']:.1f}°. Alta sensibilidad a inflación de insumos."
            )

        # ═══════════════════════════════════════════════════════════════
        # RIESGO FINANCIERO INTEGRADO
        # ═══════════════════════════════════════════════════════════════

        financial_risk = None
        if financial_metrics:
            volatility = financial_metrics.get("volatility", 0.0)
            roi = financial_metrics.get("roi", 0.0)

            if roi < -0.1:
                financial_risk = "CRÍTICO"
            elif roi < 0:
                financial_risk = "ALTO"
            elif volatility > 0.3:
                financial_risk = "ALTO"
            elif volatility > 0.2:
                financial_risk = "MEDIO"
            else:
                financial_risk = "BAJO"

            # Escalamiento por factores topológicos
            if financial_risk in ["ALTO", "MEDIO"]:
                if metrics.beta_1 > 2 or synergy["synergy_detected"]:
                    financial_risk = "CATÁSTROFICO"
                elif spectral.get("resonance_risk", False):
                    if financial_risk == "MEDIO":
                        financial_risk = "ALTO"

        # Narrativa estratégica
        strategic_narrative = self._generate_strategic_narrative(
            metrics, synergy, pyramid_stability, financial_risk, thermal, spectral
        )

        return ConstructionRiskReport(
            integrity_score=integrity_score,
            waste_alerts=waste_alerts,
            circular_risks=circular_risks,
            complexity_level=complexity_level,
            financial_risk_level=financial_risk,
            strategic_narrative=strategic_narrative,
            details={
                "metrics": asdict(metrics),
                "cycles": cycles,
                "cycles_truncated": truncated,
                "anomalies": anomalies,
                "synergy_risk": synergy,
                "connectivity": connectivity,
                "pyramid_stability": pyramid_stability,
                "density": round(density, 6),
                "convection_risk": convection,
                "spectral_analysis": spectral,
                "thermal_analysis": {
                    "system_temperature": thermal["system_temperature"],
                    "thermal_risk_level": thermal["thermal_risk_level"],
                    "hotspots": thermal["hotspots"][:5],
                },
                "scoring_breakdown": {
                    "base_score": round(base_score, 2),
                    "total_penalty": round(penalty, 2),
                    "penalty_details": penalty_details,
                    "factor_weights": weights,
                    "factors": {
                        "euler": round(euler_factor, 4),
                        "stability": round(stability_factor, 4),
                        "density": round(density_factor, 4),
                        "spectral": round(spectral_factor, 4),
                    },
                },
            },
        )

    def _generate_strategic_narrative(
        self,
        metrics: TopologicalMetrics,
        synergy: Dict[str, Any],
        stability: float,
        financial_risk: Optional[str],
        thermal: Dict[str, Any],
        spectral: Dict[str, Any],
    ) -> str:
        """
        Genera una narrativa estratégica con el tono del 'Consejo de Sabios' (Propuesta 2).
        Integra los conceptos de 'El Intérprete Diplomático'.
        """
        sections = []

        # 1. DIAGNÓSTICO ESTRUCTURAL
        if stability > 0.7:
            sections.append(
                "🏗️ **ESTRUCTURA SÓLIDA**: La pirámide presupuestaria presenta una base "
                "robusta con adecuada distribución de insumos por APU."
            )
        elif stability > 0.4:
            sections.append(
                "⚠️ **ESTRUCTURA MODERADA**: La relación base-altura es aceptable pero "
                "existen oportunidades de consolidación en la distribución de recursos."
            )
        else:
            sections.append(
                "🚨 **ALERTA ESTRUCTURAL (PIRÁMIDE INVERTIDA)**: La base de insumos es "
                "insuficiente para la complejidad de APUs definidos. Riesgo de colapso "
                "ante variaciones de mercado."
            )

        # 2. INTEGRIDAD LÓGICA
        if metrics.beta_1 == 0:
            sections.append(
                "✅ **TRAZABILIDAD LIMPIA**: Sin ciclos de dependencia. El flujo de costos "
                "es unidireccional y auditable."
            )
        elif metrics.beta_1 <= 2:
            sections.append(
                f"⚠️ **CICLOS DETECTADOS**: {metrics.beta_1} ciclo(s) de dependencia "
                f"identificado(s). Requieren revisión para evitar cálculos circulares."
            )
        else:
            sections.append(
                f"🚨 **COMPLEJIDAD CÍCLICA CRÍTICA**: {metrics.beta_1} ciclos de dependencia "
                f"comprometen la integridad del cálculo. Auditoría inmediata requerida."
            )

        # 3. RIESGO SISTÉMICO
        if synergy.get("synergy_detected"):
            risk_level = synergy.get("risk_level", "DETECTADO")
            bridge_count = len(synergy.get("bridge_nodes", []))
            sections.append(
                f"☣️ **RIESGO DE CONTAGIO ({risk_level})**: {bridge_count} nodo(s) puente "
                f"conectan múltiples ciclos. Un fallo en estos puntos desencadenaría "
                f"efecto dominó en cascada."
            )

        # 4. SENSIBILIDAD TÉRMICA
        if thermal.get("thermal_risk_level") in ["CRÍTICO", "ALTO"]:
            temp = thermal.get("system_temperature", 0)
            sections.append(
                f"🌡️ **ALTA SENSIBILIDAD INFLACIONARIA**: Temperatura sistémica de {temp:.0f}°. "
                f"El presupuesto es vulnerable a fluctuaciones de precios en insumos volátiles."
            )

        # 5. ESTABILIDAD ESPECTRAL
        if spectral.get("resonance_risk"):
            sections.append(
                "🔊 **VULNERABILIDAD ESPECTRAL**: Concentración anómala en el espectro "
                "del Laplaciano. El sistema podría amplificar perturbaciones externas "
                "(efecto resonancia)."
            )

        # 6. VEREDICTO FINANCIERO
        if financial_risk:
            if financial_risk == "CATÁSTROFICO":
                sections.append(
                    "💀 **ALERTA CRÍTICA DE VIABILIDAD**: El perfil de riesgo financiero "
                    "combinado con la estructura topológica indica probabilidad significativa "
                    "de fracaso del proyecto. Suspender compromisos hasta revisión profunda."
                )
            elif financial_risk == "CRÍTICO":
                sections.append(
                    "📉 **RIESGO FINANCIERO CRÍTICO**: Los indicadores económicos "
                    "requieren atención inmediata. Considerar reestructuración."
                )
            elif financial_risk == "ALTO":
                sections.append(
                    "📊 **PRECAUCIÓN FINANCIERA**: Volatilidad elevada en componentes "
                    "críticos. Implementar coberturas o contingencias."
                )
            elif financial_risk == "BAJO":
                sections.append(
                    "💰 **SALUD FINANCIERA**: Los indicadores económicos respaldan "
                    "la viabilidad técnica del proyecto."
                )

        return " ".join(sections)

    def analyze_structural_integrity(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Wrapper de análisis compatible con el pipeline actual y telemetría."""
        report = self.generate_executive_report(graph)
        metrics = TopologicalMetrics(**report.details["metrics"])

        flat_results = {
            "business.integrity_score": report.integrity_score,
            "business.pyramid_stability": report.details["pyramid_stability"],
            "business.betti_b0": metrics.beta_0,
            "business.betti_b1": metrics.beta_1,
            "business.euler_characteristic": metrics.euler_characteristic,
            "business.euler_efficiency": metrics.euler_efficiency,
            "business.cycles_count": len(report.details["cycles"]),
            "business.synergy_detected": 1
            if report.details["synergy_risk"]["synergy_detected"]
            else 0,
            "business.convection_risk_nodes": len(report.details["convection_risk"]["high_risk_nodes"]),
            "business.is_dag": 1 if report.details["connectivity"]["is_dag"] else 0,
            "business.isolated_count": len(report.details["anomalies"]["isolated_nodes"]),
            "business.orphan_insumos_count": len(
                report.details["anomalies"]["orphan_insumos"]
            ),
            "business.empty_apus_count": len(report.details["anomalies"]["empty_apus"]),
            "details": {
                "executive_report": asdict(report),
                "topology": {"betti_numbers": asdict(metrics)},
                "cycles": {"list": report.details["cycles"]},
                "connectivity": report.details["connectivity"],
                "anomalies": report.details["anomalies"],
                "critical_resources": self._identify_critical_resources(graph),
                "graph_summary": {
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "density": report.details["density"],
                    "pyramid_stability": report.details["pyramid_stability"],
                },
            },
        }

        # Emisión de Telemetría
        if self.telemetry:
            for k, v in flat_results.items():
                if isinstance(v, (int, float)):
                    self.telemetry.record_metric(k, v)

        return flat_results

    def get_audit_report(self, analysis_result_or_graph: Any) -> List[str]:
        """Genera un reporte ASCII art profesional."""
        if isinstance(analysis_result_or_graph, nx.DiGraph):
            analysis = self.analyze_structural_integrity(analysis_result_or_graph)
        else:
            analysis = analysis_result_or_graph

        report_dict = analysis.get("details", {}).get("executive_report", {})
        if not report_dict:
            return ["Error: No se pudo generar el reporte."]

        lines = []
        lines.append("┌──────────────────────────────────────────────────┐")
        lines.append("│      AUDITORÍA ESTRUCTURAL DEL PRESUPUESTO       │")
        lines.append("├──────────────────────────────────────────────────┤")
        lines.append(
            f"│ PUNTUACIÓN DE INTEGRIDAD: {report_dict.get('integrity_score', 0):>6.1f} / 100.0          │"
        )
        lines.append(
            f"│ Nivel de Complejidad:     {report_dict.get('complexity_level', ''):<23}│"
        )
        lines.append("├──────────────────────────────────────────────────┤")

        metrics = report_dict.get("details", {}).get("metrics", {})
        lines.append("│ [MÉTRICAS TÉCNICAS]                              │")
        lines.append(f"│ Ciclos de Costo:           {metrics.get('beta_1', 0):<22}│")
        lines.append(
            f"│ Eficiencia de Euler:       {metrics.get('euler_efficiency', 0.0):<22.2f}│"
        )
        lines.append("└──────────────────────────────────────────────────┘")

        if report_dict.get("circular_risks"):
            lines.append("│ [ALERTA CRÍTICA] Referencias circulares detectadas! │")
            for risk in report_dict["circular_risks"]:
                wrapped_lines = textwrap.wrap(risk, width=44)
                for line in wrapped_lines:
                    lines.append(f"│ ❌ {line:<44} │")

        waste_alerts = report_dict.get("waste_alerts", [])
        anomalies = analysis.get("details", {}).get("anomalies", {})
        iso_count = len(anomalies.get("isolated_nodes", []))
        orphan_count = len(anomalies.get("orphan_insumos", []))
        empty_count = len(anomalies.get("empty_apus", []))

        if waste_alerts or iso_count > 0 or orphan_count > 0 or empty_count > 0:
            lines.append("├──────────────────────────────────────────────────┤")
            lines.append("│ [POSIBLE DESPERDICIO / ALERTAS]                  │")
            if iso_count > 0:
                lines.append(f"│ ⚠ Recursos Fantasma (Sin uso): {iso_count:<18}│")
            if empty_count > 0:
                lines.append(f"│ ⚠ APUs Vacíos:          {empty_count:<25}│")
            for alert in waste_alerts:
                wrapped_lines = textwrap.wrap(alert, width=44)
                for line in wrapped_lines:
                    lines.append(f"│ ⚠ {line:<44} │")
            lines.append("└──────────────────────────────────────────────────┘")

        return lines
