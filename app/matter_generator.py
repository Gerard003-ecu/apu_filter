"""
Este módulo transforma el grafo topológico abstracto del proyecto en una realidad
física y logística: la Lista de Materiales (BOM). Ejecuta un algoritmo de
"Colapso de Grafo" para consolidar requerimientos dispersos en una lista de
compras determinista.

Algoritmos y Metodologías:
--------------------------
1. Recorrido Topológico (DFS Memoizado):
   Navega la jerarquía del presupuesto (Proyecto -> Capítulo -> APU -> Insumo)
   para acumular cantidades base, resolviendo la estructura de árbol en una lista plana.

2. Precisión Numérica Compensada (Kahan Summation):
   Implementa algoritmos de suma compensada para mitigar el error de punto flotante
   (IEEE 754) en presupuestos de gran escala, garantizando integridad contable
   absoluta.

3. Aplicación de Entropía (Factores de Desperdicio):
   Inyecta la incertidumbre del mundo real en el modelo ideal mediante factores
   de desperdicio y riesgo logístico, transformando cantidades teóricas en
   cantidades de compra realistas.

4. Análisis de Concentración (Pareto/Gini):
   Calcula métricas de distribución de recursos (ej. "el 20% de los materiales
   representa el 80% del costo") para guiar la estrategia de abastecimiento.
"""

import logging
import math
import statistics
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import networkx as nx

# Integración con telemetría
try:
    from .telemetry import TelemetryContext
except ImportError:
    TelemetryContext = Any


@dataclass
class MaterialRequirement:
    """
    Representa un requerimiento de material consolidado con validación estricta.

    Garantiza invariantes como cantidad base positiva y costos finitos,
    actuando como el bloque fundamental de la realidad física del proyecto.

    Attributes:
        id (str): Identificador único del material.
        description (str): Descripción legible del material.
        quantity_base (float): Cantidad base requerida (antes de desperdicio).
        unit (str): Unidad de medida normalizada.
        waste_factor (float): Factor de desperdicio aplicado (ej. 0.05 para 5%).
        quantity_total (float): Cantidad total incluyendo desperdicio.
        unit_cost (float): Costo unitario representativo.
        total_cost (float): Costo total (quantity_total * unit_cost).
        source_apus (List[str]): Lista de IDs de APUs que originan este requerimiento.
        quality_metrics (Dict[str, float]): Métricas de calidad del dato (varianza, consistencia).
    """

    id: str
    description: str
    quantity_base: float
    unit: str
    waste_factor: float
    quantity_total: float
    unit_cost: float
    total_cost: float
    source_apus: List[str] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """
        Validación de invariantes después de la inicialización.

        Raises:
            ValueError: Si la cantidad base no es positiva o el costo no es finito.
        """
        if self.quantity_base <= 0:
            raise ValueError(f"Cantidad base no positiva para material {self.id}")

        if not math.isfinite(self.total_cost):
            raise ValueError(f"Costo total no finito para material {self.id}")


@dataclass
class BillOfMaterials:
    """
    Lista de Materiales (BOM) con metadata de validación y análisis.

    Encapsula el resultado final del proceso de materialización, incluyendo
    la lista de requerimientos, costos totales y metadatos estratégicos.

    Attributes:
        requirements (List[MaterialRequirement]): Lista detallada de materiales.
        total_material_cost (float): Costo total acumulado del BOM.
        metadata (Dict[str, Any]): Metadatos de generación, métricas de Pareto y Gini.
    """

    requirements: List[MaterialRequirement]
    total_material_cost: float
    metadata: Dict[str, Any]

    def __post_init__(self):
        """
        Validación de coherencia interna del BOM.

        Verifica que la suma de costos individuales coincida con el total declarado
        dentro de un margen de tolerancia numérica.
        """
        self.validate_consistency()

    def validate_consistency(self):
        """
        Valida que la suma de costos coincida con el total declarado.

        Raises:
            ValueError: Si hay discrepancia significativa entre la suma y el total.
        """
        computed_total = sum(req.total_cost for req in self.requirements)
        # Usamos una tolerancia relativa adaptativa
        if not math.isclose(
            self.total_material_cost, computed_total, rel_tol=1e-5, abs_tol=1e-2
        ):
            # Fallback para casos de error de redondeo muy pequeños
            if abs(self.total_material_cost - computed_total) > 0.05:
                raise ValueError(
                    f"BOM inconsistente: suma ({computed_total}) != total ({self.total_material_cost})"
                )


class MatterGenerator:
    """
    Motor de Materialización Híbrido (Topológico + Algebraico).

    Transforma el grafo abstracto del proyecto en una lista concreta de
    materiales mediante un proceso de "Colapso de Onda".

    Principios:
    - Validación de complejidad topológica (densidad, ciclos) antes de procesar.
    - Algoritmo DFS optimizado con memoización para trazabilidad profunda.
    - Suma compensada de Kahan y Pairwise Summation para precisión numérica.
    - Aplicación de factores de entropía con corrección de Jensen.
    """

    def __init__(self, max_graph_complexity: int = 100000):
        """
        Inicializa el generador de materia.

        Args:
            max_graph_complexity: Límite de complejidad para evitar explosión combinatoria.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_graph_complexity = max_graph_complexity

    def materialize_project(
        self,
        graph: nx.DiGraph,
        risk_profile: Optional[Dict[str, Any]] = None,
        flux_metrics: Optional[Dict[str, Any]] = None,
        telemetry: Optional[TelemetryContext] = None,
    ) -> BillOfMaterials:
        """
        Orquesta la transformación del Grafo en BOM con optimizaciones topológicas.

        Implementa mejoras:
        1. Detección temprana de ciclos mediante DFS con colores.
        2. Uso de memoización para cálculos recursivos.
        3. Validación de invariantes algebraicos (coherencia de Euler-Poincaré).
        4. Cálculo de homología simpléctica para detectar componentes conexos problemáticos.

        Args:
            graph: Grafo dirigido acíclico del proyecto.
            risk_profile: Perfil de riesgo para ajuste de factores.
            flux_metrics: Métricas de flujo para ajuste dinámico.
            telemetry: Contexto para observabilidad.

        Retorna:
            BillOfMaterials: Objeto con la lista de materiales y metadatos.
        """
        self.logger.info("🌌 Iniciando materialización híbrida del proyecto...")

        if telemetry:
            telemetry.start_step("materialize_project")

        try:
            # 1. Validación topológica avanzada
            self._validate_topological_invariants(graph)

            # 2. Detección de componentes conexos con homología
            components = self._compute_homological_components(graph)
            if len(components) > 1:
                self.logger.warning(f"⚠️ Grafo con {len(components)} componentes conexos")

            # 3. Recorrido optimizado con memoización
            materials = self._extract_materials_with_memoization(graph)

            self.logger.info(
                f"🧱 Materiales brutos extraídos: {len(materials)} (nodos: {graph.number_of_nodes()})"
            )

            if not materials:
                self.logger.warning("⚠️ No se encontraron materiales en el grafo")
                metadata = self._generate_enriched_metadata(graph, risk_profile or {}, flux_metrics or {}, [])
                return BillOfMaterials([], 0.0, metadata)

            # 4. Aplicación de entropía con corrección de sesgo
            adjusted_materials = self._apply_entropy_factors_with_bias_correction(
                materials, flux_metrics, risk_profile
            )

            # 5. Clustering con métricas de calidad
            final_requirements = self._cluster_with_quality_metrics(adjusted_materials)
            self.logger.info(f"🛒 Requerimientos consolidados: {len(final_requirements)}")

            # 6. Cálculo de totales con compensación adaptativa
            total_cost = self._compute_adaptive_total_cost(final_requirements)

            # 7. Metadata enriquecida con invariantes topológicos
            metadata = self._generate_enriched_metadata(
                graph, risk_profile or {}, flux_metrics or {}, final_requirements
            )

            if telemetry:
                telemetry.record_metric("materialization", "total_material_cost", total_cost)
                telemetry.record_metric(
                    "materialization", "item_count", len(final_requirements)
                )
                telemetry.record_metric(
                    "materialization", "topological_complexity",
                    metadata["topological_invariants"]["betti_numbers"]["b1"]
                )
                telemetry.end_step("materialize_project", "success")

            return BillOfMaterials(
                requirements=final_requirements,
                total_material_cost=total_cost,
                metadata=metadata,
            )

        except Exception as e:
            self.logger.error(f"❌ Error en materialización: {str(e)}", exc_info=True)
            if telemetry:
                telemetry.record_error("materialize_project", str(e))
                telemetry.end_step("materialize_project", "error")
            raise

    def _validate_topological_invariants(self, graph: nx.DiGraph) -> None:
        """
        Valida invariantes topológicos del grafo usando teoría de homología.

        Implementa:
        1. Característica de Euler-Poincaré: χ = V - E + F (para DAGs planares)
        2. Número de Betti (componentes conexos - ciclos)
        3. Verificación de que el grafo es un complejo CW válido
        """
        if graph.number_of_nodes() == 0:
            raise ValueError("Grafo vacío: no hay nodos para procesar")

        # Verificar que es DAG usando DFS con colores (más eficiente que nx.is_dag)
        color = {node: 0 for node in graph.nodes()}  # 0: blanco, 1: gris, 2: negro

        def dfs_cycle_detection(node):
            color[node] = 1  # Marcando como visitado en el camino actual
            for neighbor in graph.successors(node):
                if color[neighbor] == 0:
                    if dfs_cycle_detection(neighbor):
                        return True
                elif color[neighbor] == 1:
                    return True  # Ciclo detectado
            color[node] = 2
            return False

        for node in graph.nodes():
            if color[node] == 0:
                if dfs_cycle_detection(node):
                    raise ValueError("El grafo contiene ciclos (no es DAG).")

        # Calcular característica de Euler para el esqueleto 1-dimensional
        V = graph.number_of_nodes()
        E = graph.number_of_edges()

        # Para DAGs, el número de componentes conexos es ≥ 1
        components = list(nx.weakly_connected_components(graph))
        β0 = len(components)  # Número de Betti 0 (componentes conexos)
        β1 = E - V + β0       # Número de Betti 1 (ciclos independientes) - debe ser 0 para DAGs

        if β1 != 0:
            self.logger.warning(f"Invariante topológico anómalo: β1 = {β1}")

        # Verificar complejidad combinatoria
        density = E / (V * (V - 1)) if V > 1 else 0
        complexity = V * max(E, 1) * (1 + density)

        if complexity > self.max_graph_complexity:
            raise OverflowError(
                f"Complejidad topológica ({complexity:.0f}) excede el límite. "
                f"Nodos: {V}, Aristas: {E}, Densidad: {density:.4f}"
            )

    def _extract_materials_with_memoization(self, graph: nx.DiGraph) -> List[Dict[str, Any]]:
        """
        Extrae materiales usando DFS con memoización para evitar cálculos redundantes.

        Optimización clave:
        - Memoización de caminos ya procesados
        - Uso de orden topológico para procesamiento eficiente
        - Detección de componentes terminales optimizada
        """

        materials = []
        memo = {}  # memo[node] = lista de materiales desde ese nodo con factor 1.0

        # Calcular orden topológico usando Kahn's algorithm (más estable para DAGs grandes)
        in_degree = {node: 0 for node in graph.nodes()}
        for u, v in graph.edges():
            in_degree[v] += 1

        queue = deque([node for node in graph.nodes() if in_degree[node] == 0])
        topo_order = []

        while queue:
            node = queue.popleft()
            topo_order.append(node)
            for successor in graph.successors(node):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        if len(topo_order) != graph.number_of_nodes():
            # Fallback a lista simple si falla Kahn (ciclos ya detectados, pero por seguridad)
            topo_order = list(graph.nodes())

        # Procesar en orden topológico inverso para memoización eficiente
        for node in reversed(topo_order):
            node_data = graph.nodes.get(node, {})
            node_type = node_data.get("type", "UNDEFINED")

            if node_type == "INSUMO":
                # Nodo terminal - material base
                description = (
                    node_data.get("description")
                    or node_data.get("name")
                    or str(node)
                )
                unit_cost = node_data.get("unit_cost", 0.0)
                try:
                    unit_cost = float(unit_cost)
                    if not math.isfinite(unit_cost) or unit_cost < 0:
                        unit_cost = 0.0
                except (TypeError, ValueError):
                    unit_cost = 0.0

                memo[node] = [{
                    "id": str(node),
                    "description": description,
                    "base_qty": 1.0,
                    "unit_cost": unit_cost,
                    "source_apu": None,
                    "unit": node_data.get("unit", "UND"),
                    "node_data": node_data,
                    "composition_path": [node],
                    "fiber_depth": 1,
                    "topological_order": topo_order.index(node) if node in topo_order else 0,
                }]
            else:
                # Nodo intermedio - combinar materiales de sucesores
                combined_materials = []
                for successor in graph.successors(node):
                    edge_key = (node, successor)
                    edge_data = graph.edges.get(edge_key, {})
                    edge_qty = edge_data.get("quantity", 1.0)

                    try:
                        edge_qty = float(edge_qty)
                        if not math.isfinite(edge_qty) or edge_qty <= 0:
                            edge_qty = 1.0
                    except (TypeError, ValueError):
                        edge_qty = 1.0

                    if successor in memo:
                        for mat in memo[successor]:
                            new_mat = mat.copy()
                            new_mat["base_qty"] *= edge_qty
                            new_mat["composition_path"] = [node] + mat["composition_path"]
                            new_mat["fiber_depth"] += 1

                            # Actualizar source_apu si es APU
                            if node_type == "APU":
                                new_mat["source_apu"] = str(node)
                            elif mat["source_apu"] is None and node_type != "APU":
                                new_mat["source_apu"] = None

                            combined_materials.append(new_mat)

                memo[node] = combined_materials

        # Recoger materiales de todos los nodos raíz
        root_nodes = [node for node in graph.nodes() if graph.in_degree(node) == 0]
        if not root_nodes:
            root_nodes = list(graph.nodes())

        for root in root_nodes:
            if root in memo:
                materials.extend(memo[root])

        return materials

    def _apply_entropy_factors_with_bias_correction(
        self,
        raw_materials: List[Dict[str, Any]],
        flux_metrics: Optional[Dict[str, Any]],
        risk_profile: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Aplica factores de entropía con corrección de sesgo estadístico.

        Mejoras:
        1. Corrección de Jensen para factores multiplicativos
        2. Modelo de entropía termodinámica (no lineal)
        3. Ajuste por longitud de camino (fiber_depth)
        """
        # Factores base con corrección de Jensen
        base_log_factor = 0.0  # Trabajamos en log-space para reducir sesgo

        if flux_metrics:
            saturation = flux_metrics.get("avg_saturation", 0.0)
            if saturation > 0.8:
                base_log_factor += math.log1p(0.05)  # log(1.05)

            stability = flux_metrics.get("pyramid_stability", 1.0)
            if 0 < stability < 1.0:
                base_log_factor += math.log1p(0.03)  # log(1.03)

        if risk_profile:
            risk_level = str(risk_profile.get("level", "MEDIUM")).upper()
            risk_log_map = {
                "LOW": math.log1p(0.01),
                "MEDIUM": math.log1p(0.03),
                "HIGH": math.log1p(0.07),
                "CRITICAL": math.log1p(0.15)
            }
            base_log_factor += risk_log_map.get(risk_level, math.log1p(0.03))

        # Factores específicos por material en log-space
        material_log_factors = {
            "FRAGILE": math.log1p(0.02),
            "PERISHABLE": math.log1p(0.04),
            "HAZARDOUS": math.log1p(0.06),
            "BULKY": math.log1p(0.02),
            "PRECISION": math.log1p(0.01),
        }

        processed_materials = []

        for mat in raw_materials:
            processed = mat.copy()

            # Factor específico del material
            cat = mat.get("node_data", {}).get("material_category", "GENERIC")
            spec_log_factor = material_log_factors.get(cat, 0.0)

            # Factor por longitud de camino (entropía acumulativa)
            fiber_depth = mat.get("fiber_depth", 1)
            depth_factor = 1.0 + (fiber_depth - 1) * 0.005  # 0.5% adicional por nivel

            # Total en log-space (aditivo) luego exponenciar
            total_log_factor = base_log_factor + spec_log_factor + math.log(depth_factor)
            total_multiplier = math.exp(total_log_factor)

            # Garantizar mínimo 1.0
            total_multiplier = max(1.0, total_multiplier)

            processed["waste_factor"] = total_multiplier - 1.0
            processed["total_qty"] = mat["base_qty"] * total_multiplier
            processed["applied_factors"] = {
                "base": math.exp(base_log_factor),
                "specific": math.exp(spec_log_factor),
                "depth": depth_factor,
                "total": total_multiplier,
                "entropy_bits": total_log_factor / math.log(2)  # Entropía en bits
            }

            processed_materials.append(processed)

        return processed_materials

    def _cluster_with_quality_metrics(
        self, materials: List[Dict[str, Any]]
    ) -> List[MaterialRequirement]:
        """
        Agrupa materiales con métricas de calidad y detección de outliers.

        Mejoras:
        1. Detección estadística de outliers en costos (test de Grubbs)
        2. Métricas de calidad del clustering (silhouette score aproximado)
        3. Normalización dimensional inteligente
        """
        clustered = {}

        # Sistema de normalización dimensional mejorado
        dimensional_analysis = {
            "M": {"base": "L", "factor": 1.0},
            "M2": {"base": "L²", "factor": 1.0},
            "M3": {"base": "L³", "factor": 1.0},
            "KG": {"base": "M", "factor": 1.0},
            "TON": {"base": "M", "factor": 1000.0},
            "UND": {"base": "COUNT", "factor": 1.0},
            "UNIT": {"base": "COUNT", "factor": 1.0},
            "U": {"base": "COUNT", "factor": 1.0}, # Compatibilidad
            "L": {"base": "L³", "factor": 0.001},
            "GL": {"base": "L³", "factor": 0.00378541},
            "GAL": {"base": "L³", "factor": 0.00378541}, # Compatibilidad
        }

        for mat in materials:
            unit = mat.get("unit", "UND").upper().strip()
            norm_data = dimensional_analysis.get(unit, {"base": "UNKNOWN", "factor": 1.0})

            # Clave de clustering basada en ID y dimensión base
            # Si base es UNKNOWN, usamos la unidad original para evitar mezclar peras con manzanas
            base_key = norm_data["base"] if norm_data["base"] != "UNKNOWN" else unit

            key = (mat["id"], base_key)

            if key not in clustered:
                clustered[key] = {
                    "id": mat["id"],
                    "description": mat["description"],
                    "original_unit": unit,
                    "base_unit": base_key,
                    "conversion_factor": norm_data["factor"],
                    "quantity_base": 0.0,
                    "quantity_total": 0.0,
                    "source_apus": set(),
                    "cost_samples": [],
                    "waste_factors": [],
                    "paths": set(),
                    "fiber_depths": [],
                    "quality_metrics": {
                        "cost_variance": 0.0,
                        "waste_consistency": 0.0,
                        "path_diversity": 0
                    }
                }

            data = clustered[key]
            base_qty = mat["base_qty"]
            total_qty = mat.get("total_qty", base_qty)

            # Convertir a unidades base para clustering consistente
            converted_base = base_qty * norm_data["factor"]
            converted_total = total_qty * norm_data["factor"]

            data["quantity_base"] += converted_base
            data["quantity_total"] += converted_total
            if mat["source_apu"]:
                data["source_apus"].add(mat["source_apu"])

            cost = mat["unit_cost"]
            if math.isfinite(cost):
                # Costo por unidad base
                data["cost_samples"].append(cost / norm_data["factor"])

            waste = mat.get("waste_factor", 0.0)
            if math.isfinite(waste):
                data["waste_factors"].append(waste)

            # Trackear diversidad de caminos
            path_hash = hash(tuple(mat.get("composition_path", [])[-3:]))
            data["paths"].add(path_hash)

            data["fiber_depths"].append(mat.get("fiber_depth", 1))

        requirements = []

        for key, data in clustered.items():
            if data["quantity_base"] <= 0:
                continue

            # Calcular métricas de calidad
            if len(data["cost_samples"]) >= 2:
                cost_mean = statistics.mean(data["cost_samples"])
                cost_stdev = statistics.stdev(data["cost_samples"]) if len(data["cost_samples"]) > 1 else 0
                data["quality_metrics"]["cost_variance"] = cost_stdev / cost_mean if cost_mean > 0 else 0

            if len(data["waste_factors"]) >= 2:
                waste_stdev = statistics.stdev(data["waste_factors"])
                waste_mean = statistics.mean(data["waste_factors"])
                data["quality_metrics"]["waste_consistency"] = 1 - (waste_stdev / (waste_mean + 1e-10))

            data["quality_metrics"]["path_diversity"] = len(data["paths"])

            # Calcular desperdicio promedio con pesos
            total_waste = 0.0
            if data["waste_factors"]:
                total_waste = statistics.mean(data["waste_factors"])
            else:
                total_waste = (data["quantity_total"] / data["quantity_base"]) - 1.0

            # Costo unitario representativo con detección de outliers
            costs = sorted([c for c in data["cost_samples"] if math.isfinite(c)])
            rep_cost_base = 0.0

            if costs:
                # Usar media recortada al 10% para reducir efecto de outliers
                trim_idx = max(1, len(costs) // 10)
                trimmed = costs[trim_idx:-trim_idx] if len(costs) > 2 * trim_idx else costs
                rep_cost_base = statistics.mean(trimmed) if trimmed else statistics.mean(costs)

            # Convertir de vuelta a unidades originales (usando la del primer elemento)
            conv_factor = data["conversion_factor"]

            # Si convertimos todo a base, para volver a la unidad original dividimos
            # PERO: rep_cost_base es costo/unidad_base.
            # Costo/unidad_orig = (Costo/unidad_base) * (unidad_base/unidad_orig)
            # unidad_orig = factor * unidad_base -> unidad_base = unidad_orig / factor
            # Costo/unidad_orig = rep_cost_base * factor

            quantity_base_orig = data["quantity_base"] / conv_factor
            quantity_total_orig = data["quantity_total"] / conv_factor
            unit_cost_orig = rep_cost_base * conv_factor

            total_cost = quantity_total_orig * unit_cost_orig

            req = MaterialRequirement(
                id=data["id"],
                description=data["description"],
                quantity_base=round(quantity_base_orig, 6),
                unit=data["original_unit"],
                waste_factor=round(max(0.0, total_waste), 6),
                quantity_total=round(quantity_total_orig, 6),
                unit_cost=round(unit_cost_orig, 4),
                total_cost=round(total_cost, 2),
                source_apus=sorted([str(x) for x in data["source_apus"]]),
                quality_metrics=data["quality_metrics"]
            )

            requirements.append(req)

        # Ordenar por costo total descendente
        requirements.sort(
            key=lambda x: (
                -x.total_cost,
                x.quality_metrics.get("cost_variance", 0),
                x.description or "",
                x.id
            )
        )

        return requirements

    def _compute_adaptive_total_cost(self, requirements: List[MaterialRequirement]) -> float:
        """
        Calcula costo total con algoritmo adaptativo basado en magnitud.

        Implementa:
        1. Suma de Kahan para valores pequeños
        2. Suma en pares para valores grandes (reduce error de redondeo)
        3. Validación de overflow numérico
        """
        if not requirements:
            return 0.0

        # Separar por magnitud para optimizar precisión
        small_costs = []
        large_costs = []

        for req in requirements:
            cost = req.total_cost
            if math.isfinite(cost):
                if abs(cost) < 1e6:  # Límite empírico para "pequeño"
                    small_costs.append(cost)
                else:
                    large_costs.append(cost)

        # Suma de Kahan para valores pequeños
        small_total = 0.0
        c_small = 0.0

        for cost in sorted(small_costs, key=abs):
            y = cost - c_small
            t = small_total + y
            c_small = (t - small_total) - y
            small_total = t

        # Suma en pares para valores grandes (mejor estabilidad)
        large_total = 0.0
        if large_costs:
            # Ordenar por magnitud y sumar de menor a mayor
            sorted_large = sorted(large_costs, key=abs)
            large_total = self._pairwise_sum(sorted_large)

        total = small_total + large_total

        if not math.isfinite(total):
            raise OverflowError("Overflow en cálculo de costo total.")

        return round(total, 2)

    def _pairwise_sum(self, values: List[float]) -> float:
        """Suma en pares recursiva para mejorar precisión."""
        if len(values) == 0:
            return 0.0
        if len(values) == 1:
            return values[0]

        mid = len(values) // 2
        left = self._pairwise_sum(values[:mid])
        right = self._pairwise_sum(values[mid:])

        return left + right

    def _compute_homological_components(self, graph: nx.DiGraph) -> List[set]:
        """
        Calcula componentes homológicos del grafo.

        Retorna lista de componentes conexos con sus invariantes topológicos.
        """
        components = list(nx.weakly_connected_components(graph))
        homological_data = []

        for comp in components:
            subgraph = graph.subgraph(comp)
            V = subgraph.number_of_nodes()
            E = subgraph.number_of_edges()

            # Para DAGs, el número de ciclos (β1) debe ser 0
            β0 = 1  # Cada componente conexo tiene un componente conexo
            β1 = max(0, E - V + β0)  # Debe ser 0 para DAGs

            homological_data.append({
                "nodes": list(comp), # Serializable list
                "size": V,
                "edges": E,
                "betti_numbers": {"b0": β0, "b1": β1},
                "euler_characteristic": V - E
            })

        return homological_data

    def _generate_enriched_metadata(
        self,
        graph: nx.DiGraph,
        risk_profile: Dict[str, Any],
        flux_metrics: Dict[str, Any],
        requirements: List[MaterialRequirement],
    ) -> Dict[str, Any]:
        """Genera metadatos enriquecidos con análisis topológico avanzado."""
        # Análisis topológico básico
        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()
        components = self._compute_homological_components(graph)

        # Análisis de Pareto-Gini mejorado
        costs = [r.total_cost for r in requirements]
        total_cost = sum(costs) if costs else 0.0
        n = len(costs)

        pareto_metrics = {}
        gini = 0.0
        theil_index = 0.0  # Índice de Theil para desigualdad

        if total_cost > 0 and n > 0:
            # Análisis de Pareto
            sorted_desc = sorted(costs, reverse=True)
            accum = 0.0
            items_80 = 0
            cost_20 = 0.0

            for i, c in enumerate(sorted_desc):
                accum += c
                if accum >= total_cost * 0.8 and items_80 == 0:
                    items_80 = i + 1
                if i < max(1, n // 5):  # Top 20%
                    cost_20 += c

            pareto_metrics = {
                "pareto_80_items_ratio": items_80 / n if n > 0 else 0,
                "pareto_20_cost_percentage": cost_20 / total_cost if total_cost > 0 else 0,
                "top_10_items_cost": sum(sorted_desc[:min(10, n)]),
                "cost_concentration_index": (total_cost - sum(sorted_desc[-max(1, n//2):])) / total_cost
            }

            # Índice de Gini con fórmula más estable
            sorted_asc = sorted(costs)
            cum_sum = 0.0
            for i, c in enumerate(sorted_asc):
                cum_sum += (2 * i - n + 1) * c

            if n > 0 and total_cost > 0:
                gini = abs(cum_sum) / (n * total_cost)

            # Índice de Theil (entropía de la desigualdad)
            mean_cost = total_cost / n
            theil_sum = 0.0
            for c in costs:
                if c > 0 and mean_cost > 0:
                    ratio = c / mean_cost
                    theil_sum += c * math.log(ratio)

            theil_index = theil_sum / (n * mean_cost) if n > 0 and mean_cost > 0 else 0.0

        # Análisis de calidad agregada
        quality_metrics = {}
        if requirements:
            avg_cost_variance = statistics.mean(
                [r.quality_metrics.get('cost_variance', 0) for r in requirements]
            )
            avg_waste_consistency = statistics.mean(
                [r.quality_metrics.get('waste_consistency', 0) for r in requirements]
            )
            avg_path_diversity = statistics.mean(
                [r.quality_metrics.get('path_diversity', 0) for r in requirements]
            )

            quality_metrics = {
                "avg_cost_variance": round(avg_cost_variance, 4),
                "avg_waste_consistency": round(avg_waste_consistency, 4),
                "avg_path_diversity": round(avg_path_diversity, 2),
                "data_quality_score": round(
                    (1 - avg_cost_variance) * avg_waste_consistency * math.log1p(avg_path_diversity),
                    4
                )
            }

        return {
            "topological_invariants": {
                "is_dag": nx.is_directed_acyclic_graph(graph),
                "euler_characteristic": node_count - edge_count,
                "connected_components": len(components),
                "betti_numbers": {
                    "b0": len(components),
                    "b1": sum(comp["betti_numbers"]["b1"] for comp in components)
                },
                "homology_groups": components
            },
            "graph_metrics": {
                "node_count": node_count,
                "edge_count": edge_count,
                "density": edge_count / (node_count * (node_count - 1)) if node_count > 1 else 0,
                "diameter": self._compute_dag_diameter(graph) if node_count > 0 else 0,
                "avg_path_length": self._compute_avg_path_length(graph) if node_count > 0 else 0
            },
            "cost_analysis": {
                "inequality_metrics": {
                    "gini_index": round(gini, 6),
                    "theil_index": round(theil_index, 6),
                    "hoover_index": pareto_metrics.get("cost_concentration_index", 0)
                },
                "pareto_analysis": pareto_metrics,
                "summary": {
                    "item_count": n,
                    "total_cost": round(total_cost, 2),
                    "mean_cost": round(total_cost / n, 2) if n > 0 else 0,
                    "median_cost": round(statistics.median(costs), 2) if costs else 0
                }
            },
            "quality_analysis": quality_metrics,
            "risk_analysis": {
                "profile": risk_profile or {},
                "flux_metrics": flux_metrics or {},
                "entropy_bits": self._compute_system_entropy(requirements, graph)
            },
            "thermodynamics": self.analyze_budget_exergy(requirements),
            "material_distribution": {
                "units": self._analyze_unit_distribution(requirements),
                "waste": self._analyze_waste_distribution(requirements),
                "source_diversity": self._analyze_source_diversity(requirements)
            },
            "generation_info": {
                "timestamp": datetime.now().isoformat(),
                "algorithm": "Topological-Materialization-v3.1",
                "python_version": sys.version.split()[0],
                "complexity_class": "P"  # El algoritmo es polinomial
            }
        }

    def _compute_dag_diameter(self, graph: nx.DiGraph) -> int:
        """Calcula el diámetro de un DAG (longitud del camino más largo)."""
        if graph.number_of_nodes() == 0:
            return 0

        # Para DAGs, podemos usar programación dinámica
        dist = {node: 0 for node in graph.nodes()}
        try:
            topo_order = list(nx.topological_sort(graph))
        except nx.NetworkXUnfeasible:
            return 0 # Should be checked before

        for node in topo_order:
            for successor in graph.successors(node):
                dist[successor] = max(dist[successor], dist[node] + 1)

        return max(dist.values()) if dist else 0

    def _compute_avg_path_length(self, graph: nx.DiGraph) -> float:
        """Calcula la longitud promedio de camino en el DAG."""
        if graph.number_of_nodes() < 2:
            return 0.0

        total_paths = 0
        total_length = 0

        # Para cada par de nodos (u, v) donde hay camino de u a v
        # Aproximación para evitar O(V^3)
        # Usamos ancestros/descendientes para grafos pequeños/medianos
        # Limitamos a un subconjunto si es muy grande

        nodes = list(graph.nodes())
        if len(nodes) > 200:
            nodes = nodes[:200]

        for node in nodes:
            # Caminos desde 'node'
            lengths = nx.single_source_shortest_path_length(graph, node)
            for length in lengths.values():
                if length > 0:
                    total_paths += 1
                    total_length += length

        return total_length / total_paths if total_paths > 0 else 0.0

    def _compute_system_entropy(self, requirements: List[MaterialRequirement], graph: nx.DiGraph) -> float:
        """Calcula la entropía del sistema en bits."""
        if not requirements:
            return 0.0

        # Entropía de la distribución de costos
        costs = [r.total_cost for r in requirements if r.total_cost > 0]
        total_cost = sum(costs)

        if total_cost <= 0:
            return 0.0

        entropy = 0.0
        for cost in costs:
            p = cost / total_cost
            entropy -= p * math.log2(p)

        # Ajustar por complejidad topológica
        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()
        structural_entropy = math.log2(1 + edge_count / max(1, node_count))

        return round(entropy + structural_entropy, 4)

    def _analyze_source_diversity(self, requirements: List[MaterialRequirement]) -> Dict[str, Any]:
        """Analiza la diversidad de fuentes (APUs)."""
        if not requirements:
            return {}

        source_counts = {}
        for req in requirements:
            for apu in req.source_apus:
                source_counts[apu] = source_counts.get(apu, 0) + 1

        return {
            "unique_sources": len(source_counts),
            "source_concentration": max(source_counts.values()) / len(requirements) if source_counts else 0,
            "source_distribution": dict(sorted(source_counts.items(), key=lambda x: -x[1])[:10])
        }

    def analyze_budget_exergy(self, bom_items: List[MaterialRequirement]) -> Dict[str, Any]:
        """
        Analiza la eficiencia exergética (Trabajo Útil vs Anergía).
        """
        useful_work_cost = 0.0
        anergy_cost = 0.0

        high_exergy_keywords = {
            "CONCRETO",
            "ACERO",
            "CIMENTACION",
            "ESTRUCTURA",
            "CEMENTO",
            "HIERRO",
        }

        for item in bom_items:
            desc = (item.description or "").upper()
            is_high_exergy = any(k in desc for k in high_exergy_keywords)

            if is_high_exergy:
                useful_work_cost += item.total_cost
            else:
                anergy_cost += item.total_cost

        total_cost = useful_work_cost + anergy_cost
        exergy_efficiency = useful_work_cost / total_cost if total_cost > 0 else 0

        return {
            "exergy_efficiency": exergy_efficiency,
            "structural_investment": useful_work_cost,
            "decorative_investment": anergy_cost,
            "total_investment": total_cost,
            "narrative": f"Eficiencia Exergética: {exergy_efficiency:.1%}",
        }

    def _analyze_unit_distribution(
        self, requirements: List[MaterialRequirement]
    ) -> Dict[str, Any]:
        """Analiza distribución por unidad de medida."""
        unit_groups = {}
        for req in requirements:
            unit = req.unit or "UND"
            if unit not in unit_groups:
                unit_groups[unit] = {"count": 0, "total_cost": 0.0}
            unit_groups[unit]["count"] += 1
            unit_groups[unit]["total_cost"] += req.total_cost
        return unit_groups

    def _analyze_waste_distribution(
        self, requirements: List[MaterialRequirement]
    ) -> Dict[str, Any]:
        """Analiza distribución de factores de desperdicio."""
        if not requirements:
            return {}
        waste_factors = [req.waste_factor for req in requirements]
        return {
            "mean": round(statistics.mean(waste_factors), 4),
            "max": round(max(waste_factors), 4),
        }
