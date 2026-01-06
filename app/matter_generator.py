import logging
import math
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import networkx as nx

# Integración con telemetría
try:
    from .telemetry import TelemetryContext
except ImportError:
    TelemetryContext = Any


@dataclass
class MaterialRequirement:
    """
    Representa un requerimiento de material consolidado con validación.

    Garantiza invariantes como cantidad base positiva y costos finitos.

    Attributes:
        id (str): Identificador único del material.
        description (str): Descripción legible del material.
        quantity_base (float): Cantidad base requerida (antes de desperdicio).
        unit (str): Unidad de medida.
        waste_factor (float): Factor de desperdicio aplicado (ej. 0.05 para 5%).
        quantity_total (float): Cantidad total incluyendo desperdicio.
        unit_cost (float): Costo unitario representativo.
        total_cost (float): Costo total (quantity_total * unit_cost).
        source_apus (List[str]): Lista de IDs de APUs que requieren este material.
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

    def __post_init__(self):
        """Validación de invariantes después de la inicialización."""
        if self.quantity_base <= 0:
            raise ValueError(f"Cantidad base no positiva para material {self.id}")

        if not math.isfinite(self.total_cost):
            raise ValueError(f"Costo total no finito para material {self.id}")


@dataclass
class BillOfMaterials:
    """
    Lista de Materiales (BOM) con metadata de validación.

    Attributes:
        requirements (List[MaterialRequirement]): Lista de materiales.
        total_material_cost (float): Costo total acumulado.
        metadata (Dict[str, Any]): Metadatos de generación y análisis.
    """

    requirements: List[MaterialRequirement]
    total_material_cost: float
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Validación de coherencia interna."""
        self.validate_consistency()

    def validate_consistency(self):
        """Valida que la suma de costos coincida con el total declarado."""
        computed_total = sum(req.total_cost for req in self.requirements)
        if not math.isclose(
            self.total_material_cost, computed_total, rel_tol=1e-5, abs_tol=1e-2
        ):
            raise ValueError(
                f"BOM inconsistente: suma ({computed_total}) != total ({self.total_material_cost})"
            )


class MatterGenerator:
    """
    Motor de Materialización Híbrido (Topológico + Algebraico).

    Transforma el grafo abstracto del proyecto en una lista concreta de
    materiales (Colapso de Onda).

    Características:
    - Validación de complejidad topológica (densidad, ciclos).
    - Algoritmo DFS optimizado para trazabilidad profunda.
    - Suma compensada de Kahan para precisión numérica.
    - Aplicación de factores de entropía (desperdicio).
    """

    def __init__(self, max_graph_complexity: int = 100000):
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
        Orquesta la transformación del Grafo en BOM.

        Args:
            graph: Grafo del proyecto.
            risk_profile: Perfil de riesgo externo.
            flux_metrics: Métricas de flujo (estabilidad piramidal, etc.).
            telemetry: Contexto para registrar métricas de ejecución.

        Returns:
            BillOfMaterials: Objeto BOM validado.
        """
        self.logger.info("🌌 Iniciando materialización híbrida del proyecto...")

        if telemetry:
            telemetry.start_step("materialize_project")

        try:
            node_count = graph.number_of_nodes()
            edge_count = graph.number_of_edges()

            if node_count == 0:
                raise ValueError("Grafo vacío: no hay nodos para procesar")

            density = edge_count / (node_count * (node_count - 1)) if node_count > 1 else 0
            complexity = node_count * max(edge_count, 1) * (1 + density)

            if complexity > self.max_graph_complexity:
                raise OverflowError(
                    f"Complejidad del grafo ({complexity:.0f}) excede el límite. "
                    f"Nodos: {node_count}, Aristas: {edge_count}"
                )

            if not nx.is_directed_acyclic_graph(graph):
                raise ValueError("El grafo contiene ciclos (no es DAG).")

            # 1. Colapso de Onda con DFS (Búsqueda en Profundidad)
            materials = []

            # Identificación de raíces (Nodos sin aristas entrantes)
            root_nodes = [node for node, in_degree in graph.in_degree() if in_degree == 0]

            if not root_nodes:
                if node_count > 0:
                    self.logger.warning(
                        "Grafo sin raíces formales - usando todos los nodos como raíces"
                    )
                    root_nodes = list(graph.nodes())
                else:
                    empty_meta = self._generate_metadata(
                        graph, risk_profile or {}, flux_metrics or {}, []
                    )
                    return BillOfMaterials([], 0.0, empty_meta)

            stack = [(root, 1.0, frozenset(), [], None, 0) for root in root_nodes]
            max_depth = node_count * 2

            visited_edges = set()
            iteration_count = 0
            max_iterations = self.max_graph_complexity * 2

            while stack:
                iteration_count += 1
                if iteration_count > max_iterations:
                    raise RuntimeError("Límite de iteraciones excedido (Posible ciclo).")

                (
                    current_node,
                    current_qty,
                    path_set,
                    path_list,
                    parent_apu,
                    depth,
                ) = stack.pop()

                if depth > max_depth:
                    continue

                if current_node in path_set:
                    continue  # Ciclo detectado en camino actual

                node_data = graph.nodes.get(current_node, {})
                node_type = node_data.get("type", "UNDEFINED")

                new_path_set = path_set | {current_node}
                new_path_list = path_list + [current_node]

                # Objeto terminal (Insumo/Material)
                if node_type == "INSUMO":
                    description = (
                        node_data.get("description")
                        or node_data.get("name")
                        or str(current_node)
                    )

                    unit_cost = node_data.get("unit_cost", 0.0)
                    try:
                        unit_cost = float(unit_cost)
                        if not math.isfinite(unit_cost) or unit_cost < 0:
                            unit_cost = 0.0
                    except (TypeError, ValueError):
                        unit_cost = 0.0

                    if not math.isfinite(current_qty) or current_qty <= 0:
                        current_qty = 1.0

                    materials.append(
                        {
                            "id": str(current_node),
                            "description": description,
                            "base_qty": current_qty,
                            "unit_cost": unit_cost,
                            "source_apu": parent_apu or "ROOT",
                            "unit": node_data.get("unit", "UND"),
                            "node_data": node_data,
                            "composition_path": new_path_list,
                            "fiber_depth": len(new_path_list),
                            "topological_order": depth,
                        }
                    )
                    continue

                next_parent_apu = current_node if node_type == "APU" else parent_apu

                for successor in graph.successors(current_node):
                    edge_key = (current_node, successor)
                    edge_data = graph.edges.get(edge_key, {})
                    edge_qty = edge_data.get("quantity", 1.0)

                    try:
                        edge_qty = float(edge_qty)
                        if not math.isfinite(edge_qty) or edge_qty <= 0:
                            edge_qty = 1.0
                    except (TypeError, ValueError):
                        edge_qty = 1.0

                    new_qty = current_qty * edge_qty

                    if not math.isfinite(new_qty):
                        continue

                    stack.append(
                        (
                            successor,
                            new_qty,
                            new_path_set,
                            new_path_list,
                            next_parent_apu,
                            depth + 1,
                        )
                    )

            raw_materials = materials
            self.logger.info(
                f"🧱 Materiales brutos extraídos: {len(raw_materials)} (nodos: {node_count})"
            )

            if not raw_materials:
                self.logger.warning("⚠️ No se encontraron materiales en el grafo")
                metadata = self._generate_metadata(graph, risk_profile, flux_metrics, [])
                return BillOfMaterials([], 0.0, metadata)

            # 2. Aplicación de Entropía (Factores de Desperdicio)
            adjusted_materials = self._apply_entropy_factors(
                raw_materials, flux_metrics, risk_profile
            )

            # 3. Clustering Semántico
            final_requirements = self._cluster_semantically(adjusted_materials)
            self.logger.info(f"🛒 Requerimientos consolidados: {len(final_requirements)}")

            # 4. Cálculo de Totales (Algoritmo Kahan)
            total_cost = self._compute_total_cost(final_requirements)

            # 5. Metadata Estratégica
            metadata = self._generate_metadata(
                graph, risk_profile, flux_metrics, final_requirements
            )

            if telemetry:
                telemetry.record_metric("materialization", "total_material_cost", total_cost)
                telemetry.record_metric(
                    "materialization", "item_count", len(final_requirements)
                )
                telemetry.end_step("materialization", "success")

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

    def _apply_entropy_factors(
        self,
        raw_materials: List[Dict[str, Any]],
        flux_metrics: Optional[Dict[str, Any]],
        risk_profile: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Aplica factores de entropía (desperdicio) basados en riesgo y flujo.

        Args:
            raw_materials: Lista de materiales crudos.
            flux_metrics: Métricas del Flujo (saturación, estabilidad).
            risk_profile: Perfil de riesgo del proyecto.

        Returns:
            List[Dict]: Materiales con cantidades ajustadas por entropía.
        """
        base_factor = 1.0
        factor_trace = {"identity": 1.0}

        if flux_metrics:
            saturation = flux_metrics.get("avg_saturation", 0.0)
            if saturation > 0.8:
                base_factor *= 1.05
                factor_trace["saturation"] = 1.05

            stability = flux_metrics.get("pyramid_stability", 1.0)
            if 0 < stability < 1.0:
                base_factor *= 1.03
                factor_trace["instability"] = 1.03

        if risk_profile:
            risk_level = str(risk_profile.get("level", "MEDIUM")).upper()
            risk_map = {"LOW": 1.01, "MEDIUM": 1.03, "HIGH": 1.07, "CRITICAL": 1.15}
            risk_mult = risk_map.get(risk_level, 1.03)
            base_factor *= risk_mult
            factor_trace["risk"] = risk_mult

        processed_materials = []

        material_factors = {
            "FRAGILE": 1.02,
            "PERISHABLE": 1.04,
            "HAZARDOUS": 1.06,
            "BULKY": 1.02,
            "PRECISION": 1.01,
        }

        for mat in raw_materials:
            processed = mat.copy()

            cat = mat.get("node_data", {}).get("material_category", "GENERIC")
            spec_factor = material_factors.get(cat, 1.0)

            total_multiplier = base_factor * spec_factor

            if total_multiplier < 1.0:
                total_multiplier = 1.0

            processed["waste_factor"] = total_multiplier - 1.0
            processed["total_qty"] = mat["base_qty"] * total_multiplier
            processed["applied_factors"] = {
                **factor_trace,
                "specific": spec_factor,
                "total": total_multiplier,
            }

            processed_materials.append(processed)

        return processed_materials

    def _cluster_semantically(
        self, materials: List[Dict[str, Any]]
    ) -> List[MaterialRequirement]:
        """
        Agrupa materiales semánticamente (por ID y Unidad).

        Utiliza estadísticas robustas (mediana) para determinar el costo unitario
        en caso de discrepancias.
        """
        clustered = {}
        unit_normalization = {
            "M": 1.0,
            "M2": 1.0,
            "M3": 1.0,
            "KG": 1.0,
            "TON": 1000.0,
            "UND": 1.0,
            "UNIT": 1.0,
            "L": 1.0,
            "GL": 3.78541,
        }

        for mat in materials:
            unit = mat.get("unit", "UND").upper()
            norm_unit = unit_normalization.get(unit, 1.0)
            key = (mat["id"], unit)

            if key not in clustered:
                clustered[key] = {
                    "id": mat["id"],
                    "description": mat["description"],
                    "unit": unit,
                    "quantity_base": 0.0,
                    "quantity_total": 0.0,
                    "source_apus": set(),
                    "cost_samples": [],
                    "waste_factors": [],
                    "paths": [],
                    "normalization_factor": norm_unit,
                }

            data = clustered[key]
            base_qty = mat["base_qty"]
            total_qty = mat.get("total_qty", base_qty)

            data["quantity_base"] += base_qty
            data["quantity_total"] += total_qty
            data["source_apus"].add(mat["source_apu"])

            cost = mat["unit_cost"]
            if math.isfinite(cost):
                data["cost_samples"].append(cost)

            waste = mat.get("waste_factor", 0.0)
            if math.isfinite(waste):
                data["waste_factors"].append(waste)

            if len(data["paths"]) < 3:
                data["paths"].append(mat.get("composition_path", [])[-3:])

        requirements = []

        for (mid, unit), data in clustered.items():
            if data["quantity_base"] <= 0:
                continue

            # Cálculo de desperdicio promedio
            total_waste = 0.0
            if data["waste_factors"]:
                total_waste = sum(data["waste_factors"]) / len(data["waste_factors"])
            else:
                total_waste = (data["quantity_total"] / data["quantity_base"]) - 1.0

            # Costo unitario representativo (Mediana robusta)
            costs = sorted([c for c in data["cost_samples"] if math.isfinite(c)])
            rep_cost = 0.0

            if costs:
                mid_idx = len(costs) // 2
                if len(costs) % 2 == 1:
                    rep_cost = costs[mid_idx]
                else:
                    rep_cost = (costs[mid_idx - 1] + costs[mid_idx]) / 2.0

            if not math.isfinite(rep_cost) or rep_cost < 0:
                rep_cost = 0.0

            total_cost = data["quantity_total"] * rep_cost
            if not math.isfinite(total_cost):
                total_cost = 0.0

            req = MaterialRequirement(
                id=data["id"],
                description=data["description"],
                quantity_base=round(data["quantity_base"], 6),
                unit=data["unit"],
                waste_factor=round(max(0.0, total_waste), 6),
                quantity_total=round(data["quantity_total"], 6),
                unit_cost=round(rep_cost, 4),
                total_cost=round(total_cost, 2),
                source_apus=sorted([str(x) for x in data["source_apus"]]),
            )
            requirements.append(req)

        requirements.sort(key=lambda x: (-x.total_cost, x.description or "", x.id))
        return requirements

    def analyze_budget_exergy(self, bom_items: List[MaterialRequirement]) -> Dict[str, Any]:
        """
        Analiza la eficiencia exergética (Trabajo Útil vs Anergía).

        Clasifica la inversión en estructura/funcionalidad (alta exergía)
        vs acabados/desperdicio (anergía).
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

    def _compute_total_cost(self, requirements: List[MaterialRequirement]) -> float:
        """
        Calcula el costo total usando Suma Compensada de Kahan.

        Minimiza el error de punto flotante al sumar muchos valores pequeños.
        """
        total = 0.0
        c = 0.0

        # Ordenar por magnitud para estabilidad
        costs = sorted(
            [req.total_cost for req in requirements if math.isfinite(req.total_cost)],
            key=abs,
        )

        for cost in costs:
            y = cost - c
            t = total + y
            c = (t - total) - y
            total = t

        if not math.isfinite(total):
            raise OverflowError("Overflow en cálculo de costo total.")

        return round(total, 2)

    def _generate_metadata(
        self,
        graph: nx.DiGraph,
        risk_profile: Dict[str, Any],
        flux_metrics: Dict[str, Any],
        requirements: List[MaterialRequirement],
    ) -> Dict[str, Any]:
        """Genera metadatos estratégicos y estadísticas."""
        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()

        # Análisis de Pareto
        costs = [r.total_cost for r in requirements]
        total_cost = sum(costs)
        n = len(costs)

        pareto_metrics = {
            "pareto_80_items_ratio": 0.0,
            "pareto_20_cost_percentage": 0.0,
            "top_10_items_cost": 0.0,
            "cost_concentration_index": 0.0,
        }

        gini = 0.0

        if total_cost > 0 and n > 0:
            sorted_desc = sorted(costs, reverse=True)
            accum = 0.0
            items_80 = 0

            for c in sorted_desc:
                accum += c
                items_80 += 1
                if accum >= total_cost * 0.8:
                    break

            pareto_metrics["pareto_80_items_ratio"] = items_80 / n

            # Gini
            sorted_asc = sorted(costs)
            cum_weighted = sum((i + 1) * c for i, c in enumerate(sorted_asc))
            gini = (2.0 * cum_weighted) / (n * total_cost) - (n + 1.0) / n
            gini = max(0.0, min(1.0, gini))

        return {
            "topological_analysis": {
                "is_dag": nx.is_directed_acyclic_graph(graph),
                "euler_characteristic": node_count - edge_count,
            },
            "topological_invariants": { # Alias for testing V3.0
                "is_dag": nx.is_directed_acyclic_graph(graph),
                "euler_characteristic": node_count - edge_count,
            },
            "graph_metrics": {
                "node_count": node_count,
                "edge_count": edge_count,
            },
            "cost_analysis": {
                "pareto_analysis": pareto_metrics,
                "inequality_metrics": {"gini_index": round(gini, 4)},
                "summary": {"item_count": n, "total_cost": round(total_cost, 2)},
                # Compatibilidad
                "item_count": n,
                "total_cost": round(total_cost, 2),
                "gini_index": round(gini, 4),
            },
            "risk_analysis": {
                "profile": risk_profile or {},
                "flux_metrics": flux_metrics or {},
            },
            "thermodynamics": self.analyze_budget_exergy(requirements),
            "generation_info": {
                "timestamp": datetime.now().isoformat(),
                "algorithm": "Hybrid-Topological-Algebraic-v2.1",
                "python_version": sys.version.split()[0],
            },
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
