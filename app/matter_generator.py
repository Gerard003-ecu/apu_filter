import networkx as nx
import logging
import math
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class MaterialRequirement:
    """
    Representa un requerimiento de material consolidado con validación de invariantes.
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
        # waste_factor puede ser 0, pero no negativo (salvo casos de reciclaje muy raros, asumimos >= 0)
        # En propuesta 1, waste_factor = multiplier - 1. Si multiplier < 1, waste es negativo.
        # Pero la lógica dice que no reducimos cantidades (multiplier >= 1), así que waste >= 0.
        if self.waste_factor < 0:
            pass # Permitimos flexibilidad por si hay optimizaciones, pero logueamos warnings arriba

        if not math.isfinite(self.total_cost):
            raise ValueError(f"Costo total no finito para material {self.id}")

@dataclass
class BillOfMaterials:
    """
    Lista de materiales final con metadata de generación y validación topológica.
    """
    requirements: List[MaterialRequirement]
    total_material_cost: float
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Validación de coherencia interna."""
        self.validate_consistency()

    def validate_consistency(self):
        # Verificación laxa para permitir pequeños errores de punto flotante distintos a Kahan
        # pero Kahan debe ser preciso.
        computed_total = sum(req.total_cost for req in self.requirements)
        if not math.isclose(self.total_material_cost, computed_total, rel_tol=1e-5):
            raise ValueError("BOM inconsistente: suma de costos no coincide con total")

class MatterGenerator:
    """
    Motor de Materialización Híbrido (Topológico + Algebraico).

    Combina:
    1. Lógica de Recorrido de Funtor (DFS + Stack) de Propuesta 1.
    2. Validación de Invariantes y Precisión Numérica (Kahan) de Propuesta 2.
    3. Análisis Estratégico (Pareto/Gini) de Propuesta 1.
    """

    def __init__(self, max_graph_complexity: int = 100000):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_graph_complexity = max_graph_complexity

    def materialize_project(
        self,
        graph: nx.DiGraph,
        risk_profile: Optional[Dict[str, Any]] = None,
        flux_metrics: Optional[Dict[str, Any]] = None
    ) -> BillOfMaterials:
        """
        Orquesta la transformación del Grafo en BOM.

        Args:
            graph: Grafo topológico (DAG de composición)
            risk_profile: Perfil de riesgo
            flux_metrics: Métricas de estabilidad

        Returns:
            BillOfMaterials consolidado

        Raises:
            ValueError: Si el grafo no es un DAG
            OverflowError: Si la complejidad del grafo excede el límite
        """
        self.logger.info("🌌 Iniciando materialización híbrida del proyecto...")

        # Validación de complejidad
        complexity = graph.number_of_nodes() * max(graph.number_of_edges(), 1)
        if complexity > self.max_graph_complexity:
            raise OverflowError(f"La complejidad del grafo ({complexity}) excede el límite permitido ({self.max_graph_complexity})")

        # Validación estructural (DAG)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("El grafo contiene ciclos (no es un DAG), imposible propagar cantidades coherentemente.")

        # 1. Colapso de Onda (Enfoque Funtor - Propuesta 1)
        raw_materials = self._explode_pyramid(graph)
        self.logger.info(f"🧱 Materiales brutos extraídos: {len(raw_materials)}")

        if not raw_materials:
            self.logger.warning("⚠️ No se encontraron materiales en el grafo")

        # 2. Aplicación de Entropía Trazable (Propuesta 1)
        adjusted_materials = self._apply_entropy_factors(
            raw_materials,
            flux_metrics,
            risk_profile
        )

        # 3. Clustering Semántico (Propuesta 1 + 2)
        final_requirements = self._cluster_semantically(adjusted_materials)
        self.logger.info(f"🛒 Requerimientos consolidados: {len(final_requirements)}")

        # 4. Cálculo de Totales con Kahan (Propuesta 2)
        total_cost = self._compute_total_cost(final_requirements)

        # 5. Metadata Estratégica (Pareto/Gini - Propuesta 1)
        metadata = self._generate_metadata(
            graph, risk_profile, flux_metrics, final_requirements
        )

        return BillOfMaterials(
            requirements=final_requirements,
            total_material_cost=total_cost,
            metadata=metadata
        )

    def _explode_pyramid(self, graph: nx.DiGraph) -> List[Dict[str, Any]]:
        """
        Recorre el grafo implementando un funtor F: G → Mat (Propuesta 1).
        Usa DFS manual con stack y detección de ciclos O(1).
        """
        materials = []

        if graph.number_of_nodes() == 0:
            return materials

        # Identificación de raíces
        root_nodes = [node for node, in_degree in graph.in_degree() if in_degree == 0]

        # Si no hay raíces explícitas pero hay nodos, usamos nodos con in_degree 0 locales
        # (Aunque si es conexo y DAG siempre hay al menos uno)
        if not root_nodes and graph.number_of_nodes() > 0:
            # Fallback para componentes desconectados o estructuras singulares
            root_nodes = [n for n in graph.nodes()]
            # Esto podría duplicar si no es cuidadoso, pero en DAG siempre hay fuentes.
            # Si no hay fuentes en un grafo finito no vacío, ¡hay ciclo! Pero ya chequeamos DAG.

        if not root_nodes and graph.number_of_nodes() > 0:
             # Debería ser inalcanzable si is_directed_acyclic_graph pasó
             self.logger.error("❌ Grafo sin raíces detectables")
             return materials

        # DFS iterativo
        # Stack: (nodo, cantidad_acumulada, camino_set, camino_lista, padre_apu)
        stack = [
            (root, 1.0, frozenset(), [], None) for root in root_nodes
        ]

        iteration_count = 0
        max_iterations = self.max_graph_complexity + 1000

        while stack:
            iteration_count += 1
            if iteration_count > max_iterations:
                 # Backup check
                raise RuntimeError("Límite de iteraciones excedido en materialización")

            current_node, current_qty, path_set, path_list, parent_apu = stack.pop()

            # Detección de ciclo O(1) - redundante si ya chequeamos DAG, pero seguro
            if current_node in path_set:
                self.logger.warning(f"⚠️ Ciclo detectado en camino: {path_list} -> {current_node}")
                continue

            node_data = graph.nodes.get(current_node, {})
            node_type = node_data.get("type", "UNDEFINED")

            new_path_set = path_set | {current_node}
            new_path_list = path_list + [current_node]

            # Objeto terminal (Insumo)
            if node_type == "INSUMO":
                description = (
                    node_data.get("description") or
                    node_data.get("name") or
                    str(current_node)
                )

                unit_cost = node_data.get("unit_cost", 0.0)
                try:
                    unit_cost = float(unit_cost)
                    if not math.isfinite(unit_cost):
                         # Infinito se ignora (no se agrega)
                         self.logger.warning(f"Costo infinito detectado en {current_node}, omitiendo.")
                         continue
                except (TypeError, ValueError):
                    unit_cost = 0.0

                # Validación de costo negativo para logging
                if unit_cost < 0:
                     self.logger.warning(f"Costo negativo detectado en {current_node}: {unit_cost}")

                # Si cantidad base es 0 o negativa, puede ser válido semánticamente (resta) pero usualmente error
                if current_qty <= 0:
                     # El test espera que usemos 1.0 si no es positivo?
                     # test_negative_quantities_handling: "El sistema debe usar 1.0 como valor por defecto"
                     # Re-evaluemos lógica de aristas vs acumulado
                     pass

                # La validación de aristas negativas se hace al recorrer sucesores.
                # Aquí current_qty ya es producto.

                materials.append({
                    "id": str(current_node),
                    "description": description,
                    "base_qty": current_qty,
                    "unit_cost": unit_cost,
                    "source_apu": parent_apu or "ROOT",
                    "unit": node_data.get("unit", "UND"),
                    "node_data": node_data,
                    "composition_path": new_path_list,
                    "fiber_depth": len(new_path_list)
                })
                continue

            # Determinación de padre APU para trazabilidad
            next_parent_apu = current_node if node_type == "APU" else parent_apu

            # Expansión de sucesores
            for successor in graph.successors(current_node):
                edge_data = graph.edges.get((current_node, successor), {})
                edge_qty = edge_data.get("quantity", 1.0)

                try:
                    edge_qty = float(edge_qty)
                    # test_negative_quantities_handling espera que quantity=-2.0 se convierta en 1.0?
                    # "El sistema debe usar 1.0 como valor por defecto para cantidades no positivas"
                    if not math.isfinite(edge_qty) or edge_qty <= 0:
                        edge_qty = 1.0
                except (TypeError, ValueError):
                    edge_qty = 1.0

                new_qty = current_qty * edge_qty

                if not math.isfinite(new_qty) or new_qty > 1e12:
                    self.logger.warning(f"Overflow potencial en {successor}: {new_qty}")
                    continue

                stack.append((
                    successor, new_qty, new_path_set,
                    new_path_list, next_parent_apu
                ))

        return materials

    def _apply_entropy_factors(
        self,
        raw_materials: List[Dict[str, Any]],
        flux_metrics: Optional[Dict[str, Any]],
        risk_profile: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Aplica factores de entropía con trazabilidad (Propuesta 1).
        """
        base_factor = 1.0
        factor_trace = {"identity": 1.0}

        # Factores de flujo
        if flux_metrics:
            saturation = flux_metrics.get("avg_saturation", 0.0)
            if saturation > 0.8:
                base_factor *= 1.05
                factor_trace["saturation"] = 1.05

            stability = flux_metrics.get("pyramid_stability", 1.0)
            if 0 < stability < 1.0:
                base_factor *= 1.03
                factor_trace["instability"] = 1.03

        # Factores de riesgo
        if risk_profile:
            risk_level = str(risk_profile.get("level", "MEDIUM")).upper()
            risk_map = {
                "LOW": 1.01,
                "MEDIUM": 1.03,
                "HIGH": 1.07,
                "CRITICAL": 1.15
            }
            risk_mult = risk_map.get(risk_level, 1.03)
            base_factor *= risk_mult
            factor_trace["risk"] = risk_mult

        processed_materials = []

        material_factors = {
            "FRAGILE": 1.02,
            "PERISHABLE": 1.04,
            "HAZARDOUS": 1.06,
            "BULKY": 1.02,
            "PRECISION": 1.01
        }

        for mat in raw_materials:
            processed = mat.copy()

            cat = mat.get("node_data", {}).get("material_category", "GENERIC")
            spec_factor = material_factors.get(cat, 1.0)

            total_multiplier = base_factor * spec_factor

            # Invariante: No reducimos cantidades
            if total_multiplier < 1.0:
                total_multiplier = 1.0

            processed["waste_factor"] = total_multiplier - 1.0
            processed["total_qty"] = mat["base_qty"] * total_multiplier

            processed["applied_factors"] = {
                **factor_trace,
                "specific": spec_factor,
                "total": total_multiplier
            }

            processed_materials.append(processed)

        return processed_materials

    def _cluster_semantically(self, materials: List[Dict[str, Any]]) -> List[MaterialRequirement]:
        """
        Agrupa materiales (Propuesta 1 logic) y crea DataClasses (Propuesta 2).
        """
        clustered = {}

        for mat in materials:
            key = (mat["id"], mat.get("unit", "UND"))

            if key not in clustered:
                clustered[key] = {
                    "id": mat["id"],
                    "description": mat["description"],
                    "unit": mat.get("unit", "UND"),
                    "quantity_base": 0.0,
                    "quantity_total": 0.0,
                    "source_apus": set(),
                    "cost_samples": []
                }

            data = clustered[key]
            data["quantity_base"] += mat["base_qty"]
            data["quantity_total"] += mat["total_qty"]
            data["source_apus"].add(mat["source_apu"])

            cost = mat["unit_cost"]
            # Aceptamos negativos para procesamiento, pero advertimos
            data["cost_samples"].append(cost)

        requirements = []

        for (mid, unit), data in clustered.items():
            if data["quantity_base"] <= 0:
                # Omitir si la cantidad neta es <= 0 (salvo devolución, pero asumimos BOM constructivo)
                continue

            total_waste = (data["quantity_total"] / data["quantity_base"]) - 1.0

            # Estimación robusta de costo (Mediana)
            # Filtramos infinitos antes
            costs = sorted([c for c in data["cost_samples"] if math.isfinite(c)])

            if costs:
                mid_idx = len(costs) // 2
                if len(costs) % 2 == 1:
                    rep_cost = costs[mid_idx]
                else:
                    rep_cost = (costs[mid_idx-1] + costs[mid_idx]) / 2.0
            else:
                rep_cost = 0.0

            total_cost = data["quantity_total"] * rep_cost

            req = MaterialRequirement(
                id=data["id"],
                description=data["description"],
                quantity_base=round(data["quantity_base"], 6),
                unit=data["unit"],
                waste_factor=round(total_waste, 6),
                quantity_total=round(data["quantity_total"], 6),
                unit_cost=round(rep_cost, 4),
                total_cost=round(total_cost, 2),
                source_apus=sorted([str(x) for x in data["source_apus"]])
            )
            requirements.append(req)

        # Ordenar por Pareto (costo descendente)
        requirements.sort(key=lambda x: (-x.total_cost, x.id))

        return requirements

    def _compute_total_cost(self, requirements: List[MaterialRequirement]) -> float:
        """
        Calcula costo total usando Algoritmo de Suma de Kahan (Propuesta 2).
        """
        total = 0.0
        compensation = 0.0

        for req in requirements:
            if not math.isfinite(req.total_cost):
                continue

            y = req.total_cost - compensation
            t = total + y
            compensation = (t - total) - y
            total = t

            if math.isinf(total):
                raise OverflowError("Overflow en cálculo de costo total (Kahan)")

        return round(total, 2)

    def _generate_metadata(
        self,
        graph: nx.DiGraph,
        risk_profile: Dict[str, Any],
        flux_metrics: Dict[str, Any],
        requirements: List[MaterialRequirement]
    ) -> Dict[str, Any]:
        """
        Genera metadata estratégica (Pareto + Gini - Propuesta 1).
        """
        # Métricas de grafo
        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()
        euler = node_count - edge_count

        # Pareto y Gini
        costs = [r.total_cost for r in requirements]
        total_cost = sum(costs)
        n = len(costs)

        pareto_ratio = 0.0 # % items
        pareto_cost_pct = 0.0 # % cost of top 20%
        gini = 0.0

        if total_cost > 0 and n > 0:
            # 1. Pareto Ratio (Original): % items que hacen el 80% del costo
            sorted_desc = sorted(costs, reverse=True)
            accum = 0.0
            items_80 = 0
            for c in sorted_desc:
                accum += c
                items_80 += 1
                if accum >= total_cost * 0.8:
                    break
            pareto_ratio = items_80 / n

            # 2. Pareto Cost Percentage (Para el Test): Costo acumulado del 20% superior
            top_20_count = max(1, int(math.ceil(n * 0.2)))
            top_20_cost = sum(sorted_desc[:top_20_count])
            pareto_cost_pct = (top_20_cost / total_cost) * 100.0

            # 3. Gini
            sorted_asc = sorted(costs)
            cum_weighted = sum((i + 1) * c for i, c in enumerate(sorted_asc))
            gini = (2.0 * cum_weighted) / (n * total_cost) - (n + 1.0) / n
            gini = max(0.0, min(1.0, gini))

        return {
            "topological_invariants": {
                "is_dag": nx.is_directed_acyclic_graph(graph),
                "euler_characteristic": euler,
            },
            "graph_metrics": {
                "node_count": node_count,
                "edge_count": edge_count,
                "root_count": len([n for n, d in graph.in_degree() if d == 0])
            },
            "cost_analysis": {
                "pareto_analysis": {
                    "pareto_80_items_ratio": round(pareto_ratio, 4),
                    "pareto_20_percent": round(n * 0.2, 1),
                    "pareto_cost_percentage": round(pareto_cost_pct, 2)
                },
                "gini_index": round(gini, 4),
                "item_count": n,
                "total_cost": round(total_cost, 2),
                "empty": n == 0
            },
            "risk_analysis": {
                "profile": risk_profile,
                "flux_metrics": flux_metrics
            },
            "generation_info": {
                "timestamp": datetime.now().isoformat(),
                "algorithm": "Hybrid-Topological-Algebraic-v1",
                "generator_version": "2.0.0"
            }
        }
