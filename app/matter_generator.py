import networkx as nx
import logging
import math
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class MaterialRequirement:
    """
    Representa un requerimiento de material consolidado.
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

@dataclass
class BillOfMaterials:
    """
    Lista de materiales final con metadata de generación.
    """
    requirements: List[MaterialRequirement]
    total_material_cost: float
    metadata: Dict[str, Any]

class MatterGenerator:
    """
    Motor de Materialización con fundamentos topológicos robustos.
    Transforma la estructura del grafo en BOM usando un colapso de onda deterministico.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def materialize_project(
        self,
        graph: nx.DiGraph,
        risk_profile: Optional[Dict[str, Any]] = None,
        flux_metrics: Optional[Dict[str, Any]] = None
    ) -> BillOfMaterials:
        """
        Orquesta la transformación del Grafo en BOM con validaciones robustas.

        Args:
            graph: Grafo topológico (presumiblemente un DAG de composición)
            risk_profile: Perfil de riesgo para ajuste de factores
            flux_metrics: Métricas de estabilidad/fricción

        Returns:
            BillOfMaterials: La lista de compras consolidada con metadata completa

        Raises:
            ValueError: Si el grafo no es un DAG o contiene ciclos prohibidos
        """
        self.logger.info("🌌 Iniciando materialización del proyecto...")

        # Validación de la estructura del grafo (debe ser DAG para composición)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("El grafo debe ser un DAG para garantizar consistencia composicional")

        # 1. Colapso de Onda (Recorrido topológico del Grafo)
        raw_materials = self._explode_pyramid(graph)
        self.logger.info(f"🧱 Materiales brutos extraídos: {len(raw_materials)}")

        if not raw_materials:
            self.logger.warning("⚠️ No se encontraron materiales en el grafo")

        # 2. Aplicación de Entropía (Factores de Seguridad con monoides)
        adjusted_materials = self._apply_entropy_factors(
            raw_materials,
            flux_metrics,
            risk_profile
        )

        # 3. Clustering Semántico (Agrupación con kernel categórico)
        final_requirements = self._cluster_semantically(adjusted_materials)
        self.logger.info(f"🛒 Requerimientos consolidados: {len(final_requirements)}")

        # 4. Cálculo de Totales con validación numérica
        total_cost = self._compute_total_cost(final_requirements)

        # 5. Generación de metadata con invariantes topológicos
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
        Recorre el grafo desde raíces usando teoría de categorías para propagación.

        Implementa un funtor desde la categoría del grafo a la categoría de materiales.
        Cada camino único corresponde a una composición de monoides.
        """
        materials = []

        # Identificación de raíces: nodos con degree de entrada 0
        root_nodes = [node for node, in_degree in graph.in_degree() if in_degree == 0]

        if not root_nodes:
            self.logger.error("❌ Grafo no tiene raíces (nodos sin predecesores)")
            return materials

        self.logger.debug(f"Encontradas {len(root_nodes)} raíces: {root_nodes}")

        # Usamos DFS con stack para preservar el orden topológico inverso
        # Estructura: (nodo, cantidad_acumulada, camino, padre_apu)
        stack = [(root, 1.0, [], None) for root in root_nodes]

        # Track de ciclos (aunque debería ser DAG, verificamos defensivamente)
        visited_edges = set()
        steps = 0
        MAX_STEPS = len(graph.nodes()) * 10  # Límite basado en complejidad

        while stack:
            current_node, current_qty, current_path, parent_apu = stack.pop()
            steps += 1

            if steps > MAX_STEPS:
                raise RuntimeError("Posible ciclo detectado o grafo demasiado complejo")

            # Verificación de ciclo en el camino actual
            if current_node in current_path:
                self.logger.warning(f"Ciclo detectado en camino: {current_path + [current_node]}")
                continue

            node_data = graph.nodes[current_node]
            node_type = node_data.get("type", "UNDEFINED")
            new_path = current_path + [current_node]

            # Si es nodo terminal (INSUMO), registramos material
            if node_type == "INSUMO":
                # Validamos que tenemos los atributos necesarios
                description = node_data.get("description", "")
                if not description:
                    description = node_data.get("name", current_node)

                materials.append({
                    "id": current_node,
                    "description": description,
                    "base_qty": current_qty,
                    "unit_cost": float(node_data.get("unit_cost", 0.0)),
                    "source_apu": parent_apu or "ROOT",
                    "unit": node_data.get("unit", "UND"),
                    "node_data": node_data,  # Preservamos metadata completa
                    "composition_path": new_path  # Para debugging/trazabilidad
                })
                continue

            # Expandimos hijos con propagación de cantidades
            for successor in graph.successors(current_node):
                edge_key = (current_node, successor)

                # OJO: En un DAG, podemos visitar un nodo múltiples veces desde diferentes caminos.
                # No debemos bloquear edge_key globalmente, sino por camino.
                # PERO para _explode_pyramid queremos todas las instancias de material.
                # visited_edges aquí podría prevenir bucles infinitos en grafos con ciclos mal formados,
                # pero en un árbol de expansión queremos recorrer todo.
                # Sin embargo, si es un DAG, no necesitamos visited_edges global para evitar ciclos.
                # La validación de ciclo en camino (linea 94) es suficiente.

                edge_data = graph.edges[edge_key]
                edge_qty = float(edge_data.get("quantity", 1.0))

                # Validación: cantidad debe ser positiva
                if edge_qty <= 0:
                    self.logger.warning(f"Cantidad no positiva en arista {edge_key}: {edge_qty}")
                    edge_qty = 1.0

                # Si el nodo actual es APU, lo registramos como padre para materiales hijos
                next_parent = current_node if node_type == "APU" else parent_apu

                # Calculamos nueva cantidad (composición de morfismos)
                new_qty = current_qty * edge_qty

                # Verificación de desbordamiento numérico
                if new_qty > 1e10:  # Límite razonable
                    self.logger.warning(f"Cantidad muy grande en nodo {successor}: {new_qty}")

                stack.append((successor, new_qty, new_path, next_parent))

        return materials

    def _apply_entropy_factors(
        self,
        raw_materials: List[Dict[str, Any]],
        flux_metrics: Optional[Dict[str, Any]],
        risk_profile: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Aplica factores de entropía usando un monoide de transformación.

        El espacio de factores forma un grupo conmutativo bajo multiplicación,
        permitiendo composición jerárquica de factores de riesgo.
        """
        # Grupo de factores base con elemento identidad 1.0
        base_factor = 1.0  # Elemento neutro del grupo

        # Aplicamos factores de flux_metrics (si existen)
        if flux_metrics:
            # Factor de saturación (fricción del sistema)
            saturation = flux_metrics.get("avg_saturation", 0.0)
            if saturation > 0.8:
                base_factor *= 1.05  # +5% por saturación

            # Factor de estabilidad estructural
            stability = flux_metrics.get("pyramid_stability", 1.0)
            if stability < 1.0:
                base_factor *= 1.03  # +3% por inestabilidad

        # Aplicamos factores de riesgo (si existen)
        if risk_profile:
            risk_level = risk_profile.get("level", "MEDIUM")
            risk_multipliers = {
                "LOW": 1.01,    # +1%
                "MEDIUM": 1.03,  # +3%
                "HIGH": 1.07,    # +7%
                "CRITICAL": 1.15 # +15%
            }
            base_factor *= risk_multipliers.get(risk_level, 1.03)

        # Factor de material específico (podría basarse en tipo de material)
        processed_materials = []

        for mat in raw_materials:
            # Clonamos el material para no modificar el original
            processed = mat.copy()

            # Calculamos factor de desperdicio (waste_factor = base_factor - 1)
            # waste_factor es el exceso.

            # Aplicamos factor específico si el material tiene metadata
            material_type = mat.get("node_data", {}).get("material_category", "GENERIC")
            type_multipliers = {
                "FRAGILE": 1.02,
                "PERISHABLE": 1.04,
                "HAZARDOUS": 1.06
            }
            specific_factor = type_multipliers.get(material_type, 1.0)

            # Composición de factores: f_total = f_base * f_especifico
            total_multiplier = base_factor * specific_factor
            total_waste_factor = total_multiplier - 1.0

            processed["waste_factor"] = total_waste_factor
            processed["total_qty"] = mat["base_qty"] * total_multiplier

            # Preservamos los factores aplicados para trazabilidad
            processed["applied_factors"] = {
                "base": base_factor,
                "specific": specific_factor,
                "material_type": material_type
            }

            processed_materials.append(processed)

        return processed_materials

    def _cluster_semantically(self, materials: List[Dict[str, Any]]) -> List[MaterialRequirement]:
        """
        Agrupa materiales usando un kernel de equivalencia categórica.

        Define una relación de equivalencia basada en (id, unit) y
        realiza el colapso al cociente con operaciones de monoide aditivo.
        """
        # Diccionario de agrupación: clave -> datos agregados
        clustered = {}

        for mat in materials:
            # Clave canónica: tupla (id, unidad) para agrupación fuerte
            canonical_key = (mat["id"], mat.get("unit", "UND"))

            if canonical_key not in clustered:
                clustered[canonical_key] = {
                    "id": mat["id"],
                    "description": mat["description"],
                    "quantity_base": 0.0,
                    "quantity_total": 0.0,
                    "unit_cost": mat["unit_cost"],
                    "waste_factor": 0.0,  # Será recalculado
                    "source_apus": set(),
                    "unit": mat.get("unit", "UND"),
                    "unit_cost_samples": [],  # Para análisis estadístico
                    "composition_paths": []   # Para trazabilidad
                }

            data = clustered[canonical_key]

            # Acumulación de cantidades (monoide aditivo)
            data["quantity_base"] += mat["base_qty"]
            data["quantity_total"] += mat["total_qty"]

            # Acumulación de APUs de origen
            data["source_apus"].add(mat["source_apu"])

            # Muestras de costo unitario para análisis
            data["unit_cost_samples"].append(mat["unit_cost"])

            # Preservamos algunos paths para debugging
            if len(data["composition_paths"]) < 3:  # Limitamos para no sobrecargar
                data["composition_paths"].append(mat.get("composition_path", []))

        # Procesamiento post-agrupación
        requirements = []

        for (mat_id, unit), data in clustered.items():
            # Validación: cantidad base debe ser positiva
            if data["quantity_base"] <= 0:
                self.logger.warning(f"Material {mat_id} tiene cantidad base no positiva")
                continue

            # Recalculamos waste_factor a partir de cantidades agregadas
            # Esto es más preciso que promediar factores individuales
            total_waste = (data["quantity_total"] / data["quantity_base"]) - 1.0

            # Calculamos costo unitario representativo
            # Podríamos usar mediana para ser robustos a outliers
            cost_samples = data["unit_cost_samples"]
            if cost_samples:
                # Usamos la mediana para evitar influencia de outliers
                sorted_costs = sorted(cost_samples)
                mid = len(sorted_costs) // 2
                if len(sorted_costs) % 2 == 0:
                    representative_cost = (sorted_costs[mid-1] + sorted_costs[mid]) / 2
                else:
                    representative_cost = sorted_costs[mid]
            else:
                representative_cost = data["unit_cost"]

            # Validamos el costo unitario
            if representative_cost <= 0:
                self.logger.warning(f"Material {mat_id} tiene costo unitario no positivo")
                representative_cost = 0.0

            # Calculamos costo total
            total_cost = data["quantity_total"] * representative_cost

            # Creamos el requerimiento
            req = MaterialRequirement(
                id=data["id"],
                description=data["description"],
                quantity_base=data["quantity_base"],
                unit=data["unit"],
                waste_factor=total_waste,
                quantity_total=data["quantity_total"],
                unit_cost=representative_cost,
                total_cost=total_cost,
                source_apus=list(data["source_apus"])
            )

            requirements.append(req)

        # Ordenamiento por criterio de Pareto (costo total descendente)
        requirements.sort(key=lambda x: (x.total_cost, -len(x.source_apus)), reverse=True)

        return requirements

    def _compute_total_cost(self, requirements: List[MaterialRequirement]) -> float:
        """
        Calcula el costo total con validación numérica robusta.

        Implementa suma en doble precisión con verificación de overflow.
        """
        total = 0.0

        for req in requirements:
            # Validación: costo debe ser finito
            if not math.isfinite(req.total_cost):
                self.logger.error(f"Costo no finito en material {req.id}: {req.total_cost}")
                continue

            # Suma con verificación de overflow
            new_total = total + req.total_cost
            if math.isinf(new_total):
                raise OverflowError("Overflow en cálculo de costo total")

            total = new_total

        # Redondeo a 2 decimales para representación monetaria
        return round(total, 2)

    def _generate_metadata(
        self,
        graph: nx.DiGraph,
        risk_profile: Optional[Dict[str, Any]],
        flux_metrics: Optional[Dict[str, Any]],
        requirements: List[MaterialRequirement]
    ) -> Dict[str, Any]:
        """
        Genera metadata con invariantes topológicos y métricas de calidad.
        """
        # Calcular invariantes topológicos del grafo
        try:
            is_dag = nx.is_directed_acyclic_graph(graph)
            longest_path = nx.dag_longest_path_length(graph) if is_dag else None
        except Exception:
            is_dag = False
            longest_path = None

        # Métricas de distribución de costos
        if requirements:
            costs = [r.total_cost for r in requirements]
            total_cost = sum(costs)
            if total_cost > 0:
                pareto_80_idx = int(len(costs) * 0.2)  # Índice para 80% del costo
                sorted_costs = sorted(costs, reverse=True)
                pareto_cost = sum(sorted_costs[:pareto_80_idx])
                pareto_percentage = (pareto_cost / total_cost) * 100
            else:
                pareto_percentage = 0.0
        else:
            pareto_percentage = 0.0

        metadata = {
            "graph_metrics": {
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "is_dag": is_dag,
                "longest_path": longest_path,
                "root_count": len([n for n, d in graph.in_degree() if d == 0])
            },
            "cost_distribution": {
                "item_count": len(requirements),
                "pareto_80_percentage": round(pareto_percentage, 2),
                "unique_apus": len(set().union(*[r.source_apus for r in requirements]))
            },
            "risk_profile": risk_profile,
            "flux_metrics": flux_metrics,
            "generation_timestamp": datetime.now().isoformat(),
            "version": "2.0-topology"
        }

        return metadata
