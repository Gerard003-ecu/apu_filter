import logging
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.constants import ColumnNames

logger = logging.getLogger(__name__)

@dataclass
class MaterialRequirement:
    """
    Representa un requerimiento de material individual derivado del grafo.
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
    Lista de Materiales (BOM) final lista para compras.
    """
    requirements: List[MaterialRequirement]
    total_material_cost: float
    generation_date: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

class MatterGenerator:
    """
    Motor de Materialización (Wave Collapse Engine).
    Transforma la topología del proyecto en una realidad determinista (BOM).
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
        Orquesta la transformación del Grafo en BOM.

        Args:
            graph: El grafo topológico del proyecto.
            risk_profile: Perfil de riesgo (opcional) para ajuste de factores.
            flux_metrics: Métricas de fricción/estabilidad (opcional).

        Returns:
            BillOfMaterials: La lista de compras consolidada.
        """
        self.logger.info("🌌 Iniciando materialización del proyecto...")

        # 1. Colapso de Onda (Recorrido del Grafo)
        raw_materials = self._explode_pyramid(graph)
        self.logger.info(f"🧱 Materiales brutos extraídos: {len(raw_materials)}")

        # 2. Aplicación de Entropía (Factores de Seguridad)
        adjusted_materials = self._apply_entropy_factors(raw_materials, flux_metrics)

        # 3. Clustering Semántico (Agrupación)
        final_requirements = self._cluster_semantically(adjusted_materials)
        self.logger.info(f"🛒 Requerimientos consolidados: {len(final_requirements)}")

        # 4. Cálculo de Totales
        total_cost = sum(req.total_cost for req in final_requirements)

        return BillOfMaterials(
            requirements=final_requirements,
            total_material_cost=total_cost,
            metadata={
                "risk_profile": risk_profile,
                "flux_metrics": flux_metrics,
                "node_count": graph.number_of_nodes()
            }
        )

    def _explode_pyramid(self, graph: nx.DiGraph) -> List[Dict[str, Any]]:
        """
        Recorre el grafo desde la raíz calculando cantidades en cascada.
        "Aplana" la pirámide.
        """
        materials = []
        root_nodes = [n for n, d in graph.nodes(data=True) if d.get("level") == 0]

        if not root_nodes:
            self.logger.warning("⚠️ No se encontró nodo raíz (Level 0). Buscando APUs directos.")
            # Fallback: Iterar sobre APUs si no hay raíz definida
            # Pero la arquitectura define PROYECTO_TOTAL. Asumiremos que si no está, el grafo está mal formado o es parcial.
            # Intentemos buscar nodos sin predecesores (raíces implícitas)
            root_nodes = [n for n, d in graph.in_degree() if d == 0]

        # Usaremos un recorrido BFS o DFS modificado para llevar la cuenta de la cantidad multiplicada
        # Stack: (node_id, cumulative_quantity, parent_id)
        # parent_id es None para la raíz
        stack = [(root, 1.0, None) for root in root_nodes]

        steps = 0
        MAX_STEPS = 100000 # Safety brake

        while stack:
            current_node, current_qty, parent_node = stack.pop()
            steps += 1
            if steps > MAX_STEPS:
                self.logger.error("🔥 Max steps reached in explode_pyramid. Possible cycle?")
                break

            # Si es un nodo de INSUMO (Hoja o Nivel 3), lo registramos
            node_data = graph.nodes[current_node]
            if node_data.get("type") == "INSUMO":
                materials.append({
                    "id": current_node,
                    "description": node_data.get("description", current_node),
                    "base_qty": current_qty,
                    "unit_cost": node_data.get("unit_cost", 0.0),
                    # Guardamos el padre inmediato como origen (APU)
                    "source_apu": parent_node if parent_node else "UNKNOWN",
                    # Add unit info if available
                    "unit": node_data.get("unit", "UND")
                })
                continue # Los insumos son hojas en este contexto de materialización

            # Expandir hijos
            # networkx graph[u][v] contiene los atributos de la arista u->v
            for neighbor in graph.successors(current_node):
                edge_data = graph[current_node][neighbor]
                edge_qty = edge_data.get("quantity", 1.0)

                # Calcular nueva cantidad acumulada
                new_qty = current_qty * edge_qty

                stack.append((neighbor, new_qty, current_node))

        return materials

    def _apply_entropy_factors(
        self,
        raw_materials: List[Dict[str, Any]],
        flux_metrics: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Aplica factores de desperdicio basados en métricas de flujo y estabilidad.
        """
        # Valores por defecto
        base_waste = 0.0 # 0% por defecto

        if flux_metrics:
            # Si hay alta saturación o fricción, aumentamos desperdicio
            saturation = flux_metrics.get("avg_saturation", 0.0)
            if saturation > 0.8:
                base_waste += 0.05 # +5% si el sistema estaba saturado (datos sucios/complejos)

            # Si la estabilidad es baja (pirámide invertida), asumimos ineficiencia en compras
            stability = flux_metrics.get("pyramid_stability", 10.0)
            if stability < 1.0:
                base_waste += 0.03 # +3% por riesgo estructural

        processed_materials = []
        for mat in raw_materials:
            # Podríamos tener lógica específica por tipo de material aquí
            # Por ahora aplicamos el factor general
            waste_factor = base_waste

            mat["waste_factor"] = waste_factor
            mat["total_qty"] = mat["base_qty"] * (1 + waste_factor)
            processed_materials.append(mat)

        return processed_materials

    def _cluster_semantically(self, materials: List[Dict[str, Any]]) -> List[MaterialRequirement]:
        """
        Agrupa materiales por ID/Descripción sumando cantidades.
        """
        clustered = {}

        for mat in materials:
            # Clave de agrupación: Descripción normalizada o ID
            # Aquí asumimos que el ID (nombre del nodo insumo) es único y representativo
            key = mat["id"]

            if key not in clustered:
                clustered[key] = {
                    "id": key,
                    "description": mat["description"],
                    "quantity_base": 0.0,
                    "quantity_total": 0.0,
                    "unit_cost": mat["unit_cost"],
                    "waste_factor": mat["waste_factor"], # Usamos el último (deberían ser iguales si es global)
                    "source_apus": set(),
                    "unit": mat.get("unit", "UND")
                }

            clustered[key]["quantity_base"] += mat["base_qty"]
            clustered[key]["quantity_total"] += mat["total_qty"]
            clustered[key]["source_apus"].add(mat["source_apu"])

            # Promedio ponderado del costo unitario si varía?
            # Por simplicidad mantenemos el costo unitario del último (deberían ser consistentes)

        # Convertir a objetos MaterialRequirement
        requirements = []
        for key, data in clustered.items():
            req = MaterialRequirement(
                id=data["id"],
                description=data["description"],
                quantity_base=data["quantity_base"],
                unit=data["unit"],
                waste_factor=data["waste_factor"],
                quantity_total=data["quantity_total"],
                unit_cost=data["unit_cost"],
                total_cost=data["quantity_total"] * data["unit_cost"],
                source_apus=list(data["source_apus"])
            )
            requirements.append(req)

        # Ordenar por costo total descendente (Pareto)
        requirements.sort(key=lambda x: x.total_cost, reverse=True)

        return requirements
