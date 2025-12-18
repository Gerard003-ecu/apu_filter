"""
Módulo de visualización de topología para la aplicación Flask.
Este módulo extiende app.py añadiendo endpoints para Cytoscape.js.
"""

from flask import Blueprint, jsonify, session, current_app
import networkx as nx
import logging

from agent.business_topology import BudgetGraphBuilder, BusinessTopologicalAnalyzer

topology_bp = Blueprint('topology', __name__)
logger = logging.getLogger(__name__)

@topology_bp.route('/api/visualization/topology', methods=['GET'])
def get_topology_data():
    """
    Endpoint para obtener la estructura del grafo para Cytoscape.js.
    """
    if "processed_data" not in session:
        return jsonify({"error": "No hay datos de sesión", "elements": []}), 404

    try:
        data = session["processed_data"]

        # Reconstruir el grafo desde los datos en sesión
        builder = BudgetGraphBuilder()

        # Convertir listas de dicts a DataFrames
        import pandas as pd
        df_presupuesto = pd.DataFrame(data.get("presupuesto", []))
        df_apus_detail = pd.DataFrame(data.get("apus_detail", []))

        graph = builder.build(df_presupuesto, df_apus_detail)

        # Analizar para colorear nodos usando método público
        analyzer = BusinessTopologicalAnalyzer()
        # Usamos analyze_structural_integrity que devuelve los detalles necesarios
        analysis_result = analyzer.analyze_structural_integrity(graph)

        details = analysis_result.get("details", {})
        anomalies = details.get("anomalies", {
            "isolated_nodes": [],
            "orphan_insumos": [],
            "empty_apus": []
        })
        cycles_list = details.get("cycles", {}).get("list", [])

        # Extraer nodos en ciclos para marcarlos
        nodes_in_cycles = set()
        for cycle_str in cycles_list:
            parts = cycle_str.split(" → ")
            # El último es repetido, lo ignoramos para el set
            nodes_in_cycles.update(parts[:-1])

        isolated_ids = {n["id"] for n in anomalies["isolated_nodes"]}
        orphan_ids = {n["id"] for n in anomalies["orphan_insumos"]}
        empty_ids = {n["id"] for n in anomalies["empty_apus"]}

        elements = []

        # Nodos
        for node, attrs in graph.nodes(data=True):
            node_type = attrs.get("type", "UNKNOWN")

            # Determinar clase/color
            classes = [node_type]

            if node in nodes_in_cycles:
                classes.append("cycle")
            elif node in isolated_ids or node in orphan_ids:
                classes.append("isolated")
            elif node in empty_ids:
                classes.append("empty")
            else:
                classes.append("normal")

            # Etiqueta
            label = str(node)
            if attrs.get("description"):
                desc = str(attrs.get("description"))
                label = f"{node}\n{desc[:20]}..." if len(desc) > 20 else f"{node}\n{desc}"

            elements.append({
                "data": {
                    "id": str(node),
                    "label": label,
                    "type": node_type,
                    "level": attrs.get("level", 0),
                    "cost": attrs.get("total_cost", 0) or attrs.get("unit_cost", 0)
                },
                "classes": " ".join(classes)
            })

        # Aristas
        for u, v, attrs in graph.edges(data=True):
            elements.append({
                "data": {
                    "source": str(u),
                    "target": str(v),
                    "cost": attrs.get("total_cost", 0)
                }
            })

        return jsonify(elements)

    except Exception as e:
        logger.error(f"Error generando visualización: {e}", exc_info=True)
        return jsonify({"error": str(e), "elements": []}), 500
