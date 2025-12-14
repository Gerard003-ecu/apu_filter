import logging
import sys

from agent.topological_analyzer import SystemTopology

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verification")


def verify_visualization():
    logger.info("Iniciando verificación de visualización topológica...")

    # Instanciar topología
    topology = SystemTopology()

    # Agregar algunos nodos y conexiones para simular estado parcial
    # (Agent -> Core existe, pero faltan las otras conexiones del agente)
    topology.add_node("Agent")
    topology.add_node("Core")
    topology.add_node("Redis")
    topology.add_node("Filesystem")

    # Conexiones existentes (parciales)
    topology.update_connectivity(
        [("Agent", "Core"), ("Core", "Redis"), ("Core", "Filesystem")]
    )

    output_path = "data/topology_verification.png"

    # Generar visualización
    success = topology.visualize_topology(output_path)

    if success:
        logger.info(f"¡Éxito! Visualización generada en {output_path}")
        # Verificar que el archivo existe
        import os

        if os.path.exists(output_path):
            logger.info("Archivo confirmado en disco.")
        else:
            logger.error("El archivo reportado como generado no se encuentra.")
            sys.exit(1)
    else:
        logger.error("Fallo al generar la visualización.")
        sys.exit(1)


if __name__ == "__main__":
    verify_visualization()
