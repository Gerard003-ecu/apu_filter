#!/bin/bash
# ==============================================================================
# Script de Limpieza Profunda (Nuke)
# ==============================================================================

set -euo pipefail

LOG_DIR="./logs"
COMPOSE_FILE="infrastructure/compose.yaml"
COLOR_RED='\033[0;31m'
COLOR_GREEN='\033[0;32m'
COLOR_RESET='\033[0m'

main() {
    echo -e "${COLOR_RED}⚠️  ADVERTENCIA: Esto eliminará contenedores, redes y volúmenes (datos de Redis).${COLOR_RESET}"
    read -p "¿Continuar? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then exit 1; fi

    echo "Limpiando..."
    podman-compose -f "$COMPOSE_FILE" down -v --remove-orphans
    
    # Limpiar logs viejos
    rm -f "$LOG_DIR"/*.log
    
    echo -e "${COLOR_GREEN}=== Sistema Limpiado ===${COLOR_RESET}"
}

main