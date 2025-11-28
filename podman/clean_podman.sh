#!/bin/bash
# ==============================================================================
# Script de Limpieza Profunda (Nuke) para APU Filter
# ==============================================================================

set -euo pipefail

LOG_DIR="./logs"
COMPOSE_FILE="infrastructure/compose.yaml"

# Colors
COLOR_RESET='\033[0m'
COLOR_RED='\033[0;31m'
COLOR_GREEN='\033[0;32m'

log_alert() { echo -e "${COLOR_RED}[ALERT]${COLOR_RESET} $1"; }
log_success() { echo -e "${COLOR_GREEN}[CLEAN]${COLOR_RESET} $1"; }

main() {
    echo ""
    log_alert "ESTO ELIMINARÁ TODOS LOS CONTENEDORES, REDES Y VOLÚMENES DE APU FILTER."
    log_alert "Los datos persistentes en Redis se perderán."
    echo ""
    read -p "⚠️  ¿Estás seguro de continuar? (y/n): " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Operación cancelada."
        exit 1
    fi

    log_alert "Iniciando limpieza profunda..."

    # Bajar todo y borrar volúmenes
    podman-compose -f "$COMPOSE_FILE" down -v --remove-orphans

    # Limpiar logs antiguos
    if [ -d "$LOG_DIR" ]; then
        rm -f "$LOG_DIR"/*.log
        log_success "Logs antiguos eliminados."
    fi

    log_success "=== Sistema APU Filter limpiado y reseteado ==="
}

main