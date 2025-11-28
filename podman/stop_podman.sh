#!/bin/bash
# ==============================================================================
# Script de Detención para APU Filter Ecosystem
# ==============================================================================

set -euo pipefail

LOG_DIR="./logs"
COMPOSE_FILE="infrastructure/compose.yaml"

# Colors
COLOR_RESET='\033[0m'
COLOR_YELLOW='\033[0;33m'
COLOR_BLUE='\033[0;34m'

log_info() { echo -e "${COLOR_BLUE}[INFO]${COLOR_RESET} $1"; }
log_warn() { echo -e "${COLOR_YELLOW}[WARN]${COLOR_RESET} $1"; }

main() {
    mkdir -p "$LOG_DIR"
    log_info "=== Deteniendo APU Filter Ecosystem ==="

    if ! podman-compose -f "$COMPOSE_FILE" stop; then
        log_warn "Hubo un problema al detener los servicios, intentando forzar..."
        podman-compose -f "$COMPOSE_FILE" kill
    fi

    log_info "🛑 APU Filter Ecosystem detenido."
}

main