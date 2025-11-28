#!/bin/bash
# ==============================================================================
# Script de Orquestación para APU Filter Ecosystem con podman-compose
# ==============================================================================

# --- Strict Mode & Error Handling ---
set -euo pipefail

# --- Configuration ---
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/podman_start_$(date +%Y%m%d_%H%M%S).log"
COMPOSE_FILE="infrastructure/compose.yaml"

# --- Colors for Logging ---
COLOR_RESET='\033[0m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[0;33m'
COLOR_RED='\033[0;31m'
COLOR_BLUE='\033[0;34m'

# --- Logging Functions ---
log_info() { echo -e "${COLOR_BLUE}[INFO]${COLOR_RESET} $1" | tee -a "$LOG_FILE"; }
log_success() { echo -e "${COLOR_GREEN}[SUCCESS]${COLOR_RESET} $1" | tee -a "$LOG_FILE"; }
log_warn() { echo -e "${COLOR_YELLOW}[WARN]${COLOR_RESET} $1" | tee -a "$LOG_FILE"; }
log_error() { echo -e "${COLOR_RED}[ERROR]${COLOR_RESET} $1" | tee -a "$LOG_FILE"; }

# --- Utility Functions ---
setup_logging() {
    mkdir -p "$LOG_DIR"
    >"$LOG_FILE"
    log_info "Logging to ${LOG_FILE}"
}

# --- Main Execution ---
main() {
    setup_logging
    log_info "=== Iniciando Despliegue de APU Filter Ecosystem ==="

    # 1. Limpieza previa
    log_info "Deteniendo contenedores huérfanos o previos..."
    podman-compose -f "$COMPOSE_FILE" down --remove-orphans >> "$LOG_FILE" 2>&1 || log_warn "Limpieza previa no necesaria o fallida. Continuando..."

    # 2. Construcción de imágenes
    log_info "Construyendo imágenes de servicios (Core & Agent)..."
    if ! podman-compose -f "$COMPOSE_FILE" build >> "$LOG_FILE" 2>&1; then
        log_error "Fallo en la construcción. Revisa: ${LOG_FILE}"
        exit 1
    fi
    log_success "Imágenes construidas correctamente."

    # 3. Iniciar servicios
    log_info "Levantando servicios en modo detached..."
    if ! podman-compose -f "$COMPOSE_FILE" up -d >> "$LOG_FILE" 2>&1; then
        log_error "Fallo al iniciar servicios. Revisa: ${LOG_FILE}"
        podman-compose -f "$COMPOSE_FILE" logs >> "$LOG_FILE" 2>&1
        exit 1
    fi
    log_success "Servicios iniciados."

    # 4. Verificación de Salud
    log_info "Esperando estabilización (10s)..."
    sleep 10

    log_info "Estado final de los contenedores:"
    podman-compose -f "$COMPOSE_FILE" ps | tee -a "$LOG_FILE"

    log_success "=== APU Filter Ecosystem Operativo ==="
}

main