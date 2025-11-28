#!/bin/bash
# ==============================================================================
# Script de Orquestación para APU Filter Ecosystem
# Versión: 2.0.0 (Adapted)
# ==============================================================================

# --- Strict Mode ---
set -euo pipefail
IFS=$'\n\t'

# --- Configuration ---
# Rutas relativas desde la raíz del proyecto
readonly LOG_DIR="./logs"
readonly COMPOSE_FILE="compose.yaml"
readonly SCRIPT_PID=$$
readonly LOG_FILE="${LOG_DIR}/podman_start_$(date +%Y%m%d_%H%M%S).log"

# --- Colors ---
declare -A COLORS=(
    [RESET]='\033[0m'
    [RED]='\033[0;31m'
    [GREEN]='\033[0;32m'
    [YELLOW]='\033[0;33m'
    [BLUE]='\033[0;34m'
)

# --- Logging ---
_log() {
    local level="$1"
    local color="$2"
    local message="$3"
    # Consola
    printf "${color}[%-7s]${COLORS[RESET]} %s\n" "$level" "$message"
    # Archivo
    if [[ -d "$LOG_DIR" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $message" >> "$LOG_FILE"
    fi
}

log_info()    { _log "INFO"    "${COLORS[BLUE]}"   "$1"; }
log_success() { _log "SUCCESS" "${COLORS[GREEN]}"  "$1"; }
log_warn()    { _log "WARN"    "${COLORS[YELLOW]}" "$1"; }
log_error()   { _log "ERROR"   "${COLORS[RED]}"    "$1" >&2; }

# --- Main Logic ---
main() {
    mkdir -p "$LOG_DIR"
    touch "$LOG_FILE"
    
    log_info "=== Iniciando Despliegue de APU Filter Ecosystem ==="
    log_info "Log file: $LOG_FILE"

    # 1. Validar archivo compose
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "No se encuentra el archivo: $COMPOSE_FILE"
        log_error "Asegúrate de ejecutar este script desde la raíz del proyecto."
        exit 1
    fi

    # 2. Limpieza previa
    log_info "Limpiando contenedores previos..."
    podman-compose -f "$COMPOSE_FILE" down --remove-orphans >> "$LOG_FILE" 2>&1 || true

    # 3. Construcción
    log_info "Construyendo imágenes (Core & Agent)..."
    if ! podman-compose -f "$COMPOSE_FILE" build >> "$LOG_FILE" 2>&1; then
        log_error "Fallo en la construcción. Revisa el log."
        exit 1
    fi
    log_success "Imágenes construidas."

    # 4. Inicio
    log_info "Levantando servicios..."
    if ! podman-compose -f "$COMPOSE_FILE" up -d >> "$LOG_FILE" 2>&1; then
        log_error "Fallo al iniciar servicios."
        exit 1
    fi

    # 5. Verificación de Salud
    log_info "Esperando estabilización (10s)..."
    sleep 10

    log_info "Estado de los contenedores:"
    podman-compose -f "$COMPOSE_FILE" ps
    
    log_success "=== APU Filter Ecosystem Operativo ==="
}

# Ejecutar main sin argumentos para evitar errores de parsing
main