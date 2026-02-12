#!/usr/bin/env bash

# ==============================================================================
# Script de Detención Segura para APU Filter Ecosystem
# Versión: 2.3.0 (Mejora de Seguridad y Robustez)
# ==============================================================================

# --- Strict Mode & Signal Handling ---
set -euo pipefail
IFS=$'\n\t'

# Capturar señales para limpieza segura
trap 'cleanup_on_exit' EXIT INT TERM

# --- Configuration ---
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Detect if running from root or scripts/ subdirectory
if [[ -f "$SCRIPT_DIR/compose.yaml" ]]; then
    readonly PROJECT_ROOT="$SCRIPT_DIR"
else
    readonly PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
readonly LOG_DIR="${PROJECT_ROOT}/logs"
readonly COMPOSE_FILE="${PROJECT_ROOT}/compose.yaml"
readonly SCRIPT_PID=$$
readonly TIMESTAMP=$(date +%Y%m%d_%H%M%S)
readonly LOG_FILE="${LOG_DIR}/podman_stop_${TIMESTAMP}.log"

# --- Colors ---
declare -A COLORS=(
    [RESET]='\033[0m'
    [RED]='\033[0;31m'
    [GREEN]='\033[0;32m'
    [YELLOW]='\033[0;33m'
    [BLUE]='\033[0;34m'
    [PURPLE]='\033[0;35m'
    [CYAN]='\033[0;36m'
)

# --- Dependencies Check ---
check_dependencies() {
    local deps=("podman" "podman-compose")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_error "Dependencia requerida no encontrada: $dep"
            exit 1
        fi
    done
}

# --- Logging ---
_log() {
    local level="$1"
    local color="$2"
    local message="$3"
    
    # Consola
    printf "${color}[%-7s]${COLORS[RESET]} [%s] %s\n" "$level" "$(date '+%H:%M:%S')" "$message"
    
    # Archivo
    if [[ -d "$LOG_DIR" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] PID:${SCRIPT_PID} $message" >> "$LOG_FILE"
    fi
}

log_info()    { _log "INFO"    "${COLORS[BLUE]}"   "$1"; }
log_success() { _log "SUCCESS" "${COLORS[GREEN]}"  "$1"; }
log_warn()    { _log "WARN"    "${COLORS[YELLOW]}" "$1"; }
log_error()   { _log "ERROR"   "${COLORS[RED]}"    "$1" >&2; }

# --- Cleanup ---
cleanup_on_exit() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Script terminado con código de error: $exit_code"
        log_info "Revisar log: $LOG_FILE"
    else
        log_info "Script de detención finalizado exitosamente"
    fi
}

# --- Safe Stop Operations ---
perform_safe_stop() {
    local graceful_timeout="${STOP_TIMEOUT:-30}"
    local force_kill="${FORCE_KILL_ON_STOP:-false}"
    
    log_info "Iniciando detención de servicios (timeout: ${graceful_timeout}s)..."
    
    # Intentar detención suave primero
    if podman-compose -f "$COMPOSE_FILE" stop -t "$graceful_timeout" > /dev/null 2>&1; then
        log_success "Servicios detenidos correctamente."
        return 0
    fi
    
    # Si la detención suave falla, verificar si se permite fuerza bruta
    if [[ "$force_kill" == "true" ]]; then
        log_warn "Detención suave fallida. Intentando kill forzoso..."
        if podman-compose -f "$COMPOSE_FILE" kill > /dev/null 2>&1; then
            log_success "Servicios detenidos con kill forzoso."
            return 0
        else
            log_error "No se pudieron detener los servicios ni con kill forzoso."
            return 1
        fi
    else
        log_warn "Detención suave fallida. Para forzar detención, establece FORCE_KILL_ON_STOP=true"
        log_error "No se pudieron detener los servicios de forma segura."
        return 1
    fi
}

# --- Resource Cleanup ---
cleanup_resources() {
    local cleanup_volumes="${CLEANUP_VOLUMES:-false}"
    local cleanup_networks="${CLEANUP_NETWORKS:-false}"
    
    log_info "Realizando limpieza de recursos..."
    
    # Detener contenedores huérfanos
    podman-compose -f "$COMPOSE_FILE" down --remove-orphans > /dev/null 2>&1 || true
    
    # Opcionalmente limpiar volúmenes
    if [[ "$cleanup_volumes" == "true" ]]; then
        log_info "Eliminando volúmenes definidos en compose..."
        podman-compose -f "$COMPOSE_FILE" down --remove-orphans --volumes > /dev/null 2>&1 || true
        log_success "Volúmenes eliminados según configuración."
    fi
    
    # Opcionalmente limpiar redes
    if [[ "$cleanup_networks" == "true" ]]; then
        log_info "Eliminando redes definidas en compose..."
        podman-compose -f "$COMPOSE_FILE" down --remove-orphans --rmi local > /dev/null 2>&1 || true
        log_success "Redes y imágenes locales eliminadas según configuración."
    fi
    
    log_success "Limpieza de recursos completada."
}

# --- Status Verification ---
verify_stopped() {
    log_info "Verificando estado de servicios..."
    
    local running_containers
    running_containers=$(podman-compose -f "$COMPOSE_FILE" ps -q 2>/dev/null | wc -l)
    
    if [[ "$running_containers" -eq 0 ]]; then
        log_success "✅ Todos los servicios han sido detenidos correctamente."
        return 0
    else
        local container_list
        container_list=$(podman-compose -f "$COMPOSE_FILE" ps --filter status=running 2>/dev/null | head -n -1 | tail -n +2)
        log_warn "⚠️  Algunos contenedores aún están activos:"
        echo "$container_list" | while read -r line; do
            [[ -n "$line" ]] && log_warn "   $line"
        done
        return 1
    fi
}

# --- Main Logic ---
main() {
    # 1. Verificar dependencias
    check_dependencies
    log_info "Dependencias verificadas correctamente"
    
    # 2. Crear directorios y archivo de log
    mkdir -p "$LOG_DIR"
    touch "$LOG_FILE"
    
    log_info "=== Iniciando Detención Segura de APU Filter Ecosystem ==="
    log_info "Log file: $LOG_FILE"
    log_info "Timestamp: $TIMESTAMP"
    
    # 3. Validar archivo compose
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "No se encuentra el archivo: $COMPOSE_FILE"
        log_error "Asegúrate de ejecutar este script desde la raíz del proyecto."
        exit 1
    fi

    # 4. Realizar detención segura
    if ! perform_safe_stop; then
        log_error "Fallo en la detención de servicios"
        exit 1
    fi

    # 5. Verificar estado post-detención
    verify_stopped
    
    # 6. Opcionalmente limpiar recursos adicionales
    cleanup_resources
    
    # 7. Mostrar estado final
    log_info "Estado final de contenedores:"
    podman-compose -f "$COMPOSE_FILE" ps || true
    
    log_success "=== APU Filter Ecosystem Detenido Correctamente ==="
    log_info "Log completo disponible en: $LOG_FILE"
}

# Ejecutar main si no estamos siendo sourceados
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi