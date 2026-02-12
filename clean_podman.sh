#!/usr/bin/env bash

# ==============================================================================
# Script de Limpieza Profunda Segura (Nuke)
# Versión: 2.3.0 (Mejora de Seguridad y Confirmación Robusta)
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
readonly LOG_FILE="${LOG_DIR}/podman_clean_${TIMESTAMP}.log"

# Configuración sensible
readonly CONFIRMATION_TIMEOUT=30  # segundos
readonly PRESERVE_RECENT_LOGS_HOURS=24  # horas

# --- Colors ---
declare -A COLORS=(
    [RESET]='\033[0m'
    [RED]='\033[0;31m'
    [GREEN]='\033[0;32m'
    [YELLOW]='\033[0;33m'
    [BLUE]='\033[0;34m'
    [PURPLE]='\033[0;35m'
    [CYAN]='\033[0;36m'
    [BOLD]='\033[1m'
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
    printf "${color}${COLORS[BOLD]}[%-7s]${COLORS[RESET]} [%s] %s\n" "$level" "$(date '+%H:%M:%S')" "$message"
    
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
        log_info "Script de limpieza finalizado exitosamente"
    fi
}

# --- Confirmation with Timeout ---
get_user_confirmation() {
    local timeout="${CLEAN_CONFIRMATION_TIMEOUT:-$CONFIRMATION_TIMEOUT}"
    local response=""
    
    log_warn "⚠️  ADVERTENCIA CRÍTICA: Esta operación ELIMINARÁ permanentemente:"
    log_warn "   • Contenedores activos e inactivos"
    log_warn "   • Volúmenes (pueden contener datos de Redis u otros servicios)"
    log_warn "   • Redes creadas por compose"
    log_warn "   • Imágenes locales no utilizadas (opcional)"
    log_warn ""
    log_warn "Tiempo restante para confirmación: ${timeout}s"
    log_warn ""
    
    # Usar select para confirmación con timeout
    local start_time=$(date +%s)
    local end_time=$((start_time + timeout))
    
    while [[ $(date +%s) -lt $end_time ]]; do
        local remaining=$((end_time - $(date +%s)))
        printf "${COLORS[YELLOW]}¿Deseas continuar con la limpieza? (y/N) [tiempo: %ds]: ${COLORS[RESET]}" "$remaining"
        
        # Leer respuesta con timeout
        if read -t 1 -n 1 -r response; then
            case $response in
                [Yy]) 
                    echo
                    log_success "Confirmación recibida: Sí"
                    return 0
                    ;;
                [Nn]|"")
                    echo
                    log_info "Confirmación recibida: No o vacío"
                    return 1
                    ;;
                *)
                    echo
                    log_warn "Respuesta inválida: '$response'. Por favor ingresa 'y' o 'n'."
                    ;;
            esac
        fi
    done
    
    echo
    log_error "Timeout alcanzado. Operación cancelada por seguridad."
    return 1
}

# --- System State Snapshot ---
take_system_snapshot() {
    log_info "Tomando snapshot del sistema antes de limpieza..."
    
    local snapshot_file="${LOG_DIR}/system_snapshot_before_clean_${TIMESTAMP}.txt"
    
    {
        echo "Snapshot tomado: $(date)"
        echo "Contenedores activos:"
        podman ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "N/A"
        echo ""
        echo "Volúmenes asociados:"
        podman volume ls --format "table {{.Name}}\t{{.Driver}}" 2>/dev/null || echo "N/A"
        echo ""
        echo "Redes activas:"
        podman network ls --format "table {{.Name}}\t{{.Driver}}" 2>/dev/null || echo "N/A"
        echo ""
        echo "Imágenes locales:"
        podman images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}" 2>/dev/null || echo "N/A"
    } > "$snapshot_file"
    
    log_success "Snapshot guardado en: $snapshot_file"
}

# --- Clean Operations ---
perform_compose_cleanup() {
    log_info "Ejecutando limpieza de compose..."
    
    if [[ -f "$COMPOSE_FILE" ]]; then
        log_info "Deteniendo y removiendo recursos de compose..."
        if podman-compose -f "$COMPOSE_FILE" down -v --remove-orphans > /dev/null 2>&1; then
            log_success "Recursos de compose limpiados exitosamente."
        else
            log_warn "Algunos recursos de compose podrían no haberse eliminado completamente."
        fi
    else
        log_warn "Archivo compose no encontrado: $COMPOSE_FILE. Saltando limpieza de compose."
    fi
}

perform_system_cleanup() {
    local cleanup_images="${CLEAN_UNUSED_IMAGES:-false}"
    local cleanup_all_containers="${CLEAN_ALL_CONTAINERS:-false}"
    
    # Limpiar contenedores huérfanos
    if [[ "$cleanup_all_containers" == "true" ]]; then
        log_info "Removiendo todos los contenedores detenidos..."
        podman container prune -f > /dev/null 2>&1 || true
        log_success "Contenedores huérfanos eliminados."
    fi
    
    # Opcionalmente limpiar imágenes no utilizadas
    if [[ "$cleanup_images" == "true" ]]; then
        log_info "Removiendo imágenes no utilizadas..."
        podman image prune -f > /dev/null 2>&1 || true
        log_success "Imágenes no utilizadas eliminadas."
    fi
}

perform_logs_cleanup() {
    local preserve_recent_logs="${PRESERVE_RECENT_LOGS:-true}"
    local hours_to_preserve="${PRESERVE_RECENT_LOGS_HOURS:-24}"
    
    log_info "Realizando limpieza de logs..."
    
    if [[ "$preserve_recent_logs" == "true" ]]; then
        log_info "Preservando logs recientes (${hours_to_preserve}h)..."
        find "$LOG_DIR" -name "*.log" -type f -mmin +$((hours_to_preserve * 60)) -delete 2>/dev/null || true
        log_success "Logs antiguos eliminados, preservados recientes."
    else
        log_warn "Eliminando TODOS los logs en $LOG_DIR..."
        find "$LOG_DIR" -name "*.log" -type f -not -path "$LOG_FILE" -delete 2>/dev/null || true
        log_success "Todos los logs eliminados excepto el actual."
    fi
}

# --- Verification ---
verify_cleanup() {
    log_info "Verificando resultados de limpieza..."
    
    local active_containers
    active_containers=$(podman ps -q 2>/dev/null | wc -l)
    
    local compose_containers
    if [[ -f "$COMPOSE_FILE" ]]; then
        compose_containers=$(podman-compose -f "$COMPOSE_FILE" ps -q 2>/dev/null | wc -l)
    else
        compose_containers=0
    fi
    
    log_info "Contenedores activos: $active_containers"
    log_info "Contenedores compose activos: $compose_containers"
    
    if [[ "$active_containers" -eq 0 ]] && [[ "$compose_containers" -eq 0 ]]; then
        log_success "✅ Limpieza completada exitosamente - No hay contenedores activos."
        return 0
    else
        log_warn "⚠️  Aún hay contenedores activos después de la limpieza."
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
    
    log_info "=== Iniciando Limpieza Profunda Segura ==="
    log_info "Log file: $LOG_FILE"
    log_info "Timestamp: $TIMESTAMP"
    
    # 3. Confirmación segura con timeout
    if ! get_user_confirmation; then
        log_info "Operación cancelada por el usuario o timeout."
        exit 0
    fi
    
    # 4. Tomar snapshot del sistema
    take_system_snapshot
    
    # 5. Realizar limpiezas
    perform_compose_cleanup
    perform_system_cleanup
    perform_logs_cleanup
    
    # 6. Verificar resultados
    verify_cleanup
    
    # 7. Mensaje final
    log_success "=== Limpieza Profunda Completada ==="
    log_info "Log completo disponible en: $LOG_FILE"
    log_info "Snapshot del sistema antes de limpieza: ${LOG_DIR}/system_snapshot_before_clean_${TIMESTAMP}.txt"
}

# Ejecutar main si no estamos siendo sourceados
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi