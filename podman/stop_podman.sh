#!/bin/bash
# ==============================================================================
# Script de Detención para APU Filter Ecosystem con podman-compose
# Versión: 2.0.0
# ==============================================================================

# --- Strict Mode & Error Handling ---
set -euo pipefail
IFS=$'\n\t'

# --- Script Metadata ---
readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_VERSION="2.0.0"
readonly SCRIPT_PID=$$

# --- Default Configuration (overridable via environment) ---
: "${LOG_DIR:=./logs}"
: "${COMPOSE_FILE:=infrastructure/compose.yaml}"
: "${STOP_TIMEOUT:=30}"
: "${VERBOSE:=false}"

readonly LOCK_FILE="/tmp/apu_ecosystem_stop.lock"

# --- Runtime State ---
LOG_FILE=""
LOCK_FD=""

# --- Operation Flags ---
FORCE_STOP=false
REMOVE_VOLUMES=false
REMOVE_IMAGES=false
REMOVE_ORPHANS=true
FULL_DOWN=false
DRY_RUN=false

# --- Terminal Colors (only for interactive output) ---
declare -A COLORS=(
    [RESET]='\033[0m'
    [BOLD]='\033[1m'
    [RED]='\033[0;31m'
    [GREEN]='\033[0;32m'
    [YELLOW]='\033[0;33m'
    [BLUE]='\033[0;34m'
    [CYAN]='\033[0;36m'
    [MAGENTA]='\033[0;35m'
)

# ==============================================================================
# LOGGING FUNCTIONS
# ==============================================================================

is_terminal() {
    [[ -t 1 ]]
}

get_timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

_log() {
    local level="$1"
    local color="$2"
    local message="$3"
    local timestamp
    timestamp="$(get_timestamp)"

    # Terminal output (with colors if interactive)
    if is_terminal; then
        printf "${color}[%-7s]${COLORS[RESET]} %s\n" "$level" "$message"
    else
        printf "[%-7s] %s\n" "$level" "$message"
    fi

    # File output (without colors, with timestamp)
    if [[ -n "${LOG_FILE:-}" ]] && [[ -w "$(dirname "$LOG_FILE")" ]]; then
        printf "[%s] [%-7s] %s\n" "$timestamp" "$level" "$message" >> "$LOG_FILE"
    fi
}

log_info()    { _log "INFO"    "${COLORS[BLUE]}"    "$1"; }
log_success() { _log "SUCCESS" "${COLORS[GREEN]}"   "$1"; }
log_warn()    { _log "WARN"    "${COLORS[YELLOW]}"  "$1"; }
log_error()   { _log "ERROR"   "${COLORS[RED]}"     "$1" >&2; }
log_debug()   { [[ "$VERBOSE" == "true" ]] && _log "DEBUG" "${COLORS[CYAN]}" "$1" || true; }

log_separator() {
    local char="${1:-=}"
    local msg="${2:-}"
    local line
    line=$(printf '%*s' 50 '' | tr ' ' "$char")
    
    if [[ -n "$msg" ]]; then
        log_info "$line"
        log_info " $msg"
        log_info "$line"
    else
        log_info "$line"
    fi
}

# ==============================================================================
# ERROR HANDLING & CLEANUP
# ==============================================================================

cleanup() {
    local exit_code=$?
    
    # Prevent recursive cleanup
    trap - EXIT INT TERM

    log_debug "Ejecutando cleanup (exit_code: $exit_code, PID: $SCRIPT_PID)"

    # Release lock file
    release_lock

    if [[ $exit_code -ne 0 ]]; then
        log_error "Script terminado con errores (código: $exit_code)"
        [[ -n "${LOG_FILE:-}" ]] && log_error "Revisa el log: $LOG_FILE"
    fi

    exit "$exit_code"
}

die() {
    log_error "$1"
    exit "${2:-1}"
}

setup_traps() {
    trap cleanup EXIT
    trap 'log_warn "Interrumpido por usuario (SIGINT)"; exit 130' INT
    trap 'log_warn "Terminado por señal (SIGTERM)"; exit 143' TERM
}

# ==============================================================================
# LOCK MANAGEMENT
# ==============================================================================

acquire_lock() {
    log_debug "Adquiriendo lock: $LOCK_FILE"

    # Create lock file descriptor
    exec {LOCK_FD}>"$LOCK_FILE" || die "No se puede crear lock file: $LOCK_FILE"

    # Try to acquire exclusive lock (non-blocking)
    if ! flock -n "$LOCK_FD"; then
        local existing_pid
        existing_pid=$(cat "$LOCK_FILE" 2>/dev/null || echo "desconocido")
        die "Otra instancia de stop ya está ejecutándose (PID: $existing_pid)"
    fi

    # Write current PID to lock file
    echo "$SCRIPT_PID" >&"$LOCK_FD"
    log_debug "Lock adquirido (PID: $SCRIPT_PID)"
}

release_lock() {
    if [[ -n "${LOCK_FD:-}" ]]; then
        exec {LOCK_FD}>&- 2>/dev/null || true
        rm -f "$LOCK_FILE" 2>/dev/null || true
        log_debug "Lock liberado"
    fi
}

# ==============================================================================
# SETUP & VALIDATION
# ==============================================================================

setup_logging() {
    # Validate and create log directory
    if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
        echo "[ERROR] No se puede crear directorio de logs: $LOG_DIR" >&2
        exit 1
    fi

    if [[ ! -w "$LOG_DIR" ]]; then
        echo "[ERROR] Sin permisos de escritura en: $LOG_DIR" >&2
        exit 1
    fi

    # Initialize log file with unique timestamp
    LOG_FILE="${LOG_DIR}/podman_stop_$(date +%Y%m%d_%H%M%S)_${SCRIPT_PID}.log"

    if ! touch "$LOG_FILE" 2>/dev/null; then
        echo "[ERROR] No se puede crear archivo de log: $LOG_FILE" >&2
        exit 1
    fi

    # Write header to log file
    {
        echo "=============================================="
        echo "APU Filter Ecosystem - Stop Log"
        echo "Script Version: $SCRIPT_VERSION"
        echo "Started: $(get_timestamp)"
        echo "PID: $SCRIPT_PID"
        echo "=============================================="
        echo ""
    } >> "$LOG_FILE"

    log_debug "Logging inicializado: ${LOG_FILE}"
}

check_dependencies() {
    log_debug "Verificando dependencias del sistema..."

    local -a required_commands=("podman" "podman-compose" "flock")
    local -a missing=()

    for cmd in "${required_commands[@]}"; do
        if command -v "$cmd" &>/dev/null; then
            log_debug "✓ $cmd encontrado"
        else
            missing+=("$cmd")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        die "Dependencias faltantes: ${missing[*]}"
    fi

    # Verify podman is functional
    if ! podman info &>/dev/null; then
        die "Podman no está operativo. Verifica el servicio/socket."
    fi

    log_debug "Todas las dependencias verificadas"
}

validate_compose_file() {
    log_debug "Validando archivo compose: $COMPOSE_FILE"

    # Resolve to absolute path if relative
    if [[ ! "$COMPOSE_FILE" = /* ]]; then
        COMPOSE_FILE="${SCRIPT_DIR}/${COMPOSE_FILE}"
    fi

    # Check existence
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        die "Archivo compose no encontrado: $COMPOSE_FILE"
    fi

    # Check readability
    if [[ ! -r "$COMPOSE_FILE" ]]; then
        die "Sin permisos de lectura: $COMPOSE_FILE"
    fi

    log_debug "Archivo compose válido: $COMPOSE_FILE"
}

# ==============================================================================
# STATUS FUNCTIONS
# ==============================================================================

get_running_containers() {
    podman-compose -f "$COMPOSE_FILE" ps -q --filter status=running 2>/dev/null | wc -l || echo 0
}

get_all_containers() {
    podman-compose -f "$COMPOSE_FILE" ps -q 2>/dev/null | wc -l || echo 0
}

show_current_status() {
    log_info "Estado actual de contenedores:"
    
    if ! podman-compose -f "$COMPOSE_FILE" ps 2>&1 | tee -a "$LOG_FILE"; then
        log_warn "No se pudo obtener el estado de los contenedores"
    fi
    echo ""
}

check_containers_exist() {
    local total_containers
    total_containers=$(get_all_containers)

    if [[ "$total_containers" -eq 0 ]]; then
        log_info "No hay contenedores del proyecto para detener"
        return 1
    fi

    local running_containers
    running_containers=$(get_running_containers)

    log_info "Contenedores encontrados: $total_containers (corriendo: $running_containers)"
    return 0
}

# ==============================================================================
# STOP FUNCTIONS
# ==============================================================================

stop_services_graceful() {
    log_info "Deteniendo servicios (timeout: ${STOP_TIMEOUT}s)..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] podman-compose -f $COMPOSE_FILE stop -t $STOP_TIMEOUT"
        return 0
    fi

    local stop_start
    stop_start=$(date +%s)

    if podman-compose -f "$COMPOSE_FILE" stop -t "$STOP_TIMEOUT" >> "$LOG_FILE" 2>&1; then
        local stop_duration=$(($(date +%s) - stop_start))
        log_success "Servicios detenidos gracefully en ${stop_duration}s"
        return 0
    else
        log_warn "Stop graceful falló o tomó demasiado tiempo"
        return 1
    fi
}

stop_services_force() {
    log_warn "Forzando detención de servicios (kill)..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] podman-compose -f $COMPOSE_FILE kill"
        return 0
    fi

    if podman-compose -f "$COMPOSE_FILE" kill >> "$LOG_FILE" 2>&1; then
        log_success "Servicios detenidos forzosamente"
        return 0
    else
        log_error "No se pudieron detener los servicios forzosamente"
        return 1
    fi
}

remove_containers() {
    log_info "Eliminando contenedores..."

    local down_args=()
    
    [[ "$REMOVE_ORPHANS" == "true" ]] && down_args+=("--remove-orphans")
    [[ "$REMOVE_VOLUMES" == "true" ]] && down_args+=("--volumes")

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] podman-compose -f $COMPOSE_FILE down ${down_args[*]}"
        return 0
    fi

    if podman-compose -f "$COMPOSE_FILE" down "${down_args[@]}" >> "$LOG_FILE" 2>&1; then
        log_success "Contenedores eliminados"
    else
        log_warn "Algunos contenedores no pudieron eliminarse completamente"
    fi
}

remove_images() {
    if [[ "$REMOVE_IMAGES" != "true" ]]; then
        return 0
    fi

    log_info "Eliminando imágenes del proyecto..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Eliminar imágenes del proyecto"
        return 0
    fi

    # Get images from compose file and remove them
    local images
    images=$(podman-compose -f "$COMPOSE_FILE" config --images 2>/dev/null || true)

    if [[ -n "$images" ]]; then
        echo "$images" | while IFS= read -r image; do
            if [[ -n "$image" ]]; then
                log_debug "Eliminando imagen: $image"
                podman rmi -f "$image" >> "$LOG_FILE" 2>&1 || log_warn "No se pudo eliminar: $image"
            fi
        done
        log_success "Imágenes del proyecto eliminadas"
    else
        log_info "No se encontraron imágenes para eliminar"
    fi

    # Optional: prune dangling images
    log_debug "Limpiando imágenes huérfanas..."
    podman image prune -f >> "$LOG_FILE" 2>&1 || true
}

cleanup_networks() {
    log_debug "Limpiando redes huérfanas..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] podman network prune -f"
        return 0
    fi

    podman network prune -f >> "$LOG_FILE" 2>&1 || true
}

# ==============================================================================
# MAIN STOP WORKFLOW
# ==============================================================================

perform_stop() {
    # Show current status before stopping
    show_current_status

    # Check if there are containers to stop
    if ! check_containers_exist; then
        if [[ "$FULL_DOWN" == "true" ]]; then
            log_info "Ejecutando cleanup de recursos residuales..."
            remove_containers
            remove_images
            cleanup_networks
        fi
        return 0
    fi

    # Decision: full down or just stop
    if [[ "$FULL_DOWN" == "true" ]]; then
        # Full down: stop + remove containers (+ volumes + images if requested)
        log_info "Ejecutando detención completa (down)..."
        
        if [[ "$FORCE_STOP" == "true" ]]; then
            # Force kill first, then down
            stop_services_force || true
        fi
        
        remove_containers
        remove_images
        cleanup_networks
    else
        # Just stop: graceful stop, with optional force fallback
        if [[ "$FORCE_STOP" == "true" ]]; then
            stop_services_force
        else
            if ! stop_services_graceful; then
                log_warn "Intentando detención forzada como fallback..."
                stop_services_force
            fi
        fi
    fi
}

verify_stopped() {
    log_info "Verificando estado final..."

    local running_after
    running_after=$(get_running_containers)

    if [[ "$running_after" -eq 0 ]]; then
        log_success "Todos los contenedores detenidos correctamente"
        return 0
    else
        log_warn "Aún hay $running_after contenedores corriendo"
        show_current_status
        return 1
    fi
}

display_final_status() {
    echo ""
    log_separator "-"

    local remaining
    remaining=$(get_all_containers)

    if [[ "$remaining" -eq 0 ]]; then
        log_success "🛑 APU Filter Ecosystem completamente detenido"
    else
        log_info "🔶 APU Filter Ecosystem detenido ($remaining contenedores permanecen parados)"
    fi

    echo ""
    log_info "Comandos útiles:"
    log_info "  Reiniciar:   ./start_podman.sh"
    log_info "  Ver estado:  podman-compose -f $COMPOSE_FILE ps -a"
    log_info "  Eliminar:    podman-compose -f $COMPOSE_FILE down -v"
    echo ""

    if [[ -n "${LOG_FILE:-}" ]]; then
        log_info "Log: $LOG_FILE"
    fi
}

# ==============================================================================
# CLI ARGUMENT PARSING
# ==============================================================================

show_help() {
    cat << EOF
${COLORS[BOLD]}NOMBRE${COLORS[RESET]}
    $SCRIPT_NAME - Detención de APU Filter Ecosystem

${COLORS[BOLD]}USO${COLORS[RESET]}
    $SCRIPT_NAME [OPCIONES]

${COLORS[BOLD]}DESCRIPCIÓN${COLORS[RESET]}
    Detiene los servicios del APU Filter Ecosystem de forma controlada.
    Por defecto realiza un stop graceful con timeout configurable.

${COLORS[BOLD]}OPCIONES${COLORS[RESET]}
    -h, --help              Muestra esta ayuda
    -V, --version           Muestra la versión
    -v, --verbose           Modo verbose con información de debug
    -f, --file FILE         Archivo compose (default: $COMPOSE_FILE)
    -t, --timeout SECONDS   Timeout para stop graceful (default: $STOP_TIMEOUT)
    
    --force                 Forzar detención inmediata (SIGKILL)
    --down                  Detener Y eliminar contenedores (como 'down')
    --volumes               Con --down: también eliminar volúmenes
    --images                Con --down: también eliminar imágenes
    --no-orphans            No eliminar contenedores huérfanos
    --dry-run               Mostrar comandos sin ejecutarlos

${COLORS[BOLD]}MODOS DE OPERACIÓN${COLORS[RESET]}
    Por defecto (sin flags):
        - Envía SIGTERM a los contenedores
        - Espera hasta --timeout segundos
        - Si falla, intenta SIGKILL automáticamente

    Con --force:
        - Envía SIGKILL inmediatamente (sin esperar)

    Con --down:
        - Detiene y elimina contenedores
        - Limpia redes huérfanas
        - Opcionalmente elimina volúmenes e imágenes

${COLORS[BOLD]}VARIABLES DE ENTORNO${COLORS[RESET]}
    LOG_DIR                  Directorio de logs (default: ./logs)
    COMPOSE_FILE             Ruta al archivo compose
    STOP_TIMEOUT             Timeout en segundos para stop graceful
    VERBOSE                  Habilitar modo debug (true/false)

${COLORS[BOLD]}EJEMPLOS${COLORS[RESET]}
    $SCRIPT_NAME                          # Stop graceful
    $SCRIPT_NAME --force                  # Kill inmediato
    $SCRIPT_NAME --down                   # Stop + eliminar contenedores
    $SCRIPT_NAME --down --volumes         # Stop + eliminar todo (contenedores + volúmenes)
    $SCRIPT_NAME --down --volumes --images  # Limpieza completa
    $SCRIPT_NAME -t 60                    # Stop con timeout de 60s

${COLORS[BOLD]}EXIT CODES${COLORS[RESET]}
    0   Éxito
    1   Error general
    130 Interrumpido (SIGINT)
    143 Terminado (SIGTERM)

EOF
}

show_version() {
    echo "$SCRIPT_NAME version $SCRIPT_VERSION"
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                show_help
                exit 0
                ;;
            -V|--version)
                show_version
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -f|--file)
                [[ -z "${2:-}" ]] && die "La opción --file requiere un argumento"
                COMPOSE_FILE="$2"
                shift 2
                ;;
            -t|--timeout)
                [[ -z "${2:-}" ]] && die "La opción --timeout requiere un argumento"
                [[ ! "$2" =~ ^[0-9]+$ ]] && die "El timeout debe ser numérico: $2"
                STOP_TIMEOUT="$2"
                shift 2
                ;;
            --force)
                FORCE_STOP=true
                shift
                ;;
            --down)
                FULL_DOWN=true
                shift
                ;;
            --volumes)
                REMOVE_VOLUMES=true
                shift
                ;;
            --images)
                REMOVE_IMAGES=true
                shift
                ;;
            --no-orphans)
                REMOVE_ORPHANS=false
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -*)
                die "Opción desconocida: $1 (usa --help para ver opciones)"
                ;;
            *)
                die "Argumento inesperado: $1"
                ;;
        esac
    done

    # Validate flag combinations
    if [[ "$REMOVE_VOLUMES" == "true" ]] && [[ "$FULL_DOWN" != "true" ]]; then
        log_warn "--volumes requiere --down, habilitando --down automáticamente"
        FULL_DOWN=true
    fi

    if [[ "$REMOVE_IMAGES" == "true" ]] && [[ "$FULL_DOWN" != "true" ]]; then
        log_warn "--images requiere --down, habilitando --down automáticamente"
        FULL_DOWN=true
    fi
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

main() {
    # Parse CLI arguments first (before any setup that could fail)
    parse_arguments "$@"

    # Initialize logging
    setup_logging

    # Setup error handling
    setup_traps

    # Acquire exclusive lock
    acquire_lock

    # Display banner
    log_separator "=" "APU Filter Ecosystem - Detención"
    log_info "Versión: $SCRIPT_VERSION"
    log_info "Compose: $COMPOSE_FILE"
    log_info "Timeout: ${STOP_TIMEOUT}s"
    [[ "$FORCE_STOP" == "true" ]] && log_info "Modo: FORCE"
    [[ "$FULL_DOWN" == "true" ]] && log_info "Modo: DOWN (eliminar contenedores)"
    [[ "$REMOVE_VOLUMES" == "true" ]] && log_info "Incluye: Eliminar volúmenes"
    [[ "$REMOVE_IMAGES" == "true" ]] && log_info "Incluye: Eliminar imágenes"
    [[ "$VERBOSE" == "true" ]] && log_info "Modo: VERBOSE"
    [[ "$DRY_RUN" == "true" ]] && log_info "Modo: DRY-RUN"
    echo ""

    # === Validation ===
    check_dependencies
    validate_compose_file

    # === Stop Workflow ===
    perform_stop

    # === Verification ===
    verify_stopped || true  # Non-fatal if some containers remain

    # === Final Status ===
    display_final_status

    return 0
}

# Entry point
main "$@"