#!/bin/bash
# ==============================================================================
# Script de Orquestación para APU Filter Ecosystem con podman-compose
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
: "${STABILIZATION_TIMEOUT:=60}"
: "${HEALTH_CHECK_INTERVAL:=5}"
: "${HEALTH_CHECK_RETRIES:=12}"
: "${VERBOSE:=false}"

readonly LOCK_FILE="/tmp/${SCRIPT_NAME%.*}.lock"

# --- Runtime State ---
LOG_FILE=""
LOCK_FD=""
SERVICES_STARTED=false
SKIP_BUILD=false
SKIP_CLEANUP=false
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

log_info()    { _log "INFO"    "${COLORS[BLUE]}"   "$1"; }
log_success() { _log "SUCCESS" "${COLORS[GREEN]}"  "$1"; }
log_warn()    { _log "WARN"    "${COLORS[YELLOW]}" "$1"; }
log_error()   { _log "ERROR"   "${COLORS[RED]}"    "$1" >&2; }
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

    # Capture service logs on failure
    if [[ $exit_code -ne 0 ]] && [[ "$SERVICES_STARTED" == "true" ]]; then
        log_warn "Capturando logs de servicios debido a error..."
        if [[ -n "${LOG_FILE:-}" ]]; then
            podman-compose -f "$COMPOSE_FILE" logs --tail=100 >> "$LOG_FILE" 2>&1 || true
        fi
    fi

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
    trap 'log_error "Interrumpido por usuario (SIGINT)"; exit 130' INT
    trap 'log_error "Terminado por señal (SIGTERM)"; exit 143' TERM
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
        die "Otra instancia ya está ejecutándose (PID: $existing_pid)"
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
    LOG_FILE="${LOG_DIR}/podman_deploy_$(date +%Y%m%d_%H%M%S)_${SCRIPT_PID}.log"

    if ! touch "$LOG_FILE" 2>/dev/null; then
        echo "[ERROR] No se puede crear archivo de log: $LOG_FILE" >&2
        exit 1
    fi

    # Write header to log file
    {
        echo "=============================================="
        echo "APU Filter Ecosystem - Deployment Log"
        echo "Script Version: $SCRIPT_VERSION"
        echo "Started: $(get_timestamp)"
        echo "PID: $SCRIPT_PID"
        echo "=============================================="
        echo ""
    } >> "$LOG_FILE"

    log_info "Logging inicializado: ${LOG_FILE}"
}

check_dependencies() {
    log_info "Verificando dependencias del sistema..."

    local -a required_commands=("podman" "podman-compose" "flock")
    local -a missing=()

    for cmd in "${required_commands[@]}"; do
        if command -v "$cmd" &>/dev/null; then
            local version
            version=$("$cmd" --version 2>/dev/null | head -1 || echo "versión desconocida")
            log_debug "✓ $cmd: $version"
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

    log_success "Todas las dependencias verificadas"
}

validate_compose_file() {
    log_info "Validando archivo compose: $COMPOSE_FILE"

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

    # Validate compose syntax
    log_debug "Validando sintaxis del compose file..."
    if ! podman-compose -f "$COMPOSE_FILE" config >/dev/null 2>&1; then
        log_warn "Advertencia en validación de compose (puede ser menor)"
        
        if [[ "$VERBOSE" == "true" ]]; then
            log_debug "Detalles de validación:"
            podman-compose -f "$COMPOSE_FILE" config 2>&1 | head -20 || true
        fi
    fi

    log_success "Archivo compose válido"
}

# ==============================================================================
# DEPLOYMENT FUNCTIONS
# ==============================================================================

cleanup_previous_deployment() {
    if [[ "$SKIP_CLEANUP" == "true" ]]; then
        log_info "Omitiendo limpieza previa (--no-cleanup)"
        return 0
    fi

    log_info "Limpiando despliegue previo..."

    local cleanup_output
    if cleanup_output=$(podman-compose -f "$COMPOSE_FILE" down --remove-orphans --volumes 2>&1); then
        log_debug "Salida de limpieza: $cleanup_output"
        log_success "Limpieza completada"
    else
        log_warn "Limpieza con advertencias (normal si no había contenedores)"
        log_debug "Salida: $cleanup_output"
    fi

    # Extra cleanup: remove dangling images related to project
    log_debug "Limpiando imágenes huérfanas..."
    podman image prune -f >> "$LOG_FILE" 2>&1 || true
}

build_images() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        log_info "Omitiendo construcción (--no-build)"
        return 0
    fi

    log_info "Construyendo imágenes de servicios..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] podman-compose -f $COMPOSE_FILE build"
        return 0
    fi

    local build_start
    build_start=$(date +%s)

    if ! podman-compose -f "$COMPOSE_FILE" build --no-cache 2>&1 | tee -a "$LOG_FILE"; then
        log_error "Fallo en la construcción de imágenes"
        log_error "Últimas líneas del log:"
        tail -30 "$LOG_FILE" | while IFS= read -r line; do
            log_error "  | $line"
        done
        exit 1
    fi

    local build_duration=$(($(date +%s) - build_start))
    log_success "Imágenes construidas en ${build_duration}s"
}

start_services() {
    log_info "Iniciando servicios en modo detached..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] podman-compose -f $COMPOSE_FILE up -d"
        return 0
    fi

    if ! podman-compose -f "$COMPOSE_FILE" up -d 2>&1 | tee -a "$LOG_FILE"; then
        log_error "Fallo al iniciar servicios"
        exit 1
    fi

    SERVICES_STARTED=true
    log_success "Comandos de inicio ejecutados"
}

wait_for_healthy_services() {
    log_info "Verificando salud de servicios (timeout: ${STABILIZATION_TIMEOUT}s)..."

    local max_attempts=$HEALTH_CHECK_RETRIES
    local interval=$HEALTH_CHECK_INTERVAL
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        log_debug "Verificación de salud: intento $attempt/$max_attempts"

        # Get container states
        local total_containers running_containers unhealthy_containers
        
        total_containers=$(podman-compose -f "$COMPOSE_FILE" ps -q 2>/dev/null | wc -l) || total_containers=0
        running_containers=$(podman-compose -f "$COMPOSE_FILE" ps -q --filter status=running 2>/dev/null | wc -l) || running_containers=0
        
        # Check for unhealthy containers
        unhealthy_containers=$(podman-compose -f "$COMPOSE_FILE" ps 2>/dev/null | grep -cE "(unhealthy|starting)" || echo 0)

        log_debug "Estado: $running_containers/$total_containers running, $unhealthy_containers unhealthy/starting"

        # All containers running and healthy
        if [[ $total_containers -gt 0 ]] && \
           [[ $running_containers -eq $total_containers ]] && \
           [[ $unhealthy_containers -eq 0 ]]; then
            log_success "Todos los servicios ($total_containers) están saludables"
            return 0
        fi

        # Check for exited/failed containers
        local failed_containers
        failed_containers=$(podman-compose -f "$COMPOSE_FILE" ps 2>/dev/null | grep -cE "(exited|dead)" || echo 0)
        
        if [[ $failed_containers -gt 0 ]]; then
            log_error "Detectados $failed_containers contenedores fallidos"
            podman-compose -f "$COMPOSE_FILE" ps 2>&1 | tee -a "$LOG_FILE"
            return 1
        fi

        sleep "$interval"
        ((attempt++))
    done

    log_warn "Timeout alcanzado. Estado actual:"
    podman-compose -f "$COMPOSE_FILE" ps 2>&1 | tee -a "$LOG_FILE"
    
    return 0  # Non-fatal: services might still be starting
}

display_final_status() {
    echo ""
    log_separator "=" "ESTADO FINAL"
    echo ""

    # Show container status
    log_info "Contenedores:"
    podman-compose -f "$COMPOSE_FILE" ps 2>&1 | tee -a "$LOG_FILE"
    echo ""

    # Show exposed ports
    log_info "Puertos expuestos:"
    podman-compose -f "$COMPOSE_FILE" ps --format "{{.Names}}: {{.Ports}}" 2>/dev/null | \
        grep -v ": $" | tee -a "$LOG_FILE" || log_info "  (ninguno detectado)"
    echo ""

    # Helpful commands
    log_info "Comandos útiles:"
    log_info "  Ver logs:    podman-compose -f $COMPOSE_FILE logs -f"
    log_info "  Detener:     podman-compose -f $COMPOSE_FILE down"
    log_info "  Reiniciar:   podman-compose -f $COMPOSE_FILE restart"
    log_info "  Estado:      podman-compose -f $COMPOSE_FILE ps"
    echo ""
}

# ==============================================================================
# CLI ARGUMENT PARSING
# ==============================================================================

show_help() {
    cat << EOF
${COLORS[BOLD]}NOMBRE${COLORS[RESET]}
    $SCRIPT_NAME - Orquestación de APU Filter Ecosystem

${COLORS[BOLD]}USO${COLORS[RESET]}
    $SCRIPT_NAME [OPCIONES]

${COLORS[BOLD]}OPCIONES${COLORS[RESET]}
    -h, --help              Muestra esta ayuda
    -V, --version           Muestra la versión
    -v, --verbose           Modo verbose con información de debug
    -f, --file FILE         Archivo compose (default: $COMPOSE_FILE)
    -t, --timeout SECONDS   Timeout de estabilización (default: $STABILIZATION_TIMEOUT)
    --no-build              Omitir construcción de imágenes
    --no-cleanup            Omitir limpieza de contenedores previos
    --dry-run               Mostrar comandos sin ejecutarlos

${COLORS[BOLD]}VARIABLES DE ENTORNO${COLORS[RESET]}
    LOG_DIR                  Directorio de logs (default: ./logs)
    COMPOSE_FILE             Ruta al archivo compose
    STABILIZATION_TIMEOUT    Timeout en segundos
    VERBOSE                  Habilitar modo debug (true/false)

${COLORS[BOLD]}EJEMPLOS${COLORS[RESET]}
    $SCRIPT_NAME                            # Despliegue completo
    $SCRIPT_NAME --verbose --timeout 120    # Con debug y timeout extendido
    $SCRIPT_NAME --no-build                 # Usar imágenes existentes
    VERBOSE=true $SCRIPT_NAME               # Debug via variable

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
                STABILIZATION_TIMEOUT="$2"
                shift 2
                ;;
            --no-build)
                SKIP_BUILD=true
                shift
                ;;
            --no-cleanup)
                SKIP_CLEANUP=true
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
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

main() {
    # Parse CLI arguments first (before any setup that could fail)
    parse_arguments "$@"

    # Initialize logging (needs to happen early)
    setup_logging

    # Setup error handling
    setup_traps

    # Acquire exclusive lock
    acquire_lock

    # Display banner
    log_separator "=" "APU Filter Ecosystem - Despliegue"
    log_info "Versión: $SCRIPT_VERSION"
    log_info "Compose: $COMPOSE_FILE"
    log_info "Timeout: ${STABILIZATION_TIMEOUT}s"
    log_info "PID: $SCRIPT_PID"
    [[ "$VERBOSE" == "true" ]] && log_info "Modo: VERBOSE"
    [[ "$DRY_RUN" == "true" ]] && log_info "Modo: DRY-RUN"
    echo ""

    # === Deployment Pipeline ===
    
    # Step 1: Verify dependencies
    check_dependencies

    # Step 2: Validate compose file
    validate_compose_file

    # Step 3: Cleanup previous deployment
    cleanup_previous_deployment

    # Step 4: Build images
    build_images

    # Step 5: Start services
    start_services

    # Step 6: Wait for healthy state
    wait_for_healthy_services

    # Step 7: Display final status
    display_final_status

    # === Success ===
    log_separator "=" "✓ APU Filter Ecosystem OPERATIVO"
    log_info "Log completo: $LOG_FILE"

    return 0
}

# Entry point
main "$@"#!/bin/bash
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