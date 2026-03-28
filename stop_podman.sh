#!/usr/bin/env bash

# ==============================================================================
# Script de Detención Segura para APU Filter Ecosystem
# Versión: 3.0.0 (Refactorización Rigurosa)
#
# Descripción:
#   Detiene los servicios del ecosistema APU Filter de forma ordenada usando
#   podman-compose, con logging estructurado, validación de estado
#   post-detención y limpieza opcional de recursos.
#
# Uso:
#   ./stop_podman.sh
#
# Variables de entorno opcionales:
#   STOP_TIMEOUT=30             Segundos de gracia para detención suave (default: 30)
#   FORCE_KILL_ON_STOP=false    Si true, usa 'kill' forzoso si 'stop' falla
#   CLEANUP_VOLUMES=false       Si true, elimina volúmenes definidos en compose
#   CLEANUP_NETWORKS=false      Si true, elimina redes e imágenes locales
#
# Autor: Equipo APU Filter
# Licencia: Propietaria
# ==============================================================================

# ==========================================================
# SECCIÓN 1: MODO ESTRICTO
# ==========================================================

set -euo pipefail
IFS=$'\n\t'

# Verificar versión mínima de Bash (4.2+ requerido)
if [[ "${BASH_VERSINFO[0]}" -lt 4 ]] || \
   { [[ "${BASH_VERSINFO[0]}" -eq 4 ]] && [[ "${BASH_VERSINFO[1]}" -lt 2 ]]; }; then
    printf 'ERROR: Se requiere Bash 4.2 o superior. Versión actual: %s\n' \
        "${BASH_VERSION}" >&2
    exit 1
fi

# ==========================================================
# SECCIÓN 2: CONFIGURACIÓN GLOBAL
# ==========================================================

readonly SCRIPT_VERSION="3.0.0"
readonly SCRIPT_PID=$$

readonly SCRIPT_DIR
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR

# Detectar raíz del proyecto (prioritiza directorio del script, luego sube un nivel)
if [[ -f "${SCRIPT_DIR}/compose.yaml" ]]; then
    readonly PROJECT_ROOT="${SCRIPT_DIR}"
else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
    readonly PROJECT_ROOT
fi

readonly LOG_DIR="${PROJECT_ROOT}/logs"
readonly COMPOSE_FILE="${PROJECT_ROOT}/compose.yaml"

readonly TIMESTAMP
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly TIMESTAMP

readonly LOG_FILE="${LOG_DIR}/podman_stop_${TIMESTAMP}.log"

declare -g LOG_INITIALIZED=false

# ==========================================================
# SECCIÓN 3: COLORES (respeta NO_COLOR y detección de TTY)
# ==========================================================

_init_colors() {
    if [[ ! -t 1 ]] || [[ -n "${NO_COLOR:-}" ]]; then
        declare -gA COLORS=(
            [RESET]='' [RED]='' [GREEN]='' [YELLOW]=''
            [BLUE]='' [PURPLE]='' [CYAN]='' [BOLD]=''
        )
    else
        declare -gA COLORS=(
            [RESET]='\033[0m'
            [RED]='\033[0;31m'
            [GREEN]='\033[0;32m'
            [YELLOW]='\033[0;33m'
            [BLUE]='\033[0;34m'
            [PURPLE]='\033[0;35m'
            [CYAN]='\033[0;36m'
            [BOLD]='\033[1m'
        )
    fi
}
_init_colors

# ==========================================================
# SECCIÓN 4: LOGGING
# ==========================================================

# _log [nivel] [color] [mensaje]
#
# Escribe a consola con color y al archivo de log sin ANSI (con PID).
# ERROR → stderr; el resto → stdout.
_log() {
    local level="$1"
    local color="$2"
    local message="$3"

    local ts_short ts_full
    ts_short="$(date '+%H:%M:%S')"
    ts_full="$(date '+%Y-%m-%d %H:%M:%S')"

    local console_line
    console_line="$(printf "${color}[%-7s]${COLORS[RESET]} [%s] %s" \
        "$level" "$ts_short" "$message")"

    if [[ "$level" == "ERROR" ]]; then
        printf '%b\n' "$console_line" >&2
    else
        printf '%b\n' "$console_line"
    fi

    if [[ "$LOG_INITIALIZED" == true ]] && [[ -f "$LOG_FILE" ]]; then
        printf '[%s] [%-7s] PID:%-6s %s\n' \
            "$ts_full" "$level" "$SCRIPT_PID" "$message" \
            >> "$LOG_FILE" 2>/dev/null || true
    fi
}

log_info()    { _log "INFO"    "${COLORS[BLUE]}"   "$1"; }
log_success() { _log "SUCCESS" "${COLORS[GREEN]}"  "$1"; }
log_warn()    { _log "WARN"    "${COLORS[YELLOW]}" "$1"; }
log_error()   { _log "ERROR"   "${COLORS[RED]}"    "$1"; }

# init_logging
#
# Crea el directorio y archivo de log con permisos restrictivos (640).
# Protege contra symlink attacks.
init_logging() {
    if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
        printf 'ERROR: No se pudo crear directorio de logs: %s\n' "$LOG_DIR" >&2
        return 1
    fi

    if [[ -L "$LOG_FILE" ]]; then
        printf 'ERROR: El archivo de log es un symlink (posible ataque): %s\n' \
            "$LOG_FILE" >&2
        return 1
    fi

    (
        umask 0137   # 0666 & ~0137 = 0640
        : > "$LOG_FILE"
    )

    if [[ ! -f "$LOG_FILE" ]]; then
        printf 'ERROR: No se pudo crear archivo de log: %s\n' "$LOG_FILE" >&2
        return 1
    fi

    LOG_INITIALIZED=true
    return 0
}

# ==========================================================
# SECCIÓN 5: GESTIÓN DE SEÑALES
# ==========================================================

# cleanup_on_exit
#
# Manejador de salida: reporta éxito o error con referencia al log.
cleanup_on_exit() {
    local exit_code=$?
    trap '' EXIT INT TERM

    if [[ "$exit_code" -ne 0 ]]; then
        if [[ "$LOG_INITIALIZED" == true ]]; then
            log_error "Script terminado con código de error: ${exit_code}"
            log_info  "Revisar log: ${LOG_FILE}"
        else
            printf 'ERROR: Script terminado con código: %s\n' "$exit_code" >&2
        fi
    else
        if [[ "$LOG_INITIALIZED" == true ]]; then
            log_info "Script de detención finalizado exitosamente."
        fi
    fi
}

trap 'cleanup_on_exit' EXIT
trap 'exit 130' INT    # 128 + 2  (SIGINT)
trap 'exit 143' TERM   # 128 + 15 (SIGTERM)

# ==========================================================
# SECCIÓN 6: VALIDACIÓN DE DEPENDENCIAS
# ==========================================================

# check_dependencies
#
# Verifica que podman y podman-compose estén disponibles en el PATH.
check_dependencies() {
    local -a deps=("podman" "podman-compose")
    local -a missing=()

    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &>/dev/null; then
            missing+=("$dep")
        fi
    done

    if [[ "${#missing[@]}" -gt 0 ]]; then
        log_error "Dependencias faltantes: ${missing[*]}"
        return 1
    fi

    log_info "Dependencias verificadas correctamente."
    return 0
}

# ==========================================================
# SECCIÓN 7: OPERACIONES DE DETENCIÓN
# ==========================================================

# perform_safe_stop
#
# Detiene los servicios con gracia (podman-compose stop) y, si se configura,
# aplica un kill forzoso como respaldo.
#
# Variables de entorno:
#   STOP_TIMEOUT=30            Segundos de gracia (default: 30)
#   FORCE_KILL_ON_STOP=false   Si true, habilita kill forzoso
#
# Retorna: 0 si detenidos correctamente, 1 si falló
perform_safe_stop() {
    local graceful_timeout="${STOP_TIMEOUT:-30}"
    local force_kill="${FORCE_KILL_ON_STOP:-false}"

    # Validar timeout
    if ! [[ "$graceful_timeout" =~ ^[1-9][0-9]*$ ]]; then
        log_warn "STOP_TIMEOUT='${graceful_timeout}' inválido. Usando default: 30"
        graceful_timeout=30
    fi

    log_info "Iniciando detención de servicios (timeout: ${graceful_timeout}s)..."

    if podman-compose -f "$COMPOSE_FILE" stop -t "$graceful_timeout" \
        >> "$LOG_FILE" 2>&1; then
        log_success "Servicios detenidos correctamente."
        return 0
    fi

    log_warn "Detención suave fallida."

    # Normalizar a minúsculas (Bash 4.2+)
    force_kill="${force_kill,,}"

    if [[ "$force_kill" == "true" ]]; then
        log_warn "Intentando kill forzoso..."
        if podman-compose -f "$COMPOSE_FILE" kill >> "$LOG_FILE" 2>&1; then
            log_success "Servicios detenidos con kill forzoso."
            return 0
        else
            log_error "No se pudieron detener los servicios ni con kill forzoso."
            return 1
        fi
    else
        log_warn "Para forzar detención, establece: FORCE_KILL_ON_STOP=true"
        log_error "No se pudieron detener los servicios de forma segura."
        return 1
    fi
}

# cleanup_resources
#
# Elimina contenedores huérfanos y, opcionalmente, volúmenes y redes.
#
# Variables de entorno:
#   CLEANUP_VOLUMES=false   Si true, elimina volúmenes
#   CLEANUP_NETWORKS=false  Si true, elimina redes e imágenes locales
cleanup_resources() {
    local cleanup_volumes="${CLEANUP_VOLUMES:-false}"
    local cleanup_networks="${CLEANUP_NETWORKS:-false}"

    # Normalizar a minúsculas
    cleanup_volumes="${cleanup_volumes,,}"
    cleanup_networks="${cleanup_networks,,}"

    log_info "Realizando limpieza de recursos..."

    # Siempre eliminar huérfanos (nunca falla fatalmente)
    podman-compose -f "$COMPOSE_FILE" down --remove-orphans \
        >> "$LOG_FILE" 2>&1 || true

    if [[ "$cleanup_volumes" == "true" ]]; then
        log_info "Eliminando volúmenes definidos en compose..."
        podman-compose -f "$COMPOSE_FILE" down --remove-orphans --volumes \
            >> "$LOG_FILE" 2>&1 || true
        log_success "Volúmenes eliminados."
    fi

    if [[ "$cleanup_networks" == "true" ]]; then
        log_info "Eliminando redes e imágenes locales definidas en compose..."
        # NOTA: --rmi local elimina imágenes locales (no de registros remotos)
        podman-compose -f "$COMPOSE_FILE" down --remove-orphans --rmi local \
            >> "$LOG_FILE" 2>&1 || true
        log_success "Redes e imágenes locales eliminadas."
    fi

    log_success "Limpieza de recursos completada."
}

# ==========================================================
# SECCIÓN 8: VERIFICACIÓN POST-DETENCIÓN
# ==========================================================

# verify_stopped
#
# Verifica que no haya contenedores del compose activos.
# Usa 'podman ps --filter' en lugar de 'podman-compose ps -q' para
# una consulta más confiable y directa.
#
# Retorna: 0 si todos detenidos, 1 si quedan contenedores activos
verify_stopped() {
    log_info "Verificando estado de servicios..."

    # Contar contenedores activos del proyecto usando el label de compose
    local project_name
    project_name="$(basename "$PROJECT_ROOT")"
    local running_count
    running_count="$(podman ps --filter \
        "label=com.docker.compose.project=${project_name}" \
        --format '{{.Names}}' 2>/dev/null | wc -l)"

    if [[ "$running_count" -eq 0 ]]; then
        log_success "✅ Todos los servicios han sido detenidos correctamente."
        return 0
    else
        log_warn "⚠️  ${running_count} contenedor(es) aún activos:"
        podman ps --filter "label=com.docker.compose.project=${project_name}" \
            --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' \
            2>/dev/null | while IFS= read -r line; do
            [[ -n "$line" ]] && log_warn "   ${line}"
        done
        return 1
    fi
}

# ==========================================================
# SECCIÓN 9: FUNCIÓN PRINCIPAL
# ==========================================================

# main [$@]
#
# Orquesta la detención segura en 6 pasos:
#   1. Verificar dependencias
#   2. Inicializar logging
#   3. Validar compose file
#   4. Detener servicios
#   5. Limpiar recursos
#   6. Verificar estado final
main() {
    # ── Paso 1: Dependencias ────────────────────────────────────────────────
    check_dependencies || exit 1

    # ── Paso 2: Logging ─────────────────────────────────────────────────────
    if ! init_logging; then
        printf 'FATAL: No se pudo inicializar el sistema de logging.\n' >&2
        exit 1
    fi

    log_info "══════════════════════════════════════════════════════════"
    log_info "  Iniciando Detención Segura de APU Filter Ecosystem"
    log_info "  Versión: ${SCRIPT_VERSION} | PID: ${SCRIPT_PID}"
    log_info "══════════════════════════════════════════════════════════"
    log_info "Log file  : ${LOG_FILE}"
    log_info "Timestamp : ${TIMESTAMP}"

    # ── Paso 3: Validar compose file ────────────────────────────────────────
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "No se encuentra el archivo: ${COMPOSE_FILE}"
        log_error "Asegúrate de ejecutar este script desde la raíz del proyecto."
        exit 1
    fi
    log_info "Compose file validado: ${COMPOSE_FILE}"

    # ── Paso 4: Detención segura ────────────────────────────────────────────
    if ! perform_safe_stop; then
        log_error "Fallo en la detención de servicios."
        exit 1
    fi

    # ── Paso 5: Limpiar recursos adicionales ────────────────────────────────
    cleanup_resources

    # ── Paso 6: Verificar estado post-detención ────────────────────────────
    # verify_stopped es informativo; no aborta el script si quedan contenedores
    verify_stopped || true

    log_info "Estado final de contenedores:"
    podman-compose -f "$COMPOSE_FILE" ps 2>/dev/null || true

    log_success "══════════════════════════════════════════════════════════"
    log_success "  APU Filter Ecosystem Detenido Correctamente"
    log_success "══════════════════════════════════════════════════════════"
    log_info "Log completo disponible en: ${LOG_FILE}"
}

# Ejecutar main si no estamos siendo sourceados
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi