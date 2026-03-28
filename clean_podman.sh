#!/usr/bin/env bash

# ==============================================================================
# Script de Limpieza Profunda Segura (Nuke) para APU Filter Ecosystem
# Versión: 3.0.0 (Refactorización Rigurosa)
#
# Descripción:
#   Elimina TODOS los recursos del ecosistema APU Filter: contenedores,
#   volúmenes, redes e imágenes opcionales. Requiere confirmación explícita
#   con timeout de seguridad. Toma un snapshot del estado previo.
#
# Uso:
#   ./clean_podman.sh
#
# Variables de entorno opcionales:
#   CLEAN_CONFIRMATION_TIMEOUT=30    Segundos máximos para confirmar (default: 30)
#   PRESERVE_RECENT_LOGS=true        Si false, elimina todos los logs
#   CLEAN_UNUSED_IMAGES=false        Si true, elimina imágenes no utilizadas
#   CLEAN_ALL_CONTAINERS=false       Si true, elimina todos los contenedores detenidos
#
# ADVERTENCIA:
#   Esta operación es DESTRUCTIVA e IRREVERSIBLE. Úsala con precaución.
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

# Detectar raíz del proyecto
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

readonly LOG_FILE="${LOG_DIR}/podman_clean_${TIMESTAMP}.log"

# Configuración sensible
readonly CONFIRMATION_TIMEOUT=30          # Segundos de espera para confirmación
readonly PRESERVE_RECENT_LOGS_HOURS=24    # Horas de logs recientes a preservar

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
# Escribe a consola con color ANSI y al archivo de log sin ANSI.
# ERROR → stderr; el resto → stdout.
_log() {
    local level="$1"
    local color="$2"
    local message="$3"

    local ts_short ts_full
    ts_short="$(date '+%H:%M:%S')"
    ts_full="$(date '+%Y-%m-%d %H:%M:%S')"

    local console_line
    console_line="$(printf "${color}${COLORS[BOLD]}[%-7s]${COLORS[RESET]} [%s] %s" \
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
            log_info "Script de limpieza finalizado exitosamente."
        fi
    fi
}

trap 'cleanup_on_exit' EXIT
trap 'exit 130' INT    # 128 + 2  (SIGINT)
trap 'exit 143' TERM   # 128 + 15 (SIGTERM)

# ==========================================================
# SECCIÓN 6: VALIDACIÓN DE DEPENDENCIAS
# ==========================================================

check_dependencies() {
    local -a deps=("podman" "podman-compose" "find")
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
# SECCIÓN 7: CONFIRMACIÓN CON TIMEOUT
# ==========================================================

# get_user_confirmation
#
# Solicita confirmación explícita del usuario con conteo regresivo.
# El loop espera entrada de 1 carácter con timeout de 1 segundo en cada
# iteración. Si el tiempo total se agota, cancela por seguridad.
#
# Acepta: [Yy] para continuar, [Nn] o Enter para cancelar.
#
# Retorna: 0 si confirmado, 1 si cancelado o timeout
get_user_confirmation() {
    local timeout="${CLEAN_CONFIRMATION_TIMEOUT:-$CONFIRMATION_TIMEOUT}"

    # Validar timeout
    if ! [[ "$timeout" =~ ^[1-9][0-9]*$ ]]; then
        log_warn "CLEAN_CONFIRMATION_TIMEOUT='${timeout}' inválido. Usando default: ${CONFIRMATION_TIMEOUT}"
        timeout="$CONFIRMATION_TIMEOUT"
    fi

    # Advertencia detallada de la operación destructiva
    log_warn "⚠️  ADVERTENCIA CRÍTICA: Esta operación ELIMINARÁ permanentemente:"
    log_warn "   • Contenedores activos e inactivos del proyecto"
    log_warn "   • Volúmenes (pueden contener datos de Redis u otros servicios)"
    log_warn "   • Redes creadas por compose"
    log_warn "   • Imágenes locales no utilizadas (si CLEAN_UNUSED_IMAGES=true)"
    log_warn ""
    log_warn "Tiempo disponible para confirmar: ${timeout}s"
    log_warn ""

    local start_time end_time
    start_time="$(date +%s)"
    end_time=$(( start_time + timeout ))

    local response=""
    while [[ "$(date +%s)" -lt "$end_time" ]]; do
        local remaining=$(( end_time - $(date +%s) ))
        # Mostrar prompt en stderr para no contaminar stdout con la pregunta
        printf "${COLORS[YELLOW]}¿Deseas continuar con la limpieza? (y/N) [restante: %ds]: ${COLORS[RESET]}" \
            "$remaining" >&2

        # read -t 1: timeout de 1 segundo; -n 1: leer solo 1 carácter; -r: no escape
        if read -r -t 1 -n 1 response 2>/dev/null; then
            printf '\n' >&2
            case "$response" in
                [Yy])
                    log_success "Confirmación recibida: continuar."
                    return 0
                    ;;
                [Nn]|"")
                    log_info "Confirmación recibida: cancelar."
                    return 1
                    ;;
                *)
                    log_warn "Respuesta inválida: '${response}'. Ingresa 'y' para continuar o 'n' para cancelar."
                    ;;
            esac
        fi
        # Si read devuelve error por timeout de 1s, continuar el loop
        true
    done

    printf '\n' >&2
    log_error "Timeout alcanzado (${timeout}s). Operación cancelada por seguridad."
    return 1
}

# ==========================================================
# SECCIÓN 8: SNAPSHOT DEL SISTEMA
# ==========================================================

# take_system_snapshot
#
# Registra el estado completo del sistema Podman antes de la limpieza:
# contenedores, volúmenes, redes e imágenes. El snapshot se guarda en LOG_DIR.
take_system_snapshot() {
    log_info "Tomando snapshot del sistema antes de la limpieza..."

    local snapshot_file="${LOG_DIR}/system_snapshot_before_clean_${TIMESTAMP}.txt"

    {
        printf 'Snapshot generado: %s\n\n' "$(date)"

        printf 'CONTENEDORES ACTIVOS:\n'
        podman ps --format \
            "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" \
            2>/dev/null || printf 'N/A\n'

        printf '\nCONTENEDORES DETENIDOS:\n'
        podman ps -a --filter status=exited \
            --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" \
            2>/dev/null || printf 'N/A\n'

        printf '\nVOLÚMENES:\n'
        podman volume ls --format \
            "table {{.Name}}\t{{.Driver}}\t{{.MountPoint}}" \
            2>/dev/null || printf 'N/A\n'

        printf '\nREDES:\n'
        podman network ls --format \
            "table {{.Name}}\t{{.Driver}}\t{{.Internal}}" \
            2>/dev/null || printf 'N/A\n'

        printf '\nIMÁGENES LOCALES:\n'
        podman images --format \
            "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.Created}}" \
            2>/dev/null || printf 'N/A\n'
    } > "$snapshot_file" 2>/dev/null

    log_success "Snapshot guardado en: ${snapshot_file}"
}

# ==========================================================
# SECCIÓN 9: OPERACIONES DE LIMPIEZA
# ==========================================================

# perform_compose_cleanup
#
# Detiene y elimina todos los recursos gestionados por el compose file
# (contenedores, redes, volúmenes anónimos).
perform_compose_cleanup() {
    log_info "Ejecutando limpieza de recursos compose..."

    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_warn "Archivo compose no encontrado: ${COMPOSE_FILE}. Saltando limpieza de compose."
        return 0
    fi

    log_info "Deteniendo y removiendo recursos de compose..."
    if podman-compose -f "$COMPOSE_FILE" down -v --remove-orphans \
        >> "$LOG_FILE" 2>&1; then
        log_success "Recursos de compose limpiados exitosamente."
    else
        log_warn "Algunos recursos de compose podrían no haberse eliminado completamente."
    fi
}

# perform_system_cleanup
#
# Elimina opcionalmente contenedores detenidos e imágenes no utilizadas
# a nivel de sistema (no solo del proyecto).
#
# Variables de entorno:
#   CLEAN_ALL_CONTAINERS=false   Si true, elimina todos los contenedores detenidos
#   CLEAN_UNUSED_IMAGES=false    Si true, elimina imágenes no utilizadas
perform_system_cleanup() {
    local cleanup_containers="${CLEAN_ALL_CONTAINERS:-false}"
    local cleanup_images="${CLEAN_UNUSED_IMAGES:-false}"

    # Normalizar a minúsculas (Bash 4.2+)
    cleanup_containers="${cleanup_containers,,}"
    cleanup_images="${cleanup_images,,}"

    if [[ "$cleanup_containers" == "true" ]]; then
        log_info "Removiendo todos los contenedores detenidos del sistema..."
        podman container prune -f >> "$LOG_FILE" 2>&1 || true
        log_success "Contenedores detenidos eliminados."
    fi

    if [[ "$cleanup_images" == "true" ]]; then
        log_info "Removiendo imágenes no utilizadas del sistema..."
        podman image prune -f >> "$LOG_FILE" 2>&1 || true
        log_success "Imágenes no utilizadas eliminadas."
    fi
}

# perform_logs_cleanup
#
# Elimina archivos de log según la política configurada:
# - Si PRESERVE_RECENT_LOGS=true (default): elimina logs con
#   antigüedad > PRESERVE_RECENT_LOGS_HOURS (default: 24h).
# - Si PRESERVE_RECENT_LOGS=false: elimina todos los logs excepto el actual.
#
# NOTA: El archivo de log activo (LOG_FILE) NUNCA se elimina.
perform_logs_cleanup() {
    local preserve_recent="${PRESERVE_RECENT_LOGS:-true}"
    local hours_to_preserve="${PRESERVE_RECENT_LOGS_HOURS:-$PRESERVE_RECENT_LOGS_HOURS}"

    # Normalizar a minúsculas
    preserve_recent="${preserve_recent,,}"

    # Validar horas de preservación
    if ! [[ "$hours_to_preserve" =~ ^[1-9][0-9]*$ ]]; then
        log_warn "PRESERVE_RECENT_LOGS_HOURS inválido. Usando default: ${PRESERVE_RECENT_LOGS_HOURS}"
        hours_to_preserve="$PRESERVE_RECENT_LOGS_HOURS"
    fi

    log_info "Realizando limpieza de logs..."

    if [[ "$preserve_recent" == "true" ]]; then
        local minutes_threshold=$(( hours_to_preserve * 60 ))
        log_info "Preservando logs de las últimas ${hours_to_preserve}h..."
        find "$LOG_DIR" -name "*.log" -type f \
            -mmin +"$minutes_threshold" \
            ! -path "$LOG_FILE" \
            -delete 2>/dev/null || true
        log_success "Logs antiguos eliminados (preservados recientes)."
    else
        log_warn "Eliminando TODOS los logs excepto el actual..."
        find "$LOG_DIR" -name "*.log" -type f \
            ! -path "$LOG_FILE" \
            -delete 2>/dev/null || true
        log_success "Logs eliminados (excepto el actual)."
    fi
}

# ==========================================================
# SECCIÓN 10: VERIFICACIÓN POST-LIMPIEZA
# ==========================================================

# verify_cleanup
#
# Verifica que no queden contenedores activos a nivel de sistema
# ni en el compose file. Es informativa (no aborta el script).
#
# Retorna: 0 si limpieza completa, 1 si quedan contenedores activos
verify_cleanup() {
    log_info "Verificando resultados de limpieza..."

    local active_containers=0
    active_containers="$(podman ps -q 2>/dev/null | wc -l)"

    local compose_containers=0
    if [[ -f "$COMPOSE_FILE" ]]; then
        compose_containers="$(podman-compose -f "$COMPOSE_FILE" ps -q \
            2>/dev/null | wc -l)"
    fi

    log_info "Contenedores activos (sistema)  : ${active_containers}"
    log_info "Contenedores activos (compose)  : ${compose_containers}"

    if [[ "$active_containers" -eq 0 ]] && [[ "$compose_containers" -eq 0 ]]; then
        log_success "✅ Limpieza completada exitosamente — sin contenedores activos."
        return 0
    else
        log_warn "⚠️  Aún hay contenedores activos después de la limpieza."
        return 1
    fi
}

# ==========================================================
# SECCIÓN 11: FUNCIÓN PRINCIPAL
# ==========================================================

# main [$@]
#
# Orquesta la limpieza profunda en 7 pasos:
#   1. Verificar dependencias
#   2. Inicializar logging
#   3. Solicitar confirmación
#   4. Tomar snapshot del sistema
#   5. Ejecutar limpiezas (compose, sistema, logs)
#   6. Verificar resultados
#   7. Mostrar mensaje final
main() {
    # ── Paso 1: Dependencias ────────────────────────────────────────────────
    check_dependencies || exit 1

    # ── Paso 2: Logging ─────────────────────────────────────────────────────
    if ! init_logging; then
        printf 'FATAL: No se pudo inicializar el sistema de logging.\n' >&2
        exit 1
    fi

    log_info "══════════════════════════════════════════════════════════"
    log_info "  Iniciando Limpieza Profunda Segura (Nuke)"
    log_info "  Versión: ${SCRIPT_VERSION} | PID: ${SCRIPT_PID}"
    log_info "══════════════════════════════════════════════════════════"
    log_info "Log file  : ${LOG_FILE}"
    log_info "Timestamp : ${TIMESTAMP}"

    # ── Paso 3: Confirmación del usuario ────────────────────────────────────
    if ! get_user_confirmation; then
        log_info "Operación cancelada por el usuario o por timeout de seguridad."
        exit 0    # Salida limpia: el usuario decidió no proceder
    fi

    # ── Paso 4: Snapshot del estado previo ──────────────────────────────────
    take_system_snapshot

    # ── Paso 5: Ejecutar limpiezas ──────────────────────────────────────────
    perform_compose_cleanup
    perform_system_cleanup
    perform_logs_cleanup

    # ── Paso 6: Verificar resultados (informativo) ──────────────────────────
    verify_cleanup || true

    # ── Paso 7: Mensaje final ───────────────────────────────────────────────
    log_success "══════════════════════════════════════════════════════════"
    log_success "  Limpieza Profunda Completada"
    log_success "══════════════════════════════════════════════════════════"
    log_info "Log completo: ${LOG_FILE}"
    log_info "Snapshot previo: ${LOG_DIR}/system_snapshot_before_clean_${TIMESTAMP}.txt"
}

# Ejecutar main si no estamos siendo sourceados
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi