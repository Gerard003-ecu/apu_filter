#!/usr/bin/env bash

# ==============================================================================
# Script de Limpieza Profunda Segura (Nuke) para APU Filter Ecosystem
# Versión: 4.0.0
#
# Descripción:
#   Elimina TODOS los recursos del ecosistema APU Filter: contenedores,
#   volúmenes, redes e imágenes opcionales. Requiere confirmación explícita
#   con timeout de seguridad. Toma un snapshot del estado previo y aplica
#   política configurable de rotación de logs.
#
# Uso:
#   ./clean_podman.sh [--help] [--yes] [--dry-run]
#
# Opciones:
#   --help      Muestra esta ayuda y termina.
#   --yes       Omite la confirmación interactiva (modo no-interactivo/CI).
#   --dry-run   Muestra las operaciones que se ejecutarían sin ejecutarlas.
#
# Variables de entorno opcionales:
#   CLEAN_CONFIRMATION_TIMEOUT=30    Segundos máximos para confirmar (default: 30)
#   PRESERVE_RECENT_LOGS=true        Si false, elimina todos los logs excepto el actual
#   PRESERVE_RECENT_LOGS_HOURS=24    Horas de retención para logs recientes (default: 24)
#   CLEAN_UNUSED_IMAGES=false        Si true, elimina imágenes no utilizadas del sistema
#   CLEAN_ALL_CONTAINERS=false       Si true, elimina todos los contenedores detenidos
#
# ADVERTENCIA:
#   Esta operación es DESTRUCTIVA e IRREVERSIBLE. El ecosistema deberá
#   ser reconstruido con 'start_podman.sh' tras su ejecución.
#
# Autor: Equipo APU Filter
# Licencia: Propietaria
# ==============================================================================

# ==========================================================
# SECCIÓN 1: MODO ESTRICTO
# ==========================================================

set -euo pipefail
IFS=$'\n\t'

# Verificar versión mínima de Bash (4.2+ requerido para arrays
# asociativos y el operador de minúsculas ${var,,}).
if [[ "${BASH_VERSINFO[0]}" -lt 4 ]] || \
   { [[ "${BASH_VERSINFO[0]}" -eq 4 ]] && [[ "${BASH_VERSINFO[1]}" -lt 2 ]]; }; then
    printf 'ERROR: Se requiere Bash 4.2 o superior. Versión actual: %s\n' \
        "${BASH_VERSION}" >&2
    exit 1
fi

# ==========================================================
# SECCIÓN 2: CONFIGURACIÓN GLOBAL
# ==========================================================

readonly SCRIPT_VERSION="4.0.0"
readonly SCRIPT_PID=$$

# Resolución robusta: resiste llamadas con rutas relativas o desde symlinks.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR

# Detectar raíz del proyecto: prioritiza el directorio del script.
if [[ -f "${SCRIPT_DIR}/compose.yaml" ]]; then
    PROJECT_ROOT="${SCRIPT_DIR}"
else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
fi
readonly PROJECT_ROOT

readonly LOG_DIR="${PROJECT_ROOT}/logs"
readonly COMPOSE_FILE="${PROJECT_ROOT}/compose.yaml"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly TIMESTAMP

readonly LOG_FILE="${LOG_DIR}/podman_clean_${TIMESTAMP}.log"

# Constantes de retención y confirmación (sujetas a override por env vars).
readonly DEFAULT_CONFIRMATION_TIMEOUT=30
readonly DEFAULT_PRESERVE_HOURS=24

# Flag global de inicialización del sistema de logging.
declare -g LOG_INITIALIZED=false

# Flags de comportamiento (modificados por parseo de argumentos).
declare -g OPT_YES=false
declare -g OPT_DRY_RUN=false

# ==========================================================
# SECCIÓN 3: COLORES (respeta NO_COLOR y detección de TTY)
# ==========================================================

# _init_colors
#
# Inicializa el array asociativo COLORS con secuencias ANSI o cadenas
# vacías si stdout no es un TTY o si NO_COLOR está definido.
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
# SECCIÓN 4: LOGGING ESTRUCTURADO
# ==========================================================

# _log NIVEL COLOR MENSAJE
#
# Emite una línea con formato enriquecido a consola y otra sin ANSI al
# archivo de log. Los mensajes de nivel ERROR se envían a stderr.
#
# Formato consola : [NIVEL  ] [HH:MM:SS] MENSAJE
# Formato archivo : [YYYY-MM-DD HH:MM:SS] [NIVEL  ] PID:NNNNN MENSAJE
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

    # Solo escribir al archivo si el logging está inicializado y el archivo
    # existe (protege contra condiciones de carrera durante el arranque).
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

# log_dry_run MENSAJE
#
# Emite un mensaje de acción simulada cuando OPT_DRY_RUN=true.
# No realiza ninguna operación destructiva.
log_dry_run() {
    _log "DRY-RUN" "${COLORS[CYAN]}" "[SIMULADO] $1"
}

# init_logging
#
# Crea el directorio de logs (750) y el archivo de log (640).
# Aborta si el destino de LOG_FILE es un symlink (prevención de ataque).
#
# Retorna: 0 si inicializado correctamente, 1 si falla.
init_logging() {
    if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
        printf 'ERROR: No se pudo crear el directorio de logs: %s\n' \
            "$LOG_DIR" >&2
        return 1
    fi

    chmod 750 "$LOG_DIR" 2>/dev/null || true

    # Defenderse contra symlinks hostiles antes de crear el archivo.
    if [[ -L "$LOG_FILE" ]]; then
        printf 'ERROR: El archivo de log es un symlink (posible ataque): %s\n' \
            "$LOG_FILE" >&2
        return 1
    fi

    # Crear el archivo con permisos 640 (umask 0137 = 0666 & ~0137).
    ( umask 0137; : > "$LOG_FILE" )

    if [[ ! -f "$LOG_FILE" ]]; then
        printf 'ERROR: No se pudo crear el archivo de log: %s\n' \
            "$LOG_FILE" >&2
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
# Manejador de trampa EXIT. Emite el código de salida y referencia el log.
cleanup_on_exit() {
    local exit_code=$?
    # Suprimir recursión de traps durante el manejador.
    trap '' EXIT INT TERM

    if [[ "$exit_code" -ne 0 ]]; then
        if [[ "$LOG_INITIALIZED" == true ]]; then
            log_error "Script terminado con código de error: ${exit_code}"
            log_info  "Revisar log: ${LOG_FILE}"
        else
            printf 'ERROR: Script terminado con código: %s\n' \
                "$exit_code" >&2
        fi
    else
        if [[ "$LOG_INITIALIZED" == true ]]; then
            log_info "Script de limpieza finalizado exitosamente."
        fi
    fi
}

trap 'cleanup_on_exit' EXIT
trap 'exit 130' INT    # 128 + SIGINT(2)
trap 'exit 143' TERM   # 128 + SIGTERM(15)

# ==========================================================
# SECCIÓN 6: PARSEO DE ARGUMENTOS
# ==========================================================

# parse_arguments [$@]
#
# Procesa las opciones de línea de comandos. Los flags son:
#   --help      Imprime la cabecera del script y termina (código 0).
#   --yes       Habilita OPT_YES; omite la confirmación interactiva.
#   --dry-run   Habilita OPT_DRY_RUN; ninguna operación destructiva se ejecuta.
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help|-h)
                sed -n '2,/^# ===\+$/p' "${BASH_SOURCE[0]}" | \
                    grep '^#' | sed 's/^# \{0,1\}//'
                exit 0
                ;;
            --yes|-y)
                OPT_YES=true
                shift
                ;;
            --dry-run|-n)
                OPT_DRY_RUN=true
                shift
                ;;
            --)
                shift
                break
                ;;
            *)
                printf 'ERROR: Opción desconocida: %s\n' "$1" >&2
                printf 'Usa --help para ver las opciones disponibles.\n' >&2
                exit 1
                ;;
        esac
    done
}

# ==========================================================
# SECCIÓN 7: VALIDACIÓN DE DEPENDENCIAS
# ==========================================================

# check_dependencies
#
# Verifica que podman, podman-compose y find estén disponibles en el PATH.
#
# Retorna: 0 si todas presentes, 1 si alguna falta.
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

    log_success "Todas las dependencias verificadas correctamente."
    return 0
}

# ==========================================================
# SECCIÓN 8: CONFIRMACIÓN CON TIMEOUT
# ==========================================================

# get_user_confirmation
#
# Solicita confirmación explícita del usuario con conteo regresivo visible.
# En modo --yes (OPT_YES=true) o --dry-run, la confirmación se omite
# automáticamente. El prompt se emite siempre a stderr para no contaminar
# stdout. Si el timeout se agota, cancela por seguridad.
#
# Acepta: [Yy] para continuar, cualquier otra entrada o Enter para cancelar.
#
# Retorna: 0 si confirmado (o modo auto-yes), 1 si cancelado o timeout.
get_user_confirmation() {
    # Modo no-interactivo o de simulación: confirmar automáticamente.
    if [[ "$OPT_YES" == true ]]; then
        log_info "Confirmación automática (--yes activo)."
        return 0
    fi
    if [[ "$OPT_DRY_RUN" == true ]]; then
        log_info "Confirmación automática (--dry-run activo; ningún cambio real)."
        return 0
    fi

    local timeout
    timeout="${CLEAN_CONFIRMATION_TIMEOUT:-$DEFAULT_CONFIRMATION_TIMEOUT}"

    # Validar que el timeout sea un entero positivo.
    if ! [[ "$timeout" =~ ^[1-9][0-9]*$ ]]; then
        log_warn "CLEAN_CONFIRMATION_TIMEOUT='${timeout}' inválido. Usando default: ${DEFAULT_CONFIRMATION_TIMEOUT}"
        timeout="$DEFAULT_CONFIRMATION_TIMEOUT"
    fi

    # Advertencia detallada de las consecuencias destructivas.
    printf '\n' >&2
    log_warn "⚠️  OPERACIÓN DESTRUCTIVA E IRREVERSIBLE ⚠️"
    log_warn "   Se eliminarán permanentemente:"
    log_warn "   • Contenedores activos e inactivos del proyecto"
    log_warn "   • Volúmenes (pueden contener datos de Redis u otros servicios)"
    log_warn "   • Redes creadas por compose"
    log_warn "   • Imágenes locales no utilizadas (si CLEAN_UNUSED_IMAGES=true)"
    log_warn "   • Contenedores detenidos del sistema (si CLEAN_ALL_CONTAINERS=true)"
    printf '\n' >&2
    log_warn "Tiempo disponible para confirmar: ${timeout}s"
    printf '\n' >&2

    local start_time end_time
    start_time="$(date +%s)"
    end_time=$(( start_time + timeout ))

    local response=""
    local current_time

    while true; do
        current_time="$(date +%s)"
        [[ "$current_time" -ge "$end_time" ]] && break

        local remaining=$(( end_time - current_time ))

        # El prompt va a stderr para no contaminar stdout.
        printf "${COLORS[YELLOW]}¿Deseas continuar con la limpieza? (y/N) [%ds restantes]: ${COLORS[RESET]}" \
            "$remaining" >&2

        # -t 1: timeout de 1s por iteración; -n 1: leer 1 carácter; -r: no escape
        if IFS= read -r -t 1 -n 1 response <&2 2>/dev/null; then
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
                    log_warn "Respuesta inválida '${response}'. Ingresa 'y' para continuar o 'n'/'Enter' para cancelar."
                    ;;
            esac
        fi
        # Si read expira (timeout de 1s), continuar el loop.
    done

    printf '\n' >&2
    log_error "Timeout de confirmación alcanzado (${timeout}s). Operación cancelada por seguridad."
    return 1
}

# ==========================================================
# SECCIÓN 9: SNAPSHOT DEL SISTEMA
# ==========================================================

# take_system_snapshot
#
# Registra el estado completo del sistema antes de la limpieza:
# contenedores activos e inactivos, volúmenes, redes e imágenes.
# El snapshot se guarda en LOG_DIR con su propio timestamp.
# Los errores individuales de cada sección son capturados sin abortar
# el snapshot completo.
take_system_snapshot() {
    log_info "Tomando snapshot del sistema antes de la limpieza..."

    local snapshot_file="${LOG_DIR}/system_snapshot_before_clean_${TIMESTAMP}.txt"

    # Crear el archivo de snapshot con permisos 640.
    ( umask 0137; : > "$snapshot_file" ) 2>/dev/null || {
        log_warn "No se pudo crear el archivo de snapshot: ${snapshot_file}. Continuando sin snapshot."
        return 0
    }

    {
        printf '# Snapshot generado: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
        printf '# Script: %s v%s\n\n' "${BASH_SOURCE[0]}" "$SCRIPT_VERSION"

        printf '## CONTENEDORES ACTIVOS\n'
        podman ps \
            --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" \
            2>/dev/null || printf '(sin contenedores activos o error al consultar)\n'

        printf '\n## CONTENEDORES DETENIDOS\n'
        podman ps -a --filter status=exited \
            --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" \
            2>/dev/null || printf '(ninguno o error al consultar)\n'

        printf '\n## VOLÚMENES\n'
        podman volume ls \
            --format "table {{.Name}}\t{{.Driver}}\t{{.MountPoint}}" \
            2>/dev/null || printf '(ninguno o error al consultar)\n'

        printf '\n## REDES\n'
        podman network ls \
            --format "table {{.Name}}\t{{.Driver}}\t{{.Internal}}" \
            2>/dev/null || printf '(ninguna o error al consultar)\n'

        printf '\n## IMÁGENES LOCALES\n'
        podman images \
            --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.Created}}" \
            2>/dev/null || printf '(ninguna o error al consultar)\n'
    } >> "$snapshot_file" 2>/dev/null

    log_success "Snapshot guardado en: ${snapshot_file}"
}

# ==========================================================
# SECCIÓN 10: OPERACIONES DE LIMPIEZA
# ==========================================================

# _run_or_dry CMD [ARGS...]
#
# Ejecuta el comando dado si OPT_DRY_RUN=false; de lo contrario registra
# la acción simulada sin ejecutarla. Toda salida se redirige al log.
#
# Retorna: el código de salida del comando real, o 0 en modo dry-run.
_run_or_dry() {
    if [[ "$OPT_DRY_RUN" == true ]]; then
        log_dry_run "$(printf '%q ' "$@")"
        return 0
    fi
    "$@" >> "$LOG_FILE" 2>&1
}

# perform_compose_cleanup
#
# Detiene y elimina todos los recursos gestionados por el compose file:
# contenedores, redes y volúmenes anónimos. Si el compose file no existe
# emite una advertencia y retorna sin error (condición degradada).
perform_compose_cleanup() {
    log_info "Ejecutando limpieza de recursos compose..."

    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_warn "Archivo compose no encontrado: ${COMPOSE_FILE}. Omitiendo esta fase."
        return 0
    fi

    log_info "Deteniendo y removiendo todos los recursos del compose..."
    if _run_or_dry podman-compose -f "$COMPOSE_FILE" down \
        --volumes --remove-orphans; then
        log_success "Recursos de compose eliminados exitosamente."
    else
        log_warn "Algunos recursos de compose podrían no haberse eliminado. Consultar el log."
    fi
}

# perform_system_cleanup
#
# Elimina opcionalmente contenedores detenidos e imágenes no utilizadas
# a nivel de sistema (no solo los del proyecto).
#
# Variables de entorno:
#   CLEAN_ALL_CONTAINERS=false   Si true, ejecuta 'podman container prune'
#   CLEAN_UNUSED_IMAGES=false    Si true, ejecuta 'podman image prune'
perform_system_cleanup() {
    local cleanup_containers="${CLEAN_ALL_CONTAINERS:-false}"
    local cleanup_images="${CLEAN_UNUSED_IMAGES:-false}"

    # Normalizar a minúsculas (Bash 4.2+).
    cleanup_containers="${cleanup_containers,,}"
    cleanup_images="${cleanup_images,,}"

    if [[ "$cleanup_containers" == "true" ]]; then
        log_info "Removiendo todos los contenedores detenidos del sistema..."
        if _run_or_dry podman container prune -f; then
            log_success "Contenedores detenidos eliminados."
        else
            log_warn "Error al eliminar contenedores detenidos. Consultar el log."
        fi
    fi

    if [[ "$cleanup_images" == "true" ]]; then
        log_info "Removiendo imágenes no utilizadas del sistema..."
        if _run_or_dry podman image prune -f; then
            log_success "Imágenes no utilizadas eliminadas."
        else
            log_warn "Error al eliminar imágenes. Consultar el log."
        fi
    fi
}

# perform_logs_cleanup
#
# Elimina archivos de log según la política configurada:
#   PRESERVE_RECENT_LOGS=true  (default): elimina logs más antiguos que
#       PRESERVE_RECENT_LOGS_HOURS (default: 24h), preservando los recientes.
#   PRESERVE_RECENT_LOGS=false: elimina TODOS los logs excepto el activo.
#
# El archivo de log activo (LOG_FILE) NUNCA se elimina.
# En modo --dry-run, solo emite las rutas que serían eliminadas.
perform_logs_cleanup() {
    local preserve_recent="${PRESERVE_RECENT_LOGS:-true}"

    # Leer el valor de horas desde la variable de entorno; si no está
    # definida o es inválida, usar el default de la constante global.
    local hours_to_preserve="${PRESERVE_RECENT_LOGS_HOURS:-$DEFAULT_PRESERVE_HOURS}"

    # Normalizar a minúsculas.
    preserve_recent="${preserve_recent,,}"

    # Validar que sea un entero positivo.
    if ! [[ "$hours_to_preserve" =~ ^[1-9][0-9]*$ ]]; then
        log_warn "PRESERVE_RECENT_LOGS_HOURS='${hours_to_preserve}' inválido. Usando default: ${DEFAULT_PRESERVE_HOURS}"
        hours_to_preserve="$DEFAULT_PRESERVE_HOURS"
    fi

    log_info "Realizando limpieza de logs..."

    # Construir el predicado de búsqueda según la política.
    local -a find_args=("$LOG_DIR" "-name" "*.log" "-type" "f" "!" "-path" "$LOG_FILE")

    if [[ "$preserve_recent" == "true" ]]; then
        local minutes_threshold=$(( hours_to_preserve * 60 ))
        log_info "Preservando logs de las últimas ${hours_to_preserve}h (${minutes_threshold}min)..."
        find_args+=("-mmin" "+${minutes_threshold}")
    else
        log_warn "Eliminando TODOS los logs excepto el activo..."
    fi

    if [[ "$OPT_DRY_RUN" == true ]]; then
        # En dry-run, listar los archivos que serían eliminados.
        local count=0
        while IFS= read -r log_path; do
            log_dry_run "Eliminar: ${log_path}"
            (( count++ )) || true
        done < <(find "${find_args[@]}" 2>/dev/null)
        log_dry_run "Total de logs que serían eliminados: ${count}"
    else
        find "${find_args[@]}" -delete 2>/dev/null || true
        log_success "Limpieza de logs completada."
    fi
}

# ==========================================================
# SECCIÓN 11: VERIFICACIÓN POST-LIMPIEZA
# ==========================================================

# verify_cleanup
#
# Verifica el estado post-limpieza reportando:
# - Contenedores activos a nivel de sistema
# - Volúmenes del proyecto que pueden haber quedado
# - Estado general del compose (si el archivo existe)
#
# Es informativa: nunca aborta el script (retorna siempre).
verify_cleanup() {
    log_info "Verificando resultados de limpieza..."

    # Contar contenedores activos a nivel de sistema.
    local active_containers=0
    local line
    while IFS= read -r line; do
        [[ -n "$line" ]] && (( active_containers++ )) || true
    done < <(podman ps -q 2>/dev/null)

    # Contar volúmenes del proyecto por label.
    local project_volumes=0
    local project_name
    project_name="$(basename "$PROJECT_ROOT")"
    while IFS= read -r line; do
        [[ -n "$line" ]] && (( project_volumes++ )) || true
    done < <(
        podman volume ls \
            --filter "label=io.podman.compose.project=${project_name}" \
            --format '{{.Name}}' \
            2>/dev/null
    )

    log_info "Contenedores activos (sistema)  : ${active_containers}"
    log_info "Volúmenes del proyecto restantes: ${project_volumes}"

    if [[ "$active_containers" -eq 0 ]] && [[ "$project_volumes" -eq 0 ]]; then
        log_success "✅ Limpieza completada — sin contenedores activos ni volúmenes residuales."
        return 0
    else
        log_warn "⚠️  Recursos residuales detectados. Revisar el log para detalles."
        return 0  # Informativo: nunca abortar.
    fi
}

# ==========================================================
# SECCIÓN 12: FUNCIÓN PRINCIPAL
# ==========================================================

# main [$@]
#
# Orquesta la limpieza profunda en 8 pasos:
#   1. Parsear argumentos de línea de comandos
#   2. Verificar dependencias del sistema
#   3. Inicializar el sistema de logging
#   4. Solicitar confirmación explícita del usuario
#   5. Tomar snapshot del estado previo
#   6. Ejecutar limpiezas (compose, sistema, logs)
#   7. Verificar resultados (informativo)
#   8. Emitir reporte final
main() {
    # ── Paso 1: Parsear argumentos ──────────────────────────────────────────
    parse_arguments "$@"

    # ── Paso 2: Dependencias ────────────────────────────────────────────────
    check_dependencies || exit 1

    # ── Paso 3: Logging ─────────────────────────────────────────────────────
    if ! init_logging; then
        printf 'FATAL: No se pudo inicializar el sistema de logging.\n' >&2
        exit 1
    fi

    log_info "══════════════════════════════════════════════════════════"
    log_info "  Iniciando Limpieza Profunda Segura (Nuke)"
    log_info "  Versión: ${SCRIPT_VERSION} | PID: ${SCRIPT_PID}"
    log_info "══════════════════════════════════════════════════════════"
    log_info "Log file    : ${LOG_FILE}"
    log_info "Project root: ${PROJECT_ROOT}"
    [[ "$OPT_DRY_RUN" == true ]] && log_warn "Modo        : DRY-RUN (ningún cambio real se aplicará)"
    [[ "$OPT_YES"     == true ]] && log_warn "Modo        : --yes (confirmación automática)"

    # ── Paso 4: Confirmación del usuario ────────────────────────────────────
    if ! get_user_confirmation; then
        log_info "Operación cancelada por el usuario o por timeout de seguridad."
        exit 0    # Salida limpia: decisión válida del usuario.
    fi

    # ── Paso 5: Snapshot del estado previo ──────────────────────────────────
    take_system_snapshot

    # ── Paso 6: Ejecutar limpiezas ──────────────────────────────────────────
    perform_compose_cleanup
    perform_system_cleanup
    perform_logs_cleanup

    # ── Paso 7: Verificar resultados (informativo, nunca falla el script) ──
    verify_cleanup

    # ── Paso 8: Reporte final ───────────────────────────────────────────────
    log_success "══════════════════════════════════════════════════════════"
    if [[ "$OPT_DRY_RUN" == true ]]; then
        log_success "  DRY-RUN completado — ningún recurso fue modificado ✓"
    else
        log_success "  Limpieza Profunda Completada ✓"
    fi
    log_success "══════════════════════════════════════════════════════════"
    log_info "Log completo: ${LOG_FILE}"
    log_info "Snapshot    : ${LOG_DIR}/system_snapshot_before_clean_${TIMESTAMP}.txt"
}

# Ejecutar main solo si el script es invocado directamente (no sourced).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi