#!/usr/bin/env bash

# ==============================================================================
# Script de Detención Segura para APU Filter Ecosystem
# Versión: 4.0.0
#
# Descripción:
#   Detiene los servicios del ecosistema APU Filter de forma ordenada usando
#   podman-compose, con logging estructurado, validación de argumentos,
#   verificación post-detención y limpieza opcional de recursos.
#
# Uso:
#   ./stop_podman.sh [--help] [--skip-cleanup] [--no-verify]
#
# Opciones:
#   --help          Muestra esta ayuda y termina.
#   --skip-cleanup  Omite la fase de cleanup_resources.
#   --no-verify     Omite la verificación post-detención.
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

# Verificar versión mínima de Bash (4.2+ requerido para arrays asociativos
# y el operador de minúsculas ${var,,}).
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

# Resolución robusta del directorio del script (resiste llamadas con
# rutas relativas o desde symlinks).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR

# Detectar raíz del proyecto: prioritiza el directorio del script;
# sube un nivel si compose.yaml no está junto al script.
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

readonly LOG_FILE="${LOG_DIR}/podman_stop_${TIMESTAMP}.log"

# Flag global de inicialización del sistema de logging.
# Evita llamadas a _log antes de que el archivo exista.
declare -g LOG_INITIALIZED=false

# Flags de comportamiento (modificados por parseo de argumentos)
declare -g OPT_SKIP_CLEANUP=false
declare -g OPT_NO_VERIFY=false

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
# archivo de log. Los mensajes de nivel ERROR se envían a stderr; el resto
# a stdout.
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
    console_line="$(printf "${color}[%-7s]${COLORS[RESET]} [%s] %s" \
        "$level" "$ts_short" "$message")"

    if [[ "$level" == "ERROR" ]]; then
        printf '%b\n' "$console_line" >&2
    else
        printf '%b\n' "$console_line"
    fi

    # Solo escribir al archivo si el logging está inicializado y el archivo
    # existe (protege contra condiciones de carrera en arranque).
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
# Crea el directorio de logs (750) y el archivo de log (640).
# Aborta si el destino es un symlink (prevención de symlink attack).
#
# Retorna: 0 si inicializado correctamente, 1 (+ mensaje a stderr) si falla.
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

    # Crear el archivo con permisos 640 (umask = 0666 & ~0026 = 0640).
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
# Manejador de trampa EXIT. Reporta el código de salida y referencia
# el log cuando el logging está disponible.
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
            log_info "Script de detención finalizado exitosamente."
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
# Procesa los argumentos de línea de comandos. Establece los flags
# globales OPT_SKIP_CLEANUP y OPT_NO_VERIFY.
#
# Argumentos soportados:
#   --help          Muestra el bloque de cabecera y termina con código 0.
#   --skip-cleanup  No ejecutar cleanup_resources.
#   --no-verify     No ejecutar verify_stopped.
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help|-h)
                sed -n '2,/^# ===\+$/p' "${BASH_SOURCE[0]}" | \
                    grep '^#' | sed 's/^# \{0,1\}//'
                exit 0
                ;;
            --skip-cleanup)
                OPT_SKIP_CLEANUP=true
                shift
                ;;
            --no-verify)
                OPT_NO_VERIFY=true
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
# Verifica que podman y podman-compose estén disponibles en el PATH.
# Registra cada dependencia faltante antes de retornar el error.
#
# Retorna: 0 si todas presentes, 1 si alguna falta.
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

    log_success "Todas las dependencias verificadas correctamente."
    return 0
}

# ==========================================================
# SECCIÓN 8: VALIDACIÓN DEL ENTORNO
# ==========================================================

# validate_environment
#
# Verifica que el archivo compose.yaml exista y sea legible.
# Verificar legibilidad (no solo existencia) previene errores
# silenciosos cuando el archivo existe pero sus permisos son
# incorrectos.
#
# Retorna: 0 si válido, 1 si inválido.
validate_environment() {
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "Archivo compose no encontrado: ${COMPOSE_FILE}"
        log_error "Asegúrate de ejecutar este script desde la raíz del proyecto."
        return 1
    fi

    if [[ ! -r "$COMPOSE_FILE" ]]; then
        log_error "Archivo compose no legible: ${COMPOSE_FILE}"
        log_error "Verifica los permisos del archivo."
        return 1
    fi

    log_info "Compose file validado: ${COMPOSE_FILE}"
    return 0
}

# ==========================================================
# SECCIÓN 9: OPERACIONES DE DETENCIÓN
# ==========================================================

# perform_safe_stop
#
# Detiene los servicios con gracia mediante 'podman-compose stop'.
# Si la detención suave falla y FORCE_KILL_ON_STOP=true, escala
# a 'podman-compose kill' como respaldo.
#
# Variables de entorno:
#   STOP_TIMEOUT=30           Segundos de gracia (default: 30; debe ser entero > 0)
#   FORCE_KILL_ON_STOP=false  Si true, habilita kill forzoso como respaldo
#
# Retorna: 0 si detenidos correctamente, 1 si falló.
perform_safe_stop() {
    local graceful_timeout="${STOP_TIMEOUT:-30}"
    local force_kill="${FORCE_KILL_ON_STOP:-false}"

    # Validar que el timeout sea un entero positivo.
    if ! [[ "$graceful_timeout" =~ ^[1-9][0-9]*$ ]]; then
        log_warn "STOP_TIMEOUT='${graceful_timeout}' inválido. Usando default: 30"
        graceful_timeout=30
    fi

    # Normalizar a minúsculas (Bash 4.2+).
    force_kill="${force_kill,,}"

    log_info "Iniciando detención de servicios (timeout: ${graceful_timeout}s)..."

    if podman-compose -f "$COMPOSE_FILE" stop -t "$graceful_timeout" \
        >> "$LOG_FILE" 2>&1; then
        log_success "Servicios detenidos correctamente."
        return 0
    fi

    log_warn "Detención suave fallida."

    if [[ "$force_kill" == "true" ]]; then
        log_warn "Intentando kill forzoso..."
        if podman-compose -f "$COMPOSE_FILE" kill \
            >> "$LOG_FILE" 2>&1; then
            log_success "Servicios detenidos mediante kill forzoso."
            return 0
        else
            log_error "No se pudieron detener los servicios ni con kill forzoso."
            return 1
        fi
    else
        log_warn "Para forzar la detención, establece: FORCE_KILL_ON_STOP=true"
        log_error "No se pudieron detener los servicios de forma segura."
        return 1
    fi
}

# cleanup_resources
#
# Elimina contenedores huérfanos del proyecto y, opcionalmente,
# volúmenes y redes/imágenes definidos en el compose file.
#
# Esta función no duplica la llamada a 'down': primero ejecuta
# --remove-orphans sin --volumes; si se pide CLEANUP_VOLUMES, ejecuta
# --volumes en una segunda pasada; si CLEANUP_NETWORKS, usa --rmi local.
#
# Variables de entorno:
#   CLEANUP_VOLUMES=false   Si true, elimina volúmenes nombrados
#   CLEANUP_NETWORKS=false  Si true, elimina redes e imágenes locales
cleanup_resources() {
    local cleanup_volumes="${CLEANUP_VOLUMES:-false}"
    local cleanup_networks="${CLEANUP_NETWORKS:-false}"

    # Normalizar a minúsculas.
    cleanup_volumes="${cleanup_volumes,,}"
    cleanup_networks="${cleanup_networks,,}"

    log_info "Realizando limpieza de recursos..."

    # Eliminar siempre los huérfanos; la falla no es fatal.
    log_info "Eliminando contenedores huérfanos del proyecto..."
    podman-compose -f "$COMPOSE_FILE" down --remove-orphans \
        >> "$LOG_FILE" 2>&1 || true

    # Eliminar volúmenes si se solicita, en pasada separada para evitar
    # conflictos con la eliminación de redes.
    if [[ "$cleanup_volumes" == "true" ]]; then
        log_info "Eliminando volúmenes definidos en compose..."
        podman-compose -f "$COMPOSE_FILE" down --volumes \
            >> "$LOG_FILE" 2>&1 || true
        log_success "Volúmenes eliminados."
    fi

    # Eliminar redes e imágenes locales como paso final.
    if [[ "$cleanup_networks" == "true" ]]; then
        log_info "Eliminando redes e imágenes locales definidas en compose..."
        # --rmi local afecta solo imágenes sin registry remoto.
        podman-compose -f "$COMPOSE_FILE" down --rmi local \
            >> "$LOG_FILE" 2>&1 || true
        log_success "Redes e imágenes locales eliminadas."
    fi

    log_success "Limpieza de recursos completada."
}

# ==========================================================
# SECCIÓN 10: VERIFICACIÓN POST-DETENCIÓN
# ==========================================================

# verify_stopped
#
# Verifica que no haya contenedores del proyecto aún activos, usando
# el label de compose como filtro canónico. Se apoya en la cuenta de
# líneas no vacías para evitar falsos positivos por lineas en blanco
# que produce el formato por defecto de 'podman ps'.
#
# Retorna: 0 si todos detenidos, 1 si quedan contenedores activos.
verify_stopped() {
    log_info "Verificando estado post-detención..."

    local project_name
    project_name="$(basename "$PROJECT_ROOT")"

    # Contar solo líneas no vacías para evitar el falso positivo que
    # produce la cabecera o líneas en blanco de 'podman ps'.
    local running_count=0
    local line
    while IFS= read -r line; do
        [[ -n "$line" ]] && (( running_count++ )) || true
    done < <(
        podman ps \
            --filter "label=com.docker.compose.project=${project_name}" \
            --format '{{.Names}}' \
            2>/dev/null
    )

    if [[ "$running_count" -eq 0 ]]; then
        log_success "✅ Todos los servicios del proyecto han sido detenidos."
        return 0
    else
        log_warn "⚠️  ${running_count} contenedor(es) del proyecto aún activos:"
        podman ps \
            --filter "label=com.docker.compose.project=${project_name}" \
            --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' \
            2>/dev/null \
        | while IFS= read -r line; do
            [[ -n "$line" ]] && log_warn "   ${line}" || true
        done
        return 1
    fi
}

# ==========================================================
# SECCIÓN 11: FUNCIÓN PRINCIPAL
# ==========================================================

# main [$@]
#
# Orquesta la detención segura en 6 pasos:
#   1. Parsear argumentos de línea de comandos
#   2. Verificar dependencias del sistema
#   3. Inicializar el sistema de logging
#   4. Validar el entorno (compose file)
#   5. Detener servicios (con escalado a kill si se configura)
#   6. Limpiar recursos adicionales (opcional)   [--skip-cleanup]
#   7. Verificar estado post-detención            [--no-verify]
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
    log_info "  Iniciando Detención Segura de APU Filter Ecosystem"
    log_info "  Versión: ${SCRIPT_VERSION} | PID: ${SCRIPT_PID}"
    log_info "══════════════════════════════════════════════════════════"
    log_info "Log file    : ${LOG_FILE}"
    log_info "Project root: ${PROJECT_ROOT}"
    [[ "$OPT_SKIP_CLEANUP" == true ]] && log_info "Modo        : --skip-cleanup activo"
    [[ "$OPT_NO_VERIFY"    == true ]] && log_info "Modo        : --no-verify activo"

    # ── Paso 4: Validar entorno ─────────────────────────────────────────────
    validate_environment || exit 1

    # ── Paso 5: Detención segura ────────────────────────────────────────────
    if ! perform_safe_stop; then
        log_error "Fallo en la detención de servicios."
        exit 1
    fi

    # ── Paso 6: Limpiar recursos adicionales ────────────────────────────────
    if [[ "$OPT_SKIP_CLEANUP" == false ]]; then
        cleanup_resources
    else
        log_info "Limpieza de recursos omitida (--skip-cleanup)."
    fi

    # ── Paso 7: Verificar estado post-detención ────────────────────────────
    if [[ "$OPT_NO_VERIFY" == false ]]; then
        # Informativo: no aborta el script si quedan contenedores.
        verify_stopped || true
    else
        log_info "Verificación post-detención omitida (--no-verify)."
    fi

    log_info "Estado final de contenedores del proyecto:"
    podman ps \
        --filter "label=com.docker.compose.project=$(basename "$PROJECT_ROOT")" \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' \
        2>/dev/null || true

    log_success "══════════════════════════════════════════════════════════"
    log_success "  APU Filter Ecosystem Detenido Correctamente ✓"
    log_success "══════════════════════════════════════════════════════════"
    log_info "Log completo disponible en: ${LOG_FILE}"
}

# Ejecutar main solo si el script es invocado directamente (no sourced).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi