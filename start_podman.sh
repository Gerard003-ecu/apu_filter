#!/usr/bin/env bash

# ==============================================================================
# Script de Orquestación Segura para APU Filter Ecosystem
# Versión: 4.0.0 (Refactorización Rigurosa)
#
# Descripción:
#   Construye imágenes, configura registros y despliega servicios del ecosistema
#   APU Filter usando Podman y podman-compose, con políticas de seguridad
#   estrictas, health checks robustos y mecanismo de rollback.
#
# Uso:
#   ./start_podman.sh [opciones]
#
# Opciones:
#   --dry-run              Simula la ejecución sin realizar cambios
#   --skip-build           Omite la construcción de imágenes
#   --skip-registry        Omite la configuración de registros
#   --timeout <seg>        Timeout total para health checks (default: 150)
#   --log-level <nivel>    Nivel de log: DEBUG, INFO, WARN, ERROR (default: INFO)
#   --help                 Muestra esta ayuda
#
# Variables de entorno opcionales:
#   ALLOW_UNCONFINED_SECCOMP=true  Permite seccomp=unconfined como fallback
#   APU_LOG_RETENTION_DAYS=30      Días de retención de logs antiguos
#   STOP_TIMEOUT=30                Segundos de gracia para detención en rollback
#
# Autor: Equipo APU Filter
# Licencia: Propietaria
# ==============================================================================

# ==========================================================
# SECCIÓN 1: MODO ESTRICTO Y CONFIGURACIÓN INICIAL DE SHELL
# ==========================================================

# Activar modo estricto de Bash:
#   -e        : Salir inmediatamente si un comando falla
#   -u        : Tratar variables no definidas como error
#   -o pipefail: El código de retorno de un pipeline es el del último
#                comando que falle (no el último ejecutado)
set -euo pipefail

# Separador de campos: solo newline y tab (evita problemas con espacios en nombres)
IFS=$'\n\t'

# Verificar versión mínima de Bash (4.2+ requerido: arrays asociativos + ${var,,})
if [[ "${BASH_VERSINFO[0]}" -lt 4 ]] || \
   { [[ "${BASH_VERSINFO[0]}" -eq 4 ]] && [[ "${BASH_VERSINFO[1]}" -lt 2 ]]; }; then
    printf 'ERROR: Se requiere Bash 4.2 o superior. Versión actual: %s\n' \
        "${BASH_VERSION}" >&2
    exit 1
fi

# ====================================================
# SECCIÓN 2: CONSTANTES Y CONFIGURACIÓN GLOBAL
# ====================================================

# --- Metadatos del script ---
readonly SCRIPT_VERSION="4.0.0"
readonly SCRIPT_PID=$$
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# --- Detección de rutas ---
# Resuelve el directorio real del script (maneja symlinks)
readonly SCRIPT_DIR
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR

# Detectar raíz del proyecto: busca compose.yaml ascendiendo desde SCRIPT_DIR
_detect_project_root() {
    local dir="$SCRIPT_DIR"
    local max_depth=5
    local depth=0
    local parent

    while [[ "$depth" -lt "$max_depth" ]]; do
        if [[ -f "${dir}/compose.yaml" ]]; then
            printf '%s' "$dir"
            return 0
        fi
        # Evitar subir más allá de /
        parent="$(cd "${dir}/.." && pwd -P)"
        if [[ "$parent" == "$dir" ]]; then
            break
        fi
        dir="$parent"
        (( depth++ )) || true
    done

    # Fallback: directorio del script
    printf '%s' "$SCRIPT_DIR"
    return 1
}

PROJECT_ROOT="$(_detect_project_root)" || {
    printf 'WARN: No se encontró compose.yaml. Usando directorio del script.\n' >&2
}
readonly PROJECT_ROOT

# --- Rutas derivadas ---
readonly LOG_DIR="${PROJECT_ROOT}/logs"
readonly DATA_DIR="${PROJECT_ROOT}/data"
readonly COMPOSE_FILE="${PROJECT_ROOT}/compose.yaml"
readonly REGISTRY_SETUP_SCRIPT="${PROJECT_ROOT}/scripts/setup_podman_registry.sh"
readonly LOG_FILE="${LOG_DIR}/podman_start_${TIMESTAMP}.log"
readonly INFRASTRUCTURE_DIR="${PROJECT_ROOT}/infrastructure"

# --- Permisos (notación octal documentada) ---
# 750: propietario=rwx, grupo=r-x, otros=---
readonly SAFE_DIR_PERMS="750"
# 640: propietario=rw-, grupo=r--, otros=---
readonly SAFE_FILE_PERMS="640"

# --- Health Check: configuración por defecto ---
readonly DEFAULT_HEALTH_TIMEOUT=150     # Timeout total en segundos
readonly HEALTH_INITIAL_INTERVAL=2      # Intervalo inicial entre reintentos
readonly HEALTH_MAX_INTERVAL=15         # Intervalo máximo (backoff exponencial)
# Factor de backoff escalado: 15 representa 1.5 (aritmética entera: i * 15 / 10)
readonly HEALTH_BACKOFF_FACTOR=15

# --- Retención de logs ---
readonly DEFAULT_LOG_RETENTION_DAYS=30

# --- Definición de imágenes a construir ---
# Formato: "nombre_imagen:tag|ruta_relativa_dockerfile"
readonly -a IMAGE_DEFINITIONS=(
    "apu-core:latest|infrastructure/Dockerfile.core"
    "apu-agent:latest|infrastructure/Dockerfile.agent"
)

# --- Servicios críticos que requieren health check ---
readonly -a CRITICAL_SERVICES=(
    "apu-core"
    "apu-agent"
)

# --- Colores (con detección de terminal y respeto a NO_COLOR) ---
# Ref: https://no-color.org/
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

# --- Estado global mutable (para rollback y trazabilidad) ---
declare -g  SERVICES_STARTED=false
declare -ga IMAGES_BUILT=()
declare -g  LOG_INITIALIZED=false
declare -g  DRY_RUN=false
declare -g  SKIP_BUILD=false
declare -g  SKIP_REGISTRY=false
declare -g  HEALTH_TIMEOUT="$DEFAULT_HEALTH_TIMEOUT"
declare -g  LOG_LEVEL="INFO"

# Mapa numérico de niveles de log para comparación eficiente O(1)
declare -gA LOG_LEVEL_MAP=(
    [DEBUG]=0
    [INFO]=1
    [WARN]=2
    [ERROR]=3
)

# ====================================================
# SECCIÓN 3: SISTEMA DE LOGGING ROBUSTO
# ====================================================

# _log [nivel] [color] [mensaje]
#
# Función interna de logging con protección contra ciclos y fallos.
# - Escribe a consola con color ANSI (respeta NO_COLOR y TTY)
# - Escribe a archivo de log sin color (con PID y timestamp completo)
# - Redirige ERROR a stderr; el resto va a stdout
# - Filtra mensajes según LOG_LEVEL configurado
#
# SUCCESS se trata internamente como INFO para el filtrado de nivel.
_log() {
    local level="$1"
    local color="$2"
    local message="$3"

    # Filtrar por nivel de log (SUCCESS equivale a INFO numéricamente)
    local effective_level="${level}"
    [[ "$effective_level" == "SUCCESS" ]] && effective_level="INFO"
    local configured_num="${LOG_LEVEL_MAP[${LOG_LEVEL}]:-1}"
    local message_num="${LOG_LEVEL_MAP[${effective_level}]:-1}"
    if [[ "$message_num" -lt "$configured_num" ]]; then
        return 0
    fi

    local ts_short ts_full
    ts_short="$(date '+%H:%M:%S')"
    ts_full="$(date '+%Y-%m-%d %H:%M:%S')"

    # Formato de consola: [NIVEL  ] [HH:MM:SS] mensaje
    local console_line
    console_line="$(printf "${color}[%-7s]${COLORS[RESET]} [%s] %s" \
        "$level" "$ts_short" "$message")"

    # Escribir a consola (ERROR → stderr, resto → stdout)
    if [[ "$level" == "ERROR" ]]; then
        printf '%b\n' "$console_line" >&2
    else
        printf '%b\n' "$console_line"
    fi

    # Escribir a archivo de log (sin ANSI, con PID para trazabilidad forense)
    if [[ "$LOG_INITIALIZED" == true ]] && [[ -f "$LOG_FILE" ]]; then
        printf '[%s] [%-7s] PID:%-6s %s\n' \
            "$ts_full" "$level" "$SCRIPT_PID" "$message" \
            >> "$LOG_FILE" 2>/dev/null || true
    fi
}

log_debug()   { _log "DEBUG"   "${COLORS[CYAN]}"   "$1"; }
log_info()    { _log "INFO"    "${COLORS[BLUE]}"    "$1"; }
log_success() { _log "SUCCESS" "${COLORS[GREEN]}"   "$1"; }
log_warn()    { _log "WARN"    "${COLORS[YELLOW]}"  "$1"; }
log_error()   { _log "ERROR"   "${COLORS[RED]}"     "$1"; }

# init_logging
#
# Crea el directorio de logs y el archivo de log con permisos restrictivos.
# Protege contra ataques de symlink (TOCTOU mitigado con umask atómico).
#
# Retorna: 0 si exitoso, 1 si falla
init_logging() {
    if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
        printf 'ERROR: No se pudo crear directorio de logs: %s\n' "$LOG_DIR" >&2
        return 1
    fi

    # Protección anti-symlink: si el destino ya es un enlace simbólico, abortar
    if [[ -L "$LOG_FILE" ]]; then
        printf 'ERROR: El archivo de log es un symlink (posible ataque): %s\n' \
            "$LOG_FILE" >&2
        return 1
    fi

    # Creación atómica del archivo con permisos restrictivos (640 = rw-r-----)
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

# rotate_logs
#
# Elimina logs de inicio más antiguos que APU_LOG_RETENTION_DAYS días.
# Valida que el valor sea un entero positivo antes de usarlo.
rotate_logs() {
    local retention_days="${APU_LOG_RETENTION_DAYS:-$DEFAULT_LOG_RETENTION_DAYS}"

    if ! [[ "$retention_days" =~ ^[1-9][0-9]*$ ]]; then
        log_warn "APU_LOG_RETENTION_DAYS='${retention_days}' inválido. Usando default: ${DEFAULT_LOG_RETENTION_DAYS}"
        retention_days="$DEFAULT_LOG_RETENTION_DAYS"
    fi

    local count=0
    count=$(find "$LOG_DIR" -name "podman_start_*.log" -type f \
        -mtime +"$retention_days" 2>/dev/null | wc -l)

    if [[ "$count" -gt 0 ]]; then
        log_info "Rotando ${count} archivo(s) de log con más de ${retention_days} días..."
        find "$LOG_DIR" -name "podman_start_*.log" -type f \
            -mtime +"$retention_days" -delete 2>/dev/null || \
            log_warn "No se pudieron eliminar algunos logs antiguos."
    else
        log_debug "No hay logs antiguos para rotar (retención: ${retention_days} días)."
    fi
}

# ====================================================
# SECCIÓN 4: VALIDACIÓN DE DEPENDENCIAS
# ====================================================

# check_dependencies
#
# Verifica que todas las dependencias requeridas estén disponibles en PATH.
# También comprueba que Podman sea >= 4.0.
#
# Retorna: 0 si todas las dependencias están presentes, 1 si falta alguna
check_dependencies() {
    local -a required_deps=("podman" "podman-compose" "find" "date" "awk")
    local -a missing_deps=()

    for dep in "${required_deps[@]}"; do
        if ! command -v "$dep" &>/dev/null; then
            missing_deps+=("$dep")
        fi
    done

    if [[ "${#missing_deps[@]}" -gt 0 ]]; then
        log_error "Dependencias faltantes: ${missing_deps[*]}"
        log_error "Instala las dependencias requeridas antes de continuar."
        return 1
    fi

    # Verificar versión mínima de podman (4.0+) usando awk (evita grep -oP/PCRE)
    local podman_version
    podman_version="$(podman --version 2>/dev/null | awk '{print $3}' | head -1)"
    if [[ -n "$podman_version" ]]; then
        local major_version="${podman_version%%.*}"
        if [[ "$major_version" -lt 4 ]]; then
            log_warn "Podman ${podman_version} detectado. Se recomienda 4.0+."
        else
            log_debug "Podman ${podman_version} detectado (OK)."
        fi
    fi

    log_success "Todas las dependencias verificadas correctamente."
    return 0
}

# ====================================================
# SECCIÓN 5: CONFIGURACIÓN SEGURA DE DIRECTORIOS
# ====================================================

# setup_secure_directories
#
# Crea LOG_DIR y DATA_DIR con permisos restrictivos (750).
# Reporta anomalías pero no falla si el chmod no es posible.
# No ejecuta chown para evitar requerir privilegios de root.
setup_secure_directories() {
    log_info "Configurando directorios con permisos seguros..."

    local -a target_dirs=("$LOG_DIR" "$DATA_DIR")

    for dir in "${target_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            if [[ "$DRY_RUN" == true ]]; then
                log_info "[DRY-RUN] Crearía directorio: ${dir}"
                continue
            fi
            mkdir -p "$dir" || {
                log_error "No se pudo crear directorio: ${dir}"
                return 1
            }
            log_debug "Directorio creado: ${dir}"
        fi

        if [[ "$DRY_RUN" == true ]]; then
            log_info "[DRY-RUN] Aplicaría permisos ${SAFE_DIR_PERMS} a: ${dir}"
            continue
        fi

        chmod "$SAFE_DIR_PERMS" "$dir" 2>/dev/null || \
            log_warn "No se pudieron establecer permisos ${SAFE_DIR_PERMS} en: ${dir}"

        # Reportar si "others" tiene permisos de escritura (bit 1 en último dígito octal)
        local actual_perms
        actual_perms="$(stat -c '%a' "$dir" 2>/dev/null || echo "000")"
        local others_perm="${actual_perms: -1}"
        if [[ "$others_perm" -ge 2 ]]; then
            log_warn "Directorio ${dir} tiene permisos de escritura para 'others' (${actual_perms})"
        fi
    done

    log_success "Directorios configurados con permisos seguros (${SAFE_DIR_PERMS})."
}

# ====================================================
# SECCIÓN 6: GESTIÓN DE SEÑALES Y LIMPIEZA
# ====================================================

# cleanup_on_exit
#
# Manejador de limpieza ejecutado al salir del script (trap EXIT).
# - Código 0: log de éxito.
# - Código != 0: log de error + rollback si hay servicios iniciados.
#
# Los traps INT/TERM establecen su código de salida antes de activar EXIT:
#   SIGINT  → exit 130  (128 + 2)
#   SIGTERM → exit 143  (128 + 15)
cleanup_on_exit() {
    local exit_code=$?
    # Deshabilitar trap recursivo
    trap '' EXIT INT TERM

    if [[ "$exit_code" -ne 0 ]]; then
        if [[ "$LOG_INITIALIZED" == true ]]; then
            log_error "Script terminado con código de error: ${exit_code}"
            if [[ "$SERVICES_STARTED" == true ]]; then
                log_warn "Ejecutando rollback: deteniendo servicios iniciados..."
                local stop_timeout="${STOP_TIMEOUT:-30}"
                podman-compose -f "$COMPOSE_FILE" stop -t "$stop_timeout" \
                    >> "$LOG_FILE" 2>&1 || true
                podman-compose -f "$COMPOSE_FILE" down --remove-orphans \
                    >> "$LOG_FILE" 2>&1 || true
                log_info "Rollback completado."
            fi
            log_info "Revisar log para detalles: ${LOG_FILE}"
        else
            printf 'ERROR: Script terminado con código: %s\n' "$exit_code" >&2
            printf 'El sistema de logging no se inicializó.\n' >&2
        fi
    else
        if [[ "$LOG_INITIALIZED" == true ]]; then
            log_info "Script finalizado exitosamente."
        fi
    fi
}

# Registrar traps DESPUÉS de definir la función de cleanup
trap 'cleanup_on_exit' EXIT
trap 'exit 130' INT    # 128 + 2  (SIGINT  / Ctrl-C)
trap 'exit 143' TERM   # 128 + 15 (SIGTERM)

# ====================================================
# SECCIÓN 7: HEALTH CHECKS CON BACKOFF EXPONENCIAL
# ====================================================

# wait_for_service_health [nombre_servicio] [timeout_segundos?]
#
# Espera a que un servicio alcance el estado "healthy" o "running".
# Implementa backoff exponencial (aritmética entera) para reducir carga:
#
#   intervalo_n = min(intervalo_{n-1} * FACTOR / 10,  MAX_INTERVAL)
#   FACTOR = HEALTH_BACKOFF_FACTOR = 15  →  equivale a ×1.5
#
# Retorna: 0 si saludable, 1 si timeout agotado o estado de error detectado
wait_for_service_health() {
    local service_name="$1"
    local timeout="${2:-$HEALTH_TIMEOUT}"

    if ! [[ "$timeout" =~ ^[1-9][0-9]*$ ]]; then
        log_error "Timeout inválido: '${timeout}'. Debe ser un entero positivo."
        return 1
    fi

    log_info "Verificando salud de '${service_name}' (timeout: ${timeout}s)..."

    local elapsed=0
    local interval="$HEALTH_INITIAL_INTERVAL"
    local attempt=1

    while [[ "$elapsed" -lt "$timeout" ]]; do
        local service_status=""
        service_status="$(podman-compose -f "$COMPOSE_FILE" ps 2>/dev/null || true)"

        if [[ -z "$service_status" ]]; then
            log_warn "No se pudo obtener estado de servicios (intento ${attempt})"
        else
            # Estado saludable: "healthy" (con HEALTHCHECK) o "running" (sin él)
            if printf '%s\n' "$service_status" | \
                grep -qE "${service_name}.*(healthy|running)"; then
                log_success "Servicio '${service_name}' operativo (${elapsed}s, ${attempt} intentos)."
                return 0
            fi

            # Estado de error irreversible: detener la espera inmediatamente
            if printf '%s\n' "$service_status" | \
                grep -qE "${service_name}.*(exited|dead|error)"; then
                log_error "Servicio '${service_name}' está en estado de error."
                _dump_service_logs "$service_name" 20
                return 1
            fi
        fi

        log_debug "Esperando '${service_name}'... (${elapsed}s/${timeout}s, intento ${attempt}, próximo en ${interval}s)"
        sleep "$interval"
        elapsed=$(( elapsed + interval ))
        (( attempt++ )) || true

        # Backoff exponencial: interval = interval * 15 / 10  (equivale a ×1.5)
        interval=$(( (interval * HEALTH_BACKOFF_FACTOR) / 10 ))
        if [[ "$interval" -gt "$HEALTH_MAX_INTERVAL" ]]; then
            interval="$HEALTH_MAX_INTERVAL"
        fi
    done

    log_error "Timeout agotado esperando '${service_name}' (${timeout}s, ${attempt} intentos)."
    _dump_service_logs "$service_name" 30
    return 1
}

# _dump_service_logs [nombre_servicio] [num_lineas]
#
# Función auxiliar: vuelca las últimas N líneas de logs de un servicio al log de error.
_dump_service_logs() {
    local service_name="$1"
    local lines="${2:-20}"
    log_error "Últimos ${lines} líneas de log del servicio '${service_name}':"
    podman-compose -f "$COMPOSE_FILE" logs --tail="$lines" "$service_name" \
        2>/dev/null | while IFS= read -r line; do
        log_error "  ${line}"
    done || true
}

# ====================================================
# SECCIÓN 8: CONSTRUCCIÓN DE IMÁGENES CON SEGURIDAD
# ====================================================

# build_image_with_security_check [dockerfile] [tag] [contexto?]
#
# Construye una imagen con fallback escalonado de seguridad:
#   1. Construcción estándar con seguridad por defecto
#   2. Si falla y ALLOW_UNCONFINED_SECCOMP=true: reintenta con seccomp=unconfined
#   3. Si ambos fallan: error fatal
#
# Valida el formato del tag con una regex robusta antes de construir.
# Los logs de construcción se redirigen al LOG_FILE (no a stdout).
#
# Retorna: 0 si éxito, 1 si fallo
build_image_with_security_check() {
    local dockerfile_path="$1"
    local image_tag="$2"
    local context_path="${3:-$PROJECT_ROOT}"

    # Validar existencia del Dockerfile
    if [[ ! -f "$dockerfile_path" ]]; then
        log_error "Dockerfile no encontrado: ${dockerfile_path}"
        return 1
    fi

    # Validar formato del tag de imagen (OCI Image Spec)
    # nombre[:tag] donde nombre puede incluir registry/path
    if ! [[ "$image_tag" =~ ^[a-zA-Z0-9][a-zA-Z0-9._/:-]*(:[a-zA-Z0-9._-]+)?$ ]]; then
        log_error "Tag de imagen inválido: '${image_tag}'"
        return 1
    fi

    log_info "Construyendo imagen: ${image_tag} (Dockerfile: ${dockerfile_path})"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Ejecutaría: podman build -f ${dockerfile_path} -t ${image_tag} ${context_path}"
        IMAGES_BUILT+=("$image_tag")
        return 0
    fi

    # Intento 1: Construcción estándar
    log_debug "Intento 1: Construcción con seguridad por defecto..."
    if podman build \
        --pull=newer \
        --no-cache=false \
        -f "$dockerfile_path" \
        -t "$image_tag" \
        "$context_path" >> "$LOG_FILE" 2>&1; then
        log_success "Imagen '${image_tag}' construida con seguridad por defecto."
        IMAGES_BUILT+=("$image_tag")
        return 0
    fi

    log_warn "Construcción estándar fallida para '${image_tag}'."

    # Intento 2: Con seccomp=unconfined (solo si explícitamente permitido)
    # Sanitizar y normalizar la variable de entorno a minúsculas
    local allow_unconfined="${ALLOW_UNCONFINED_SECCOMP:-false}"
    allow_unconfined="${allow_unconfined,,}"

    if [[ "$allow_unconfined" == "true" ]]; then
        log_warn "Reintentando con --security-opt seccomp=unconfined (RIESGO DE SEGURIDAD)"
        log_warn "Esto desactiva el filtrado de syscalls durante la construcción."
        if podman build \
            --security-opt seccomp=unconfined \
            --pull=newer \
            -f "$dockerfile_path" \
            -t "$image_tag" \
            "$context_path" >> "$LOG_FILE" 2>&1; then
            log_warn "Imagen '${image_tag}' construida con seccomp=unconfined."
            log_warn "ACCIÓN REQUERIDA: Investigar por qué la construcción estándar falla."
            IMAGES_BUILT+=("$image_tag")
            return 0
        fi
    else
        log_error "Establece ALLOW_UNCONFINED_SECCOMP=true para habilitar el fallback (con precaución)."
    fi

    log_error "Falló la construcción de '${image_tag}' en todos los intentos."
    log_error "Consulta el log para detalles: ${LOG_FILE}"
    return 1
}

# build_all_images
#
# Construye todas las imágenes definidas en IMAGE_DEFINITIONS (fail-fast).
# Si una imagen falla, el proceso completo se detiene.
#
# Retorna: 0 si todas exitosas, 1 si alguna falló
build_all_images() {
    local total="${#IMAGE_DEFINITIONS[@]}"
    log_info "Iniciando construcción de ${total} imagen(es)..."

    local build_count=0
    for definition in "${IMAGE_DEFINITIONS[@]}"; do
        # Parsear definición: "tag|dockerfile_relativo"
        local image_tag="${definition%%|*}"
        local dockerfile_rel="${definition##*|}"
        local dockerfile_path="${PROJECT_ROOT}/${dockerfile_rel}"

        if ! build_image_with_security_check \
            "$dockerfile_path" "$image_tag" "$PROJECT_ROOT"; then
            log_error "Fallo en imagen $((build_count + 1))/${total}: ${image_tag}"
            return 1
        fi

        (( build_count++ )) || true
        log_info "Progreso: ${build_count}/${total}"
    done

    log_success "Todas las imágenes (${build_count}) construidas correctamente."
    return 0
}

# ====================================================
# SECCIÓN 9: CONFIGURACIÓN DE REGISTROS
# ====================================================

# setup_registries
#
# Ejecuta el script de configuración de registros de Podman si existe.
# Valida que el script sea un archivo regular (no symlink) y tenga shebang.
#
# Retorna: 0 si configurado o no era necesario, 1 si la configuración falló
setup_registries() {
    log_info "Verificando configuración de registros de Podman..."

    if [[ ! -f "$REGISTRY_SETUP_SCRIPT" ]]; then
        log_warn "Script de registros no encontrado: ${REGISTRY_SETUP_SCRIPT}"
        log_warn "Asumiendo configuración manual de registros."
        return 0
    fi

    # Verificaciones de seguridad del script
    if [[ -L "$REGISTRY_SETUP_SCRIPT" ]]; then
        log_error "El script de registros es un symlink (posible riesgo de seguridad)."
        return 1
    fi

    if [[ ! -r "$REGISTRY_SETUP_SCRIPT" ]]; then
        log_error "Sin permisos de lectura en: ${REGISTRY_SETUP_SCRIPT}"
        return 1
    fi

    # Verificar shebang como señal de script de shell válido
    local first_line
    first_line="$(head -1 "$REGISTRY_SETUP_SCRIPT" 2>/dev/null)"
    if [[ "$first_line" != "#!/"* ]]; then
        log_warn "El script de registros no tiene shebang estándar."
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Ejecutaría: bash ${REGISTRY_SETUP_SCRIPT}"
        return 0
    fi

    log_info "Ejecutando configuración de registros..."
    if bash "$REGISTRY_SETUP_SCRIPT" >> "$LOG_FILE" 2>&1; then
        log_success "Configuración de registros aplicada correctamente."
        return 0
    else
        log_error "Fallo en la configuración de registros."
        return 1
    fi
}

# ====================================================
# SECCIÓN 10: GESTIÓN DE SERVICIOS
# ====================================================

# cleanup_previous_deployment
#
# Detiene y elimina contenedores, redes y volúmenes de despliegues previos.
# Los fallos no son fatales (pueden ser normales en primera ejecución).
cleanup_previous_deployment() {
    log_info "Limpiando contenedores y redes de despliegues previos..."

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Ejecutaría: podman-compose -f ${COMPOSE_FILE} down --remove-orphans --volumes"
        return 0
    fi

    podman-compose -f "$COMPOSE_FILE" down --remove-orphans --volumes \
        >> "$LOG_FILE" 2>&1 || {
        log_warn "La limpieza previa generó advertencias (puede ser normal en primera ejecución)."
    }

    log_debug "Limpieza previa completada."
}

# start_services
#
# Inicia los servicios definidos en compose.yaml en modo detached.
# En caso de fallo, vuelca las últimas líneas del log para diagnóstico.
#
# Retorna: 0 si exitoso, 1 si falló
start_services() {
    log_info "Levantando servicios desde: ${COMPOSE_FILE}"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Ejecutaría: podman-compose -f ${COMPOSE_FILE} up -d --force-recreate"
        SERVICES_STARTED=true
        return 0
    fi

    if ! podman-compose -f "$COMPOSE_FILE" up -d --force-recreate \
        >> "$LOG_FILE" 2>&1; then
        log_error "Fallo al iniciar servicios."
        log_error "Últimas 20 líneas del log de arranque:"
        tail -20 "$LOG_FILE" 2>/dev/null | while IFS= read -r line; do
            log_error "  ${line}"
        done
        return 1
    fi

    SERVICES_STARTED=true
    log_success "Servicios iniciados correctamente."
    return 0
}

# verify_critical_services
#
# Espera la inicialización inicial y luego verifica la salud de cada servicio
# en CRITICAL_SERVICES. Acumula los fallos para reportarlos todos juntos.
#
# Retorna: 0 si todos saludables, 1 si alguno falló
verify_critical_services() {
    local init_wait=5
    log_info "Esperando inicialización de contenedores (${init_wait}s)..."

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Verificaría salud de: ${CRITICAL_SERVICES[*]}"
        return 0
    fi

    sleep "$init_wait"

    local services_status=""
    services_status="$(podman-compose -f "$COMPOSE_FILE" ps 2>/dev/null || true)"

    local -a failed_services=()

    for service in "${CRITICAL_SERVICES[@]}"; do
        # Omitir servicios no presentes en el despliegue
        if ! printf '%s\n' "$services_status" | grep -q "$service"; then
            log_warn "Servicio '${service}' no encontrado en el despliegue. Omitiendo health check."
            continue
        fi

        if ! wait_for_service_health "$service"; then
            failed_services+=("$service")
        fi
    done

    if [[ "${#failed_services[@]}" -gt 0 ]]; then
        log_error "Servicios que fallaron health check: ${failed_services[*]}"
        return 1
    fi

    log_success "Todos los servicios críticos están saludables."
    return 0
}

# print_deployment_report
#
# Muestra un resumen del despliegue: versión, modo, imágenes construidas
# y estado actual de los contenedores.
print_deployment_report() {
    local separator="═══════════════════════════════════════════════"
    local thin_sep="───────────────────────────────────────────────"
    local mode
    mode="$( [[ "$DRY_RUN" == true ]] && printf 'DRY-RUN' || printf 'PRODUCCIÓN' )"

    log_info "$separator"
    log_info "          REPORTE DE DESPLIEGUE"
    log_info "$separator"
    log_info "Versión del script  : ${SCRIPT_VERSION}"
    log_info "Timestamp           : ${TIMESTAMP}"
    log_info "Modo                : ${mode}"
    log_info "Imágenes construidas: ${#IMAGES_BUILT[@]}"

    for img in "${IMAGES_BUILT[@]}"; do
        log_info "  ✓ ${img}"
    done

    log_info "$thin_sep"

    if [[ "$DRY_RUN" == false ]]; then
        log_info "Estado de los contenedores:"
        podman-compose -f "$COMPOSE_FILE" ps 2>/dev/null || \
            log_warn "No se pudo obtener el estado de los contenedores"
    fi

    log_info "$thin_sep"
    log_info "Log completo: ${LOG_FILE}"
    log_info "$separator"
}

# ====================================================
# SECCIÓN 11: PARSEO DE ARGUMENTOS
# ====================================================

# parse_arguments [$@]
#
# Parsea los argumentos de línea de comandos del script.
# Modifica las variables globales: DRY_RUN, SKIP_BUILD, SKIP_REGISTRY,
# HEALTH_TIMEOUT y LOG_LEVEL.
#
# Termina con exit 1 ante argumentos inválidos; exit 0 con --help.
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run)
                DRY_RUN=true
                # No usar log_info aquí: logging aún no inicializado
                printf 'INFO: Modo DRY-RUN activado: no se realizarán cambios.\n'
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-registry)
                SKIP_REGISTRY=true
                shift
                ;;
            --timeout)
                if [[ -z "${2:-}" ]] || ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
                    printf 'ERROR: --timeout requiere un entero positivo.\n' >&2
                    exit 1
                fi
                HEALTH_TIMEOUT="$2"
                shift 2
                ;;
            --log-level)
                if [[ -z "${2:-}" ]]; then
                    printf 'ERROR: --log-level requiere un valor: DEBUG, INFO, WARN, ERROR\n' >&2
                    exit 1
                fi
                local level_upper="${2^^}"
                if [[ -z "${LOG_LEVEL_MAP[$level_upper]+exists}" ]]; then
                    printf 'ERROR: Nivel de log inválido: '"'"'%s'"'"'. Válidos: DEBUG, INFO, WARN, ERROR\n' \
                        "$2" >&2
                    exit 1
                fi
                LOG_LEVEL="$level_upper"
                shift 2
                ;;
            --help|-h)
                # Extraer el bloque de documentación del encabezado del script
                sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" \
                    | grep '^#' \
                    | sed 's/^# \?//'
                exit 0
                ;;
            *)
                printf 'ERROR: Argumento desconocido: %s\n' "$1" >&2
                printf 'Usa --help para ver las opciones disponibles.\n' >&2
                exit 1
                ;;
        esac
    done
}

# ====================================================
# SECCIÓN 12: FUNCIÓN PRINCIPAL (ORQUESTACIÓN)
# ====================================================

# main [$@]
#
# Orquesta el proceso completo de despliegue en 12 pasos secuenciales:
#
#   1.  Parsear argumentos de línea de comandos
#   2.  Inicializar sistema de logging
#   3.  Verificar dependencias del sistema
#   4.  Configurar directorios con permisos seguros
#   5.  Rotar logs antiguos
#   6.  Configurar registros de Podman
#   7.  Validar archivo compose
#   8.  Limpiar despliegue previo
#   9.  Construir imágenes de contenedores
#   10. Iniciar servicios
#   11. Verificar salud de servicios críticos
#   12. Generar reporte de despliegue
#
# Códigos de salida:
#   0 - Despliegue exitoso
#   1 - Error en alguna etapa del despliegue
main() {
    # ── Paso 1: Parsear argumentos ──────────────────────────────────────────
    parse_arguments "$@"

    # ── Paso 2: Inicializar logging ─────────────────────────────────────────
    if ! init_logging; then
        printf 'FATAL: No se pudo inicializar el sistema de logging.\n' >&2
        exit 1
    fi

    log_info "══════════════════════════════════════════════════════════"
    log_info "  Iniciando Despliegue Seguro de APU Filter Ecosystem"
    log_info "  Versión: ${SCRIPT_VERSION} | PID: ${SCRIPT_PID}"
    log_info "══════════════════════════════════════════════════════════"
    log_info "Log file    : ${LOG_FILE}"
    log_info "Project root: ${PROJECT_ROOT}"

    # ── Paso 3: Verificar dependencias ─────────────────────────────────────
    check_dependencies || exit 1

    # ── Paso 4: Configurar directorios ──────────────────────────────────────
    setup_secure_directories || exit 1

    # ── Paso 5: Rotar logs antiguos ─────────────────────────────────────────
    rotate_logs

    # ── Paso 6: Configurar registros ────────────────────────────────────────
    if [[ "$SKIP_REGISTRY" == false ]]; then
        setup_registries || exit 1
    else
        log_info "Configuración de registros omitida (--skip-registry)."
    fi

    # ── Paso 7: Validar compose file ────────────────────────────────────────
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "Archivo compose no encontrado: ${COMPOSE_FILE}"
        log_error "Asegúrate de ejecutar desde la raíz del proyecto o que compose.yaml exista."
        exit 1
    fi
    log_debug "Archivo compose validado: ${COMPOSE_FILE}"

    # ── Paso 8: Limpieza previa ──────────────────────────────────────────────
    cleanup_previous_deployment

    # ── Paso 9: Construcción de imágenes ────────────────────────────────────
    if [[ "$SKIP_BUILD" == false ]]; then
        build_all_images || exit 1
    else
        log_info "Construcción de imágenes omitida (--skip-build)."
    fi

    # ── Paso 10: Inicio de servicios ────────────────────────────────────────
    start_services || exit 1

    # ── Paso 11: Health checks ──────────────────────────────────────────────
    verify_critical_services || exit 1

    # ── Paso 12: Reporte final ──────────────────────────────────────────────
    print_deployment_report

    log_success "══════════════════════════════════════════════════════════"
    log_success "  APU Filter Ecosystem: Operativo y Verificado ✓"
    log_success "══════════════════════════════════════════════════════════"
}

# ====================================================
# SECCIÓN 13: PUNTO DE ENTRADA
# ====================================================

# Ejecutar main solo si el script se invoca directamente
# (no cuando se importa con 'source' para testing)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi