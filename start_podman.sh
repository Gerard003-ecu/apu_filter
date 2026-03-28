#!/usr/bin/env bash

# ==============================================================================
# Script de Orquestación Segura para APU Filter Ecosystem
# Versión: 3.0.0 (Rediseño Riguroso)
#
# Descripción:
#   Construye imágenes, configura registros y despliega servicios del ecosistema
#   APU Filter usando Podman y podman-compose, con políticas de seguridad
#   estrictas, health checks robustos y mecanismo de rollback.
#
# Uso:
#   ./start.sh [opciones]
#
# Opciones:
#   --dry-run           Simula la ejecución sin realizar cambios
#   --skip-build        Omite la construcción de imágenes
#   --skip-registry     Omite la configuración de registros
#   --timeout <seg>     Timeout total para health checks (default: 150)
#   --log-level <nivel> Nivel de log: DEBUG, INFO, WARN, ERROR (default: INFO)
#   --help              Muestra esta ayuda
#
# Variables de entorno opcionales:
#   ALLOW_UNCONFINED_SECCOMP=true  Permite seccomp=unconfined como fallback
#   APU_LOG_RETENTION_DAYS=30      Días de retención de logs antiguos
#
# Autor: Equipo APU Filter
# Licencia: Propietaria
# ==============================================================================

# ==========================================================
# SECCIÓN 1: MODO ESTRICTO Y CONFIGURACIÓN INICIAL DE SHELL
# ==========================================================

# Activar modo estricto de Bash:
#   -e  : Salir inmediatamente si un comando falla
#   -u  : Tratar variables no definidas como error
#   -o pipefail : El código de retorno de un pipeline es el del último comando
#                 que falle (no el último ejecutado)
set -euo pipefail

# Separador de campos: solo newline y tab (evita problemas con espacios en nombres)
IFS=$'\n\t'

# Verificar versión mínima de Bash (4.0+ requerido para arrays asociativos)
if [[ "${BASH_VERSINFO[0]}" -lt 4 ]]; then
    echo "ERROR: Se requiere Bash 4.0 o superior. Versión actual: ${BASH_VERSION}" >&2
    exit 1
fi

# ====================================================
# SECCIÓN 2: CONSTANTES Y CONFIGURACIÓN GLOBAL
# ====================================================

# --- Metadatos del script ---
readonly SCRIPT_VERSION="3.0.0"
readonly SCRIPT_PID=$$
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# --- Detección de rutas ---
# Resuelve el directorio real del script (maneja symlinks)
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

# Detectar raíz del proyecto: busca compose.yaml ascendiendo desde SCRIPT_DIR
_detect_project_root() {
    local dir="$SCRIPT_DIR"
    local max_depth=5
    local depth=0

    while [[ "$depth" -lt "$max_depth" ]]; do
        if [[ -f "${dir}/compose.yaml" ]]; then
            echo "$dir"
            return 0
        fi
        # Evitar subir más allá de /
        local parent
        parent="$(cd "${dir}/.." && pwd -P)"
        if [[ "$parent" == "$dir" ]]; then
            break
        fi
        dir="$parent"
        ((depth++))
    done

    # Fallback: directorio del script
    echo "$SCRIPT_DIR"
    return 1
}

readonly PROJECT_ROOT="$(_detect_project_root)" || {
    echo "WARN: No se encontró compose.yaml. Usando directorio del script." >&2
}

# --- Rutas derivadas ---
readonly LOG_DIR="${PROJECT_ROOT}/logs"
readonly DATA_DIR="${PROJECT_ROOT}/data"
readonly COMPOSE_FILE="${PROJECT_ROOT}/compose.yaml"
readonly REGISTRY_SETUP_SCRIPT="${PROJECT_ROOT}/scripts/setup_podman_registry.sh"
readonly LOG_FILE="${LOG_DIR}/podman_start_${TIMESTAMP}.log"
readonly INFRASTRUCTURE_DIR="${PROJECT_ROOT}/infrastructure"

# --- Permisos (notación octal documentada) ---
# 750: propietario=rwx, grupo=r-x, otros=--- (ejecutable para cd en directorios)
readonly SAFE_DIR_PERMS="750"
# 640: propietario=rw-, grupo=r--, otros=--- (lectura para grupo, nada para otros)
readonly SAFE_FILE_PERMS="640"

# --- Health Check: configuración por defecto ---
readonly DEFAULT_HEALTH_TIMEOUT=150    # Timeout total en segundos
readonly HEALTH_INITIAL_INTERVAL=2     # Intervalo inicial entre reintentos (segundos)
readonly HEALTH_MAX_INTERVAL=15        # Intervalo máximo (backoff exponencial)
readonly HEALTH_BACKOFF_FACTOR=15      # Factor de backoff: 1.5 (representado como 15/10)

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

# --- Colores (con detección de terminal) ---
_init_colors() {
    # Desactivar colores si no hay terminal o si NO_COLOR está definido
    # Ref: https://no-color.org/
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

# --- Estado global mutable (para rollback) ---
declare -g SERVICES_STARTED=false
declare -g IMAGES_BUILT=()
declare -g LOG_INITIALIZED=false
declare -g DRY_RUN=false
declare -g SKIP_BUILD=false
declare -g SKIP_REGISTRY=false
declare -g HEALTH_TIMEOUT="$DEFAULT_HEALTH_TIMEOUT"
declare -g LOG_LEVEL="INFO"

# Mapa numérico de niveles de log para comparación eficiente
declare -gA LOG_LEVEL_MAP=(
    [DEBUG]=0
    [INFO]=1
    [WARN]=2
    [ERROR]=3
)

# ====================================================
# SECCIÓN 3: SISTEMA DE LOGGING ROBUSTO
# ====================================================

# Función interna de log con protección contra fallos circulares.
#
# Parámetros:
#   $1 - Nivel: DEBUG, INFO, SUCCESS, WARN, ERROR
#   $2 - Código de color ANSI
#   $3 - Mensaje a registrar
#
# Comportamiento:
#   - Escribe a consola con formato coloreado
#   - Escribe a archivo de log si está inicializado
#   - Redirige ERROR a stderr
#   - Respeta el nivel de log configurado
_log() {
    local level="$1"
    local color="$2"
    local message="$3"

    # Filtrar por nivel de log (SUCCESS se trata como INFO)
    local effective_level="${level}"
    [[ "$effective_level" == "SUCCESS" ]] && effective_level="INFO"
    local configured_num="${LOG_LEVEL_MAP[${LOG_LEVEL}]:-1}"
    local message_num="${LOG_LEVEL_MAP[${effective_level}]:-1}"
    if [[ "$message_num" -lt "$configured_num" ]]; then
        return 0
    fi

    local timestamp
    timestamp="$(date '+%H:%M:%S')"
    local full_timestamp
    full_timestamp="$(date '+%Y-%m-%d %H:%M:%S')"

    # Formato de consola
    local console_line
    console_line="$(printf "${color}[%-7s]${COLORS[RESET]} [%s] %s" \
        "$level" "$timestamp" "$message")"

    # Escribir a consola (ERROR va a stderr)
    if [[ "$level" == "ERROR" ]]; then
        echo -e "$console_line" >&2
    else
        echo -e "$console_line"
    fi

    # Escribir a archivo de log (sin colores, con PID para trazabilidad)
    if [[ "$LOG_INITIALIZED" == true ]] && [[ -f "$LOG_FILE" ]]; then
        printf "[%s] [%-7s] PID:%-6s %s\n" \
            "$full_timestamp" "$level" "$SCRIPT_PID" "$message" >> "$LOG_FILE" 2>/dev/null || true
    fi
}

log_debug()   { _log "DEBUG"   "${COLORS[CYAN]}"   "$1"; }
log_info()    { _log "INFO"    "${COLORS[BLUE]}"    "$1"; }
log_success() { _log "SUCCESS" "${COLORS[GREEN]}"   "$1"; }
log_warn()    { _log "WARN"    "${COLORS[YELLOW]}"  "$1"; }
log_error()   { _log "ERROR"   "${COLORS[RED]}"     "$1"; }

# Inicializa el sistema de logging de forma segura.
#
# Crea el directorio de logs, el archivo de log con permisos restrictivos
# y protege contra ataques de symlink.
#
# Retorna:
#   0 si la inicialización es exitosa
#   1 si falla la creación del directorio o archivo
init_logging() {
    # Crear directorio de logs
    if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
        echo "ERROR: No se pudo crear directorio de logs: $LOG_DIR" >&2
        return 1
    fi

    # Protección contra symlink attack: si el archivo existe como symlink, abortar
    if [[ -L "$LOG_FILE" ]]; then
        echo "ERROR: El archivo de log es un symlink (posible ataque): $LOG_FILE" >&2
        return 1
    fi

    # Crear archivo de log con permisos restrictivos (atómico con umask)
    (
        umask 0137  # Resultado: 640 (rw-r-----)
        : > "$LOG_FILE"
    )

    if [[ -f "$LOG_FILE" ]]; then
        LOG_INITIALIZED=true
        return 0
    else
        echo "ERROR: No se pudo crear archivo de log: $LOG_FILE" >&2
        return 1
    fi
}

# Rota logs antiguos eliminando archivos que excedan el período de retención.
#
# Usa la variable de entorno APU_LOG_RETENTION_DAYS o el valor por defecto.
rotate_logs() {
    local retention_days="${APU_LOG_RETENTION_DAYS:-$DEFAULT_LOG_RETENTION_DAYS}"

    # Validar que retention_days sea un entero positivo
    if ! [[ "$retention_days" =~ ^[1-9][0-9]*$ ]]; then
        log_warn "APU_LOG_RETENTION_DAYS='$retention_days' inválido. Usando default: $DEFAULT_LOG_RETENTION_DAYS"
        retention_days="$DEFAULT_LOG_RETENTION_DAYS"
    fi

    local count
    count=$(find "$LOG_DIR" -name "podman_start_*.log" -type f \
        -mtime +"$retention_days" 2>/dev/null | wc -l)

    if [[ "$count" -gt 0 ]]; then
        log_info "Rotando $count archivo(s) de log con más de $retention_days días..."
        find "$LOG_DIR" -name "podman_start_*.log" -type f \
            -mtime +"$retention_days" -delete 2>/dev/null || \
            log_warn "No se pudieron eliminar algunos logs antiguos"
    else
        log_debug "No hay logs antiguos para rotar (retención: ${retention_days} días)"
    fi
}

# ====================================================
# SECCIÓN 4: VALIDACIÓN DE DEPENDENCIAS
# ====================================================

# Verifica que todas las dependencias requeridas estén disponibles en el PATH.
#
# Dependencias verificadas:
#   - podman: Motor de contenedores
#   - podman-compose: Orquestador de servicios
#   - find: Utilidad para rotación de logs
#   - date: Timestamps
#
# Retorna:
#   0 si todas las dependencias están presentes
#   1 si falta alguna dependencia
check_dependencies() {
    local -a required_deps=("podman" "podman-compose" "find" "date")
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

    # Verificar versión mínima de podman (4.0+)
    local podman_version
    podman_version="$(podman --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' | head -1)"
    if [[ -n "$podman_version" ]]; then
        local major_version
        major_version="${podman_version%%.*}"
        if [[ "$major_version" -lt 4 ]]; then
            log_warn "Podman ${podman_version} detectado. Se recomienda 4.0+."
        else
            log_debug "Podman ${podman_version} detectado."
        fi
    fi

    log_success "Todas las dependencias verificadas correctamente."
    return 0
}

# ====================================================
# SECCIÓN 5: CONFIGURACIÓN SEGURA DE DIRECTORIOS
# ====================================================

# Configura directorios del proyecto con permisos seguros.
#
# Operaciones:
#   1. Crea LOG_DIR y DATA_DIR si no existen
#   2. Aplica permisos restrictivos (750 para directorios)
#   3. Detecta y reporta anomalías de permisos
#
# Nota: No ejecuta chown para evitar requerir privilegios de root.
#       El directorio hereda la propiedad del usuario que lo crea.
setup_secure_directories() {
    log_info "Configurando directorios con permisos seguros..."

    local -a target_dirs=("$LOG_DIR" "$DATA_DIR")

    for dir in "${target_dirs[@]}"; do
        # Crear directorio si no existe
        if [[ ! -d "$dir" ]]; then
            if [[ "$DRY_RUN" == true ]]; then
                log_info "[DRY-RUN] Crearía directorio: $dir"
                continue
            fi
            mkdir -p "$dir" || {
                log_error "No se pudo crear directorio: $dir"
                return 1
            }
            log_debug "Directorio creado: $dir"
        fi

        if [[ "$DRY_RUN" == true ]]; then
            log_info "[DRY-RUN] Aplicaría permisos $SAFE_DIR_PERMS a: $dir"
            continue
        fi

        # Aplicar permisos seguros
        chmod "$SAFE_DIR_PERMS" "$dir" 2>/dev/null || {
            log_warn "No se pudieron establecer permisos $SAFE_DIR_PERMS en: $dir"
        }

        # Verificación: reportar si los permisos actuales son excesivos
        local actual_perms
        actual_perms="$(stat -c '%a' "$dir" 2>/dev/null || stat -f '%Lp' "$dir" 2>/dev/null)"
        if [[ -n "$actual_perms" ]]; then
            # Verificar que el componente "others" no tenga escritura
            local others_perm="${actual_perms: -1}"
            if [[ "$others_perm" -ge 2 ]]; then
                log_warn "Directorio $dir tiene permisos de escritura para 'others' ($actual_perms)"
            fi
        fi
    done

    log_success "Directorios configurados con permisos seguros ($SAFE_DIR_PERMS)."
}

# ====================================================
# SECCIÓN 6: GESTIÓN DE SEÑALES Y LIMPIEZA
# ====================================================

# Manejador de limpieza ejecutado al salir del script.
#
# Comportamiento según código de salida:
#   - 0: Log de finalización exitosa
#   - != 0: Log de error + intento de rollback si hay servicios iniciados
#
# Parámetros:
#   Ninguno (usa $? implícitamente)
cleanup_on_exit() {
    local exit_code=$?

    # Deshabilitar trap recursivo
    trap '' EXIT INT TERM

    if [[ "$exit_code" -ne 0 ]]; then
        # Usar echo directo si el logging no está inicializado
        if [[ "$LOG_INITIALIZED" == true ]]; then
            log_error "Script terminado con código de error: $exit_code"

            # Intentar rollback si se iniciaron servicios
            if [[ "$SERVICES_STARTED" == true ]]; then
                log_warn "Ejecutando rollback: deteniendo servicios iniciados..."
                podman-compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
                log_info "Rollback completado."
            fi

            log_info "Revisar log para detalles: $LOG_FILE"
        else
            echo "ERROR: Script terminado con código: $exit_code" >&2
            echo "El sistema de logging no se inicializó." >&2
        fi
    else
        if [[ "$LOG_INITIALIZED" == true ]]; then
            log_info "Script finalizado exitosamente."
        fi
    fi
}

# Registrar traps DESPUÉS de definir la función
trap 'cleanup_on_exit' EXIT
trap 'exit 130' INT     # 128 + 2 (SIGINT)
trap 'exit 143' TERM    # 128 + 15 (SIGTERM)

# ====================================================
# SECCIÓN 7: HEALTH CHECKS CON BACKOFF EXPONENCIAL
# ====================================================

# Espera a que un servicio alcance el estado saludable.
#
# Implementa backoff exponencial para reducir la carga en el sistema
# durante la espera, con un intervalo máximo configurable.
#
# Fórmula de backoff:
#   intervalo_n = min(intervalo_{n-1} * FACTOR, MAX_INTERVAL)
#   donde FACTOR = HEALTH_BACKOFF_FACTOR / 10 (aritmética entera)
#
# Parámetros:
#   $1 - Nombre del servicio a verificar
#   $2 - Timeout total en segundos (opcional, default: HEALTH_TIMEOUT)
#
# Retorna:
#   0 si el servicio está saludable
#   1 si el timeout se agotó
wait_for_service_health() {
    local service_name="$1"
    local timeout="${2:-$HEALTH_TIMEOUT}"

    # Validar timeout
    if ! [[ "$timeout" =~ ^[1-9][0-9]*$ ]]; then
        log_error "Timeout inválido: '$timeout'. Debe ser un entero positivo."
        return 1
    fi

    log_info "Verificando salud de '$service_name' (timeout: ${timeout}s)..."

    local elapsed=0
    local interval="$HEALTH_INITIAL_INTERVAL"
    local attempt=1

    while [[ "$elapsed" -lt "$timeout" ]]; do
        # Obtener estado del servicio
        local service_status
        service_status="$(podman-compose -f "$COMPOSE_FILE" ps 2>/dev/null || echo "")"

        if [[ -z "$service_status" ]]; then
            log_warn "No se pudo obtener estado de servicios (intento $attempt)"
        else
            # Verificar si está saludable
            if echo "$service_status" | grep -qE "${service_name}.*(healthy|running)"; then
                log_success "Servicio '$service_name' está saludable (${elapsed}s, $attempt intentos)."
                return 0
            fi

            # Verificar si el servicio está en estado de error
            if echo "$service_status" | grep -qE "${service_name}.*(exited|dead|error)"; then
                log_error "Servicio '$service_name' está en estado de error."
                log_error "Logs del servicio:"
                podman-compose -f "$COMPOSE_FILE" logs --tail=20 "$service_name" 2>/dev/null | \
                    while IFS= read -r line; do log_error "  $line"; done || true
                return 1
            fi
        fi

        log_debug "Esperando '$service_name'... (${elapsed}s/${timeout}s, intento $attempt, próximo en ${interval}s)"
        sleep "$interval"
        elapsed=$((elapsed + interval))
        ((attempt++))

        # Backoff exponencial con aritmética entera
        # interval = interval * 15 / 10 (equivale a *1.5)
        interval=$(( (interval * HEALTH_BACKOFF_FACTOR) / 10 ))
        # Limitar al máximo
        if [[ "$interval" -gt "$HEALTH_MAX_INTERVAL" ]]; then
            interval="$HEALTH_MAX_INTERVAL"
        fi
    done

    log_error "Timeout agotado esperando servicio '$service_name' (${timeout}s, $attempt intentos)."
    log_error "Últimos logs del servicio:"
    podman-compose -f "$COMPOSE_FILE" logs --tail=30 "$service_name" 2>/dev/null | \
        while IFS= read -r line; do log_error "  $line"; done || true
    return 1
}

# ====================================================
# SECCIÓN 8: CONSTRUCCIÓN DE IMÁGENES CON SEGURIDAD
# ====================================================

# Construye una imagen de contenedor con políticas de seguridad escalonadas.
#
# Estrategia de construcción (fallback escalonado):
#   1. Construcción estándar con seguridad por defecto
#   2. Si falla y ALLOW_UNCONFINED_SECCOMP=true: reintenta con seccomp=unconfined
#   3. Si todo falla: error fatal
#
# Parámetros:
#   $1 - Ruta al Dockerfile (relativa a PROJECT_ROOT o absoluta)
#   $2 - Tag de la imagen (formato: nombre:tag)
#   $3 - Contexto de construcción (opcional, default: PROJECT_ROOT)
#
# Retorna:
#   0 si la imagen se construyó exitosamente
#   1 si la construcción falló
build_image_with_security_check() {
    local dockerfile_path="$1"
    local image_tag="$2"
    local context_path="${3:-$PROJECT_ROOT}"

    # Validar existencia del Dockerfile
    if [[ ! -f "$dockerfile_path" ]]; then
        log_error "Dockerfile no encontrado: $dockerfile_path"
        return 1
    fi

    # Validar formato del tag
    if ! [[ "$image_tag" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*(:[a-zA-Z0-9._-]+)?$ ]]; then
        log_error "Tag de imagen inválido: '$image_tag'"
        return 1
    fi

    log_info "Construyendo imagen: $image_tag (Dockerfile: $dockerfile_path)"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Ejecutaría: podman build -f $dockerfile_path -t $image_tag $context_path"
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
        log_success "Imagen '$image_tag' construida con seguridad por defecto."
        IMAGES_BUILT+=("$image_tag")
        return 0
    fi

    log_warn "Construcción estándar fallida para '$image_tag'."

    # Intento 2: Con seccomp=unconfined (solo si explícitamente permitido)
    # Sanitizar variable de entorno
    local allow_unconfined="${ALLOW_UNCONFINED_SECCOMP:-false}"
    allow_unconfined="${allow_unconfined,,}"  # Convertir a minúsculas

    if [[ "$allow_unconfined" == "true" ]]; then
        log_warn "Reintentando con --security-opt seccomp=unconfined (RIESGO DE SEGURIDAD)"
        log_warn "Esto desactiva el filtrado de syscalls durante la construcción."

        if podman build \
            --security-opt seccomp=unconfined \
            --pull=newer \
            -f "$dockerfile_path" \
            -t "$image_tag" \
            "$context_path" >> "$LOG_FILE" 2>&1; then
            log_warn "Imagen '$image_tag' construida con seccomp=unconfined."
            log_warn "ACCIÓN REQUERIDA: Investigar por qué la construcción estándar falla."
            IMAGES_BUILT+=("$image_tag")
            return 0
        fi
    else
        log_error "Establece ALLOW_UNCONFINED_SECCOMP=true si es absolutamente necesario."
    fi

    log_error "Falló la construcción de '$image_tag' en todos los intentos."
    log_error "Consulta el log para detalles: $LOG_FILE"
    return 1
}

# Construye todas las imágenes definidas en IMAGE_DEFINITIONS.
#
# Itera sobre el array de definiciones y construye cada imagen.
# Si alguna falla, detiene el proceso completo (fail-fast).
#
# Retorna:
#   0 si todas las imágenes se construyeron
#   1 si alguna imagen falló
build_all_images() {
    log_info "Iniciando construcción de ${#IMAGE_DEFINITIONS[@]} imagen(es)..."

    local build_count=0
    for definition in "${IMAGE_DEFINITIONS[@]}"; do
        # Parsear definición: "tag|dockerfile_relativo"
        local image_tag="${definition%%|*}"
        local dockerfile_rel="${definition##*|}"
        local dockerfile_path="${PROJECT_ROOT}/${dockerfile_rel}"

        if ! build_image_with_security_check "$dockerfile_path" "$image_tag" "$PROJECT_ROOT"; then
            log_error "Fallo en imagen ${build_count}/${#IMAGE_DEFINITIONS[@]}: $image_tag"
            return 1
        fi

        ((build_count++))
        log_info "Progreso de construcción: ${build_count}/${#IMAGE_DEFINITIONS[@]}"
    done

    log_success "Todas las imágenes (${build_count}) construidas correctamente."
    return 0
}

# ====================================================
# SECCIÓN 9: CONFIGURACIÓN DE REGISTROS
# ====================================================

# Configura los registros de contenedores de Podman.
#
# Ejecuta el script de configuración si existe y es válido.
# Valida que el script tenga permisos adecuados antes de ejecutarlo.
#
# Retorna:
#   0 si la configuración se aplicó o no era necesaria
#   1 si la configuración falló
setup_registries() {
    log_info "Verificando configuración de registros de Podman..."

    if [[ ! -f "$REGISTRY_SETUP_SCRIPT" ]]; then
        log_warn "Script de registros no encontrado: $REGISTRY_SETUP_SCRIPT"
        log_warn "Asumiendo configuración manual de registros."
        return 0
    fi

    # Verificaciones de seguridad del script
    if [[ -L "$REGISTRY_SETUP_SCRIPT" ]]; then
        log_error "El script de registros es un symlink (posible riesgo de seguridad)."
        return 1
    fi

    if [[ ! -r "$REGISTRY_SETUP_SCRIPT" ]]; then
        log_error "Sin permisos de lectura en: $REGISTRY_SETUP_SCRIPT"
        return 1
    fi

    # Verificar que sea un script de shell válido (primera línea)
    local first_line
    first_line="$(head -1 "$REGISTRY_SETUP_SCRIPT" 2>/dev/null)"
    if [[ "$first_line" != "#!/"* ]]; then
        log_warn "El script de registros no tiene shebang estándar."
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Ejecutaría: bash $REGISTRY_SETUP_SCRIPT"
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

# Detiene y limpia contenedores y redes previas.
#
# Ejecuta podman-compose down con limpieza de huérfanos y volúmenes.
# Los fallos en esta etapa no son fatales (se ignoran).
cleanup_previous_deployment() {
    log_info "Limpiando contenedores y redes de despliegues previos..."

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Ejecutaría: podman-compose -f $COMPOSE_FILE down --remove-orphans --volumes"
        return 0
    fi

    podman-compose -f "$COMPOSE_FILE" down --remove-orphans --volumes >> "$LOG_FILE" 2>&1 || {
        log_warn "La limpieza previa generó advertencias (puede ser normal en primera ejecución)."
    }

    log_debug "Limpieza previa completada."
}

# Inicia los servicios definidos en el archivo compose.
#
# Retorna:
#   0 si los servicios se iniciaron correctamente
#   1 si el inicio falló
start_services() {
    log_info "Levantando servicios desde: $COMPOSE_FILE"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Ejecutaría: podman-compose -f $COMPOSE_FILE up -d --force-recreate"
        SERVICES_STARTED=true
        return 0
    fi

    if ! podman-compose -f "$COMPOSE_FILE" up -d --force-recreate >> "$LOG_FILE" 2>&1; then
        log_error "Fallo al iniciar servicios."
        log_error "Últimas líneas del log:"
        tail -20 "$LOG_FILE" 2>/dev/null | while IFS= read -r line; do
            log_error "  $line"
        done
        return 1
    fi

    SERVICES_STARTED=true
    log_success "Servicios iniciados correctamente."
    return 0
}

# Verifica la salud de todos los servicios críticos.
#
# Espera un período inicial para permitir la inicialización,
# luego verifica cada servicio definido en CRITICAL_SERVICES.
#
# Retorna:
#   0 si todos los servicios están saludables
#   1 si algún servicio falló el health check
verify_critical_services() {
    local init_wait=5
    log_info "Esperando inicialización de contenedores (${init_wait}s)..."

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Verificaría salud de: ${CRITICAL_SERVICES[*]}"
        return 0
    fi

    sleep "$init_wait"

    local services_status
    services_status="$(podman-compose -f "$COMPOSE_FILE" ps 2>/dev/null || echo "")"

    local failed_services=()

    for service in "${CRITICAL_SERVICES[@]}"; do
        # Verificar si el servicio existe en el compose
        if ! echo "$services_status" | grep -q "$service"; then
            log_warn "Servicio '$service' no encontrado en el despliegue. Omitiendo health check."
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

# Muestra el estado final de todos los servicios desplegados.
print_deployment_report() {
    log_info "═══════════════════════════════════════════════"
    log_info "          REPORTE DE DESPLIEGUE"
    log_info "═══════════════════════════════════════════════"
    log_info "Versión del script : $SCRIPT_VERSION"
    log_info "Timestamp          : $TIMESTAMP"
    log_info "Modo               : $(if [[ "$DRY_RUN" == true ]]; then echo 'DRY-RUN'; else echo 'PRODUCCIÓN'; fi)"
    log_info "Imágenes construidas: ${#IMAGES_BUILT[@]}"

    for img in "${IMAGES_BUILT[@]}"; do
        log_info "  ✓ $img"
    done

    log_info "───────────────────────────────────────────────"

    if [[ "$DRY_RUN" == false ]]; then
        log_info "Estado de los contenedores:"
        podman-compose -f "$COMPOSE_FILE" ps 2>/dev/null || \
            log_warn "No se pudo obtener el estado de los contenedores"
    fi

    log_info "───────────────────────────────────────────────"
    log_info "Log completo: $LOG_FILE"
    log_info "═══════════════════════════════════════════════"
}

# ====================================================
# SECCIÓN 11: PARSEO DE ARGUMENTOS
# ====================================================

# Parsea los argumentos de línea de comandos.
#
# Parámetros:
#   $@ - Argumentos del script
#
# Modifica variables globales:
#   DRY_RUN, SKIP_BUILD, SKIP_REGISTRY, HEALTH_TIMEOUT, LOG_LEVEL
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run)
                DRY_RUN=true
                log_info "Modo DRY-RUN activado: no se realizarán cambios."
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
                    log_error "El argumento --timeout requiere un entero positivo."
                    exit 1
                fi
                HEALTH_TIMEOUT="$2"
                shift 2
                ;;
            --log-level)
                if [[ -z "${2:-}" ]]; then
                    log_error "El argumento --log-level requiere un valor: DEBUG, INFO, WARN, ERROR"
                    exit 1
                fi
                local level_upper="${2^^}"
                if [[ -z "${LOG_LEVEL_MAP[$level_upper]+exists}" ]]; then
                    log_error "Nivel de log inválido: '$2'. Valores válidos: DEBUG, INFO, WARN, ERROR"
                    exit 1
                fi
                LOG_LEVEL="$level_upper"
                shift 2
                ;;
            --help|-h)
                # Extraer y mostrar la documentación del encabezado
                grep '^#' "${BASH_SOURCE[0]}" | head -25 | sed 's/^# \?//'
                exit 0
                ;;
            *)
                log_error "Argumento desconocido: $1"
                log_error "Usa --help para ver las opciones disponibles."
                exit 1
                ;;
        esac
    done
}

# ====================================================
# SECCIÓN 12: FUNCIÓN PRINCIPAL (ORQUESTACIÓN)
# ====================================================

# Función principal que orquesta todo el proceso de despliegue.
#
# Flujo de ejecución:
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
# Parámetros:
#   $@ - Argumentos del script (pasados desde la invocación)
#
# Códigos de salida:
#   0 - Despliegue exitoso
#   1 - Error en alguna etapa del despliegue
main() {
    # ── Paso 1: Parsear argumentos ──
    parse_arguments "$@"

    # ── Paso 2: Inicializar logging ──
    if ! init_logging; then
        echo "FATAL: No se pudo inicializar el sistema de logging." >&2
        exit 1
    fi

    log_info "══════════════════════════════════════════════════════════"
    log_info "  Iniciando Despliegue Seguro de APU Filter Ecosystem"
    log_info "  Versión: $SCRIPT_VERSION | PID: $SCRIPT_PID"
    log_info "══════════════════════════════════════════════════════════"
    log_info "Log file: $LOG_FILE"
    log_info "Project root: $PROJECT_ROOT"

    # ── Paso 3: Verificar dependencias ──
    if ! check_dependencies; then
        exit 1
    fi

    # ── Paso 4: Configurar directorios ──
    if ! setup_secure_directories; then
        exit 1
    fi

    # ── Paso 5: Rotar logs antiguos ──
    rotate_logs

    # ── Paso 6: Configurar registros ──
    if [[ "$SKIP_REGISTRY" == false ]]; then
        if ! setup_registries; then
            exit 1
        fi
    else
        log_info "Configuración de registros omitida (--skip-registry)."
    fi

    # ── Paso 7: Validar compose file ──
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "Archivo compose no encontrado: $COMPOSE_FILE"
        log_error "Asegúrate de ejecutar desde la raíz del proyecto o que compose.yaml exista."
        exit 1
    fi
    log_debug "Archivo compose validado: $COMPOSE_FILE"

    # ── Paso 8: Limpieza previa ──
    cleanup_previous_deployment

    # ── Paso 9: Construcción de imágenes ──
    if [[ "$SKIP_BUILD" == false ]]; then
        if ! build_all_images; then
            exit 1
        fi
    else
        log_info "Construcción de imágenes omitida (--skip-build)."
    fi

    # ── Paso 10: Inicio de servicios ──
    if ! start_services; then
        exit 1
    fi

    # ── Paso 11: Health checks ──
    if ! verify_critical_services; then
        exit 1
    fi

    # ── Paso 12: Reporte final ──
    print_deployment_report

    log_success "══════════════════════════════════════════════════════════"
    log_success "  APU Filter Ecosystem: Operativo y Verificado ✓"
    log_success "══════════════════════════════════════════════════════════"
}

# ====================================================
# SECCIÓN 13: PUNTO DE ENTRADA
# ====================================================

# Ejecutar main solo si el script se invoca directamente
# (no cuando se importa con 'source')
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi