#!/usr/bin/env bash

# ==============================================================================
# Script de Orquestación Segura para APU Filter Ecosystem
# Versión: 2.3.0 (Seguridad Mejorada & Optimización)
# ==============================================================================

# --- Strict Mode & Signal Handling ---
set -euo pipefail
IFS=$'\n\t'

# Capturar señales para limpieza segura
trap 'cleanup_on_exit' EXIT INT TERM

# --- Configuration ---
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly LOG_DIR="${PROJECT_ROOT}/logs"
readonly DATA_DIR="${PROJECT_ROOT}/data"
readonly COMPOSE_FILE="${PROJECT_ROOT}/compose.yaml"
readonly REGISTRY_SETUP_SCRIPT="${PROJECT_ROOT}/scripts/setup_podman_registry.sh"
readonly SCRIPT_PID=$$
readonly TIMESTAMP=$(date +%Y%m%d_%H%M%S)
readonly LOG_FILE="${LOG_DIR}/podman_start_${TIMESTAMP}.log"

# Configuración de permisos más seguros
readonly SAFE_DATA_PERMS="755"
readonly SAFE_LOG_PERMS="755"

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

# --- Security & Permissions ---
setup_secure_directories() {
    log_info "Configurando directorios con permisos seguros..."
    
    # Crear directorios si no existen
    mkdir -p "$LOG_DIR" "$DATA_DIR"
    
    # Establecer propietario y grupo (usando usuario actual)
    local current_user=$(id -u)
    local current_group=$(id -g)
    
    chown -R "$current_user:$current_group" "$LOG_DIR" "$DATA_DIR"
    
    # Establecer permisos seguros
    chmod "$SAFE_LOG_PERMS" "$LOG_DIR"
    chmod "$SAFE_DATA_PERMS" "$DATA_DIR"
    
    log_success "Directorios configurados con permisos seguros (logs: $SAFE_LOG_PERMS, data: $SAFE_DATA_PERMS)"
}

# --- Cleanup ---
cleanup_on_exit() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Script terminado con código de error: $exit_code"
        log_info "Revisar log: $LOG_FILE"
    else
        log_info "Script finalizado exitosamente"
    fi
}

# --- Health Checks ---
wait_for_service_health() {
    local service_name="$1"
    local max_attempts=30
    local attempt=1
    local healthy=false
    
    log_info "Verificando salud del servicio: $service_name..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if podman-compose -f "$COMPOSE_FILE" ps | grep -q "${service_name}.*Up.*healthy"; then
            log_success "Servicio $service_name está saludable"
            healthy=true
            break
        fi
        
        log_info "Esperando salud del servicio $service_name... (intento $attempt/$max_attempts)"
        sleep 5
        ((attempt++))
    done
    
    if [[ "$healthy" == false ]]; then
        log_error "Tiempo de espera agotado para el servicio $service_name"
        return 1
    fi
}

# --- Image Building with Security Policy ---
build_image_with_security_check() {
    local dockerfile_path="$1"
    local image_tag="$2"
    local context_path="${3:-.}"
    
    log_info "Construyendo imagen: $image_tag"
    
    # Primero intentar sin deshabilitar seccomp
    if podman build \
        -f "$dockerfile_path" \
        -t "$image_tag" \
        "$context_path" >> "$LOG_FILE" 2>&1; then
        log_success "Imagen $image_tag construida con seguridad por defecto"
        return 0
    fi
    
    # Si falla, registrar advertencia y permitir opción con seccomp modificado
    log_warn "Construcción estándar fallida para $image_tag, intentando con ajustes de seguridad..."
    
    # Opción configurable para seguridad reducida (solo si necesario)
    if [[ "${ALLOW_UNCONFINED_SECCOMP:-false}" == "true" ]]; then
        log_warn "Usando --security-opt seccomp=unconfined para $image_tag (RIESGO DE SEGURIDAD)"
        if podman build \
            --security-opt seccomp=unconfined \
            -f "$dockerfile_path" \
            -t "$image_tag" \
            "$context_path" >> "$LOG_FILE" 2>&1; then
            log_warn "Imagen $image_tag construida con seccomp=unconfined (requiere revisión de seguridad)"
            return 0
        fi
    else
        log_error "Construcción fallida para $image_tag. Establece ALLOW_UNCONFINED_SECCOMP=true si es absolutamente necesario."
        return 1
    fi
    
    log_error "Fallo en la construcción de $image_tag después de todos los intentos"
    return 1
}

# --- Main Logic ---
main() {
    # 1. Verificar dependencias
    check_dependencies
    log_info "Dependencias verificadas correctamente"
    
    # 2. Crear log directory y archivo
    mkdir -p "$LOG_DIR"
    touch "$LOG_FILE"
    
    log_info "=== Iniciando Despliegue Seguro de APU Filter Ecosystem ==="
    log_info "Log file: $LOG_FILE"
    log_info "Timestamp: $TIMESTAMP"
    
    # 3. Configurar directorios con permisos seguros
    setup_secure_directories
    
    # 4. Configurar Registries de Podman
    log_info "Verificando configuración de registros..."
    if [[ -f "$REGISTRY_SETUP_SCRIPT" ]]; then
        log_info "Ejecutando setup_podman_registry.sh..."
        if bash "$REGISTRY_SETUP_SCRIPT" >> "$LOG_FILE" 2>&1; then
            log_success "Configuración de registros aplicada."
        else
            log_error "Fallo en la configuración de registros"
            exit 1
        fi
    else
        log_warn "No se encontró $REGISTRY_SETUP_SCRIPT. Asumiendo configuración manual."
    fi

    # 5. Validar archivo compose
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "No se encuentra el archivo: $COMPOSE_FILE"
        log_error "Asegúrate de ejecutar este script desde la raíz del proyecto."
        exit 1
    fi

    # 6. Limpieza previa
    log_info "Limpiando contenedores y redes previas..."
    podman-compose -f "$COMPOSE_FILE" down --remove-orphans --volumes >> "$LOG_FILE" 2>&1 || true

    # 7. Construcción de imágenes con políticas de seguridad
    log_info "Construyendo imágenes con políticas de seguridad..."
    
    # Construir Core
    if ! build_image_with_security_check \
        "${PROJECT_ROOT}/infrastructure/Dockerfile.core" \
        "apu-core:latest" \
        "$PROJECT_ROOT"; then
        log_error "Fallo en la construcción de Core. Revisa el log."
        exit 1
    fi
    log_success "Imagen apu-core:latest construida correctamente."

    # Construir Agent
    if ! build_image_with_security_check \
        "${PROJECT_ROOT}/infrastructure/Dockerfile.agent" \
        "apu-agent:latest" \
        "$PROJECT_ROOT"; then
        log_error "Fallo en la construcción de Agent. Revisa el log."
        exit 1
    fi
    log_success "Imagen apu-agent:latest construida correctamente."

    # 8. Inicio de servicios (Orquestación)
    log_info "Levantando servicios..."
    if ! podman-compose -f "$COMPOSE_FILE" up -d --force-recreate >> "$LOG_FILE" 2>&1; then
        log_error "Fallo al iniciar servicios."
        exit 1
    fi

    # 9. Verificación de salud de servicios críticos
    log_info "Esperando inicialización de contenedores (5s)..."
    sleep 5
    
    # Verificar servicios específicos si están definidos en el compose
    declare -a critical_services=("apu-core" "apu-agent")
    for service in "${critical_services[@]}"; do
        if podman-compose -f "$COMPOSE_FILE" ps | grep -q "$service"; then
            wait_for_service_health "$service" || {
                log_error "Servicio crítico $service no alcanzó estado saludable"
                exit 1
            }
        fi
    done

    # 10. Reporte final
    log_info "Estado de los contenedores:"
    podman-compose -f "$COMPOSE_FILE" ps
    
    log_success "=== APU Filter Ecosystem Operativo y Verificado ==="
    log_info "Log completo disponible en: $LOG_FILE"
}

# Ejecutar main si no estamos siendo sourceados
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi