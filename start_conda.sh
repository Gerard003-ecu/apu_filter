#!/usr/bin/env bash

# ==============================================================================
# Script de Inicialización de Entorno Local (Conda + UV)
# ==============================================================================
# Versión: 3.0.0
# Coherencia: alineado con Dockerfile.core v7.0 y Dockerfile.agent v7.0
# Cambios vs 2.1.0:
#   - Fix: --verbose ahora activa log_debug (variable unificada)
#   - Fix: lock file usa nombre fijo (protección real contra concurrencia)
#   - Fix: PyTorch versionado idéntico al Docker stack (2.5.1+cpu)
#   - Fix: trap único (elimina doble ejecución de cleanup)
#   - Fix: run_pip() reemplaza get_pip_command() (elimina word-splitting)
#   - Fix: ABSOLUTE_LOG_DIR robusto ante directorios inexistentes
#   - Fix: env_exists usa awk+grep -xF (sin regex injection)
#   - Fix: ANSI solo en terminal interactiva (show_help, display_final)
#   - Add: constraints.txt local (coherencia con Docker stack)
#   - Add: verificación de download.pytorch.org
#   - Fix: check_base_dependencies no verifica python pre-activación
#   - Fix: paquetes de visualización usan run_pip
# ==============================================================================

# --- Strict Mode ---
set -euo pipefail
IFS=$'\n\t'

# --- Script Metadata ---
readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly SCRIPT_VERSION="3.0.0"
readonly SCRIPT_PID=$$
readonly TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# --- Default Configuration (overridable via environment) ---
: "${ENV_NAME:=apu_filter_env}"
: "${PYTHON_VERSION:=3.10}"
: "${LOG_DIR:=./logs}"
: "${PYTORCH_INDEX_URL:=https://download.pytorch.org/whl/cpu}"
: "${VERBOSE:=false}"

# ─── Versiones centralizadas (idénticas a Dockerfiles v7.0) ─────────────────
: "${TORCH_VERSION:=2.5.1}"
: "${TORCHVISION_VERSION:=0.20.1}"
: "${TORCHAUDIO_VERSION:=2.5.1}"
: "${SYMPY_VERSION:=1.13.1}"

# --- Resolver paths absolutos (robusto ante directorios inexistentes) ---
case "$LOG_DIR" in
    /*) readonly ABSOLUTE_LOG_DIR="$LOG_DIR" ;;
    *)  readonly ABSOLUTE_LOG_DIR="${PROJECT_ROOT}/${LOG_DIR#./}" ;;
esac

readonly REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements.txt"
readonly REQUIREMENTS_DEV_FILE="${PROJECT_ROOT}/requirements-dev.txt"
readonly CONSTRAINTS_FILE="${PROJECT_ROOT}/constraints-local.txt"
readonly PROCESSED_DATA_FILE="${PROJECT_ROOT}/data/processed_apus.json"
readonly EMBEDDINGS_SCRIPT="${PROJECT_ROOT}/scripts/generate_embeddings.py"

# --- Lock Management (nombre fijo = protección real contra concurrencia) ---
readonly LOCK_FILE="/tmp/${SCRIPT_NAME%.*}.lock"

# --- Runtime State ---
LOG_FILE=""
LOCK_FD=""

declare -A RUNTIME_STATE=(
    [CONDA_INITIALIZED]="false"
    [ENV_CREATED]="false"
    [ENV_ACTIVATED]="false"
)

# --- Operation Flags ---
declare -A OPERATION_FLAGS=(
    [SKIP_PYTORCH]="false"
    [SKIP_EMBEDDINGS]="false"
    [SKIP_DEV_DEPS]="false"
    [FORCE_RECREATE]="false"
    [CLEAN_ENV]="false"
    [UPDATE_ONLY]="false"
    [DRY_RUN]="false"
)

# --- Terminal Colors ---
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

    # Terminal output (con colores solo si es interactivo)
    if is_terminal; then
        printf "${color}[%-7s]${COLORS[RESET]} [%s] %s\n" \
            "$level" "$(date '+%H:%M:%S')" "$message"
    else
        printf "[%-7s] [%s] %s\n" \
            "$level" "$(date '+%H:%M:%S')" "$message"
    fi

    # File output (sin colores, con timestamp completo)
    if [[ -n "${LOG_FILE:-}" ]] && [[ -d "$(dirname "$LOG_FILE" 2>/dev/null)" ]]; then
        printf "[%s] [%-7s] PID:%s %s\n" \
            "$timestamp" "$level" "$SCRIPT_PID" "$message" \
            >> "$LOG_FILE" 2>/dev/null || true
    fi
}

log_info()    { _log "INFO"    "${COLORS[BLUE]}"    "$1"; }
log_step()    { _log "STEP"    "${COLORS[CYAN]}"    "→ $1"; }
log_success() { _log "SUCCESS" "${COLORS[GREEN]}"   "✓ $1"; }
log_warn()    { _log "WARN"    "${COLORS[YELLOW]}"  "⚠ $1"; }
log_error()   { _log "ERROR"   "${COLORS[RED]}"     "✗ $1" >&2; }

log_debug() {
    # Variable unificada: --verbose y VERBOSE=true convergen aquí
    [[ "$VERBOSE" == "true" ]] && _log "DEBUG" "${COLORS[MAGENTA]}" "$1" || true
}

log_separator() {
    local char="${1:-=}"
    local msg="${2:-}"
    local line
    line=$(printf '%*s' 60 '' | tr ' ' "$char")
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

cleanup_on_exit() {
    local exit_code=$?

    # Entorno creado pero instalación incompleta → avisar
    if [[ $exit_code -ne 0 ]] \
        && [[ "${RUNTIME_STATE[ENV_CREATED]}" == "true" ]] \
        && [[ "${RUNTIME_STATE[ENV_ACTIVATED]}" != "completed" ]]; then
        log_warn "La instalación falló. El entorno '$ENV_NAME' puede estar incompleto."
        log_info "Para eliminarlo: conda env remove -n $ENV_NAME"
    fi

    # Liberar lock file
    release_lock

    # Limpiar constraints temporal
    rm -f "$CONSTRAINTS_FILE" 2>/dev/null || true

    if [[ $exit_code -ne 0 ]]; then
        log_error "Script terminado con errores (código: $exit_code)"
        [[ -n "${LOG_FILE:-}" ]] && log_error "Revisa el log: $LOG_FILE"
    else
        log_info "Script de inicialización finalizado exitosamente"
    fi
}

die() {
    log_error "$1"
    exit "${2:-1}"
}

# --- Trap único: EXIT ejecuta cleanup; INT/TERM solo fijan exit code ---
# Esto evita que cleanup_on_exit se ejecute dos veces.
trap 'cleanup_on_exit' EXIT
trap 'log_error "Interrumpido por usuario (SIGINT)"; exit 130' INT
trap 'log_error "Terminado por señal (SIGTERM)"; exit 143' TERM

# ==============================================================================
# LOCK MANAGEMENT
# ==============================================================================

acquire_lock() {
    log_debug "Adquiriendo lock: $LOCK_FILE"

    # Crear lock file descriptor
    exec {LOCK_FD}>"$LOCK_FILE" || die "No se puede crear lock file: $LOCK_FILE"

    # Intentar adquirir lock exclusivo (non-blocking)
    if ! flock -n "$LOCK_FD"; then
        local existing_pid
        existing_pid=$(cat "$LOCK_FILE" 2>/dev/null || echo "desconocido")
        die "Otra instancia ya está ejecutándose (PID: $existing_pid)"
    fi

    # Escribir PID actual
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
# PIP ABSTRACTION (elimina word-splitting de "uv pip")
# ==============================================================================

run_pip() {
    # Función única para invocar pip: prefiere uv si está disponible.
    # Elimina el patrón frágil de $pip_cmd con word-splitting.
    if command -v uv &>/dev/null; then
        uv pip "$@"
    else
        pip "$@"
    fi
}

# ==============================================================================
# SETUP & VALIDATION
# ==============================================================================

setup_logging() {
    if ! mkdir -p "$ABSOLUTE_LOG_DIR" 2>/dev/null; then
        echo "[ERROR] No se puede crear directorio de logs: $ABSOLUTE_LOG_DIR" >&2
        exit 1
    fi

    if [[ ! -w "$ABSOLUTE_LOG_DIR" ]]; then
        echo "[ERROR] Sin permisos de escritura en: $ABSOLUTE_LOG_DIR" >&2
        exit 1
    fi

    LOG_FILE="${ABSOLUTE_LOG_DIR}/conda_setup_${TIMESTAMP}_${SCRIPT_PID}.log"
    if ! touch "$LOG_FILE" 2>/dev/null; then
        echo "[ERROR] No se puede crear archivo de log: $LOG_FILE" >&2
        exit 1
    fi

    {
        echo "=============================================="
        echo "Conda Environment Setup Log"
        echo "Script Version: $SCRIPT_VERSION"
        echo "Environment: $ENV_NAME"
        echo "Python Version: $PYTHON_VERSION"
        echo "Torch Version: ${TORCH_VERSION}+cpu"
        echo "Started: $(get_timestamp)"
        echo "PID: $SCRIPT_PID"
        echo "=============================================="
        echo ""
    } >> "$LOG_FILE"

    log_debug "Logging inicializado: ${LOG_FILE}"
}

check_base_dependencies() {
    log_step "Verificando dependencias base del sistema..."

    # python NO se verifica aquí: conda lo instala en el entorno.
    # Se verifica después de activar en verify_python_available.
    local -a required_commands=("flock" "conda")
    local -a missing=()

    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &>/dev/null; then
            missing+=("$cmd")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        die "Dependencias del sistema faltantes: ${missing[*]}"
    fi
    log_debug "Dependencias base verificadas: ${required_commands[*]}"
}

validate_python_version() {
    local version_pattern='^[0-9]+\.[0-9]+$'
    if ! [[ "$PYTHON_VERSION" =~ $version_pattern ]]; then
        die "Versión de Python inválida: $PYTHON_VERSION. Formato esperado: X.Y"
    fi
    log_debug "Versión de Python válida: $PYTHON_VERSION"
}

check_conda_installation() {
    log_step "Verificando instalación de Conda..."

    if ! command -v conda &>/dev/null; then
        log_error "Conda no está instalado o no está en el PATH."
        log_info ""
        log_info "Para instalar Miniconda:"
        log_info "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        log_info "  bash Miniconda3-latest-Linux-x86_64.sh"
        log_info ""
        log_info "Después de instalar, reinicia tu terminal o ejecuta:"
        log_info "  source ~/.bashrc"
        die "Conda requerido pero no encontrado"
    fi

    local conda_version
    conda_version=$(conda --version 2>/dev/null | awk '{print $2}' || echo "desconocida")

    local conda_base
    conda_base=$(conda info --base 2>/dev/null || echo "")
    if [[ -z "$conda_base" ]]; then
        die "No se pudo determinar la ruta base de Conda"
    fi

    log_debug "Conda base: $conda_base"
    log_success "Conda encontrado: v$conda_version"
}

initialize_conda_shell() {
    log_step "Inicializando Conda para este shell..."

    local conda_sh=""
    local -a conda_paths=(
        "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
        "$HOME/miniconda3/etc/profile.d/conda.sh"
        "$HOME/anaconda3/etc/profile.d/conda.sh"
        "/opt/conda/etc/profile.d/conda.sh"
    )

    for path in "${conda_paths[@]}"; do
        if [[ -f "$path" ]]; then
            conda_sh="$path"
            break
        fi
    done

    if [[ -n "$conda_sh" ]]; then
        log_debug "Sourcing: $conda_sh"
        # shellcheck source=/dev/null
        source "$conda_sh"
    else
        log_debug "Usando conda shell.bash hook"
        eval "$(conda shell.bash hook 2>/dev/null)" \
            || die "No se pudo inicializar conda shell hook"
    fi

    if ! type conda | grep -q "function"; then
        die "Conda no se inicializó correctamente como función"
    fi

    RUNTIME_STATE[CONDA_INITIALIZED]="true"
    log_success "Conda inicializado correctamente"
}

check_network_connectivity() {
    log_debug "Verificando conectividad de red..."

    local -a endpoints=(
        "repo.anaconda.com"
        "pypi.org"
        "download.pytorch.org"
    )

    local has_network=false
    for endpoint in "${endpoints[@]}"; do
        if timeout 5 bash -c "echo >/dev/tcp/$endpoint/443" 2>/dev/null; then
            has_network=true
            log_debug "Conectividad OK: $endpoint"
            break
        fi
    done

    if [[ "$has_network" != "true" ]]; then
        log_warn "No se detectó conexión a internet. Algunas instalaciones pueden fallar."
    else
        log_debug "Conectividad de red verificada"
    fi
}

# ==============================================================================
# CONSTRAINTS (coherencia con Docker stack)
# ==============================================================================

generate_constraints() {
    log_step "Generando constraints (coherencia con Docker stack v7.0)..."

    if [[ "${OPERATION_FLAGS[DRY_RUN]}" == "true" ]]; then
        log_info "[DRY-RUN] Generaría: $CONSTRAINTS_FILE"
        return 0
    fi

    # Mismo patrón que Dockerfiles v7.0: printf atómico
    printf '%s\n' \
        "torch==${TORCH_VERSION}+cpu" \
        "torchvision==${TORCHVISION_VERSION}+cpu" \
        "torchaudio==${TORCHAUDIO_VERSION}+cpu" \
        "sympy==${SYMPY_VERSION}" \
        > "$CONSTRAINTS_FILE"

    log_debug "Constraints generados: $(cat "$CONSTRAINTS_FILE" | tr '\n' ' ')"
    log_success "Constraints alineados con Docker stack"
}

# ==============================================================================
# ENVIRONMENT MANAGEMENT
# ==============================================================================

env_exists() {
    # awk extrae la primera columna; grep -xF hace match exacto sin regex
    conda env list 2>/dev/null | awk '{print $1}' | grep -qxF "$ENV_NAME"
}

get_env_python_version() {
    if env_exists; then
        conda run -n "$ENV_NAME" python --version 2>/dev/null \
            | awk '{print $2}' | cut -d. -f1,2 || echo ""
    fi
}

remove_environment() {
    log_step "Eliminando entorno existente '$ENV_NAME'..."

    if [[ "${OPERATION_FLAGS[DRY_RUN]}" == "true" ]]; then
        log_info "[DRY-RUN] conda env remove -n $ENV_NAME -y"
        return 0
    fi

    if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]]; then
        log_debug "Desactivando entorno actual..."
        conda deactivate 2>/dev/null || true
    fi

    if conda env remove -n "$ENV_NAME" -y >> "$LOG_FILE" 2>&1; then
        log_success "Entorno '$ENV_NAME' eliminado"
    else
        log_warn "No se pudo eliminar completamente el entorno"
    fi
}

create_environment() {
    log_step "Creando entorno '$ENV_NAME' con Python $PYTHON_VERSION..."

    if [[ "${OPERATION_FLAGS[DRY_RUN]}" == "true" ]]; then
        log_info "[DRY-RUN] conda create -n $ENV_NAME python=$PYTHON_VERSION -y"
        return 0
    fi

    local create_start
    create_start=$(date +%s)

    if ! conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y >> "$LOG_FILE" 2>&1; then
        log_error "Fallo al crear el entorno conda"
        log_error "Últimas líneas del log:"
        tail -20 "$LOG_FILE" | while IFS= read -r line; do
            log_error "  | $line"
        done
        exit 1
    fi

    RUNTIME_STATE[ENV_CREATED]="true"
    local create_duration=$(($(date +%s) - create_start))
    log_success "Entorno creado en ${create_duration}s"
}

activate_environment() {
    log_step "Activando entorno '$ENV_NAME'..."

    if [[ "${OPERATION_FLAGS[DRY_RUN]}" == "true" ]]; then
        log_info "[DRY-RUN] conda activate $ENV_NAME"
        return 0
    fi

    if ! conda activate "$ENV_NAME" 2>> "$LOG_FILE"; then
        die "No se pudo activar el entorno '$ENV_NAME'"
    fi

    if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
        die "El entorno no se activó correctamente (actual: ${CONDA_DEFAULT_ENV:-none})"
    fi

    RUNTIME_STATE[ENV_ACTIVATED]="true"

    local python_path python_version
    python_path=$(which python 2>/dev/null || echo "no encontrado")
    python_version=$(python --version 2>/dev/null || echo "desconocida")
    log_debug "Python: $python_version ($python_path)"
    log_success "Entorno '$ENV_NAME' activado"
}

verify_python_available() {
    log_debug "Verificando disponibilidad de Python en entorno activo..."

    if ! command -v python &>/dev/null; then
        die "Python no disponible después de activar el entorno"
    fi

    if ! command -v pip &>/dev/null; then
        die "pip no disponible después de activar el entorno"
    fi

    local actual_version
    actual_version=$(python --version 2>/dev/null | awk '{print $2}' | cut -d. -f1,2)
    if [[ "$actual_version" != "$PYTHON_VERSION" ]]; then
        log_warn "Versión de Python ($actual_version) difiere de la solicitada ($PYTHON_VERSION)"
    fi

    log_debug "Python $actual_version disponible"
}

# ==============================================================================
# PACKAGE INSTALLATION
# ==============================================================================

install_conda_packages() {
    log_step "Instalando paquetes base via Conda (faiss-cpu, redis)..."

    if [[ "${OPERATION_FLAGS[DRY_RUN]}" == "true" ]]; then
        log_info "[DRY-RUN] conda install -c pytorch -c conda-forge faiss-cpu redis -y"
        return 0
    fi

    local install_start
    install_start=$(date +%s)

    if ! conda install -c pytorch -c conda-forge faiss-cpu redis -y >> "$LOG_FILE" 2>&1; then
        log_error "Fallo al instalar paquetes conda"
        log_error "Revisa el log para más detalles: $LOG_FILE"
        exit 1
    fi

    local install_duration=$(($(date +%s) - install_start))
    log_success "Paquetes conda instalados en ${install_duration}s"
}

install_uv() {
    log_step "Instalando 'uv' (acelerador de pip)..."

    if [[ "${OPERATION_FLAGS[DRY_RUN]}" == "true" ]]; then
        log_info "[DRY-RUN] pip install uv"
        return 0
    fi

    if command -v uv &>/dev/null; then
        local uv_version
        uv_version=$(uv --version 2>/dev/null | head -1 || echo "instalado")
        log_info "uv ya está instalado: $uv_version"
        return 0
    fi

    if ! pip install uv >> "$LOG_FILE" 2>&1; then
        log_warn "No se pudo instalar uv, continuando con pip estándar"
        return 1
    fi

    log_success "uv instalado correctamente"
    return 0
}

install_pytorch() {
    if [[ "${OPERATION_FLAGS[SKIP_PYTORCH]}" == "true" ]]; then
        log_info "Omitiendo instalación de PyTorch (--skip-pytorch)"
        return 0
    fi

    log_step "Instalando PyTorch CPU (torch==${TORCH_VERSION}+cpu)..."

    if [[ "${OPERATION_FLAGS[DRY_RUN]}" == "true" ]]; then
        log_info "[DRY-RUN] run_pip install torch==${TORCH_VERSION}+cpu torchvision==${TORCHVISION_VERSION}+cpu torchaudio==${TORCHAUDIO_VERSION}+cpu --index-url $PYTORCH_INDEX_URL"
        return 0
    fi

    local install_start
    install_start=$(date +%s)

    # Versiones pinneadas idénticas a Dockerfile.agent/core v7.0
    if ! run_pip install \
            "torch==${TORCH_VERSION}+cpu" \
            "torchvision==${TORCHVISION_VERSION}+cpu" \
            "torchaudio==${TORCHAUDIO_VERSION}+cpu" \
            --index-url "$PYTORCH_INDEX_URL" \
            >> "$LOG_FILE" 2>&1; then
        log_error "Fallo al instalar PyTorch"
        exit 1
    fi

    # Verificación idéntica a Dockerfiles v7.0
    if python -c "import torch; v=torch.__version__; assert v=='${TORCH_VERSION}+cpu', f'Esperado ${TORCH_VERSION}+cpu, obtenido {v}'" 2>/dev/null; then
        local torch_version
        torch_version=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
        log_debug "PyTorch verificado: $torch_version"
    else
        log_warn "PyTorch instalado pero la versión no coincide con ${TORCH_VERSION}+cpu"
    fi

    local install_duration=$(($(date +%s) - install_start))
    log_success "PyTorch instalado en ${install_duration}s"
}

install_ml_packages() {
    if [[ "${OPERATION_FLAGS[SKIP_PYTORCH]}" == "true" ]]; then
        log_info "Omitiendo paquetes ML (dependen de PyTorch)"
        return 0
    fi

    log_step "Instalando paquetes ML con constraints (coherencia Docker stack)..."

    if [[ "${OPERATION_FLAGS[DRY_RUN]}" == "true" ]]; then
        log_info "[DRY-RUN] run_pip install -c $CONSTRAINTS_FILE sentence-transformers transformers accelerate huggingface-hub safetensors"
        return 0
    fi

    local install_start
    install_start=$(date +%s)

    # Mismos paquetes y constraints que Dockerfiles v7.0 FASE 4
    if ! run_pip install \
            -c "$CONSTRAINTS_FILE" \
            sentence-transformers \
            transformers \
            accelerate \
            huggingface-hub \
            safetensors \
            >> "$LOG_FILE" 2>&1; then
        log_error "Fallo al instalar paquetes ML"
        exit 1
    fi

    local install_duration=$(($(date +%s) - install_start))
    log_success "Paquetes ML instalados en ${install_duration}s"
}

install_requirements() {
    local req_file="$1"
    local description="$2"
    local is_optional="${3:-false}"

    # Resolver path si es relativo
    if [[ ! "$req_file" = /* ]]; then
        req_file="${PROJECT_ROOT}/${req_file}"
    fi

    if [[ ! -f "$req_file" ]]; then
        if [[ "$is_optional" == "true" ]]; then
            log_info "Archivo opcional no encontrado: $(basename "$req_file")"
            return 0
        else
            log_warn "Archivo no encontrado: $req_file"
            return 1
        fi
    fi

    log_step "Instalando $description desde $(basename "$req_file")..."

    if [[ "${OPERATION_FLAGS[DRY_RUN]}" == "true" ]]; then
        log_info "[DRY-RUN] run_pip install -c $CONSTRAINTS_FILE -r $req_file"
        return 0
    fi

    local install_start
    install_start=$(date +%s)

    # Constraints aplicados a TODAS las instalaciones (coherencia Docker stack)
    if ! run_pip install -c "$CONSTRAINTS_FILE" -r "$req_file" >> "$LOG_FILE" 2>&1; then
        log_error "Fallo al instalar dependencias desde $req_file"
        return 1
    fi

    local install_duration=$(($(date +%s) - install_start))
    log_success "$description instaladas en ${install_duration}s"
}

install_project_dependencies() {
    # Dependencias principales (con constraints)
    install_requirements "$REQUIREMENTS_FILE" "dependencias del proyecto" false || {
        log_error "Las dependencias principales son requeridas"
        exit 1
    }

    # Dependencias de desarrollo (opcionales, con constraints)
    if [[ "${OPERATION_FLAGS[SKIP_DEV_DEPS]}" != "true" ]]; then
        install_requirements "$REQUIREMENTS_DEV_FILE" "dependencias de desarrollo" true
    else
        log_info "Omitiendo dependencias de desarrollo (--skip-dev)"
    fi

    # Paquetes de visualización y ciencia (opcionales, no críticos)
    log_step "Instalando librerías de visualización y ciencia..."
    if [[ "${OPERATION_FLAGS[DRY_RUN]}" == "true" ]]; then
        log_info "[DRY-RUN] run_pip install -c constraints matplotlib scipy networkx"
    else
        if ! run_pip install \
                -c "$CONSTRAINTS_FILE" \
                matplotlib scipy networkx \
                >> "$LOG_FILE" 2>&1; then
            log_warn "No se pudieron instalar paquetes de visualización (no crítico)"
        else
            log_success "Paquetes de visualización instalados"
        fi
    fi
}

# ==============================================================================
# POST-INSTALLATION TASKS
# ==============================================================================

generate_embeddings() {
    if [[ "${OPERATION_FLAGS[SKIP_EMBEDDINGS]}" == "true" ]]; then
        log_info "Omitiendo generación de embeddings (--skip-embeddings)"
        return 0
    fi

    log_step "Generando artefactos de búsqueda semántica..."

    if [[ ! -f "$PROCESSED_DATA_FILE" ]]; then
        log_warn "No se encontró: $PROCESSED_DATA_FILE"
        log_info "Ejecuta el pipeline de carga (upload) primero para generar los datos base."
        return 0
    fi

    if [[ ! -f "$EMBEDDINGS_SCRIPT" ]]; then
        log_warn "Script de embeddings no encontrado: $EMBEDDINGS_SCRIPT"
        return 0
    fi

    if [[ "${OPERATION_FLAGS[DRY_RUN]}" == "true" ]]; then
        log_info "[DRY-RUN] python $EMBEDDINGS_SCRIPT --input $PROCESSED_DATA_FILE"
        return 0
    fi

    local gen_start
    gen_start=$(date +%s)

    if python "$EMBEDDINGS_SCRIPT" --input "$PROCESSED_DATA_FILE" >> "$LOG_FILE" 2>&1; then
        local gen_duration=$(($(date +%s) - gen_start))
        log_success "Embeddings generados en ${gen_duration}s"
    else
        log_warn "Fallo en la generación de embeddings (no crítico)"
        log_info "Ejecución manual: python $EMBEDDINGS_SCRIPT --input $PROCESSED_DATA_FILE"
    fi
}

verify_installation() {
    log_step "Verificando instalación..."

    local -a checks_passed=()
    local -a checks_failed=()

    # Verificar Python
    if python --version &>/dev/null; then
        checks_passed+=("Python")
    else
        checks_failed+=("Python")
    fi

    # Verificar paquetes clave
    local -a packages_to_check=("faiss" "redis" "flask")
    if [[ "${OPERATION_FLAGS[SKIP_PYTORCH]}" != "true" ]]; then
        packages_to_check+=("torch" "sentence_transformers")
    fi

    for pkg in "${packages_to_check[@]}"; do
        if python -c "import $pkg" 2>/dev/null; then
            checks_passed+=("$pkg")
        else
            checks_failed+=("$pkg")
        fi
    done

    # Verificar coherencia de versión PyTorch con Docker stack
    if [[ "${OPERATION_FLAGS[SKIP_PYTORCH]}" != "true" ]]; then
        local torch_ver
        torch_ver=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
        if [[ "$torch_ver" == "${TORCH_VERSION}+cpu" ]]; then
            checks_passed+=("torch-version-coherence")
            log_debug "PyTorch $torch_ver coincide con Docker stack"
        elif [[ -n "$torch_ver" ]]; then
            checks_failed+=("torch-version-coherence(${torch_ver}≠${TORCH_VERSION}+cpu)")
        fi
    fi

    # Verificar uv
    if command -v uv &>/dev/null; then
        checks_passed+=("uv")
    else
        checks_passed+=("uv(ausente,usando pip)")
    fi

    # Reportar resultados
    log_debug "Verificaciones pasadas: ${checks_passed[*]}"
    if [[ ${#checks_failed[@]} -gt 0 ]]; then
        log_warn "Verificaciones fallidas: ${checks_failed[*]}"
    fi

    if [[ ${#checks_passed[@]} -gt 0 ]]; then
        log_success "Instalación verificada: ${#checks_passed[@]} componentes OK"
    fi
}

# ==============================================================================
# DISPLAY FUNCTIONS
# ==============================================================================

_color_or_plain() {
    # Emite color solo si stdout es terminal interactivo
    local color="$1"
    local text="$2"
    if is_terminal; then
        echo -e "${color}${text}${COLORS[RESET]}"
    else
        echo "$text"
    fi
}

display_final_status() {
    echo ""
    log_separator "=" "CONFIGURACIÓN COMPLETADA"
    echo ""

    log_info "Entorno: $ENV_NAME"
    log_info "Python: $(python --version 2>/dev/null || echo 'N/A')"
    log_info "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
    if command -v uv &>/dev/null; then
        log_info "Gestor de paquetes: uv ($(uv --version 2>/dev/null | head -1))"
    else
        log_info "Gestor de paquetes: pip"
    fi
    echo ""
    log_separator "-"
    echo ""

    log_info "Para activar el entorno manualmente:"
    _color_or_plain "${COLORS[GREEN]}" "  conda activate $ENV_NAME"
    echo ""

    log_info "Comandos útiles:"
    _color_or_plain "${COLORS[CYAN]}" "  python -m flask run --port=5002    # Iniciar servidor"
    _color_or_plain "${COLORS[CYAN]}" "  conda deactivate                   # Desactivar entorno"
    _color_or_plain "${COLORS[CYAN]}" "  conda env remove -n $ENV_NAME      # Eliminar entorno"
    echo ""

    if [[ -n "${LOG_FILE:-}" ]]; then
        log_info "Log completo: $LOG_FILE"
    fi
}

# ==============================================================================
# OPERATION MODES
# ==============================================================================

run_setup() {
    # Paso 1: Gestión del entorno
    if env_exists; then
        if [[ "${OPERATION_FLAGS[FORCE_RECREATE]}" == "true" ]]; then
            log_warn "Entorno '$ENV_NAME' existe. Recreando (--force)..."
            remove_environment
            create_environment
        elif [[ "${OPERATION_FLAGS[UPDATE_ONLY]}" == "true" ]]; then
            log_info "Entorno '$ENV_NAME' existe. Actualizando dependencias..."
        else
            log_info "Entorno '$ENV_NAME' ya existe."
            local existing_version
            existing_version=$(get_env_python_version)
            if [[ -n "$existing_version" ]] && [[ "$existing_version" != "$PYTHON_VERSION" ]]; then
                log_warn "Python existente ($existing_version) difiere del solicitado ($PYTHON_VERSION)"
                log_info "Usa --force para recrear con la versión correcta"
            fi
        fi
    else
        create_environment
    fi

    # Paso 2: Activar entorno
    activate_environment
    verify_python_available

    # Paso 3: Generar constraints (antes de cualquier pip install)
    generate_constraints

    # Paso 4: Paquetes conda (solo si entorno nuevo o no update-only)
    if [[ "${OPERATION_FLAGS[UPDATE_ONLY]}" != "true" ]] \
        || [[ "${RUNTIME_STATE[ENV_CREATED]}" == "true" ]]; then
        install_conda_packages
    fi

    # Paso 5: Instalar uv
    install_uv || true  # No fatal

    # Paso 6: Instalar PyTorch (versionado, con constraints)
    install_pytorch

    # Paso 7: Instalar paquetes ML (con constraints)
    install_ml_packages

    # Paso 8: Instalar dependencias del proyecto (con constraints)
    install_project_dependencies

    # Paso 9: Tareas post-instalación
    generate_embeddings

    # Paso 10: Verificar instalación
    verify_installation

    # Marcar como completado
    RUNTIME_STATE[ENV_ACTIVATED]="completed"
}

run_clean() {
    log_info "Modo limpieza: eliminando entorno '$ENV_NAME'..."

    if ! env_exists; then
        log_info "El entorno '$ENV_NAME' no existe"
        return 0
    fi

    remove_environment
    log_success "Limpieza completada"
}

# ==============================================================================
# CLI ARGUMENT PARSING
# ==============================================================================

show_help() {
    # Colores condicionales para --help en pipe/redirección
    local b="" r="" c=""
    if is_terminal; then
        b="${COLORS[BOLD]}"
        r="${COLORS[RESET]}"
        c="${COLORS[CYAN]}"
    fi

    cat << EOF
${b}NOMBRE${r}
    $SCRIPT_NAME - Inicialización de Entorno Conda + UV

${b}USO${r}
    $SCRIPT_NAME [OPCIONES]

${b}DESCRIPCIÓN${r}
    Configura un entorno Conda con Python, instala dependencias del proyecto
    usando UV (acelerador de pip), y prepara los artefactos necesarios.
    Versiones de PyTorch alineadas con Docker stack v7.0.

${b}OPCIONES${r}
    -h, --help              Muestra esta ayuda
    -V, --version           Muestra la versión
    -v, --verbose           Modo verbose con información de debug
    -n, --name NAME         Nombre del entorno (default: $ENV_NAME)
    -p, --python VERSION    Versión de Python (default: $PYTHON_VERSION)

${b}MODOS DE OPERACIÓN${r}
    --clean                 Solo eliminar el entorno existente
    --force                 Forzar recreación del entorno si existe
    --update                Solo actualizar dependencias (no recrear entorno)

${b}OPCIONES DE INSTALACIÓN${r}
    --skip-pytorch          Omitir instalación de PyTorch
    --skip-embeddings       Omitir generación de embeddings
    --skip-dev              Omitir dependencias de desarrollo
    --dry-run               Mostrar comandos sin ejecutarlos

${b}VARIABLES DE ENTORNO${r}
    ENV_NAME                Nombre del entorno conda
    PYTHON_VERSION          Versión de Python
    TORCH_VERSION           Versión de PyTorch (default: $TORCH_VERSION)
    LOG_DIR                 Directorio de logs
    PYTORCH_INDEX_URL       URL del índice de PyTorch
    VERBOSE                 Habilitar modo debug (true/false)

${b}EJEMPLOS${r}
    ${c}$SCRIPT_NAME${r}                          # Setup completo
    ${c}$SCRIPT_NAME --force${r}                   # Recrear entorno desde cero
    ${c}$SCRIPT_NAME --update${r}                  # Solo actualizar dependencias
    ${c}$SCRIPT_NAME --skip-pytorch${r}            # Sin PyTorch (más rápido)
    ${c}$SCRIPT_NAME --clean${r}                   # Eliminar entorno
    ${c}$SCRIPT_NAME -n myenv -p 3.11${r}          # Entorno personalizado
    ${c}$SCRIPT_NAME --verbose --dry-run${r}       # Ver qué haría

${b}EXIT CODES${r}
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
                # Establece la VARIABLE que log_debug realmente lee
                VERBOSE="true"
                shift
                ;;
            -n|--name)
                [[ -z "${2:-}" ]] && die "La opción --name requiere un argumento"
                ENV_NAME="$2"
                shift 2
                ;;
            -p|--python)
                [[ -z "${2:-}" ]] && die "La opción --python requiere un argumento"
                PYTHON_VERSION="$2"
                shift 2
                ;;
            --clean)
                OPERATION_FLAGS[CLEAN_ENV]="true"
                shift
                ;;
            --force)
                OPERATION_FLAGS[FORCE_RECREATE]="true"
                shift
                ;;
            --update)
                OPERATION_FLAGS[UPDATE_ONLY]="true"
                shift
                ;;
            --skip-pytorch)
                OPERATION_FLAGS[SKIP_PYTORCH]="true"
                shift
                ;;
            --skip-embeddings)
                OPERATION_FLAGS[SKIP_EMBEDDINGS]="true"
                shift
                ;;
            --skip-dev)
                OPERATION_FLAGS[SKIP_DEV_DEPS]="true"
                shift
                ;;
            --dry-run)
                OPERATION_FLAGS[DRY_RUN]="true"
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

    # Validar opciones mutuamente excluyentes
    if [[ "${OPERATION_FLAGS[FORCE_RECREATE]}" == "true" ]] \
        && [[ "${OPERATION_FLAGS[UPDATE_ONLY]}" == "true" ]]; then
        die "--force y --update son mutuamente excluyentes"
    fi

    if [[ "${OPERATION_FLAGS[CLEAN_ENV]}" == "true" ]] \
        && [[ "${OPERATION_FLAGS[FORCE_RECREATE]}" == "true" ]]; then
        die "--clean y --force son mutuamente excluyentes (--clean solo elimina)"
    fi
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

main() {
    parse_arguments "$@"
    setup_logging
    validate_python_version
    acquire_lock

    # Banner
    log_separator "=" "Configuración de Entorno Local (Conda + UV)"
    log_info "Versión: $SCRIPT_VERSION"
    log_info "Entorno: $ENV_NAME"
    log_info "Python: $PYTHON_VERSION"
    log_info "PyTorch: ${TORCH_VERSION}+cpu (coherente con Docker stack v7.0)"
    [[ "$VERBOSE" == "true" ]]                         && log_info "Modo: VERBOSE"
    [[ "${OPERATION_FLAGS[DRY_RUN]}" == "true" ]]      && log_info "Modo: DRY-RUN"
    [[ "${OPERATION_FLAGS[CLEAN_ENV]}" == "true" ]]     && log_info "Modo: CLEAN"
    [[ "${OPERATION_FLAGS[FORCE_RECREATE]}" == "true" ]] && log_info "Modo: FORCE RECREATE"
    [[ "${OPERATION_FLAGS[UPDATE_ONLY]}" == "true" ]]   && log_info "Modo: UPDATE ONLY"
    echo ""

    # Validación
    check_base_dependencies
    check_conda_installation
    initialize_conda_shell
    check_network_connectivity

    # Ejecutar operación
    if [[ "${OPERATION_FLAGS[CLEAN_ENV]}" == "true" ]]; then
        run_clean
    else
        run_setup
        display_final_status
    fi

    log_success "=== Operación completada exitosamente ==="
    return 0
}

# Punto de entrada
main "$@"