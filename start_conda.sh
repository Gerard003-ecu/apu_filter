#!/bin/bash
# ==============================================================================
# Script de Inicialización de Entorno Local (Conda + UV)
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
: "${ENV_NAME:=apu_filter_env}"
: "${PYTHON_VERSION:=3.10}"
: "${LOG_DIR:=./logs}"
: "${PYTORCH_INDEX_URL:=https://download.pytorch.org/whl/cpu}"
: "${VERBOSE:=false}"

# --- Paths (resolved relative to script directory) ---
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
REQUIREMENTS_DEV_FILE="${SCRIPT_DIR}/requirements-dev.txt"
PROCESSED_DATA_FILE="${SCRIPT_DIR}/data/processed_apus.json"
EMBEDDINGS_SCRIPT="${SCRIPT_DIR}/scripts/generate_embeddings.py"

readonly LOCK_FILE="/tmp/${SCRIPT_NAME%.*}.lock"

# --- Runtime State ---
LOG_FILE=""
LOCK_FD=""
CONDA_INITIALIZED=false
ENV_CREATED=false
ENV_ACTIVATED=false

# --- Operation Flags ---
SKIP_PYTORCH=false
SKIP_EMBEDDINGS=false
SKIP_DEV_DEPS=false
FORCE_RECREATE=false
CLEAN_ENV=false
UPDATE_ONLY=false
DRY_RUN=false

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

    # Terminal output (with colors if interactive)
    if is_terminal; then
        printf "${color}[%-7s]${COLORS[RESET]} %s\n" "$level" "$message"
    else
        printf "[%-7s] %s\n" "$level" "$message"
    fi

    # File output (without colors, with timestamp)
    if [[ -n "${LOG_FILE:-}" ]] && [[ -w "$(dirname "$LOG_FILE" 2>/dev/null || echo ".")" ]]; then
        printf "[%s] [%-7s] %s\n" "$timestamp" "$level" "$message" >> "$LOG_FILE" 2>/dev/null || true
    fi
}

log_info()    { _log "INFO"    "${COLORS[BLUE]}"    "$1"; }
log_step()    { _log "STEP"    "${COLORS[CYAN]}"    "→ $1"; }
log_success() { _log "SUCCESS" "${COLORS[GREEN]}"   "✓ $1"; }
log_warn()    { _log "WARN"    "${COLORS[YELLOW]}"  "⚠ $1"; }
log_error()   { _log "ERROR"   "${COLORS[RED]}"     "✗ $1" >&2; }
log_debug()   { [[ "$VERBOSE" == "true" ]] && _log "DEBUG" "${COLORS[MAGENTA]}" "$1" || true; }

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

cleanup() {
    local exit_code=$?
    
    # Prevent recursive cleanup
    trap - EXIT INT TERM

    log_debug "Ejecutando cleanup (exit_code: $exit_code, PID: $SCRIPT_PID)"

    # If we created an environment but failed before completion, offer to remove it
    if [[ $exit_code -ne 0 ]] && [[ "$ENV_CREATED" == "true" ]] && [[ "$ENV_ACTIVATED" != "completed" ]]; then
        log_warn "La instalación falló. El entorno '$ENV_NAME' puede estar incompleto."
        log_info "Para eliminarlo: conda env remove -n $ENV_NAME"
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
    # Resolve LOG_DIR relative to script directory if not absolute
    if [[ ! "$LOG_DIR" = /* ]]; then
        LOG_DIR="${SCRIPT_DIR}/${LOG_DIR}"
    fi

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
    LOG_FILE="${LOG_DIR}/conda_setup_$(date +%Y%m%d_%H%M%S)_${SCRIPT_PID}.log"

    if ! touch "$LOG_FILE" 2>/dev/null; then
        echo "[ERROR] No se puede crear archivo de log: $LOG_FILE" >&2
        exit 1
    fi

    # Write header to log file
    {
        echo "=============================================="
        echo "Conda Environment Setup Log"
        echo "Script Version: $SCRIPT_VERSION"
        echo "Environment: $ENV_NAME"
        echo "Python Version: $PYTHON_VERSION"
        echo "Started: $(get_timestamp)"
        echo "PID: $SCRIPT_PID"
        echo "=============================================="
        echo ""
    } >> "$LOG_FILE"

    log_debug "Logging inicializado: ${LOG_FILE}"
}

check_base_dependencies() {
    log_step "Verificando dependencias base del sistema..."

    local -a required_commands=("flock")
    local -a missing=()

    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &>/dev/null; then
            missing+=("$cmd")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        die "Dependencias del sistema faltantes: ${missing[*]}"
    fi

    log_debug "Dependencias base verificadas"
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

    # Get conda version
    local conda_version
    conda_version=$(conda --version 2>/dev/null | awk '{print $2}' || echo "desconocida")
    log_debug "Conda versión: $conda_version"

    # Check conda base path
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

    # Try to initialize conda shell hooks
    local conda_sh=""
    
    # Find conda.sh in common locations
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
        eval "$(conda shell.bash hook 2>/dev/null)" || die "No se pudo inicializar conda shell hook"
    fi

    # Verify conda activate is now available
    if ! type conda | grep -q "function"; then
        die "Conda no se inicializó correctamente como función"
    fi

    CONDA_INITIALIZED=true
    log_success "Conda inicializado correctamente"
}

check_network_connectivity() {
    log_debug "Verificando conectividad de red..."

    # Quick check to common endpoints
    local -a endpoints=(
        "repo.anaconda.com"
        "pypi.org"
    )

    local has_network=false
    for endpoint in "${endpoints[@]}"; do
        if timeout 5 bash -c "echo >/dev/tcp/$endpoint/443" 2>/dev/null; then
            has_network=true
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
# ENVIRONMENT MANAGEMENT
# ==============================================================================

env_exists() {
    conda env list 2>/dev/null | grep -qE "^${ENV_NAME}\s" 
}

get_env_python_version() {
    if env_exists; then
        conda run -n "$ENV_NAME" python --version 2>/dev/null | awk '{print $2}' | cut -d. -f1,2 || echo ""
    fi
}

remove_environment() {
    log_step "Eliminando entorno existente '$ENV_NAME'..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] conda env remove -n $ENV_NAME -y"
        return 0
    fi

    # Deactivate if currently active
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

    if [[ "$DRY_RUN" == "true" ]]; then
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

    ENV_CREATED=true
    local create_duration=$(($(date +%s) - create_start))
    log_success "Entorno creado en ${create_duration}s"
}

activate_environment() {
    log_step "Activando entorno '$ENV_NAME'..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] conda activate $ENV_NAME"
        return 0
    fi

    if ! conda activate "$ENV_NAME" 2>> "$LOG_FILE"; then
        die "No se pudo activar el entorno '$ENV_NAME'"
    fi

    # Verify activation
    if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
        die "El entorno no se activó correctamente (actual: ${CONDA_DEFAULT_ENV:-none})"
    fi

    ENV_ACTIVATED=true

    # Log environment info
    local python_path python_version
    python_path=$(which python 2>/dev/null || echo "no encontrado")
    python_version=$(python --version 2>/dev/null || echo "desconocida")
    
    log_debug "Python: $python_version ($python_path)"
    log_success "Entorno '$ENV_NAME' activado"
}

verify_python_available() {
    log_debug "Verificando disponibilidad de Python..."

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

    if [[ "$DRY_RUN" == "true" ]]; then
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

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] pip install uv"
        return 0
    fi

    # Check if uv is already installed
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

get_pip_command() {
    # Returns the best available pip command (uv pip or regular pip)
    if command -v uv &>/dev/null; then
        echo "uv pip"
    else
        echo "pip"
    fi
}

install_pytorch() {
    if [[ "$SKIP_PYTORCH" == "true" ]]; then
        log_info "Omitiendo instalación de PyTorch (--skip-pytorch)"
        return 0
    fi

    log_step "Instalando PyTorch (versión CPU)..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] $(get_pip_command) install torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL"
        return 0
    fi

    local pip_cmd
    pip_cmd=$(get_pip_command)

    local install_start
    install_start=$(date +%s)

    if ! $pip_cmd install torch torchvision torchaudio --index-url "$PYTORCH_INDEX_URL" >> "$LOG_FILE" 2>&1; then
        log_error "Fallo al instalar PyTorch"
        exit 1
    fi

    # Verify installation
    if ! python -c "import torch; print(f'PyTorch {torch.__version__}')" >> "$LOG_FILE" 2>&1; then
        log_warn "PyTorch instalado pero no se pudo importar correctamente"
    else
        local torch_version
        torch_version=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
        log_debug "PyTorch versión: $torch_version"
    fi

    local install_duration=$(($(date +%s) - install_start))
    log_success "PyTorch instalado en ${install_duration}s"
}

install_requirements() {
    local req_file="$1"
    local description="$2"
    local is_optional="${3:-false}"

    # Resolve path if relative
    if [[ ! "$req_file" = /* ]]; then
        req_file="${SCRIPT_DIR}/${req_file}"
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

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] $(get_pip_command) install -r $req_file"
        return 0
    fi

    local pip_cmd
    pip_cmd=$(get_pip_command)

    local install_start
    install_start=$(date +%s)

    if ! $pip_cmd install -r "$req_file" >> "$LOG_FILE" 2>&1; then
        log_error "Fallo al instalar dependencias desde $req_file"
        return 1
    fi

    local install_duration=$(($(date +%s) - install_start))
    log_success "$description instaladas en ${install_duration}s"
}

install_project_dependencies() {
    # Main requirements
    install_requirements "$REQUIREMENTS_FILE" "dependencias del proyecto" false || {
        log_error "Las dependencias principales son requeridas"
        exit 1
    }

    # Development requirements (optional)
    if [[ "$SKIP_DEV_DEPS" != "true" ]]; then
        install_requirements "$REQUIREMENTS_DEV_FILE" "dependencias de desarrollo" true
    else
        log_info "Omitiendo dependencias de desarrollo (--skip-dev)"
    fi
}

# ==============================================================================
# POST-INSTALLATION TASKS
# ==============================================================================

generate_embeddings() {
    if [[ "$SKIP_EMBEDDINGS" == "true" ]]; then
        log_info "Omitiendo generación de embeddings (--skip-embeddings)"
        return 0
    fi

    log_step "Generando artefactos de búsqueda semántica..."

    # Check for data file
    if [[ ! -f "$PROCESSED_DATA_FILE" ]]; then
        log_warn "No se encontró: $PROCESSED_DATA_FILE"
        log_info "Ejecuta el pipeline de carga (upload) primero para generar los datos base."
        return 0
    fi

    # Check for embeddings script
    if [[ ! -f "$EMBEDDINGS_SCRIPT" ]]; then
        log_warn "Script de embeddings no encontrado: $EMBEDDINGS_SCRIPT"
        return 0
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
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
        log_info "Puedes ejecutarlo manualmente después: python $EMBEDDINGS_SCRIPT --input $PROCESSED_DATA_FILE"
    fi
}

verify_installation() {
    log_step "Verificando instalación..."

    local -a checks_passed=()
    local -a checks_failed=()

    # Check Python
    if python --version &>/dev/null; then
        checks_passed+=("Python")
    else
        checks_failed+=("Python")
    fi

    # Check key packages
    local -a packages_to_check=("faiss" "redis" "flask")
    
    if [[ "$SKIP_PYTORCH" != "true" ]]; then
        packages_to_check+=("torch")
    fi

    for pkg in "${packages_to_check[@]}"; do
        if python -c "import $pkg" 2>/dev/null; then
            checks_passed+=("$pkg")
        else
            checks_failed+=("$pkg")
        fi
    done

    # Check uv
    if command -v uv &>/dev/null; then
        checks_passed+=("uv")
    else
        checks_failed+=("uv (opcional)")
    fi

    # Report results
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

display_final_status() {
    echo ""
    log_separator "=" "CONFIGURACIÓN COMPLETADA"
    echo ""

    # Environment info
    log_info "Entorno: $ENV_NAME"
    log_info "Python: $(python --version 2>/dev/null || echo 'N/A')"
    
    if command -v uv &>/dev/null; then
        log_info "Gestor de paquetes: uv ($(uv --version 2>/dev/null | head -1))"
    else
        log_info "Gestor de paquetes: pip"
    fi
    
    echo ""
    log_separator "-"
    echo ""

    # Activation instructions
    log_info "Para activar el entorno manualmente:"
    echo -e "  ${COLORS[GREEN]}conda activate $ENV_NAME${COLORS[RESET]}"
    echo ""

    # Common commands
    log_info "Comandos útiles:"
    echo -e "  ${COLORS[CYAN]}python -m flask run --port=5002${COLORS[RESET]}  # Iniciar servidor"
    echo -e "  ${COLORS[CYAN]}conda deactivate${COLORS[RESET]}                  # Desactivar entorno"
    echo -e "  ${COLORS[CYAN]}conda env remove -n $ENV_NAME${COLORS[RESET]}     # Eliminar entorno"
    echo ""

    if [[ -n "${LOG_FILE:-}" ]]; then
        log_info "Log completo: $LOG_FILE"
    fi
}

# ==============================================================================
# OPERATION MODES
# ==============================================================================

run_setup() {
    # Step 1: Check if environment exists
    if env_exists; then
        if [[ "$FORCE_RECREATE" == "true" ]]; then
            log_warn "Entorno '$ENV_NAME' existe. Recreando (--force)..."
            remove_environment
            create_environment
        elif [[ "$UPDATE_ONLY" == "true" ]]; then
            log_info "Entorno '$ENV_NAME' existe. Actualizando dependencias..."
        else
            log_info "Entorno '$ENV_NAME' ya existe."
            
            # Check Python version match
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

    # Step 2: Activate environment
    activate_environment
    verify_python_available

    # Step 3: Install packages (only if not update-only or if newly created)
    if [[ "$UPDATE_ONLY" != "true" ]] || [[ "$ENV_CREATED" == "true" ]]; then
        install_conda_packages
    fi

    # Step 4: Install uv
    install_uv || true  # Non-fatal

    # Step 5: Install PyTorch
    install_pytorch

    # Step 6: Install project dependencies
    install_project_dependencies

    # Step 7: Post-installation tasks
    generate_embeddings

    # Step 8: Verify installation
    verify_installation

    # Mark as completed
    ENV_ACTIVATED="completed"
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
    cat << EOF
${COLORS[BOLD]}NOMBRE${COLORS[RESET]}
    $SCRIPT_NAME - Inicialización de Entorno Conda + UV

${COLORS[BOLD]}USO${COLORS[RESET]}
    $SCRIPT_NAME [OPCIONES]

${COLORS[BOLD]}DESCRIPCIÓN${COLORS[RESET]}
    Configura un entorno Conda con Python, instala dependencias del proyecto
    usando UV (acelerador de pip), y prepara los artefactos necesarios.

${COLORS[BOLD]}OPCIONES${COLORS[RESET]}
    -h, --help              Muestra esta ayuda
    -V, --version           Muestra la versión
    -v, --verbose           Modo verbose con información de debug
    -n, --name NAME         Nombre del entorno (default: $ENV_NAME)
    -p, --python VERSION    Versión de Python (default: $PYTHON_VERSION)

${COLORS[BOLD]}MODOS DE OPERACIÓN${COLORS[RESET]}
    --clean                 Solo eliminar el entorno existente
    --force                 Forzar recreación del entorno si existe
    --update                Solo actualizar dependencias (no recrear entorno)

${COLORS[BOLD]}OPCIONES DE INSTALACIÓN${COLORS[RESET]}
    --skip-pytorch          Omitir instalación de PyTorch
    --skip-embeddings       Omitir generación de embeddings
    --skip-dev              Omitir dependencias de desarrollo
    --dry-run               Mostrar comandos sin ejecutarlos

${COLORS[BOLD]}VARIABLES DE ENTORNO${COLORS[RESET]}
    ENV_NAME                Nombre del entorno conda
    PYTHON_VERSION          Versión de Python
    LOG_DIR                 Directorio de logs
    PYTORCH_INDEX_URL       URL del índice de PyTorch
    VERBOSE                 Habilitar modo debug (true/false)

${COLORS[BOLD]}EJEMPLOS${COLORS[RESET]}
    $SCRIPT_NAME                          # Setup completo
    $SCRIPT_NAME --force                  # Recrear entorno desde cero
    $SCRIPT_NAME --update                 # Solo actualizar dependencias
    $SCRIPT_NAME --skip-pytorch           # Sin PyTorch (más rápido)
    $SCRIPT_NAME --clean                  # Eliminar entorno
    $SCRIPT_NAME -n myenv -p 3.11         # Entorno personalizado
    $SCRIPT_NAME --verbose --dry-run      # Ver qué haría

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
                CLEAN_ENV=true
                shift
                ;;
            --force)
                FORCE_RECREATE=true
                shift
                ;;
            --update)
                UPDATE_ONLY=true
                shift
                ;;
            --skip-pytorch)
                SKIP_PYTORCH=true
                shift
                ;;
            --skip-embeddings)
                SKIP_EMBEDDINGS=true
                shift
                ;;
            --skip-dev)
                SKIP_DEV_DEPS=true
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

    # Validate mutually exclusive options
    if [[ "$FORCE_RECREATE" == "true" ]] && [[ "$UPDATE_ONLY" == "true" ]]; then
        die "--force y --update son mutuamente excluyentes"
    fi

    if [[ "$CLEAN_ENV" == "true" ]] && [[ "$FORCE_RECREATE" == "true" ]]; then
        die "--clean y --force son mutuamente excluyentes (--clean solo elimina)"
    fi
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

main() {
    # Parse CLI arguments first
    parse_arguments "$@"

    # Initialize logging
    setup_logging

    # Setup error handling
    setup_traps

    # Acquire exclusive lock
    acquire_lock

    # Display banner
    log_separator "=" "Configuración de Entorno Local (Conda + UV)"
    log_info "Versión: $SCRIPT_VERSION"
    log_info "Entorno: $ENV_NAME"
    log_info "Python: $PYTHON_VERSION"
    [[ "$VERBOSE" == "true" ]] && log_info "Modo: VERBOSE"
    [[ "$DRY_RUN" == "true" ]] && log_info "Modo: DRY-RUN"
    [[ "$CLEAN_ENV" == "true" ]] && log_info "Modo: CLEAN"
    [[ "$FORCE_RECREATE" == "true" ]] && log_info "Modo: FORCE RECREATE"
    [[ "$UPDATE_ONLY" == "true" ]] && log_info "Modo: UPDATE ONLY"
    echo ""

    # === Validation ===
    check_base_dependencies
    check_conda_installation
    initialize_conda_shell
    check_network_connectivity

    # === Execute Operation ===
    if [[ "$CLEAN_ENV" == "true" ]]; then
        run_clean
    else
        run_setup
        display_final_status
    fi

    log_success "=== Operación completada exitosamente ==="
    return 0
}

# Entry point
main "$@"