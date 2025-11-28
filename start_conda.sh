#!/bin/bash
# ==============================================================================
# Script de Inicialización de Entorno Local (Conda + Uv)
# ==============================================================================

# --- Strict Mode ---
set -euo pipefail

# --- Configuration ---
ENV_NAME="apu_filter_env"
PYTHON_VERSION="3.10"
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/conda_setup_$(date +%Y%m%d_%H%M%S).log"

# --- Colors ---
COLOR_RESET='\033[0m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[0;33m'
COLOR_RED='\033[0;31m'
COLOR_BLUE='\033[0;34m'
COLOR_CYAN='\033[0;36m'

# --- Logging Functions ---
log_info() { echo -e "${COLOR_BLUE}[INFO]${COLOR_RESET} $1" | tee -a "$LOG_FILE"; }
log_step() { echo -e "${COLOR_CYAN}[STEP]${COLOR_RESET} $1" | tee -a "$LOG_FILE"; }
log_success() { echo -e "${COLOR_GREEN}[SUCCESS]${COLOR_RESET} $1" | tee -a "$LOG_FILE"; }
log_warn() { echo -e "${COLOR_YELLOW}[WARN]${COLOR_RESET} $1" | tee -a "$LOG_FILE"; }
log_error() { echo -e "${COLOR_RED}[ERROR]${COLOR_RESET} $1" | tee -a "$LOG_FILE"; }

setup_logging() {
    mkdir -p "$LOG_DIR"
    >"$LOG_FILE"
}

# --- Main Execution ---
main() {
    setup_logging
    log_info "=== Iniciando Configuración de Entorno Local (Conda) ==="

    # 1. Verificar instalación de Conda
    if ! command -v conda &> /dev/null; then
        log_error "Conda no está instalado o no está en el PATH."
        log_info "Por favor instala Miniconda o Anaconda primero."
        exit 1
    fi

    # 2. Inicializar Conda en el script
    # Esto es necesario para poder usar 'conda activate' dentro del script bash
    eval "$(conda shell.bash hook)"

    # 3. Crear o Actualizar Entorno
    if conda info --envs | grep -q "$ENV_NAME"; then
        log_warn "El entorno '$ENV_NAME' ya existe."
        log_step "Activando entorno existente..."
        conda activate "$ENV_NAME"
    else
        log_step "Creando entorno '$ENV_NAME' con Python $PYTHON_VERSION..."
        conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y >> "$LOG_FILE" 2>&1
        log_step "Activando entorno..."
        conda activate "$ENV_NAME"
    fi

    # 4. Instalar Cimientos (Conda Forge / Pytorch)
    log_step "Instalando dependencias base (Faiss, Redis)..."
    # Faiss-cpu es mejor instalarlo por conda para evitar problemas de compilación C++
    conda install -c pytorch -c conda-forge faiss-cpu redis -y >> "$LOG_FILE" 2>&1

    # 5. Instalar UV (Acelerador de PIP)
    log_step "Instalando 'uv' para gestión rápida de paquetes..."
    pip install uv >> "$LOG_FILE" 2>&1

    # 6. Instalar PyTorch (Versión CPU Ligera)
    log_step "Instalando PyTorch (Versión CPU)..."
    # Usamos uv pip install para velocidad
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu >> "$LOG_FILE" 2>&1

    # 7. Instalar Resto de Dependencias
    log_step "Instalando dependencias del proyecto (requirements.txt)..."
    if [ -f "requirements.txt" ]; then
        uv pip install -r requirements.txt >> "$LOG_FILE" 2>&1
    else
        log_warn "No se encontró requirements.txt"
    fi

    if [ -f "requirements-dev.txt" ]; then
        log_step "Instalando dependencias de desarrollo..."
        uv pip install -r requirements-dev.txt >> "$LOG_FILE" 2>&1
    fi

    # 8. Generar Inteligencia (Embeddings)
    log_step "Generando artefactos de Búsqueda Semántica..."
    if [ -f "data/processed_apus.json" ]; then
        python scripts/generate_embeddings.py --input data/processed_apus.json >> "$LOG_FILE" 2>&1
        log_success "Embeddings generados correctamente."
    else
        log_warn "No se encontró 'data/processed_apus.json'. Se omitió la generación de embeddings."
        log_info "Ejecuta el pipeline de carga (upload) para generar los datos base primero."
    fi

    # 9. Resumen Final
    echo ""
    log_success "=== Configuración Completada Exitosamente ==="
    echo -e "Para activar el entorno manualmente, ejecuta:"
    echo -e "${COLOR_GREEN}conda activate $ENV_NAME${COLOR_RESET}"
    echo -e "Para iniciar el servidor:"
    echo -e "${COLOR_GREEN}python -m flask run --port=5002${COLOR_RESET}"
    echo ""
}

main