#!/usr/bin/env bash

# ==============================================================================
# Script de Configuración de Entorno APU Filter (Objetivo 1)
# ==============================================================================
set -euo pipefail

echo "[INFO] Iniciando configuración de entorno automatizada..."

# 1. Descargar e Instalar Miniconda
if [ ! -d "$HOME/miniconda3" ]; then
    echo "[STEP 1] Descargando Miniconda..."
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

    echo "[STEP 1] Instalando Miniconda..."
    bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"

    echo "[STEP 1] Inicializando Conda..."
    "$HOME/miniconda3/bin/conda" init bash
else
    echo "[INFO] Miniconda ya está instalado."
fi

# 2. Aceptar ToS de Anaconda
echo "[STEP 2] Aceptando Términos de Servicio de Anaconda..."
"$HOME/miniconda3/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
"$HOME/miniconda3/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# 3. Ejecutar start_conda.sh para levantar el entorno
echo "[STEP 3] Ejecutando start_conda.sh para crear el entorno 'apu_filter_env'..."
chmod +x start_conda.sh
./start_conda.sh

# 4. Instalar dependencias adicionales si es necesario
echo "[STEP 4] Instalando dependencias de análisis..."
"$HOME/miniconda3/envs/apu_filter_env/bin/pip" install objgraph

echo "[SUCCESS] Entorno configurado correctamente."
echo "[INFO] Para activar el entorno use: conda activate apu_filter_env"
