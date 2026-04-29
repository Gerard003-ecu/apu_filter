#!/usr/bin/env bash

# ==============================================================================
# GOBERNANZA CI/CD: ORQUESTADOR DE VARIEDAD DIFERENCIABLE DIKΩαW
# ==============================================================================
# Este script impone la Ley de Clausura Transitiva. Su ejecución es monótona:
# 1. Esterilización del espacio vectorial (Thread Isolation).
# 2. Certificación del Funtor Estático (Transitive Closure).
# 3. Evaluación Asintótica bajo Estrés (Dynamic Ergodicity).
# ==============================================================================

set -euo pipefail

# Find python in the conda environment
PYTHON_BIN="$HOME/miniconda3/envs/apu_filter_env/bin/python"

echo "[⚙️] FASE 0: Induciendo Vacío Termodinámico (Aislamiento OMP/BLAS/LAPACK)..."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Variables de entorno adicionales para purga de memoria
export PYTHONHASHSEED=42  # Determinismo en diccionarios
export PYTHONDONTWRITEBYTECODE=1

echo -e "\n[📐] FASE I: Verificación del Funtor Estático (Cohomología y Clausura Transitiva)..."
# Ejecuta validaciones estructurales, topológicas y categóricas.
# Se excluyen explícitamente las métricas de estrés y de baja velocidad.
$PYTHON_BIN -m pytest tests/integration/boole/test_gamma_transitive_closure.py \
    -v \
    --tb=short \
    -m "integration and not stress and not slow"

# El flag 'set -e' garantiza que si el comando anterior no retorna 0,
# el script colapsa aquí. Se respeta el Fast-Fail axiomático.

echo -e "\n[🌪️] FASE II: Inyección de Turbulencia Estocástica (Ergodicidad Dinámica)..."
# Si la topología es estable (β₁=0, λ₂>0), procedemos a someter el colector
# a vuelos de Lévy y evaluar el exponente máximo de Lyapunov.
$PYTHON_BIN -m pytest tests/integration/dynamic_stress/test_gamma_dynamic_ergodicity.py \
    -v \
    --tb=short \
    -m "integration and stress"

echo -e "\n[✅] EL COLAPSO DE LA FUNCIÓN DE ONDA ES EXITOSO."
echo "La Variedad Diferenciable preserva sus invariantes topológicos y termodinámicos."
exit 0
