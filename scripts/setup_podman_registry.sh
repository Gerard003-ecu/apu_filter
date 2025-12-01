#!/bin/bash
# scripts/setup_podman_registry.sh

# Crear directorio de configuración si no existe
mkdir -p ~/.config/containers/registries.conf.d/

# Crear archivo de configuración
cat > ~/.config/containers/registries.conf.d/000-local.conf << 'EOF'
# =============================================================================
# Configuración de Registries para Podman
# =============================================================================
# Este archivo permite usar nombres cortos (sin prefijo de registry)
# para imágenes construidas localmente.
# =============================================================================

# Permitir nombres cortos sin preguntar
# Opciones: "enforcing" (rechaza), "permissive" (permite), "disabled"
short-name-mode = "permissive"

# Lista de registries donde buscar imágenes sin prefijo
# (en orden de prioridad)
unqualified-search-registries = ["docker.io"]
EOF

echo "Configuración de registros de Podman aplicada correctamente."
