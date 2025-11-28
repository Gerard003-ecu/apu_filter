#!/bin/bash
set -e

echo "=== Iniciando APU Filter (Podman) ==="

COMPOSE_FILE="infrastructure/compose.yaml"

if [ ! -f "$COMPOSE_FILE" ]; then
    echo "❌ Error: No se encuentra $COMPOSE_FILE"
    exit 1
fi

echo "🚀 Construyendo y levantando..."
podman-compose -f "$COMPOSE_FILE" up --build -d

echo "✅ Estado:"
podman-compose -f "$COMPOSE_FILE" ps
