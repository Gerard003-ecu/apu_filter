import os
import sys

import pytest

# Añadir el directorio raíz al path para asegurar que los módulos se encuentren
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.app import create_app


@pytest.fixture(scope="module")
def app():
    """Crea y configura una nueva instancia de la aplicación para cada módulo de prueba."""
    # Usar la configuración de 'testing'
    app = create_app("testing")

    # Establecer el contexto de la aplicación
    with app.app_context():
        yield app


@pytest.fixture(scope="module")
def client(app):
    """Un cliente de prueba para la aplicación."""
    return app.test_client()
