import os

from dotenv import load_dotenv

# Cargar variables de entorno desde un archivo .env
load_dotenv()


class Config:
    """Configuración base."""

    SECRET_KEY = os.environ.get("SECRET_KEY") or "una-clave-secreta-muy-dificil-de-adivinar"
    DEBUG = False
    TESTING = False
    # Otras configuraciones globales
    UPLOAD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "uploads")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    REDIS_URL = os.environ.get("REDIS_URL") or "redis://127.0.0.1:6379/0"


class DevelopmentConfig(Config):
    """Configuración para desarrollo."""

    DEBUG = True
    FLASK_ENV = "development"


class ProductionConfig(Config):
    """Configuración para producción."""

    FLASK_ENV = "production"
    # Aquí se podrían añadir configuraciones específicas de producción,
    # como la configuración de una base de datos, logging, etc.


class TestingConfig(Config):
    """Configuración para pruebas."""

    TESTING = True
    SECRET_KEY = "test-secret-key"
    UPLOAD_FOLDER = "test_uploads"


# Mapeo de nombres de configuración a clases
config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}
