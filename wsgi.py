import os

from app.app import create_app

# Obtener el entorno de la variable de entorno FLASK_ENV.
# Si no está definida, se usará 'development' por defecto.
config_name = os.getenv("FLASK_ENV", "default")
app = create_app(config_name)

if __name__ == "__main__":
    # Esta parte es útil para ejecutar la aplicación directamente con `python wsgi.py`
    # para pruebas locales en un entorno de producción simulado.
    # Gunicorn no usará esta sección.
    app.run(use_reloader=False)
