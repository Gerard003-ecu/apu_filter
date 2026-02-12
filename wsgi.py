import os
from app.app import create_app

# En el contenedor, siempre queremos producción a menos que se diga lo contrario
config_name = os.getenv("CONFIG_NAME", "production")
app = create_app(config_name)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)