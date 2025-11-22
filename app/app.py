"""
Módulo principal de la aplicación Flask - VERSIÓN CORREGIDA.

Gestiona el procesamiento de archivos CSV, estimaciones y simulaciones Monte Carlo.
"""

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Optional, Tuple

# --- Dependencias para Búsqueda Semántica ---
import faiss
from flask import Flask, current_app, jsonify, render_template, request, session
from flask_cors import CORS
from markupsafe import escape
from sentence_transformers import SentenceTransformer
from werkzeug.utils import secure_filename

# Configuración del path del sistema
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config_by_name
from models.probability_models import run_monte_carlo_simulation

from .estimator import calculate_estimate, SearchArtifacts
from .presenters import APUPresenter
from .procesador_csv import process_all_files
from .utils import sanitize_for_json

# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================

SESSION_TIMEOUT = 3600  # 1 hora
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB máximo por archivo
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


# ============================================================================
# CONFIGURACIÓN DE LOGGING MEJORADA
# ============================================================================


def setup_logging(app: Flask, log_file: str = "app.log") -> None:
    """
    Configura el sistema de logging con rotación y formatos mejorados.

    Args:
        app: Instancia de Flask.
        log_file: Nombre del archivo de log.
    """
    # Crear directorio de logs si no existe
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Formato detallado para archivo
    file_formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Formato simple para consola
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    # Handler para archivo con rotación
    from logging.handlers import RotatingFileHandler

    file_handler = RotatingFileHandler(
        log_dir / log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    # Configurar logger de la aplicación
    app.logger.handlers.clear()
    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)
    app.logger.setLevel(logging.DEBUG)

    # Configurar logger raíz
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)


# ============================================================================
# DECORADORES Y UTILIDADES - VERSIÓN CORREGIDA
# ============================================================================


def require_session(f):
    """
    Decorador que verifica la existencia y validez de una sesión con datos procesados.
    
    CORRECCIÓN: Ahora usa la interfaz nativa de Flask-Session en lugar de 
    gestión manual de Redis, eliminando el "agujero negro" de sincronización.
    
    Flask-Session automáticamente:
    - Crea la sesión cuando haces session['key'] = value
    - La persiste en Redis con el prefijo configurado
    - La carga automáticamente en cada request
    - Maneja la expiración según PERMANENT_SESSION_LIFETIME
    
    Returns:
        Una respuesta JSON de error con código 401 si no hay datos de sesión,
        o el resultado de la función decorada si la validación es exitosa.
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Verificar que la sesión contiene datos procesados
        if 'processed_data' not in session:
            response = {
                "error": "Sesión no iniciada o datos no encontrados. "
                        "Por favor, cargue los archivos nuevamente.",
                "code": "SESSION_MISSING"
            }
            current_app.logger.warning(
                f"Intento de acceso sin sesión válida desde {request.remote_addr}"
            )
            return jsonify(response), 401

        # Verificar que los datos son válidos (no None ni vacíos)
        if not session['processed_data'] or not isinstance(session['processed_data'], dict):
            response = {
                "error": "Datos de sesión corruptos. Por favor, recargue los archivos.",
                "code": "SESSION_CORRUPTED"
            }
            current_app.logger.error(
                f"Datos de sesión inválidos detectados: {type(session.get('processed_data'))}"
            )
            # Limpiar la sesión corrupta
            session.clear()
            return jsonify(response), 401

        # Preparar datos para la función decorada
        session_data = {"data": session['processed_data']}

        # Marcar la sesión como modificada para refrescar el TTL en Redis
        # Esto extiende automáticamente la vida de la sesión
        session.modified = True

        current_app.logger.debug(
            f"Sesión válida encontrada: {session.sid[:8]}... "
            f"con {len(session['processed_data'])} claves"
        )

        return f(session_data=session_data, *args, **kwargs)

    return decorated_function


def handle_errors(f):
    """Decorador para manejo centralizado de errores."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            current_app.logger.warning(f"Error de validación en {f.__name__}: {str(e)}")
            response = {"error": str(e), "code": "VALIDATION_ERROR"}
            return jsonify(response), 400
        except KeyError as e:
            current_app.logger.error(f"Clave faltante en {f.__name__}: {str(e)}")
            response = {
                "error": f"Dato requerido faltante: {str(e)}",
                "code": "MISSING_KEY"
            }
            return jsonify(response), 400
        except Exception as e:
            current_app.logger.error(
                f"Error no controlado en {f.__name__}: {str(e)}", exc_info=True
            )
            response = {
                "error": "Error interno del servidor",
                "code": "INTERNAL_ERROR"
            }
            return jsonify(response), 500

    return decorated_function


@contextmanager
def temporary_upload_directory(base_path: Path, session_id: str):
    """Context manager para manejar directorios temporales de forma segura."""
    user_dir = base_path / session_id
    user_dir.mkdir(parents=True, exist_ok=True)

    try:
        yield user_dir
    finally:
        # Limpiar archivos temporales
        if user_dir.exists():
            for file in user_dir.iterdir():
                try:
                    file.unlink()
                except Exception as e:
                    logging.warning(f"No se pudo eliminar {file}: {e}")

            try:
                user_dir.rmdir()
            except Exception as e:
                logging.warning(f"No se pudo eliminar directorio {user_dir}: {e}")


# ============================================================================
# VALIDADORES
# ============================================================================


class FileValidator:
    """Valida archivos subidos."""

    @staticmethod
    def validate_file(file) -> Tuple[bool, Optional[str]]:
        """
        Valida que un archivo sea válido para procesamiento.

        Returns:
            Tupla (es_válido, mensaje_error).
        """
        if not file or file.filename == "":
            return False, "Archivo no seleccionado"

        filename = secure_filename(file.filename)
        if not filename:
            return False, "Nombre de archivo inválido"

        extension = Path(filename).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            return False, f"Extensión no permitida: {extension}"

        # Verificar el tamaño del archivo
        file.seek(0, 2)  # Mover al final
        file_size = file.tell()
        file.seek(0)  # Volver al inicio

        if file_size > MAX_CONTENT_LENGTH:
            return False, f"Archivo demasiado grande: {file_size / (1024 * 1024):.2f}MB"

        return True, None


# ============================================================================
# CARGA DE MODELOS DE BÚSQUEDA SEMÁNTICA
# ============================================================================


def load_semantic_search_artifacts(app: Flask):
    """
    Carga el índice FAISS, el mapeo de IDs y el modelo de embeddings.
    Lee el nombre del modelo desde metadata.json para carga dinámica.
    Si los archivos no existen, registra una advertencia y desactiva la función.
    """
    app.logger.info("Iniciando carga de artefactos de búsqueda semántica...")

    embeddings_dir = Path(__file__).parent / "embeddings"
    index_path = embeddings_dir / "faiss.index"
    map_path = embeddings_dir / "id_map.json"
    metadata_path = embeddings_dir / "metadata.json"

    # Establecer valores por defecto en None
    app.config["FAISS_INDEX"] = None
    app.config["ID_MAP"] = None
    app.config["EMBEDDING_MODEL"] = None

    try:
        # 1. Validar que todos los archivos necesarios existan
        if not all([index_path.exists(), map_path.exists(), metadata_path.exists()]):
            app.logger.warning(
                "No se encontraron todos los archivos de embeddings. "
                "La búsqueda semántica estará desactivada. "
                "Ejecute 'scripts/generate_embeddings.py' para generarlos."
            )
            return

        # 2. Cargar metadata para obtener el nombre del modelo dinámicamente
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        model_name = metadata.get("model_name")
        if not model_name:
            raise ValueError("El 'model_name' no se encontró en metadata.json")

        # 3. Cargar los artefactos
        faiss_index = faiss.read_index(str(index_path))

        with open(map_path, "r", encoding="utf-8") as f:
            id_map = json.load(f)

        embedding_model = SentenceTransformer(model_name)

        # 4. Guardar en la configuración de la app solo si todo fue exitoso
        app.config["FAISS_INDEX"] = faiss_index
        app.config["ID_MAP"] = id_map
        app.config["EMBEDDING_MODEL"] = embedding_model

        app.logger.info(
            f"✅ Búsqueda semántica lista. Modelo: '{model_name}', "
            f"Vectores: {faiss_index.ntotal}"
        )

    except Exception as e:
        app.logger.error(
            f"❌ Error crítico al cargar artefactos de búsqueda semántica: {e}",
            exc_info=True,
        )
        # Asegurar que la configuración quede limpia en caso de error
        app.config["FAISS_INDEX"] = None
        app.config["ID_MAP"] = None
        app.config["EMBEDDING_MODEL"] = None
        app.logger.warning(
            "La funcionalidad de búsqueda semántica ha sido desactivada debido a un error."
        )


# ============================================================================
# FACTORY DE APLICACIÓN MEJORADA
# ============================================================================


def create_app(config_name: str) -> Flask:
    """
    Crea y configura una instancia de la aplicación Flask.

    Args:
        config_name: Nombre del entorno de configuración.

    Returns:
        Instancia configurada de Flask.
    """
    app = Flask(__name__, static_folder='../static', static_url_path='/static')

    # Configuración básica
    app.config.from_object(config_by_name[config_name])
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    # SECRET_KEY estable
    app.config["SECRET_KEY"] = os.environ.get(
        "SECRET_KEY", "dev_key_fija_y_secreta_12345"
    )

    # Configurar logging
    setup_logging(app)
    app.logger.info(f"Iniciando aplicación en modo: {config_name}")

    # ========================================================================
    # CONFIGURACIÓN DE SESIÓN Y CORS (VERSIÓN CORREGIDA)
    # ========================================================================

    # 1. CORS Permisivo
    CORS(app, supports_credentials=True)

    # 2. Redis y Flask-Session
    import redis
    from flask_session import Session

    app.config["SESSION_TYPE"] = "redis"
    app.config["SESSION_PERMANENT"] = True
    app.config["SESSION_USE_SIGNER"] = True
    app.config["SESSION_KEY_PREFIX"] = "apu_filter:session:"
    app.config["SESSION_REDIS"] = redis.from_url(app.config["REDIS_URL"])
    app.config["PERMANENT_SESSION_LIFETIME"] = SESSION_TIMEOUT

    # 3. Configuración de Cookie
    app.config["SESSION_COOKIE_NAME"] = "apu_session"
    app.config["SESSION_COOKIE_PATH"] = "/"
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_DOMAIN"] = None

    if config_name == "production":
        app.config["SESSION_COOKIE_SECURE"] = True
        app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    else:
        app.config["SESSION_COOKIE_SECURE"] = False
        app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    Session(app)

    # Cargar configuración específica de la aplicación
    config_path = Path(__file__).parent / "config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            app.config["APP_CONFIG"] = json.load(f)
            app.logger.info("Configuración cargada exitosamente desde config.json")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        app.logger.error(f"Error al cargar config.json: {e}")
        app.config["APP_CONFIG"] = {}

    # Configurar carpeta de plantillas
    project_root = Path(__file__).parent.parent
    app.template_folder = str(project_root / "templates")

    # Crear directorio de uploads si no existe
    upload_folder = Path(app.config.get("UPLOAD_FOLDER", "uploads"))
    upload_folder.mkdir(exist_ok=True)
    app.config["UPLOAD_FOLDER"] = str(upload_folder)

    # Inicializar presentador de APU
    apu_presenter = APUPresenter(app.logger)

    # Cargar artefactos de búsqueda semántica
    with app.app_context():
        load_semantic_search_artifacts(app)

    # ========================================================================
    # MIDDLEWARE Y HOOKS
    # ========================================================================

    @app.before_request
    def before_request_func():
        """Registra información de la solicitud antes de que se procese."""
        app.logger.debug(
            f"Request: {request.method} {request.path} | "
            f"Session: {'✓' if 'processed_data' in session else '✗'} | "
            f"IP: {request.remote_addr}"
        )

    @app.after_request
    def after_request_func(response):
        """Añade cabeceras de seguridad a la respuesta."""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response

    # ========================================================================
    # RUTAS PRINCIPALES - VERSIÓN CORREGIDA
    # ========================================================================

    @app.route("/")
    def index():
        """Renderiza la página principal."""
        return render_template("index.html")

    @app.route("/api/health", methods=["GET"])
    def health_check():
        """Endpoint de verificación de estado."""
        redis_client = app.config["SESSION_REDIS"]

        try:
            # Contar sesiones activas (keys con el prefijo de sesión)
            prefix = app.config["SESSION_KEY_PREFIX"]
            active_sessions = len(redis_client.keys(f"{prefix}*"))
        except Exception as e:
            app.logger.error(f"Error al obtener métricas de Redis: {e}")
            active_sessions = -1

        return jsonify(
            {
                "status": "healthy",
                "active_sessions": active_sessions,
                "timestamp": time.time(),
                "version": app.config.get("APP_CONFIG", {}).get("version", "1.0.0"),
                "session_timeout": SESSION_TIMEOUT,
            }
        )

    @app.route("/upload", methods=["POST"])
    @handle_errors
    def upload_files():
        """
        Gestiona la carga de archivos de presupuesto, APU e insumos.
        
        CORRECCIÓN: Ahora guarda los datos en session['processed_data'] 
        en lugar de gestionar Redis manualmente.

        Returns:
            Un objeto JSON con los datos procesados si el proceso es exitoso,
            o un mensaje de error en caso contrario.
        """
        # Validar archivos requeridos
        required_files = ["presupuesto", "apus", "insumos"]
        missing_files = [f for f in required_files if f not in request.files]

        if missing_files:
            return jsonify(
                {
                    "error": f"Faltan archivos: {', '.join(missing_files)}",
                    "code": "MISSING_FILES",
                }
            ), 400

        # Validar cada archivo
        file_validator = FileValidator()
        files_to_process = {}

        for file_name in required_files:
            file = request.files[file_name]
            is_valid, error_msg = file_validator.validate_file(file)

            if not is_valid:
                return jsonify(
                    {
                        "error": f"Error en archivo {file_name}: {error_msg}",
                        "code": "INVALID_FILE",
                    }
                ), 400

            files_to_process[file_name] = file

        # Flask-Session crea automáticamente el session_id al asignar datos
        # Generar un ID temporal para el directorio de uploads
        import uuid
        temp_id = str(uuid.uuid4())

        # Procesar archivos en directorio temporal
        upload_path = Path(app.config["UPLOAD_FOLDER"])

        with temporary_upload_directory(upload_path, temp_id) as user_dir:
            file_paths = {}

            # Guardar archivos temporalmente
            for name, file in files_to_process.items():
                filename = secure_filename(file.filename)
                file_path = user_dir / filename
                file.save(str(file_path))
                file_paths[name] = str(file_path)

            # Procesar archivos
            app.logger.info("Procesando archivos para nueva sesión...")

            processed_data = process_all_files(
                file_paths["presupuesto"],
                file_paths["apus"],
                file_paths["insumos"],
                config=app.config.get("APP_CONFIG", {}),
            )

        # Verificar errores en el procesamiento
        if "error" in processed_data:
            app.logger.error(f"Error de procesamiento: {processed_data['error']}")
            return jsonify(
                {"error": processed_data["error"], "code": "PROCESSING_ERROR"}
            ), 500

        # ===================================================================
        # CAMBIO CRÍTICO: Guardar en session en lugar de Redis manual
        # ===================================================================

        # Sanitizar datos para JSON
        sanitized_data = sanitize_for_json(processed_data)

        # Guardar en la sesión de Flask
        # Flask-Session automáticamente:
        # 1. Crea el session.sid si no existe
        # 2. Serializa los datos
        # 3. Los guarda en Redis con el prefijo configurado
        # 4. Crea/actualiza la cookie en el navegador
        session['processed_data'] = sanitized_data
        session.permanent = True  # Usar PERMANENT_SESSION_LIFETIME

        app.logger.info(
            f"✅ Archivos procesados y guardados en sesión {session.sid[:8]}... | "
            f"Presupuesto: {len(sanitized_data.get('presupuesto', []))} ítems | "
            f"APUs: {len(sanitized_data.get('apus_detail', []))} detalles"
        )

        # Preparar respuesta
        response_data = sanitized_data.copy()
        response_data["session_id"] = session.sid
        response_data["session_timeout"] = SESSION_TIMEOUT

        return jsonify(response_data)

    @app.route("/api/apu/<code>", methods=["GET"])
    @require_session
    @handle_errors
    def get_apu_detail(code: str, session_data: dict = None):
        """
        Recupera y procesa los detalles de un Análisis de Precios Unitarios (APU).

        Args:
            code: El código del APU a consultar.
            session_data: Los datos de la sesión del usuario, inyectados por el
                          decorador `require_session`.

        Returns:
            Un objeto JSON con el desglose detallado del APU, los resultados de
            la simulación y metadatos asociados.
        """
        app.logger.info(f"Solicitud de detalle para APU: {code}")

        # Decodificar código
        apu_code = code.replace("%2C", ",")

        # Obtener datos del APU
        user_data = session_data["data"]
        all_apu_details = user_data.get("apus_detail", [])

        # Filtrar por código (lógica de búsqueda robusta)
        apu_details = [
            item for item in all_apu_details if item.get("CODIGO_APU") == apu_code
        ]

        if not apu_details:
            # Intento alternativo (cambiar punto por coma o viceversa)
            alt_code = (
                apu_code.replace(".", ",") if "." in apu_code
                else apu_code.replace(",", ".")
            )
            apu_details = [
                item for item in all_apu_details if item.get("CODIGO_APU") == alt_code
            ]
            if apu_details:
                apu_code = alt_code

        if not apu_details:
            app.logger.warning(f"APU no encontrado con ninguna variante: {code}")
            return jsonify(
                {"error": f"APU no encontrado: {code}", "code": "APU_NOT_FOUND"}
            ), 404

        # Procesar detalles del APU
        processed_data = apu_presenter.process_apu_details(apu_details, apu_code)

        # Obtener información del presupuesto
        presupuesto_data = user_data.get("presupuesto", [])
        presupuesto_item = next(
            (item for item in presupuesto_data if item.get("CODIGO_APU") == apu_code),
            None
        )

        # Ejecutar simulación Monte Carlo
        simulation_results = run_monte_carlo_simulation(processed_data["items"])

        # Preparar respuesta
        response = {
            "codigo": apu_code,
            "descripcion": (
                presupuesto_item.get("original_description", "")
                if presupuesto_item
                else ""
            ),
            "desglose": processed_data["desglose"],
            "simulation": simulation_results,
            "metadata": {
                "total_items": processed_data["total_items"],
                "categorias": list(processed_data["desglose"].keys()),
            },
        }

        app.logger.info(
            f"Detalle para APU {apu_code} generado exitosamente "
            f"con {processed_data['total_items']} items"
        )

        return jsonify(sanitize_for_json(response))

    @app.route("/api/estimate", methods=["POST"])
    @require_session
    @handle_errors
    def get_estimate(session_data: dict = None):
        """
        Calcula una estimación de costos y rendimientos para un proyecto.

        Args:
            session_data: Los datos de la sesión del usuario, inyectados por el
                          decorador `require_session`.

        Returns:
            Un objeto JSON con los resultados de la estimación.
        """
        # Validar request
        if not request.is_json:
            return jsonify(
                {
                    "error": "Content-Type debe ser application/json",
                    "code": "INVALID_CONTENT_TYPE",
                }
            ), 400

        params = request.get_json()
        if not params:
            return jsonify(
                {"error": "No se proporcionaron parámetros", "code": "NO_PARAMS"}
            ), 400

        app.logger.info(f"Solicitud de estimación con parámetros: {params}")

        # Construir artefactos de búsqueda
        search_artifacts = SearchArtifacts(
            model=app.config.get("EMBEDDING_MODEL"),
            faiss_index=app.config.get("FAISS_INDEX"),
            id_map=app.config.get("ID_MAP"),
        )

        # Calcular estimación
        user_data = session_data["data"]

        result = calculate_estimate(
            params=params,
            data_store=user_data,
            config=app.config.get("APP_CONFIG", {}),
            search_artifacts=search_artifacts
        )

        if "error" in result:
            app.logger.warning(f"Error en estimación: {result['error']}")
            return jsonify(result), 400

        app.logger.info(
            f"Estimación calculada: Construcción="
            f"${result.get('valor_construccion', 0):,.2f}, "
            f"Rendimiento={result.get('rendimiento_m2_por_dia', 0):.2f}"
        )

        return jsonify(sanitize_for_json(result))

    @app.route("/api/session/clear", methods=["POST"])
    def clear_session():
        """
        Limpia la sesión actual del usuario.
        Útil para debugging y permitir que el usuario "reinicie" sin cerrar el navegador.
        """
        had_session = 'processed_data' in session
        session.clear()

        return jsonify({
            "success": True,
            "message": "Sesión limpiada exitosamente" if had_session else "No había sesión activa",
            "had_session": had_session
        })

    @app.route("/api/session/info", methods=["GET"])
    def session_info():
        """
        Endpoint de debugging para inspeccionar el estado de la sesión.
        NOTA: Deshabilitar en producción por seguridad.
        """
        if config_name == "production":
            return jsonify({"error": "Endpoint deshabilitado en producción"}), 403

        return jsonify({
            "sid": session.sid if hasattr(session, 'sid') else None,
            "has_data": 'processed_data' in session,
            "data_keys": list(session.get('processed_data', {}).keys()) if 'processed_data' in session else [],
            "permanent": session.permanent,
            "modified": session.modified,
        })

    # ========================================================================
    # MANEJADORES DE ERRORES
    # ========================================================================

    @app.errorhandler(404)
    def not_found(error):
        """Maneja errores 404."""
        return jsonify(
            {
                "error": "Recurso no encontrado",
                "code": "NOT_FOUND",
                "path": escape(request.path),
            }
        ), 404

    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Maneja errores de tamaño de archivo."""
        max_mb = MAX_CONTENT_LENGTH / (1024 * 1024)
        return jsonify(
            {
                "error": f"Archivo demasiado grande. Máximo: {max_mb:.1f}MB",
                "code": "FILE_TOO_LARGE",
            }
        ), 413

    @app.errorhandler(500)
    def internal_error(error):
        """Maneja errores internos del servidor."""
        app.logger.error(f"Error 500: {str(error)}", exc_info=True)
        return jsonify(
            {
                "error": "Error interno del servidor",
                "code": "INTERNAL_ERROR",
                "message": "Por favor, contacte al administrador si el problema persiste",
            }
        ), 500

    return app


# ============================================================================
# PUNTO DE ENTRADA PARA DESARROLLO
# ============================================================================

if __name__ == "__main__":
    # Solo para desarrollo/debugging
    app = create_app("development")
    app.run(debug=True, host="0.0.0.0", port=5000)
