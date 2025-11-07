"""
Módulo principal de la aplicación Flask.
Gestiona el procesamiento de archivos CSV, estimaciones y simulaciones Monte Carlo.
"""

import json
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, session
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# Configuración del path del sistema
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config_by_name
from models.probability_models import run_monte_carlo_simulation
from .estimator import calculate_estimate
from .procesador_csv import process_all_files
from .utils import sanitize_for_json

# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================

SESSION_TIMEOUT = 3600  # 1 hora
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB máximo por archivo
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}

# ============================================================================
# GESTOR DE SESIONES MEJORADO
# ============================================================================

class SessionManager:
    """Gestiona las sesiones de usuario con expiración automática."""
    
    def __init__(self, timeout: int = SESSION_TIMEOUT):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self.timeout = timeout
        self._logger = logging.getLogger(__name__)
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Crea una nueva sesión con ID único."""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        self._sessions[session_id] = {
            "data": {},
            "last_activity": time.time(),
            "created_at": time.time()
        }
        self._logger.info(f"Nueva sesión creada: {session_id[:8]}...")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene los datos de una sesión si existe y no ha expirado."""
        if session_id not in self._sessions:
            return None
        
        session_data = self._sessions[session_id]
        if self._is_expired(session_data):
            self.remove_session(session_id)
            return None
        
        # Actualizar actividad
        session_data["last_activity"] = time.time()
        return session_data
    
    def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Actualiza los datos de una sesión existente."""
        if session_id not in self._sessions:
            return False
        
        self._sessions[session_id]["data"] = data
        self._sessions[session_id]["last_activity"] = time.time()
        return True
    
    def remove_session(self, session_id: str) -> bool:
        """Elimina una sesión del almacenamiento."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._logger.debug(f"Sesión eliminada: {session_id[:8]}...")
            return True
        return False
    
    def cleanup_expired_sessions(self) -> int:
        """Elimina todas las sesiones expiradas y retorna el número eliminado."""
        current_time = time.time()
        expired = [
            sid for sid, data in self._sessions.items()
            if self._is_expired(data)
        ]
        
        for sid in expired:
            self.remove_session(sid)
        
        if expired:
            self._logger.info(f"Limpieza: {len(expired)} sesiones expiradas eliminadas")
        
        return len(expired)
    
    def _is_expired(self, session_data: Dict[str, Any]) -> bool:
        """Verifica si una sesión ha expirado."""
        return time.time() - session_data.get("last_activity", 0) > self.timeout
    
    @property
    def active_count(self) -> int:
        """Retorna el número de sesiones activas."""
        return len(self._sessions)

# Instancia global del gestor de sesiones
session_manager = SessionManager()

# ============================================================================
# CONFIGURACIÓN DE LOGGING MEJORADA
# ============================================================================

def setup_logging(app: Flask, log_file: str = "app.log") -> None:
    """
    Configura el sistema de logging con rotación y formatos mejorados.
    
    Args:
        app: Instancia de Flask
        log_file: Nombre del archivo de log
    """
    # Crear directorio de logs si no existe
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Formato detallado para archivo
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Formato simple para consola
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Handler para archivo con rotación
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_dir / log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
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
# DECORADORES Y UTILIDADES
# ============================================================================

def require_session(f):
    """Decorador que requiere una sesión válida para acceder al endpoint."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        session_id = session.get("session_id")
        if not session_id:
            return jsonify({"error": "Sesión no iniciada", "code": "NO_SESSION"}), 401
        
        session_data = session_manager.get_session(session_id)
        if not session_data:
            return jsonify({"error": "Sesión expirada", "code": "SESSION_EXPIRED"}), 401
        
        # Pasar los datos de sesión a la función
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
            return jsonify({"error": str(e), "code": "VALIDATION_ERROR"}), 400
        except KeyError as e:
            current_app.logger.error(f"Clave faltante en {f.__name__}: {str(e)}")
            return jsonify({"error": f"Dato requerido faltante: {str(e)}", "code": "MISSING_KEY"}), 400
        except Exception as e:
            current_app.logger.error(f"Error no controlado en {f.__name__}: {str(e)}", exc_info=True)
            return jsonify({"error": "Error interno del servidor", "code": "INTERNAL_ERROR"}), 500
    
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
            Tupla (es_válido, mensaje_error)
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
            return False, f"Archivo demasiado grande: {file_size / (1024*1024):.2f}MB"
        
        return True, None

# ============================================================================
# PROCESADORES DE DATOS MEJORADOS
# ============================================================================

class APUProcessor:
    """Procesa y agrupa datos de APU."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def process_apu_details(self, apu_details: list, apu_code: str) -> dict:
        """
        Procesa los detalles de un APU específico.
        
        Args:
            apu_details: Lista de detalles del APU
            apu_code: Código del APU
            
        Returns:
            Diccionario con los datos procesados
        """
        if not apu_details:
            raise ValueError(f"No se encontraron detalles para el APU {apu_code}")
        
        # Convertir a DataFrame y sanitizar
        df = pd.DataFrame(apu_details)
        df.replace({np.nan: None}, inplace=True)
        
        # Procesar por categorías
        processed_items = self._group_by_category(df)
        
        # Organizar desglose
        desglose = self._organize_breakdown(processed_items)
        
        return {
            "items": processed_items,
            "desglose": desglose,
            "total_items": len(processed_items)
        }
    
    def _group_by_category(self, df: pd.DataFrame) -> list:
        """Agrupa los items por categoría y descripción."""
        processed = []
        
        for categoria in df["CATEGORIA"].unique():
            df_categoria = df[df["CATEGORIA"] == categoria]
            
            if df_categoria.empty:
                continue
            
            # Definir agregaciones
            aggregations = {
                'CANTIDAD_APU': 'sum',
                'VALOR_TOTAL_APU': 'sum',
                'RENDIMIENTO': 'sum',
                'UNIDAD_APU': 'first',
                'PRECIO_UNIT_APU': 'first',
                'CATEGORIA': 'first',
                'CODIGO_APU': 'first',
                'UNIDAD_INSUMO': 'first'
            }
            
            # Agregar manejo de alertas si existe
            if "alerta" in df_categoria.columns:
                aggregations["alerta"] = lambda x: " | ".join(x.dropna().unique())
            
            # Agrupar
            df_grouped = (
                df_categoria.groupby("DESCRIPCION_INSUMO")
                .agg(aggregations)
                .reset_index()
            )
            
            # Renombrar columnas
            df_grouped.rename(columns={
                "DESCRIPCION_INSUMO": "DESCRIPCION",
                "CANTIDAD_APU": "CANTIDAD",
                "VALOR_TOTAL_APU": "VR_TOTAL",
                "UNIDAD_INSUMO": "UNIDAD",
                "PRECIO_UNIT_APU": "VR_UNITARIO"
            }, inplace=True)
            
            processed.extend(df_grouped.to_dict("records"))
        
        return processed
    
    def _organize_breakdown(self, items: list) -> dict:
        """Organiza los items en un desglose por categoría."""
        desglose = {}
        for item in items:
            categoria = item.get("CATEGORIA", "INDEFINIDO")
            if categoria not in desglose:
                desglose[categoria] = []
            desglose[categoria].append(item)
        
        return desglose

# ============================================================================
# FACTORY DE APLICACIÓN MEJORADA
# ============================================================================

def create_app(config_name: str) -> Flask:
    """
    Crea y configura una instancia de la aplicación Flask.
    
    Args:
        config_name: Nombre del entorno de configuración
        
    Returns:
        Instancia configurada de Flask
    """
    app = Flask(__name__)
    
    # Configuración básica
    app.config.from_object(config_by_name[config_name])
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
    app.config['SESSION_COOKIE_SECURE'] = config_name == 'production'
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    
    # Configurar logging
    setup_logging(app)
    app.logger.info(f"Iniciando aplicación en modo: {config_name}")
    
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
    
    # Inicializar procesador de APU
    apu_processor = APUProcessor(app.logger)
    
    # ========================================================================
    # MIDDLEWARE Y HOOKS
    # ========================================================================
    
    @app.before_request
    def before_request_func():
        """Mantenimiento antes de cada solicitud."""
        # Limpiar sesiones expiradas cada 100 requests (optimización)
        if hasattr(app, 'request_count'):
            app.request_count += 1
        else:
            app.request_count = 1
        
        if app.request_count % 100 == 0:
            session_manager.cleanup_expired_sessions()
        
        # Logging de request
        app.logger.debug(f"Request: {request.method} {request.path}")
    
    @app.after_request
    def after_request_func(response):
        """Procesamiento después de cada solicitud."""
        # Headers de seguridad
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # CORS si es necesario (configurar según necesidades)
        if app.config.get('ENABLE_CORS'):
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        
        return response
    
    # ========================================================================
    # RUTAS PRINCIPALES
    # ========================================================================
    
    @app.route("/")
    def index():
        """Renderiza la página principal."""
        return render_template("index.html")
    
    @app.route("/api/health", methods=["GET"])
    def health_check():
        """Endpoint de verificación de estado."""
        return jsonify({
            "status": "healthy",
            "active_sessions": session_manager.active_count,
            "timestamp": time.time(),
            "version": app.config.get("APP_CONFIG", {}).get("version", "1.0.0")
        })
    
    @app.route("/upload", methods=["POST"])
    @handle_errors
    def upload_files():
        """
        Maneja la carga y procesamiento de archivos.
        
        Returns:
            JSON con los datos procesados o error
        """
        # Validar archivos requeridos
        required_files = ["presupuesto", "apus", "insumos"]
        missing_files = [f for f in required_files if f not in request.files]
        
        if missing_files:
            return jsonify({
                "error": f"Faltan archivos: {', '.join(missing_files)}",
                "code": "MISSING_FILES"
            }), 400
        
        # Validar cada archivo
        file_validator = FileValidator()
        files_to_process = {}
        
        for file_name in required_files:
            file = request.files[file_name]
            is_valid, error_msg = file_validator.validate_file(file)
            
            if not is_valid:
                return jsonify({
                    "error": f"Error en archivo {file_name}: {error_msg}",
                    "code": "INVALID_FILE"
                }), 400
            
            files_to_process[file_name] = file
        
        # Crear o recuperar sesión
        if "session_id" not in session:
            session["session_id"] = session_manager.create_session()
        
        session_id = session["session_id"]
        
        # Procesar archivos en directorio temporal
        upload_path = Path(app.config["UPLOAD_FOLDER"])
        
        with temporary_upload_directory(upload_path, session_id) as user_dir:
            file_paths = {}
            
            # Guardar archivos temporalmente
            for name, file in files_to_process.items():
                filename = secure_filename(file.filename)
                file_path = user_dir / filename
                file.save(str(file_path))
                file_paths[name] = str(file_path)
            
            # Procesar archivos
            app.logger.info(f"Procesando archivos para sesión {session_id[:8]}...")
            
            processed_data = process_all_files(
                file_paths["presupuesto"],
                file_paths["apus"],
                file_paths["insumos"],
                config=app.config.get("APP_CONFIG", {})
            )
        
        # Verificar errores en el procesamiento
        if "error" in processed_data:
            app.logger.error(f"Error de procesamiento: {processed_data['error']}")
            return jsonify({
                "error": processed_data['error'],
                "code": "PROCESSING_ERROR"
            }), 500
        
        # Actualizar sesión con datos procesados
        session_manager.update_session(session_id, processed_data)
        
        app.logger.info(f"Archivos procesados exitosamente para sesión {session_id[:8]}")
        
        # Preparar respuesta
        response_data = sanitize_for_json(processed_data)
        response_data["session_id"] = session_id[:8]  # Enviar ID parcial por seguridad
        
        return jsonify(response_data)
    
    @app.route("/api/apu/<code>", methods=["GET"])
    @require_session
    @handle_errors
    def get_apu_detail(code: str, session_data: dict = None):
        """
        Obtiene detalles de un APU específico.
        
        Args:
            code: Código del APU
            session_data: Datos de la sesión (inyectado por decorador)
            
        Returns:
            JSON con detalles del APU
        """
        app.logger.info(f"Solicitud de detalle para APU: {code}")
        
        # Decodificar código
        apu_code = code.replace("%2C", ",")
        
        # Obtener datos del APU
        user_data = session_data["data"]
        all_apu_details = user_data.get("apus_detail", [])
        
        # Filtrar por código
        apu_details = [
            item for item in all_apu_details 
            if item.get("CODIGO_APU") == apu_code
        ]
        
        if not apu_details:
            app.logger.warning(f"APU no encontrado: {apu_code}")
            return jsonify({
                "error": f"APU no encontrado: {apu_code}",
                "code": "APU_NOT_FOUND"
            }), 404
        
        # Procesar detalles del APU
        processed_data = apu_processor.process_apu_details(apu_details, apu_code)
        
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
            "descripcion": presupuesto_item.get("original_description", "") if presupuesto_item else "",
            "desglose": processed_data["desglose"],
            "simulation": simulation_results,
            "metadata": {
                "total_items": processed_data["total_items"],
                "categorias": list(processed_data["desglose"].keys())
            }
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
        Calcula estimación basada en parámetros.
        
        Args:
            session_data: Datos de la sesión (inyectado por decorador)
            
        Returns:
            JSON con resultados de la estimación
        """
        # Validar request
        if not request.is_json:
            return jsonify({
                "error": "Content-Type debe ser application/json",
                "code": "INVALID_CONTENT_TYPE"
            }), 400
        
        params = request.get_json()
        if not params:
            return jsonify({
                "error": "No se proporcionaron parámetros",
                "code": "NO_PARAMS"
            }), 400
        
        app.logger.info(f"Solicitud de estimación con parámetros: {params}")
        
        # Calcular estimación
        user_data = session_data["data"]
        result = calculate_estimate(
            params,
            user_data,
            app.config.get("APP_CONFIG", {})
        )
        
        if "error" in result:
            app.logger.warning(f"Error en estimación: {result['error']}")
            return jsonify(result), 400
        
        app.logger.info(
            f"Estimación calculada: Construcción=${result.get('valor_construccion', 0):,.2f}, "
            f"Rendimiento={result.get('rendimiento_m2_por_dia', 0):.2f}"
        )
        
        return jsonify(sanitize_for_json(result))
    
    # ========================================================================
    # MANEJADORES DE ERRORES
    # ========================================================================
    
    @app.errorhandler(404)
    def not_found(error):
        """Maneja errores 404."""
        return jsonify({
            "error": "Recurso no encontrado",
            "code": "NOT_FOUND",
            "path": request.path
        }), 404
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Maneja errores de tamaño de archivo."""
        return jsonify({
            "error": f"Archivo demasiado grande. Máximo: {MAX_CONTENT_LENGTH / (1024*1024):.1f}MB",
            "code": "FILE_TOO_LARGE"
        }), 413
    
    @app.errorhandler(500)
    def internal_error(error):
        """Maneja errores internos del servidor."""
        app.logger.error(f"Error 500: {str(error)}", exc_info=True)
        return jsonify({
            "error": "Error interno del servidor",
            "code": "INTERNAL_ERROR",
            "message": "Por favor, contacte al administrador si el problema persiste"
        }), 500
    
    return app


# ============================================================================
# PUNTO DE ENTRADA PARA DESARROLLO
# ============================================================================

if __name__ == "__main__":
    # Solo para desarrollo/debugging
    app = create_app("development")
    app.run(debug=True, host="0.0.0.0", port=5000)