"""
Este módulo actúa como el "Plano de Control" expuesto al mundo exterior. No procesa
la lógica de negocio profunda (eso lo hacen los Sabios), sino que establece el entorno
físico y temporal necesario para que el Consejo delibere. Es el responsable de instanciar
el "Pasaporte de Telemetría" y mantener la integridad del estado entre transacciones.

Arquitectura y Responsabilidades:
---------------------------------
1. Transparencia "Glass Box" (Caja de Cristal):
   Implementa una estrategia de observabilidad total donde cada decisión del sistema
   es trazable y auditable, exponiendo no solo el resultado final sino la narrativa
   física y matemática que condujo a él.

2. Gestión de Espacio-Tiempo (SessionMetadata):
   Administra el ciclo de vida de la deliberación. Utiliza `SessionMetadata` para
   garantizar la continuidad temporal (persistencia en Redis) y la integridad
   referencial (hashing de datos) entre las distintas fases del pipeline (Carga ->
   Diagnóstico -> Estimación).

2. Inyección de Telemetría ("El Pasaporte"):
   Inicializa el `TelemetryContext` [1] que viaja con cada solicitud. Este contexto
   actúa como un pasaporte diplomático que acumula sellos (métricas, errores)
   mientras atraviesa las fronteras de los diferentes microservicios (Guardian,
   Arquitecto, Alquimista).

3. Válvulas de Presión (Rate Limiting):
   Implementa límites de velocidad adaptativos (`@limiter`) que actúan como
   resistencias variables en el circuito de flujo de datos, protegiendo al
   `FluxPhysicsEngine` de sobrecargas térmicas por exceso de solicitudes [2].

4. Despacho de Herramientas (Tool Dispatcher):
   Expone los vectores de actuación de la Matriz de Interacción Central (MIC),
   permitiendo invocar capacidades específicas como `diagnose_file` (Diagnóstico)
   o `financial_analysis` (Oráculo) bajo demanda [3].
"""

import hashlib
import json
import logging
import mimetypes
import os

# Forzar modo secuencial para evitar deadlocks con Gunicorn
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from flask import Flask, current_app, g, jsonify, render_template, request, session
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from markupsafe import escape
from werkzeug.utils import secure_filename

# Configuración del path del sistema
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config_app import config_by_name
from models.probability_models import run_monte_carlo_simulation

from .data_loader import LoadStatus, load_data  # Nueva importación para carga robusta
from .pipeline_director import process_all_files  # Ahora usa la versión refactorizada
from .presenters import APUPresenter
from .telemetry import TelemetryContext  # Nueva importación
from .data_validator import PyramidalValidator  # Importar el validador piramidal
from .tools_interface import (
    MICRegistry,
    analyze_financial_viability,
    clean_file,
    diagnose_file,
    get_telemetry_status,
    register_core_vectors,
)
from .schemas import Stratum
from .topology_viz import topology_bp  # Importar el nuevo blueprint
from .utils import sanitize_for_json
from .laplace_oracle import LaplaceOracle, ConfigurationError as OracleConfigurationError

# ============================================================================
# INYECCIÓN DE SALUD PIRAMIDAL
# ============================================================================


def _inject_pyramidal_health(response_data: dict, session_data: dict):
    """
    Inyecta el diagnóstico de estabilidad piramidal en la respuesta API.
    Fuente: LENGUAJE_CONSEJO.md (Termodinámica Estructural) [7]
    Utiliza PyramidalValidator para centralizar la lógica.
    """
    try:
        # Obtener DataFrames de la sesión
        # Nota: processed_apus es lista de dicts, raw_insumos_df es DataFrame
        processed_apus = session_data["data"].get("processed_apus", [])
        raw_insumos_df = session_data["data"].get("raw_insumos_df")

        # Convertir processed_apus a DataFrame si es necesario
        if isinstance(processed_apus, list):
            apus_df = pd.DataFrame(processed_apus)
        else:
            apus_df = processed_apus

        # Asegurar que raw_insumos_df sea DataFrame
        if not isinstance(raw_insumos_df, pd.DataFrame):
            # Intentar obtener desde insumos si raw_insumos_df no está disponible
            insumos_data = session_data["data"].get("insumos", [])
            if isinstance(insumos_data, dict):
                # Si está agrupado por categoría, aplanar
                all_insumos = []
                for cat_list in insumos_data.values():
                    all_insumos.extend(cat_list)
                raw_insumos_df = pd.DataFrame(all_insumos)
            elif isinstance(insumos_data, list):
                raw_insumos_df = pd.DataFrame(insumos_data)
            else:
                raw_insumos_df = pd.DataFrame()

        # Instanciar validador y ejecutar análisis
        validator = PyramidalValidator()
        metrics = validator.validate_structure(apus_df, raw_insumos_df)

        # Determinar estado basado en Psi
        psi = metrics.pyramid_stability_index
        stability_status = "SÓLIDA"
        if psi < 1.0:
            stability_status = "CRÍTICA (Pirámide Invertida)"
        elif psi < 3.0:
            stability_status = "FRÁGIL"

        # Mensaje con detalle de nodos flotantes
        floating_count = len(metrics.floating_nodes)
        message = (
            "Riesgo de Colapso Logístico" if psi < 1.0 else "Estructura Estable"
        )
        if floating_count > 0:
            message += f". Detectados {floating_count} nodos flotantes."

        # Inyección en el payload de respuesta
        response_data["structural_health"] = {
            "psi_index": round(psi, 2),
            "status": stability_status,
            "base_width": metrics.base_width,  # Nivel 3
            "apex_load": metrics.structure_load,  # Nivel 2
            "floating_nodes_count": floating_count,
            "floating_nodes_sample": metrics.floating_nodes[:5],  # Muestra
            "message": message,
        }
    except Exception as e:
        current_app.logger.warning(f"No se pudo calcular salud piramidal: {e}")


# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================

SESSION_TIMEOUT = 3600  # 1 hora
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB máximo por archivo
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
ALLOWED_MIME_TYPES = {
    "text/csv",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

# Límites de rate limiting
RATE_LIMIT_UPLOAD = "10 per hour"
RATE_LIMIT_API = "100 per minute"
RATE_LIMIT_ESTIMATE = "30 per hour"

# Configuración de validación de datos
MIN_ROWS_REQUIRED = 1
MAX_ROWS_ALLOWED = 50000
REQUIRED_COLUMNS = {
    "presupuesto": [],  # Validación flexible en FileValidator
    "apus": ["CODIGO_APU"],
    "insumos": ["CODIGO_INSUMO"],
}


# ============================================================================
# DATACLASSES PARA VALIDACIÓN DE ESQUEMAS
# ============================================================================


@dataclass
class SessionMetadata:
    """Metadata de sesión para validación de integridad."""

    session_id: str
    created_at: float
    last_accessed: float
    data_hash: str
    version: str = "2.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionMetadata":
        return cls(**data)

    def is_expired(self, timeout: int = SESSION_TIMEOUT) -> bool:
        """Verifica si la sesión ha expirado."""
        return (time.time() - self.last_accessed) > timeout

    def refresh(self) -> None:
        """Actualiza el timestamp de último acceso."""
        self.last_accessed = time.time()


@dataclass
class FileValidationResult:
    """Resultado de validación de archivo."""

    is_valid: bool
    filename: str
    file_type: str
    size_bytes: int
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    errors: List[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


# ============================================================================
# CONFIGURACIÓN DE LOGGING MEJORADA CON REQUEST ID
# ============================================================================


class RequestIdFilter(logging.Filter):
    """Filtro para añadir request_id a los logs."""

    def filter(self, record):
        try:
            record.request_id = g.get("request_id", "N/A")
            record.session_id = (
                session.get("sid", "N/A")[:8] if hasattr(session, "sid") else "N/A"
            )
        except RuntimeError:
            # Handle cases outside of application context (e.g., startup logging)
            record.request_id = "SYSTEM"
            record.session_id = "N/A"
        return True


def setup_logging(app: Flask, log_file: str = "app.log") -> None:
    """
    Configura el sistema de logging con rotación, formatos mejorados y request IDs.

    Args:
        app: Instancia de Flask.
        log_file: Nombre del archivo de log.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Formato detallado con request_id
    file_formatter = logging.Formatter(
        "%(asctime)s | %(request_id)s | %(session_id)s | %(name)s | "
        "%(levelname)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_formatter = logging.Formatter(
        "%(asctime)s [%(request_id)s] - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    from logging.handlers import RotatingFileHandler

    file_handler = RotatingFileHandler(
        log_dir / log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    file_handler.addFilter(RequestIdFilter())

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(RequestIdFilter())

    app.logger.handlers.clear()
    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)
    app.logger.setLevel(logging.DEBUG)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addFilter(RequestIdFilter())
    root_logger.setLevel(logging.INFO)


# ============================================================================
# UTILIDADES DE SEGURIDAD Y VALIDACIÓN
# ============================================================================


def generate_data_hash(data: Dict[str, Any]) -> str:
    """
    Genera un hash SHA256 de los datos para verificar integridad.

    Args:
        data: Diccionario de datos a hashear.

    Returns:
        Hash hexadecimal.
    """
    data_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(data_str.encode()).hexdigest()


def validate_data_integrity(data: Dict[str, Any], expected_hash: str) -> bool:
    """
    Valida que los datos no hayan sido modificados.

    Args:
        data: Datos a validar.
        expected_hash: Hash esperado.

    Returns:
        True si los datos son íntegros.
    """
    current_hash = generate_data_hash(data)
    return current_hash == expected_hash


def validate_data_schema(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Valida que los datos procesados tengan la estructura esperada.

    Args:
        data: Datos a validar.

    Returns:
        Tupla (es_válido, lista_de_errores).
    """
    errors = []

    # Validar listas
    for key in ["presupuesto", "apus_detail"]:
        if key not in data:
            errors.append(f"Clave requerida faltante: {key}")
        elif not isinstance(data[key], list):
            errors.append(f"La clave '{key}' debe ser una lista")

    # Validar diccionario de insumos
    if "insumos" not in data:
        errors.append("Clave requerida faltante: insumos")
    elif not isinstance(data["insumos"], dict):
        errors.append("La clave 'insumos' debe ser un diccionario agrupado")

    return len(errors) == 0, errors


# ============================================================================
# CLASE DE MÉTRICAS DE RENDIMIENTO
# ============================================================================


class PerformanceMetrics:
    """Recolector de métricas de rendimiento."""

    def __init__(self):
        self.metrics = defaultdict(list)

    def record(self, metric_name: str, value: float):
        """Registra una métrica."""
        self.metrics[metric_name].append({"value": value, "timestamp": time.time()})

    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """Obtiene estadísticas de una métrica."""
        values = [m["value"] for m in self.metrics.get(metric_name, [])]
        if not values:
            return {}

        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    def cleanup_old_metrics(self, max_age_seconds: int = 3600):
        """Limpia métricas antiguas."""
        cutoff = time.time() - max_age_seconds
        for metric_name in list(self.metrics.keys()):
            self.metrics[metric_name] = [
                m for m in self.metrics[metric_name] if m["timestamp"] > cutoff
            ]


# ============================================================================
# DECORADORES MEJORADOS
# ============================================================================


def timed(metric_name: str = None):
    """
    Decorador que mide el tiempo de ejecución de una función.

    Args:
        metric_name: Nombre de la métrica (si None, usa el nombre de la función).
    """

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            start_time = time.time()
            result = f(*args, **kwargs)
            elapsed = time.time() - start_time

            name = metric_name or f.__name__
            if hasattr(current_app, "metrics"):
                current_app.metrics.record(name, elapsed)

            current_app.logger.debug(f"⏱️ {name} ejecutado en {elapsed:.3f}s")
            return result

        return decorated_function

    return decorator


def require_session(f):
    """
    Decorador mejorado que verifica sesión con validación de integridad.

    Verifica:
    - Existencia de datos de sesión
    - Validez del esquema de datos
    - Integridad de datos (opcional)
    - Expiración de sesión
    - Renueva automáticamente el TTL

    Returns:
        Función decorada con validación de sesión robusta.
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 1. Verificar existencia de datos
        if "processed_data" not in session:
            current_app.logger.warning(
                f"Intento de acceso sin sesión válida desde {request.remote_addr}"
            )
            return jsonify(
                {
                    "error": "Sesión no iniciada o expirada. Por favor, cargue los archivos nuevamente.",
                    "code": "SESSION_MISSING",
                }
            ), 401

        # 2. Verificar metadata de sesión
        if "session_metadata" not in session:
            current_app.logger.error("Sesión sin metadata detectada")
            session.clear()
            return jsonify(
                {
                    "error": "Sesión corrupta. Por favor, recargue los archivos.",
                    "code": "SESSION_CORRUPTED",
                }
            ), 401

        try:
            metadata = SessionMetadata.from_dict(session["session_metadata"])
        except (TypeError, KeyError) as e:
            current_app.logger.error(f"Error al parsear metadata de sesión: {e}")
            session.clear()
            return jsonify(
                {
                    "error": "Metadata de sesión inválida.",
                    "code": "SESSION_INVALID",
                }
            ), 401

        # 3. Verificar expiración
        if metadata.is_expired():
            current_app.logger.info(
                f"Sesión expirada: {metadata.session_id[:8]}... "
                f"(última actividad hace {time.time() - metadata.last_accessed:.0f}s)"
            )
            session.clear()
            return jsonify(
                {
                    "error": "Sesión expirada. Por favor, cargue los archivos nuevamente.",
                    "code": "SESSION_EXPIRED",
                }
            ), 401

        # 4. Validar esquema de datos
        processed_data = session["processed_data"]
        is_valid, errors = validate_data_schema(processed_data)

        if not is_valid:
            current_app.logger.error(f"Esquema de datos inválido: {errors}")
            session.clear()
            return jsonify(
                {
                    "error": "Datos de sesión corruptos.",
                    "code": "SCHEMA_INVALID",
                    "details": errors,
                }
            ), 401

        # 5. Validar integridad (opcional, puede ser costoso)
        if current_app.config.get("VALIDATE_SESSION_INTEGRITY", False):
            if not validate_data_integrity(processed_data, metadata.data_hash):
                current_app.logger.error(
                    f"Integridad de datos comprometida para sesión {metadata.session_id[:8]}..."
                )
                session.clear()
                return jsonify(
                    {
                        "error": "Integridad de datos comprometida.",
                        "code": "INTEGRITY_FAILED",
                    }
                ), 401

        # 6. Renovar sesión
        metadata.refresh()
        session["session_metadata"] = metadata.to_dict()
        session.modified = True

        current_app.logger.debug(
            f"✓ Sesión válida: {metadata.session_id[:8]}... | "
            f"Edad: {time.time() - metadata.created_at:.0f}s"
        )

        # 7. Inyectar datos en la función
        session_data = {"data": processed_data, "metadata": metadata}
        return f(session_data=session_data, *args, **kwargs)

    return decorated_function


def handle_errors(f):
    """Decorador mejorado para manejo centralizado de errores con contexto."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)

        except ValueError as e:
            current_app.logger.warning(
                f"Error de validación en {f.__name__}: {str(e)}",
                extra={"endpoint": request.endpoint, "method": request.method},
            )
            return jsonify(
                {
                    "error": str(e),
                    "code": "VALIDATION_ERROR",
                    "endpoint": request.endpoint,
                }
            ), 400

        except KeyError as e:
            current_app.logger.error(
                f"Clave faltante en {f.__name__}: {str(e)}",
                extra={"endpoint": request.endpoint},
            )
            return jsonify(
                {
                    "error": f"Dato requerido faltante: {str(e)}",
                    "code": "MISSING_KEY",
                    "key": str(e),
                }
            ), 400

        except FileNotFoundError as e:
            current_app.logger.error(f"Archivo no encontrado: {str(e)}")
            return jsonify(
                {
                    "error": "Archivo requerido no encontrado",
                    "code": "FILE_NOT_FOUND",
                }
            ), 404

        except PermissionError as e:
            current_app.logger.error(f"Error de permisos: {str(e)}")
            return jsonify(
                {
                    "error": "Error de permisos al acceder a recursos",
                    "code": "PERMISSION_DENIED",
                }
            ), 500

        except json.JSONDecodeError as e:
            current_app.logger.error(f"Error al parsear JSON: {str(e)}")
            return jsonify(
                {
                    "error": "Formato JSON inválido",
                    "code": "INVALID_JSON",
                    "details": str(e),
                }
            ), 400

        except pd.errors.EmptyDataError as e:
            current_app.logger.error(f"Archivo vacío: {str(e)}")
            return jsonify(
                {
                    "error": "El archivo proporcionado está vacío",
                    "code": "EMPTY_FILE",
                }
            ), 400

        except pd.errors.ParserError as e:
            current_app.logger.error(f"Error al parsear CSV/Excel: {str(e)}")
            return jsonify(
                {
                    "error": "Error al leer el archivo. Verifique el formato.",
                    "code": "PARSE_ERROR",
                }
            ), 400

        except Exception as e:
            current_app.logger.error(
                f"Error no controlado en {f.__name__}: {str(e)}",
                exc_info=True,
                extra={
                    "endpoint": request.endpoint,
                    "method": request.method,
                    "remote_addr": request.remote_addr,
                },
            )
            return jsonify(
                {
                    "error": "Error interno del servidor",
                    "code": "INTERNAL_ERROR",
                    "request_id": g.get("request_id", "N/A"),
                }
            ), 500

    return decorated_function


@contextmanager
def temporary_upload_directory(base_path: Path, session_id: str):
    """
    Context manager mejorado para manejar directorios temporales de forma segura.

    Mejoras:
    - Múltiples intentos de limpieza
    - Logging detallado de errores
    - Validación de paths
    - Manejo de archivos bloqueados
    """
    user_dir = base_path / session_id

    # Validar que el path no escape del directorio base
    try:
        user_dir = user_dir.resolve()
        base_path = base_path.resolve()
        if not str(user_dir).startswith(str(base_path)):
            raise ValueError("Path traversal detectado")
    except (ValueError, OSError) as e:
        logging.error(f"Error de seguridad en path: {e}")
        raise

    user_dir.mkdir(parents=True, exist_ok=True)

    try:
        yield user_dir
    finally:
        # Intentar limpiar con múltiples intentos
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if user_dir.exists():
                    # Eliminar archivos
                    for file_path in user_dir.iterdir():
                        try:
                            if file_path.is_file():
                                file_path.unlink()
                            elif file_path.is_dir():
                                import shutil

                                shutil.rmtree(file_path)
                        except Exception as e:
                            logging.warning(
                                f"Intento {attempt + 1}/{max_attempts}: "
                                f"No se pudo eliminar {file_path}: {e}"
                            )

                    # Eliminar directorio
                    user_dir.rmdir()
                    logging.debug(f"Directorio temporal limpiado: {user_dir}")
                    break

            except Exception as e:
                if attempt == max_attempts - 1:
                    logging.error(
                        f"No se pudo limpiar {user_dir} después de {max_attempts} intentos: {e}"
                    )
                else:
                    time.sleep(0.1)  # Pequeña pausa antes de reintentar


# ============================================================================
# VALIDADORES MEJORADOS
# ============================================================================


class FileValidator:
    """Validador robusto de archivos con verificación de contenido."""

    @staticmethod
    def validate_file_metadata(file) -> Tuple[bool, Optional[str]]:
        """
        Valida metadata básica del archivo.

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

        # Verificar tamaño
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)

        if file_size == 0:
            return False, "El archivo está vacío"

        if file_size > MAX_CONTENT_LENGTH:
            size_mb = file_size / (1024 * 1024)
            max_mb = MAX_CONTENT_LENGTH / (1024 * 1024)
            return False, f"Archivo demasiado grande: {size_mb:.2f}MB (máx: {max_mb:.2f}MB)"

        return True, None

    @staticmethod
    def validate_mime_type(file) -> Tuple[bool, Optional[str]]:
        """
        Valida el tipo MIME del archivo.

        Returns:
            Tupla (es_válido, mensaje_error).
        """
        # Obtener MIME type desde el nombre del archivo
        mime_type, _ = mimetypes.guess_type(file.filename)

        if mime_type not in ALLOWED_MIME_TYPES:
            return False, f"Tipo MIME no permitido: {mime_type}"

        return True, None

    @staticmethod
    def validate_file_content(
        file_path: Path, file_type: str, required_columns: List[str]
    ) -> FileValidationResult:
        """
        Valida el contenido del archivo CSV/Excel utilizando data_loader robusto.

        Args:
            file_path: Ruta del archivo.
            file_type: Tipo de archivo ('presupuesto', 'apus', 'insumos').
            required_columns: Columnas requeridas.

        Returns:
            FileValidationResult con detalles de validación.
        """
        result = FileValidationResult(
            is_valid=False,
            filename=file_path.name,
            file_type=file_type,
            size_bytes=file_path.stat().st_size,
        )

        try:
            # Usar load_data para validación robusta y detección automática de separadores
            # Leemos solo una muestra inicial para validar estructura
            # Nota: load_data maneja automáticamente nrows para CSV si se pasa en kwargs,
            # pero para validación completa inicial cargamos todo con manejo de errores optimizado en data_loader
            # Sin embargo, para no saturar memoria en validación, idealmente data_loader soportaría nrows.
            # load_data pasa **kwargs a pd.read_csv/excel.

            # Primero intentamos carga ligera para validación de estructura
            load_result = load_data(file_path, nrows=10)

            if load_result.status.value != "SUCCESS":
                error_msg = (
                    load_result.error_message or "Error desconocido al cargar archivo"
                )
                if load_result.status == LoadStatus.EMPTY:
                    result.errors.append("El archivo está vacío")
                else:
                    result.errors.append(f"Error de carga: {error_msg}")
                return result

            # Obtener DataFrame (puede ser dict si es excel con múltiples hojas, tomamos la primera o única)
            df = load_result.data
            if isinstance(df, dict):
                # Si hay múltiples hojas, usamos la primera que no esté vacía
                first_sheet = next(iter(df.values()))
                df = first_sheet

            if df is None or df.empty:
                result.errors.append("El archivo no contiene datos legibles")
                return result

            result.column_count = len(df.columns)

            # Verificar columnas requeridas
            if file_type == "presupuesto":
                # Lógica flexible para presupuesto: acepta ITEM o CODIGO_APU
                has_item = "ITEM" in df.columns
                has_codigo = "CODIGO_APU" in df.columns

                if not (has_item or has_codigo):
                    result.errors.append(
                        "Columnas faltantes: Se requiere 'CODIGO_APU' o 'ITEM'"
                    )
                    return result
            else:
                missing_columns = set(required_columns) - set(df.columns)
                if missing_columns:
                    result.errors.append(f"Columnas faltantes: {', '.join(missing_columns)}")
                    return result

            # Si la estructura es válida, cargamos el archivo completo para validación de volumen y nulos
            # Esto es necesario para tener el row_count real y estadísticas de nulos
            # Si el archivo es muy grande, load_data ya tiene advertencias, pero aquí necesitamos números exactos
            full_load_result = load_data(file_path)

            if full_load_result.status.value != "SUCCESS":
                result.errors.append("Error al leer el archivo completo para validación")
                return result

            df_full = full_load_result.data
            if isinstance(df_full, dict):
                df_full = next(iter(df_full.values()))

            result.row_count = len(df_full)

            # Validar cantidad de filas
            if result.row_count < MIN_ROWS_REQUIRED:
                result.errors.append(
                    f"Archivo con muy pocas filas: {result.row_count} (mín: {MIN_ROWS_REQUIRED})"
                )
                return result

            if result.row_count > MAX_ROWS_ALLOWED:
                result.warnings.append(
                    f"Archivo grande: {result.row_count} filas (máx recomendado: {MAX_ROWS_ALLOWED})"
                )

            # Verificar valores nulos en columnas críticas
            for col in required_columns:
                if col in df_full.columns:
                    null_count = df_full[col].isnull().sum()
                    if null_count > 0:
                        null_pct = (null_count / result.row_count) * 100
                        if null_pct > 50:
                            result.errors.append(
                                f"Columna '{col}' tiene {null_pct:.1f}% valores nulos"
                            )
                        elif null_pct > 10:
                            result.warnings.append(
                                f"Columna '{col}' tiene {null_pct:.1f}% valores nulos"
                            )

            # Si no hay errores, el archivo es válido
            result.is_valid = len(result.errors) == 0

        except Exception as e:
            result.errors.append(f"Error inesperado durante validación: {str(e)}")

        return result

    @classmethod
    def validate_complete(
        cls, file, file_type: str
    ) -> Tuple[FileValidationResult, Optional[str]]:
        """
        Validación completa: metadata + MIME + contenido.

        Args:
            file: Objeto de archivo de werkzeug.
            file_type: Tipo de archivo.

        Returns:
            Tupla (FileValidationResult, path_temporal).
        """
        # 1. Validar metadata
        is_valid, error = cls.validate_file_metadata(file)
        if not is_valid:
            return (
                FileValidationResult(
                    is_valid=False,
                    filename=file.filename,
                    file_type=file_type,
                    size_bytes=0,
                    errors=[error],
                ),
                None,
            )

        # 2. Validar MIME type
        is_valid, error = cls.validate_mime_type(file)
        if not is_valid:
            return (
                FileValidationResult(
                    is_valid=False,
                    filename=file.filename,
                    file_type=file_type,
                    size_bytes=0,
                    errors=[error],
                ),
                None,
            )

        # 3. Guardar temporalmente y validar contenido
        import tempfile

        temp_dir = Path(tempfile.gettempdir()) / "apu_validation"
        temp_dir.mkdir(exist_ok=True)

        temp_file = temp_dir / secure_filename(file.filename)

        try:
            file.save(str(temp_file))

            required_cols = REQUIRED_COLUMNS.get(file_type, [])
            content_result = cls.validate_file_content(temp_file, file_type, required_cols)

            return content_result, str(temp_file)

        except Exception as e:
            return (
                FileValidationResult(
                    is_valid=False,
                    filename=file.filename,
                    file_type=file_type,
                    size_bytes=0,
                    errors=[f"Error al guardar archivo temporal: {str(e)}"],
                ),
                None,
            )




# ============================================================================
# FACTORY DE APLICACIÓN MEJORADA
# ============================================================================


def create_app(config_name: str) -> Flask:
    """
    Crea y configura una instancia de la aplicación Flask con configuración robusta.

    Mejoras:
    - Validación de configuración al inicio
    - Sistema de métricas integrado
    - Rate limiting configurado
    - Logging con request IDs
    - Gestión de sesiones mejorada

    Args:
        config_name: Nombre del entorno de configuración.

    Returns:
        Instancia configurada de Flask.
    """
    app = Flask(__name__, static_folder="../static", static_url_path="/static")

    # ========================================================================
    # CONFIGURACIÓN BÁSICA
    # ========================================================================

    app.config.from_object(config_by_name[config_name])
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.config["VALIDATE_SESSION_INTEGRITY"] = config_name == "production"

    # SECRET_KEY desde variable de entorno (obligatoria en producción)
    secret_key = os.environ.get("SECRET_KEY")
    if config_name == "production" and not secret_key:
        raise ValueError("SECRET_KEY es obligatoria en producción")

    app.config["SECRET_KEY"] = secret_key or "dev_key_fija_y_secreta_12345"

    # ========================================================================
    # CONFIGURACIÓN DE LOGGING
    # ========================================================================

    setup_logging(app)
    app.logger.info(f"{'=' * 60}")
    app.logger.info(f"Iniciando aplicación en modo: {config_name.upper()}")
    app.logger.info(f"{'=' * 60}")

    # ========================================================================
    # SISTEMA DE MÉTRICAS
    # ========================================================================

    app.metrics = PerformanceMetrics()

    # ========================================================================
    # CONFIGURACIÓN DE CORS
    # ========================================================================

    CORS(app, supports_credentials=True)

    # ========================================================================
    # CONFIGURACIÓN DE REDIS Y SESIONES
    # ========================================================================

    import redis
    from flask_session import Session

    redis_url = app.config.get("REDIS_URL", "redis://localhost:6379/0")

    try:
        redis_client = redis.from_url(redis_url, decode_responses=False)
        redis_client.ping()
        app.logger.info(f"✅ Conexión a Redis exitosa: {redis_url}")
    except redis.ConnectionError as e:
        app.logger.error(f"❌ Error de conexión a Redis: {e}")
        raise

    app.config["SESSION_TYPE"] = "redis"
    app.config["SESSION_PERMANENT"] = True
    app.config["SESSION_USE_SIGNER"] = True
    app.config["SESSION_KEY_PREFIX"] = "apu_filter:session:"
    app.config["SESSION_REDIS"] = redis_client
    app.config["PERMANENT_SESSION_LIFETIME"] = SESSION_TIMEOUT

    # Configuración de cookies
    app.config["SESSION_COOKIE_NAME"] = "apu_session"
    app.config["SESSION_COOKIE_PATH"] = "/"
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_DOMAIN"] = None

    if config_name == "production":
        app.config["SESSION_COOKIE_SECURE"] = True
        app.config["SESSION_COOKIE_SAMESITE"] = "Strict"
    else:
        app.config["SESSION_COOKIE_SECURE"] = False
        app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    Session(app)

    # ========================================================================
    # RATE LIMITING
    # ========================================================================

    def is_internal_traffic():
        """
        Determina si el tráfico es interno y debe ser eximido del rate limiting.
        """
        # Eximir por User-Agent específico (Alternativa simple y robusta)
        if request.headers.get("User-Agent") == "APU-Agent-Internal":
            return True

        # Eximir por IP interna (Opcional, si se requiere más adelante)
        # remote_addr = get_remote_address()
        # return remote_addr.startswith(("10.", "172.16.", "192.168."))
        return False

    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"],
        storage_uri=redis_url,
        default_limits_exempt_when=is_internal_traffic,
    )

    # ========================================================================
    # CONFIGURACIÓN DE LA APLICACIÓN
    # ========================================================================

    config_path = Path(__file__).parent.parent / "config" / "config_rules.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            app.config["APP_CONFIG"] = json.load(f)
            app.logger.info("✅ Configuración cargada desde config_rules.json")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        app.logger.error(f"❌ Error al cargar config_rules.json: {e}")
        app.config["APP_CONFIG"] = {}

    # ========================================================================
    # CONFIGURACIÓN DE DIRECTORIOS
    # ========================================================================

    project_root = Path(__file__).parent.parent
    app.template_folder = str(project_root / "templates")

    if os.environ.get("FLASK_ENV") == "production":
        upload_folder = Path("/app/data/uploads")
    else:
        upload_folder = Path(app.config.get("UPLOAD_FOLDER", "data/uploads"))

    upload_folder.mkdir(parents=True, exist_ok=True)
    app.config["UPLOAD_FOLDER"] = str(upload_folder)

    # ========================================================================
    # INICIALIZACIÓN DE COMPONENTES
    # ========================================================================

    apu_presenter = APUPresenter(app.logger)

    # Inicializar la MIC (Matriz de Interacción Central)
    mic = MICRegistry()

    # Registrar Vectores Base
    mic.register_vector("diagnose", Stratum.PHYSICS, diagnose_file)
    mic.register_vector("clean", Stratum.PHYSICS, clean_file)
    mic.register_vector("get_telemetry_status", Stratum.PHYSICS, get_telemetry_status)
    mic.register_vector("financial_analysis", Stratum.STRATEGY, analyze_financial_viability)

    # Wrapper para Laplace Oracle (Vector Físico/Táctico - TACTICS según plan)
    # TACTICS es intermedio. PHYSICS < TACTICS < STRATEGY.
    def laplace_handler(R, L, C):
        oracle = LaplaceOracle(R=R, L=L, C=C)
        return oracle.get_laplace_pyramid()

    mic.register_vector("oracle_analyze", Stratum.TACTICS, laplace_handler)

    # Registrar Vectores del Núcleo (incluye Semánticos si config está presente)
    register_core_vectors(mic, app.config.get("APP_CONFIG"))

    # Inyectar MIC en la app
    app.mic = mic

    # ========================================================================
    # MIDDLEWARE Y HOOKS
    # ========================================================================

    @app.before_request
    def before_request_func():
        """Middleware para añadir request_id y logging."""
        # Generar request_id único
        g.request_id = str(uuid.uuid4())[:8]

        # Timestamp de inicio
        g.start_time = time.time()

        # Inicializar TelemetryContext
        g.telemetry = TelemetryContext(request_id=g.request_id)

        # Logging
        app.logger.debug(
            f"➡️  Request iniciado: {request.method} {request.path} | "
            f"Session: {'✓' if 'processed_data' in session else '✗'} | "
            f"IP: {request.remote_addr}"
        )

    @app.after_request
    def after_request_func(response):
        """Middleware para añadir cabeceras de seguridad y métricas."""
        # Cabeceras de seguridad
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["X-Request-ID"] = g.get("request_id", "N/A")

        # Métricas de tiempo de respuesta
        if hasattr(g, "start_time"):
            elapsed = time.time() - g.start_time
            app.metrics.record("request_duration", elapsed)

            app.logger.debug(
                f"⬅️  Response: {response.status_code} | "
                f"Tiempo: {elapsed:.3f}s | "
                f"Size: {response.content_length or 0} bytes"
            )

        return response

    @app.teardown_appcontext
    def cleanup_resources(exception=None):
        """Cleanup de recursos al finalizar el request."""
        if exception:
            app.logger.error(f"Exception en teardown: {exception}")
            if hasattr(g, "telemetry"):
                g.telemetry.record_error("app_teardown", str(exception))

        # Log telemetría
        if hasattr(g, "telemetry"):
            telemetry_data = g.telemetry.to_dict()
            app.logger.info(
                f"📊 Telemetry for {g.request_id}: {json.dumps(telemetry_data, default=str)}"
            )

        # Cleanup de métricas antiguas cada 100 requests
        if hasattr(app, "metrics"):
            import random

            if random.randint(1, 100) == 1:
                app.metrics.cleanup_old_metrics()

    # Registrar Blueprints
    app.register_blueprint(topology_bp)

    # ========================================================================
    # RUTAS PRINCIPALES
    # ========================================================================

    @app.route("/")
    def index():
        """Renderiza la página principal."""
        return render_template("index.html")

    @app.route("/health", methods=["GET"])
    def health_check():
        """Endpoint de verificación de estado con métricas detalladas."""
        redis_client = app.config["SESSION_REDIS"]

        try:
            prefix = app.config["SESSION_KEY_PREFIX"]
            active_sessions = len(redis_client.keys(f"{prefix}*"))
            redis_healthy = True
        except Exception as e:
            app.logger.error(f"Error al obtener métricas de Redis: {e}")
            active_sessions = -1
            redis_healthy = False

        # Métricas de rendimiento
        metrics = {}
        if hasattr(app, "metrics"):
            metrics = {
                "request_duration": app.metrics.get_stats("request_duration"),
                "upload_duration": app.metrics.get_stats("upload_files"),
                "estimate_duration": app.metrics.get_stats("get_estimate"),
            }

        return jsonify(
            {
                "status": "healthy" if redis_healthy else "degraded",
                "timestamp": time.time(),
                "version": app.config.get("APP_CONFIG", {}).get("version", "2.0.0"),
                "environment": config_name,
                "redis": {
                    "healthy": redis_healthy,
                    "active_sessions": active_sessions,
                },
                "session_config": {
                    "timeout_seconds": SESSION_TIMEOUT,
                    "max_file_size_mb": MAX_CONTENT_LENGTH / (1024 * 1024),
                },
                "semantic_search": {
                    "enabled": app.mic.is_registered("semantic_match"),
                    "model": app.config.get("APP_CONFIG", {}).get("embedding_metadata", {}).get(
                        "model_name", "N/A"
                    ),
                },
                "metrics": metrics,
            }
        )

    @app.route("/upload", methods=["POST"])
    @limiter.limit(RATE_LIMIT_UPLOAD, exempt_when=lambda: current_app.config.get("TESTING"))
    @handle_errors
    @timed("upload_files")
    def upload_files():
        """
        Gestiona la carga de archivos con validación exhaustiva y telemetría.
        """
        app.logger.info("📤 Iniciando proceso de carga de archivos")
        g.telemetry.start_step("upload_request_validation")

        # Validar archivos requeridos
        required_files = ["presupuesto", "apus", "insumos"]
        missing_files = [f for f in required_files if f not in request.files]

        if missing_files:
            g.telemetry.record_error("upload_request_validation", "Missing files")
            g.telemetry.end_step("upload_request_validation", "error")
            return jsonify(
                {
                    "error": f"Faltan archivos: {', '.join(missing_files)}",
                    "code": "MISSING_FILES",
                    "required": required_files,
                }
            ), 400

        # Validar cada archivo completamente
        validator = FileValidator()
        files_to_process = {}
        validation_results = {}

        for file_type in required_files:
            file = request.files[file_type]

            # 1. Validar metadata básica (tamaño, extensión)
            is_valid_meta, error_meta = validator.validate_file_metadata(file)
            if not is_valid_meta:
                g.telemetry.record_error(
                    "upload_request_validation", f"Invalid metadata: {file_type}"
                )
                return jsonify(
                    {"error": error_meta, "code": "INVALID_FILE", "file": file_type}
                ), 400

            # 2. Validar MIME type
            is_valid_mime, error_mime = validator.validate_mime_type(file)
            if not is_valid_mime:
                g.telemetry.record_error(
                    "upload_request_validation", f"Invalid mime: {file_type}"
                )
                return jsonify(
                    {"error": error_mime, "code": "INVALID_MIME", "file": file_type}
                ), 400

            # 3. Guardar temporalmente SIN validar contenido estricto (permitiendo columnas flexibles)
            import tempfile

            temp_dir = Path(tempfile.gettempdir()) / "apu_validation"
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / secure_filename(file.filename)

            try:
                file.save(str(temp_path))
            except Exception as e:
                app.logger.error(f"Error al guardar archivo temporal {file_type}: {e}")
                return jsonify(
                    {"error": "Error interno al guardar archivo", "code": "SAVE_ERROR"}
                ), 500

            # Registrar resultado exitoso para mantener compatibilidad
            validation_results[file_type] = FileValidationResult(
                is_valid=True,
                filename=file.filename,
                file_type=file_type,
                size_bytes=Path(temp_path).stat().st_size,
            )

            files_to_process[file_type] = str(temp_path)

        g.telemetry.end_step("upload_request_validation", "success")

        # Generar ID de sesión
        session_id = str(uuid.uuid4())

        # Procesar archivos
        upload_path = Path(app.config["UPLOAD_FOLDER"])

        with temporary_upload_directory(upload_path, session_id) as user_dir:
            # Copiar archivos validados al directorio de sesión
            file_paths = {}
            for file_type, temp_path in files_to_process.items():
                import shutil

                dest_path = user_dir / Path(temp_path).name
                shutil.copy2(temp_path, dest_path)
                file_paths[file_type] = str(dest_path)

                # Limpiar archivo temporal de validación
                try:
                    Path(temp_path).unlink()
                except Exception as e:
                    app.logger.warning(f"No se pudo eliminar temp file: {e}")

            # Procesar archivos
            app.logger.info(f"Procesando archivos para sesión {session_id[:8]}...")

            start_processing = time.time()
            # Pasamos g.telemetry al procesador
            processed_data = process_all_files(
                file_paths["presupuesto"],
                file_paths["apus"],
                file_paths["insumos"],
                config=app.config.get("APP_CONFIG", {}),
                telemetry=g.telemetry,
            )
            processing_time = time.time() - start_processing
            g.telemetry.record_metric("app", "total_processing_time", processing_time)

            # Persistir métricas globales en Redis para visibilidad del Agente
            try:
                redis_client = current_app.config.get("SESSION_REDIS")
                if redis_client:
                    metrics_data = json.dumps(g.telemetry.metrics)
                    redis_client.set("apu_filter:global_metrics", metrics_data, ex=3600)
                    app.logger.info(
                        "📡 Métricas globales persistidas en Redis para el Agente"
                    )
            except Exception as e:
                app.logger.warning(f"⚠️ No se pudo persistir telemetría global: {e}")

            app.logger.info(f"Procesamiento completado en {processing_time:.2f}s")

        # Verificar errores en procesamiento
        if "error" in processed_data:
            app.logger.error(f"Error de procesamiento: {processed_data['error']}")
            g.telemetry.record_error("processing_pipeline", processed_data["error"])
            return jsonify(
                {
                    "error": processed_data["error"],
                    "code": "PROCESSING_ERROR",
                }
            ), 500

        g.telemetry.start_step("response_preparation")

        # Detectar si es un Data Product (QFS) y desempaquetar para validación y almacenamiento
        if processed_data.get("kind") == "DataProduct" and "payload" in processed_data:
            app.logger.info(
                "📦 Data Product (QFS) detectado: Desempaquetando payload para flujo legacy"
            )
            processed_data = processed_data["payload"]

        # Validar esquema de datos procesados
        is_valid_schema, schema_errors = validate_data_schema(processed_data)
        if not is_valid_schema:
            app.logger.error(f"Esquema inválido: {schema_errors}")
            g.telemetry.record_error("response_preparation", "Invalid schema")
            return jsonify(
                {
                    "error": "Datos procesados con esquema inválido",
                    "code": "INVALID_SCHEMA",
                    "details": schema_errors,
                }
            ), 500

        # Sanitizar datos
        sanitized_data = sanitize_for_json(processed_data)

        # Generar metadata de sesión
        data_hash = generate_data_hash(sanitized_data)
        metadata = SessionMetadata(
            session_id=session_id,
            created_at=time.time(),
            last_accessed=time.time(),
            data_hash=data_hash,
        )

        # Guardar en sesión
        session["processed_data"] = sanitized_data
        session["session_metadata"] = metadata.to_dict()
        # MIC: Carga exitosa valida el nivel PHYSICS
        session["validated_strata"] = {Stratum.PHYSICS}
        session.permanent = True

        # Preparar respuesta
        response_data = {
            "success": True,
            "session_id": session_id,
            "metadata": {
                "created_at": metadata.created_at,
                "session_timeout": SESSION_TIMEOUT,
                "data_hash": data_hash[:16],  # Primeros 16 caracteres
            },
            "summary": {
                "presupuesto_items": len(sanitized_data.get("presupuesto", [])),
                "apu_details": len(sanitized_data.get("apus_detail", [])),
                "insumos": len(sanitized_data.get("insumos", [])),
            },
            "executive_summary": g.telemetry.get_business_report(),  # Capa 1: Traducción Ejecutiva
            "technical_audit": sanitized_data.get(
                "audit_report"
            ),  # Capa 3: Auditoría Técnica
            "validation": {
                file_type: {
                    "rows": result.row_count,
                    "columns": result.column_count,
                    "warnings": result.warnings,
                }
                for file_type, result in validation_results.items()
            },
            "processing_time": processing_time,
        }

        # Inyectar salud piramidal
        _inject_pyramidal_health(
            response_data,
            {"data": processed_data}
        )

        # Incluir datos completos si se solicita
        if request.args.get("include_data") == "true":
            response_data["data"] = sanitized_data

        app.logger.info(
            f"✅ Archivos procesados exitosamente | Sesión: {session_id[:8]}... | "
            f"Items: {response_data['summary']}"
        )
        g.telemetry.end_step("response_preparation", "success")

        return jsonify(response_data)

    @app.route("/api/apu/<code>", methods=["GET"])
    @limiter.limit(RATE_LIMIT_API, exempt_when=lambda: current_app.config.get("TESTING"))
    @require_session
    @handle_errors
    @timed("get_apu_detail")
    def get_apu_detail(code: str, session_data: dict = None):
        """
        Recupera y procesa los detalles de un APU con manejo robusto de variantes de código.
        """
        g.telemetry.start_step("get_apu_detail")
        app.logger.info(f"🔍 Solicitud de detalle para APU: {code}")

        # Normalizar código
        apu_code = code.replace("%2C", ",").strip()

        # Obtener datos
        user_data = session_data["data"]
        all_apu_details = user_data.get("apus_detail", [])

        # Estrategia de búsqueda multi-nivel
        search_variants = [
            apu_code,
            apu_code.replace(".", ","),
            apu_code.replace(",", "."),
            apu_code.upper(),
            apu_code.lower(),
        ]

        apu_details = None
        matched_code = None

        for variant in search_variants:
            matches = [
                item
                for item in all_apu_details
                if item.get("CODIGO_APU", "").strip() == variant
            ]
            if matches:
                apu_details = matches
                matched_code = variant
                app.logger.debug(f"APU encontrado con variante: {variant}")
                break

        if not apu_details:
            app.logger.warning(
                f"❌ APU no encontrado después de intentar variantes: {search_variants}"
            )
            g.telemetry.record_error("get_apu_detail", "APU not found")
            g.telemetry.end_step("get_apu_detail", "not_found")
            return jsonify(
                {
                    "error": f"APU no encontrado: {code}",
                    "code": "APU_NOT_FOUND",
                    "searched_variants": search_variants,
                }
            ), 404

        # Procesar detalles
        processed_data = apu_presenter.process_apu_details(apu_details, matched_code)

        # Buscar en presupuesto
        presupuesto_data = user_data.get("presupuesto", [])
        presupuesto_item = next(
            (
                item
                for item in presupuesto_data
                if item.get("CODIGO_APU", "").strip() == matched_code
            ),
            None,
        )

        # Ejecutar simulación Monte Carlo
        simulation_results = run_monte_carlo_simulation(processed_data["items"])

        # Preparar respuesta
        response = {
            "codigo": matched_code,
            "codigo_original": apu_code,
            "descripcion": (
                presupuesto_item.get("original_description", "") if presupuesto_item else ""
            ),
            "desglose": processed_data["desglose"],
            "simulation": simulation_results,
            "metadata": {
                "total_items": processed_data["total_items"],
                "categorias": list(processed_data["desglose"].keys()),
                "presupuesto_encontrado": presupuesto_item is not None,
            },
        }

        app.logger.info(
            f"✅ Detalle para APU {matched_code} generado exitosamente | "
            f"Items: {processed_data['total_items']}"
        )
        g.telemetry.end_step("get_apu_detail", "success")

        return jsonify(sanitize_for_json(response))

    @app.route("/api/estimate", methods=["POST"])
    @limiter.limit(
        RATE_LIMIT_ESTIMATE, exempt_when=lambda: current_app.config.get("TESTING")
    )
    @require_session
    @handle_errors
    @timed("get_estimate")
    def get_estimate(session_data: dict = None):
        """
        Calcula estimación de costos con validación de parámetros.
        """
        g.telemetry.start_step("get_estimate")

        # Validar Content-Type
        if not request.is_json:
            g.telemetry.end_step("get_estimate", "error")
            return jsonify(
                {
                    "error": "Content-Type debe ser application/json",
                    "code": "INVALID_CONTENT_TYPE",
                }
            ), 400

        params = request.get_json()
        if not params:
            g.telemetry.end_step("get_estimate", "error")
            return jsonify(
                {
                    "error": "No se proporcionaron parámetros",
                    "code": "NO_PARAMS",
                }
            ), 400

        # Validar parámetros requeridos (aceptar m2 como alias de area_m2)
        if "m2" in params and "area_m2" not in params:
            params["area_m2"] = params["m2"]

        required_params = ["area_m2"]
        missing_params = [p for p in required_params if p not in params]

        if missing_params:
            g.telemetry.record_error("get_estimate", f"Missing params: {missing_params}")
            g.telemetry.end_step("get_estimate", "error")
            return jsonify(
                {
                    "error": f"Parámetros faltantes: {', '.join(missing_params)}",
                    "code": "MISSING_PARAMS",
                    "required": required_params,
                }
            ), 400

        app.logger.info(f"📊 Solicitud de estimación con parámetros: {params}")

        # Calcular estimación vía MIC (Tactical Advisor)
        user_data = session_data["data"]

        payload = {
            "params": params,
            "data_store": user_data
        }

        # Contexto para MIC
        mic_context = {
            "validated_strata": session.get("validated_strata", set()),
            "telemetry_context": g.telemetry
        }

        # En testing permitimos bypass
        if current_app.config.get("TESTING"):
             mic_context["force_physics_override"] = True

        try:
            # Proyección Algebraica al Estrato Táctico
            projection = app.mic.project_intent("tactical_estimate", payload, mic_context)

            if not projection.get("success"):
                error_msg = projection.get("error", "Unknown error in tactical estimate")
                app.logger.warning(f"⚠️ Error en estimación táctica: {error_msg}")
                g.telemetry.record_error("get_estimate", error_msg)

                # Manejo de error de permisos (Jerarquía)
                if projection.get("error_category") == "mic_hierarchy_violation":
                     return jsonify(projection), 403

                return jsonify(projection), 400

            result = projection.get("estimate", {})

            # MIC: Actualizar sesión si el estrato fue validado
            if projection.get("_mic_validation_update"):
                 if "validated_strata" not in session:
                     session["validated_strata"] = {Stratum.TACTICS}
                 elif isinstance(session.get("validated_strata"), set):
                     session["validated_strata"].add(Stratum.TACTICS)

            # Añadir metadata
            result["metadata"] = {
                "semantic_search_used": True,
                "timestamp": time.time(),
                "request_id": g.get("request_id", "N/A"),
                "mic_stratum": projection.get("stratum", "UNKNOWN")
            }

            app.logger.info(
                f"✅ Estimación calculada | Construcción: ${result.get('valor_construccion', 0):,.2f} | "
                f"Rendimiento: {result.get('rendimiento_m2_por_dia', 0):.2f}"
            )

            g.telemetry.end_step("get_estimate", "success")

            return jsonify(sanitize_for_json(result))

        except Exception as e:
            app.logger.error(f"Critical error in estimate projection: {e}")
            return jsonify({"error": str(e), "code": "PROJECTION_ERROR"}), 500

    @app.route("/api/session/clear", methods=["POST"])
    @handle_errors
    def clear_session():
        """Limpia la sesión actual con logging."""
        session_id = session.get("sid", "N/A")[:8] if hasattr(session, "sid") else "N/A"
        had_session = "processed_data" in session

        session.clear()

        app.logger.info(
            f"🗑️  Sesión limpiada | ID: {session_id} | Tenía datos: {had_session}"
        )

        return jsonify(
            {
                "success": True,
                "message": "Sesión limpiada exitosamente"
                if had_session
                else "No había sesión activa",
                "had_session": had_session,
                "session_id": session_id,
            }
        )

    @app.route("/api/session/info", methods=["GET"])
    def session_info():
        """Endpoint de debugging para inspeccionar sesión (solo dev)."""
        if config_name == "production":
            return jsonify({"error": "Endpoint deshabilitado en producción"}), 403

        has_data = "processed_data" in session
        has_metadata = "session_metadata" in session

        info = {
            "sid": session.sid if hasattr(session, "sid") else None,
            "has_data": has_data,
            "has_metadata": has_metadata,
            "permanent": session.permanent,
            "modified": session.modified,
        }

        if has_data:
            info["data_keys"] = list(session["processed_data"].keys())
            info["data_summary"] = {
                key: len(value) if isinstance(value, list) else type(value).__name__
                for key, value in session["processed_data"].items()
            }

        if has_metadata:
            metadata = SessionMetadata.from_dict(session["session_metadata"])
            info["metadata"] = {
                "session_id": metadata.session_id[:8],
                "created_at": datetime.fromtimestamp(metadata.created_at).isoformat(),
                "last_accessed": datetime.fromtimestamp(metadata.last_accessed).isoformat(),
                "age_seconds": time.time() - metadata.created_at,
                "is_expired": metadata.is_expired(),
            }

        return jsonify(info)

    @app.route("/api/metrics", methods=["GET"])
    def get_metrics():
        """Endpoint para obtener métricas de rendimiento (solo dev)."""
        if config_name == "production":
            return jsonify({"error": "Endpoint deshabilitado en producción"}), 403

        if not hasattr(app, "metrics"):
            return jsonify({"error": "Sistema de métricas no disponible"}), 503

        all_metrics = {}
        for metric_name in [
            "request_duration",
            "upload_files",
            "get_estimate",
            "get_apu_detail",
        ]:
            stats = app.metrics.get_stats(metric_name)
            if stats:
                all_metrics[metric_name] = stats

        return jsonify(
            {
                "metrics": all_metrics,
                "timestamp": time.time(),
            }
        )

    # ========================================================================
    # ENDPOINTS DE HERRAMIENTAS (Pivotes del Agente)
    # ========================================================================

    @app.route("/api/tools/diagnose", methods=["POST"])
    @limiter.limit("20 per minute", exempt_when=lambda: current_app.config.get("TESTING"))
    @handle_errors
    def tool_diagnose():
        """
        Pivote 2: Diagnóstico de archivos.
        Recibe un archivo y devuelve su análisis estructural.
        """
        if "file" not in request.files:
            return jsonify({"error": "No file part", "code": "MISSING_FILE"}), 400

        file = request.files["file"]
        file_type = request.form.get("type", "apus")  # apus, insumos, presupuesto

        if file.filename == "":
            return jsonify({"error": "No selected file", "code": "NO_FILE"}), 400

        # Guardar temporalmente
        import tempfile

        temp_dir = Path(tempfile.gettempdir()) / "apu_tools"
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / secure_filename(file.filename)

        try:
            file.save(str(temp_path))

            # Proyectar intención en la MIC
            # Contexto: Agregamos force_physics_override porque diagnose ES el validador físico
            context = {
                "validated_strata": session.get("validated_strata", []),
                "force_physics_override": True
            }

            payload = {
                "file_path": temp_path,
                "file_type": file_type
            }

            result = app.mic.project_intent("diagnose", payload, context)

            # MIC: Actualizar sesión si hay validación
            if result.get("_mic_validation_update"):
                if "validated_strata" not in session:
                    session["validated_strata"] = {Stratum.PHYSICS}
                elif isinstance(session.get("validated_strata"), set):
                    session["validated_strata"].add(Stratum.PHYSICS)

            # Limpiar
            try:
                temp_path.unlink()
            except Exception as e:
                app.logger.warning(f"Failed to delete temp file: {e}")

            if not result.get("success", False):
                return jsonify(result), 400

            return jsonify(result)

        except Exception as e:
            app.logger.error(f"Error in diagnose tool: {e}")
            return jsonify({"error": str(e), "code": "TOOL_ERROR"}), 500

    @app.route("/api/tools/clean", methods=["POST"])
    @limiter.limit("10 per minute", exempt_when=lambda: current_app.config.get("TESTING"))
    @handle_errors
    def tool_clean():
        """
        Pivote 3: Saneamiento de archivos.
        Recibe un archivo sucio y devuelve estadísticas de limpieza.
        (En un escenario real, devolvería el archivo limpio o un link para descargarlo).
        """
        if "file" not in request.files:
            return jsonify({"error": "No file part", "code": "MISSING_FILE"}), 400

        file = request.files["file"]
        delimiter = request.form.get("delimiter", ";")
        encoding = request.form.get("encoding", "utf-8")

        if file.filename == "":
            return jsonify({"error": "No selected file", "code": "NO_FILE"}), 400

        # Guardar temporalmente
        import tempfile

        temp_dir = Path(tempfile.gettempdir()) / "apu_tools"
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / secure_filename(file.filename)
        output_path = temp_dir / f"clean_{secure_filename(file.filename)}"

        try:
            file.save(str(temp_path))

            # Proyectar intención en la MIC
            # Contexto: Limpieza es un proceso Físico.
            context = {
                "validated_strata": session.get("validated_strata", []),
                "force_physics_override": True
            }

            payload = {
                "input_path": temp_path,
                "output_path": output_path,
                "delimiter": delimiter,
                "encoding": encoding
            }

            result = app.mic.project_intent("clean", payload, context)

            # MIC: Actualizar sesión
            if result.get("_mic_validation_update"):
                if "validated_strata" not in session:
                    session["validated_strata"] = {Stratum.PHYSICS}
                elif isinstance(session.get("validated_strata"), set):
                    session["validated_strata"].add(Stratum.PHYSICS)

            # Limpiar input
            try:
                temp_path.unlink()
            except Exception:
                pass

            if result.get("success"):
                try:
                    Path(result["output_path"]).unlink()
                except Exception:
                    pass
                result["message"] = "File cleaned successfully (temp file deleted)"
            else:
                return jsonify(result), 400

            return jsonify(result)

        except Exception as e:
            app.logger.error(f"Error in clean tool: {e}")
            return jsonify({"error": str(e), "code": "TOOL_ERROR"}), 500

    @app.route("/api/telemetry/status", methods=["GET"])
    @handle_errors
    def tool_telemetry_status():
        """
        Pivote 1: Telemetría (El Sensor).
        Devuelve el Vector de Estado del sistema.
        """
        context = getattr(g, "telemetry", None)

        if context and not context.metrics:
            try:
                redis_client = current_app.config.get("SESSION_REDIS")
                if redis_client:
                    global_metrics_json = redis_client.get("apu_filter:global_metrics")
                    if global_metrics_json:
                        global_metrics = json.loads(global_metrics_json)
                        context.metrics = global_metrics
                        current_app.logger.debug("📡 Telemetría global recuperada de Redis")
            except Exception as e:
                current_app.logger.warning(f"⚠️ Error leyendo telemetría global: {e}")

        # Proyectar en MIC
        payload = {"telemetry_context": context}
        context_mic = {"validated_strata": session.get("validated_strata", [])}

        # Telemetry is Physics/Observation, usually always allowed or base
        context_mic["force_physics_override"] = True

        result = app.mic.project_intent("get_telemetry_status", payload, context_mic)

        # Unwrap if success response wrapped (get_telemetry_status returns dict directly usually)
        # But project_intent returns what handler returns.

        return jsonify(result)

    @app.route("/api/tools/financial_analysis", methods=["POST"])
    @limiter.limit("30 per minute", exempt_when=lambda: current_app.config.get("TESTING"))
    @handle_errors
    def tool_financial_analysis():
        """
        Pivote 4: Análisis de Viabilidad Financiera (Nivel 1 - ESTRATEGIA).
        Recibe un monto, desviación y tiempo, y devuelve un análisis financiero.
        """
        if not request.is_json:
            return jsonify(
                {
                    "error": "Content-Type must be application/json",
                    "code": "INVALID_CONTENT_TYPE",
                }
            ), 400

        data = request.get_json()
        amount = data.get("amount")
        std_dev = data.get("std_dev")
        time_years = data.get("time")

        if not all([amount, std_dev, time_years]):
            return jsonify(
                {
                    "error": "Faltan parámetros: amount, std_dev, time",
                    "code": "MISSING_PARAMS",
                }
            ), 400

        try:
            payload = {
                "amount": float(amount),
                "std_dev": float(std_dev),
                "time_years": int(time_years)
            }
        except (ValueError, TypeError):
            return jsonify(
                {"error": "Parámetros deben ser numéricos", "code": "INVALID_PARAMS"}
            ), 400

        # MIC: Contexto Estratégico
        context = {
            "validated_strata": session.get("validated_strata", set())
        }

        # Bypass en testing
        if current_app.config.get("TESTING"):
             context["force_physics_override"] = True

        result = app.mic.project_intent("financial_analysis", payload, context)

        if not result.get("success"):
            status_code = 400
            # Si es error de permiso (violación de jerarquía), 403
            if result.get("error_category") == "mic_hierarchy_violation":
                status_code = 403
            return jsonify(result), status_code

        return jsonify(result)

    @app.route("/api/oracle/analyze", methods=["POST"])
    @limiter.limit("30 per minute", exempt_when=lambda: current_app.config.get("TESTING"))
    @handle_errors
    def oracle_analyze():
        """
        Pivote 5: Oráculo de Laplace (Nivel 3 - FÍSICA).
        Recibe parámetros físicos (R, L, C) y devuelve la Pirámide de Laplace.
        """
        if not request.is_json:
            return jsonify(
                {
                    "error": "Content-Type must be application/json",
                    "code": "INVALID_CONTENT_TYPE",
                }
            ), 400

        data = request.get_json()
        R = data.get("R")
        L = data.get("L")
        C = data.get("C")

        if any(p is None for p in [R, L, C]):
            return jsonify(
                {
                    "error": "Faltan parámetros: R, L, C",
                    "code": "MISSING_PARAMS",
                }
            ), 400

        try:
            payload = {
                "R": float(R),
                "L": float(L),
                "C": float(C)
            }
        except (ValueError, TypeError):
            return jsonify(
                {"error": "Parámetros deben ser numéricos", "code": "INVALID_PARAMS"}
            ), 400

        # Contexto: Laplace es TACTICS (Nivel 2) en el plan, pero requiere PHYSICS validado
        # Sin embargo, como herramienta aislada, a veces se permite si es 'Calculadora'.
        # El plan dice: oracle_analyze -> Stratum.TACTICS
        context = {
            "validated_strata": session.get("validated_strata", set())
        }

        # Si es TACTICS, requiere PHYSICS.
        # Si el usuario no ha hecho login/upload, validated_strata es vacio.
        # Permitimos bypass si es herramienta de prueba o agregamos override.
        if current_app.config.get("TESTING"):
             context["force_physics_override"] = True

        result = app.mic.project_intent("oracle_analyze", payload, context)

        # Update session with TACTICS if successful (and PHYSICS logic passes)
        # Note: In project_intent, it auto-adds `_mic_validation_update` ONLY if stratum is PHYSICS
        # Here we manually upgrade if we trust TACTICS success implies PHYSICS was valid or bypassed.
        if "success" not in result:
             # Laplace returns dict directly, assume success if keys present
             if "stability_index" in result:
                 if "validated_strata" not in session:
                     session["validated_strata"] = {Stratum.TACTICS}
                 elif isinstance(session.get("validated_strata"), set):
                     session["validated_strata"].add(Stratum.TACTICS)
        else:
             if result.get("success"):
                 if "validated_strata" not in session:
                     session["validated_strata"] = {Stratum.TACTICS}
                 elif isinstance(session.get("validated_strata"), set):
                     session["validated_strata"].add(Stratum.TACTICS)

        if not result.get("success") and "success" in result:
             # Si hubo error de gatekeeper
             status_code = 400
             if result.get("error_category") == "mic_hierarchy_violation":
                status_code = 403
             return jsonify(result), status_code

        # Laplace devuelve el dict directamente, project_intent lo envuelve.
        # Si es éxito, el resultado es el retorno del handler.
        return jsonify(result)

    # ========================================================================
    # MANEJADORES DE ERRORES
    # ========================================================================

    @app.errorhandler(404)
    def not_found(error):
        """Maneja errores 404."""
        app.logger.warning(f"404 Not Found: {request.path}")
        return jsonify(
            {
                "error": "Recurso no encontrado",
                "code": "NOT_FOUND",
                "path": escape(request.path),
                "request_id": g.get("request_id", "N/A"),
            }
        ), 404

    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Maneja errores de tamaño de archivo."""
        max_mb = MAX_CONTENT_LENGTH / (1024 * 1024)
        app.logger.warning(f"413 Request Too Large: {request.path}")
        return jsonify(
            {
                "error": f"Archivo demasiado grande. Máximo: {max_mb:.1f}MB",
                "code": "FILE_TOO_LARGE",
                "max_size_bytes": MAX_CONTENT_LENGTH,
            }
        ), 413

    @app.errorhandler(429)
    def ratelimit_handler(error):
        """Maneja errores de rate limiting."""
        app.logger.warning(
            f"429 Rate Limit Exceeded: {request.remote_addr} - {request.path}"
        )
        return jsonify(
            {
                "error": "Demasiadas solicitudes. Por favor, intente más tarde.",
                "code": "RATE_LIMIT_EXCEEDED",
                "retry_after": error.description,
            }
        ), 429

    @app.errorhandler(500)
    def internal_error(error):
        """Maneja errores internos del servidor."""
        app.logger.error(f"Error 500: {str(error)}", exc_info=True)
        return jsonify(
            {
                "error": "Error interno del servidor",
                "code": "INTERNAL_ERROR",
                "message": "Por favor, contacte al administrador si el problema persiste",
                "request_id": g.get("request_id", "N/A"),
            }
        ), 500

    # ========================================================================
    # FINALIZACIÓN
    # ========================================================================

    app.logger.info("✅ Aplicación inicializada exitosamente")
    app.logger.info(f"{'=' * 60}")

    return app


# ============================================================================
# PUNTO DE ENTRADA PARA DESARROLLO
# ============================================================================

if __name__ == "__main__":
    app = create_app("development")
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=True)
