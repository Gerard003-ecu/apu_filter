import json
import os
import sys
import time
import uuid

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, session
from werkzeug.utils import secure_filename

# Asegurarse de que el directorio raíz esté en el sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config_by_name
from models.probability_models import run_monte_carlo_simulation

from .estimator import calculate_estimate
from .procesador_csv import process_all_files
from .utils import sanitize_for_json

# Almacenamiento en memoria para sesiones de usuario
user_sessions = {}
SESSION_TIMEOUT = 3600  # 1 hora

import logging

def create_app(config_name):
    """Crea y configura una instancia de la aplicación Flask.

    Utiliza un patrón de fábrica para inicializar la aplicación, cargar la
    configuración desde un objeto y un archivo JSON, y registrar las rutas y
    manejadores de errores.

    Args:
        config_name (str): El nombre del entorno de configuración a utilizar
                           (ej. 'development', 'production').

    Returns:
        Flask: La instancia de la aplicación Flask configurada.
    """
    logging.basicConfig(level=logging.DEBUG)
    app = Flask(__name__)

    # Cargar configuración desde el objeto de configuración
    app.config.from_object(config_by_name[config_name])

    # Cargar configuración específica de la aplicación desde config.json
    try:
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            app.config["APP_CONFIG"] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        app.logger.error(f"Error crítico: No se pudo cargar config.json. {e}")
        app.config["APP_CONFIG"] = {}

    # Configurar la carpeta de plantillas de forma explícita
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    app.template_folder = os.path.join(project_root, "templates")

    def cleanup_expired_sessions():
        """Elimina del almacenamiento en memoria las sesiones de usuario expiradas.

        Una sesión se considera expirada si ha estado inactiva por más tiempo
        del definido en `SESSION_TIMEOUT`.
        """
        current_time = time.time()
        expired_sessions = [
            session_id
            for session_id, data in user_sessions.items()
            if current_time - data.get("last_activity", 0) > SESSION_TIMEOUT
        ]
        for session_id in expired_sessions:
            if session_id in user_sessions:
                del user_sessions[session_id]

    @app.route("/")
    def index():
        """Renderiza y sirve la página principal de la aplicación.

        Returns:
            str: El contenido HTML de la página de inicio (index.html).
        """
        return render_template("index.html")

    @app.before_request
    def before_request_func():
        """Realiza tareas de mantenimiento antes de procesar cada solicitud.

        Se encarga de limpiar sesiones expiradas y actualizar el tiempo de
        última actividad para la sesión actual.
        """
        cleanup_expired_sessions()
        if "session_id" in session and session["session_id"] in user_sessions:
            user_sessions[session["session_id"]]["last_activity"] = time.time()

    @app.route("/upload", methods=["POST"])
    def upload_files():
        """Maneja la subida y procesamiento de archivos de datos.

        Recibe los archivos 'presupuesto', 'apus' e 'insumos', los guarda
        temporalmente, los procesa para extraer datos estructurados y almacena
        el resultado en la sesión del usuario.

        Returns:
            Response: Un objeto JSON con los datos procesados o un mensaje de
                      error si falla la subida o el procesamiento.
        """
        try:
            required_files = ["presupuesto", "apus", "insumos"]
            if not all(file in request.files for file in required_files):
                return jsonify({"error": "Faltan archivos"}), 400

            files = {name: request.files[name] for name in required_files}
            if any(file.filename == "" for file in files.values()):
                return jsonify({"error": "Archivos no seleccionados"}), 400

            if "session_id" not in session:
                session["session_id"] = str(uuid.uuid4())
            session_id = session["session_id"]

            user_dir = os.path.join(app.config["UPLOAD_FOLDER"], session_id)
            os.makedirs(user_dir, exist_ok=True)

            file_paths = {}
            for name, file in files.items():
                filename = secure_filename(file.filename)
                file_path = os.path.join(user_dir, filename)
                file.save(file_path)
                file_paths[name] = file_path

            processed_data = process_all_files(
                file_paths["presupuesto"],
                file_paths["apus"],
                file_paths["insumos"],
                config=app.config.get("APP_CONFIG", {}),
            )

            user_sessions[session_id] = {
                "data": processed_data,
                "last_activity": time.time(),
            }

            for file_path in file_paths.values():
                if os.path.exists(file_path):
                    os.remove(file_path)
            if not os.listdir(user_dir):
                os.rmdir(user_dir)

            if "error" in processed_data:
                app.logger.error(f"Error de procesamiento: {processed_data['error']}")
                return jsonify(processed_data), 500
            return jsonify(sanitize_for_json(processed_data))

        except Exception as e:
            import traceback
            with open("upload_error.log", "w") as f:
                f.write(traceback.format_exc())
            app.logger.error(f"Excepción no controlada en upload_files: {str(e)}", exc_info=True)
            return jsonify({"error": "Error interno del servidor"}), 500

    def get_user_data():
        """Recupera los datos procesados asociados a la sesión del usuario.

        Returns:
            tuple: Una tupla con los datos del usuario (dict), una respuesta
                   de error (Response o None) y un código de estado.

        """
        if "session_id" not in session:
            return None, jsonify({"error": "Sesión no iniciada"}), 401

        session_id = session["session_id"]
        if session_id not in user_sessions:
            return None, jsonify({"error": "Sesión expirada"}), 401

        return user_sessions[session_id]["data"], None, 200

    @app.route("/api/apu/<code>", methods=["GET"])
    def get_apu_detail(code):
        """Obtiene y devuelve los detalles de un APU específico.

        Busca el APU por su código, sanea los datos, agrupa los insumos de
        'MANO DE OBRA' para consolidar costos y cantidades, y ejecuta una
        simulación de Monte Carlo sobre los costos.

        Args:
            code (str): El código del APU a consultar.

        Returns:
            Response: Un objeto JSON con el desglose detallado del APU,
                      incluyendo resultados de la simulación, o un error si
                      el APU no se encuentra.
        """
        logger = app.logger
        try:
            logger.debug(f"Iniciando get_apu_detail para el código: {code}")

            user_data, error_response, status_code = get_user_data()
            if error_response:
                return error_response, status_code

            apu_code = code.replace("%2C", ",")
            all_apu_details = user_data.get("apus_detail", [])

            apu_details_for_code_raw = [
                item for item in all_apu_details if item.get("CODIGO_APU") == apu_code
            ]

            if not apu_details_for_code_raw:
                return jsonify({"error": "APU no encontrado"}), 404

            df_temp = pd.DataFrame(apu_details_for_code_raw)
            df_temp.replace({np.nan: None}, inplace=True)
            apu_details_for_code = df_temp.to_dict("records")
            logger.debug(
                f"Datos sanitizados. Encontrados {len(apu_details_for_code)} "
                f"insumos para el APU {apu_code}"
            )

            df_details = pd.DataFrame(apu_details_for_code)

            # Generalizar la agrupación para TODAS las categorías
            logger.debug("Agrupando todos los insumos por categoría y descripción...")
            apu_details_procesados = []

            # Iterar sobre cada categoría y aplicar la lógica de agrupación
            for categoria in df_details["CATEGORIA"].unique():
                df_categoria = df_details[df_details["CATEGORIA"] == categoria]

                if df_categoria.empty:
                    continue

                # Agrupar por descripción de insumo
                # Definir las agregaciones base
                aggregations = {
                    'CANTIDAD_APU': ('CANTIDAD_APU', 'sum'),
                    'VALOR_TOTAL_APU': ('VALOR_TOTAL_APU', 'sum'),
                    'RENDIMIENTO': ('RENDIMIENTO', 'sum'),
                    'UNIDAD_APU': ('UNIDAD_APU', 'first'),
                    'PRECIO_UNIT_APU': ('PRECIO_UNIT_APU', 'first'),
                    'CATEGORIA': ('CATEGORIA', 'first'),
                    'CODIGO_APU': ('CODIGO_APU', 'first'),
                    'UNIDAD_INSUMO': ('UNIDAD_INSUMO', 'first')
                }

                # Añadir agregación para 'alerta' si la columna existe
                if "alerta" in df_categoria.columns:
                    # Usamos una función lambda para unir las alertas únicas y no nulas
                    aggregations["alerta"] = (
                        "alerta",
                        lambda x: " | ".join(x.dropna().unique()),
                    )

                df_agrupado = (
                    df_categoria.groupby("DESCRIPCION_INSUMO")
                    .agg(**aggregations)
                    .reset_index()
                )

                # Renombrar columnas para consistencia en el frontend
                df_agrupado.rename(
                    columns={
                        "DESCRIPCION_INSUMO": "DESCRIPCION",
                        "CANTIDAD_APU": "CANTIDAD",
                        "VALOR_TOTAL_APU": "VR_TOTAL",
                        "UNIDAD_INSUMO": "UNIDAD",
                        "PRECIO_UNIT_APU": "VR_UNITARIO",
                    },
                    inplace=True,
                )

                apu_details_procesados.extend(df_agrupado.to_dict("records"))

            presupuesto_data = user_data.get("presupuesto", [])
            presupuesto_item = next(
                (
                    item
                    for item in presupuesto_data
                    if item.get("CODIGO_APU") == apu_code
                ),
                None,
            )

            desglose = {}
            for item in apu_details_procesados:
                categoria = item.get("CATEGORIA", "INDEFINIDO")
                if categoria not in desglose:
                    desglose[categoria] = []
                desglose[categoria].append(item)

            simulation_results = run_monte_carlo_simulation(apu_details_procesados)

            descripcion = (
                presupuesto_item.get("original_description", "")
                if presupuesto_item
                else ""
            )
            response = {
                "codigo": apu_code,
                "descripcion": descripcion,
                "desglose": desglose,
                "simulation": simulation_results,
            }
            return jsonify(sanitize_for_json(response))

        except Exception as e:
            logger.error(
                f"Excepción no controlada en get_apu_detail "
                f"para el código {code}: {e}",
                exc_info=True,
            )
            error_msg = (
                "Error interno del servidor al procesar los detalles del APU."
            )
            return jsonify({"error": error_msg}), 500

    @app.route("/api/estimate", methods=["POST"])
    def get_estimate():
        """Calcula una estimación de costos basada en parámetros de entrada.

        Utiliza los datos de la sesión del usuario y los parámetros
        proporcionados en el cuerpo de la solicitud JSON para llamar a la
        función `calculate_estimate`.

        Returns:
            Response: Un objeto JSON con el resultado de la estimación o un
                      mensaje de error si los parámetros son inválidos o
                      ocurre un error en el cálculo.
        """
        user_data, error_response, status_code = get_user_data()
        if error_response:
            return error_response, status_code

        if not request.is_json:
            return jsonify({"error": "La solicitud debe ser JSON"}), 400

        params = request.get_json()
        if not params:
            return jsonify({"error": "No se proporcionaron parámetros"}), 400

        try:
            result = calculate_estimate(
                params, user_data, app.config.get("APP_CONFIG", {})
            )
            if "error" in result:
                return jsonify(result), 400
            return jsonify(sanitize_for_json(result))
        except Exception as e:
            app.logger.error(f"Error en get_estimate: {str(e)}")
            return jsonify({"error": "Error interno al calcular"}), 500

    @app.route("/api/health", methods=["GET"])
    def health_check():
        """Proporciona un endpoint de verificación de estado.

        Returns:
            Response: Un objeto JSON indicando el estado de la aplicación y
                      el número de sesiones activas.
        """
        return jsonify({"status": "ok", "active_sessions": len(user_sessions)})

    # Manejadores de errores
    @app.errorhandler(404)
    def not_found(error):
        """Maneja los errores de 'No Encontrado' (404).

        Args:
            error: El objeto de excepción.

        Returns:
            Response: Un objeto JSON con un mensaje de error 404.
        """
        return jsonify({"error": "Endpoint no encontrado"}), 404

    @app.errorhandler(500)
    def internal_error(error):
        """Maneja los errores internos del servidor (500).

        Args:
            error: El objeto de excepción.

        Returns:
            Response: Un objeto JSON con un mensaje de error 500.
        """
        app.logger.error(f"Error interno del servidor: {str(error)}")
        return jsonify({"error": "Error interno del servidor"}), 500

    @app.errorhandler(413)
    def too_large(error):
        """Maneja los errores de 'Payload Too Large' (413).

        Args:
            error: El objeto de excepción.

        Returns:
            Response: Un objeto JSON con un mensaje de error 413.
        """
        return jsonify({"error": "El archivo es demasiado grande"}), 413

    return app
