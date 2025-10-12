import json
import os
import sys
import time
import uuid

from flask import Flask, jsonify, render_template, request, session
from werkzeug.utils import secure_filename

# Asegurarse de que el directorio raíz esté en el sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config_by_name
from models.probability_models import run_monte_carlo_simulation

from .estimator import calculate_estimate
from .procesador_csv import process_all_files

# Almacenamiento en memoria para sesiones de usuario
user_sessions = {}
SESSION_TIMEOUT = 3600  # 1 hora

def create_app(config_name):
    """Fábrica de aplicaciones que crea y configura la aplicación Flask."""
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

    # Registrar blueprints y configurar extensiones aquí si las hubiera
    # Ejemplo: app.register_blueprint(mi_blueprint)

    def cleanup_expired_sessions():
        """Elimina sesiones expiradas."""
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
        """Sirve la página principal."""
        return render_template("index.html")

    @app.before_request
    def before_request_func():
        """Se ejecuta antes de cada solicitud."""
        cleanup_expired_sessions()
        if "session_id" in session and session["session_id"] in user_sessions:
            user_sessions[session["session_id"]]["last_activity"] = time.time()

    @app.route("/upload", methods=["POST"])
    def upload_files():
        """Endpoint para la subida de archivos."""
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
                file_paths["presupuesto"], file_paths["apus"], file_paths["insumos"]
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
                return jsonify(processed_data), 500

            # --- INICIO DE LA CORRECCIÓN ---
            try:
                # Forzar una doble conversión para limpiar tipos de datos no serializables
                # 1. Convertir el diccionario a una cadena JSON (esto manejará NaNs, etc.)
                json_string = json.dumps(processed_data, allow_nan=False)
                # 2. Convertir la cadena de nuevo a un objeto Python puro
                clean_data = json.loads(json_string)
                # 3. Devolver el objeto limpio con jsonify
                return jsonify(clean_data)
            except (TypeError, ValueError) as e:
                app.logger.error(f"Error de serialización JSON en /upload: {e}")
                return jsonify({"error": "Los datos procesados no pudieron ser convertidos a JSON."}), 500
            # --- FIN DE LA CORRECCIÓN ---

        except Exception as e:
            app.logger.error(f"Error en upload_files: {str(e)}")
            return jsonify({"error": "Error interno del servidor"}), 500

    def get_user_data():
        """Obtiene los datos de la sesión del usuario."""
        if "session_id" not in session:
            return None, jsonify({"error": "Sesión no iniciada"}), 401

        session_id = session["session_id"]
        if session_id not in user_sessions:
            return None, jsonify({"error": "Sesión expirada"}), 401

        return user_sessions[session_id]["data"], None, 200

    @app.route("/api/apu/<code>", methods=["GET"])
    def get_apu_detail(code):
        """
        Devuelve los detalles de un APU, con agrupación de mano de obra
        y asegurando consistencia de datos para la simulación. VERSIÓN FINAL.
        """
        user_data, error_response, status_code = get_user_data()
        if error_response:
            return error_response, status_code

        apu_code = code.replace("%2C", ",")

        all_apu_details = user_data.get("apus_detail", [])
        if not isinstance(all_apu_details, list):
            return jsonify({"error": "Formato de datos de apus_detail incorrecto"}), 500

        apu_details_for_code = [
            item for item in all_apu_details if item.get("CODIGO_APU") == apu_code
        ]

        if not apu_details_for_code:
            return jsonify({"error": "APU no encontrado"}), 404

        # --- INICIO DE LA LÓGICA DE AGRUPACIÓN Y CONSISTENCIA ---
        import pandas as pd

        # Usar Pandas para una agrupación robusta
        df_details = pd.DataFrame(apu_details_for_code)

        df_mano_de_obra = df_details[df_details['CATEGORIA'] == 'MANO DE OBRA']
        df_otros = df_details[df_details['CATEGORIA'] != 'MANO DE OBRA']

        # La lista final que se usará para TODO
        apu_details_procesados = df_otros.to_dict('records')

        if not df_mano_de_obra.empty:
            # --- INICIO DE LA CORRECCIÓN ---
            df_mo_agrupado = df_mano_de_obra.groupby('DESCRIPCION').agg(
                CANTIDAD=('CANTIDAD', 'sum'),
                VR_TOTAL=('VR_TOTAL', 'sum'),
                UNIDAD=('UNIDAD', 'first'),
                VR_UNITARIO=('VR_UNITARIO', 'first'),
                CATEGORIA=('CATEGORIA', 'first')
            ).reset_index()
            # --- FIN DE LA CORRECCIÓN ---

            apu_details_procesados.extend(df_mo_agrupado.to_dict('records'))

        # --- FIN DE LA LÓGICA DE AGRUPACIÓN Y CONSISTENCIA ---

        presupuesto_item = next(
            (item for item in user_data.get("presupuesto", []) if item.get("CODIGO_APU") == apu_code),
            None,
        )

        desglose = {}
        for item in apu_details_procesados: # Usar la lista procesada
            categoria = item.get("CATEGORIA", "INDEFINIDO")
            if categoria not in desglose:
                desglose[categoria] = []
            desglose[categoria].append(item)

        # Usar la misma lista procesada para la simulación
        simulation_results = run_monte_carlo_simulation(apu_details_procesados)

        response = {
            "codigo": apu_code,
            "descripcion": (
                presupuesto_item.get("original_description", "")
                if presupuesto_item
                else ""
            ),
            "desglose": desglose,
            "simulation": simulation_results,
        }
        return jsonify(response)

    @app.route("/api/estimate", methods=["POST"])
    def get_estimate():
        """Calcula una estimación basada en los parámetros."""
        user_data, error_response, status_code = get_user_data()
        if error_response:
            return error_response, status_code

        if not request.is_json:
            return jsonify({"error": "La solicitud debe ser JSON"}), 400

        params = request.get_json()
        if not params:
            return jsonify({"error": "No se proporcionaron parámetros"}), 400

        try:
            result = calculate_estimate(params, user_data)
            if "error" in result:
                return jsonify(result), 400
            return jsonify(result)
        except Exception as e:
            app.logger.error(f"Error en get_estimate: {str(e)}")
            return jsonify({"error": "Error interno al calcular"}), 500

    @app.route("/api/health", methods=["GET"])
    def health_check():
        """Verifica el estado de la aplicación."""
        return jsonify({"status": "ok", "active_sessions": len(user_sessions)})

    # Manejadores de errores
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Endpoint no encontrado"}), 404

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"Error interno del servidor: {str(error)}")
        return jsonify({"error": "Error interno del servidor"}), 500

    @app.errorhandler(413)
    def too_large(error):
        return jsonify({"error": "El archivo es demasiado grande"}), 413

    return app
