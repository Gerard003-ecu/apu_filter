import os
import time
import uuid
from datetime import timedelta

from flask import Flask, jsonify, render_template, request, session
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

# Importamos la nueva función orquestadora y la de estimación
from procesador_csv import calculate_estimate, process_all_files

# Configuración de la aplicación
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB limit
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=1)
app.secret_key = os.environ.get("SECRET_KEY", "dev-key-change-in-production")

# Almacenamiento en memoria con expiración (para múltiples usuarios)
user_sessions = {}
SESSION_TIMEOUT = 3600  # 1 hora en segundos


def cleanup_expired_sessions():
    """Elimina sesiones expiradas periódicamente."""
    current_time = time.time()
    expired_sessions = [
        session_id
        for session_id, data in user_sessions.items()
        if current_time - data["last_activity"] > SESSION_TIMEOUT
    ]
    for session_id in expired_sessions:
        if session_id in user_sessions:
            del user_sessions[session_id]


@app.route("/")
def index():
    """Sirve la página principal del dashboard interactivo."""
    return render_template("index.html")


@app.before_request
def before_request():
    """Limpia sesiones expiradas antes de cada solicitud."""
    cleanup_expired_sessions()

    # Actualizar actividad de la sesión actual
    if "session_id" in session and session["session_id"] in user_sessions:
        user_sessions[session["session_id"]]["last_activity"] = time.time()


@app.route("/upload", methods=["POST"])
def upload_files():
    """Endpoint mejorado para subida de archivos con mejor manejo de errores."""
    try:
        # Verificar que todos los archivos estén presentes
        required_files = ["presupuesto", "apus", "insumos"]
        if not all(file in request.files for file in required_files):
            return jsonify({"error": "Faltan uno o más archivos en la solicitud"}), 400

        # Verificar que los archivos no estén vacíos
        files = {name: request.files[name] for name in required_files}
        if any(file.filename == "" for file in files.values()):
            return jsonify({"error": "Uno o más archivos no fueron seleccionados"}), 400

        # Verificar extensiones de archivo
        for name, file in files.items():
            if not file.filename.lower().endswith(".csv"):
                return jsonify({"error": f"El archivo {name} debe ser un CSV"}), 400

        # Crear directorio de sesión único para el usuario
        if "session_id" not in session:
            session["session_id"] = str(uuid.uuid4())

        session_id = session["session_id"]
        user_dir = os.path.join(app.config["UPLOAD_FOLDER"], session_id)
        os.makedirs(user_dir, exist_ok=True)

        # Guardar archivos con nombres únicos
        file_paths = {}
        for name, file in files.items():
            filename = f"{name}_{int(time.time())}_{secure_filename(file.filename)}"
            file_path = os.path.join(user_dir, filename)
            file.save(file_path)
            file_paths[name] = file_path

        # Procesar archivos
        processed_data = process_all_files(
            file_paths["presupuesto"], file_paths["apus"], file_paths["insumos"]
        )

        # Almacenar datos en la sesión del usuario
        user_sessions[session_id] = {
            "data": processed_data,
            "last_activity": time.time(),
        }

        # Limpiar archivos temporales
        for file_path in file_paths.values():
            if os.path.exists(file_path):
                os.remove(file_path)

        # Limpiar directorio si está vacío
        if not os.listdir(user_dir):
            os.rmdir(user_dir)

        # Manejar errores de procesamiento
        if "error" in processed_data:
            return jsonify(processed_data), 500

        return jsonify(processed_data)

    except HTTPException as e:
        # Las excepciones HTTP deben ser manejadas por los error handlers de Flask.
        # Las relanzamos para que el decorador @app.errorhandler las atrape.
        raise e
    except Exception as e:
        app.logger.error(f"Error en upload_files: {str(e)}")
        return jsonify(
            {"error": "Error interno del servidor al procesar archivos"}
        ), 500


def get_user_data():
    """Obtiene los datos del usuario actual o devuelve error."""
    if "session_id" not in session:
        return None, jsonify({"error": "Sesión no iniciada"}), 401

    session_id = session["session_id"]
    if session_id not in user_sessions:
        return None, jsonify({"error": "Sesión expirada o no válida"}), 401

    return user_sessions[session_id]["data"], None, 200


@app.route("/api/apu/<code>", methods=["GET"])
def get_apu_detail(code):
    """Endpoint mejorado para detalles de APU."""
    user_data, error_response, status_code = get_user_data()
    if error_response:
        return error_response, status_code

    # Decodificar código APU
    apu_code = code.replace("%2C", ",")

    # Buscar detalles del APU
    apu_details = user_data.get("apus_detail", {}).get(apu_code)
    if not apu_details:
        return jsonify({"error": "No se encontró el APU solicitado"}), 404

    # Buscar ítem en el presupuesto
    presupuesto_item = next(
        (
            item
            for item in user_data.get("presupuesto", [])
            if item.get("CODIGO_APU") == apu_code
        ),
        None,
    )

    # Agrupar por categoría
    desglose = {}
    for item in apu_details:
        categoria = item.get("CATEGORIA", "INDEFINIDO")
        if categoria not in desglose:
            desglose[categoria] = []
        desglose[categoria].append(item)

    response = {
        "codigo": apu_code,
        "descripcion": presupuesto_item.get("DESCRIPCION_APU", "")
        if presupuesto_item
        else "",
        "desglose": desglose,
    }

    return jsonify(response)


@app.route("/api/estimate", methods=["POST"])
def get_estimate():
    """Endpoint mejorado para estimaciones."""
    user_data, error_response, status_code = get_user_data()
    if error_response:
        return error_response, status_code

    # Validar parámetros
    if not request.is_json:
        return jsonify({"error": "La solicitud debe ser JSON"}), 400

    params = request.get_json()
    if not params:
        return jsonify({"error": "No se proporcionaron parámetros"}), 400

    # Validar parámetros requeridos
    required_params = ["tipo", "material"]
    missing_params = [p for p in required_params if p not in params]
    if missing_params:
        return jsonify(
            {"error": f"Parámetros requeridos faltantes: {missing_params}"}
        ), 400

    try:
        result = calculate_estimate(params, user_data)
        if "error" in result:
            return jsonify(result), 404
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Error en get_estimate: {str(e)}")
        return jsonify({"error": "Error interno al calcular la estimación"}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    """Endpoint para verificar el estado de la aplicación."""
    return jsonify(
        {
            "status": "ok",
            "timestamp": time.time(),
            "active_sessions": len(user_sessions),
        }
    )


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


if __name__ == "__main__":
    # Ejecuta la aplicación en el puerto 5002 y en modo debug
    app.run(port=5002, debug=True)
