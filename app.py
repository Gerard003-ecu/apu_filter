import os

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

# Importamos la nueva función orquestadora del procesador
from procesador_csv import process_all_files

app = Flask(__name__)

# Configuración para la carpeta donde se guardarán temporalmente los archivos subidos
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# Crea la carpeta si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Almacenaremos los datos procesados en memoria para la sesión del usuario.
# Esto es simple; para una app multiusuario se usaría una base de datos.
processed_data_store = {}


@app.route("/")
def index():
    """Sirve la página principal del dashboard interactivo."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_files():
    """
    Recibe los tres archivos CSV desde el frontend, los guarda temporalmente,
    los procesa usando el módulo `procesador_csv`, y devuelve los datos
    consolidados en formato JSON.
    """
    global processed_data_store  # Usamos la variable global para almacenar los datos

    # 1. Verificar que todos los archivos esperados estén en la solicitud
    if (
        "presupuesto" not in request.files
        or "apus" not in request.files
        or "insumos" not in request.files
    ):
        return jsonify({"error": "Faltan uno o más archivos en la solicitud"}), 400

    presupuesto_file = request.files["presupuesto"]
    apus_file = request.files["apus"]
    insumos_file = request.files["insumos"]

    if (
        presupuesto_file.filename == ""
        or apus_file.filename == ""
        or insumos_file.filename == ""
    ):
        return jsonify({"error": "Uno o más archivos no fueron seleccionados"}), 400

    # 2. Guardar los archivos de forma segura en el servidor
    presupuesto_path = os.path.join(
        app.config["UPLOAD_FOLDER"], secure_filename(presupuesto_file.filename)
    )
    apus_path = os.path.join(
        app.config["UPLOAD_FOLDER"], secure_filename(apus_file.filename)
    )
    insumos_path = os.path.join(
        app.config["UPLOAD_FOLDER"], secure_filename(insumos_file.filename)
    )

    presupuesto_file.save(presupuesto_path)
    apus_file.save(apus_path)
    insumos_file.save(insumos_path)

    # 3. Procesar los archivos usando la lógica importada
    processed_data_store = process_all_files(presupuesto_path, apus_path, insumos_path)

    # 4. Limpiar los archivos temporales después del procesamiento
    os.remove(presupuesto_path)
    os.remove(apus_path)
    os.remove(insumos_path)

    # Si hubo un error durante el procesamiento, devolverlo al frontend
    if "error" in processed_data_store:
        return jsonify(processed_data_store), 500

    # Si todo salió bien, devolver el diccionario completo de datos procesados
    return jsonify(processed_data_store)


@app.route("/api/apu/<code>", methods=["GET"])
def get_apu_detail(code):
    """
    Endpoint de la API que devuelve el desglose detallado para un código APU específico.
    El frontend llama a esta ruta cuando el usuario hace clic en una
    fila del presupuesto.
    """
    global processed_data_store

    # El código del APU puede llegar con comas codificadas como '%2C' en la URL.
    apu_code = code.replace("%2C", ",")

    apu_details = processed_data_store.get("apus_detail", {}).get(apu_code)
    presupuesto_item = next(
        (
            item
            for item in processed_data_store.get("presupuesto", [])
            if item["Código APU"] == apu_code
        ),
        None,
    )

    if not apu_details or not presupuesto_item:
        return jsonify(
            {"error": "No se encontró el detalle para el código APU proporcionado"}
        ), 404

    # Agrupar los insumos del APU por categoría para mostrarlos en el modal
    desglose = {}
    for item in apu_details:
        categoria = item.get("CATEGORIA", "INDEFINIDO")
        if categoria not in desglose:
            desglose[categoria] = []
        desglose[categoria].append(item)

    response = {
        "codigo": apu_code,
        "descripcion": presupuesto_item.get("Descripción"),
        "desglose": desglose,
    }

    return jsonify(response)


if __name__ == "__main__":
    # Ejecuta la aplicación en el puerto 5002 y en modo debug
    app.run(port=5002, debug=True)
