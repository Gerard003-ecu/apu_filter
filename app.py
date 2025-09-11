import pandas as pd
from flask import Flask, jsonify, render_template

# Importa la función desde tu otro archivo
from procesador_datos import process_files

app = Flask(__name__)

# --- Cargar y procesar los datos una sola vez al inicio ---
try:
    # Define las rutas a tus archivos Excel
    presupuesto_path = "presupuesto.xlsx"
    apus_path = "apus.xlsx"
    insumos_path = "insumos.xlsx"

    # Llama a tu función para obtener el DataFrame final y procesado
    df_consolidado = process_files(presupuesto_path, apus_path, insumos_path)

    if not df_consolidado.empty:
        print("DataFrame consolidado cargado y procesado exitosamente.")
        print(f"Columnas del DataFrame final: {df_consolidado.columns.tolist()}")
    else:
        print(
            "El procesamiento de archivos resultó en un DataFrame vacío."
            "Revisa 'procesador_datos.py' y los archivos de Excel."
        )

except FileNotFoundError:
    print(
        "Error: No se encontró el archivo necesario.",
            "Asegúrate de que los archivos .xlsx estén ",
            "presentes en la misma carpeta que app.py. Detalle: {e}"
    )
    df_consolidado = (
        pd.DataFrame()
    )  # Crea un DataFrame vacío para evitar errores en la ejecución de la app


@app.route("/")
def home():
    # Asegúrate de tener un template 'index.html'
    return render_template("index.html")


@app.route("/filtrar", methods=["POST"])
def filtrar_datos():
    # La lógica de filtrado puede ser más compleja, pero por ahora devolvemos todo
    if not df_consolidado.empty:
        # Convertimos los NaN (Not a Number) a None para que JSON no de errores
        data_to_return = df_consolidado.where(pd.notna(df_consolidado), None).to_dict(
            "records"
        )
        total_records = len(data_to_return)
    else:
        data_to_return = []
        total_records = 0

    return jsonify({"data": data_to_return, "total": total_records})


if __name__ == "__main__":
    app.run(port=5001, debug=True)
