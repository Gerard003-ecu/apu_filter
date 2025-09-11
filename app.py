import pandas as pd
from flask import Flask, jsonify, render_template

# Importa la nueva función desde procesador_csv
from procesador_csv import process_csv_files

app = Flask(__name__)

# --- Cargar y procesar los datos una sola vez al inicio ---
df_consolidado = pd.DataFrame()
try:
    # Asegúrate de que los nombres de archivo coincidan con los que tienes
    presupuesto_path = "presupuesto.csv"
    apus_path = "apus.csv"
    insumos_path = "insumos.csv"

    # Llama a la nueva función para procesar los archivos CSV
    df_consolidado = process_csv_files(presupuesto_path, apus_path, insumos_path)

    if not df_consolidado.empty:
        print("DataFrame consolidado desde CSV cargado y procesado exitosamente.")
    else:
        print("El procesamiento de archivos CSV resultó en un DataFrame vacío.")

except FileNotFoundError as e:
    print(f"Error: No se encontró el archivo necesario. Detalle: {e}")
except Exception as e:
    print(f"Ocurrió un error general al iniciar la app: {e}")


@app.route("/")
def home():
    """Renderiza la página principal."""
    return render_template("index.html")


@app.route("/filtrar", methods=["POST"])
def filtrar_datos():
    """Devuelve los datos procesados en formato JSON."""
    total = 0
    if not df_consolidado.empty:
        # Rellena valores nulos (NaN) para evitar errores en JSON
        df_display = df_consolidado.fillna(0)
        data_to_return = df_display.to_dict("records")
        total = df_display["Valor Total"].sum()
    else:
        data_to_return = []

    return jsonify({"data": data_to_return, "total": total})


if __name__ == "__main__":
    app.run(port=5001, debug=True)
