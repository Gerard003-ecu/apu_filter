import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Cargar y preparar los datos una sola vez al inicio
try:
    df_presupuesto = pd.read_csv('presupuesto.csv')
    df_apus = pd.read_csv('apus.csv')
    df_insumos = pd.read_csv('insumos.csv')

    # Aquí se consolidan los DataFrames según la lógica de tu aplicación
    # Por ejemplo, una simple concatenación para este ejemplo
    df_consolidado = pd.concat([df_presupuesto, df_apus, df_insumos], ignore_index=True)
    print("DataFrame consolidado cargado exitosamente.")

except FileNotFoundError as e:
    print(f"Error: No se encontró el archivo necesario. Asegúrate de que los archivos .csv estén en la misma carpeta que app.py. Detalle: {e}")
    df_consolidado = pd.DataFrame() # Crea un DataFrame vacío para evitar errores

@app.route('/')
def home():
    return "¡La aplicación está en funcionamiento!"

@app.route('/filtrar', methods=['POST'])
def filtrar_datos():
    # Ahora la variable df_consolidado está disponible en este ámbito
    data_to_return = df_consolidado.to_dict('records')

    return jsonify({
        'data': data_to_return
    })

if __name__ == '__main__':
    app.run(port=5002, debug=True)