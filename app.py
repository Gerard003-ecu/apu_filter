import pandas as pd
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

# Cargar y preparar los datos una sola vez al inicio
try:
    # A
    df_presupuesto = pd.read_csv('presupuesto.csv', encoding='latin1')
    df_apus = pd.read_csv('apus.csv', encoding='latin1')
    df_insumos = pd.read_csv('insumos.csv', encoding='latin1')

    # B
    df_consolidado = pd.concat([df_presupuesto, df_apus, df_insumos], ignore_index=True)
    print("DataFrame consolidado cargado exitosamente.")
    print(f"Columnas del DataFrame: {df_consolidado.columns.tolist()}")

except FileNotFoundError as e:
    print(f"Error: No se encontró el archivo necesario. Asegúrate de que los archivos .csv estén en la misma carpeta que app.py. Detalle: {e}")
    df_consolidado = pd.DataFrame() # Crea un DataFrame vacío para evitar errores

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/filtrar', methods=['POST'])
def filtrar_datos():
    data_to_return = df_consolidado.to_dict('records')

    return jsonify({
        'data': data_to_return,
        'total': len(data_to_return)
    })

if __name__ == '__main__':
    app.run(port=5001, debug=True)