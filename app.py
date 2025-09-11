from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from procesador_datos import process_files # Importar la nueva función

app = Flask(__name__)

# La función consolidar_datos simulada se elimina, ya que ahora usaremos process_files.

# Ruta principal para la página de la GUI
@app.route('/')
def index():
    return render_template('index.html')

# Nueva ruta para procesar los archivos y mostrar los resultados
@app.route('/process')
def process_data():
    # Definir las rutas a los archivos. En una aplicación real, esto vendría de un formulario de carga.
    base_path = os.path.dirname(__file__)
    presupuesto_path = os.path.join(base_path, 'fuentes', 'presupuesto.csv')
    apus_path = os.path.join(base_path, 'fuentes', 'apus.csv')
    insumos_path = os.path.join(base_path, 'fuentes', 'insumos.csv')

    # Llamar a la función de procesamiento
    df_resultado = process_files(presupuesto_path, apus_path, insumos_path)

    # Si el DataFrame está vacío, significa que hubo un error o no hay datos.
    if df_resultado.empty:
        return "Error al procesar los archivos o no se encontraron datos.", 500

    # Renderizar la plantilla de resultados, pasando el DataFrame convertido a HTML.
    return render_template('results.html', tables=[df_resultado.to_html(classes='data', header="true")])


# La ruta /filtrar se mantiene por si es usada por la GUI existente, pero ahora está desacoplada del procesamiento principal.
@app.route('/filtrar', methods=['POST'])
def filtrar_datos():
    # Esta ruta puede ser adaptada o eliminada si la nueva funcionalidad la reemplaza.
    # Por ahora, la dejamos con su comportamiento de simulación.
    data = {
        'CODIGO_APU': ['APU001', 'APU002', 'APU003', 'APU004'],
        'NOMBRE_APU': ['REMATE CURVO', 'TEJA ARQUITECTONICA', 'REMATE CON PINTURA', 'TEJA ARQUITECTONICA'],
        'VR. TOTAL': [227179, 194453, 164844, 113889],
        'ZONA': ['ZONA O', 'ZONA 1', 'ZONA 2', 'ZONA 3'],
        'TIPO_COSTO': ['VALOR M2 SUMINISTRO + AIU', 'VALOR M2 INSTALACION + AIU', 'VALOR M2 SUMINISTRO + AIU', 'VALOR M2 INSTALACION + AIU']
    }
    df = pd.DataFrame(data)
    
    zona = request.json.get('zona')
    tipo_costo = request.json.get('tipo_costo')
    busqueda = request.json.get('busqueda')

    df_filtrado = df.copy()
    if zona and zona != 'Todas las Zonas':
        df_filtrado = df_filtrado[df_filtrado['ZONA'] == zona]
    if tipo_costo:
        df_filtrado = df_filtrado[df_filtrado['TIPO_COSTO'] == tipo_costo]
    if busqueda:
        df_filtrado = df_filtrado[df_filtrado['CODIGO_APU'].str.contains(busqueda, case=False) | 
                                 df_filtrado['NOMBRE_APU'].str.contains(busqueda, case=False)]

    resultados_json = df_filtrado.to_dict('records')
    valor_total_consolidado = df_filtrado['VR. TOTAL'].sum()

    return jsonify({
        "data": resultados_json,
        "total": valor_total_consolidado
    })

if __name__ == '__main__':
    app.run(debug=True, port=5002)