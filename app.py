from flask import Flask, render_template, request, jsonify
import pandas as pd
import io

app = Flask(__name__)

# Función para consolidar y limpiar los datos (versión mejorada de nuestro script)
def consolidar_datos(presupuesto_file, apus_file, insumos_file):
    try:
        # Aquí iría el código para leer y consolidar los 3 archivos
        # Por simplicidad, usaremos un DataFrame de ejemplo
        
        # Simulación de lectura y consolidación de datos
        data = {
            'CODIGO_APU': ['APU001', 'APU002', 'APU003', 'APU004'],
            'NOMBRE_APU': ['REMATE CURVO', 'TEJA ARQUITECTONICA', 'REMATE CON PINTURA', 'TEJA ARQUITECTONICA'],
            'VR. TOTAL': [227179, 194453, 164844, 113889],
            'ZONA': ['ZONA O', 'ZONA 1', 'ZONA 2', 'ZONA 3'],
            'TIPO_COSTO': ['VALOR M2 SUMINISTRO + AIU', 'VALOR M2 INSTALACION + AIU', 'VALOR M2 SUMINISTRO + AIU', 'VALOR M2 INSTALACION + AIU']
        }
        df = pd.DataFrame(data)
        return df

    except Exception as e:
        return None

# Ruta principal para la página de la GUI
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para procesar la subida de archivos y aplicar filtros
@app.route('/filtrar', methods=['POST'])
def filtrar_datos():
    # La lógica de carga de archivos iría aquí
    # presupuesto_file = request.files['presupuesto_file']
    # apus_file = request.files['apus_file']
    # insumos_file = request.files['insumos_file']

    # Simulamos la carga y consolidación
    df = consolidar_datos(None, None, None)
    
    if df is None:
        return jsonify({"error": "No se pudieron procesar los archivos. Asegúrate de que son válidos."}), 400

    # Obtener los filtros de la solicitud del usuario
    zona = request.json.get('zona')
    tipo_costo = request.json.get('tipo_costo')
    busqueda = request.json.get('busqueda')

    # Aplicar los filtros
    df_filtrado = df.copy()
    if zona and zona != 'Todas las Zonas':
        df_filtrado = df_filtrado[df_filtrado['ZONA'] == zona]
    if tipo_costo:
        df_filtrado = df_filtrado[df_filtrado['TIPO_COSTO'] == tipo_costo]
    if busqueda:
        df_filtrado = df_filtrado[df_filtrado['CODIGO_APU'].str.contains(busqueda, case=False) | 
                                 df_filtrado['NOMBRE_APU'].str.contains(busqueda, case=False)]

    # Convertir el DataFrame filtrado a un formato JSON para la respuesta
    resultados_json = df_filtrado.to_dict('records')

    # Calcular el valor total consolidado
    valor_total_consolidado = df_filtrado['VR. TOTAL'].sum()

    return jsonify({
        "data": resultados_json,
        "total": valor_total_consolidado
    })

if __name__ == '__main__':
    app.run(debug=True, port=5002)