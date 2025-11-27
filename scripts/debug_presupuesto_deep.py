import json
import os
import sys

# Añadir el directorio raíz al path para importar módulos de app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.data_loader import load_data
from app.utils import find_and_rename_columns


def debug_presupuesto():
    # 1. Configuración
    file_path = "data/presupuesto_clean.csv"  # Asegúrate que esta ruta sea correcta
    config_path = "config/config_rules.json"

    print(f"🔍 Analizando archivo: {file_path}")

    # 2. Cargar Configuración
    with open(config_path, "r") as f:
        config = json.load(f)
    column_map = config.get("presupuesto_column_map", {})
    print(f"📋 Mapa de columnas esperado: {column_map}")

    # 3. Intentar Carga con Data Loader (Auto-detección)
    print("\n--- INTENTO 1: Carga Automática (Data Loader) ---")
    result = load_data(file_path)

    if result.status.value != "SUCCESS":
        print(f"❌ Falló la carga: {result.error_message}")
        return

    df = result.data
    print(f"✅ Carga exitosa. Dimensiones: {df.shape}")
    print(f"Separador detectado: '{result.delimiter_used}'")
    print(f"Encoding usado: '{result.encoding_used}'")

    print("\n--- COLUMNAS ENCONTRADAS ---")
    print(df.columns.tolist())

    print("\n--- PRIMERAS 3 FILAS (RAW) ---")
    print(df.head(3).to_string())

    # 4. Simular Renombrado
    print("\n--- SIMULACIÓN DE RENOMBRADO ---")
    try:
        df_renamed = find_and_rename_columns(df, column_map)
        print("Columnas después de renombrar:")
        print(df_renamed.columns.tolist())

        if "CODIGO_APU" in df_renamed.columns:
            print("\n✅ ÉXITO: Se encontró la columna CODIGO_APU (o ITEM).")
        else:
            print("\n❌ FALLO: No se logró mapear 'CODIGO_APU'.")
            print(
                "Posible causa: El nombre de la columna en el CSV no coincide con el mapa."
            )

    except Exception as e:
        print(f"❌ Error durante renombrado: {e}")


if __name__ == "__main__":
    debug_presupuesto()
