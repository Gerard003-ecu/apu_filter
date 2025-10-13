# En models/probability_models.py

import numpy as np
import pandas as pd

# --- INICIO DE LA CORRECCIÓN ---
def sanitize_value(value):
    """Convierte NaN de numpy a None de Python, si no, devuelve el valor original."""
    if pd.isna(value):
        return None
    return value
# --- FIN DE LA CORRECCIÓN ---

def run_monte_carlo_simulation(apu_details, num_simulations=1000):
    if not apu_details:
        return {
            'mean': 0, 'std_dev': 0, 'percentile_5': 0, 'percentile_95': 0
        }

    df = pd.DataFrame(apu_details)

    # Asegurarse de que las columnas existen y son numéricas
    if 'VR_TOTAL' not in df.columns or 'CANTIDAD' not in df.columns:
         return {
            'mean': 0, 'std_dev': 0, 'percentile_5': 0, 'percentile_95': 0
        }

    df['VR_TOTAL'] = pd.to_numeric(df['VR_TOTAL'], errors='coerce').fillna(0)
    df['CANTIDAD'] = pd.to_numeric(df['CANTIDAD'], errors='coerce').fillna(0)

    total_costs = []
    if df.empty or df['VR_TOTAL'].sum() == 0:
        # Si no hay datos o costos, devolver ceros sanitizados
        return {
            'mean': 0, 'std_dev': 0, 'percentile_5': 0, 'percentile_95': 0
        }

    for _ in range(num_simulations):
        simulated_cost = 0
        for _, row in df.iterrows():
            # Simular variabilidad solo si hay un costo base
            if row['VR_TOTAL'] > 0:
                # Asumir una desviación estándar del 10% para la simulación
                simulated_item_cost = np.random.normal(loc=row['VR_TOTAL'], scale=row['VR_TOTAL'] * 0.1)
                simulated_cost += max(0, simulated_item_cost) # Evitar costos negativos

        total_costs.append(simulated_cost)

    # Si total_costs está vacío, np.mean y otros devolverán NaN.
    if not total_costs:
        return {'mean': 0, 'std_dev': 0, 'percentile_5': 0, 'percentile_95': 0}

    # --- INICIO DE LA CORRECCIÓN ---
    # Aplicar la sanitización a cada resultado antes de devolverlo
    results = {
        'mean': sanitize_value(np.mean(total_costs)),
        'std_dev': sanitize_value(np.std(total_costs)),
        'percentile_5': sanitize_value(np.percentile(total_costs, 5)),
        'percentile_95': sanitize_value(np.percentile(total_costs, 95)),
    }
    # --- FIN DE LA CORRECCIÓN ---

    return results