# models/probability_models.py

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def sanitize_value(value: Any) -> Optional[float]:
    """
    Convierte valores NaN de numpy/pandas a None, dejando otros valores sin modificar.
    Útil para serialización JSON o integración con sistemas que no manejan NaN.
    
    Args:
        value: Cualquier valor (int, float, np.nan, pd.NA, etc.)
    
    Returns:
        None si es NaN/NA, valor original en otro caso.
    """
    # Si es un iterable no string, devolver tal cual
    if isinstance(value, (list, tuple)):
        return value

    if pd.isna(value):
        return None

    return float(value) if isinstance(value, (np.floating, np.integer)) else value


def run_monte_carlo_simulation(
    apu_details: List[Dict[str, Any]],
    num_simulations: int = 1000,
    volatility_factor: float = 0.10,
    min_cost_threshold: float = 0.0,
    log_warnings: bool = False
) -> Dict[str, Optional[float]]:
    """
    Ejecuta una simulación de Monte Carlo para estimar el costo total de una lista de APU.
    
    Modelo:
        - Cada APU tiene un costo base (VR_TOTAL) y una cantidad (CANTIDAD).
        - El costo total simulado = Σ (VR_TOTAL * CANTIDAD * factor_de_variabilidad)
        - La variabilidad se modela con una distribución normal centrada en el costo base,
          con desviación estándar = VR_TOTAL * CANTIDAD * volatility_factor.
        - Los valores negativos se truncan a 0 (costos no pueden ser negativos).
        - Se ignora cualquier APU con VR_TOTAL <= min_cost_threshold o CANTIDAD <= 0.
    
    Args:
        apu_details: Lista de diccionarios con claves 'VR_TOTAL' y 'CANTIDAD'.
        num_simulations: Número de simulaciones a realizar (default: 1000).
        volatility_factor: Factor de volatilidad relativo al costo base (default: 0.10 = 10%).
        min_cost_threshold: Umbral mínimo para considerar un costo válido (default: 0.0).
        log_warnings: Si True, imprime advertencias sobre datos problemáticos.
    
    Returns:
        Dict con estadísticas de la simulación:
            - mean: Valor promedio del costo total simulado
            - std_dev: Desviación estándar del costo total
            - percentile_5: Percentil 5
            - percentile_95: Percentil 95
        Todos los valores son float o None si no hay datos válidos.
    
    Raises:
        ValueError: Si num_simulations <= 0 o volatility_factor < 0.
    """

    # Validación de entradas
    if not isinstance(apu_details, list):
        if log_warnings:
            print("WARNING: apu_details no es una lista. Devolviendo ceros.")
        return {
            'mean': None,
            'std_dev': None,
            'percentile_5': None,
            'percentile_95': None
        }

    if not isinstance(num_simulations, int) or num_simulations <= 0:
        raise ValueError("num_simulations debe ser un entero positivo.")

    if not isinstance(volatility_factor, (int, float)) or volatility_factor < 0:
        raise ValueError("volatility_factor debe ser un número no negativo.")

    # Convertir a DataFrame y limpiar
    df = pd.DataFrame(apu_details)

    # Validar columnas requeridas
    required_cols = {'VR_TOTAL', 'CANTIDAD'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        if log_warnings:
            print(f"WARNING: Faltan columnas requeridas: {missing_cols}. Devolviendo ceros.")
        return {
            'mean': None,
            'std_dev': None,
            'percentile_5': None,
            'percentile_95': None
        }

    # Convertir a numérico y manejar errores
    df['VR_TOTAL'] = pd.to_numeric(df['VR_TOTAL'], errors='coerce')
    df['CANTIDAD'] = pd.to_numeric(df['CANTIDAD'], errors='coerce')

    # Filtrar filas inválidas
    valid_mask = (
        df['VR_TOTAL'].notna() &
        df['CANTIDAD'].notna() &
        (df['VR_TOTAL'] > min_cost_threshold) &
        (df['CANTIDAD'] > 0)
    )
    df_valid = df[valid_mask].copy()

    # Log de datos descartados
    if log_warnings and len(df) > 0:
        discarded = len(df) - len(df_valid)
        if discarded > 0:
            print(f"WARNING: Se descartaron {discarded} filas por valores inválidos "
                  f"(VR_TOTAL <= {min_cost_threshold} o CANTIDAD <= 0 o NaN).")

    # Si no hay datos válidos, retornar None
    if len(df_valid) == 0:
        if log_warnings:
            print("WARNING: No hay datos válidos después del filtrado. Devolviendo ceros.")
        return {
            'mean': None,
            'std_dev': None,
            'percentile_5': None,
            'percentile_95': None
        }

    # Calcular costo base por fila: VR_TOTAL * CANTIDAD
    df_valid['base_cost'] = df_valid['VR_TOTAL'] * df_valid['CANTIDAD']

    # Vectorizar la simulación: matriz de (num_simulations, n_rows)
    # Cada fila es una simulación, cada columna es un APU
    base_costs = df_valid['base_cost'].values  # shape: (n_apus,)
    scales = base_costs * volatility_factor   # desviación estándar por APU

    # Generar matriz de simulaciones: normal(loc=base, scale=scale)
    # shape: (num_simulations, n_apus)
    simulated_costs_matrix = np.random.normal(
        loc=base_costs,
        scale=scales,
        size=(num_simulations, len(base_costs))
    )

    # Truncar negativos a 0
    simulated_costs_matrix = np.maximum(simulated_costs_matrix, 0)

    # Sumar por fila (por simulación)
    total_simulated_costs = simulated_costs_matrix.sum(axis=1)

    # Validar que no sea un array vacío (aunque debería ser imposible)
    if len(total_simulated_costs) == 0:
        if log_warnings:
            print("WARNING: Resultado de simulación vacío. Esto no debería ocurrir.")
        return {
            'mean': None,
            'std_dev': None,
            'percentile_5': None,
            'percentile_95': None
        }

    # Calcular estadísticas
    mean_cost = float(np.mean(total_simulated_costs))
    std_dev_cost = float(np.std(total_simulated_costs))
    p5 = float(np.percentile(total_simulated_costs, 5))
    p95 = float(np.percentile(total_simulated_costs, 95))

    # Sanitizar resultados para evitar NaN en JSON
    return {
        'mean': sanitize_value(mean_cost),
        'std_dev': sanitize_value(std_dev_cost),
        'percentile_5': sanitize_value(p5),
        'percentile_95': sanitize_value(p95)
    }
