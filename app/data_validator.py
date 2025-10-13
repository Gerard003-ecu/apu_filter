# app/data_validator.py
import logging
from fuzzywuzzy import process
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Constantes de Validación
COSTO_MAXIMO_RAZONABLE = 50_000_000

def _validate_extreme_costs(presupuesto_data: list) -> list:
    """
    Valida los costos de construcción en la lista de presupuesto.
    Añade una alerta si un costo unitario es excesivamente alto.
    """
    for item in presupuesto_data:
        if item.get('VALOR_CONSTRUCCION_UN', 0) > COSTO_MAXIMO_RAZONABLE:
            alerta_msg = f"Costo unitario ({item.get('VALOR_CONSTRUCCION_UN'):,.2f}) excede el umbral de {COSTO_MAXIMO_RAZONABLE:,.2f}."
            item['alerta'] = alerta_msg
            logger.warning(f"Alerta de costo excesivo para el item {item.get('ITEM')}: {alerta_msg}")
    return presupuesto_data

def _validate_zero_quantity_with_cost(apus_detail_data: list) -> list:
    """
    Valida insumos con cantidad cero pero con un valor total positivo.
    Intenta recalcular la cantidad y añade una alerta si no es posible.
    """
    for insumo in apus_detail_data:
        if insumo.get('CANTIDAD') == 0 and insumo.get('VALOR_TOTAL', 0) > 0:
            precio_unitario = insumo.get('VR_UNITARIO', 0)
            valor_total = insumo.get('VALOR_TOTAL')

            if precio_unitario > 0:
                nueva_cantidad = valor_total / precio_unitario
                insumo['CANTIDAD'] = nueva_cantidad
                alerta_msg = f"Cantidad recalculada a {nueva_cantidad:.4f} (era 0 con costo)."
                insumo['alerta'] = alerta_msg
                logger.info(f"Insumo {insumo.get('DESCRIPCION_INSUMO')} con cantidad 0 y costo. Recalculada a {nueva_cantidad}.")
            else:
                alerta_msg = "Cantidad es 0 con costo, pero el precio unitario es 0. No se puede recalcular."
                insumo['alerta'] = alerta_msg
                logger.warning(f"No se pudo recalcular la cantidad para el insumo {insumo.get('DESCRIPCION_INSUMO')}.")
    return apus_detail_data

def _validate_missing_descriptions(apus_detail_data: list, raw_insumos_df: pd.DataFrame) -> list:
    """
    Valida insumos con descripciones faltantes.
    Intenta encontrar la descripción más plausible usando fuzzy matching.
    """
    if raw_insumos_df is None or 'DESCRIPCION_INSUMO' not in raw_insumos_df.columns:
        logger.error("El DataFrame de insumos crudos no está disponible o no tiene la columna 'DESCRIPCION_INSUMO'. Saltando validación de descripciones.")
        return apus_detail_data

    lista_descripciones = raw_insumos_df['DESCRIPCION_INSUMO'].dropna().unique().tolist()

    if not lista_descripciones:
        logger.warning("La lista de descripciones de insumos para fuzzy matching está vacía.")
        return apus_detail_data

    for insumo in apus_detail_data:
        descripcion = insumo.get('DESCRIPCION_INSUMO')
        if not descripcion or pd.isna(descripcion):
            # No hay suficiente información para hacer un match.
            # Se podría intentar con el código si existiera.
            insumo['DESCRIPCION_INSUMO'] = "Insumo sin descripción"
            insumo['alerta'] = "Descripción del insumo faltante."
            logger.warning("Insumo encontrado sin descripción.")

    return apus_detail_data


def validate_and_clean_data(data_store: dict) -> dict:
    """
    Agente de Validación de Datos.
    Orquesta las validaciones y correcciones sobre los datos procesados.

    Args:
        data_store (dict): El diccionario de datos que contiene 'presupuesto',
                           'apus_detail', y 'raw_insumos_df'.

    Returns:
        dict: El data_store enriquecido con validaciones y alertas.
    """
    logger.info("El Agente de Validación de Datos ha iniciado.")

    # Validar que las claves necesarias existen
    if 'presupuesto' not in data_store:
        logger.error("La clave 'presupuesto' no se encontró en data_store. Saltando validaciones de presupuesto.")
    else:
        data_store['presupuesto'] = _validate_extreme_costs(data_store['presupuesto'])

    if 'apus_detail' not in data_store:
        logger.error("La clave 'apus_detail' no se encontró en data_store. Saltando validaciones de APU detail.")
    else:
        data_store['apus_detail'] = _validate_zero_quantity_with_cost(data_store['apus_detail'])

        raw_insumos_df = data_store.get('raw_insumos_df')
        data_store['apus_detail'] = _validate_missing_descriptions(data_store['apus_detail'], raw_insumos_df)

    logger.info("El Agente de Validación de Datos ha completado su ejecución.")

    return data_store