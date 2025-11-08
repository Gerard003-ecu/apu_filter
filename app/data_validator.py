# app/data_validator.py
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional

import pandas as pd

# Optional fuzzy matching (install with: pip install fuzzywuzzy python-Levenshtein)
try:
    from fuzzywuzzy import process
    HAS_FUZZY = True
except ImportError:
    HAS_FUZZY = False

logger = logging.getLogger(__name__)

# Constantes de Validación
COSTO_MAXIMO_RAZONABLE = 50_000_000  # 50 millones de unidades monetarias


def _validate_extreme_costs(presupuesto_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Valida costos unitarios de construcción en el presupuesto.
    Añade alertas si un costo unitario excede el umbral razonable.
    No modifica los datos originales; devuelve una copia con alertas agregadas.

    Args:
        presupuesto_data (List[Dict]): Lista de diccionarios con claves 'VALOR_CONSTRUCCION_UN' y 'ITEM'.

    Returns:
        List[Dict]: Copia de los datos con alertas agregadas bajo la clave 'alertas' (lista).
    """
    if not isinstance(presupuesto_data, list):
        logger.error("Entrada a _validate_extreme_costs no es una lista. Retornando sin cambios.")
        return presupuesto_data

    result = deepcopy(presupuesto_data)

    for idx, item in enumerate(result):
        if not isinstance(item, dict):
            logger.warning(f"Item en presupuesto[{idx}] no es un diccionario. Saltando.")
            continue

        valor = item.get("VALOR_CONSTRUCCION_UN")
        if not isinstance(valor, (int, float)) or pd.isna(valor):
            logger.warning(f"Valor de construcción unitario inválido en item {item.get('ITEM', 'desconocido')}: {valor}")
            continue

        if valor > COSTO_MAXIMO_RAZONABLE:
            alerta_msg = (
                f"Costo unitario ({valor:,.2f}) excede el umbral de {COSTO_MAXIMO_RAZONABLE:,.2f}."
            )
            item.setdefault("alertas", []).append(alerta_msg)
            logger.warning(
                f"Alerta de costo excesivo para el item {item.get('ITEM', 'desconocido')}: {alerta_msg}"
            )

    return result


def _validate_zero_quantity_with_cost(apus_detail_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Valida insumos con cantidad cero pero valor total positivo.
    Intenta recalcular la cantidad si el precio unitario es válido.
    No sobrescribe alertas existentes; las acumula.

    Args:
        apus_detail_data (List[Dict]): Lista de diccionarios con claves 'CANTIDAD', 'VALOR_TOTAL', 'VR_UNITARIO', 'DESCRIPCION_INSUMO'.

    Returns:
        List[Dict]: Copia de los datos con alertas y posibles correcciones de cantidad.
    """
    if not isinstance(apus_detail_data, list):
        logger.error("Entrada a _validate_zero_quantity_with_cost no es una lista. Retornando sin cambios.")
        return apus_detail_data

    result = deepcopy(apus_detail_data)

    for idx, insumo in enumerate(result):
        if not isinstance(insumo, dict):
            logger.warning(f"Item en apus_detail[{idx}] no es un diccionario. Saltando.")
            continue

        cantidad = insumo.get("CANTIDAD")
        valor_total = insumo.get("VALOR_TOTAL", 0)
        precio_unitario = insumo.get("VR_UNITARIO", 0)

        # Validar tipos numéricos
        if not isinstance(cantidad, (int, float)) or pd.isna(cantidad):
            logger.warning(f"Cantidad inválida en insumo {insumo.get('DESCRIPCION_INSUMO', 'desconocido')}: {cantidad}")
            continue

        if not isinstance(valor_total, (int, float)) or pd.isna(valor_total):
            logger.warning(f"Valor total inválido en insumo {insumo.get('DESCRIPCION_INSUMO', 'desconocido')}: {valor_total}")
            continue

        if not isinstance(precio_unitario, (int, float)) or pd.isna(precio_unitario):
            logger.warning(f"Precio unitario inválido en insumo {insumo.get('DESCRIPCION_INSUMO', 'desconocido')}: {precio_unitario}")
            continue

        # Caso: cantidad == 0 pero valor_total > 0
        if cantidad == 0 and valor_total > 0:
            if precio_unitario > 0:
                nueva_cantidad = valor_total / precio_unitario
                insumo["CANTIDAD"] = nueva_cantidad
                alerta_msg = f"Cantidad recalculada a {nueva_cantidad:.4f} (era 0 con costo total de {valor_total:,.2f})."
                insumo.setdefault("alertas", []).append(alerta_msg)
                logger.info(
                    f"Insumo '{insumo.get('DESCRIPCION_INSUMO', 'desconocido')}' con cantidad 0 y costo. Recalculada a {nueva_cantidad:.4f}."
                )
            else:
                alerta_msg = (
                    "Cantidad es 0 con costo total positivo, pero precio unitario es 0 o inválido. "
                    "No se puede recalcular la cantidad."
                )
                insumo.setdefault("alertas", []).append(alerta_msg)
                logger.warning(
                    f"No se pudo recalcular cantidad para insumo '{insumo.get('DESCRIPCION_INSUMO', 'desconocido')}'."
                )

    return result


def _validate_missing_descriptions(
    apus_detail_data: List[Dict[str, Any]],
    raw_insumos_df: Optional[pd.DataFrame]
) -> List[Dict[str, Any]]:
    """
    Valida insumos con descripciones faltantes o nulas.
    Intenta reemplazarlas con una descripción similar del DataFrame de insumos crudos.
    Usa fuzzy matching si está disponible; de lo contrario, usa fallback genérico.

    Args:
        apus_detail_data (List[Dict]): Lista de diccionarios con clave 'DESCRIPCION_INSUMO'.
        raw_insumos_df (pd.DataFrame, optional): DataFrame con columna 'DESCRIPCION_INSUMO'.

    Returns:
        List[Dict]: Copia de los datos con descripciones corregidas y alertas agregadas.
    """
    if not isinstance(apus_detail_data, list):
        logger.error("Entrada a _validate_missing_descriptions no es una lista. Retornando sin cambios.")
        return apus_detail_data

    result = deepcopy(apus_detail_data)

    if not isinstance(raw_insumos_df, pd.DataFrame):
        logger.warning(
            "raw_insumos_df no es un DataFrame válido. No se puede realizar fuzzy matching. "
            "Se usarán descripciones genéricas para faltantes."
        )
        for idx, insumo in enumerate(result):
            if not isinstance(insumo, dict):
                continue
            descripcion = insumo.get("DESCRIPCION_INSUMO")
            if not descripcion or pd.isna(descripcion):
                insumo["DESCRIPCION_INSUMO"] = "Insumo sin descripción"
                insumo.setdefault("alertas", []).append("Descripción del insumo faltante y no se pudo corregir (DataFrame inválido).")
                logger.warning(f"Insumo {idx} sin descripción y sin DataFrame para corrección.")
        return result

    # Extraer descripciones válidas del DataFrame
    if "DESCRIPCION_INSUMO" not in raw_insumos_df.columns:
        logger.error(
            "Columna 'DESCRIPCION_INSUMO' no existe en raw_insumos_df. No se puede realizar fuzzy matching."
        )
        for idx, insumo in enumerate(result):
            if not isinstance(insumo, dict):
                continue
            descripcion = insumo.get("DESCRIPCION_INSUMO")
            if not descripcion or pd.isna(descripcion):
                insumo["DESCRIPCION_INSUMO"] = "Insumo sin descripción"
                insumo.setdefault("alertas", []).append("Descripción del insumo faltante y columna faltante en DataFrame.")
                logger.warning(f"Insumo {idx} sin descripción y columna 'DESCRIPCION_INSUMO' ausente en DataFrame.")
        return result

    lista_descripciones = (
        raw_insumos_df["DESCRIPCION_INSUMO"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    if not lista_descripciones:
        logger.warning("La lista de descripciones de insumos para fuzzy matching está vacía.")
        for idx, insumo in enumerate(result):
            if not isinstance(insumo, dict):
                continue
            descripcion = insumo.get("DESCRIPCION_INSUMO")
            if not descripcion or pd.isna(descripcion):
                insumo["DESCRIPCION_INSUMO"] = "Insumo sin descripción"
                insumo.setdefault("alertas", []).append("Descripción faltante y no hay referencias para corrección.")
                logger.warning(f"Insumo {idx} sin descripción y sin referencias disponibles.")
        return result

    # Procesar cada insumo
    for idx, insumo in enumerate(result):
        if not isinstance(insumo, dict):
            continue

        descripcion_actual = insumo.get("DESCRIPCION_INSUMO")
        if not descripcion_actual or pd.isna(descripcion_actual):
            # Intentar fuzzy matching
            if HAS_FUZZY:
                match = process.extractOne(str(descripcion_actual), lista_descripciones, score_cutoff=70)
                if match:
                    insumo["DESCRIPCION_INSUMO"] = match[0]
                    insumo.setdefault("alertas", []).append(
                        f"Descripción corregida por fuzzy matching: '{match[0]}' (similitud: {match[1]}%)."
                    )
                    logger.info(
                        f"Insumo {idx}: Descripción corregida a '{match[0]}' (similitud: {match[1]}%)."
                    )
                else:
                    insumo["DESCRIPCION_INSUMO"] = "Insumo sin descripción"
                    insumo.setdefault("alertas", []).append(
                        "Descripción faltante. Fuzzy matching no encontró coincidencia suficiente."
                    )
                    logger.warning(f"Insumo {idx}: Sin descripción y sin match fuzzy.")
            else:
                # Fallback sin fuzzywuzzy
                insumo["DESCRIPCION_INSUMO"] = "Insumo sin descripción"
                insumo.setdefault("alertas", []).append(
                    "Descripción faltante. Fuzzy matching no disponible. Instale 'fuzzywuzzy' para mejoras."
                )
                logger.warning(f"Insumo {idx}: Descripción faltante (fuzzywuzzy no instalado).")

    return result


def validate_and_clean_data(data_store: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orquesta las validaciones y correcciones sobre los datos procesados.
    Garantiza inmutabilidad: no modifica el diccionario original.

    Args:
        data_store (Dict): Diccionario con claves:
            - 'presupuesto': List[Dict] con datos de presupuesto.
            - 'apus_detail': List[Dict] con detalles de APU.
            - 'raw_insumos_df': pd.DataFrame opcional con descripciones de insumos.

    Returns:
        Dict: Nuevo diccionario con datos validados, corregidos y enriquecidos con alertas.
    """
    logger.info("El Agente de Validación de Datos ha iniciado.")

    # Asegurar que data_store es un diccionario
    if not isinstance(data_store, dict):
        logger.error("data_store no es un diccionario. Retornando copia vacía.")
        return {}

    # Crear copia profunda para evitar efectos secundarios
    result = deepcopy(data_store)

    # Validar presupuesto
    if "presupuesto" in result:
        try:
            result["presupuesto"] = _validate_extreme_costs(result["presupuesto"])
        except Exception as e:
            logger.error(f"Error al validar presupuesto: {str(e)}", exc_info=True)
            result["presupuesto"] = []
    else:
        logger.warning("Clave 'presupuesto' no encontrada en data_store. Saltando validación.")

    # Validar apus_detail y usar raw_insumos_df si está disponible
    if "apus_detail" in result:
        try:
            result["apus_detail"] = _validate_zero_quantity_with_cost(result["apus_detail"])
            raw_insumos_df = result.get("raw_insumos_df")
            result["apus_detail"] = _validate_missing_descriptions(result["apus_detail"], raw_insumos_df)
        except Exception as e:
            logger.error(f"Error al validar apus_detail: {str(e)}", exc_info=True)
            result["apus_detail"] = []
    else:
        logger.warning("Clave 'apus_detail' no encontrada en data_store. Saltando validación de APU.")

    logger.info("El Agente de Validación de Datos ha completado su ejecución.")
    return result
