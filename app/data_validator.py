# app/data_validator.py
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

# Optional fuzzy matching (install with: pip install fuzzywuzzy python-Levenshtein)
try:
    from fuzzywuzzy import process
    HAS_FUZZY = True
except ImportError:
    HAS_FUZZY = False

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES DE VALIDACIÓN
# ============================================================================
COSTO_MAXIMO_RAZONABLE = 50_000_000  # 50 millones de unidades monetarias
COSTO_MINIMO_VALIDO = 0  # No se permiten costos negativos
CANTIDAD_MINIMA_VALIDA = 0  # No se permiten cantidades negativas
FUZZY_MATCH_THRESHOLD = 70  # Umbral mínimo de similitud para fuzzy matching
TOLERANCIA_COMPARACION_FLOAT = 1e-6  # Tolerancia para comparaciones de punto flotante
TOLERANCIA_PORCENTUAL_COHERENCIA = 0.01  # 1% de tolerancia para validación matemática
DESCRIPCION_GENERICA_FALLBACK = "Insumo sin descripción"
MAX_DESCRIPCION_LENGTH = 500  # Longitud máxima razonable para descripción


# ============================================================================
# ENUMS Y DATACLASSES
# ============================================================================
class TipoAlerta(Enum):
    """Tipos de alertas para clasificación"""
    COSTO_EXCESIVO = "COSTO_EXCESIVO"
    COSTO_NEGATIVO = "COSTO_NEGATIVO"
    CANTIDAD_RECALCULADA = "CANTIDAD_RECALCULADA"
    CANTIDAD_INVALIDA = "CANTIDAD_INVALIDA"
    DESCRIPCION_CORREGIDA = "DESCRIPCION_CORREGIDA"
    DESCRIPCION_FALTANTE = "DESCRIPCION_FALTANTE"
    INCOHERENCIA_MATEMATICA = "INCOHERENCIA_MATEMATICA"
    VALOR_INFINITO = "VALOR_INFINITO"
    CAMPO_REQUERIDO_FALTANTE = "CAMPO_REQUERIDO_FALTANTE"


@dataclass
class ValidationMetrics:
    """Métricas de validación para reporting"""
    total_items_procesados: int = 0
    items_con_alertas: int = 0
    costos_excesivos: int = 0
    costos_negativos: int = 0
    cantidades_recalculadas: int = 0
    descripciones_corregidas: int = 0
    incoherencias_matematicas: int = 0
    valores_infinitos: int = 0
    items_con_errores: int = 0
    alertas_por_tipo: Dict[str, int] = field(default_factory=dict)

    def agregar_alerta(self, tipo_alerta: TipoAlerta) -> None:
        """Registra una alerta en las métricas"""
        tipo_str = tipo_alerta.value
        self.alertas_por_tipo[tipo_str] = self.alertas_por_tipo.get(tipo_str, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        """Convierte las métricas a diccionario"""
        return {
            "total_items_procesados": self.total_items_procesados,
            "items_con_alertas": self.items_con_alertas,
            "costos_excesivos": self.costos_excesivos,
            "costos_negativos": self.costos_negativos,
            "cantidades_recalculadas": self.cantidades_recalculadas,
            "descripciones_corregidas": self.descripciones_corregidas,
            "incoherencias_matematicas": self.incoherencias_matematicas,
            "valores_infinitos": self.valores_infinitos,
            "items_con_errores": self.items_con_errores,
            "alertas_por_tipo": self.alertas_por_tipo
        }


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================
def _es_numero_valido(valor: Any) -> bool:
    """
    Verifica si un valor es un número válido (no None, no NaN, no infinito).
    
    Args:
        valor: Valor a verificar
        
    Returns:
        bool: True si es un número válido, False en caso contrario
    """
    if valor is None:
        return False
    if not isinstance(valor, (int, float)):
        return False
    if pd.isna(valor):
        return False
    if np.isinf(valor):
        return False
    return True


def _agregar_alerta(
    item: Dict[str, Any],
    mensaje: str,
    tipo_alerta: TipoAlerta,
    metrics: Optional[ValidationMetrics] = None
) -> None:
    """
    Agrega una alerta a un item y actualiza métricas.
    
    Args:
        item: Diccionario del item
        mensaje: Mensaje de alerta
        tipo_alerta: Tipo de alerta (enum)
        metrics: Objeto de métricas opcional
    """
    if "alertas" not in item:
        item["alertas"] = []
    
    alerta_completa = {
        "tipo": tipo_alerta.value,
        "mensaje": mensaje
    }
    item["alertas"].append(alerta_completa)
    
    if metrics:
        metrics.agregar_alerta(tipo_alerta)


def _validar_campos_requeridos(
    item: Dict[str, Any],
    campos_requeridos: List[str],
    nombre_item: str = "desconocido"
) -> List[str]:
    """
    Valida que un item contenga todos los campos requeridos.
    
    Args:
        item: Diccionario a validar
        campos_requeridos: Lista de nombres de campos requeridos
        nombre_item: Nombre del item para logging
        
    Returns:
        List[str]: Lista de campos faltantes
    """
    campos_faltantes = []
    for campo in campos_requeridos:
        if campo not in item:
            campos_faltantes.append(campo)
            logger.warning(
                f"Campo requerido '{campo}' faltante en item '{nombre_item}'"
            )
    return campos_faltantes


def _validar_coherencia_matematica(
    cantidad: float,
    precio_unitario: float,
    valor_total: float,
    tolerancia_porcentual: float = TOLERANCIA_PORCENTUAL_COHERENCIA
) -> Tuple[bool, Optional[float]]:
    """
    Valida que la relación cantidad * precio_unitario ≈ valor_total sea coherente.
    
    Args:
        cantidad: Cantidad del insumo
        precio_unitario: Precio unitario
        valor_total: Valor total reportado
        tolerancia_porcentual: Tolerancia porcentual permitida (default 1%)
        
    Returns:
        Tuple[bool, Optional[float]]: (es_coherente, diferencia_porcentual)
    """
    if not all(_es_numero_valido(v) for v in [cantidad, precio_unitario, valor_total]):
        return True, None  # No validar si hay valores inválidos
    
    # Caso especial: todos son cero
    if cantidad == 0 and precio_unitario == 0 and valor_total == 0:
        return True, 0.0
    
    # Calcular valor esperado
    valor_esperado = cantidad * precio_unitario
    
    # Si el valor esperado es cero, verificar que valor_total también lo sea
    if abs(valor_esperado) < TOLERANCIA_COMPARACION_FLOAT:
        es_coherente = abs(valor_total) < TOLERANCIA_COMPARACION_FLOAT
        return es_coherente, None
    
    # Calcular diferencia porcentual
    diferencia = abs(valor_esperado - valor_total)
    diferencia_porcentual = (diferencia / abs(valor_esperado)) * 100
    
    es_coherente = diferencia_porcentual <= (tolerancia_porcentual * 100)
    
    return es_coherente, diferencia_porcentual


def _limpiar_y_validar_descripcion(descripcion: Any) -> Tuple[Optional[str], bool]:
    """
    Limpia y valida una descripción.
    
    Args:
        descripcion: Descripción a validar
        
    Returns:
        Tuple[Optional[str], bool]: (descripcion_limpia, es_valida)
    """
    # Verificar si es None o NaN
    if descripcion is None or (isinstance(descripcion, float) and pd.isna(descripcion)):
        return None, False
    
    # Convertir a string y limpiar
    descripcion_str = str(descripcion).strip()
    
    # Verificar si está vacía
    if not descripcion_str or descripcion_str.lower() in ['nan', 'none', 'null', '']:
        return None, False
    
    # Validar longitud razonable
    if len(descripcion_str) > MAX_DESCRIPCION_LENGTH:
        logger.warning(
            f"Descripción excesivamente larga ({len(descripcion_str)} caracteres). "
            f"Truncando a {MAX_DESCRIPCION_LENGTH}."
        )
        descripcion_str = descripcion_str[:MAX_DESCRIPCION_LENGTH]
    
    return descripcion_str, True


def _obtener_identificador_item(item: Dict[str, Any]) -> str:
    """
    Obtiene un identificador legible para un item para logging.
    
    Args:
        item: Diccionario del item
        
    Returns:
        str: Identificador del item
    """
    posibles_ids = ['ITEM', 'ID', 'CODIGO', 'DESCRIPCION_INSUMO', 'DESCRIPCION']
    for campo in posibles_ids:
        if campo in item and item[campo]:
            return str(item[campo])
    return "desconocido"


# ============================================================================
# FUNCIONES DE VALIDACIÓN PRINCIPALES
# ============================================================================
def _validate_extreme_costs(
    presupuesto_data: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], ValidationMetrics]:
    """
    Valida costos unitarios de construcción en el presupuesto.
    Detecta costos excesivos, negativos e infinitos.
    
    Args:
        presupuesto_data: Lista de diccionarios con claves
                         'VALOR_CONSTRUCCION_UN' y 'ITEM'.
                         
    Returns:
        Tuple[List[Dict], ValidationMetrics]: (datos_validados, métricas)
    """
    metrics = ValidationMetrics()
    
    if not isinstance(presupuesto_data, list):
        logger.error(
            f"Entrada a _validate_extreme_costs no es una lista (tipo: {type(presupuesto_data)}). "
            "Retornando sin cambios."
        )
        return presupuesto_data, metrics
    
    if not presupuesto_data:
        logger.info("Lista de presupuesto vacía. No hay nada que validar.")
        return presupuesto_data, metrics
    
    result = deepcopy(presupuesto_data)
    metrics.total_items_procesados = len(result)
    
    for idx, item in enumerate(result):
        if not isinstance(item, dict):
            logger.warning(
                f"Item en presupuesto[{idx}] no es un diccionario (tipo: {type(item)}). Saltando."
            )
            metrics.items_con_errores += 1
            continue
        
        item_id = _obtener_identificador_item(item)
        valor = item.get("VALOR_CONSTRUCCION_UN")
        
        # Validar que el valor sea un número válido
        if not isinstance(valor, (int, float)):
            logger.warning(
                f"Valor de construcción unitario no es numérico en item '{item_id}': "
                f"{valor} (tipo: {type(valor)})"
            )
            _agregar_alerta(
                item,
                f"Valor de construcción unitario no es numérico: {valor}",
                TipoAlerta.CAMPO_REQUERIDO_FALTANTE,
                metrics
            )
            metrics.items_con_errores += 1
            continue
        
        # Validar NaN
        if pd.isna(valor):
            logger.warning(f"Valor de construcción unitario es NaN en item '{item_id}'")
            _agregar_alerta(
                item,
                "Valor de construcción unitario es NaN",
                TipoAlerta.CAMPO_REQUERIDO_FALTANTE,
                metrics
            )
            metrics.items_con_errores += 1
            continue
        
        # Validar infinito
        if np.isinf(valor):
            logger.error(
                f"Valor de construcción unitario es infinito en item '{item_id}': {valor}"
            )
            _agregar_alerta(
                item,
                f"Valor de construcción unitario es infinito: {valor}",
                TipoAlerta.VALOR_INFINITO,
                metrics
            )
            metrics.valores_infinitos += 1
            continue
        
        # Validar valores negativos
        if valor < COSTO_MINIMO_VALIDO:
            alerta_msg = (
                f"Costo unitario negativo detectado: {valor:,.2f}. "
                "Los costos no deben ser negativos."
            )
            _agregar_alerta(item, alerta_msg, TipoAlerta.COSTO_NEGATIVO, metrics)
            logger.error(f"Costo negativo en item '{item_id}': {valor:,.2f}")
            metrics.costos_negativos += 1
        
        # Validar valores excesivos
        if valor > COSTO_MAXIMO_RAZONABLE:
            alerta_msg = (
                f"Costo unitario ({valor:,.2f}) excede el umbral razonable de "
                f"{COSTO_MAXIMO_RAZONABLE:,.2f}. Verificar si es correcto."
            )
            _agregar_alerta(item, alerta_msg, TipoAlerta.COSTO_EXCESIVO, metrics)
            logger.warning(f"Costo excesivo en item '{item_id}': {valor:,.2f}")
            metrics.costos_excesivos += 1
    
    # Actualizar contador de items con alertas
    metrics.items_con_alertas = sum(1 for item in result if item.get("alertas"))
    
    logger.info(
        f"Validación de costos extremos completada: {metrics.total_items_procesados} items procesados, "
        f"{metrics.items_con_alertas} con alertas, {metrics.costos_excesivos} costos excesivos, "
        f"{metrics.costos_negativos} costos negativos."
    )
    
    return result, metrics


def _validate_zero_quantity_with_cost(
    apus_detail_data: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], ValidationMetrics]:
    """
    Valida insumos con cantidades inválidas y coherencia matemática.
    - Detecta cantidad cero con costo positivo e intenta recalcular
    - Valida cantidades negativas
    - Verifica coherencia entre cantidad * precio_unitario vs valor_total
    
    Args:
        apus_detail_data: Lista de diccionarios con claves
                         'CANTIDAD', 'VALOR_TOTAL', 'VR_UNITARIO', 'DESCRIPCION_INSUMO'.
                         
    Returns:
        Tuple[List[Dict], ValidationMetrics]: (datos_validados, métricas)
    """
    metrics = ValidationMetrics()
    
    if not isinstance(apus_detail_data, list):
        logger.error(
            f"Entrada a _validate_zero_quantity_with_cost no es una lista (tipo: {type(apus_detail_data)}). "
            "Retornando sin cambios."
        )
        return apus_detail_data, metrics
    
    if not apus_detail_data:
        logger.info("Lista de APUs detalle vacía. No hay nada que validar.")
        return apus_detail_data, metrics
    
    result = deepcopy(apus_detail_data)
    metrics.total_items_procesados = len(result)
    
    for idx, insumo in enumerate(result):
        if not isinstance(insumo, dict):
            logger.warning(
                f"Item en apus_detail[{idx}] no es un diccionario (tipo: {type(insumo)}). Saltando."
            )
            metrics.items_con_errores += 1
            continue
        
        item_id = _obtener_identificador_item(insumo)
        
        # Extraer valores
        cantidad = insumo.get("CANTIDAD")
        valor_total = insumo.get("VALOR_TOTAL", 0)
        precio_unitario = insumo.get("VR_UNITARIO", 0)
        
        # Validar tipos y valores numéricos
        valores_validos = {
            "CANTIDAD": _es_numero_valido(cantidad),
            "VALOR_TOTAL": _es_numero_valido(valor_total),
            "VR_UNITARIO": _es_numero_valido(precio_unitario)
        }
        
        campos_invalidos = [campo for campo, valido in valores_validos.items() if not valido]
        
        if campos_invalidos:
            logger.warning(
                f"Campos numéricos inválidos en insumo '{item_id}': {', '.join(campos_invalidos)}"
            )
            for campo in campos_invalidos:
                _agregar_alerta(
                    insumo,
                    f"Campo '{campo}' tiene un valor inválido: {insumo.get(campo)}",
                    TipoAlerta.CAMPO_REQUERIDO_FALTANTE,
                    metrics
                )
            metrics.items_con_errores += 1
            continue
        
        # A partir de aquí, sabemos que los valores son numéricos válidos
        cantidad = float(cantidad)
        valor_total = float(valor_total)
        precio_unitario = float(precio_unitario)
        
        # Validar cantidad negativa
        if cantidad < CANTIDAD_MINIMA_VALIDA:
            alerta_msg = (
                f"Cantidad negativa detectada: {cantidad}. "
                "Las cantidades no deben ser negativas."
            )
            _agregar_alerta(insumo, alerta_msg, TipoAlerta.CANTIDAD_INVALIDA, metrics)
            logger.error(f"Cantidad negativa en insumo '{item_id}': {cantidad}")
        
        # Validar valor_total negativo
        if valor_total < 0:
            alerta_msg = f"Valor total negativo detectado: {valor_total:,.2f}"
            _agregar_alerta(insumo, alerta_msg, TipoAlerta.COSTO_NEGATIVO, metrics)
            logger.error(f"Valor total negativo en insumo '{item_id}': {valor_total}")
            metrics.costos_negativos += 1
        
        # Validar precio_unitario negativo
        if precio_unitario < 0:
            alerta_msg = f"Precio unitario negativo detectado: {precio_unitario:,.2f}"
            _agregar_alerta(insumo, alerta_msg, TipoAlerta.COSTO_NEGATIVO, metrics)
            logger.error(f"Precio unitario negativo en insumo '{item_id}': {precio_unitario}")
            metrics.costos_negativos += 1
        
        # Caso especial: cantidad == 0 pero valor_total > 0
        if abs(cantidad) < TOLERANCIA_COMPARACION_FLOAT and valor_total > TOLERANCIA_COMPARACION_FLOAT:
            if precio_unitario > TOLERANCIA_COMPARACION_FLOAT:
                nueva_cantidad = valor_total / precio_unitario
                insumo["CANTIDAD"] = nueva_cantidad
                alerta_msg = (
                    f"Cantidad recalculada de 0 a {nueva_cantidad:.6f} "
                    f"(valor total: {valor_total:,.2f}, precio unitario: {precio_unitario:,.2f})"
                )
                _agregar_alerta(insumo, alerta_msg, TipoAlerta.CANTIDAD_RECALCULADA, metrics)
                logger.info(f"Insumo '{item_id}': {alerta_msg}")
                metrics.cantidades_recalculadas += 1
                
                # Actualizar cantidad para la validación de coherencia
                cantidad = nueva_cantidad
            else:
                alerta_msg = (
                    f"Cantidad es 0 con valor total {valor_total:,.2f}, pero precio unitario "
                    f"es {precio_unitario}. No se puede recalcular la cantidad."
                )
                _agregar_alerta(insumo, alerta_msg, TipoAlerta.CANTIDAD_INVALIDA, metrics)
                logger.warning(f"Insumo '{item_id}': {alerta_msg}")
                continue  # No validar coherencia si no se pudo recalcular
        
        # Validar coherencia matemática: cantidad * precio_unitario ≈ valor_total
        es_coherente, diferencia_pct = _validar_coherencia_matematica(
            cantidad, precio_unitario, valor_total
        )
        
        if not es_coherente and diferencia_pct is not None:
            alerta_msg = (
                f"Incoherencia matemática detectada: "
                f"cantidad ({cantidad:.6f}) × precio unitario ({precio_unitario:,.2f}) = "
                f"{cantidad * precio_unitario:,.2f}, pero valor total reportado es {valor_total:,.2f}. "
                f"Diferencia: {diferencia_pct:.2f}%"
            )
            _agregar_alerta(insumo, alerta_msg, TipoAlerta.INCOHERENCIA_MATEMATICA, metrics)
            logger.warning(f"Insumo '{item_id}': {alerta_msg}")
            metrics.incoherencias_matematicas += 1
    
    # Actualizar contador de items con alertas
    metrics.items_con_alertas = sum(1 for item in result if item.get("alertas"))
    
    logger.info(
        f"Validación de cantidades completada: {metrics.total_items_procesados} items procesados, "
        f"{metrics.items_con_alertas} con alertas, {metrics.cantidades_recalculadas} cantidades recalculadas, "
        f"{metrics.incoherencias_matematicas} incoherencias matemáticas."
    )
    
    return result, metrics


def _validate_missing_descriptions(
    apus_detail_data: List[Dict[str, Any]],
    raw_insumos_df: Optional[pd.DataFrame]
) -> Tuple[List[Dict[str, Any]], ValidationMetrics]:
    """
    Valida y corrige descripciones faltantes o inválidas en insumos.
    Usa fuzzy matching cuando está disponible para encontrar descripciones similares.
    
    Args:
        apus_detail_data: Lista de diccionarios con clave 'DESCRIPCION_INSUMO'.
        raw_insumos_df: DataFrame opcional con columna 'DESCRIPCION_INSUMO'.
        
    Returns:
        Tuple[List[Dict], ValidationMetrics]: (datos_validados, métricas)
    """
    metrics = ValidationMetrics()
    
    if not isinstance(apus_detail_data, list):
        logger.error(
            f"Entrada a _validate_missing_descriptions no es una lista (tipo: {type(apus_detail_data)}). "
            "Retornando sin cambios."
        )
        return apus_detail_data, metrics
    
    if not apus_detail_data:
        logger.info("Lista de APUs detalle vacía. No hay descripciones que validar.")
        return apus_detail_data, metrics
    
    result = deepcopy(apus_detail_data)
    metrics.total_items_procesados = len(result)
    
    # Preparar lista de descripciones válidas para fuzzy matching
    lista_descripciones: List[str] = []
    
    if isinstance(raw_insumos_df, pd.DataFrame) and not raw_insumos_df.empty:
        if "DESCRIPCION_INSUMO" in raw_insumos_df.columns:
            try:
                descripciones_df = (
                    raw_insumos_df["DESCRIPCION_INSUMO"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )
                # Filtrar descripciones vacías y duplicadas
                descripciones_df = descripciones_df[descripciones_df != ""]
                lista_descripciones = descripciones_df.unique().tolist()
                
                logger.info(
                    f"Se cargaron {len(lista_descripciones)} descripciones únicas "
                    f"para fuzzy matching desde raw_insumos_df."
                )
            except Exception as e:
                logger.error(
                    f"Error al extraer descripciones de raw_insumos_df: {str(e)}",
                    exc_info=True
                )
                lista_descripciones = []
        else:
            logger.warning(
                "Columna 'DESCRIPCION_INSUMO' no existe en raw_insumos_df. "
                "No se puede realizar fuzzy matching."
            )
    else:
        logger.warning(
            "raw_insumos_df no es un DataFrame válido o está vacío. "
            "No se puede realizar fuzzy matching."
        )
    
    tiene_referencias = len(lista_descripciones) > 0
    puede_fuzzy_match = HAS_FUZZY and tiene_referencias
    
    if not puede_fuzzy_match and HAS_FUZZY:
        logger.info(
            "Fuzzy matching está disponible pero no hay descripciones de referencia."
        )
    elif not HAS_FUZZY:
        logger.info(
            "Fuzzy matching no disponible. Instale 'fuzzywuzzy' y 'python-Levenshtein' "
            "para habilitar corrección automática de descripciones."
        )
    
    # Procesar cada insumo
    for idx, insumo in enumerate(result):
        if not isinstance(insumo, dict):
            logger.warning(
                f"Item en apus_detail[{idx}] no es un diccionario (tipo: {type(insumo)}). Saltando."
            )
            metrics.items_con_errores += 1
            continue
        
        descripcion_original = insumo.get("DESCRIPCION_INSUMO")
        descripcion_limpia, es_valida = _limpiar_y_validar_descripcion(descripcion_original)
        
        if es_valida:
            # La descripción es válida, actualizar si fue limpiada
            if descripcion_limpia != descripcion_original:
                insumo["DESCRIPCION_INSUMO"] = descripcion_limpia
            continue
        
        # Descripción faltante o inválida - intentar corrección
        item_id = insumo.get("ID", insumo.get("CODIGO", f"índice_{idx}"))
        
        if puede_fuzzy_match:
            try:
                # Intentar fuzzy matching con una descripción vacía es inútil,
                # pero podríamos intentar con otros campos si están disponibles
                candidato_busqueda = None
                
                # Buscar campos alternativos que podrían contener información útil
                campos_alternativos = ['CODIGO', 'ID', 'TIPO', 'CATEGORIA']
                for campo in campos_alternativos:
                    valor_campo = insumo.get(campo)
                    if valor_campo and isinstance(valor_campo, str) and valor_campo.strip():
                        candidato_busqueda = str(valor_campo).strip()
                        break
                
                if candidato_busqueda:
                    match = process.extractOne(
                        candidato_busqueda,
                        lista_descripciones,
                        score_cutoff=FUZZY_MATCH_THRESHOLD
                    )
                    
                    if match:
                        descripcion_corregida, similitud = match[0], match[1]
                        insumo["DESCRIPCION_INSUMO"] = descripcion_corregida
                        alerta_msg = (
                            f"Descripción faltante. Corregida por fuzzy matching usando campo alternativo: "
                            f"'{descripcion_corregida}' (similitud: {similitud}%, "
                            f"basado en: {candidato_busqueda})"
                        )
                        _agregar_alerta(
                            insumo,
                            alerta_msg,
                            TipoAlerta.DESCRIPCION_CORREGIDA,
                            metrics
                        )
                        logger.info(f"Insumo '{item_id}': {alerta_msg}")
                        metrics.descripciones_corregidas += 1
                        continue
                
                # Si llegamos aquí, fuzzy matching no encontró coincidencia
                insumo["DESCRIPCION_INSUMO"] = DESCRIPCION_GENERICA_FALLBACK
                alerta_msg = (
                    f"Descripción faltante (valor original: {descripcion_original}). "
                    f"Fuzzy matching no encontró coincidencia suficiente (umbral: {FUZZY_MATCH_THRESHOLD}%)."
                )
                _agregar_alerta(
                    insumo,
                    alerta_msg,
                    TipoAlerta.DESCRIPCION_FALTANTE,
                    metrics
                )
                logger.warning(f"Insumo '{item_id}': {alerta_msg}")
                
            except Exception as e:
                logger.error(
                    f"Error en fuzzy matching para insumo '{item_id}': {str(e)}",
                    exc_info=True
                )
                insumo["DESCRIPCION_INSUMO"] = DESCRIPCION_GENERICA_FALLBACK
                _agregar_alerta(
                    insumo,
                    f"Descripción faltante. Error en fuzzy matching: {str(e)}",
                    TipoAlerta.DESCRIPCION_FALTANTE,
                    metrics
                )
        else:
            # Fuzzy matching no disponible o sin referencias
            insumo["DESCRIPCION_INSUMO"] = DESCRIPCION_GENERICA_FALLBACK
            razon = "no hay referencias disponibles" if HAS_FUZZY else "fuzzy matching no está instalado"
            alerta_msg = (
                f"Descripción faltante (valor original: {descripcion_original}). "
                f"No se pudo corregir automáticamente ({razon})."
            )
            _agregar_alerta(
                insumo,
                alerta_msg,
                TipoAlerta.DESCRIPCION_FALTANTE,
                metrics
            )
            logger.warning(f"Insumo '{item_id}': {alerta_msg}")
    
    # Actualizar contador de items con alertas
    metrics.items_con_alertas = sum(1 for item in result if item.get("alertas"))
    
    logger.info(
        f"Validación de descripciones completada: {metrics.total_items_procesados} items procesados, "
        f"{metrics.items_con_alertas} con alertas, {metrics.descripciones_corregidas} descripciones corregidas."
    )
    
    return result, metrics


# ============================================================================
# FUNCIÓN ORQUESTADORA PRINCIPAL
# ============================================================================
def validate_and_clean_data(data_store: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orquesta las validaciones y correcciones sobre los datos procesados.
    Garantiza inmutabilidad: no modifica el diccionario original.
    
    Validaciones realizadas:
    - Costos extremos (excesivos, negativos, infinitos)
    - Cantidades inválidas y coherencia matemática
    - Descripciones faltantes o inválidas
    
    Args:
        data_store: Diccionario con claves esperadas:
            - 'presupuesto': List[Dict] con datos de presupuesto
            - 'apus_detail': List[Dict] con detalles de APU
            - 'raw_insumos_df': pd.DataFrame opcional con descripciones de insumos
            
    Returns:
        Dict: Nuevo diccionario con:
            - Datos validados y corregidos
            - 'validation_summary': Resumen de validaciones realizadas
            - 'validation_metrics': Métricas detalladas por tipo de validación
    """
    logger.info("="*80)
    logger.info("Iniciando Agente de Validación de Datos")
    logger.info("="*80)
    
    # Validar entrada
    if not isinstance(data_store, dict):
        logger.error(
            f"data_store no es un diccionario (tipo: {type(data_store)}). "
            "Retornando diccionario vacío con información del error."
        )
        return {
            "error": "Entrada inválida: data_store debe ser un diccionario",
            "validation_summary": {
                "exito": False,
                "mensaje": "Validación fallida por entrada inválida"
            }
        }
    
    if not data_store:
        logger.warning("data_store está vacío. No hay datos para validar.")
        return {
            "validation_summary": {
                "exito": True,
                "mensaje": "No había datos para validar",
                "total_items_procesados": 0
            }
        }
    
    # Crear copia profunda para evitar efectos secundarios
    try:
        result = deepcopy(data_store)
    except Exception as e:
        logger.error(f"Error al crear copia profunda de data_store: {str(e)}", exc_info=True)
        return {
            "error": f"Error al copiar datos: {str(e)}",
            "validation_summary": {
                "exito": False,
                "mensaje": "Validación fallida por error en copia de datos"
            }
        }
    
    # Inicializar contenedor de métricas
    metricas_totales = {
        "presupuesto": None,
        "apus_detail_cantidad": None,
        "apus_detail_descripcion": None
    }
    
    # ========================================================================
    # Validar PRESUPUESTO
    # ========================================================================
    if "presupuesto" in result:
        logger.info("-" * 80)
        logger.info("Validando PRESUPUESTO (costos extremos)")
        logger.info("-" * 80)
        
        try:
            presupuesto_original = result["presupuesto"]
            result["presupuesto"], metrics_presupuesto = _validate_extreme_costs(
                presupuesto_original
            )
            metricas_totales["presupuesto"] = metrics_presupuesto.to_dict()
            logger.info(
                f"Validación de presupuesto exitosa. Métricas: {metrics_presupuesto.to_dict()}"
            )
        except Exception as e:
            logger.error(
                f"Error crítico al validar presupuesto: {str(e)}",
                exc_info=True
            )
            # Mantener datos originales en caso de error, no asignar lista vacía
            logger.warning("Manteniendo datos originales de presupuesto debido al error.")
            metricas_totales["presupuesto"] = {
                "error": str(e),
                "items_procesados": 0
            }
    else:
        logger.warning(
            "Clave 'presupuesto' no encontrada en data_store. "
            "Saltando validación de presupuesto."
        )
    
    # ========================================================================
    # Validar APUS_DETAIL
    # ========================================================================
    if "apus_detail" in result:
        logger.info("-" * 80)
        logger.info("Validando APUS_DETAIL (cantidades y coherencia)")
        logger.info("-" * 80)
        
        try:
            apus_original = result["apus_detail"]
            result["apus_detail"], metrics_cantidad = _validate_zero_quantity_with_cost(
                apus_original
            )
            metricas_totales["apus_detail_cantidad"] = metrics_cantidad.to_dict()
            logger.info(
                f"Validación de cantidades exitosa. Métricas: {metrics_cantidad.to_dict()}"
            )
        except Exception as e:
            logger.error(
                f"Error crítico al validar cantidades en apus_detail: {str(e)}",
                exc_info=True
            )
            logger.warning("Manteniendo datos originales de apus_detail debido al error.")
            metricas_totales["apus_detail_cantidad"] = {
                "error": str(e),
                "items_procesados": 0
            }
        
        # Validar descripciones
        logger.info("-" * 80)
        logger.info("Validando APUS_DETAIL (descripciones)")
        logger.info("-" * 80)
        
        try:
            raw_insumos_df = result.get("raw_insumos_df")
            result["apus_detail"], metrics_descripcion = _validate_missing_descriptions(
                result["apus_detail"],
                raw_insumos_df
            )
            metricas_totales["apus_detail_descripcion"] = metrics_descripcion.to_dict()
            logger.info(
                f"Validación de descripciones exitosa. Métricas: {metrics_descripcion.to_dict()}"
            )
        except Exception as e:
            logger.error(
                f"Error crítico al validar descripciones en apus_detail: {str(e)}",
                exc_info=True
            )
            logger.warning("Continuando con datos parcialmente validados.")
            metricas_totales["apus_detail_descripcion"] = {
                "error": str(e),
                "items_procesados": 0
            }
    else:
        logger.warning(
            "Clave 'apus_detail' no encontrada en data_store. "
            "Saltando validación de APU."
        )
    
    # ========================================================================
    # Generar resumen de validación
    # ========================================================================
    total_items = sum(
        m.get("total_items_procesados", 0)
        for m in metricas_totales.values()
        if m and isinstance(m, dict)
    )
    
    total_alertas = sum(
        m.get("items_con_alertas", 0)
        for m in metricas_totales.values()
        if m and isinstance(m, dict)
    )
    
    total_errores = sum(
        m.get("items_con_errores", 0)
        for m in metricas_totales.values()
        if m and isinstance(m, dict)
    )
    
    tiene_errores = any(
        "error" in m
        for m in metricas_totales.values()
        if m and isinstance(m, dict)
    )
    
    resumen = {
        "exito": not tiene_errores,
        "total_items_procesados": total_items,
        "total_items_con_alertas": total_alertas,
        "total_items_con_errores": total_errores,
        "porcentaje_items_con_alertas": (
            round((total_alertas / total_items * 100), 2) if total_items > 0 else 0
        ),
        "validaciones_realizadas": [
            k for k, v in metricas_totales.items() if v is not None
        ],
        "mensaje": (
            "Validación completada exitosamente"
            if not tiene_errores
            else "Validación completada con errores"
        )
    }
    
    result["validation_summary"] = resumen
    result["validation_metrics"] = metricas_totales
    
    logger.info("="*80)
    logger.info("Agente de Validación de Datos completado")
    logger.info(f"Resumen: {resumen}")
    logger.info("="*80)
    
    return result