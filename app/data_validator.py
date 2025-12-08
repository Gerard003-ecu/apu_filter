# app/data_validator.py
import logging
import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Telemetry integration
try:
    from .telemetry import TelemetryContext
except ImportError:
    TelemetryContext = Any  # Fallback for typing if circular import

# Optional fuzzy matching
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
MAX_ALERTAS_POR_ITEM = 50  # Límite razonable de alertas por item
MAX_IDENTIFICADOR_LENGTH = 100
MAX_FUZZY_ATTEMPTS_PER_BATCH = 1000

# Constante extendida de valores a tratar como inválidos en descripciones
VALORES_DESCRIPCION_INVALIDOS = frozenset([
    "nan", "none", "null", "", "n/a", "na", "n.a.", "n.a",
    "-", "--", "---", ".", "..", "...",
    "undefined", "sin descripcion", "sin descripción",
    "no aplica", "no disponible", "pendiente",
])

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
            "alertas_por_tipo": self.alertas_por_tipo,
        }

@dataclass
class CampoFaltanteInfo:
    """Información detallada sobre un campo faltante o inválido"""
    nombre: str
    motivo: str  # "faltante", "none", "vacio", "nan"
    valor_actual: Any = None


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
    # Verificación rápida de None primero (operación O(1))
    if valor is None:
        return False

    # Manejar tipos numéricos nativos y numpy
    if isinstance(valor, (int, float, np.integer, np.floating)):
        # Verificar NaN e infinito (orden optimizado: NaN es más común)
        try:
            if pd.isna(valor) or np.isinf(valor):
                return False
            return True
        except (TypeError, ValueError):
            # Algunos tipos numpy pueden fallar en estas verificaciones
            return False

    # Manejar numpy arrays escalares
    if isinstance(valor, np.ndarray):
        if valor.ndim == 0:  # Escalar numpy
            try:
                return _es_numero_valido(valor.item())
            except (ValueError, IndexError):
                return False
        return False  # Arrays no escalares no son válidos

    # Rechazar cualquier otro tipo (strings, listas, etc.)
    return False


def _agregar_alerta(
    item: Dict[str, Any],
    mensaje: str,
    tipo_alerta: TipoAlerta,
    metrics: Optional[ValidationMetrics] = None,
    permitir_duplicados: bool = False,
) -> bool:
    """
    Agrega una alerta a un item y actualiza métricas.

    Args:
        item: Diccionario del item
        mensaje: Mensaje de alerta
        tipo_alerta: Tipo de alerta (enum)
        metrics: Objeto de métricas opcional
        permitir_duplicados: Si False, no agrega alertas con mismo tipo y mensaje

    Returns:
        bool: True si la alerta fue agregada, False si fue ignorada (duplicado o límite)
    """
    if not isinstance(item, dict):
        logger.warning(f"Intento de agregar alerta a un item que no es diccionario: {type(item)}")
        return False

    if "alertas" not in item:
        item["alertas"] = []

    # Verificar límite de alertas
    if len(item["alertas"]) >= MAX_ALERTAS_POR_ITEM:
        logger.warning(
            f"Item ha alcanzado el límite máximo de alertas ({MAX_ALERTAS_POR_ITEM}). "
            f"Alerta ignorada: {tipo_alerta.value} - {mensaje[:50]}..."
        )
        return False

    alerta_completa = {
        "tipo": tipo_alerta.value,
        "mensaje": mensaje,
    }

    # Verificar duplicados si no se permiten
    if not permitir_duplicados:
        for alerta_existente in item["alertas"]:
            if (
                alerta_existente.get("tipo") == alerta_completa["tipo"]
                and alerta_existente.get("mensaje") == alerta_completa["mensaje"]
            ):
                # logger.debug(f"Alerta duplicada ignorada: {tipo_alerta.value}")
                return False

    item["alertas"].append(alerta_completa)

    if metrics is not None:
        metrics.agregar_alerta(tipo_alerta)

    return True


def _validar_campos_requeridos(
    item: Dict[str, Any],
    campos_requeridos: List[str],
    nombre_item: str = "desconocido",
    considerar_none_como_faltante: bool = True,
    considerar_vacio_como_faltante: bool = True,
) -> List[CampoFaltanteInfo]:
    """
    Valida que un item contenga todos los campos requeridos con valores válidos.

    Args:
        item: Diccionario a validar
        campos_requeridos: Lista de nombres de campos requeridos
        nombre_item: Nombre del item para logging
        considerar_none_como_faltante: Si True, None se considera como faltante
        considerar_vacio_como_faltante: Si True, strings vacíos se consideran faltantes

    Returns:
        List[CampoFaltanteInfo]: Lista de información sobre campos faltantes/inválidos
    """
    if not isinstance(item, dict):
        logger.error(f"Item '{nombre_item}' no es un diccionario: {type(item)}")
        return [
            CampoFaltanteInfo(campo, "item_invalido", None)
            for campo in campos_requeridos
        ]

    campos_problematicos = []

    for campo in campos_requeridos:
        if campo not in item:
            campos_problematicos.append(CampoFaltanteInfo(campo, "faltante", None))
            logger.warning(f"Campo requerido '{campo}' faltante en item '{nombre_item}'")
            continue

        valor = item[campo]

        # Verificar None
        if valor is None and considerar_none_como_faltante:
            campos_problematicos.append(CampoFaltanteInfo(campo, "none", valor))
            logger.warning(f"Campo '{campo}' es None en item '{nombre_item}'")
            continue

        # Verificar NaN para valores numéricos
        if isinstance(valor, (float, np.floating)) and pd.isna(valor):
            campos_problematicos.append(CampoFaltanteInfo(campo, "nan", valor))
            logger.warning(f"Campo '{campo}' es NaN en item '{nombre_item}'")
            continue

        # Verificar strings vacíos
        if considerar_vacio_como_faltante and isinstance(valor, str):
            if not valor.strip():
                campos_problematicos.append(CampoFaltanteInfo(campo, "vacio", valor))
                logger.warning(f"Campo '{campo}' está vacío en item '{nombre_item}'")

    return campos_problematicos


def _validar_coherencia_matematica(
    cantidad: float,
    precio_unitario: float,
    valor_total: float,
    tolerancia_porcentual: float = TOLERANCIA_PORCENTUAL_COHERENCIA,
) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    Valida que la relación cantidad * precio_unitario ≈ valor_total sea coherente.

    Args:
        cantidad: Cantidad del insumo
        precio_unitario: Precio unitario
        valor_total: Valor total reportado
        tolerancia_porcentual: Tolerancia porcentual permitida (0.01 = 1%)

    Returns:
        Tuple[bool, Optional[float], Optional[str]]:
            (es_coherente, diferencia_porcentual, mensaje_detalle)
    """
    # Validar que todos los valores sean números válidos
    valores = {"cantidad": cantidad, "precio_unitario": precio_unitario, "valor_total": valor_total}
    for nombre, valor in valores.items():
        if not _es_numero_valido(valor):
            return True, None, f"Valor inválido en {nombre}: {valor}"

    # Convertir a float para cálculos consistentes
    cantidad = float(cantidad)
    precio_unitario = float(precio_unitario)
    valor_total = float(valor_total)

    # Caso especial: todos son cero o muy cercanos a cero
    todos_cercanos_a_cero = all(
        abs(v) < TOLERANCIA_COMPARACION_FLOAT
        for v in [cantidad, precio_unitario, valor_total]
    )
    if todos_cercanos_a_cero:
        return True, 0.0, "Todos los valores son cero o cercanos a cero"

    # Calcular valor esperado con detección de overflow
    try:
        valor_esperado = cantidad * precio_unitario
        if np.isinf(valor_esperado):
            return False, None, f"Overflow en cálculo: {cantidad} × {precio_unitario}"
    except (OverflowError, FloatingPointError):
        return False, None, f"Error de overflow: {cantidad} × {precio_unitario}"

    # Caso: valor_esperado ≈ 0 pero valor_total no
    if abs(valor_esperado) < TOLERANCIA_COMPARACION_FLOAT:
        if abs(valor_total) >= TOLERANCIA_COMPARACION_FLOAT:
            return False, None, (
                f"Valor esperado es ~0 (cantidad={cantidad}, precio={precio_unitario}) "
                f"pero valor_total={valor_total}"
            )
        return True, 0.0, "Valor esperado y valor total son ambos ~0"

    # Caso: valor_total ≈ 0 pero valor_esperado no
    if abs(valor_total) < TOLERANCIA_COMPARACION_FLOAT:
        if abs(valor_esperado) >= TOLERANCIA_COMPARACION_FLOAT:
            diferencia_pct = 100.0  # 100% de diferencia
            return False, diferencia_pct, (
                f"Valor total es ~0 pero valor esperado es {valor_esperado:.2f}"
            )

    # Calcular diferencia porcentual usando el valor más grande como referencia
    referencia = max(abs(valor_esperado), abs(valor_total))
    diferencia = abs(valor_esperado - valor_total)
    diferencia_porcentual = (diferencia / referencia) * 100

    # Comparar con tolerancia (tolerancia_porcentual ya es fracción, ej: 0.01 = 1%)
    umbral_porcentaje = tolerancia_porcentual * 100
    es_coherente = diferencia_porcentual <= umbral_porcentaje

    mensaje = (
        f"Esperado: {valor_esperado:.2f}, Reportado: {valor_total:.2f}, "
        f"Diferencia: {diferencia_porcentual:.2f}% (umbral: {umbral_porcentaje:.2f}%)"
    )

    return es_coherente, diferencia_porcentual, mensaje


def _limpiar_y_validar_descripcion(
    descripcion: Any,
    max_length: int = MAX_DESCRIPCION_LENGTH,
    normalizar_unicode: bool = True,
) -> Tuple[Optional[str], bool, Optional[str]]:
    """
    Limpia y valida una descripción.

    Args:
        descripcion: Descripción a validar
        max_length: Longitud máxima permitida
        normalizar_unicode: Si True, normaliza caracteres unicode

    Returns:
        Tuple[Optional[str], bool, Optional[str]]:
            (descripcion_limpia, es_valida, motivo_si_invalida)
    """
    # Verificar si es None o NaN
    if descripcion is None:
        return None, False, "Valor es None"

    if isinstance(descripcion, float) and pd.isna(descripcion):
        return None, False, "Valor es NaN"

    # Intentar convertir a string de forma segura
    try:
        descripcion_str = str(descripcion)
    except (ValueError, TypeError) as e:
        return None, False, f"No se puede convertir a string: {e}"

    # Normalizar unicode si está habilitado
    if normalizar_unicode:
        try:
            # Normalizar a forma NFC (composición canónica)
            descripcion_str = unicodedata.normalize("NFC", descripcion_str)
        except (TypeError, ValueError):
            pass  # Continuar sin normalización si falla

    # Eliminar caracteres de control (excepto espacios, tabs, newlines)
    descripcion_str = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', descripcion_str)

    # Normalizar espacios: múltiples espacios/tabs/newlines a un solo espacio
    descripcion_str = re.sub(r'\s+', ' ', descripcion_str)

    # Strip de espacios al inicio y final
    descripcion_str = descripcion_str.strip()

    # Verificar si está vacía después de limpieza
    if not descripcion_str:
        return None, False, "Descripción vacía después de limpieza"

    # Verificar contra lista de valores inválidos (case-insensitive)
    if descripcion_str.lower() in VALORES_DESCRIPCION_INVALIDOS:
        return None, False, f"Valor considerado inválido: '{descripcion_str}'"

    # Validar longitud mínima (al menos 2 caracteres significativos)
    if len(descripcion_str) < 2:
        return None, False, f"Descripción muy corta: '{descripcion_str}'"

    # Validar y truncar longitud máxima
    fue_truncada = False
    if len(descripcion_str) > max_length:
        logger.warning(
            f"Descripción excesivamente larga ({len(descripcion_str)} caracteres). "
            f"Truncando a {max_length}."
        )
        # Truncar en un límite de palabra si es posible
        descripcion_truncada = descripcion_str[:max_length]
        ultimo_espacio = descripcion_truncada.rfind(' ')
        if ultimo_espacio > max_length * 0.8:  # Solo si no perdemos mucho
            descripcion_str = descripcion_truncada[:ultimo_espacio].rstrip() + "..."
        else:
            descripcion_str = descripcion_truncada.rstrip() + "..."
        fue_truncada = True

    motivo = "Truncada por longitud" if fue_truncada else None
    return descripcion_str, True, motivo


def _obtener_identificador_item(
    item: Dict[str, Any],
    campos_prioritarios: Optional[List[str]] = None,
    max_length: int = MAX_IDENTIFICADOR_LENGTH,
) -> str:
    """
    Obtiene un identificador legible para un item para logging.

    Args:
        item: Diccionario del item
        campos_prioritarios: Lista de campos a buscar en orden de prioridad
        max_length: Longitud máxima del identificador

    Returns:
        str: Identificador del item (nunca vacío)
    """
    if not isinstance(item, dict):
        return f"<no-dict:{type(item).__name__}>"

    if campos_prioritarios is None:
        campos_prioritarios = ["ITEM", "ID", "CODIGO", "DESCRIPCION_INSUMO", "DESCRIPCION", "NOMBRE"]

    for campo in campos_prioritarios:
        if campo not in item:
            continue

        valor = item[campo]

        # Manejar None explícitamente
        if valor is None:
            continue

        # Manejar NaN
        if isinstance(valor, float) and pd.isna(valor):
            continue

        # Convertir a string
        try:
            valor_str = str(valor).strip()
        except (ValueError, TypeError):
            continue

        # Verificar que no esté vacío después de strip
        if not valor_str or valor_str.lower() in ("nan", "none", "null"):
            continue

        # Truncar si es necesario
        if len(valor_str) > max_length:
            valor_str = valor_str[: max_length - 3] + "..."

        return valor_str

    # Fallback: intentar construir identificador con hash parcial del contenido
    try:
        contenido_repr = repr(dict(list(item.items())[:3]))  # Primeros 3 items
        hash_parcial = hash(contenido_repr) % 10000
        return f"item_hash_{hash_parcial:04d}"
    except Exception:
        return "item_desconocido"


# ============================================================================
# FUNCIONES DE VALIDACIÓN PRINCIPALES
# ============================================================================
def _validate_extreme_costs(
    presupuesto_data: List[Dict[str, Any]],
    campo_costo: str = "VALOR_CONSTRUCCION_UN",
    costo_maximo: float = COSTO_MAXIMO_RAZONABLE,
    costo_minimo: float = COSTO_MINIMO_VALIDO,
) -> Tuple[List[Dict[str, Any]], ValidationMetrics]:
    """
    Valida costos unitarios de construcción en el presupuesto.
    Detecta costos excesivos, negativos e infinitos.

    Args:
        presupuesto_data: Lista de diccionarios con datos de presupuesto.
        campo_costo: Nombre del campo que contiene el costo a validar.
        costo_maximo: Umbral máximo razonable para costos.
        costo_minimo: Umbral mínimo válido para costos.

    Returns:
        Tuple[List[Dict], ValidationMetrics]: (datos_validados, métricas)
    """
    metrics = ValidationMetrics()

    # Validación de entrada
    if not isinstance(presupuesto_data, list):
        logger.error(
            f"Entrada a _validate_extreme_costs no es una lista (tipo: {type(presupuesto_data)}). "
            "Retornando sin cambios."
        )
        return presupuesto_data if presupuesto_data is not None else [], metrics

    if not presupuesto_data:
        logger.info("Lista de presupuesto vacía. No hay nada que validar.")
        return [], metrics

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
        tiene_error_en_item = False

        # Verificar si el campo existe
        if campo_costo not in item:
            logger.warning(f"Campo '{campo_costo}' no existe en item '{item_id}'")
            _agregar_alerta(
                item,
                f"Campo de costo '{campo_costo}' no existe en el item",
                TipoAlerta.CAMPO_REQUERIDO_FALTANTE,
                metrics,
            )
            metrics.items_con_errores += 1
            continue

        valor = item[campo_costo]

        # Intentar conversión si es string numérico
        if isinstance(valor, str):
            valor_limpio = valor.strip().replace(",", "").replace("$", "")
            try:
                valor = float(valor_limpio)
                item[campo_costo] = valor  # Actualizar con valor convertido
                logger.debug(f"Valor convertido de string a float en item '{item_id}': {valor}")
            except (ValueError, TypeError):
                logger.warning(
                    f"Valor de costo no convertible en item '{item_id}': '{valor}'"
                )
                _agregar_alerta(
                    item,
                    f"Valor de costo no es numérico y no se puede convertir: {valor}",
                    TipoAlerta.CAMPO_REQUERIDO_FALTANTE,
                    metrics,
                )
                metrics.items_con_errores += 1
                continue

        # Validar que el valor sea un número válido
        if not isinstance(valor, (int, float, np.integer, np.floating)):
            logger.warning(
                f"Valor de costo no es numérico en item '{item_id}': "
                f"{valor} (tipo: {type(valor).__name__})"
            )
            _agregar_alerta(
                item,
                f"Valor de costo no es numérico: {valor}",
                TipoAlerta.CAMPO_REQUERIDO_FALTANTE,
                metrics,
            )
            metrics.items_con_errores += 1
            continue

        # Convertir a float nativo para operaciones consistentes
        try:
            valor = float(valor)
        except (ValueError, TypeError, OverflowError) as e:
            logger.error(f"Error al convertir valor a float en item '{item_id}': {e}")
            metrics.items_con_errores += 1
            continue

        # Validar NaN
        if pd.isna(valor):
            logger.warning(f"Valor de costo es NaN en item '{item_id}'")
            _agregar_alerta(
                item,
                "Valor de costo es NaN",
                TipoAlerta.CAMPO_REQUERIDO_FALTANTE,
                metrics,
            )
            tiene_error_en_item = True
            # No continue: puede haber más validaciones útiles

        # Validar infinito
        elif np.isinf(valor):
            logger.error(f"Valor de costo es infinito en item '{item_id}': {valor}")
            _agregar_alerta(
                item,
                f"Valor de costo es infinito: {valor}",
                TipoAlerta.VALOR_INFINITO,
                metrics,
            )
            metrics.valores_infinitos += 1
            tiene_error_en_item = True

        else:
            # Validar valores negativos
            if valor < costo_minimo:
                alerta_msg = (
                    f"Costo unitario negativo detectado: {valor:,.2f}. "
                    "Los costos no deben ser negativos."
                )
                _agregar_alerta(item, alerta_msg, TipoAlerta.COSTO_NEGATIVO, metrics)
                logger.error(f"Costo negativo en item '{item_id}': {valor:,.2f}")
                metrics.costos_negativos += 1

            # Validar valores excesivos (independiente del check de negativo)
            if valor > costo_maximo:
                alerta_msg = (
                    f"Costo unitario ({valor:,.2f}) excede el umbral razonable de "
                    f"{costo_maximo:,.2f}. Verificar si es correcto."
                )
                _agregar_alerta(item, alerta_msg, TipoAlerta.COSTO_EXCESIVO, metrics)
                logger.warning(f"Costo excesivo en item '{item_id}': {valor:,.2f}")
                metrics.costos_excesivos += 1

        if tiene_error_en_item:
            metrics.items_con_errores += 1

    # Actualizar contador de items con alertas
    metrics.items_con_alertas = sum(
        1 for item in result
        if isinstance(item, dict) and item.get("alertas")
    )

    logger.info(
        f"Validación de costos extremos completada: "
        f"{metrics.total_items_procesados} items procesados, "
        f"{metrics.items_con_alertas} con alertas, "
        f"{metrics.costos_excesivos} costos excesivos, "
        f"{metrics.costos_negativos} costos negativos, "
        f"{metrics.items_con_errores} items con errores."
    )

    return result, metrics


def _validate_zero_quantity_with_cost(
    apus_detail_data: List[Dict[str, Any]],
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
            f"Entrada no es una lista (tipo: {type(apus_detail_data)}). "
            "Retornando sin cambios."
        )
        return apus_detail_data if apus_detail_data is not None else [], metrics

    if not apus_detail_data:
        logger.info("Lista de APUs detalle vacía. No hay nada que validar.")
        return [], metrics

    result = deepcopy(apus_detail_data)
    metrics.total_items_procesados = len(result)

    def _convertir_a_numero(valor: Any, nombre_campo: str) -> Tuple[Optional[float], bool]:
        """Intenta convertir un valor a número float."""
        if valor is None:
            return None, False

        if isinstance(valor, (int, float, np.integer, np.floating)):
            try:
                valor_float = float(valor)
                if pd.isna(valor_float) or np.isinf(valor_float):
                    return None, False
                return valor_float, True
            except (ValueError, TypeError, OverflowError):
                return None, False

        if isinstance(valor, str):
            try:
                valor_limpio = valor.strip().replace(",", "").replace("$", "")
                return float(valor_limpio), True
            except (ValueError, TypeError):
                return None, False

        return None, False

    for idx, insumo in enumerate(result):
        if not isinstance(insumo, dict):
            logger.warning(
                f"Item en apus_detail[{idx}] no es un diccionario (tipo: {type(insumo)}). Saltando."
            )
            metrics.items_con_errores += 1
            continue

        item_id = _obtener_identificador_item(insumo)
        alertas_agregadas = 0

        # Extraer y convertir valores
        cantidad_raw = insumo.get("CANTIDAD")
        valor_total_raw = insumo.get("VALOR_TOTAL", 0)
        precio_unitario_raw = insumo.get("VR_UNITARIO", 0)

        cantidad, cantidad_valida = _convertir_a_numero(cantidad_raw, "CANTIDAD")
        valor_total, valor_total_valido = _convertir_a_numero(valor_total_raw, "VALOR_TOTAL")
        precio_unitario, precio_unitario_valido = _convertir_a_numero(precio_unitario_raw, "VR_UNITARIO")

        # Registrar campos con problemas de conversión
        campos_con_problemas = []
        if not cantidad_valida:
            campos_con_problemas.append(("CANTIDAD", cantidad_raw))
        if not valor_total_valido:
            campos_con_problemas.append(("VALOR_TOTAL", valor_total_raw))
        if not precio_unitario_valido:
            campos_con_problemas.append(("VR_UNITARIO", precio_unitario_raw))

        if campos_con_problemas:
            for campo, valor in campos_con_problemas:
                _agregar_alerta(
                    insumo,
                    f"Campo '{campo}' tiene un valor no numérico o inválido: {valor}",
                    TipoAlerta.CAMPO_REQUERIDO_FALTANTE,
                    metrics,
                )
            logger.warning(
                f"Campos numéricos inválidos en insumo '{item_id}': "
                f"{[c[0] for c in campos_con_problemas]}"
            )
            metrics.items_con_errores += 1
            # Continuar con valores por defecto donde sea posible
            cantidad = cantidad if cantidad is not None else 0.0
            valor_total = valor_total if valor_total is not None else 0.0
            precio_unitario = precio_unitario if precio_unitario is not None else 0.0

        # A partir de aquí, trabajamos con valores float (pueden ser 0 por defecto)

        # === Validaciones de signo ===

        # Validar cantidad negativa
        if cantidad < CANTIDAD_MINIMA_VALIDA:
            alerta_msg = (
                f"Cantidad negativa detectada: {cantidad}. "
                "Las cantidades no deben ser negativas."
            )
            _agregar_alerta(insumo, alerta_msg, TipoAlerta.CANTIDAD_INVALIDA, metrics)
            logger.error(f"Cantidad negativa en insumo '{item_id}': {cantidad}")
            alertas_agregadas += 1

        # Validar valor_total negativo
        if valor_total < 0:
            alerta_msg = f"Valor total negativo detectado: {valor_total:,.2f}"
            _agregar_alerta(insumo, alerta_msg, TipoAlerta.COSTO_NEGATIVO, metrics)
            logger.error(f"Valor total negativo en insumo '{item_id}': {valor_total}")
            metrics.costos_negativos += 1
            alertas_agregadas += 1

        # Validar precio_unitario negativo
        if precio_unitario < 0:
            alerta_msg = f"Precio unitario negativo detectado: {precio_unitario:,.2f}"
            _agregar_alerta(insumo, alerta_msg, TipoAlerta.COSTO_NEGATIVO, metrics)
            logger.error(f"Precio unitario negativo en insumo '{item_id}': {precio_unitario}")
            metrics.costos_negativos += 1
            alertas_agregadas += 1

        # === Recálculo de cantidad si es necesario ===

        cantidad_fue_recalculada = False

        # Caso: cantidad == 0 pero valor_total > 0
        if (
            abs(cantidad) < TOLERANCIA_COMPARACION_FLOAT
            and valor_total > TOLERANCIA_COMPARACION_FLOAT
        ):
            if precio_unitario > TOLERANCIA_COMPARACION_FLOAT:
                nueva_cantidad = valor_total / precio_unitario

                # Verificar que el resultado sea razonable (no infinito ni demasiado grande)
                if np.isfinite(nueva_cantidad) and nueva_cantidad < 1e12:
                    cantidad_anterior = cantidad
                    cantidad = nueva_cantidad
                    insumo["CANTIDAD"] = nueva_cantidad
                    cantidad_fue_recalculada = True

                    alerta_msg = (
                        f"Cantidad recalculada de {cantidad_anterior} a {nueva_cantidad:.6f} "
                        f"(valor total: {valor_total:,.2f}, precio unitario: {precio_unitario:,.2f})"
                    )
                    _agregar_alerta(insumo, alerta_msg, TipoAlerta.CANTIDAD_RECALCULADA, metrics)
                    logger.info(f"Insumo '{item_id}': {alerta_msg}")
                    metrics.cantidades_recalculadas += 1
                else:
                    alerta_msg = (
                        f"Cantidad calculada es inválida o muy grande: {nueva_cantidad}. "
                        f"No se modificó el valor original."
                    )
                    _agregar_alerta(insumo, alerta_msg, TipoAlerta.CANTIDAD_INVALIDA, metrics)
                    logger.warning(f"Insumo '{item_id}': {alerta_msg}")
            else:
                alerta_msg = (
                    f"Cantidad es 0 con valor total {valor_total:,.2f}, pero precio unitario "
                    f"es {precio_unitario}. No se puede recalcular la cantidad."
                )
                _agregar_alerta(insumo, alerta_msg, TipoAlerta.CANTIDAD_INVALIDA, metrics)
                logger.warning(f"Insumo '{item_id}': {alerta_msg}")

        # === Validar coherencia matemática ===
        # Solo si no hubo errores críticos y los valores son válidos para comparar

        if cantidad_valida or cantidad_fue_recalculada:
            es_coherente, diferencia_pct, mensaje_detalle = _validar_coherencia_matematica(
                cantidad, precio_unitario, valor_total
            )

            if not es_coherente:
                alerta_msg = (
                    f"Incoherencia matemática detectada: "
                    f"cantidad ({cantidad:.6f}) × precio unitario ({precio_unitario:,.2f}) = "
                    f"{cantidad * precio_unitario:,.2f}, pero valor total reportado es {valor_total:,.2f}."
                )
                if diferencia_pct is not None:
                    alerta_msg += f" Diferencia: {diferencia_pct:.2f}%"
                if mensaje_detalle:
                    alerta_msg += f" ({mensaje_detalle})"

                _agregar_alerta(insumo, alerta_msg, TipoAlerta.INCOHERENCIA_MATEMATICA, metrics)
                logger.warning(f"Insumo '{item_id}': {alerta_msg}")
                metrics.incoherencias_matematicas += 1

    # Actualizar contador de items con alertas
    metrics.items_con_alertas = sum(
        1 for item in result
        if isinstance(item, dict) and item.get("alertas")
    )

    logger.info(
        f"Validación de cantidades completada: "
        f"{metrics.total_items_procesados} items procesados, "
        f"{metrics.items_con_alertas} con alertas, "
        f"{metrics.cantidades_recalculadas} cantidades recalculadas, "
        f"{metrics.incoherencias_matematicas} incoherencias matemáticas, "
        f"{metrics.items_con_errores} items con errores."
    )

    return result, metrics


def _validate_missing_descriptions(
    apus_detail_data: List[Dict[str, Any]],
    raw_insumos_df: Optional[pd.DataFrame],
    fuzzy_threshold: int = FUZZY_MATCH_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], ValidationMetrics]:
    """
    Valida y corrige descripciones faltantes o inválidas en insumos.
    Usa fuzzy matching cuando está disponible para encontrar descripciones similares.

    Args:
        apus_detail_data: Lista de diccionarios con clave 'DESCRIPCION_INSUMO'.
        raw_insumos_df: DataFrame opcional con columna 'DESCRIPCION_INSUMO'.
        fuzzy_threshold: Umbral mínimo de similitud para aceptar coincidencia.

    Returns:
        Tuple[List[Dict], ValidationMetrics]: (datos_validados, métricas)
    """
    metrics = ValidationMetrics()

    if not isinstance(apus_detail_data, list):
        logger.error(
            f"Entrada no es una lista (tipo: {type(apus_detail_data)}). "
            "Retornando sin cambios."
        )
        return apus_detail_data if apus_detail_data is not None else [], metrics

    if not apus_detail_data:
        logger.info("Lista de APUs detalle vacía. No hay descripciones que validar.")
        return [], metrics

    result = deepcopy(apus_detail_data)
    metrics.total_items_procesados = len(result)

    # Preparar lista de descripciones válidas para fuzzy matching
    lista_descripciones: List[str] = []
    mapa_descripciones: Dict[str, str] = {}  # lowercase -> original

    if isinstance(raw_insumos_df, pd.DataFrame) and not raw_insumos_df.empty:
        if "DESCRIPCION_INSUMO" in raw_insumos_df.columns:
            try:
                descripciones_series = (
                    raw_insumos_df["DESCRIPCION_INSUMO"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )
                descripciones_series = descripciones_series[
                    (descripciones_series != "") &
                    (descripciones_series.str.len() >= 2)
                ]

                for desc in descripciones_series.unique():
                    lista_descripciones.append(desc)
                    mapa_descripciones[desc.lower()] = desc

                logger.info(
                    f"Se cargaron {len(lista_descripciones)} descripciones únicas "
                    f"para fuzzy matching."
                )
            except Exception as e:
                logger.error(
                    f"Error al extraer descripciones de raw_insumos_df: {str(e)}",
                    exc_info=True,
                )
        else:
            logger.warning("Columna 'DESCRIPCION_INSUMO' no existe en raw_insumos_df.")
    else:
        logger.warning("raw_insumos_df no es un DataFrame válido o está vacío.")

    tiene_referencias = len(lista_descripciones) > 0
    puede_fuzzy_match = HAS_FUZZY and tiene_referencias

    if not puede_fuzzy_match:
        razon = "sin referencias" if HAS_FUZZY else "librería no instalada"
        logger.info(f"Fuzzy matching no disponible ({razon}).")

    # Cache para resultados de fuzzy matching
    cache_fuzzy: Dict[str, Optional[Tuple[str, int]]] = {}
    fuzzy_attempts = 0

    def _buscar_con_fuzzy(texto_busqueda: str) -> Optional[Tuple[str, int]]:
        """Busca coincidencia fuzzy con cache."""
        nonlocal fuzzy_attempts

        if not puede_fuzzy_match:
            return None

        if texto_busqueda in cache_fuzzy:
            return cache_fuzzy[texto_busqueda]

        if fuzzy_attempts >= MAX_FUZZY_ATTEMPTS_PER_BATCH:
            return None

        fuzzy_attempts += 1

        try:
            # Primero buscar coincidencia exacta (case-insensitive)
            texto_lower = texto_busqueda.lower()
            if texto_lower in mapa_descripciones:
                resultado = (mapa_descripciones[texto_lower], 100)
                cache_fuzzy[texto_busqueda] = resultado
                return resultado

            # Fuzzy matching
            match = process.extractOne(
                texto_busqueda,
                lista_descripciones,
                score_cutoff=fuzzy_threshold,
            )

            if match:
                resultado = (match[0], match[1])
            else:
                resultado = None

            cache_fuzzy[texto_busqueda] = resultado
            return resultado

        except Exception as e:
            logger.debug(f"Error en fuzzy matching para '{texto_busqueda[:30]}...': {e}")
            cache_fuzzy[texto_busqueda] = None
            return None

    # Procesar cada insumo
    for idx, insumo in enumerate(result):
        if not isinstance(insumo, dict):
            logger.warning(
                f"Item en apus_detail[{idx}] no es un diccionario (tipo: {type(insumo)}). Saltando."
            )
            metrics.items_con_errores += 1
            continue

        descripcion_original = insumo.get("DESCRIPCION_INSUMO")
        descripcion_limpia, es_valida, motivo = _limpiar_y_validar_descripcion(descripcion_original)

        if es_valida and descripcion_limpia:
            # La descripción es válida
            if descripcion_limpia != descripcion_original:
                insumo["DESCRIPCION_INSUMO"] = descripcion_limpia
                # Solo loggear si hubo cambio significativo
                if motivo:
                    logger.debug(f"Descripción limpiada ({motivo}): '{descripcion_original}' -> '{descripcion_limpia}'")
            continue

        # Descripción faltante o inválida - intentar corrección
        item_id = insumo.get("ID", insumo.get("CODIGO", f"índice_{idx}"))
        descripcion_encontrada = False

        if puede_fuzzy_match:
            # Estrategia 1: Buscar usando la descripción original si tiene algo útil
            if descripcion_original and isinstance(descripcion_original, str):
                texto_busqueda = str(descripcion_original).strip()
                if len(texto_busqueda) >= 3:  # Mínimo 3 caracteres para búsqueda
                    resultado = _buscar_con_fuzzy(texto_busqueda)
                    if resultado:
                        desc_corregida, similitud = resultado
                        insumo["DESCRIPCION_INSUMO"] = desc_corregida
                        alerta_msg = (
                            f"Descripción corregida por fuzzy matching: "
                            f"'{descripcion_original}' -> '{desc_corregida}' "
                            f"(similitud: {similitud}%)"
                        )
                        _agregar_alerta(insumo, alerta_msg, TipoAlerta.DESCRIPCION_CORREGIDA, metrics)
                        logger.info(f"Insumo '{item_id}': {alerta_msg}")
                        metrics.descripciones_corregidas += 1
                        descripcion_encontrada = True

            # Estrategia 2: Buscar usando campos alternativos (solo si no encontró)
            if not descripcion_encontrada:
                campos_alternativos = ["CODIGO", "NOMBRE", "TIPO"]
                for campo in campos_alternativos:
                    valor_campo = insumo.get(campo)
                    if not valor_campo or not isinstance(valor_campo, str):
                        continue

                    texto_busqueda = valor_campo.strip()
                    if len(texto_busqueda) < 3:
                        continue

                    resultado = _buscar_con_fuzzy(texto_busqueda)
                    if resultado and resultado[1] >= fuzzy_threshold + 10:  # Umbral más alto para campos alternativos
                        desc_corregida, similitud = resultado
                        insumo["DESCRIPCION_INSUMO"] = desc_corregida
                        alerta_msg = (
                            f"Descripción inferida desde campo '{campo}': "
                            f"'{texto_busqueda}' -> '{desc_corregida}' "
                            f"(similitud: {similitud}%)"
                        )
                        _agregar_alerta(insumo, alerta_msg, TipoAlerta.DESCRIPCION_CORREGIDA, metrics)
                        logger.info(f"Insumo '{item_id}': {alerta_msg}")
                        metrics.descripciones_corregidas += 1
                        descripcion_encontrada = True
                        break

        # Si no se encontró descripción, usar fallback
        if not descripcion_encontrada:
            insumo["DESCRIPCION_INSUMO"] = DESCRIPCION_GENERICA_FALLBACK

            razon_detallada = motivo if motivo else "valor inválido o faltante"
            if puede_fuzzy_match:
                razon_detallada += "; fuzzy matching no encontró coincidencia"
            elif HAS_FUZZY:
                razon_detallada += "; sin datos de referencia para fuzzy matching"
            else:
                razon_detallada += "; fuzzy matching no instalado"

            alerta_msg = (
                f"Descripción faltante (original: {repr(descripcion_original)}). "
                f"Razón: {razon_detallada}. "
                f"Usando valor por defecto: '{DESCRIPCION_GENERICA_FALLBACK}'"
            )
            _agregar_alerta(insumo, alerta_msg, TipoAlerta.DESCRIPCION_FALTANTE, metrics)
            logger.warning(f"Insumo '{item_id}': Descripción faltante, usando fallback")

    # Actualizar contador de items con alertas
    metrics.items_con_alertas = sum(
        1 for item in result
        if isinstance(item, dict) and item.get("alertas")
    )

    logger.info(
        f"Validación de descripciones completada: "
        f"{metrics.total_items_procesados} items procesados, "
        f"{metrics.items_con_alertas} con alertas, "
        f"{metrics.descripciones_corregidas} descripciones corregidas. "
        f"Fuzzy matching: {fuzzy_attempts} búsquedas, {len(cache_fuzzy)} en cache."
    )

    return result, metrics


# ============================================================================
# FUNCIÓN ORQUESTADORA PRINCIPAL
# ============================================================================
def validate_and_clean_data(
    data_store: Dict[str, Any],
    skip_on_error: bool = True,
    validaciones_habilitadas: Optional[Dict[str, bool]] = None,
    telemetry_context: Optional[TelemetryContext] = None,
) -> Dict[str, Any]:
    """
    Orquesta las validaciones y correcciones sobre los datos procesados.
    Garantiza inmutabilidad: no modifica el diccionario original.

    Args:
        data_store: Diccionario con claves esperadas.
        skip_on_error: Si True, continúa con otras validaciones si una falla.
        validaciones_habilitadas: Dict para habilitar/deshabilitar validaciones específicas.
            Claves: 'presupuesto_costos', 'apus_cantidades', 'apus_descripciones'
        telemetry_context: Contexto de telemetría para registro centralizado.

    Returns:
        Dict: Nuevo diccionario con datos validados y metadatos de validación.
    """
    logger.info("=" * 80)
    logger.info("Iniciando Agente de Validación de Datos")
    logger.info("=" * 80)

    if telemetry_context:
        telemetry_context.start_step("validate_data")

    # Configuración por defecto de validaciones
    config_validaciones = {
        "presupuesto_costos": True,
        "apus_cantidades": True,
        "apus_descripciones": True,
    }
    if validaciones_habilitadas:
        config_validaciones.update(validaciones_habilitadas)

    # Validar entrada
    if data_store is None:
        logger.error("data_store es None. Retornando error.")
        error_result = {
            "error": "Entrada inválida: data_store es None",
            "validation_summary": {
                "exito": False,
                "mensaje": "Validación fallida: entrada None",
                "errores": ["data_store es None"],
            },
        }
        if telemetry_context:
            telemetry_context.record_error("validate_data", "data_store is None")
            telemetry_context.end_step("validate_data", "failure", metadata=error_result["validation_summary"])
        return error_result

    if not isinstance(data_store, dict):
        logger.error(f"data_store no es un diccionario (tipo: {type(data_store).__name__}).")
        error_result = {
            "error": f"Entrada inválida: data_store debe ser un diccionario, recibido {type(data_store).__name__}",
            "validation_summary": {
                "exito": False,
                "mensaje": "Validación fallida por entrada inválida",
                "errores": [f"Tipo incorrecto: {type(data_store).__name__}"],
            },
        }
        if telemetry_context:
            telemetry_context.record_error("validate_data", f"Invalid data_store type: {type(data_store)}")
            telemetry_context.end_step("validate_data", "failure", metadata=error_result["validation_summary"])
        return error_result

    if not data_store:
        logger.warning("data_store está vacío. No hay datos para validar.")
        result = {
            "validation_summary": {
                "exito": True,
                "mensaje": "No había datos para validar",
                "total_items_procesados": 0,
                "advertencias": ["data_store vacío"],
            }
        }
        if telemetry_context:
            telemetry_context.end_step("validate_data", "success", metadata=result["validation_summary"])
        return result

    # Crear copia profunda para evitar efectos secundarios
    try:
        result = deepcopy(data_store)
    except Exception as e:
        logger.error(f"Error al crear copia profunda de data_store: {str(e)}", exc_info=True)
        error_result = {
            "error": f"Error al copiar datos: {str(e)}",
            "validation_summary": {
                "exito": False,
                "mensaje": "Validación fallida por error en copia de datos",
                "errores": [str(e)],
            },
            # Retornar datos originales como fallback
            **{k: v for k, v in data_store.items() if k not in ("validation_summary", "validation_metrics")},
        }
        if telemetry_context:
            telemetry_context.record_error("validate_data", f"Deepcopy failed: {e}")
            telemetry_context.end_step("validate_data", "failure", metadata=error_result["validation_summary"])
        return error_result

    # Inicializar contenedor de métricas y errores
    metricas_totales: Dict[str, Optional[Dict[str, Any]]] = {}
    errores_validacion: List[str] = []
    advertencias_validacion: List[str] = []

    # ========================================================================
    # Validar PRESUPUESTO
    # ========================================================================
    if "presupuesto" in result and config_validaciones.get("presupuesto_costos", True):
        logger.info("-" * 80)
        logger.info("Validando PRESUPUESTO (costos extremos)")
        logger.info("-" * 80)

        if telemetry_context:
            telemetry_context.start_step("validate_presupuesto")

        presupuesto_data = result.get("presupuesto")

        # Validar que presupuesto tenga el tipo esperado
        if not isinstance(presupuesto_data, list):
            error_msg = f"'presupuesto' no es una lista (tipo: {type(presupuesto_data).__name__})"
            logger.error(error_msg)
            errores_validacion.append(error_msg)
            metricas_totales["presupuesto"] = {"error": error_msg, "items_procesados": 0}
            if telemetry_context:
                telemetry_context.record_error("validate_presupuesto", error_msg)
        else:
            try:
                result["presupuesto"], metrics_presupuesto = _validate_extreme_costs(
                    presupuesto_data
                )
                metricas_totales["presupuesto"] = metrics_presupuesto.to_dict()
                logger.info(f"Validación de presupuesto exitosa.")
                if telemetry_context:
                    telemetry_context.record_metric("validation", "presupuesto_items", metrics_presupuesto.total_items_procesados)
                    telemetry_context.record_metric("validation", "presupuesto_alerts", metrics_presupuesto.items_con_alertas)
            except Exception as e:
                error_msg = f"Error crítico al validar presupuesto: {str(e)}"
                logger.error(error_msg, exc_info=True)
                errores_validacion.append(error_msg)
                metricas_totales["presupuesto"] = {"error": str(e), "items_procesados": 0}
                if telemetry_context:
                    telemetry_context.record_error("validate_presupuesto", error_msg)

                if not skip_on_error:
                    if telemetry_context:
                        telemetry_context.end_step("validate_presupuesto", "failure")
                        telemetry_context.end_step("validate_data", "failure")
                    raise RuntimeError(error_msg) from e

                logger.warning("Continuando con datos originales de presupuesto.")

        if telemetry_context:
            status = "failure" if metricas_totales.get("presupuesto", {}).get("error") else "success"
            telemetry_context.end_step("validate_presupuesto", status)

    elif "presupuesto" not in result:
        advertencias_validacion.append("Clave 'presupuesto' no encontrada en data_store")
        logger.warning("Clave 'presupuesto' no encontrada. Saltando validación.")

    # ========================================================================
    # Validar APUS_DETAIL - Cantidades
    # ========================================================================
    if "apus_detail" in result and config_validaciones.get("apus_cantidades", True):
        logger.info("-" * 80)
        logger.info("Validando APUS_DETAIL (cantidades y coherencia)")
        logger.info("-" * 80)

        if telemetry_context:
            telemetry_context.start_step("validate_apus_quantities")

        apus_data = result.get("apus_detail")

        if not isinstance(apus_data, list):
            error_msg = f"'apus_detail' no es una lista (tipo: {type(apus_data).__name__})"
            logger.error(error_msg)
            errores_validacion.append(error_msg)
            metricas_totales["apus_detail_cantidad"] = {"error": error_msg, "items_procesados": 0}
            if telemetry_context:
                telemetry_context.record_error("validate_apus_quantities", error_msg)
        else:
            try:
                result["apus_detail"], metrics_cantidad = _validate_zero_quantity_with_cost(
                    apus_data
                )
                metricas_totales["apus_detail_cantidad"] = metrics_cantidad.to_dict()
                logger.info("Validación de cantidades exitosa.")
                if telemetry_context:
                    telemetry_context.record_metric("validation", "apus_items", metrics_cantidad.total_items_procesados)
                    telemetry_context.record_metric("validation", "apus_recalculated", metrics_cantidad.cantidades_recalculadas)
            except Exception as e:
                error_msg = f"Error crítico al validar cantidades: {str(e)}"
                logger.error(error_msg, exc_info=True)
                errores_validacion.append(error_msg)
                metricas_totales["apus_detail_cantidad"] = {"error": str(e), "items_procesados": 0}
                if telemetry_context:
                    telemetry_context.record_error("validate_apus_quantities", error_msg)

                if not skip_on_error:
                    if telemetry_context:
                        telemetry_context.end_step("validate_apus_quantities", "failure")
                        telemetry_context.end_step("validate_data", "failure")
                    raise RuntimeError(error_msg) from e

                logger.warning("Continuando con datos parcialmente validados.")

        if telemetry_context:
            status = "failure" if metricas_totales.get("apus_detail_cantidad", {}).get("error") else "success"
            telemetry_context.end_step("validate_apus_quantities", status)

    # ========================================================================
    # Validar APUS_DETAIL - Descripciones
    # ========================================================================
    if "apus_detail" in result and config_validaciones.get("apus_descripciones", True):
        logger.info("-" * 80)
        logger.info("Validando APUS_DETAIL (descripciones)")
        logger.info("-" * 80)

        if telemetry_context:
            telemetry_context.start_step("validate_apus_descriptions")

        apus_data = result.get("apus_detail")

        if not isinstance(apus_data, list):
            # Si ya se registró error arriba, no duplicar
            if "apus_detail_cantidad" not in metricas_totales or "error" not in metricas_totales.get("apus_detail_cantidad", {}):
                error_msg = f"'apus_detail' no es una lista (tipo: {type(apus_data).__name__})"
                logger.error(error_msg)
                errores_validacion.append(error_msg)
            metricas_totales["apus_detail_descripcion"] = {"error": "apus_detail inválido", "items_procesados": 0}
            if telemetry_context:
                telemetry_context.record_error("validate_apus_descriptions", "apus_detail is not a list")
        else:
            try:
                raw_insumos_df = result.get("raw_insumos_df")

                # Validar tipo de raw_insumos_df
                if raw_insumos_df is not None and not isinstance(raw_insumos_df, pd.DataFrame):
                    logger.warning(
                        f"raw_insumos_df no es un DataFrame (tipo: {type(raw_insumos_df).__name__}). "
                        "Fuzzy matching no estará disponible."
                    )
                    advertencias_validacion.append("raw_insumos_df tiene tipo incorrecto")
                    raw_insumos_df = None

                result["apus_detail"], metrics_descripcion = _validate_missing_descriptions(
                    apus_data, raw_insumos_df
                )
                metricas_totales["apus_detail_descripcion"] = metrics_descripcion.to_dict()
                logger.info("Validación de descripciones exitosa.")
                if telemetry_context:
                    telemetry_context.record_metric("validation", "descriptions_corrected", metrics_descripcion.descripciones_corregidas)
            except Exception as e:
                error_msg = f"Error crítico al validar descripciones: {str(e)}"
                logger.error(error_msg, exc_info=True)
                errores_validacion.append(error_msg)
                metricas_totales["apus_detail_descripcion"] = {"error": str(e), "items_procesados": 0}
                if telemetry_context:
                    telemetry_context.record_error("validate_apus_descriptions", error_msg)

                if not skip_on_error:
                    if telemetry_context:
                        telemetry_context.end_step("validate_apus_descriptions", "failure")
                        telemetry_context.end_step("validate_data", "failure")
                    raise RuntimeError(error_msg) from e

                logger.warning("Continuando con datos parcialmente validados.")

        if telemetry_context:
            status = "failure" if metricas_totales.get("apus_detail_descripcion", {}).get("error") else "success"
            telemetry_context.end_step("validate_apus_descriptions", status)

    elif "apus_detail" not in result:
        advertencias_validacion.append("Clave 'apus_detail' no encontrada en data_store")
        logger.warning("Clave 'apus_detail' no encontrada. Saltando validación de APU.")

    # ========================================================================
    # Generar resumen de validación
    # ========================================================================

    def _safe_get_metric(metricas: Dict, key: str, default: int = 0) -> int:
        """Obtiene una métrica de forma segura."""
        if not metricas or not isinstance(metricas, dict):
            return default
        if "error" in metricas:
            return default
        return metricas.get(key, default)

    total_items = sum(
        _safe_get_metric(m, "total_items_procesados")
        for m in metricas_totales.values()
    )

    total_alertas = sum(
        _safe_get_metric(m, "items_con_alertas")
        for m in metricas_totales.values()
    )

    total_errores_items = sum(
        _safe_get_metric(m, "items_con_errores")
        for m in metricas_totales.values()
    )

    tiene_errores_criticos = len(errores_validacion) > 0

    # Calcular porcentaje de forma segura
    porcentaje_alertas = 0.0
    if total_items > 0:
        porcentaje_alertas = round((total_alertas / total_items * 100), 2)

    validaciones_exitosas = [
        k for k, v in metricas_totales.items()
        if v and isinstance(v, dict) and "error" not in v
    ]

    validaciones_fallidas = [
        k for k, v in metricas_totales.items()
        if v and isinstance(v, dict) and "error" in v
    ]

    resumen = {
        "exito": not tiene_errores_criticos,
        "total_items_procesados": total_items,
        "total_items_con_alertas": total_alertas,
        "total_items_con_errores": total_errores_items,
        "porcentaje_items_con_alertas": porcentaje_alertas,
        "validaciones_exitosas": validaciones_exitosas,
        "validaciones_fallidas": validaciones_fallidas,
        "errores": errores_validacion if errores_validacion else None,
        "advertencias": advertencias_validacion if advertencias_validacion else None,
        "mensaje": (
            "Validación completada exitosamente"
            if not tiene_errores_criticos
            else f"Validación completada con {len(errores_validacion)} error(es)"
        ),
    }

    result["validation_summary"] = resumen
    result["validation_metrics"] = metricas_totales

    logger.info("=" * 80)
    logger.info("Agente de Validación de Datos completado")
    logger.info(f"Éxito: {resumen['exito']}, Items: {total_items}, Alertas: {total_alertas}")
    if errores_validacion:
        logger.warning(f"Errores encontrados: {errores_validacion}")
    logger.info("=" * 80)

    if telemetry_context:
        telemetry_context.record_metric("validation", "total_processed", total_items)
        telemetry_context.record_metric("validation", "total_alerts", total_alertas)
        telemetry_context.record_metric("validation", "total_errors", total_errores_items)
        telemetry_context.end_step("validate_data", "success" if resumen["exito"] else "warning", metadata=resumen)

    return result
