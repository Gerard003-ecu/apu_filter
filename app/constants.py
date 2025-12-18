from enum import Enum
from typing import Final


class ColumnNames:
    """Nombres de columnas estandarizados en todo el sistema."""

    # Identificadores y Básicos
    CODIGO_APU: Final[str] = "CODIGO_APU"
    DESCRIPCION_APU: Final[str] = "DESCRIPCION_APU"
    DESCRIPCION_SECUNDARIA: Final[str] = "descripcion_secundaria"
    ORIGINAL_DESCRIPTION: Final[str] = "original_description"
    UNIDAD_APU: Final[str] = "UNIDAD_APU"

    # Cantidades
    CANTIDAD_APU: Final[str] = "CANTIDAD_APU"
    CANTIDAD_PRESUPUESTO: Final[str] = "CANTIDAD_PRESUPUESTO"

    # Insumos
    GRUPO_INSUMO: Final[str] = "GRUPO_INSUMO"
    DESCRIPCION_INSUMO: Final[str] = "DESCRIPCION_INSUMO"
    DESCRIPCION_INSUMO_NORM: Final[str] = "DESCRIPCION_INSUMO_NORM"
    VR_UNITARIO_INSUMO: Final[str] = "VR_UNITARIO_INSUMO"
    UNIDAD_INSUMO: Final[str] = "UNIDAD_INSUMO"
    TIPO_INSUMO: Final[str] = "TIPO_INSUMO"
    COSTO_INSUMO_EN_APU: Final[str] = "COSTO_INSUMO_EN_APU"

    # Procesamiento de Texto
    NORMALIZED_DESC: Final[str] = "NORMALIZED_DESC"
    DESC_NORMALIZED: Final[str] = "DESC_NORMALIZED"  # Alias para compatibilidad

    # Costos Calculados
    VR_UNITARIO_FINAL: Final[str] = "VR_UNITARIO_FINAL"
    VALOR_TOTAL_APU: Final[str] = "VALOR_TOTAL_APU"
    PRECIO_UNIT_APU: Final[str] = "PRECIO_UNIT_APU"

    # Categorías de Costo
    MATERIALES: Final[str] = "MATERIALES"
    MANO_DE_OBRA: Final[str] = "MANO DE OBRA"
    EQUIPO: Final[str] = "EQUIPO"
    OTROS: Final[str] = "OTROS"

    # Valores Unitarios Desglosados
    VALOR_SUMINISTRO_UN: Final[str] = "VALOR_SUMINISTRO_UN"
    VALOR_INSTALACION_UN: Final[str] = "VALOR_INSTALACION_UN"
    VALOR_CONSTRUCCION_UN: Final[str] = "VALOR_CONSTRUCCION_UN"

    # Valores Totales Desglosados
    VALOR_SUMINISTRO_TOTAL: Final[str] = "VALOR_SUMINISTRO_TOTAL"
    VALOR_INSTALACION_TOTAL: Final[str] = "VALOR_INSTALACION_TOTAL"
    VALOR_CONSTRUCCION_TOTAL: Final[str] = "VALOR_CONSTRUCCION_TOTAL"

    # Clasificación y Tiempos
    TIPO_APU: Final[str] = "tipo_apu"
    CATEGORIA: Final[str] = "CATEGORIA"
    TIEMPO_INSTALACION: Final[str] = "TIEMPO_INSTALACION"
    RENDIMIENTO: Final[str] = "RENDIMIENTO"
    RENDIMIENTO_DIA: Final[str] = "RENDIMIENTO_DIA"

    # Legacy/Old compatibility (keeping what was there to be safe, if still needed by other parts)
    DESCRIPCION: Final[str] = "DESCRIPCION"
    UNIDAD: Final[str] = "UNIDAD"
    COSTO_TOTAL: Final[str] = "COSTO_TOTAL"
    APU_ID: Final[str] = "APU_ID"
    CODIGO: Final[str] = "CODIGO"
    VALOR_MATERIALES: Final[str] = "Valor_Materiales"
    VALOR_MANO_OBRA: Final[str] = "Valor_Mano_de_Obra"
    VALOR_EQUIPO: Final[str] = "Valor_Equipo"
    VALOR_OTROS: Final[str] = "Valor_Otros"


class InsumoType(Enum):
    """Tipos de insumos soportados."""

    MATERIAL = "MATERIALES"
    MANO_DE_OBRA = "MANO_DE_OBRA"
    EQUIPO = "EQUIPO"
    TRANSPORTE = "TRANSPORTE"
    HERRAMIENTA = "HERRAMIENTA"
    SUBCONTRATO = "SUBCONTRATO"
    OTROS = "OTROS"


class ProcessingThresholds:
    """Umbrales y configuraciones de procesamiento."""

    MIN_MATCH_CONFIDENCE: float = 0.8
    MIN_CLASSIFICATION_COVERAGE: float = 0.95
    MAX_UNCLASSIFIED_RATIO: float = 0.05
