from enum import Enum
from typing import Final


class ColumnNames:
    """Nombres de columnas estandarizados."""
    DESCRIPCION: Final[str] = 'DESCRIPCION'
    UNIDAD: Final[str] = 'UNIDAD'
    TIPO_INSUMO: Final[str] = 'TIPO_INSUMO'
    COSTO_TOTAL: Final[str] = 'COSTO_TOTAL'
    APU_ID: Final[str] = 'APU_ID'
    CATEGORIA: Final[str] = 'CATEGORIA'
    GRUPO_INSUMO: Final[str] = 'GRUPO_INSUMO'
    CODIGO: Final[str] = 'CODIGO'

    # Columnas usadas en PipelineDirector
    CODIGO_APU: Final[str] = "CODIGO_APU"
    DESCRIPCION_APU: Final[str] = "DESCRIPCION_APU"
    CANTIDAD_PRESUPUESTO: Final[str] = "CANTIDAD_PRESUPUESTO"
    DESCRIPCION_INSUMO: Final[str] = "DESCRIPCION_INSUMO"
    COSTO_INSUMO_EN_APU: Final[str] = "COSTO_INSUMO_EN_APU"
    CANTIDAD_APU: Final[str] = "CANTIDAD_APU"

    # Columnas calculadas
    VALOR_MATERIALES: Final[str] = 'Valor_Materiales'
    VALOR_MANO_OBRA: Final[str] = 'Valor_Mano_de_Obra'
    VALOR_EQUIPO: Final[str] = 'Valor_Equipo'
    VALOR_OTROS: Final[str] = 'Valor_Otros'

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
