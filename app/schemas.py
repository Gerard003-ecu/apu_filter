"""
Esquemas de datos para Análisis de Precios Unitarios (APU).

Este módulo define las estructuras de datos robustas y validadas para representar
insumos de construcción en diferentes categorías: Mano de Obra, Equipo, Suministro,
Transporte y Otros.

Características principales:
- Validación automática de datos con mensajes descriptivos
- Normalización consistente de unidades y descripciones
- Type hints completos para mejor IDE support
- Inmutabilidad donde sea apropiado
- Factory pattern para creación dinámica
- Logging detallado de inconsistencias
"""

from __future__ import annotations

import logging
import re
import unicodedata
import warnings
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from functools import lru_cache
from typing import Any, ClassVar, Dict, Final, Optional, Protocol, Set, Type, Union

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================

# Tolerancias para validaciones numéricas
VALOR_TOTAL_TOLERANCE: Final[float] = 0.01  # 1% tolerancia para valor_total
CANTIDAD_RENDIMIENTO_TOLERANCE: Final[float] = 0.0001  # Tolerancia para cálculos de rendimiento

# Límites de valores
MIN_CANTIDAD: Final[float] = 0.0
MAX_CANTIDAD: Final[float] = 1_000_000.0
MIN_PRECIO: Final[float] = 0.0
MAX_PRECIO: Final[float] = 1_000_000_000.0
MIN_RENDIMIENTO: Final[float] = 0.0
MAX_RENDIMIENTO: Final[float] = 1000.0

# Longitud máxima de strings
MAX_CODIGO_LENGTH: Final[int] = 50
MAX_DESCRIPCION_LENGTH: Final[int] = 500

# Unidades válidas por categoría
UNIDADES_TIEMPO: Final[Set[str]] = frozenset({
    'HORA', 'DIA', 'SEMANA', 'MES', 'JOR'
})

UNIDADES_MASA: Final[Set[str]] = frozenset({
    'KG', 'GR', 'TON', 'LB'
})

UNIDADES_VOLUMEN: Final[Set[str]] = frozenset({
    'M3', 'L', 'ML', 'GAL'
})

UNIDADES_AREA: Final[Set[str]] = frozenset({
    'M2', 'CM2'
})

UNIDADES_LONGITUD: Final[Set[str]] = frozenset({
    'M', 'KM', 'CM', 'MM'
})

UNIDADES_TRANSPORTE: Final[Set[str]] = frozenset({
    'KM', 'VIAJE', 'VIAJES', 'MILLA', 'TON-KM'
})

UNIDADES_GENERICAS: Final[Set[str]] = frozenset({
    'UNIDAD', 'UND', 'U', 'PAR', 'JUEGO', 'KIT'
})

# Mapeo completo de normalización de unidades
UNIDAD_NORMALIZADA_MAP: Final[Dict[str, str]] = {
    # Tiempo
    'hora': 'HORA', 'hr': 'HORA', 'hrs': 'HORA', 'h': 'HORA',
    'dia': 'DIA', 'dias': 'DIA', 'd': 'DIA',
    'semana': 'SEMANA', 'sem': 'SEMANA', 'semanas': 'SEMANA',
    'mes': 'MES', 'meses': 'MES',
    'jornal': 'JOR', 'jornales': 'JOR',
    
    # Masa
    'kg': 'KG', 'kilogramo': 'KG', 'kilogramos': 'KG', 'kilo': 'KG', 'kilos': 'KG',
    'gramo': 'GR', 'gramos': 'GR', 'gr': 'GR', 'g': 'GR',
    'tonelada': 'TON', 'toneladas': 'TON', 'ton': 'TON', 't': 'TON',
    'libra': 'LB', 'libras': 'LB', 'lb': 'LB',
    
    # Volumen
    'm3': 'M3', 'm³': 'M3', 'metro cubico': 'M3', 'metros cubicos': 'M3',
    'litro': 'L', 'litros': 'L', 'l': 'L', 'lt': 'L', 'lts': 'L',
    'mililitro': 'ML', 'mililitros': 'ML', 'ml': 'ML',
    'galon': 'GAL', 'galones': 'GAL', 'gal': 'GAL',
    
    # Área
    'm2': 'M2', 'm²': 'M2', 'metro cuadrado': 'M2', 'metros cuadrados': 'M2',
    'cm2': 'CM2', 'cm²': 'CM2',
    
    # Longitud
    'metro': 'M', 'metros': 'M', 'm': 'M', 'mts': 'M',
    'kilometro': 'KM', 'kilometros': 'KM', 'km': 'KM', 'kilómetro': 'KM', 'kilómetros': 'KM',
    'centimetro': 'CM', 'centimetros': 'CM', 'cm': 'CM',
    'milimetro': 'MM', 'milimetros': 'MM', 'mm': 'MM',
    
    # Transporte
    'viaje': 'VIAJE', 'viajes': 'VIAJES', 'vje': 'VIAJE',
    'milla': 'MILLA', 'millas': 'MILLA',
    'ton-km': 'TON-KM', 'ton km': 'TON-KM', 'tonelada-kilometro': 'TON-KM',
    
    # Genéricas
    'unidad': 'UNIDAD', 'unidades': 'UNIDAD', 'und': 'UNIDAD', 'u': 'UNIDAD', 'un': 'UNIDAD',
    'par': 'PAR', 'pares': 'PAR',
    'juego': 'JUEGO', 'juegos': 'JUEGO',
    'kit': 'KIT', 'kits': 'KIT',
}

# Formatos de origen válidos
FORMATOS_VALIDOS: Final[Set[str]] = frozenset({
    'FORMATO_A', 'FORMATO_B', 'FORMATO_C', 'GENERIC'
})

# ============================================================================
# ENUMERACIONES
# ============================================================================

class TipoInsumo(str, Enum):
    """Tipos válidos de insumos en un APU."""
    
    MANO_DE_OBRA = "MANO_DE_OBRA"
    EQUIPO = "EQUIPO"
    SUMINISTRO = "SUMINISTRO"
    TRANSPORTE = "TRANSPORTE"
    OTRO = "OTRO"
    
    @classmethod
    def from_string(cls, value: str) -> TipoInsumo:
        """
        Convierte string a TipoInsumo con normalización.
        
        Args:
            value: String a convertir
            
        Returns:
            TipoInsumo correspondiente
            
        Raises:
            InvalidTipoInsumoError: Si el valor no es válido
        """
        if isinstance(value, cls):
            return value
        
        normalized = str(value).strip().upper().replace(' ', '_')
        
        try:
            return cls(normalized)
        except ValueError:
            raise InvalidTipoInsumoError(
                f"Tipo de insumo inválido: '{value}'. "
                f"Valores válidos: {', '.join(t.value for t in cls)}"
            )
    
    @classmethod
    def get_valid_values(cls) -> Set[str]:
        """Retorna conjunto de valores válidos."""
        return {t.value for t in cls}


# ============================================================================
# EXCEPCIONES CUSTOM
# ============================================================================

class SchemaError(Exception):
    """Excepción base para errores de schema."""
    pass


class ValidationError(SchemaError):
    """Error de validación de datos."""
    pass


class InvalidTipoInsumoError(ValidationError):
    """Error cuando el tipo de insumo no es válido."""
    pass


class InsumoDataError(ValidationError):
    """Error en los datos del insumo."""
    pass


class UnitNormalizationError(SchemaError):
    """Error al normalizar unidades."""
    pass


# ============================================================================
# FUNCIONES DE NORMALIZACIÓN CON CACHE
# ============================================================================

@lru_cache(maxsize=512)
def normalize_unit(unit: Optional[str]) -> str:
    """
    Normaliza una unidad de medida a su forma estándar con cache.
    
    Args:
        unit: Unidad a normalizar
        
    Returns:
        Unidad normalizada en mayúsculas
        
    Examples:
        >>> normalize_unit('kg')
        'KG'
        >>> normalize_unit('metros')
        'M'
        >>> normalize_unit('hora')
        'HORA'
    """
    if unit is None or not isinstance(unit, str):
        return 'UNIDAD'  # Default fallback
    
    # Normalizar a minúsculas y limpiar espacios
    normalized = unit.strip().lower()
    
    if not normalized:
        return 'UNIDAD'
    
    # Buscar en mapeo
    result = UNIDAD_NORMALIZADA_MAP.get(normalized)
    
    if result:
        return result
    
    # Si no está en el mapeo, retornar en mayúsculas
    return unit.strip().upper()


@lru_cache(maxsize=1024)
def normalize_description(desc: Optional[str]) -> str:
    """
    Normaliza descripción con cache: elimina espacios extra, convierte a mayúsculas, 
    elimina tildes y caracteres especiales.
    
    Args:
        desc: Descripción a normalizar
        
    Returns:
        Descripción normalizada
        
    Examples:
        >>> normalize_description('  Concreto  f\'c=280  ')
        'CONCRETO FC=280'
        >>> normalize_description('Ácido Nitrógeno')
        'ACIDO NITROGENO'
    """
    if not desc or not isinstance(desc, str):
        return ""
    
    # Eliminar tildes (NFD normalization)
    desc = unicodedata.normalize('NFD', desc)
    desc = desc.encode('ASCII', 'ignore').decode('ASCII')
    
    # Limpiar espacios múltiples y caracteres especiales innecesarios
    desc = re.sub(r'[^\w\s\-./()=]', '', desc)
    desc = re.sub(r'\s+', ' ', desc.strip())
    
    return desc.upper()


@lru_cache(maxsize=256)
def normalize_codigo(codigo: Optional[str]) -> str:
    """
    Normaliza un código APU.
    
    Args:
        codigo: Código a normalizar
        
    Returns:
        Código normalizado
        
    Raises:
        ValidationError: Si el código es inválido
    """
    if not codigo or not isinstance(codigo, str):
        raise ValidationError("Código APU no puede estar vacío")
    
    # Limpiar y convertir a mayúsculas
    codigo_clean = codigo.strip().upper()
    
    # Validar longitud
    if len(codigo_clean) > MAX_CODIGO_LENGTH:
        raise ValidationError(
            f"Código APU demasiado largo: {len(codigo_clean)} caracteres "
            f"(máximo: {MAX_CODIGO_LENGTH})"
        )
    
    if len(codigo_clean) < 1:
        raise ValidationError("Código APU no puede estar vacío")
    
    # Remover caracteres no permitidos (mantener alfanuméricos, guiones, puntos)
    codigo_clean = re.sub(r'[^\w\-.]', '', codigo_clean)
    
    return codigo_clean


# ============================================================================
# VALIDADORES REUTILIZABLES
# ============================================================================

class NumericValidator:
    """Validador de valores numéricos con rangos configurables."""
    
    @staticmethod
    def validate_non_negative(
        value: Union[int, float, Decimal],
        field_name: str,
        min_value: float = 0.0,
        max_value: Optional[float] = None
    ) -> float:
        """
        Valida que un valor sea numérico y esté en el rango válido.
        
        Args:
            value: Valor a validar
            field_name: Nombre del campo para mensajes de error
            min_value: Valor mínimo permitido
            max_value: Valor máximo permitido (None = sin límite)
            
        Returns:
            Valor validado como float
            
        Raises:
            ValidationError: Si el valor es inválido
        """
        # Validar tipo
        if not isinstance(value, (int, float, Decimal)):
            raise ValidationError(
                f"{field_name} debe ser numérico, recibido: {type(value).__name__}"
            )
        
        # Convertir a float
        try:
            float_value = float(value)
        except (ValueError, TypeError) as e:
            raise ValidationError(
                f"No se puede convertir {field_name} a número: {e}"
            )
        
        # Validar NaN e infinito
        if not (float('-inf') < float_value < float('inf')):
            raise ValidationError(
                f"{field_name} debe ser un número finito, recibido: {float_value}"
            )
        
        # Validar rango mínimo
        if float_value < min_value:
            raise ValidationError(
                f"{field_name} no puede ser menor que {min_value}, recibido: {float_value}"
            )
        
        # Validar rango máximo
        if max_value is not None and float_value > max_value:
            raise ValidationError(
                f"{field_name} no puede ser mayor que {max_value}, recibido: {float_value}"
            )
        
        return float_value
    
    @staticmethod
    def validate_positive(value: Union[int, float], field_name: str) -> float:
        """Valida que un valor sea positivo (> 0)."""
        validated = NumericValidator.validate_non_negative(value, field_name)
        
        if validated <= 0:
            raise ValidationError(f"{field_name} debe ser mayor que 0")
        
        return validated


class StringValidator:
    """Validador de strings."""
    
    @staticmethod
    def validate_non_empty(
        value: str,
        field_name: str,
        max_length: Optional[int] = None
    ) -> str:
        """
        Valida que un string no esté vacío.
        
        Args:
            value: String a validar
            field_name: Nombre del campo
            max_length: Longitud máxima permitida
            
        Returns:
            String validado
            
        Raises:
            ValidationError: Si el string es inválido
        """
        if not isinstance(value, str):
            raise ValidationError(
                f"{field_name} debe ser string, recibido: {type(value).__name__}"
            )
        
        if not value or not value.strip():
            raise ValidationError(f"{field_name} no puede estar vacío")
        
        if max_length and len(value) > max_length:
            raise ValidationError(
                f"{field_name} demasiado largo: {len(value)} caracteres "
                f"(máximo: {max_length})"
            )
        
        return value.strip()


# ============================================================================
# PROTOCOLO PARA INSUMOS
# ============================================================================

class InsumoProtocol(Protocol):
    """Protocolo que define la interfaz de un insumo."""
    
    codigo_apu: str
    descripcion_apu: str
    unidad_apu: str
    descripcion_insumo: str
    unidad_insumo: str
    cantidad: float
    precio_unitario: float
    valor_total: float
    categoria: str
    tipo_insumo: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el insumo a diccionario."""
        ...
    
    def validate(self) -> bool:
        """Valida el insumo."""
        ...


# ============================================================================
# CLASE BASE INMUTABLE
# ============================================================================

@dataclass(frozen=False)  # No frozen para permitir __post_init__
class InsumoProcesado:
    """
    Estructura base para cualquier insumo de APU.
    
    Proporciona validación automática, normalización y serialización.
    
    Attributes:
        codigo_apu: Código único del APU
        descripcion_apu: Descripción del APU
        unidad_apu: Unidad de medida del APU
        descripcion_insumo: Descripción del insumo
        unidad_insumo: Unidad de medida del insumo
        cantidad: Cantidad del insumo
        precio_unitario: Precio unitario del insumo
        valor_total: Valor total calculado
        categoria: Categoría del insumo (sincronizado con tipo_insumo)
        tipo_insumo: Tipo de insumo (enum)
        formato_origen: Formato de origen de los datos
        rendimiento: Rendimiento (para mano de obra)
        normalized_desc: Descripción normalizada (generada automáticamente)
    """
    
    # Campos obligatorios
    codigo_apu: str
    descripcion_apu: str
    unidad_apu: str
    descripcion_insumo: str
    unidad_insumo: str
    cantidad: float
    precio_unitario: float
    valor_total: float
    tipo_insumo: str
    
    # Campos opcionales con defaults
    categoria: str = field(default="OTRO")
    formato_origen: str = field(default="GENERIC")
    rendimiento: float = field(default=0.0)
    normalized_desc: str = field(default="", init=False)
    
    # Metadatos de validación
    _validated: bool = field(default=False, init=False, repr=False)
    
    # Constantes de clase
    EXPECTED_UNITS: ClassVar[Set[str]] = UNIDADES_GENERICAS
    REQUIRES_RENDIMIENTO: ClassVar[bool] = False
    
    def __post_init__(self):
        """
        Validación y normalización automática después de inicialización.
        
        Orden de operaciones:
        1. Normalizar campos
        2. Validar campos requeridos
        3. Validar coherencia lógica
        4. Marcar como validado
        """
        try:
            self._normalize_all_fields()
            self._validate_required_fields()
            self._validate_numeric_fields()
            self._validate_consistency()
            self._post_validation_hook()
            
            # Marcar como validado
            object.__setattr__(self, '_validated', True)
            
        except Exception as e:
            logger.error(
                f"Error inicializando {self.__class__.__name__} "
                f"para APU {getattr(self, 'codigo_apu', 'UNKNOWN')}: {e}"
            )
            raise
    
    def _normalize_all_fields(self):
        """Normaliza todos los campos del insumo."""
        # Normalizar código
        try:
            object.__setattr__(self, 'codigo_apu', normalize_codigo(self.codigo_apu))
        except ValidationError as e:
            raise ValidationError(f"Error en codigo_apu: {e}")
        
        # Normalizar descripciones
        object.__setattr__(
            self, 'descripcion_apu',
            normalize_description(self.descripcion_apu)
        )
        object.__setattr__(
            self, 'descripcion_insumo',
            normalize_description(self.descripcion_insumo)
        )
        object.__setattr__(
            self, 'normalized_desc',
            normalize_description(self.descripcion_insumo)
        )
        
        # Normalizar unidades
        object.__setattr__(self, 'unidad_apu', normalize_unit(self.unidad_apu))
        object.__setattr__(self, 'unidad_insumo', normalize_unit(self.unidad_insumo))
        
        # Normalizar tipo_insumo
        tipo_enum = TipoInsumo.from_string(self.tipo_insumo)
        object.__setattr__(self, 'tipo_insumo', tipo_enum.value)
        
        # Sincronizar categoria con tipo_insumo
        object.__setattr__(self, 'categoria', tipo_enum.value)
        
        # Normalizar formato_origen
        formato_upper = str(self.formato_origen).strip().upper()
        if formato_upper not in FORMATOS_VALIDOS:
            logger.warning(
                f"Formato origen '{self.formato_origen}' no reconocido, "
                f"usando 'GENERIC'"
            )
            formato_upper = "GENERIC"
        object.__setattr__(self, 'formato_origen', formato_upper)
    
    def _validate_required_fields(self):
        """Valida campos obligatorios."""
        # Validar codigo_apu (ya validado en normalización)
        StringValidator.validate_non_empty(
            self.codigo_apu, "codigo_apu", MAX_CODIGO_LENGTH
        )
        
        # Validar descripción insumo
        StringValidator.validate_non_empty(
            self.descripcion_insumo, "descripcion_insumo", MAX_DESCRIPCION_LENGTH
        )
    
    def _validate_numeric_fields(self):
        """Valida todos los campos numéricos."""
        # Validar cantidad
        cantidad_validated = NumericValidator.validate_non_negative(
            self.cantidad, "cantidad", MIN_CANTIDAD, MAX_CANTIDAD
        )
        object.__setattr__(self, 'cantidad', cantidad_validated)
        
        # Validar precio_unitario
        precio_validated = NumericValidator.validate_non_negative(
            self.precio_unitario, "precio_unitario", MIN_PRECIO, MAX_PRECIO
        )
        object.__setattr__(self, 'precio_unitario', precio_validated)
        
        # Validar valor_total
        valor_validated = NumericValidator.validate_non_negative(
            self.valor_total, "valor_total", MIN_PRECIO, MAX_PRECIO
        )
        object.__setattr__(self, 'valor_total', valor_validated)
        
        # Validar rendimiento
        rendimiento_validated = NumericValidator.validate_non_negative(
            self.rendimiento, "rendimiento", MIN_RENDIMIENTO, MAX_RENDIMIENTO
        )
        object.__setattr__(self, 'rendimiento', rendimiento_validated)
    
    def _validate_consistency(self):
        """
        Valida coherencia lógica entre campos.
        Puede ser sobreescrito por subclases.
        """
        # Validar coherencia valor_total
        self._validate_valor_total_consistency()
        
        # Validar unidades esperadas (si están definidas)
        if self.EXPECTED_UNITS:
            self._validate_expected_units()
        
        # Validar rendimiento si es requerido
        if self.REQUIRES_RENDIMIENTO:
            self._validate_rendimiento_required()
    
    def _validate_valor_total_consistency(self):
        """Valida que valor_total sea coherente con cantidad * precio_unitario."""
        expected_total = self.cantidad * self.precio_unitario
        
        # Calcular diferencia relativa
        if expected_total > 0:
            diff_rel = abs(self.valor_total - expected_total) / expected_total
        else:
            diff_rel = abs(self.valor_total - expected_total)
        
        if diff_rel > VALOR_TOTAL_TOLERANCE:
            warnings.warn(
                f"{self.__class__.__name__} [{self.codigo_apu}]: "
                f"valor_total ({self.valor_total:.4f}) difiere del esperado "
                f"({expected_total:.4f}). Diferencia relativa: {diff_rel:.2%}",
                UserWarning,
                stacklevel=3
            )
    
    def _validate_expected_units(self):
        """Valida que las unidades estén en el conjunto esperado."""
        if (self.unidad_apu not in self.EXPECTED_UNITS and
            self.unidad_insumo not in self.EXPECTED_UNITS):
            
            warnings.warn(
                f"{self.__class__.__name__} [{self.codigo_apu}]: "
                f"unidades '{self.unidad_apu}'/'{self.unidad_insumo}' no son típicas. "
                f"Unidades esperadas: {self.EXPECTED_UNITS}",
                UserWarning,
                stacklevel=3
            )
    
    def _validate_rendimiento_required(self):
        """Valida que rendimiento sea > 0 si es requerido."""
        if self.rendimiento <= 0:
            warnings.warn(
                f"{self.__class__.__name__} [{self.codigo_apu}]: "
                f"rendimiento debería ser > 0 para este tipo de insumo",
                UserWarning,
                stacklevel=3
            )
    
    def _post_validation_hook(self):
        """
        Hook para validaciones adicionales en subclases.
        Se ejecuta después de todas las validaciones básicas.
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte el insumo a diccionario para serialización.
        
        Returns:
            Diccionario con todos los campos del insumo
        """
        data = asdict(self)
        # Remover campos internos
        data.pop('_validated', None)
        return data
    
    def validate(self) -> bool:
        """
        Verifica si el insumo está validado.
        
        Returns:
            True si el insumo pasó todas las validaciones
        """
        return self._validated
    
    def __repr__(self) -> str:
        """Representación string mejorada."""
        return (
            f"{self.__class__.__name__}("
            f"codigo_apu='{self.codigo_apu}', "
            f"tipo='{self.tipo_insumo}', "
            f"cantidad={self.cantidad:.4f}, "
            f"valor_total={self.valor_total:.2f})"
        )


# ============================================================================
# CLASES ESPECÍFICAS POR TIPO DE INSUMO
# ============================================================================

@dataclass(frozen=False)
class ManoDeObra(InsumoProcesado):
    """
    Insumo de Mano de Obra.
    
    Características específicas:
    - Requiere rendimiento > 0
    - Unidades típicas: HORA, DIA, JOR, SEMANA, MES
    - Puede incluir jornal (campo opcional adicional)
    """
    
    # Sobrescribir constantes de clase
    EXPECTED_UNITS: ClassVar[Set[str]] = UNIDADES_TIEMPO
    REQUIRES_RENDIMIENTO: ClassVar[bool] = True
    
    # Campo adicional específico
    jornal: float = field(default=0.0)
    
    def _post_validation_hook(self):
        """Validaciones específicas de Mano de Obra."""
        super()._post_validation_hook()
        
        # Validar jornal si existe
        if self.jornal != 0.0:
            jornal_validated = NumericValidator.validate_non_negative(
                self.jornal, "jornal", MIN_PRECIO, MAX_PRECIO
            )
            object.__setattr__(self, 'jornal', jornal_validated)
        
        # Validar coherencia rendimiento-cantidad
        if self.rendimiento > 0:
            expected_cantidad = 1.0 / self.rendimiento
            diff = abs(self.cantidad - expected_cantidad)
            
            if diff > CANTIDAD_RENDIMIENTO_TOLERANCE:
                warnings.warn(
                    f"ManoDeObra [{self.codigo_apu}]: "
                    f"rendimiento={self.rendimiento:.4f} sugiere "
                    f"cantidad≈{expected_cantidad:.4f}, pero cantidad={self.cantidad:.4f}",
                    UserWarning,
                    stacklevel=4
                )


@dataclass(frozen=False)
class Equipo(InsumoProcesado):
    """
    Insumo de Equipo.
    
    Características específicas:
    - Unidades típicas: HORA, DIA
    - Rendimiento no aplica (debe ser 0)
    """
    
    EXPECTED_UNITS: ClassVar[Set[str]] = UNIDADES_TIEMPO
    REQUIRES_RENDIMIENTO: ClassVar[bool] = False
    
    def _post_validation_hook(self):
        """Validaciones específicas de Equipo."""
        super()._post_validation_hook()
        
        # Advertir si rendimiento != 0
        if self.rendimiento != 0:
            warnings.warn(
                f"Equipo [{self.codigo_apu}]: "
                f"rendimiento={self.rendimiento} no es relevante para equipos. "
                f"Se recomienda usar 0.0",
                UserWarning,
                stacklevel=4
            )


@dataclass(frozen=False)
class Suministro(InsumoProcesado):
    """
    Insumo de Suministro.
    
    Características específicas:
    - Unidades típicas: KG, M2, M3, L, UNIDAD
    - Cantidad debería ser > 0 (salvo casos especiales)
    - Rendimiento no aplica
    """
    
    EXPECTED_UNITS: ClassVar[Set[str]] = (
        UNIDADES_MASA | UNIDADES_VOLUMEN | UNIDADES_AREA | 
        UNIDADES_LONGITUD | UNIDADES_GENERICAS
    )
    REQUIRES_RENDIMIENTO: ClassVar[bool] = False
    
    def _post_validation_hook(self):
        """Validaciones específicas de Suministro."""
        super()._post_validation_hook()
        
        # Advertir si cantidad == 0
        if self.cantidad == 0:
            warnings.warn(
                f"Suministro [{self.codigo_apu}]: "
                f"cantidad=0.0. ¿Es intencional? Los suministros suelen tener cantidad > 0",
                UserWarning,
                stacklevel=4
            )
        
        # Advertir si rendimiento != 0
        if self.rendimiento != 0:
            warnings.warn(
                f"Suministro [{self.codigo_apu}]: "
                f"rendimiento={self.rendimiento} no es relevante. Se recomienda usar 0.0",
                UserWarning,
                stacklevel=4
            )


@dataclass(frozen=False)
class Transporte(InsumoProcesado):
    """
    Insumo de Transporte.
    
    Características específicas:
    - Unidades típicas: KM, VIAJE, TON-KM
    - Rendimiento no aplica
    """
    
    EXPECTED_UNITS: ClassVar[Set[str]] = UNIDADES_TRANSPORTE
    REQUIRES_RENDIMIENTO: ClassVar[bool] = False
    
    def _post_validation_hook(self):
        """Validaciones específicas de Transporte."""
        super()._post_validation_hook()
        
        # Advertir si rendimiento != 0
        if self.rendimiento != 0:
            warnings.warn(
                f"Transporte [{self.codigo_apu}]: "
                f"rendimiento={self.rendimiento} no es relevante. Se recomienda usar 0.0",
                UserWarning,
                stacklevel=4
            )


@dataclass(frozen=False)
class Otro(InsumoProcesado):
    """
    Insumo de categoría Otro.
    
    Categoría genérica para insumos que no encajan en las categorías estándar.
    Validaciones mínimas.
    """
    
    EXPECTED_UNITS: ClassVar[Set[str]] = set()  # Cualquier unidad es válida
    REQUIRES_RENDIMIENTO: ClassVar[bool] = False


# ============================================================================
# MAPEO DE TIPOS A CLASES
# ============================================================================

INSUMO_CLASS_MAP: Final[Dict[str, Type[InsumoProcesado]]] = {
    TipoInsumo.MANO_DE_OBRA.value: ManoDeObra,
    TipoInsumo.EQUIPO.value: Equipo,
    TipoInsumo.SUMINISTRO.value: Suministro,
    TipoInsumo.TRANSPORTE.value: Transporte,
    TipoInsumo.OTRO.value: Otro,
}


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_insumo(tipo_insumo: Union[str, TipoInsumo], **kwargs) -> InsumoProcesado:
    """
    Factory function para crear instancias de insumos dinámicamente.
    
    Args:
        tipo_insumo: Tipo de insumo a crear (string o enum)
        **kwargs: Argumentos para la clase específica
        
    Returns:
        Instancia validada de la clase de insumo correspondiente
        
    Raises:
        InvalidTipoInsumoError: Si el tipo_insumo no es válido
        ValidationError: Si los argumentos son inválidos
        
    Examples:
        >>> insumo = create_insumo(
        ...     'MANO_DE_OBRA',
        ...     codigo_apu='APU-001',
        ...     descripcion_apu='Concreto',
        ...     unidad_apu='M3',
        ...     descripcion_insumo='Oficial',
        ...     unidad_insumo='HORA',
        ...     cantidad=1.5,
        ...     precio_unitario=15000,
        ...     valor_total=22500,
        ...     rendimiento=0.67
        ... )
    """
    try:
        # Normalizar tipo_insumo
        tipo_enum = TipoInsumo.from_string(tipo_insumo)
        tipo_normalizado = tipo_enum.value
        
        # Forzar coherencia en kwargs
        kwargs['tipo_insumo'] = tipo_normalizado
        kwargs['categoria'] = tipo_normalizado
        
        # Obtener clase correspondiente
        insumo_class = INSUMO_CLASS_MAP.get(tipo_normalizado)
        
        if not insumo_class:
            raise InvalidTipoInsumoError(
                f"No hay clase definida para tipo: {tipo_normalizado}"
            )
        
        # Crear instancia
        logger.debug(f"Creando insumo tipo {tipo_normalizado} para APU {kwargs.get('codigo_apu', 'UNKNOWN')}")
        return insumo_class(**kwargs)
        
    except InvalidTipoInsumoError:
        raise
    except TypeError as e:
        # Error en los argumentos
        raise ValidationError(
            f"Error creando insumo tipo {tipo_insumo}: {e}. "
            f"Argumentos recibidos: {list(kwargs.keys())}"
        ) from e
    except Exception as e:
        # Otros errores
        logger.error(f"Error inesperado creando insumo: {e}", exc_info=True)
        raise InsumoDataError(f"Error creando insumo: {e}") from e


def validate_insumo_data(insumo_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida y limpia datos crudos para crear un insumo.
    
    Normaliza campos, asigna defaults, valida existencia de campos críticos
    y prepara los datos para pasarlos a create_insumo().
    
    Args:
        insumo_data: Diccionario con datos crudos del insumo
        
    Returns:
        Diccionario validado y limpiado listo para create_insumo()
        
    Raises:
        ValidationError: Si faltan campos obligatorios o tienen tipos inválidos
        
    Examples:
        >>> raw_data = {
        ...     'codigo_apu': 'APU-001',
        ...     'descripcion_apu': 'Concreto',
        ...     'unidad_apu': 'm3',
        ...     'descripcion_insumo': 'Cemento',
        ...     'unidad_insumo': 'kg',
        ...     'tipo_insumo': 'suministro',
        ...     'cantidad': '50',
        ...     'precio_unitario': '35000'
        ... }
        >>> cleaned = validate_insumo_data(raw_data)
        >>> cleaned['cantidad']
        50.0
        >>> cleaned['tipo_insumo']
        'SUMINISTRO'
    """
    if not isinstance(insumo_data, dict):
        raise ValidationError("insumo_data debe ser un diccionario")
    
    # Campos obligatorios
    required_fields = [
        'codigo_apu', 'descripcion_apu', 'unidad_apu',
        'descripcion_insumo', 'unidad_insumo', 'tipo_insumo'
    ]
    
    # Verificar campos requeridos
    missing_fields = [f for f in required_fields if f not in insumo_data or insumo_data[f] is None]
    
    if missing_fields:
        raise ValidationError(
            f"Campos requeridos faltantes o nulos: {', '.join(missing_fields)}"
        )
    
    # Valores por defecto
    defaults = {
        'cantidad': 0.0,
        'precio_unitario': 0.0,
        'valor_total': 0.0,
        'categoria': 'OTRO',
        'formato_origen': 'GENERIC',
        'rendimiento': 0.0,
        'jornal': 0.0
    }
    
    # Crear diccionario limpio
    cleaned_data = {}
    
    # Procesar todos los campos del input
    for key, value in insumo_data.items():
        if value is None:
            # Usar default si existe
            if key in defaults:
                cleaned_data[key] = defaults[key]
            continue
        
        # Convertir campos numéricos
        if key in ['cantidad', 'precio_unitario', 'valor_total', 'rendimiento', 'jornal']:
            try:
                cleaned_data[key] = float(value)
            except (ValueError, TypeError) as e:
                raise ValidationError(
                    f"Campo '{key}' debe ser numérico. Valor recibido: {value}"
                ) from e
        else:
            # Mantener otros campos como están
            cleaned_data[key] = value
    
    # Agregar defaults para campos no proporcionados
    for field, default_value in defaults.items():
        if field not in cleaned_data:
            cleaned_data[field] = default_value
    
    # Validar y normalizar tipo_insumo
    try:
        tipo_enum = TipoInsumo.from_string(cleaned_data['tipo_insumo'])
        cleaned_data['tipo_insumo'] = tipo_enum.value
        cleaned_data['categoria'] = tipo_enum.value  # Sincronizar
    except InvalidTipoInsumoError as e:
        raise ValidationError(f"Tipo de insumo inválido: {e}") from e
    
    # Calcular valor_total si no se proporcionó o es 0
    if cleaned_data.get('valor_total', 0) == 0:
        cantidad = cleaned_data.get('cantidad', 0)
        precio = cleaned_data.get('precio_unitario', 0)
        cleaned_data['valor_total'] = cantidad * precio
    
    logger.debug(
        f"Datos validados para insumo {cleaned_data.get('codigo_apu', 'UNKNOWN')}: "
        f"tipo={cleaned_data.get('tipo_insumo')}"
    )
    
    return cleaned_data


def create_insumo_from_raw(raw_data: Dict[str, Any]) -> InsumoProcesado:
    """
    Crea un insumo directamente desde datos crudos.
    
    Combina validate_insumo_data() y create_insumo() en una sola operación.
    
    Args:
        raw_data: Diccionario con datos crudos del insumo
        
    Returns:
        Instancia validada de InsumoProcesado
        
    Raises:
        ValidationError: Si los datos son inválidos
    """
    cleaned_data = validate_insumo_data(raw_data)
    tipo_insumo = cleaned_data.pop('tipo_insumo')
    return create_insumo(tipo_insumo, **cleaned_data)


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def get_tipo_insumo_class(tipo: Union[str, TipoInsumo]) -> Type[InsumoProcesado]:
    """
    Obtiene la clase correspondiente a un tipo de insumo.
    
    Args:
        tipo: Tipo de insumo
        
    Returns:
        Clase correspondiente
        
    Raises:
        InvalidTipoInsumoError: Si el tipo no es válido
    """
    tipo_enum = TipoInsumo.from_string(tipo)
    insumo_class = INSUMO_CLASS_MAP.get(tipo_enum.value)
    
    if not insumo_class:
        raise InvalidTipoInsumoError(
            f"No hay clase definida para tipo: {tipo_enum.value}"
        )
    
    return insumo_class


def get_all_tipo_insumo_values() -> Set[str]:
    """Retorna todos los valores válidos de tipo_insumo."""
    return TipoInsumo.get_valid_values()


def is_valid_tipo_insumo(tipo: str) -> bool:
    """
    Verifica si un tipo de insumo es válido.
    
    Args:
        tipo: Tipo a verificar
        
    Returns:
        True si es válido, False en caso contrario
    """
    try:
        TipoInsumo.from_string(tipo)
        return True
    except InvalidTipoInsumoError:
        return False


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Excepciones
    'SchemaError',
    'ValidationError',
    'InvalidTipoInsumoError',
    'InsumoDataError',
    'UnitNormalizationError',
    
    # Enums
    'TipoInsumo',
    
    # Clases principales
    'InsumoProcesado',
    'ManoDeObra',
    'Equipo',
    'Suministro',
    'Transporte',
    'Otro',
    
    # Factory functions
    'create_insumo',
    'create_insumo_from_raw',
    'validate_insumo_data',
    
    # Funciones de normalización
    'normalize_unit',
    'normalize_description',
    'normalize_codigo',
    
    # Validadores
    'NumericValidator',
    'StringValidator',
    
    # Utilidades
    'get_tipo_insumo_class',
    'get_all_tipo_insumo_values',
    'is_valid_tipo_insumo',
    
    # Constantes útiles
    'UNIDADES_TIEMPO',
    'UNIDADES_MASA',
    'UNIDADES_VOLUMEN',
    'UNIDADES_AREA',
    'UNIDADES_LONGITUD',
    'UNIDADES_TRANSPORTE',
    'UNIDADES_GENERICAS',
]