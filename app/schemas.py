"""
Este módulo define la "Constitución de los Datos". Establece las estructuras de datos
inmutables y fuertemente tipadas que representan los átomos del presupuesto (Insumos,
APUs). Implementa el patrón Factory para garantizar que ningún objeto se instancie
sin pasar por validaciones de invariantes de dominio.

Invariantes y Reglas de Dominio:
--------------------------------
1. Tipología Estricta (InsumoProcesado):
   Define la jerarquía de clases (ManoDeObra, Equipo, Suministro).

2. Normalización Canónica:
   Asegura que todas las unidades y descripciones se conviertan a un lenguaje común.

3. Validación Reactiva (__post_init__):
   Los objetos se autovalidan al nacer.

4. Estabilidad Topológica:
   Mide la entropía y diversidad de los componentes de un APU para determinar su robustez.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
import warnings
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from enum import Enum, IntEnum
from functools import lru_cache
from typing import Any, ClassVar, Dict, Final, List, Optional, Protocol, Set, Type, Union

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================

# Tolerancias para validaciones numéricas
VALOR_TOTAL_WARNING_TOLERANCE: Final[float] = 0.01  # 1% tolerancia (Warning)
VALOR_TOTAL_ERROR_TOLERANCE: Final[float] = 0.05    # 5% tolerancia (Error crítico)
CANTIDAD_RENDIMIENTO_TOLERANCE: Final[float] = 0.0001

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
UNIDADES_TIEMPO: Final[Set[str]] = frozenset({"HORA", "DIA", "SEMANA", "MES", "JOR"})
UNIDADES_MASA: Final[Set[str]] = frozenset({"KG", "GR", "TON", "LB"})
UNIDADES_VOLUMEN: Final[Set[str]] = frozenset({"M3", "L", "ML", "GAL"})
UNIDADES_AREA: Final[Set[str]] = frozenset({"M2", "CM2"})
UNIDADES_LONGITUD: Final[Set[str]] = frozenset({"M", "KM", "CM", "MM"})
UNIDADES_TRANSPORTE: Final[Set[str]] = frozenset({"KM", "VIAJE", "VIAJES", "MILLA", "TON-KM"})
UNIDADES_GENERICAS: Final[Set[str]] = frozenset({"UNIDAD", "UND", "U", "PAR", "JUEGO", "KIT", "%"})

# Mapeo completo de normalización de unidades
UNIDAD_NORMALIZADA_MAP: Final[Dict[str, str]] = {
    # Tiempo
    "hora": "HORA", "hr": "HORA", "hrs": "HORA", "h": "HORA",
    "dia": "DIA", "dias": "DIA", "d": "DIA",
    "semana": "SEMANA", "sem": "SEMANA",
    "mes": "MES", "meses": "MES",
    "jornal": "JOR", "jornales": "JOR",
    # Masa
    "kg": "KG", "kilogramo": "KG", "kilogramos": "KG", "kilo": "KG",
    "gramo": "GR", "gr": "GR", "g": "GR",
    "tonelada": "TON", "ton": "TON", "t": "TON",
    "libra": "LB", "lb": "LB",
    # Volumen
    "m3": "M3", "m³": "M3", "metro cubico": "M3",
    "litro": "L", "litros": "L", "l": "L", "lt": "L",
    "galon": "GAL", "galones": "GAL", "gal": "GAL",
    # Área
    "m2": "M2", "m²": "M2", "metro cuadrado": "M2",
    "cm2": "CM2", "cm²": "CM2",
    # Longitud
    "metro": "M", "metros": "M", "m": "M", "mts": "M",
    "kilometro": "KM", "km": "KM",
    "centimetro": "CM", "cm": "CM",
    "milimetro": "MM", "mm": "MM",
    # Transporte
    "viaje": "VIAJE", "viajes": "VIAJES", "vje": "VIAJE",
    "ton-km": "TON-KM", "ton km": "TON-KM",
    # Genéricas
    "unidad": "UNIDAD", "unidades": "UNIDAD", "und": "UNIDAD", "u": "UNIDAD", "un": "UNIDAD",
    "par": "PAR", "juego": "JUEGO", "kit": "KIT",
}

FORMATOS_VALIDOS: Final[Set[str]] = frozenset({"FORMATO_A", "FORMATO_B", "FORMATO_C", "GENERIC"})


# ============================================================================
# ENUMERACIONES
# ============================================================================

class Stratum(IntEnum):
    """Niveles de abstracción en la topología del negocio."""
    ROOT = 0
    STRATEGY = 1
    TACTIC = 2
    LOGISTICS = 3


class TipoInsumo(str, Enum):
    """Tipos válidos de insumos en un APU."""
    MANO_DE_OBRA = "MANO_DE_OBRA"
    EQUIPO = "EQUIPO"
    SUMINISTRO = "SUMINISTRO"
    TRANSPORTE = "TRANSPORTE"
    OTRO = "OTRO"

    @classmethod
    def from_string(cls, value: str) -> TipoInsumo:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().upper().replace(" ", "_")
        try:
            return cls(normalized)
        except ValueError:
            raise InvalidTipoInsumoError(f"Tipo de insumo inválido: '{value}'")

    @classmethod
    def get_valid_values(cls) -> Set[str]:
        return {t.value for t in cls}


# ============================================================================
# EXCEPCIONES CUSTOM
# ============================================================================

class SchemaError(Exception): pass
class ValidationError(SchemaError): pass
class InvalidTipoInsumoError(ValidationError): pass
class InsumoDataError(ValidationError): pass
class UnitNormalizationError(SchemaError): pass


# ============================================================================
# FUNCIONES DE NORMALIZACIÓN
# ============================================================================

@lru_cache(maxsize=512)
def normalize_unit(unit: Optional[str]) -> str:
    if not isinstance(unit, str) or not unit.strip():
        return "UNIDAD"
    normalized = unit.strip().lower()
    result = UNIDAD_NORMALIZADA_MAP.get(normalized)
    if result:
        return result
    return unit.strip().upper()

@lru_cache(maxsize=1024)
def normalize_description(desc: Optional[str]) -> str:
    if not isinstance(desc, str) or not desc:
        return ""
    desc = unicodedata.normalize("NFD", desc)
    desc = desc.encode("ASCII", "ignore").decode("ASCII")
    desc = re.sub(r"[^\w\s\-./()=]", "", desc)
    desc = re.sub(r"\s+", " ", desc.strip())
    return desc.upper()

@lru_cache(maxsize=256)
def normalize_codigo(codigo: Optional[str]) -> str:
    if not codigo or not isinstance(codigo, str):
        raise ValidationError("Código APU no puede estar vacío")
    codigo_clean = codigo.strip().upper()
    if len(codigo_clean) > MAX_CODIGO_LENGTH:
        raise ValidationError(f"Código APU demasiado largo: {len(codigo_clean)}")
    if len(codigo_clean) < 1:
        raise ValidationError("Código APU no puede estar vacío")
    codigo_clean = re.sub(r"[^\w\-.]", "", codigo_clean)
    return codigo_clean


# ============================================================================
# VALIDADORES REUTILIZABLES
# ============================================================================

class NumericValidator:
    @staticmethod
    def validate_non_negative(value: Union[int, float, Decimal], field_name: str, min_value: float = 0.0, max_value: Optional[float] = None) -> float:
        if not isinstance(value, (int, float, Decimal)):
            raise ValidationError(f"{field_name} debe ser numérico")
        try:
            float_value = float(value)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"No se puede convertir {field_name}: {e}")

        if not (float("-inf") < float_value < float("inf")):
            raise ValidationError(f"{field_name} debe ser finito")

        if float_value < min_value:
            raise ValidationError(f"{field_name} < {min_value}")
        if max_value is not None and float_value > max_value:
            raise ValidationError(f"{field_name} > {max_value}")

        return float_value

class StringValidator:
    @staticmethod
    def validate_non_empty(value: str, field_name: str, max_length: Optional[int] = None) -> str:
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} debe ser string")
        if not value or not value.strip():
            raise ValidationError(f"{field_name} no puede estar vacío")
        if max_length and len(value) > max_length:
            raise ValidationError(f"{field_name} demasiado largo")
        return value.strip()


# ============================================================================
# CLASE BASE INMUTABLE
# ============================================================================

@dataclass(kw_only=True)
class TopologicalNode:
    id: str = field(default="")
    stratum: Stratum = field(default=Stratum.LOGISTICS)
    description: str = field(default="")
    structural_health: float = 1.0
    is_floating: bool = False

    def validate_connectivity(self):
        if self.stratum == Stratum.LOGISTICS and hasattr(self, "children") and self.children:
            raise ValueError(f"Violación de Invariante: Nodo Logístico ({self.id}) con hijos.")


@dataclass(frozen=False)
class InsumoProcesado(TopologicalNode):
    """
    Estructura base para cualquier insumo de APU.
    """
    codigo_apu: str
    descripcion_apu: str
    unidad_apu: str
    descripcion_insumo: str
    unidad_insumo: str
    cantidad: float
    precio_unitario: float
    valor_total: float
    tipo_insumo: str

    capitulo: str = field(default="GENERAL")
    categoria: str = field(default="OTRO", init=True)
    formato_origen: str = field(default="GENERIC")
    rendimiento: float = field(default=0.0)
    normalized_desc: str = field(default="", init=False)

    _validated: bool = field(default=False, init=False, repr=False)

    EXPECTED_UNITS: ClassVar[Set[str]] = UNIDADES_GENERICAS
    REQUIRES_RENDIMIENTO: ClassVar[bool] = False

    def __post_init__(self):
        try:
            self.stratum = Stratum.LOGISTICS
            self.description = self.descripcion_insumo
            self.id = f"{self.codigo_apu}_{self.descripcion_insumo[:20]}"

            self._normalize_all_fields()
            self._validate_required_fields()
            self._validate_numeric_fields()
            self._validate_consistency()
            self._post_validation_hook()

            if self.precio_unitario < 0:
                raise ValueError(f"Física Violada: Precio negativo en {self.id}")

            object.__setattr__(self, "_validated", True)

        except Exception as e:
            logger.error(f"Error inicializando {self.__class__.__name__}: {e}")
            raise

    @property
    def total_cost(self) -> float:
        return self.cantidad * self.precio_unitario

    def _normalize_all_fields(self):
        object.__setattr__(self, "codigo_apu", normalize_codigo(self.codigo_apu))
        object.__setattr__(self, "descripcion_apu", normalize_description(self.descripcion_apu))
        object.__setattr__(self, "descripcion_insumo", normalize_description(self.descripcion_insumo))
        object.__setattr__(self, "normalized_desc", normalize_description(self.descripcion_insumo))
        object.__setattr__(self, "unidad_apu", normalize_unit(self.unidad_apu))
        object.__setattr__(self, "unidad_insumo", normalize_unit(self.unidad_insumo))

        tipo_enum = TipoInsumo.from_string(self.tipo_insumo)
        object.__setattr__(self, "tipo_insumo", tipo_enum.value)
        object.__setattr__(self, "categoria", tipo_enum.value)

        formato_upper = str(self.formato_origen).strip().upper()
        if formato_upper not in FORMATOS_VALIDOS:
            formato_upper = "GENERIC"
        object.__setattr__(self, "formato_origen", formato_upper)

    def _validate_required_fields(self):
        StringValidator.validate_non_empty(self.codigo_apu, "codigo_apu", MAX_CODIGO_LENGTH)
        StringValidator.validate_non_empty(self.descripcion_insumo, "descripcion_insumo", MAX_DESCRIPCION_LENGTH)

    def _validate_numeric_fields(self):
        object.__setattr__(self, "cantidad", NumericValidator.validate_non_negative(self.cantidad, "cantidad", MIN_CANTIDAD, MAX_CANTIDAD))
        object.__setattr__(self, "precio_unitario", NumericValidator.validate_non_negative(self.precio_unitario, "precio_unitario", MIN_PRECIO, MAX_PRECIO))
        object.__setattr__(self, "valor_total", NumericValidator.validate_non_negative(self.valor_total, "valor_total", MIN_PRECIO, MAX_PRECIO))
        object.__setattr__(self, "rendimiento", NumericValidator.validate_non_negative(self.rendimiento, "rendimiento", MIN_RENDIMIENTO, MAX_RENDIMIENTO))

    def _validate_consistency(self):
        self._validate_valor_total_consistency()
        if self.EXPECTED_UNITS:
            self._validate_expected_units()
        if self.REQUIRES_RENDIMIENTO:
            self._validate_rendimiento_required()

    def _validate_valor_total_consistency(self):
        """
        Valida que valor_total sea coherente con cantidad * precio_unitario.
        Diferencia WARNING de ERROR.
        """
        expected_total = self.cantidad * self.precio_unitario

        if expected_total == 0 and self.valor_total == 0:
            return

        if expected_total != 0:
            diff_rel = abs(self.valor_total - expected_total) / expected_total
        else:
            diff_rel = float('inf') if self.valor_total > 0 else 0.0

        if diff_rel > VALOR_TOTAL_ERROR_TOLERANCE:
            # Tolerancia de error (5%) -> Levanta Excepción
            # Salvo que el valor sea muy pequeño (ruido numérico)
            if abs(self.valor_total - expected_total) > 0.01:
                raise ValidationError(
                    f"{self.__class__.__name__} [{self.codigo_apu}]: "
                    f"Inconsistencia grave: valor_total={self.valor_total:.2f} vs calculado={expected_total:.2f} "
                    f"(Diff: {diff_rel:.2%})"
                )
        elif diff_rel > VALOR_TOTAL_WARNING_TOLERANCE:
            # Tolerancia de warning (1%) -> Solo avisa
             if abs(self.valor_total - expected_total) > 0.01:
                warnings.warn(
                    f"{self.__class__.__name__} [{self.codigo_apu}]: "
                    f"Divergencia matemática: valor_total={self.valor_total:.2f} vs calculado={expected_total:.2f}",
                    UserWarning, stacklevel=3
                )

    def _validate_expected_units(self):
        if self.unidad_apu not in self.EXPECTED_UNITS and self.unidad_insumo not in self.EXPECTED_UNITS:
            pass # Silenciado por ahora para reducir ruido, o usar logger debug

    def _validate_rendimiento_required(self):
        if self.rendimiento <= 0:
            warnings.warn(f"Rendimiento debería ser > 0 en {self.codigo_apu}", UserWarning, stacklevel=3)

    def _post_validation_hook(self):
        pass

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("_validated", None)
        return data

    def validate(self) -> bool:
        return self._validated


@dataclass
class APUStructure(TopologicalNode):
    """
    Representa una actividad constructiva en el Nivel 2 (Táctica).
    """
    unit: str = ""
    quantity: float = 0.0
    resources: List[InsumoProcesado] = field(default_factory=list)

    def __post_init__(self):
        self.stratum = Stratum.TACTIC

    @property
    def support_base_width(self) -> int:
        return len(self.resources)

    @property
    def is_inverted_pyramid(self) -> bool:
        return self.quantity > 1000 and self.support_base_width == 1

    def topological_stability_index(self) -> float:
        """
        Calcula el Índice de Estabilidad Topológica (0.0 - 1.0).
        Basado en la diversidad de insumos (Tipos) y la entropía de su distribución de costo.
        """
        if not self.resources:
            return 0.0

        # 1. Diversidad de Tipos (Robustez cualitativa)
        tipos = set(r.tipo_insumo for r in self.resources)
        # Normalizamos: si tiene > 3 tipos, es muy diverso (1.0).
        diversidad_score = min(len(tipos) / 3.0, 1.0)

        # 2. Entropía de Shannon de los costos (Robustez cuantitativa)
        # Si un solo insumo representa el 99% del costo, el sistema es frágil (baja entropía).
        # Si el costo está distribuido equitativamente, es estable (alta entropía).
        valores = [max(r.valor_total, 0.0) for r in self.resources]
        total_valor = sum(valores)

        if total_valor <= 0:
            return 0.0

        probs = [v / total_valor for v in valores]
        # Shannon Entropy: H = -sum(p * log(p))
        entropia = -sum(p * math.log(p) for p in probs if p > 0)

        # Normalizar entropía (H / H_max) donde H_max = log(N)
        n = len(valores)
        entropia_max = math.log(n) if n > 1 else 1.0

        entropia_norm = entropia / entropia_max if entropia_max > 0 else 0.0

        # Índice combinado
        return (diversidad_score * 0.4) + (entropia_norm * 0.6)

    def add_resource(self, resource: InsumoProcesado):
        if resource.stratum != Stratum.LOGISTICS:
            raise TypeError(f"Solo se admiten Insumos en APUStructure, recibido {resource.stratum}")
        self.resources.append(resource)


# ============================================================================
# CLASES ESPECÍFICAS
# ============================================================================

@dataclass(frozen=False)
class ManoDeObra(InsumoProcesado):
    EXPECTED_UNITS: ClassVar[Set[str]] = UNIDADES_TIEMPO
    REQUIRES_RENDIMIENTO: ClassVar[bool] = True
    jornal: float = field(default=0.0)

    def _post_validation_hook(self):
        super()._post_validation_hook()
        if self.jornal != 0.0:
            object.__setattr__(self, "jornal", NumericValidator.validate_non_negative(self.jornal, "jornal", MIN_PRECIO, MAX_PRECIO))
        if self.rendimiento > 0:
            expected = 1.0 / self.rendimiento
            if abs(self.cantidad - expected) > CANTIDAD_RENDIMIENTO_TOLERANCE:
                warnings.warn(f"Discrepancia Rendimiento/Cantidad en {self.codigo_apu}", UserWarning, stacklevel=4)

@dataclass(frozen=False)
class Equipo(InsumoProcesado):
    EXPECTED_UNITS: ClassVar[Set[str]] = UNIDADES_TIEMPO
    REQUIRES_RENDIMIENTO: ClassVar[bool] = False

@dataclass(frozen=False)
class Suministro(InsumoProcesado):
    EXPECTED_UNITS: ClassVar[Set[str]] = UNIDADES_MASA | UNIDADES_VOLUMEN | UNIDADES_AREA | UNIDADES_LONGITUD | UNIDADES_GENERICAS
    REQUIRES_RENDIMIENTO: ClassVar[bool] = False

    def _post_validation_hook(self):
        super()._post_validation_hook()
        if self.cantidad == 0:
             warnings.warn(f"Suministro cantidad=0 en {self.codigo_apu}", UserWarning, stacklevel=4)

@dataclass(frozen=False)
class Transporte(InsumoProcesado):
    EXPECTED_UNITS: ClassVar[Set[str]] = UNIDADES_TRANSPORTE
    REQUIRES_RENDIMIENTO: ClassVar[bool] = False

@dataclass(frozen=False)
class Otro(InsumoProcesado):
    EXPECTED_UNITS: ClassVar[Set[str]] = set()
    REQUIRES_RENDIMIENTO: ClassVar[bool] = False


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

def validate_insumo_data(insumo_data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(insumo_data, dict):
        raise ValidationError("insumo_data debe ser dict")

    required = ["codigo_apu", "descripcion_apu", "unidad_apu", "descripcion_insumo", "unidad_insumo", "tipo_insumo"]
    missing = [f for f in required if f not in insumo_data or insumo_data[f] is None]
    if missing:
        raise ValidationError(f"Campos faltantes: {missing}")

    defaults = {
        "cantidad": 0.0, "precio_unitario": 0.0, "valor_total": 0.0,
        "capitulo": "GENERAL", "categoria": "OTRO", "formato_origen": "GENERIC",
        "rendimiento": 0.0, "jornal": 0.0,
    }

    cleaned = {}
    for k, v in insumo_data.items():
        if v is None:
            if k in defaults: cleaned[k] = defaults[k]
            continue

        if k in ["cantidad", "precio_unitario", "valor_total", "rendimiento", "jornal"]:
            try:
                val = float(v)
                if val < 0: raise ValidationError(f"{k} negativo")
                cleaned[k] = val
            except (ValueError, TypeError):
                raise ValidationError(f"{k} debe ser numérico")
        else:
            cleaned[k] = v

    for k, v in defaults.items():
        if k not in cleaned: cleaned[k] = v

    try:
        tipo = TipoInsumo.from_string(cleaned["tipo_insumo"])
        cleaned["tipo_insumo"] = tipo.value
        cleaned["categoria"] = tipo.value
    except InvalidTipoInsumoError as e:
        raise ValidationError(str(e))

    if cleaned.get("valor_total", 0) == 0:
        cleaned["valor_total"] = cleaned.get("cantidad", 0) * cleaned.get("precio_unitario", 0)

    return cleaned

def create_insumo(**kwargs) -> InsumoProcesado:
    try:
        if "tipo_insumo" not in kwargs: raise ValidationError("Falta tipo_insumo")
        tipo = TipoInsumo.from_string(kwargs["tipo_insumo"]).value
        kwargs["tipo_insumo"] = tipo
        kwargs["categoria"] = tipo

        cls = INSUMO_CLASS_MAP.get(tipo)
        if not cls: raise InvalidTipoInsumoError(f"Clase no hallada para {tipo}")

        import inspect
        sig = inspect.signature(cls)
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        # Check required args that don't have defaults
        for name, param in sig.parameters.items():
            if param.default is inspect.Parameter.empty and name not in valid_kwargs:
                 raise ValidationError(f"Falta argumento obligatorio '{name}' para {cls.__name__}")

        return cls(**valid_kwargs)

    except (InvalidTipoInsumoError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"Error creando insumo: {e}", exc_info=True)
        raise InsumoDataError(f"Error fatal creando insumo: {e}")

def create_insumo_from_raw(raw_data: Dict[str, Any]) -> InsumoProcesado:
    return create_insumo(**validate_insumo_data(raw_data))

# ============================================================================
# FUNCIONES DE UTILIDAD (Legacy Compatibility)
# ============================================================================

def get_tipo_insumo_class(tipo: Union[str, TipoInsumo]) -> Type[InsumoProcesado]:
    tipo_enum = TipoInsumo.from_string(tipo)
    insumo_class = INSUMO_CLASS_MAP.get(tipo_enum.value)
    if not insumo_class:
        raise InvalidTipoInsumoError(f"No hay clase definida para tipo: {tipo_enum.value}")
    return insumo_class

def get_all_tipo_insumo_values() -> Set[str]:
    return TipoInsumo.get_valid_values()

def is_valid_tipo_insumo(tipo: str) -> bool:
    try:
        TipoInsumo.from_string(tipo)
        return True
    except InvalidTipoInsumoError:
        return False

# Exports
__all__ = [
    "SchemaError", "ValidationError", "InvalidTipoInsumoError", "InsumoDataError",
    "TipoInsumo", "Stratum",
    "InsumoProcesado", "APUStructure",
    "ManoDeObra", "Equipo", "Suministro", "Transporte", "Otro",
    "create_insumo", "create_insumo_from_raw", "validate_insumo_data",
    "normalize_unit", "normalize_description", "normalize_codigo",
    "NumericValidator", "StringValidator",
    "UNIDADES_TIEMPO", "UNIDADES_MASA",
    "get_tipo_insumo_class", "get_all_tipo_insumo_values", "is_valid_tipo_insumo"
]
