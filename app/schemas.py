"""
Este módulo define la "Constitución de los Datos" del sistema. Establece las estructuras de datos
inmutables y fuertemente tipadas que representan los átomos del presupuesto (Insumos, APUs).
Implementa el patrón Factory para garantizar que ningún objeto se instancie sin pasar por
validaciones de invariantes de dominio matemáticas y topológicas.

Invariantes y Reglas de Dominio:
--------------------------------
1. Tipología Estricta (InsumoProcesado):
   Define la jerarquía de clases (ManoDeObra, Equipo, Suministro) para el modelado preciso.

2. Normalización Canónica:
   Asegura que todas las unidades y descripciones se conviertan a un lenguaje común mediante
   mapeos deterministas.

3. Validación Reactiva (__post_init__):
   Los objetos se autovalidan al nacer, garantizando consistencia interna inmediata.

4. Estabilidad Topológica:
   Mide la entropía y diversidad de los componentes de un APU para determinar su robustez
   estructural utilizando teoría de la información.
"""

from __future__ import annotations

import math
import logging
import math
import re
import unicodedata
import warnings
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from enum import Enum, IntEnum
from functools import lru_cache
from typing import Any, ClassVar, Dict, Final, List, Optional, Protocol, Set, Tuple, Type, Union

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

# Límites de valores físicos
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
    """
    Niveles de abstracción en la topología piramidal del negocio.
    Representa la jerarquía de profundidad desde la estrategia hasta la logística.
    """
    ROOT = 0      # Raíz del proyecto
    STRATEGY = 1  # Capítulos principales
    TACTIC = 2    # APUs (Análisis de Precios Unitarios)
    LOGISTICS = 3 # Insumos (Recursos básicos)


_TIPO_INSUMO_CACHE: Dict[str, 'TipoInsumo'] = {}

class TipoInsumo(str, Enum):
    """
    Tipos válidos de insumos en un APU.
    Define la naturaleza ontológica del recurso.
    """
    MANO_DE_OBRA = "MANO_DE_OBRA"
    EQUIPO = "EQUIPO"
    SUMINISTRO = "SUMINISTRO"
    TRANSPORTE = "TRANSPORTE"
    OTRO = "OTRO"

    @classmethod
    def from_string(cls, value: str) -> TipoInsumo:
        """
        Convierte string a TipoInsumo utilizando una caché para evitar re-procesamiento costoso.

        Args:
            value (str): Cadena representando el tipo de insumo (ej. "Mano de Obra").

        Returns:
            TipoInsumo: Miembro de la enumeración correspondiente.

        Raises:
            InvalidTipoInsumoError: Si el valor no coincide con ningún tipo conocido.
        """
        if isinstance(value, cls):
            return value

        if not isinstance(value, str):
            raise InvalidTipoInsumoError(f"Tipo debe ser string, recibido: {type(value).__name__}")

        normalized = value.strip().upper().replace(" ", "_").replace("-", "_")

        if normalized in _TIPO_INSUMO_CACHE:
            return _TIPO_INSUMO_CACHE[normalized]

        try:
            result = cls(normalized)
            _TIPO_INSUMO_CACHE[normalized] = result
            return result
        except ValueError:
            raise InvalidTipoInsumoError(
                f"Tipo de insumo inválido: '{value}'. "
                f"Valores válidos: {cls.get_valid_values()}"
            )

    @classmethod
    def get_valid_values(cls) -> Set[str]:
        return {member.value for member in cls}


# ============================================================================
# EXCEPCIONES CUSTOM
# ============================================================================

class SchemaError(Exception):
    """Clase base para errores de esquema."""
    pass

class ValidationError(SchemaError):
    """Error de validación de datos."""
    pass

class InvalidTipoInsumoError(ValidationError):
    """El tipo de insumo no es válido."""
    pass

class InsumoDataError(ValidationError):
    """Error en los datos constitutivos del insumo."""
    pass

class UnitNormalizationError(SchemaError):
    """Error durante la normalización de unidades."""
    pass


# ============================================================================
# FUNCIONES DE NORMALIZACIÓN
# ============================================================================

@lru_cache(maxsize=512)
def normalize_unit(unit: Optional[str]) -> str:
    """
    Normaliza una unidad de medida a su forma canónica.
    Usa una caché case-insensitive para optimizar búsquedas repetitivas.

    Args:
        unit (str | None): Unidad de entrada (ej. "Mts", "kgs").

    Returns:
        str: Unidad normalizada (ej. "M", "KG") o "UNIDAD" por defecto.
    """
    if not isinstance(unit, str) or not unit.strip():
        return "UNIDAD"

    normalized_key = unit.strip().lower()

    if normalized_key in UNIDAD_NORMALIZADA_MAP:
        return UNIDAD_NORMALIZADA_MAP[normalized_key]

    return unit.strip().upper()


@lru_cache(maxsize=1024)
def normalize_description(desc: Optional[str]) -> str:
    """
    Normaliza descripciones eliminando acentos y caracteres especiales no permitidos.

    Args:
        desc (str | None): Descripción original.

    Returns:
        str: Descripción normalizada en ASCII mayúsculas.
    """
    if not isinstance(desc, str) or not desc:
        return ""
    desc = unicodedata.normalize("NFD", desc)
    desc = desc.encode("ASCII", "ignore").decode("ASCII")
    desc = re.sub(r"[^\w\s\-./()=]", "", desc)
    desc = re.sub(r"\s+", " ", desc.strip())
    return desc.upper()


@lru_cache(maxsize=256)
def normalize_codigo(codigo: Optional[str]) -> str:
    """
    Normaliza un código APU eliminando caracteres inválidos y validando longitud.

    Orden de operaciones:
    1. Validar existencia.
    2. Limpiar caracteres no alfanuméricos (excepto . y -).
    3. Validar longitud resultante.

    Args:
        codigo (str | None): Código original.

    Returns:
        str: Código normalizado.

    Raises:
        ValidationError: Si el código es inválido o vacío tras limpieza.
    """
    if not codigo or not isinstance(codigo, str):
        raise ValidationError("Código APU no puede estar vacío")

    codigo_clean = re.sub(r"[^\w\-.]", "", codigo.strip().upper())

    if not codigo_clean:
        raise ValidationError("Código APU vacío tras normalización")

    if len(codigo_clean) > MAX_CODIGO_LENGTH:
        raise ValidationError(
            f"Código APU excede límite: {len(codigo_clean)} > {MAX_CODIGO_LENGTH}"
        )

    return codigo_clean


# ============================================================================
# VALIDADORES REUTILIZABLES
# ============================================================================

class NumericValidator:
    """Validador numérico con soporte para tolerancias relativas y absolutas y chequeo de finitud."""

    @staticmethod
    def validate_non_negative(
        value: Union[int, float, Decimal],
        field_name: str,
        min_value: float = 0.0,
        max_value: Optional[float] = None
    ) -> float:
        """
        Valida que un valor sea numérico, finito y no negativo dentro de un rango.

        Args:
            value: Valor a validar.
            field_name: Nombre del campo para el mensaje de error.
            min_value: Límite inferior (inclusivo).
            max_value: Límite superior (opcional, inclusivo).

        Returns:
            float: Valor validado convertido a float.

        Raises:
            ValidationError: Si el valor no cumple las condiciones.
        """
        if not isinstance(value, (int, float, Decimal)):
            raise ValidationError(f"{field_name} debe ser numérico, recibido: {type(value).__name__}")

        try:
            float_value = float(value)
        except (ValueError, TypeError, OverflowError) as e:
            raise ValidationError(f"No se puede convertir {field_name} a float: {e}")

        if not math.isfinite(float_value):
            raise ValidationError(f"{field_name} debe ser finito, recibido: {float_value}")

        if float_value < min_value:
            raise ValidationError(f"{field_name}={float_value} < mínimo={min_value}")

        if max_value is not None and float_value > max_value:
            raise ValidationError(f"{field_name}={float_value} > máximo={max_value}")

        return float_value

    @staticmethod
    def relative_difference(actual: float, expected: float, epsilon: float = 1e-10) -> float:
        """
        Calcula la diferencia relativa simétrica entre dos valores.

        Fórmula:
        $$ \Delta_{rel} = \frac{|a - b|}{\max(|a|, |b|, \epsilon)} $$

        Usa el máximo de los valores absolutos como denominador para asegurar simetría
        y estabilidad numérica cerca de cero.
        """
        if actual == expected:
            return 0.0

        denominator = max(abs(actual), abs(expected), epsilon)
        return abs(actual - expected) / denominator

    @staticmethod
    def values_consistent(
        actual: float,
        expected: float,
        rel_tolerance: float = 0.01,
        abs_tolerance: float = 0.01
    ) -> Tuple[bool, float]:
        """
        Verifica consistencia entre valores usando una tolerancia híbrida (relativa + absoluta).

        Args:
            actual: Valor observado.
            expected: Valor esperado.
            rel_tolerance: Tolerancia relativa (porcentaje).
            abs_tolerance: Tolerancia absoluta (magnitud fija).

        Returns:
            Tuple[bool, float]: (Es consistente, Diferencia relativa calculada)
        """
        abs_diff = abs(actual - expected)

        if abs_diff <= abs_tolerance:
            return True, 0.0

        rel_diff = NumericValidator.relative_difference(actual, expected)
        return rel_diff <= rel_tolerance, rel_diff


class StringValidator:
    """Validador de cadenas de texto con capacidades de normalización."""

    @staticmethod
    def validate_non_empty(
        value: str,
        field_name: str,
        max_length: Optional[int] = None,
        normalize: bool = True
    ) -> str:
        """
        Valida que un string no sea vacío ni nulo.

        Args:
            value: Valor string.
            field_name: Nombre del campo.
            max_length: Longitud máxima permitida.
            normalize: Si se debe aplicar strip().

        Returns:
            str: String validado.
        """
        if not isinstance(value, str):
            raise ValidationError(
                f"{field_name} debe ser string, recibido: {type(value).__name__}"
            )

        cleaned = value.strip() if normalize else value

        if not cleaned:
            raise ValidationError(f"{field_name} no puede estar vacío")

        if max_length is not None and len(cleaned) > max_length:
            raise ValidationError(
                f"{field_name} excede longitud máxima: {len(cleaned)} > {max_length}"
            )

        return cleaned


# ============================================================================
# CLASE BASE INMUTABLE
# ============================================================================

@dataclass(kw_only=True)
class TopologicalNode:
    """
    Nodo base para el grafo topológico del proyecto.
    Define las propiedades comunes de cualquier entidad en la jerarquía.
    """
    id: str = field(default="")
    stratum: Stratum = field(default=Stratum.LOGISTICS)
    description: str = field(default="")
    structural_health: float = 1.0
    is_floating: bool = False

    def validate_connectivity(self):
        """Valida las reglas de conectividad topológica."""
        if self.stratum == Stratum.LOGISTICS and hasattr(self, "children") and self.children:
            raise ValueError(f"Violación de Invariante: Nodo Logístico ({self.id}) con hijos.")


@dataclass(frozen=False)
class InsumoProcesado(TopologicalNode):
    """
    Estructura base para cualquier insumo de APU.
    Representa un nodo hoja en el grafo de dependencias (Nivel 3 - Logística).
    Implementa validación reactiva durante la inicialización.
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
        """Inicialización con validación completa de invariantes de dominio."""
        try:
            self.stratum = Stratum.LOGISTICS
            self.description = self.descripcion_insumo

            self._normalize_all_fields()
            self._validate_required_fields()
            self._validate_numeric_fields()
            self._validate_consistency()

            # Generar ID único basado en contenido para trazabilidad
            self.id = f"{self.codigo_apu}_{hash(self.descripcion_insumo) % 10000:04d}"

            self._post_validation_hook()
            self._validated = True

        except Exception as e:
            logger.error(f"Error inicializando {self.__class__.__name__}: {e}")
            raise

    @property
    def total_cost(self) -> float:
        """
        Costo total calculado del insumo.
        $$ Costo = Cantidad \times Precio Unitario $$
        """
        return self.cantidad * self.precio_unitario

    def _normalize_all_fields(self):
        """Normaliza todos los campos de texto a su forma canónica."""
        self.codigo_apu = normalize_codigo(self.codigo_apu)
        self.descripcion_apu = normalize_description(self.descripcion_apu)
        self.descripcion_insumo = normalize_description(self.descripcion_insumo)
        self.normalized_desc = self.descripcion_insumo
        self.unidad_apu = normalize_unit(self.unidad_apu)
        self.unidad_insumo = normalize_unit(self.unidad_insumo)

        tipo_enum = TipoInsumo.from_string(self.tipo_insumo)
        self.tipo_insumo = tipo_enum.value
        self.categoria = tipo_enum.value

        formato_upper = str(self.formato_origen).strip().upper()
        self.formato_origen = formato_upper if formato_upper in FORMATOS_VALIDOS else "GENERIC"

    def _validate_required_fields(self):
        """Valida la presencia y longitud de campos string obligatorios."""
        StringValidator.validate_non_empty(
            self.codigo_apu, "codigo_apu", MAX_CODIGO_LENGTH
        )
        StringValidator.validate_non_empty(
            self.descripcion_insumo, "descripcion_insumo", MAX_DESCRIPCION_LENGTH
        )

    def _validate_numeric_fields(self):
        """Valida rangos y tipos de campos numéricos."""
        self.cantidad = NumericValidator.validate_non_negative(
            self.cantidad, "cantidad", MIN_CANTIDAD, MAX_CANTIDAD
        )
        self.precio_unitario = NumericValidator.validate_non_negative(
            self.precio_unitario, "precio_unitario", MIN_PRECIO, MAX_PRECIO
        )
        self.valor_total = NumericValidator.validate_non_negative(
            self.valor_total, "valor_total", MIN_PRECIO, MAX_PRECIO
        )
        self.rendimiento = NumericValidator.validate_non_negative(
            self.rendimiento, "rendimiento", MIN_RENDIMIENTO, MAX_RENDIMIENTO
        )

        if self.precio_unitario < 0:
            raise ValidationError(f"Precio negativo no permitido: {self.precio_unitario}")

    def _validate_consistency(self):
        """Valida consistencia lógica y matemática entre campos."""
        self._validate_valor_total_consistency()

        if self.REQUIRES_RENDIMIENTO and self.rendimiento <= 0:
            warnings.warn(
                f"Rendimiento debería ser > 0 en {self.codigo_apu}",
                UserWarning,
                stacklevel=4
            )

    def _validate_valor_total_consistency(self):
        """
        Valida la coherencia algebraica: $$ ValorTotal \approx Cantidad \times PrecioUnitario $$

        Utiliza una tolerancia híbrida (relativa + absoluta) para manejar correctamente
        tanto valores pequeños (ruido numérico) como grandes.
        """
        expected_total = self.cantidad * self.precio_unitario

        if expected_total == 0.0 and self.valor_total == 0.0:
            return

        if expected_total == 0.0 and self.valor_total > 0.0:
            raise ValidationError(
                f"{self.__class__.__name__} [{self.codigo_apu}]: "
                f"valor_total={self.valor_total:.2f} pero cantidad×precio=0"
            )

        is_consistent, rel_diff = NumericValidator.values_consistent(
            self.valor_total,
            expected_total,
            rel_tolerance=VALOR_TOTAL_ERROR_TOLERANCE,
            abs_tolerance=0.01
        )

        if not is_consistent:
            raise ValidationError(
                f"{self.__class__.__name__} [{self.codigo_apu}]: "
                f"Inconsistencia grave: valor_total={self.valor_total:.2f} vs "
                f"calculado={expected_total:.2f} (Δ={rel_diff:.2%})"
            )

        is_warning_level, rel_diff = NumericValidator.values_consistent(
            self.valor_total,
            expected_total,
            rel_tolerance=VALOR_TOTAL_WARNING_TOLERANCE,
            abs_tolerance=0.01
        )

        if not is_warning_level:
            warnings.warn(
                f"{self.__class__.__name__} [{self.codigo_apu}]: "
                f"Divergencia: valor_total={self.valor_total:.2f} vs "
                f"calculado={expected_total:.2f}",
                UserWarning,
                stacklevel=4
            )

    def _post_validation_hook(self):
        """Hook para validaciones específicas de subclases."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Serializa el objeto a diccionario, excluyendo campos internos."""
        data = asdict(self)
        data.pop("_validated", None)
        return data

    def validate(self) -> bool:
        """Retorna el estado de validación del objeto."""
        return self._validated


@dataclass
class APUStructure(TopologicalNode):
    """
    Representa una actividad constructiva en el Nivel 2 (Táctica).
    Agrega recursos (insumos) y calcula métricas de estabilidad topológica.
    """
    unit: str = ""
    quantity: float = 0.0
    resources: List[InsumoProcesado] = field(default_factory=list)

    def __post_init__(self):
        self.stratum = Stratum.TACTIC

    @property
    def support_base_width(self) -> int:
        """Número de recursos que componen el APU (ancho de la base de soporte)."""
        return len(self.resources)

    @property
    def is_inverted_pyramid(self) -> bool:
        """
        Detecta si el APU es una 'Pirámide Invertida'.
        Esto ocurre cuando hay una gran cantidad táctica soportada por un único recurso logístico.
        """
        return self.quantity > 1000 and self.support_base_width == 1

    def topological_stability_index(self) -> float:
        """
        Calcula el Índice de Estabilidad Topológica ($\Psi$) en el intervalo [0.0, 1.0].

        Combina dos métricas fundamentales:
        1. **Diversidad de tipos (Cualitativa):** Penaliza la mono-dependencia de un solo tipo de recurso.
        2. **Entropía de Shannon (Cuantitativa):** Mide la distribución de costos entre los recursos.

        Fórmula de Entropía Normalizada:
        $$ H_{norm} = \\frac{-\\sum p_i \cdot \ln(p_i)}{\ln(n)} $$
        donde $p_i$ es la fracción del costo total del recurso $i$, y $n$ es el número de recursos.

        Returns:
            float: Índice de estabilidad (Mayor es más estable).
        """
        if not self.resources:
            return 0.0

        n = len(self.resources)

        tipos_unicos = {r.tipo_insumo for r in self.resources}
        num_tipos = len(tipos_unicos)
        diversidad_score = min(num_tipos / 3.0, 1.0)

        if n == 1:
            # Si solo hay un recurso, la entropía es 0, retornamos solo la diversidad ponderada.
            return diversidad_score * 0.4

        valores = [max(r.valor_total, 0.0) for r in self.resources]
        total_valor = sum(valores)

        if total_valor <= 0:
            return diversidad_score * 0.4

        probabilidades = [v / total_valor for v in valores]

        entropia = 0.0
        for p in probabilidades:
            if p > 0:
                entropia -= p * math.log(p)

        # Normalizar H con H_max = ln(n)
        entropia_maxima = math.log(n)
        entropia_normalizada = entropia / entropia_maxima if entropia_maxima > 0 else 0.0
        entropia_normalizada = min(max(entropia_normalizada, 0.0), 1.0)

        # Ponderación: 40% Diversidad, 60% Entropía de Costos
        return (diversidad_score * 0.4) + (entropia_normalizada * 0.6)

    def add_resource(self, resource: InsumoProcesado) -> None:
        """
        Agrega un recurso a la estructura, validando el invariante de estrato.
        Solo se permiten nodos de nivel LOGISTICS (Insumos) en nodos TACTIC (APUs).
        """
        if not isinstance(resource, InsumoProcesado):
            raise TypeError(f"Se esperaba InsumoProcesado, recibido: {type(resource).__name__}")

        if resource.stratum != Stratum.LOGISTICS:
            raise TypeError(
                f"Solo se admiten nodos LOGISTICS en APUStructure, "
                f"recibido: {resource.stratum.name}"
            )

        self.resources.append(resource)

    def get_cost_breakdown(self) -> Dict[str, float]:
        """Retorna la distribución de costos agrupada por tipo de insumo."""
        breakdown: Dict[str, float] = {}
        for resource in self.resources:
            tipo = resource.tipo_insumo
            breakdown[tipo] = breakdown.get(tipo, 0.0) + resource.valor_total
        return breakdown


# ============================================================================
# CLASES ESPECÍFICAS
# ============================================================================

@dataclass(frozen=False)
class ManoDeObra(InsumoProcesado):
    """
    Insumo de tipo Mano de Obra.
    Requiere validación estricta de rendimiento (1/Rendimiento ≈ Cantidad).
    """
    EXPECTED_UNITS: ClassVar[Set[str]] = UNIDADES_TIEMPO
    REQUIRES_RENDIMIENTO: ClassVar[bool] = True
    jornal: float = field(default=0.0)

    def _post_validation_hook(self):
        super()._post_validation_hook()

        if self.jornal != 0.0:
            self.jornal = NumericValidator.validate_non_negative(
                self.jornal, "jornal", MIN_PRECIO, MAX_PRECIO
            )

        if self.rendimiento > 0:
            expected_cantidad = 1.0 / self.rendimiento

            is_consistent, rel_diff = NumericValidator.values_consistent(
                self.cantidad,
                expected_cantidad,
                rel_tolerance=0.05,
                abs_tolerance=CANTIDAD_RENDIMIENTO_TOLERANCE
            )

            if not is_consistent:
                warnings.warn(
                    f"Discrepancia Rendimiento/Cantidad en {self.codigo_apu}: "
                    f"cantidad={self.cantidad:.4f}, esperada={expected_cantidad:.4f} "
                    f"(rendimiento={self.rendimiento})",
                    UserWarning,
                    stacklevel=4
                )

@dataclass(frozen=False)
class Equipo(InsumoProcesado):
    """Insumo de tipo Maquinaria y Equipo."""
    EXPECTED_UNITS: ClassVar[Set[str]] = UNIDADES_TIEMPO
    REQUIRES_RENDIMIENTO: ClassVar[bool] = False

@dataclass(frozen=False)
class Suministro(InsumoProcesado):
    """Insumo de tipo Material/Suministro."""
    EXPECTED_UNITS: ClassVar[Set[str]] = UNIDADES_MASA | UNIDADES_VOLUMEN | UNIDADES_AREA | UNIDADES_LONGITUD | UNIDADES_GENERICAS
    REQUIRES_RENDIMIENTO: ClassVar[bool] = False

    def _post_validation_hook(self):
        super()._post_validation_hook()
        if self.cantidad == 0:
             warnings.warn(f"Suministro cantidad=0 en {self.codigo_apu}", UserWarning, stacklevel=4)

@dataclass(frozen=False)
class Transporte(InsumoProcesado):
    """Insumo de tipo Transporte (Fletes)."""
    EXPECTED_UNITS: ClassVar[Set[str]] = UNIDADES_TRANSPORTE
    REQUIRES_RENDIMIENTO: ClassVar[bool] = False

@dataclass(frozen=False)
class Otro(InsumoProcesado):
    """Insumo de tipo genérico u otros costos indirectos."""
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

_INSUMO_SIGNATURES: Dict[str, Set[str]] = {}

def _get_class_params(cls: Type[InsumoProcesado]) -> Set[str]:
    """Obtiene los parámetros válidos del constructor de una clase, con caché."""
    cls_name = cls.__name__
    if cls_name not in _INSUMO_SIGNATURES:
        import inspect
        sig = inspect.signature(cls)
        _INSUMO_SIGNATURES[cls_name] = set(sig.parameters.keys())
    return _INSUMO_SIGNATURES[cls_name]


def validate_insumo_data(insumo_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida, limpia y normaliza un diccionario de datos crudos de insumo.
    Aplica valores por defecto y convierte tipos.

    Args:
        insumo_data: Diccionario con datos del insumo (raw).

    Returns:
        Dict[str, Any]: Datos validados y tipados listos para el constructor.

    Raises:
        ValidationError: Si faltan campos obligatorios o hay tipos inválidos.
    """
    if not isinstance(insumo_data, dict):
        raise ValidationError(f"insumo_data debe ser dict, recibido: {type(insumo_data).__name__}")

    required_fields = frozenset({
        "codigo_apu", "descripcion_apu", "unidad_apu",
        "descripcion_insumo", "unidad_insumo", "tipo_insumo"
    })

    missing = [f for f in required_fields if f not in insumo_data or insumo_data[f] is None]
    if missing:
        raise ValidationError(f"Campos obligatorios faltantes: {missing}")

    defaults = {
        "cantidad": 0.0,
        "precio_unitario": 0.0,
        "valor_total": 0.0,
        "capitulo": "GENERAL",
        "categoria": "OTRO",
        "formato_origen": "GENERIC",
        "rendimiento": 0.0,
        "jornal": 0.0,
    }

    numeric_fields = frozenset({"cantidad", "precio_unitario", "valor_total", "rendimiento", "jornal"})

    cleaned: Dict[str, Any] = {}

    for key, value in insumo_data.items():
        if value is None:
            if key in defaults:
                cleaned[key] = defaults[key]
            continue

        if key in numeric_fields:
            try:
                float_val = float(value)
                if float_val < 0:
                    raise ValidationError(f"Campo '{key}' no puede ser negativo: {float_val}")
                if not math.isfinite(float_val):
                    raise ValidationError(f"Campo '{key}' debe ser finito: {float_val}")
                cleaned[key] = float_val
            except (ValueError, TypeError) as e:
                raise ValidationError(f"Campo '{key}' debe ser numérico: {e}")
        else:
            cleaned[key] = value

    for key, default_val in defaults.items():
        if key not in cleaned:
            cleaned[key] = default_val

    try:
        tipo = TipoInsumo.from_string(cleaned["tipo_insumo"])
        cleaned["tipo_insumo"] = tipo.value
        cleaned["categoria"] = tipo.value
    except InvalidTipoInsumoError as e:
        raise ValidationError(str(e))

    if cleaned.get("valor_total", 0.0) == 0.0:
        cleaned["valor_total"] = cleaned.get("cantidad", 0.0) * cleaned.get("precio_unitario", 0.0)

    return cleaned


def create_insumo(**kwargs) -> InsumoProcesado:
    """
    Factory function principal para instanciar Insumos.
    Determina la clase correcta basada en 'tipo_insumo' y filtra los argumentos.

    Args:
        **kwargs: Argumentos para el insumo (deben incluir 'tipo_insumo').

    Returns:
        InsumoProcesado: Instancia de la subclase correcta (ManoDeObra, Equipo, etc.).

    Raises:
        ValidationError: Si falta 'tipo_insumo' o hay error de validación interna.
        InsumoDataError: Para errores fatales inesperados.
    """
    if "tipo_insumo" not in kwargs:
        raise ValidationError("Falta campo obligatorio: tipo_insumo")

    try:
        tipo_enum = TipoInsumo.from_string(kwargs["tipo_insumo"])
        tipo_value = tipo_enum.value
    except InvalidTipoInsumoError:
        raise

    kwargs["tipo_insumo"] = tipo_value
    kwargs["categoria"] = tipo_value

    insumo_class = INSUMO_CLASS_MAP.get(tipo_value)
    if insumo_class is None:
        raise InvalidTipoInsumoError(f"Clase no registrada para tipo: {tipo_value}")

    valid_params = _get_class_params(insumo_class)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    try:
        return insumo_class(**filtered_kwargs)
    except TypeError as e:
        raise ValidationError(f"Error instanciando {insumo_class.__name__}: {e}")
    except Exception as e:
        logger.error(f"Error creando insumo tipo {tipo_value}: {e}", exc_info=True)
        raise InsumoDataError(f"Error fatal creando insumo: {e}")

def create_insumo_from_raw(raw_data: Dict[str, Any]) -> InsumoProcesado:
    """
    Crea una instancia de insumo directamente desde datos crudos.

    Pipeline de construcción:
    1. `validate_insumo_data`: Limpieza y tipado básico.
    2. `create_insumo`: Selección de clase Factory y construcción del objeto.
    3. `__post_init__` (interno): Validación de invariantes de dominio.

    Args:
        raw_data: Diccionario de datos crudos.

    Returns:
        InsumoProcesado: Instancia validada.
    """
    validated = validate_insumo_data(raw_data)
    return create_insumo(**validated)

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
