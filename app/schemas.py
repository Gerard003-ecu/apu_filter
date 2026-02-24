"""
Constitución de los Datos — Esquema Canónico del Sistema de Presupuesto.

Define las estructuras de datos fuertemente tipadas que representan los átomos
del presupuesto (Insumos, APUs). Implementa el patrón Factory para garantizar
que ningún objeto se instancie sin pasar por validaciones de invariantes de
dominio matemáticas y topológicas.

Invariantes y Reglas de Dominio
-------------------------------
1. **Tipología Estricta**: Jerarquía cerrada de clases (ManoDeObra, Equipo,
   Suministro, Transporte, Otro) modela con precisión la naturaleza ontológica
   de cada recurso.

2. **Normalización Canónica**: Unidades y descripciones se proyectan a un
   espacio normalizado mediante mapeos deterministas idempotentes.

3. **Validación Reactiva** (``__post_init__``): Cada objeto se autovalida al
   nacer, garantizando consistencia interna inmediata sin estados intermedios
   inválidos.

4. **Estabilidad Topológica**: Métricas de entropía de Shannon y diversidad
   categórica cuantifican la robustez estructural de un APU, detectando
   configuraciones degeneradas (mono-dependencia, pirámides invertidas).
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import math
import re
import unicodedata
import warnings
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from enum import Enum, IntEnum
from functools import lru_cache
from typing import (
    Any,
    ClassVar,
    Dict,
    Final,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)


# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================

# --- Tolerancias numéricas ---
VALOR_TOTAL_WARNING_TOLERANCE: Final[float] = 0.01   # 1 % → advertencia
VALOR_TOTAL_ERROR_TOLERANCE: Final[float] = 0.05     # 5 % → error crítico
CANTIDAD_RENDIMIENTO_TOLERANCE: Final[float] = 0.0001

# --- Límites físicos ---
MIN_CANTIDAD: Final[float] = 0.0
MAX_CANTIDAD: Final[float] = 1_000_000.0
MIN_PRECIO: Final[float] = 0.0
MAX_PRECIO: Final[float] = 1_000_000_000.0
MIN_RENDIMIENTO: Final[float] = 0.0
MAX_RENDIMIENTO: Final[float] = 1_000.0

# --- Longitudes de cadenas ---
MAX_CODIGO_LENGTH: Final[int] = 50
MAX_DESCRIPCION_LENGTH: Final[int] = 500

# --- Unidades válidas por categoría dimensional ---
UNIDADES_TIEMPO: Final[FrozenSet[str]] = frozenset(
    {"HORA", "DIA", "SEMANA", "MES", "JOR"}
)
UNIDADES_MASA: Final[FrozenSet[str]] = frozenset(
    {"KG", "GR", "TON", "LB"}
)
UNIDADES_VOLUMEN: Final[FrozenSet[str]] = frozenset(
    {"M3", "L", "ML", "GAL"}
)
UNIDADES_AREA: Final[FrozenSet[str]] = frozenset({"M2", "CM2"})
UNIDADES_LONGITUD: Final[FrozenSet[str]] = frozenset(
    {"M", "KM", "CM", "MM"}
)
UNIDADES_TRANSPORTE: Final[FrozenSet[str]] = frozenset(
    {"KM", "VIAJE", "VIAJES", "MILLA", "TON-KM"}
)
UNIDADES_GENERICAS: Final[FrozenSet[str]] = frozenset(
    {"UNIDAD", "UND", "U", "PAR", "JUEGO", "KIT", "%"}
)

# Unión de todas las unidades para validación global
_ALL_KNOWN_UNITS: Final[FrozenSet[str]] = (
    UNIDADES_TIEMPO
    | UNIDADES_MASA
    | UNIDADES_VOLUMEN
    | UNIDADES_AREA
    | UNIDADES_LONGITUD
    | UNIDADES_TRANSPORTE
    | UNIDADES_GENERICAS
)

# --- Mapeo de normalización de unidades ---
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
    "m3": "M3", "m\u00b3": "M3", "metro cubico": "M3",
    "litro": "L", "litros": "L", "l": "L", "lt": "L",
    "galon": "GAL", "galones": "GAL", "gal": "GAL",
    # Área
    "m2": "M2", "m\u00b2": "M2", "metro cuadrado": "M2",
    "cm2": "CM2", "cm\u00b2": "CM2",
    # Longitud
    "metro": "M", "metros": "M", "m": "M", "mts": "M",
    "kilometro": "KM", "km": "KM",
    "centimetro": "CM", "cm": "CM",
    "milimetro": "MM", "mm": "MM",
    # Transporte
    "viaje": "VIAJE", "viajes": "VIAJES", "vje": "VIAJE",
    "ton-km": "TON-KM", "ton km": "TON-KM",
    # Genéricas
    "unidad": "UNIDAD", "unidades": "UNIDAD", "und": "UNIDAD",
    "u": "UNIDAD", "un": "UNIDAD",
    "par": "PAR", "juego": "JUEGO", "kit": "KIT",
}

FORMATOS_VALIDOS: Final[FrozenSet[str]] = frozenset(
    {"FORMATO_A", "FORMATO_B", "FORMATO_C", "GENERIC"}
)

# Número de tipos de insumo existentes (para normalizar diversidad)
_NUM_TIPOS_INSUMO: Final[int] = 5  # Se actualiza automáticamente abajo


# ============================================================================
# ENUMERACIONES
# ============================================================================

class Stratum(IntEnum):
    """
    Niveles de abstracción en la topología piramidal del negocio.

    La jerarquía refleja una cadena de mando descendente::

        WISDOM (0)  →  Decisión / Agente
        STRATEGY (1) →  Finanzas / Capítulos
        TACTICS (2)  →  APUs / Estructura
        PHYSICS (3)  →  Insumos / Logística
    """

    WISDOM = 0
    STRATEGY = 1
    TACTICS = 2
    PHYSICS = 3


class TipoInsumo(str, Enum):
    """
    Tipos válidos de insumos en un APU.

    La enumeración hereda de ``str`` para permitir comparaciones directas
    con cadenas y serialización JSON transparente.
    """

    MANO_DE_OBRA = "MANO_DE_OBRA"
    EQUIPO = "EQUIPO"
    SUMINISTRO = "SUMINISTRO"
    TRANSPORTE = "TRANSPORTE"
    OTRO = "OTRO"

    # Caché de nivel de clase (thread-safe para lecturas en CPython por el GIL)
    _cache: ClassVar[Dict[str, TipoInsumo]] = {}

    @classmethod
    def from_string(cls, value: Union[str, TipoInsumo]) -> TipoInsumo:
        """
        Convierte una cadena a ``TipoInsumo`` con caché interna.

        Parameters
        ----------
        value : str | TipoInsumo
            Representación textual del tipo (e.g. ``"Mano de Obra"``).

        Returns
        -------
        TipoInsumo

        Raises
        ------
        InvalidTipoInsumoError
            Si *value* no se puede mapear a ningún miembro.
        """
        if isinstance(value, cls):
            return value

        if not isinstance(value, str):
            raise InvalidTipoInsumoError(
                f"Tipo debe ser str, recibido: {type(value).__name__}"
            )

        normalized = value.strip().upper().replace(" ", "_").replace("-", "_")

        cached = cls._cache.get(normalized)
        if cached is not None:
            return cached

        try:
            result = cls(normalized)
        except ValueError:
            raise InvalidTipoInsumoError(
                f"Tipo de insumo inválido: '{value}'. "
                f"Valores válidos: {cls.valid_values()}"
            ) from None

        cls._cache[normalized] = result
        return result

    @classmethod
    def valid_values(cls) -> FrozenSet[str]:
        """Retorna el conjunto inmutable de valores válidos."""
        return frozenset(member.value for member in cls)


# Actualizar constante de diversidad ahora que la enumeración existe
_NUM_TIPOS_INSUMO: Final[int] = len(TipoInsumo)  # type: ignore[misc]


# ============================================================================
# EXCEPCIONES
# ============================================================================

class SchemaError(Exception):
    """Clase raíz para errores del esquema de datos."""


class ValidationError(SchemaError):
    """Datos que violan un invariante de dominio."""


class InvalidTipoInsumoError(ValidationError):
    """El tipo de insumo proporcionado no pertenece a la enumeración."""


class InsumoDataError(ValidationError):
    """Error en los datos constitutivos de un insumo."""


class UnitNormalizationError(SchemaError):
    """Fallo irrecuperable durante la normalización de unidades."""


# ============================================================================
# FUNCIONES DE NORMALIZACIÓN (puras e idempotentes)
# ============================================================================

@lru_cache(maxsize=512)
def normalize_unit(unit: Optional[str]) -> str:
    """
    Proyecta una unidad de medida a su forma canónica.

    La función es **idempotente**: ``normalize_unit(normalize_unit(x)) == normalize_unit(x)``.

    Parameters
    ----------
    unit : str | None
        Unidad cruda (e.g. ``"Mts"``, ``"kgs"``).

    Returns
    -------
    str
        Forma canónica (e.g. ``"M"``, ``"KG"``) o ``"UNIDAD"`` por defecto.
    """
    if not isinstance(unit, str) or not unit.strip():
        return "UNIDAD"

    key = unit.strip().lower()
    return UNIDAD_NORMALIZADA_MAP.get(key, unit.strip().upper())


@lru_cache(maxsize=1024)
def normalize_description(desc: Optional[str]) -> str:
    """
    Normaliza descripciones: elimina diacríticos, caracteres no permitidos
    y trunca a ``MAX_DESCRIPCION_LENGTH``.

    Parameters
    ----------
    desc : str | None

    Returns
    -------
    str
        Cadena ASCII en mayúsculas, limpia y acotada.
    """
    if not isinstance(desc, str) or not desc:
        return ""

    nfkd = unicodedata.normalize("NFD", desc)
    ascii_only = nfkd.encode("ASCII", "ignore").decode("ASCII")
    cleaned = re.sub(r"[^\w\s\-./()=]", "", ascii_only)
    collapsed = re.sub(r"\s+", " ", cleaned.strip())
    return collapsed.upper()[:MAX_DESCRIPCION_LENGTH]


@lru_cache(maxsize=256)
def normalize_codigo(codigo: Optional[str]) -> str:
    """
    Normaliza un código APU: elimina caracteres inválidos y valida longitud.

    Parameters
    ----------
    codigo : str | None

    Returns
    -------
    str
        Código normalizado en mayúsculas.

    Raises
    ------
    ValidationError
        Si el código es nulo, vacío tras limpieza, o excede la longitud máxima.
    """
    if not codigo or not isinstance(codigo, str):
        raise ValidationError("Código APU no puede estar vacío")

    clean = re.sub(r"[^\w\-.]", "", codigo.strip().upper())

    if not clean:
        raise ValidationError(
            f"Código APU vacío tras normalización (original: '{codigo}')"
        )

    if len(clean) > MAX_CODIGO_LENGTH:
        raise ValidationError(
            f"Código APU excede límite: {len(clean)} > {MAX_CODIGO_LENGTH}"
        )

    return clean


def _deterministic_short_hash(text: str, length: int = 8) -> str:
    """
    Genera un hash corto determinista basado en SHA-256.

    A diferencia de ``hash()`` de Python, el resultado es reproducible
    entre ejecuciones y plataformas.

    Parameters
    ----------
    text : str
        Texto de entrada.
    length : int
        Longitud del hash hexadecimal resultante (máx. 64).

    Returns
    -------
    str
        Subcadena hexadecimal de ``length`` caracteres.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:length]


# ============================================================================
# VALIDADORES REUTILIZABLES
# ============================================================================

class NumericValidator:
    """
    Validador numérico con soporte para tolerancias relativas y absolutas.

    Todos los métodos son estáticos y sin estado para facilitar la composición
    y el testing.
    """

    @staticmethod
    def validate_non_negative(
        value: Union[int, float, Decimal],
        field_name: str,
        min_value: float = 0.0,
        max_value: Optional[float] = None,
    ) -> float:
        """
        Valida que *value* sea numérico, finito y esté en ``[min_value, max_value]``.

        Parameters
        ----------
        value : int | float | Decimal
        field_name : str
            Nombre del campo (para mensajes de error).
        min_value : float
            Cota inferior inclusiva.
        max_value : float | None
            Cota superior inclusiva (``None`` = sin cota).

        Returns
        -------
        float
            Valor convertido a ``float``.

        Raises
        ------
        ValidationError
        """
        if not isinstance(value, (int, float, Decimal)):
            raise ValidationError(
                f"{field_name} debe ser numérico, recibido: {type(value).__name__}"
            )

        try:
            fval = float(value)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValidationError(
                f"No se puede convertir {field_name} a float: {exc}"
            ) from exc

        if not math.isfinite(fval):
            raise ValidationError(
                f"{field_name} debe ser finito, recibido: {fval}"
            )

        if fval < min_value:
            raise ValidationError(
                f"{field_name}={fval} < mínimo permitido={min_value}"
            )

        if max_value is not None and fval > max_value:
            raise ValidationError(
                f"{field_name}={fval} > máximo permitido={max_value}"
            )

        return fval

    @staticmethod
    def relative_difference(
        actual: float, expected: float, epsilon: float = 1e-10
    ) -> float:
        r"""
        Diferencia relativa simétrica:

        .. math::
            \Delta_{\text{rel}} = \frac{|a - b|}{\max(|a|,\;|b|,\;\varepsilon)}

        El denominador usa ``epsilon`` como piso para evitar división por cero
        cuando ambos operandos son cercanos a cero.
        """
        if actual == expected:
            return 0.0
        denom = max(abs(actual), abs(expected), epsilon)
        return abs(actual - expected) / denom

    @staticmethod
    def values_consistent(
        actual: float,
        expected: float,
        rel_tolerance: float = 0.01,
        abs_tolerance: float = 0.01,
    ) -> Tuple[bool, float]:
        """
        Consistencia híbrida: pasa si la diferencia absoluta ≤ ``abs_tolerance``
        **o** la diferencia relativa ≤ ``rel_tolerance``.

        Returns
        -------
        tuple[bool, float]
            ``(es_consistente, diferencia_relativa)``
        """
        abs_diff = abs(actual - expected)

        if abs_diff <= abs_tolerance:
            return True, 0.0

        rel_diff = NumericValidator.relative_difference(actual, expected)
        return rel_diff <= rel_tolerance, rel_diff


class StringValidator:
    """Validador de cadenas de texto."""

    @staticmethod
    def validate_non_empty(
        value: Any,
        field_name: str,
        max_length: Optional[int] = None,
        *,
        strip: bool = True,
    ) -> str:
        """
        Garantiza que *value* sea ``str``, no vacío y dentro de longitud.

        Parameters
        ----------
        value : Any
        field_name : str
        max_length : int | None
        strip : bool
            Aplicar ``str.strip()`` antes de validar.

        Returns
        -------
        str

        Raises
        ------
        ValidationError
        """
        if not isinstance(value, str):
            raise ValidationError(
                f"{field_name} debe ser str, recibido: {type(value).__name__}"
            )

        cleaned = value.strip() if strip else value

        if not cleaned:
            raise ValidationError(f"{field_name} no puede estar vacío")

        if max_length is not None and len(cleaned) > max_length:
            raise ValidationError(
                f"{field_name} excede longitud máxima: {len(cleaned)} > {max_length}"
            )

        return cleaned


# ============================================================================
# NODO TOPOLÓGICO BASE
# ============================================================================

@dataclass
class TopologicalNode:
    """
    Nodo base en el grafo topológico del proyecto.

    Define las propiedades comunes de cualquier entidad jerárquica.
    El campo ``structural_health`` ∈ [0, 1] resume la calidad estructural
    del nodo.
    """

    id: str = field(default="")
    stratum: Stratum = field(default=Stratum.PHYSICS)
    description: str = field(default="")
    structural_health: float = field(default=1.0)


# ============================================================================
# INSUMO PROCESADO (Nodo hoja — Nivel 3: Física)
# ============================================================================

@dataclass
class InsumoProcesado(TopologicalNode):
    """
    Estructura base para cualquier recurso atómico de un APU.

    Representa un nodo hoja (sin hijos) en el grafo de dependencias.
    La validación reactiva en ``__post_init__`` garantiza que el objeto
    **nunca** exista en un estado inválido.
    """

    # --- Campos obligatorios ---
    codigo_apu: str = ""
    descripcion_apu: str = ""
    unidad_apu: str = ""
    descripcion_insumo: str = ""
    unidad_insumo: str = ""
    cantidad: float = 0.0
    precio_unitario: float = 0.0
    valor_total: float = 0.0
    tipo_insumo: str = ""

    # --- Campos con valor por defecto ---
    capitulo: str = field(default="GENERAL")
    categoria: str = field(default="OTRO")
    formato_origen: str = field(default="GENERIC")
    rendimiento: float = field(default=0.0)
    normalized_desc: str = field(default="", init=False)

    # --- Estado interno (no serializable) ---
    _validated: bool = field(default=False, init=False, repr=False, compare=False)

    # --- Invariantes de clase (sobreescritos en subclases) ---
    EXPECTED_UNITS: ClassVar[FrozenSet[str]] = UNIDADES_GENERICAS
    REQUIRES_RENDIMIENTO: ClassVar[bool] = False

    def __post_init__(self) -> None:
        """Validación completa de invariantes de dominio al nacer."""
        try:
            self.stratum = Stratum.PHYSICS
            self.description = self.descripcion_insumo

            self._normalize_fields()
            self._validate_required_fields()
            self._validate_numeric_fields()
            self._validate_consistency()

            # ID determinista basado en contenido
            content_key = f"{self.codigo_apu}|{self.descripcion_insumo}"
            self.id = f"{self.codigo_apu}_{_deterministic_short_hash(content_key)}"

            self._post_validation_hook()
            self._validated = True

        except Exception:
            logger.error(
                "Error inicializando %s (codigo_apu=%r)",
                self.__class__.__name__,
                getattr(self, "codigo_apu", "?"),
                exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def total_cost(self) -> float:
        r"""
        Costo calculado: :math:`C = q \times p`.
        """
        return self.cantidad * self.precio_unitario

    # ------------------------------------------------------------------
    # Normalización
    # ------------------------------------------------------------------

    def _normalize_fields(self) -> None:
        """Proyecta todos los campos de texto a su forma canónica."""
        self.codigo_apu = normalize_codigo(self.codigo_apu)
        self.descripcion_apu = normalize_description(self.descripcion_apu)
        self.descripcion_insumo = normalize_description(self.descripcion_insumo)
        self.normalized_desc = self.descripcion_insumo
        self.unidad_apu = normalize_unit(self.unidad_apu)
        self.unidad_insumo = normalize_unit(self.unidad_insumo)

        tipo_enum = TipoInsumo.from_string(self.tipo_insumo)
        self.tipo_insumo = tipo_enum.value
        self.categoria = tipo_enum.value

        fmt = str(self.formato_origen).strip().upper()
        self.formato_origen = fmt if fmt in FORMATOS_VALIDOS else "GENERIC"

    # ------------------------------------------------------------------
    # Validaciones
    # ------------------------------------------------------------------

    def _validate_required_fields(self) -> None:
        """Valida presencia y longitud de campos string obligatorios."""
        StringValidator.validate_non_empty(
            self.codigo_apu, "codigo_apu", MAX_CODIGO_LENGTH
        )
        StringValidator.validate_non_empty(
            self.descripcion_insumo, "descripcion_insumo", MAX_DESCRIPCION_LENGTH
        )

    def _validate_numeric_fields(self) -> None:
        """Valida rangos de todos los campos numéricos."""
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

    def _validate_consistency(self) -> None:
        """Valida coherencia lógica y matemática entre campos."""
        self._validate_valor_total_consistency()
        self._validate_unit_category()

        if self.REQUIRES_RENDIMIENTO and self.rendimiento <= 0:
            warnings.warn(
                f"Rendimiento debería ser > 0 en {self.codigo_apu}",
                UserWarning,
                stacklevel=5,
            )

    def _validate_valor_total_consistency(self) -> None:
        r"""
        Verifica: :math:`\text{valor\_total} \approx \text{cantidad} \times \text{precio\_unitario}`.

        Aplica tolerancia híbrida (absoluta + relativa) en dos niveles:
        advertencia (1 %) y error (5 %).
        """
        expected = self.cantidad * self.precio_unitario

        # Caso trivial: ambos son cero
        if expected == 0.0 and self.valor_total == 0.0:
            return

        # Caso imposible: producto es cero pero valor_total no
        if expected == 0.0 and self.valor_total > 0.0:
            raise ValidationError(
                f"{self.__class__.__name__} [{self.codigo_apu}]: "
                f"valor_total={self.valor_total:.2f} pero cantidad×precio=0"
            )

        # Nivel ERROR (5 %)
        ok_error, rel_diff = NumericValidator.values_consistent(
            self.valor_total,
            expected,
            rel_tolerance=VALOR_TOTAL_ERROR_TOLERANCE,
            abs_tolerance=0.01,
        )
        if not ok_error:
            raise ValidationError(
                f"{self.__class__.__name__} [{self.codigo_apu}]: "
                f"Inconsistencia grave: valor_total={self.valor_total:.2f} vs "
                f"calculado={expected:.2f} (Δ={rel_diff:.2%})"
            )

        # Nivel WARNING (1 %)
        ok_warn, rel_diff = NumericValidator.values_consistent(
            self.valor_total,
            expected,
            rel_tolerance=VALOR_TOTAL_WARNING_TOLERANCE,
            abs_tolerance=0.01,
        )
        if not ok_warn:
            warnings.warn(
                f"{self.__class__.__name__} [{self.codigo_apu}]: "
                f"Divergencia leve: valor_total={self.valor_total:.2f} vs "
                f"calculado={expected:.2f} (Δ={rel_diff:.2%})",
                UserWarning,
                stacklevel=6,
            )

    def _validate_unit_category(self) -> None:
        """
        Verifica que la unidad del insumo pertenezca a las categorías
        esperadas por la subclase.

        Emite advertencia (no error) para no bloquear formatos desconocidos.
        """
        if not self.EXPECTED_UNITS:
            return  # Subclase sin restricción (e.g., Otro)

        if self.unidad_insumo not in self.EXPECTED_UNITS:
            warnings.warn(
                f"{self.__class__.__name__} [{self.codigo_apu}]: "
                f"Unidad '{self.unidad_insumo}' no está en las unidades esperadas "
                f"{self.EXPECTED_UNITS}",
                UserWarning,
                stacklevel=5,
            )

    def _post_validation_hook(self) -> None:
        """Hook para validaciones específicas de subclases."""

    # ------------------------------------------------------------------
    # Serialización
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario, excluyendo campos internos con prefijo ``_``."""
        data = asdict(self)
        return {k: v for k, v in data.items() if not k.startswith("_")}

    @property
    def is_valid(self) -> bool:
        """Estado de validación del objeto."""
        return self._validated


# ============================================================================
# ESTRUCTURA APU (Nodo interno — Nivel 2: Táctica)
# ============================================================================

@dataclass
class APUStructure(TopologicalNode):
    """
    Actividad constructiva en el Nivel 2 (Táctica).

    Agrega recursos (``InsumoProcesado``) y calcula métricas de
    estabilidad topológica basadas en entropía de Shannon y diversidad
    categórica.
    """

    unit: str = ""
    quantity: float = 0.0
    resources: List[InsumoProcesado] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.stratum = Stratum.TACTICS

    # ------------------------------------------------------------------
    # Propiedades estructurales
    # ------------------------------------------------------------------

    @property
    def support_base_width(self) -> int:
        """Número de recursos (ancho de la base de soporte)."""
        return len(self.resources)

    @property
    def is_inverted_pyramid(self) -> bool:
        """
        Detecta configuración degenerada: cantidad táctica grande
        soportada por un único recurso.
        """
        return self.quantity > 1000 and self.support_base_width == 1

    @property
    def total_cost(self) -> float:
        """Costo total del APU como suma de valores de sus recursos."""
        return sum(r.valor_total for r in self.resources)

    # ------------------------------------------------------------------
    # Métricas topológicas
    # ------------------------------------------------------------------

    def topological_stability_index(self) -> float:
        r"""
        Índice de Estabilidad Topológica :math:`\Psi \in [0, 1]`.

        Combina dos componentes complementarias:

        1. **Diversidad categórica** (*cualitativa*):

           .. math::
               D = \frac{|\{\text{tipos únicos}\}|}{|\text{TipoInsumo}|}

           Penaliza mono-dependencia de un solo tipo de recurso.  Se normaliza
           contra el número total de tipos existentes en la enumeración (5),
           no un valor arbitrario.

        2. **Entropía de Shannon normalizada** (*cuantitativa*):

           .. math::
               H_{\text{norm}} = \frac{-\sum_{i} p_i \ln p_i}{\ln n}

           donde :math:`p_i = v_i / \sum v_j` es la fracción de costo del
           recurso *i* y *n* es el número de recursos.

        La ponderación final sigue una media geométrica ponderada para
        que ambos factores sean *necesarios* (no solo *suficientes*):

        .. math::
            \Psi = D^{0.4} \cdot H_{\text{norm}}^{0.6}

        Esto evita que un APU con alta entropía pero un solo tipo de
        insumo obtenga un índice artificialmente alto.

        Returns
        -------
        float
            Índice en [0, 1].  Mayor = más estable.
        """
        if not self.resources:
            return 0.0

        n = len(self.resources)
        num_tipos_existentes = len(TipoInsumo)

        # --- Diversidad ---
        tipos_unicos = {r.tipo_insumo for r in self.resources}
        diversidad = len(tipos_unicos) / num_tipos_existentes

        if n == 1:
            # Un solo recurso → entropía = 0, retornamos solo diversidad escalada
            return diversidad ** 0.4 * 0.0  # 0^0.6 = 0 → resultado 0

        # --- Entropía de Shannon ---
        valores = [max(r.valor_total, 0.0) for r in self.resources]
        total_valor = sum(valores)

        if total_valor <= 0:
            return 0.0

        entropia = 0.0
        for v in valores:
            if v > 0:
                p = v / total_valor
                entropia -= p * math.log(p)

        h_max = math.log(n)
        h_norm = (entropia / h_max) if h_max > 0 else 0.0
        h_norm = max(0.0, min(h_norm, 1.0))

        # Media geométrica ponderada: ambos factores deben contribuir
        return (diversidad ** 0.4) * (h_norm ** 0.6)

    # ------------------------------------------------------------------
    # Mutación controlada
    # ------------------------------------------------------------------

    def add_resource(self, resource: InsumoProcesado) -> None:
        """
        Agrega un recurso validando el invariante de estrato.

        Parameters
        ----------
        resource : InsumoProcesado
            Debe tener ``stratum == Stratum.PHYSICS``.

        Raises
        ------
        TypeError
            Si *resource* no es ``InsumoProcesado`` o tiene estrato incorrecto.
        """
        if not isinstance(resource, InsumoProcesado):
            raise TypeError(
                f"Se esperaba InsumoProcesado, recibido: {type(resource).__name__}"
            )

        if resource.stratum != Stratum.PHYSICS:
            raise TypeError(
                f"Solo nodos PHYSICS se admiten en APUStructure, "
                f"recibido: {resource.stratum.name}"
            )

        self.resources.append(resource)

    # ------------------------------------------------------------------
    # Consultas
    # ------------------------------------------------------------------

    def get_cost_breakdown(self) -> Dict[str, float]:
        """Distribución de costos agrupada por tipo de insumo."""
        breakdown: Dict[str, float] = {}
        for r in self.resources:
            breakdown[r.tipo_insumo] = breakdown.get(r.tipo_insumo, 0.0) + r.valor_total
        return breakdown

    def get_cost_fractions(self) -> Dict[str, float]:
        """
        Fracción porcentual de cada tipo de insumo respecto al costo total.

        Returns
        -------
        dict[str, float]
            Valores en [0, 1] que suman ≈ 1.0.
        """
        breakdown = self.get_cost_breakdown()
        total = sum(breakdown.values())
        if total <= 0:
            return {k: 0.0 for k in breakdown}
        return {k: v / total for k, v in breakdown.items()}


# ============================================================================
# SUBCLASES ESPECIALIZADAS
# ============================================================================

@dataclass
class ManoDeObra(InsumoProcesado):
    """
    Recurso de tipo Mano de Obra.

    Valida que :math:`\text{cantidad} \approx 1 / \text{rendimiento}` cuando
    el rendimiento es positivo.
    """

    EXPECTED_UNITS: ClassVar[FrozenSet[str]] = UNIDADES_TIEMPO
    REQUIRES_RENDIMIENTO: ClassVar[bool] = True

    jornal: float = field(default=0.0)

    def _post_validation_hook(self) -> None:
        super()._post_validation_hook()

        if self.jornal != 0.0:
            self.jornal = NumericValidator.validate_non_negative(
                self.jornal, "jornal", MIN_PRECIO, MAX_PRECIO
            )

        if self.rendimiento > 0:
            expected_qty = 1.0 / self.rendimiento
            ok, rel_diff = NumericValidator.values_consistent(
                self.cantidad,
                expected_qty,
                rel_tolerance=0.05,
                abs_tolerance=CANTIDAD_RENDIMIENTO_TOLERANCE,
            )
            if not ok:
                warnings.warn(
                    f"Discrepancia Rendimiento/Cantidad en {self.codigo_apu}: "
                    f"cantidad={self.cantidad:.4f}, esperada={expected_qty:.4f} "
                    f"(rendimiento={self.rendimiento}, Δ={rel_diff:.2%})",
                    UserWarning,
                    stacklevel=5,
                )


@dataclass
class Equipo(InsumoProcesado):
    """Recurso de tipo Maquinaria y Equipo."""

    EXPECTED_UNITS: ClassVar[FrozenSet[str]] = UNIDADES_TIEMPO
    REQUIRES_RENDIMIENTO: ClassVar[bool] = False


@dataclass
class Suministro(InsumoProcesado):
    """Recurso de tipo Material / Suministro."""

    EXPECTED_UNITS: ClassVar[FrozenSet[str]] = (
        UNIDADES_MASA
        | UNIDADES_VOLUMEN
        | UNIDADES_AREA
        | UNIDADES_LONGITUD
        | UNIDADES_GENERICAS
    )
    REQUIRES_RENDIMIENTO: ClassVar[bool] = False

    def _post_validation_hook(self) -> None:
        super()._post_validation_hook()
        if self.cantidad == 0:
            warnings.warn(
                f"Suministro con cantidad=0 en {self.codigo_apu}",
                UserWarning,
                stacklevel=5,
            )


@dataclass
class Transporte(InsumoProcesado):
    """Recurso de tipo Transporte (Fletes)."""

    EXPECTED_UNITS: ClassVar[FrozenSet[str]] = UNIDADES_TRANSPORTE
    REQUIRES_RENDIMIENTO: ClassVar[bool] = False


@dataclass
class Otro(InsumoProcesado):
    """Recurso genérico u otros costos indirectos."""

    EXPECTED_UNITS: ClassVar[FrozenSet[str]] = frozenset()
    REQUIRES_RENDIMIENTO: ClassVar[bool] = False


# --- Registro de clases por tipo ---
INSUMO_CLASS_MAP: Final[Dict[str, Type[InsumoProcesado]]] = {
    TipoInsumo.MANO_DE_OBRA.value: ManoDeObra,
    TipoInsumo.EQUIPO.value: Equipo,
    TipoInsumo.SUMINISTRO.value: Suministro,
    TipoInsumo.TRANSPORTE.value: Transporte,
    TipoInsumo.OTRO.value: Otro,
}


# ============================================================================
# CACHÉ DE FIRMAS DE CONSTRUCTOR
# ============================================================================

@lru_cache(maxsize=32)
def _get_class_params(cls: Type[InsumoProcesado]) -> FrozenSet[str]:
    """
    Obtiene los nombres de parámetros válidos del constructor de *cls*.

    Usa ``lru_cache`` para evitar introspección repetida y elimina la
    necesidad de un diccionario mutable global.
    """
    sig = inspect.signature(cls)
    return frozenset(sig.parameters.keys())


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

# Campos obligatorios para validación de datos crudos
_REQUIRED_RAW_FIELDS: Final[FrozenSet[str]] = frozenset({
    "codigo_apu",
    "descripcion_apu",
    "unidad_apu",
    "descripcion_insumo",
    "unidad_insumo",
    "tipo_insumo",
})

# Campos numéricos con sus valores por defecto
_NUMERIC_FIELDS: Final[Dict[str, float]] = {
    "cantidad": 0.0,
    "precio_unitario": 0.0,
    "valor_total": 0.0,
    "rendimiento": 0.0,
    "jornal": 0.0,
}

# Valores por defecto para campos no numéricos
_STRING_DEFAULTS: Final[Dict[str, str]] = {
    "capitulo": "GENERAL",
    "categoria": "OTRO",
    "formato_origen": "GENERIC",
}


def validate_insumo_data(insumo_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida, limpia y normaliza un diccionario de datos crudos.

    Pipeline:
      1. Verificar tipo del contenedor.
      2. Verificar campos obligatorios.
      3. Convertir y validar campos numéricos.
      4. Aplicar valores por defecto.
      5. Normalizar ``tipo_insumo``.
      6. Calcular ``valor_total`` si está ausente.

    Parameters
    ----------
    insumo_data : dict[str, Any]
        Datos crudos del insumo.

    Returns
    -------
    dict[str, Any]
        Diccionario limpio y tipado, listo para el constructor.

    Raises
    ------
    ValidationError
        Si faltan campos obligatorios o hay valores inválidos.
    """
    if not isinstance(insumo_data, dict):
        raise ValidationError(
            f"insumo_data debe ser dict, recibido: {type(insumo_data).__name__}"
        )

    # 1. Campos obligatorios
    missing = [
        f for f in _REQUIRED_RAW_FIELDS
        if f not in insumo_data or insumo_data[f] is None
    ]
    if missing:
        raise ValidationError(f"Campos obligatorios faltantes: {sorted(missing)}")

    cleaned: Dict[str, Any] = {}

    # 2. Procesar cada campo
    for key, value in insumo_data.items():
        if value is None:
            # Aplicar default si existe
            if key in _NUMERIC_FIELDS:
                cleaned[key] = _NUMERIC_FIELDS[key]
            elif key in _STRING_DEFAULTS:
                cleaned[key] = _STRING_DEFAULTS[key]
            continue

        if key in _NUMERIC_FIELDS:
            try:
                fval = float(value)
            except (ValueError, TypeError) as exc:
                raise ValidationError(
                    f"Campo '{key}' debe ser numérico: {exc}"
                ) from exc

            if not math.isfinite(fval):
                raise ValidationError(f"Campo '{key}' debe ser finito: {fval}")
            if fval < 0:
                raise ValidationError(
                    f"Campo '{key}' no puede ser negativo: {fval}"
                )
            cleaned[key] = fval
        else:
            cleaned[key] = value

    # 3. Defaults para campos ausentes
    for key, default in {**_NUMERIC_FIELDS, **_STRING_DEFAULTS}.items():
        cleaned.setdefault(key, default)

    # 4. Normalizar tipo
    try:
        tipo = TipoInsumo.from_string(cleaned["tipo_insumo"])
        cleaned["tipo_insumo"] = tipo.value
        cleaned["categoria"] = tipo.value
    except InvalidTipoInsumoError as exc:
        raise ValidationError(str(exc)) from exc

    # 5. Calcular valor_total si no fue proporcionado
    if cleaned.get("valor_total", 0.0) == 0.0:
        cleaned["valor_total"] = (
            cleaned.get("cantidad", 0.0) * cleaned.get("precio_unitario", 0.0)
        )

    return cleaned


def create_insumo(**kwargs: Any) -> InsumoProcesado:
    """
    Factory principal: instancia la subclase correcta de ``InsumoProcesado``.

    Parameters
    ----------
    **kwargs
        Deben incluir ``tipo_insumo``.

    Returns
    -------
    InsumoProcesado
        Instancia validada de la subclase correspondiente.

    Raises
    ------
    ValidationError
        Si falta ``tipo_insumo`` o la validación interna falla.
    InsumoDataError
        Para errores fatales inesperados.
    """
    if "tipo_insumo" not in kwargs:
        raise ValidationError("Falta campo obligatorio: tipo_insumo")

    tipo_enum = TipoInsumo.from_string(kwargs["tipo_insumo"])
    tipo_value = tipo_enum.value

    kwargs["tipo_insumo"] = tipo_value
    kwargs["categoria"] = tipo_value

    insumo_class = INSUMO_CLASS_MAP.get(tipo_value)
    if insumo_class is None:
        raise InvalidTipoInsumoError(
            f"Clase no registrada para tipo: {tipo_value}"
        )

    valid_params = _get_class_params(insumo_class)
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}

    try:
        return insumo_class(**filtered)
    except ValidationError:
        raise
    except TypeError as exc:
        raise ValidationError(
            f"Error instanciando {insumo_class.__name__}: {exc}"
        ) from exc
    except Exception as exc:
        logger.error(
            "Error creando insumo tipo %s: %s", tipo_value, exc, exc_info=True
        )
        raise InsumoDataError(f"Error fatal creando insumo: {exc}") from exc


def create_insumo_from_raw(raw_data: Dict[str, Any]) -> InsumoProcesado:
    """
    Crea un insumo desde datos crudos (pipeline completo).

    Pipeline::

        raw_data → validate_insumo_data() → create_insumo() → __post_init__()

    Parameters
    ----------
    raw_data : dict[str, Any]
        Datos crudos provenientes de un parser o API.

    Returns
    -------
    InsumoProcesado
    """
    validated = validate_insumo_data(raw_data)
    return create_insumo(**validated)


# ============================================================================
# FUNCIONES DE UTILIDAD (Compatibilidad)
# ============================================================================

def get_tipo_insumo_class(tipo: Union[str, TipoInsumo]) -> Type[InsumoProcesado]:
    """Resuelve la clase correspondiente a un tipo de insumo."""
    tipo_enum = TipoInsumo.from_string(tipo)
    cls = INSUMO_CLASS_MAP.get(tipo_enum.value)
    if cls is None:
        raise InvalidTipoInsumoError(
            f"No hay clase definida para tipo: {tipo_enum.value}"
        )
    return cls


def get_all_tipo_insumo_values() -> FrozenSet[str]:
    """Retorna todos los valores válidos de ``TipoInsumo``."""
    return TipoInsumo.valid_values()


def is_valid_tipo_insumo(tipo: str) -> bool:
    """Verifica si *tipo* es un ``TipoInsumo`` válido sin lanzar excepción."""
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
    "SchemaError",
    "ValidationError",
    "InvalidTipoInsumoError",
    "InsumoDataError",
    "UnitNormalizationError",
    # Enumeraciones
    "TipoInsumo",
    "Stratum",
    # Estructuras de datos
    "TopologicalNode",
    "InsumoProcesado",
    "APUStructure",
    "ManoDeObra",
    "Equipo",
    "Suministro",
    "Transporte",
    "Otro",
    # Factories
    "create_insumo",
    "create_insumo_from_raw",
    "validate_insumo_data",
    # Normalización
    "normalize_unit",
    "normalize_description",
    "normalize_codigo",
    # Validadores
    "NumericValidator",
    "StringValidator",
    # Constantes
    "UNIDADES_TIEMPO",
    "UNIDADES_MASA",
    "UNIDADES_VOLUMEN",
    "UNIDADES_AREA",
    "UNIDADES_LONGITUD",
    "UNIDADES_TRANSPORTE",
    "UNIDADES_GENERICAS",
    "INSUMO_CLASS_MAP",
    # Utilidades
    "get_tipo_insumo_class",
    "get_all_tipo_insumo_values",
    "is_valid_tipo_insumo",
]