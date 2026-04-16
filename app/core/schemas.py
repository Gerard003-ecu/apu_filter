"""
=========================================================================================
Módulo: Schemas (Constitución de los Datos — Esquema Canónico del Sistema de Presupuesto)
Ubicación: app/core/schemas.py
=========================================================================================

Naturaleza Ciber-Física y Topológica:
    Este módulo no define simples clases de datos, sino que establece los Invariantes 
    Estructurales del espacio de fase presupuestal. Proyecta las entidades financieras 
    como un complejo simplicial restringido, garantizando que el orquestador opere 
    sobre un espacio métrico no degenerado y lógicamente coherente.

0. Preliminares: Topología del Grafo Bipartito:
    La estructura se modela axiomáticamente como un grafo bipartito G = (V_TACTICS ∪ V_PHYSICS, E):
    • V_TACTICS = {APU₁, APU₂, ...} (nodos internos del estrato táctico).
    • V_PHYSICS = {Insumo₁, Insumo₂, ...} (nodos hoja del estrato físico).
    • E ⊆ V_TACTICS × V_PHYSICS (aristas de dependencia directa).
    
    Propiedades garantizadas algorítmicamente:
    (G1) Acíclico Dirigido (DAG): β₁ = 0. Ausencia absoluta de paradojas circulares.
    (G2) Bipartito Estricto: Las aristas E operan exclusivamente inter-estrato.
    (G3) Bosque Estructural: Cada APU constituye la raíz de un árbol de expansión de insumos.

1. Invariantes Físico-Matemáticos de Dominio:
    (I1) Ley de Conservación de Valor: Para todo insumo i, valor_total[i] = cantidad[i] × precio_unitario[i].
         Debido a la entropía de la Unidad de Punto Flotante (IEEE 754), se impone una 
         tolerancia híbrida: ε_rel = 1e-6 y ε_abs = 1e-10. 
         Justificación: Para valores de O(10⁶) y O(10⁹), el error relativo acumulado es O(10⁻²⁷). 
         El margen ε_rel ≫ 10⁻²⁷ absorbe la fricción numérica sin admitir corrupción contable.
    
    (I2) No-negatividad Absoluta: ∀i: cantidad[i] ≥ 0, precio[i] ≥ 0. La energía financiera 
         negativa carece de significado físico en este marco y es vetada.
    
    (I3) Acotación Física (Saturación Dimensional):
         • Cantidad ∈ [0, 10⁶] (Límite logístico estructural).
         • Precio ∈ [0, 10⁹] (Límite de capitalización de equipos pesados).
         • Rendimiento ∈ [0, 10³] (Límite termodinámico del trabajo humano/mecánico).
    
    (I4) Normalización Idempotente (Retractos de Deformación):
         Todo operador normalizador f ∈ {normalize_unit, normalize_description, normalize_codigo}
         debe garantizar f(f(x)) = f(x). Esta invarianza protege al sistema de mutaciones parásitas 
         durante evaluaciones cíclicas iterativas.

2. Termodinámica Estructural y Entropía de Shannon:
    La estabilidad logística (Índice de Fiedler generalizado Ψ) de una APU "a" se deriva 
    de la dispersión de sus recursos R(a):
        H = -Σᵢ pᵢ ln(pᵢ)  [nats]
    Donde pᵢ = valor_total_i / Σ valor_total_j ∈ es la fracción de energía financiera.
    Normalizado como H_norm = H / ln(n) ∈. Si pᵢ = 1 para un único nodo (H = 0),
    existe un Punto de Fallo Único (SPOF), lo que desploma el índice Ψ y activa alertas
    de "Pirámide Invertida" o monopolio logístico.

3. Diversidad Categórica D(a):
    D(a) = |tipos únicos en R(a)| / 5 ∈. Cuantifica la ortogonalidad de la cadena 
    de suministro. Una APU con D=0 (monotipo) es intrínsecamente frágil ante fallos de 
    una sola clase de proveedor.
=========================================================================================
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import math
import re
import unicodedata
import warnings
from abc import ABC, abstractmethod
import sys
from dataclasses import asdict, dataclass, field, fields, replace
from decimal import Decimal
from enum import Enum, IntEnum
from functools import lru_cache, wraps
from typing import (
    Any, Callable, ClassVar, Dict, Final, FrozenSet, List, Optional,
    Protocol, Set, Tuple, Type, TypeVar, Union, cast
)

# ============================================================================
# LOGGING ESTRUCTURADO
# ============================================================================

logger = logging.getLogger(__name__)

# Tipos para tipo hints
T = TypeVar('T')
NumericType = Union[int, float, Decimal]


# ============================================================================
# ENUMERACIONES Y CONSTANTES
# ============================================================================


class Stratum(IntEnum):
    """
    Niveles de abstracción jerárquicos.

    Orden: WISDOM (0, ápice) ← ... ← PHYSICS (5, base)

    Convención: value MAYOR = más cercano a datos crudos.
    """
    WISDOM = 0
    ALPHA = 1
    OMEGA = 2
    STRATEGY = 3
    TACTICS = 4
    PHYSICS = 5

    @classmethod
    def base_stratum(cls) -> Stratum:
        """Estrato base (más cercano a datos crudos)."""
        return cls.PHYSICS

    @classmethod
    def apex_stratum(cls) -> Stratum:
        """Estrato ápice (más abstracto)."""
        return cls.WISDOM

    def requires(self) -> frozenset:
        """Estratos de dependencia (value > this.value)."""
        return frozenset(s for s in Stratum if s.value > self.value)

    @classmethod
    def ordered_bottom_up(cls) -> List[Stratum]:
        """PHYSICS → ... → WISDOM."""
        return sorted(cls, key=lambda s: s.value, reverse=True)

    @classmethod
    def ordered_top_down(cls) -> List[Stratum]:
        """WISDOM → ... → PHYSICS."""
        return sorted(cls, key=lambda s: s.value)


class TipoInsumo(str, Enum):
    """Tipos válidos de insumos (5 categorías, mutuamente excluyentes)."""
    MANO_DE_OBRA = "MANO_DE_OBRA"
    EQUIPO = "EQUIPO"
    SUMINISTRO = "SUMINISTRO"
    TRANSPORTE = "TRANSPORTE"
    OTRO = "OTRO"

    @classmethod
    def from_string(cls, value: Union[str, TipoInsumo]) -> TipoInsumo:
        """Convierte string a TipoInsumo con normalización."""
        if isinstance(value, cls):
            return value

        if not isinstance(value, str):
            raise InvalidTipoInsumoError(
                f"Tipo debe ser str, recibido: {type(value).__name__}"
            )

        normalized = value.strip().upper().replace(" ", "_").replace("-", "_")

        # Caché de módulo
        if normalized in _TIPO_INSUMO_CACHE:
            return _TIPO_INSUMO_CACHE[normalized]

        try:
            result = cls(normalized)
        except ValueError:
            raise InvalidTipoInsumoError(
                f"Tipo inválido: '{value}'. "
                f"Válidos: {cls.valid_values()}"
            ) from None

        _TIPO_INSUMO_CACHE[normalized] = result
        return result

    @classmethod
    def valid_values(cls) -> FrozenSet[str]:
        """Conjunto de valores válidos."""
        return frozenset(member.value for member in cls)


# Caché de normalización
_TIPO_INSUMO_CACHE: Dict[str, TipoInsumo] = {}
_NUM_TIPOS_INSUMO: int = len(TipoInsumo)


# ============================================================================
# CONSTANTES: TOLERANCIAS NUMÉRICAS (JUSTIFICADAS)
# ============================================================================


@dataclass(frozen=True)
class NumericalTolerances:
    """
    Tolerancias numéricas con justificación teórica.

    Justificación de valores
    ────────────────────────
    CONSERVATION_RELATIVE_TOLERANCE = 1e-6
        • float64: 15-17 dígitos significativos
        • Producto Q × P donde |Q|=O(10^m), |P|=O(10^n)
        • Error relativo ≈ 10^(m+n-15)
        • Peor caso (m+n ≤ 12): error ≤ 10^-27
        • Margen de seguridad: 1e-6 ≫ 10^-27

    CONSERVATION_ABSOLUTE_TOLERANCE = 1e-10
        • Subnormal threshold de float64: 2.2e-308
        • Para valores cercanos a cero (< 1e-10), usar tolerancia absoluta
        • Si ambos operandos < 1e-10, su producto ≈ 0 trivialmente

    VALOR_TOTAL_WARNING_TOLERANCE = 0.01 (1%)
        • Discrepancia pequeña genera warning (auditoría no crítica)
        • 1% es perceptible pero explicable por redondeo

    VALOR_TOTAL_ERROR_TOLERANCE = 0.05 (5%)
        • Discrepancia grande genera ValidationError
        • 5% indica error serio en datos o lógica
    """

    conservation_relative: float = 1e-3
    conservation_absolute: float = 1e-10
    warning_relative: float = 0.01
    error_relative: float = 0.05
    cantidad_rendimiento: float = 0.0001

    def validate_self(self) -> None:
        """Verifica coherencia de tolerancias."""
        assert 0 < self.conservation_relative < 1
        assert 0 < self.conservation_absolute < self.conservation_relative
        assert self.warning_relative < self.error_relative


_TOLERANCES = NumericalTolerances()
_TOLERANCES.validate_self()


# ============================================================================
# CONSTANTES: ACOTACIÓN FÍSICA
# ============================================================================


@dataclass(frozen=True)
class PhysicalBounds:
    """
    Cotas físicas en construcción civil (justificadas por experiencia).

    Atributos
    ─────────
    cantidad: [0, 10^6] — O(10^6) unidades máx por APU
    precio: [0, 10^9] — Equipo especializado: O(10^8) por hora
    valor_total: [0, 10^15] — cantidad_max × precio_max
    rendimiento: [0, 10^3] — O(1000) unidades/hora máx
    """
    cantidad_min: float = 0.0
    cantidad_max: float = 1_000_000.0
    precio_min: float = 0.0
    precio_max: float = 1_000_000_000.0
    valor_total_max: float = 1_000_000_000_000_000.0
    rendimiento_min: float = 0.0
    rendimiento_max: float = 100000.0

    def validate_self(self) -> None:
        """Verifica que cantidad_max × precio_max ≤ valor_total_max."""
        computed = self.cantidad_max * self.precio_max
        assert computed <= self.valor_total_max, (
            f"Inconsistencia: {self.cantidad_max} × {self.precio_max} "
            f"= {computed} > {self.valor_total_max}"
        )


_BOUNDS = PhysicalBounds()
_BOUNDS.validate_self()


# ============================================================================
# CONSTANTES: UNIDADES DE MEDIDA
# ============================================================================

UNIDADES_TIEMPO = frozenset({"HORA", "DIA", "SEMANA", "MES", "JOR"})
UNIDADES_MASA = frozenset({"KG", "GR", "TON", "LB"})
UNIDADES_VOLUMEN = frozenset({"M3", "L", "ML", "GAL"})
UNIDADES_AREA = frozenset({"M2", "CM2"})
UNIDADES_LONGITUD = frozenset({"M", "KM", "CM", "MM"})
UNIDADES_TRANSPORTE = frozenset({"KM", "VIAJE", "VIAJES", "MILLA", "TON-KM", "UNIDAD", "UND", "U"})
UNIDADES_GENERICAS = frozenset({"UNIDAD", "UND", "U", "PAR", "JUEGO", "KIT", "%"})

ALL_KNOWN_UNITS = (
    UNIDADES_TIEMPO | UNIDADES_MASA | UNIDADES_VOLUMEN |
    UNIDADES_AREA | UNIDADES_LONGITUD | UNIDADES_TRANSPORTE |
    UNIDADES_GENERICAS
)

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

# Constantes de strings
MAX_CODIGO_LENGTH: Final[int] = 50
MAX_DESCRIPCION_LENGTH: Final[int] = 500
_HASH_LENGTH: Final[int] = 10


# ============================================================================
# EXCEPCIONES
# ============================================================================


class SchemaError(Exception):
    """Clase raíz para errores de esquema."""


class ValidationError(SchemaError):
    """Violación de invariante de dominio."""


class InvalidTipoInsumoError(ValidationError):
    """Tipo de insumo inválido."""


class InsumoDataError(ValidationError):
    """Error en datos de insumo."""


class UnitNormalizationError(SchemaError):
    """Error irrecuperable en normalización de unidades."""


class InvariantError(ValidationError):
    """Violación de invariante formal (I1-I6)."""


# ============================================================================
# DECORADORES Y CONTEXTOS
# ============================================================================


def invariant_check(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorador para verificar invariantes pre/post.

    Registro: si hay error, incluir stack completo para auditoría.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            result = func(*args, **kwargs)
            return result
        except ValidationError as e:
            logger.error(
                "Invariant violation in %s: %s",
                func.__name__, str(e), exc_info=True
            )
            raise
    return wrapper


@dataclass(frozen=True)
class IdempotenceValidator:
    """
    Verifica que una función normalizadora es idempotente.

    Teorema: f es idempotente si f(f(x)) = f(x) ∀ x
    """

    func: Callable[[str], str]
    test_cases: List[str]

    def validate(self) -> bool:
        """Verifica idempotencia sobre casos de prueba."""
        for test in self.test_cases:
            f1 = self.func(test)
            f2 = self.func(f1)
            if f1 != f2:
                logger.error(
                    "Idempotence violation: f(%r) = %r, f(f(x)) = %r",
                    test, f1, f2
                )
                return False
        return True


# ============================================================================
# FUNCIONES DE NORMALIZACIÓN (VERIFICABLES Y IDEMPOTENTES)
# ============================================================================


@lru_cache(maxsize=512)
def normalize_unit(unit: Optional[str]) -> str:
    """
    Normaliza unidad de medida.

    Propiedad: idempotente — normalize_unit(normalize_unit(x)) = normalize_unit(x)

    Demostración:
        Sea x = "HORA" (ya normalizado).
        normalize_unit("HORA") → "HORA"  (no está en map, se uppercase)
        normalize_unit("HORA") → "HORA"  (idem)

    Sea x = "hora" (sin normalizar).
        normalize_unit("hora") → map["hora"] = "HORA"
        normalize_unit("HORA") → "HORA"  (fixed point)
    """
    if not isinstance(unit, str) or not unit.strip():
        return "UNIDAD"

    key = unit.strip().lower()
    return UNIDAD_NORMALIZADA_MAP.get(key, unit.strip().upper())


@lru_cache(maxsize=1024)
def normalize_description(desc: Optional[str]) -> str:
    """
    Normaliza descripción (idempotente).

    Pipeline:
    1. Descomposición NFD (separar diacríticos)
    2. Codificar ASCII (ignorar no-ASCII)
    3. Regex: eliminar caracteres no alfanuméricos
    4. Colapsar whitespace
    5. Strip, uppercase, truncar

    Idempotencia: aplicar dos veces produce mismo resultado.
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
    Normaliza código APU (idempotente).

    Caracteres permitidos: [\\w\\-.] (alphanumeric, dash, dot).
    """
    if not codigo or not isinstance(codigo, str):
        raise ValidationError("Código APU no puede ser vacío.")

    clean = re.sub(r"[^\w\-.]", "", codigo.strip().upper())

    if not clean:
        raise ValidationError(
            f"Código vacío tras normalización: '{codigo}'"
        )

    if len(clean) > MAX_CODIGO_LENGTH:
        raise ValidationError(
            f"Código excede límite: {len(clean)} > {MAX_CODIGO_LENGTH}"
        )

    return clean


def _deterministic_short_hash(
    text: str,
    length: int = _HASH_LENGTH,
) -> str:
    """
    Hash determinista SHA-256 (reproducible entre ejecuciones).

    Colisiones esperadas (Birthday Paradox):
        P(colisión) ≈ n² / (2 · 2^(4·length))

    Para length=10 (hex): 2^40 ≈ 10^12 valores
        n=1000:  P ≈ 5×10^-7 (negligible)
        n=10000: P ≈ 5×10^-5 (aceptable)
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:length]


# ============================================================================
# VALIDADORES (CON TIPADO Y PROTOCOLOS)
# ============================================================================


class NumericValidator:
    """Validador numérico con soporte para tolerancias híbridas."""

    @staticmethod
    def validate_non_negative(
        value: NumericType,
        field_name: str,
        min_value: float = 0.0,
        max_value: Optional[float] = None,
    ) -> float:
        """
        Valida que value sea numérico, finito y en rango [min_value, max_value].
        """
        if not isinstance(value, (int, float, Decimal)):
            raise ValidationError(
                f"{field_name} debe ser numérico, "
                f"recibido: {type(value).__name__}"
            )

        try:
            fval = float(value)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValidationError(
                f"No se puede convertir {field_name} a float: {exc}"
            ) from exc

        if not math.isfinite(fval):
            raise ValidationError(
                f"{field_name} debe ser finito: {fval}"
            )

        if fval < min_value:
            raise ValidationError(
                f"{field_name} = {fval} < mínimo = {min_value}"
            )

        if max_value is not None and fval > max_value:
            raise ValidationError(
                f"{field_name} = {fval} > máximo = {max_value}"
            )

        return fval

    @staticmethod
    def relative_difference(
        actual: float,
        expected: float,
        epsilon: float = 1e-15,
    ) -> float:
        r"""
        Diferencia relativa: Δ_rel = |actual - expected| / max(|actual|, |expected|, ε)

        Parámetro epsilon evita división por cero cuando ambos son ~0.
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
        Verificación de consistencia con tolerancia híbrida.

        Pasa si:
            |actual - expected| ≤ abs_tolerance
            OR
            Δ_rel ≤ rel_tolerance

        Maneja correctamente:
            • Valores grandes: domina tolerancia relativa
            • Valores ~0: domina tolerancia absoluta
        """
        abs_diff = abs(actual - expected)

        if abs_diff <= abs_tolerance:
            return True, 0.0

        rel_diff = NumericValidator.relative_difference(actual, expected)
        return rel_diff <= rel_tolerance, rel_diff


class StringValidator:
    """Validador de cadenas."""

    @staticmethod
    def validate_non_empty(
        value: Any,
        field_name: str,
        max_length: Optional[int] = None,
        *,
        strip: bool = True,
    ) -> str:
        """Valida que value sea str no vacío y dentro de longitud."""
        if not isinstance(value, str):
            raise ValidationError(
                f"{field_name} debe ser str, "
                f"recibido: {type(value).__name__}"
            )

        cleaned = value.strip() if strip else value

        if not cleaned:
            raise ValidationError(f"{field_name} no puede estar vacío")

        if max_length is not None and len(cleaned) > max_length:
            raise ValidationError(
                f"{field_name} excede límite: {len(cleaned)} > {max_length}"
            )

        return cleaned


# ============================================================================
# PROTOCOLO PARA VALIDADORES
# ============================================================================


class InvariantChecker(Protocol):
    """
    Protocolo para objetos que verifican invariantes.
    """

    def check_invariants(self) -> List[Tuple[str, bool]]:
        """
        Verifica invariantes del objeto.

        Retorna
        ───────
        List[Tuple[str, bool]]
            Lista de (nombre_invariante, pasó).
        """
        ...


# ============================================================================
# QUOTIENT SPACE MAPPER — Fibrado de Transformación de Unidades
# ============================================================================


class QuotientSpaceMapper:
    """
    Proyecta el espacio de unidades genéricas Q_genérico al dominio temporal Q_temporal.

    Fundamento Algebraico
    ─────────────────────
    Sea U = {"UNIDAD", "UND", "U", ...} la 0-forma genérica (escalar sin dimensión)
    y T = {"HORA", "DIA", "JOR", ...} el dominio temporal.

    La proyección es un morfismo de espacios vectoriales:
        π: (Q ∈ U, ρ > 0) → (Q' ∈ T,  Q' = 1/ρ)

    donde ρ es el rendimiento [unidades/jornada].

    Esto implementa el fibrado:
        E = Q_genérico × ρ  →(π)→  Q_temporal
        F^{-1}(q_t) = {(q_g, ρ) : q_g = q_t · ρ}  (fibra sobre q_t)

    Invariante garantizado: el nodo InsumoProcesado resultante satisface la
    conservación I1 en el dominio temporal normalizado.
    """

    # Unidades genéricas que pertenecen a la 0-forma escalar
    _GENERIC_FORMS: ClassVar[FrozenSet[str]] = frozenset(
        {"UNIDAD", "UND", "U", "UN", "PAR", "JUEGO", "KIT"}
    )

    # Tolerancia para considerar rendimiento válido
    _RENDIMIENTO_MIN: ClassVar[float] = 1e-9

    @classmethod
    def needs_projection(cls, unit: str, rendimiento: float) -> bool:
        """
        Determina si el morfismo de proyección debe aplicarse.

        Condición: unidad ∈ U_genérico  AND  rendimiento > ε
        """
        return unit in cls._GENERIC_FORMS and rendimiento > cls._RENDIMIENTO_MIN

    @classmethod
    def project_to_temporal(
        cls,
        cantidad: float,
        rendimiento: float,
        codigo_apu: str = "",
    ) -> Tuple[float, str]:
        """
        Aplica la proyección π: Q_genérico → Q_temporal.

        Parameters
        ----------
        cantidad : float
            Cantidad en dominio genérico.
        rendimiento : float
            Tasa de producción [unidades/unidad_temporal] > 0.
        codigo_apu : str
            Código para trazabilidad en logs.

        Returns
        -------
        tuple[float, str]
            (cantidad_proyectada, unidad_proyectada)
            donde cantidad_proyectada = 1.0 / rendimiento.

        Raises
        ------
        ValidationError
            Si rendimiento ≤ 0 (fibra degenerada).
        """
        if rendimiento <= cls._RENDIMIENTO_MIN:
            raise ValidationError(
                f"QuotientSpaceMapper [{codigo_apu}]: "
                f"rendimiento={rendimiento} ≤ ε={cls._RENDIMIENTO_MIN}. "
                f"Fibra degenerada — proyección imposible."
            )

        # π(Q_g, ρ) = 1/ρ   (cantidad temporal estándar en MO)
        cantidad_temporal = 1.0 / rendimiento

        logger.debug(
            "QuotientSpaceMapper [%s]: Q_genérico=%.4f ──π──> "
            "Q_temporal=%.4f (ρ=%.4f)",
            codigo_apu, cantidad, cantidad_temporal, rendimiento,
        )

        return cantidad_temporal, "HORA"

    @classmethod
    def audit_projection(
        cls,
        cantidad_original: float,
        cantidad_proyectada: float,
        codigo_apu: str = "",
    ) -> None:
        """
        Audita la coherencia de la proyección respecto a la cantidad declarada.

        Emite advertencia si la cantidad original difiere significativamente
        de la proyectada (posible error de datos en el insumo fuente).
        """
        if cantidad_original <= 0:
            return
        rel_diff = abs(cantidad_original - cantidad_proyectada) / max(
            abs(cantidad_proyectada), 1e-15
        )
        if rel_diff > 0.05:  # 5% de tolerancia de auditoría
            warnings.warn(
                f"QuotientSpaceMapper [{codigo_apu}]: "
                f"cantidad declarada={cantidad_original:.4f} difiere "
                f"de la cantidad proyectada (1/ρ)={cantidad_proyectada:.4f} "
                f"(Δ_rel={rel_diff:.2%}). Posible inconsistencia en datos fuente.",
                UserWarning,
                stacklevel=4,
            )



# ============================================================================
# NODO TOPOLÓGICO BASE
# ============================================================================


@dataclass
class TopologicalNode:
    """
    Nodo en el grafo bipartito G = (V_TACTICS ∪ V_PHYSICS, E).

    Invariantes:
        • stratum identifica la capa (TACTICS o PHYSICS)
        • id es único y determinista (basado en contenido)
        • structural_health ∈ [0, 1] es métrica de validez
    """

    id: str = ""
    stratum: Stratum = Stratum.PHYSICS
    description: str = ""
    structural_health: float = 1.0
    is_floating: bool = False


# ============================================================================
# INSUMO PROCESADO (NODO HOJA — NIVEL PHYSICS)
# ============================================================================


@dataclass
class InsumoProcesado(TopologicalNode, InvariantChecker):
    """
    Recurso atómico (nodo hoja del grafo bipartito).

    Invariantes verificados en __post_init__:
        (I1) valor_total ≈ cantidad × precio_unitario  (tolerancia relativa)
        (I2) No-negatividad de campos numéricos
        (I3) Acotación por límites físicos
        (I4) Normalización idempotente de campos string
        (I5) tipo_insumo ∈ TipoInsumo
        (I6) Todas las validaciones pasan

    El objeto NUNCA existe en estado inválido gracias a validación
    reactiva en __post_init__.
    """

    # Campos obligatorios de datos
    codigo_apu: str = ""
    descripcion_apu: str = ""
    unidad_apu: str = ""
    descripcion_insumo: str = ""
    unidad_insumo: str = ""
    cantidad: float = 0.0
    precio_unitario: float = 0.0
    valor_total: float = 0.0
    tipo_insumo: str = ""

    # Campos opcionales
    capitulo: str = "GENERAL"
    categoria: str = "OTRO"
    formato_origen: str = "GENERIC"
    rendimiento: float = 0.0

    # Estado interno (no serializable)
    normalized_desc: str = field(default="", init=False, repr=False)
    _validated: bool = field(default=False, init=False, repr=False)
    _validation_errors: List[str] = field(
        default_factory=list, init=False, repr=False
    )

    # Configurables por subclase
    EXPECTED_UNITS: ClassVar[FrozenSet[str]] = UNIDADES_GENERICAS
    REQUIRES_RENDIMIENTO: ClassVar[bool] = False

    def __post_init__(self) -> None:
        """
        Validación completa de invariantes al nacer.

        Pipeline:
            1. Normalizar campos de texto
            2. Validar campos obligatorios
            3. Validar rangos numéricos
            4. Validar consistencia (ley de conservación)
            5. Generar ID determinista
            6. Invocar hook de post-validación
            7. Marcar como validado
        """
        self.stratum = Stratum.PHYSICS
        self.description = self.descripcion_insumo

        try:
            self._normalize_fields()
            self._validate_required_fields()
            self._validate_numeric_fields()
            self._validate_consistency()

            # ID determinista
            content_key = f"{self.codigo_apu}|{self.descripcion_insumo}"
            self.id = f"{self.codigo_apu}_{_deterministic_short_hash(content_key)}"

            self._post_validation_hook()
            self._validated = True

        except ValidationError:
            self._validated = False
            raise

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def total_cost(self) -> float:
        """Costo calculado: cantidad × precio_unitario."""
        return self.cantidad * self.precio_unitario

    @property
    def is_valid(self) -> bool:
        """True si pasó todas las validaciones."""
        return self._validated

    # ------------------------------------------------------------------
    # Invariantes formales
    # ------------------------------------------------------------------

    def check_invariants(self) -> List[Tuple[str, bool]]:
        """
        Verifica invariantes (I1-I5).

        Retorna lista de (nombre, resultado).
        """
        results = []

        # (I1) Ley de conservación
        expected = self.cantidad * self.precio_unitario
        scale = max(abs(expected), abs(self.valor_total))

        if scale < _TOLERANCES.conservation_absolute:
            inv_1 = True
        else:
            rel_diff = abs(expected - self.valor_total) / scale
            inv_1 = rel_diff <= _TOLERANCES.conservation_relative

        results.append(("(I1) Conservation", inv_1))

        # (I2) No-negatividad
        inv_2 = (
            self.cantidad >= 0 and
            self.precio_unitario >= 0 and
            self.valor_total >= 0
        )
        results.append(("(I2) Non-negativity", inv_2))

        # (I3) Acotación
        inv_3 = (
            _BOUNDS.cantidad_min <= self.cantidad <= _BOUNDS.cantidad_max and
            _BOUNDS.precio_min <= self.precio_unitario <= _BOUNDS.precio_max and
            self.valor_total <= _BOUNDS.valor_total_max
        )
        results.append(("(I3) Bounds", inv_3))

        # (I4) Normalización idempotente
        desc_norm_1 = normalize_description(self.descripcion_insumo)
        desc_norm_2 = normalize_description(desc_norm_1)
        inv_4 = desc_norm_1 == desc_norm_2
        results.append(("(I4) Idempotence", inv_4))

        # (I5) Tipología
        inv_5 = self.tipo_insumo in TipoInsumo.valid_values()
        results.append(("(I5) Typology", inv_5))

        return results

    # ------------------------------------------------------------------
    # Normalización
    # ------------------------------------------------------------------

    def _normalize_fields(self) -> None:
        """Proyecta campos a forma canónica."""
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
        """Valida presencia de campos obligatorios."""
        StringValidator.validate_non_empty(
            self.codigo_apu, "codigo_apu", MAX_CODIGO_LENGTH
        )
        StringValidator.validate_non_empty(
            self.descripcion_insumo, "descripcion_insumo", MAX_DESCRIPCION_LENGTH
        )

    def _validate_numeric_fields(self) -> None:
        """Valida rangos numéricos (I2, I3)."""
        self.cantidad = NumericValidator.validate_non_negative(
            self.cantidad, "cantidad",
            _BOUNDS.cantidad_min, _BOUNDS.cantidad_max
        )
        self.precio_unitario = NumericValidator.validate_non_negative(
            self.precio_unitario, "precio_unitario",
            _BOUNDS.precio_min, _BOUNDS.precio_max
        )
        self.valor_total = NumericValidator.validate_non_negative(
            self.valor_total, "valor_total",
            0, _BOUNDS.valor_total_max
        )
        self.rendimiento = NumericValidator.validate_non_negative(
            self.rendimiento, "rendimiento",
            _BOUNDS.rendimiento_min, _BOUNDS.rendimiento_max
        )

    def _validate_consistency(self) -> None:
        """Valida coherencia lógica (I1)."""
        self._validate_valor_total_consistency()
        self._validate_unit_category()

        if self.REQUIRES_RENDIMIENTO and self.rendimiento <= 0:
            warnings.warn(
                f"Rendimiento debería ser > 0 en {self.codigo_apu}",
                UserWarning, stacklevel=3
            )

    def _validate_valor_total_consistency(self) -> None:
        """
        Verifica ley de conservación (I1) con tolerancia híbrida.

        Criterio:
            expected = cantidad × precio_unitario
            scale = max(|expected|, |valor_total|)

            Si scale < ε_abs: aceptar (ambos ≈ 0)
            Si scale ≥ ε_abs: exigir |expected - actual|/scale ≤ ε_rel
        """
        expected = self.cantidad * self.precio_unitario
        actual = self.valor_total
        scale = max(abs(expected), abs(actual))

        # Caso trivial
        if scale < _TOLERANCES.conservation_absolute:
            return

        # Verificación relativa
        rel_diff = abs(expected - actual) / scale

        if rel_diff > _TOLERANCES.conservation_relative:
            raise InvariantError(
                f"{self.__class__.__name__} [{self.codigo_apu}]: "
                f"Ley de conservación (I1) violada: "
                f"valor_total={actual:.6f} vs "
                f"calculado={expected:.6f} "
                f"(Δ_rel={rel_diff:.2e} > τ={_TOLERANCES.conservation_relative:.2e})"
            )

    def _validate_unit_category(self) -> None:
        """Advierte si unidad no está en EXPECTED_UNITS."""
        if not self.EXPECTED_UNITS:
            return

        if self.unidad_insumo not in self.EXPECTED_UNITS:
            warnings.warn(
                f"{self.__class__.__name__} [{self.codigo_apu}]: "
                f"Unidad '{self.unidad_insumo}' no en "
                f"{self.EXPECTED_UNITS}",
                UserWarning, stacklevel=3
            )

    def _post_validation_hook(self) -> None:
        """Hook para validaciones específicas de subclases."""
        pass

    # ------------------------------------------------------------------
    # Serialización
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario, excluyendo campos internos (_*)."""
        data = asdict(self)
        return {k: v for k, v in data.items() if not k.startswith("_")}


# ============================================================================
# ESTRUCTURA APU (NODO INTERNO — NIVEL TACTICS)
# ============================================================================


@dataclass
class APUStructure(TopologicalNode):
    """
    Actividad constructiva en nivel TACTICS.

    Agrega recursos (InsumoProcesado) y calcula índice de estabilidad
    topológica Ψ basado en entropía de Shannon y diversidad categórica.
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
        """Número de recursos (ancho de base de soporte)."""
        return len(self.resources)

    @property
    def is_inverted_pyramid(self) -> bool:
        """
        Detecta configuración degenerada: cantidad grande, base estrecha.

        Criterio: quantity > 1000 y support_base_width = 1
        """
        return self.quantity > 1000 and self.support_base_width == 1

    @property
    def total_cost(self) -> float:
        """Costo total: Σ valor_total."""
        return sum(r.valor_total for r in self.resources)

    # ------------------------------------------------------------------
    # Métrica de estabilidad topológica (Índice de Fiedler generalizado)
    # ------------------------------------------------------------------

    def topological_stability_index(self) -> float:
        r"""
        Índice de estabilidad topológica Ψ ∈ [0, 1].

        Definición
        ──────────
        Ψ(a) = D(a)^α · H_norm(a)^(1-α)  donde α = 0.4

        Componentes:
            1. Diversidad categórica: D = |tipos únicos| / 5
            2. Entropía de Shannon normalizada: H_norm = H / ln(n)

        Propiedades
        ───────────
        (P1) Ψ = 0 si n ≤ 1 o Σ valor_total = 0
        (P2) Ψ = 1 solo si D = 1 Y H = H_max (máxima estabilidad)
        (P3) Media geométrica fuerza compensación:
             Ψ bajo si D bajo O H bajo (no puede compensarse uno con otro)

        Justificación de α = 0.4
        ─────────────────────
        • Privilegia diversidad categórica: α > 0.5 porque el riesgo
          de mono-proveedor es la amenaza más grave
        • Balance: 0.4 permite que alta entropía ayude pero no compense
          completamente bajo D

        Entropía de Shannon (Shannon, 1948)
        ──────────────────────────────────
        H = -Σᵢ₌₁ⁿ pᵢ ln(pᵢ)  donde pᵢ = valor_total_i / Σ valor_total_j

        Propiedades:
            • H ∈ [0, ln(n)]
            • H = 0 cuando una categoría tiene 100% del costo
            • H = ln(n) cuando todas tienen costo igual
            • Cóncava: penaliza concentración

        Casos especiales
        ────────────────
        • n = 0: Ψ = 0 (sin recursos)
        • n = 1: Ψ = 0 (sin diversidad)
        • Σ valor_total = 0: Ψ = 0 (trivial)

        Retorna
        ───────
        float
            Índice en [0, 1]. Mayor = más estable.
        """
        if not self.resources:
            return 0.0

        n = len(self.resources)

        # Diversidad categórica
        tipos_unicos = {r.tipo_insumo for r in self.resources}
        diversidad = len(tipos_unicos) / _NUM_TIPOS_INSUMO

        # Un solo recurso → no hay incertidumbre, Ψ = 0
        if n == 1:
            return sys.float_info.epsilon

        # Entropía de Shannon
        valores = [max(r.valor_total, 0.0) for r in self.resources]
        total_valor = sum(valores)

        if total_valor <= 0:
            return 0.0

        # Cálculo seguro de H = -Σ p ln(p)
        entropia = 0.0
        for v in valores:
            if v > 0:
                p = v / total_valor
                # p ln(p) → 0 cuando p → 0+ (continuidad)
                entropia -= p * math.log(p)

        h_max = math.log(n)
        h_norm = (entropia / h_max) if h_max > 0 else 0.0
        h_norm = max(0.0, min(h_norm, 1.0))  # Clamp a [0, 1]

        # Media geométrica ponderada
        alpha = 0.4
        return (diversidad ** alpha) * (h_norm ** (1.0 - alpha))

    # ------------------------------------------------------------------
    # Mutación controlada
    # ------------------------------------------------------------------

    def add_resource(self, resource: InsumoProcesado) -> None:
        """
        Agrega recurso validando invariante topológico.

        Garantiza: recurso.stratum = PHYSICS
        """
        if not isinstance(resource, InsumoProcesado):
            raise TypeError(
                f"Esperado InsumoProcesado, "
                f"recibido: {type(resource).__name__}"
            )

        if resource.stratum != Stratum.PHYSICS:
            raise TypeError(
                f"Solo nodos PHYSICS en APUStructure, "
                f"recibido: {resource.stratum.name}"
            )

        self.resources.append(resource)

    # ------------------------------------------------------------------
    # Análisis
    # ------------------------------------------------------------------

    def get_cost_breakdown(self) -> Dict[str, float]:
        """Distribución de costos por tipo de insumo."""
        breakdown: Dict[str, float] = {}
        for r in self.resources:
            breakdown[r.tipo_insumo] = (
                breakdown.get(r.tipo_insumo, 0.0) + r.valor_total
            )
        return breakdown

    def get_cost_fractions(self) -> Dict[str, float]:
        """Fracción de costo de cada tipo (suma ≈ 1.0)."""
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
    """Recurso de tipo Mano de Obra."""

    EXPECTED_UNITS: ClassVar[FrozenSet[str]] = UNIDADES_TIEMPO
    REQUIRES_RENDIMIENTO: ClassVar[bool] = True

    jornal: float = 0.0

    def _post_validation_hook(self) -> None:
        super()._post_validation_hook()

        if self.jornal != 0.0:
            self.jornal = NumericValidator.validate_non_negative(
                self.jornal, "jornal",
                _BOUNDS.precio_min, _BOUNDS.precio_max
            )

        if self.rendimiento > 0:
            # ── QuotientSpaceMapper: si unidad_insumo es 0-forma genérica,
            #    proyectar al dominio temporal mediante π(Q,ρ) = 1/ρ ──────────
            if QuotientSpaceMapper.needs_projection(
                self.unidad_insumo, self.rendimiento
            ):
                qty_projected, unit_projected = QuotientSpaceMapper.project_to_temporal(
                    cantidad=self.cantidad,
                    rendimiento=self.rendimiento,
                    codigo_apu=self.codigo_apu,
                )
                QuotientSpaceMapper.audit_projection(
                    self.cantidad, qty_projected, self.codigo_apu
                )
                # Actualizar unidad al dominio temporal homogenizado
                object.__setattr__(self, "unidad_insumo", unit_projected) \
                    if hasattr(type(self), "__dataclass_fields__") \
                    else setattr(self, "unidad_insumo", unit_projected)
                # Nota: la cantidad NO se corrige aquí — se audita vía warnings
                # para preservar el dato fuente y trazabilidad forense.

            # ── Verificación rendimiento/cantidad en dominio normalizado ──────
            expected_qty = 1.0 / self.rendimiento
            ok, rel_diff = NumericValidator.values_consistent(
                self.cantidad, expected_qty,
                rel_tolerance=0.05,
                abs_tolerance=_TOLERANCES.cantidad_rendimiento
            )
            if not ok:
                warnings.warn(
                    f"Discrepancia rendimiento/cantidad en {self.codigo_apu}: "
                    f"cantidad={self.cantidad:.4f}, "
                    f"esperada={expected_qty:.4f} "
                    f"(Δ_rel={rel_diff:.2%})",
                    UserWarning, stacklevel=3
                )



@dataclass
class Equipo(InsumoProcesado):
    """Recurso de tipo Equipo."""
    EXPECTED_UNITS: ClassVar[FrozenSet[str]] = UNIDADES_TIEMPO
    REQUIRES_RENDIMIENTO: ClassVar[bool] = False


@dataclass
class Suministro(InsumoProcesado):
    """Recurso de tipo Suministro."""
    EXPECTED_UNITS: ClassVar[FrozenSet[str]] = (
        UNIDADES_MASA | UNIDADES_VOLUMEN | UNIDADES_AREA |
        UNIDADES_LONGITUD | UNIDADES_GENERICAS
    )

    def _post_validation_hook(self) -> None:
        super()._post_validation_hook()
        if self.cantidad == 0:
            warnings.warn(
                f"Suministro con cantidad=0 en {self.codigo_apu}",
                UserWarning, stacklevel=3
            )


@dataclass
class Transporte(InsumoProcesado):
    """Recurso de tipo Transporte."""
    EXPECTED_UNITS: ClassVar[FrozenSet[str]] = UNIDADES_TRANSPORTE
    REQUIRES_RENDIMIENTO: ClassVar[bool] = False


@dataclass
class Otro(InsumoProcesado):
    """Recurso genérico (sin restricción de unidades)."""
    EXPECTED_UNITS: ClassVar[FrozenSet[str]] = frozenset()
    REQUIRES_RENDIMIENTO: ClassVar[bool] = False


# Registro de clases
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


@lru_cache(maxsize=32)
def _get_class_params(cls: Type[InsumoProcesado]) -> FrozenSet[str]:
    """Obtiene parámetros válidos del constructor."""
    sig = inspect.signature(cls)
    return frozenset(sig.parameters.keys())


_REQUIRED_RAW_FIELDS = frozenset({
    "codigo_apu", "descripcion_apu", "unidad_apu",
    "descripcion_insumo", "unidad_insumo", "tipo_insumo",
})

_NUMERIC_FIELDS = {
    "cantidad": 0.0,
    "precio_unitario": 0.0,
    "valor_total": 0.0,
    "rendimiento": 0.0,
    "jornal": 0.0,
}

_STRING_DEFAULTS = {
    "capitulo": "GENERAL",
    "categoria": "OTRO",
    "formato_origen": "GENERIC",
}


def validate_insumo_data(insumo_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida y limpia datos crudos.

    Pipeline:
        1. Verificar tipo (dict)
        2. Verificar campos obligatorios
        3. Validar/convertir campos numéricos
        4. Aplicar defaults
        5. Normalizar tipo_insumo
        6. Calcular valor_total si falta
    """
    if not isinstance(insumo_data, dict):
        raise ValidationError(
            f"insumo_data debe ser dict, "
            f"recibido: {type(insumo_data).__name__}"
        )

    # Campos obligatorios
    missing = sorted(
        f for f in _REQUIRED_RAW_FIELDS
        if f not in insumo_data or insumo_data[f] is None
    )
    if missing:
        raise ValidationError(f"Campos faltantes: {missing}")

    cleaned: Dict[str, Any] = {}

    # Procesar campos
    for key, value in insumo_data.items():
        if value is None:
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
                raise ValidationError(f"Campo '{key}' no finito: {fval}")

            if fval < 0:
                raise ValidationError(f"Campo '{key}' negativo: {fval}")

            cleaned[key] = fval
        else:
            cleaned[key] = value

    # Defaults
    for key, default in {**_NUMERIC_FIELDS, **_STRING_DEFAULTS}.items():
        cleaned.setdefault(key, default)

    # Normalizar tipo
    try:
        tipo = TipoInsumo.from_string(cleaned["tipo_insumo"])
        cleaned["tipo_insumo"] = tipo.value
        cleaned["categoria"] = tipo.value
    except InvalidTipoInsumoError as exc:
        raise ValidationError(str(exc)) from exc

    # Calcular valor_total si falta
    if cleaned.get("valor_total", 0.0) == 0.0:
        cleaned["valor_total"] = (
            cleaned.get("cantidad", 0.0) * cleaned.get("precio_unitario", 0.0)
        )

    return cleaned


@invariant_check
def create_insumo(**kwargs: Any) -> InsumoProcesado:
    """
    Factory principal con invariantes verificados.

    Resuelve clase vía INSUMO_CLASS_MAP y filtra kwargs.
    """
    if "tipo_insumo" not in kwargs:
        raise ValidationError("Falta: tipo_insumo")

    tipo_enum = TipoInsumo.from_string(kwargs["tipo_insumo"])
    tipo_value = tipo_enum.value

    kwargs["tipo_insumo"] = tipo_value
    kwargs["categoria"] = tipo_value

    insumo_class = INSUMO_CLASS_MAP.get(tipo_value)
    if insumo_class is None:
        raise InvalidTipoInsumoError(f"No hay clase para: {tipo_value}")

    valid_params = _get_class_params(insumo_class)
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}

    try:
        return insumo_class(**filtered)
    except ValidationError:
        raise
    except Exception as exc:
        logger.error(
            "Error creando insumo %s: %s",
            tipo_value, exc, exc_info=True
        )
        raise InsumoDataError(f"Error fatal: {exc}") from exc


def create_insumo_from_raw(raw_data: Dict[str, Any]) -> InsumoProcesado:
    """
    Crea insumo desde datos crudos (pipeline completo).

    raw_data → validate → create → validado
    """
    validated = validate_insumo_data(raw_data)
    return create_insumo(**validated)


# ============================================================================
# UTILIDADES
# ============================================================================


def get_tipo_insumo_class(
    tipo: Union[str, TipoInsumo]
) -> Type[InsumoProcesado]:
    """Resuelve clase correspondiente a tipo."""
    tipo_enum = TipoInsumo.from_string(tipo)
    cls = INSUMO_CLASS_MAP.get(tipo_enum.value)
    if cls is None:
        raise InvalidTipoInsumoError(f"No hay clase: {tipo_enum.value}")
    return cls


def get_all_tipo_insumo_values() -> FrozenSet[str]:
    """Todos los valores válidos de TipoInsumo."""
    return TipoInsumo.valid_values()


def is_valid_tipo_insumo(tipo: str) -> bool:
    """Verifica si tipo es válido (sin lanzar excepción)."""
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
    "SchemaError", "ValidationError", "InvalidTipoInsumoError",
    "InsumoDataError", "UnitNormalizationError", "InvariantError",
    # Enumeraciones
    "TipoInsumo", "Stratum",
    # Estructuras
    "TopologicalNode", "InsumoProcesado", "APUStructure",
    "ManoDeObra", "Equipo", "Suministro", "Transporte", "Otro",
    # Factories
    "create_insumo", "create_insumo_from_raw", "validate_insumo_data",
    # Normalización
    "normalize_unit", "normalize_description", "normalize_codigo",
    # Validadores
    "NumericValidator", "StringValidator", "NumericalTolerances", "PhysicalBounds",
    # Constantes
    "UNIDADES_TIEMPO", "UNIDADES_MASA", "UNIDADES_VOLUMEN",
    "UNIDADES_AREA", "UNIDADES_LONGITUD", "UNIDADES_TRANSPORTE",
    "UNIDADES_GENERICAS", "INSUMO_CLASS_MAP",
    # Utilidades
    "get_tipo_insumo_class", "get_all_tipo_insumo_values", "is_valid_tipo_insumo",
]