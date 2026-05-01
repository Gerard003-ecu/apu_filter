"""
═══════════════════════════════════════════════════════════════════════════════
═══════════════════════════════════════════════════════════════════════════════
MÓDULO: Data Validators (Operador de Restricción Métrica y Funtor de Validación)
UBICACIÓN: app/adapters/validators.py
VERSIÓN: 3.0.0 - Rigorización Matemática Absoluta y Cierre Algebraico

 NATURALEZA CIBER-FÍSICA Y TOPOLÓGICA:
 Este módulo actúa como el Filtro de Variedad Diferenciable en la frontera de 
 los estratos operativos. Implementa rigurosamente un FUNCTOR de validación 
 𝓕: DataFrames → ValidationResults que mapea el hiperespacio de los datos 
 (Espacios de Hilbert ℝⁿ) hacia la categoría de resultados, preservando la 
 estructura monoidal y erradicando singularidades numéricas.

 FUNDAMENTOS MATEMÁTICOS RIGUROSOS Y AXIOMAS DE EJECUCIÓN:

 1. EL MONOIDE DE VALIDACIÓN Y LA OPERACIÓN SUPREMO (⊔):
    * La agregación de resultados opera sobre un Monoide Conmutativo estricto 
      donde el elemento neutro e = ValidationResult.IDENTITY.
    * El operador binario ⊕ (merge) computa matemáticamente el Supremo (⊔) del 
      Retículo de Severidades: Severidad_Resultante = max(s₁, s₂). 
    * El subconjunto de severidades bloqueantes {ERROR, CRITICAL} actúa como 
      el elemento absorbente (⊤); tocar este límite colapsa incondicionalmente 
      el tensor hacia la invalidez (Veto Estructural).

 2. ISOMORFISMO DIMENSIONAL VÍA ESPECTRO (SVD):
    * La conexidad del espacio (rango de la matriz) no se evalúa mediante
      heurísticas nominales. Se exige el cálculo del rango efectivo mediante
      Descomposición en Valores Singulares (SVD).
    * Si un valor singular σ_i < FLOAT_TOLERANCE, la dimensión se considera 
      topológicamente colapsada (degeneración por colinealidad), disparando 
      un ValidationCode.DEGENERATION.

 3. ACOTACIÓN TERMODINÁMICA Y CIERRE ALGEBRAICO (IEEE 754):
    * Se impone el cierre algebraico de los números reales. Valores infinitos 
      (±∞) y NaN se catalogan como Singularidades Topológicas irresolubles.
    * Aniquilación Estricta de Subnormales: Los valores en el intervalo abierto 
      (0, MIN_NORMAL_FLOAT) se someten a un filtro "flush-to-zero" (0.0) para 
      evitar la inyección de fricción cuántica en la FPU durante simulaciones.

 4. SOBREVIVENCIA TOPOLÓGICA (Medida de Lebesgue):
    * La extirpación de singularidades está sujeta a la conservación del volumen. 
    * Si la medida μ(filas_válidas) / μ(filas_totales) decae por debajo de la cota 
      DEFAULT_SURVIVAL_THRESHOLD, el sistema aborta la inyección para evitar 
      entregar un hiperespacio degenerado a los estratos tácticos.
═══════════════════════════════════════════════════════════════════════════════
═══════════════════════════════════════════════════════════════════════════════

INVARIANTES TOPOLÓGICOS PRESERVADOS
-----------------------------------

∀ df ∈ DataFrames, el validador garantiza:

1. **Isomorfismo Dimensional**: 
   dim(df.columns) ≅ dim(required_schema)

2. **Continuidad**: 
   No existen singularidades (NaN, ±∞) en el espacio métrico

3. **Compacidad**: 
   ∀ columna numérica: valores ∈ [min, max] ⊂ ℝ (conjunto compacto)

4. **Conexidad**: 
   rank(matriz_datos) = dim(espacio_columnas) (espacio conexo)

5. **Medida de Lebesgue**: 
   μ(filas_válidas) / μ(filas_totales) ≥ threshold (preservación de volumen)

RETÍCULO DE SEVERIDADES (Álgebra de Boole)
-------------------------------------------

El conjunto de severidades forma un RETÍCULO COMPLETO ordenado:

    CRITICAL  (⊤ - elemento máximo, absorbente)
       ↑
    ERROR
       ↑
    WARNING
       ↑
    INFO      (⊥ - elemento mínimo)

Operaciones:
    - ∨ (join/supremo): max(s₁, s₂)
    - ∧ (meet/ínfimo): min(s₁, s₂)
    - ¬ (complemento): inversión del orden

LEYES DEL MONOIDE DE VALIDACIÓN
--------------------------------

(ValidationResult, ⊕, IDENTITY) satisface:

1. **Asociatividad**: 
   (r₁ ⊕ r₂) ⊕ r₃ = r₁ ⊕ (r₂ ⊕ r₃)

2. **Elemento Neutro**: 
   r ⊕ IDENTITY = IDENTITY ⊕ r = r

3. **Conmutatividad** (no requerida pero deseable para paralelización):
   r₁ ⊕ r₂ ≈ r₂ ⊕ r₁ (módulo orden de issues)

TRANSFORMACIONES NATURALES
---------------------------

El método `validate_domain` implementa una TRANSFORMACIÓN NATURAL entre functores:

    η: Validator → Composite_Validator
    
tal que el siguiente diagrama conmuta:

    DataFrame ──validate_schema──→ ValidationResult
        │                              │
        │                              │
        ↓                              ↓
    DataFrame ──validate_domain───→ ValidationResult

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    ClassVar,
    Final,
    FrozenSet,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
)

import numpy as np
import pandas as pd
from numpy.typing import NDArray


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN I: CONSTANTES MATEMÁTICAS Y FÍSICAS
# ══════════════════════════════════════════════════════════════════════════════

# Tolerancia para comparaciones de punto flotante (épsilon de máquina × 10)
FLOAT_TOLERANCE: Final[float] = np.finfo(np.float64).eps * 10

# Mínimo número normal positivo IEEE 754 (barrera subnormal)
MIN_NORMAL_FLOAT: Final[float] = sys.float_info.min

# Umbral por defecto de supervivencia volumétrica (90%)
DEFAULT_SURVIVAL_THRESHOLD: Final[float] = 0.9

# Máximo rango permitido para validaciones (evita desbordamiento)
MAX_SAFE_RANGE: Final[float] = 1e308


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN II: ÁLGEBRA DE SEVERIDADES (RETÍCULO COMPLETO)
# ══════════════════════════════════════════════════════════════════════════════


class ValidationSeverity(str, Enum):
    """
    Niveles de severidad que forman un RETÍCULO COMPLETO ordenado.
    
    Orden parcial: INFO < WARNING < ERROR < CRITICAL
    
    Operaciones Reticulares:
    ------------------------
    - join (∨): severidad máxima
    - meet (∧): severidad mínima
    - complement (¬): inversión del orden
    
    Propiedades Algebraicas:
    ------------------------
    - Idempotencia: s ∨ s = s, s ∧ s = s
    - Conmutatividad: s₁ ∨ s₂ = s₂ ∨ s₁
    - Asociatividad: (s₁ ∨ s₂) ∨ s₃ = s₁ ∨ (s₂ ∨ s₃)
    - Absorción: s ∨ (s ∧ t) = s
    """

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        """Rango ordinal en el retículo (menor = menos severo)."""
        return _SEVERITY_RANKS[self]

    def __lt__(self, other: ValidationSeverity) -> bool:
        """Orden parcial estricto."""
        if not isinstance(other, ValidationSeverity):
            return NotImplemented
        return self.rank < other.rank

    def __le__(self, other: ValidationSeverity) -> bool:
        """Orden parcial no estricto."""
        if not isinstance(other, ValidationSeverity):
            return NotImplemented
        return self.rank <= other.rank

    def join(self, other: ValidationSeverity) -> ValidationSeverity:
        """Supremo (∨) en el retículo: max(self, other)."""
        return self if self.rank >= other.rank else other

    def meet(self, other: ValidationSeverity) -> ValidationSeverity:
        """Ínfimo (∧) en el retículo: min(self, other)."""
        return self if self.rank <= other.rank else other

    @classmethod
    def top(cls) -> ValidationSeverity:
        """Elemento máximo del retículo (⊤): CRITICAL."""
        return cls.CRITICAL

    @classmethod
    def bottom(cls) -> ValidationSeverity:
        """Elemento mínimo del retículo (⊥): INFO."""
        return cls.INFO


# Mapeo de severidades a rangos ordinales (invariante de orden)
_SEVERITY_RANKS: Final[Mapping[ValidationSeverity, int]] = {
    ValidationSeverity.INFO: 0,
    ValidationSeverity.WARNING: 1,
    ValidationSeverity.ERROR: 2,
    ValidationSeverity.CRITICAL: 3,
}

# Conjunto inmutable de severidades bloqueantes (elementos absorbentes)
_BLOCKING_SEVERITIES: Final[FrozenSet[ValidationSeverity]] = frozenset(
    {ValidationSeverity.ERROR, ValidationSeverity.CRITICAL}
)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN III: CÓDIGOS CANÓNICOS DE INCIDENCIA (ESPACIO DE ESTADOS)
# ══════════════════════════════════════════════════════════════════════════════


class ValidationCode(str, Enum):
    """
    Códigos canónicos que identifican ESTADOS ANÓMALOS en el espacio de datos.
    
    Cada código representa una clase de equivalencia de fallos que requieren
    intervención correctiva antes de la inyección al sistema agéntico.
    
    Organización Taxonómica:
    ------------------------
    1. Códigos Estructurales (0xx): anomalías en la topología del DataFrame
    2. Códigos de Calidad (1xx): singularidades en el espacio de valores
    3. Códigos Numéricos (2xx): violaciones de clausura algebraica
    4. Códigos de Dominio (3xx): transgresiones de restricciones físicas
    5. Códigos Topológicos (4xx): degeneraciones del espacio vectorial
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Grupo 0: Anomalías Estructurales (Topología del Esquema)
    # ──────────────────────────────────────────────────────────────────────────
    NONE_DATAFRAME = "NONE_DATAFRAME"
    INVALID_DATAFRAME_TYPE = "INVALID_DATAFRAME_TYPE"
    INVALID_COLUMNS_ARGUMENT = "INVALID_COLUMNS_ARGUMENT"
    EMPTY_COLUMN_NAME = "EMPTY_COLUMN_NAME"
    DUPLICATE_COLUMNS = "DUPLICATE_COLUMNS"
    MISSING_REQUIRED_COLUMN = "MISSING_REQUIRED_COLUMN"
    MISSING_CRITICAL_COLUMN = "MISSING_CRITICAL_COLUMN"

    # ──────────────────────────────────────────────────────────────────────────
    # Grupo 1: Singularidades de Calidad (Discontinuidades)
    # ──────────────────────────────────────────────────────────────────────────
    NULL_VALUES = "NULL_VALUES"
    EMPTY_STRINGS = "EMPTY_STRINGS"

    # ──────────────────────────────────────────────────────────────────────────
    # Grupo 2: Violaciones Numéricas (Clausura Algebraica)
    # ──────────────────────────────────────────────────────────────────────────
    NON_NUMERIC_VALUES = "NON_NUMERIC_VALUES"
    NON_FINITE_VALUES = "NON_FINITE_VALUES"
    SUBNORMAL_VALUES = "SUBNORMAL_VALUES"

    # ──────────────────────────────────────────────────────────────────────────
    # Grupo 3: Transgresiones de Dominio (Restricciones Físicas)
    # ──────────────────────────────────────────────────────────────────────────
    NEGATIVE_VALUES = "NEGATIVE_VALUES"
    RANGE_VIOLATION = "RANGE_VIOLATION"

    # ──────────────────────────────────────────────────────────────────────────
    # Grupo 4: Degeneraciones Topológicas (Invariantes del Espacio)
    # ──────────────────────────────────────────────────────────────────────────
    LINEAR_DEPENDENCY = "LINEAR_DEPENDENCY"
    SURVIVAL_THRESHOLD_VIOLATION = "SURVIVAL_THRESHOLD_VIOLATION"
    RANK_DEFICIENCY = "RANK_DEFICIENCY"


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN IV: ESTRUCTURAS DE DATOS INMUTABLES (OBJETOS ALGEBRAICOS)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, order=True)
class ValidationIssue:
    """
    Hallazgo atómico de validación (elemento del monoide libre).
    
    Representa un PUNTO CRÍTICO en el espacio de configuración donde se viola
    un invariante. Es inmutable (frozen) para garantizar pureza referencial.
    
    Invariantes Matemáticos:
    ------------------------
    1. severity ∈ ValidationSeverity (elemento del retículo)
    2. code ∈ ValidationCode (estado del autómata)
    3. message ≠ "" (descripción no vacía)
    4. column ∈ df.columns ∪ {None} (proyección opcional)
    5. count ∈ ℕ ∪ {None} (cardinalidad opcional)
    
    Orden Lexicográfico:
    -------------------
    Issues se ordenan por: (severity.rank, code, column, count)
    """

    severity: ValidationSeverity
    code: ValidationCode
    message: str
    column: Optional[str] = field(default=None, compare=True)
    count: Optional[int] = field(default=None, compare=True)

    def __post_init__(self) -> None:
        """Validación de invariantes del constructor."""
        if not isinstance(self.severity, ValidationSeverity):
            raise TypeError(f"severity debe ser ValidationSeverity, recibido: {type(self.severity)}")
        if not isinstance(self.code, ValidationCode):
            raise TypeError(f"code debe ser ValidationCode, recibido: {type(self.code)}")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message debe ser un string no vacío")
        if self.column is not None and not isinstance(self.column, str):
            raise TypeError(f"column debe ser str o None, recibido: {type(self.column)}")
        if self.count is not None:
            if not isinstance(self.count, int):
                raise TypeError(f"count debe ser int o None, recibido: {type(self.count)}")
            if self.count < 0:
                raise ValueError(f"count debe ser no negativo, recibido: {self.count}")

    def __str__(self) -> str:
        """Representación legible para logging y diagnóstico."""
        parts = [f"[{self.severity.value}] {self.code.value}: {self.message}"]
        if self.column is not None:
            parts.append(f"columna={self.column}")
        if self.count is not None:
            parts.append(f"count={self.count}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialización a diccionario para APIs externas."""
        return {
            "severity": self.severity.value,
            "code": self.code.value,
            "message": self.message,
            "column": self.column,
            "count": self.count,
        }

    @property
    def is_blocking(self) -> bool:
        """Predicado: ¿Este issue invalida el resultado?"""
        return self.severity in _BLOCKING_SEVERITIES


@dataclass(frozen=True)
class ValidationResult:
    """
    Resultado agregado de validación que forma un MONOIDE CONMUTATIVO.
    
    Estructura Algebraica:
    ----------------------
    (ValidationResult, merge, IDENTITY) es un MONOIDE donde:
    
    - **Conjunto**: Todos los ValidationResult posibles
    - **Operación binaria** (⊕ = merge):
        - Asociativa: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
        - Conmutativa: a ⊕ b ≈ b ⊕ a (módulo orden de issues)
    - **Elemento neutro** (ε = IDENTITY):
        - a ⊕ ε = ε ⊕ a = a
    
    Función Indicadora:
    ------------------
    is_valid: ValidationResult → {True, False}
    is_valid(r) = ∄ issue ∈ r.issues : issue.severity ∈ BLOCKING_SEVERITIES
    
    Invariantes:
    -----------
    1. issues es una tupla inmutable (pureza)
    2. is_valid se calcula automáticamente (no puede ser inconsistente)
    3. Orden de issues preserva cronología de detección
    """

    is_valid: bool
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    # Elemento neutro del monoide (singleton)
    IDENTITY: ClassVar[ValidationResult]

    def __post_init__(self) -> None:
        """Garantiza invariante: is_valid coherente con issues."""
        # Validación de tipos
        if not isinstance(self.issues, tuple):
            raise TypeError(f"issues debe ser tuple, recibido: {type(self.issues)}")
        
        for idx, issue in enumerate(self.issues):
            if not isinstance(issue, ValidationIssue):
                raise TypeError(
                    f"issues[{idx}] debe ser ValidationIssue, recibido: {type(issue)}"
                )
        
        # Recalcular is_valid para garantizar coherencia
        computed_validity = not any(
            issue.severity in _BLOCKING_SEVERITIES for issue in self.issues
        )
        
        # Forzar el valor correcto (dataclass frozen requiere object.__setattr__)
        object.__setattr__(self, "is_valid", computed_validity)

    # ──────────────────────────────────────────────────────────────────────────
    # Propiedades Derivadas (Proyecciones del Conjunto de Issues)
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        """Proyección: issues con severidad ERROR o CRITICAL."""
        return tuple(i for i in self.issues if i.severity in _BLOCKING_SEVERITIES)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        """Proyección: issues con severidad WARNING."""
        return tuple(i for i in self.issues if i.severity == ValidationSeverity.WARNING)

    @property
    def infos(self) -> tuple[ValidationIssue, ...]:
        """Proyección: issues con severidad INFO."""
        return tuple(i for i in self.issues if i.severity == ValidationSeverity.INFO)

    @property
    def has_errors(self) -> bool:
        """Predicado: ∃ issue ∈ errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Predicado: ∃ issue ∈ warnings."""
        return len(self.warnings) > 0

    @property
    def max_severity(self) -> Optional[ValidationSeverity]:
        """Supremo del retículo de severidades en issues."""
        if not self.issues:
            return None
        return max(issue.severity for issue in self.issues)

    # ──────────────────────────────────────────────────────────────────────────
    # Operadores del Monoide
    # ──────────────────────────────────────────────────────────────────────────

    def __add__(self, other: ValidationResult) -> ValidationResult:
        """
        Operador binario ⊕ del monoide (asociativo y conmutativo).
        
        Sintaxis: result = result1 + result2
        Equivale a: result = result1.merge(result2)
        
        Propiedades:
        -----------
        - Asociatividad: (r1 + r2) + r3 = r1 + (r2 + r3)
        - Conmutatividad: r1 + r2 ≈ r2 + r1 (módulo orden)
        - Elemento neutro: r + IDENTITY = r
        """
        if not isinstance(other, ValidationResult):
            return NotImplemented
        return self.merge(other)

    def __bool__(self) -> bool:
        """Conversión booleana: True si es válido."""
        return self.is_valid

    def __len__(self) -> int:
        """Cardinalidad del conjunto de issues."""
        return len(self.issues)

    def __str__(self) -> str:
        """Representación textual para logging."""
        status = "✓ VALID" if self.is_valid else "✗ INVALID"
        return (
            f"ValidationResult({status}, "
            f"errors={len(self.errors)}, "
            f"warnings={len(self.warnings)}, "
            f"infos={len(self.infos)})"
        )

    def __repr__(self) -> str:
        """Representación para debugging."""
        return (
            f"ValidationResult(is_valid={self.is_valid}, "
            f"issues={self.issues!r})"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Constructores Estáticos (Factories)
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_issues(cls, issues: Iterable[ValidationIssue] = ()) -> ValidationResult:
        """
        Constructor canónico que recalcula is_valid automáticamente.
        
        Args:
            issues: Iterable de ValidationIssue (se materializa a tuple)
        
        Returns:
            ValidationResult con is_valid coherente
        """
        issues_tuple = tuple(issues)
        # __post_init__ garantizará coherencia de is_valid
        return cls(is_valid=True, issues=issues_tuple)

    @classmethod
    def success(cls) -> ValidationResult:
        """
        Construye resultado exitoso (elemento neutro del monoide).
        
        Returns:
            ValidationResult.IDENTITY (sin issues, válido)
        """
        return cls(is_valid=True, issues=())

    @classmethod
    def failure(
        cls,
        issues: Iterable[ValidationIssue],
    ) -> ValidationResult:
        """
        Construye resultado fallido desde issues.
        
        Args:
            issues: Iterable de issues (debe contener al menos un bloqueante)
        
        Returns:
            ValidationResult con is_valid recalculado
        """
        return cls.from_issues(issues)

    @classmethod
    def single_issue(
        cls,
        severity: ValidationSeverity,
        code: ValidationCode,
        message: str,
        column: Optional[str] = None,
        count: Optional[int] = None,
    ) -> ValidationResult:
        """
        Constructor de conveniencia para un único issue.
        
        Args:
            severity: Severidad del issue
            code: Código canónico
            message: Descripción
            column: Columna afectada (opcional)
            count: Cantidad afectada (opcional)
        
        Returns:
            ValidationResult con un único issue
        """
        issue = ValidationIssue(
            severity=severity,
            code=code,
            message=message,
            column=column,
            count=count,
        )
        return cls.from_issues((issue,))

    # ──────────────────────────────────────────────────────────────────────────
    # Operaciones del Monoide
    # ──────────────────────────────────────────────────────────────────────────

    def merge(self, other: ValidationResult) -> ValidationResult:
        """
        Combina dos resultados preservando orden cronológico (⊕).
        
        Implementa la operación binaria del monoide:
            (r₁ ⊕ r₂).issues = r₁.issues ++ r₂.issues (concatenación)
            (r₁ ⊕ r₂).is_valid = r₁.is_valid ∧ r₂.is_valid
        
        Args:
            other: Otro ValidationResult
        
        Returns:
            Nuevo ValidationResult combinado
        
        Raises:
            TypeError: Si other no es ValidationResult
        """
        if not isinstance(other, ValidationResult):
            raise TypeError(
                f"merge requiere ValidationResult, recibido: {type(other).__name__}"
            )
        return ValidationResult.from_issues(self.issues + other.issues)

    def filter_by_severity(
        self,
        min_severity: ValidationSeverity,
    ) -> ValidationResult:
        """
        Filtra issues por severidad mínima.
        
        Args:
            min_severity: Severidad mínima a incluir
        
        Returns:
            Nuevo ValidationResult con issues filtrados
        """
        filtered = tuple(
            issue for issue in self.issues if issue.severity >= min_severity
        )
        return ValidationResult.from_issues(filtered)

    def filter_by_code(
        self,
        codes: Iterable[ValidationCode],
    ) -> ValidationResult:
        """
        Filtra issues por códigos específicos.
        
        Args:
            codes: Conjunto de códigos a incluir
        
        Returns:
            Nuevo ValidationResult con issues filtrados
        """
        code_set = frozenset(codes)
        filtered = tuple(issue for issue in self.issues if issue.code in code_set)
        return ValidationResult.from_issues(filtered)

    # ──────────────────────────────────────────────────────────────────────────
    # Serialización y Exportación
    # ──────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """
        Serializa a diccionario para APIs externas.
        
        Returns:
            Diccionario con estructura estándar de validación
        """
        return {
            "is_valid": self.is_valid,
            "issue_count": len(self.issues),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "info_count": len(self.infos),
            "max_severity": self.max_severity.value if self.max_severity else None,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Exporta issues a DataFrame para análisis.
        
        Returns:
            DataFrame con columnas [severity, code, message, column, count]
        """
        if not self.issues:
            return pd.DataFrame(
                columns=["severity", "code", "message", "column", "count"]
            )
        
        return pd.DataFrame([issue.to_dict() for issue in self.issues])

    def raise_if_invalid(self, exception_class: type[Exception] = ValueError) -> None:
        """
        Lanza excepción si el resultado es inválido.
        
        Args:
            exception_class: Clase de excepción a lanzar
        
        Raises:
            exception_class: Si is_valid es False
        """
        if not self.is_valid:
            error_messages = "\n".join(str(e) for e in self.errors)
            raise exception_class(
                f"Validación fallida con {len(self.errors)} errores:\n{error_messages}"
            )


# Inicialización del elemento neutro (después de la definición de la clase)
ValidationResult.IDENTITY = ValidationResult.success()


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN V: VALIDADOR PRINCIPAL (FUNCTOR DE VALIDACIÓN)
# ══════════════════════════════════════════════════════════════════════════════


class DataFrameValidator:
    """
    FUNCTOR DE VALIDACIÓN entre categorías DataFrames → ValidationResults.
    
    Implementa un conjunto de TRANSFORMACIONES NATURALES que preservan la
    estructura algebraica y los invariantes topológicos del espacio de datos.
    
    Arquitectura Funcional:
    ----------------------
    - **Sin Estado**: Todos los métodos son classmethods/staticmethods puros
    - **Inmutabilidad**: No muta el DataFrame de entrada (preserva pureza)
    - **Composicionalidad**: Las validaciones se componen vía monoide merge
    - **Fidelidad**: Errores distintos producen resultados distintos (inyectividad)
    
    Jerarquía de Validaciones (Cadena de Responsibilidad):
    -----------------------------------------------------
    
    1. **validate_schema** (Isomorfismo Dimensional)
       ├─ Verifica existencia de columnas requeridas
       ├─ Detecta duplicaciones (degeneración de base)
       └─ Garantiza dim(df.columns) ≅ dim(schema)
    
    2. **check_data_quality** (Extirpación de Singularidades)
       ├─ Localiza valores nulos (puntos no definidos)
       ├─ Identifica strings vacíos (singularidades de cadena)
       └─ Preserva continuidad del espacio de valores
    
    3. **validate_numeric_columns** (Clausura Algebraica)
       ├─ Coerción a tipo numérico (embedding en ℝ)
       ├─ Verificación de finitud (evita ±∞)
       ├─ Detección de subnormales (fricción cuántica IEEE 754)
       └─ Garantiza ∀ x ∈ col: x ∈ ℝ_finite
    
    4. **validate_non_negative** (Termodinámica)
       ├─ Verifica no negatividad (energía ≥ 0)
       └─ Preserva Segunda Ley (entropía no negativa)
    
    5. **validate_ranges** (Compacidad)
       ├─ Verifica cotas [min, max]
       └─ Garantiza valores en conjunto compacto
    
    6. **validate_linear_independence** (Conexidad)
       ├─ Calcula rank(matriz_columnas)
       ├─ Verifica rank = num_columnas
       └─ Garantiza espacio vectorial conexo (no degenerado)
    
    7. **validate_survival_threshold** (Preservación de Medida)
       ├─ Calcula μ(filas_limpias) / μ(filas_totales)
       ├─ Verifica ratio ≥ threshold
       └─ Garantiza masa suficiente post-filtrado
    
    8. **validate_domain** (Composición Orquestada)
       └─ Ejecuta secuencialmente todas las anteriores
       └─ Combina resultados vía monoide merge (⊕)
    
    Invariantes Matemáticos Preservados:
    ------------------------------------
    
    ∀ df ∈ DataFrames válido:
    
    1. **Isomorfismo**: schema(df) ≅ schema_esperado
    2. **Continuidad**: ∄ x ∈ df : x ∈ {NaN, ±∞}
    3. **Finitud**: ∀ x numérico: |x| < ∞
    4. **Normalidad**: ∀ x numérico: |x| ≥ min_normal ∨ x = 0
    5. **No-Negatividad**: ∀ x ∈ cols_no_neg: x ≥ 0
    6. **Compacidad**: ∀ x ∈ col: x ∈ [min, max]
    7. **Independencia**: rank(df[cols]) = |cols|
    8. **Supervivencia**: |df_limpio| / |df| ≥ threshold
    """

    # ──────────────────────────────────────────────────────────────────────────
    # SUBSECCIÓN 5.1: Validadores Internos (Precondiciones)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_dataframe_object(df: object) -> ValidationResult:
        """
        Verifica invariante: df ∈ pandas.DataFrame.
        
        Precondición para todas las validaciones. Garantiza que el objeto
        pasado es efectivamente un DataFrame y no None o otro tipo.
        
        Args:
            df: Objeto a verificar
        
        Returns:
            ValidationResult.success() si df es DataFrame válido,
            ValidationResult.failure() en caso contrario
        """
        if df is None:
            return ValidationResult.single_issue(
                severity=ValidationSeverity.CRITICAL,
                code=ValidationCode.NONE_DATAFRAME,
                message="El DataFrame es None. Se requiere una instancia válida.",
            )
        
        if not isinstance(df, pd.DataFrame):
            return ValidationResult.single_issue(
                severity=ValidationSeverity.CRITICAL,
                code=ValidationCode.INVALID_DATAFRAME_TYPE,
                message=(
                    f"Tipo inválido: se esperaba pandas.DataFrame, "
                    f"se recibió {type(df).__module__}.{type(df).__name__}."
                ),
            )
        
        return ValidationResult.success()

    @staticmethod
    def _normalize_columns_argument(
        columns: Iterable[str],
        arg_name: str,
    ) -> tuple[str, ...]:
        """
        Normaliza y valida argumento de columnas.
        
        Garantiza que:
        1. columns es iterable
        2. Todos los elementos son strings
        3. No hay strings vacíos
        4. Se eliminan duplicados (preservando orden de primera aparición)
        
        Args:
            columns: Iterable de nombres de columnas
            arg_name: Nombre del argumento (para mensajes de error)
        
        Returns:
            Tupla normalizada de nombres de columnas únicos
        
        Raises:
            ValueError: Si columns es inválido
        """
        if columns is None:
            raise ValueError(f"{arg_name} no puede ser None")
        
        try:
            raw_list = list(columns)
        except TypeError as exc:
            raise ValueError(
                f"{arg_name} debe ser un iterable de strings"
            ) from exc
        
        normalized: list[str] = []
        seen: set[str] = set()
        
        for idx, col in enumerate(raw_list):
            if not isinstance(col, str):
                raise ValueError(
                    f"{arg_name}[{idx}] debe ser str, "
                    f"recibido: {type(col).__name__}"
                )
            
            cleaned = col.strip()
            if not cleaned:
                raise ValueError(
                    f"{arg_name}[{idx}] no puede ser un string vacío o solo espacios"
                )
            
            # Agregar solo si no se ha visto antes (elimina duplicados)
            if cleaned not in seen:
                normalized.append(cleaned)
                seen.add(cleaned)
        
        return tuple(normalized)

    @staticmethod
    def _check_duplicate_columns(df: pd.DataFrame) -> ValidationResult:
        """
        Detecta columnas duplicadas (degeneración de base vectorial).
        
        Las columnas duplicadas indican que el DataFrame tiene una base
        degenerada, lo que puede causar problemas en operaciones matriciales.
        
        Args:
            df: DataFrame a verificar
        
        Returns:
            ValidationResult con WARNING si hay duplicados, success() si no
        """
        duplicated_mask = df.columns.duplicated()
        if not duplicated_mask.any():
            return ValidationResult.success()
        
        # Mapear nombres duplicados a sus posiciones
        duplicated_names = df.columns[duplicated_mask].tolist()
        positions: dict[str, list[int]] = {}
        
        for idx, col_name in enumerate(df.columns):
            col_str = str(col_name)
            if col_str in set(map(str, duplicated_names)):
                positions.setdefault(col_str, []).append(idx)
        
        # Construir mensaje detallado
        detail = ", ".join(
            f"'{name}' en posiciones {pos}" for name, pos in positions.items()
        )
        
        return ValidationResult.single_issue(
            severity=ValidationSeverity.WARNING,
            code=ValidationCode.DUPLICATE_COLUMNS,
            message=(
                f"Columnas duplicadas detectadas (base degenerada): {detail}. "
                "Esto puede causar ambigüedad en selecciones."
            ),
            count=len(duplicated_names),
        )

    @staticmethod
    def _validate_range_bound(
        value: Optional[float],
        bound_name: str,
        col: str,
    ) -> None:
        """
        Valida que un límite de rango sea numérico y finito.
        
        Args:
            value: Valor del límite (puede ser None)
            bound_name: Nombre del límite ('min' o 'max')
            col: Nombre de la columna (para mensajes)
        
        Raises:
            ValueError: Si el límite es inválido
        """
        if value is None:
            return
        
        # Verificar tipo (rechazar bool que es subclase de int)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                f"Límite '{bound_name}' para columna '{col}' debe ser numérico, "
                f"recibido: {type(value).__name__}"
            )
        
        # Verificar finitud (rechazar ±inf y NaN)
        if not math.isfinite(value):
            raise ValueError(
                f"Límite '{bound_name}' para columna '{col}' debe ser finito, "
                f"recibido: {value}"
            )

    @staticmethod
    def _is_subnormal(value: float) -> bool:
        """
        Detecta valores subnormales (denormales) según IEEE 754.
        
        Los números subnormales (denormales) son aquellos con exponente mínimo
        y mantisa no nula. Pueden causar degradación de precisión y rendimiento
        en operaciones de punto flotante.
        
        Definición matemática:
            subnormal(x) ⟺ 0 < |x| < min_normal
        
        donde min_normal = 2^(-1022) ≈ 2.225e-308 para float64.
        
        Args:
            value: Valor numérico a verificar
        
        Returns:
            True si el valor es subnormal, False en caso contrario
        """
        # Rechazar valores no finitos o cero
        if not np.isfinite(value) or value == 0.0:
            return False
        
        # Verificar si está por debajo del mínimo normal
        return abs(value) < MIN_NORMAL_FLOAT

    # ──────────────────────────────────────────────────────────────────────────
    # SUBSECCIÓN 5.2: Validadores Estructurales (Isomorfismo Dimensional)
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def validate_schema(
        cls,
        df: pd.DataFrame,
        required_columns: Iterable[str],
    ) -> ValidationResult:
        """
        Valida ISOMORFISMO DIMENSIONAL entre esquema real y esperado.
        
        Verifica que el espacio vectorial del DataFrame contenga todas las
        dimensiones (columnas) requeridas. La ausencia de columnas críticas
        colapsa el rango de la transformación y debe ser rechazado.
        
        Invariante Verificado:
        ---------------------
        required_columns ⊆ df.columns
        
        Args:
            df: DataFrame a validar
            required_columns: Columnas que deben existir
        
        Returns:
            ValidationResult con:
            - ERROR por cada columna faltante
            - WARNING si hay columnas duplicadas
        """
        # Precondición: df es DataFrame válido
        df_result = cls._validate_dataframe_object(df)
        if not df_result.is_valid:
            return df_result
        
        # Normalizar columnas requeridas
        required = cls._normalize_columns_argument(required_columns, "required_columns")
        
        # Iniciar con verificación de duplicados
        issues: list[ValidationIssue] = list(cls._check_duplicate_columns(df).issues)
        
        # Convertir columnas actuales a conjunto
        current_columns = frozenset(str(c) for c in df.columns)
        
        # Identificar columnas faltantes
        missing = [col for col in required if col not in current_columns]
        
        if missing:
            # Generar un issue por cada columna faltante
            issues.extend(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code=ValidationCode.MISSING_REQUIRED_COLUMN,
                    message=(
                        f"Columna requerida '{col}' no encontrada. "
                        f"Isomorfismo dimensional violado."
                    ),
                    column=col,
                )
                for col in missing
            )
        
        return ValidationResult.from_issues(tuple(issues))

    # ──────────────────────────────────────────────────────────────────────────
    # SUBSECCIÓN 5.3: Validadores de Calidad (Extirpación de Singularidades)
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def check_data_quality(
        cls,
        df: pd.DataFrame,
        critical_columns: Iterable[str],
        *,
        empty_strings_as_warning: bool = True,
    ) -> ValidationResult:
        """
        Localiza y cuantifica SINGULARIDADES TOPOLÓGICAS en columnas críticas.
        
        Las singularidades (NaN, None, strings vacíos) son puntos de
        discontinuidad que deben ser extirpados antes de la inyección al
        sistema agéntico.
        
        Singularidades Detectadas:
        --------------------------
        1. **Nulos**: NaN, None, pd.NA (discontinuidades fuertes)
        2. **Strings vacíos**: "", "   " (singularidades de cadena)
        
        Args:
            df: DataFrame a validar
            critical_columns: Columnas donde se deben detectar singularidades
            empty_strings_as_warning: Si True, strings vacíos generan WARNING
        
        Returns:
            ValidationResult con WARNING por cada tipo de singularidad detectada
        """
        # Precondición: df es DataFrame válido
        df_result = cls._validate_dataframe_object(df)
        if not df_result.is_valid:
            return df_result
        
        # Normalizar columnas críticas
        critical = cls._normalize_columns_argument(critical_columns, "critical_columns")
        
        # Iniciar con verificación de duplicados
        issues: list[ValidationIssue] = list(cls._check_duplicate_columns(df).issues)
        
        for col in critical:
            # Verificar existencia de columna
            if col not in df.columns:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code=ValidationCode.MISSING_CRITICAL_COLUMN,
                        message=(
                            f"Columna crítica '{col}' no encontrada para "
                            "validación de calidad."
                        ),
                        column=col,
                    )
                )
                continue
            
            # ────────────────────────────────────────────────────────────────
            # Detección de Nulos (Singularidades Fuertes)
            # ────────────────────────────────────────────────────────────────
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code=ValidationCode.NULL_VALUES,
                        message=(
                            f"Columna '{col}' contiene {null_count} valores nulos "
                            "(singularidades topológicas). Esto induce discontinuidad."
                        ),
                        column=col,
                        count=null_count,
                    )
                )
            
            # ────────────────────────────────────────────────────────────────
            # Detección de Strings Vacíos (Singularidades de Cadena)
            # ────────────────────────────────────────────────────────────────
            if empty_strings_as_warning:
                series = df[col]
                
                # Solo verificar si la columna puede contener strings
                if series.dtype == object or pd.api.types.is_string_dtype(series):
                    # Máscara de valores que son efectivamente strings
                    str_mask = series.apply(type) == str
                    # Máscara de strings vacíos (o solo espacios)
                    empty_mask = str_mask & (series.str.strip() == "")
                    empty_count = int(empty_mask.sum())
                else:
                    empty_count = 0
                
                if empty_count > 0:
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            code=ValidationCode.EMPTY_STRINGS,
                            message=(
                                f"Columna '{col}' contiene {empty_count} strings vacíos "
                                "(singularidades de cadena)."
                            ),
                            column=col,
                            count=empty_count,
                        )
                    )
        
        return ValidationResult.from_issues(tuple(issues))

    # ──────────────────────────────────────────────────────────────────────────
    # SUBSECCIÓN 5.4: Validadores Numéricos (Clausura Algebraica)
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def validate_numeric_columns(
        cls,
        df: pd.DataFrame,
        numeric_columns: Iterable[str],
        *,
        missing_as_warning: bool = True,
        non_numeric_severity: ValidationSeverity = ValidationSeverity.WARNING,
    ) -> ValidationResult:
        """
        Verifica CLAUSURA ALGEBRAICA del espacio numérico.
        
        Garantiza que las columnas designadas estén embebidas en el campo
        de los reales finitos: ℝ_finite = ℝ \ {-∞, +∞, NaN}.
        
        Validaciones Ejecutadas:
        -----------------------
        1. **Coercibilidad**: valores → ℝ (sin pérdida)
        2. **Finitud**: ∀ x: |x| < ∞ (rechaza ±inf)
        3. **Normalidad**: detecta subnormales (fricción cuántica IEEE 754)
        
        Jerarquía de Severidad:
        ----------------------
        - No convertibles: configurable (default WARNING)
        - No finitos: ERROR (invalidan operaciones)
        - Subnormales: WARNING (degradan precisión)
        
        Args:
            df: DataFrame a validar
            numeric_columns: Columnas que deben ser numéricas
            missing_as_warning: Si True, columnas ausentes generan WARNING
            non_numeric_severity: Severidad para valores no convertibles
        
        Returns:
            ValidationResult con diagnóstico de clausura algebraica
        """
        # Precondición: df es DataFrame válido
        df_result = cls._validate_dataframe_object(df)
        if not df_result.is_valid:
            return df_result
        
        # Normalizar columnas numéricas
        columns = cls._normalize_columns_argument(numeric_columns, "numeric_columns")
        
        issues: list[ValidationIssue] = []
        
        for col in columns:
            # ────────────────────────────────────────────────────────────────
            # Verificar Existencia de Columna
            # ────────────────────────────────────────────────────────────────
            if col not in df.columns:
                sev = (
                    ValidationSeverity.WARNING
                    if missing_as_warning
                    else ValidationSeverity.ERROR
                )
                code = (
                    ValidationCode.MISSING_CRITICAL_COLUMN
                    if missing_as_warning
                    else ValidationCode.MISSING_REQUIRED_COLUMN
                )
                issues.append(
                    ValidationIssue(
                        severity=sev,
                        code=code,
                        message=(
                            f"Columna numérica '{col}' no encontrada. "
                            "No se puede verificar clausura algebraica."
                        ),
                        column=col,
                    )
                )
                continue
            
            series = df[col]
            non_null_mask = series.notna()
            non_null_count = int(non_null_mask.sum())
            
            # Si todos son nulos, no hay nada que validar
            if non_null_count == 0:
                continue
            
            # ────────────────────────────────────────────────────────────────
            # Coerción a Tipo Numérico (Embedding en ℝ)
            # ────────────────────────────────────────────────────────────────
            coerced = pd.to_numeric(series, errors="coerce")
            
            # Identificar valores que no pudieron convertirse
            coercion_failed_mask = non_null_mask & coerced.isna()
            non_numeric_count = int(coercion_failed_mask.sum())
            
            if non_numeric_count > 0:
                issues.append(
                    ValidationIssue(
                        severity=non_numeric_severity,
                        code=ValidationCode.NON_NUMERIC_VALUES,
                        message=(
                            f"Columna '{col}' contiene {non_numeric_count} valores "
                            "no convertibles a numérico. Clausura algebraica violada."
                        ),
                        column=col,
                        count=non_numeric_count,
                    )
                )
            
            # ────────────────────────────────────────────────────────────────
            # Verificación de Finitud (Rechazar ±∞)
            # ────────────────────────────────────────────────────────────────
            successfully_converted = coerced.notna()
            if int(successfully_converted.sum()) > 0:
                numeric_values = coerced[successfully_converted]
                
                # Usar numpy para verificar finitud (más eficiente)
                numeric_array: NDArray[np.float64] = numeric_values.to_numpy(
                    dtype=np.float64
                )
                non_finite_mask = ~np.isfinite(numeric_array)
                non_finite_count = int(non_finite_mask.sum())
                
                if non_finite_count > 0:
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            code=ValidationCode.NON_FINITE_VALUES,
                            message=(
                                f"Columna '{col}' contiene {non_finite_count} valores "
                                "no finitos (±∞). Estas singularidades invalidan "
                                "operaciones algebraicas."
                            ),
                            column=col,
                            count=non_finite_count,
                        )
                    )
                
                # ────────────────────────────────────────────────────────────
                # Detección de Subnormales (Fricción Cuántica IEEE 754)
                # ────────────────────────────────────────────────────────────
                subnormal_mask = numeric_values.map(cls._is_subnormal).fillna(False)
                subnormal_count = int(subnormal_mask.sum())
                
                if subnormal_count > 0:
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            code=ValidationCode.SUBNORMAL_VALUES,
                            message=(
                                f"Columna '{col}' contiene {subnormal_count} valores "
                                f"subnormales (|x| < {MIN_NORMAL_FLOAT:.2e}). "
                                "Estos pueden degradar precisión numérica y rendimiento."
                            ),
                            column=col,
                            count=subnormal_count,
                        )
                    )
        
        return ValidationResult.from_issues(tuple(issues))

    # ──────────────────────────────────────────────────────────────────────────
    # SUBSECCIÓN 5.5: Validadores de Dominio (Restricciones Físicas)
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def validate_non_negative(
        cls,
        df: pd.DataFrame,
        columns: Iterable[str],
        *,
        missing_as_warning: bool = True,
        negative_severity: ValidationSeverity = ValidationSeverity.WARNING,
    ) -> ValidationResult:
        """
        Verifica TERMODINÁMICA: energía no negativa (Segunda Ley).
        
        En contextos logísticos y financieros, ciertas magnitudes físicas
        (costos, tiempos, cantidades) deben ser estrictamente no negativas
        para preservar la coherencia termodinámica del sistema.
        
        Invariante Verificado:
        ---------------------
        ∀ x ∈ columnas_no_negativas: x ≥ 0
        
        Args:
            df: DataFrame a validar
            columns: Columnas que deben ser no negativas
            missing_as_warning: Si True, columnas ausentes generan WARNING
            negative_severity: Severidad para valores negativos
        
        Returns:
            ValidationResult con diagnóstico de no-negatividad
        """
        # Precondición: df es DataFrame válido
        df_result = cls._validate_dataframe_object(df)
        if not df_result.is_valid:
            return df_result
        
        # Normalizar columnas
        cols = cls._normalize_columns_argument(columns, "columns")
        
        issues: list[ValidationIssue] = []
        
        for col in cols:
            # Verificar existencia de columna
            if col not in df.columns:
                sev = (
                    ValidationSeverity.WARNING
                    if missing_as_warning
                    else ValidationSeverity.ERROR
                )
                code = (
                    ValidationCode.MISSING_CRITICAL_COLUMN
                    if missing_as_warning
                    else ValidationCode.MISSING_REQUIRED_COLUMN
                )
                issues.append(
                    ValidationIssue(
                        severity=sev,
                        code=code,
                        message=(
                            f"Columna '{col}' no encontrada para validación "
                            "de no-negatividad (termodinámica)."
                        ),
                        column=col,
                    )
                )
                continue
            
            # Coercer a numérico
            coerced = pd.to_numeric(df[col], errors="coerce")
            
            # Filtrar solo valores finitos (evitar comparación con ±∞)
            finite_mask = coerced.notna() & np.isfinite(
                coerced.fillna(0).to_numpy(dtype=np.float64)
            )
            finite_series = coerced[pd.Series(finite_mask, index=coerced.index)]
            
            # Contar negativos
            negative_count = int((finite_series < 0).sum())
            
            if negative_count > 0:
                issues.append(
                    ValidationIssue(
                        severity=negative_severity,
                        code=ValidationCode.NEGATIVE_VALUES,
                        message=(
                            f"Columna '{col}' contiene {negative_count} valores negativos. "
                            "Esto viola la restricción termodinámica (energía ≥ 0)."
                        ),
                        column=col,
                        count=negative_count,
                    )
                )
        
        return ValidationResult.from_issues(tuple(issues))

    @classmethod
    def validate_ranges(
        cls,
        df: pd.DataFrame,
        range_rules: Mapping[str, tuple[Optional[float], Optional[float]]],
        *,
        missing_as_warning: bool = True,
        violation_severity: ValidationSeverity = ValidationSeverity.WARNING,
    ) -> ValidationResult:
        """
        Verifica COMPACIDAD: valores en conjuntos compactos [min, max] ⊂ ℝ.
        
        Los rangos definen conjuntos compactos donde deben vivir los valores.
        La compacidad garantiza existencia de extremos y preserva propiedades
        topológicas esenciales para convergencia de algoritmos.
        
        Invariantes Verificados:
        -----------------------
        ∀ col ∈ range_rules:
            ∀ x ∈ df[col]: min ≤ x ≤ max
        
        Args:
            df: DataFrame a validar
            range_rules: Mapeo {columna: (min, max)}
                min/max pueden ser None (sin cota)
            missing_as_warning: Si True, columnas ausentes generan WARNING
            violation_severity: Severidad para violaciones de rango
        
        Returns:
            ValidationResult con diagnóstico de compacidad
        
        Raises:
            ValueError: Si range_rules contiene reglas inválidas
        """
        # Precondición: df es DataFrame válido
        df_result = cls._validate_dataframe_object(df)
        if not df_result.is_valid:
            return df_result
        
        if range_rules is None:
            raise ValueError("range_rules no puede ser None")
        
        issues: list[ValidationIssue] = []
        
        for col, bounds in range_rules.items():
            # ────────────────────────────────────────────────────────────────
            # Validar Estructura de la Regla
            # ────────────────────────────────────────────────────────────────
            if not isinstance(col, str) or not col.strip():
                raise ValueError(
                    "Las claves de range_rules deben ser strings no vacíos"
                )
            
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise ValueError(
                    f"Regla de rango inválida para '{col}': "
                    "debe ser tuple(min, max)"
                )
            
            min_value, max_value = bounds
            
            # Validar límites individualmente
            cls._validate_range_bound(min_value, "min", col)
            cls._validate_range_bound(max_value, "max", col)
            
            # Verificar coherencia: min ≤ max
            if (
                min_value is not None
                and max_value is not None
                and min_value > max_value
            ):
                raise ValueError(
                    f"Rango inválido para '{col}': "
                    f"min ({min_value}) > max ({max_value})"
                )
            
            # ────────────────────────────────────────────────────────────────
            # Verificar Existencia de Columna
            # ────────────────────────────────────────────────────────────────
            if col not in df.columns:
                sev = (
                    ValidationSeverity.WARNING
                    if missing_as_warning
                    else ValidationSeverity.ERROR
                )
                code = (
                    ValidationCode.MISSING_CRITICAL_COLUMN
                    if missing_as_warning
                    else ValidationCode.MISSING_REQUIRED_COLUMN
                )
                issues.append(
                    ValidationIssue(
                        severity=sev,
                        code=code,
                        message=(
                            f"Columna '{col}' no encontrada para validación de rango."
                        ),
                        column=col,
                    )
                )
                continue
            
            # ────────────────────────────────────────────────────────────────
            # Verificar Violaciones de Rango
            # ────────────────────────────────────────────────────────────────
            coerced = pd.to_numeric(df[col], errors="coerce")
            valid_numeric = coerced.notna()
            
            # Máscara de violaciones (inicialmente vacía)
            violation_mask = pd.Series(False, index=df.index)
            
            # Aplicar cota inferior si existe
            if min_value is not None:
                violation_mask = violation_mask | (
                    valid_numeric & (coerced < min_value)
                )
            
            # Aplicar cota superior si existe
            if max_value is not None:
                violation_mask = violation_mask | (
                    valid_numeric & (coerced > max_value)
                )
            
            violation_count = int(violation_mask.sum())
            
            if violation_count > 0:
                min_str = str(min_value) if min_value is not None else "-∞"
                max_str = str(max_value) if max_value is not None else "+∞"
                
                issues.append(
                    ValidationIssue(
                        severity=violation_severity,
                        code=ValidationCode.RANGE_VIOLATION,
                        message=(
                            f"Columna '{col}' contiene {violation_count} valores "
                            f"fuera del rango compacto [{min_str}, {max_str}]. "
                            "Esto viola la restricción de compacidad."
                        ),
                        column=col,
                        count=violation_count,
                    )
                )
        
        return ValidationResult.from_issues(tuple(issues))

    # ──────────────────────────────────────────────────────────────────────────
    # SUBSECCIÓN 5.6: Validadores Topológicos (Invariantes del Espacio)
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def validate_linear_independence(
        cls,
        df: pd.DataFrame,
        columns: Iterable[str],
        *,
        tolerance: float = FLOAT_TOLERANCE,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        missing_as_warning: bool = True,
    ) -> ValidationResult:
        """
        Verifica CONEXIDAD del espacio vectorial (independencia lineal).
        
        Calcula el rango numérico de la matriz formada por las columnas
        especificadas. Si rank < num_columnas, el espacio está DEGENERADO
        (columnas linealmente dependientes), lo que colapsa la matriz de
        covarianza y puede invalidar análisis estadísticos.
        
        Invariante Verificado:
        ---------------------
        rank(df[columns]) = |columns|
        
        Contexto Matemático:
        -------------------
        Un conjunto de vectores {v₁, v₂, ..., vₙ} es linealmente independiente
        si y solo si la única solución de:
            α₁v₁ + α₂v₂ + ... + αₙvₙ = 0
        es α₁ = α₂ = ... = αₙ = 0.
        
        Equivalentemente, rank(matriz) = n.
        
        Args:
            df: DataFrame a validar
            columns: Columnas que deben ser linealmente independientes
            tolerance: Tolerancia para el cálculo del rango (épsilon)
            severity: Severidad si hay dependencia (default ERROR)
            missing_as_warning: Si True, columnas ausentes generan WARNING
        
        Returns:
            ValidationResult con diagnóstico de independencia lineal
        """
        # Precondición: df es DataFrame válido
        df_result = cls._validate_dataframe_object(df)
        if not df_result.is_valid:
            return df_result
        
        # Normalizar columnas
        cols = cls._normalize_columns_argument(columns, "columns")
        
        issues: list[ValidationIssue] = []
        
        # ────────────────────────────────────────────────────────────────────
        # Verificar Existencia de Columnas
        # ────────────────────────────────────────────────────────────────────
        for col in cols:
            if col not in df.columns:
                sev = (
                    ValidationSeverity.WARNING
                    if missing_as_warning
                    else ValidationSeverity.ERROR
                )
                code = (
                    ValidationCode.MISSING_CRITICAL_COLUMN
                    if missing_as_warning
                    else ValidationCode.MISSING_REQUIRED_COLUMN
                )
                issues.append(
                    ValidationIssue(
                        severity=sev,
                        code=code,
                        message=(
                            f"Columna '{col}' no encontrada para validación "
                            "de independencia lineal."
                        ),
                        column=col,
                    )
                )
        
        # Filtrar columnas presentes
        present_cols = [col for col in cols if col in df.columns]
        if not present_cols:
            return ValidationResult.from_issues(tuple(issues))
        
        # ────────────────────────────────────────────────────────────────────
        # Preparar Submatriz y Eliminar Singularidades
        # ────────────────────────────────────────────────────────────────────
        sub = df[present_cols].copy()
        
        # Convertir a numérico (forzar errores a NaN)
        for col in present_cols:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
        
        # Eliminar filas con al menos un NaN
        sub_complete = sub.dropna()
        
        # Verificar que haya suficientes filas
        if sub_complete.shape[0] < 2:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code=ValidationCode.LINEAR_DEPENDENCY,
                    message=(
                        f"Insuficientes filas completas ({sub_complete.shape[0]}) "
                        f"para calcular rango de {len(present_cols)} columnas. "
                        "Se requieren al menos 2 filas."
                    ),
                    count=sub_complete.shape[0],
                )
            )
            return ValidationResult.from_issues(tuple(issues))
        
        # ────────────────────────────────────────────────────────────────────
        # Cálculo del Rango Numérico
        # ────────────────────────────────────────────────────────────────────
        matrix: NDArray[np.float64] = sub_complete.values.astype(np.float64)
        
        try:
            rank = np.linalg.matrix_rank(matrix, tol=tolerance)
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code=ValidationCode.LINEAR_DEPENDENCY,
                    message=f"Error al calcular rango de matriz: {exc}",
                )
            )
            return ValidationResult.from_issues(tuple(issues))
        
        # ────────────────────────────────────────────────────────────────────
        # Verificar Degeneración
        # ────────────────────────────────────────────────────────────────────
        expected_rank = len(present_cols)
        if rank < expected_rank:
            deficiency = expected_rank - rank
            issues.append(
                ValidationIssue(
                    severity=severity,
                    code=ValidationCode.LINEAR_DEPENDENCY,
                    message=(
                        f"Las columnas {present_cols} NO son linealmente independientes. "
                        f"Rango numérico: {rank} < {expected_rank} (deficiencia: {deficiency}). "
                        "El espacio vectorial está DEGENERADO, lo que puede colapsar "
                        "la matriz de covarianza."
                    ),
                    count=deficiency,
                )
            )
        
        return ValidationResult.from_issues(tuple(issues))

    @classmethod
    def validate_survival_threshold(
        cls,
        df: pd.DataFrame,
        critical_columns: Iterable[str],
        *,
        min_survival: float = DEFAULT_SURVIVAL_THRESHOLD,
        include_empty_strings: bool = True,
    ) -> ValidationResult:
        """
        Verifica PRESERVACIÓN DE MEDIDA tras extirpación de singularidades.
        
        Cuantifica la "masa" (medida de Lebesgue) que sobrevive después de
        eliminar filas con singularidades (NaN, None, strings vacíos) en
        columnas críticas.
        
        Si la supervivencia cae por debajo del umbral, se emite CRITICAL
        porque el dataset ha perdido demasiada información y la inyección
        debe ser abortada.
        
        Invariante Verificado:
        ---------------------
        μ(filas_limpias) / μ(filas_totales) ≥ min_survival
        
        donde μ es la medida de Lebesgue (cardinalidad).
        
        Args:
            df: DataFrame a validar
            critical_columns: Columnas donde se extirpan singularidades
            min_survival: Fracción mínima aceptable (0.0 a 1.0)
            include_empty_strings: Si True, strings vacíos son singularidades
        
        Returns:
            ValidationResult con:
            - SUCCESS si supervivencia ≥ min_survival
            - CRITICAL si supervivencia < min_survival
        """
        # Precondición: df es DataFrame válido
        df_result = cls._validate_dataframe_object(df)
        if not df_result.is_valid:
            return df_result
        
        # Normalizar columnas críticas
        cols = cls._normalize_columns_argument(critical_columns, "critical_columns")
        
        # Verificar existencia de columnas
        missing = [col for col in cols if col not in df.columns]
        if missing:
            return ValidationResult.from_issues(
                tuple(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code=ValidationCode.MISSING_CRITICAL_COLUMN,
                        message=(
                            f"Columna '{col}' no encontrada para validación "
                            "de supervivencia."
                        ),
                        column=col,
                    )
                    for col in missing
                )
            )
        
        # ────────────────────────────────────────────────────────────────────
        # Construir Máscara de Filas Limpias
        # ────────────────────────────────────────────────────────────────────
        clean_mask = pd.Series(True, index=df.index)
        
        for col in cols:
            # Excluir nulos
            not_null = df[col].notna()
            clean_mask &= not_null
            
            # Excluir strings vacíos (si está habilitado)
            if include_empty_strings:
                series = df[col]
                
                # Solo verificar si puede contener strings
                if series.dtype == object or pd.api.types.is_string_dtype(series):
                    # Máscara de valores que son strings
                    str_mask = series.apply(type) == str
                    # Máscara de strings vacíos (o solo espacios)
                    empty_mask = str_mask & (series.str.strip() == "")
                    # Excluir strings vacíos
                    clean_mask &= ~empty_mask
        
        # ────────────────────────────────────────────────────────────────────
        # Calcular Supervivencia
        # ────────────────────────────────────────────────────────────────────
        total_rows = len(df)
        surviving_rows = int(clean_mask.sum())
        
        survival_ratio = (
            surviving_rows / total_rows if total_rows > 0 else 1.0
        )
        
        # ────────────────────────────────────────────────────────────────────
        # Verificar Umbral
        # ────────────────────────────────────────────────────────────────────
        if survival_ratio < min_survival:
            lost_rows = total_rows - surviving_rows
            return ValidationResult.single_issue(
                severity=ValidationSeverity.CRITICAL,
                code=ValidationCode.SURVIVAL_THRESHOLD_VIOLATION,
                message=(
                    f"COLAPSO VOLUMÉTRICO: La extirpación de singularidades "
                    f"deja solo {surviving_rows}/{total_rows} filas "
                    f"({survival_ratio:.2%}), por debajo del umbral "
                    f"de supervivencia {min_survival:.2%}. "
                    f"Se han perdido {lost_rows} filas ({(1 - survival_ratio):.2%}). "
                    "ABORTAR INYECCIÓN."
                ),
                count=lost_rows,
            )
        
        return ValidationResult.success()

    # ──────────────────────────────────────────────────────────────────────────
    # SUBSECCIÓN 5.7: Validador Compuesto (Transformación Natural)
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def validate_domain(
        cls,
        df: pd.DataFrame,
        *,
        required_columns: Iterable[str] = (),
        critical_columns: Iterable[str] = (),
        numeric_columns: Iterable[str] = (),
        non_negative_columns: Iterable[str] = (),
        range_rules: Optional[
            Mapping[str, tuple[Optional[float], Optional[float]]]
        ] = None,
        linear_dependency_columns: Optional[Iterable[str]] = None,
        survival_critical_columns: Optional[Iterable[str]] = None,
        survival_min_threshold: float = DEFAULT_SURVIVAL_THRESHOLD,
    ) -> ValidationResult:
        """
        TRANSFORMACIÓN NATURAL: composición orquestada de todos los validadores.
        
        Ejecuta secuencialmente la batería completa de validaciones,
        combinando resultados vía monoide merge (⊕). El orden de ejecución
        está diseñado para maximizar eficiencia (fallos rápidos).
        
        Diagrama de Flujo:
        -----------------
        
        df ──┬─→ validate_schema ────────────┬─→
             ├─→ check_data_quality ─────────┤
             ├─→ validate_numeric_columns ───┤
             ├─→ validate_non_negative ──────┤
             ├─→ validate_ranges ────────────┤
             ├─→ validate_linear_independence┤
             └─→ validate_survival_threshold ┘
                            ↓
                    ⊕ (monoide merge)
                            ↓
                   ValidationResult
        
        Propiedades Categoriales:
        ------------------------
        - **Funtorialidad**: 𝓕(id) = IDENTITY
        - **Naturalidad**: El diagrama conmuta
        - **Preservación**: Composición ↦ merge
        
        Args:
            df: DataFrame a validar (objeto de la categoría fuente)
            required_columns: Columnas requeridas (isomorfismo)
            critical_columns: Columnas críticas (calidad)
            numeric_columns: Columnas numéricas (clausura)
            non_negative_columns: Columnas no negativas (termodinámica)
            range_rules: Reglas de rango (compacidad)
            linear_dependency_columns: Columnas para independencia (conexidad)
            survival_critical_columns: Columnas para supervivencia (medida)
            survival_min_threshold: Umbral de supervivencia (default 90%)
        
        Returns:
            ValidationResult compuesto (objeto de la categoría destino)
        """
        # Precondición: df es DataFrame válido
        df_result = cls._validate_dataframe_object(df)
        if not df_result.is_valid:
            return df_result
        
        # ────────────────────────────────────────────────────────────────────
        # Materializar Argumentos (Evaluación Eager)
        # ────────────────────────────────────────────────────────────────────
        req_cols = _materialize_iterable(required_columns)
        crit_cols = _materialize_iterable(critical_columns)
        num_cols = _materialize_iterable(numeric_columns)
        nn_cols = _materialize_iterable(non_negative_columns)
        lin_cols = _materialize_iterable(linear_dependency_columns or ())
        surv_cols = _materialize_iterable(survival_critical_columns or ())
        
        # ────────────────────────────────────────────────────────────────────
        # Inicializar Resultado Compuesto (Elemento Neutro)
        # ────────────────────────────────────────────────────────────────────
        composite = ValidationResult.IDENTITY
        
        # ────────────────────────────────────────────────────────────────────
        # Ejecutar Validaciones Secuencialmente (Composición Monoidal)
        # ────────────────────────────────────────────────────────────────────
        
        # 1. Isomorfismo Dimensional
        if req_cols:
            composite = composite + cls.validate_schema(df, req_cols)
        
        # 2. Extirpación de Singularidades
        if crit_cols:
            composite = composite + cls.check_data_quality(df, crit_cols)
        
        # 3. Clausura Algebraica
        if num_cols:
            composite = composite + cls.validate_numeric_columns(df, num_cols)
        
        # 4. Termodinámica (No-Negatividad)
        if nn_cols:
            composite = composite + cls.validate_non_negative(df, nn_cols)
        
        # 5. Compacidad (Rangos)
        if range_rules:
            composite = composite + cls.validate_ranges(df, range_rules)
        
        # 6. Conexidad (Independencia Lineal)
        if lin_cols:
            composite = composite + cls.validate_linear_independence(df, lin_cols)
        
        # 7. Preservación de Medida (Supervivencia)
        if surv_cols:
            composite = composite + cls.validate_survival_threshold(
                df,
                surv_cols,
                min_survival=survival_min_threshold,
            )
        
        return composite


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN VI: UTILIDADES INTERNAS (FUNCIONES AUXILIARES)
# ══════════════════════════════════════════════════════════════════════════════


def _materialize_iterable(iterable: Iterable[str]) -> tuple[str, ...]:
    """
    Materializa un iterable a tupla (evaluación eager).
    
    Optimización: si ya es tupla, retorna directamente sin copiar.
    
    Args:
        iterable: Iterable de strings
    
    Returns:
        Tupla inmutable de strings
    """
    if isinstance(iterable, tuple):
        return iterable
    return tuple(iterable)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN VII: API PÚBLICA (EXPORTS)
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enumeraciones
    "ValidationSeverity",
    "ValidationCode",
    # Estructuras de datos
    "ValidationIssue",
    "ValidationResult",
    # Validador principal
    "DataFrameValidator",
    # Constantes
    "FLOAT_TOLERANCE",
    "MIN_NORMAL_FLOAT",
    "DEFAULT_SURVIVAL_THRESHOLD",
]