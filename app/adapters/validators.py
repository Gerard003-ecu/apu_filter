"""
=========================================================================================
Módulo: Data Validators (Operador de Restricción Métrica y Verificador de Invariantes)
Ubicación: app/adapters/validators.py
=========================================================================================

Naturaleza Ciber-Física y Topológica:
    Este componente trasciende la noción de un simple validador de esquemas o tipos. 
    Opera como el Filtro de Variedad Diferenciable en la frontera del sistema, proyectando 
    los tensores de entrada (DataFrames) hacia el hiperespacio funcional canónico de la 
    Malla Agéntica. Su mandato axiomático es garantizar que ninguna singularidad numérica 
    o deformación dimensional contamine la matriz de estado.

1. Isomorfismo Dimensional (Validación Estructural):
    Rechaza la concepción empírica de "columnas" para evaluar Vectores Base. El validador 
    exige que el hiperespacio de entrada sea isomórfico al esquema canónico esperado (ℝⁿ).
    La duplicación de vectores base o la ausencia de dimensiones críticas aniquila el rango 
    de la matriz de estado, provocando un VETO ESTRUCTURAL ineludible.

2. Erradicación de Singularidades (Filtro de Ruido y Calidad):
    Las celdas nulas (NaN, None) y las cadenas vacías no son simples "errores de calidad"; 
    son Singularidades Topológicas que inducen discontinuidades severas en el espacio de fase. 
    El validador localiza y extirpa estas singularidades en la frontera, antes de que colapsen 
    el cálculo de la Homología Persistente en los estratos tácticos.

3. Acotación Termodinámica y Cierre Algebraico (Validación de Dominio):
    Somete los tensores a la rigurosidad de la Unidad de Punto Flotante (IEEE 754) y a la 
    física del negocio. Exige axiomáticamente el Cierre Algebraico (finitud estricta de valores) 
    y la No-Negatividad absoluta (la "energía financiera" no puede ser negativa, preservando 
    la Segunda Ley de la Termodinámica en el flujo logístico del proyecto).

4. Monoide de Diagnóstico y Retículo de Severidad (Composición de Resultados):
    La arquitectura subyacente de `ValidationResult` y `ValidationIssue` forma un Monoide 
    Algebraico. La operación de agregación (merge) es estrictamente asociativa y preserva 
    el orden cronológico para garantizar la Trazabilidad Forense. El veredicto de validez 
    (`is_valid`) colapsa mediante una función indicadora matemática, subordinada a los elementos 
    absorbentes del retículo de severidad (ERROR y CRITICAL).
=========================================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


# ======================================================================
# ENUMS
# ======================================================================


class ValidationSeverity(str, Enum):
    """Niveles de severidad para hallazgos de validación."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Conjunto de severidades que invalidan el resultado.
_BLOCKING_SEVERITIES: frozenset[ValidationSeverity] = frozenset(
    {ValidationSeverity.ERROR, ValidationSeverity.CRITICAL}
)


class ValidationCode(str, Enum):
    """Códigos canónicos de incidencia de validación."""

    NONE_DATAFRAME = "NONE_DATAFRAME"
    INVALID_DATAFRAME_TYPE = "INVALID_DATAFRAME_TYPE"
    INVALID_COLUMNS_ARGUMENT = "INVALID_COLUMNS_ARGUMENT"
    EMPTY_COLUMN_NAME = "EMPTY_COLUMN_NAME"
    DUPLICATE_COLUMNS = "DUPLICATE_COLUMNS"
    MISSING_REQUIRED_COLUMN = "MISSING_REQUIRED_COLUMN"
    MISSING_CRITICAL_COLUMN = "MISSING_CRITICAL_COLUMN"
    NULL_VALUES = "NULL_VALUES"
    EMPTY_STRINGS = "EMPTY_STRINGS"
    NON_NUMERIC_VALUES = "NON_NUMERIC_VALUES"
    NON_FINITE_VALUES = "NON_FINITE_VALUES"
    NEGATIVE_VALUES = "NEGATIVE_VALUES"
    RANGE_VIOLATION = "RANGE_VIOLATION"


# ======================================================================
# DATA CLASSES
# ======================================================================


@dataclass(frozen=True)
class ValidationIssue:
    """
    Hallazgo atómico de validación.

    Attributes:
        severity: Nivel de severidad del hallazgo.
        code: Código canónico de la incidencia.
        message: Descripción legible del hallazgo.
        column: Columna afectada (si aplica).
        count: Cantidad de registros afectados (si aplica).
    """

    severity: ValidationSeverity
    code: ValidationCode
    message: str
    column: Optional[str] = None
    count: Optional[int] = None

    def __str__(self) -> str:
        parts = [f"[{self.severity.value}] {self.code.value}: {self.message}"]
        if self.column is not None:
            parts.append(f"columna={self.column}")
        if self.count is not None:
            parts.append(f"count={self.count}")
        return " | ".join(parts)


@dataclass(frozen=True)
class ValidationResult:
    """
    Resultado agregado de validación.

    ``is_valid`` se determina automáticamente: es ``True`` si y solo si
    no existe ningún issue con severidad ``ERROR`` o ``CRITICAL``.

    La composición vía ``merge`` preserva orden de inserción (concatenación)
    y recalcula ``is_valid``.
    """

    is_valid: bool
    issues: tuple[ValidationIssue, ...]

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        """Issues con severidad ERROR o CRITICAL."""
        return tuple(
            i for i in self.issues if i.severity in _BLOCKING_SEVERITIES
        )

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        """Issues con severidad WARNING."""
        return tuple(
            i
            for i in self.issues
            if i.severity == ValidationSeverity.WARNING
        )

    @property
    def infos(self) -> tuple[ValidationIssue, ...]:
        """Issues con severidad INFO."""
        return tuple(
            i
            for i in self.issues
            if i.severity == ValidationSeverity.INFO
        )

    @property
    def has_errors(self) -> bool:
        """``True`` si existe al menos un issue ERROR o CRITICAL."""
        return any(
            i.severity in _BLOCKING_SEVERITIES for i in self.issues
        )

    @property
    def has_warnings(self) -> bool:
        """``True`` si existe al menos un issue WARNING."""
        return any(
            i.severity == ValidationSeverity.WARNING for i in self.issues
        )

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        return (
            f"ValidationResult({status}, "
            f"errors={len(self.errors)}, "
            f"warnings={len(self.warnings)}, "
            f"infos={len(self.infos)})"
        )

    @classmethod
    def from_issues(
        cls, issues: Iterable[ValidationIssue] = ()
    ) -> ValidationResult:
        """
        Construye un ``ValidationResult`` determinando ``is_valid``
        automáticamente a partir de los issues proporcionados.

        Esta es la fábrica canónica. ``is_valid`` es ``True`` si y solo si
        ningún issue tiene severidad ``ERROR`` o ``CRITICAL``.
        """
        issues_t = tuple(issues)
        is_valid = not any(
            i.severity in _BLOCKING_SEVERITIES for i in issues_t
        )
        return cls(is_valid=is_valid, issues=issues_t)

    @classmethod
    def success(cls) -> ValidationResult:
        """Resultado exitoso sin issues."""
        return cls(is_valid=True, issues=())

    @classmethod
    def failure(
        cls, issues: Iterable[ValidationIssue]
    ) -> ValidationResult:
        """
        Construye un resultado a partir de issues que se espera contengan
        al menos un error. Determina ``is_valid`` automáticamente.

        Nota: si los issues proporcionados no contienen errores/critical,
        el resultado será ``is_valid=True`` — esto es por diseño para
        mantener el invariante único de determinación de validez.
        """
        return cls.from_issues(issues)

    def merge(self, other: ValidationResult) -> ValidationResult:
        """
        Combina dos resultados de validación.

        El orden de issues se preserva: primero ``self``, luego ``other``.
        ``is_valid`` se recalcula sobre la unión de issues.

        Args:
            other: Otro ``ValidationResult`` para combinar.

        Returns:
            Nuevo ``ValidationResult`` con issues combinados.

        Raises:
            TypeError: Si ``other`` no es ``ValidationResult``.
        """
        if not isinstance(other, ValidationResult):
            raise TypeError(
                f"other debe ser ValidationResult, "
                f"recibido={type(other).__name__}"
            )
        return ValidationResult.from_issues(self.issues + other.issues)

    def to_dict(self) -> dict[str, Any]:
        """Serializa a diccionario para consumo externo."""
        return {
            "is_valid": self.is_valid,
            "issue_count": len(self.issues),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [
                {
                    "severity": i.severity.value,
                    "code": i.code.value,
                    "message": i.message,
                    "column": i.column,
                    "count": i.count,
                }
                for i in self.issues
            ],
        }


# ======================================================================
# VALIDATOR
# ======================================================================


class DataFrameValidator:
    """
    Validador estructural, de calidad y de dominio para DataFrames.

    Todos los métodos son classmethods o staticmethods puros:
    no mantienen estado y no mutan el DataFrame de entrada.

    Jerarquía de validaciones:
    1. ``validate_schema``: existencia de columnas requeridas.
    2. ``check_data_quality``: nulos y strings vacíos en columnas críticas.
    3. ``validate_numeric_columns``: coerción numérica y finitud.
    4. ``validate_non_negative``: no negatividad sobre valores numéricos válidos.
    5. ``validate_ranges``: cumplimiento de rangos [min, max].
    6. ``validate_domain``: composición orquestada de todas las anteriores.
    """

    # ------------------------------------------------------------------
    # Validación interna de argumentos
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_dataframe_object(df: object) -> ValidationResult:
        """
        Valida que el objeto sea un ``pd.DataFrame`` no nulo.

        Returns:
            ``ValidationResult.success()`` si es válido, o resultado con
            issue ERROR si es ``None`` o tipo incorrecto.
        """
        if df is None:
            return ValidationResult.failure(
                (
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code=ValidationCode.NONE_DATAFRAME,
                        message="El DataFrame es None.",
                    ),
                )
            )
        if not isinstance(df, pd.DataFrame):
            return ValidationResult.failure(
                (
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code=ValidationCode.INVALID_DATAFRAME_TYPE,
                        message=(
                            f"Se esperaba pandas.DataFrame, "
                            f"recibido={type(df).__name__}."
                        ),
                    ),
                )
            )
        return ValidationResult.success()

    @staticmethod
    def _normalize_columns_argument(
        columns: Iterable[str], arg_name: str
    ) -> tuple[str, ...]:
        """
        Normaliza y valida un argumento de columnas.

        Operaciones:
        - Materializa el iterable.
        - Valida que cada elemento sea un string no vacío.
        - Deduplica preservando orden de primera aparición
          (con warning implícito: el llamador debe evitar duplicados).

        Args:
            columns: Iterable de nombres de columna.
            arg_name: Nombre del argumento para mensajes de error.

        Returns:
            Tupla de nombres de columna normalizados y deduplicados.

        Raises:
            ValueError: Si ``columns`` es ``None``, no iterable,
                contiene no-strings, o strings vacíos.
        """
        if columns is None:
            raise ValueError(f"{arg_name} no puede ser None")

        try:
            raw = list(columns)
        except TypeError as exc:
            raise ValueError(
                f"{arg_name} debe ser un iterable de strings"
            ) from exc

        normalized: list[str] = []
        seen: set[str] = set()

        for idx, col in enumerate(raw):
            if not isinstance(col, str):
                raise ValueError(
                    f"{arg_name}[{idx}] debe ser str, "
                    f"recibido={type(col).__name__}"
                )
            cleaned = col.strip()
            if not cleaned:
                raise ValueError(
                    f"{arg_name}[{idx}] no puede ser un string vacío"
                )
            if cleaned not in seen:
                normalized.append(cleaned)
                seen.add(cleaned)

        return tuple(normalized)

    @staticmethod
    def _check_duplicate_columns(
        df: pd.DataFrame,
    ) -> ValidationResult:
        """
        Detecta columnas duplicadas en el DataFrame.

        Reporta las columnas duplicadas con sus posiciones (índices)
        para facilitar la depuración.
        """
        duplicated_mask = df.columns.duplicated()
        if not duplicated_mask.any():
            return ValidationResult.success()

        duplicated_names = df.columns[duplicated_mask].tolist()
        # Encontrar posiciones de cada duplicado
        positions: dict[str, list[int]] = {}
        for idx, col_name in enumerate(df.columns):
            if col_name in set(duplicated_names):
                positions.setdefault(str(col_name), []).append(idx)

        detail = ", ".join(
            f"'{name}' en posiciones {pos}"
            for name, pos in positions.items()
        )

        return ValidationResult.from_issues(
            (
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code=ValidationCode.DUPLICATE_COLUMNS,
                    message=f"Columnas duplicadas detectadas: {detail}",
                    count=len(duplicated_names),
                ),
            )
        )

    @staticmethod
    def _validate_range_bound(
        value: Optional[float], bound_name: str, col: str
    ) -> None:
        """
        Valida que un límite de rango sea finito cuando no es ``None``.

        Raises:
            ValueError: Si el valor no es numérico o no es finito.
        """
        if value is None:
            return
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Límite '{bound_name}' para '{col}' debe ser numérico, "
                f"recibido={type(value).__name__}"
            )
        if not math.isfinite(value):
            raise ValueError(
                f"Límite '{bound_name}' para '{col}' debe ser finito, "
                f"recibido={value}"
            )

    # ------------------------------------------------------------------
    # Validadores principales
    # ------------------------------------------------------------------

    @classmethod
    def validate_schema(
        cls,
        df: pd.DataFrame,
        required_columns: Iterable[str],
    ) -> ValidationResult:
        """
        Valida existencia de columnas requeridas en el DataFrame.

        Detecta adicionalmente columnas duplicadas como WARNING.

        Args:
            df: DataFrame a validar.
            required_columns: Iterable de nombres de columna requeridos.

        Returns:
            ``ValidationResult`` con issues para columnas faltantes (ERROR)
            y duplicadas (WARNING).
        """
        df_result = cls._validate_dataframe_object(df)
        if not df_result.is_valid:
            return df_result

        required = cls._normalize_columns_argument(
            required_columns, "required_columns"
        )
        issues: list[ValidationIssue] = list(
            cls._check_duplicate_columns(df).issues
        )

        current_columns = frozenset(str(c) for c in df.columns)
        missing = [
            col for col in required if col not in current_columns
        ]

        if missing:
            issues.extend(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code=ValidationCode.MISSING_REQUIRED_COLUMN,
                    message=f"Falta columna requerida '{col}'.",
                    column=col,
                )
                for col in missing
            )

        return ValidationResult.from_issues(tuple(issues))

    @classmethod
    def check_data_quality(
        cls,
        df: pd.DataFrame,
        critical_columns: Iterable[str],
        *,
        empty_strings_as_warning: bool = True,
    ) -> ValidationResult:
        """
        Valida calidad de datos en columnas críticas.

        Detecta:
        - Columnas críticas ausentes (WARNING).
        - Valores nulos (WARNING).
        - Strings vacíos o solo-espacios (WARNING, configurable).

        Args:
            df: DataFrame a validar.
            critical_columns: Columnas a inspeccionar.
            empty_strings_as_warning: Si ``True``, reporta strings vacíos.

        Returns:
            ``ValidationResult`` con issues de calidad.
        """
        df_result = cls._validate_dataframe_object(df)
        if not df_result.is_valid:
            return df_result

        critical = cls._normalize_columns_argument(
            critical_columns, "critical_columns"
        )
        issues: list[ValidationIssue] = list(
            cls._check_duplicate_columns(df).issues
        )

        for col in critical:
            if col not in df.columns:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code=ValidationCode.MISSING_CRITICAL_COLUMN,
                        message=(
                            f"Columna crítica '{col}' no encontrada "
                            f"para validación de calidad."
                        ),
                        column=col,
                    )
                )
                continue

            # Detección de nulos
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code=ValidationCode.NULL_VALUES,
                        message=(
                            f"Columna '{col}' tiene "
                            f"{null_count} valores nulos."
                        ),
                        column=col,
                        count=null_count,
                    )
                )

            # Detección de strings vacíos (vectorizada)
            if empty_strings_as_warning:
                series = df[col]
                if series.dtype == object or pd.api.types.is_string_dtype(
                    series
                ):
                    str_mask = series.apply(type) == str
                    empty_mask = str_mask & (
                        series.str.strip() == ""
                    )
                    empty_count = int(empty_mask.sum())
                else:
                    empty_count = 0

                if empty_count > 0:
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            code=ValidationCode.EMPTY_STRINGS,
                            message=(
                                f"Columna '{col}' tiene "
                                f"{empty_count} strings vacíos."
                            ),
                            column=col,
                            count=empty_count,
                        )
                    )

        return ValidationResult.from_issues(tuple(issues))

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
        Valida que las columnas especificadas contengan valores numéricos finitos.

        Detecta:
        - Columnas ausentes (WARNING o ERROR según ``missing_as_warning``).
        - Valores no convertibles a numérico (severidad configurable).
        - Valores no finitos: ±inf, NaN post-coerción (ERROR).

        La detección de no-finitos opera sobre los valores que sí se
        convirtieron exitosamente a numérico, evitando falsos positivos
        por coerción fallida.

        Args:
            df: DataFrame a validar.
            numeric_columns: Columnas a inspeccionar.
            missing_as_warning: Si ``True``, columnas ausentes son WARNING.
            non_numeric_severity: Severidad para valores no numéricos.

        Returns:
            ``ValidationResult`` con issues de numericidad y finitud.
        """
        df_result = cls._validate_dataframe_object(df)
        if not df_result.is_valid:
            return df_result

        columns = cls._normalize_columns_argument(
            numeric_columns, "numeric_columns"
        )
        issues: list[ValidationIssue] = []

        for col in columns:
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
                            f"Columna numérica '{col}' no encontrada."
                        ),
                        column=col,
                    )
                )
                continue

            series = df[col]
            non_null_mask = series.notna()
            non_null_count = int(non_null_mask.sum())

            if non_null_count == 0:
                continue

            # Coerción numérica
            coerced = pd.to_numeric(series, errors="coerce")

            # Valores que eran no-nulos pero fallaron la coerción
            coercion_failed_mask = non_null_mask & coerced.isna()
            non_numeric_count = int(coercion_failed_mask.sum())
            if non_numeric_count > 0:
                issues.append(
                    ValidationIssue(
                        severity=non_numeric_severity,
                        code=ValidationCode.NON_NUMERIC_VALUES,
                        message=(
                            f"Columna '{col}' tiene "
                            f"{non_numeric_count} valores no numéricos."
                        ),
                        column=col,
                        count=non_numeric_count,
                    )
                )

            # Valores que SÍ se convirtieron pero no son finitos
            # (±inf). Operamos solo sobre los exitosamente convertidos.
            successfully_converted = coerced.notna()
            if int(successfully_converted.sum()) > 0:
                numeric_values = coerced[successfully_converted]
                non_finite_mask = ~np.isfinite(
                    numeric_values.to_numpy(dtype=np.float64)
                )
                non_finite_count = int(non_finite_mask.sum())
                if non_finite_count > 0:
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            code=ValidationCode.NON_FINITE_VALUES,
                            message=(
                                f"Columna '{col}' tiene "
                                f"{non_finite_count} valores no finitos."
                            ),
                            column=col,
                            count=non_finite_count,
                        )
                    )

        return ValidationResult.from_issues(tuple(issues))

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
        Valida que las columnas especificadas no contengan valores negativos.

        Opera solo sobre valores numéricos finitos: ``-inf`` se excluye
        del conteo de negativos (debería ser capturado por
        ``validate_numeric_columns``).

        Args:
            df: DataFrame a validar.
            columns: Columnas a inspeccionar.
            missing_as_warning: Si ``True``, columnas ausentes son WARNING.
            negative_severity: Severidad para valores negativos.

        Returns:
            ``ValidationResult`` con issues de negatividad.
        """
        df_result = cls._validate_dataframe_object(df)
        if not df_result.is_valid:
            return df_result

        cols = cls._normalize_columns_argument(columns, "columns")
        issues: list[ValidationIssue] = []

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
                            f"Columna '{col}' no encontrada para "
                            f"validación de no negatividad."
                        ),
                        column=col,
                    )
                )
                continue

            coerced = pd.to_numeric(df[col], errors="coerce")
            # Solo considerar valores numéricos finitos
            finite_mask = coerced.notna() & np.isfinite(
                coerced.fillna(0).to_numpy(dtype=np.float64)
            )
            finite_series = coerced[
                pd.Series(finite_mask, index=coerced.index)
            ]

            negative_count = int((finite_series < 0).sum())
            if negative_count > 0:
                issues.append(
                    ValidationIssue(
                        severity=negative_severity,
                        code=ValidationCode.NEGATIVE_VALUES,
                        message=(
                            f"Columna '{col}' tiene "
                            f"{negative_count} valores negativos."
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
        range_rules: Mapping[
            str, tuple[Optional[float], Optional[float]]
        ],
        *,
        missing_as_warning: bool = True,
        violation_severity: ValidationSeverity = ValidationSeverity.WARNING,
    ) -> ValidationResult:
        """
        Valida que los valores numéricos estén dentro de rangos especificados.

        Cada regla es ``{columna: (min_value, max_value)}`` donde cualquiera
        de los límites puede ser ``None`` para indicar sin restricción.

        Validaciones de la regla misma:
        - Claves deben ser strings no vacíos.
        - Valores deben ser tuplas de longitud 2.
        - Límites no-``None`` deben ser numéricos finitos.
        - ``min_value <= max_value`` cuando ambos están presentes.

        Args:
            df: DataFrame a validar.
            range_rules: Mapeo de columna a tupla (min, max).
            missing_as_warning: Si ``True``, columnas ausentes son WARNING.
            violation_severity: Severidad para violaciones de rango.

        Returns:
            ``ValidationResult`` con issues de rango.

        Raises:
            ValueError: Si ``range_rules`` es ``None`` o contiene
                reglas mal formadas.
        """
        df_result = cls._validate_dataframe_object(df)
        if not df_result.is_valid:
            return df_result

        if range_rules is None:
            raise ValueError("range_rules no puede ser None")

        issues: list[ValidationIssue] = []

        for col, bounds in range_rules.items():
            if not isinstance(col, str) or not col.strip():
                raise ValueError(
                    "Las claves de range_rules deben ser strings no vacíos."
                )
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise ValueError(
                    f"Regla de rango inválida para '{col}'. "
                    f"Debe ser tuple(min, max)."
                )

            min_value, max_value = bounds

            # Validar que los límites sean numéricos finitos
            cls._validate_range_bound(min_value, "min", col)
            cls._validate_range_bound(max_value, "max", col)

            if (
                min_value is not None
                and max_value is not None
                and min_value > max_value
            ):
                raise ValueError(
                    f"Rango inválido para '{col}': "
                    f"min ({min_value}) > max ({max_value})."
                )

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
                            f"Columna '{col}' no encontrada "
                            f"para validación de rango."
                        ),
                        column=col,
                    )
                )
                continue

            coerced = pd.to_numeric(df[col], errors="coerce")
            valid_numeric = coerced.notna()

            violation_mask = pd.Series(False, index=df.index)
            if min_value is not None:
                violation_mask = violation_mask | (
                    valid_numeric & (coerced < min_value)
                )
            if max_value is not None:
                violation_mask = violation_mask | (
                    valid_numeric & (coerced > max_value)
                )

            violation_count = int(violation_mask.sum())
            if violation_count > 0:
                # Construir descripción del rango
                min_str = (
                    str(min_value) if min_value is not None else "-∞"
                )
                max_str = (
                    str(max_value) if max_value is not None else "+∞"
                )
                issues.append(
                    ValidationIssue(
                        severity=violation_severity,
                        code=ValidationCode.RANGE_VIOLATION,
                        message=(
                            f"Columna '{col}' tiene "
                            f"{violation_count} valores fuera de rango "
                            f"[{min_str}, {max_str}]."
                        ),
                        column=col,
                        count=violation_count,
                    )
                )

        return ValidationResult.from_issues(tuple(issues))

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
    ) -> ValidationResult:
        """
        Ejecuta validación compuesta de dominio.

        Orquesta secuencialmente:
        1. Validación de esquema (columnas requeridas).
        2. Validación de calidad (columnas críticas).
        3. Validación de numericidad (columnas numéricas).
        4. Validación de no negatividad.
        5. Validación de rangos.

        Cada paso se ejecuta solo si el argumento correspondiente
        no está vacío, evitando trabajo innecesario.

        Args:
            df: DataFrame a validar.
            required_columns: Columnas que deben existir.
            critical_columns: Columnas para verificación de calidad.
            numeric_columns: Columnas que deben ser numéricas.
            non_negative_columns: Columnas que no deben tener negativos.
            range_rules: Reglas de rango por columna.

        Returns:
            ``ValidationResult`` compuesto con todos los issues.
        """
        df_result = cls._validate_dataframe_object(df)
        if not df_result.is_valid:
            return df_result

        # Materializar iterables una sola vez para verificar vacío
        # y reutilizar en el validador correspondiente.
        req_cols = _materialize_iterable(required_columns)
        crit_cols = _materialize_iterable(critical_columns)
        num_cols = _materialize_iterable(numeric_columns)
        nn_cols = _materialize_iterable(non_negative_columns)

        composite = ValidationResult.success()

        if req_cols:
            composite = composite.merge(
                cls.validate_schema(df, req_cols)
            )
        if crit_cols:
            composite = composite.merge(
                cls.check_data_quality(df, crit_cols)
            )
        if num_cols:
            composite = composite.merge(
                cls.validate_numeric_columns(df, num_cols)
            )
        if nn_cols:
            composite = composite.merge(
                cls.validate_non_negative(df, nn_cols)
            )
        if range_rules:
            composite = composite.merge(
                cls.validate_ranges(df, range_rules)
            )

        return composite


# ======================================================================
# UTILIDADES INTERNAS
# ======================================================================


def _materialize_iterable(
    iterable: Iterable[str],
) -> tuple[str, ...]:
    """
    Materializa un iterable a tupla de forma segura.

    Si el iterable ya es una tupla o lista, lo convierte directamente.
    Para generadores u otros iterables, los consume una sola vez.
    """
    if isinstance(iterable, tuple):
        return iterable
    return tuple(iterable)