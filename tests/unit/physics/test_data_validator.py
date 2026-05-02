"""
Suite de pruebas rigurosa para ``validators.py``.

Estructura:
- Fixtures reutilizables para DataFrames y configuraciones.
- Pruebas unitarias por método/clase.
- Pruebas de integración para ``validate_domain``.
- Pruebas de propiedades (inmutabilidad, determinismo, invariantes).

Convenciones:
- Cada clase ``Test*`` agrupa pruebas de una unidad lógica.
- Los nombres siguen ``test_<condición>_<resultado_esperado>``.
- Se usa ``pytest.mark.parametrize`` para variantes sin duplicación.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest

from app.adapters.validators import (
    DataFrameValidator,
    ValidationCode,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    _materialize_iterable,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def valid_df() -> pd.DataFrame:
    """DataFrame válido con columnas mixtas."""
    return pd.DataFrame(
        {
            "ID": ["A", "B", "C", "D"],
            "CANTIDAD": [10.0, 20.0, 30.0, 40.0],
            "PRECIO": [100.0, 200.0, 300.0, 400.0],
            "DESCRIPCION": ["Item A", "Item B", "Item C", "Item D"],
            "RENDIMIENTO": [1.0, 0.5, 0.8, 1.2],
        }
    )


@pytest.fixture()
def df_with_nulls() -> pd.DataFrame:
    """DataFrame con valores nulos en varias columnas."""
    return pd.DataFrame(
        {
            "ID": ["A", None, "C", "D"],
            "CANTIDAD": [10.0, np.nan, 30.0, np.nan],
            "PRECIO": [100.0, 200.0, None, 400.0],
            "DESCRIPCION": ["Item A", "", "  ", "Item D"],
        }
    )


@pytest.fixture()
def df_with_negatives() -> pd.DataFrame:
    """DataFrame con valores negativos."""
    return pd.DataFrame(
        {
            "CANTIDAD": [10.0, -5.0, 30.0, -1.0],
            "PRECIO": [100.0, 200.0, -50.0, 400.0],
            "VALOR": [1.0, 2.0, 3.0, 4.0],
        }
    )


@pytest.fixture()
def df_with_non_numeric() -> pd.DataFrame:
    """DataFrame con valores no numéricos en columnas esperadas numéricas."""
    return pd.DataFrame(
        {
            "CANTIDAD": [10.0, "abc", 30.0, "def"],
            "PRECIO": [100.0, 200.0, "N/A", 400.0],
            "TEXTO": ["a", "b", "c", "d"],
        }
    )


@pytest.fixture()
def df_with_inf() -> pd.DataFrame:
    """DataFrame con valores infinitos."""
    return pd.DataFrame(
        {
            "CANTIDAD": [10.0, float("inf"), 30.0, float("-inf")],
            "PRECIO": [100.0, 200.0, float("inf"), 400.0],
        }
    )


@pytest.fixture()
def df_with_duplicated_columns() -> pd.DataFrame:
    """DataFrame con columnas duplicadas."""
    df = pd.DataFrame(
        {
            "A": [1, 2],
            "B": [3, 4],
        }
    )
    # Forzar columnas duplicadas
    df.columns = ["COL", "COL"]
    return df


# ======================================================================
# Tests: ValidationSeverity
# ======================================================================


class TestValidationSeverity:
    """Pruebas para el enum de severidad."""

    def test_all_values_present(self) -> None:
        expected = {"INFO", "WARNING", "ERROR", "CRITICAL"}
        actual = {s.value for s in ValidationSeverity}
        assert actual == expected

    def test_string_enum_comparison(self) -> None:
        assert ValidationSeverity.ERROR == "ERROR"
        assert ValidationSeverity.WARNING == "WARNING"

    def test_ordering_by_severity(self) -> None:
        severities = list(ValidationSeverity)
        assert ValidationSeverity.INFO in severities
        assert ValidationSeverity.CRITICAL in severities


# ======================================================================
# Tests: ValidationCode
# ======================================================================


class TestValidationCode:
    """Pruebas para el enum de códigos."""

    def test_all_codes_present(self) -> None:
        expected_codes = {
            "NONE_DATAFRAME",
            "INVALID_DATAFRAME_TYPE",
            "INVALID_COLUMNS_ARGUMENT",
            "EMPTY_COLUMN_NAME",
            "DUPLICATE_COLUMNS",
            "MISSING_REQUIRED_COLUMN",
            "MISSING_CRITICAL_COLUMN",
            "NULL_VALUES",
            "EMPTY_STRINGS",
            "NON_NUMERIC_VALUES",
            "NON_FINITE_VALUES",
            "NEGATIVE_VALUES",
            "RANGE_VIOLATION",
            "SUBNORMAL_VALUES",
            "LINEAR_DEPENDENCY",
            "RANK_DEFICIENCY",
            "SURVIVAL_THRESHOLD_VIOLATION",
        }
        actual = {c.value for c in ValidationCode}
        assert actual == expected_codes

    def test_string_enum_comparison(self) -> None:
        assert ValidationCode.NULL_VALUES == "NULL_VALUES"


# ======================================================================
# Tests: ValidationIssue
# ======================================================================


class TestValidationIssue:
    """Pruebas para hallazgos atómicos de validación."""

    def test_creation_with_all_fields(self) -> None:
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code=ValidationCode.NULL_VALUES,
            message="Test message",
            column="COL_A",
            count=5,
        )
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.code == ValidationCode.NULL_VALUES
        assert issue.message == "Test message"
        assert issue.column == "COL_A"
        assert issue.count == 5

    def test_creation_with_defaults(self) -> None:
        issue = ValidationIssue(
            severity=ValidationSeverity.INFO,
            code=ValidationCode.DUPLICATE_COLUMNS,
            message="Info message",
        )
        assert issue.column is None
        assert issue.count is None

    def test_frozen_immutability(self) -> None:
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code=ValidationCode.EMPTY_STRINGS,
            message="Test",
        )
        with pytest.raises(AttributeError):
            issue.message = "Changed"  # type: ignore[misc]

    def test_str_representation_with_all_fields(self) -> None:
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code=ValidationCode.NEGATIVE_VALUES,
            message="Valores negativos",
            column="PRECIO",
            count=3,
        )
        text = str(issue)
        assert "ERROR" in text
        assert "NEGATIVE_VALUES" in text
        assert "Valores negativos" in text
        assert "PRECIO" in text
        assert "3" in text

    def test_str_representation_minimal(self) -> None:
        issue = ValidationIssue(
            severity=ValidationSeverity.INFO,
            code=ValidationCode.DUPLICATE_COLUMNS,
            message="Duplicados",
        )
        text = str(issue)
        assert "INFO" in text
        assert "Duplicados" in text
        assert "columna=" not in text
        assert "count=" not in text


# ======================================================================
# Tests: ValidationResult
# ======================================================================


class TestValidationResult:
    """Pruebas para resultados agregados de validación."""

    def test_success_factory(self) -> None:
        result = ValidationResult.success()
        assert result.is_valid is True
        assert len(result.issues) == 0
        assert result.has_errors is False
        assert result.has_warnings is False

    def test_from_issues_no_errors(self) -> None:
        issues = (
            ValidationIssue(
                severity=ValidationSeverity.INFO,
                code=ValidationCode.DUPLICATE_COLUMNS,
                message="Info",
            ),
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code=ValidationCode.NULL_VALUES,
                message="Warning",
            ),
        )
        result = ValidationResult.from_issues(issues)
        assert result.is_valid is True
        assert result.has_warnings is True
        assert result.has_errors is False

    def test_from_issues_with_error(self) -> None:
        issues = (
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code=ValidationCode.MISSING_REQUIRED_COLUMN,
                message="Error",
            ),
        )
        result = ValidationResult.from_issues(issues)
        assert result.is_valid is False
        assert result.has_errors is True

    def test_from_issues_with_critical(self) -> None:
        issues = (
            ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                code=ValidationCode.NONE_DATAFRAME,
                message="Critical",
            ),
        )
        result = ValidationResult.from_issues(issues)
        assert result.is_valid is False
        assert result.has_errors is True

    def test_failure_factory_with_errors(self) -> None:
        issues = (
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code=ValidationCode.NONE_DATAFRAME,
                message="Fail",
            ),
        )
        result = ValidationResult.failure(issues)
        assert result.is_valid is False

    def test_failure_factory_without_errors_is_valid(self) -> None:
        """failure() sin errores produce is_valid=True por invariante."""
        issues = (
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code=ValidationCode.NULL_VALUES,
                message="Solo warning",
            ),
        )
        result = ValidationResult.failure(issues)
        assert result.is_valid is True

    def test_errors_property(self) -> None:
        issues = (
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code=ValidationCode.MISSING_REQUIRED_COLUMN,
                message="E1",
            ),
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code=ValidationCode.NULL_VALUES,
                message="W1",
            ),
            ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                code=ValidationCode.NONE_DATAFRAME,
                message="C1",
            ),
        )
        result = ValidationResult.from_issues(issues)
        errors = result.errors
        assert len(errors) == 2
        assert all(
            e.severity
            in {ValidationSeverity.ERROR, ValidationSeverity.CRITICAL}
            for e in errors
        )

    def test_warnings_property(self) -> None:
        issues = (
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code=ValidationCode.NULL_VALUES,
                message="W1",
            ),
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code=ValidationCode.EMPTY_STRINGS,
                message="W2",
            ),
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code=ValidationCode.MISSING_REQUIRED_COLUMN,
                message="E1",
            ),
        )
        result = ValidationResult.from_issues(issues)
        assert len(result.warnings) == 2

    def test_infos_property(self) -> None:
        issues = (
            ValidationIssue(
                severity=ValidationSeverity.INFO,
                code=ValidationCode.DUPLICATE_COLUMNS,
                message="I1",
            ),
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code=ValidationCode.NONE_DATAFRAME,
                message="E1",
            ),
        )
        result = ValidationResult.from_issues(issues)
        assert len(result.infos) == 1

    def test_frozen_immutability(self) -> None:
        result = ValidationResult.success()
        with pytest.raises(AttributeError):
            result.is_valid = False  # type: ignore[misc]

    def test_str_representation(self) -> None:
        result = ValidationResult.from_issues(
            (
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code=ValidationCode.NONE_DATAFRAME,
                    message="E",
                ),
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code=ValidationCode.NULL_VALUES,
                    message="W",
                ),
            )
        )
        text = str(result)
        assert "INVALID" in text
        assert "errors=1" in text
        assert "warnings=1" in text

    def test_str_valid(self) -> None:
        text = str(ValidationResult.success())
        assert "VALID" in text

    # --- merge ---

    def test_merge_two_valid(self) -> None:
        r1 = ValidationResult.from_issues(
            (
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    code=ValidationCode.DUPLICATE_COLUMNS,
                    message="I1",
                ),
            )
        )
        r2 = ValidationResult.from_issues(
            (
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code=ValidationCode.NULL_VALUES,
                    message="W1",
                ),
            )
        )
        merged = r1.merge(r2)
        assert merged.is_valid is True
        assert len(merged.issues) == 2

    def test_merge_propagates_invalidity(self) -> None:
        valid = ValidationResult.success()
        invalid = ValidationResult.from_issues(
            (
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code=ValidationCode.NONE_DATAFRAME,
                    message="E",
                ),
            )
        )
        merged = valid.merge(invalid)
        assert merged.is_valid is False
        assert len(merged.errors) == 1

    def test_merge_preserves_order(self) -> None:
        r1 = ValidationResult.from_issues(
            (
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    code=ValidationCode.DUPLICATE_COLUMNS,
                    message="First",
                ),
            )
        )
        r2 = ValidationResult.from_issues(
            (
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code=ValidationCode.NULL_VALUES,
                    message="Second",
                ),
            )
        )
        merged = r1.merge(r2)
        assert merged.issues[0].message == "First"
        assert merged.issues[1].message == "Second"

    def test_merge_non_validation_result_raises(self) -> None:
        result = ValidationResult.success()
        with pytest.raises(TypeError, match="ValidationResult"):
            result.merge("not_a_result")  # type: ignore[arg-type]

    def test_merge_with_dict_raises(self) -> None:
        result = ValidationResult.success()
        with pytest.raises(TypeError):
            result.merge({"is_valid": True})  # type: ignore[arg-type]

    # --- to_dict ---

    def test_to_dict_structure(self) -> None:
        result = ValidationResult.from_issues(
            (
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code=ValidationCode.MISSING_REQUIRED_COLUMN,
                    message="Missing",
                    column="COL_A",
                    count=1,
                ),
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code=ValidationCode.NULL_VALUES,
                    message="Nulls",
                    column="COL_B",
                    count=5,
                ),
            )
        )
        d = result.to_dict()
        assert d["is_valid"] is False
        assert d["issue_count"] == 2
        assert d["error_count"] == 1
        assert d["warning_count"] == 1
        assert len(d["issues"]) == 2

        issue_dict = d["issues"][0]
        assert issue_dict["severity"] == "ERROR"
        assert issue_dict["code"] == "MISSING_REQUIRED_COLUMN"
        assert issue_dict["column"] == "COL_A"
        assert issue_dict["count"] == 1

    def test_to_dict_empty(self) -> None:
        d = ValidationResult.success().to_dict()
        assert d["is_valid"] is True
        assert d["issue_count"] == 0
        assert d["error_count"] == 0
        assert d["warning_count"] == 0
        assert d["issues"] == []


# ======================================================================
# Tests: _validate_dataframe_object
# ======================================================================


class TestValidateDataframeObject:
    """Pruebas para validación del objeto DataFrame."""

    def test_none_returns_error(self) -> None:
        result = DataFrameValidator._validate_dataframe_object(None)
        assert result.is_valid is False
        assert result.errors[0].code == ValidationCode.NONE_DATAFRAME

    def test_non_dataframe_returns_error(self) -> None:
        result = DataFrameValidator._validate_dataframe_object(
            {"a": [1, 2]}
        )
        assert result.is_valid is False
        assert (
            result.errors[0].code
            == ValidationCode.INVALID_DATAFRAME_TYPE
        )

    @pytest.mark.parametrize(
        "invalid_input",
        [42, "string", [1, 2, 3], True, np.array([1, 2])],
    )
    def test_various_non_dataframe_types(
        self, invalid_input: Any
    ) -> None:
        result = DataFrameValidator._validate_dataframe_object(
            invalid_input
        )
        assert result.is_valid is False

    def test_valid_dataframe(self, valid_df: pd.DataFrame) -> None:
        result = DataFrameValidator._validate_dataframe_object(valid_df)
        assert result.is_valid is True
        assert len(result.issues) == 0

    def test_empty_dataframe_is_valid(self) -> None:
        result = DataFrameValidator._validate_dataframe_object(
            pd.DataFrame()
        )
        assert result.is_valid is True


# ======================================================================
# Tests: _normalize_columns_argument
# ======================================================================


class TestNormalizeColumnsArgument:
    """Pruebas para normalización de argumentos de columnas."""

    def test_basic_normalization(self) -> None:
        result = DataFrameValidator._normalize_columns_argument(
            ["A", "B", "C"], "test"
        )
        assert result == ("A", "B", "C")

    def test_strips_whitespace(self) -> None:
        result = DataFrameValidator._normalize_columns_argument(
            ["  A  ", " B "], "test"
        )
        assert result == ("A", "B")

    def test_deduplicates(self) -> None:
        result = DataFrameValidator._normalize_columns_argument(
            ["A", "B", "A", "C", "B"], "test"
        )
        assert result == ("A", "B", "C")

    def test_dedup_after_strip(self) -> None:
        result = DataFrameValidator._normalize_columns_argument(
            ["A", " A ", "  A  "], "test"
        )
        assert result == ("A",)

    def test_none_raises(self) -> None:
        with pytest.raises(ValueError, match="no puede ser None"):
            DataFrameValidator._normalize_columns_argument(None, "test")

    def test_non_iterable_raises(self) -> None:
        with pytest.raises(ValueError, match="iterable de strings"):
            DataFrameValidator._normalize_columns_argument(42, "test")  # type: ignore[arg-type]

    def test_non_string_element_raises(self) -> None:
        with pytest.raises(ValueError, match="debe ser str"):
            DataFrameValidator._normalize_columns_argument(
                ["A", 42], "test"  # type: ignore[list-item]
            )

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="string vacío"):
            DataFrameValidator._normalize_columns_argument(
                ["A", ""], "test"
            )

    def test_whitespace_only_string_raises(self) -> None:
        with pytest.raises(ValueError, match="string vacío"):
            DataFrameValidator._normalize_columns_argument(
                ["A", "   "], "test"
            )

    def test_empty_list_returns_empty_tuple(self) -> None:
        result = DataFrameValidator._normalize_columns_argument(
            [], "test"
        )
        assert result == ()

    def test_generator_input(self) -> None:
        gen = (x for x in ["A", "B"])
        result = DataFrameValidator._normalize_columns_argument(
            gen, "test"
        )
        assert result == ("A", "B")

    def test_frozenset_input(self) -> None:
        result = DataFrameValidator._normalize_columns_argument(
            frozenset({"A", "B"}), "test"
        )
        assert set(result) == {"A", "B"}
        assert len(result) == 2

    def test_index_in_error_message(self) -> None:
        with pytest.raises(ValueError, match=r"\[1\]"):
            DataFrameValidator._normalize_columns_argument(
                ["OK", 999], "my_arg"  # type: ignore[list-item]
            )


# ======================================================================
# Tests: _check_duplicate_columns
# ======================================================================


class TestCheckDuplicateColumns:
    """Pruebas para detección de columnas duplicadas."""

    def test_no_duplicates(self, valid_df: pd.DataFrame) -> None:
        result = DataFrameValidator._check_duplicate_columns(valid_df)
        assert result.is_valid is True
        assert len(result.issues) == 0

    def test_with_duplicates(
        self, df_with_duplicated_columns: pd.DataFrame
    ) -> None:
        result = DataFrameValidator._check_duplicate_columns(
            df_with_duplicated_columns
        )
        assert len(result.issues) == 1
        issue = result.issues[0]
        assert issue.severity == ValidationSeverity.WARNING
        assert issue.code == ValidationCode.DUPLICATE_COLUMNS
        assert issue.count is not None and issue.count > 0

    def test_duplicate_message_contains_positions(
        self, df_with_duplicated_columns: pd.DataFrame
    ) -> None:
        result = DataFrameValidator._check_duplicate_columns(
            df_with_duplicated_columns
        )
        message = result.issues[0].message
        assert "posiciones" in message.lower() or "posicion" in message.lower()

    def test_duplicates_still_valid(
        self, df_with_duplicated_columns: pd.DataFrame
    ) -> None:
        """Duplicados son WARNING, no ERROR → is_valid=True."""
        result = DataFrameValidator._check_duplicate_columns(
            df_with_duplicated_columns
        )
        assert result.is_valid is True


# ======================================================================
# Tests: _validate_range_bound
# ======================================================================


class TestValidateRangeBound:
    """Pruebas para validación de límites de rango."""

    def test_none_is_valid(self) -> None:
        DataFrameValidator._validate_range_bound(None, "min", "COL")

    def test_int_is_valid(self) -> None:
        DataFrameValidator._validate_range_bound(0, "min", "COL")
        DataFrameValidator._validate_range_bound(100, "max", "COL")

    def test_float_is_valid(self) -> None:
        DataFrameValidator._validate_range_bound(3.14, "min", "COL")

    def test_negative_is_valid(self) -> None:
        DataFrameValidator._validate_range_bound(-10.0, "min", "COL")

    def test_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="finito"):
            DataFrameValidator._validate_range_bound(
                float("inf"), "max", "COL"
            )

    def test_neg_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="finito"):
            DataFrameValidator._validate_range_bound(
                float("-inf"), "min", "COL"
            )

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="finito"):
            DataFrameValidator._validate_range_bound(
                float("nan"), "min", "COL"
            )

    def test_string_raises(self) -> None:
        with pytest.raises(ValueError, match="numérico"):
            DataFrameValidator._validate_range_bound(
                "10", "min", "COL"  # type: ignore[arg-type]
            )

    def test_bool_raises(self) -> None:
        with pytest.raises(ValueError, match="numérico"):
            DataFrameValidator._validate_range_bound(
                True, "min", "COL"  # type: ignore[arg-type]
            )


# ======================================================================
# Tests: validate_schema
# ======================================================================


class TestValidateSchema:
    """Pruebas para validación de esquema."""

    def test_all_columns_present(self, valid_df: pd.DataFrame) -> None:
        result = DataFrameValidator.validate_schema(
            valid_df, ["ID", "CANTIDAD", "PRECIO"]
        )
        assert result.is_valid is True

    def test_missing_columns(self, valid_df: pd.DataFrame) -> None:
        result = DataFrameValidator.validate_schema(
            valid_df, ["ID", "MISSING_A", "MISSING_B"]
        )
        assert result.is_valid is False
        missing_issues = [
            i
            for i in result.issues
            if i.code == ValidationCode.MISSING_REQUIRED_COLUMN
        ]
        assert len(missing_issues) == 2
        missing_cols = {i.column for i in missing_issues}
        assert missing_cols == {"MISSING_A", "MISSING_B"}

    def test_none_dataframe(self) -> None:
        result = DataFrameValidator.validate_schema(
            None, ["A"]  # type: ignore[arg-type]
        )
        assert result.is_valid is False

    def test_detects_duplicate_columns(
        self, df_with_duplicated_columns: pd.DataFrame
    ) -> None:
        result = DataFrameValidator.validate_schema(
            df_with_duplicated_columns, ["COL"]
        )
        dup_issues = [
            i
            for i in result.issues
            if i.code == ValidationCode.DUPLICATE_COLUMNS
        ]
        assert len(dup_issues) == 1

    def test_empty_required_columns(
        self, valid_df: pd.DataFrame
    ) -> None:
        result = DataFrameValidator.validate_schema(valid_df, [])
        assert result.is_valid is True

    def test_extra_columns_ignored(
        self, valid_df: pd.DataFrame
    ) -> None:
        result = DataFrameValidator.validate_schema(valid_df, ["ID"])
        assert result.is_valid is True

    def test_invalid_columns_argument_raises(
        self, valid_df: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError):
            DataFrameValidator.validate_schema(valid_df, None)  # type: ignore[arg-type]


# ======================================================================
# Tests: check_data_quality
# ======================================================================


class TestCheckDataQuality:
    """Pruebas para validación de calidad de datos."""

    def test_clean_data(self, valid_df: pd.DataFrame) -> None:
        result = DataFrameValidator.check_data_quality(
            valid_df, ["ID", "CANTIDAD"]
        )
        null_issues = [
            i for i in result.issues if i.code == ValidationCode.NULL_VALUES
        ]
        assert len(null_issues) == 0

    def test_detects_nulls(self, df_with_nulls: pd.DataFrame) -> None:
        result = DataFrameValidator.check_data_quality(
            df_with_nulls, ["ID", "CANTIDAD"]
        )
        null_issues = [
            i for i in result.issues if i.code == ValidationCode.NULL_VALUES
        ]
        assert len(null_issues) == 2  # ID y CANTIDAD tienen nulos

        id_issue = next(i for i in null_issues if i.column == "ID")
        assert id_issue.count == 1

        cant_issue = next(
            i for i in null_issues if i.column == "CANTIDAD"
        )
        assert cant_issue.count == 2

    def test_detects_empty_strings(
        self, df_with_nulls: pd.DataFrame
    ) -> None:
        result = DataFrameValidator.check_data_quality(
            df_with_nulls, ["DESCRIPCION"]
        )
        empty_issues = [
            i
            for i in result.issues
            if i.code == ValidationCode.EMPTY_STRINGS
        ]
        assert len(empty_issues) == 1
        assert empty_issues[0].count == 2  # "" y "  "

    def test_empty_strings_disabled(
        self, df_with_nulls: pd.DataFrame
    ) -> None:
        result = DataFrameValidator.check_data_quality(
            df_with_nulls,
            ["DESCRIPCION"],
            empty_strings_as_warning=False,
        )
        empty_issues = [
            i
            for i in result.issues
            if i.code == ValidationCode.EMPTY_STRINGS
        ]
        assert len(empty_issues) == 0

    def test_missing_critical_column(
        self, valid_df: pd.DataFrame
    ) -> None:
        result = DataFrameValidator.check_data_quality(
            valid_df, ["NONEXISTENT"]
        )
        missing_issues = [
            i
            for i in result.issues
            if i.code == ValidationCode.MISSING_CRITICAL_COLUMN
        ]
        assert len(missing_issues) == 1
        assert result.is_valid is True  # Missing critical es WARNING

    def test_none_dataframe(self) -> None:
        result = DataFrameValidator.check_data_quality(
            None, ["A"]  # type: ignore[arg-type]
        )
        assert result.is_valid is False

    def test_numeric_column_no_empty_strings(self) -> None:
        """Columnas numéricas no deberían reportar empty strings."""
        df = pd.DataFrame({"CANTIDAD": [1.0, 2.0, 3.0]})
        result = DataFrameValidator.check_data_quality(
            df, ["CANTIDAD"]
        )
        empty_issues = [
            i
            for i in result.issues
            if i.code == ValidationCode.EMPTY_STRINGS
        ]
        assert len(empty_issues) == 0


# ======================================================================
# Tests: validate_numeric_columns
# ======================================================================


class TestValidateNumericColumns:
    """Pruebas para validación de columnas numéricas."""

    def test_valid_numeric_columns(
        self, valid_df: pd.DataFrame
    ) -> None:
        result = DataFrameValidator.validate_numeric_columns(
            valid_df, ["CANTIDAD", "PRECIO"]
        )
        assert result.is_valid is True
        assert len(result.issues) == 0

    def test_detects_non_numeric(
        self, df_with_non_numeric: pd.DataFrame
    ) -> None:
        result = DataFrameValidator.validate_numeric_columns(
            df_with_non_numeric, ["CANTIDAD", "PRECIO"]
        )
        non_num_issues = [
            i
            for i in result.issues
            if i.code == ValidationCode.NON_NUMERIC_VALUES
        ]
        assert len(non_num_issues) == 2

        cant_issue = next(
            i for i in non_num_issues if i.column == "CANTIDAD"
        )
        assert cant_issue.count == 2  # "abc" y "def"

        precio_issue = next(
            i for i in non_num_issues if i.column == "PRECIO"
        )
        assert precio_issue.count == 1  # "N/A"

    def test_detects_non_finite(
        self, df_with_inf: pd.DataFrame
    ) -> None:
        result = DataFrameValidator.validate_numeric_columns(
            df_with_inf, ["CANTIDAD", "PRECIO"]
        )
        inf_issues = [
            i
            for i in result.issues
            if i.code == ValidationCode.NON_FINITE_VALUES
        ]
        assert len(inf_issues) == 2
        assert result.is_valid is False  # Non-finite es ERROR

        cant_issue = next(
            i for i in inf_issues if i.column == "CANTIDAD"
        )
        assert cant_issue.count == 2  # inf y -inf

    def test_missing_column_as_warning(
        self, valid_df: pd.DataFrame
    ) -> None:
        result = DataFrameValidator.validate_numeric_columns(
            valid_df, ["NONEXISTENT"]
        )
        assert result.is_valid is True
        assert len(result.issues) == 1
        assert (
            result.issues[0].code
            == ValidationCode.MISSING_CRITICAL_COLUMN
        )

    def test_missing_column_as_error(
        self, valid_df: pd.DataFrame
    ) -> None:
        result = DataFrameValidator.validate_numeric_columns(
            valid_df, ["NONEXISTENT"], missing_as_warning=False
        )
        assert result.is_valid is False
        assert (
            result.issues[0].code
            == ValidationCode.MISSING_REQUIRED_COLUMN
        )

    def test_all_null_column_skipped(self) -> None:
        df = pd.DataFrame({"COL": [None, np.nan, None]})
        result = DataFrameValidator.validate_numeric_columns(
            df, ["COL"]
        )
        assert len(result.issues) == 0

    def test_custom_non_numeric_severity(
        self, df_with_non_numeric: pd.DataFrame
    ) -> None:
        result = DataFrameValidator.validate_numeric_columns(
            df_with_non_numeric,
            ["CANTIDAD"],
            non_numeric_severity=ValidationSeverity.ERROR,
        )
        non_num = [
            i
            for i in result.issues
            if i.code == ValidationCode.NON_NUMERIC_VALUES
        ]
        assert non_num[0].severity == ValidationSeverity.ERROR
        assert result.is_valid is False

    def test_none_dataframe(self) -> None:
        result = DataFrameValidator.validate_numeric_columns(
            None, ["A"]  # type: ignore[arg-type]
        )
        assert result.is_valid is False

    def test_mixed_nan_and_inf(self) -> None:
        """NaN originales no se cuentan como non-finite post-coerción."""
        df = pd.DataFrame(
            {"COL": [1.0, np.nan, float("inf"), 4.0]}
        )
        result = DataFrameValidator.validate_numeric_columns(
            df, ["COL"]
        )
        inf_issues = [
            i
            for i in result.issues
            if i.code == ValidationCode.NON_FINITE_VALUES
        ]
        assert len(inf_issues) == 1
        assert inf_issues[0].count == 1  # Solo inf, no NaN


# ======================================================================
# Tests: validate_non_negative
# ======================================================================


class TestValidateNonNegative:
    """Pruebas para validación de no negatividad."""

    def test_all_positive(self, valid_df: pd.DataFrame) -> None:
        result = DataFrameValidator.validate_non_negative(
            valid_df, ["CANTIDAD", "PRECIO"]
        )
        assert result.is_valid is True
        assert len(result.issues) == 0

    def test_detects_negatives(
        self, df_with_negatives: pd.DataFrame
    ) -> None:
        result = DataFrameValidator.validate_non_negative(
            df_with_negatives, ["CANTIDAD", "PRECIO"]
        )
        neg_issues = [
            i
            for i in result.issues
            if i.code == ValidationCode.NEGATIVE_VALUES
        ]
        assert len(neg_issues) == 2

        cant_issue = next(
            i for i in neg_issues if i.column == "CANTIDAD"
        )
        assert cant_issue.count == 2  # -5.0 y -1.0

        precio_issue = next(
            i for i in neg_issues if i.column == "PRECIO"
        )
        assert precio_issue.count == 1  # -50.0

    def test_negative_inf_excluded(self) -> None:
        """-inf no se cuenta como negativo (es no-finito)."""
        df = pd.DataFrame(
            {"COL": [1.0, float("-inf"), -5.0, 10.0]}
        )
        result = DataFrameValidator.validate_non_negative(df, ["COL"])
        neg_issues = [
            i
            for i in result.issues
            if i.code == ValidationCode.NEGATIVE_VALUES
        ]
        assert len(neg_issues) == 1
        assert neg_issues[0].count == 1  # Solo -5.0, no -inf

    def test_zero_is_not_negative(self) -> None:
        df = pd.DataFrame({"COL": [0.0, 0, 0.0]})
        result = DataFrameValidator.validate_non_negative(df, ["COL"])
        assert len(result.issues) == 0

    def test_missing_column_as_warning(
        self, valid_df: pd.DataFrame
    ) -> None:
        result = DataFrameValidator.validate_non_negative(
            valid_df, ["NONEXISTENT"]
        )
        assert result.is_valid is True

    def test_missing_column_as_error(
        self, valid_df: pd.DataFrame
    ) -> None:
        result = DataFrameValidator.validate_non_negative(
            valid_df, ["NONEXISTENT"], missing_as_warning=False
        )
        assert result.is_valid is False

    def test_custom_negative_severity(
        self, df_with_negatives: pd.DataFrame
    ) -> None:
        result = DataFrameValidator.validate_non_negative(
            df_with_negatives,
            ["CANTIDAD"],
            negative_severity=ValidationSeverity.ERROR,
        )
        neg = [
            i
            for i in result.issues
            if i.code == ValidationCode.NEGATIVE_VALUES
        ]
        assert neg[0].severity == ValidationSeverity.ERROR

    def test_non_numeric_values_ignored(self) -> None:
        """Valores no numéricos no se cuentan como negativos."""
        df = pd.DataFrame({"COL": [1.0, "abc", -3.0, "xyz"]})
        result = DataFrameValidator.validate_non_negative(df, ["COL"])
        neg_issues = [
            i
            for i in result.issues
            if i.code == ValidationCode.NEGATIVE_VALUES
        ]
        assert len(neg_issues) == 1
        assert neg_issues[0].count == 1  # Solo -3.0

    def test_none_dataframe(self) -> None:
        result = DataFrameValidator.validate_non_negative(
            None, ["A"]  # type: ignore[arg-type]
        )
        assert result.is_valid is False


# ======================================================================
# Tests: validate_ranges
# ======================================================================


class TestValidateRanges:
    """Pruebas para validación de rangos."""

    def test_within_range(self, valid_df: pd.DataFrame) -> None:
        result = DataFrameValidator.validate_ranges(
            valid_df, {"CANTIDAD": (0.0, 100.0), "PRECIO": (0.0, 500.0)}
        )
        assert result.is_valid is True
        assert len(result.issues) == 0

    def test_below_min(self) -> None:
        df = pd.DataFrame({"COL": [5.0, -1.0, 10.0]})
        result = DataFrameValidator.validate_ranges(
            df, {"COL": (0.0, 100.0)}
        )
        violations = [
            i
            for i in result.issues
            if i.code == ValidationCode.RANGE_VIOLATION
        ]
        assert len(violations) == 1
        assert violations[0].count == 1

    def test_above_max(self) -> None:
        df = pd.DataFrame({"COL": [5.0, 150.0, 10.0]})
        result = DataFrameValidator.validate_ranges(
            df, {"COL": (0.0, 100.0)}
        )
        violations = [
            i
            for i in result.issues
            if i.code == ValidationCode.RANGE_VIOLATION
        ]
        assert len(violations) == 1
        assert violations[0].count == 1

    def test_open_min(self) -> None:
        """min=None → sin restricción inferior."""
        df = pd.DataFrame({"COL": [-1000.0, 50.0, 200.0]})
        result = DataFrameValidator.validate_ranges(
            df, {"COL": (None, 100.0)}
        )
        violations = [
            i
            for i in result.issues
            if i.code == ValidationCode.RANGE_VIOLATION
        ]
        assert len(violations) == 1
        assert violations[0].count == 1  # 200.0 > 100.0

    def test_open_max(self) -> None:
        """max=None → sin restricción superior."""
        df = pd.DataFrame({"COL": [-5.0, 50.0, 99999.0]})
        result = DataFrameValidator.validate_ranges(
            df, {"COL": (0.0, None)}
        )
        violations = [
            i
            for i in result.issues
            if i.code == ValidationCode.RANGE_VIOLATION
        ]
        assert len(violations) == 1
        assert violations[0].count == 1  # -5.0 < 0.0

    def test_both_open(self) -> None:
        """Ambos None → sin restricción."""
        df = pd.DataFrame(
            {"COL": [-9999.0, 0.0, 9999.0]}
        )
        result = DataFrameValidator.validate_ranges(
            df, {"COL": (None, None)}
        )
        assert len(result.issues) == 0

    def test_range_message_contains_infinity_symbols(self) -> None:
        df = pd.DataFrame({"COL": [200.0]})
        result = DataFrameValidator.validate_ranges(
            df, {"COL": (None, 100.0)}
        )
        violations = [
            i
            for i in result.issues
            if i.code == ValidationCode.RANGE_VIOLATION
        ]
        assert "∞" in violations[0].message

    def test_min_greater_than_max_raises(self) -> None:
        df = pd.DataFrame({"COL": [5.0]})
        with pytest.raises(ValueError, match="min.*>.*max"):
            DataFrameValidator.validate_ranges(
                df, {"COL": (100.0, 0.0)}
            )

    def test_non_finite_bound_raises(self) -> None:
        df = pd.DataFrame({"COL": [5.0]})
        with pytest.raises(ValueError, match="finito"):
            DataFrameValidator.validate_ranges(
                df, {"COL": (float("inf"), 100.0)}
            )

    def test_nan_bound_raises(self) -> None:
        df = pd.DataFrame({"COL": [5.0]})
        with pytest.raises(ValueError, match="finito"):
            DataFrameValidator.validate_ranges(
                df, {"COL": (0.0, float("nan"))}
            )

    def test_non_numeric_bound_raises(self) -> None:
        df = pd.DataFrame({"COL": [5.0]})
        with pytest.raises(ValueError, match="numérico"):
            DataFrameValidator.validate_ranges(
                df, {"COL": ("0", 100.0)}  # type: ignore[dict-item]
            )

    def test_invalid_bounds_tuple_raises(self) -> None:
        df = pd.DataFrame({"COL": [5.0]})
        with pytest.raises(ValueError, match="tuple"):
            DataFrameValidator.validate_ranges(
                df, {"COL": [0.0, 100.0]}  # type: ignore[dict-item]
            )

    def test_bounds_wrong_length_raises(self) -> None:
        df = pd.DataFrame({"COL": [5.0]})
        with pytest.raises(ValueError, match="tuple"):
            DataFrameValidator.validate_ranges(
                df, {"COL": (0.0, 50.0, 100.0)}  # type: ignore[dict-item]
            )

    def test_empty_column_key_raises(self) -> None:
        df = pd.DataFrame({"COL": [5.0]})
        with pytest.raises(ValueError, match="strings no vacíos"):
            DataFrameValidator.validate_ranges(
                df, {"": (0.0, 100.0)}
            )

    def test_none_range_rules_raises(self) -> None:
        df = pd.DataFrame({"COL": [5.0]})
        with pytest.raises(ValueError, match="None"):
            DataFrameValidator.validate_ranges(
                df, None  # type: ignore[arg-type]
            )

    def test_missing_column(self, valid_df: pd.DataFrame) -> None:
        result = DataFrameValidator.validate_ranges(
            valid_df, {"NONEXISTENT": (0.0, 100.0)}
        )
        assert result.is_valid is True  # missing_as_warning=True
        assert len(result.issues) == 1

    def test_missing_column_as_error(
        self, valid_df: pd.DataFrame
    ) -> None:
        result = DataFrameValidator.validate_ranges(
            valid_df,
            {"NONEXISTENT": (0.0, 100.0)},
            missing_as_warning=False,
        )
        assert result.is_valid is False

    def test_none_dataframe(self) -> None:
        result = DataFrameValidator.validate_ranges(
            None, {"A": (0.0, 1.0)}  # type: ignore[arg-type]
        )
        assert result.is_valid is False

    def test_custom_violation_severity(self) -> None:
        df = pd.DataFrame({"COL": [200.0]})
        result = DataFrameValidator.validate_ranges(
            df,
            {"COL": (0.0, 100.0)},
            violation_severity=ValidationSeverity.ERROR,
        )
        violations = [
            i
            for i in result.issues
            if i.code == ValidationCode.RANGE_VIOLATION
        ]
        assert violations[0].severity == ValidationSeverity.ERROR
        assert result.is_valid is False

    def test_non_numeric_values_ignored(self) -> None:
        """Valores no numéricos no se cuentan como violaciones de rango."""
        df = pd.DataFrame({"COL": [5.0, "abc", 200.0]})
        result = DataFrameValidator.validate_ranges(
            df, {"COL": (0.0, 100.0)}
        )
        violations = [
            i
            for i in result.issues
            if i.code == ValidationCode.RANGE_VIOLATION
        ]
        assert violations[0].count == 1  # Solo 200.0

    def test_multiple_rules(self) -> None:
        df = pd.DataFrame(
            {"A": [5.0, 200.0], "B": [-1.0, 50.0]}
        )
        result = DataFrameValidator.validate_ranges(
            df, {"A": (0.0, 100.0), "B": (0.0, 100.0)}
        )
        violations = [
            i
            for i in result.issues
            if i.code == ValidationCode.RANGE_VIOLATION
        ]
        assert len(violations) == 2


# ======================================================================
# Tests: validate_domain (integración)
# ======================================================================


class TestValidateDomain:
    """Pruebas de integración para validación compuesta."""

    def test_all_clean(self, valid_df: pd.DataFrame) -> None:
        result = DataFrameValidator.validate_domain(
            valid_df,
            required_columns=["ID", "CANTIDAD", "PRECIO"],
            critical_columns=["ID"],
            numeric_columns=["CANTIDAD", "PRECIO"],
            non_negative_columns=["CANTIDAD", "PRECIO"],
            range_rules={"CANTIDAD": (0.0, 100.0)},
        )
        assert result.is_valid is True

    def test_missing_required_column(
        self, valid_df: pd.DataFrame
    ) -> None:
        result = DataFrameValidator.validate_domain(
            valid_df,
            required_columns=["ID", "NONEXISTENT"],
        )
        assert result.is_valid is False
        assert result.has_errors is True

    def test_quality_issues_detected(
        self, df_with_nulls: pd.DataFrame
    ) -> None:
        result = DataFrameValidator.validate_domain(
            df_with_nulls,
            critical_columns=["ID", "CANTIDAD"],
        )
        assert result.has_warnings is True
        null_issues = [
            i
            for i in result.issues
            if i.code == ValidationCode.NULL_VALUES
        ]
        assert len(null_issues) > 0

    def test_numeric_and_negative(
        self, df_with_negatives: pd.DataFrame
    ) -> None:
        result = DataFrameValidator.validate_domain(
            df_with_negatives,
            numeric_columns=["CANTIDAD", "PRECIO"],
            non_negative_columns=["CANTIDAD"],
        )
        neg_issues = [
            i
            for i in result.issues
            if i.code == ValidationCode.NEGATIVE_VALUES
        ]
        assert len(neg_issues) == 1
        assert neg_issues[0].column == "CANTIDAD"

    def test_range_violations(self, valid_df: pd.DataFrame) -> None:
        result = DataFrameValidator.validate_domain(
            valid_df,
            range_rules={"CANTIDAD": (0.0, 5.0)},
        )
        violations = [
            i
            for i in result.issues
            if i.code == ValidationCode.RANGE_VIOLATION
        ]
        assert len(violations) == 1

    def test_no_validations_specified(
        self, valid_df: pd.DataFrame
    ) -> None:
        result = DataFrameValidator.validate_domain(valid_df)
        assert result.is_valid is True
        assert len(result.issues) == 0

    def test_none_dataframe(self) -> None:
        result = DataFrameValidator.validate_domain(
            None,  # type: ignore[arg-type]
            required_columns=["A"],
        )
        assert result.is_valid is False

    def test_all_validators_run(self) -> None:
        """Verifica que todos los validadores se ejecutan."""
        df = pd.DataFrame(
            {
                "ID": ["A", None],
                "CANTIDAD": [10.0, -5.0],
                "PRECIO": [float("inf"), 200.0],
            }
        )
        result = DataFrameValidator.validate_domain(
            df,
            required_columns=["ID", "CANTIDAD", "PRECIO", "MISSING"],
            critical_columns=["ID"],
            numeric_columns=["PRECIO"],
            non_negative_columns=["CANTIDAD"],
            range_rules={"CANTIDAD": (0.0, 100.0)},
        )
        codes_found = {i.code for i in result.issues}
        assert ValidationCode.MISSING_REQUIRED_COLUMN in codes_found
        assert ValidationCode.NULL_VALUES in codes_found
        assert ValidationCode.NON_FINITE_VALUES in codes_found
        assert ValidationCode.NEGATIVE_VALUES in codes_found

    def test_generator_columns_consumed_once(
        self, valid_df: pd.DataFrame
    ) -> None:
        """Generadores se materializan correctamente sin doble consumo."""

        def gen_required():
            yield "ID"
            yield "CANTIDAD"

        def gen_numeric():
            yield "CANTIDAD"
            yield "PRECIO"

        result = DataFrameValidator.validate_domain(
            valid_df,
            required_columns=gen_required(),
            numeric_columns=gen_numeric(),
        )
        assert result.is_valid is True

    def test_composite_issue_count(self) -> None:
        df = pd.DataFrame(
            {
                "A": [None, "x"],
                "B": [1.0, "bad"],
                "C": [-1.0, 200.0],
            }
        )
        result = DataFrameValidator.validate_domain(
            df,
            required_columns=["A", "B", "C", "D"],
            critical_columns=["A"],
            numeric_columns=["B"],
            non_negative_columns=["C"],
            range_rules={"C": (0.0, 100.0)},
        )
        assert len(result.issues) > 0
        d = result.to_dict()
        assert d["issue_count"] == len(result.issues)
        assert d["error_count"] == len(result.errors)
        assert d["warning_count"] == len(result.warnings)


# ======================================================================
# Tests: _materialize_iterable
# ======================================================================


class TestMaterializeIterable:
    """Pruebas para la utilidad de materialización."""

    def test_tuple_passthrough(self) -> None:
        t = ("a", "b", "c")
        assert _materialize_iterable(t) is t

    def test_list_to_tuple(self) -> None:
        result = _materialize_iterable(["a", "b"])
        assert result == ("a", "b")
        assert isinstance(result, tuple)

    def test_generator_consumed(self) -> None:
        gen = (x for x in ["a", "b", "c"])
        result = _materialize_iterable(gen)
        assert result == ("a", "b", "c")

    def test_empty(self) -> None:
        assert _materialize_iterable([]) == ()
        assert _materialize_iterable(()) == ()

    def test_frozenset_to_tuple(self) -> None:
        result = _materialize_iterable(frozenset({"a", "b"}))
        assert isinstance(result, tuple)
        assert set(result) == {"a", "b"}


# ======================================================================
# Tests: Propiedades transversales del sistema
# ======================================================================


class TestSystemProperties:
    """Pruebas de propiedades transversales."""

    def test_dataframe_not_mutated_by_schema(
        self, valid_df: pd.DataFrame
    ) -> None:
        original = valid_df.copy()
        DataFrameValidator.validate_schema(
            valid_df, ["ID", "CANTIDAD"]
        )
        pd.testing.assert_frame_equal(valid_df, original)

    def test_dataframe_not_mutated_by_quality(
        self, df_with_nulls: pd.DataFrame
    ) -> None:
        original = df_with_nulls.copy()
        DataFrameValidator.check_data_quality(
            df_with_nulls, ["ID", "CANTIDAD"]
        )
        pd.testing.assert_frame_equal(df_with_nulls, original)

    def test_dataframe_not_mutated_by_numeric(
        self, df_with_non_numeric: pd.DataFrame
    ) -> None:
        original = df_with_non_numeric.copy()
        DataFrameValidator.validate_numeric_columns(
            df_with_non_numeric, ["CANTIDAD"]
        )
        pd.testing.assert_frame_equal(df_with_non_numeric, original)

    def test_dataframe_not_mutated_by_non_negative(
        self, df_with_negatives: pd.DataFrame
    ) -> None:
        original = df_with_negatives.copy()
        DataFrameValidator.validate_non_negative(
            df_with_negatives, ["CANTIDAD"]
        )
        pd.testing.assert_frame_equal(df_with_negatives, original)

    def test_dataframe_not_mutated_by_ranges(
        self, valid_df: pd.DataFrame
    ) -> None:
        original = valid_df.copy()
        DataFrameValidator.validate_ranges(
            valid_df, {"CANTIDAD": (0.0, 100.0)}
        )
        pd.testing.assert_frame_equal(valid_df, original)

    def test_dataframe_not_mutated_by_domain(
        self, valid_df: pd.DataFrame
    ) -> None:
        original = valid_df.copy()
        DataFrameValidator.validate_domain(
            valid_df,
            required_columns=["ID"],
            numeric_columns=["CANTIDAD"],
            non_negative_columns=["PRECIO"],
            range_rules={"CANTIDAD": (0.0, 100.0)},
        )
        pd.testing.assert_frame_equal(valid_df, original)

    def test_deterministic_output(
        self, valid_df: pd.DataFrame
    ) -> None:
        """Dos ejecuciones idénticas producen el mismo resultado."""
        r1 = DataFrameValidator.validate_domain(
            valid_df,
            required_columns=["ID", "CANTIDAD"],
            numeric_columns=["CANTIDAD", "PRECIO"],
            non_negative_columns=["CANTIDAD"],
            range_rules={"CANTIDAD": (0.0, 100.0)},
        )
        r2 = DataFrameValidator.validate_domain(
            valid_df,
            required_columns=["ID", "CANTIDAD"],
            numeric_columns=["CANTIDAD", "PRECIO"],
            non_negative_columns=["CANTIDAD"],
            range_rules={"CANTIDAD": (0.0, 100.0)},
        )
        assert r1.is_valid == r2.is_valid
        assert len(r1.issues) == len(r2.issues)
        for i1, i2 in zip(r1.issues, r2.issues):
            assert i1.severity == i2.severity
            assert i1.code == i2.code
            assert i1.message == i2.message
            assert i1.column == i2.column
            assert i1.count == i2.count

    def test_validation_result_invariant_is_valid(self) -> None:
        """is_valid es consistente con la presencia de errores."""
        # Sin errores → válido
        r1 = ValidationResult.from_issues(
            (
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code=ValidationCode.NULL_VALUES,
                    message="W",
                ),
            )
        )
        assert r1.is_valid is True
        assert r1.has_errors is False

        # Con error → inválido
        r2 = ValidationResult.from_issues(
            (
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code=ValidationCode.MISSING_REQUIRED_COLUMN,
                    message="E",
                ),
            )
        )
        assert r2.is_valid is False
        assert r2.has_errors is True

        # Merge propaga invalidez
        merged = r1.merge(r2)
        assert merged.is_valid is False
        assert merged.has_errors is True
        assert merged.has_warnings is True

    def test_merge_associativity(self) -> None:
        """(a.merge(b)).merge(c) produce mismos issues que a.merge(b.merge(c))."""
        a = ValidationResult.from_issues(
            (
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    code=ValidationCode.DUPLICATE_COLUMNS,
                    message="A",
                ),
            )
        )
        b = ValidationResult.from_issues(
            (
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code=ValidationCode.NULL_VALUES,
                    message="B",
                ),
            )
        )
        c = ValidationResult.from_issues(
            (
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code=ValidationCode.MISSING_REQUIRED_COLUMN,
                    message="C",
                ),
            )
        )

        left = a.merge(b).merge(c)
        right = a.merge(b.merge(c))

        assert len(left.issues) == len(right.issues)
        assert left.is_valid == right.is_valid
        for li, ri in zip(left.issues, right.issues):
            assert li.code == ri.code
            assert li.message == ri.message

    def test_merge_identity(self) -> None:
        """Merge con resultado vacío es identidad."""
        original = ValidationResult.from_issues(
            (
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code=ValidationCode.NULL_VALUES,
                    message="W",
                ),
            )
        )
        identity = ValidationResult.success()
        merged = original.merge(identity)
        assert len(merged.issues) == len(original.issues)
        assert merged.is_valid == original.is_valid

    def test_all_issue_codes_are_unique(self) -> None:
        """No hay códigos duplicados en el enum."""
        values = [c.value for c in ValidationCode]
        assert len(values) == len(set(values))

    def test_all_severity_values_are_unique(self) -> None:
        values = [s.value for s in ValidationSeverity]
        assert len(values) == len(set(values))

    def test_to_dict_round_trip_consistency(self) -> None:
        """to_dict produce estructura consistente con las propiedades."""
        issues = (
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code=ValidationCode.MISSING_REQUIRED_COLUMN,
                message="E",
                column="X",
                count=1,
            ),
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code=ValidationCode.NULL_VALUES,
                message="W",
                column="Y",
                count=5,
            ),
            ValidationIssue(
                severity=ValidationSeverity.INFO,
                code=ValidationCode.DUPLICATE_COLUMNS,
                message="I",
            ),
        )
        result = ValidationResult.from_issues(issues)
        d = result.to_dict()

        assert d["is_valid"] == result.is_valid
        assert d["issue_count"] == len(result.issues)
        assert d["error_count"] == len(result.errors)
        assert d["warning_count"] == len(result.warnings)
        assert len(d["issues"]) == len(result.issues)

        for issue_dict, issue_obj in zip(d["issues"], result.issues):
            assert issue_dict["severity"] == issue_obj.severity.value
            assert issue_dict["code"] == issue_obj.code.value
            assert issue_dict["message"] == issue_obj.message
            assert issue_dict["column"] == issue_obj.column
            assert issue_dict["count"] == issue_obj.count
# ======================================================================
# Tests: Spectral Independence (SVD)
# ======================================================================

class TestSpectralIndependence:
    """Verifica la detección de deficiencia de rango vía SVD."""

    def test_linear_independence_valid(self) -> None:
        """Matriz identidad debe ser independiente."""
        df = pd.DataFrame({
            "A": [1.0, 0.0, 0.0],
            "B": [0.0, 1.0, 0.0],
            "C": [0.0, 0.0, 1.0],
        })
        result = DataFrameValidator.validate_linear_independence(df, ["A", "B", "C"])
        assert result.is_valid
        assert len(result.issues) == 0

    def test_linear_dependency_detected(self) -> None:
        """Detección de colinealidad exacta."""
        df = pd.DataFrame({
            "A": [1.0, 2.0, 3.0],
            "B": [2.0, 4.0, 6.0], # B = 2*A
            "C": [1.0, 0.0, 1.0],
        })
        result = DataFrameValidator.validate_linear_independence(
            df, ["A", "B", "C"], severity=ValidationSeverity.ERROR
        )
        assert not result.is_valid
        assert any(issue.code == ValidationCode.LINEAR_DEPENDENCY for issue in result.issues)
        assert any("κ=" in issue.message for issue in result.issues)

    def test_near_singularity_detected(self) -> None:
        """Detección de casi-singularidad (mal condicionamiento)."""
        eps = 1e-12
        df = pd.DataFrame({
            "A": [1.0, 0.0],
            "B": [1.0, eps], # Casi igual a A
        })
        # κ ≈ 1/eps ≈ 1e12. Si threshold es 1e10, debe fallar.
        result = DataFrameValidator.validate_linear_independence(
            df, ["A", "B"], condition_threshold=1e10
        )
        assert not result.is_valid
        assert any(issue.code == ValidationCode.LINEAR_DEPENDENCY for issue in result.issues)


class TestNumericalPathologies:
    """Verifica la detección de singularidades numéricas (Subnormales, etc)."""

    def test_subnormal_values_detected(self) -> None:
        """Valores < float_info.min deben detectarse."""
        import sys
        subnormal = sys.float_info.min / 10.0
        df = pd.DataFrame({"A": [1.0, subnormal, 0.0]}) # 0.0 no es subnormal

        # validate_numeric_values detecta subnormales
        result = DataFrameValidator.validate_numeric_columns(df, ["A"])

        assert any(issue.code == ValidationCode.SUBNORMAL_VALUES for issue in result.issues)
        assert any(issue.count == 1 for issue in result.issues) # Solo el subnormal
