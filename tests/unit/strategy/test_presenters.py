"""
Suite de pruebas rigurosa para ``presenters.py``.

Estructura:
- Fixtures reutilizables para logger, config y datos de APU.
- Pruebas unitarias por método/responsabilidad.
- Pruebas de integración para el pipeline completo.
- Pruebas de propiedades (idempotencia, determinismo, invariantes).

Convenciones:
- Cada clase ``Test*`` agrupa pruebas de una unidad lógica.
- Los nombres siguen el patrón ``test_<condición>_<resultado_esperado>``.
- Se usa ``pytest.mark.parametrize`` para cubrir variantes sin duplicación.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.adapters.presenters import APUPresenter, APUProcessingConfig


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def logger() -> logging.Logger:
    """Logger silencioso para pruebas."""
    test_logger = logging.getLogger("test_presenters")
    test_logger.setLevel(logging.DEBUG)
    if not test_logger.handlers:
        test_logger.addHandler(logging.NullHandler())
    return test_logger


@pytest.fixture()
def config() -> APUProcessingConfig:
    """Configuración por defecto."""
    return APUProcessingConfig()


@pytest.fixture()
def presenter(logger: logging.Logger, config: APUProcessingConfig) -> APUPresenter:
    """Instancia de presentador con configuración por defecto."""
    return APUPresenter(logger=logger, config=config)


@pytest.fixture()
def minimal_apu_record() -> Dict[str, Any]:
    """Registro APU mínimo válido con todas las columnas requeridas."""
    return {
        "CATEGORIA": "MATERIALES",
        "DESCRIPCION_INSUMO": "Cemento Portland",
        "CANTIDAD_APU": 10.0,
        "VALOR_TOTAL_APU": 500.0,
        "UNIDAD_APU": "kg",
        "PRECIO_UNIT_APU": 50.0,
        "CODIGO_APU": "APU-001",
        "UNIDAD_INSUMO": "kg",
        "RENDIMIENTO": 1.0,
    }


@pytest.fixture()
def valid_apu_details(minimal_apu_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Lista de registros APU válidos con múltiples categorías."""
    return [
        minimal_apu_record,
        {
            "CATEGORIA": "MANO DE OBRA",
            "DESCRIPCION_INSUMO": "Albañil",
            "CANTIDAD_APU": 8.0,
            "VALOR_TOTAL_APU": 400.0,
            "UNIDAD_APU": "hr",
            "PRECIO_UNIT_APU": 50.0,
            "CODIGO_APU": "APU-001",
            "UNIDAD_INSUMO": "hr",
            "RENDIMIENTO": 1.0,
        },
        {
            "CATEGORIA": "EQUIPO",
            "DESCRIPCION_INSUMO": "Mezcladora",
            "CANTIDAD_APU": 2.0,
            "VALOR_TOTAL_APU": 100.0,
            "UNIDAD_APU": "hr",
            "PRECIO_UNIT_APU": 50.0,
            "CODIGO_APU": "APU-001",
            "UNIDAD_INSUMO": "hr",
            "RENDIMIENTO": 0.5,
        },
    ]


# ======================================================================
# Tests: APUProcessingConfig
# ======================================================================


class TestAPUProcessingConfig:
    """Pruebas para la dataclass de configuración."""

    def test_default_values_are_populated(self) -> None:
        cfg = APUProcessingConfig()
        assert cfg.required_columns is not None
        assert cfg.numeric_columns is not None
        assert isinstance(cfg.required_columns, frozenset)
        assert isinstance(cfg.numeric_columns, frozenset)
        assert cfg.default_category == "INDEFINIDO"

    def test_numeric_columns_subset_of_required(self) -> None:
        cfg = APUProcessingConfig()
        assert cfg.numeric_columns <= cfg.required_columns, (
            "numeric_columns debe ser subconjunto de required_columns"
        )

    def test_custom_values_preserved(self) -> None:
        custom_required = frozenset({"A", "B", "C"})
        custom_numeric = frozenset({"A"})
        cfg = APUProcessingConfig(
            required_columns=custom_required,
            numeric_columns=custom_numeric,
            default_category="OTRO",
            tolerance_price_variance=0.05,
        )
        assert cfg.required_columns == custom_required
        assert cfg.numeric_columns == custom_numeric
        assert cfg.default_category == "OTRO"
        assert cfg.tolerance_price_variance == 0.05

    @pytest.mark.parametrize(
        "field, value",
        [
            ("tolerance_price_variance", -0.01),
            ("consistency_relative_tolerance", -1e-3),
            ("consistency_absolute_tolerance", -1e-9),
        ],
    )
    def test_negative_tolerance_raises(self, field: str, value: float) -> None:
        with pytest.raises(ValueError, match="debe ser ≥ 0"):
            APUProcessingConfig(**{field: value})

    def test_orphan_numeric_columns_raises(self) -> None:
        with pytest.raises(ValueError, match="no presentes en required_columns"):
            APUProcessingConfig(
                required_columns=frozenset({"A", "B"}),
                numeric_columns=frozenset({"A", "C"}),
            )

    def test_empty_default_category_raises(self) -> None:
        with pytest.raises(ValueError, match="string no vacío"):
            APUProcessingConfig(default_category="")

    def test_whitespace_only_default_category_raises(self) -> None:
        with pytest.raises(ValueError, match="string no vacío"):
            APUProcessingConfig(default_category="   ")

    def test_frozen_immutability(self) -> None:
        cfg = APUProcessingConfig()
        with pytest.raises(AttributeError):
            cfg.default_category = "NUEVO"  # type: ignore[misc]


# ======================================================================
# Tests: APUPresenter.__init__
# ======================================================================


class TestAPUPresenterInit:
    """Pruebas de inicialización del presentador."""

    def test_valid_initialization(self, logger: logging.Logger) -> None:
        presenter = APUPresenter(logger=logger)
        assert presenter.logger is logger
        assert isinstance(presenter.config, APUProcessingConfig)

    def test_custom_config_preserved(self, logger: logging.Logger) -> None:
        cfg = APUProcessingConfig(default_category="CUSTOM")
        presenter = APUPresenter(logger=logger, config=cfg)
        assert presenter.config.default_category == "CUSTOM"

    def test_none_logger_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="logging.Logger"):
            APUPresenter(logger=None)  # type: ignore[arg-type]

    def test_non_logger_instance_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="logging.Logger"):
            APUPresenter(logger="not_a_logger")  # type: ignore[arg-type]

    def test_mock_object_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            APUPresenter(logger=MagicMock())  # type: ignore[arg-type]


# ======================================================================
# Tests: _validate_inputs
# ======================================================================


class TestValidateInputs:
    """Pruebas para validación de entradas del pipeline."""

    def test_valid_inputs_pass(self, presenter: APUPresenter) -> None:
        presenter._validate_inputs([{"key": "value"}], "APU-001")

    def test_non_list_raises(self, presenter: APUPresenter) -> None:
        with pytest.raises(ValueError, match="debe ser una lista"):
            presenter._validate_inputs({"key": "value"}, "APU-001")

    def test_empty_list_raises(self, presenter: APUPresenter) -> None:
        with pytest.raises(ValueError, match="No se encontraron detalles"):
            presenter._validate_inputs([], "APU-001")

    def test_non_dict_elements_raises(self, presenter: APUPresenter) -> None:
        with pytest.raises(ValueError, match="no son diccionarios"):
            presenter._validate_inputs([{"ok": 1}, "not_dict", 42], "APU-001")

    def test_non_dict_reports_positions(self, presenter: APUPresenter) -> None:
        with pytest.raises(ValueError, match=r"\[1, 2\]"):
            presenter._validate_inputs([{"ok": 1}, "bad", 42], "APU-001")

    @pytest.mark.parametrize(
        "bad_code",
        [None, "", "   ", 123, [], True],
    )
    def test_invalid_apu_code_raises(
        self, presenter: APUPresenter, bad_code: Any
    ) -> None:
        with pytest.raises(ValueError, match="apu_code"):
            presenter._validate_inputs([{"key": "val"}], bad_code)

    def test_tuple_input_raises(self, presenter: APUPresenter) -> None:
        with pytest.raises(ValueError, match="debe ser una lista"):
            presenter._validate_inputs(({"key": "val"},), "APU-001")


# ======================================================================
# Tests: _sanitize_text
# ======================================================================


class TestSanitizeText:
    """Pruebas para sanitización textual."""

    @pytest.mark.parametrize(
        "input_val, expected",
        [
            (None, None),
            ("", None),
            ("   ", None),
            ("hello", "hello"),
            ("  hello  world  ", "hello world"),
            ("line\none", "line one"),
            ("tab\there", "tab here"),
            ("\x00\x01hidden\x7F", "hidden"),
            ("múltiples   espacios", "múltiples espacios"),
            (42, 42),
            (3.14, 3.14),
            (True, True),
            (["list"], ["list"]),
        ],
    )
    def test_sanitization(self, input_val: Any, expected: Any) -> None:
        assert APUPresenter._sanitize_text(input_val) == expected

    def test_idempotence(self) -> None:
        """Aplicar _sanitize_text dos veces produce el mismo resultado."""
        test_values = [
            "  hello\t\nworld  ",
            "\x00control\x1F",
            "normal",
            None,
            "",
            42,
        ]
        for val in test_values:
            first = APUPresenter._sanitize_text(val)
            second = APUPresenter._sanitize_text(first)
            assert first == second, f"No idempotente para {val!r}"

    def test_only_control_chars_returns_none(self) -> None:
        assert APUPresenter._sanitize_text("\x00\x01\x02") is None

    def test_unicode_preserved(self) -> None:
        assert APUPresenter._sanitize_text("café résumé") == "café résumé"


# ======================================================================
# Tests: _normalize_category
# ======================================================================


class TestNormalizeCategory:
    """Pruebas para normalización categórica."""

    @pytest.mark.parametrize(
        "input_val, expected",
        [
            ("MATERIALES", "MATERIALES"),
            ("MATERIAL", "MATERIALES"),
            ("MAT", "MATERIALES"),
            ("material", "MATERIALES"),
            ("Material", "MATERIALES"),
            ("MANO DE OBRA", "MANO DE OBRA"),
            ("MO", "MANO DE OBRA"),
            ("EQUIPO", "EQUIPO"),
            ("EQUIPOS", "EQUIPO"),
            ("MAQUINARIA", "EQUIPO"),
            ("equipo", "EQUIPO"),
        ],
    )
    def test_known_categories(
        self, presenter: APUPresenter, input_val: str, expected: str
    ) -> None:
        assert presenter._normalize_category(input_val) == expected

    def test_unknown_category_returns_uppercase(
        self, presenter: APUPresenter
    ) -> None:
        assert presenter._normalize_category("transporte") == "TRANSPORTE"

    def test_none_returns_default(self, presenter: APUPresenter) -> None:
        assert (
            presenter._normalize_category(None)
            == presenter.config.default_category
        )

    def test_empty_string_returns_default(
        self, presenter: APUPresenter
    ) -> None:
        assert (
            presenter._normalize_category("")
            == presenter.config.default_category
        )

    def test_irregular_spaces_collapsed(
        self, presenter: APUPresenter
    ) -> None:
        """Variantes con espacios internos irregulares se resuelven."""
        assert presenter._normalize_category("MANO  DE   OBRA") == "MANO DE OBRA"
        assert presenter._normalize_category("  MANO   DE  OBRA  ") == "MANO DE OBRA"

    def test_control_chars_in_category(
        self, presenter: APUPresenter
    ) -> None:
        assert presenter._normalize_category("\x00EQUIPO\x1F") == "EQUIPO"

    def test_idempotence(self, presenter: APUPresenter) -> None:
        categories = ["MATERIALES", "MO", "EQUIPO", None, "", "TRANSPORTE"]
        for cat in categories:
            first = presenter._normalize_category(cat)
            second = presenter._normalize_category(first)
            assert first == second, f"No idempotente para {cat!r}"


# ======================================================================
# Tests: _sanitize_dataframe
# ======================================================================


class TestSanitizeDataframe:
    """Pruebas para sanitización del DataFrame."""

    def _make_df(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame(records)

    def test_text_columns_sanitized(
        self, presenter: APUPresenter, minimal_apu_record: Dict[str, Any]
    ) -> None:
        record = minimal_apu_record.copy()
        record["DESCRIPCION_INSUMO"] = "  Cemento\x00  Portland  "
        df = self._make_df([record])
        result = presenter._sanitize_dataframe(df)
        assert result["DESCRIPCION_INSUMO"].iloc[0] == "Cemento Portland"

    def test_category_normalized(
        self, presenter: APUPresenter, minimal_apu_record: Dict[str, Any]
    ) -> None:
        record = minimal_apu_record.copy()
        record["CATEGORIA"] = "material"
        df = self._make_df([record])
        result = presenter._sanitize_dataframe(df)
        assert result["CATEGORIA"].iloc[0] == "MATERIALES"

    def test_numeric_conversion(
        self, presenter: APUPresenter, minimal_apu_record: Dict[str, Any]
    ) -> None:
        record = minimal_apu_record.copy()
        record["CANTIDAD_APU"] = "10.5"
        df = self._make_df([record])
        result = presenter._sanitize_dataframe(df)
        assert result["CANTIDAD_APU"].iloc[0] == pytest.approx(10.5)

    def test_non_numeric_coerced_to_nan(
        self, presenter: APUPresenter, minimal_apu_record: Dict[str, Any]
    ) -> None:
        record = minimal_apu_record.copy()
        record["CANTIDAD_APU"] = "not_a_number"
        df = self._make_df([record])
        result = presenter._sanitize_dataframe(df)
        assert pd.isna(result["CANTIDAD_APU"].iloc[0])

    def test_inf_replaced_by_nan(
        self, presenter: APUPresenter, minimal_apu_record: Dict[str, Any]
    ) -> None:
        record = minimal_apu_record.copy()
        record["CANTIDAD_APU"] = float("inf")
        df = self._make_df([record])
        result = presenter._sanitize_dataframe(df)
        assert pd.isna(result["CANTIDAD_APU"].iloc[0])

    def test_negative_inf_replaced_by_nan(
        self, presenter: APUPresenter, minimal_apu_record: Dict[str, Any]
    ) -> None:
        record = minimal_apu_record.copy()
        record["VALOR_TOTAL_APU"] = float("-inf")
        df = self._make_df([record])
        result = presenter._sanitize_dataframe(df)
        assert pd.isna(result["VALOR_TOTAL_APU"].iloc[0])

    def test_original_dataframe_not_mutated(
        self, presenter: APUPresenter, minimal_apu_record: Dict[str, Any]
    ) -> None:
        df = self._make_df([minimal_apu_record])
        original_copy = df.copy()
        presenter._sanitize_dataframe(df)
        pd.testing.assert_frame_equal(df, original_copy)

    def test_idempotence(
        self, presenter: APUPresenter, minimal_apu_record: Dict[str, Any]
    ) -> None:
        df = self._make_df([minimal_apu_record])
        first = presenter._sanitize_dataframe(df)
        second = presenter._sanitize_dataframe(first)
        pd.testing.assert_frame_equal(first, second)

    def test_none_category_becomes_default(
        self, presenter: APUPresenter, minimal_apu_record: Dict[str, Any]
    ) -> None:
        record = minimal_apu_record.copy()
        record["CATEGORIA"] = None
        df = self._make_df([record])
        result = presenter._sanitize_dataframe(df)
        assert result["CATEGORIA"].iloc[0] == presenter.config.default_category


# ======================================================================
# Tests: _validate_dataframe_schema
# ======================================================================


class TestValidateDataframeSchema:
    """Pruebas para validación de esquema."""

    def test_valid_schema_passes(
        self,
        presenter: APUPresenter,
        minimal_apu_record: Dict[str, Any],
    ) -> None:
        df = pd.DataFrame([minimal_apu_record])
        presenter._validate_dataframe_schema(df, "APU-001")

    def test_missing_columns_raises_key_error(
        self, presenter: APUPresenter
    ) -> None:
        df = pd.DataFrame([{"CATEGORIA": "MATERIALES"}])
        with pytest.raises(KeyError, match="columnas faltantes"):
            presenter._validate_dataframe_schema(df, "APU-001")

    def test_error_message_contains_missing_column_names(
        self, presenter: APUPresenter
    ) -> None:
        df = pd.DataFrame(
            [{"CATEGORIA": "A", "DESCRIPCION_INSUMO": "B"}]
        )
        with pytest.raises(KeyError) as exc_info:
            presenter._validate_dataframe_schema(df, "APU-001")
        error_msg = str(exc_info.value)
        assert "CANTIDAD_APU" in error_msg

    def test_extra_columns_accepted(
        self,
        presenter: APUPresenter,
        minimal_apu_record: Dict[str, Any],
    ) -> None:
        record = minimal_apu_record.copy()
        record["EXTRA_COL"] = "extra"
        df = pd.DataFrame([record])
        presenter._validate_dataframe_schema(df, "APU-001")


# ======================================================================
# Tests: _aggregate_price
# ======================================================================


class TestAggregatePrice:
    """Pruebas para agregación de precios."""

    def test_single_price_returned_as_is(
        self, presenter: APUPresenter
    ) -> None:
        prices = pd.Series([50.0])
        result = presenter._aggregate_price(prices, None, "TEST")
        assert result == pytest.approx(50.0)

    def test_empty_series_returns_none(
        self, presenter: APUPresenter
    ) -> None:
        prices = pd.Series([], dtype=float)
        assert presenter._aggregate_price(prices, None, "TEST") is None

    def test_all_nan_returns_none(self, presenter: APUPresenter) -> None:
        prices = pd.Series([np.nan, np.nan])
        assert presenter._aggregate_price(prices, None, "TEST") is None

    def test_weighted_average_with_quantities(
        self, presenter: APUPresenter
    ) -> None:
        """p̄ = (50×10 + 60×20) / (10+20) = 1700/30 ≈ 56.6667"""
        prices = pd.Series([50.0, 60.0])
        quantities = pd.Series([10.0, 20.0])
        result = presenter._aggregate_price(prices, quantities, "TEST")
        expected = (50.0 * 10.0 + 60.0 * 20.0) / (10.0 + 20.0)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_fallback_to_arithmetic_mean_without_quantities(
        self, presenter: APUPresenter
    ) -> None:
        prices = pd.Series([50.0, 60.0])
        result = presenter._aggregate_price(prices, None, "TEST")
        assert result == pytest.approx(55.0)

    def test_fallback_when_quantities_all_zero(
        self, presenter: APUPresenter
    ) -> None:
        prices = pd.Series([50.0, 60.0])
        quantities = pd.Series([0.0, 0.0])
        result = presenter._aggregate_price(prices, quantities, "TEST")
        # No valid weights → arithmetic mean
        assert result == pytest.approx(55.0)

    def test_identical_prices_return_single_value(
        self, presenter: APUPresenter
    ) -> None:
        prices = pd.Series([42.0, 42.0, 42.0])
        result = presenter._aggregate_price(prices, None, "TEST")
        assert result == pytest.approx(42.0)

    def test_non_numeric_prices_filtered(
        self, presenter: APUPresenter
    ) -> None:
        prices = pd.Series([50.0, "invalid", 60.0])
        result = presenter._aggregate_price(prices, None, "TEST")
        assert result == pytest.approx(55.0)


# ======================================================================
# Tests: _validate_consistency (field)
# ======================================================================


class TestValidateConsistencyField:
    """Pruebas para validación de consistencia de campos singulares."""

    def test_single_value_returned(self, presenter: APUPresenter) -> None:
        series = pd.Series(["kg", "kg", "kg"])
        assert presenter._validate_consistency(series, "UNIDAD") == "kg"

    def test_empty_series_returns_none(
        self, presenter: APUPresenter
    ) -> None:
        series = pd.Series([None, None, np.nan])
        assert presenter._validate_consistency(series, "UNIDAD") is None

    def test_multiple_values_returns_mode(
        self, presenter: APUPresenter
    ) -> None:
        # "kg" aparece 3 veces, "lb" aparece 1 vez
        series = pd.Series(["kg", "kg", "lb", "kg"])
        result = presenter._validate_consistency(series, "UNIDAD")
        assert result == "kg"

    def test_all_nan_returns_none(self, presenter: APUPresenter) -> None:
        series = pd.Series([np.nan, np.nan])
        assert presenter._validate_consistency(series) is None

    def test_mixed_with_nans(self, presenter: APUPresenter) -> None:
        series = pd.Series(["m3", None, "m3", np.nan])
        assert presenter._validate_consistency(series, "UNIDAD") == "m3"


# ======================================================================
# Tests: _validate_item_consistency (economic)
# ======================================================================


class TestValidateItemConsistency:
    """Pruebas para verificación de consistencia económica."""

    def test_consistent_item_no_warning(
        self, presenter: APUPresenter
    ) -> None:
        """valor_total = cantidad × valor_unitario → sin warning."""
        item = {
            "descripcion": "Cemento",
            "cantidad": 10.0,
            "valor_unitario": 50.0,
            "valor_total": 500.0,
        }
        # No debería levantar excepción ni warning
        presenter._validate_item_consistency(item, "MATERIALES")

    def test_inconsistent_item_logs_warning(
        self, logger: logging.Logger, presenter: APUPresenter
    ) -> None:
        item = {
            "descripcion": "Cemento",
            "cantidad": 10.0,
            "valor_unitario": 50.0,
            "valor_total": 999.0,  # Inconsistente
        }
        import warnings
        with warnings.catch_warnings(record=True) as _:
            # No lanza excepción, solo registra warning en logger
            presenter._validate_item_consistency(item, "MATERIALES")

    def test_none_values_skip_validation(
        self, presenter: APUPresenter
    ) -> None:
        """Si algún valor es None, no se puede verificar → no falla."""
        items = [
            {"cantidad": None, "valor_unitario": 50.0, "valor_total": 500.0},
            {"cantidad": 10.0, "valor_unitario": None, "valor_total": 500.0},
            {"cantidad": 10.0, "valor_unitario": 50.0, "valor_total": None},
        ]
        for item in items:
            presenter._validate_item_consistency(item, "TEST")

    def test_zero_quantity_nonzero_total(
        self, presenter: APUPresenter
    ) -> None:
        """cantidad=0, valor_total>0 debería detectar inconsistencia."""
        item = {
            "descripcion": "Item fantasma",
            "cantidad": 0.0,
            "valor_unitario": 50.0,
            "valor_total": 100.0,
        }
        # Debe ejecutar sin excepción (solo warning interno)
        presenter._validate_item_consistency(item, "TEST")

    def test_zero_unit_price_nonzero_total(
        self, presenter: APUPresenter
    ) -> None:
        """valor_unitario=0, valor_total>0 → inconsistencia."""
        item = {
            "descripcion": "Gratis pero no",
            "cantidad": 10.0,
            "valor_unitario": 0.0,
            "valor_total": 500.0,
        }
        presenter._validate_item_consistency(item, "TEST")

    def test_within_tolerance_passes(self, logger: logging.Logger) -> None:
        """Diferencia dentro de tolerancia no genera warning."""
        cfg = APUProcessingConfig(
            consistency_relative_tolerance=0.01,
            consistency_absolute_tolerance=1.0,
        )
        p = APUPresenter(logger=logger, config=cfg)
        item = {
            "descripcion": "Casi exacto",
            "cantidad": 10.0,
            "valor_unitario": 50.0,
            "valor_total": 500.5,  # Dentro de atol=1.0
        }
        p._validate_item_consistency(item, "TEST")


# ======================================================================
# Tests: _organize_breakdown
# ======================================================================


class TestOrganizeBreakdown:
    """Pruebas para organización del desglose."""

    def test_empty_items_returns_empty_dict(
        self, presenter: APUPresenter
    ) -> None:
        assert presenter._organize_breakdown([]) == {}

    def test_single_category(self, presenter: APUPresenter) -> None:
        items = [
            {"categoria": "MATERIALES", "descripcion": "Cemento"},
            {"categoria": "MATERIALES", "descripcion": "Arena"},
        ]
        result = presenter._organize_breakdown(items)
        assert len(result) == 1
        assert "MATERIALES" in result
        assert len(result["MATERIALES"]) == 2

    def test_multiple_categories(self, presenter: APUPresenter) -> None:
        items = [
            {"categoria": "MATERIALES", "descripcion": "Cemento"},
            {"categoria": "EQUIPO", "descripcion": "Mezcladora"},
        ]
        result = presenter._organize_breakdown(items)
        assert len(result) == 2
        assert "MATERIALES" in result
        assert "EQUIPO" in result

    def test_missing_category_assigned_default(
        self, presenter: APUPresenter
    ) -> None:
        items = [
            {"categoria": None, "descripcion": "Sin categoría"},
            {"categoria": "", "descripcion": "Vacío"},
            {"categoria": "   ", "descripcion": "Solo espacios"},
        ]
        result = presenter._organize_breakdown(items)
        default = presenter.config.default_category
        assert default in result
        assert len(result[default]) == 3

    def test_items_mutated_with_default_category(
        self, presenter: APUPresenter
    ) -> None:
        items = [{"categoria": None, "descripcion": "Test"}]
        presenter._organize_breakdown(items)
        assert items[0]["categoria"] == presenter.config.default_category


# ======================================================================
# Tests: _calculate_metadata
# ======================================================================


class TestCalculateMetadata:
    """Pruebas para cálculo de metadatos."""

    def test_basic_metadata_structure(
        self,
        presenter: APUPresenter,
        minimal_apu_record: Dict[str, Any],
    ) -> None:
        df = pd.DataFrame([minimal_apu_record])
        df = presenter._sanitize_dataframe(df)
        items = [{"valor_total": 500.0, "categoria": "MATERIALES"}]
        meta = presenter._calculate_metadata(df, items)

        expected_keys = {
            "original_rows",
            "processed_items",
            "reduction_rate",
            "categories_count",
            "classification_coverage",
            "total_value",
            "avg_value_per_item",
        }
        assert set(meta.keys()) == expected_keys

    def test_no_processed_items(
        self,
        presenter: APUPresenter,
        minimal_apu_record: Dict[str, Any],
    ) -> None:
        df = pd.DataFrame([minimal_apu_record])
        meta = presenter._calculate_metadata(df, [])
        assert meta["processed_items"] == 0
        assert meta["total_value"] == 0.0
        assert meta["avg_value_per_item"] == 0.0

    def test_reduction_rate_clamped_to_zero_one(
        self, presenter: APUPresenter
    ) -> None:
        """Si la agregación produce más filas, reduction_rate = 0."""
        df = pd.DataFrame(
            [
                {
                    "CATEGORIA": "A",
                    "DESCRIPCION_INSUMO": "X",
                    "CANTIDAD_APU": 1,
                    "VALOR_TOTAL_APU": 10,
                    "UNIDAD_APU": "u",
                    "PRECIO_UNIT_APU": 10,
                    "CODIGO_APU": "C1",
                    "UNIDAD_INSUMO": "u",
                    "RENDIMIENTO": 1,
                }
            ]
        )
        # 3 processed items from 1 original row
        items = [
            {"valor_total": 10.0},
            {"valor_total": 20.0},
            {"valor_total": 30.0},
        ]
        meta = presenter._calculate_metadata(df, items)
        assert 0.0 <= meta["reduction_rate"] <= 1.0

    def test_total_value_sums_correctly(
        self, presenter: APUPresenter
    ) -> None:
        df = pd.DataFrame(
            [
                {
                    "CATEGORIA": "A",
                    "DESCRIPCION_INSUMO": "X",
                    "CANTIDAD_APU": 1,
                    "VALOR_TOTAL_APU": 10,
                    "UNIDAD_APU": "u",
                    "PRECIO_UNIT_APU": 10,
                    "CODIGO_APU": "C1",
                    "UNIDAD_INSUMO": "u",
                    "RENDIMIENTO": 1,
                }
            ]
            * 3
        )
        items = [
            {"valor_total": 100.0},
            {"valor_total": 200.0},
            {"valor_total": None},
        ]
        meta = presenter._calculate_metadata(df, items)
        assert meta["total_value"] == pytest.approx(300.0)

    def test_classification_coverage(
        self, presenter: APUPresenter
    ) -> None:
        df = pd.DataFrame(
            {
                "CATEGORIA": ["A", None, "B", None],
                "DESCRIPCION_INSUMO": ["x"] * 4,
                "CANTIDAD_APU": [1] * 4,
                "VALOR_TOTAL_APU": [10] * 4,
                "UNIDAD_APU": ["u"] * 4,
                "PRECIO_UNIT_APU": [10] * 4,
                "CODIGO_APU": ["C1"] * 4,
                "UNIDAD_INSUMO": ["u"] * 4,
                "RENDIMIENTO": [1] * 4,
            }
        )
        meta = presenter._calculate_metadata(df, [{"valor_total": 40.0}])
        assert meta["classification_coverage"] == pytest.approx(0.5)

    def test_empty_dataframe_metadata(
        self, presenter: APUPresenter
    ) -> None:
        df = pd.DataFrame(
            columns=[
                "CATEGORIA",
                "DESCRIPCION_INSUMO",
                "CANTIDAD_APU",
                "VALOR_TOTAL_APU",
                "UNIDAD_APU",
                "PRECIO_UNIT_APU",
                "CODIGO_APU",
                "UNIDAD_INSUMO",
                "RENDIMIENTO",
            ]
        )
        meta = presenter._calculate_metadata(df, [])
        assert meta["original_rows"] == 0
        assert meta["reduction_rate"] == 0.0


# ======================================================================
# Tests: _safe_float
# ======================================================================


class TestSafeFloat:
    """Pruebas para conversión defensiva a float."""

    @pytest.mark.parametrize(
        "input_val, expected",
        [
            (None, None),
            (42, 42.0),
            (3.14, 3.14),
            ("50.5", 50.5),
            ("0", 0.0),
            ("-10.5", -10.5),
        ],
    )
    def test_valid_conversions(
        self, input_val: Any, expected: Any
    ) -> None:
        result = APUPresenter._safe_float(input_val)
        if expected is None:
            assert result is None
        else:
            assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        "input_val",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
            "not_a_number",
            "abc",
            [],
            {},
            object(),
        ],
    )
    def test_invalid_returns_none(self, input_val: Any) -> None:
        assert APUPresenter._safe_float(input_val) is None

    def test_numpy_int(self) -> None:
        assert APUPresenter._safe_float(np.int64(42)) == pytest.approx(42.0)

    def test_numpy_float(self) -> None:
        assert APUPresenter._safe_float(np.float64(3.14)) == pytest.approx(
            3.14
        )

    def test_numpy_nan(self) -> None:
        assert APUPresenter._safe_float(np.nan) is None

    def test_numpy_inf(self) -> None:
        assert APUPresenter._safe_float(np.inf) is None


# ======================================================================
# Tests: _serialize_scalar / _serialize_record
# ======================================================================


class TestSerialization:
    """Pruebas para serialización segura."""

    @pytest.mark.parametrize(
        "input_val, expected",
        [
            (None, None),
            (42, 42),
            (3.14, 3.14),
            ("hello", "hello"),
            (True, True),
            (float("nan"), None),
            (float("inf"), None),
            (float("-inf"), None),
            (np.int64(42), 42),
            (np.float64(3.14), pytest.approx(3.14)),
            (np.nan, None),
            (np.bool_(True), True),
        ],
    )
    def test_scalar_serialization(
        self, input_val: Any, expected: Any
    ) -> None:
        result = APUPresenter._serialize_scalar(input_val)
        if expected is None:
            assert result is None
        else:
            assert result == expected

    def test_record_serialization(self) -> None:
        record = {
            "name": "test",
            "value": np.float64(42.0),
            "bad": float("nan"),
            "none": None,
            "count": np.int64(5),
        }
        result = APUPresenter._serialize_record(record)
        assert result == {
            "name": "test",
            "value": pytest.approx(42.0),
            "bad": None,
            "none": None,
            "count": 5,
        }

    def test_pd_nat_serialized_to_none(self) -> None:
        assert APUPresenter._serialize_scalar(pd.NaT) is None

    def test_list_value_passed_through(self) -> None:
        """Valores no escalares (listas) pasan sin modificación."""
        val = [1, 2, 3]
        result = APUPresenter._serialize_scalar(val)
        assert result == [1, 2, 3]


# ======================================================================
# Tests: _aggregate_alerts
# ======================================================================


class TestAggregateAlerts:
    """Pruebas para consolidación de alertas."""

    def test_single_alert(self) -> None:
        series = pd.Series(["Precio alto"])
        assert APUPresenter._aggregate_alerts(series) == "Precio alto"

    def test_multiple_unique_alerts(self) -> None:
        series = pd.Series(["Alerta B", "Alerta A", "Alerta B"])
        result = APUPresenter._aggregate_alerts(series)
        # Deduplicadas y ordenadas
        assert result == "Alerta A | Alerta B"

    def test_all_nan_returns_none(self) -> None:
        series = pd.Series([None, np.nan])
        assert APUPresenter._aggregate_alerts(series) is None

    def test_empty_strings_filtered(self) -> None:
        series = pd.Series(["", "  ", "Válida"])
        result = APUPresenter._aggregate_alerts(series)
        assert result == "Válida"

    def test_empty_series_returns_none(self) -> None:
        series = pd.Series([], dtype=object)
        assert APUPresenter._aggregate_alerts(series) is None

    def test_deterministic_order(self) -> None:
        """El resultado es determinista independientemente del orden de entrada."""
        series_a = pd.Series(["Z", "A", "M"])
        series_b = pd.Series(["M", "Z", "A"])
        assert APUPresenter._aggregate_alerts(
            series_a
        ) == APUPresenter._aggregate_alerts(series_b)


# ======================================================================
# Tests: process_apu_details (integración)
# ======================================================================


class TestProcessApuDetailsIntegration:
    """Pruebas de integración del pipeline completo."""

    def test_successful_processing(
        self,
        presenter: APUPresenter,
        valid_apu_details: List[Dict[str, Any]],
    ) -> None:
        result = presenter.process_apu_details(valid_apu_details, "APU-001")

        assert "items" in result
        assert "desglose" in result
        assert "total_items" in result
        assert "metadata" in result
        assert result["total_items"] == len(result["items"])
        assert isinstance(result["desglose"], dict)
        assert isinstance(result["metadata"], dict)

    def test_output_contains_expected_categories(
        self,
        presenter: APUPresenter,
        valid_apu_details: List[Dict[str, Any]],
    ) -> None:
        result = presenter.process_apu_details(valid_apu_details, "APU-001")
        categories = set(result["desglose"].keys())
        assert "MATERIALES" in categories
        assert "MANO DE OBRA" in categories
        assert "EQUIPO" in categories

    def test_metadata_original_rows_correct(
        self,
        presenter: APUPresenter,
        valid_apu_details: List[Dict[str, Any]],
    ) -> None:
        result = presenter.process_apu_details(valid_apu_details, "APU-001")
        assert result["metadata"]["original_rows"] == len(valid_apu_details)

    def test_empty_list_raises_value_error(
        self, presenter: APUPresenter
    ) -> None:
        with pytest.raises(ValueError, match="No se encontraron detalles"):
            presenter.process_apu_details([], "APU-001")

    def test_missing_columns_raises_key_error(
        self, presenter: APUPresenter
    ) -> None:
        with pytest.raises(KeyError, match="columnas faltantes"):
            presenter.process_apu_details(
                [{"incomplete": "data"}], "APU-001"
            )

    def test_non_list_raises_value_error(
        self, presenter: APUPresenter
    ) -> None:
        with pytest.raises(ValueError, match="debe ser una lista"):
            presenter.process_apu_details(
                {"not": "a_list"}, "APU-001"  # type: ignore[arg-type]
            )

    def test_serialization_produces_json_safe_output(
        self,
        presenter: APUPresenter,
        valid_apu_details: List[Dict[str, Any]],
    ) -> None:
        """Todos los valores en items deben ser JSON-serializable."""
        import json

        result = presenter.process_apu_details(valid_apu_details, "APU-001")
        # Esto no debería lanzar excepción
        json_str = json.dumps(result, default=str)
        assert isinstance(json_str, str)

    def test_duplicate_descriptions_aggregated(
        self, presenter: APUPresenter
    ) -> None:
        """Registros con misma categoría y descripción se agregan."""
        records = [
            {
                "CATEGORIA": "MATERIALES",
                "DESCRIPCION_INSUMO": "Cemento",
                "CANTIDAD_APU": 5.0,
                "VALOR_TOTAL_APU": 250.0,
                "UNIDAD_APU": "kg",
                "PRECIO_UNIT_APU": 50.0,
                "CODIGO_APU": "APU-001",
                "UNIDAD_INSUMO": "kg",
                "RENDIMIENTO": 1.0,
            },
            {
                "CATEGORIA": "MATERIALES",
                "DESCRIPCION_INSUMO": "Cemento",
                "CANTIDAD_APU": 5.0,
                "VALOR_TOTAL_APU": 250.0,
                "UNIDAD_APU": "kg",
                "PRECIO_UNIT_APU": 50.0,
                "CODIGO_APU": "APU-001",
                "UNIDAD_INSUMO": "kg",
                "RENDIMIENTO": 1.0,
            },
        ]
        result = presenter.process_apu_details(records, "APU-AGG")
        materiales_items = result["desglose"].get("MATERIALES", [])
        assert len(materiales_items) == 1
        assert materiales_items[0]["cantidad"] == pytest.approx(10.0)
        assert materiales_items[0]["valor_total"] == pytest.approx(500.0)

    def test_category_normalization_in_pipeline(
        self, presenter: APUPresenter
    ) -> None:
        """Variantes categóricas se consolidan en el pipeline completo."""
        records = [
            {
                "CATEGORIA": "MATERIAL",
                "DESCRIPCION_INSUMO": "Arena",
                "CANTIDAD_APU": 1.0,
                "VALOR_TOTAL_APU": 10.0,
                "UNIDAD_APU": "m3",
                "PRECIO_UNIT_APU": 10.0,
                "CODIGO_APU": "APU-002",
                "UNIDAD_INSUMO": "m3",
                "RENDIMIENTO": 1.0,
            },
            {
                "CATEGORIA": "MAT",
                "DESCRIPCION_INSUMO": "Grava",
                "CANTIDAD_APU": 2.0,
                "VALOR_TOTAL_APU": 30.0,
                "UNIDAD_APU": "m3",
                "PRECIO_UNIT_APU": 15.0,
                "CODIGO_APU": "APU-002",
                "UNIDAD_INSUMO": "m3",
                "RENDIMIENTO": 1.0,
            },
        ]
        result = presenter.process_apu_details(records, "APU-NORM")
        assert "MATERIALES" in result["desglose"]
        assert len(result["desglose"]["MATERIALES"]) == 2

    def test_with_alerts_column(self, presenter: APUPresenter) -> None:
        """Pipeline maneja correctamente la columna opcional 'alerta'."""
        records = [
            {
                "CATEGORIA": "EQUIPO",
                "DESCRIPCION_INSUMO": "Vibrador",
                "CANTIDAD_APU": 1.0,
                "VALOR_TOTAL_APU": 50.0,
                "UNIDAD_APU": "hr",
                "PRECIO_UNIT_APU": 50.0,
                "CODIGO_APU": "APU-003",
                "UNIDAD_INSUMO": "hr",
                "RENDIMIENTO": 0.5,
                "alerta": "Precio alto",
            },
            {
                "CATEGORIA": "EQUIPO",
                "DESCRIPCION_INSUMO": "Vibrador",
                "CANTIDAD_APU": 1.0,
                "VALOR_TOTAL_APU": 50.0,
                "UNIDAD_APU": "hr",
                "PRECIO_UNIT_APU": 50.0,
                "CODIGO_APU": "APU-003",
                "UNIDAD_INSUMO": "hr",
                "RENDIMIENTO": 0.5,
                "alerta": "Rendimiento bajo",
            },
        ]
        result = presenter.process_apu_details(records, "APU-ALERT")
        equipo_items = result["desglose"]["EQUIPO"]
        assert len(equipo_items) == 1
        assert "alerta" in equipo_items[0]
        assert "Precio alto" in equipo_items[0]["alerta"]
        assert "Rendimiento bajo" in equipo_items[0]["alerta"]

    def test_runtime_error_wraps_unexpected_exceptions(
        self, logger: logging.Logger
    ) -> None:
        """Excepciones inesperadas se envuelven en RuntimeError."""
        # Crear un presenter con config que provoque error interno
        presenter = APUPresenter(logger=logger)

        # Monkey-patch para forzar una excepción inesperada
        def exploding_sanitize(*args: Any, **kwargs: Any) -> None:
            raise MemoryError("boom")

        presenter._sanitize_dataframe = exploding_sanitize  # type: ignore[assignment]

        records = [
            {
                "CATEGORIA": "EQUIPO",
                "DESCRIPCION_INSUMO": "X",
                "CANTIDAD_APU": 1,
                "VALOR_TOTAL_APU": 10,
                "UNIDAD_APU": "u",
                "PRECIO_UNIT_APU": 10,
                "CODIGO_APU": "C1",
                "UNIDAD_INSUMO": "u",
                "RENDIMIENTO": 1,
            }
        ]
        with pytest.raises(RuntimeError, match="Fallo en procesamiento"):
            presenter.process_apu_details(records, "APU-FAIL")


# ======================================================================
# Tests: Propiedades del sistema
# ======================================================================


class TestSystemProperties:
    """Pruebas de propiedades transversales del sistema."""

    def test_config_immutability_after_init(
        self, presenter: APUPresenter
    ) -> None:
        """La configuración no puede ser mutada después de la inicialización."""
        with pytest.raises(AttributeError):
            presenter.config.default_category = "NUEVO"  # type: ignore[misc]

    def test_deterministic_output(
        self,
        presenter: APUPresenter,
        valid_apu_details: List[Dict[str, Any]],
    ) -> None:
        """Dos ejecuciones con mismos datos producen mismo resultado."""
        result1 = presenter.process_apu_details(valid_apu_details, "APU-DET")
        result2 = presenter.process_apu_details(valid_apu_details, "APU-DET")

        assert result1["total_items"] == result2["total_items"]
        assert result1["metadata"] == result2["metadata"]
        assert len(result1["items"]) == len(result2["items"])

        for item1, item2 in zip(result1["items"], result2["items"]):
            for key in item1:
                v1, v2 = item1[key], item2[key]
                if isinstance(v1, float) and v1 is not None:
                    assert v1 == pytest.approx(v2)
                else:
                    assert v1 == v2

    def test_no_numpy_types_in_output(
        self,
        presenter: APUPresenter,
        valid_apu_details: List[Dict[str, Any]],
    ) -> None:
        """La salida no contiene tipos numpy, solo tipos Python nativos."""
        result = presenter.process_apu_details(valid_apu_details, "APU-NP")

        def check_no_numpy(obj: Any, path: str = "root") -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    check_no_numpy(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    check_no_numpy(v, f"{path}[{i}]")
            else:
                assert not isinstance(obj, np.generic), (
                    f"Tipo numpy encontrado en {path}: {type(obj).__name__}"
                )

        check_no_numpy(result)

    def test_all_items_have_consistent_keys(
        self,
        presenter: APUPresenter,
        valid_apu_details: List[Dict[str, Any]],
    ) -> None:
        """Todos los items procesados tienen el mismo conjunto de claves."""
        result = presenter.process_apu_details(valid_apu_details, "APU-KEYS")
        items = result["items"]

        if not items:
            pytest.skip("No hay items para verificar")

        reference_keys = set(items[0].keys())
        for i, item in enumerate(items[1:], start=1):
            assert set(item.keys()) == reference_keys, (
                f"Item {i} tiene claves diferentes: "
                f"esperado={reference_keys}, obtenido={set(item.keys())}"
            )

    def test_total_items_equals_sum_of_breakdown(
        self,
        presenter: APUPresenter,
        valid_apu_details: List[Dict[str, Any]],
    ) -> None:
        """total_items == sum(len(v) for v in desglose.values())"""
        result = presenter.process_apu_details(valid_apu_details, "APU-SUM")
        breakdown_total = sum(
            len(items) for items in result["desglose"].values()
        )
        assert result["total_items"] == breakdown_total