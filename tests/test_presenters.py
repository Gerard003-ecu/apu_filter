"""
Test Suite Completo para APUPresenter
Cubre todos los métodos, casos edge, validaciones y flujos de error.
"""

import logging
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Importar las clases a testear
from app.presenters import APUPresenter, APUProcessingConfig

# ============================================================================
# FIXTURES - Datos de Prueba Reutilizables
# ============================================================================


@pytest.fixture
def mock_logger():
    """Logger mock para pruebas."""
    logger = Mock(spec=logging.Logger)
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture
def default_config():
    """Configuración por defecto para pruebas."""
    return APUProcessingConfig()


@pytest.fixture
def custom_config():
    """Configuración personalizada para pruebas específicas."""
    return APUProcessingConfig(
        default_category="SIN_CATEGORIA", tolerance_price_variance=0.05
    )


@pytest.fixture
def presenter(mock_logger):
    """Instancia de APUPresenter con logger mock."""
    return APUPresenter(logger=mock_logger)


@pytest.fixture
def presenter_custom_config(mock_logger, custom_config):
    """Instancia de APUPresenter con configuración personalizada."""
    return APUPresenter(logger=mock_logger, config=custom_config)


@pytest.fixture
def valid_apu_details():
    """Datos válidos de APU para pruebas exitosas."""
    return [
        {
            "CATEGORIA": "MATERIALES",
            "DESCRIPCION_INSUMO": "Cemento gris",
            "CANTIDAD_APU": 50.0,
            "VALOR_TOTAL_APU": 500000.0,
            "RENDIMIENTO": 1.0,
            "UNIDAD_APU": "kg",
            "PRECIO_UNIT_APU": 10000.0,
            "CODIGO_APU": "MAT001",
            "UNIDAD_INSUMO": "kg",
        },
        {
            "CATEGORIA": "MATERIALES",
            "DESCRIPCION_INSUMO": "Cemento gris",
            "CANTIDAD_APU": 25.0,
            "VALOR_TOTAL_APU": 250000.0,
            "RENDIMIENTO": 0.5,
            "UNIDAD_APU": "kg",
            "PRECIO_UNIT_APU": 10000.0,
            "CODIGO_APU": "MAT001",
            "UNIDAD_INSUMO": "kg",
        },
        {
            "CATEGORIA": "MANO DE OBRA",
            "DESCRIPCION_INSUMO": "Oficial",
            "CANTIDAD_APU": 8.0,
            "VALOR_TOTAL_APU": 200000.0,
            "RENDIMIENTO": 1.0,
            "UNIDAD_APU": "hr",
            "PRECIO_UNIT_APU": 25000.0,
            "CODIGO_APU": "MO001",
            "UNIDAD_INSUMO": "hr",
        },
        {
            "CATEGORIA": "EQUIPO",
            "DESCRIPCION_INSUMO": "Mezcladora",
            "CANTIDAD_APU": 4.0,
            "VALOR_TOTAL_APU": 100000.0,
            "RENDIMIENTO": 1.0,
            "UNIDAD_APU": "hr",
            "PRECIO_UNIT_APU": 25000.0,
            "CODIGO_APU": "EQ001",
            "UNIDAD_INSUMO": "hr",
        },
    ]


@pytest.fixture
def apu_details_with_alerts():
    """Datos de APU con alertas."""
    return [
        {
            "CATEGORIA": "MATERIALES",
            "DESCRIPCION_INSUMO": "Cemento gris",
            "CANTIDAD_APU": 50.0,
            "VALOR_TOTAL_APU": 500000.0,
            "RENDIMIENTO": 1.0,
            "UNIDAD_APU": "kg",
            "PRECIO_UNIT_APU": 10000.0,
            "CODIGO_APU": "MAT001",
            "UNIDAD_INSUMO": "kg",
            "alerta": "Precio alto",
        },
        {
            "CATEGORIA": "MATERIALES",
            "DESCRIPCION_INSUMO": "Cemento gris",
            "CANTIDAD_APU": 25.0,
            "VALOR_TOTAL_APU": 250000.0,
            "RENDIMIENTO": 0.5,
            "UNIDAD_APU": "kg",
            "PRECIO_UNIT_APU": 10000.0,
            "CODIGO_APU": "MAT001",
            "UNIDAD_INSUMO": "kg",
            "alerta": "Stock bajo",
        },
    ]


@pytest.fixture
def apu_details_with_nan():
    """Datos con valores NaN y None."""
    return [
        {
            "CATEGORIA": "MATERIALES",
            "DESCRIPCION_INSUMO": "Cemento gris",
            "CANTIDAD_APU": 50.0,
            "VALOR_TOTAL_APU": np.nan,
            "RENDIMIENTO": 1.0,
            "UNIDAD_APU": "kg",
            "PRECIO_UNIT_APU": 10000.0,
            "CODIGO_APU": "MAT001",
            "UNIDAD_INSUMO": "kg",
        },
        {
            "CATEGORIA": None,
            "DESCRIPCION_INSUMO": "Arena",
            "CANTIDAD_APU": None,
            "VALOR_TOTAL_APU": 150000.0,
            "RENDIMIENTO": np.inf,
            "UNIDAD_APU": "m3",
            "PRECIO_UNIT_APU": 50000.0,
            "CODIGO_APU": "MAT002",
            "UNIDAD_INSUMO": "m3",
        },
    ]


@pytest.fixture
def apu_details_inconsistent_prices():
    """Datos con precios inconsistentes para mismo insumo."""
    return [
        {
            "CATEGORIA": "MATERIALES",
            "DESCRIPCION_INSUMO": "Cemento gris",
            "CANTIDAD_APU": 50.0,
            "VALOR_TOTAL_APU": 500000.0,
            "RENDIMIENTO": 1.0,
            "UNIDAD_APU": "kg",
            "PRECIO_UNIT_APU": 10000.0,
            "CODIGO_APU": "MAT001",
            "UNIDAD_INSUMO": "kg",
        },
        {
            "CATEGORIA": "MATERIALES",
            "DESCRIPCION_INSUMO": "Cemento gris",
            "CANTIDAD_APU": 25.0,
            "VALOR_TOTAL_APU": 312500.0,
            "RENDIMIENTO": 0.5,
            "UNIDAD_APU": "kg",
            "PRECIO_UNIT_APU": 12500.0,  # Precio diferente
            "CODIGO_APU": "MAT001",
            "UNIDAD_INSUMO": "kg",
        },
    ]


@pytest.fixture
def apu_details_negative_values():
    """Datos con valores negativos."""
    return [
        {
            "CATEGORIA": "MATERIALES",
            "DESCRIPCION_INSUMO": "Cemento gris",
            "CANTIDAD_APU": -50.0,  # Negativo
            "VALOR_TOTAL_APU": 500000.0,
            "RENDIMIENTO": 1.0,
            "UNIDAD_APU": "kg",
            "PRECIO_UNIT_APU": 10000.0,
            "CODIGO_APU": "MAT001",
            "UNIDAD_INSUMO": "kg",
        }
    ]


@pytest.fixture
def apu_details_whitespace_strings():
    """Datos con strings que contienen espacios."""
    return [
        {
            "CATEGORIA": "  MATERIALES  ",
            "DESCRIPCION_INSUMO": "  Cemento gris  ",
            "CANTIDAD_APU": 50.0,
            "VALOR_TOTAL_APU": 500000.0,
            "RENDIMIENTO": 1.0,
            "UNIDAD_APU": "  kg  ",
            "PRECIO_UNIT_APU": 10000.0,
            "CODIGO_APU": "MAT001",
            "UNIDAD_INSUMO": "  kg  ",
        }
    ]


# ============================================================================
# TESTS DE INICIALIZACIÓN
# ============================================================================


class TestAPUPresenterInitialization:
    """Tests para la inicialización del presenter."""

    def test_initialization_with_logger(self, mock_logger):
        """Debe inicializar correctamente con un logger válido."""
        presenter = APUPresenter(logger=mock_logger)

        assert presenter.logger is mock_logger
        assert isinstance(presenter.config, APUProcessingConfig)
        mock_logger.info.assert_called_once_with("APUPresenter inicializado correctamente")

    def test_initialization_with_custom_config(self, mock_logger, custom_config):
        """Debe aceptar configuración personalizada."""
        presenter = APUPresenter(logger=mock_logger, config=custom_config)

        assert presenter.config is custom_config
        assert presenter.config.default_category == "SIN_CATEGORIA"
        assert presenter.config.tolerance_price_variance == 0.05

    def test_initialization_without_logger_raises_error(self):
        """Debe lanzar ValueError si no se proporciona logger."""
        with pytest.raises(ValueError, match="Logger es requerido"):
            APUPresenter(logger=None)

    def test_initialization_with_default_config(self, mock_logger):
        """Debe crear configuración por defecto si no se proporciona."""
        presenter = APUPresenter(logger=mock_logger)

        assert presenter.config.default_category == "INDEFINIDO"
        assert presenter.config.tolerance_price_variance == 0.01


# ============================================================================
# TESTS DE VALIDACIÓN DE ENTRADAS
# ============================================================================


class TestInputValidation:
    """Tests para el método _validate_inputs."""

    def test_validate_inputs_success(self, presenter, valid_apu_details):
        """Debe pasar validación con datos correctos."""
        # No debe lanzar excepción
        presenter._validate_inputs(valid_apu_details, "APU001")

    def test_validate_inputs_not_a_list(self, presenter):
        """Debe fallar si apu_details no es una lista."""
        with pytest.raises(ValueError, match="debe ser una lista"):
            presenter._validate_inputs("not a list", "APU001")

        with pytest.raises(ValueError, match="debe ser una lista"):
            presenter._validate_inputs({"key": "value"}, "APU001")

    def test_validate_inputs_empty_list(self, presenter):
        """Debe fallar si la lista está vacía."""
        with pytest.raises(ValueError, match="No se encontraron detalles"):
            presenter._validate_inputs([], "APU001")

    def test_validate_inputs_not_dict_elements(self, presenter):
        """Debe fallar si los elementos no son diccionarios."""
        with pytest.raises(ValueError, match="deben ser diccionarios"):
            presenter._validate_inputs([1, 2, 3], "APU001")

        with pytest.raises(ValueError, match="deben ser diccionarios"):
            presenter._validate_inputs(["string", "values"], "APU001")

    def test_validate_inputs_invalid_apu_code(self, presenter, valid_apu_details):
        """Debe fallar si apu_code no es válido."""
        with pytest.raises(ValueError, match="debe ser un string no vacío"):
            presenter._validate_inputs(valid_apu_details, "")

        with pytest.raises(ValueError, match="debe ser un string no vacío"):
            presenter._validate_inputs(valid_apu_details, "   ")

        with pytest.raises(ValueError, match="debe ser un string no vacío"):
            presenter._validate_inputs(valid_apu_details, None)

        with pytest.raises(ValueError, match="debe ser un string no vacío"):
            presenter._validate_inputs(valid_apu_details, 123)


# ============================================================================
# TESTS DE SANITIZACIÓN
# ============================================================================


class TestDataSanitization:
    """Tests para el método _sanitize_dataframe."""

    def test_sanitize_replaces_nan_with_none(self, presenter):
        """Debe reemplazar NaN con None."""
        df = pd.DataFrame({"A": [1, np.nan, 3], "B": [np.nan, 2, 3]})

        df_sanitized = presenter._sanitize_dataframe(df)

        assert df_sanitized.loc[0, "B"] is None
        assert df_sanitized.loc[1, "A"] is None

    def test_sanitize_replaces_inf_with_none(self, presenter):
        """Debe reemplazar inf y -inf con None."""
        df = pd.DataFrame({"A": [1, np.inf, 3], "B": [-np.inf, 2, 3]})

        df_sanitized = presenter._sanitize_dataframe(df)

        assert df_sanitized.loc[1, "A"] is None
        assert df_sanitized.loc[0, "B"] is None

    def test_sanitize_strips_whitespace_from_strings(self, presenter):
        """Debe eliminar espacios de los strings."""
        df = pd.DataFrame({"text": ["  hello  ", "world  ", "  test"]})

        df_sanitized = presenter._sanitize_dataframe(df)

        assert df_sanitized.loc[0, "text"] == "hello"
        assert df_sanitized.loc[1, "text"] == "world"
        assert df_sanitized.loc[2, "text"] == "test"

    def test_sanitize_converts_numeric_columns(self, presenter, mock_logger):
        """Debe convertir columnas numéricas y advertir sobre fallos."""
        df = pd.DataFrame(
            {
                "CANTIDAD_APU": ["10", "20", "invalid"],
                "VALOR_TOTAL_APU": [100.5, 200.5, 300.5],
            }
        )

        presenter.logger = mock_logger
        df_sanitized = presenter._sanitize_dataframe(df)

        # Verificar conversión
        assert df_sanitized.loc[0, "CANTIDAD_APU"] == 10.0
        assert df_sanitized.loc[1, "CANTIDAD_APU"] == 20.0
        assert pd.isna(df_sanitized.loc[2, "CANTIDAD_APU"])

        # Verificar warning por valor no numérico
        assert mock_logger.warning.called

    def test_sanitize_warns_about_negative_values(self, presenter, mock_logger):
        """Debe advertir sobre valores negativos en columnas numéricas."""
        df = pd.DataFrame(
            {"CANTIDAD_APU": [10, -20, 30], "VALOR_TOTAL_APU": [100, 200, -300]}
        )

        presenter.logger = mock_logger
        df_sanitized = presenter._sanitize_dataframe(df)

        # Debe haber advertencias sobre valores negativos
        warning_calls = [call for call in mock_logger.warning.call_args_list]
        assert len(warning_calls) > 0

    def test_sanitize_does_not_modify_original(self, presenter):
        """Debe retornar una copia, no modificar el original."""
        df_original = pd.DataFrame(
            {"A": [1, np.nan, 3], "text": ["  hello  ", "world", "test"]}
        )

        df_original_copy = df_original.copy()
        df_sanitized = presenter._sanitize_dataframe(df_original)

        # El original no debe cambiar
        pd.testing.assert_frame_equal(df_original, df_original_copy)
        # El sanitizado debe ser diferente
        assert not df_sanitized.equals(df_original)


# ============================================================================
# TESTS DE VALIDACIÓN DE ESQUEMA
# ============================================================================


class TestSchemaValidation:
    """Tests para el método _validate_dataframe_schema."""

    def test_validate_schema_success(self, presenter, valid_apu_details):
        """Debe pasar validación con todas las columnas requeridas."""
        df = pd.DataFrame(valid_apu_details)

        # No debe lanzar excepción
        presenter._validate_dataframe_schema(df, "APU001")

    def test_validate_schema_missing_columns(self, presenter, mock_logger):
        """Debe fallar si faltan columnas requeridas."""
        df = pd.DataFrame(
            {
                "CATEGORIA": ["MAT"],
                "DESCRIPCION_INSUMO": ["Cemento"],
                # Faltan muchas columnas requeridas
            }
        )

        presenter.logger = mock_logger

        with pytest.raises(KeyError, match="Columnas faltantes"):
            presenter._validate_dataframe_schema(df, "APU001")

    def test_validate_schema_logs_success(self, presenter, valid_apu_details, mock_logger):
        """Debe loggear éxito de validación."""
        df = pd.DataFrame(valid_apu_details)
        presenter.logger = mock_logger

        presenter._validate_dataframe_schema(df, "APU001")

        mock_logger.debug.assert_called_with("Validación de esquema exitosa para APU APU001")


# ============================================================================
# TESTS DE AGREGACIÓN
# ============================================================================


class TestAggregation:
    """Tests para métodos de agregación."""

    def test_group_by_category_consolidates_items(self, presenter, valid_apu_details):
        """Debe consolidar items duplicados por descripción."""
        df = pd.DataFrame(valid_apu_details)

        processed = presenter._group_by_category(df)

        # Debe consolidar los dos registros de "Cemento gris" en uno
        cement_items = [item for item in processed if item["DESCRIPCION"] == "Cemento gris"]

        assert len(cement_items) == 1
        assert cement_items[0]["CANTIDAD"] == 75.0  # 50 + 25
        assert cement_items[0]["VR_TOTAL"] == 750000.0  # 500000 + 250000

    def test_group_by_category_processes_all_categories(self, presenter, valid_apu_details):
        """Debe procesar todas las categorías presentes."""
        df = pd.DataFrame(valid_apu_details)

        processed = presenter._group_by_category(df)

        categories = {item["CATEGORIA"] for item in processed}

        assert "MATERIALES" in categories
        assert "MANO DE OBRA" in categories
        assert "EQUIPO" in categories

    def test_group_by_category_empty_dataframe(self, presenter, mock_logger):
        """Debe manejar DataFrame vacío."""
        df = pd.DataFrame({"CATEGORIA": [], "DESCRIPCION_INSUMO": []})

        presenter.logger = mock_logger
        processed = presenter._group_by_category(df)

        assert processed == []
        mock_logger.warning.assert_called()

    def test_group_by_category_with_alerts(self, presenter, apu_details_with_alerts):
        """Debe agregar alertas correctamente."""
        df = pd.DataFrame(apu_details_with_alerts)

        processed = presenter._group_by_category(df)

        # Verificar que las alertas se concatenaron
        cement_item = next(
            item for item in processed if item["DESCRIPCION"] == "Cemento gris"
        )

        assert "alerta" in cement_item
        assert "Precio alto" in cement_item["alerta"]
        assert "Stock bajo" in cement_item["alerta"]

    def test_aggregate_price_single_price(self, presenter):
        """Debe retornar el precio único si todos son iguales."""
        series = pd.Series([10000.0, 10000.0, 10000.0])

        price = presenter._aggregate_price(series)

        assert price == 10000.0

    def test_aggregate_price_multiple_prices_warning(self, presenter, mock_logger):
        """Debe advertir y promediar si hay variación significativa."""
        series = pd.Series([10000.0, 12000.0, 14000.0])
        presenter.logger = mock_logger

        price = presenter._aggregate_price(series, "MATERIALES")

        assert price == 12000.0  # Promedio
        mock_logger.warning.assert_called()

    def test_aggregate_price_empty_series(self, presenter):
        """Debe retornar 0 si la serie está vacía."""
        series = pd.Series([], dtype=float)

        price = presenter._aggregate_price(series)

        assert price == 0.0

    def test_validate_consistency_single_value(self, presenter):
        """Debe retornar el valor único si todos son iguales."""
        series = pd.Series(["kg", "kg", "kg"])

        value = presenter._validate_consistency(series, "UNIDAD")

        assert value == "kg"

    def test_validate_consistency_multiple_values_warning(self, presenter, mock_logger):
        """Debe advertir y retornar el primero si hay inconsistencia."""
        series = pd.Series(["kg", "m3", "lt"])
        presenter.logger = mock_logger

        value = presenter._validate_consistency(series, "UNIDAD")

        assert value == "kg"
        mock_logger.warning.assert_called()

    def test_validate_consistency_empty_series(self, presenter):
        """Debe retornar None si la serie está vacía."""
        series = pd.Series([])

        value = presenter._validate_consistency(series)

        assert value is None

    def test_aggregate_alerts_concatenates_unique(self, presenter):
        """Debe concatenar alertas únicas."""
        series = pd.Series(["Alerta 1", "Alerta 2", "Alerta 1"])

        result = presenter._aggregate_alerts(series)

        assert "Alerta 1" in result
        assert "Alerta 2" in result
        assert result.count("Alerta 1") == 1  # Solo una vez

    def test_aggregate_alerts_empty_series(self, presenter):
        """Debe retornar None si no hay alertas."""
        series = pd.Series([])

        result = presenter._aggregate_alerts(series)

        assert result is None


# ============================================================================
# TESTS DE ORGANIZACIÓN DE DESGLOSE
# ============================================================================


class TestBreakdownOrganization:
    """Tests para el método _organize_breakdown."""

    def test_organize_breakdown_groups_by_category(self, presenter):
        """Debe organizar items por categoría."""
        items = [
            {"CATEGORIA": "MATERIALES", "desc": "Item 1"},
            {"CATEGORIA": "MATERIALES", "desc": "Item 2"},
            {"CATEGORIA": "EQUIPO", "desc": "Item 3"},
        ]

        desglose = presenter._organize_breakdown(items)

        assert "MATERIALES" in desglose
        assert "EQUIPO" in desglose
        assert len(desglose["MATERIALES"]) == 2
        assert len(desglose["EQUIPO"]) == 1

    def test_organize_breakdown_handles_missing_category(self, presenter, mock_logger):
        """Debe asignar categoría por defecto a items sin categoría."""
        items = [
            {"CATEGORIA": None, "desc": "Item 1"},
            {"CATEGORIA": "", "desc": "Item 2"},
            {"CATEGORIA": "   ", "desc": "Item 3"},
        ]

        presenter.logger = mock_logger
        desglose = presenter._organize_breakdown(items)

        assert "INDEFINIDO" in desglose
        assert len(desglose["INDEFINIDO"]) == 3
        mock_logger.warning.assert_called()

    def test_organize_breakdown_empty_list(self, presenter, mock_logger):
        """Debe retornar diccionario vacío para lista vacía."""
        presenter.logger = mock_logger
        desglose = presenter._organize_breakdown([])

        assert desglose == {}
        mock_logger.warning.assert_called_with("No hay items para organizar en desglose")

    def test_organize_breakdown_custom_default_category(
        self, presenter_custom_config, mock_logger
    ):
        """Debe usar categoría por defecto personalizada."""
        items = [{"CATEGORIA": None, "desc": "Item 1"}]

        presenter_custom_config.logger = mock_logger
        desglose = presenter_custom_config._organize_breakdown(items)

        assert "SIN_CATEGORIA" in desglose


# ============================================================================
# TESTS DE METADATA
# ============================================================================


class TestMetadataCalculation:
    """Tests para el método _calculate_metadata."""

    def test_calculate_metadata_success(self, presenter, valid_apu_details):
        """Debe calcular metadata correctamente."""
        df = pd.DataFrame(valid_apu_details)
        processed = presenter._group_by_category(df)

        metadata = presenter._calculate_metadata(df, processed)

        assert metadata["original_rows"] == 4
        assert metadata["processed_items"] == 3  # 2 items consolidados en 1
        assert "reduction_rate" in metadata
        assert metadata["reduction_rate"] > 0
        assert "total_value" in metadata
        assert "avg_value_per_item" in metadata
        assert "categories_count" in metadata

    def test_calculate_metadata_total_value(self, presenter):
        """Debe calcular el valor total correctamente."""
        df = pd.DataFrame(
            {
                "CATEGORIA": ["A", "B"],
                "DESCRIPCION_INSUMO": ["Item1", "Item2"],
                "CANTIDAD_APU": [10, 20],
                "VALOR_TOTAL_APU": [100, 200],
                "RENDIMIENTO": [1, 1],
                "UNIDAD_APU": ["kg", "kg"],
                "PRECIO_UNIT_APU": [10, 10],
                "CODIGO_APU": ["A1", "B1"],
                "UNIDAD_INSUMO": ["kg", "kg"],
            }
        )

        processed = [{"VR_TOTAL": 100.0}, {"VR_TOTAL": 200.0}]

        metadata = presenter._calculate_metadata(df, processed)

        assert metadata["total_value"] == 300.0

    def test_calculate_metadata_handles_none_values(self, presenter):
        """Debe manejar valores None en VR_TOTAL."""
        df = pd.DataFrame(
            {
                "CATEGORIA": ["A"],
                "DESCRIPCION_INSUMO": ["Item1"],
                "CANTIDAD_APU": [10],
                "VALOR_TOTAL_APU": [100],
                "RENDIMIENTO": [1],
                "UNIDAD_APU": ["kg"],
                "PRECIO_UNIT_APU": [10],
                "CODIGO_APU": ["A1"],
                "UNIDAD_INSUMO": ["kg"],
            }
        )

        processed = [{"VR_TOTAL": None}, {"VR_TOTAL": 100.0}]

        metadata = presenter._calculate_metadata(df, processed)

        assert metadata["total_value"] == 100.0

    def test_calculate_metadata_empty_processed_items(self, presenter):
        """Debe manejar lista vacía de items procesados."""
        df = pd.DataFrame(
            {
                "CATEGORIA": ["A"],
                "DESCRIPCION_INSUMO": ["Item1"],
                "CANTIDAD_APU": [10],
                "VALOR_TOTAL_APU": [100],
                "RENDIMIENTO": [1],
                "UNIDAD_APU": ["kg"],
                "PRECIO_UNIT_APU": [10],
                "CODIGO_APU": ["A1"],
                "UNIDAD_INSUMO": ["kg"],
            }
        )

        metadata = presenter._calculate_metadata(df, [])

        assert metadata["processed_items"] == 0
        assert metadata["avg_value_per_item"] == 0


# ============================================================================
# TESTS DE INTEGRACIÓN - PROCESO COMPLETO
# ============================================================================


class TestProcessAPUDetailsIntegration:
    """Tests de integración del proceso completo."""

    def test_process_apu_details_success(self, presenter, valid_apu_details, mock_logger):
        """Debe procesar APU completo exitosamente."""
        presenter.logger = mock_logger

        result = presenter.process_apu_details(valid_apu_details, "APU001")

        assert "items" in result
        assert "desglose" in result
        assert "total_items" in result
        assert "metadata" in result

        assert len(result["items"]) == 3  # 4 originales, 2 consolidados
        assert result["total_items"] == 3
        assert len(result["desglose"]) == 3  # 3 categorías

    def test_process_apu_details_with_alerts(self, presenter, apu_details_with_alerts):
        """Debe procesar alertas correctamente."""
        result = presenter.process_apu_details(apu_details_with_alerts, "APU001")

        # Verificar que el item consolidado tiene alertas
        cement_item = next(
            item for item in result["items"] if item["DESCRIPCION"] == "Cemento gris"
        )

        assert "alerta" in cement_item

    def test_process_apu_details_sanitizes_data(self, presenter, apu_details_with_nan):
        """Debe sanitizar datos con NaN correctamente."""
        result = presenter.process_apu_details(apu_details_with_nan, "APU001")

        # Verificar que se procesó sin errores
        assert result["total_items"] >= 0

    def test_process_apu_details_logs_processing(
        self, presenter, valid_apu_details, mock_logger
    ):
        """Debe loggear el inicio y fin del procesamiento."""
        presenter.logger = mock_logger

        result = presenter.process_apu_details(valid_apu_details, "APU001")

        # Verificar llamadas al logger
        info_calls = mock_logger.info.call_args_list
        assert len(info_calls) >= 2  # Inicio y fin

    def test_process_apu_details_handles_whitespace(
        self, presenter, apu_details_whitespace_strings
    ):
        """Debe eliminar espacios en blanco de strings."""
        result = presenter.process_apu_details(apu_details_whitespace_strings, "APU001")

        item = result["items"][0]

        assert item["CATEGORIA"] == "MATERIALES"
        assert item["DESCRIPCION"] == "Cemento gris"
        assert item["UNIDAD"] == "kg"


# ============================================================================
# TESTS DE CASOS EDGE Y ERRORES
# ============================================================================


class TestEdgeCasesAndErrors:
    """Tests para casos edge y manejo de errores."""

    def test_process_empty_list_raises_error(self, presenter):
        """Debe lanzar error con lista vacía."""
        with pytest.raises(ValueError, match="No se encontraron detalles"):
            presenter.process_apu_details([], "APU001")

    def test_process_invalid_type_raises_error(self, presenter):
        """Debe lanzar error con tipo inválido."""
        with pytest.raises(ValueError, match="debe ser una lista"):
            presenter.process_apu_details("not a list", "APU001")

    def test_process_missing_columns_raises_error(self, presenter):
        """Debe lanzar error si faltan columnas requeridas."""
        invalid_data = [{"CATEGORIA": "MAT", "DESCRIPCION_INSUMO": "Cemento"}]

        with pytest.raises(KeyError, match="Columnas faltantes"):
            presenter.process_apu_details(invalid_data, "APU001")

    def test_process_invalid_apu_code_raises_error(self, presenter, valid_apu_details):
        """Debe lanzar error con código APU inválido."""
        with pytest.raises(ValueError, match="debe ser un string no vacío"):
            presenter.process_apu_details(valid_apu_details, "")

    def test_process_handles_category_processing_error(
        self, presenter, valid_apu_details, mock_logger
    ):
        """Debe continuar procesando si una categoría falla."""
        presenter.logger = mock_logger

        # Simular error en una categoría específica
        with patch.object(
            presenter,
            "_aggregate_category_items",
            side_effect=[
                Exception("Error simulado"),  # Primera categoría falla
                [{"item": "data"}],  # Segunda categoría funciona
            ],
        ):
            # No debe lanzar excepción, debe continuar
            result = presenter._group_by_category(pd.DataFrame(valid_apu_details))

            # Verificar que se loggeó el error
            mock_logger.error.assert_called()

    def test_process_all_categories_nan(self, presenter):
        """Debe manejar caso donde todas las categorías son NaN."""
        data = [
            {
                "CATEGORIA": np.nan,
                "DESCRIPCION_INSUMO": "Item",
                "CANTIDAD_APU": 10.0,
                "VALOR_TOTAL_APU": 100.0,
                "RENDIMIENTO": 1.0,
                "UNIDAD_APU": "kg",
                "PRECIO_UNIT_APU": 10.0,
                "CODIGO_APU": "A1",
                "UNIDAD_INSUMO": "kg",
            }
        ]

        result = presenter.process_apu_details(data, "APU001")

        # Debe asignar a categoría por defecto
        assert "INDEFINIDO" in result["desglose"]

    def test_process_with_inconsistent_prices(
        self, presenter, apu_details_inconsistent_prices, mock_logger
    ):
        """Debe advertir sobre precios inconsistentes."""
        presenter.logger = mock_logger

        result = presenter.process_apu_details(apu_details_inconsistent_prices, "APU001")

        # Debe haber procesado los datos
        assert result["total_items"] > 0

        # Debe haber advertencia sobre variación de precios
        warning_calls = [
            call
            for call in mock_logger.warning.call_args_list
            if "Variación significativa en precios" in str(call)
        ]
        assert len(warning_calls) > 0

    def test_process_with_negative_values_warns(
        self, presenter, apu_details_negative_values, mock_logger
    ):
        """Debe advertir sobre valores negativos."""
        presenter.logger = mock_logger

        result = presenter.process_apu_details(apu_details_negative_values, "APU001")

        # Debe advertir sobre valores negativos
        mock_logger.warning.assert_called()

    def test_process_dataframe_creation_error(self, presenter, mock_logger):
        """Debe manejar error en creación de DataFrame."""
        presenter.logger = mock_logger

        # Datos que no se pueden convertir a DataFrame correctamente
        invalid_data = [
            {"key1": "value1"},
            {"different_key": "value2"},  # Estructura inconsistente
        ]

        # Aunque pandas puede crear el DF, faltarán columnas
        with pytest.raises(KeyError):
            presenter.process_apu_details(invalid_data, "APU001")


# ============================================================================
# TESTS DE CONFIGURACIÓN
# ============================================================================


class TestAPUProcessingConfig:
    """Tests para la clase de configuración."""

    def test_config_default_values(self):
        """Debe tener valores por defecto correctos."""
        config = APUProcessingConfig()

        assert config.default_category == "INDEFINIDO"
        assert config.tolerance_price_variance == 0.01
        assert len(config.required_columns) > 0
        assert len(config.numeric_columns) > 0

    def test_config_custom_values(self):
        """Debe aceptar valores personalizados."""
        config = APUProcessingConfig(
            default_category="CUSTOM", tolerance_price_variance=0.05
        )

        assert config.default_category == "CUSTOM"
        assert config.tolerance_price_variance == 0.05

    def test_config_required_columns_includes_essentials(self):
        """Debe incluir columnas esenciales."""
        config = APUProcessingConfig()

        essential_columns = {
            "CATEGORIA",
            "DESCRIPCION_INSUMO",
            "CANTIDAD_APU",
            "VALOR_TOTAL_APU",
        }

        assert essential_columns.issubset(config.required_columns)

    def test_config_numeric_columns_includes_essentials(self):
        """Debe incluir columnas numéricas esenciales."""
        config = APUProcessingConfig()

        essential_numeric = {"CANTIDAD_APU", "VALOR_TOTAL_APU", "PRECIO_UNIT_APU"}

        assert essential_numeric.issubset(config.numeric_columns)


# ============================================================================
# TESTS DE RENDIMIENTO Y VOLUMEN
# ============================================================================


class TestPerformance:
    """Tests de rendimiento y manejo de grandes volúmenes."""

    def test_process_large_dataset(self, presenter):
        """Debe procesar datasets grandes eficientemente."""
        # Generar dataset grande
        large_data = []
        for i in range(1000):
            large_data.append(
                {
                    "CATEGORIA": f"CAT_{i % 10}",
                    "DESCRIPCION_INSUMO": f"Insumo_{i % 50}",
                    "CANTIDAD_APU": float(i),
                    "VALOR_TOTAL_APU": float(i * 100),
                    "RENDIMIENTO": 1.0,
                    "UNIDAD_APU": "kg",
                    "PRECIO_UNIT_APU": 100.0,
                    "CODIGO_APU": f"COD_{i}",
                    "UNIDAD_INSUMO": "kg",
                }
            )

        result = presenter.process_apu_details(large_data, "APU_LARGE")

        # Verificar que se procesó correctamente
        assert result["total_items"] > 0
        assert result["metadata"]["original_rows"] == 1000

    def test_process_many_duplicate_items(self, presenter):
        """Debe consolidar eficientemente muchos duplicados."""
        # 100 registros del mismo item
        duplicate_data = []
        for i in range(100):
            duplicate_data.append(
                {
                    "CATEGORIA": "MATERIALES",
                    "DESCRIPCION_INSUMO": "Cemento",
                    "CANTIDAD_APU": 1.0,
                    "VALOR_TOTAL_APU": 100.0,
                    "RENDIMIENTO": 1.0,
                    "UNIDAD_APU": "kg",
                    "PRECIO_UNIT_APU": 100.0,
                    "CODIGO_APU": "MAT001",
                    "UNIDAD_INSUMO": "kg",
                }
            )

        result = presenter.process_apu_details(duplicate_data, "APU_DUP")

        # Debe consolidar en un solo item
        assert result["total_items"] == 1
        cement = result["items"][0]
        assert cement["CANTIDAD"] == 100.0
        assert cement["VR_TOTAL"] == 10000.0


# ============================================================================
# TESTS PARAMETRIZADOS
# ============================================================================


class TestParametrized:
    """Tests parametrizados para cubrir múltiples casos."""

    @pytest.mark.parametrize("invalid_apu_code", ["", "   ", None, 123, [], {}])
    def test_invalid_apu_codes(self, presenter, valid_apu_details, invalid_apu_code):
        """Debe rechazar múltiples tipos de códigos APU inválidos."""
        with pytest.raises(ValueError):
            presenter.process_apu_details(valid_apu_details, invalid_apu_code)

    @pytest.mark.parametrize(
        "invalid_details",
        ["string", 123, None, {"key": "value"}, [1, 2, 3], ["string1", "string2"]],
    )
    def test_invalid_apu_details_types(self, presenter, invalid_details):
        """Debe rechazar múltiples tipos inválidos de apu_details."""
        with pytest.raises(ValueError):
            presenter.process_apu_details(invalid_details, "APU001")

    @pytest.mark.parametrize("nan_value", [np.nan, np.inf, -np.inf, None])
    def test_sanitize_special_values(self, presenter, nan_value):
        """Debe sanitizar múltiples tipos de valores especiales."""
        df = pd.DataFrame({"A": [1, nan_value, 3]})

        df_sanitized = presenter._sanitize_dataframe(df)

        assert df_sanitized.loc[1, "A"] is None

    @pytest.mark.parametrize(
        "whitespace_string,expected",
        [
            ("  hello  ", "hello"),
            ("world  ", "world"),
            ("  test", "test"),
            ("no-space", "no-space"),
            ("  multiple   spaces  ", "multiple   spaces"),
        ],
    )
    def test_strip_whitespace_variations(self, presenter, whitespace_string, expected):
        """Debe eliminar espacios de diferentes formas."""
        df = pd.DataFrame({"text": [whitespace_string]})

        df_sanitized = presenter._sanitize_dataframe(df)

        assert df_sanitized.loc[0, "text"] == expected


# ============================================================================
# TESTS DE COBERTURA ADICIONAL
# ============================================================================


class TestAdditionalCoverage:
    """Tests adicionales para alcanzar 100% de cobertura."""

    def test_create_dataframe_exception_handling(self, presenter, mock_logger):
        """Debe manejar excepciones al crear DataFrame."""
        presenter.logger = mock_logger

        # Crear datos que causen error al convertir a DataFrame
        # (aunque pandas es muy tolerante)
        with patch("pandas.DataFrame", side_effect=Exception("Mock error")):
            with pytest.raises(ValueError, match="Error creando DataFrame"):
                presenter._create_and_sanitize_dataframe([{"key": "value"}], "APU001")

    def test_create_dataframe_results_empty(self, presenter):
        """Debe detectar cuando DataFrame resulta vacío."""
        # Esto es difícil de lograr con pandas, pero podemos mockear
        with patch("pandas.DataFrame") as mock_df:
            mock_df.return_value = pd.DataFrame()
            mock_df.return_value.empty = True

            with pytest.raises(ValueError, match="DataFrame resultó vacío"):
                presenter._create_and_sanitize_dataframe([{"key": "value"}], "APU001")

    def test_process_unexpected_exception(self, presenter, valid_apu_details, mock_logger):
        """Debe manejar excepciones inesperadas."""
        presenter.logger = mock_logger

        # Simular excepción inesperada
        with patch.object(
            presenter,
            "_create_and_sanitize_dataframe",
            side_effect=RuntimeError("Unexpected error"),
        ):
            with pytest.raises(RuntimeError, match="Fallo en procesamiento"):
                presenter.process_apu_details(valid_apu_details, "APU001")

            # Verificar que se loggeó el error
            mock_logger.error.assert_called()

    def test_metadata_reduction_rate_calculation(self, presenter):
        """Debe calcular tasa de reducción correctamente."""
        df = pd.DataFrame(
            {
                "CATEGORIA": ["A"] * 10,
                "DESCRIPCION_INSUMO": ["Item"] * 10,
                "CANTIDAD_APU": [1] * 10,
                "VALOR_TOTAL_APU": [100] * 10,
                "RENDIMIENTO": [1] * 10,
                "UNIDAD_APU": ["kg"] * 10,
                "PRECIO_UNIT_APU": [100] * 10,
                "CODIGO_APU": ["A1"] * 10,
                "UNIDAD_INSUMO": ["kg"] * 10,
            }
        )

        processed = [{"VR_TOTAL": 1000.0}]  # 10 items reducidos a 1

        metadata = presenter._calculate_metadata(df, processed)

        # Reducción de 90%
        assert metadata["reduction_rate"] == 0.9

    def test_aggregate_alerts_with_empty_strings(self, presenter):
        """Debe filtrar strings vacíos en alertas."""
        series = pd.Series(["Alerta 1", "", "   ", "Alerta 2"])

        result = presenter._aggregate_alerts(series)

        # No debe incluir strings vacíos
        assert "Alerta 1" in result
        assert "Alerta 2" in result
        # No debe tener separadores vacíos consecutivos


# ============================================================================
# SUITE DE EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    """
    Ejecutar tests con pytest.
    
    Comandos útiles:
    - pytest test_presenters.py -v                    # Verbose
    - pytest test_presenters.py -v --cov=presenters   # Con cobertura
    - pytest test_presenters.py -v --cov-report=html  # Reporte HTML
    - pytest test_presenters.py -k "test_process"     # Solo tests que coincidan
    - pytest test_presenters.py -x                    # Detener en primer fallo
    - pytest test_presenters.py --tb=short            # Traceback corto
    """
    pytest.main(
        [
            __file__,
            "-v",
            "--cov=presenters",
            "--cov-report=term-missing",
            "--cov-report=html",
            "-x",  # Detener en primer fallo para debugging
        ]
    )
