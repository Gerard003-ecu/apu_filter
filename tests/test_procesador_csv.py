"""
Suite de pruebas robusta para el procesador CSV.

Incluye:
- Pruebas unitarias para cada clase especializada
- Pruebas de integración end-to-end
- Fixtures reutilizables
- Validaciones exhaustivas
- Manejo de casos edge
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.procesador_csv import (
    APUCostCalculator,
    APUTypes,
    ColumnNames,
    DataMerger,
    DataValidator,
    FileValidator,
    InsumosProcessor,
    InsumoTypes,
    PresupuestoProcessor,
    ProcessingThresholds,
    build_processed_apus_dataframe,
    calculate_insumo_costs,
    calculate_total_costs,
    group_and_split_description,
    process_all_files,
    synchronize_data_sources,
)

# Importar datos de prueba centralizados
from tests.test_data import TEST_CONFIG

# ==================== FIXTURES Y HELPERS ====================


class TestDataBuilder:
    """Constructor de datos de prueba reutilizables."""

    @staticmethod
    def create_presupuesto_csv(path: str, data: str = None) -> str:
        """Crea archivo CSV de presupuesto de prueba."""
        if data is None:
            data = (
                "ITEM;DESCRIPCION;UND;CANT.;VR. UNIT;VR.TOTAL\n"
                "1.1;INSTALACION TUBERIA PVC 3/4;ML;100;50000;5000000\n"
                "1.2;EXCAVACION MANUAL;M3;50;25000;1250000\n"
                "2.1;SUM. CEMENTO;UND;200;15000;3000000\n"
                "2.2;TRANSPORTE MATERIAL;VIAJE;10;80000;800000\n"
            )

        with open(path, "w", encoding="latin1") as f:
            f.write(data)

        return path

    @staticmethod
    def create_apus_csv(path: str, data: str = None) -> str:
        """Crea archivo CSV de APUs de prueba."""
        if data is None:
            data = (
                "ITEM: 1.1; UNIDAD: ML\n"
                "INSTALACION TUBERIA PVC 3/4\n"
                "MATERIALES\n"
                '"TUBERIA PVC 3/4";ML;1.05;;38095;40000\n'
                '"CEMENTO";UND;0.5;;10000;5000\n'
                "MANO DE OBRA\n"
                '"OFICIAL";50000;0;75000;8.0;9375\n'
                "EQUIPO\n"
                '"HERRAMIENTA MENOR";DIA;0.1;;50000;5000\n'
            )

        with open(path, "w", encoding="latin1") as f:
            f.write(data)

        return path

    @staticmethod
    def create_insumos_csv(path: str, data: str = None) -> str:
        """Crea archivo CSV de insumos de prueba."""
        if data is None:
            data = (
                "G;MATERIALES\n"
                "CODIGO;DESCRIPCION;UND;CANT.;VR. UNIT.\n"
                "M001;TUBERIA PVC 3/4;ML;1;40000\n"
                "M002;CEMENTO;UND;1;12000\n"
                "G;MANO DE OBRA\n"
                "CODIGO;DESCRIPCION;UND;CANT.;VR. UNIT.\n"
                "MO01;OFICIAL;JOR;1;80000\n"
                "G;EQUIPO\n"
                "CODIGO;DESCRIPCION;UND;CANT.;VR. UNIT.\n"
                "E001;HERRAMIENTA MENOR;DIA;1;55000\n"
            )

        with open(path, "w", encoding="latin1") as f:
            f.write(data)

        return path

    @staticmethod
    def create_sample_presupuesto_df() -> pd.DataFrame:
        """Crea DataFrame de presupuesto de muestra."""
        return pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["1.1", "1.2", "2.1"],
                ColumnNames.DESCRIPCION_APU: [
                    "INSTALACION TUBERIA",
                    "EXCAVACION MANUAL",
                    "SUMINISTRO CEMENTO",
                ],
                ColumnNames.CANTIDAD_PRESUPUESTO: [100, 50, 200],
            }
        )

    @staticmethod
    def create_sample_insumos_df() -> pd.DataFrame:
        """Crea DataFrame de insumos de muestra."""
        return pd.DataFrame(
            {
                ColumnNames.GRUPO_INSUMO: ["MATERIALES", "MATERIALES", "MANO DE OBRA"],
                ColumnNames.DESCRIPCION_INSUMO: ["TUBERIA PVC 3/4", "CEMENTO", "OFICIAL"],
                ColumnNames.VR_UNITARIO_INSUMO: [40000, 12000, 80000],
                ColumnNames.DESCRIPCION_INSUMO_NORM: [
                    "tuberia pvc 3/4",
                    "cemento",
                    "oficial",
                ],
            }
        )

    @staticmethod
    def create_sample_apus_df() -> pd.DataFrame:
        """Crea DataFrame de APUs procesados de muestra."""
        return pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["1.1", "1.1", "1.2"],
                ColumnNames.DESCRIPCION_APU: [
                    "INSTALACION TUBERIA",
                    "INSTALACION TUBERIA",
                    "EXCAVACION MANUAL",
                ],
                ColumnNames.UNIDAD_APU: ["ML", "ML", "M3"],
                ColumnNames.DESCRIPCION_INSUMO: ["TUBERIA PVC 3/4", "OFICIAL", "PEON"],
                ColumnNames.CANTIDAD_APU: [1.05, 0.125, 0.5],
                ColumnNames.PRECIO_UNIT_APU: [38095, 75000, 50000],
                ColumnNames.VALOR_TOTAL_APU: [40000, 9375, 25000],
                ColumnNames.CATEGORIA: ["MATERIALES", "MANO DE OBRA", "MANO DE OBRA"],
                ColumnNames.TIPO_INSUMO: [
                    InsumoTypes.SUMINISTRO,
                    InsumoTypes.MANO_DE_OBRA,
                    InsumoTypes.MANO_DE_OBRA,
                ],
                ColumnNames.NORMALIZED_DESC: ["tuberia pvc 3/4", "oficial", "peon"],
                ColumnNames.RENDIMIENTO: [0, 8.0, 10.0],
            }
        )


class TempFileManager:
    """Gestor de archivos temporales para pruebas."""

    def __init__(self):
        self.temp_files = []
        self.temp_dir = tempfile.mkdtemp()

    def create_temp_file(self, suffix: str = ".csv") -> str:
        """Crea un archivo temporal y registra su ruta."""
        fd, path = tempfile.mkstemp(suffix=suffix, dir=self.temp_dir)
        os.close(fd)
        self.temp_files.append(path)
        return path

    def cleanup(self):
        """Limpia todos los archivos temporales."""
        for path in self.temp_files:
            if os.path.exists(path):
                os.remove(path)

        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)


# ==================== PRUEBAS UNITARIAS ====================


class TestColumnNames(unittest.TestCase):
    """Pruebas para la clase de constantes ColumnNames."""

    def test_column_names_are_strings(self):
        """Verifica que todas las constantes sean strings."""
        for attr_name in dir(ColumnNames):
            if not attr_name.startswith("_"):
                attr_value = getattr(ColumnNames, attr_name)
                self.assertIsInstance(attr_value, str, f"{attr_name} debe ser un string")

    def test_no_duplicate_column_names(self):
        """Verifica que no haya nombres de columna duplicados."""
        column_values = [
            getattr(ColumnNames, attr)
            for attr in dir(ColumnNames)
            if not attr.startswith("_")
        ]

        self.assertEqual(
            len(column_values),
            len(set(column_values)),
            "No debe haber nombres de columna duplicados",
        )


class TestProcessingThresholds(unittest.TestCase):
    """Pruebas para la clase ProcessingThresholds."""

    def test_default_thresholds(self):
        """Verifica que los umbrales por defecto sean razonables."""
        thresholds = ProcessingThresholds()

        self.assertEqual(thresholds.outlier_std_multiplier, 3.0)
        self.assertEqual(thresholds.max_quantity, 1e6)
        self.assertEqual(thresholds.max_cost_per_item, 1e9)
        self.assertEqual(thresholds.max_total_cost, 1e11)
        self.assertEqual(thresholds.instalacion_mo_threshold, 75.0)

    def test_custom_thresholds(self):
        """Verifica que se puedan personalizar los umbrales."""
        thresholds = ProcessingThresholds(outlier_std_multiplier=2.5, max_total_cost=5e11)

        self.assertEqual(thresholds.outlier_std_multiplier, 2.5)
        self.assertEqual(thresholds.max_total_cost, 5e11)


class TestDataValidator(unittest.TestCase):
    """Pruebas para la clase DataValidator."""

    def setUp(self):
        self.validator = DataValidator()

    def test_validate_dataframe_not_empty_success(self):
        """Prueba validación exitosa de DataFrame no vacío."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        is_valid, error = self.validator.validate_dataframe_not_empty(df, "test_df")

        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_validate_dataframe_not_empty_failure(self):
        """Prueba validación fallida con DataFrame vacío."""
        df = pd.DataFrame()
        is_valid, error = self.validator.validate_dataframe_not_empty(df, "test_df")

        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
        self.assertIn("vacío", error.lower())

    def test_validate_dataframe_none(self):
        """Prueba validación con DataFrame None."""
        is_valid, error = self.validator.validate_dataframe_not_empty(None, "test_df")

        self.assertFalse(is_valid)
        self.assertIsNotNone(error)

    def test_validate_required_columns_success(self):
        """Prueba validación exitosa de columnas requeridas."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4], "col3": [5, 6]})

        is_valid, error = self.validator.validate_required_columns(
            df, ["col1", "col2"], "test_df"
        )

        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_validate_required_columns_failure(self):
        """Prueba validación fallida con columnas faltantes."""
        df = pd.DataFrame({"col1": [1, 2]})

        is_valid, error = self.validator.validate_required_columns(
            df, ["col1", "col2", "col3"], "test_df"
        )

        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
        self.assertIn("col2", error)
        self.assertIn("col3", error)

    def test_detect_and_log_duplicates_no_duplicates(self):
        """Prueba detección sin duplicados."""
        df = pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})

        result = self.validator.detect_and_log_duplicates(df, ["id"], "test_df")

        self.assertEqual(len(result), 3)
        pd.testing.assert_frame_equal(result, df)

    def test_detect_and_log_duplicates_with_duplicates(self):
        """Prueba detección y eliminación de duplicados."""
        df = pd.DataFrame({"id": [1, 2, 2, 3], "value": ["a", "b", "c", "d"]})

        with self.assertLogs("app.procesador_csv", level="WARNING"):
            result = self.validator.detect_and_log_duplicates(
                df, ["id"], "test_df", keep="first"
            )

        self.assertEqual(len(result), 3)
        self.assertListEqual(result["id"].tolist(), [1, 2, 3])

    def test_validate_numeric_range_success(self):
        """Prueba validación de rango numérico exitosa."""
        series = pd.Series([10, 20, 30, 40])

        is_valid = self.validator.validate_numeric_range(
            series, "test_column", max_value=50, min_value=5
        )

        self.assertTrue(is_valid)

    def test_validate_numeric_range_failure(self):
        """Prueba validación de rango numérico fallida."""
        series = pd.Series([10, 20, 100, 40])

        with self.assertLogs("app.procesador_csv", level="WARNING"):
            is_valid = self.validator.validate_numeric_range(
                series, "test_column", max_value=50, min_value=5
            )

        self.assertFalse(is_valid)


class TestFileValidator(unittest.TestCase):
    """Pruebas para la clase FileValidator."""

    def setUp(self):
        self.validator = FileValidator()
        self.temp_manager = TempFileManager()

    def tearDown(self):
        self.temp_manager.cleanup()

    def test_validate_file_exists_success(self):
        """Prueba validación exitosa de archivo existente."""
        temp_file = self.temp_manager.create_temp_file()

        is_valid, error = self.validator.validate_file_exists(temp_file, "test")

        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_validate_file_exists_failure(self):
        """Prueba validación fallida con archivo inexistente."""
        non_existent = "/path/to/nonexistent/file.csv"

        is_valid, error = self.validator.validate_file_exists(non_existent, "test")

        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
        self.assertIn("no encontrado", error.lower())

    def test_validate_file_is_directory(self):
        """Prueba validación fallida cuando la ruta es un directorio."""
        is_valid, error = self.validator.validate_file_exists(
            self.temp_manager.temp_dir, "test"
        )

        self.assertFalse(is_valid)
        self.assertIsNotNone(error)


class TestPresupuestoProcessor(unittest.TestCase):
    """Pruebas para la clase PresupuestoProcessor."""

    def setUp(self):
        self.config = TEST_CONFIG.copy()
        self.thresholds = ProcessingThresholds()
        # CAMBIO: Extraer y pasar el perfil
        profile = self.config.get("file_profiles", {}).get("presupuesto_default", {})
        self.processor = PresupuestoProcessor(self.config, self.thresholds, profile)
        self.temp_manager = TempFileManager()

    def tearDown(self):
        self.temp_manager.cleanup()

    @patch("app.procesador_csv.load_data")
    def test_process_success(self, mock_load_data):
        """Prueba procesamiento exitoso de presupuesto."""
        # Configurar el mock para que devuelva un DataFrame de prueba
        mock_df = pd.DataFrame(
            [
                ["", "", "", ""],
                ["Proyecto X", "", "", ""],
                ["ITEM", "DESCRIPCION", "UND", "CANT."],
                ["1.1", "APU 1", "ML", "100"],
            ]
        )
        mock_load_data.return_value = mock_df

        # Crear un archivo temporal (aunque su contenido no se usará)
        temp_file = self.temp_manager.create_temp_file()

        result = self.processor.process(temp_file)

        self.assertFalse(result.empty)
        self.assertIn(ColumnNames.CODIGO_APU, result.columns)
        self.assertIn(ColumnNames.DESCRIPCION_APU, result.columns)
        self.assertIn(ColumnNames.CANTIDAD_PRESUPUESTO, result.columns)
        self.assertEqual(len(result), 1)

    def test_process_with_duplicates(self):
        """Prueba que se eliminen duplicados correctamente."""
        temp_file = self.temp_manager.create_temp_file()
        data = (
            "ITEM;DESCRIPCION;UND;CANT.;VR. UNIT;VR.TOTAL\n"
            "1.1;APU 1;ML;100;50000;5000000\n"
            "1.1;APU 1 DUPLICADO;ML;150;60000;9000000\n"
            "1.2;APU 2;M3;50;25000;1250000\n"
        )
        TestDataBuilder.create_presupuesto_csv(temp_file, data)

        with self.assertLogs("app.procesador_csv", level="WARNING"):
            result = self.processor.process(temp_file)

        # Debe conservar solo el primero
        self.assertEqual(len(result), 2)
        self.assertEqual(result[ColumnNames.CODIGO_APU].tolist(), ["1.1", "1.2"])

    def test_process_invalid_file(self):
        """Prueba manejo de archivo inválido."""
        result = self.processor.process("/path/to/nonexistent.csv")

        self.assertTrue(result.empty)

    def test_find_and_set_header_success(self):
        """Prueba búsqueda exitosa de encabezado."""
        df = pd.DataFrame(
            [
                ["", "", "", ""],
                ["Proyecto X", "", "", ""],
                ["ITEM", "DESCRIPCION", "UND", "CANT."],
                ["1.1", "APU 1", "ML", "100"],
            ]
        )

        result = self.processor._find_and_set_header(df)

        self.assertFalse(result.empty)
        self.assertIn("ITEM", result.columns)
        self.assertEqual(len(result), 1)

    def test_find_and_set_header_not_found(self):
        """Prueba cuando no se encuentra encabezado."""
        df = pd.DataFrame([["datos", "sin", "formato"], ["invalido", "para", "presupuesto"]])

        with self.assertLogs("app.procesador_csv", level="ERROR"):
            result = self.processor._find_and_set_header(df)

        self.assertTrue(result.empty)

    def test_process_with_single_digit_item(self):
        """Prueba que procese correctamente un ITEM de un solo dígito."""
        temp_file = self.temp_manager.create_temp_file()
        data = (
            "ITEM;DESCRIPCION;UND;CANT.\n1;APU de un digito;ML;100\n1.1;APU normal;M3;50\n"
        )
        TestDataBuilder.create_presupuesto_csv(temp_file, data)

        result = self.processor.process(temp_file)

        self.assertEqual(len(result), 2)
        self.assertIn("1", result[ColumnNames.CODIGO_APU].tolist())
        self.assertEqual(result.iloc[0][ColumnNames.CODIGO_APU], "1")


class TestInsumosProcessor(unittest.TestCase):
    """Pruebas para la clase InsumosProcessor."""

    def setUp(self):
        self.thresholds = ProcessingThresholds()
        # CAMBIO: Extraer y pasar el perfil
        profile = TEST_CONFIG.get("file_profiles", {}).get("insumos_default", {})
        self.processor = InsumosProcessor(self.thresholds, profile)
        self.temp_manager = TempFileManager()

    def tearDown(self):
        self.temp_manager.cleanup()

    def test_process_success(self):
        """Prueba procesamiento exitoso de insumos."""
        temp_file = self.temp_manager.create_temp_file()
        TestDataBuilder.create_insumos_csv(temp_file)

        result = self.processor.process(temp_file)

        self.assertFalse(result.empty)
        self.assertIn(ColumnNames.GRUPO_INSUMO, result.columns)
        self.assertIn(ColumnNames.DESCRIPCION_INSUMO, result.columns)
        self.assertIn(ColumnNames.VR_UNITARIO_INSUMO, result.columns)
        self.assertIn(ColumnNames.DESCRIPCION_INSUMO_NORM, result.columns)

    def test_process_with_duplicates(self):
        """Prueba eliminación de duplicados conservando el de mayor precio."""
        temp_file = self.temp_manager.create_temp_file()
        data = (
            "G;MATERIALES\n"
            "CODIGO;DESCRIPCION;UND;CANT.;VR. UNIT.\n"
            "M001;CEMENTO;UND;1;10000\n"
            "M002;CEMENTO;UND;1;12000\n"  # Mayor precio
            "M003;CEMENTO;UND;1;11000\n"
        )
        TestDataBuilder.create_insumos_csv(temp_file, data)

        with self.assertLogs("app.procesador_csv", level="WARNING"):
            result = self.processor.process(temp_file)

        # Debe conservar el de mayor precio
        cemento = result[result[ColumnNames.DESCRIPCION_INSUMO] == "CEMENTO"]
        self.assertEqual(len(cemento), 1)
        self.assertEqual(cemento[ColumnNames.VR_UNITARIO_INSUMO].iloc[0], 12000)

    def test_parse_file_structure(self):
        """Prueba el parseo de la estructura del archivo."""
        temp_file = self.temp_manager.create_temp_file()
        TestDataBuilder.create_insumos_csv(temp_file)

        records = self.processor._parse_file(temp_file)

        self.assertGreater(len(records), 0)
        self.assertIn(ColumnNames.GRUPO_INSUMO, records[0])


class TestAPUCostCalculator(unittest.TestCase):
    """Pruebas para la clase APUCostCalculator."""

    def setUp(self):
        self.config = TEST_CONFIG.copy()
        self.thresholds = ProcessingThresholds()
        self.calculator = APUCostCalculator(self.config, self.thresholds)

    def test_calculate_complete_flow(self):
        """Prueba el flujo completo de cálculo de costos."""
        df_merged = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["1.1", "1.1", "1.2"],
                ColumnNames.TIPO_INSUMO: [
                    InsumoTypes.SUMINISTRO,
                    InsumoTypes.MANO_DE_OBRA,
                    InsumoTypes.SUMINISTRO,
                ],
                ColumnNames.COSTO_INSUMO_EN_APU: [40000, 9375, 25000],
                ColumnNames.CANTIDAD_APU: [1.05, 0.125, 1.0],
                ColumnNames.RENDIMIENTO: [0, 8.0, 10.0],
            }
        )

        df_costos, df_tiempo, df_rendimiento = self.calculator.calculate(df_merged)

        # Verificar estructura de costos
        self.assertFalse(df_costos.empty)
        self.assertIn(ColumnNames.CODIGO_APU, df_costos.columns)
        self.assertIn(ColumnNames.VALOR_SUMINISTRO_UN, df_costos.columns)
        self.assertIn(ColumnNames.VALOR_INSTALACION_UN, df_costos.columns)
        self.assertIn(ColumnNames.VALOR_CONSTRUCCION_UN, df_costos.columns)
        self.assertIn(ColumnNames.TIPO_APU, df_costos.columns)

        # Verificar tiempo
        self.assertFalse(df_tiempo.empty)
        self.assertIn(ColumnNames.TIEMPO_INSTALACION, df_tiempo.columns)

        # Verificar rendimiento
        self.assertFalse(df_rendimiento.empty)
        self.assertIn(ColumnNames.RENDIMIENTO_DIA, df_rendimiento.columns)

    def test_classify_apus_instalacion(self):
        """Prueba clasificación de APU tipo Instalación."""
        df = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["1.1"],
                ColumnNames.MATERIALES: [10000],
                ColumnNames.MANO_DE_OBRA: [70000],
                ColumnNames.EQUIPO: [10000],
                ColumnNames.OTROS: [0],
            }
        )

        df = self.calculator._calculate_unit_values(df)
        df = self.calculator._classify_apus(df)

        self.assertEqual(df[ColumnNames.TIPO_APU].iloc[0], APUTypes.INSTALACION)

    def test_classify_apus_suministro(self):
        """Prueba clasificación de APU tipo Suministro."""
        df = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["1.1"],
                ColumnNames.MATERIALES: [80000],
                ColumnNames.MANO_DE_OBRA: [9000],
                ColumnNames.EQUIPO: [5000],
                ColumnNames.OTROS: [0],
            }
        )

        df = self.calculator._calculate_unit_values(df)
        df = self.calculator._classify_apus(df)

        self.assertEqual(df[ColumnNames.TIPO_APU].iloc[0], APUTypes.SUMINISTRO)

    def test_classify_apus_obra_completa(self):
        """Prueba clasificación de APU tipo Obra Completa."""
        df = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["1.1"],
                ColumnNames.MATERIALES: [40000],
                ColumnNames.MANO_DE_OBRA: [30000],
                ColumnNames.EQUIPO: [20000],
                ColumnNames.OTROS: [10000],
            }
        )

        df = self.calculator._calculate_unit_values(df)
        df = self.calculator._classify_apus(df)

        self.assertEqual(df[ColumnNames.TIPO_APU].iloc[0], APUTypes.OBRA_COMPLETA)

    def test_detect_cost_outliers(self):
        """Prueba detección de valores atípicos."""
        df = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["1.1", "1.2", "1.3", "1.4"],
                ColumnNames.VALOR_CONSTRUCCION_UN: [100000, 110000, 105000, 1000000],
            }
        )

        result = self.calculator._detect_cost_outliers(df)

        self.assertEqual(len(result), 4)


class TestDataMerger(unittest.TestCase):
    """Pruebas para la clase DataMerger."""

    def setUp(self):
        self.thresholds = ProcessingThresholds()
        self.merger = DataMerger(self.thresholds)

    def test_merge_apus_with_insumos_success(self):
        """Prueba merge exitoso de APUs con insumos."""
        df_apus = TestDataBuilder.create_sample_apus_df()
        df_insumos = TestDataBuilder.create_sample_insumos_df()

        result = self.merger.merge_apus_with_insumos(df_apus, df_insumos)

        self.assertFalse(result.empty)
        self.assertIn(ColumnNames.TIPO_INSUMO, result.columns)

    def test_merge_apus_without_normalized_desc(self):
        """Prueba merge cuando falta NORMALIZED_DESC (fallback)."""
        df_apus = TestDataBuilder.create_sample_apus_df()
        df_apus = df_apus.drop(columns=[ColumnNames.NORMALIZED_DESC])
        df_insumos = TestDataBuilder.create_sample_insumos_df()

        with self.assertLogs("app.procesador_csv", level="WARNING"):
            result = self.merger.merge_apus_with_insumos(df_apus, df_insumos)

        self.assertIn(ColumnNames.NORMALIZED_DESC, result.columns)

    def test_merge_with_presupuesto_success(self):
        """Prueba merge exitoso 1:1 con presupuesto."""
        df_presupuesto = TestDataBuilder.create_sample_presupuesto_df()

        df_costos = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["1.1", "1.2", "2.1"],
                ColumnNames.VALOR_SUMINISTRO_UN: [50000, 30000, 15000],
                ColumnNames.VALOR_INSTALACION_UN: [20000, 15000, 5000],
                ColumnNames.VALOR_CONSTRUCCION_UN: [70000, 45000, 20000],
            }
        )

        result = self.merger.merge_with_presupuesto(df_presupuesto, df_costos)

        self.assertEqual(len(result), 3)
        self.assertIn(ColumnNames.VALOR_CONSTRUCCION_UN, result.columns)

    def test_merge_with_presupuesto_duplicates(self):
        """Prueba que falle con duplicados (explosión cartesiana)."""
        df_presupuesto = TestDataBuilder.create_sample_presupuesto_df()

        df_costos = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["1.1", "1.1", "1.2"],  # Duplicado
                ColumnNames.VALOR_SUMINISTRO_UN: [50000, 55000, 30000],
                ColumnNames.VALOR_INSTALACION_UN: [20000, 22000, 15000],
                ColumnNames.VALOR_CONSTRUCCION_UN: [70000, 77000, 45000],
            }
        )

        with self.assertRaises(pd.errors.MergeError):
            with self.assertLogs("app.procesador_csv", level="ERROR"):
                self.merger.merge_with_presupuesto(df_presupuesto, df_costos)


# ==================== PRUEBAS DE FUNCIONES AUXILIARES ====================


class TestAuxiliaryFunctions(unittest.TestCase):
    """Pruebas para funciones auxiliares."""

    def test_group_and_split_description(self):
        """Prueba división de descripciones."""
        df = pd.DataFrame(
            {
                ColumnNames.DESCRIPCION_APU: [
                    "INSTALACION / TUBERIA PVC",
                    "SUMINISTRO CEMENTO",
                    "EXCAVACION / MANUAL / PROFUNDA",
                ]
            }
        )

        result = group_and_split_description(df)

        self.assertIn(ColumnNames.ORIGINAL_DESCRIPTION, result.columns)
        self.assertIn(ColumnNames.DESCRIPCION_SECUNDARIA, result.columns)
        self.assertEqual(result[ColumnNames.DESCRIPCION_APU].iloc[0], "INSTALACION")
        self.assertEqual(result[ColumnNames.DESCRIPCION_SECUNDARIA].iloc[0], "TUBERIA PVC")

    def test_calculate_insumo_costs(self):
        """Prueba cálculo de costos de insumos."""
        df = pd.DataFrame(
            {
                ColumnNames.CANTIDAD_APU: [1.05, 0.5],
                ColumnNames.VR_UNITARIO_INSUMO: [40000, 12000],
                ColumnNames.VALOR_TOTAL_APU: [42000, 6000],
            }
        )

        thresholds = ProcessingThresholds()
        result = calculate_insumo_costs(df, thresholds)

        self.assertIn(ColumnNames.COSTO_INSUMO_EN_APU, result.columns)
        self.assertIn(ColumnNames.VR_UNITARIO_FINAL, result.columns)
        self.assertEqual(result[ColumnNames.COSTO_INSUMO_EN_APU].iloc[0], 42000)

    def test_calculate_total_costs(self):
        """Prueba cálculo de valores totales."""
        df = pd.DataFrame(
            {
                ColumnNames.CANTIDAD_PRESUPUESTO: [100, 50],
                ColumnNames.VALOR_SUMINISTRO_UN: [50000, 30000],
                ColumnNames.VALOR_INSTALACION_UN: [20000, 15000],
                ColumnNames.VALOR_CONSTRUCCION_UN: [70000, 45000],
            }
        )

        thresholds = ProcessingThresholds()
        result = calculate_total_costs(df, thresholds)

        self.assertIn(ColumnNames.VALOR_SUMINISTRO_TOTAL, result.columns)
        self.assertIn(ColumnNames.VALOR_INSTALACION_TOTAL, result.columns)
        self.assertIn(ColumnNames.VALOR_CONSTRUCCION_TOTAL, result.columns)

        self.assertEqual(result[ColumnNames.VALOR_CONSTRUCCION_TOTAL].iloc[0], 7000000)

    def test_calculate_total_costs_exceeds_threshold(self):
        """Prueba que lance error cuando se excede el umbral."""
        df = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["BIG.1"],
                ColumnNames.DESCRIPCION_APU: ["APU GIGANTE"],
                ColumnNames.CANTIDAD_PRESUPUESTO: [1e6],
                ColumnNames.VALOR_SUMINISTRO_UN: [1e6],
                ColumnNames.VALOR_INSTALACION_UN: [0],
                ColumnNames.VALOR_CONSTRUCCION_UN: [1e6],
            }
        )

        thresholds = ProcessingThresholds(max_total_cost=1e10)

        with self.assertRaises(ValueError):
            with self.assertLogs("app.procesador_csv", level="ERROR"):
                calculate_total_costs(df, thresholds)

    def test_synchronize_data_sources(self):
        """Prueba sincronización de fuentes de datos."""
        df_merged = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["1.1", "1.2", "1.3", "2.1"],
                "data": ["a", "b", "c", "d"],
            }
        )

        df_final = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["1.1", "1.2", "2.1"]  # Sin 1.3
            }
        )

        with self.assertLogs("app.procesador_csv", level="INFO"):
            result = synchronize_data_sources(df_merged, df_final)

        self.assertEqual(len(result), 3)
        self.assertNotIn("1.3", result[ColumnNames.CODIGO_APU].tolist())

    def test_build_processed_apus_dataframe(self):
        """Prueba construcción de DataFrame de APUs procesados."""
        df_costos = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["1.1", "1.2"],
                ColumnNames.VALOR_CONSTRUCCION_UN: [70000, 45000],
            }
        )

        df_apus_raw = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["1.1", "1.2"],
                ColumnNames.DESCRIPCION_APU: ["APU 1", "APU 2"],
                ColumnNames.UNIDAD_APU: ["ML", "M3"],
            }
        )

        df_tiempo = pd.DataFrame(
            {ColumnNames.CODIGO_APU: ["1.1"], ColumnNames.TIEMPO_INSTALACION: [0.125]}
        )

        df_rendimiento = pd.DataFrame(
            {ColumnNames.CODIGO_APU: ["1.1"], ColumnNames.RENDIMIENTO_DIA: [8.0]}
        )

        result = build_processed_apus_dataframe(
            df_costos, df_apus_raw, df_tiempo, df_rendimiento
        )

        self.assertFalse(result.empty)
        self.assertIn("UNIDAD", result.columns)
        self.assertIn("DESC_NORMALIZED", result.columns)
        self.assertIn(ColumnNames.ORIGINAL_DESCRIPTION, result.columns)


# ==================== PRUEBAS DE INTEGRACIÓN ====================


class TestProcessAllFilesIntegration(unittest.TestCase):
    """Pruebas de integración end-to-end."""

    def setUp(self):
        """Configura archivos temporales para pruebas."""
        self.temp_manager = TempFileManager()

        self.presupuesto_path = self.temp_manager.create_temp_file()
        self.apus_path = self.temp_manager.create_temp_file()
        self.insumos_path = self.temp_manager.create_temp_file()

        TestDataBuilder.create_presupuesto_csv(self.presupuesto_path)
        TestDataBuilder.create_apus_csv(self.apus_path)
        TestDataBuilder.create_insumos_csv(self.insumos_path)

    def tearDown(self):
        """Limpia archivos temporales."""
        self.temp_manager.cleanup()

    @patch("app.procesador_csv._save_output_files")
    def test_process_all_files_success(self, mock_save_output_files):
        """Prueba procesamiento completo exitoso, simulando guardado de archivos."""
        # Crear un mock de Path para evitar el error 'str' object has no attribute 'exists'
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.stat.return_value.st_size = 12345
        # Simulamos que _save_output_files devuelve un diccionario con mocks de Path
        mock_save_output_files.return_value = {
            "processed_apus": mock_path,
            "presupuesto_final": mock_path,
            "insumos_detalle": mock_path,
        }

        with (
            patch("app.procesador_csv.ReportParserCrudo") as mock_parser_class,
            patch("app.procesador_csv.APUProcessor") as mock_processor_class,
        ):
            mock_parser = MagicMock()
            mock_parser_class.return_value = mock_parser
            mock_parser.parse_to_raw.return_value = []

            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor
            mock_processor.process_all.return_value = TestDataBuilder.create_sample_apus_df()

            resultado = process_all_files(
                self.presupuesto_path, self.apus_path, self.insumos_path, TEST_CONFIG
            )

            self.assertIsInstance(resultado, dict)
            self.assertNotIn(
                "error",
                resultado,
                f"Se encontró un error inesperado: {resultado.get('error')}",
            )

            expected_keys = [
                "presupuesto",
                "insumos",
                "apus_detail",
                "all_apus",
                "processed_apus",
                "output_files",
            ]
            for key in expected_keys:
                self.assertIn(key, resultado, f"Falta clave: {key}")

            # Verificar la nueva estructura de salida
            self.assertIn("processed_apus", resultado["output_files"])
            mock_save_output_files.assert_called_once()

    def test_process_all_files_file_not_found(self):
        """Prueba manejo de archivos no encontrados."""
        resultado = process_all_files(
            "/nonexistent/presupuesto.csv", self.apus_path, self.insumos_path, TEST_CONFIG
        )

        self.assertIn("error", resultado)
        self.assertIn("no encontrado", resultado["error"].lower())

    def test_process_all_files_empty_presupuesto(self):
        """Prueba manejo de presupuesto vacío."""
        with patch("app.procesador_csv.PresupuestoProcessor") as mock_class:
            mock_processor = MagicMock()
            mock_class.return_value = mock_processor
            mock_processor.process.return_value = pd.DataFrame()

            resultado = process_all_files(
                self.presupuesto_path, self.apus_path, self.insumos_path, TEST_CONFIG
            )

            self.assertIn("error", resultado)
            self.assertIn("presupuesto", resultado["error"].lower())

    def test_process_all_files_custom_thresholds(self):
        """Prueba aplicación de umbrales personalizados."""
        custom_config = TEST_CONFIG.copy()
        custom_config["processing_thresholds"] = {
            "outlier_std_multiplier": 2.5,
            "max_total_cost": 5e11,
        }

        with patch("app.procesador_csv.ReportParserCrudo"):
            with patch("app.procesador_csv.APUProcessor") as mock_processor_class:
                mock_processor = MagicMock()
                mock_processor_class.return_value = mock_processor
                mock_processor.process_all.return_value = pd.DataFrame()  # Return empty df

                resultado = process_all_files(
                    self.presupuesto_path, self.apus_path, self.insumos_path, custom_config
                )

                # Debe procesar sin errores con los umbrales personalizados
                self.assertIsInstance(resultado, dict)

    def test_process_all_files_merge_error(self):
        """
        Prueba el manejo de un pd.errors.MergeError durante el merge final.

        Esta prueba valida que si el merge entre el presupuesto y los costos de APU falla,
        el pipeline se detiene de forma controlada y devuelve un diccionario de error.
        """
        # --- Configuración de Mocks ---
        # 1. Mockeamos procesadores iniciales para que devuelvan DataFrames
        #    válidos y no vacíos. Así, el pipeline progresa.
        # 2. Mockeamos el método que queremos que falle: `merge_with_presupuesto`.

        with (
            patch("app.procesador_csv.ReportParserCrudo") as mock_parser_class,
            patch("app.procesador_csv.APUProcessor") as mock_processor_class,
            patch(
                "app.procesador_csv.DataMerger.merge_with_presupuesto"
            ) as mock_final_merge,
        ):
            # Configurar mock de APUProcessor para que devuelva un DF realista.
            # Esto es crucial para que los pasos intermedios funcionen.
            mock_processor_instance = MagicMock()
            mock_processor_instance.process_all.return_value = (
                TestDataBuilder.create_sample_apus_df()
            )
            mock_processor_class.return_value = mock_processor_instance

            # El mock del ReportParserCrudo solo necesita devolver algo no vacío.
            mock_parser_instance = MagicMock()
            mock_parser_instance.parse_to_raw.return_value = [{"data": "dummy"}]
            mock_parser_class.return_value = mock_parser_instance

            # --- ¡LA CLAVE DE LA SOLUCIÓN! ---
            # Hacemos que `merge_with_presupuesto` lance el error deseado.
            # El resto de DataMerger funcionará normalmente.
            mock_final_merge.side_effect = pd.errors.MergeError(
                "Simulated 1:1 merge error due to duplicates"
            )

            # --- Ejecución ---
            # Llamamos a la función principal que orquesta todo el proceso.
            resultado = process_all_files(
                self.presupuesto_path, self.apus_path, self.insumos_path, TEST_CONFIG
            )

            # --- Aserciones ---
            # 1. Verificar que la función devolvió un diccionario y no se estrelló.
            self.assertIsInstance(resultado, dict)

            # 2. Verificar que el diccionario contiene la clave 'error'.
            self.assertIn("error", resultado)

            # 3. Verificar que el mensaje de error esperado está presente.
            self.assertIn("merge", resultado["error"].lower())

            # 4. Verificar que el método mockeado fue llamado.
            mock_final_merge.assert_called_once()

    def test_process_all_files_apu_load_failure_triggers_diagnostic(self):
        """Prueba que el fallo en carga de APUs active el diagnóstico."""
        with patch("app.procesador_csv.LoadDataStep.execute") as mock_execute:
            mock_execute.side_effect = ValueError("Error de carga de apus")

            with patch("app.procesador_csv.APUFileDiagnostic") as mock_diagnostic_class:
                mock_diagnostic_instance = MagicMock()
                mock_diagnostic_class.return_value = mock_diagnostic_instance

                resultado = process_all_files(
                    self.presupuesto_path, self.apus_path, self.insumos_path, TEST_CONFIG
                )

                self.assertIn("error", resultado)
                self.assertIn("apus", resultado["error"].lower())

                # Verificar que el diagnóstico fue llamado
                mock_diagnostic_class.assert_called_once_with(self.apus_path)
                mock_diagnostic_instance.diagnose.assert_called_once()


class TestEdgeCases(unittest.TestCase):
    """Pruebas de casos límite y escenarios especiales."""

    def setUp(self):
        self.temp_manager = TempFileManager()
        self.thresholds = ProcessingThresholds()
        # CAMBIO: Añadir el perfil para la prueba que lo necesita
        self.insumos_profile = TEST_CONFIG.get("file_profiles", {}).get(
            "insumos_default", {}
        )

    def tearDown(self):
        self.temp_manager.cleanup()

    def test_empty_insumos_group(self):
        """Prueba manejo de grupos de insumos vacíos."""
        temp_file = self.temp_manager.create_temp_file()
        data = (
            "G;MATERIALES\n"
            "CODIGO;DESCRIPCION;UND;CANT.;VR. UNIT.\n"
            "G;MANO DE OBRA\n"  # Grupo sin datos
            "CODIGO;DESCRIPCION;UND;CANT.;VR. UNIT.\n"
            "MO01;OFICIAL;JOR;1;80000\n"
        )
        TestDataBuilder.create_insumos_csv(temp_file, data)

        # CAMBIO: Pasar el perfil al crear el procesador
        processor = InsumosProcessor(self.thresholds, self.insumos_profile)
        result = processor.process(temp_file)

        self.assertFalse(result.empty)
        self.assertIn("OFICIAL", result[ColumnNames.DESCRIPCION_INSUMO].values)

    def test_special_characters_in_descriptions(self):
        """Prueba manejo de caracteres especiales."""
        df = pd.DataFrame(
            {
                ColumnNames.DESCRIPCION_APU: [
                    'TUBERÍA PVC 3/4" Ø20mm',
                    "EXCAVACIÓN (MANUAL) - TIPO A",
                    "CEMENTO 50kg @ $15.000",
                ]
            }
        )

        result = group_and_split_description(df)

        self.assertEqual(len(result), 3)
        self.assertIn(ColumnNames.ORIGINAL_DESCRIPTION, result.columns)

    def test_zero_quantities(self):
        """Prueba manejo de cantidades en cero."""
        df = pd.DataFrame(
            {
                ColumnNames.CANTIDAD_APU: [0, 1.05, 0],
                ColumnNames.VR_UNITARIO_INSUMO: [40000, 12000, 5000],
                ColumnNames.VALOR_TOTAL_APU: [0, 12600, 0],
            }
        )

        result = calculate_insumo_costs(df, self.thresholds)

        # No debe generar errores ni NaN
        self.assertFalse(result[ColumnNames.COSTO_INSUMO_EN_APU].isna().any())
        self.assertFalse(result[ColumnNames.VR_UNITARIO_FINAL].isna().any())

    def test_very_long_descriptions(self):
        """Prueba manejo de descripciones muy largas."""
        long_desc = "A" * 1000
        df = pd.DataFrame({ColumnNames.DESCRIPCION_APU: [long_desc]})

        result = group_and_split_description(df)

        self.assertEqual(len(result[ColumnNames.DESCRIPCION_APU].iloc[0]), 1000)

    def test_negative_costs(self):
        """Prueba detección de costos negativos."""
        df = pd.DataFrame(
            {
                ColumnNames.CANTIDAD_APU: [1.0],
                ColumnNames.VR_UNITARIO_INSUMO: [-1000],  # Negativo
                ColumnNames.VALOR_TOTAL_APU: [1000],
            }
        )

        result = calculate_insumo_costs(df, self.thresholds)

        # Debe calcular sin crash (aunque sea anómalo)
        self.assertIsNotNone(result[ColumnNames.COSTO_INSUMO_EN_APU].iloc[0])


# ==================== SUITE DE PRUEBAS ====================


def suite():
    """Construye la suite completa de pruebas."""
    test_suite = unittest.TestSuite()

    # Pruebas unitarias
    test_suite.addTest(unittest.makeSuite(TestColumnNames))
    test_suite.addTest(unittest.makeSuite(TestProcessingThresholds))
    test_suite.addTest(unittest.makeSuite(TestDataValidator))
    test_suite.addTest(unittest.makeSuite(TestFileValidator))
    test_suite.addTest(unittest.makeSuite(TestPresupuestoProcessor))
    test_suite.addTest(unittest.makeSuite(TestInsumosProcessor))
    test_suite.addTest(unittest.makeSuite(TestAPUCostCalculator))
    test_suite.addTest(unittest.makeSuite(TestDataMerger))
    test_suite.addTest(unittest.makeSuite(TestAuxiliaryFunctions))

    # Pruebas de integración
    test_suite.addTest(unittest.makeSuite(TestProcessAllFilesIntegration))
    test_suite.addTest(unittest.makeSuite(TestEdgeCases))

    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2, failfast=False)
    runner.run(suite())
