"""
Suite de pruebas para el módulo utils.py

Este módulo contiene pruebas exhaustivas y perfectamente alineadas
con la implementación actual del módulo utils.py

"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

# Importar el módulo a probar
from app import utils

# ============================================================================
# FIXTURES Y UTILIDADES DE PRUEBA MEJORADAS
# ============================================================================

class TestDataFactory:
    """Factory mejorada para generar datos de prueba consistentes."""

    @staticmethod
    def create_sample_dataframe(n_rows: int = 100, seed: int = 42) -> pd.DataFrame:
        """Crea un DataFrame de muestra para pruebas."""
        np.random.seed(seed)
        return pd.DataFrame({
            'codigo': [f'APU{i:03d}' for i in range(n_rows)],
            'descripcion': [f'Descripción {i}' for i in range(n_rows)],
            'unidad': np.random.choice(['M', 'M2', 'KG', 'HR'], n_rows),
            'cantidad': np.random.uniform(0.1, 100, n_rows),
            'precio': np.random.uniform(10, 1000, n_rows),
            'texto_mixto': [f'Texto con Ñ, é, ü #{i}' for i in range(n_rows)]
        })

    @staticmethod
    def create_numeric_series_with_outliers() -> pd.Series:
        """Crea una serie numérica con valores atípicos."""
        np.random.seed(42)
        normal_data = np.random.normal(100, 15, 95)
        outliers = [500, -100, 1000, 0.001, 999]
        return pd.Series(np.concatenate([normal_data, outliers]))

    @staticmethod
    def create_mixed_type_series() -> pd.Series:
        """Crea una serie con tipos mixtos para pruebas de robustez."""
        return pd.Series([
            "100", 200, 300.5, None, pd.NA, np.nan,
            "400,50", "$500", "600.789", True, False
        ])

# ============================================================================
# PRUEBAS DE NORMALIZACIÓN DE TEXTO - REFINADAS
# ============================================================================

class TestTextNormalization(unittest.TestCase):
    """Pruebas refinadas para funciones de normalización de texto."""

    def setUp(self):
        """Configuración inicial para cada prueba."""
        utils.normalize_text.cache_clear()

    def test_normalize_text_basic(self):
        """Prueba normalización básica de texto."""
        test_cases = [
            ("TEXTO EN MAYÚSCULAS", "texto en mayusculas"),
            ("  espacios  extras  ", "espacios extras"),
            ("Ñoño con eñes", "nono con enes"),
            ("Café, té y más!", "cafe te y mas"),
            ("", ""),
            ("123 números 456", "123 numeros 456"),
            ("texto-con-guiones", "textoconguiones"),  # Sin preserve_special_chars
            ("texto_con_underscore", "textoconunderscore"),
        ]

        for input_text, expected in test_cases:
            with self.subTest(input=input_text):
                result = utils.normalize_text(input_text, preserve_special_chars=False)
                self.assertEqual(result, expected)

    def test_normalize_text_with_special_chars_preserved(self):
        """Prueba normalización preservando caracteres especiales."""
        test_cases = [
            ("archivo_2024.txt", "archivo_2024.txt"),
            ("user@email.com", "user@email.com"),
            ("path/to/file", "path/to/file"),
            ("item#123", "item#123"),
            ("test-case", "test-case"),
            ("value_underscore", "value_underscore"),
        ]

        for input_text, expected in test_cases:
            with self.subTest(input=input_text):
                result = utils.normalize_text(input_text, preserve_special_chars=True)
                # Los caracteres especiales se preservan
                self.assertTrue(any(char in result for char in '#-_/.@' if char in input_text.lower()))

    def test_normalize_text_type_conversion(self):
        """Prueba conversión de tipos a string."""
        test_cases = [
            (123, "123"),
            (45.67, "4567"),  # El punto se elimina sin preserve_special_chars
            (True, "true"),
            (None, "none"),
            ([], ""),  # Lista vacía se convierte a "[]" y luego se limpian los brackets
        ]

        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = utils.normalize_text(input_val)
                self.assertEqual(result, expected)

    def test_normalize_text_unicode_handling(self):
        """Prueba manejo de caracteres Unicode."""
        test_cases = [
            ("こんにちは", "konnichiha"),  # Japonés
            ("Здравствуйте", "zdravstvuite"),  # Ruso
            ("مرحبا", "mrhb"),  # Árabe
            ("🚀 Rocket", "rocket"),  # Emoji
        ]

        for input_text, expected in test_cases:
            with self.subTest(input=input_text):
                result = utils.normalize_text(input_text)
                # Verificar que no hay caracteres no ASCII
                self.assertTrue(all(ord(c) < 128 for c in result))

    def test_normalize_text_series_with_none(self):
        """Prueba normalización de Series con valores None."""
        series = pd.Series(["TEXTO 1", "Texto 2", None, 123, ""])
        result = utils.normalize_text_series(series)

        # None se convierte a "None" como string y luego se normaliza
        expected = pd.Series(["texto 1", "texto 2", "none", "123", ""])
        assert_series_equal(result, expected)

    def test_normalize_text_series_large_chunking(self):
        """Prueba procesamiento por chunks de series grandes."""
        # Crear serie grande
        large_series = pd.Series([f"Texto Número {i}" for i in range(15000)])

        # Procesar con chunks pequeños para forzar múltiples chunks
        result = utils.normalize_text_series(large_series, chunk_size=5000)

        self.assertEqual(len(result), 15000)
        # Verificar algunos valores específicos
        self.assertEqual(result.iloc[0], "texto numero 0")
        self.assertEqual(result.iloc[14999], "texto numero 14999")

    def test_safe_normalize_error_handling(self):
        """Prueba manejo de errores en _safe_normalize."""
        # Crear un objeto que cause error al convertir a string
        class BadObject:
            def __str__(self):
                raise ValueError("Cannot convert")

        series = pd.Series([BadObject(), "texto normal"])

        with patch('utils.logger.warning') as mock_warning:
            result = utils.normalize_text_series(series)
            # Debería registrar warning pero no fallar
            self.assertTrue(mock_warning.called)

# ============================================================================
# PRUEBAS DE CONVERSIÓN NUMÉRICA - REFINADAS
# ============================================================================

class TestNumericConversion(unittest.TestCase):
    """Pruebas refinadas para funciones de conversión numérica."""

    def test_parse_number_basic_types(self):
        """Prueba conversión de tipos básicos."""
        test_cases = [
            # Strings numéricos básicos
            ("123", 123.0),
            ("123.45", 123.45),
            ("-67.89", -67.89),
            ("1e3", 1000.0),
            ("1.23E-4", 0.000123),
            # Tipos numéricos directos
            (123, 123.0),
            (45.67, 45.67),
            (np.int32(100), 100.0),
            (np.float64(200.5), 200.5),
        ]

        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = utils.parse_number(input_val)
                self.assertAlmostEqual(result, expected, places=7)

    def test_parse_number_special_values(self):
        """Prueba manejo de valores especiales."""
        test_cases = [
            (None, 0.0),
            ("", 0.0),
            ("   ", 0.0),
            (float('nan'), 0.0),
            (np.nan, 0.0),
            (float('inf'), 0.0),
            (float('-inf'), 0.0),
            (pd.NA, 0.0),  # Pandas NA
        ]

        for input_val, expected in test_cases:
            with self.subTest(input=str(input_val)):
                result = utils.parse_number(input_val)
                self.assertEqual(result, expected)

    def test_parse_number_currency_formats(self):
        """Prueba conversión con símbolos de moneda."""
        test_cases = [
            ("$1,234.56", 1234.56),
            ("€ 1.234,56", 1234.56),  # Formato europeo
            ("£999.99", 999.99),
            ("¥10,000", 10000.0),
            ("50%", 50.0),
            ("$ 100 ", 100.0),
        ]

        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = utils.parse_number(input_val)
                self.assertAlmostEqual(result, expected, places=2)

    def test_parse_number_decimal_separator_detection(self):
        """Prueba detección automática del separador decimal."""
        test_cases = [
            ("1,234.56", 1234.56),    # Formato US: coma para miles
            ("1.234,56", 1234.56),    # Formato EU: punto para miles
            ("1234,56", 1234.56),     # Solo coma decimal
            ("1234.56", 1234.56),     # Solo punto decimal
            ("1,000", 1000.0),        # Coma para miles (3 dígitos después)
            ("1,50", 1.50),           # Coma decimal (menos de 3 dígitos)
        ]

        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = utils.parse_number(input_val, decimal_separator="auto")
                self.assertAlmostEqual(result, expected, places=2)

    def test_parse_number_explicit_separator(self):
        """Prueba con separador decimal explícito."""
        # Forzar interpretación con coma como decimal
        result = utils.parse_number("1.234,56", decimal_separator="comma")
        self.assertAlmostEqual(result, 1234.56)

        # Forzar interpretación con punto como decimal
        result = utils.parse_number("1,234.56", decimal_separator="dot")
        self.assertAlmostEqual(result, 1234.56)

    def test_parse_number_edge_cases(self):
        """Prueba casos límite en conversión numérica."""
        test_cases = [
            ("abc", 0.0),                    # Texto no numérico
            ("12.34.56.78", 12345678.0),     # Múltiples puntos
            ("1.234.567,89", 1234567.89),    # Formato EU complejo
            ("-", 0.0),                      # Solo signo
            ("+", 0.0),                      # Solo signo positivo
            ("1.2.3", 12.3),                 # Múltiples puntos, último como decimal
        ]

        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = utils.parse_number(input_val)
                self.assertAlmostEqual(result, expected, places=2)

    def test_detect_decimal_separator(self):
        """Prueba función interna de detección de separador."""
        test_cases = [
            ("1,234.56", "dot"),      # Formato US claro
            ("1.234,56", "comma"),    # Formato EU claro
            ("1234.56", "dot"),       # Solo punto
            ("1234,56", "comma"),     # Solo coma (menos de 3 dígitos después)
            ("1,000", "dot"),         # Coma con 3 dígitos = miles
            ("1,50", "comma"),        # Coma con 2 dígitos = decimal
        ]

        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = utils._detect_decimal_separator(input_val)
                self.assertEqual(result, expected)

# ============================================================================
# PRUEBAS DE CÓDIGOS APU - REFINADAS
# ============================================================================

class TestAPUCode(unittest.TestCase):
    """Pruebas refinadas para funciones de códigos APU."""

    def setUp(self):
        utils.clean_apu_code.cache_clear()

    def test_clean_apu_code_basic_cleaning(self):
        """Prueba limpieza básica de códigos APU."""
        test_cases = [
            ("apu-001", "APU-001"),
            ("  APU.002  ", "APU.002"),
            ("APU#003$%", "APU003"),  # Caracteres especiales removidos
            ("apu_004-.", "APU_004"),  # Guión y punto al final removidos
            ("APU.005.", "APU.005"),   # Punto al final removido
            ("APU-006-", "APU-006"),   # Guión al final removido
            ("test_code_123", "TEST_CODE_123"),  # Underscore preservado
        ]

        for input_code, expected in test_cases:
            with self.subTest(input=input_code):
                result = utils.clean_apu_code(input_code, validate_format=False)
                self.assertEqual(result, expected)

    def test_clean_apu_code_validation_valid(self):
        """Prueba validación de códigos APU válidos."""
        valid_codes = [
            "APU001",
            "APU-002",
            "APU.003",
            "APU_004",
            "A1B2C3",
            "123ABC",
            "ITEM_2024.01",
        ]

        for code in valid_codes:
            with self.subTest(code=code):
                result = utils.clean_apu_code(code, validate_format=True)
                self.assertIsNotNone(result)
                self.assertIsInstance(result, str)
                self.assertGreaterEqual(len(result), 2)

    def test_clean_apu_code_validation_invalid(self):
        """Prueba validación de códigos APU inválidos."""
        # Código vacío
        with self.assertRaises(ValueError) as context:
            utils.clean_apu_code("", validate_format=True)
        self.assertIn("no puede estar vacío", str(context.exception))

        # Código demasiado corto
        with self.assertRaises(ValueError) as context:
            utils.clean_apu_code("A", validate_format=True)
        self.assertIn("demasiado corto", str(context.exception))

        # Código sin números (debería generar warning pero no error)
        with patch('utils.logger.warning') as mock_warning:
            result = utils.clean_apu_code("ABC", validate_format=True)
            self.assertEqual(result, "ABC")
            mock_warning.assert_called()

    def test_clean_apu_code_type_conversion(self):
        """Prueba conversión de tipos en códigos APU."""
        test_cases = [
            (123, "123"),
            (45.67, "45.67"),
            (True, "TRUE"),  # Boolean a string
        ]

        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = utils.clean_apu_code(input_val, validate_format=False)
                self.assertEqual(result, expected)

    def test_clean_apu_code_type_error(self):
        """Prueba error de tipo cuando no se puede convertir."""
        class UnconvertibleObject:
            def __str__(self):
                raise Exception("Cannot convert")

        with self.assertRaises(TypeError):
            utils.clean_apu_code(UnconvertibleObject())

    def test_clean_apu_code_cache_functionality(self):
        """Prueba funcionamiento del cache LRU."""
        code = "test-123"

        # Primera llamada
        result1 = utils.clean_apu_code(code)

        # Segunda llamada (debe venir del cache)
        result2 = utils.clean_apu_code(code)

        self.assertEqual(result1, result2)
        self.assertGreater(utils.clean_apu_code.cache_info().hits, 0)

# ============================================================================
# PRUEBAS DE NORMALIZACIÓN DE UNIDADES - REFINADAS
# ============================================================================

class TestUnitNormalization(unittest.TestCase):
    """Pruebas refinadas para normalización de unidades."""

    def setUp(self):
        utils.normalize_unit.cache_clear()

    def test_normalize_unit_standard_units(self):
        """Prueba normalización de unidades estándar."""
        # Probar todas las unidades en STANDARD_UNITS
        for unit in utils.STANDARD_UNITS:
            with self.subTest(unit=unit):
                result = utils.normalize_unit(unit.lower())
                self.assertEqual(result, unit)

    def test_normalize_unit_mapping(self):
        """Prueba mapeo completo de unidades equivalentes."""
        # Probar todo el mapeo definido en UNIT_MAPPING
        for input_unit, expected_unit in utils.UNIT_MAPPING.items():
            with self.subTest(input=input_unit):
                result = utils.normalize_unit(input_unit)
                self.assertEqual(result, expected_unit)

    def test_normalize_unit_case_insensitive(self):
        """Prueba insensibilidad a mayúsculas."""
        test_cases = [
            ("m", "M"),
            ("M", "M"),
            ("kg", "KG"),
            ("Kg", "KG"),
            ("KG", "KG"),
            ("hora", "HR"),
            ("HORA", "HR"),
        ]

        for input_unit, expected in test_cases:
            with self.subTest(input=input_unit):
                result = utils.normalize_unit(input_unit)
                self.assertEqual(result, expected)

    def test_normalize_unit_with_extra_chars(self):
        """Prueba normalización con caracteres extra."""
        test_cases = [
            ("M.", "M"),      # Con punto
            ("KG.", "KG"),    # Con punto
            ("M2.", "M2"),    # M2 con punto
            ("-HR-", "HR"),   # Con guiones
            ("(UND)", "UND"), # Con paréntesis
        ]

        for input_unit, expected in test_cases:
            with self.subTest(input=input_unit):
                result = utils.normalize_unit(input_unit)
                self.assertEqual(result, expected)

    def test_normalize_unit_invalid_returns_und(self):
        """Prueba que unidades inválidas retornan 'UND'."""
        test_cases = [
            "",
            None,
            "INVALID_UNIT_XYZ",
            "@#$%",
            "123",
            "ABC",
            12345,  # Número
        ]

        for input_unit in test_cases:
            with self.subTest(input=input_unit):
                result = utils.normalize_unit(input_unit)
                self.assertEqual(result, 'UND')

    def test_normalize_unit_logging(self):
        """Prueba que se registren logs para unidades no reconocidas."""
        with patch('utils.logger.debug') as mock_debug:
            # Unidad no reconocida y no trivial
            utils.normalize_unit("METRO_CUBICO_ESPECIAL")
            mock_debug.assert_called()

            # Reset mock
            mock_debug.reset_mock()

            # Unidades triviales no deben generar log
            utils.normalize_unit("")
            utils.normalize_unit("UND")
            utils.normalize_unit("X")  # Solo 1 caracter
            mock_debug.assert_not_called()

# ============================================================================
# PRUEBAS DE LECTURA DE ARCHIVOS - REFINADAS
# ============================================================================

class TestFileReading(unittest.TestCase):
    """Pruebas refinadas para funciones de lectura de archivos."""

    def setUp(self):
        """Crear archivos temporales para pruebas."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Crear CSV de prueba
        self.csv_file = self.temp_path / "test.csv"
        df = TestDataFactory.create_sample_dataframe(20)
        df.to_csv(self.csv_file, index=False)

        # Crear CSV con diferente separador
        self.csv_semicolon = self.temp_path / "test_semicolon.csv"
        df.to_csv(self.csv_semicolon, index=False, sep=';')

        # Crear Excel de prueba
        self.excel_file = self.temp_path / "test.xlsx"
        df.to_excel(self.excel_file, index=False)

        # Crear CSV con encoding diferente
        self.csv_latin1 = self.temp_path / "test_latin1.csv"
        df_latin = pd.DataFrame({'texto': ['año', 'niño', 'café']})
        df_latin.to_csv(self.csv_latin1, index=False, encoding='latin1')

    def tearDown(self):
        """Limpiar archivos temporales."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_safe_read_csv_basic(self):
        """Prueba lectura básica de CSV."""
        df = utils.safe_read_dataframe(self.csv_file)

        self.assertFalse(df.empty)
        self.assertEqual(len(df), 20)
        self.assertIn('codigo', df.columns)
        self.assertIn('descripcion', df.columns)

    def test_safe_read_csv_with_parameters(self):
        """Prueba lectura de CSV con parámetros específicos."""
        df = utils.safe_read_dataframe(
            self.csv_file,
            header=0,
            nrows=5,
            usecols=['codigo', 'descripcion']
        )

        self.assertEqual(len(df), 5)
        self.assertEqual(list(df.columns), ['codigo', 'descripcion'])

    def test_safe_read_excel(self):
        """Prueba lectura de archivo Excel."""
        df = utils.safe_read_dataframe(self.excel_file)

        self.assertFalse(df.empty)
        self.assertEqual(len(df), 20)
        self.assertIn('codigo', df.columns)

    def test_safe_read_nonexistent_file(self):
        """Prueba manejo de archivo inexistente."""
        nonexistent = self.temp_path / "nonexistent.csv"

        with patch('utils.logger.error') as mock_error:
            df = utils.safe_read_dataframe(nonexistent)

            self.assertTrue(df.empty)
            mock_error.assert_called()
            self.assertIn("Archivo no encontrado", mock_error.call_args[0][0])

    def test_safe_read_unsupported_format(self):
        """Prueba manejo de formato no soportado."""
        txt_file = self.temp_path / "test.txt"
        txt_file.write_text("contenido de texto")

        with patch('utils.logger.error') as mock_error:
            df = utils.safe_read_dataframe(txt_file)

            self.assertTrue(df.empty)
            mock_error.assert_called()
            self.assertIn("Formato no soportado", mock_error.call_args[0][0])

    def test_detect_file_encoding(self):
        """Prueba detección de encoding de archivo."""
        # UTF-8 (archivo CSV normal)
        encoding = utils._detect_file_encoding(self.csv_file)
        self.assertEqual(encoding, 'utf-8')

        # Latin1
        encoding = utils._detect_file_encoding(self.csv_latin1)
        self.assertIn(encoding, ['utf-8', 'latin1'])  # Puede detectar cualquiera

    def test_detect_csv_separator(self):
        """Prueba detección de separador CSV."""
        # Coma
        sep = utils._detect_csv_separator(self.csv_file, 'utf-8')
        self.assertEqual(sep, ',')

        # Punto y coma
        sep = utils._detect_csv_separator(self.csv_semicolon, 'utf-8')
        self.assertEqual(sep, ';')

    def test_read_csv_robust_with_bad_lines(self):
        """Prueba lectura robusta con líneas malformadas."""
        # Crear CSV con líneas problemáticas
        bad_csv = self.temp_path / "bad.csv"
        content = "col1,col2,col3\n1,2,3\n4,5\n6,7,8,9,10\n11,12,13"
        bad_csv.write_text(content)

        # Debe leer sin errores, saltando líneas malas
        df = utils.safe_read_dataframe(bad_csv)
        self.assertFalse(df.empty)
        # Solo debe tener las líneas válidas
        self.assertGreaterEqual(len(df), 2)  # Al menos las líneas válidas

    def test_path_conversion(self):
        """Prueba conversión de string a Path."""
        # Con string
        df = utils.safe_read_dataframe(str(self.csv_file))
        self.assertFalse(df.empty)

        # Con Path object
        df = utils.safe_read_dataframe(self.csv_file)
        self.assertFalse(df.empty)

# ============================================================================
# PRUEBAS DE VALIDACIÓN - REFINADAS
# ============================================================================

class TestValidation(unittest.TestCase):
    """Pruebas refinadas para funciones de validación."""

    def test_validate_numeric_value_types(self):
        """Prueba validación de diferentes tipos numéricos."""
        valid_types = [
            100,              # int
            100.5,            # float
            np.int32(100),    # numpy int32
            np.int64(100),    # numpy int64
            np.float32(100),  # numpy float32
            np.float64(100),  # numpy float64
        ]

        for value in valid_types:
            with self.subTest(type=type(value).__name__):
                is_valid, error = utils.validate_numeric_value(value)
                self.assertTrue(is_valid)
                self.assertEqual(error, "")

    def test_validate_numeric_value_invalid_types(self):
        """Prueba rechazo de tipos no numéricos."""
        invalid_types = [
            "100",      # string
            [100],      # list
            {'v': 100}, # dict
            None,       # None (tipo incorrecto, no NA)
        ]

        for value in invalid_types:
            with self.subTest(type=type(value).__name__):
                is_valid, error = utils.validate_numeric_value(value)
                self.assertFalse(is_valid)
                self.assertIn("debe ser numérico", error)

    def test_validate_numeric_value_special_values(self):
        """Prueba validación de valores especiales."""
        # NaN
        is_valid, error = utils.validate_numeric_value(float('nan'))
        self.assertFalse(is_valid)
        self.assertIn("no puede ser nulo", error)

        # Infinito sin permitir
        is_valid, error = utils.validate_numeric_value(float('inf'), allow_inf=False)
        self.assertFalse(is_valid)
        self.assertIn("no puede ser infinito", error)

        # Infinito permitido
        is_valid, error = utils.validate_numeric_value(float('inf'), allow_inf=True)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")

    def test_validate_numeric_value_constraints(self):
        """Prueba restricciones de validación."""
        test_cases = [
            # (valor, kwargs, debería_ser_válido)
            (0, {'allow_zero': False}, False),
            (0, {'allow_zero': True}, True),
            (-5, {'allow_negative': False}, False),
            (-5, {'allow_negative': True}, True),
            (150, {'max_value': 100}, False),
            (50, {'max_value': 100}, True),
            (5, {'min_value': 10}, False),
            (15, {'min_value': 10}, True),
        ]

        for value, kwargs, should_be_valid in test_cases:
            with self.subTest(value=value, kwargs=kwargs):
                is_valid, error = utils.validate_numeric_value(value, **kwargs)
                self.assertEqual(is_valid, should_be_valid)
                if not should_be_valid:
                    self.assertNotEqual(error, "")

    def test_validate_numeric_value_field_name(self):
        """Prueba personalización del nombre del campo en mensajes."""
        is_valid, error = utils.validate_numeric_value(
            -5,
            field_name="cantidad",
            allow_negative=False
        )

        self.assertFalse(is_valid)
        self.assertIn("cantidad", error)

    def test_validate_series_with_mask(self):
        """Prueba validación de series retornando máscara."""
        series = pd.Series([1, 2, 0, -1, None, 100, float('inf')])

        mask = utils.validate_series(
            series,
            return_mask=True,
            allow_zero=False,
            allow_negative=False,
            allow_inf=False
        )

        expected = pd.Series([True, True, False, False, False, True, False])
        assert_series_equal(mask, expected)

    def test_validate_series_with_details(self):
        """Prueba validación de series con DataFrame detallado."""
        series = pd.Series([10, -5, 0])

        df_result = utils.validate_series(
            series,
            return_mask=False,
            allow_negative=False,
            allow_zero=False
        )

        # Verificar estructura del DataFrame
        self.assertEqual(len(df_result), 3)
        self.assertIn('value', df_result.columns)
        self.assertIn('is_valid', df_result.columns)
        self.assertIn('error_message', df_result.columns)

        # Verificar resultados específicos
        self.assertTrue(df_result.iloc[0]['is_valid'])   # 10 es válido
        self.assertFalse(df_result.iloc[1]['is_valid'])  # -5 no es válido
        self.assertFalse(df_result.iloc[2]['is_valid'])  # 0 no es válido

    def test_validate_empty_series(self):
        """Prueba validación de serie vacía."""
        empty_series = pd.Series([])

        # Con máscara
        mask = utils.validate_series(empty_series, return_mask=True)
        self.assertTrue(mask.empty)

        # Con detalles
        df_result = utils.validate_series(empty_series, return_mask=False)
        self.assertTrue(df_result.empty)

# ============================================================================
# PRUEBAS DE ANÁLISIS Y DETECCIÓN - REFINADAS
# ============================================================================

class TestAnalysisDetection(unittest.TestCase):
    """Pruebas refinadas para funciones de análisis y detección."""

    def test_create_apu_signature_default_fields(self):
        """Prueba creación de firma con campos por defecto."""
        apu_data = {
            'CODIGO_APU': 'APU-001',
            'DESCRIPCION_APU': 'Excavación Manual',
            'UNIDAD_APU': 'M3',
            'PRECIO': 100.0,  # Este campo no se incluye por defecto
        }

        signature = utils.create_apu_signature(apu_data)

        # La firma debe incluir los campos normalizados
        self.assertIn('apu001', signature)  # Código normalizado sin guión
        self.assertIn('excavacion manual', signature)
        self.assertIn('m3', signature)
        self.assertNotIn('100', signature)  # Precio no incluido

        # Verificar formato de separación
        parts = signature.split('|')
        self.assertEqual(len(parts), 3)

    def test_create_apu_signature_custom_fields(self):
        """Prueba creación de firma con campos personalizados."""
        apu_data = {
            'campo1': 'Valor 1',
            'campo2': 123.45,
            'campo3': None,
            'campo4': '',
        }

        signature = utils.create_apu_signature(
            apu_data,
            key_fields=['campo1', 'campo2', 'campo3', 'campo4']
        )

        # Solo incluye campos con valor
        self.assertIn('valor 1', signature)
        self.assertIn('123.45', signature)
        self.assertNotIn('none', signature)  # None no se incluye

        parts = signature.split('|')
        self.assertEqual(len(parts), 2)  # Solo 2 campos válidos

    def test_create_apu_signature_empty_data(self):
        """Prueba creación de firma con datos vacíos."""
        # Diccionario vacío
        signature = utils.create_apu_signature({})
        self.assertEqual(signature, 'empty_signature')

        # Campos no existentes
        signature = utils.create_apu_signature(
            {'otro_campo': 'valor'},
            key_fields=['campo_inexistente']
        )
        self.assertEqual(signature, 'empty_signature')

    def test_detect_outliers_iqr_method(self):
        """Prueba detección de outliers con método IQR."""
        series = TestDataFactory.create_numeric_series_with_outliers()

        outliers, bounds = utils.detect_outliers(
            series,
            method='iqr',
            threshold=1.5,
            return_bounds=True
        )

        # Verificar estructura
        self.assertEqual(len(outliers), len(series))
        self.assertIn('Q1', bounds)
        self.assertIn('Q3', bounds)
        self.assertIn('IQR', bounds)
        self.assertIn('lower_bound', bounds)
        self.assertIn('upper_bound', bounds)

        # Verificar que detecta outliers extremos
        self.assertTrue(outliers.iloc[-3])  # 1000 es outlier
        self.assertTrue(outliers.iloc[-5])  # -100 es outlier

    def test_detect_outliers_zscore_method(self):
        """Prueba detección de outliers con z-score."""
        series = TestDataFactory.create_numeric_series_with_outliers()

        outliers, bounds = utils.detect_outliers(
            series,
            method='zscore',
            threshold=3,
            return_bounds=True
        )

        # Verificar bounds
        self.assertIn('mean', bounds)
        self.assertIn('std', bounds)
        self.assertIn('threshold', bounds)

        # Los valores más extremos deben ser outliers
        self.assertTrue(outliers.iloc[-1])  # 999
        self.assertTrue(outliers.iloc[-3])  # 1000

    def test_detect_outliers_modified_zscore(self):
        """Prueba detección con modified z-score (robusto)."""
        # Serie con outliers y valores constantes
        series = pd.Series([50] * 20 + [100, 200, 300])

        outliers, bounds = utils.detect_outliers(
            series,
            method='modified_zscore',
            threshold=3.5,
            return_bounds=True
        )

        self.assertIn('median', bounds)
        self.assertIn('mad', bounds)

        # Los valores extremos deben ser outliers
        self.assertTrue(outliers.iloc[-1])  # 300
        self.assertTrue(outliers.iloc[-2])  # 200

    def test_detect_outliers_empty_series(self):
        """Prueba detección con serie vacía."""
        empty_series = pd.Series([])

        outliers = utils.detect_outliers(empty_series)
        self.assertTrue(outliers.empty)
        self.assertEqual(outliers.dtype, bool)

        outliers, bounds = utils.detect_outliers(empty_series, return_bounds=True)
        self.assertTrue(outliers.empty)
        self.assertEqual(bounds, {})

    def test_detect_outliers_all_nan(self):
        """Prueba detección con serie de solo NaN."""
        nan_series = pd.Series([np.nan, np.nan, np.nan])

        outliers = utils.detect_outliers(nan_series)
        self.assertEqual(len(outliers), 3)
        self.assertFalse(outliers.any())  # Todos False

    def test_detect_outliers_constant_values(self):
        """Prueba detección con valores constantes."""
        constant_series = pd.Series([100] * 50)

        # Con z-score (std = 0)
        outliers = utils.detect_outliers(constant_series, method='zscore')
        self.assertFalse(outliers.any())

        # Con modified z-score (MAD = 0)
        outliers = utils.detect_outliers(constant_series, method='modified_zscore')
        self.assertFalse(outliers.any())

    def test_detect_outliers_invalid_method(self):
        """Prueba error con método no soportado."""
        series = pd.Series([1, 2, 3, 4, 5])

        with self.assertRaises(ValueError) as context:
            utils.detect_outliers(series, method='invalid_method')

        self.assertIn("Método no soportado", str(context.exception))

# ============================================================================
# PRUEBAS DE MANIPULACIÓN DE DATAFRAMES - REFINADAS
# ============================================================================

class TestDataFrameManipulation(unittest.TestCase):
    """Pruebas refinadas para manipulación de DataFrames."""

    def test_find_and_rename_columns_basic(self):
        """Prueba básica de búsqueda y renombrado."""
        df = pd.DataFrame({
            'cod_apu': [1, 2],
            'descripcion_item': ['a', 'b'],
            'unidad_medida': ['M', 'KG']
        })

        column_map = {
            'CODIGO': ['cod', 'codigo'],
            'DESCRIPCION': ['desc', 'descripcion'],
            'UNIDAD': ['unidad']
        }

        result = utils.find_and_rename_columns(df, column_map)

        # Verificar columnas renombradas
        self.assertIn('CODIGO', result.columns)
        self.assertIn('DESCRIPCION', result.columns)
        self.assertIn('UNIDAD', result.columns)

        # Verificar que los datos se mantienen
        self.assertEqual(result['CODIGO'].tolist(), [1, 2])

    def test_find_and_rename_columns_partial_match(self):
        """Prueba coincidencia parcial en nombres."""
        df = pd.DataFrame({
            'codigo_del_apu': [1],
            'descripcion_completa': ['texto'],
            'otra_columna': ['valor']
        })

        column_map = {
            'CODIGO': ['codigo'],
            'DESCRIPCION': ['descripcion']
        }

        result = utils.find_and_rename_columns(df, column_map)

        # Debe encontrar por coincidencia parcial
        self.assertIn('CODIGO', result.columns)
        self.assertIn('DESCRIPCION', result.columns)
        self.assertIn('otra_columna', result.columns)  # No mapeada

    def test_find_and_rename_columns_case_sensitivity(self):
        """Prueba sensibilidad a mayúsculas."""
        df = pd.DataFrame({
            'CODIGO': [1],
            'Descripcion': ['texto']
        })

        column_map = {
            'codigo_nuevo': ['codigo'],
            'descripcion_nueva': ['descripcion']
        }

        # Sin sensibilidad (encuentra las columnas)
        result1 = utils.find_and_rename_columns(df, column_map, case_sensitive=False)
        self.assertIn('codigo_nuevo', result1.columns)
        self.assertIn('descripcion_nueva', result1.columns)

        # Con sensibilidad (no encuentra las columnas)
        result2 = utils.find_and_rename_columns(df, column_map, case_sensitive=True)
        self.assertNotIn('codigo_nuevo', result2.columns)
        self.assertIn('CODIGO', result2.columns)  # Original se mantiene

    def test_find_and_rename_columns_conflict_warning(self):
        """Prueba warning cuando hay conflictos de mapeo."""
        df = pd.DataFrame({
            'codigo_1': [1],
            'codigo_2': [2],
            'descripcion': ['texto']
        })

        column_map = {
            'CODIGO': ['codigo']  # Ambas columnas coinciden
        }

        with patch('utils.logger.warning') as mock_warning:
            result = utils.find_and_rename_columns(df, column_map)

            # Solo una columna debe ser mapeada
            self.assertEqual(sum(1 for col in result.columns if col == 'CODIGO'), 1)
            # Debe haber generado warning
            mock_warning.assert_called()

    def test_find_and_rename_columns_empty_dataframe(self):
        """Prueba con DataFrame vacío."""
        df = pd.DataFrame()
        column_map = {'CODIGO': ['cod']}

        result = utils.find_and_rename_columns(df, column_map)
        self.assertTrue(result.empty)

    def test_find_and_rename_unmapped_logging(self):
        """Prueba logging de columnas no mapeadas."""
        df = pd.DataFrame({
            'col1': [1],
            'col2': [2],
            'col3': [3],
            'col4': [4]
        })

        column_map = {
            'MAPPED': ['col1']
        }

        with patch('utils.logger.debug') as mock_debug:
            utils.find_and_rename_columns(df, column_map)

            # Debe loggear las columnas no mapeadas (3 columnas)
            mock_debug.assert_called()
            call_args = str(mock_debug.call_args)
            self.assertIn('col2', call_args)

# ============================================================================
# PRUEBAS DE SERIALIZACIÓN - REFINADAS
# ============================================================================

class TestSerialization(unittest.TestCase):
    """Pruebas refinadas para funciones de serialización."""

    def test_sanitize_basic_types(self):
        """Prueba sanitización de tipos básicos."""
        data = {
            'int': 42,
            'float': 3.14,
            'str': 'texto',
            'bool': True,
            'none': None,
            'list': [1, 2, 3],
            'dict': {'key': 'value'}
        }

        result = utils.sanitize_for_json(data)

        # Verificar tipos nativos
        self.assertIsInstance(result['int'], int)
        self.assertIsInstance(result['float'], float)
        self.assertIsInstance(result['str'], str)
        self.assertIsInstance(result['bool'], bool)
        self.assertIsNone(result['none'])

        # Debe ser serializable
        json_str = json.dumps(result)
        self.assertIsNotNone(json_str)

    def test_sanitize_numpy_types(self):
        """Prueba sanitización de tipos NumPy."""
        data = {
            'np_int32': np.int32(42),
            'np_int64': np.int64(100),
            'np_float32': np.float32(3.14),
            'np_float64': np.float64(2.71),
            'np_bool': np.bool_(True),
            'np_array': np.array([1, 2, 3]),
            'np_nan': np.nan,
            'np_inf': np.inf,
            'np_ninf': -np.inf,
        }

        result = utils.sanitize_for_json(data)

        # Verificar conversión a tipos nativos
        self.assertIsInstance(result['np_int32'], int)
        self.assertIsInstance(result['np_int64'], int)
        self.assertIsInstance(result['np_float32'], float)
        self.assertIsInstance(result['np_float64'], float)
        self.assertIsInstance(result['np_bool'], bool)
        self.assertIsInstance(result['np_array'], list)
        self.assertIsNone(result['np_nan'])
        self.assertIsNone(result['np_inf'])
        self.assertIsNone(result['np_ninf'])

        # Verificar valores
        self.assertEqual(result['np_int32'], 42)
        self.assertEqual(result['np_array'], [1, 2, 3])

    def test_sanitize_pandas_objects(self):
        """Prueba sanitización de objetos Pandas."""
        data = {
            'series': pd.Series([1, 2, 3, np.nan]),
            'dataframe': pd.DataFrame({'col1': [4, 5], 'col2': [6, 7]}),
            'pd_na': pd.NA,
            'pd_nat': pd.NaT,
        }

        result = utils.sanitize_for_json(data)

        # Series -> lista
        self.assertIsInstance(result['series'], list)
        self.assertEqual(result['series'][:3], [1, 2, 3])
        self.assertIsNone(result['series'][3])  # NaN -> None

        # DataFrame -> lista de diccionarios
        self.assertIsInstance(result['dataframe'], list)
        self.assertEqual(len(result['dataframe']), 2)
        self.assertEqual(result['dataframe'][0], {'col1': 4, 'col2': 6})

        # Valores especiales de Pandas
        self.assertIsNone(result['pd_na'])
        self.assertIsNone(result['pd_nat'])

    def test_sanitize_nested_structures(self):
        """Prueba sanitización de estructuras anidadas complejas."""
        data = {
            'level1': {
                'level2': {
                    'series': pd.Series([1, np.nan, 3]),
                    'array': np.array([4, 5, 6]),
                    'level3': {
                        'value': np.int32(100)
                    }
                },
                'list': [np.float64(1.1), np.float64(2.2)]
            }
        }

        result = utils.sanitize_for_json(data)

        # Verificar estructura anidada
        self.assertIsInstance(result['level1']['level2']['series'], list)
        self.assertIsNone(result['level1']['level2']['series'][1])
        self.assertEqual(result['level1']['level2']['level3']['value'], 100)
        self.assertIsInstance(result['level1']['list'][0], float)

    def test_sanitize_max_depth_limit(self):
        """Prueba límite de profundidad de recursión."""
        # Crear estructura profundamente anidada
        def create_nested(depth):
            if depth == 0:
                return "value"
            return {"next": create_nested(depth - 1)}

        # Profundidad dentro del límite
        data = create_nested(50)
        result = utils.sanitize_for_json(data, max_depth=60)
        self.assertIsNotNone(result)

        # Profundidad excede el límite
        data = create_nested(150)
        with self.assertRaises(RecursionError):
            utils.sanitize_for_json(data, max_depth=100)

    def test_sanitize_datetime_objects(self):
        """Prueba sanitización de objetos datetime."""
        from datetime import date, datetime

        data = {
            'datetime': datetime(2024, 1, 1, 12, 0, 0),
            'date': date(2024, 1, 1),
        }

        result = utils.sanitize_for_json(data)

        # Fechas deben convertirse a ISO format
        self.assertIsInstance(result['datetime'], str)
        self.assertIn('2024-01-01', result['datetime'])
        self.assertIsInstance(result['date'], str)

    def test_sanitize_object_with_dict(self):
        """Prueba sanitización de objetos con __dict__."""
        class CustomObject:
            def __init__(self):
                self.value = np.int32(42)
                self.text = "hello"

        data = {'obj': CustomObject()}
        result = utils.sanitize_for_json(data)

        # Objeto convertido a diccionario
        self.assertIsInstance(result['obj'], dict)
        self.assertEqual(result['obj']['value'], 42)
        self.assertEqual(result['obj']['text'], "hello")

# ============================================================================
# PRUEBAS DE FUNCIONES ADICIONALES - REFINADAS
# ============================================================================

class TestAdditionalFunctions(unittest.TestCase):
    """Pruebas refinadas para funciones adicionales."""

    def test_calculate_statistics_complete(self):
        """Prueba cálculo completo de estadísticas."""
        series = pd.Series([1, 2, 3, 4, 5, None, 7, 8, 9, 10])
        stats = utils.calculate_statistics(series)

        # Verificar todas las estadísticas
        self.assertEqual(stats['count'], 9)
        self.assertAlmostEqual(stats['mean'], 5.444, places=2)
        self.assertAlmostEqual(stats['std'], 3.2, places=1)
        self.assertEqual(stats['min'], 1.0)
        self.assertEqual(stats['max'], 10.0)
        self.assertEqual(stats['median'], 5.0)
        self.assertEqual(stats['q1'], 2.5)
        self.assertEqual(stats['q3'], 8.5)
        self.assertEqual(stats['null_count'], 1)
        self.assertEqual(stats['null_percentage'], 10.0)

    def test_calculate_statistics_empty_series(self):
        """Prueba estadísticas con serie vacía."""
        empty_series = pd.Series([])
        stats = utils.calculate_statistics(empty_series)

        self.assertEqual(stats['count'], 0)
        self.assertIsNone(stats['mean'])
        self.assertIsNone(stats['std'])
        self.assertIsNone(stats['min'])
        self.assertIsNone(stats['max'])
        self.assertIsNone(stats['median'])

    def test_calculate_statistics_all_nulls(self):
        """Prueba estadísticas con todos valores nulos."""
        null_series = pd.Series([None, np.nan, pd.NA])
        stats = utils.calculate_statistics(null_series)

        self.assertEqual(stats['count'], 0)
        self.assertIsNone(stats['mean'])
        # null_percentage debe manejarse correctamente
        self.assertIn('null_percentage', stats)

    def test_batch_process_dataframe_small(self):
        """Prueba procesamiento de DataFrame pequeño."""
        df = TestDataFactory.create_sample_dataframe(10)

        def process_func(batch_df):
            batch_df['doubled'] = batch_df['cantidad'] * 2
            return batch_df

        result = utils.batch_process_dataframe(df, process_func, batch_size=100)

        # No debe dividir en lotes
        self.assertEqual(len(result), 10)
        self.assertIn('doubled', result.columns)
        self.assertTrue((result['doubled'] == result['cantidad'] * 2).all())

    def test_batch_process_dataframe_large(self):
        """Prueba procesamiento por lotes de DataFrame grande."""
        df = TestDataFactory.create_sample_dataframe(2500)

        def process_func(batch_df, multiplier=3):
            batch_df['new_col'] = batch_df['cantidad'] * multiplier
            return batch_df

        result = utils.batch_process_dataframe(
            df,
            process_func,
            batch_size=500,
            multiplier=3
        )

        # Verificar resultado
        self.assertEqual(len(result), 2500)
        self.assertIn('new_col', result.columns)
        self.assertTrue((result['new_col'] == result['cantidad'] * 3).all())

        # Verificar que se mantiene el índice correcto
        self.assertEqual(result.index[0], 0)
        self.assertEqual(result.index[-1], 2499)

    def test_batch_process_preserves_dtypes(self):
        """Prueba que el procesamiento por lotes preserva tipos de datos."""
        df = pd.DataFrame({
            'int_col': range(100),
            'float_col': np.random.random(100),
            'str_col': ['text'] * 100,
            'bool_col': [True, False] * 50
        })

        # Función que no modifica tipos
        def identity_func(batch_df):
            return batch_df

        result = utils.batch_process_dataframe(df, identity_func, batch_size=25)

        # Verificar que los tipos se preservan
        self.assertEqual(result['int_col'].dtype, df['int_col'].dtype)
        self.assertEqual(result['float_col'].dtype, df['float_col'].dtype)
        self.assertEqual(result['str_col'].dtype, df['str_col'].dtype)
        self.assertEqual(result['bool_col'].dtype, df['bool_col'].dtype)

# ============================================================================
# PRUEBAS DE INTEGRACIÓN Y END-TO-END
# ============================================================================

class TestIntegrationE2E(unittest.TestCase):
    """Pruebas de integración end-to-end."""

    def test_complete_apu_processing_pipeline(self):
        """Prueba pipeline completo de procesamiento APU."""
        # Datos de entrada simulando un APU real
        raw_apu_data = {
            'CODIGO_APU': '  APU-2024.001  ',
            'DESCRIPCION_APU': 'Excavación Manual en Tierra Común',
            'UNIDAD_APU': 'METROS CUBICOS',
            'CANTIDAD': '1.234,56',  # Formato EU
            'PRECIO': '$2,500.00',   # Con símbolo de moneda
            'RENDIMIENTO': '8 horas',
        }

        # 1. Limpiar código APU
        clean_code = utils.clean_apu_code(raw_apu_data['CODIGO_APU'])
        self.assertEqual(clean_code, 'APU-2024.001')

        # 2. Normalizar descripción
        norm_desc = utils.normalize_text(raw_apu_data['DESCRIPCION_APU'])
        self.assertEqual(norm_desc, 'excavacion manual en tierra comun')

        # 3. Normalizar unidad
        norm_unit = utils.normalize_unit(raw_apu_data['UNIDAD_APU'])
        self.assertEqual(norm_unit, 'M3')

        # 4. Parsear números
        cantidad = utils.parse_number(raw_apu_data['CANTIDAD'])
        precio = utils.parse_number(raw_apu_data['PRECIO'])
        self.assertAlmostEqual(cantidad, 1234.56, places=2)
        self.assertAlmostEqual(precio, 2500.00, places=2)

        # 5. Crear firma única
        processed_apu = {
            'CODIGO_APU': clean_code,
            'DESCRIPCION_APU': norm_desc,
            'UNIDAD_APU': norm_unit,
            'CANTIDAD': cantidad,
            'PRECIO': precio
        }

        signature = utils.create_apu_signature(processed_apu)
        self.assertIsNotNone(signature)
        self.assertIn('apu2024001', signature)  # Código normalizado en firma

        # 6. Validar valores numéricos
        is_valid_cantidad, _ = utils.validate_numeric_value(
            cantidad,
            field_name="cantidad",
            min_value=0,
            max_value=10000
        )
        self.assertTrue(is_valid_cantidad)

        # 7. Sanitizar para JSON
        json_safe = utils.sanitize_for_json(processed_apu)
        json_str = json.dumps(json_safe)
        self.assertIsNotNone(json_str)

        # Verificar que se puede deserializar
        recovered = json.loads(json_str)
        self.assertEqual(recovered['CODIGO_APU'], clean_code)

    def test_dataframe_batch_processing_with_outliers(self):
        """Prueba procesamiento completo con detección de outliers."""
        # Crear DataFrame con datos mixtos
        np.random.seed(42)
        n_rows = 1000

        df = pd.DataFrame({
            'codigo': [f'APU{i:04d}' for i in range(n_rows)],
            'descripcion': [f'Ítem número {i}' for i in range(n_rows)],
            'unidad': np.random.choice(['metros', 'kilogramos', 'horas', 'unidades'], n_rows),
            'cantidad': np.concatenate([
                np.random.normal(100, 10, n_rows-5),
                [1000, 2000, 0.001, -50, 5000]  # Outliers
            ]),
            'precio': np.random.lognormal(5, 1, n_rows)
        })

        # 1. Normalizar texto en lotes
        df['descripcion_norm'] = utils.normalize_text_series(
            df['descripcion'],
            chunk_size=200
        )

        # 2. Limpiar códigos APU
        df['codigo_limpio'] = df['codigo'].apply(
            lambda x: utils.clean_apu_code(x, validate_format=False)
        )

        # 3. Normalizar unidades
        df['unidad_norm'] = df['unidad'].apply(utils.normalize_unit)

        # 4. Detectar outliers en cantidad
        outliers_mask = utils.detect_outliers(
            df['cantidad'],
            method='iqr',
            threshold=1.5
        )

        # Verificar outliers detectados
        self.assertTrue(outliers_mask.iloc[-1])  # 5000 es outlier
        self.assertTrue(outliers_mask.iloc[-4])  # -50 es outlier

        # 5. Validar serie de precios
        validation_results = utils.validate_series(
            df['precio'],
            return_mask=False,
            min_value=0,
            allow_negative=False
        )

        self.assertEqual(len(validation_results), n_rows)
        self.assertTrue(validation_results['is_valid'].all())  # lognormal siempre positivo

        # 6. Calcular estadísticas
        stats = utils.calculate_statistics(df['cantidad'])

        self.assertIn('mean', stats)
        self.assertIn('median', stats)
        self.assertGreater(stats['count'], 0)

        # 7. Crear firma para cada APU
        def create_apu_dict(row):
            return {
                'CODIGO_APU': row['codigo_limpio'],
                'DESCRIPCION_APU': row['descripcion_norm'],
                'UNIDAD_APU': row['unidad_norm']
            }

        df['firma'] = df.apply(
            lambda row: utils.create_apu_signature(create_apu_dict(row)),
            axis=1
        )

        # Verificar unicidad de firmas (deberían ser únicas)
        self.assertEqual(len(df['firma'].unique()), n_rows)

# ============================================================================
# SUITE DE PRUEBAS PRINCIPAL
# ============================================================================

def create_test_suite():
    """Crea y retorna la suite completa de pruebas."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Lista completa de clases de prueba
    test_classes = [
        TestTextNormalization,
        TestNumericConversion,
        TestAPUCode,
        TestUnitNormalization,
        TestFileReading,
        TestValidation,
        TestAnalysisDetection,
        TestDataFrameManipulation,
        TestSerialization,
        TestAdditionalFunctions,
        TestIntegrationE2E,
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    return suite

# ============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================================

if __name__ == '__main__':
    # Configurar logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_utils.log', mode='w')
        ]
    )

    # Ejecutar pruebas
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)

    # Imprimir resumen detallado
    print("\n" + "="*80)
    print("📊 RESUMEN DE PRUEBAS")
    print("="*80)
    print(f"✅ Pruebas ejecutadas: {result.testsRun}")
    print(f"✅ Exitosas: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Fallos: {len(result.failures)}")
    print(f"⚠️  Errores: {len(result.errors)}")
    print(f"⏭️  Omitidas: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n🎉 ¡TODAS LAS PRUEBAS PASARON EXITOSAMENTE!")
    else:
        print("\n❗ Algunas pruebas fallaron. Revisa los detalles arriba.")

    # Calcular cobertura si está disponible
    try:
        import coverage
        print("\n" + "="*80)
        print("📈 ANÁLISIS DE COBERTURA")
        print("="*80)

        cov = coverage.Coverage()
        cov.start()

        # Re-ejecutar con cobertura
        suite = create_test_suite()
        runner = unittest.TextTestRunner(verbosity=0)
        runner.run(suite)

        cov.stop()
        cov.save()

        # Generar reporte
        print("\n📋 Reporte de Cobertura:")
        cov.report(include=['utils.py'])

        # Generar HTML si es posible
        cov.html_report(directory='htmlcov', include=['utils.py'])
        print("\n📁 Reporte HTML generado en: ./htmlcov/index.html")

    except ImportError:
        print("\n💡 Tip: Instala 'coverage' para análisis de cobertura:")
        print("   pip install coverage")
        print("   coverage run test_utils.py")
        print("   coverage report")
        print("   coverage html")

    # Exit code
    sys.exit(0 if result.wasSuccessful() else 1)
