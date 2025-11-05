import os
import sys
import unittest
import tempfile
import pandas as pd
import numpy as np

# Añadir el directorio raíz del proyecto al sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.utils import (
    normalize_text,
    normalize_text_series,
    parse_number,
    clean_apu_code,
    normalize_unit,
    safe_read_dataframe,
    validate_numeric_value,
    validate_series,
    create_apu_signature,
    detect_outliers,
    find_and_rename_columns,
    sanitize_for_json,
)


class TestUtils(unittest.TestCase):
    """
    Pruebas unitarias robustas para las funciones utilitarias mejoradas.

    Cubre todas las funciones mejoradas de utils.py con casos edge,
    validaciones exhaustivas y compatibilidad con la nueva arquitectura.
    """

    def test_normalize_text_comprehensive(self):
        """Prueba exhaustiva de la normalización de texto."""
        test_cases = [
            # (input, expected_output, preserve_special_chars)
            ("Hola Mundo", "hola mundo", False),
            ("Café con AÇÃO", "cafe con acao", False),
            ("  espacios   múltiples  ", "espacios multiples", False),
            ("a,b.c-d_e@f#g", "a b c d e f g", False),
            ("TUBERÍA PVC 3/4\"", "tuberia pvc 3 4", False),
            ("Precio: $1,000.00", "precio 1 000 00", False),
            # Con preservación de caracteres especiales
            ("a,b.c-d_e@f/g", "a b c-d e@f g", True),
            ("TUBERÍA PVC 3/4\"", "tuberia pvc 3/4", True),
            ("Precio: $1,000.00", "precio 1,000.00", True),
        ]

        for input_text, expected, preserve_chars in test_cases:
            with self.subTest(f"normalize_text: '{input_text}'"):
                result = normalize_text(input_text, preserve_special_chars=preserve_chars)
                self.assertEqual(result, expected)

    def test_normalize_text_edge_cases(self):
        """Prueba casos edge en la normalización de texto."""
        edge_cases = [
            ("", ""),
            ("   ", ""),
            (None, ""),
            ("123", "123"),
            ("A\nB\tC", "a b c"),
            ("Ñoño", "nono"),
            ("M2 M3", "m2 m3"),
        ]

        for input_text, expected in edge_cases:
            with self.subTest(f"normalize_text edge: '{input_text}'"):
                result = normalize_text(input_text)
                self.assertEqual(result, expected)

    def test_normalize_text_series_robust(self):
        """Prueba robusta de normalización de series de texto."""
        # Serie normal
        series = pd.Series(["Hola Mundo", "Café", "  espacios  ", "TUBERÍA PVC"])
        normalized = normalize_text_series(series)
        expected = pd.Series(["hola mundo", "cafe", "espacios", "tuberia pvc"])
        pd.testing.assert_series_equal(normalized, expected)

        # Serie con valores nulos y mixed types
        mixed_series = pd.Series(["Texto", 123, None, "M2", np.nan])
        normalized_mixed = normalize_text_series(mixed_series)
        self.assertEqual(len(normalized_mixed), 5)
        self.assertTrue(all(isinstance(x, str) for x in normalized_mixed))

    def test_parse_number_comprehensive(self):
        """Prueba exhaustiva del parseo de números con diferentes formatos."""
        number_cases = [
            # (input, expected, decimal_separator)
            ("1.000,00", 1000.0, "auto"),
            ("1,000.00", 1000.0, "auto"),
            ("1.500.000,50", 1500000.5, "auto"),
            ("0,75", 0.75, "auto"),
            ("0.75", 0.75, "auto"),
            ("100", 100.0, "auto"),
            ("1 000", 1000.0, "auto"),
            ("$1,000.00", 1000.0, "auto"),
            ("1.000,00€", 1000.0, "auto"),
            ("80.000", 80000.0, "auto"),  # Punto como separador de miles
            ("0,125", 0.125, "auto"),
            # Casos edge
            ("", 0.0, "auto"),
            ("abc", 0.0, "auto"),
            (None, 0.0, "auto"),
            ("1.2.3", 0.0, "auto"),  # Múltiples puntos inconsistentes
            # Forzando separadores
            ("1.000,00", 1000.0, "comma"),
            ("1,000.00", 1000.0, "dot"),
        ]

        for input_val, expected, separator in number_cases:
            with self.subTest(f"parse_number: '{input_val}'"):
                result = parse_number(input_val, decimal_separator=separator)
                self.assertAlmostEqual(result, expected, places=2,
                                    msg=f"Failed for input: {input_val}")

    def test_parse_number_scientific_notation(self):
        """Prueba el parseo de notación científica."""
        sci_cases = [
            ("1e3", 1000.0),
            ("1.5e-2", 0.015),
            ("2E+3", 2000.0),
        ]

        for input_val, expected in sci_cases:
            with self.subTest(f"parse_number scientific: '{input_val}'"):
                result = parse_number(input_val)
                self.assertAlmostEqual(result, expected, places=6)

    def test_clean_apu_code_robust(self):
        """Prueba robusta de limpieza de códigos APU."""
        code_cases = [
            # (input, expected, validate_format)
            ("1.1", "1.1", True),
            ("1.1.1", "1.1.1", True),
            ("1.1-1", "1.1-1", True),
            (" 1.1 ", "1.1", True),
            ("1.1.", "1.1", True),
            ("1.1-", "1.1", True),
            ("APU-001", "APU-001", True),
            ("ITEM 1.1", "ITEM1.1", True),
            # Casos que deben fallar con validación
            ("", "", False),  # Vacío sin validación
            ("A", "A", False),  # Muy corto sin validación
        ]

        for input_code, expected, validate in code_cases:
            with self.subTest(f"clean_apu_code: '{input_code}'"):
                if validate and not expected:  # Casos que deben fallar
                    with self.assertRaises(ValueError):
                        clean_apu_code(input_code, validate_format=validate)
                else:
                    result = clean_apu_code(input_code, validate_format=validate)
                    self.assertEqual(result, expected)

    def test_clean_apu_code_validation_errors(self):
        """Prueba los errores de validación en códigos APU."""
        invalid_codes = [
            ("", "Código APU no puede estar vacío"),
            ("A", "Código APU demasiado corto"),
            ("   ", "Código APU no puede estar vacío"),
        ]

        for code, expected_error in invalid_codes:
            with self.subTest(f"clean_apu_code validation: '{code}'"):
                with self.assertRaises(ValueError) as context:
                    clean_apu_code(code, validate_format=True)
                self.assertIn(expected_error, str(context.exception))

    def test_normalize_unit_comprehensive(self):
        """Prueba exhaustiva de normalización de unidades."""
        unit_cases = [
            # (input, expected)
            ("und", "UND"),
            ("m2", "M2"),
            ("metro cubico", "M3"),
            ("metro cúbico", "M3"),
            ("jornal", "JOR"),
            ("unidad", "UND"),
            ("dias", "DIA"),
            ("días", "DIA"),
            ("hora", "HR"),
            ("horas", "HR"),
            ("kilogramo", "KG"),
            ("tonelada", "TON"),
            ("galon", "GAL"),
            ("litro", "L"),
            ("viaje", "VIAJE"),
            ("viajes", "VIAJE"),
            # Casos edge
            ("", "UND"),
            (None, "UND"),
            ("invalid_unit", "UND"),
            ("UNIDADES", "UND"),
            ("METROS", "M"),
            ("MTS", "M"),
        ]

        for input_unit, expected in unit_cases:
            with self.subTest(f"normalize_unit: '{input_unit}'"):
                result = normalize_unit(input_unit)
                self.assertEqual(result, expected)

    def test_safe_read_dataframe_robust(self):
        """Prueba robusta de lectura segura de DataFrames."""
        # Crear archivos temporales para pruebas
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\n1,2\n3,4")
            csv_path = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as f:
            df_temp = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
            df_temp.to_excel(f.name, index=False)
            excel_path = f.name

        try:
            # Probar CSV
            df_csv = safe_read_dataframe(csv_path)
            self.assertEqual(len(df_csv), 2)
            self.assertListEqual(df_csv.columns.tolist(), ["col1", "col2"])

            # Probar Excel
            df_excel = safe_read_dataframe(excel_path)
            self.assertEqual(len(df_excel), 2)
            self.assertIn("A", df_excel.columns)

            # Probar archivo inexistente
            df_nonexistent = safe_read_dataframe("nonexistent_file.csv")
            self.assertTrue(df_nonexistent.empty)

            # Probar formato no soportado
            df_unsupported = safe_read_dataframe("file.txt")
            self.assertTrue(df_unsupported.empty)

        finally:
            # Limpiar archivos temporales
            for path in [csv_path, excel_path]:
                if os.path.exists(path):
                    os.remove(path)

    def test_safe_read_dataframe_encoding_detection(self):
        """Prueba la detección automática de encoding."""
        # Crear archivo con encoding específico
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', encoding='latin1', delete=False) as f:
            f.write("col1,col2\ncafé,niño")
            csv_path = f.name

        try:
            df = safe_read_dataframe(csv_path, encoding="auto")
            self.assertEqual(len(df), 1)
            self.assertIn("café", df["col1"].iloc[0])
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_validate_numeric_value_comprehensive(self):
        """Prueba exhaustiva de validación de valores numéricos."""
        validation_cases = [
            # (value, field_name, min_val, max_val, allow_zero, expected_valid, expected_msg_contains)
            (100, "test", 0, 200, True, True, ""),
            (-1, "test", 0, 200, True, False, "menor"),
            (300, "test", 0, 200, True, False, "mayor"),
            (0, "test", 0, 200, False, False, "cero"),
            (0, "test", 0, 200, True, True, ""),
            (np.inf, "test", 0, 200, True, False, "infinito"),
            (-np.inf, "test", 0, 200, True, False, "infinito"),
            (np.nan, "test", 0, 200, True, False, "nulo"),
            ("not_number", "test", 0, 200, True, False, "numérico"),
            (None, "test", 0, 200, True, False, "numérico"),
        ]

        for value, field, min_v, max_v, allow_zero, expected_valid, expected_msg in validation_cases:
            with self.subTest(f"validate_numeric: {value}"):
                is_valid, message = validate_numeric_value(
                    value, field, min_v, max_v, allow_zero
                )
                self.assertEqual(is_valid, expected_valid)
                if not expected_valid:
                    self.assertIn(expected_msg, message.lower())

    def test_validate_series_robust(self):
        """Prueba robusta de validación de series."""
        series = pd.Series([1, 2, 3, 1000, -5, 0, np.nan])
        validated = validate_series(
            series,
            field_name="test_series",
            min_value=0,
            max_value=100,
            allow_zero=True
        )

        expected = pd.Series([True, True, True, False, False, True, False])
        pd.testing.assert_series_equal(validated, expected)

    def test_create_apu_signature_robust(self):
        """Prueba robusta de creación de firmas APU."""
        apu_cases = [
            {
                "data": {
                    "CODIGO_APU": "1.1",
                    "DESCRIPCION_APU": "Instalación Tubería PVC",
                    "UNIDAD_APU": "ML"
                },
                "expected": "1.1|instalacion tuberia pvc|ml"
            },
            {
                "data": {
                    "CODIGO_APU": "2.1",
                    "DESCRIPCION_APU": "Excavación Manual",
                    "UNIDAD_APU": "M3"
                },
                "expected": "2.1|excavacion manual|m3"
            },
            {
                "data": {
                    "CODIGO_APU": "3.1",
                    "DESCRIPCION_APU": "",  # Descripción vacía
                    "UNIDAD_APU": "UND"
                },
                "expected": "3.1|und"
            },
        ]

        for case in apu_cases:
            with self.subTest(f"create_apu_signature: {case['data']['CODIGO_APU']}"):
                result = create_apu_signature(case["data"])
                self.assertEqual(result, case["expected"])

    def test_detect_outliers_comprehensive(self):
        """Prueba exhaustiva de detección de outliers."""
        # Serie con outliers obvios
        series = pd.Series([1, 2, 3, 4, 5, 100])

        # Método IQR
        outliers_iqr = detect_outliers(series, method="iqr")
        expected_iqr = pd.Series([False, False, False, False, False, True])
        pd.testing.assert_series_equal(outliers_iqr, expected_iqr)

        # Método Z-Score
        outliers_zscore = detect_outliers(series, method="zscore")
        expected_zscore = pd.Series([False, False, False, False, False, True])
        pd.testing.assert_series_equal(outliers_zscore, expected_zscore)

        # Serie sin outliers
        series_no_outliers = pd.Series([1, 2, 3, 4, 5])
        outliers_none = detect_outliers(series_no_outliers, method="iqr")
        self.assertFalse(outliers_none.any())

        # Método no soportado
        with self.assertRaises(ValueError):
            detect_outliers(series, method="unsupported_method")

    def test_find_and_rename_columns_robust(self):
        """Prueba robusta de búsqueda y renombrado de columnas."""
        df = pd.DataFrame({
            "Item": [1, 2],
            "Descripción del ítem": ["A", "B"],
            "Unidad de Medida": ["UND", "M2"],
            "Cantidad Presupuestada": [10, 20],
            "Columna Sin Mapeo": [100, 200]
        })

        column_map = {
            "CODIGO_APU": ["item", "codigo"],
            "DESCRIPCION_APU": ["descripción", "descripcion"],
            "UNIDAD_APU": ["unidad"],
            "CANTIDAD_PRESUPUESTO": ["cantidad", "cant"]
        }

        renamed_df = find_and_rename_columns(df, column_map)

        expected_columns = [
            "CODIGO_APU", "DESCRIPCION_APU", "UNIDAD_APU",
            "CANTIDAD_PRESUPUESTO", "Columna Sin Mapeo"
        ]

        self.assertListEqual(list(renamed_df.columns), expected_columns)

        # Verificar que los datos se mantienen
        self.assertEqual(renamed_df["CODIGO_APU"].iloc[0], 1)
        self.assertEqual(renamed_df["DESCRIPCION_APU"].iloc[0], "A")

    def test_sanitize_for_json_comprehensive(self):
        """Prueba exhaustiva de sanitización para JSON."""
        test_data = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"key": "value"},
            "numpy_int": np.int64(100),
            "numpy_float": np.float64(2.5),
            "numpy_nan": np.nan,
            "numpy_inf": np.inf,
            "pandas_na": pd.NA,
            "nested": {
                "numpy_array": np.array([1, 2, 3]),
                "mixed_list": [1, "two", np.float64(3.0), np.nan]
            }
        }

        sanitized = sanitize_for_json(test_data)

        # Verificar tipos convertidos
        self.assertIsInstance(sanitized["numpy_int"], int)
        self.assertIsInstance(sanitized["numpy_float"], float)
        self.assertIsNone(sanitized["numpy_nan"])
        self.assertIsNone(sanitized["pandas_na"])

        # Verificar estructura mantenida
        self.assertEqual(sanitized["string"], "hello")
        self.assertEqual(sanitized["int"], 42)
        self.assertEqual(sanitized["float"], 3.14)
        self.assertEqual(sanitized["bool"], True)
        self.assertIsNone(sanitized["none"])
        self.assertIsInstance(sanitized["list"], list)
        self.assertIsInstance(sanitized["dict"], dict)

        # Verificar nested structures
        self.assertIsInstance(sanitized["nested"]["mixed_list"], list)
        self.assertEqual(sanitized["nested"]["mixed_list"][0], 1)
        self.assertEqual(sanitized["nested"]["mixed_list"][1], "two")
        self.assertEqual(sanitized["nested"]["mixed_list"][2], 3.0)
        self.assertIsNone(sanitized["nested"]["mixed_list"][3])

    def test_sanitize_for_json_edge_cases(self):
        """Prueba casos edge en sanitización para JSON."""
        edge_cases = [
            # (input, expected_type_or_value)
            (np.int32(100), int),
            (np.float32(2.5), float),
            (pd.NaT, type(None)),  # Not a Time
            (complex(1, 2), complex),  # No cambia
            ([np.nan, np.inf, -np.inf], list),
        ]

        for input_val, expected in edge_cases:
            with self.subTest(f"sanitize_for_json edge: {type(input_val)}"):
                result = sanitize_for_json(input_val)
                if isinstance(expected, type):
                    self.assertIsInstance(result, expected)
                else:
                    self.assertEqual(result, expected)

    def test_integration_workflow(self):
        """Prueba de flujo de trabajo integrado con múltiples funciones."""
        # Datos de ejemplo similares a los del mundo real
        sample_data = {
            "CODIGO_APU": "1.1",
            "DESCRIPCION_APU": "Instalación de Tubería PVC 3/4\"",
            "UNIDAD_APU": "und",  # Unidad necesita normalización
            "CANTIDAD": "1,5",   # Número con formato europeo
            "PRECIO": "$1.000,50",
            "VALOR_TOTAL": "1.500,75"
        }

        # Aplicar pipeline de procesamiento
        codigo_limpio = clean_apu_code(sample_data["CODIGO_APU"])
        descripcion_normalizada = normalize_text(sample_data["DESCRIPCION_APU"])
        unidad_normalizada = normalize_unit(sample_data["UNIDAD_APU"])
        cantidad_parseada = parse_number(sample_data["CANTIDAD"])
        precio_parseado = parse_number(sample_data["PRECIO"])
        valor_total_parseado = parse_number(sample_data["VALOR_TOTAL"])

        # Crear firma APU
        apu_data = {
            "CODIGO_APU": codigo_limpio,
            "DESCRIPCION_APU": descripcion_normalizada,
            "UNIDAD_APU": unidad_normalizada
        }
        firma = create_apu_signature(apu_data)

        # Validaciones
        cantidad_valida, _ = validate_numeric_value(cantidad_parseada, "cantidad", 0, 1000)
        precio_valido, _ = validate_numeric_value(precio_parseado, "precio", 0, 1000000)

        # Verificar resultados
        self.assertEqual(codigo_limpio, "1.1")
        self.assertEqual(descripcion_normalizada, "instalacion de tuberia pvc 3 4")
        self.assertEqual(unidad_normalizada, "UND")
        self.assertAlmostEqual(cantidad_parseada, 1.5)
        self.assertAlmostEqual(precio_parseado, 1000.5)
        self.assertAlmostEqual(valor_total_parseado, 1500.75)
        self.assertEqual(firma, "1.1|instalacion de tuberia pvc 3 4|und")
        self.assertTrue(cantidad_valida)
        self.assertTrue(precio_valido)


if __name__ == "__main__":
    # Ejecutar pruebas con cobertura completa
    unittest.main(verbosity=2, failfast=False)
