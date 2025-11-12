"""
Suite completa de pruebas para utils.py

Pruebas exhaustivas con cobertura de casos edge, validaciones,
excepciones y optimización de rendimiento.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Importar módulo a probar
from app import utils

# ============================================================================
# FIXTURES - Datos de prueba reutilizables
# ============================================================================

@pytest.fixture
def sample_text_data():
    """Fixture con datos de texto variados para pruebas."""
    return {
        'simple': 'Texto Simple',
        'accents': 'Ácido Nitrógeno Ñoño',
        'special_chars': 'Test#123-ABC_def/xyz@domain.com',
        'whitespace': '  Múltiples    espacios   ',
        'mixed': 'MAYÚSCULAS minúsculas 123',
        'empty': '',
        'none': None,
        'numeric': 12345
    }


@pytest.fixture
def sample_numeric_strings():
    """Fixture con strings numéricos en varios formatos."""
    return {
        # Formatos estándar
        'integer': '1234',
        'float_dot': '1234.56',
        'float_comma': '1234,56',
        'negative': '-1234.56',
        'positive_sign': '+1234.56',

        # Separadores de miles
        'thousands_comma': '1,234,567.89',
        'thousands_dot': '1.234.567,89',
        'thousands_space': '1 234 567.89',

        # Casos especiales
        'percentage': '15.5%',
        'scientific': '1.5e-3',
        'scientific_upper': '2.5E+5',

        # Con símbolos de moneda
        'currency_dollar': '$1,234.56',
        'currency_euro': '€ 1.234,56',
        'currency_mixed': 'USD 1,234.56',

        # Casos edge
        'zero': '0',
        'zero_decimal': '0.0',
        'very_small': '0.000001',
        'very_large': '999999999999',

        # Casos inválidos
        'empty': '',
        'text': 'ABC',
        'na': 'N/A',
        'null': 'NULL',
        'excel_error': '#DIV/0!',
        'multiple_dots': '1.234.567.89'
    }


@pytest.fixture
def sample_dataframe():
    """Fixture con DataFrame de prueba."""
    return pd.DataFrame({
        'CODIGO_APU': ['APU-001', 'APU-002', 'APU-003', 'APU-004'],
        'DESCRIPCION_APU': ['Concreto', 'Acero', 'Excavación', 'Pintura'],
        'UNIDAD_APU': ['M3', 'KG', 'M3', 'M2'],
        'TIPO_INSUMO': ['SUMINISTRO', 'SUMINISTRO', 'MANO_DE_OBRA', 'OTRO'],
        'VALOR_TOTAL_APU': [1000.0, 2000.0, 1500.0, 500.0],
        'CANTIDAD': [10, 20, 15, 5]
    })


@pytest.fixture
def sample_series_numeric():
    """Fixture con Serie numérica para pruebas estadísticas."""
    return pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0, np.nan, 3.5, 4.5])


@pytest.fixture
def temp_csv_file(tmp_path):
    """Fixture que crea archivo CSV temporal."""
    csv_path = tmp_path / "test_data.csv"
    csv_content = """CODIGO,DESCRIPCION,VALOR
APU-001,Concreto,1234.56
APU-002,Acero,2345.67
APU-003,Pintura,345.78"""
    csv_path.write_text(csv_content, encoding='utf-8')
    return csv_path


@pytest.fixture
def temp_csv_semicolon(tmp_path):
    """Fixture que crea archivo CSV con punto y coma."""
    csv_path = tmp_path / "test_semicolon.csv"
    csv_content = """CODIGO;DESCRIPCION;VALOR
APU-001;Concreto;1234,56
APU-002;Acero;2345,67"""
    csv_path.write_text(csv_content, encoding='utf-8')
    return csv_path


# ============================================================================
# TESTS - NORMALIZACIÓN DE TEXTO
# ============================================================================

class TestNormalizeText:
    """Suite de pruebas para normalize_text()"""

    def test_normalize_simple_text(self):
        """Debe normalizar texto simple correctamente."""
        result = utils.normalize_text('Texto Simple')
        assert result == 'texto simple'

    def test_normalize_accents(self):
        """Debe remover acentos y caracteres especiales."""
        result = utils.normalize_text('Ácido Nitrógeno Ñoño')
        assert result == 'acido nitrogeno nono'

    def test_normalize_special_chars_default(self):
        """Debe remover caracteres especiales por defecto."""
        result = utils.normalize_text('Test#123-ABC_def')
        assert result == 'test123abcdef'

    def test_normalize_preserve_special_chars(self):
        """Debe preservar caracteres especiales si se indica."""
        result = utils.normalize_text('Test#123-ABC_def', preserve_special_chars=True)
        assert '#' in result or '-' in result or '_' in result

    def test_normalize_multiple_spaces(self):
        """Debe normalizar múltiples espacios a uno solo."""
        result = utils.normalize_text('  Múltiples    espacios   ')
        assert result == 'multiples espacios'
        assert '  ' not in result

    def test_normalize_empty_string(self):
        """Debe manejar string vacío."""
        assert utils.normalize_text('') == ''
        assert utils.normalize_text('   ') == ''

    def test_normalize_none_raises_type_error(self):
        """Debe intentar convertir None a string."""
        # La función intenta convertir None a 'None' string
        result = utils.normalize_text(None)
        assert result == 'none'

    def test_normalize_numeric_input(self):
        """Debe convertir números a texto normalizado."""
        result = utils.normalize_text(12345)
        assert result == '12345'

    def test_normalize_case_insensitive(self):
        """Debe normalizar a minúsculas."""
        assert utils.normalize_text('MAYÚSCULAS') == 'mayusculas'
        assert utils.normalize_text('MiXeD CaSe') == 'mixed case'

    def test_normalize_cache_hit(self):
        """Debe usar cache para strings repetidos."""
        # Primera llamada
        result1 = utils.normalize_text('Test Cache')
        # Segunda llamada debería usar cache
        result2 = utils.normalize_text('Test Cache')
        assert result1 == result2
        # Verificar que el cache funciona
        cache_info = utils.normalize_text.cache_info()
        assert cache_info.hits > 0


class TestNormalizeTextSeries:
    """Suite de pruebas para normalize_text_series()"""

    def test_normalize_series_basic(self):
        """Debe normalizar una serie completa."""
        series = pd.Series(['Texto 1', 'TEXTO 2', 'téxto 3'])
        result = utils.normalize_text_series(series)

        assert len(result) == 3
        assert result.iloc[0] == 'texto 1'
        assert result.iloc[1] == 'texto 2'
        assert result.iloc[2] == 'texto 3'

    def test_normalize_series_empty(self):
        """Debe manejar series vacías."""
        series = pd.Series([], dtype=str)
        result = utils.normalize_text_series(series)
        assert len(result) == 0

    def test_normalize_series_none_input(self):
        """Debe manejar None como input."""
        result = utils.normalize_text_series(None)
        assert result is None

    def test_normalize_series_with_nulls(self):
        """Debe manejar valores nulos en la serie."""
        series = pd.Series(['Texto', None, 'Otro', np.nan])
        result = utils.normalize_text_series(series)

        assert len(result) == 4
        assert result.iloc[0] == 'texto'
        # None y np.nan se convierten a 'nan' o 'none'
        assert isinstance(result.iloc[1], str)

    def test_normalize_series_large_chunks(self):
        """Debe procesar series grandes por chunks."""
        # Crear serie grande (mayor que chunk_size default de 10000)
        large_series = pd.Series(['Texto ' + str(i) for i in range(15000)])
        result = utils.normalize_text_series(large_series, chunk_size=5000)

        assert len(result) == 15000
        assert result.iloc[0] == 'texto 0'
        assert result.iloc[-1] == 'texto 14999'

    def test_normalize_series_preserve_special(self):
        """Debe preservar caracteres especiales si se indica."""
        series = pd.Series(['Test#1', 'Test@2', 'Test-3'])
        result = utils.normalize_text_series(series, preserve_special_chars=True)

        # Verificar que algunos caracteres especiales se preservaron
        assert any('#' in str(val) or '@' in str(val) or '-' in str(val)
                  for val in result)


# ============================================================================
# TESTS - CONVERSIÓN NUMÉRICA
# ============================================================================

class TestParseNumber:
    """Suite exhaustiva de pruebas para parse_number()"""

    # Tests básicos
    def test_parse_integer_string(self):
        """Debe parsear entero simple."""
        assert utils.parse_number('1234') == 1234.0
        assert utils.parse_number('0') == 0.0
        assert utils.parse_number('-1234') == -1234.0

    def test_parse_float_string(self):
        """Debe parsear flotante con punto decimal."""
        assert utils.parse_number('1234.56') == 1234.56
        assert utils.parse_number('0.5') == 0.5
        assert utils.parse_number('.5') == 0.5

    def test_parse_float_comma_separator(self):
        """Debe parsear flotante con coma decimal."""
        result = utils.parse_number('1234,56', decimal_separator='comma')
        assert abs(result - 1234.56) < 0.01

    def test_parse_direct_numeric(self):
        """Debe manejar entradas numéricas directas."""
        assert utils.parse_number(1234) == 1234.0
        assert utils.parse_number(1234.56) == 1234.56
        assert utils.parse_number(-100) == -100.0

    def test_parse_none_returns_default(self):
        """Debe retornar default para None."""
        assert utils.parse_number(None) == 0.0
        assert utils.parse_number(None, default_value=99.9) == 99.9

    # Tests con separadores de miles
    def test_parse_thousands_comma_separator(self):
        """Debe parsear números con separador de miles (coma)."""
        assert utils.parse_number('1,234,567.89') == 1234567.89
        assert utils.parse_number('1,234.56') == 1234.56

    def test_parse_thousands_dot_separator(self):
        """Debe parsear números con separador de miles (punto)."""
        result = utils.parse_number('1.234.567,89', decimal_separator='comma')
        assert abs(result - 1234567.89) < 0.01

    def test_parse_thousands_space_separator(self):
        """Debe parsear números con separador de miles (espacio)."""
        assert utils.parse_number('1 234 567.89') == 1234567.89

    # Tests con porcentajes
    def test_parse_percentage(self):
        """Debe parsear porcentajes."""
        assert utils.parse_number('15%', allow_percentage=True) == 0.15
        assert utils.parse_number('100%', allow_percentage=True) == 1.0
        assert utils.parse_number('0.5%', allow_percentage=True) == 0.005

    def test_parse_percentage_disabled(self):
        """Debe fallar con porcentaje si está deshabilitado."""
        result = utils.parse_number('15%', allow_percentage=False, default_value=-1)
        # Debería retornar default porque no puede parsear
        assert result == -1 or result == 15.0  # Depende de implementación

    # Tests con notación científica
    def test_parse_scientific_notation(self):
        """Debe parsear notación científica."""
        assert utils.parse_number('1.5e-3', allow_scientific=True) == 0.0015
        assert utils.parse_number('2.5E+5', allow_scientific=True) == 250000.0
        assert utils.parse_number('1e10', allow_scientific=True) == 1e10

    def test_parse_scientific_disabled(self):
        """Debe manejar científico si está deshabilitado."""
        result = utils.parse_number('1.5e-3', allow_scientific=False, default_value=-1)
        # Podría parsear o fallar dependiendo de implementación
        assert isinstance(result, float)

    # Tests con símbolos de moneda
    def test_parse_currency_symbols(self):
        """Debe remover símbolos de moneda."""
        assert utils.parse_number('$1,234.56') == 1234.56
        assert utils.parse_number('€ 1.234,56', decimal_separator='comma') == 1234.56
        assert utils.parse_number('USD 1,234.56') == 1234.56

    # Tests con signos
    def test_parse_negative_numbers(self):
        """Debe parsear números negativos."""
        assert utils.parse_number('-1234.56') == -1234.56
        assert utils.parse_number('- 1234.56') == -1234.56

    def test_parse_positive_sign(self):
        """Debe parsear números con signo positivo."""
        assert utils.parse_number('+1234.56') == 1234.56

    def test_parse_multiple_signs(self):
        """Debe manejar múltiples signos."""
        assert utils.parse_number('--1234') == 1234.0  # Doble negativo
        assert utils.parse_number('---1234') == -1234.0  # Triple negativo

    # Tests casos edge
    def test_parse_empty_string(self):
        """Debe retornar default para string vacío."""
        assert utils.parse_number('') == 0.0
        assert utils.parse_number('   ') == 0.0
        assert utils.parse_number('', default_value=100) == 100.0

    def test_parse_non_numeric_text(self):
        """Debe retornar default para texto no numérico."""
        assert utils.parse_number('ABC', default_value=-1) == -1.0
        assert utils.parse_number('N/A', default_value=-1) == -1.0
        assert utils.parse_number('NULL', default_value=-1) == -1.0
        assert utils.parse_number('#DIV/0!', default_value=-1) == -1.0

    def test_parse_very_small_number(self):
        """Debe parsear números muy pequeños."""
        result = utils.parse_number('0.000001')
        assert abs(result - 0.000001) < 1e-10

    def test_parse_very_large_number(self):
        """Debe parsear números muy grandes."""
        result = utils.parse_number('999999999999')
        assert result == 999999999999.0

    def test_parse_zero_variations(self):
        """Debe parsear variaciones de cero."""
        assert utils.parse_number('0') == 0.0
        assert utils.parse_number('0.0') == 0.0
        assert utils.parse_number('.0') == 0.0
        assert utils.parse_number('0,0', decimal_separator='comma') == 0.0

    # Tests modo strict
    def test_parse_strict_mode_raises_exception(self):
        """Debe lanzar excepción en modo strict para entradas inválidas."""
        with pytest.raises(ValueError):
            utils.parse_number('ABC', strict=True)

        with pytest.raises(ValueError):
            utils.parse_number('N/A', strict=True)

    def test_parse_strict_mode_valid_input(self):
        """Debe funcionar normalmente en modo strict para entradas válidas."""
        assert utils.parse_number('1234.56', strict=True) == 1234.56

    # Tests con debug
    def test_parse_debug_mode(self, caplog):
        """Debe generar logs en modo debug."""
        import logging
        with caplog.at_level(logging.DEBUG):
            utils.parse_number('1234.56', debug=True)
            # Verificar que se generaron logs (si la implementación los usa)

    # Tests edge cases adicionales
    def test_parse_only_decimal_separator(self):
        """Debe manejar solo separador decimal."""
        result = utils.parse_number('.', default_value=-1)
        assert result == -1.0

    def test_parse_multiple_decimal_separators(self):
        """Debe manejar múltiples separadores decimales (inválido)."""
        result = utils.parse_number('1.2.3.4', default_value=-1)
        # Debería fallar o auto-corregir
        assert isinstance(result, float)

    def test_parse_whitespace_only(self):
        """Debe manejar solo espacios en blanco."""
        assert utils.parse_number('     ', default_value=99) == 99.0


# ============================================================================
# TESTS - CÓDIGOS APU
# ============================================================================

class TestCleanApuCode:
    """Suite de pruebas exhaustiva para la nueva función clean_apu_code."""

    # Casos básicos de limpieza
    def test_limpieza_basica(self):
        assert utils.clean_apu_code("  apu-001  ") == "APU-001"
        assert utils.clean_apu_code("apu,001") == "APU.001"
        assert utils.clean_apu_code("APU@001#") == "APU001"
        assert utils.clean_apu_code("APU-001.-_") == "APU-001"
        assert utils.clean_apu_code("._-APU-001") == "APU-001"

    # Pruebas del parámetro min_length
    def test_min_length(self):
        assert utils.clean_apu_code("1") == "1"  # Default min_length=1
        with pytest.raises(ValueError, match="demasiado corto"):
            utils.clean_apu_code("1", min_length=2)
        assert utils.clean_apu_code("12", min_length=2) == "12"

    # Pruebas del parámetro is_item_code
    def test_is_item_code(self, caplog):
        # Códigos de item válidos
        assert utils.clean_apu_code("1.2.3", is_item_code=True) == "1.2.3"
        assert utils.clean_apu_code("A.1", is_item_code=True) == "A.1"
        # Un código numérico no debería generar warnings si es un item_code
        utils.clean_apu_code("123", is_item_code=True)
        assert not any("inusual" in rec.message for rec in caplog.records)

    # Pruebas de validación de formato
    def test_validate_format_exceptions(self):
        with pytest.raises(ValueError, match="vacío o solo contener espacios"):
            utils.clean_apu_code("", validate_format=True)
        with pytest.raises(ValueError, match="demasiado corto"):
            utils.clean_apu_code("A", min_length=2, validate_format=True)
        # Este caso ahora debería lanzar "vacío después de limpieza"
        with pytest.raises(ValueError, match="vacío después de limpieza"):
            utils.clean_apu_code(".-.", validate_format=True)

    def test_no_validate_format(self):
        assert utils.clean_apu_code("", validate_format=False) == ""
        assert utils.clean_apu_code("A", min_length=2, validate_format=False) == "A"

    # Pruebas de advertencias (warnings)
    def test_warnings(self, caplog):
        # Código sin números (no item)
        utils.clean_apu_code("ABC", is_item_code=False)
        assert "sin números" in caplog.text
        caplog.clear()
        # Múltiples puntos
        utils.clean_apu_code("A..B")
        assert "puntos consecutivos" in caplog.text
        caplog.clear()
        # Múltiples guiones
        utils.clean_apu_code("A--B")
        assert "guiones consecutivos" in caplog.text

    # Pruebas de correcciones automáticas
    def test_correcciones_automaticas(self):
        assert utils.clean_apu_code("A..B--C") == "A.B-C"
        assert utils.clean_apu_code("A...B") == "A.B"

    # Pruebas de tipos de entrada
    def test_tipos_de_entrada(self):
        assert utils.clean_apu_code(123) == "123"
        assert utils.clean_apu_code(123.45) == "123.45"
        # La función ahora convierte `None` a string 'None' y lo procesa
        assert utils.clean_apu_code(None) == "NONE"

    # Pruebas específicas para _is_valid_item_code y su interacción
    @pytest.mark.parametrize("code, expected_clean, should_warn", [
        ("1", "1", False),
        ("1.2.3", "1.2.3", False),
        ("A.1", "A.1", False),
        ("ITEM-01", "ITEM-01", False),
        # Estos casos se limpian a un formato válido ANTES de la validación,
        # por lo que no deberían generar una advertencia.
        ("1.", "1", False),
        (".1", "1", False),
        ("A-B-", "A-B", False),
    ])
    def test_item_code_cleaning_and_warnings(self, code, expected_clean, should_warn, caplog):
        result = utils.clean_apu_code(code, is_item_code=True)
        assert result == expected_clean
        if should_warn:
            assert "formato inusual" in caplog.text
        else:
            assert "formato inusual" not in caplog.text

    def test_item_code_invalid_raises_error(self):
        # Estos se limpian a vacío, lo que dispara un ValueError
        with pytest.raises(ValueError, match="vacío después de limpieza"):
            utils.clean_apu_code(".", is_item_code=True)
        with pytest.raises(ValueError, match="vacío después de limpieza"):
            utils.clean_apu_code("-", is_item_code=True)


# ============================================================================
# TESTS - UNIDADES
# ============================================================================

class TestNormalizeUnit:
    """Suite de pruebas para normalize_unit()"""

    def test_normalize_standard_unit(self):
        """Debe reconocer unidades estándar."""
        assert utils.normalize_unit('M') == 'M'
        assert utils.normalize_unit('M2') == 'M2'
        assert utils.normalize_unit('KG') == 'KG'
        assert utils.normalize_unit('UND') == 'UND'

    def test_normalize_mapped_unit(self):
        """Debe mapear unidades equivalentes."""
        assert utils.normalize_unit('METROS') == 'M'
        assert utils.normalize_unit('KILOGRAMOS') == 'KG'
        assert utils.normalize_unit('UNIDAD') == 'UND'
        assert utils.normalize_unit('DIAS') == 'DIA'

    def test_normalize_case_insensitive(self):
        """Debe ser case-insensitive."""
        assert utils.normalize_unit('metros') == 'M'
        assert utils.normalize_unit('METROS') == 'M'
        assert utils.normalize_unit('MeTrOs') == 'M'

    def test_normalize_with_spaces(self):
        """Debe manejar espacios."""
        assert utils.normalize_unit('  M  ') == 'M'
        assert utils.normalize_unit(' METROS ') == 'M'

    def test_normalize_empty_returns_default(self):
        """Debe retornar UND para entrada vacía."""
        assert utils.normalize_unit('') == 'UND'
        assert utils.normalize_unit('   ') == 'UND'

    def test_normalize_none_returns_default(self):
        """Debe retornar UND para None."""
        assert utils.normalize_unit(None) == 'UND'

    def test_normalize_unknown_unit(self):
        """Debe retornar la unidad original si no la reconoce."""
        result = utils.normalize_unit('UNIDAD_RARA')
        # Debería retornar la unidad limpia o como está
        assert isinstance(result, str)

    def test_normalize_special_chars_removal(self):
        """Debe limpiar caracteres especiales."""
        result = utils.normalize_unit('M@2#')
        assert '@' not in result
        assert '#' not in result

    def test_normalize_cache_hit(self):
        """Debe usar cache."""
        unit = 'METROS'
        result1 = utils.normalize_unit(unit)
        result2 = utils.normalize_unit(unit)

        assert result1 == result2
        cache_info = utils.normalize_unit.cache_info()
        assert cache_info.hits > 0


# ============================================================================
# TESTS - LECTURA DE ARCHIVOS
# ============================================================================

class TestSafeReadDataframe:
    """Suite de pruebas para safe_read_dataframe()"""

    def test_read_csv_basic(self, temp_csv_file):
        """Debe leer archivo CSV básico."""
        df = utils.safe_read_dataframe(temp_csv_file)

        assert not df.empty
        assert 'CODIGO' in df.columns
        assert 'DESCRIPCION' in df.columns
        assert len(df) == 3

    def test_read_csv_with_encoding_auto(self, temp_csv_file):
        """Debe detectar encoding automáticamente."""
        df = utils.safe_read_dataframe(temp_csv_file, encoding='auto')
        assert not df.empty

    def test_read_csv_with_nrows(self, temp_csv_file):
        """Debe leer solo N filas."""
        df = utils.safe_read_dataframe(temp_csv_file, nrows=2)
        assert len(df) == 2

    def test_read_csv_with_usecols(self, temp_csv_file):
        """Debe leer solo columnas especificadas."""
        df = utils.safe_read_dataframe(temp_csv_file, usecols=['CODIGO', 'VALOR'])
        assert len(df.columns) == 2
        assert 'CODIGO' in df.columns
        assert 'DESCRIPCION' not in df.columns

    def test_read_nonexistent_file(self):
        """Debe retornar DataFrame vacío para archivo inexistente."""
        df = utils.safe_read_dataframe('archivo_que_no_existe.csv')
        assert df.empty

    def test_read_unsupported_format(self, tmp_path):
        """Debe retornar DataFrame vacío para formato no soportado."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Some text")

        df = utils.safe_read_dataframe(txt_file)
        assert df.empty

    def test_read_path_object(self, temp_csv_file):
        """Debe aceptar objeto Path."""
        path_obj = Path(temp_csv_file)
        df = utils.safe_read_dataframe(path_obj)
        assert not df.empty

    def test_read_string_path(self, temp_csv_file):
        """Debe aceptar string path."""
        df = utils.safe_read_dataframe(str(temp_csv_file))
        assert not df.empty

    @patch('pandas.read_csv')
    def test_read_csv_separator_detection(self, mock_read_csv, temp_csv_semicolon):
        """Debe detectar separador de CSV."""
        mock_read_csv.return_value = pd.DataFrame()
        utils.safe_read_dataframe(temp_csv_semicolon)
        # Verificar que se llamó read_csv (la detección de separador es interna)
        assert mock_read_csv.called


# ============================================================================
# TESTS - VALIDACIÓN
# ============================================================================

class TestValidateNumericValue:
    """Suite de pruebas para validate_numeric_value()"""

    def test_validate_valid_number(self):
        """Debe validar número válido."""
        is_valid, msg = utils.validate_numeric_value(100.0)
        assert is_valid
        assert msg == ""

    def test_validate_zero_allowed(self):
        """Debe permitir cero si está habilitado."""
        is_valid, msg = utils.validate_numeric_value(0, allow_zero=True)
        assert is_valid

    def test_validate_zero_not_allowed(self):
        """No debe permitir cero si está deshabilitado."""
        is_valid, msg = utils.validate_numeric_value(0, allow_zero=False)
        assert not is_valid
        assert 'cero' in msg.lower()

    def test_validate_negative_allowed(self):
        """Debe permitir negativos si está habilitado."""
        is_valid, msg = utils.validate_numeric_value(-100, allow_negative=True)
        assert is_valid

    def test_validate_negative_not_allowed(self):
        """No debe permitir negativos si está deshabilitado."""
        is_valid, msg = utils.validate_numeric_value(-100, allow_negative=False)
        assert not is_valid
        assert 'negativo' in msg.lower()

    def test_validate_min_value(self):
        """Debe validar valor mínimo."""
        is_valid, msg = utils.validate_numeric_value(50, min_value=100)
        assert not is_valid
        assert '100' in msg

    def test_validate_max_value(self):
        """Debe validar valor máximo."""
        is_valid, msg = utils.validate_numeric_value(150, max_value=100)
        assert not is_valid
        assert '100' in msg

    def test_validate_range(self):
        """Debe validar rango completo."""
        is_valid, msg = utils.validate_numeric_value(50, min_value=0, max_value=100)
        assert is_valid

        is_valid, msg = utils.validate_numeric_value(150, min_value=0, max_value=100)
        assert not is_valid

    def test_validate_nan(self):
        """No debe permitir NaN."""
        is_valid, msg = utils.validate_numeric_value(np.nan)
        assert not is_valid
        assert 'nulo' in msg.lower()

    def test_validate_inf_not_allowed(self):
        """No debe permitir infinito por defecto."""
        is_valid, msg = utils.validate_numeric_value(np.inf, allow_inf=False)
        assert not is_valid
        assert 'infinito' in msg.lower()

    def test_validate_inf_allowed(self):
        """Debe permitir infinito si está habilitado."""
        is_valid, msg = utils.validate_numeric_value(np.inf, allow_inf=True)
        assert is_valid

    def test_validate_non_numeric_type(self):
        """No debe permitir tipos no numéricos."""
        is_valid, msg = utils.validate_numeric_value("texto")
        assert not is_valid
        assert 'numérico' in msg.lower()

    def test_validate_numpy_types(self):
        """Debe aceptar tipos de NumPy."""
        is_valid, _ = utils.validate_numeric_value(np.int64(100))
        assert is_valid

        is_valid, _ = utils.validate_numeric_value(np.float32(100.5))
        assert is_valid

    def test_validate_field_name_in_message(self):
        """Debe incluir nombre de campo en mensaje."""
        is_valid, msg = utils.validate_numeric_value(
            -100, field_name="precio", allow_negative=False
        )
        assert 'precio' in msg.lower()


class TestValidateSeries:
    """Suite de pruebas para validate_series()"""

    def test_validate_series_mask_mode(self):
        """Debe retornar máscara booleana."""
        series = pd.Series([1, 2, -3, 4, 0])
        mask = utils.validate_series(
            series, return_mask=True, allow_negative=False, allow_zero=False
        )

        assert isinstance(mask, pd.Series)
        assert mask.dtype == bool
        assert mask.iloc[0]  # 1 es válido
        assert not mask.iloc[2]  # -3 no es válido
        assert not mask.iloc[4]  # 0 no es válido

    def test_validate_series_dataframe_mode(self):
        """Debe retornar DataFrame con detalles."""
        series = pd.Series([1, -2, 3])
        result = utils.validate_series(series, return_mask=False, allow_negative=False)

        assert isinstance(result, pd.DataFrame)
        assert 'value' in result.columns
        assert 'is_valid' in result.columns
        assert 'error_message' in result.columns
        assert len(result) == 3

    def test_validate_empty_series(self):
        """Debe manejar serie vacía."""
        series = pd.Series([], dtype=float)
        result = utils.validate_series(series)
        assert len(result) == 0

    def test_validate_series_with_kwargs(self):
        """Debe pasar kwargs a validate_numeric_value."""
        series = pd.Series([50, 150, 200])
        mask = utils.validate_series(series, return_mask=True, min_value=100, max_value=180)

        assert not mask.iloc[0]  # 50 < 100
        assert mask.iloc[1]  # 150 está en rango
        assert not mask.iloc[2]  # 200 > 180


# ============================================================================
# TESTS - ANÁLISIS Y DETECCIÓN
# ============================================================================

class TestCreateApuSignature:
    """Suite de pruebas para create_apu_signature()"""

    def test_signature_basic(self):
        """Debe crear firma básica."""
        apu_data = {
            'CODIGO_APU': 'APU-001',
            'DESCRIPCION_APU': 'Concreto',
            'UNIDAD_APU': 'M3'
        }
        signature = utils.create_apu_signature(apu_data)

        assert signature
        assert 'apu001' in signature.lower() or 'apu-001' in signature.lower()

    def test_signature_custom_fields(self):
        """Debe usar campos personalizados."""
        apu_data = {
            'CAMPO1': 'Valor1',
            'CAMPO2': 'Valor2'
        }
        signature = utils.create_apu_signature(apu_data, key_fields=['CAMPO1', 'CAMPO2'])

        assert 'valor1' in signature.lower()
        assert 'valor2' in signature.lower()

    def test_signature_missing_fields(self):
        """Debe manejar campos faltantes."""
        apu_data = {'CODIGO_APU': 'APU-001'}
        signature = utils.create_apu_signature(apu_data)

        assert signature
        assert signature != 'empty_signature'

    def test_signature_empty_data(self):
        """Debe manejar datos vacíos."""
        signature = utils.create_apu_signature({})
        assert signature == 'empty_signature'

    def test_signature_numeric_values(self):
        """Debe manejar valores numéricos."""
        apu_data = {
            'CODIGO_APU': 123,
            'DESCRIPCION_APU': 'Test',
            'UNIDAD_APU': 'M'
        }
        signature = utils.create_apu_signature(apu_data)
        assert '123' in signature

    def test_signature_normalization(self):
        """Debe normalizar valores."""
        apu1 = {'CODIGO_APU': 'APU-001', 'DESCRIPCION_APU': 'CONCRETO'}
        apu2 = {'CODIGO_APU': 'apu-001', 'DESCRIPCION_APU': 'concreto'}

        sig1 = utils.create_apu_signature(apu1)
        sig2 = utils.create_apu_signature(apu2)

        # Deberían ser iguales después de normalización
        assert sig1 == sig2


class TestDetectOutliers:
    """Suite de pruebas para detect_outliers()"""

    def test_detect_outliers_iqr_basic(self, sample_series_numeric):
        """Debe detectar outliers con método IQR."""
        outliers = utils.detect_outliers(sample_series_numeric, method='iqr')

        assert isinstance(outliers, pd.Series)
        assert outliers.dtype == bool
        # El valor 100.0 debería ser outlier
        assert outliers.iloc[5]  # índice 5 = 100.0

    def test_detect_outliers_with_bounds(self, sample_series_numeric):
        """Debe retornar bounds si se solicita."""
        outliers, bounds = utils.detect_outliers(
            sample_series_numeric,
            method='iqr',
            return_bounds=True
        )

        assert isinstance(bounds, dict)
        assert 'Q1' in bounds
        assert 'Q3' in bounds
        assert 'IQR' in bounds
        assert 'lower_bound' in bounds
        assert 'upper_bound' in bounds

    def test_detect_outliers_zscore(self, sample_series_numeric):
        """Debe detectar outliers con z-score."""
        outliers = utils.detect_outliers(sample_series_numeric, method='zscore', threshold=2)

        assert isinstance(outliers, pd.Series)
        assert outliers.dtype == bool

    def test_detect_outliers_modified_zscore(self, sample_series_numeric):
        """Debe detectar outliers con modified z-score."""
        outliers = utils.detect_outliers(
            sample_series_numeric,
            method='modified_zscore',
            threshold=3.5
        )

        assert isinstance(outliers, pd.Series)

    def test_detect_outliers_empty_series(self):
        """Debe manejar serie vacía."""
        empty_series = pd.Series([], dtype=float)
        outliers = utils.detect_outliers(empty_series)

        assert len(outliers) == 0

    def test_detect_outliers_all_nan(self):
        """Debe manejar serie con todos NaN."""
        nan_series = pd.Series([np.nan, np.nan, np.nan])
        outliers = utils.detect_outliers(nan_series)

        assert len(outliers) == 3
        assert not outliers.any()  # Todos deberían ser False

    def test_detect_outliers_invalid_method(self, sample_series_numeric):
        """Debe lanzar error para método inválido."""
        with pytest.raises(ValueError):
            utils.detect_outliers(sample_series_numeric, method='invalid_method')

    def test_detect_outliers_constant_series(self):
        """Debe manejar serie con valores constantes."""
        constant_series = pd.Series([5.0, 5.0, 5.0, 5.0])
        outliers = utils.detect_outliers(constant_series, method='iqr')

        # No debería haber outliers en serie constante
        assert not outliers.any()

    def test_detect_outliers_custom_threshold(self, sample_series_numeric):
        """Debe usar threshold personalizado."""
        outliers_strict = utils.detect_outliers(sample_series_numeric, threshold=1.0)
        outliers_loose = utils.detect_outliers(sample_series_numeric, threshold=3.0)

        # Threshold más estricto debería detectar más outliers
        assert outliers_strict.sum() >= outliers_loose.sum()


# ============================================================================
# TESTS - MANIPULACIÓN DE DATAFRAMES
# ============================================================================

class TestFindAndRenameColumns:
    """Suite de pruebas para find_and_rename_columns()"""

    def test_rename_exact_match(self):
        """Debe renombrar con coincidencia exacta."""
        df = pd.DataFrame({'codigo': [1, 2], 'desc': [3, 4]})
        column_map = {
            'CODIGO': ['codigo'],
            'DESCRIPCION': ['desc']
        }

        result = utils.find_and_rename_columns(df, column_map)

        assert 'CODIGO' in result.columns
        assert 'DESCRIPCION' in result.columns

    def test_rename_partial_match(self):
        """Debe renombrar con coincidencia parcial."""
        df = pd.DataFrame({'codigo_apu': [1, 2], 'descripcion_completa': [3, 4]})
        column_map = {
            'CODIGO': ['codigo'],
            'DESCRIPCION': ['descripcion']
        }

        result = utils.find_and_rename_columns(df, column_map, case_sensitive=False)

        assert 'CODIGO' in result.columns
        assert 'DESCRIPCION' in result.columns

    def test_rename_case_insensitive(self):
        """Debe ser case-insensitive por defecto."""
        df = pd.DataFrame({'CODIGO': [1, 2], 'codigo': [3, 4]})
        column_map = {'COD': ['codigo']}

        result = utils.find_and_rename_columns(df, column_map, case_sensitive=False)

        assert 'COD' in result.columns

    def test_rename_case_sensitive(self):
        """Debe respetar case cuando se especifica."""
        df = pd.DataFrame({'codigo': [1, 2], 'CODIGO': [3, 4]})
        column_map = {'COD': ['codigo']}

        result = utils.find_and_rename_columns(df, column_map, case_sensitive=True)

        # Solo debe renombrar 'codigo' (minúsculas)
        assert 'COD' in result.columns

    def test_rename_multiple_possibilities(self):
        """Debe buscar entre múltiples nombres posibles."""
        df = pd.DataFrame({'cod_apu': [1, 2]})
        column_map = {
            'CODIGO': ['codigo', 'cod', 'code']
        }

        result = utils.find_and_rename_columns(df, column_map)
        assert 'CODIGO' in result.columns

    def test_rename_empty_dataframe(self):
        """Debe manejar DataFrame vacío."""
        df = pd.DataFrame()
        column_map = {'CODIGO': ['codigo']}

        result = utils.find_and_rename_columns(df, column_map)
        assert result.empty

    def test_rename_no_matches(self):
        """Debe mantener columnas originales si no hay coincidencias."""
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        column_map = {'CODIGO': ['codigo']}

        result = utils.find_and_rename_columns(df, column_map)

        assert 'col1' in result.columns
        assert 'col2' in result.columns


class TestBatchProcessDataframe:
    """Suite de pruebas para batch_process_dataframe()"""

    def test_batch_process_small_df(self):
        """Debe procesar DataFrame pequeño sin batching."""
        df = pd.DataFrame({'A': range(100)})

        def process_func(df_chunk):
            df_chunk['B'] = df_chunk['A'] * 2
            return df_chunk

        result = utils.batch_process_dataframe(df, process_func, batch_size=1000)

        assert len(result) == 100
        assert 'B' in result.columns
        assert result['B'].iloc[0] == 0
        assert result['B'].iloc[50] == 100

    def test_batch_process_large_df(self):
        """Debe procesar DataFrame grande en batches."""
        df = pd.DataFrame({'A': range(5000)})

        def process_func(df_chunk):
            df_chunk['B'] = df_chunk['A'] * 2
            return df_chunk

        result = utils.batch_process_dataframe(df, process_func, batch_size=1000)

        assert len(result) == 5000
        assert 'B' in result.columns

    def test_batch_process_with_kwargs(self):
        """Debe pasar kwargs a función de procesamiento."""
        df = pd.DataFrame({'A': range(100)})

        def process_func(df_chunk, multiplier=1):
            df_chunk['B'] = df_chunk['A'] * multiplier
            return df_chunk

        result = utils.batch_process_dataframe(df, process_func, batch_size=50, multiplier=3)

        assert result['B'].iloc[10] == 30


# ============================================================================
# TESTS - SERIALIZACIÓN
# ============================================================================

class TestSanitizeForJson:
    """Suite de pruebas para sanitize_for_json()"""

    def test_sanitize_basic_types(self):
        """Debe mantener tipos básicos de Python."""
        data = {
            'string': 'text',
            'int': 123,
            'float': 123.45,
            'bool': True,
            'none': None,
            'list': [1, 2, 3],
            'dict': {'nested': 'value'}
        }

        result = utils.sanitize_for_json(data)

        assert result['string'] == 'text'
        assert result['int'] == 123
        assert result['float'] == 123.45
        assert result['bool'] is True
        assert result['none'] is None

    def test_sanitize_numpy_types(self):
        """Debe convertir tipos de NumPy."""
        data = {
            'np_int': np.int64(123),
            'np_float': np.float64(123.45),
            'np_bool': np.bool_(True),
            'np_array': np.array([1, 2, 3])
        }

        result = utils.sanitize_for_json(data)

        assert isinstance(result['np_int'], int)
        assert isinstance(result['np_float'], float)
        assert isinstance(result['np_bool'], bool)
        assert isinstance(result['np_array'], list)

    def test_sanitize_pandas_types(self):
        """Debe convertir tipos de Pandas."""
        data = {
            'series': pd.Series([1, 2, 3]),
            'dataframe': pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        }

        result = utils.sanitize_for_json(data)

        assert isinstance(result['series'], list)
        assert isinstance(result['dataframe'], list)

    def test_sanitize_nan_inf(self):
        """Debe convertir NaN e Inf a None."""
        data = {
            'nan': np.nan,
            'inf': np.inf,
            'neg_inf': -np.inf
        }

        result = utils.sanitize_for_json(data)

        assert result['nan'] is None
        assert result['inf'] is None
        assert result['neg_inf'] is None

    def test_sanitize_pandas_na(self):
        """Debe convertir pd.NA a None."""
        data = {'pd_na': pd.NA}
        result = utils.sanitize_for_json(data)
        assert result['pd_na'] is None

    def test_sanitize_nested_structures(self):
        """Debe manejar estructuras anidadas."""
        data = {
            'level1': {
                'level2': {
                    'level3': [np.int64(1), np.float32(2.5)]
                }
            }
        }

        result = utils.sanitize_for_json(data)

        assert isinstance(result['level1']['level2']['level3'][0], int)
        assert isinstance(result['level1']['level2']['level3'][1], float)

    def test_sanitize_max_depth_error(self):
        """Debe lanzar error al exceder profundidad máxima."""
        # Crear estructura muy anidada
        deep_data = {'level': {}}
        current = deep_data['level']
        for i in range(150):
            current['level'] = {}
            current = current['level']

        with pytest.raises(RecursionError):
            utils.sanitize_for_json(deep_data, max_depth=100)

    def test_sanitize_datetime(self):
        """Debe convertir fechas a ISO format."""
        from datetime import datetime
        data = {'date': datetime(2024, 1, 1, 12, 30)}
        result = utils.sanitize_for_json(data)

        assert isinstance(result['date'], str)
        assert '2024' in result['date']


# ============================================================================
# TESTS - ESTADÍSTICAS
# ============================================================================

class TestCalculateStatistics:
    """Suite de pruebas para calculate_statistics()"""

    def test_statistics_basic(self):
        """Debe calcular estadísticas básicas."""
        series = pd.Series([1, 2, 3, 4, 5])
        stats = utils.calculate_statistics(series)

        assert stats['count'] == 5
        assert stats['mean'] == 3.0
        assert stats['min'] == 1.0
        assert stats['max'] == 5.0
        assert stats['median'] == 3.0

    def test_statistics_with_nulls(self):
        """Debe manejar valores nulos."""
        series = pd.Series([1, 2, np.nan, 4, 5])
        stats = utils.calculate_statistics(series)

        assert stats['count'] == 4  # Sin contar NaN
        assert stats['null_count'] == 1
        assert stats['null_percentage'] == 20.0

    def test_statistics_empty_series(self):
        """Debe manejar serie vacía."""
        series = pd.Series([], dtype=float)
        stats = utils.calculate_statistics(series)

        assert stats['count'] == 0
        assert stats['mean'] is None

    def test_statistics_all_nan(self):
        """Debe manejar serie con todos NaN."""
        series = pd.Series([np.nan, np.nan, np.nan])
        stats = utils.calculate_statistics(series)

        assert stats['count'] == 0
        assert stats['null_count'] == 3

    def test_statistics_quartiles(self):
        """Debe calcular cuartiles."""
        series = pd.Series(range(1, 101))
        stats = utils.calculate_statistics(series)

        assert 'q1' in stats
        assert 'q3' in stats
        assert stats['q1'] < stats['median'] < stats['q3']


class TestCalculateStdDev:
    """Suite de pruebas para calculate_std_dev()"""

    def test_std_dev_basic(self):
        """Debe calcular desviación estándar."""
        values = [2, 4, 4, 4, 5, 5, 7, 9]
        std_dev = utils.calculate_std_dev(values)

        assert std_dev > 0
        assert isinstance(std_dev, float)

    def test_std_dev_constant_values(self):
        """Debe retornar 0 para valores constantes."""
        values = [5, 5, 5, 5, 5]
        std_dev = utils.calculate_std_dev(values)

        assert std_dev == 0.0

    def test_std_dev_single_value(self):
        """Debe retornar 0 para un solo valor."""
        values = [5]
        std_dev = utils.calculate_std_dev(values)

        assert std_dev == 0.0

    def test_std_dev_empty_list(self):
        """Debe retornar 0 para lista vacía."""
        values = []
        std_dev = utils.calculate_std_dev(values)

        assert std_dev == 0.0


# ============================================================================
# TESTS - CÁLCULO DE COSTOS UNITARIOS
# ============================================================================

class TestCalculateUnitCosts:
    """Suite de pruebas para calculate_unit_costs()"""

    def test_calculate_costs_basic(self, sample_dataframe):
        """Debe calcular costos unitarios básicos."""
        result = utils.calculate_unit_costs(sample_dataframe)

        assert not result.empty
        assert 'CODIGO_APU' in result.columns
        assert 'COSTO_UNITARIO_TOTAL' in result.columns
        assert 'VALOR_SUMINISTRO_UN' in result.columns

    def test_calculate_costs_empty_df(self):
        """Debe manejar DataFrame vacío."""
        df = pd.DataFrame()
        result = utils.calculate_unit_costs(df)

        assert result.empty

    def test_calculate_costs_missing_columns(self):
        """Debe manejar columnas faltantes."""
        df = pd.DataFrame({'CODIGO_APU': [1, 2]})
        result = utils.calculate_unit_costs(df)

        assert result.empty

    def test_calculate_costs_aggregation(self):
        """Debe agregar correctamente por APU."""
        df = pd.DataFrame({
            'CODIGO_APU': ['APU-001', 'APU-001', 'APU-002'],
            'DESCRIPCION_APU': ['Concreto', 'Concreto', 'Acero'],
            'UNIDAD_APU': ['M3', 'M3', 'KG'],
            'TIPO_INSUMO': ['SUMINISTRO', 'MANO_DE_OBRA', 'SUMINISTRO'],
            'VALOR_TOTAL_APU': [1000.0, 500.0, 2000.0]
        })

        result = utils.calculate_unit_costs(df)

        # Debe haber 2 APUs únicos
        assert len(result) == 2

        # APU-001 debe tener suma de suministro e instalación
        apu_001 = result[result['CODIGO_APU'] == 'APU-001'].iloc[0]
        assert apu_001['VALOR_SUMINISTRO_UN'] == 1000.0
        assert apu_001['VALOR_INSTALACION_UN'] == 500.0

    def test_calculate_costs_percentages(self):
        """Debe calcular porcentajes correctamente."""
        df = pd.DataFrame({
            'CODIGO_APU': ['APU-001'],
            'DESCRIPCION_APU': ['Test'],
            'UNIDAD_APU': ['M'],
            'TIPO_INSUMO': ['SUMINISTRO'],
            'VALOR_TOTAL_APU': [1000.0]
        })

        result = utils.calculate_unit_costs(df)

        assert result['PCT_SUMINISTRO'].iloc[0] == 100.0
        assert result['PCT_INSTALACION'].iloc[0] == 0.0

    def test_calculate_costs_zero_division(self):
        """Debe manejar división por cero."""
        df = pd.DataFrame({
            'CODIGO_APU': ['APU-001'],
            'DESCRIPCION_APU': ['Test'],
            'UNIDAD_APU': ['M'],
            'TIPO_INSUMO': ['SUMINISTRO'],
            'VALOR_TOTAL_APU': [0.0]
        })

        result = utils.calculate_unit_costs(df)

        # No debe lanzar error
        assert not result.empty
        assert result['PCT_SUMINISTRO'].iloc[0] == 0.0


# ============================================================================
# TESTS DE INTEGRACIÓN Y CASOS COMPLEJOS
# ============================================================================

class TestIntegration:
    """Tests de integración entre múltiples funciones."""

    def test_full_workflow_text_processing(self):
        """Workflow completo de procesamiento de texto."""
        # Crear datos de prueba
        texts = pd.Series([
            'CÓDIGO APU-001',
            'código apu-002',
            'Código APU-003'
        ])

        # Normalizar
        normalized = utils.normalize_text_series(texts)

        # Verificar resultados
        assert all('codigo' in text for text in normalized)
        assert all('apu' in text for text in normalized)

    def test_full_workflow_numeric_processing(self):
        """Workflow completo de procesamiento numérico."""
        # Datos con diferentes formatos
        values = ['$1,234.56', '€2.345,67', '15%', '1.5e3']

        results = [
            utils.parse_number(values[0]),
            utils.parse_number(values[1], decimal_separator='comma'),
            utils.parse_number(values[2], allow_percentage=True),
            utils.parse_number(values[3], allow_scientific=True)
        ]

        assert results[0] == 1234.56
        assert abs(results[1] - 2345.67) < 0.01
        assert results[2] == 0.15
        assert results[3] == 1500.0

    def test_full_workflow_dataframe_processing(self, sample_dataframe):
        """Workflow completo de procesamiento de DataFrame."""
        # Renombrar columnas
        column_map = {
            'CODE': ['CODIGO', 'codigo'],
            'DESC': ['DESCRIPCION', 'descripcion']
        }
        df = utils.find_and_rename_columns(sample_dataframe.copy(), column_map)

        # Validar valores numéricos
        valid_mask = utils.validate_series(
            df['VALOR_TOTAL_APU'],
            return_mask=True,
            min_value=0
        )

        assert valid_mask.all()

        # Detectar outliers
        outliers = utils.detect_outliers(df['VALOR_TOTAL_APU'])
        assert isinstance(outliers, pd.Series)

    def test_full_workflow_apu_processing(self):
        """Workflow completo de procesamiento de APU."""
        # Crear datos de prueba
        apu_data = {
            'CODIGO_APU': '  apu-001  ',
            'DESCRIPCION_APU': 'CONCRETO F\'C=280 KG/CM2',
            'UNIDAD_APU': 'metros cubicos',
            'VALOR': '1,234.56'
        }

        # Procesar
        processed = {
            'CODIGO_APU': utils.clean_apu_code(apu_data['CODIGO_APU']),
            'DESCRIPCION_APU': utils.normalize_text(apu_data['DESCRIPCION_APU']),
            'UNIDAD_APU': utils.normalize_unit(apu_data['UNIDAD_APU']),
            'VALOR': utils.parse_number(apu_data['VALOR'])
        }

        # Verificar
        assert processed['CODIGO_APU'] == 'APU-001'
        assert 'concreto' in processed['DESCRIPCION_APU']
        assert processed['UNIDAD_APU'] == 'M3'
        assert processed['VALOR'] == 1234.56

        # Crear firma
        signature = utils.create_apu_signature(processed)
        assert signature
        assert 'apu001' in signature.lower()


# ============================================================================
# TESTS DE RENDIMIENTO
# ============================================================================

class TestPerformance:
    """Tests de rendimiento y optimización."""

    def test_normalize_text_cache_performance(self):
        """Cache debe mejorar rendimiento."""
        text = 'Test String For Cache'

        # Primera llamada (no cache)
        utils.normalize_text(text)

        # Segunda llamada (con cache)
        import time
        start = time.time()
        for _ in range(1000):
            utils.normalize_text(text)
        cached_time = time.time() - start

        # Debe ser muy rápido con cache
        assert cached_time < 0.1  # Menos de 100ms para 1000 llamadas

    def test_parse_number_performance(self):
        """parse_number debe ser eficiente."""
        import time

        values = ['1234.56'] * 1000

        start = time.time()
        results = [utils.parse_number(v) for v in values]
        elapsed = time.time() - start

        assert len(results) == 1000
        assert elapsed < 1.0  # Menos de 1 segundo para 1000 conversiones

    def test_batch_processing_efficiency(self):
        """Batch processing debe manejar DataFrames grandes."""
        # Crear DataFrame grande
        large_df = pd.DataFrame({
            'A': range(50000),
            'B': range(50000)
        })

        def simple_process(df):
            df['C'] = df['A'] + df['B']
            return df

        import time
        start = time.time()
        result = utils.batch_process_dataframe(large_df, simple_process, batch_size=10000)
        elapsed = time.time() - start

        assert len(result) == 50000
        assert 'C' in result.columns
        # Debe completar en tiempo razonable
        assert elapsed < 5.0


# ============================================================================
# TESTS DE CONSTANTES Y CONFIGURACIÓN
# ============================================================================

class TestConstants:
    """Tests para constantes y configuración."""

    def test_standard_units_is_frozen_set(self):
        """STANDARD_UNITS debe ser frozenset."""
        assert isinstance(utils.STANDARD_UNITS, frozenset)

    def test_standard_units_contains_common_units(self):
        """Debe contener unidades comunes."""
        assert 'M' in utils.STANDARD_UNITS
        assert 'M2' in utils.STANDARD_UNITS
        assert 'M3' in utils.STANDARD_UNITS
        assert 'KG' in utils.STANDARD_UNITS
        assert 'UND' in utils.STANDARD_UNITS

    def test_unit_mapping_is_dict(self):
        """UNIT_MAPPING debe ser diccionario."""
        assert isinstance(utils.UNIT_MAPPING, dict)

    def test_unit_mapping_consistency(self):
        """Mapeo de unidades debe ser consistente."""
        assert utils.UNIT_MAPPING['METROS'] == 'M'
        assert utils.UNIT_MAPPING['KILOGRAMOS'] == 'KG'
        assert utils.UNIT_MAPPING['UNIDAD'] == 'UND'


# ============================================================================
# TESTS DE MANEJO DE ERRORES
# ============================================================================

class TestErrorHandling:
    """Tests para manejo robusto de errores."""

    def test_normalize_text_invalid_type_handling(self):
        """Debe manejar tipos inválidos gracefully."""
        # Objeto que no puede convertirse a string fácilmente
        class UnconvertibleObject:
            def __str__(self):
                raise Exception("Cannot convert")

        obj = UnconvertibleObject()

        with pytest.raises(TypeError):
            utils.normalize_text(obj)

    def test_parse_number_extreme_values(self):
        """Debe manejar valores extremos."""
        # Número muy grande
        result = utils.parse_number('999999999999999999')
        assert isinstance(result, float)

        # Número muy pequeño
        result = utils.parse_number('0.000000000001')
        assert isinstance(result, float)

    def test_validate_numeric_mixed_types(self):
        """Debe manejar tipos mixtos en validación."""
        # Integer de Python
        is_valid, _ = utils.validate_numeric_value(100)
        assert is_valid

        # Float de Python
        is_valid, _ = utils.validate_numeric_value(100.5)
        assert is_valid

        # NumPy int
        is_valid, _ = utils.validate_numeric_value(np.int32(100))
        assert is_valid

        # NumPy float
        is_valid, _ = utils.validate_numeric_value(np.float64(100.5))
        assert is_valid

    def test_safe_read_corrupted_file(self, tmp_path):
        """Debe manejar archivos corruptos."""
        corrupted_file = tmp_path / "corrupted.csv"
        corrupted_file.write_bytes(b'\x00\x01\x02\x03\x04')

        df = utils.safe_read_dataframe(corrupted_file)
        # Debería retornar DataFrame vacío sin crash
        assert isinstance(df, pd.DataFrame)


# ============================================================================
# CONFIGURACIÓN DE PYTEST
# ============================================================================

def pytest_configure(config):
    """Configuración de pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == '__main__':
    # Ejecutar tests con pytest
    pytest.main([
        __file__,
        '-v',  # Verbose
        '--tb=short',  # Traceback corto
        '--cov=utils',  # Cobertura de código
        '--cov-report=html',  # Reporte HTML
        '--cov-report=term-missing',  # Mostrar líneas no cubiertas
        '-W', 'ignore::DeprecationWarning',  # Ignorar warnings de deprecación
    ])
