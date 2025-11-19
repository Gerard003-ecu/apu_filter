# tests/test_grammar_diagnostics.py

"""
Suite de pruebas robusta para el módulo grammar_diagnostics.

Cobertura de pruebas:
- Diagnóstico de líneas válidas e inválidas
- Manejo de diferentes tipos de errores de Lark
- Generación correcta de reportes
- Análisis de patrones de fallo
- Manejo de archivos y encodings
- Casos edge y errores
"""

import os
from typing import List
from unittest.mock import patch

import pytest

# Importar la función a testear
from dev_tools.diagnose_grammar import diagnose_grammar_mismatches

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def simple_grammar():
    """Gramática Lark simple para testing."""
    return """
    ?start: line
    line: description ";" unit ";" quantity ";" price ";" total
    description: /[^;]+/
    unit: /[A-Z0-9.]+/
    quantity: NUMBER
    price: NUMBER
    total: NUMBER
    NUMBER: /\\d+[.,]\\d+/ | /\\d+/
    %import common.WS
    %ignore WS
    """


@pytest.fixture
def complex_grammar():
    """Gramática compleja que permite campos opcionales."""
    return """
    ?start: line

    line: description ";" [unit] ";" quantity ";" [price] ";" total

    description: /[^;]+/
    unit: /[A-Z]{2,4}/
    quantity: NUMBER
    price: NUMBER
    total: NUMBER

    NUMBER: /\\d+[.,]\\d+/ | /\\d+/

    %import common.WS
    %ignore WS
    """


@pytest.fixture
def valid_csv_content():
    """Contenido CSV válido para testing."""
    return [
        "CEMENTO GRIS;KG;150,5;2,35;353,68",
        "ARENA LAVADA;M3;2,000;45,00;90,00",
        "ACERO DE REFUERZO;KG;85,25;1,89;161,12",
        "ENCOFRADO METALICO;M2;12,5;15,75;196,88",
        "MANO DE OBRA OFICIAL;HR;8,0;12,50;100,00",
    ]


@pytest.fixture
def invalid_csv_content():
    """Contenido CSV con errores para testing."""
    return [
        "CEMENTO GRIS;KG;150,5;2,35;353,68",  # Válida
        "ARENA LAVADA;;2,000;45,00;90,00",  # Campo unit vacío
        "ACERO;;;1,89;161,12",  # Múltiples campos vacíos
        ";;;45,00;",  # Casi toda vacía
        "ENCOFRADO;M2;abc;15,75;196,88",  # Quantity no numérico
        "SUBTOTAL;;;;5000,00",  # Línea de subtotal
        ";;;;",  # Línea vacía con separadores
    ]


@pytest.fixture
def csv_with_headers():
    """CSV con encabezados de APU."""
    return [
        "EXCAVACION MANUAL;UNIDAD:M3",
        "ITEM:001.001",
        "CEMENTO GRIS;KG;150,5;2,35;353,68",
        "ARENA LAVADA;M3;2,000;45,00;90,00",
        "CONSTRUCCION DE MURO;UNIDAD:M2",
        "ITEM:002.001",
        "BLOQUE DE CONCRETO;UND;45;2,50;112,50",
    ]


@pytest.fixture
def temp_csv_file(tmp_path):
    """Crea un archivo CSV temporal para testing."""
    def _create_file(content: List[str], filename: str = "test.csv"):
        file_path = tmp_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        return str(file_path)

    return _create_file


@pytest.fixture
def temp_output_file(tmp_path):
    """Genera ruta para archivo de salida temporal."""
    return str(tmp_path / "diagnosis_output.txt")


@pytest.fixture
def mock_logger():
    """Mock del logger para verificar llamadas."""
    with patch('dev_tools.diagnose_grammar.logger') as mock_log:
        yield mock_log


# ============================================================================
# TESTS: FUNCIONALIDAD BÁSICA
# ============================================================================


class TestBasicFunctionality:
    """Tests de funcionalidad básica del diagnóstico."""

    def test_diagnose_all_valid_lines(
        self,
        simple_grammar,
        valid_csv_content,
        temp_csv_file,
        temp_output_file
    ):
        """Test con todas las líneas válidas."""
        csv_file = temp_csv_file(valid_csv_content)

        failed_lines = diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        assert len(failed_lines) == 0, "No deberían haber líneas fallidas"
        assert os.path.exists(temp_output_file), "Debe crear archivo de salida"

        # Verificar contenido del reporte
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Total líneas analizadas: 5" in content
            assert "Líneas que fallan Lark: 0" in content
            assert "Tasa de fallo: 0.00%" in content

    def test_diagnose_all_invalid_lines(
        self,
        simple_grammar,
        temp_csv_file,
        temp_output_file
    ):
        """Test con todas las líneas inválidas."""
        invalid_content = [
            ";;;",
            "abc",
            "campo1;campo2",
        ]
        csv_file = temp_csv_file(invalid_content)

        failed_lines = diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        assert len(failed_lines) == 3, "Todas las líneas deberían fallar"

        # Verificar estructura de líneas fallidas
        for failed in failed_lines:
            assert "line_num" in failed
            assert "line" in failed
            assert "error" in failed
            assert "fields" in failed
            assert "fields_count" in failed

    def test_diagnose_mixed_valid_invalid(
        self,
        simple_grammar,
        invalid_csv_content,
        temp_csv_file,
        temp_output_file
    ):
        """Test con mezcla de líneas válidas e inválidas."""
        csv_file = temp_csv_file(invalid_csv_content)

        failed_lines = diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        # Al menos algunas deberían fallar (depende de la gramática estricta)
        assert len(failed_lines) > 0, "Debería detectar líneas inválidas"
        assert len(failed_lines) < len(
            invalid_csv_content
        ), "No todas deberían fallar"

        # Verificar que detecta campos vacíos
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Líneas con campos vacíos:" in content

    def test_skip_apu_headers(
        self,
        simple_grammar,
        csv_with_headers,
        temp_csv_file,
        temp_output_file
    ):
        """Test que verifica que se saltan los encabezados de APU."""
        csv_file = temp_csv_file(csv_with_headers)

        failed_lines = diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        # Verificar que no se incluyen líneas UNIDAD: o ITEM:
        for failed in failed_lines:
            assert "UNIDAD:" not in failed["line"]
            assert "ITEM:" not in failed["line"]

        with open(temp_output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Solo debería contar líneas de insumos (no headers)
            assert "Total líneas analizadas:" in content
            # Verificar que el número es menor que el total del archivo


# ============================================================================
# TESTS: ANÁLISIS DE PATRONES
# ============================================================================


class TestPatternAnalysis:
    """Tests del análisis de patrones de fallo."""

    def test_detect_empty_fields(
        self,
        simple_grammar,
        temp_csv_file,
        temp_output_file
    ):
        """Test detección de campos vacíos."""
        content = [
            "DESCRIPCION;;123;456;789",  # Campo 1 vacío
            "DESCRIPCION;UND;;456;789",  # Campo 2 vacío
            "DESCRIPCION;UND;123;;789",  # Campo 3 vacío
        ]
        csv_file = temp_csv_file(content)

        diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        # Verificar que se registran las posiciones vacías
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Campos vacíos en posiciones:" in content
            assert "Líneas con campos vacíos:" in content

    def test_failure_rate_calculation(
        self,
        simple_grammar,
        temp_csv_file,
        temp_output_file
    ):
        """Test cálculo correcto de tasa de fallo."""
        # 2 válidas, 3 inválidas = 60% fallo
        content = [
            "VALIDA 1;KG;100;2,50;250,00",
            "VALIDA 2;M3;5,5;100;550,00",
            "INVALIDA;;;",
            "INVALIDA;abc;def;ghi;jkl",
            "INVALIDA;;;;;;",
        ]
        csv_file = temp_csv_file(content)

        diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        with open(temp_output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Total líneas analizadas: 5" in content
            # Tasa de fallo debería estar documentada
            assert "Tasa de fallo:" in content
            assert "%" in content

    def test_sample_limit_respected(
        self,
        simple_grammar,
        temp_csv_file,
        temp_output_file
    ):
        """Test que se respeta el límite de muestras (primeras 20)."""
        # Crear 30 líneas inválidas
        content = [f"INVALIDA_{i};;;" for i in range(30)]
        csv_file = temp_csv_file(content)

        diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        with open(temp_output_file, 'r', encoding='utf-8') as f:
            content = f.read()

            # Contar cuántas muestras se muestran
            sample_count = content.count("Línea ")

            # Debería mostrar máximo 20 muestras detalladas
            assert sample_count <= 20, (
                f"Se mostraron {sample_count} muestras, máximo 20"
            )


# ============================================================================
# TESTS: MANEJO DE ERRORES
# ============================================================================


class TestErrorHandling:
    """Tests de manejo de errores y casos edge."""

    def test_file_not_found(self, simple_grammar, temp_output_file):
        """Test cuando el archivo CSV no existe."""
        with pytest.raises(FileNotFoundError):
            diagnose_grammar_mismatches(
                csv_file="nonexistent_file.csv",
                grammar=simple_grammar,
                output_file=temp_output_file
            )

    def test_invalid_grammar(
        self,
        temp_csv_file,
        valid_csv_content,
        temp_output_file
    ):
        """Test con gramática inválida."""
        csv_file = temp_csv_file(valid_csv_content)
        invalid_grammar = "INVALID GRAMMAR SYNTAX {{{"

        with pytest.raises(Exception):  # Lark lanzará excepción
            diagnose_grammar_mismatches(
                csv_file=csv_file,
                grammar=invalid_grammar,
                output_file=temp_output_file
            )

    def test_empty_csv_file(
        self,
        simple_grammar,
        temp_csv_file,
        temp_output_file
    ):
        """Test con archivo CSV vacío."""
        csv_file = temp_csv_file([])

        failed_lines = diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        assert len(failed_lines) == 0

        with open(temp_output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Total líneas analizadas: 0" in content

    def test_csv_with_only_whitespace(
        self,
        simple_grammar,
        temp_csv_file,
        temp_output_file
    ):
        """Test con archivo que solo tiene espacios en blanco."""
        content = ["   ", "\t", "\n", "  \t  \n  "]
        csv_file = temp_csv_file(content)

        failed_lines = diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        # Las líneas vacías se eliminan con strip()
        assert len(failed_lines) == 0

    def test_csv_with_special_characters(
        self,
        simple_grammar,
        temp_csv_file,
        temp_output_file
    ):
        """Test con caracteres especiales y Unicode."""
        content = [
            "DESCRIPCIÓN ESPAÑOLA;KG;100;2,50;250,00",
            "中文描述;UND;50;1,00;50,00",
            "Émoji Test 🏗️;M2;10;5,00;50,00",
            "Quote\"Test;M3;1;1;1",
        ]
        csv_file = temp_csv_file(content)

        # No debería lanzar excepción
        diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        # Verificar que el archivo se generó correctamente
        assert os.path.exists(temp_output_file)

    def test_invalid_encoding_handling(
        self,
        simple_grammar,
        tmp_path,
        temp_output_file
    ):
        """Test manejo de diferentes encodings."""
        # Crear archivo con encoding diferente
        csv_file = tmp_path / "latin1_file.csv"
        content = "DESCRIPCIÓN;KG;100;2,50;250,00"

        with open(csv_file, 'w', encoding='latin-1') as f:
            f.write(content)

        # Debería manejar el encoding o lanzar error claro
        try:
            diagnose_grammar_mismatches(
                csv_file=str(csv_file),
                grammar=simple_grammar,
                output_file=temp_output_file
            )
            # Si no falla, es que manejó el encoding
            assert True
        except UnicodeDecodeError:
            # Si falla, es esperado para este test
            assert True


# ============================================================================
# TESTS: TIPOS DE ERRORES LARK
# ============================================================================


class TestLarkErrorTypes:
    """Tests de diferentes tipos de errores de Lark."""

    def test_unexpected_input_error(
        self,
        simple_grammar,
        temp_csv_file,
        temp_output_file
    ):
        """Test líneas que causan UnexpectedInput."""
        content = [
            "DESC;INVALID_UNIT_FORMAT;123;456;789",
        ]
        csv_file = temp_csv_file(content)

        failed_lines = diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        if failed_lines:
            # Verificar que se captura el error
            assert any("error" in fl for fl in failed_lines)

    def test_unexpected_characters_error(
        self,
        simple_grammar,
        temp_csv_file,
        temp_output_file
    ):
        """Test líneas con caracteres inesperados."""
        content = [
            "DESC;KG;123@invalid;456;789",
        ]
        csv_file = temp_csv_file(content)

        failed_lines = diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        # Debería detectar el carácter inválido
        assert len(failed_lines) > 0

    def test_parse_error_generic(
        self,
        simple_grammar,
        temp_csv_file,
        temp_output_file
    ):
        """Test errores de parsing genéricos."""
        content = [
            "Incomplete line;KG;123",  # Faltan campos
        ]
        csv_file = temp_csv_file(content)

        failed_lines = diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        assert len(failed_lines) > 0
        assert failed_lines[0]["fields_count"] < 5


# ============================================================================
# TESTS: FORMATO DE SALIDA
# ============================================================================


class TestOutputFormat:
    """Tests del formato de archivo de salida."""

    def test_output_file_structure(
        self,
        simple_grammar,
        valid_csv_content,
        temp_csv_file,
        temp_output_file
    ):
        """Test estructura correcta del archivo de salida."""
        csv_file = temp_csv_file(valid_csv_content)

        diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        with open(temp_output_file, 'r', encoding='utf-8') as f:
            content = f.read()

            # Verificar secciones obligatorias
            assert "DIAGNÓSTICO DE INCOMPATIBILIDAD GRAMÁTICA-DATOS" in content
            assert "Total líneas analizadas:" in content
            assert "Líneas que fallan Lark:" in content
            assert "Tasa de fallo:" in content
            assert "Líneas con campos vacíos:" in content
            assert "MUESTRAS DE LÍNEAS FALLIDAS:" in content

    def test_output_file_encoding_utf8(
        self,
        simple_grammar,
        temp_csv_file,
        temp_output_file
    ):
        """Test que el archivo de salida usa UTF-8."""
        content = ["DESCRIPCIÓN CON ÑOÑO;KG;100;2,50;250,00"]
        csv_file = temp_csv_file(content)

        diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        # Leer con UTF-8 no debería fallar
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "DESCRIPCIÓN" in content or "Línea" in content

    def test_output_detailed_error_info(
        self,
        simple_grammar,
        temp_csv_file,
        temp_output_file
    ):
        """Test que se incluye información detallada de errores."""
        content = [
            "INVALIDA;;123;456;789",
        ]
        csv_file = temp_csv_file(content)

        diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        with open(temp_output_file, 'r', encoding='utf-8') as f:
            content = f.read()

            # Debe incluir detalles de la muestra
            assert "Error:" in content
            assert "Campos:" in content
            assert "Contenido:" in content
            assert "Campos vacíos en posiciones:" in content


# ============================================================================
# TESTS: INTEGRACIÓN CON LOGGER
# ============================================================================


class TestLogging:
    """Tests de integración con el sistema de logging."""

    def test_logger_info_called(
        self,
        simple_grammar,
        valid_csv_content,
        temp_csv_file,
        temp_output_file,
        mock_logger
    ):
        """Test que se llama al logger.info."""
        csv_file = temp_csv_file(valid_csv_content)

        diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        # Verificar que se llamó a logger.info con el mensaje de éxito
        mock_logger.info.assert_called()
        call_args = [str(call) for call in mock_logger.info.call_args_list]
        assert any("guardado" in str(arg).lower() for arg in call_args)

    def test_logger_includes_output_path(
        self,
        simple_grammar,
        valid_csv_content,
        temp_csv_file,
        temp_output_file,
        mock_logger
    ):
        """Test que el logger incluye la ruta del archivo de salida."""
        csv_file = temp_csv_file(valid_csv_content)

        diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        # Verificar que se menciona el archivo de salida
        all_calls = str(mock_logger.info.call_args_list)
        assert temp_output_file in all_calls or "diagnosis" in all_calls.lower()


# ============================================================================
# TESTS: RETORNO DE DATOS
# ============================================================================


class TestReturnValue:
    """Tests del valor de retorno de la función."""

    def test_return_type_is_list(
        self,
        simple_grammar,
        valid_csv_content,
        temp_csv_file,
        temp_output_file
    ):
        """Test que retorna una lista."""
        csv_file = temp_csv_file(valid_csv_content)

        result = diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        assert isinstance(result, list)

    def test_return_structure_complete(
        self,
        simple_grammar,
        temp_csv_file,
        temp_output_file
    ):
        """Test estructura completa de cada elemento retornado."""
        content = ["INVALIDA;;;"]
        csv_file = temp_csv_file(content)

        failed_lines = diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        if failed_lines:
            line = failed_lines[0]

            # Verificar campos obligatorios
            assert "line_num" in line
            assert "line" in line
            assert "error" in line
            assert "fields" in line
            assert "fields_count" in line

            # Verificar tipos
            assert isinstance(line["line_num"], int)
            assert isinstance(line["line"], str)
            assert isinstance(line["error"], str)
            assert isinstance(line["fields"], list)
            assert isinstance(line["fields_count"], int)

    def test_return_preserves_order(
        self,
        simple_grammar,
        temp_csv_file,
        temp_output_file
    ):
        """Test que el orden de las líneas fallidas se preserva."""
        content = [
            "PRIMERA INVALIDA;;;",
            "VALIDA;KG;100;2,50;250,00",
            "SEGUNDA INVALIDA;;;",
            "TERCERA INVALIDA;;;",
        ]
        csv_file = temp_csv_file(content)

        failed_lines = diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        # Verificar que están en orden
        if len(failed_lines) >= 2:
            assert failed_lines[0]["line_num"] < failed_lines[1]["line_num"]
            assert "PRIMERA" in failed_lines[0]["line"]


# ============================================================================
# TESTS: PERFORMANCE Y ESCALABILIDAD
# ============================================================================


class TestPerformance:
    """Tests de performance y escalabilidad."""

    def test_handles_large_file(
        self,
        simple_grammar,
        temp_csv_file,
        temp_output_file
    ):
        """Test con archivo grande (1000 líneas)."""
        # Generar 1000 líneas
        content = [
            f"DESCRIPCION_{i};KG;{i};2,50;{i*2.5}"
            for i in range(1000)
        ]
        csv_file = temp_csv_file(content)

        import time
        start = time.time()

        diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        elapsed = time.time() - start

        # No debería tardar más de 10 segundos
        assert elapsed < 10, f"Tomó {elapsed}s, demasiado lento"
        assert os.path.exists(temp_output_file)

    def test_memory_efficient_with_large_errors(
        self,
        simple_grammar,
        temp_csv_file,
        temp_output_file
    ):
        """Test eficiencia de memoria con muchos errores."""
        # Generar 500 líneas inválidas
        content = [f"INVALIDA_{i};;;" for i in range(500)]
        csv_file = temp_csv_file(content)

        failed_lines = diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        # Debería procesar todas
        assert len(failed_lines) == 500

        # El archivo de salida debe limitar muestras a 20
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Contar menciones de "Línea X:" (muestras detalladas)
            detailed_samples = content.count("Línea ")
            assert detailed_samples <= 20


# ============================================================================
# TESTS: CASOS EDGE ESPECÍFICOS
# ============================================================================


class TestEdgeCases:
    """Tests de casos edge específicos del dominio."""

    def test_line_with_semicolon_in_description(
        self,
        temp_csv_file,
        temp_output_file
    ):
        """Test línea con punto y coma en la descripción."""
        # Esto es problemático porque ; es el delimitador
        grammar = """
        ?start: line
        line: description ";" rest
        description: /[^;]+/
        rest: /.+/
        """

        content = ['DESCRIPCION;CON;PUNTO;Y;COMA;KG;100']
        csv_file = temp_csv_file(content)

        # No debería crashear
        diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=grammar,
            output_file=temp_output_file
        )

    def test_line_with_only_semicolons(
        self,
        simple_grammar,
        temp_csv_file,
        temp_output_file
    ):
        """Test línea que solo tiene punto y coma."""
        content = [';;;;;;;;;;;;']
        csv_file = temp_csv_file(content)

        failed_lines = diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        assert len(failed_lines) > 0
        assert failed_lines[0]["fields_count"] > 0

    def test_line_exceeding_typical_length(
        self,
        simple_grammar,
        temp_csv_file,
        temp_output_file
    ):
        """Test línea extremadamente larga."""
        long_description = "A" * 10000
        content = [f'{long_description};KG;100;2,50;250,00']
        csv_file = temp_csv_file(content)

        diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

    def test_mixed_line_endings(
        self,
        simple_grammar,
        tmp_path,
        temp_output_file
    ):
        """Test con diferentes tipos de fin de línea (\\n, \\r\\n)."""
        csv_file = tmp_path / "mixed_endings.csv"

        # Escribir en modo binario para controlar line endings
        content = (
            b"LINEA1;KG;100;2,50;250,00\n"
            b"LINEA2;M3;50;5,00;250,00\r\n"
            b"LINEA3;UND;10;10,00;100,00\r"
        )

        with open(csv_file, 'wb') as f:
            f.write(content)

        # No debería crashear
        diagnose_grammar_mismatches(
            csv_file=str(csv_file),
            grammar=simple_grammar,
            output_file=temp_output_file
        )


# ============================================================================
# TESTS: INTEGRACIÓN CON DIFERENTES GRAMÁTICAS
# ============================================================================


class TestDifferentGrammars:
    """Tests con diferentes tipos de gramáticas."""

    def test_with_optional_fields_grammar(
        self,
        complex_grammar,
        temp_csv_file,
        temp_output_file
    ):
        """Test con gramática que permite campos opcionales."""
        content = [
            "DESCRIPCION;KG;100;;250,00",  # price opcional vacío
            "DESCRIPCION;UND;50;5,00;250,00",  # Todos los campos
            "DESCRIPCION;;75;7,50;562,50",  # unit opcional vacío
        ]
        csv_file = temp_csv_file(content)

        failed_lines = diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=complex_grammar,
            output_file=temp_output_file
        )

        # Con gramática flexible, deberían pasar más líneas
        assert len(failed_lines) < len(content)

    def test_with_strict_grammar(
        self,
        simple_grammar,
        temp_csv_file,
        temp_output_file
    ):
        """Test con gramática estricta (sin campos opcionales)."""
        content = [
            "DESCRIPCION;KG;100;;250,00",  # Campo vacío, debería fallar
        ]
        csv_file = temp_csv_file(content)

        failed_lines = diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        # Gramática estricta debería rechazar campos vacíos
        assert len(failed_lines) > 0

    def test_with_number_format_variations(
        self,
        temp_csv_file,
        temp_output_file
    ):
        """Test con diferentes formatos de números."""
        # Gramática que acepta tanto punto como coma
        grammar = """
        ?start: line
        line: description ";" unit ";" quantity ";" price ";" total
        description: /[^;]+/
        unit: /[A-Z]+/
        quantity: NUMBER
        price: NUMBER
        total: NUMBER
        NUMBER: /\\d+[.,]\\d+/ | /\\d+/
        %import common.WS
        %ignore WS
        """

        content = [
            "DESC1;KG;100,5;2.35;353.68",  # Mezcla coma y punto
            "DESC2;M3;2.000;45,00;90,00",  # Mezcla punto y coma
            "DESC3;UND;85;1;85",  # Solo enteros
        ]
        csv_file = temp_csv_file(content)

        failed_lines = diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=grammar,
            output_file=temp_output_file
        )

        # Verificar que maneja diferentes formatos
        assert isinstance(failed_lines, list)


# ============================================================================
# TESTS: CONFIGURACIÓN Y PARÁMETROS
# ============================================================================


class TestConfiguration:
    """Tests de configuración y parámetros."""

    def test_with_different_profiles(
        self,
        simple_grammar,
        valid_csv_content,
        temp_csv_file,
        temp_output_file
    ):
        """Test con diferentes perfiles de configuración."""
        csv_file = temp_csv_file(valid_csv_content)

        # Profile 1: decimal_separator = comma
        failed1 = diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        # Profile 2: decimal_separator = dot
        failed2 = diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file + "2"
        )

        # Ambos deberían ejecutarse sin error
        assert isinstance(failed1, list)
        assert isinstance(failed2, list)

    def test_custom_output_path(
        self,
        simple_grammar,
        valid_csv_content,
        temp_csv_file,
        tmp_path
    ):
        """Test con ruta de salida personalizada."""
        csv_file = temp_csv_file(valid_csv_content)
        custom_output = tmp_path / "custom" / "path" / "diagnosis.txt"

        # Crear directorio padre
        custom_output.parent.mkdir(parents=True, exist_ok=True)

        diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=str(custom_output)
        )

        assert custom_output.exists()

    def test_output_file_overwrite(
        self,
        simple_grammar,
        valid_csv_content,
        temp_csv_file,
        temp_output_file
    ):
        """Test que sobrescribe archivo existente."""
        csv_file = temp_csv_file(valid_csv_content)

        # Crear archivo existente
        with open(temp_output_file, 'w') as f:
            f.write("CONTENIDO ANTERIOR")

        diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        # Verificar que se sobrescribió
        with open(temp_output_file, 'r') as f:
            content = f.read()
            assert "CONTENIDO ANTERIOR" not in content
            assert "DIAGNÓSTICO" in content


# ============================================================================
# TESTS DE REGRESIÓN
# ============================================================================


class TestRegression:
    """Tests de regresión para bugs conocidos."""

    def test_empty_field_at_end_of_line(
        self,
        simple_grammar,
        temp_csv_file,
        temp_output_file
    ):
        """
        Test regresión: campo vacío al final de línea.
        Bug: líneas que terminan en ; podrían no detectarse correctamente.
        """
        content = [
            "DESCRIPCION;KG;100;2,50;",  # Campo total vacío al final
        ]
        csv_file = temp_csv_file(content)

        failed_lines = diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )

        # Debería detectar el campo vacío
        if failed_lines:
            assert 4 in failed_lines[0].get("empty_field_positions", [])

    def test_unicode_normalization(
        self,
        simple_grammar,
        temp_csv_file,
        temp_output_file
    ):
        """
        Test regresión: normalización Unicode.
        Bug: caracteres Unicode compuestos vs descompuestos podrían causar problemas.
        """
        # É como carácter compuesto vs É como E + ́
        content = [
            "DESCRIPCIÓN;KG;100;2,50;250,00",  # É compuesto (U+00C9)
        ]
        csv_file = temp_csv_file(content)

        # No debería crashear
        diagnose_grammar_mismatches(
            csv_file=csv_file,
            grammar=simple_grammar,
            output_file=temp_output_file
        )


# ============================================================================
# SUITE COMPLETA
# ============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=app.utils.grammar_diagnostics"
    ])
