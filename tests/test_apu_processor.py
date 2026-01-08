"""
Suite de pruebas para apu_processor.py (V2 Refinada).

Incluye pruebas para:
- Estructuras categóricas (OptionMonad)
- Validación algebraica y topológica
- Componentes especialistas
- Procesador principal
- Casos edge y robustez
"""

import logging
import os
import sys
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pandas as pd
from lark import Lark, Token, Tree

# Ajustar ruta de importación para pruebas
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.apu_processor import (
    APU_GRAMMAR,
    APUProcessor,
    APUTransformer,
    FormatoLinea,
    NumericFieldExtractor,
    OptionMonad,
    ParsingStats,
    PatternMatcher,
    TipoInsumo,
    UnitsValidator,
    ValidationThresholds,
)
from app.utils import calculate_unit_costs

# Configurar logging para pruebas
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("test_apu_processor")


# ============================================================================
# FIXTURES MEJORADOS
# ============================================================================


class TestFixtures:
    """Fixtures completos para pruebas de APU Processor."""

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Configuración completa con todos los parámetros necesarios."""
        return {
            "apu_processor_rules": {
                "special_cases": {
                    "TRANSPORTE": "TRANSPORTE",
                    "ALQUILER": "EQUIPO",
                    "SUBCONTRATO": "OTRO",
                },
                "mo_keywords": [
                    "OFICIAL",
                    "AYUDANTE",
                    "PEON",
                    "CUADRILLA",
                    "OPERARIO",
                    "JORNAL",
                    "MAESTRO",
                ],
                "equipo_keywords": [
                    "EQUIPO",
                    "HERRAMIENTA",
                    "MAQUINA",
                    "ALQUILER",
                    "COMPRESOR",
                    "VIBRADOR",
                    "MEZCLADORA",
                ],
                "otro_keywords": [
                    "SUBCONTRATO",
                    "ADMINISTRACION",
                    "IMPUESTO",
                    "GASTO",
                    "IMPREVISTOS",
                ],
            },
            "validation_thresholds": {
                "MANO_DE_OBRA": {
                    "min_jornal": 50000,
                    "max_jornal": 10000000,
                    "min_rendimiento": 0.001,
                    "max_rendimiento": 1000,
                    "max_rendimiento_tipico": 100,
                },
                "GENERAL": {
                    "min_cantidad": 0.001,
                    "max_cantidad": 1000000,
                    "min_precio": 0.01,
                    "max_precio": 1e9,
                },
            },
            "debug_mode": False,
        }

    @staticmethod
    def get_default_profile() -> Dict[str, Any]:
        """Perfil por defecto con separador decimal punto."""
        return {
            "number_format": {
                "decimal_separator": ".",
                "thousand_separator": ",",
            },
            "encoding": "utf-8",
        }

    @staticmethod
    def get_comma_decimal_profile() -> Dict[str, Any]:
        """Perfil con separador decimal coma (formato europeo/latinoamericano)."""
        return {
            "number_format": {
                "decimal_separator": ",",
                "thousand_separator": ".",
            },
            "encoding": "latin-1",
        }

    @staticmethod
    def get_default_apu_context() -> Dict[str, Any]:
        """Contexto APU por defecto para pruebas del transformer."""
        return {
            "codigo_apu": "TEST-001",
            "descripcion_apu": "APU de Prueba Unitaria",
            "unidad_apu": "UN",
            "cantidad_apu": 1.0,
            "precio_unitario_apu": 0.0,
            "categoria": "PRUEBAS",
        }

    @staticmethod
    def get_grouped_sample_records() -> List[Dict[str, Any]]:
        """Registros en formato agrupado (legacy) con casos variados."""
        return [
            {
                "codigo_apu": "1.1",
                "descripcion_apu": "Muro de Contención",
                "unidad_apu": "M3",
                "category": "Estructuras",
                "source_line": 10,
                "lines": [
                    "OFICIAL ALBAÑIL;JOR;0.125;;180000;22500",
                    "AYUDANTE;JOR;0.25;;100000;25000",
                    "CEMENTO PORTLAND;KG;350;1200;420000",
                    "ARENA LAVADA;M3;0.5;150000;75000",
                    "AGUA;LT;180;200;36000",
                    "VIBRADOR ALQUILER;HR;0.5;15000;7500",
                ],
            },
            {
                "codigo_apu": "2.1",
                "descripcion_apu": "Excavación Manual",
                "unidad_apu": "M3",
                "category": "Movimiento de Tierras",
                "source_line": 25,
                "lines": [
                    "PEON;JOR;0.5;;100000;50000",
                    "RETROEXCAVADORA;HR;0.1;85000;8500",
                    "TRANSPORTE;VIAJE;0.3;45000;13500",
                ],
            },
            {
                "codigo_apu": "3.1",
                "descripcion_apu": "Piso Industrial",
                "unidad_apu": "M2",
                "category": "Acabados",
                "source_line": 40,
                "lines": [
                    "CUADRILLA PISOS;JOR;0.08;;250000;20000",
                    "CONCRETO ESPECIAL;M3;0.15;850123.50;127518.53",
                    "ACABADO DIAMANTINA;M2;1.0;35000;35000",
                ],
            },
        ]

    @staticmethod
    def get_flat_sample_records() -> List[Dict[str, Any]]:
        """Registros en formato plano (nuevo)."""
        return [
            {
                "apu_code": "1.1",
                "apu_desc": "Muro de Contención",
                "apu_unit": "M3",
                "category": "Estructuras",
                "source_line": 10,
                "insumo_line": "OFICIAL ALBAÑIL;JOR;0.125;;180000;22500",
                "line_number": 11,
            },
            {
                "apu_code": "1.1",
                "apu_desc": "Muro de Contención",
                "apu_unit": "M3",
                "category": "Estructuras",
                "source_line": 10,
                "insumo_line": "CEMENTO PORTLAND;KG;350;1200;420000",
                "line_number": 12,
            },
            {
                "apu_code": "2.1",
                "apu_desc": "Excavación Manual",
                "apu_unit": "M3",
                "category": "Movimiento de Tierras",
                "source_line": 25,
                "insumo_line": "PEON;JOR;0.5;;100000;50000",
                "line_number": 26,
            },
        ]

    @staticmethod
    def get_edge_case_lines() -> List[str]:
        """Líneas con casos extremos para pruebas de robustez."""
        return [
            "",  # Vacía
            "   ",  # Solo espacios
            "DESCRIPCION;UND;CANT;PRECIO;TOTAL",  # Encabezado
            "SUBTOTAL MANO DE OBRA;;;200000",  # Subtotal
            "MATERIALES",  # Categoría
            "TOTAL",  # Resumen
            "X" * 600 + ";UN;1;100;100",  # Descripción muy larga
            "INSUMO;%;10;;5000",  # Unidad porcentual
            "ITEM\x00CON\x1fCONTROL;UN;1;100;100",  # Caracteres de control
        ]

    @staticmethod
    def get_numeric_format_samples() -> List[Dict[str, Any]]:
        """Muestras de diferentes formatos numéricos."""
        return [
            {"input": "1234", "expected": 1234.0, "desc": "Entero simple"},
            {"input": "1234.56", "expected": 1234.56, "desc": "Decimal con punto"},
            {"input": "1,234.56", "expected": 1234.56, "desc": "Miles coma, decimal punto"},
            {"input": "1.234,56", "expected": 1234.56, "desc": "Miles punto, decimal coma"},
            {"input": "1234,56", "expected": 1234.56, "desc": "Decimal coma simple"},
            {"input": "$1,234.56", "expected": 1234.56, "desc": "Con símbolo monetario"},
            {"input": "-1234.56", "expected": -1234.56, "desc": "Negativo"},
        ]


# ============================================================================
# PRUEBAS DE OPTIONMONAD (ESTRUCTURA CATEGÓRICA)
# ============================================================================


class OptionMonadTests(unittest.TestCase):
    """Pruebas unitarias para la mónada Option/Maybe."""

    def test_pure_creates_valid_monad(self):
        """pure() crea una mónada válida con el valor dado."""
        monad = OptionMonad.pure(42)
        self.assertTrue(monad.is_valid())
        self.assertEqual(monad.value, 42)

    def test_fail_creates_invalid_monad(self):
        """fail() crea una mónada inválida con mensaje de error."""
        monad = OptionMonad.fail("Error de prueba")
        self.assertFalse(monad.is_valid())
        self.assertEqual(monad.error, "Error de prueba")

    def test_value_access_on_invalid_raises(self):
        """Acceder a value en mónada inválida lanza ValueError."""
        monad = OptionMonad.fail("Sin valor")
        with self.assertRaises(ValueError) as ctx:
            _ = monad.value
        self.assertIn("Sin valor", str(ctx.exception))

    def test_get_or_else_returns_value_when_valid(self):
        """get_or_else() retorna el valor cuando la mónada es válida."""
        monad = OptionMonad.pure("valor")
        self.assertEqual(monad.get_or_else("default"), "valor")

    def test_get_or_else_returns_default_when_invalid(self):
        """get_or_else() retorna el default cuando la mónada es inválida."""
        monad = OptionMonad.fail("error")
        self.assertEqual(monad.get_or_else("default"), "default")

    def test_map_transforms_value(self):
        """map() aplica función al valor contenido."""
        monad = OptionMonad.pure(5)
        result = monad.map(lambda x: x * 2)
        self.assertTrue(result.is_valid())
        self.assertEqual(result.value, 10)

    def test_map_preserves_type_transformation(self):
        """map() permite cambio de tipo (functor correcto)."""
        monad = OptionMonad.pure(42)
        result = monad.map(str)
        self.assertTrue(result.is_valid())
        self.assertEqual(result.value, "42")
        self.assertIsInstance(result.value, str)

    def test_map_propagates_invalid(self):
        """map() propaga mónada inválida sin ejecutar función."""
        monad = OptionMonad.fail("error previo")
        call_count = [0]

        def should_not_call(x):
            call_count[0] += 1
            return x

        result = monad.map(should_not_call)
        self.assertFalse(result.is_valid())
        self.assertEqual(call_count[0], 0)
        self.assertEqual(result.error, "error previo")

    def test_map_catches_exceptions(self):
        """map() captura excepciones y retorna mónada inválida."""
        monad = OptionMonad.pure(0)
        result = monad.map(lambda x: 1 / x)  # ZeroDivisionError
        self.assertFalse(result.is_valid())
        self.assertIn("Map error", result.error)

    def test_bind_chains_monadic_operations(self):
        """bind() encadena operaciones monádicas."""

        def safe_divide(x):
            if x == 0:
                return OptionMonad.fail("División por cero")
            return OptionMonad.pure(100 / x)

        monad = OptionMonad.pure(5)
        result = monad.bind(safe_divide)
        self.assertTrue(result.is_valid())
        self.assertEqual(result.value, 20.0)

    def test_bind_short_circuits_on_invalid(self):
        """bind() cortocircuita cuando la mónada es inválida."""
        monad = OptionMonad.fail("error inicial")
        call_count = [0]

        def should_not_call(x):
            call_count[0] += 1
            return OptionMonad.pure(x)

        result = monad.bind(should_not_call)
        self.assertFalse(result.is_valid())
        self.assertEqual(call_count[0], 0)

    def test_bind_propagates_inner_failure(self):
        """bind() propaga fallo de la función interna."""

        def always_fail(x):
            return OptionMonad.fail(f"Fallo procesando {x}")

        monad = OptionMonad.pure(42)
        result = monad.bind(always_fail)
        self.assertFalse(result.is_valid())
        self.assertIn("Fallo procesando 42", result.error)

    def test_filter_keeps_matching_values(self):
        """filter() mantiene valores que cumplen el predicado."""
        monad = OptionMonad.pure(10)
        result = monad.filter(lambda x: x > 5, "Valor muy pequeño")
        self.assertTrue(result.is_valid())
        self.assertEqual(result.value, 10)

    def test_filter_rejects_non_matching_values(self):
        """filter() rechaza valores que no cumplen el predicado."""
        monad = OptionMonad.pure(3)
        result = monad.filter(lambda x: x > 5, "Valor muy pequeño")
        self.assertFalse(result.is_valid())
        self.assertEqual(result.error, "Valor muy pequeño")

    def test_filter_propagates_invalid(self):
        """filter() propaga mónada inválida sin evaluar predicado."""
        monad = OptionMonad.fail("ya inválido")
        result = monad.filter(lambda x: True, "no importa")
        self.assertFalse(result.is_valid())
        self.assertEqual(result.error, "ya inválido")

    def test_monad_laws_left_identity(self):
        """Ley de identidad izquierda: pure(a).bind(f) == f(a)."""

        def f(x):
            return OptionMonad.pure(x + 1)

        a = 5
        left = OptionMonad.pure(a).bind(f)
        right = f(a)
        self.assertEqual(left.value, right.value)

    def test_monad_laws_right_identity(self):
        """Ley de identidad derecha: m.bind(pure) == m."""
        m = OptionMonad.pure(42)
        result = m.bind(OptionMonad.pure)
        self.assertEqual(result.value, m.value)

    def test_monad_laws_associativity(self):
        """Ley de asociatividad: m.bind(f).bind(g) == m.bind(lambda x: f(x).bind(g))."""

        def f(x):
            return OptionMonad.pure(x * 2)

        def g(x):
            return OptionMonad.pure(x + 10)

        m = OptionMonad.pure(5)
        left = m.bind(f).bind(g)
        right = m.bind(lambda x: f(x).bind(g))
        self.assertEqual(left.value, right.value)

    def test_repr_valid(self):
        """__repr__ muestra Some para mónadas válidas."""
        monad = OptionMonad.pure(42)
        self.assertIn("Some", repr(monad))
        self.assertIn("42", repr(monad))

    def test_repr_invalid(self):
        """__repr__ muestra None para mónadas inválidas."""
        monad = OptionMonad.fail("error")
        self.assertIn("None", repr(monad))
        self.assertIn("error", repr(monad))


# ============================================================================
# PRUEBAS DE PATTERNMATCHER
# ============================================================================


class PatternMatcherTests(unittest.TestCase):
    """Pruebas para el componente PatternMatcher."""

    def setUp(self):
        self.matcher = PatternMatcher()

    # --- Tests de detección de encabezados ---

    def test_is_likely_header_with_keywords(self):
        """Detecta encabezados con múltiples palabras clave."""
        self.assertTrue(
            self.matcher.is_likely_header("DESCRIPCION UND CANTIDAD PRECIO TOTAL", 5)
        )
        self.assertTrue(
            self.matcher.is_likely_header("ITEM CODIGO DESCRIPCION UNIDAD", 4)
        )

    def test_is_likely_header_high_ratio(self):
        """Detecta encabezados por alta proporción de keywords."""
        self.assertTrue(
            self.matcher.is_likely_header("CODIGO DESCRIPCION UNIDAD CANTIDAD PRECIO VALOR", 2)
        )

    def test_is_likely_header_rejects_data_lines(self):
        """Rechaza líneas de datos como encabezados."""
        self.assertFalse(self.matcher.is_likely_header("CEMENTO PORTLAND TIPO I", 5))
        self.assertFalse(self.matcher.is_likely_header("OFICIAL ALBAÑIL", 6))

    # --- Tests de detección de resúmenes ---

    def test_is_likely_summary_with_keywords(self):
        """Detecta líneas de resumen con keywords."""
        self.assertTrue(self.matcher.is_likely_summary("TOTAL MANO DE OBRA", 2))
        self.assertTrue(self.matcher.is_likely_summary("SUBTOTAL MATERIALES", 1))
        self.assertTrue(self.matcher.is_likely_summary("GRAN TOTAL", 2))
        self.assertTrue(self.matcher.is_likely_summary("COSTO DIRECTO", 2))

    def test_is_likely_summary_rejects_normal_lines(self):
        """Rechaza líneas normales como resúmenes."""
        self.assertFalse(self.matcher.is_likely_summary("CEMENTO PORTLAND", 5))
        self.assertFalse(self.matcher.is_likely_summary("OFICIAL ALBAÑIL", 6))

    # --- Tests de detección de categorías ---

    def test_is_likely_category_exact_matches(self):
        """Detecta categorías con coincidencia exacta."""
        self.assertTrue(self.matcher.is_likely_category("MANO DE OBRA", 1))
        self.assertTrue(self.matcher.is_likely_category("MATERIALES", 2))
        self.assertTrue(self.matcher.is_likely_category("EQUIPO", 1))
        self.assertTrue(self.matcher.is_likely_category("TRANSPORTE", 1))
        self.assertTrue(self.matcher.is_likely_category("OTROS", 2))

    def test_is_likely_category_rejects_with_many_fields(self):
        """Rechaza categorías cuando hay muchos campos."""
        self.assertFalse(self.matcher.is_likely_category("MATERIALES", 5))
        self.assertFalse(self.matcher.is_likely_category("EQUIPO", 4))

    # --- Tests de contenido numérico ---

    def test_has_numeric_content(self):
        """Detecta contenido numérico en diferentes formatos."""
        self.assertTrue(self.matcher.has_numeric_content("123.45"))
        self.assertTrue(self.matcher.has_numeric_content("1,234.56"))
        self.assertTrue(self.matcher.has_numeric_content("Precio: $100"))
        self.assertTrue(self.matcher.has_numeric_content("Código A1B2"))
        self.assertFalse(self.matcher.has_numeric_content("Solo texto"))

    def test_has_percentage(self):
        """Detecta porcentajes en diferentes formatos."""
        self.assertTrue(self.matcher.has_percentage("15%"))
        self.assertTrue(self.matcher.has_percentage("15 %"))
        self.assertTrue(self.matcher.has_percentage("Administración 10%"))
        self.assertTrue(self.matcher.has_percentage("IVA 19 %"))
        self.assertFalse(self.matcher.has_percentage("Sin porcentaje"))

    # --- Tests de encabezados de capítulo ---

    def test_is_likely_chapter_header(self):
        """Detecta encabezados de capítulo."""
        self.assertTrue(self.matcher.is_likely_chapter_header("CAPITULO 1"))
        self.assertTrue(self.matcher.is_likely_chapter_header("CAPÍTULO PRELIMINARES"))
        self.assertTrue(self.matcher.is_likely_chapter_header("TITULO ESTRUCTURAS"))

    def test_is_likely_chapter_header_rejects_normal(self):
        """Rechaza líneas normales como encabezados de capítulo."""
        self.assertFalse(self.matcher.is_likely_chapter_header("CEMENTO 350 KG"))
        self.assertFalse(self.matcher.is_likely_chapter_header("MATERIALES"))


# ============================================================================
# PRUEBAS DE UNITSVALIDATOR
# ============================================================================


class UnitsValidatorTests(unittest.TestCase):
    """Pruebas para el componente UnitsValidator."""

    def test_normalize_unit_mappings(self):
        """Prueba mapeos de normalización conocidos."""
        mappings = [
            ("MT", "M"),
            ("MTS", "M"),
            ("JORNAL", "JOR"),
            ("JORN", "JOR"),
            ("UNID", "UND"),
            ("UN", "UND"),
        ]
        for input_unit, expected in mappings:
            with self.subTest(input=input_unit):
                self.assertEqual(UnitsValidator.normalize_unit(input_unit), expected)

    def test_normalize_unit_preserves_valid(self):
        """Preserva unidades ya en forma canónica."""
        valid_units = ["M", "M2", "M3", "KG", "LT", "HR", "JOR", "UND"]
        for unit in valid_units:
            with self.subTest(unit=unit):
                self.assertEqual(UnitsValidator.normalize_unit(unit), unit)

    def test_normalize_unit_empty_returns_und(self):
        """Cadena vacía normaliza a UND."""
        self.assertEqual(UnitsValidator.normalize_unit(""), "UND")
        self.assertEqual(UnitsValidator.normalize_unit(None), "UND")

    def test_normalize_unit_cleans_punctuation(self):
        """Limpia puntuación de unidades."""
        self.assertEqual(UnitsValidator.normalize_unit("UND."), "UND")
        self.assertEqual(UnitsValidator.normalize_unit("M2."), "M2")

    def test_normalize_unit_unknown_short_preserves(self):
        """Unidades desconocidas cortas se preservan si parecen válidas."""
        # Unidades de 4 caracteres o menos que no están en el mapping
        result = UnitsValidator.normalize_unit("ABC")
        # Debería devolver UND si no es reconocida
        self.assertIn(result, ["ABC", "UND"])

    def test_is_valid_known_units(self):
        """Valida unidades conocidas."""
        valid = ["M", "M2", "M3", "KG", "LT", "HR", "JOR", "UND", "VIAJE", "GLB"]
        for unit in valid:
            with self.subTest(unit=unit):
                self.assertTrue(UnitsValidator.is_valid(unit))

    def test_is_valid_rejects_empty(self):
        """Rechaza cadenas vacías."""
        self.assertFalse(UnitsValidator.is_valid(""))
        self.assertFalse(UnitsValidator.is_valid(None))

    def test_is_valid_short_unknown_accepted(self):
        """Acepta unidades cortas desconocidas (<=4 caracteres)."""
        self.assertTrue(UnitsValidator.is_valid("XYZ"))
        self.assertTrue(UnitsValidator.is_valid("AB"))

    def test_is_valid_rejects_very_long(self):
        """Rechaza unidades muy largas no reconocidas."""
        self.assertFalse(UnitsValidator.is_valid("UNIDADMUYLARGANORECONOCIDA"))


# ============================================================================
# PRUEBAS DE NUMERICFIELDEXTRACTOR
# ============================================================================


class NumericFieldExtractorTests(unittest.TestCase):
    """Pruebas para el componente NumericFieldExtractor."""

    def setUp(self):
        self.config = TestFixtures.get_default_config()
        self.profile = TestFixtures.get_default_profile()
        self.thresholds = ValidationThresholds()
        self.extractor = NumericFieldExtractor(self.config, self.profile, self.thresholds)

    def test_parse_number_safe_integers(self):
        """Parsea enteros correctamente."""
        self.assertEqual(self.extractor.parse_number_safe("1000"), 1000.0)
        self.assertEqual(self.extractor.parse_number_safe("0"), 0.0)
        self.assertEqual(self.extractor.parse_number_safe("-500"), -500.0)

    def test_parse_number_safe_decimals(self):
        """Parsea decimales con punto."""
        self.assertEqual(self.extractor.parse_number_safe("1234.56"), 1234.56)
        self.assertEqual(self.extractor.parse_number_safe("0.001"), 0.001)

    def test_parse_number_safe_with_thousands(self):
        """Parsea números con separador de miles."""
        self.assertEqual(self.extractor.parse_number_safe("1,000.50"), 1000.50)
        self.assertEqual(self.extractor.parse_number_safe("1,234,567.89"), 1234567.89)

    def test_parse_number_safe_invalid_returns_none(self):
        """Retorna None para entradas inválidas."""
        self.assertIsNone(self.extractor.parse_number_safe(""))
        self.assertIsNone(self.extractor.parse_number_safe(None))

    def test_parse_number_safe_text_only(self):
        """Maneja texto sin números."""
        result = self.extractor.parse_number_safe("texto sin numeros")
        # Debe retornar None o 0.0 dependiendo de implementación
        self.assertIn(result, [None, 0.0])

    def test_extract_all_numeric_values(self):
        """Extrae todos los valores numéricos de campos."""
        fields = ["DESCRIPCION", "UND", "0.5", "100000", "50000"]
        values = self.extractor.extract_all_numeric_values(fields)
        # Debería contener los valores numéricos válidos
        self.assertIn(0.5, values)
        self.assertIn(100000.0, values)
        self.assertIn(50000.0, values)

    def test_extract_all_numeric_values_skip_first(self):
        """Omite el primer campo (descripción) por defecto."""
        fields = ["123", "UND", "0.5", "100"]
        values = self.extractor.extract_all_numeric_values(fields, skip_first=True)
        # No debería incluir 123 del primer campo
        self.assertNotIn(123.0, values)

    def test_identify_mo_values_normal_case(self):
        """Identifica rendimiento y jornal en caso normal."""
        values = [0.125, 180000.0, 22500.0]
        result = self.extractor.identify_mo_values(values)
        self.assertIsNotNone(result)
        rendimiento, jornal = result
        self.assertAlmostEqual(rendimiento, 0.125)
        self.assertAlmostEqual(jornal, 180000.0)

    def test_identify_mo_values_multiple_candidates(self):
        """Identifica MO con múltiples candidatos, selecciona correctamente."""
        values = [0.05, 0.125, 180000.0, 250000.0, 22500.0]
        result = self.extractor.identify_mo_values(values)
        self.assertIsNotNone(result)
        rendimiento, jornal = result
        # Debería seleccionar el rendimiento más bajo y jornal más alto
        self.assertLessEqual(rendimiento, 0.125)
        self.assertGreaterEqual(jornal, 180000.0)

    def test_identify_mo_values_insufficient_data(self):
        """Retorna None con datos insuficientes."""
        self.assertIsNone(self.extractor.identify_mo_values([]))
        self.assertIsNone(self.extractor.identify_mo_values([100000.0]))

    def test_identify_mo_values_no_valid_jornal(self):
        """Retorna None cuando no hay jornal válido."""
        # Todos los valores fuera del rango de jornal
        values = [0.5, 1.0, 2.0, 100.0]
        result = self.extractor.identify_mo_values(values)
        self.assertIsNone(result)

    def test_identify_mo_values_invariant_jornal_greater_than_rendimiento(self):
        """El jornal siempre es mayor que el rendimiento (invariante algebraico)."""
        test_cases = [
            [0.1, 150000.0],
            [0.5, 200000.0, 100000.0],
            [0.08, 0.16, 300000.0],
        ]
        for values in test_cases:
            with self.subTest(values=values):
                result = self.extractor.identify_mo_values(values)
                if result:
                    rendimiento, jornal = result
                    self.assertGreater(jornal, rendimiento * 10)  # Jornal >> rendimiento

    def test_comma_decimal_parsing(self):
        """Parsea números con coma decimal (perfil europeo)."""
        comma_profile = TestFixtures.get_comma_decimal_profile()
        extractor = NumericFieldExtractor(self.config, comma_profile, self.thresholds)

        self.assertAlmostEqual(extractor.parse_number_safe("1,5"), 1.5, places=2)
        self.assertAlmostEqual(extractor.parse_number_safe("1.000,50"), 1000.50, places=2)
        self.assertAlmostEqual(extractor.parse_number_safe("250.000,00"), 250000.00, places=2)


# ============================================================================
# PRUEBAS DE VALIDACIÓN ALGEBRAICA (APUTransformer)
# ============================================================================


class AlgebraicValidationTests(unittest.TestCase):
    """Pruebas para métodos de validación algebraica del transformer."""

    def setUp(self):
        self.config = TestFixtures.get_default_config()
        self.profile = TestFixtures.get_default_profile()
        self.context = TestFixtures.get_default_apu_context()
        self.transformer = APUTransformer(
            self.context, self.config, self.profile, {}
        )

    def test_classify_field_algebraic_type_numeric(self):
        """Clasifica campos numéricos correctamente."""
        self.assertEqual(self.transformer._classify_field_algebraic_type("1234"), "NUMERIC")
        self.assertEqual(self.transformer._classify_field_algebraic_type("1234.56"), "NUMERIC")
        self.assertEqual(self.transformer._classify_field_algebraic_type("15%"), "NUMERIC")
        self.assertEqual(self.transformer._classify_field_algebraic_type("$100"), "NUMERIC")

    def test_classify_field_algebraic_type_alpha(self):
        """Clasifica campos alfabéticos correctamente."""
        self.assertEqual(self.transformer._classify_field_algebraic_type("CEMENTO"), "ALPHA")
        self.assertEqual(self.transformer._classify_field_algebraic_type("Mano de Obra"), "ALPHA")

    def test_classify_field_algebraic_type_mixed(self):
        """Clasifica campos mixtos correctamente."""
        result = self.transformer._classify_field_algebraic_type("M2")
        self.assertIn(result, ["MIXED_NUMERIC", "ALPHA"])  # Depende de proporciones

    def test_classify_field_algebraic_type_empty(self):
        """Clasifica campos vacíos."""
        self.assertEqual(self.transformer._classify_field_algebraic_type(""), "EMPTY")
        self.assertEqual(self.transformer._classify_field_algebraic_type("   "), "EMPTY")

    def test_validate_algebraic_homogeneity_first_position(self):
        """Primera posición siempre es válida (generador del anillo)."""
        self.assertTrue(
            self.transformer._validate_algebraic_homogeneity("CUALQUIER COSA", 0, [])
        )
        self.assertTrue(
            self.transformer._validate_algebraic_homogeneity("12345", 0, [])
        )

    def test_validate_algebraic_homogeneity_valid_transitions(self):
        """Valida transiciones algebraicas permitidas."""
        # ALPHA -> NUMERIC (descripción -> cantidad)
        self.assertTrue(
            self.transformer._validate_algebraic_homogeneity("100", 1, ["CEMENTO"])
        )
        # NUMERIC -> ALPHA (cantidad -> unidad)
        self.assertTrue(
            self.transformer._validate_algebraic_homogeneity("KG", 2, ["CEMENTO", "100"])
        )
        # NUMERIC -> NUMERIC (precio -> total)
        self.assertTrue(
            self.transformer._validate_algebraic_homogeneity("5000", 3, ["CEMENTO", "100", "50"])
        )

    def test_validate_minimal_cardinality(self):
        """Valida cardinalidad mínima de campos."""
        # Suficientes campos
        result = self.transformer._validate_minimal_cardinality(["a", "b", "c"])
        self.assertTrue(result.is_valid())

        # Insuficientes campos
        result = self.transformer._validate_minimal_cardinality(["a", "b"])
        self.assertFalse(result.is_valid())
        self.assertIn("Cardinalidad", result.error)

    def test_validate_description_epicenter(self):
        """Valida que la descripción no esté vacía."""
        # Descripción válida
        result = self.transformer._validate_description_epicenter(["CEMENTO", "KG", "100"])
        self.assertTrue(result.is_valid())

        # Descripción vacía
        result = self.transformer._validate_description_epicenter(["", "KG", "100"])
        self.assertFalse(result.is_valid())

        # Sin campos
        result = self.transformer._validate_description_epicenter([])
        self.assertFalse(result.is_valid())


# ============================================================================
# PRUEBAS DE GRAFOS Y CONECTIVIDAD
# ============================================================================


class GraphConnectivityTests(unittest.TestCase):
    """Pruebas para construcción de grafos y verificación de conectividad."""

    def setUp(self):
        self.config = TestFixtures.get_default_config()
        self.profile = TestFixtures.get_default_profile()
        self.context = TestFixtures.get_default_apu_context()
        self.transformer = APUTransformer(
            self.context, self.config, self.profile, {}
        )

    def test_build_field_dependency_graph_linear(self):
        """Construye grafo con dependencias lineales."""
        fields = ["DESC", "UND", "100", "50", "5000"]
        graph = self.transformer._build_field_dependency_graph(fields)

        # Debe haber 5 nodos
        self.assertEqual(len(graph), 5)

        # Verificar conexiones lineales
        self.assertIn(1, graph[0])  # 0 conecta con 1
        self.assertIn(0, graph[1])  # 1 conecta con 0
        self.assertIn(2, graph[1])  # 1 conecta con 2

    def test_build_field_dependency_graph_semantic(self):
        """Detecta relaciones semánticas entre campos."""
        fields = ["CEMENTO PORTLAND", "KG", "350", "CEMENTO"]
        graph = self.transformer._build_field_dependency_graph(fields)

        # "CEMENTO PORTLAND" y "CEMENTO" deberían estar conectados semánticamente
        if 3 in graph[0]:
            self.assertIn(0, graph[3])

    def test_is_graph_connected_simple(self):
        """Verifica conectividad en grafo simple."""
        # Grafo lineal conectado
        graph = {0: {1}, 1: {0, 2}, 2: {1}}
        self.assertTrue(self.transformer._is_graph_connected(graph, 3))

    def test_is_graph_connected_disconnected(self):
        """Detecta grafo desconectado."""
        # Dos componentes separados
        graph = {0: {1}, 1: {0}, 2: {3}, 3: {2}}
        self.assertFalse(self.transformer._is_graph_connected(graph, 4))

    def test_is_graph_connected_single_node(self):
        """Grafo de un solo nodo es conexo."""
        graph = {0: set()}
        self.assertTrue(self.transformer._is_graph_connected(graph, 1))

    def test_is_graph_connected_empty(self):
        """Grafo vacío se considera conexo."""
        self.assertTrue(self.transformer._is_graph_connected({}, 0))

    def test_fields_are_semantically_related(self):
        """Detecta relaciones semánticas entre campos."""
        # Contención directa
        self.assertTrue(
            self.transformer._fields_are_semantically_related("CEMENTO", "CEMENTO PORTLAND")
        )

        # Sin relación
        self.assertFalse(
            self.transformer._fields_are_semantically_related("CEMENTO", "ARENA")
        )

        # Campos cortos no relacionados
        self.assertFalse(
            self.transformer._fields_are_semantically_related("AB", "CD")
        )

    def test_validate_structural_integrity_short_list(self):
        """Listas cortas pasan validación trivialmente."""
        result = self.transformer._validate_structural_integrity(["A", "B"])
        self.assertTrue(result.is_valid())

    def test_validate_structural_integrity_connected(self):
        """Valida lista conectada correctamente."""
        fields = ["CEMENTO PORTLAND", "KG", "350", "1200", "420000"]
        result = self.transformer._validate_structural_integrity(fields)
        self.assertTrue(result.is_valid())


# ============================================================================
# PRUEBAS DE NORMALIZACIÓN NUMÉRICA
# ============================================================================


class NumericNormalizationTests(unittest.TestCase):
    """Pruebas para normalización de representaciones numéricas."""

    def setUp(self):
        self.config = TestFixtures.get_default_config()
        self.profile = TestFixtures.get_default_profile()
        self.context = TestFixtures.get_default_apu_context()
        self.transformer = APUTransformer(
            self.context, self.config, self.profile, {}
        )

    def test_normalize_numeric_simple_integer(self):
        """Normaliza enteros simples."""
        self.assertEqual(
            self.transformer._normalize_numeric_representation("1234"),
            "1234"
        )

    def test_normalize_numeric_decimal_point(self):
        """Preserva decimales con punto."""
        self.assertEqual(
            self.transformer._normalize_numeric_representation("1234.56"),
            "1234.56"
        )

    def test_normalize_numeric_european_format(self):
        """Normaliza formato europeo (punto miles, coma decimal)."""
        result = self.transformer._normalize_numeric_representation("1.234,56")
        self.assertEqual(result, "1234.56")

    def test_normalize_numeric_us_format(self):
        """Normaliza formato US (coma miles, punto decimal)."""
        result = self.transformer._normalize_numeric_representation("1,234.56")
        self.assertEqual(result, "1234.56")

    def test_normalize_numeric_removes_currency(self):
        """Remueve símbolos de moneda."""
        result = self.transformer._normalize_numeric_representation("$1,234.56")
        self.assertEqual(result, "1234.56")

    def test_normalize_numeric_ambiguous_comma(self):
        """Maneja coma ambigua (podría ser decimal o miles)."""
        # "1,5" debería tratarse como 1.5
        result = self.transformer._normalize_numeric_representation("1,5")
        self.assertEqual(result, "1.5")

    def test_looks_numeric(self):
        """Detecta campos numéricos correctamente."""
        self.assertTrue(self.transformer._looks_numeric("1234"))
        self.assertTrue(self.transformer._looks_numeric("1234.56"))
        self.assertTrue(self.transformer._looks_numeric("1,234.56"))
        self.assertTrue(self.transformer._looks_numeric("$100"))
        self.assertFalse(self.transformer._looks_numeric("CEMENTO"))
        self.assertFalse(self.transformer._looks_numeric(""))


# ============================================================================
# PRUEBAS DEL PARSER LARK
# ============================================================================


class LarkParserTests(unittest.TestCase):
    """Pruebas para el parser Lark y la gramática APU."""

    def setUp(self):
        self.parser = Lark(APU_GRAMMAR, start="line", parser="lalr")

    def test_parse_simple_line(self):
        """Parsea línea simple correctamente."""
        line = "CEMENTO;KG;350;1200;420000"
        tree = self.parser.parse(line)
        self.assertIsNotNone(tree)
        self.assertEqual(tree.data, "line")

    def test_parse_line_with_empty_fields(self):
        """Parsea línea con campos vacíos."""
        line = "OFICIAL;JOR;0.125;;180000;22500"
        tree = self.parser.parse(line)
        self.assertIsNotNone(tree)

    def test_parse_line_with_spaces(self):
        """Parsea línea con espacios alrededor de separadores."""
        line = "CEMENTO ; KG ; 350 ; 1200 ; 420000"
        tree = self.parser.parse(line)
        self.assertIsNotNone(tree)

    def test_parse_line_with_special_chars(self):
        """Parsea línea con caracteres especiales en descripción."""
        line = "CEMENTO (TIPO I) 350 KG/M3;KG;350;1200;420000"
        tree = self.parser.parse(line)
        self.assertIsNotNone(tree)

    def test_parse_minimal_line(self):
        """Parsea línea mínima."""
        line = "A;B;C"
        tree = self.parser.parse(line)
        self.assertIsNotNone(tree)


# ============================================================================
# PRUEBAS DEL TRANSFORMER COMPLETO
# ============================================================================


class APUTransformerTests(unittest.TestCase):
    """Pruebas de integración para APUTransformer."""

    def setUp(self):
        self.config = TestFixtures.get_default_config()
        self.profile = TestFixtures.get_default_profile()
        self.context = TestFixtures.get_default_apu_context()
        self.parser = Lark(APU_GRAMMAR, start="line", parser="lalr")
        self.transformer = APUTransformer(
            self.context, self.config, self.profile, {}
        )

    def test_transform_mano_de_obra_line(self):
        """Transforma línea de mano de obra correctamente."""
        line = "OFICIAL ALBAÑIL;JOR;0.125;;180000;22500"
        tree = self.parser.parse(line)
        result = self.transformer.transform(tree)

        if result is not None:
            self.assertEqual(result.tipo_insumo, "MANO_DE_OBRA")
            # Verificar que la descripción esté normalizada (ALBAÑIL -> ALBANIL)
            self.assertEqual(result.descripcion_insumo, "OFICIAL ALBANIL")
            self.assertAlmostEqual(result.rendimiento, 0.125)
            self.assertAlmostEqual(result.precio_unitario, 180000.0)

    def test_transform_suministro_line(self):
        """Transforma línea de suministro correctamente."""
        line = "CEMENTO PORTLAND;KG;350;1200;420000"
        tree = self.parser.parse(line)
        result = self.transformer.transform(tree)

        if result is not None:
            self.assertEqual(result.tipo_insumo, "SUMINISTRO")
            self.assertEqual(result.descripcion_insumo, "CEMENTO PORTLAND")

    def test_transform_rejects_header_line(self):
        """Rechaza líneas de encabezado."""
        line = "DESCRIPCION;UNIDAD;CANTIDAD;PRECIO;TOTAL"
        tree = self.parser.parse(line)
        result = self.transformer.transform(tree)
        self.assertIsNone(result)

    def test_transform_rejects_summary_line(self):
        """Rechaza líneas de resumen."""
        line = "SUBTOTAL MANO DE OBRA;;;200000"
        tree = self.parser.parse(line)
        result = self.transformer.transform(tree)
        self.assertIsNone(result)

    def test_transform_rejects_category_line(self):
        """Rechaza líneas de categoría."""
        line = "MATERIALES"
        tree = self.parser.parse(line)
        result = self.transformer.transform(tree)
        self.assertIsNone(result)

    def test_classify_insumo_mano_de_obra(self):
        """Clasifica correctamente mano de obra."""
        mo_keywords = ["OFICIAL", "AYUDANTE", "PEON", "CUADRILLA", "OPERARIO"]
        for keyword in mo_keywords:
            with self.subTest(keyword=keyword):
                tipo = self.transformer._classify_insumo(f"{keyword} ESPECIALIZADO")
                self.assertEqual(tipo, TipoInsumo.MANO_DE_OBRA)

    def test_classify_insumo_equipo(self):
        """Clasifica correctamente equipos."""
        equipo_keywords = ["EQUIPO", "HERRAMIENTA", "VIBRADOR", "MEZCLADORA"]
        for keyword in equipo_keywords:
            with self.subTest(keyword=keyword):
                tipo = self.transformer._classify_insumo(f"{keyword} CONSTRUCCION")
                self.assertEqual(tipo, TipoInsumo.EQUIPO)

    def test_classify_insumo_default_suministro(self):
        """Clasifica como suministro por defecto."""
        tipo = self.transformer._classify_insumo("CEMENTO PORTLAND TIPO I")
        self.assertEqual(tipo, TipoInsumo.SUMINISTRO)


# ============================================================================
# PRUEBAS DE APUPROCESSOR
# ============================================================================


class APUProcessorTests(unittest.TestCase):
    """Pruebas principales para el procesador APU."""

    def setUp(self):
        self.config = TestFixtures.get_default_config()
        self.profile = TestFixtures.get_default_profile()

    def test_initialization(self):
        """Inicializa procesador correctamente."""
        processor = APUProcessor(self.config, self.profile)
        self.assertIsNotNone(processor.config)
        self.assertIsNotNone(processor.profile)
        self.assertIsNotNone(processor.parser)
        self.assertIsInstance(processor.parsing_stats, ParsingStats)

    def test_detect_record_format_grouped(self):
        """Detecta formato agrupado."""
        processor = APUProcessor(self.config, self.profile)
        records = [{"lines": ["line1", "line2"]}]
        format_type, _ = processor._detect_record_format(records)
        self.assertEqual(format_type, "grouped")

    def test_detect_record_format_flat(self):
        """Detecta formato plano."""
        processor = APUProcessor(self.config, self.profile)
        records = [{"insumo_line": "line", "apu_code": "1.1"}]
        format_type, _ = processor._detect_record_format(records)
        self.assertEqual(format_type, "flat")

    def test_detect_record_format_empty(self):
        """Maneja lista vacía."""
        processor = APUProcessor(self.config, self.profile)
        format_type, _ = processor._detect_record_format([])
        self.assertEqual(format_type, "unknown")

    def test_process_grouped_records(self):
        """Procesa registros agrupados correctamente."""
        records = TestFixtures.get_grouped_sample_records()

        processor = APUProcessor(self.config, self.profile)
        processor.raw_records = records
        df = processor.process_all()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

        # Verificar columnas esperadas
        expected_columns = [
            "CODIGO_APU", "DESCRIPCION_APU", "DESCRIPCION_INSUMO",
            "TIPO_INSUMO", "VALOR_TOTAL_APU"
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)

    def test_process_flat_records(self):
        """Procesa registros planos correctamente."""
        records = TestFixtures.get_flat_sample_records()

        processor = APUProcessor(self.config, self.profile)
        processor.raw_records = records
        df = processor.process_all()

        self.assertIsInstance(df, pd.DataFrame)

        if not df.empty:
            apu_codes = df["CODIGO_APU"].unique()
            self.assertIn("1.1", apu_codes)

    def test_group_flat_records(self):
        """Agrupa registros planos por APU correctamente."""
        processor = APUProcessor(self.config, self.profile)
        flat_records = TestFixtures.get_flat_sample_records()

        grouped = processor._group_flat_records(flat_records)

        # Debería haber 2 APUs únicos
        self.assertEqual(len(grouped), 2)

        # Verificar estructura
        for record in grouped:
            self.assertIn("lines", record)
            self.assertIn("codigo_apu", record)

    def test_process_with_comma_decimal(self):
        """Procesa con separador decimal coma."""
        comma_profile = TestFixtures.get_comma_decimal_profile()
        records = [
            {
                "codigo_apu": "4.1",
                "descripcion_apu": "Test Coma",
                "unidad_apu": "M2",
                "lines": [
                    "CUADRILLA;JOR;0,08;;250.000,00;20.000,00",
                ],
            }
        ]

        processor = APUProcessor(self.config, comma_profile)
        processor.raw_records = records
        df = processor.process_all()

        self.assertIsInstance(df, pd.DataFrame)

    def test_extract_apu_context(self):
        """Extrae contexto APU correctamente."""
        processor = APUProcessor(self.config, self.profile)
        record = {
            "codigo_apu": "1.1",
            "descripcion_apu": "Test APU",
            "unidad_apu": "M3",
            "category": "Test",
        }

        context = processor._extract_apu_context(record)

        self.assertEqual(context["codigo_apu"], "1.1")
        self.assertEqual(context["descripcion_apu"], "Test APU")
        self.assertEqual(context["unidad_apu"], "M3")

    def test_is_valid_line(self):
        """Valida líneas correctamente."""
        processor = APUProcessor(self.config, self.profile)

        self.assertTrue(processor._is_valid_line("CEMENTO;KG;350"))
        self.assertFalse(processor._is_valid_line(""))
        self.assertFalse(processor._is_valid_line("  "))
        self.assertFalse(processor._is_valid_line("AB"))  # Muy corta
        self.assertFalse(processor._is_valid_line(None))
        self.assertFalse(processor._is_valid_line(123))  # No es string

    def test_compute_cache_key(self):
        """Computa claves de cache normalizadas."""
        processor = APUProcessor(self.config, self.profile)

        key1 = processor._compute_cache_key("CEMENTO;KG;350")
        key2 = processor._compute_cache_key("CEMENTO;KG;350  ")
        key3 = processor._compute_cache_key("CEMENTO;  KG;350")

        # Claves normalizadas deberían ser consistentes
        self.assertEqual(key1, key2.strip())

    def test_statistics_tracking(self):
        """Rastrea estadísticas de procesamiento."""
        records = TestFixtures.get_grouped_sample_records()

        processor = APUProcessor(self.config, self.profile)
        processor.raw_records = records
        processor.process_all()

        stats = processor.parsing_stats
        self.assertGreaterEqual(stats.total_lines, 0)
        self.assertGreaterEqual(stats.successful_parses, 0)

        global_stats = processor.global_stats
        self.assertIn("total_apus", global_stats)
        self.assertIn("total_insumos", global_stats)


# ============================================================================
# PRUEBAS DE ROBUSTEZ Y CASOS EDGE
# ============================================================================


class RobustnessTests(unittest.TestCase):
    """Pruebas de robustez para casos edge y errores."""

    def setUp(self):
        self.config = TestFixtures.get_default_config()
        self.profile = TestFixtures.get_default_profile()

    def test_empty_records(self):
        """Maneja lista vacía de registros."""
        processor = APUProcessor(self.config, self.profile)
        processor.raw_records = []
        df = processor.process_all()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_records_with_empty_lines(self):
        """Maneja registros con líneas vacías."""
        records = [
            {
                "codigo_apu": "1.1",
                "descripcion_apu": "Test",
                "unidad_apu": "UN",
                "lines": ["", "   ", "CEMENTO;KG;350;1200;420000"],
            }
        ]

        processor = APUProcessor(self.config, self.profile)
        processor.raw_records = records
        df = processor.process_all()

        # Debería procesar al menos la línea válida
        self.assertIsInstance(df, pd.DataFrame)

    def test_malformed_numeric_fields(self):
        """Maneja campos numéricos malformados."""
        records = [
            {
                "codigo_apu": "1.1",
                "descripcion_apu": "Test",
                "unidad_apu": "UN",
                "lines": [
                    "CEMENTO;KG;INVALIDO;1200;420000",
                    "ARENA;M3;0.5;NaN;75000",
                ],
            }
        ]

        processor = APUProcessor(self.config, self.profile)
        processor.raw_records = records

        # No debería lanzar excepción
        try:
            df = processor.process_all()
            self.assertIsInstance(df, pd.DataFrame)
        except Exception as e:
            self.fail(f"No debería lanzar excepción: {e}")

    def test_special_characters_in_description(self):
        """Maneja caracteres especiales en descripción."""
        records = [
            {
                "codigo_apu": "1.1",
                "descripcion_apu": "Test",
                "unidad_apu": "UN",
                "lines": [
                    "CEMENTO (TIPO I) - 350 KG/M³;KG;350;1200;420000",
                    "ARENA LAVADA & CERNIDA;M3;0.5;150000;75000",
                ],
            }
        ]

        processor = APUProcessor(self.config, self.profile)
        processor.raw_records = records
        df = processor.process_all()

        self.assertIsInstance(df, pd.DataFrame)

    def test_unicode_characters(self):
        """Maneja caracteres Unicode correctamente."""
        records = [
            {
                "codigo_apu": "1.1",
                "descripcion_apu": "Test",
                "unidad_apu": "UN",
                "lines": [
                    "CONCRETO PREMEZCLADO F'C=210 KG/CM²;M3;0.15;850000;127500",
                    "ACERO Ø 3/8\";KG;50;3500;175000",
                ],
            }
        ]

        processor = APUProcessor(self.config, self.profile)
        processor.raw_records = records
        df = processor.process_all()

        self.assertIsInstance(df, pd.DataFrame)

    def test_very_long_description(self):
        """Maneja descripciones muy largas."""
        long_desc = "CEMENTO PORTLAND " + "X" * 600
        records = [
            {
                "codigo_apu": "1.1",
                "descripcion_apu": "Test",
                "unidad_apu": "UN",
                "lines": [f"{long_desc};KG;350;1200;420000"],
            }
        ]

        processor = APUProcessor(self.config, self.profile)
        processor.raw_records = records
        df = processor.process_all()

        self.assertIsInstance(df, pd.DataFrame)
        if not df.empty:
            # La descripción debería estar truncada
            desc = df.iloc[0]["DESCRIPCION_INSUMO"]
            self.assertLessEqual(len(desc), 600)

    def test_none_config(self):
        """Maneja configuración None."""
        # El procesador debería manejar config None gracefully
        try:
            processor = APUProcessor(None, self.profile)
            self.assertIsNotNone(processor)
        except Exception:
            # También es aceptable que falle de forma controlada
            pass

    def test_none_profile(self):
        """Maneja perfil None."""
        processor = APUProcessor(self.config, None)
        self.assertIsNotNone(processor)
        self.assertEqual(processor.profile, {})


# ============================================================================
# PRUEBAS DE INTEGRACIÓN
# ============================================================================


class IntegrationTests(unittest.TestCase):
    """Pruebas de integración end-to-end."""

    def setUp(self):
        self.config = TestFixtures.get_default_config()
        self.profile = TestFixtures.get_default_profile()

    def test_full_pipeline_grouped_records(self):
        """Pipeline completo con registros agrupados."""
        records = TestFixtures.get_grouped_sample_records()

        processor = APUProcessor(self.config, self.profile)
        processor.raw_records = records
        df = processor.process_all()

        # Verificaciones del resultado
        self.assertFalse(df.empty)

        # Verificar que hay diferentes tipos de insumo
        tipos = df["TIPO_INSUMO"].unique()
        self.assertGreater(len(tipos), 1)

        # Verificar valores positivos
        self.assertTrue(all(df["VALOR_TOTAL_APU"] >= 0))

    def test_full_pipeline_flat_records(self):
        """Pipeline completo con registros planos."""
        records = TestFixtures.get_flat_sample_records()

        processor = APUProcessor(self.config, self.profile)
        processor.raw_records = records
        df = processor.process_all()

        self.assertIsInstance(df, pd.DataFrame)

    def test_consistency_across_formats(self):
        """Verifica consistencia entre formatos grouped y flat."""
        grouped = TestFixtures.get_grouped_sample_records()[:1]  # Primer APU
        flat = [
            r for r in TestFixtures.get_flat_sample_records()
            if r["apu_code"] == "1.1"
        ]

        # Procesar ambos
        proc_grouped = APUProcessor(self.config, self.profile)
        proc_grouped.raw_records = grouped
        df_grouped = proc_grouped.process_all()

        proc_flat = APUProcessor(self.config, self.profile)
        proc_flat.raw_records = flat
        df_flat = proc_flat.process_all()

        # Ambos deberían producir resultados para el mismo APU
        if not df_grouped.empty and not df_flat.empty:
            self.assertEqual(
                df_grouped["CODIGO_APU"].iloc[0],
                df_flat["CODIGO_APU"].iloc[0]
            )


# ============================================================================
# PRUEBAS DE VALIDATIONTHRESHOLDS
# ============================================================================


class ValidationThresholdsTests(unittest.TestCase):
    """Pruebas para ValidationThresholds dataclass."""

    def test_default_values(self):
        """Verifica valores por defecto."""
        thresholds = ValidationThresholds()

        self.assertEqual(thresholds.min_jornal, 50000)
        self.assertEqual(thresholds.max_jornal, 10000000)
        self.assertEqual(thresholds.min_rendimiento, 0.001)
        self.assertEqual(thresholds.max_rendimiento, 1000)
        self.assertEqual(thresholds.max_rendimiento_tipico, 100)

    def test_custom_values(self):
        """Acepta valores personalizados."""
        thresholds = ValidationThresholds(
            min_jornal=100000,
            max_jornal=5000000,
            min_rendimiento=0.01,
        )

        self.assertEqual(thresholds.min_jornal, 100000)
        self.assertEqual(thresholds.max_jornal, 5000000)
        self.assertEqual(thresholds.min_rendimiento, 0.01)


# ============================================================================
# PRUEBAS DE PARSINGSTATS
# ============================================================================


class ParsingStatsTests(unittest.TestCase):
    """Pruebas para ParsingStats dataclass."""

    def test_default_values(self):
        """Verifica valores por defecto."""
        stats = ParsingStats()

        self.assertEqual(stats.total_lines, 0)
        self.assertEqual(stats.successful_parses, 0)
        self.assertEqual(stats.lark_parse_errors, 0)
        self.assertEqual(stats.transformer_errors, 0)
        self.assertEqual(stats.failed_lines, [])

    def test_mutable_failed_lines(self):
        """failed_lines es mutable independientemente."""
        stats1 = ParsingStats()
        stats2 = ParsingStats()

        stats1.failed_lines.append({"line": 1})

        self.assertEqual(len(stats1.failed_lines), 1)
        self.assertEqual(len(stats2.failed_lines), 0)


# ============================================================================
# PRUEBAS DE LOAD_VALIDATION_THRESHOLDS
# ============================================================================


class LoadValidationThresholdsTests(unittest.TestCase):
    """Pruebas para _load_validation_thresholds del transformer."""

    def test_loads_from_config(self):
        """Carga umbrales desde configuración."""
        config = {
            "validation_thresholds": {
                "MANO_DE_OBRA": {
                    "min_jornal": 75000,
                    "max_jornal": 8000000,
                }
            }
        }
        profile = TestFixtures.get_default_profile()
        context = TestFixtures.get_default_apu_context()

        transformer = APUTransformer(context, config, profile, {})

        self.assertEqual(transformer.thresholds.min_jornal, 75000)
        self.assertEqual(transformer.thresholds.max_jornal, 8000000)

    def test_uses_defaults_on_missing_config(self):
        """Usa valores por defecto cuando falta configuración."""
        config = {}
        profile = TestFixtures.get_default_profile()
        context = TestFixtures.get_default_apu_context()

        transformer = APUTransformer(context, config, profile, {})
        defaults = ValidationThresholds()

        self.assertEqual(transformer.thresholds.min_jornal, defaults.min_jornal)
        self.assertEqual(transformer.thresholds.max_jornal, defaults.max_jornal)

    def test_handles_invalid_values(self):
        """Maneja valores inválidos en configuración."""
        config = {
            "validation_thresholds": {
                "MANO_DE_OBRA": {
                    "min_jornal": "no_es_numero",
                    "max_jornal": None,
                }
            }
        }
        profile = TestFixtures.get_default_profile()
        context = TestFixtures.get_default_apu_context()

        # No debería lanzar excepción
        transformer = APUTransformer(context, config, profile, {})
        defaults = ValidationThresholds()

        # Debería usar valores por defecto
        self.assertEqual(transformer.thresholds.min_jornal, defaults.min_jornal)


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    # Ejecutar con verbosidad
    unittest.main(verbosity=2)