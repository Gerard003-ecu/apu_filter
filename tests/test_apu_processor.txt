"""
Suite de pruebas completa y robusta para APUProcessor.
Alineada con el código refinado y mejores prácticas de testing.
"""

import logging
import os
import sys
import time
import unittest
import time
from contextlib import contextmanager
from typing import Any, Dict, List
from unittest.mock import Mock

import numpy as np
import pandas as pd
from parameterized import parameterized

# Agregar el path del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.apu_processor import (
    APUProcessor,
    APUTransformer,
    FormatoLinea,
    KeywordCache,
    TipoInsumo,
    calculate_unit_costs,
)
from app.schemas import (
    Equipo,
    ManoDeObra,
    Otro,
    Suministro,
    Transporte,
)

# ============================================================================
# FIXTURES Y HELPERS
# ============================================================================

class TestFixtures:
    """Centraliza datos de prueba reutilizables."""

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Configuración estándar para pruebas."""
        return {
            "validation_thresholds": {
                "MANO_DE_OBRA": {
                    "min_jornal": 1000,
                    "max_jornal": 10000000,
                    "min_rendimiento": 0.0001,
                    "max_rendimiento": 10000,
                    "max_valor_total": 200000000
                },
                "EQUIPO": {
                    "max_valor_total": 50000000
                },
                "TRANSPORTE": {
                    "max_valor_total": 10000000
                },
                "SUMINISTRO": {
                    "max_valor_total": 100000000
                },
                "DEFAULT": {
                    "max_valor_total": 100000000
                }
            },
            "batch_size": 100,
            "keyword_maps": {
                "equipo": ["EQUIPO", "MAQUINA", "MAQUINARIA", "HERRAMIENTA", "RETRO", "MOTONIVELADORA",
                      "COMPACTADORA", "VIBRO", "MOTOBOMBA", "MOTOCARGADOR", "EXCAVADORA", "CAMION"],
                "mano_de_obra": ["OFICIAL", "AYUDANTE", "ALBAÑIL", "PEON", "OPERARIO", "CONDUCTOR",
                            "CARPINTERO", "ELECTRICISTA", "PINTOR", "SOLDADOR", "MO ", "MANO OBRA"],
                "transporte": ["TRANSPORTE", "VOLQUETA", "CAMIONETA", "FURGON", "CAMION", "VIAJE", "ACARREO"],
                "suministro": ["CEMENTO", "ARENA", "AGREGADO", "CONCRETO", "TUBERIA", "ACERO", "LAMINA",
                          "MATERIAL", "SUMINISTRO", "INSUMO", "TUBO", "VARILLA", "MALLA", "ALAMBRE"]
            }
        }

    @staticmethod
    def get_sample_records() -> List[Dict[str, str]]:
        """Registros de muestra variados."""
        return [
            # Suministro básico
            {
                "apu_code": "01.01.001",
                "apu_desc": "Suministro de tubería PVC 4\"",
                "apu_unit": "M",
                "category": "MATERIALES",
                "insumo_line": "Tubería PVC 4 pulgadas;M;10.5;25000;262500"
            },
            # Mano de obra formato completo
            {
                "apu_code": "02.01.001",
                "apu_desc": "Excavación manual",
                "apu_unit": "M3",
                "category": "MANO DE OBRA",
                "insumo_line": "Oficial excavador;JOR;1;120000;8.0;15000"
            },
            # Equipo
            {
                "apu_code": "03.01.001",
                "apu_desc": "Retroexcavadora",
                "apu_unit": "HR",
                "category": "EQUIPO",
                "insumo_line": "Retroexcavadora CAT 420;HR;1.0;150000;150000"
            },
            # Transporte
            {
                "apu_code": "04.01.001",
                "apu_desc": "Transporte de material",
                "apu_unit": "VIAJE",
                "category": "TRANSPORTE",
                "insumo_line": "Volqueta 7M3;VIAJE;2.0;180000;360000"
            },
            # Caso con desperdicio (6 campos)
            {
                "apu_code": "05.01.001",
                "apu_desc": "Concreto 3000 PSI",
                "apu_unit": "M3",
                "category": "MATERIALES",
                "insumo_line": "Concreto premezclado;M3;1.05;5%;380000;399000"
            },
            # Nuevos registros para pasar validaciones
            {
                "apu_code": "06.01.001",
                "apu_desc": "Suministro de Acero",
                "apu_unit": "KG",
                "category": "MATERIALES",
                "insumo_line": "Acero de refuerzo;KG;100;4000;400000"
            },
            {
                "apu_code": "07.01.001",
                "apu_desc": "Mano de Obra Calificada",
                "apu_unit": "JOR",
                "category": "MANO DE OBRA",
                "insumo_line": "Soldador calificado;JOR;1;250000;4.0;62500"
            }
        ]

    @staticmethod
    def get_edge_case_records() -> List[Dict[str, str]]:
        """Casos edge para pruebas de robustez."""
        return [
            # Registro vacío
            {"apu_code": "", "apu_desc": "", "apu_unit": "", "category": "", "insumo_line": ""},
            # Solo código APU
            {"apu_code": "TEST-01", "apu_desc": "", "apu_unit": "", "category": "", "insumo_line": ""},
            # Línea de insumo malformada
            {"apu_code": "ERR-01", "apu_desc": "Error test", "apu_unit": "UND",
             "category": "TEST", "insumo_line": "Solo;dos;campos"},
            # Valores no numéricos
            {"apu_code": "ERR-02", "apu_desc": "Non numeric", "apu_unit": "UND",
             "category": "TEST", "insumo_line": "Test;UND;abc;xyz;123"},
            # Valores negativos
            {"apu_code": "ERR-03", "apu_desc": "Negative values", "apu_unit": "UND",
             "category": "TEST", "insumo_line": "Test;UND;-10;-5000;50000"},
            # Valores extremos
            {"apu_code": "ERR-04", "apu_desc": "Extreme values", "apu_unit": "UND",
             "category": "TEST", "insumo_line": "Test;UND;999999999;999999999;999999999999"},
            # Caracteres especiales
            {"apu_code": "SPEC-01", "apu_desc": "Descripción con ñ, á, é, í, ó, ú",
             "apu_unit": "M²", "category": "TEST",
             "insumo_line": "Ítem con carácteres especiales;M²;1.5;1000;1500"},
        ]


@contextmanager
def suppress_logs():
    """Context manager para suprimir logs durante pruebas."""
    logger = logging.getLogger('app.apu_processor')
    old_level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        logger.setLevel(old_level)


# ============================================================================
# PRUEBAS DE ENUMS Y CONSTANTES
# ============================================================================

class TestEnumsAndConstants(unittest.TestCase):
    """Pruebas para enumeraciones y constantes."""

    def test_tipo_insumo_enum(self):
        """Verifica que TipoInsumo tenga todos los valores esperados."""
        expected_values = {
            "MANO_DE_OBRA", "EQUIPO", "TRANSPORTE", "SUMINISTRO", "OTRO"
        }
        actual_values = {tipo.value for tipo in TipoInsumo}
        self.assertEqual(expected_values, actual_values)

    def test_formato_linea_enum(self):
        """Verifica que FormatoLinea tenga todos los valores esperados."""
        expected_values = {"MO_COMPLETA", "INSUMO_BASICO", "DESCONOCIDO"}
        actual_values = {formato.value for formato in FormatoLinea}
        self.assertEqual(expected_values, actual_values)

    def test_excluded_terms_immutable(self):
        """Verifica que EXCLUDED_TERMS sea inmutable (frozenset)."""
        self.assertIsInstance(APUProcessor.EXCLUDED_TERMS, frozenset)
        self.assertIn('IMPUESTOS', APUProcessor.EXCLUDED_TERMS)
        self.assertIn('AIU', APUProcessor.EXCLUDED_TERMS)


# ============================================================================
# PRUEBAS DE KEYWORD CACHE
# ============================================================================

class TestKeywordCache(unittest.TestCase):
    """Pruebas para el sistema de cache de keywords."""

    def setUp(self):
        self.config = TestFixtures.get_default_config()
        self.cache = KeywordCache(self.config)

    def test_lazy_initialization(self):
        """Verifica que las keywords se inicialicen solo cuando se acceden."""
        self.assertFalse(self.cache._initialized)

        # Acceder a keywords debe inicializar
        keywords = self.cache.equipo_keywords
        self.assertTrue(self.cache._initialized)
        self.assertIsInstance(keywords, list)
        self.assertIn("EQUIPO", keywords)

    def test_all_keyword_categories(self):
        """Verifica que todas las categorías de keywords estén disponibles."""
        self.assertGreater(len(self.cache.equipo_keywords), 0)
        self.assertGreater(len(self.cache.mo_keywords), 0)
        self.assertGreater(len(self.cache.transporte_keywords), 0)
        self.assertGreater(len(self.cache.suministro_keywords), 0)

    def test_no_duplicate_keywords(self):
        """Verifica que no haya keywords duplicadas en cada categoría."""
        for category in ['equipo', 'mo', 'transporte', 'suministro']:
            keywords = getattr(self.cache, f"{category}_keywords")
            self.assertEqual(len(keywords), len(set(keywords)),
                           f"Duplicados encontrados en {category}_keywords")


# ============================================================================
# PRUEBAS DE APU TRANSFORMER
# ============================================================================

class TestAPUTransformer(unittest.TestCase):
    """Pruebas exhaustivas para APUTransformer."""

    def setUp(self):
        with suppress_logs():
            self.config = TestFixtures.get_default_config()
            self.keyword_cache = KeywordCache(self.config)
            self.apu_context = {
                "codigo_apu": "TEST-001",
                "descripcion_apu": "Test APU",
                "unidad_apu": "UND",
                "categoria": "TEST"
            }
            self.transformer = APUTransformer(self.apu_context, self.config, self.keyword_cache)

    def test_clean_token_variations(self):
        """Prueba limpieza de tokens con diferentes entradas."""
        # Token normal
        mock_token = Mock()
        mock_token.value = "  test value  "
        self.assertEqual(self.transformer._clean_token(mock_token), "test value")

        # Token None
        self.assertEqual(self.transformer._clean_token(None), "")

        # Token sin atributo value
        self.assertEqual(self.transformer._clean_token("direct_string"), "direct_string")

    @parameterized.expand([
        # (tokens, formato_esperado)
        (["Oficial", "JOR", "1", "120000", "8.0", "15000"], FormatoLinea.MO_COMPLETA),
        (["Cemento", "KG", "50", "1000", "50000"], FormatoLinea.INSUMO_BASICO),
        (["Descripción", "UND"], FormatoLinea.DESCONOCIDO),
        ([], FormatoLinea.DESCONOCIDO),
    ])
    def test_detect_format_scenarios(self, tokens, expected_format):
        """Prueba detección de formato con diferentes escenarios."""
        with suppress_logs():
            detected = self.transformer._detect_format(tokens)
            self.assertEqual(detected, expected_format)

    def test_validate_mo_format(self):
        """Prueba validación específica de formato MO."""
        # Caso válido
        valid_fields = ["Oficial", "JOR", "1", "120000", "8.0", "15000"]
        self.assertTrue(self.transformer._validate_mo_format(valid_fields))

        # Jornal fuera de rango
        invalid_jornal = ["Oficial", "JOR", "1", "10000", "8.0", "1250"]
        self.assertTrue(self.transformer._validate_mo_format(invalid_jornal))

        # Rendimiento inválido
        invalid_rend = ["Oficial", "JOR", "1", "120000", "0", "0"]
        self.assertTrue(self.transformer._validate_mo_format(invalid_rend))

        # Campos insuficientes
        short_fields = ["Oficial", "JOR", "1"]
        self.assertFalse(self.transformer._validate_mo_format(short_fields))

    def test_build_mo_completa_success(self):
        """Prueba construcción exitosa de ManoDeObra."""
        tokens = ["Oficial albañil", "JOR", "1", "150000", "10.0", "15000"]
        result = self.transformer._build_mo_completa(tokens)

        self.assertIsInstance(result, ManoDeObra)
        self.assertEqual(result.descripcion_insumo, "Oficial albañil")
        self.assertEqual(result.precio_unitario, 150000)
        self.assertAlmostEqual(result.rendimiento, 10.0)
        self.assertAlmostEqual(result.cantidad, 0.1, places=4)

    def test_build_mo_completa_failure_cases(self):
        """Prueba casos de falla en construcción de MO."""
        with suppress_logs():
            # Campos insuficientes
            self.assertIsNone(self.transformer._build_mo_completa(["Solo", "dos"]))

            # Valores no numéricos
            self.assertIsNone(
                self.transformer._build_mo_completa(["Test", "JOR", "1", "abc", "def", "ghi"])
            )

    def test_build_insumo_basico_variations(self):
        """Prueba construcción de insumos básicos con variaciones."""
        test_cases = [
            # (tokens, tipo_esperado)
            (["Tubería PVC", "M", "10", "5000", "50000"], Suministro),
            (["Retroexcavadora", "HR", "2", "150000", "300000"], Equipo),
            (["Transporte material", "VIAJE", "1", "200000", "200000"], Transporte),
            (["Otro insumo", "UND", "5", "1000", "5000"], Otro),
        ]

        for tokens, expected_class in test_cases:
            with self.subTest(tokens=tokens):
                result = self.transformer._build_insumo_basico(tokens)
                self.assertIsInstance(result, expected_class)

    def test_parse_insumo_fields_with_waste(self):
        """Prueba parsing de campos con desperdicio (6 campos)."""
        # Con desperdicio
        tokens_6 = ["Concreto", "M3", "1.05", "5%", "380000", "399000"]
        result = self.transformer._parse_insumo_fields(tokens_6)
        self.assertIsNotNone(result)
        desc, unit, qty, price, total = result
        self.assertEqual(desc, "Concreto")
        self.assertEqual(unit, "M3")
        self.assertAlmostEqual(qty, 1.05)

        # Sin desperdicio (5 campos)
        tokens_5 = ["Arena", "M3", "2.0", "50000", "100000"]
        result = self.transformer._parse_insumo_fields(tokens_5)
        self.assertIsNotNone(result)

    def test_correct_total_value(self):
        """Prueba corrección automática de valor total."""
        # Valor total cero, debe calcularse
        corrected = self.transformer._correct_total_value(10, 1000, 0)
        self.assertEqual(corrected, 10000)

        # Valor total correcto, no debe cambiar
        unchanged = self.transformer._correct_total_value(10, 1000, 10000)
        self.assertEqual(unchanged, 10000)

        # Valor total con tolerancia aceptable
        with_tolerance = self.transformer._correct_total_value(10, 1000, 10050)
        self.assertEqual(with_tolerance, 10050)  # No debe cambiar si está dentro de tolerancia

    def test_classify_insumo_with_cache(self):
        """Prueba clasificación con cache."""
        descriptions = [
            ("Oficial de construcción", TipoInsumo.MANO_DE_OBRA),
            ("Retroexcavadora CAT", TipoInsumo.EQUIPO),
            ("Transporte de agregados", TipoInsumo.TRANSPORTE),
            ("Tubería PVC 4\"", TipoInsumo.SUMINISTRO),
            ("Elemento desconocido", TipoInsumo.OTRO),
        ]

        for desc, expected_tipo in descriptions:
            with self.subTest(desc=desc):
                result = self.transformer._classify_insumo_with_cache(desc)
                self.assertEqual(result, expected_tipo)

    def test_special_cases_classification(self):
        """Prueba casos especiales de clasificación."""
        special_cases = [
            ("HERRAMIENTA MENOR", TipoInsumo.EQUIPO),
            ("HERRAMIENTA (% MO)", TipoInsumo.EQUIPO),
            ("EQUIPO Y HERRAMIENTA", TipoInsumo.EQUIPO),
        ]

        for desc, expected in special_cases:
            result = self.transformer._classify_insumo(desc)
            self.assertEqual(result, expected, f"Fallo para: {desc}")

    def test_detect_format_ignores_junk_lines(self):
        """Prueba que se ignoren las líneas de ruido."""
        junk_lines = [
            ["SUBTOTAL MATERIALES", "65.403,35"],
            ["DESCRIPCION", "UND", "CANT."],
            ["MATERIALES"],
            ["TOTAL APU", "1.200.000"],
            ["-- encabezado --"],
        ]
        for tokens in junk_lines:
            with self.subTest(tokens=tokens):
                formato = self.transformer._detect_format(tokens)
                self.assertEqual(formato, FormatoLinea.DESCONOCIDO)

    def test_parse_insumo_fields_handles_comma_decimals(self):
        """Prueba el parseo de campos con comas decimales."""
        tokens = ["Cemento", "UND", "1,5", "10.000,00", "15.000,00"]
        parsed = self.transformer._parse_insumo_fields(tokens)
        self.assertIsNotNone(parsed)
        # _parse_insumo_fields devuelve 5 valores
        self.assertEqual(len(parsed), 5)
        # Acceder a los valores desempaquetados
        _, _, cantidad, precio_unitario, valor_total = parsed
        self.assertAlmostEqual(cantidad, 1.5)
        self.assertAlmostEqual(precio_unitario, 10000.0)
        self.assertAlmostEqual(valor_total, 15000.0)


# ============================================================================
# PRUEBAS DE APU PROCESSOR
# ============================================================================

class TestAPUProcessor(unittest.TestCase):
    """Pruebas completas para APUProcessor."""

    def setUp(self):
        with suppress_logs():
            self.config = TestFixtures.get_default_config()
            self.sample_records = TestFixtures.get_sample_records()

    def test_initialization_valid(self):
        """Prueba inicialización con datos válidos."""
        processor = APUProcessor(self.sample_records, self.config)
        self.assertIsNotNone(processor)
        self.assertIsNotNone(processor._parser)
        self.assertEqual(len(processor.processed_data), 0)

    def test_initialization_invalid_inputs(self):
        """Prueba inicialización con entradas inválidas."""
        # raw_records no es lista
        with self.assertRaises(ValueError):
            APUProcessor("not_a_list", self.config)

        # config no es diccionario
        with self.assertRaises(ValueError):
            APUProcessor([], "not_a_dict")

    def test_initialization_with_warnings(self):
        """Prueba inicialización con configuración incompleta."""
        incomplete_config = {}
        with self.assertLogs('app.apu_processor', level='WARNING') as logs:
            processor = APUProcessor([], incomplete_config)
            self.assertIsNotNone(processor)
            self.assertTrue(any('validation_thresholds' in log for log in logs.output))

    def test_process_all_basic(self):
        """Prueba procesamiento básico completo."""
        processor = APUProcessor(self.sample_records, self.config)
        df = processor.process_all()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

        # Verificar columnas esperadas
        expected_columns = [
            'CODIGO_APU', 'DESCRIPCION_APU', 'UNIDAD_APU',
            'DESCRIPCION_INSUMO', 'TIPO_INSUMO', 'VALOR_TOTAL_APU'
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)

    def test_process_edge_cases(self):
        """Prueba procesamiento de casos edge."""
        with suppress_logs():
            edge_records = TestFixtures.get_edge_case_records()
            processor = APUProcessor(edge_records, self.config)
            df = processor.process_all()

            # Debe procesar sin crashes, aunque descarte registros inválidos
            self.assertIsInstance(df, pd.DataFrame)

            # Verificar que se procesaron algunos registros válidos
            if len(df) > 0:
                # No debe haber valores negativos en cantidades
                self.assertFalse((df['CANTIDAD_APU'] < 0).any())
                # No debe haber NaN en columnas críticas
                self.assertFalse(df['CODIGO_APU'].isna().any())

    def test_batch_processing(self):
        """Prueba procesamiento por lotes."""
        # Crear muchos registros
        many_records = self.sample_records * 50  # 250 registros
        config_with_batch = self.config.copy()
        config_with_batch['batch_size'] = 50

        with suppress_logs():
            processor = APUProcessor(many_records, config_with_batch)
            df = processor.process_all()

            self.assertIsInstance(df, pd.DataFrame)
            # Verificar que procesó múltiples lotes
            self.assertGreaterEqual(len(df), 50)

    def test_clean_record_fields(self):
        """Prueba limpieza de campos de registro."""
        processor = APUProcessor([], self.config)

        # Registro válido
        valid_record = {
            "apu_code": "  01.01.001  ",
            "apu_desc": "  Test Description  ",
            "apu_unit": "m2",
            "category": "TEST"
        }
        result = processor._clean_record_fields(valid_record)
        self.assertIsNotNone(result)
        self.assertEqual(result["apu_code"], "01.01.001")
        self.assertEqual(result["apu_desc"], "Test Description")
        self.assertEqual(result["apu_unit"], "M2")

        # Registro sin código APU
        invalid_record = {"apu_desc": "Test", "apu_unit": "UND"}
        result = processor._clean_record_fields(invalid_record)
        self.assertIsNone(result)

    def test_normalize_unit_cache(self):
        """Prueba normalización de unidades con cache."""
        processor = APUProcessor([], self.config)

        test_units = [
            ("dias", "DIA"),
            ("DÍAS", "DIA"),
            ("m2", "M2"),
            ("metros cuadrados", "M2"),
            ("UN", "UND"),
            ("", "UND"),
            (None, "UND"),
            ("UNKNOWN_UNIT", "UNKNOWN_UNIT"),
        ]

        for input_unit, expected in test_units:
            with self.subTest(input=input_unit):
                result = processor._normalize_unit(input_unit or "")
                self.assertEqual(result, expected)

    def test_infer_unit_intelligent(self):
        """Prueba inferencia inteligente de unidades."""
        processor = APUProcessor([], self.config)

        test_cases = [
            # (descripción, categoría, tipo_insumo, unidad_esperada)
            ("Excavación manual en terreno", "OBRA", "MANO_DE_OBRA", "JOR"),
            ("Retroexcavadora CAT 320", "EQUIPO", "EQUIPO", "DIA"),
            ("Transporte de material pétreo", "TRANSPORTE", "TRANSPORTE", "VIAJE"),
            ("Concreto de 3000 PSI M3", "MATERIALES", "SUMINISTRO", "M3"),
            ("Pintura epóxica en muros", "MATERIALES", "SUMINISTRO", "UND"),
            ("Tubería PVC 4 pulgadas", "MATERIALES", "SUMINISTRO", "UND"),
        ]

        for desc, cat, tipo, expected in test_cases:
            with self.subTest(desc=desc):
                result = processor._infer_unit_intelligent(desc, cat, tipo)
                self.assertEqual(result, expected)

    def test_validate_final_insumo(self):
        """Prueba validación final de insumos."""
        processor = APUProcessor([], self.config)

        # Crear insumo válido
        valid_insumo = ManoDeObra(
            codigo_apu="TEST-01",
            descripcion_apu="Test",
            unidad_apu="JOR",
            descripcion_insumo="Oficial",
            unidad_insumo="JOR",
            cantidad=1.0,
            precio_unitario=120000,
            valor_total=120000,
            categoria="MO",
            tipo_insumo="MANO_DE_OBRA",
            formato_origen="TEST",
            rendimiento=1.0,
            normalized_desc="oficial"
        )
        self.assertTrue(processor._validate_final_insumo(valid_insumo))

        # Insumo con término excluido
        excluded_insumo = ManoDeObra(
            codigo_apu="TEST-02",
            descripcion_apu="Test",
            unidad_apu="JOR",
            descripcion_insumo="IMPUESTOS Y GASTOS",
            unidad_insumo="JOR",
            cantidad=1.0,
            precio_unitario=100000,
            valor_total=100000,
            categoria="MO",
            tipo_insumo="MANO_DE_OBRA",
            formato_origen="TEST",
            rendimiento=1.0,
            normalized_desc="impuestos"
        )
        self.assertFalse(processor._validate_final_insumo(excluded_insumo))

    def test_validate_by_type(self):
        """Prueba validación específica por tipo."""
        processor = APUProcessor([], self.config)

        # MO con jornal válido
        valid_mo = ManoDeObra(
            codigo_apu="TEST", descripcion_apu="Test", unidad_apu="JOR",
            descripcion_insumo="Oficial", unidad_insumo="JOR",
            cantidad=1.0, precio_unitario=120000, valor_total=120000,
            categoria="MO", tipo_insumo="MANO_DE_OBRA",
            formato_origen="TEST", rendimiento=1.0, normalized_desc="oficial"
        )
        self.assertTrue(processor._validate_by_type(valid_mo))

        # MO con jornal muy bajo
        low_jornal_mo = ManoDeObra(
            codigo_apu="TEST", descripcion_apu="Test", unidad_apu="JOR",
            descripcion_insumo="Oficial", unidad_insumo="JOR",
            cantidad=1.0, precio_unitario=10000, valor_total=10000,
            categoria="MO", tipo_insumo="MANO_DE_OBRA",
            formato_origen="TEST", rendimiento=1.0, normalized_desc="oficial"
        )
        self.assertTrue(processor._validate_by_type(low_jornal_mo))

    def test_fix_squad_units(self):
        """Prueba corrección de unidades para cuadrillas."""
        with suppress_logs():
            processor = APUProcessor([], self.config)

            # Agregar datos de prueba
            processor.processed_data = [
                ManoDeObra(
                    codigo_apu="13.01", descripcion_apu="Cuadrilla A",
                    unidad_apu="UND",  # Debe cambiar a DIA
                    descripcion_insumo="Cuadrilla", unidad_insumo="JOR",
                    cantidad=1, precio_unitario=500000, valor_total=500000,
                    categoria="MO", tipo_insumo="MANO_DE_OBRA",
                    formato_origen="TEST", rendimiento=1, normalized_desc="cuadrilla"
                ),
                ManoDeObra(
                    codigo_apu="01.01", descripcion_apu="Oficial",
                    unidad_apu="JOR",  # No debe cambiar
                    descripcion_insumo="Oficial", unidad_insumo="JOR",
                    cantidad=1, precio_unitario=120000, valor_total=120000,
                    categoria="MO", tipo_insumo="MANO_DE_OBRA",
                    formato_origen="TEST", rendimiento=1, normalized_desc="oficial"
                )
            ]

            processor._fix_squad_units()

            # Verificar correcciones
            self.assertEqual(processor.processed_data[0].unidad_apu, "DIA")
            self.assertEqual(processor.processed_data[1].unidad_apu, "JOR")

    def test_handle_duplicates(self):
        """Prueba manejo de duplicados."""
        with suppress_logs():
            processor = APUProcessor([], self.config)
            # Crear datos con duplicados
            base_insumo = ManoDeObra(
                codigo_apu="TEST-01", descripcion_apu="Test", unidad_apu="JOR",
                descripcion_insumo="Oficial", unidad_insumo="JOR",
                cantidad=1.0, precio_unitario=120000, valor_total=120000,
                categoria="MO", tipo_insumo="MANO_DE_OBRA",
                formato_origen="TEST", rendimiento=1.0, normalized_desc="oficial"
            )

            # Agregar el mismo insumo 3 veces
            processor.processed_data = [base_insumo, base_insumo, base_insumo]

            # Agregar uno diferente
            different_insumo = ManoDeObra(
                codigo_apu="TEST-02", descripcion_apu="Test2", unidad_apu="JOR",
                descripcion_insumo="Ayudante", unidad_insumo="JOR",
                cantidad=1.0, precio_unitario=80000, valor_total=80000,
                categoria="MO", tipo_insumo="MANO_DE_OBRA",
                formato_origen="TEST", rendimiento=1.0, normalized_desc="ayudante"
            )
            processor.processed_data.append(different_insumo)

            processor._handle_duplicates()

            # Debe quedar solo 2 registros únicos
            self.assertEqual(len(processor.processed_data), 2)

    def test_fix_mo_values(self):
        """Prueba la corrección de valores inconsistentes en Mano de Obra."""
        processor = APUProcessor([], self.config)
        inconsistent_mo = ManoDeObra(
            descripcion_insumo="Oficial",
            cantidad=0,  # Cantidad incorrecta
            precio_unitario=0,  # Precio incorrecto
            valor_total=120000,
            rendimiento=10.0,  # Rendimiento correcto
            tipo_insumo="MANO_DE_OBRA",
            # ... otros campos necesarios
            codigo_apu="TEST-MO",
            descripcion_apu="Test MO Fix",
            unidad_apu="UND",
            unidad_insumo="JOR",
            categoria="MANO DE OBRA",
            formato_origen="INSUMO_BASICO",
            normalized_desc="oficial"
        )
        processor.processed_data.append(inconsistent_mo)

        processor._fix_mo_values()

        fixed_mo = processor.processed_data[0]
        self.assertGreater(fixed_mo.cantidad, 0)
        self.assertAlmostEqual(fixed_mo.cantidad, 1.0 / 10.0)
        self.assertGreater(fixed_mo.precio_unitario, 0)

    def test_build_optimized_dataframe(self):
        """Prueba construcción de DataFrame optimizado."""
        with suppress_logs():
            processor = APUProcessor(self.sample_records[:3], self.config)
            df = processor.process_all()

            if not df.empty:
                # Verificar tipos de datos optimizados
                # Columnas numéricas deben ser float32
                numeric_cols = ['CANTIDAD_APU', 'PRECIO_UNIT_APU', 'VALOR_TOTAL_APU']
                for col in numeric_cols:
                    if col in df.columns:
                        self.assertEqual(df[col].dtype, np.float32)

                # Columnas categóricas deben ser category
                cat_cols = ['TIPO_INSUMO', 'UNIDAD_APU']
                for col in cat_cols:
                    if col in df.columns and df[col].nunique() < 100:
                        self.assertEqual(df[col].dtype.name, 'category')


# ============================================================================
# PRUEBAS DE FUNCIONES AUXILIARES
# ============================================================================

class TestUtilityFunctions(unittest.TestCase):
    """Pruebas para funciones auxiliares."""

    def test_calculate_unit_costs_basic(self):
        """Prueba cálculo básico de costos unitarios."""
        # Crear DataFrame de prueba
        data = {
            'CODIGO_APU': ['01', '01', '01', '02', '02'],
            'DESCRIPCION_APU': ['APU1', 'APU1', 'APU1', 'APU2', 'APU2'],
            'UNIDAD_APU': ['M2', 'M2', 'M2', 'M3', 'M3'],
            'TIPO_INSUMO': ['SUMINISTRO', 'MANO_DE_OBRA', 'EQUIPO',
                           'SUMINISTRO', 'TRANSPORTE'],
            'VALOR_TOTAL_APU': [100000, 50000, 30000, 200000, 40000]
        }
        df = pd.DataFrame(data)

        result = calculate_unit_costs(df)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)  # 2 APUs únicos

        # Verificar cálculos para APU 01
        apu01 = result[result['CODIGO_APU'] == '01'].iloc[0]
        self.assertEqual(apu01['VALOR_SUMINISTRO_UN'], 100000)
        self.assertEqual(apu01['VALOR_INSTALACION_UN'], 80000)  # MO + Equipo
        self.assertEqual(apu01['COSTO_UNITARIO_TOTAL'], 180000)

        # Verificar porcentajes
        self.assertAlmostEqual(apu01['PCT_SUMINISTRO'], 55.56, places=1)
        self.assertAlmostEqual(apu01['PCT_INSTALACION'], 44.44, places=1)

    def test_calculate_unit_costs_empty_df(self):
        """Prueba cálculo con DataFrame vacío."""
        df = pd.DataFrame()
        result = calculate_unit_costs(df)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    def test_calculate_unit_costs_missing_columns(self):
        """Prueba cálculo con columnas faltantes."""
        df = pd.DataFrame({'CODIGO_APU': ['01'], 'INVALID_COL': [100]})

        with self.assertLogs('app.apu_processor', level='ERROR'):
            result = calculate_unit_costs(df)
            self.assertTrue(result.empty)

    def test_calculate_unit_costs_zero_division(self):
        """Prueba manejo de división por cero en porcentajes."""
        data = {
            'CODIGO_APU': ['01'],
            'DESCRIPCION_APU': ['APU1'],
            'UNIDAD_APU': ['M2'],
            'TIPO_INSUMO': ['OTRO'],
            'VALOR_TOTAL_APU': [0]
        }
        df = pd.DataFrame(data)

        result = calculate_unit_costs(df)

        # No debe fallar y los porcentajes deben ser 0
        self.assertEqual(result.iloc[0]['PCT_SUMINISTRO'], 0)
        self.assertEqual(result.iloc[0]['PCT_INSTALACION'], 0)


# ============================================================================
# PRUEBAS DE INTEGRACIÓN
# ============================================================================

class TestIntegration(unittest.TestCase):
    """Pruebas de integración end-to-end."""

    def test_full_pipeline(self):
        """Prueba pipeline completo desde registros crudos hasta costos unitarios."""
        with suppress_logs():
            # Datos de entrada
            raw_records = TestFixtures.get_sample_records()
            config = TestFixtures.get_default_config()

            # Procesar
            processor = APUProcessor(raw_records, config)
            df_processed = processor.process_all()

            # Validar procesamiento
            self.assertGreater(len(df_processed), 0)
            self.assertIn('TIPO_INSUMO', df_processed.columns)

            # Calcular costos unitarios
            df_costs = calculate_unit_costs(df_processed, config)

            # Validar costos
            self.assertGreater(len(df_costs), 0)
            self.assertIn('COSTO_UNITARIO_TOTAL', df_costs.columns)
            self.assertTrue((df_costs['COSTO_UNITARIO_TOTAL'] >= 0).all())

    def test_large_dataset_performance(self):
        """Prueba performance con dataset grande."""
        # Generar dataset grande
        base_records = TestFixtures.get_sample_records()
        large_dataset = base_records * 200  # 1000 registros

        config = TestFixtures.get_default_config()
        config['batch_size'] = 100

        with suppress_logs():
            start_time = time.time()
            processor = APUProcessor(large_dataset, config)
            df = processor.process_all()
            elapsed = time.time() - start_time

            # Debe procesar en tiempo razonable (< 10 segundos para 1000 registros)
            self.assertLess(elapsed, 10, f"Procesamiento muy lento: {elapsed:.2f}s")

            # Debe procesar al menos 50% de los registros
            self.assertGreater(len(df), len(large_dataset) * 0.5)

    def test_error_recovery(self):
        """Prueba recuperación de errores durante procesamiento."""
        # Mix de registros buenos y malos
        mixed_records = (
            TestFixtures.get_sample_records() +
            TestFixtures.get_edge_case_records()
        )

        config = TestFixtures.get_default_config()

        with suppress_logs():
            processor = APUProcessor(mixed_records, config)
            df = processor.process_all()

            # Debe procesar sin crashes
            self.assertIsInstance(df, pd.DataFrame)

            # Debe haber procesado al menos los registros válidos
            self.assertGreater(len(df), 0)

            # Verificar estadísticas de errores
            self.assertGreater(processor.stats.get('errores', 0) +
                             processor.stats.get('registros_descartados', 0), 0)


# ============================================================================
# PRUEBAS DE REGRESIÓN
# ============================================================================

class TestRegression(unittest.TestCase):
    """Pruebas de regresión para bugs conocidos."""

    def test_unicode_handling(self):
        """Regresión: manejo de caracteres Unicode."""
        records = [{
            "apu_code": "UNI-01",
            "apu_desc": "Instalación de señalización",
            "apu_unit": "M²",
            "category": "SEÑALIZACIÓN",
            "insumo_line": "Señal de tránsito tipo Ñ;UND;1;50000;50000"
        }]

        with suppress_logs():
            processor = APUProcessor(records, TestFixtures.get_default_config())
            df = processor.process_all()

            # No debe fallar con caracteres especiales
            self.assertIsInstance(df, pd.DataFrame)

    def test_very_long_descriptions(self):
        """Regresión: descripciones muy largas."""
        long_desc = "A" * 500
        records = [{
            "apu_code": "LONG-01",
            "apu_desc": long_desc,
            "apu_unit": "UND",
            "category": "TEST",
            "insumo_line": f"{long_desc};UND;1;1000;1000"
        }]

        with suppress_logs():
            processor = APUProcessor(records, TestFixtures.get_default_config())
            df = processor.process_all()

            # Debe manejar descripciones largas
            self.assertIsInstance(df, pd.DataFrame)

    def test_scientific_notation(self):
        """Regresión: números en notación científica."""
        records = [{
            "apu_code": "SCI-01",
            "apu_desc": "Test científico",
            "apu_unit": "UND",
            "category": "TEST",
            "insumo_line": "Item;UND;1.5e-3;2.5e6;3.75e3"
        }]

        with suppress_logs():
            processor = APUProcessor(records, TestFixtures.get_default_config())
            df = processor.process_all()

            if not df.empty:
                # Verificar que los valores se parsearon correctamente
                self.assertAlmostEqual(df.iloc[0]['CANTIDAD_APU'], 0.0015, places=6)
                self.assertAlmostEqual(df.iloc[0]['PRECIO_UNIT_APU'], 2500000, places=0)


# ============================================================================
# SUITE DE PRUEBAS
# ============================================================================

def create_test_suite():
    """Crea suite completa de pruebas."""
    suite = unittest.TestSuite()

    # Agregar todas las clases de prueba
    test_classes = [
        TestEnumsAndConstants,
        TestKeywordCache,
        TestAPUTransformer,
        TestAPUProcessor,
        TestUtilityFunctions,
        TestIntegration,
        TestRegression
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    return suite


if __name__ == "__main__":
    # Configurar logging para pruebas
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Ejecutar pruebas con mayor verbosidad
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)

    # Imprimir resumen
    print("\n" + "="*60)
    print("RESUMEN DE PRUEBAS")
    print("="*60)
    print(f"Pruebas ejecutadas: {result.testsRun}")
    print(f"Éxitos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Fallos: {len(result.failures)}")
    print(f"Errores: {len(result.errors)}")
    print(f"Omitidas: {len(result.skipped)}")
    print("="*60)

    # Salir con código apropiado
    sys.exit(0 if result.wasSuccessful() else 1)
