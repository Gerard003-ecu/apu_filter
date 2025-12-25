import unittest
import logging
import pandas as pd
from unittest.mock import MagicMock, patch
import sys
import os
from typing import Dict, List, Any, Optional

# Ajustar ruta de importación para pruebas
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.apu_processor import (
    APUProcessor, 
    PatternMatcher, 
    UnitsValidator, 
    NumericFieldExtractor,
    ValidationThresholds,
    FormatoLinea,
    TipoInsumo,
    APUTransformer,
    ParsingStats
)
from app.utils import calculate_unit_costs, parse_number

# Configurar logging para pruebas
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_apu_processor")

class TestFixtures:
    """Fixtures mejorados que reflejan casos realistas y complejos"""
    
    @staticmethod
    def get_default_config():
        """Configuración completa con todos los parámetros necesarios"""
        return {
            "apu_processor_rules": {
                "special_cases": {
                    "TRANSPORTE": "TRANSPORTE",
                    "ALQUILER": "EQUIPO",
                    "SUBCONTRATO": "OTRO"
                },
                "mo_keywords": ["OFICIAL", "AYUDANTE", "PEON", "CUADRILLA", "OPERARIO", "JORNAL"],
                "equipo_keywords": ["EQUIPO", "HERRAMIENTA", "MAQUINA", "ALQUILER", "COMPRESOR"],
                "otro_keywords": ["SUBCONTRATO", "ADMINISTRACION", "IMPUESTO", "GASTO"]
            },
            "validation_thresholds": {
                "MANO_DE_OBRA": {
                    "min_jornal": 50000,
                    "max_jornal": 10000000,
                    "min_rendimiento": 0.001,
                    "max_rendimiento": 1000,
                    "max_rendimiento_tipico": 100
                }
            },
            "debug_mode": False
        }
    
    @staticmethod
    def get_grouped_sample_records():
        """Registros en formato agrupado (legacy) con casos variados"""
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
                    "VIBRADOR ALQUILER;HR;0.5;15000;7500"
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
                    "TRANSPORTE;VIAJE;0.3;45000;13500"
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
                    "ACABADO DIAMANTINA;M2;1.0;35000;35000"
                ],
            },
        ]
    
    @staticmethod
    def get_flat_sample_records():
        """Registros en formato plano (nuevo)"""
        return [
            # APU 1.1
            {
                "apu_code": "1.1",
                "apu_desc": "Muro de Contención",
                "apu_unit": "M3",
                "category": "Estructuras",
                "source_line": 10,
                "insumo_line": "OFICIAL ALBAÑIL;JOR;0.125;;180000;22500",
                "line_number": 11
            },
            {
                "apu_code": "1.1",
                "apu_desc": "Muro de Contención",
                "apu_unit": "M3",
                "category": "Estructuras",
                "source_line": 10,
                "insumo_line": "AYUDANTE;JOR;0.25;;100000;25000",
                "line_number": 12
            },
            {
                "apu_code": "1.1",
                "apu_desc": "Muro de Contención",
                "apu_unit": "M3",
                "category": "Estructuras",
                "source_line": 10,
                "insumo_line": "CEMENTO PORTLAND;KG;350;1200;420000",
                "line_number": 13
            },
            # APU 2.1
            {
                "apu_code": "2.1",
                "apu_desc": "Excavación Manual",
                "apu_unit": "M3",
                "category": "Movimiento de Tierras",
                "source_line": 25,
                "insumo_line": "PEON;JOR;0.5;;100000;50000",
                "line_number": 26
            },
            {
                "apu_code": "2.1",
                "apu_desc": "Excavación Manual",
                "apu_unit": "M3",
                "category": "Movimiento de Tierras",
                "source_line": 25,
                "insumo_line": "TRANSPORTE;VIAJE;0.3;45000;13500",
                "line_number": 27
            },
        ]
    
    @staticmethod
    def get_edge_case_records():
        """Registros con casos extremos y potenciales errores"""
        return [
            {
                "codigo_apu": "99.1",
                "descripcion_apu": "Caso Extremo",
                "unidad_apu": "UN",
                "category": "Pruebas",
                "source_line": 100,
                "lines": [
                    "",  # Línea vacía
                    "DESCRIPCION;UND;CANT;PRECIO;TOTAL",  # Línea de encabezado
                    "SUBTOTAL MANO DE OBRA;;;200000",  # Línea de subtotal
                    "MATERIALES",  # Línea de categoría
                    "OFICIAL;;0.0001;;50000000",  # Jornal fuera de rango
                    "AYUDANTE;JOR;0;;100000;0",  # Rendimiento cero
                    "AGUA;%;;;;15000",  # Unidad porcentual
                    "ADMINISTRACION;GLB;;;;45000",  # Insumo indirecto
                    "DESCRIPCION MUY LARGA " + "X" * 500,  # Descripción muy larga
                ],
            },
        ]
    
    @staticmethod
    def get_comma_decimal_records():
        """Registros con separador decimal de coma"""
        return [
            {
                "codigo_apu": "4.1",
                "descripcion_apu": "Piso con Decimales con Coma",
                "unidad_apu": "M2",
                "category": "Acabados",
                "source_line": 55,
                "lines": [
                    "CUADRILLA ESPECIAL;JOR;0,08;;250.000,00;20.000,00",
                    "CONCRETO PREMEZCLADO;M3;0,15;850.123,50;127.518,53",
                    "REFUERZO FIBRA;KG;1,25;2.500,75;3.125,94"
                ],
            },
        ]
    
    @staticmethod
    def get_profile_with_comma_decimal():
        """Perfil configurado para usar coma como separador decimal"""
        return {
            "number_format": {
                "decimal_separator": ",",
                "thousand_separator": "."
            },
            "encoding": "latin-1"
        }

class PatternMatcherTests(unittest.TestCase):
    """Pruebas específicas para el componente PatternMatcher"""
    
    def setUp(self):
        self.matcher = PatternMatcher()
    
    def test_is_likely_header(self):
        """Prueba detección de encabezados de tabla"""
        # Casos positivos
        self.assertTrue(self.matcher.is_likely_header("DESCRIPCION;UND;CANTIDAD;PRECIO;TOTAL", 5))
        self.assertTrue(self.matcher.is_likely_header("DESCRIPCIÓN UNIDAD CANTIDAD", 3))
        self.assertTrue(self.matcher.is_likely_header("ITEM CODIGO DESCRIPCION UND CANT PRECIO TOTAL", 2))
        
        # Casos negativos
        self.assertFalse(self.matcher.is_likely_header("CEMENTO PORTLAND", 3))
        self.assertFalse(self.matcher.is_likely_header("SUBTOTAL", 2))
        
    def test_is_likely_summary(self):
        """Prueba detección de líneas de resumen"""
        # Casos positivos
        self.assertTrue(self.matcher.is_likely_summary("TOTAL MANO DE OBRA", 2))
        self.assertTrue(self.matcher.is_likely_summary("SUBTOTAL MATERIALES", 1))
        self.assertTrue(self.matcher.is_likely_summary("GRAN TOTAL", 2))
        
        # Casos negativos
        self.assertFalse(self.matcher.is_likely_summary("CEMENTO PORTLAND", 3))
        self.assertFalse(self.matcher.is_likely_summary("DESCRIPCION", 1))
    
    def test_is_likely_category(self):
        """Prueba detección de líneas de categoría"""
        # Casos positivos
        self.assertTrue(self.matcher.is_likely_category("MANO DE OBRA", 1))
        self.assertTrue(self.matcher.is_likely_category("MATERIALES", 2))
        self.assertTrue(self.matcher.is_likely_category("EQUIPO", 1))
        
        # Casos negativos
        self.assertFalse(self.matcher.is_likely_category("CEMENTO PORTLAND", 3))
        self.assertFalse(self.matcher.is_likely_category("OFICIAL ALBAÑIL", 4))
    
    def test_has_numeric_content(self):
        """Prueba detección de contenido numérico"""
        self.assertTrue(self.matcher.has_numeric_content("123.45"))
        self.assertTrue(self.matcher.has_numeric_content("Precio: 100"))
        self.assertFalse(self.matcher.has_numeric_content("Solo texto"))
        self.assertTrue(self.matcher.has_numeric_content("12,345.67"))
    
    def test_has_percentage(self):
        """Prueba detección de porcentajes"""
        self.assertTrue(self.matcher.has_percentage("15%"))
        self.assertTrue(self.matcher.has_percentage("15 %"))
        self.assertTrue(self.matcher.has_percentage("Administración 10%"))
        self.assertFalse(self.matcher.has_percentage("Sin porcentaje"))

class UnitsValidatorTests(unittest.TestCase):
    """Pruebas específicas para el componente UnitsValidator"""
    
    def test_normalize_unit(self):
        """Prueba normalización de unidades"""
        # Mapeos directos
        self.assertEqual(UnitsValidator.normalize_unit("MT"), "M")
        self.assertEqual(UnitsValidator.normalize_unit("METRO"), "METRO")
        self.assertEqual(UnitsValidator.normalize_unit("JORNAL"), "JOR")
        self.assertEqual(UnitsValidator.normalize_unit("KILOGRAMO"), "KILOGRAMO")
        
        # Casos especiales
        self.assertEqual(UnitsValidator.normalize_unit(""), "UND")
        self.assertEqual(UnitsValidator.normalize_unit("METROSCUBICOS"), "METROSCUBICOS")
        self.assertEqual(UnitsValidator.normalize_unit("UND."), "UND")
        
        # Unidades no reconocidas
        self.assertEqual(UnitsValidator.normalize_unit("XYZ"), "UND")
    
    def test_is_valid(self):
        """Prueba validación de unidades"""
        # Unidades válidas
        self.assertTrue(UnitsValidator.is_valid("M"))
        self.assertTrue(UnitsValidator.is_valid("KG"))
        self.assertTrue(UnitsValidator.is_valid("JOR"))
        self.assertTrue(UnitsValidator.is_valid("M2"))
        
        # Unidades inválidas
        self.assertFalse(UnitsValidator.is_valid(""))
        self.assertFalse(UnitsValidator.is_valid("METROSCUADRADOSMUYLARGO"))  # Demasiado largo

class NumericFieldExtractorTests(unittest.TestCase):
    """Pruebas específicas para el componente NumericFieldExtractor"""
    
    def setUp(self):
        self.config = TestFixtures.get_default_config()
        self.profile = {"number_format": {"decimal_separator": "."}}
        self.thresholds = ValidationThresholds()
        self.extractor = NumericFieldExtractor(self.config, self.profile, self.thresholds)
    
    def test_parse_number_safe(self):
        """Prueba parseo seguro de números"""
        self.assertEqual(self.extractor.parse_number_safe("1000"), 1000.0)
        self.assertEqual(self.extractor.parse_number_safe("1,000.50"), 1000.50)
        self.assertIsNone(self.extractor.parse_number_safe(""))
        self.assertIsNone(self.extractor.parse_number_safe(None))
        self.assertIsNone(self.extractor.parse_number_safe("texto"))
    
    def test_extract_all_numeric_values(self):
        """Prueba extracción de todos los valores numéricos"""
        fields = ["DESCRIPCION", "UND", "0.5", "", "100000", "50000"]
        values = self.extractor.extract_all_numeric_values(fields)
        self.assertEqual(values, [0.5, 100000.0, 50000.0])
    
    def test_identify_mo_values(self):
        """Prueba identificación de valores de mano de obra"""
        # Caso normal
        values = [0.125, 180000.0, 22500.0]
        result = self.extractor.identify_mo_values(values)
        self.assertEqual(result, (0.125, 180000.0))
        
        # Caso con múltiples valores
        values = [0.08, 0.125, 250000.0, 20000.0]
        result = self.extractor.identify_mo_values(values)
        self.assertEqual(result, (0.08, 250000.0))
        
        # Caso sin valores válidos
        values = [1000.0, 2000.0, 3000.0]  # Todos fuera de rango para jornal
        result = self.extractor.identify_mo_values(values)
        self.assertIsNone(result)
    
    def test_comma_decimal_parsing(self):
        """Prueba parseo con separador decimal de coma"""
        comma_profile = TestFixtures.get_profile_with_comma_decimal()
        extractor = NumericFieldExtractor(self.config, comma_profile, self.thresholds)
        
        self.assertEqual(extractor.parse_number_safe("1,5"), 1.5)
        self.assertEqual(extractor.parse_number_safe("1.000,50"), 1000.50)
        self.assertEqual(extractor.parse_number_safe("250.000,00"), 250000.00)

class APUProcessorTests(unittest.TestCase):
    """Pruebas principales para el procesador APU"""
    
    def setUp(self):
        self.config = TestFixtures.get_default_config()
        self.default_profile = {"number_format": {"decimal_separator": "."}}
        self.comma_profile = TestFixtures.get_profile_with_comma_decimal()
    
    def test_initialization(self):
        """Prueba inicialización correcta del procesador"""
        processor = APUProcessor(self.config, self.default_profile)
        self.assertIsNotNone(processor.config)
        self.assertIsNotNone(processor.profile)
        self.assertIsNotNone(processor.parser)
        self.assertIsInstance(processor.parsing_stats, ParsingStats)
    
    def test_process_grouped_records(self):
        """Prueba procesamiento de registros en formato agrupado"""
        records = TestFixtures.get_grouped_sample_records()
        
        processor = APUProcessor(self.config, self.default_profile)
        processor.raw_records = records
        df = processor.process_all()
        
        # Verificaciones básicas
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertGreater(len(df), 0)
        
        # Verificar columnas esperadas
        expected_columns = [
            "CODIGO_APU", "DESCRIPCION_APU", "UNIDAD_APU",
            "DESCRIPCION_INSUMO", "UNIDAD_INSUMO", "CANTIDAD_APU",
            "PRECIO_UNIT_APU", "VALOR_TOTAL_APU", "RENDIMIENTO",
            "TIPO_INSUMO", "FORMATO_ORIGEN", "CATEGORIA"
        ]
        for column in expected_columns:
            self.assertIn(column, df.columns)
        
        # Verificar tipos de insumo identificados correctamente
        mo_insumos = df[df["TIPO_INSUMO"] == "MANO_DE_OBRA"]
        self.assertGreater(len(mo_insumos), 0)
        
        equipo_insumos = df[df["TIPO_INSUMO"] == "EQUIPO"]
        self.assertGreater(len(equipo_insumos), 0)
    
    def test_process_flat_records(self):
        """Prueba procesamiento de registros en formato plano"""
        records = TestFixtures.get_flat_sample_records()
        
        processor = APUProcessor(self.config, self.default_profile)
        processor.raw_records = records
        df = processor.process_all()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        
        # Verificar que los registros se agruparon correctamente por APU
        apu_codes = df["CODIGO_APU"].unique()
        self.assertEqual(len(apu_codes), 2)
        self.assertIn("1.1", apu_codes)
        self.assertIn("2.1", apu_codes)
    
    def test_process_with_comma_decimal_separator(self):
        """Prueba procesamiento con separador decimal de coma"""
        records = TestFixtures.get_comma_decimal_records()
        
        processor = APUProcessor(self.config, self.comma_profile)
        processor.raw_records = records
        
        # Probar que el parser se inicializa con la configuración correcta
        self.assertIsNotNone(processor.parser)
        
        df = processor.process_all()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        
        # Verificar parseo correcto de números con coma decimal
        concreto_row = df[df["DESCRIPCION_INSUMO"] == "CONCRETO PREMEZCLADO"].iloc[0]
        self.assertAlmostEqual(concreto_row["CANTIDAD_APU"], 0.15)
        self.assertAlmostEqual(concreto_row["PRECIO_UNIT_APU"], 850123.50)
        self.assertAlmostEqual(concreto_row["VALOR_TOTAL_APU"], 127518.53)
        
        cuadrilla_row = df[df["DESCRIPCION_INSUMO"] == "CUADRILLA ESPECIAL"].iloc[0]
        self.assertAlmostEqual(cuadrilla_row["RENDIMIENTO"], 0.08)
        self.assertAlmostEqual(cuadrilla_row["PRECIO_UNIT_APU"], 250000.00)
        self.assertAlmostEqual(cuadrilla_row["CANTIDAD_APU"], 1.0 / 0.08)
    
    def test_edge_cases_handling(self):
        """Prueba manejo de casos extremos y errores"""
        records = TestFixtures.get_edge_case_records()
        
        processor = APUProcessor(self.config, self.default_profile)
        processor.raw_records = records
        df = processor.process_all()
        
        self.assertIsInstance(df, pd.DataFrame)
        
        # Aunque haya líneas inválidas, debería haber algunos resultados válidos
        if not df.empty:
            # Verificar que no hay valores negativos o cero en campos críticos
            self.assertTrue(all(df["CANTIDAD_APU"] > 0))
            self.assertTrue(all(df["VALOR_TOTAL_APU"] > 0))
    
    def test_integration_with_calculate_unit_costs(self):
        """Prueba integración con cálculo de costos unitarios"""
        records = TestFixtures.get_grouped_sample_records()
        
        processor = APUProcessor(self.config, self.default_profile)
        processor.raw_records = records
        df = processor.process_all()
        
        # Renombrar columnas para que coincidan con lo que espera calculate_unit_costs
        df = df.rename(
            columns={
                "CODIGO_APU": "CODIGO_APU",
                "DESCRIPCION_APU": "DESCRIPCION_APU",
                "UNIDAD_APU": "UNIDAD_APU",
                "TIPO_INSUMO": "TIPO_INSUMO",
                "VALOR_TOTAL_APU": "VALOR_TOTAL_APU",
            }
        )
        
        costs_df = calculate_unit_costs(df)
        self.assertFalse(costs_df.empty)
        self.assertIn("COSTO_UNITARIO_TOTAL", costs_df.columns)
        
        # Verificar que hay un costo calculado por cada APU
        apu_count = len(df["CODIGO_APU"].unique())
        self.assertEqual(len(costs_df), apu_count)
        
        # Verificar que los costos son positivos
        self.assertTrue(all(costs_df["COSTO_UNITARIO_TOTAL"] > 0))
    
    def test_processing_statistics(self):
        """Prueba generación de estadísticas de procesamiento"""
        records = TestFixtures.get_grouped_sample_records()
        
        processor = APUProcessor(self.config, self.default_profile)
        processor.raw_records = records
        df = processor.process_all()
        
        # Verificar estadísticas
        stats = processor.parsing_stats
        self.assertGreater(stats.total_lines, 0)
        self.assertGreater(stats.successful_parses, 0)
        self.assertEqual(stats.total_lines, stats.successful_parses + len(stats.failed_lines))
        
        # Verificar estadísticas globales
        self.assertGreater(processor.global_stats["total_apus"], 0)
        self.assertGreater(processor.global_stats["total_insumos"], 0)
        self.assertEqual(processor.global_stats["format_detected"], "grouped")
    
    def test_special_cases_classification(self):
        """Prueba clasificación de casos especiales configurados"""
        records = [
            {
                "codigo_apu": "5.1",
                "descripcion_apu": "Caso Especial",
                "unidad_apu": "UN",
                "lines": [
                    "TRANSPORTE DE EQUIPO;VIAJE;2;50000;100000",
                    "SUBCONTRATO ACABADOS;GLB;;;;75000",
                    "ALQUILER MAQUINA;HR;4;25000;100000"
                ],
            },
        ]
        
        processor = APUProcessor(self.config, self.default_profile)
        processor.raw_records = records
        df = processor.process_all()
        
        self.assertFalse(df.empty)
        
        # Verificar clasificación correcta de casos especiales
        transporte_row = df[df["DESCRIPCION_INSUMO"] == "TRANSPORTE DE EQUIPO"]
        self.assertEqual(len(transporte_row), 1)
        self.assertEqual(transporte_row.iloc[0]["TIPO_INSUMO"], "TRANSPORTE")
        
        subcontrato_row = df[df["DESCRIPCION_INSUMO"] == "SUBCONTRATO ACABADOS"]
        self.assertEqual(len(subcontrato_row), 1)
        self.assertEqual(subcontrato_row.iloc[0]["TIPO_INSUMO"], "OTRO")
        
        alquiler_row = df[df["DESCRIPCION_INSUMO"] == "ALQUILER MAQUINA"]
        self.assertEqual(len(alquiler_row), 1)
        self.assertEqual(alquiler_row.iloc[0]["TIPO_INSUMO"], "EQUIPO")

class APUParserRobustnessTests(unittest.TestCase):
    """Pruebas de robustez para el parser y transformer"""
    
    def setUp(self):
        self.config = TestFixtures.get_default_config()
        self.profile = {"number_format": {"decimal_separator": "."}}
        self.thresholds = ValidationThresholds()
        self.context = {
            "codigo_apu": "TEST",
            "descripcion_apu": "APU de Prueba",
            "unidad_apu": "UN",
            "cantidad_apu": 1.0,
            "precio_unitario_apu": 0.0,
            "categoria": "PRUEBA"
        }
        self.transformer = APUTransformer(
            self.context, 
            self.config, 
            self.profile, 
            {}
        )
    
    def test_transformer_empty_line(self):
        """Prueba transformer con línea vacía"""
        with patch('lark.Tree') as mock_tree:
            mock_tree.data = "line"
            mock_tree.children = []
            result = self.transformer.transform(mock_tree)
            self.assertIsNone(result)
    
    def test_transformer_malformed_line(self):
        """Prueba transformer con línea malformada"""
        with patch('lark.Tree') as mock_tree:
            mock_tree.data = "line"
            mock_tree.children = [Token("FIELD_VALUE", "")]
            result = self.transformer.transform(mock_tree)
            self.assertIsNone(result)
    
    def test_line_with_unexpected_characters(self):
        """Prueba línea con caracteres inesperados"""
        processor = APUProcessor(self.config, self.profile)
        
        # Simular una línea con caracteres inesperados
        bad_line = "CONCRETO\xa0ESPECIAL;M3;0.15;850000;127500"
        tokens = bad_line.split(";")
        
        # Probar el PatternMatcher directamente
        matcher = PatternMatcher()
        self.assertTrue(matcher.has_numeric_content(bad_line))
        
        # Probar el extractor numérico
        extractor = NumericFieldExtractor(self.config, self.profile, self.thresholds)
        values = extractor.extract_all_numeric_values(tokens)
        self.assertEqual(len(values), 3)

if __name__ == "__main__":
    # Ejecutar pruebas con nivel de log INFO para ver el progreso
    logging.basicConfig(level=logging.INFO)
    unittest.main(verbosity=2)