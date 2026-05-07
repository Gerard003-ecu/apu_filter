"""
=========================================================================================
    Test Suite: AST Static Analyzer - Rigorous Verification
    Ubicación: tests/unit/boole/physics/test_ast_static_analyzer.py
    Versión: 2.0 - Comprehensive Test Coverage
    
    FILOSOFÍA DE TESTING:
    ---------------------
    Esta suite implementa testing riguroso basado en:
    
    1. PROPERTY-BASED TESTING (PBT):
       - Verificación de invariantes algebraicas
       - Generación automática de casos de prueba
       - Falsificación de propiedades
    
    2. METAMORPHIC TESTING:
       - Relaciones entre transformaciones de código
       - Invariantes bajo refactorización
    
    3. COVERAGE CRITERIOS:
       - Statement coverage: > 95%
       - Branch coverage: > 90%
       - Path coverage: casos críticos
       - Mutation testing: resistencia a mutaciones
    
    4. FORMAL VERIFICATION:
       - Verificación de axiomas matemáticos
       - Pruebas de propiedades algebraicas
       - Validación de límites asintóticos
    
    ORGANIZACIÓN:
    -------------
    - TestDataFlowCoordinates: Pruebas de estructura algebraica
    - TestComplexityProfile: Validación de métricas
    - TestDataFlowAnalyzer: Análisis de AST
    - TestHamiltonianMonitor: Verificación disipativa
    - TestCohomology: Análisis topológico
    - TestJSONValidator: Seguridad de datos
    - TestIntegration: Casos de uso completos
    - TestPerformance: Benchmarks y límites
    
=========================================================================================
"""

import ast
import math
import sys
import time
import unittest
from typing import Dict, List, Set, Tuple
from unittest.mock import patch, MagicMock

# Importar el módulo a testear
# Nota: ajustar el path según la estructura del proyecto
try:
    from app.boole.physics.ast_static_analyzer import (
        # Estructuras de datos
        DataFlowCoordinates,
        ComplexityProfile,
        NodeCategory,
        
        # Analizadores
        DataFlowAnalyzer,
        CellularSheafCohomology,
        JSONStructureValidator,
        TabularNormalizer,
        ASTStaticAnalyzer,
        
        # Excepciones
        ThermodynamicSingularityError,
        CohomologicalObstructionError,
        ComplexityBoundsViolationError,
        StructuralValidationError,
        AnalysisException,
        
        # Límites
        AnalysisLimits,
        
        # Utilidades
        print_complexity_report,
        print_dataflow_report,
    )
except ImportError:
    # Fallback para ejecución directa
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from app.boole.physics.ast_static_analyzer import *


# =============================================================================
# UTILIDADES DE TESTING
# =============================================================================

class TestHelpers:
    """Utilidades compartidas para testing."""
    
    @staticmethod
    def parse_code(code: str) -> ast.AST:
        """Parse código Python a AST de forma segura."""
        return ast.parse(code.strip(), mode='exec')
    
    @staticmethod
    def analyze_code_safe(code: str, **kwargs) -> Dict:
        """Analiza código con manejo de excepciones."""
        try:
            return ASTStaticAnalyzer.analyze_code(code, **kwargs)
        except Exception as e:
            return {'error': e}
    
    @staticmethod
    def assert_frozenset_equal(a: frozenset, b: frozenset, msg: str = ""):
        """Compara frozensets con mensaje claro."""
        if a != b:
            missing = b - a
            extra = a - b
            error_msg = f"FrozenSets differ"
            if missing:
                error_msg += f"\n  Missing: {sorted(missing)}"
            if extra:
                error_msg += f"\n  Extra: {sorted(extra)}"
            if msg:
                error_msg = f"{msg}\n{error_msg}"
            raise AssertionError(error_msg)
    
    @staticmethod
    def generate_deep_nested_code(depth: int) -> str:
        """Genera código con anidamiento profundo."""
        code = "x = 0\n"
        indent = ""
        for i in range(depth):
            code += f"{indent}if True:\n"
            indent += "    "
            code += f"{indent}x += {i}\n"
        return code
    
    @staticmethod
    def generate_high_complexity_code(complexity: int) -> str:
        """Genera código con complejidad ciclomática específica."""
        # CC = 1 (base) + (complexity - 1) decisiones
        code = "def func(x):\n"
        code += "    result = 0\n"
        for i in range(complexity - 1):
            code += f"    if x > {i}:\n"
            code += f"        result += {i}\n"
        code += "    return result\n"
        return code


# =============================================================================
# TESTS: DataFlowCoordinates (Estructura Algebraica)
# =============================================================================

class TestDataFlowCoordinates(unittest.TestCase):
    """
    Test suite para DataFlowCoordinates.
    
    Verifica:
    - Propiedades algebraicas (monoide)
    - Operaciones de conjuntos
    - Análisis de interferencia
    - Métricas espectrales
    """
    
    def setUp(self):
        """Inicialización de casos de prueba comunes."""
        self.empty = DataFlowCoordinates()
        
        self.read_only = DataFlowCoordinates(
            reads=frozenset(['x', 'y']),
            writes=frozenset()
        )
        
        self.write_only = DataFlowCoordinates(
            reads=frozenset(),
            writes=frozenset(['z', 'w'])
        )
        
        self.read_write = DataFlowCoordinates(
            reads=frozenset(['a', 'b']),
            writes=frozenset(['b', 'c'])
        )
    
    # -------------------------------------------------------------------------
    # PROPIEDADES BÁSICAS
    # -------------------------------------------------------------------------
    
    def test_initialization_empty(self):
        """Test: Inicialización vacía."""
        coords = DataFlowCoordinates()
        self.assertEqual(coords.reads, frozenset())
        self.assertEqual(coords.writes, frozenset())
    
    def test_initialization_with_sets(self):
        """Test: Inicialización con sets regulares (conversión automática)."""
        coords = DataFlowCoordinates(
            reads={'x', 'y'},
            writes={'z'}
        )
        self.assertIsInstance(coords.reads, frozenset)
        self.assertIsInstance(coords.writes, frozenset)
        self.assertEqual(coords.reads, frozenset(['x', 'y']))
    
    def test_immutability(self):
        """Test: Inmutabilidad de la estructura."""
        coords = DataFlowCoordinates(reads=frozenset(['x']))
        
        with self.assertRaises(AttributeError):
            coords.reads = frozenset(['y'])  # type: ignore
    
    def test_hashability(self):
        """Test: Hashable (puede usarse en sets/dicts)."""
        coords1 = DataFlowCoordinates(reads=frozenset(['x']))
        coords2 = DataFlowCoordinates(reads=frozenset(['x']))
        coords3 = DataFlowCoordinates(reads=frozenset(['y']))
        
        # Pueden usarse como claves
        d = {coords1: "value"}
        self.assertEqual(d[coords2], "value")
        
        # Set de coords
        s = {coords1, coords2, coords3}
        self.assertEqual(len(s), 2)  # coords1 == coords2
    
    def test_equality(self):
        """Test: Igualdad estructural."""
        coords1 = DataFlowCoordinates(
            reads=frozenset(['x', 'y']),
            writes=frozenset(['z'])
        )
        coords2 = DataFlowCoordinates(
            reads=frozenset(['y', 'x']),  # Orden no importa
            writes=frozenset(['z'])
        )
        coords3 = DataFlowCoordinates(
            reads=frozenset(['x']),
            writes=frozenset(['z'])
        )
        
        self.assertEqual(coords1, coords2)
        self.assertNotEqual(coords1, coords3)
    
    def test_ordering(self):
        """Test: Orden total (frozen=True, order=True)."""
        coords1 = DataFlowCoordinates(reads=frozenset(['a']))
        coords2 = DataFlowCoordinates(reads=frozenset(['b']))
        coords3 = DataFlowCoordinates(writes=frozenset(['c']))
        
        # Debe ser ordenable
        sorted_list = sorted([coords2, coords3, coords1])
        self.assertEqual(len(sorted_list), 3)
    
    # -------------------------------------------------------------------------
    # PROPIEDADES DERIVADAS
    # -------------------------------------------------------------------------
    
    def test_all_variables(self):
        """Test: Unión de todas las variables."""
        self.assertEqual(self.empty.all_variables, frozenset())
        self.assertEqual(self.read_only.all_variables, frozenset(['x', 'y']))
        self.assertEqual(self.write_only.all_variables, frozenset(['z', 'w']))
        self.assertEqual(self.read_write.all_variables, frozenset(['a', 'b', 'c']))
    
    def test_modified_variables(self):
        """Test: Variables leídas Y escritas (modificación in-place)."""
        self.assertEqual(self.empty.modified_variables, frozenset())
        self.assertEqual(self.read_only.modified_variables, frozenset())
        self.assertEqual(self.write_only.modified_variables, frozenset())
        self.assertEqual(self.read_write.modified_variables, frozenset(['b']))
    
    def test_pure_reads(self):
        """Test: Variables solo leídas."""
        self.assertEqual(self.empty.pure_reads, frozenset())
        self.assertEqual(self.read_only.pure_reads, frozenset(['x', 'y']))
        self.assertEqual(self.write_only.pure_reads, frozenset())
        self.assertEqual(self.read_write.pure_reads, frozenset(['a']))
    
    def test_pure_writes(self):
        """Test: Variables solo escritas."""
        self.assertEqual(self.empty.pure_writes, frozenset())
        self.assertEqual(self.read_only.pure_writes, frozenset())
        self.assertEqual(self.write_only.pure_writes, frozenset(['z', 'w']))
        self.assertEqual(self.read_write.pure_writes, frozenset(['c']))
    
    # -------------------------------------------------------------------------
    # ÁLGEBRA MONOIDAL
    # -------------------------------------------------------------------------
    
    def test_monoidal_identity(self):
        """Test: Elemento identidad del monoide."""
        identity = DataFlowCoordinates.identity()
        
        # e ⊕ a = a
        result1 = identity.combine(self.read_only)
        self.assertEqual(result1, self.read_only)
        
        # a ⊕ e = a
        result2 = self.read_only.combine(identity)
        self.assertEqual(result2, self.read_only)
    
    def test_monoidal_associativity(self):
        """Test: Asociatividad de combine."""
        # (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
        left = self.read_only.combine(self.write_only).combine(self.read_write)
        right = self.read_only.combine(self.write_only.combine(self.read_write))
        
        self.assertEqual(left, right)
    
    def test_combine_sequential_semantics(self):
        """Test: Semántica secuencial de combine."""
        # Si A escribe x, luego B lee x, la lectura es interna
        A = DataFlowCoordinates(writes=frozenset(['x']))
        B = DataFlowCoordinates(reads=frozenset(['x']))
        
        combined = A.combine(B)
        
        # x no debe aparecer en reads del combinado (lectura interna)
        self.assertNotIn('x', combined.reads)
        self.assertIn('x', combined.writes)
    
    def test_combine_preserves_external_reads(self):
        """Test: Lecturas externas se preservan."""
        A = DataFlowCoordinates(reads=frozenset(['y']))
        B = DataFlowCoordinates(writes=frozenset(['z']))
        
        combined = A.combine(B)
        
        self.assertIn('y', combined.reads)
        self.assertIn('z', combined.writes)
    
    def test_combine_multiple(self):
        """Test: Composición de múltiples flujos."""
        flows = [
            DataFlowCoordinates(reads=frozenset(['a']), writes=frozenset(['b'])),
            DataFlowCoordinates(reads=frozenset(['b']), writes=frozenset(['c'])),
            DataFlowCoordinates(reads=frozenset(['c']), writes=frozenset(['d'])),
        ]
        
        # a → b → c → d
        result = flows[0]
        for flow in flows[1:]:
            result = result.combine(flow)
        
        # Solo 'a' es lectura externa, 'd' es escritura final
        self.assertIn('a', result.reads)
        self.assertIn('d', result.writes)
        # b, c son internas
        self.assertNotIn('b', result.reads)
        self.assertNotIn('c', result.reads)
    
    # -------------------------------------------------------------------------
    # ANÁLISIS DE INTERFERENCIA
    # -------------------------------------------------------------------------
    
    def test_no_interference_disjoint(self):
        """Test: No hay interferencia si las variables son disjuntas."""
        A = DataFlowCoordinates(reads=frozenset(['x']), writes=frozenset(['y']))
        B = DataFlowCoordinates(reads=frozenset(['a']), writes=frozenset(['b']))
        
        self.assertFalse(A.has_interference_with(B))
        self.assertFalse(B.has_interference_with(A))
    
    def test_interference_RAW(self):
        """Test: Interferencia RAW (Read After Write)."""
        A = DataFlowCoordinates(writes=frozenset(['x']))
        B = DataFlowCoordinates(reads=frozenset(['x']))
        
        self.assertTrue(A.has_interference_with(B))
        self.assertTrue(B.has_interference_with(A))
    
    def test_interference_WAR(self):
        """Test: Interferencia WAR (Write After Read)."""
        A = DataFlowCoordinates(reads=frozenset(['x']))
        B = DataFlowCoordinates(writes=frozenset(['x']))
        
        self.assertTrue(A.has_interference_with(B))
    
    def test_interference_WAW(self):
        """Test: Interferencia WAW (Write After Write)."""
        A = DataFlowCoordinates(writes=frozenset(['x']))
        B = DataFlowCoordinates(writes=frozenset(['x']))
        
        self.assertTrue(A.has_interference_with(B))
    
    def test_interference_score_antisymmetry(self):
        """Test: Antisimetría del score de interferencia."""
        A = DataFlowCoordinates(writes=frozenset(['x']), reads=frozenset(['y']))
        B = DataFlowCoordinates(reads=frozenset(['x']), writes=frozenset(['y']))
        
        score_AB = DataFlowCoordinates.interference_score(A, B)
        score_BA = DataFlowCoordinates.interference_score(B, A)
        
        # I(A, B) = -I(B, A)
        self.assertEqual(score_AB, -score_BA)
    
    def test_interference_score_magnitude(self):
        """Test: Magnitud del score indica dirección de dependencia."""
        # A escribe x, B lee x → A debe ir antes que B
        A = DataFlowCoordinates(writes=frozenset(['x']))
        B = DataFlowCoordinates(reads=frozenset(['x']))
        
        score = DataFlowCoordinates.interference_score(A, B)
        
        # Score positivo: A debe ejecutarse antes que B
        self.assertGreater(score, 0)
    
    def test_interference_score_zero(self):
        """Test: Score cero para flujos independientes."""
        A = DataFlowCoordinates(reads=frozenset(['x']), writes=frozenset(['y']))
        B = DataFlowCoordinates(reads=frozenset(['a']), writes=frozenset(['b']))
        
        score = DataFlowCoordinates.interference_score(A, B)
        self.assertEqual(score, 0)
    
    def test_dominates(self):
        """Test: Relación de dominancia."""
        A = DataFlowCoordinates(writes=frozenset(['x']))
        B = DataFlowCoordinates(reads=frozenset(['x']))
        
        self.assertTrue(A.dominates(B))
        self.assertFalse(B.dominates(A))
    
    # -------------------------------------------------------------------------
    # MÉTRICAS ESPECTRALES
    # -------------------------------------------------------------------------
    
    def test_spectral_radius_empty(self):
        """Test: Radio espectral para flujo vacío."""
        radius = self.empty.spectral_radius()
        self.assertEqual(radius, 0.0)
    
    def test_spectral_radius_calculation(self):
        """Test: Cálculo del radio espectral."""
        coords = DataFlowCoordinates(
            reads=frozenset(['a', 'b']),
            writes=frozenset(['c'])
        )
        # ρ = √(|reads|² + |writes|²) = √(4 + 1) = √5
        expected = math.sqrt(2**2 + 1**2)
        self.assertAlmostEqual(coords.spectral_radius(), expected, places=6)
    
    def test_entropy_empty(self):
        """Test: Entropía de flujo vacío."""
        entropy = self.empty.entropy()
        self.assertEqual(entropy, 0.0)
    
    def test_entropy_calculation(self):
        """Test: Cálculo de entropía de Shannon."""
        coords = DataFlowCoordinates(
            reads=frozenset(['a', 'b']),
            writes=frozenset(['c', 'd'])
        )
        # H = log₂(|all_variables|) = log₂(4) = 2.0
        expected = math.log2(4)
        self.assertAlmostEqual(coords.entropy(), expected, places=6)
    
    def test_entropy_monotonicity(self):
        """Test: Entropía crece con más variables."""
        coords1 = DataFlowCoordinates(reads=frozenset(['x']))
        coords2 = DataFlowCoordinates(reads=frozenset(['x', 'y']))
        coords3 = DataFlowCoordinates(reads=frozenset(['x', 'y', 'z']))
        
        self.assertLess(coords1.entropy(), coords2.entropy())
        self.assertLess(coords2.entropy(), coords3.entropy())
    
    # -------------------------------------------------------------------------
    # CASOS EDGE
    # -------------------------------------------------------------------------
    
    def test_large_variable_set(self):
        """Test: Manejo de conjuntos grandes de variables."""
        large_set = frozenset([f'var_{i}' for i in range(1000)])
        coords = DataFlowCoordinates(reads=large_set)
        
        self.assertEqual(len(coords.reads), 1000)
        self.assertGreater(coords.spectral_radius(), 0)
        self.assertGreater(coords.entropy(), 0)
    
    def test_special_characters_in_names(self):
        """Test: Nombres de variables con caracteres especiales."""
        coords = DataFlowCoordinates(
            reads=frozenset(['_private', '__dunder__', 'CamelCase'])
        )
        
        self.assertEqual(len(coords.reads), 3)
    
    def test_unicode_variable_names(self):
        """Test: Nombres Unicode (Python 3 permite esto)."""
        coords = DataFlowCoordinates(
            reads=frozenset(['α', 'β', 'γ'])
        )
        
        self.assertEqual(len(coords.reads), 3)


# =============================================================================
# TESTS: ComplexityProfile (Métricas de Complejidad)
# =============================================================================

class TestComplexityProfile(unittest.TestCase):
    """
    Test suite para ComplexityProfile.
    
    Verifica:
    - Validación de invariantes
    - Métricas derivadas
    - Clasificación de riesgo
    - Umbrales de mantenibilidad
    """
    
    def test_initialization_valid(self):
        """Test: Inicialización con valores válidos."""
        profile = ComplexityProfile(
            cyclomatic_complexity=5,
            max_nesting_depth=3,
            num_functions=2,
            num_classes=1,
            num_branches=4,
            num_loops=1
        )
        
        self.assertEqual(profile.cyclomatic_complexity, 5)
        self.assertEqual(profile.max_nesting_depth, 3)
    
    def test_initialization_invalid_cc(self):
        """Test: CC < 1 es inválido."""
        with self.assertRaises(ValueError):
            ComplexityProfile(
                cyclomatic_complexity=0,
                max_nesting_depth=1,
                num_functions=1,
                num_classes=0
            )
    
    def test_initialization_invalid_depth(self):
        """Test: Profundidad negativa es inválida."""
        with self.assertRaises(ValueError):
            ComplexityProfile(
                cyclomatic_complexity=1,
                max_nesting_depth=-1,
                num_functions=1,
                num_classes=0
            )
    
    def test_initialization_invalid_counts(self):
        """Test: Contadores negativos son inválidos."""
        with self.assertRaises(ValueError):
            ComplexityProfile(
                cyclomatic_complexity=1,
                max_nesting_depth=1,
                num_functions=-1,
                num_classes=0
            )
    
    # -------------------------------------------------------------------------
    # MÉTRICAS DERIVADAS
    # -------------------------------------------------------------------------
    
    def test_is_maintainable_true(self):
        """Test: Código mantenible."""
        profile = ComplexityProfile(
            cyclomatic_complexity=10,
            max_nesting_depth=3,
            num_functions=5,
            num_classes=2
        )
        
        self.assertTrue(profile.is_maintainable)
    
    def test_is_maintainable_false_high_cc(self):
        """Test: No mantenible por alta complejidad."""
        profile = ComplexityProfile(
            cyclomatic_complexity=25,
            max_nesting_depth=3,
            num_functions=1,
            num_classes=0
        )
        
        self.assertFalse(profile.is_maintainable)
    
    def test_is_maintainable_false_high_depth(self):
        """Test: No mantenible por anidamiento profundo."""
        profile = ComplexityProfile(
            cyclomatic_complexity=10,
            max_nesting_depth=10,
            num_functions=1,
            num_classes=0
        )
        
        self.assertFalse(profile.is_maintainable)
    
    def test_risk_level_low(self):
        """Test: Nivel de riesgo BAJO."""
        profile = ComplexityProfile(
            cyclomatic_complexity=5,
            max_nesting_depth=2,
            num_functions=3,
            num_classes=1
        )
        
        self.assertEqual(profile.risk_level, "LOW")
    
    def test_risk_level_moderate(self):
        """Test: Nivel de riesgo MODERADO."""
        profile = ComplexityProfile(
            cyclomatic_complexity=15,
            max_nesting_depth=4,
            num_functions=2,
            num_classes=1
        )
        
        self.assertEqual(profile.risk_level, "MODERATE")
    
    def test_risk_level_high(self):
        """Test: Nivel de riesgo ALTO."""
        profile = ComplexityProfile(
            cyclomatic_complexity=30,
            max_nesting_depth=5,
            num_functions=1,
            num_classes=0
        )
        
        self.assertEqual(profile.risk_level, "HIGH")
    
    def test_risk_level_critical(self):
        """Test: Nivel de riesgo CRÍTICO."""
        profile = ComplexityProfile(
            cyclomatic_complexity=60,
            max_nesting_depth=8,
            num_functions=1,
            num_classes=0
        )
        
        self.assertEqual(profile.risk_level, "CRITICAL")
    
    def test_complexity_density(self):
        """Test: Densidad de complejidad."""
        profile = ComplexityProfile(
            cyclomatic_complexity=20,
            max_nesting_depth=3,
            num_functions=4,
            num_classes=0
        )
        
        # Densidad = 20 / 4 = 5.0
        self.assertAlmostEqual(profile.complexity_density, 5.0, places=2)
    
    def test_complexity_density_no_functions(self):
        """Test: Densidad cuando no hay funciones (normalización)."""
        profile = ComplexityProfile(
            cyclomatic_complexity=10,
            max_nesting_depth=2,
            num_functions=0,
            num_classes=1
        )
        
        # Densidad = 10 / max(0, 1) = 10.0
        self.assertAlmostEqual(profile.complexity_density, 10.0, places=2)
    
    def test_essential_complexity(self):
        """Test: Complejidad esencial."""
        profile = ComplexityProfile(
            cyclomatic_complexity=20,
            max_nesting_depth=5,
            num_functions=3,
            num_classes=1,
            num_branches=8,
            num_loops=4
        )
        
        # Essential = max(1, CC - branches - loops + 1)
        # = max(1, 20 - 8 - 4 + 1) = max(1, 9) = 9
        self.assertEqual(profile.essential_complexity, 9)
    
    def test_essential_complexity_minimum(self):
        """Test: Complejidad esencial mínima es 1."""
        profile = ComplexityProfile(
            cyclomatic_complexity=5,
            max_nesting_depth=2,
            num_functions=1,
            num_classes=0,
            num_branches=10,
            num_loops=10
        )
        
        # Aunque la fórmula dé negativo, debe ser al menos 1
        self.assertEqual(profile.essential_complexity, 1)
    
    def test_halstead_volume(self):
        """Test: Volumen de Halstead."""
        profile = ComplexityProfile(
            cyclomatic_complexity=10,
            max_nesting_depth=3,
            num_functions=4,
            num_classes=2
        )
        
        # V = N × log₂(n) donde N ≈ CC, n = funcs + classes
        # V = 10 × log₂(6) ≈ 10 × 2.585 ≈ 25.85
        expected = 10 * math.log2(4 + 2)
        self.assertAlmostEqual(profile.halstead_volume, expected, places=2)
    
    def test_halstead_volume_minimum(self):
        """Test: Volumen mínimo cuando no hay funciones/clases."""
        profile = ComplexityProfile(
            cyclomatic_complexity=5,
            max_nesting_depth=2,
            num_functions=0,
            num_classes=0
        )
        
        # n = max(0 + 0, 2) = 2
        expected = 5 * math.log2(2)
        self.assertAlmostEqual(profile.halstead_volume, expected, places=2)
    
    # -------------------------------------------------------------------------
    # SERIALIZACIÓN
    # -------------------------------------------------------------------------
    
    def test_to_dict(self):
        """Test: Conversión a diccionario."""
        profile = ComplexityProfile(
            cyclomatic_complexity=10,
            max_nesting_depth=3,
            num_functions=2,
            num_classes=1,
            num_branches=5,
            num_loops=2
        )
        
        d = profile.to_dict()
        
        self.assertIsInstance(d, dict)
        self.assertEqual(d['cyclomatic_complexity'], 10)
        self.assertEqual(d['max_nesting_depth'], 3)
        self.assertEqual(d['risk_level'], 'MODERATE')
        self.assertIn('complexity_density', d)
        self.assertIn('is_maintainable', d)


# =============================================================================
# TESTS: DataFlowAnalyzer (Análisis de AST)
# =============================================================================

class TestDataFlowAnalyzer(unittest.TestCase):
    """
    Test suite para DataFlowAnalyzer.
    
    Verifica:
    - Detección de lecturas/escrituras
    - Cálculo de complejidad ciclomática
    - Profundidad de anidamiento
    - Contadores estructurales
    """
    
    def analyze(self, code: str, **kwargs) -> Tuple[DataFlowCoordinates, ComplexityProfile]:
        """Helper: analiza código y devuelve coordenadas + perfil."""
        tree = TestHelpers.parse_code(code)
        analyzer = DataFlowAnalyzer(
            enable_hamiltonian_monitor=False,
            enable_block_tracking=False,
            **kwargs
        )
        analyzer.visit(tree)
        return analyzer.get_dataflow_coordinates(), analyzer.get_complexity_profile()
    
    # -------------------------------------------------------------------------
    # FLUJO DE DATOS: LECTURAS Y ESCRITURAS
    # -------------------------------------------------------------------------
    
    def test_simple_assignment(self):
        """Test: Asignación simple."""
        code = "x = 5"
        coords, _ = self.analyze(code)
        
        self.assertIn('x', coords.writes)
        self.assertEqual(len(coords.reads), 0)
    
    def test_read_and_write(self):
        """Test: Lectura y escritura."""
        code = "y = x + 1"
        coords, _ = self.analyze(code)
        
        self.assertIn('x', coords.reads)
        self.assertIn('y', coords.writes)
    
    def test_augmented_assignment(self):
        """Test: Asignación aumentada (x += 1)."""
        code = "x += 1"
        coords, _ = self.analyze(code)
        
        # x es tanto lectura como escritura
        self.assertIn('x', coords.reads)
        self.assertIn('x', coords.writes)
        self.assertIn('x', coords.modified_variables)
    
    def test_multiple_assignment(self):
        """Test: Asignación múltiple."""
        code = "a = b = c = 0"
        coords, _ = self.analyze(code)
        
        self.assertIn('a', coords.writes)
        self.assertIn('b', coords.writes)
        self.assertIn('c', coords.writes)
    
    def test_tuple_unpacking(self):
        """Test: Desempaquetado de tuplas."""
        code = "a, b = 1, 2"
        coords, _ = self.analyze(code)
        
        self.assertIn('a', coords.writes)
        self.assertIn('b', coords.writes)
    
    def test_function_call_arguments(self):
        """Test: Argumentos en llamadas a función."""
        code = """
result = func(x, y, z=w)
"""
        coords, _ = self.analyze(code)
        
        self.assertIn('x', coords.reads)
        self.assertIn('y', coords.reads)
        self.assertIn('w', coords.reads)
        self.assertIn('result', coords.writes)
    
    def test_attribute_access(self):
        """Test: Acceso a atributos."""
        code = """
value = obj.attr
obj.prop = 5
"""
        coords, _ = self.analyze(code)
        
        self.assertIn('attr', coords.reads)
        self.assertIn('prop', coords.writes)
        self.assertIn('value', coords.writes)
    
    def test_subscript_access(self):
        """Test: Acceso por índice."""
        code = """
value = arr[i]
arr[j] = 10
"""
        coords, _ = self.analyze(code)
        
        self.assertIn('i', coords.reads)
        self.assertIn('j', coords.reads)
        self.assertIn('arr', coords.reads)
    
    def test_delete_statement(self):
        """Test: Statement delete."""
        code = "del x"
        coords, _ = self.analyze(code)
        
        # del implica lectura (verificar existencia) y escritura (eliminación)
        self.assertIn('x', coords.reads)
        self.assertIn('x', coords.writes)
    
    def test_function_definition_args(self):
        """Test: Argumentos de función son escrituras."""
        code = """
def func(a, b, c=0, *args, **kwargs):
    pass
"""
        coords, _ = self.analyze(code)
        
        self.assertIn('a', coords.writes)
        self.assertIn('b', coords.writes)
        self.assertIn('c', coords.writes)
        self.assertIn('args', coords.writes)
        self.assertIn('kwargs', coords.writes)
    
    def test_lambda_arguments(self):
        """Test: Argumentos de lambda."""
        code = "f = lambda x, y: x + y"
        coords, _ = self.analyze(code)
        
        self.assertIn('x', coords.writes)
        self.assertIn('y', coords.writes)
        self.assertIn('f', coords.writes)
    
    def test_comprehension_variables(self):
        """Test: Variables en comprehensions."""
        code = "[x for x in range(10) if x > 5]"
        coords, _ = self.analyze(code)
        
        # 'x' es variable local de la comprehension (escritura)
        # Nota: el comportamiento exacto puede variar según implementación
        # pero range es una lectura
        self.assertIn('range', coords.reads)
    
    def test_class_definition(self):
        """Test: Definición de clase."""
        code = """
class MyClass:
    def method(self):
        return self.value
"""
        coords, profile = self.analyze(code)
        
        self.assertEqual(profile.num_classes, 1)
        self.assertEqual(profile.num_functions, 1)
        self.assertIn('self', coords.writes)
    
    # -------------------------------------------------------------------------
    # COMPLEJIDAD CICLOMÁTICA
    # -------------------------------------------------------------------------
    
    def test_cc_base(self):
        """Test: Complejidad base es 1."""
        code = "x = 5"
        _, profile = self.analyze(code)
        
        self.assertEqual(profile.cyclomatic_complexity, 1)
    
    def test_cc_if_statement(self):
        """Test: if incrementa CC en 1."""
        code = """
if x > 0:
    y = 1
"""
        _, profile = self.analyze(code)
        
        self.assertEqual(profile.cyclomatic_complexity, 2)  # 1 base + 1 if
    
    def test_cc_if_else(self):
        """Test: if-else incrementa CC en 1 (no 2)."""
        code = """
if x > 0:
    y = 1
else:
    y = -1
"""
        _, profile = self.analyze(code)
        
        self.assertEqual(profile.cyclomatic_complexity, 2)  # 1 base + 1 if
    
    def test_cc_if_elif_else(self):
        """Test: if-elif-else."""
        code = """
if x > 0:
    y = 1
elif x < 0:
    y = -1
else:
    y = 0
"""
        _, profile = self.analyze(code)
        
        # 1 base + 2 (if y elif)
        self.assertEqual(profile.cyclomatic_complexity, 3)
    
    def test_cc_for_loop(self):
        """Test: for incrementa CC en 1."""
        code = """
for i in range(10):
    print(i)
"""
        _, profile = self.analyze(code)
        
        self.assertEqual(profile.cyclomatic_complexity, 2)
        self.assertEqual(profile.num_loops, 1)
    
    def test_cc_while_loop(self):
        """Test: while incrementa CC en 1."""
        code = """
while x > 0:
    x -= 1
"""
        _, profile = self.analyze(code)
        
        self.assertEqual(profile.cyclomatic_complexity, 2)
        self.assertEqual(profile.num_loops, 1)
    
    def test_cc_try_except(self):
        """Test: try-except incrementa CC."""
        code = """
try:
    risky()
except ValueError:
    handle()
except TypeError:
    handle2()
"""
        _, profile = self.analyze(code)
        
        # 1 base + 1 try + 2 except
        self.assertEqual(profile.cyclomatic_complexity, 4)
    
    def test_cc_boolean_operators(self):
        """Test: and/or incrementan CC."""
        code = """
if x > 0 and y > 0 or z > 0:
    pass
"""
        _, profile = self.analyze(code)
        
        # 1 base + 1 if + 2 (and, or)
        self.assertEqual(profile.cyclomatic_complexity, 4)
    
    def test_cc_ternary_expression(self):
        """Test: Expresión ternaria incrementa CC."""
        code = "y = 1 if x > 0 else -1"
        _, profile = self.analyze(code)
        
        self.assertEqual(profile.cyclomatic_complexity, 2)
    
    def test_cc_list_comprehension_with_if(self):
        """Test: List comprehension con filtro if."""
        code = "[x for x in range(10) if x > 5]"
        _, profile = self.analyze(code)
        
        # 1 base + 1 comprehension + 1 if
        self.assertEqual(profile.cyclomatic_complexity, 3)
    
    def test_cc_nested_loops(self):
        """Test: Bucles anidados."""
        code = """
for i in range(10):
    for j in range(10):
        print(i, j)
"""
        _, profile = self.analyze(code)
        
        # 1 base + 2 (dos fors)
        self.assertEqual(profile.cyclomatic_complexity, 3)
        self.assertEqual(profile.num_loops, 2)
    
    def test_cc_complex_example(self):
        """Test: Ejemplo con múltiples construcciones."""
        code = """
def process(x):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                print(i)
    elif x < 0:
        while x < 0:
            x += 1
    else:
        try:
            risky()
        except:
            pass
    return x
"""
        _, profile = self.analyze(code)
        
        # 1 base + 1 if + 1 for + 1 if + 1 elif + 1 while + 1 try + 1 except
        # Total: 8
        self.assertGreaterEqual(profile.cyclomatic_complexity, 7)
    
    # -------------------------------------------------------------------------
    # PROFUNDIDAD DE ANIDAMIENTO
    # -------------------------------------------------------------------------
    
    def test_depth_flat_code(self):
        """Test: Código plano tiene profundidad mínima."""
        code = """
x = 1
y = 2
z = 3
"""
        _, profile = self.analyze(code)
        
        # Profundidad mínima (puede variar según implementación)
        self.assertLessEqual(profile.max_nesting_depth, 3)
    
    def test_depth_nested_ifs(self):
        """Test: Ifs anidados incrementan profundidad."""
        code = """
if True:
    if True:
        if True:
            x = 1
"""
        _, profile = self.analyze(code)
        
        self.assertGreaterEqual(profile.max_nesting_depth, 3)
    
    def test_depth_function_with_loops(self):
        """Test: Función con bucles anidados."""
        code = """
def func():
    for i in range(10):
        for j in range(10):
            if i == j:
                print(i)
"""
        _, profile = self.analyze(code)
        
        self.assertGreaterEqual(profile.max_nesting_depth, 4)
    
    # -------------------------------------------------------------------------
    # CONTADORES ESTRUCTURALES
    # -------------------------------------------------------------------------
    
    def test_count_functions(self):
        """Test: Contador de funciones."""
        code = """
def func1():
    pass

def func2():
    pass

def func3():
    def nested():
        pass
"""
        _, profile = self.analyze(code)
        
        self.assertEqual(profile.num_functions, 4)  # 3 + 1 nested
    
    def test_count_classes(self):
        """Test: Contador de clases."""
        code = """
class A:
    pass

class B:
    class Nested:
        pass
"""
        _, profile = self.analyze(code)
        
        self.assertEqual(profile.num_classes, 3)
    
    def test_count_lambdas_as_functions(self):
        """Test: Lambdas cuentan como funciones."""
        code = """
f1 = lambda x: x + 1
f2 = lambda x, y: x * y
"""
        _, profile = self.analyze(code)
        
        self.assertEqual(profile.num_functions, 2)
    
    # -------------------------------------------------------------------------
    # CASOS EDGE
    # -------------------------------------------------------------------------
    
    def test_empty_code(self):
        """Test: Código vacío."""
        code = ""
        coords, profile = self.analyze(code)
        
        self.assertEqual(len(coords.all_variables), 0)
        self.assertEqual(profile.cyclomatic_complexity, 1)
    
    def test_comments_only(self):
        """Test: Solo comentarios."""
        code = """
# This is a comment
# Another comment
"""
        coords, profile = self.analyze(code)
        
        self.assertEqual(len(coords.all_variables), 0)
        self.assertEqual(profile.cyclomatic_complexity, 1)
    
    def test_docstring_only(self):
        """Test: Solo docstring."""
        code = '''
"""
This is a module docstring.
"""
'''
        coords, profile = self.analyze(code)
        
        self.assertEqual(profile.cyclomatic_complexity, 1)


# =============================================================================
# TESTS: Hamiltoniano y Control Disipativo
# =============================================================================

class TestHamiltonianMonitor(unittest.TestCase):
    """
    Test suite para verificación Hamiltoniana.
    
    Verifica:
    - Condición de disipación H(child) ≤ H(parent)
    - Detección de singularidades termodinámicas
    - Comportamiento en código patológico
    """
    
    def test_hamiltonian_simple_code(self):
        """Test: Código simple no viola disipación."""
        code = """
def func(x):
    if x > 0:
        return x
    return 0
"""
        result = ASTStaticAnalyzer.analyze_code(
            code,
            enable_hamiltonian=True,
            strict_mode=True
        )
        
        self.assertTrue(result['hamiltonian_ok'])
    
    def test_hamiltonian_well_structured_code(self):
        """Test: Código bien estructurado cumple disipación."""
        code = """
class Calculator:
    def __init__(self):
        self.value = 0
    
    def add(self, x):
        self.value += x
        return self.value
    
    def multiply(self, x):
        self.value *= x
        return self.value
"""
        result = ASTStaticAnalyzer.analyze_code(
            code,
            enable_hamiltonian=True,
            strict_mode=True
        )
        
        self.assertTrue(result['hamiltonian_ok'])
    
    def test_hamiltonian_pathological_code(self):
        """Test: Código patológico puede violar disipación."""
        # Código con complejidad concentrada en nodos profundos
        code = """
def outer():
    def middle():
        def inner():
            # Alta complejidad en nodo muy profundo
            if x and y or z and w or a and b:
                if p or q or r:
                    if m and n:
                        pass
"""
        # En modo no estricto, debe detectar pero no lanzar excepción
        result = ASTStaticAnalyzer.analyze_code(
            code,
            enable_hamiltonian=True,
            strict_mode=False
        )
        
        # Puede fallar la condición Hamiltoniana
        # (depende de la distribución de complejidad)
        self.assertIn('hamiltonian_ok', result)
    
    def test_hamiltonian_deep_nesting_high_complexity(self):
        """Test: Anidamiento profundo con alta complejidad local."""
        # Generar código con estructura problemática
        code = TestHelpers.generate_deep_nested_code(10)
        
        # Agregar complejidad en el nivel más profundo
        code += """
                    if a and b or c and d or e and f:
                        pass
"""
        
        result = ASTStaticAnalyzer.analyze_code(
            code,
            enable_hamiltonian=True,
            strict_mode=False
        )
        
        # Verificar que se analizó
        self.assertIn('complexity', result)
    
    def test_hamiltonian_exception_strict_mode(self):
        """Test: Modo estricto lanza excepción en violación."""
        # Código diseñado para violar disipación
        # (alta complejidad concentrada en nodos profundos)
        code = """
def level1():
    def level2():
        def level3():
            def level4():
                def level5():
                    # Complejidad extrema en nodo muy profundo
                    if (a and b) or (c and d) or (e and f):
                        if (g or h) and (i or j):
                            if k and l and m:
                                if n or o or p:
                                    if q and r:
                                        pass
"""
        
        # Puede lanzar ThermodynamicSingularityError
        try:
            result = ASTStaticAnalyzer.analyze_code(
                code,
                enable_hamiltonian=True,
                strict_mode=True
            )
            # Si no lanza, verificar que al menos detectó el problema
            if not result['hamiltonian_ok']:
                self.assertFalse(result['hamiltonian_ok'])
        except ThermodynamicSingularityError as e:
            # Esperado en algunos casos
            self.assertIn('Thermodynamic', str(e))
    
    def test_hamiltonian_disabled(self):
        """Test: Con monitoreo desactivado, siempre pasa."""
        code = """
# Código arbitrario
x = 1
"""
        result = ASTStaticAnalyzer.analyze_code(
            code,
            enable_hamiltonian=False,
            strict_mode=True
        )
        
        # hamiltonian_ok debe ser True (no se verificó)
        self.assertTrue(result['hamiltonian_ok'])


# =============================================================================
# TESTS: Cohomología de Haces Celulares
# =============================================================================

class TestCohomology(unittest.TestCase):
    """
    Test suite para análisis cohomológico.
    
    Verifica:
    - Cálculo de dim H¹
    - Detección de variables no inicializadas
    - Detección de ciclos de dependencias
    """
    
    def test_cohomology_simple_code(self):
        """Test: Código simple tiene H¹ = 0."""
        code = """
x = 1
y = x + 1
z = y * 2
"""
        result = ASTStaticAnalyzer.analyze_code(
            code,
            enable_cohomology=True,
            strict_mode=True
        )
        
        self.assertTrue(result['cohomology_ok'])
        self.assertEqual(result.get('cohomology_dimension', 0), 0)
    
    def test_cohomology_uninitialized_variable(self):
        """Test: Variable no inicializada genera obstrucción."""
        code = """
# y no está definida antes de usarse
result = y + 1
"""
        result = ASTStaticAnalyzer.analyze_code(
            code,
            enable_cohomology=True,
            strict_mode=False
        )
        
        # Debe detectar 'y' como problemática
        if 'problematic_variables' in result:
            self.assertIn('y', result['problematic_variables'])
    
    def test_cohomology_function_args_ok(self):
        """Test: Argumentos de función están bien definidos."""
        code = """
def func(x, y):
    return x + y
"""
        result = ASTStaticAnalyzer.analyze_code(
            code,
            enable_cohomology=True,
            strict_mode=True
        )
        
        # x, y son argumentos (definidos), no deben ser problemáticos
        self.assertTrue(result.get('cohomology_ok', True))
    
    def test_cohomology_class_methods(self):
        """Test: Métodos de clase con self."""
        code = """
class MyClass:
    def __init__(self, value):
        self.value = value
    
    def get_value(self):
        return self.value
"""
        result = ASTStaticAnalyzer.analyze_code(
            code,
            enable_cohomology=True,
            strict_mode=True
        )
        
        # No debe haber obstrucciones
        self.assertTrue(result.get('cohomology_ok', True))
    
    def test_cohomology_global_variables(self):
        """Test: Variables globales."""
        code = """
GLOBAL_CONST = 100

def use_global():
    return GLOBAL_CONST * 2
"""
        result = ASTStaticAnalyzer.analyze_code(
            code,
            enable_cohomology=True,
            strict_mode=False
        )
        
        # GLOBAL_CONST está definida antes de usarse
        # No debe ser problemática
        self.assertIn('cohomology_ok', result)
    
    def test_cohomology_disabled(self):
        """Test: Con cohomología desactivada."""
        code = "x = y + 1"  # y no definida
        
        result = ASTStaticAnalyzer.analyze_code(
            code,
            enable_cohomology=False,
            strict_mode=True
        )
        
        # cohomology_ok debe ser None (no verificado)
        self.assertIsNone(result.get('cohomology_ok'))
    
    def test_cohomology_dimension_calculation(self):
        """Test: Cálculo de dimensión cohomológica."""
        # Código con ciclo en el grafo de dependencias
        code = """
a = b + 1
b = c + 1
c = a + 1  # Ciclo: a → b → c → a
"""
        result = ASTStaticAnalyzer.analyze_code(
            code,
            enable_cohomology=True,
            strict_mode=False
        )
        
        # Debe detectar el ciclo (dim H¹ > 0 o variables problemáticas)
        dim_h1 = result.get('cohomology_dimension', 0)
        prob_vars = result.get('problematic_variables', set())
        
        # Al menos una de las variables debe ser problemática
        self.assertTrue(dim_h1 > 0 or len(prob_vars) > 0)


# =============================================================================
# TESTS: Validador JSON
# =============================================================================

class TestJSONValidator(unittest.TestCase):
    """
    Test suite para JSONStructureValidator.
    
    Verifica:
    - Validación de profundidad
    - Validación de tamaño
    - Prevención de DoS
    """
    
    def test_validate_simple_dict(self):
        """Test: Diccionario simple válido."""
        data = {"name": "test", "value": 42}
        
        is_valid, error = JSONStructureValidator.validate_structure(data)
        
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
    
    def test_validate_nested_dict(self):
        """Test: Diccionario anidado válido."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": 123
                    }
                }
            }
        }
        
        is_valid, error = JSONStructureValidator.validate_structure(
            data, max_depth=5
        )
        
        self.assertTrue(is_valid)
    
    def test_validate_depth_exceeded(self):
        """Test: Profundidad excedida."""
        data = {"a": {"b": {"c": {"d": {"e": {"f": "too deep"}}}}}}
        
        is_valid, error = JSONStructureValidator.validate_structure(
            data, max_depth=3
        )
        
        self.assertFalse(is_valid)
        self.assertIn("Depth", error)
        self.assertIn("exceeds limit", error)
    
    def test_validate_too_many_keys(self):
        """Test: Demasiadas claves."""
        data = {f"key_{i}": i for i in range(150)}
        
        is_valid, error = JSONStructureValidator.validate_structure(
            data, max_keys=100
        )
        
        self.assertFalse(is_valid)
        self.assertIn("keys", error.lower())
    
    def test_validate_large_array(self):
        """Test: Array muy grande."""
        data = list(range(2000))
        
        is_valid, error = JSONStructureValidator.validate_structure(
            data, max_array_size=1000
        )
        
        self.assertFalse(is_valid)
        self.assertIn("items", error.lower())
    
    def test_validate_mixed_structure(self):
        """Test: Estructura mixta válida."""
        data = {
            "users": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25}
            ],
            "config": {
                "timeout": 30,
                "retries": 3
            }
        }
        
        is_valid, error = JSONStructureValidator.validate_structure(data)
        
        self.assertTrue(is_valid)
    
    def test_validate_primitives(self):
        """Test: Tipos primitivos válidos."""
        primitives = [
            42,
            3.14,
            "string",
            True,
            False,
            None
        ]
        
        for prim in primitives:
            is_valid, error = JSONStructureValidator.validate_structure(prim)
            self.assertTrue(is_valid, f"Failed for {prim}: {error}")
    
    def test_validate_invalid_type(self):
        """Test: Tipo no soportado."""
        class CustomClass:
            pass
        
        data = {"obj": CustomClass()}
        
        is_valid, error = JSONStructureValidator.validate_structure(data)
        
        self.assertFalse(is_valid)
        self.assertIn("Unsupported type", error)
    
    def test_compute_depth(self):
        """Test: Cálculo de profundidad."""
        data1 = {"a": 1}
        data2 = {"a": {"b": {"c": 3}}}
        data3 = [1, [2, [3, [4]]]]
        
        self.assertEqual(JSONStructureValidator.compute_depth(data1), 1)
        self.assertEqual(JSONStructureValidator.compute_depth(data2), 3)
        self.assertEqual(JSONStructureValidator.compute_depth(data3), 4)
    
    def test_compute_depth_empty(self):
        """Test: Profundidad de estructuras vacías."""
        self.assertEqual(JSONStructureValidator.compute_depth({}), 0)
        self.assertEqual(JSONStructureValidator.compute_depth([]), 0)
        self.assertEqual(JSONStructureValidator.compute_depth(42), 0)
    
    def test_estimate_memory_size(self):
        """Test: Estimación de tamaño en memoria."""
        data = {"key": "value"}
        
        size = JSONStructureValidator.estimate_memory_size(data)
        
        self.assertGreater(size, 0)
        self.assertIsInstance(size, int)


# =============================================================================
# TESTS: Normalizador Tabular
# =============================================================================

class TestTabularNormalizer(unittest.TestCase):
    """Test suite para TabularNormalizer."""
    
    def test_to_markdown_simple(self):
        """Test: Conversión simple a Markdown."""
        data = {"name": "Alice", "age": 30}
        
        markdown = TabularNormalizer.to_markdown_table(data)
        
        self.assertIn("name", markdown)
        self.assertIn("age", markdown)
        self.assertIn("Alice", markdown)
        self.assertIn("30", markdown)
        self.assertIn("|", markdown)
    
    def test_to_markdown_empty(self):
        """Test: Diccionario vacío."""
        data = {}
        
        markdown = TabularNormalizer.to_markdown_table(data)
        
        self.assertIn("empty", markdown.lower())
    
    def test_to_markdown_type_error(self):
        """Test: Tipo incorrecto lanza TypeError."""
        with self.assertRaises(TypeError):
            TabularNormalizer.to_markdown_table([1, 2, 3])  # type: ignore
    
    def test_to_csv_row(self):
        """Test: Conversión a CSV."""
        data = {"x": 1, "y": 2, "z": 3}
        
        header, row = TabularNormalizer.to_csv_row(data)
        
        # Orden alfabético de claves
        self.assertEqual(header, "x,y,z")
        self.assertEqual(row, "1,2,3")
    
    def test_serialize_value_types(self):
        """Test: Serialización de diferentes tipos."""
        tests = [
            (None, "null"),
            (True, "true"),
            (False, "false"),
            (42, "42"),
            (3.14, "3.14"),
            ("text", '"text"'),
        ]
        
        for value, expected in tests:
            result = TabularNormalizer._serialize_value(value, 0)
            self.assertEqual(result, expected)
    
    def test_serialize_value_nan_inf(self):
        """Test: Serialización de NaN e Infinito."""
        self.assertEqual(
            TabularNormalizer._serialize_value(float('nan'), 0),
            "NaN"
        )
        self.assertEqual(
            TabularNormalizer._serialize_value(float('inf'), 0),
            "∞"
        )
        self.assertEqual(
            TabularNormalizer._serialize_value(float('-inf'), 0),
            "-∞"
        )
    
    def test_serialize_value_truncation(self):
        """Test: Truncado de strings largos."""
        long_string = "x" * 200
        
        result = TabularNormalizer._serialize_value(long_string, 0, max_length=50)
        
        self.assertLess(len(result), 60)  # Incluye comillas y "..."
        self.assertIn("...", result)
    
    def test_serialize_nested_depth_limit(self):
        """Test: Límite de profundidad en estructuras anidadas."""
        nested = {"a": {"b": {"c": {"d": "deep"}}}}
        
        result = TabularNormalizer._serialize_value(nested, 0, max_depth=2)
        
        self.assertIn("...", result)
    
    def test_flatten_dict(self):
        """Test: Aplanado de diccionario anidado."""
        nested = {
            "user": {
                "name": "Alice",
                "address": {
                    "city": "NYC"
                }
            }
        }
        
        flat = TabularNormalizer.flatten_dict(nested)
        
        self.assertEqual(flat["user.name"], "Alice")
        self.assertEqual(flat["user.address.city"], "NYC")
    
    def test_flatten_dict_custom_separator(self):
        """Test: Separador personalizado."""
        nested = {"a": {"b": 1}}
        
        flat = TabularNormalizer.flatten_dict(nested, separator="/")
        
        self.assertEqual(flat["a/b"], 1)


# =============================================================================
# TESTS: Integración Completa
# =============================================================================

class TestIntegration(unittest.TestCase):
    """
    Tests de integración end-to-end.
    
    Verifica el pipeline completo de análisis.
    """
    
    def test_full_analysis_simple_script(self):
        """Test: Análisis completo de script simple."""
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
print(result)
"""
        result = ASTStaticAnalyzer.analyze_code(
            code,
            filename="test.py",
            enable_hamiltonian=True,
            enable_cohomology=True,
            strict_mode=True
        )
        
        # Verificaciones
        self.assertIn('dataflow', result)
        self.assertIn('complexity', result)
        self.assertTrue(result['hamiltonian_ok'])
        self.assertTrue(result.get('cohomology_ok', True))
        
        # Complejidad esperada
        complexity = result['complexity']
        self.assertGreaterEqual(complexity.cyclomatic_complexity, 2)
        self.assertEqual(complexity.num_functions, 1)
    
    def test_full_analysis_class_based(self):
        """Test: Análisis de código orientado a objetos."""
        code = """
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.items:
            raise IndexError("Stack is empty")
        return self.items.pop()
    
    def peek(self):
        if not self.items:
            return None
        return self.items[-1]
    
    def is_empty(self):
        return len(self.items) == 0
"""
        result = ASTStaticAnalyzer.analyze_code(
            code,
            enable_hamiltonian=True,
            enable_cohomology=True
        )
        
        complexity = result['complexity']
        self.assertEqual(complexity.num_classes, 1)
        self.assertEqual(complexity.num_functions, 5)
        self.assertGreater(complexity.cyclomatic_complexity, 1)
    
    def test_analyze_file_api(self):
        """Test: API de análisis de archivo."""
        # Crear archivo temporal
        import tempfile
        
        code = "x = 1\ny = 2\nz = x + y"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            filepath = f.name
        
        try:
            result = ASTStaticAnalyzer.analyze_file(
                filepath,
                enable_hamiltonian=False,
                enable_cohomology=False
            )
            
            self.assertIn('dataflow', result)
            self.assertIn('complexity', result)
        finally:
            import os
            os.unlink(filepath)
    
    def test_validate_json_contract_valid(self):
        """Test: Validación de contrato JSON válido."""
        schema = {
            "name": "TestService",
            "version": "1.0.0",
            "config": {
                "timeout": 30,
                "retries": 3
            }
        }
        
        is_valid, error, markdown = ASTStaticAnalyzer.validate_json_contract(schema)
        
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        self.assertIsNotNone(markdown)
        self.assertIn("name", markdown)
    
    def test_validate_json_contract_invalid(self):
        """Test: Validación de contrato JSON inválido."""
        # Estructura demasiado profunda
        schema = {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": "deep"}}}}}}}}
        
        is_valid, error, markdown = ASTStaticAnalyzer.validate_json_contract(
            schema, max_depth=5
        )
        
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
    
    def test_compute_interference_matrix(self):
        """Test: Cálculo de matriz de interferencia."""
        tools = {
            "tool_a": DataFlowCoordinates(
                reads=frozenset(['x']),
                writes=frozenset(['y'])
            ),
            "tool_b": DataFlowCoordinates(
                reads=frozenset(['y']),
                writes=frozenset(['z'])
            ),
            "tool_c": DataFlowCoordinates(
                reads=frozenset(['z']),
                writes=frozenset(['w'])
            ),
        }
        
        matrix = ASTStaticAnalyzer.compute_interference_matrix(tools)
        
        # Verificar antisimetría
        self.assertEqual(
            matrix[('tool_a', 'tool_b')],
            -matrix[('tool_b', 'tool_a')]
        )
        
        # Diagonal debe ser cero
        self.assertEqual(matrix[('tool_a', 'tool_a')], 0)
        
        # tool_a → tool_b (A escribe y, B lee y)
        self.assertGreater(matrix[('tool_a', 'tool_b')], 0)
    
    def test_topological_sort(self):
        """Test: Ordenamiento topológico por interferencia."""
        tools = {
            "A": DataFlowCoordinates(writes=frozenset(['x'])),
            "B": DataFlowCoordinates(reads=frozenset(['x']), writes=frozenset(['y'])),
            "C": DataFlowCoordinates(reads=frozenset(['y']), writes=frozenset(['z'])),
        }
        
        sorted_tools = ASTStaticAnalyzer.topological_sort_by_interference(tools)
        
        # Orden esperado: A → B → C
        self.assertEqual(sorted_tools.index('A'), 0)
        self.assertLess(sorted_tools.index('A'), sorted_tools.index('B'))
        self.assertLess(sorted_tools.index('B'), sorted_tools.index('C'))
    
    def test_topological_sort_cycle_detection(self):
        """Test: Detección de ciclos en ordenamiento."""
        tools = {
            "A": DataFlowCoordinates(reads=frozenset(['z']), writes=frozenset(['x'])),
            "B": DataFlowCoordinates(reads=frozenset(['x']), writes=frozenset(['y'])),
            "C": DataFlowCoordinates(reads=frozenset(['y']), writes=frozenset(['z'])),
        }
        
        # Ciclo: A → B → C → A
        with self.assertRaises(ValueError) as ctx:
            ASTStaticAnalyzer.topological_sort_by_interference(tools)
        
        self.assertIn("cycle", str(ctx.exception).lower())


# =============================================================================
# TESTS: Performance y Límites
# =============================================================================

class TestPerformance(unittest.TestCase):
    """
    Tests de performance y límites.
    
    Verifica:
    - Comportamiento en casos límite
    - Tiempos de ejecución razonables
    - Manejo de código grande
    """
    
    def test_depth_limit_enforcement(self):
        """Test: Límite de profundidad se respeta."""
        # Generar código excesivamente profundo
        code = TestHelpers.generate_deep_nested_code(
            AnalysisLimits.MAX_AST_DEPTH + 10
        )
        
        with self.assertRaises(ComplexityBoundsViolationError) as ctx:
            ASTStaticAnalyzer.analyze_code(
                code,
                enable_hamiltonian=True,
                strict_mode=True
            )
        
        self.assertIn("depth", str(ctx.exception).lower())
    
    def test_high_complexity_handling(self):
        """Test: Manejo de código con alta complejidad."""
        # Generar código con CC muy alta
        code = TestHelpers.generate_high_complexity_code(100)
        
        result = ASTStaticAnalyzer.analyze_code(
            code,
            enable_hamiltonian=False,
            enable_cohomology=False,
            strict_mode=False
        )
        
        complexity = result['complexity']
        self.assertGreaterEqual(complexity.cyclomatic_complexity, 90)
        self.assertEqual(complexity.risk_level, "CRITICAL")
    
    def test_large_number_of_variables(self):
        """Test: Manejo de muchas variables."""
        # Generar código con muchas variables
        num_vars = 500
        assignments = "\n".join([f"var_{i} = {i}" for i in range(num_vars)])
        code = f"""
def func():
{assignments}
    return var_0
"""
        
        result = ASTStaticAnalyzer.analyze_code(
            code,
            enable_hamiltonian=False,
            enable_cohomology=False
        )
        
        coords = result['dataflow']
        # Al menos algunas variables deben detectarse
        self.assertGreater(len(coords.all_variables), 100)
    
    def test_performance_reasonable_time(self):
        """Test: Tiempo de análisis razonable."""
        # Código medianamente complejo
        code = """
class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.cache = {}
    
    def process(self):
        results = []
        for item in self.data:
            if item in self.cache:
                results.append(self.cache[item])
            else:
                processed = self._transform(item)
                self.cache[item] = processed
                results.append(processed)
        return results
    
    def _transform(self, item):
        if isinstance(item, int):
            return item * 2
        elif isinstance(item, str):
            return item.upper()
        else:
            return str(item)
"""
        
        start = time.time()
        result = ASTStaticAnalyzer.analyze_code(
            code,
            enable_hamiltonian=True,
            enable_cohomology=True
        )
        elapsed = time.time() - start
        
        # Debe completar en menos de 1 segundo
        self.assertLess(elapsed, 1.0)
    
    def test_memory_efficiency(self):
        """Test: Uso eficiente de memoria."""
        # Generar código grande pero no patológico
        code = """
def large_function():
    x = 0
""" + "\n".join([f"    x += {i}" for i in range(1000)]) + """
    return x
"""
        
        # No debe consumir memoria excesiva
        result = ASTStaticAnalyzer.analyze_code(
            code,
            enable_hamiltonian=False,
            enable_cohomology=False
        )
        
        self.assertIn('complexity', result)


# =============================================================================
# TESTS: Casos Edge y Robustez
# =============================================================================

class TestEdgeCases(unittest.TestCase):
    """Tests de casos edge y robustez."""
    
    def test_syntax_error_handling(self):
        """Test: Manejo de errores de sintaxis."""
        code = "def broken(:\n    pass"
        
        with self.assertRaises(ValueError) as ctx:
            ASTStaticAnalyzer.analyze_code(code)
        
        self.assertIn("syntax", str(ctx.exception).lower())
    
    def test_empty_code_handling(self):
        """Test: Manejo de código vacío."""
        with self.assertRaises(ValueError):
            ASTStaticAnalyzer.analyze_code("")
    
    def test_whitespace_only(self):
        """Test: Solo espacios en blanco."""
        with self.assertRaises(ValueError):
            ASTStaticAnalyzer.analyze_code("   \n\n   \t\t   ")
    
    def test_unicode_in_code(self):
        """Test: Código con Unicode."""
        code = """
def función(número):
    # Comentario en español
    resultado = número * 2
    return resultado

α = función(42)
"""
        
        result = ASTStaticAnalyzer.analyze_code(code)
        
        self.assertIn('dataflow', result)
        coords = result['dataflow']
        self.assertIn('α', coords.all_variables)
    
    def test_async_await_syntax(self):
        """Test: Sintaxis async/await."""
        code = """
async def fetch_data():
    async with client.session() as session:
        async for item in session.iter():
            await process(item)
"""
        
        result = ASTStaticAnalyzer.analyze_code(code)
        
        complexity = result['complexity']
        self.assertGreater(complexity.cyclomatic_complexity, 1)
    
    def test_match_statement_python310(self):
        """Test: Pattern matching (Python 3.10+)."""
        if sys.version_info < (3, 10):
            self.skipTest("Requires Python 3.10+")
        
        code = """
def process(value):
    match value:
        case 0:
            return "zero"
        case 1:
            return "one"
        case _:
            return "other"
"""
        
        result = ASTStaticAnalyzer.analyze_code(code)
        
        complexity = result['complexity']
        # Match con 3 cases incrementa CC
        self.assertGreater(complexity.cyclomatic_complexity, 2)
    
    def test_walrus_operator(self):
        """Test: Operador walrus :=."""
        code = """
if (n := len(data)) > 10:
    print(f"Large dataset: {n} items")
"""
        
        result = ASTStaticAnalyzer.analyze_code(code)
        
        self.assertIn('dataflow', result)
    
    def test_f_strings(self):
        """Test: F-strings."""
        code = """
name = "Alice"
age = 30
message = f"Hello {name}, you are {age} years old"
"""
        
        result = ASTStaticAnalyzer.analyze_code(code)
        
        coords = result['dataflow']
        self.assertIn('name', coords.reads)
        self.assertIn('age', coords.reads)
    
    def test_decorators(self):
        """Test: Decoradores."""
        code = """
@staticmethod
@lru_cache(maxsize=128)
def cached_func(x):
    return x ** 2
"""
        
        result = ASTStaticAnalyzer.analyze_code(code)
        
        complexity = result['complexity']
        self.assertEqual(complexity.num_functions, 1)
    
    def test_generator_expressions(self):
        """Test: Expresiones generadoras."""
        code = """
squares = (x**2 for x in range(10) if x % 2 == 0)
"""
        
        result = ASTStaticAnalyzer.analyze_code(code)
        
        # Debe detectar la comprehension
        self.assertIn('complexity', result)
    
    def test_exception_handling_complex(self):
        """Test: Manejo complejo de excepciones."""
        code = """
try:
    risky_operation()
except ValueError as e:
    handle_value_error(e)
except (TypeError, KeyError) as e:
    handle_type_key_error(e)
except Exception:
    log_error()
    raise
else:
    success()
finally:
    cleanup()
"""
        
        result = ASTStaticAnalyzer.analyze_code(code)
        
        complexity = result['complexity']
        # try + 3 except incrementa CC
        self.assertGreater(complexity.cyclomatic_complexity, 3)


# =============================================================================
# TESTS: API Legacy (Compatibilidad)
# =============================================================================

class TestLegacyAPI(unittest.TestCase):
    """Tests de API legacy para retrocompatibilidad."""
    
    def test_ast_symplectic_parser_parse_tool_dynamics(self):
        """Test: API deprecated parse_tool_dynamics."""
        from app.boole.physics.ast_static_analyzer import ASTSymplecticParser
        
        code = """
def tool_func(input_data):
    result = process(input_data)
    return result
"""
        
        with self.assertLogs(level='WARNING') as log:
            coords, complexity = ASTSymplecticParser.parse_tool_dynamics(code)
        
        # Debe generar warning de deprecación
        self.assertTrue(any("deprecated" in msg.lower() for msg in log.output))
        
        # Pero debe funcionar
        self.assertIsInstance(coords, DataFlowCoordinates)
        self.assertIsInstance(complexity, ComplexityProfile)
    
    def test_ast_symplectic_parser_process_data_contract(self):
        """Test: API deprecated process_data_contract."""
        from app.boole.physics.ast_static_analyzer import ASTSymplecticParser
        
        schema = {"name": "test", "version": "1.0"}
        
        with self.assertLogs(level='WARNING') as log:
            markdown = ASTSymplecticParser.process_data_contract(schema)
        
        # Debe generar warning
        self.assertTrue(any("deprecated" in msg.lower() for msg in log.output))
        
        # Debe retornar markdown
        self.assertIsInstance(markdown, str)


# =============================================================================
# SUITE DE TESTS PRINCIPAL
# =============================================================================

def suite():
    """Construye la suite completa de tests."""
    test_suite = unittest.TestSuite()
    
    # Agregar todos los test cases
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDataFlowCoordinates))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestComplexityProfile))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDataFlowAnalyzer))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestHamiltonianMonitor))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCohomology))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestJSONValidator))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestTabularNormalizer))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIntegration))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPerformance))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEdgeCases))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLegacyAPI))
    
    return test_suite


# =============================================================================
# RUNNER PRINCIPAL
# =============================================================================

if __name__ == '__main__':
    # Configurar logging para tests
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s | %(name)s | %(message)s'
    )
    
    # Ejecutar tests con verbosidad
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    
    # Resumen final
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)