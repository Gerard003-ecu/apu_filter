"""
=========================================================================================
Módulo: Suite de Pruebas para AST Static Analyzer
Ubicación: tests/boole/physics/test_ast_static_analyzer.py
Versión: 1.0 - Suite Rigurosa de Testing
=========================================================================================

COBERTURA DE PRUEBAS:
---------------------
1. Estructuras de datos (DataFlowCoordinates, ComplexityProfile, etc.)
2. Análisis de flujo de datos (reads/writes)
3. Complejidad ciclomática (McCabe)
4. Validación de JSON/Schema
5. Normalización tabular
6. Análisis completo end-to-end
7. Casos extremos y edge cases
8. Performance y escalabilidad

EJECUCIÓN:
----------
    python -m pytest test_ast_symplectic_parser.py -v --cov=ast_symplectic_parser
    python -m unittest test_ast_symplectic_parser.TestDataFlowCoordinates -v

=========================================================================================
"""

import ast
import logging
import sys
import unittest
from contextlib import contextmanager

# Importar módulo a testear
from app.boole.physics.ast_static_analyzer import (
    AnalysisLimits,
    ASTStaticAnalyzer,
    ComplexityProfile,
    DataFlowAnalyzer,
    DataFlowCoordinates,
    JSONStructureValidator,
    TabularNormalizer,
)
from app.boole.wisdom.semantic_validator import (
    BusinessPurpose,
    ConfidenceFilter,
    ConstraintMapper,
    LLMOutput,
    PurposeValidator,
    RiskProfile,
    SemanticValidationEngine,
    Verdict,
    create_default_knowledge_graph,
)

# Suprimir logs durante testing
logging.getLogger("Gamma.Physics.ASTAnalyzer.v2.0").setLevel(logging.CRITICAL)
logging.getLogger("Gamma.Wisdom.SemanticValidator.v3.0").setLevel(logging.CRITICAL)


# ========================================================================================
# UTILIDADES DE TESTING
# ========================================================================================


class TestBase(unittest.TestCase):
    """Clase base con utilidades comunes."""

    @contextmanager
    def assertNotRaises(self, exc_type):
        """Verifica que NO se lance una excepción."""
        try:
            yield
        except exc_type as e:
            self.fail(f"Se lanzó {exc_type.__name__} inesperadamente: {e}")

    def assertFrozenSetEqual(self, set1, set2, msg=None):
        """Compara frozensets."""
        self.assertEqual(set1, set2, msg)


# ========================================================================================
# TESTS: DataFlowCoordinates
# ========================================================================================


class TestDataFlowCoordinates(TestBase):
    """Tests para DataFlowCoordinates."""

    def test_construction_basic(self):
        """Construcción básica."""
        coords = DataFlowCoordinates(
            reads=frozenset(["x", "y"]), writes=frozenset(["z"])
        )

        self.assertEqual(coords.reads, frozenset(["x", "y"]))
        self.assertEqual(coords.writes, frozenset(["z"]))

    def test_construction_empty(self):
        """Construcción vacía."""
        coords = DataFlowCoordinates()

        self.assertEqual(coords.reads, frozenset())
        self.assertEqual(coords.writes, frozenset())

    def test_construction_normalization(self):
        """Normalización a frozenset."""
        coords = DataFlowCoordinates(
            reads={"x", "y"}, writes=["z"]  # set regular  # lista
        )

        self.assertIsInstance(coords.reads, frozenset)
        self.assertIsInstance(coords.writes, frozenset)

    def test_immutability(self):
        """Las coordenadas son inmutables."""
        coords = DataFlowCoordinates(reads=frozenset(["x"]))

        with self.assertRaises(AttributeError):
            coords.reads = frozenset(["y"])

    def test_hashability(self):
        """Las coordenadas son hasheables."""
        coords1 = DataFlowCoordinates(reads=frozenset(["x"]), writes=frozenset(["y"]))
        coords2 = DataFlowCoordinates(reads=frozenset(["x"]), writes=frozenset(["y"]))

        # Mismo contenido → mismo hash
        self.assertEqual(hash(coords1), hash(coords2))

        # Se pueden usar en sets
        s = {coords1, coords2}
        self.assertEqual(len(s), 1)

    def test_all_variables(self):
        """Propiedad all_variables."""
        coords = DataFlowCoordinates(
            reads=frozenset(["x", "y"]), writes=frozenset(["y", "z"])
        )

        self.assertEqual(coords.all_variables, frozenset(["x", "y", "z"]))

    def test_has_interference_with_raw(self):
        """Interferencia RAW (Read After Write)."""
        a = DataFlowCoordinates(writes=frozenset(["x"]))
        b = DataFlowCoordinates(reads=frozenset(["x"]))

        self.assertTrue(a.has_interference_with(b))
        self.assertTrue(b.has_interference_with(a))

    def test_has_interference_with_war(self):
        """Interferencia WAR (Write After Read)."""
        a = DataFlowCoordinates(reads=frozenset(["x"]))
        b = DataFlowCoordinates(writes=frozenset(["x"]))

        self.assertTrue(a.has_interference_with(b))
        self.assertTrue(b.has_interference_with(a))

    def test_has_interference_with_waw(self):
        """Interferencia WAW (Write After Write)."""
        a = DataFlowCoordinates(writes=frozenset(["x"]))
        b = DataFlowCoordinates(writes=frozenset(["x"]))

        self.assertTrue(a.has_interference_with(b))

    def test_no_interference(self):
        """Sin interferencia."""
        a = DataFlowCoordinates(reads=frozenset(["x"]), writes=frozenset(["y"]))
        b = DataFlowCoordinates(reads=frozenset(["z"]), writes=frozenset(["w"]))

        self.assertFalse(a.has_interference_with(b))
        self.assertFalse(b.has_interference_with(a))

    def test_interference_score_asymmetric(self):
        """Score de interferencia es asimétrico."""
        a = DataFlowCoordinates(writes=frozenset(["x"]))
        b = DataFlowCoordinates(reads=frozenset(["x"]))

        score_ab = DataFlowCoordinates.interference_score(a, b)
        score_ba = DataFlowCoordinates.interference_score(b, a)

        # Antisimetría
        self.assertEqual(score_ab, -score_ba)
        self.assertEqual(score_ab, 1)  # A escribe lo que B lee

    def test_interference_score_symmetric(self):
        """Score simétrico cuando interferencia es igual."""
        a = DataFlowCoordinates(reads=frozenset(["x"]), writes=frozenset(["y"]))
        b = DataFlowCoordinates(reads=frozenset(["y"]), writes=frozenset(["x"]))

        score_ab = DataFlowCoordinates.interference_score(a, b)
        score_ba = DataFlowCoordinates.interference_score(b, a)

        # Simétrico (cada uno escribe lo que el otro lee)
        self.assertEqual(score_ab, 0)
        self.assertEqual(score_ba, 0)

    def test_interference_score_zero(self):
        """Score cero sin interferencia."""
        a = DataFlowCoordinates(reads=frozenset(["x"]))
        b = DataFlowCoordinates(reads=frozenset(["y"]))

        score = DataFlowCoordinates.interference_score(a, b)
        self.assertEqual(score, 0)


# ========================================================================================
# TESTS: ComplexityProfile
# ========================================================================================


class TestComplexityProfile(TestBase):
    """Tests para ComplexityProfile."""

    def test_construction(self):
        """Construcción básica."""
        profile = ComplexityProfile(
            cyclomatic_complexity=10,
            max_nesting_depth=3,
            num_functions=2,
            num_classes=1,
        )

        self.assertEqual(profile.cyclomatic_complexity, 10)
        self.assertEqual(profile.max_nesting_depth, 3)

    def test_is_maintainable_true(self):
        """Código mantenible."""
        profile = ComplexityProfile(
            cyclomatic_complexity=15,
            max_nesting_depth=4,
            num_functions=3,
            num_classes=1,
        )

        self.assertTrue(profile.is_maintainable)

    def test_is_maintainable_false_complexity(self):
        """No mantenible por complejidad."""
        profile = ComplexityProfile(
            cyclomatic_complexity=25,
            max_nesting_depth=3,
            num_functions=1,
            num_classes=0,
        )

        self.assertFalse(profile.is_maintainable)

    def test_is_maintainable_false_depth(self):
        """No mantenible por profundidad."""
        profile = ComplexityProfile(
            cyclomatic_complexity=10,
            max_nesting_depth=7,
            num_functions=1,
            num_classes=0,
        )

        self.assertFalse(profile.is_maintainable)

    def test_risk_level_low(self):
        """Riesgo bajo."""
        profile = ComplexityProfile(
            cyclomatic_complexity=5, max_nesting_depth=2, num_functions=1, num_classes=0
        )

        self.assertEqual(profile.risk_level, "LOW")

    def test_risk_level_moderate(self):
        """Riesgo moderado."""
        profile = ComplexityProfile(
            cyclomatic_complexity=15,
            max_nesting_depth=3,
            num_functions=1,
            num_classes=0,
        )

        self.assertEqual(profile.risk_level, "MODERATE")

    def test_risk_level_high(self):
        """Riesgo alto."""
        profile = ComplexityProfile(
            cyclomatic_complexity=30,
            max_nesting_depth=4,
            num_functions=1,
            num_classes=0,
        )

        self.assertEqual(profile.risk_level, "HIGH")

    def test_risk_level_critical(self):
        """Riesgo crítico."""
        profile = ComplexityProfile(
            cyclomatic_complexity=60,
            max_nesting_depth=8,
            num_functions=1,
            num_classes=0,
        )

        self.assertEqual(profile.risk_level, "CRITICAL")

    def test_complexity_density(self):
        """Densidad de complejidad."""
        profile = ComplexityProfile(
            cyclomatic_complexity=20,
            max_nesting_depth=3,
            num_functions=4,
            num_classes=0,
        )

        self.assertEqual(profile.complexity_density, 5.0)  # 20 / 4

    def test_complexity_density_no_functions(self):
        """Densidad sin funciones (usa 1 como base)."""
        profile = ComplexityProfile(
            cyclomatic_complexity=10,
            max_nesting_depth=2,
            num_functions=0,
            num_classes=0,
        )

        self.assertEqual(profile.complexity_density, 10.0)  # 10 / 1


# ========================================================================================
# TESTS: DataFlowAnalyzer
# ========================================================================================


class TestDataFlowAnalyzer(TestBase):
    """Tests para DataFlowAnalyzer."""

    def test_simple_assignment(self):
        """Asignación simple."""
        code = "x = 5"
        tree = ast.parse(code)

        analyzer = DataFlowAnalyzer()
        analyzer.visit(tree)

        coords = analyzer.get_dataflow_coordinates()

        self.assertIn("x", coords.writes)
        self.assertEqual(len(coords.reads), 0)

    def test_simple_read(self):
        """Lectura simple."""
        code = "y = x + 1"
        tree = ast.parse(code)

        analyzer = DataFlowAnalyzer()
        analyzer.visit(tree)

        coords = analyzer.get_dataflow_coordinates()

        self.assertIn("x", coords.reads)
        self.assertIn("y", coords.writes)

    def test_multiple_reads_writes(self):
        """Múltiples lecturas y escrituras."""
        code = """
x = 1
y = 2
z = x + y
"""
        tree = ast.parse(code)

        analyzer = DataFlowAnalyzer()
        analyzer.visit(tree)

        coords = analyzer.get_dataflow_coordinates()

        self.assertFrozenSetEqual(coords.reads, frozenset(["x", "y"]))
        self.assertFrozenSetEqual(coords.writes, frozenset(["x", "y", "z"]))

    def test_function_arguments(self):
        """Argumentos de función son writes."""
        code = """
def foo(a, b, c=10):
    return a + b + c
"""
        tree = ast.parse(code)

        analyzer = DataFlowAnalyzer()
        analyzer.visit(tree)

        coords = analyzer.get_dataflow_coordinates()

        self.assertIn("a", coords.writes)
        self.assertIn("b", coords.writes)
        self.assertIn("c", coords.writes)

    def test_function_varargs(self):
        """*args y **kwargs."""
        code = """
def foo(*args, **kwargs):
    pass
"""
        tree = ast.parse(code)

        analyzer = DataFlowAnalyzer()
        analyzer.visit(tree)

        coords = analyzer.get_dataflow_coordinates()

        self.assertIn("args", coords.writes)
        self.assertIn("kwargs", coords.writes)

    def test_attribute_access(self):
        """Acceso a atributos."""
        code = """
obj.value = 10
x = obj.value
"""
        tree = ast.parse(code)

        analyzer = DataFlowAnalyzer()
        analyzer.visit(tree)

        coords = analyzer.get_dataflow_coordinates()

        self.assertIn("value", coords.reads)
        self.assertIn("value", coords.writes)
        self.assertIn("x", coords.writes)

    def test_cyclomatic_if(self):
        """If incrementa complejidad."""
        code = """
if x > 0:
    y = 1
else:
    y = 0
"""
        tree = ast.parse(code)

        analyzer = DataFlowAnalyzer()
        analyzer.visit(tree)

        profile = analyzer.get_complexity_profile()

        self.assertEqual(profile.cyclomatic_complexity, 2)  # 1 base + 1 if

    def test_cyclomatic_for(self):
        """For incrementa complejidad."""
        code = """
for i in range(10):
    print(i)
"""
        tree = ast.parse(code)

        analyzer = DataFlowAnalyzer()
        analyzer.visit(tree)

        profile = analyzer.get_complexity_profile()

        self.assertEqual(profile.cyclomatic_complexity, 2)  # 1 base + 1 for

    def test_cyclomatic_while(self):
        """While incrementa complejidad."""
        code = """
while x < 10:
    x += 1
"""
        tree = ast.parse(code)

        analyzer = DataFlowAnalyzer()
        analyzer.visit(tree)

        profile = analyzer.get_complexity_profile()

        self.assertEqual(profile.cyclomatic_complexity, 2)  # 1 base + 1 while

    def test_cyclomatic_try_except(self):
        """Try/except incrementa complejidad."""
        code = """
try:
    x = risky_function()
except ValueError:
    x = 0
except KeyError:
    x = 1
"""
        tree = ast.parse(code)

        analyzer = DataFlowAnalyzer()
        analyzer.visit(tree)

        profile = analyzer.get_complexity_profile()

        # 1 base + 1 try + 2 except
        self.assertEqual(profile.cyclomatic_complexity, 4)

    def test_cyclomatic_boolean_or(self):
        """Boolean OR incrementa complejidad."""
        code = """
if a or b or c:
    pass
"""
        tree = ast.parse(code)

        analyzer = DataFlowAnalyzer()
        analyzer.visit(tree)

        profile = analyzer.get_complexity_profile()

        # 1 base + 1 if + 2 (or con 3 operandos = 2 decisiones)
        self.assertEqual(profile.cyclomatic_complexity, 4)

    def test_cyclomatic_boolean_and(self):
        """Boolean AND incrementa complejidad."""
        code = """
if a and b and c and d:
    pass
"""
        tree = ast.parse(code)

        analyzer = DataFlowAnalyzer()
        analyzer.visit(tree)

        profile = analyzer.get_complexity_profile()

        # 1 base + 1 if + 3 (and con 4 operandos = 3 decisiones)
        self.assertEqual(profile.cyclomatic_complexity, 5)

    def test_cyclomatic_ternary(self):
        """Expresión ternaria incrementa complejidad."""
        code = "x = a if condition else b"
        tree = ast.parse(code)

        analyzer = DataFlowAnalyzer()
        analyzer.visit(tree)

        profile = analyzer.get_complexity_profile()

        self.assertEqual(profile.cyclomatic_complexity, 2)  # 1 base + 1 ternary

    def test_cyclomatic_lambda(self):
        """Lambda incrementa complejidad y función."""
        code = "f = lambda x: x + 1"
        tree = ast.parse(code)

        analyzer = DataFlowAnalyzer()
        analyzer.visit(tree)

        profile = analyzer.get_complexity_profile()

        self.assertEqual(
            profile.cyclomatic_complexity, 1
        )  # 1 base (lambda no añade ramas extra a Betti_1)
        self.assertEqual(profile.num_functions, 1)

    def test_cyclomatic_comprehension(self):
        """Comprensión incrementa complejidad."""
        code = "[x for x in range(10) if x % 2 == 0]"
        tree = ast.parse(code)

        analyzer = DataFlowAnalyzer()
        analyzer.visit(tree)

        profile = analyzer.get_complexity_profile()

        # 1 base + 1 for + 1 if
        self.assertEqual(profile.cyclomatic_complexity, 3)

    def test_max_depth_simple(self):
        """Profundidad simple."""
        code = "x = 1"
        tree = ast.parse(code)

        analyzer = DataFlowAnalyzer()
        analyzer.visit(tree)

        profile = analyzer.get_complexity_profile()

        self.assertGreater(profile.max_nesting_depth, 0)

    def test_max_depth_nested(self):
        """Profundidad con anidamiento."""
        code = """
def outer():
    def inner():
        if True:
            for i in range(10):
                pass
"""
        tree = ast.parse(code)

        analyzer = DataFlowAnalyzer()
        analyzer.visit(tree)

        profile = analyzer.get_complexity_profile()

        # Debe tener profundidad considerable
        self.assertGreater(profile.max_nesting_depth, 5)

    def test_depth_limit_exceeded(self):
        """Límite de profundidad excedido."""
        # Crear código con anidamiento profundo
        depth = AnalysisLimits.MAX_AST_DEPTH + 5
        code = "x = " + "(" * depth + "1" + ")" * depth

        tree = ast.parse(code)
        analyzer = DataFlowAnalyzer()

        with self.assertRaises(RecursionError):
            # Nested parentheses in python AST are parsed away and won't increase depth. Let's use nested conditionals
            depth = AnalysisLimits.MAX_AST_DEPTH + 5
            code = (
                "if True:\n"
                + "".join([f"{'    ' * i}if True:\n" for i in range(1, depth)])
                + f"{'    ' * depth}pass"
            )
            tree = ast.parse(code)
            analyzer.visit(tree)

    def test_num_functions(self):
        """Conteo de funciones."""
        code = """
def foo():
    pass

def bar():
    pass

async def baz():
    pass

f = lambda x: x
"""
        tree = ast.parse(code)

        analyzer = DataFlowAnalyzer()
        analyzer.visit(tree)

        profile = analyzer.get_complexity_profile()

        self.assertEqual(profile.num_functions, 4)  # foo, bar, baz, lambda

    def test_num_classes(self):
        """Conteo de clases."""
        code = """
class A:
    pass

class B:
    class C:
        pass
"""
        tree = ast.parse(code)

        analyzer = DataFlowAnalyzer()
        analyzer.visit(tree)

        profile = analyzer.get_complexity_profile()

        self.assertEqual(profile.num_classes, 3)  # A, B, C


# ========================================================================================
# TESTS: JSONStructureValidator
# ========================================================================================


class TestJSONStructureValidator(TestBase):
    """Tests para JSONStructureValidator."""

    def test_validate_simple_dict(self):
        """Validación de diccionario simple."""
        data = {"key": "value"}

        is_valid, error = JSONStructureValidator.validate_structure(data)

        self.assertTrue(is_valid)
        self.assertEqual(error, "")

    def test_validate_nested_dict(self):
        """Validación de diccionario anidado."""
        data = {"level1": {"level2": {"level3": "value"}}}

        is_valid, error = JSONStructureValidator.validate_structure(data, max_depth=5)

        self.assertTrue(is_valid)

    def test_validate_depth_exceeded(self):
        """Profundidad excedida."""
        data = {"l1": {"l2": {"l3": {"l4": {"l5": {"l6": "too deep"}}}}}}

        is_valid, error = JSONStructureValidator.validate_structure(data, max_depth=4)

        self.assertFalse(is_valid)
        self.assertIn("Depth", error)

    def test_validate_too_many_keys(self):
        """Demasiadas claves."""
        data = {f"key{i}": i for i in range(150)}

        is_valid, error = JSONStructureValidator.validate_structure(data, max_keys=100)

        self.assertFalse(is_valid)
        self.assertIn("keys", error)

    def test_validate_list(self):
        """Validación de lista."""
        data = [1, 2, 3, {"nested": "value"}]

        is_valid, error = JSONStructureValidator.validate_structure(data)

        self.assertTrue(is_valid)

    def test_validate_list_too_long(self):
        """Lista demasiado larga."""
        data = list(range(2000))

        is_valid, error = JSONStructureValidator.validate_structure(data, max_keys=100)

        self.assertFalse(is_valid)
        self.assertIn("items", error)

    def test_validate_mixed_structure(self):
        """Estructura mixta válida."""
        data = {
            "string": "value",
            "number": 42,
            "list": [1, 2, 3],
            "nested": {"inner": "data"},
        }

        is_valid, error = JSONStructureValidator.validate_structure(data)

        self.assertTrue(is_valid)

    def test_compute_depth_simple(self):
        """Profundidad simple."""
        data = {"key": "value"}

        depth = JSONStructureValidator.compute_depth(data)

        self.assertEqual(depth, 1)

    def test_compute_depth_nested(self):
        """Profundidad anidada."""
        data = {"l1": {"l2": {"l3": "value"}}}

        depth = JSONStructureValidator.compute_depth(data)

        self.assertEqual(depth, 3)

    def test_compute_depth_empty(self):
        """Profundidad de estructura vacía."""
        self.assertEqual(JSONStructureValidator.compute_depth({}), 0)
        self.assertEqual(JSONStructureValidator.compute_depth([]), 0)

    def test_compute_depth_primitive(self):
        """Profundidad de primitivo."""
        self.assertEqual(JSONStructureValidator.compute_depth(42), 0)
        self.assertEqual(JSONStructureValidator.compute_depth("string"), 0)


# ========================================================================================
# TESTS: TabularNormalizer
# ========================================================================================


class TestTabularNormalizer(TestBase):
    """Tests para TabularNormalizer."""

    def test_serialize_primitive(self):
        """Serialización de primitivos."""
        self.assertEqual(TabularNormalizer._serialize_value(None, 0), "null")
        self.assertEqual(TabularNormalizer._serialize_value(True, 0), "true")
        self.assertEqual(TabularNormalizer._serialize_value(False, 0), "false")
        self.assertEqual(TabularNormalizer._serialize_value(42, 0), "42")
        self.assertEqual(TabularNormalizer._serialize_value(3.14, 0), "3.14")

    def test_serialize_string(self):
        """Serialización de strings."""
        self.assertEqual(TabularNormalizer._serialize_value("hello", 0), '"hello"')

        # String largo se trunca
        long_str = "a" * 100
        result = TabularNormalizer._serialize_value(long_str, 0)
        self.assertTrue(result.endswith('..."'))

    def test_serialize_nan_inf(self):
        """Serialización de NaN e infinito."""
        self.assertEqual(TabularNormalizer._serialize_value(float("nan"), 0), "NaN")
        self.assertEqual(TabularNormalizer._serialize_value(float("inf"), 0), "∞")
        self.assertEqual(TabularNormalizer._serialize_value(float("-inf"), 0), "-∞")

    def test_serialize_dict_shallow(self):
        """Serialización de diccionario poco profundo."""
        data = {"a": 1, "b": 2}
        result = TabularNormalizer._serialize_value(data, 0)

        self.assertIn("a: 1", result)
        self.assertIn("b: 2", result)

    def test_serialize_dict_deep(self):
        """Serialización de diccionario profundo (truncado)."""
        data = {"a": {"b": {"c": "deep"}}}
        result = TabularNormalizer._serialize_value(data, 0, max_depth=1)

        # Debe truncarse
        self.assertIn("keys", result)

    def test_serialize_list_shallow(self):
        """Serialización de lista."""
        data = [1, 2, 3]
        result = TabularNormalizer._serialize_value(data, 0)

        self.assertEqual(result, "[1, 2, 3]")

    def test_serialize_list_long(self):
        """Lista larga se trunca."""
        data = list(range(20))
        result = TabularNormalizer._serialize_value(data, 0)

        # Solo primeros 5 + ...
        self.assertIn("...", result)

    def test_to_markdown_table_simple(self):
        """Tabla Markdown simple."""
        data = {"name": "Alice", "age": 30}
        result = TabularNormalizer.to_markdown_table(data)

        # Verificar estructura
        lines = result.split("\n")
        self.assertEqual(len(lines), 3)  # header, separator, row

        # Verificar contenido
        self.assertIn("age", lines[0])
        self.assertIn("name", lines[0])
        self.assertIn("30", lines[2])
        self.assertIn("Alice", lines[2])

    def test_to_markdown_table_empty(self):
        """Tabla vacía."""
        result = TabularNormalizer.to_markdown_table({})

        self.assertIn("empty", result)

    def test_to_markdown_table_invalid_type(self):
        """Tipo inválido."""
        with self.assertRaises(TypeError):
            TabularNormalizer.to_markdown_table([1, 2, 3])

    def test_to_csv_row(self):
        """Conversión a CSV."""
        data = {"name": "Bob", "score": 95}
        header, row = TabularNormalizer.to_csv_row(data)

        self.assertIn("name", header)
        self.assertIn("score", header)
        self.assertIn("Bob", row)
        self.assertIn("95", row)


# ========================================================================================
# TESTS: ASTStaticAnalyzer
# ========================================================================================


class TestASTStaticAnalyzer(TestBase):
    """Tests para ASTStaticAnalyzer."""

    def test_analyze_simple_code(self):
        """Análisis de código simple."""
        code = """
def add(a, b):
    return a + b
"""

        dataflow, complexity = ASTStaticAnalyzer.analyze_code(code)

        # Verificar que se detectaron reads/writes
        self.assertIn("a", dataflow.reads)
        self.assertIn("b", dataflow.reads)

        # Complejidad ciclomática base
        self.assertGreaterEqual(complexity.cyclomatic_complexity, 1)

        # Una función definida
        self.assertEqual(complexity.num_functions, 1)

    def test_analyze_complex_code(self):
        """Análisis de código complejo."""
        code = """
def process_data(items):
    results = []
    for item in items:
        if item > 0:
            results.append(item * 2)
        else:
            results.append(0)
    return results
"""

        dataflow, complexity = ASTStaticAnalyzer.analyze_code(code)

        # Complejidad: 1 base + 1 for + 1 if = 3
        self.assertEqual(complexity.cyclomatic_complexity, 3)

    def test_analyze_empty_code(self):
        """Código vacío."""
        with self.assertRaises(ValueError):
            ASTStaticAnalyzer.analyze_code("")

    def test_analyze_invalid_syntax(self):
        """Sintaxis inválida."""
        code = "def foo( invalid syntax"

        with self.assertRaises(ValueError):
            ASTStaticAnalyzer.analyze_code(code)

    def test_analyze_deep_nesting(self):
        """Anidamiento profundo."""
        # Crear código muy anidado
        code = "x = 1\n"
        for i in range(60):
            code += f"{'  ' * i}if x > {i}:\n"
            code += f"{'  ' * (i+1)}x += 1\n"

        with self.assertRaises(RecursionError):
            ASTStaticAnalyzer.analyze_code(code)

    def test_validate_json_contract_valid(self):
        """Validación de JSON válido."""
        schema = {"name": "TestTool", "version": "1.0", "config": {"timeout": 30}}

        is_valid, error, markdown = ASTStaticAnalyzer.validate_json_contract(schema)

        self.assertTrue(is_valid)
        self.assertIsNone(error)
        self.assertIsNotNone(markdown)

    def test_validate_json_contract_too_deep(self):
        """JSON demasiado profundo."""
        # Crear estructura muy anidada
        schema = {}
        current = schema
        for i in range(20):
            current["nested"] = {}
            current = current["nested"]

        is_valid, error, markdown = ASTStaticAnalyzer.validate_json_contract(
            schema, max_depth=10
        )

        self.assertFalse(is_valid)
        self.assertIsNotNone(error)

    def test_validate_json_contract_list(self):
        """JSON como lista."""
        schema = [1, 2, 3]

        is_valid, error, markdown = ASTStaticAnalyzer.validate_json_contract(schema)

        self.assertTrue(is_valid)
        self.assertEqual(markdown, "(Not a dict, cannot normalize to table)")

    def test_compute_interference_matrix(self):
        """Matriz de interferencia."""
        tools = {
            "tool_a": DataFlowCoordinates(
                reads=frozenset(["x"]), writes=frozenset(["y"])
            ),
            "tool_b": DataFlowCoordinates(
                reads=frozenset(["y"]), writes=frozenset(["z"])
            ),
        }

        matrix = ASTStaticAnalyzer.compute_interference_matrix(tools)

        # tool_a escribe y, tool_b lee y → interferencia
        self.assertEqual(matrix[("tool_a", "tool_b")], 1)
        self.assertEqual(matrix[("tool_b", "tool_a")], -1)


# ========================================================================================
# TESTS: Semantic Validation Components
# ========================================================================================


class TestBusinessPurpose(TestBase):
    """Tests para BusinessPurpose."""

    def test_construction_valid(self):
        """Construcción válida."""
        purpose = BusinessPurpose(
            concept="caching",
            business_problem="LATENCY_REDUCTION",
            strength=0.9,
            confidence=0.95,
        )

        self.assertEqual(purpose.concept, "caching")
        self.assertEqual(purpose.effective_strength, 0.9 * 0.95)

    def test_construction_invalid_strength(self):
        """Strength inválido."""
        with self.assertRaises(ValueError):
            BusinessPurpose("concept", "problem", strength=1.5)

    def test_construction_invalid_confidence(self):
        """Confidence inválido."""
        with self.assertRaises(ValueError):
            BusinessPurpose("concept", "problem", strength=0.5, confidence=-0.1)


class TestRiskProfile(TestBase):
    """Tests para RiskProfile."""

    def test_construction_valid(self):
        """Construcción válida."""
        profile = RiskProfile(
            risk_tolerance=0.7, domain_criticality=0.3, acceptable_failure_rate=0.05
        )

        self.assertEqual(profile.risk_tolerance, 0.7)

    def test_effective_tolerance(self):
        """Tolerancia efectiva."""
        profile = RiskProfile(risk_tolerance=0.8, domain_criticality=0.6)

        # 0.8 * (1 - 0.5 * 0.6) = 0.8 * 0.7 = 0.56
        self.assertAlmostEqual(profile.effective_tolerance, 0.56)

    def test_risk_category(self):
        """Categorización de riesgo."""
        self.assertEqual(
            RiskProfile(risk_tolerance=0.1).risk_category, "HIGHLY_CONSERVATIVE"
        )
        self.assertEqual(
            RiskProfile(risk_tolerance=0.6, domain_criticality=0).risk_category,
            "AGGRESSIVE",
        )
        self.assertEqual(RiskProfile(risk_tolerance=0.5).risk_category, "CONSERVATIVE")
        self.assertEqual(
            RiskProfile(risk_tolerance=0.9, domain_criticality=0).risk_category,
            "HIGHLY_AGGRESSIVE",
        )


class TestPurposeValidator(TestBase):
    """Tests para PurposeValidator."""

    def test_validate_strong_purpose(self):
        """Propósito fuerte."""
        validator = PurposeValidator()

        purposes = [
            BusinessPurpose("caching", "COST_REDUCTION", strength=0.9, confidence=0.95)
        ]

        is_valid, strength, reason = validator.validate(purposes)

        self.assertTrue(is_valid)
        self.assertGreater(strength, 0.7)

    def test_validate_weak_purpose(self):
        """Propósito débil."""
        validator = PurposeValidator()

        purposes = [
            BusinessPurpose("unknown", "COST_REDUCTION", strength=0.3, confidence=0.5)
        ]

        is_valid, strength, reason = validator.validate(purposes)

        self.assertFalse(is_valid)

    def test_validate_non_canonical_problem(self):
        """Problema no canónico."""
        validator = PurposeValidator()

        purposes = [BusinessPurpose("concept", "UNKNOWN_PROBLEM", strength=0.9)]

        is_valid, strength, reason = validator.validate(purposes)

        self.assertFalse(is_valid)

    def test_compute_purpose_score(self):
        """Cálculo de score de propósito."""
        validator = PurposeValidator()

        purposes = [
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.9),
            BusinessPurpose("caching", "COST_REDUCTION", strength=0.7),
        ]

        score = validator.compute_purpose_score(purposes)

        self.assertGreater(score, 0.7)
        self.assertLessEqual(score, 1.0)


class TestConfidenceFilter(TestBase):
    """Tests para ConfidenceFilter."""

    def test_validate_high_confidence(self):
        """Alta confianza."""
        filter = ConfidenceFilter()

        llm_output = LLMOutput(
            entropy=0.5, confidence=0.9, temperature=0.7, num_tokens=100
        )

        is_valid, score, reason = filter.validate(llm_output)

        self.assertTrue(is_valid)
        self.assertGreater(score, 0.7)

    def test_validate_low_confidence(self):
        """Baja confianza."""
        filter = ConfidenceFilter()

        llm_output = LLMOutput(
            entropy=1.0, confidence=0.4, temperature=1.0, num_tokens=100
        )

        is_valid, score, reason = filter.validate(llm_output)

        self.assertFalse(is_valid)

    def test_validate_high_entropy(self):
        """Alta entropía."""
        filter = ConfidenceFilter()

        llm_output = LLMOutput(
            entropy=5.0, confidence=0.8, temperature=1.0, num_tokens=100
        )

        is_valid, score, reason = filter.validate(llm_output)

        self.assertFalse(is_valid)


class TestConstraintMapper(TestBase):
    """Tests para ConstraintMapper."""

    def test_map_to_constraints_conservative(self):
        """Mapeo conservador."""
        mapper = ConstraintMapper()

        profile = RiskProfile(risk_tolerance=0.2, domain_criticality=0.8)

        constraints = mapper.map_to_constraints(profile)

        # Debe tener límites estrictos
        self.assertLess(constraints["cyclomatic"], 20)

    def test_map_to_constraints_aggressive(self):
        """Mapeo agresivo."""
        mapper = ConstraintMapper()

        profile = RiskProfile(risk_tolerance=0.9, domain_criticality=0.1)

        constraints = mapper.map_to_constraints(profile)

        # Límites más relajados
        self.assertGreater(constraints["cyclomatic"], 30)

    def test_compute_constraint_score_satisfied(self):
        """Restricciones satisfechas."""
        mapper = ConstraintMapper()

        profile = RiskProfile(risk_tolerance=0.5)
        metrics = {"cyclomatic": 10, "depth": 3, "loc": 50}

        score = mapper.compute_constraint_score(metrics, profile)

        self.assertGreaterEqual(score, 0.9)

    def test_compute_constraint_score_violated(self):
        """Restricciones violadas."""
        mapper = ConstraintMapper()

        profile = RiskProfile(risk_tolerance=0.3)
        metrics = {"cyclomatic": 50, "depth": 10, "loc": 500}

        score = mapper.compute_constraint_score(metrics, profile)

        self.assertLess(score, 0.5)


class TestSemanticValidationEngine(TestBase):
    """Tests para SemanticValidationEngine."""

    def test_validate_viable(self):
        """Validación VIABLE."""
        engine = SemanticValidationEngine(
            knowledge_graph=create_default_knowledge_graph(),
            risk_profile=RiskProfile(risk_tolerance=0.7),
        )

        purposes = [
            BusinessPurpose(
                "caching", "LATENCY_REDUCTION", strength=0.9, confidence=0.95
            )
        ]

        llm_output = LLMOutput(
            entropy=0.5, confidence=0.9, temperature=0.7, num_tokens=100
        )

        code_metrics = {"cyclomatic": 8, "depth": 3, "loc": 50}

        result = engine.validate(purposes, llm_output, code_metrics)

        self.assertEqual(result.verdict, Verdict.VIABLE)
        self.assertGreater(result.overall_score, 0.75)

    def test_validate_reject_no_purpose(self):
        """Rechazo por falta de propósito."""
        engine = SemanticValidationEngine()

        purposes = [BusinessPurpose("unknown", "UNKNOWN_PROBLEM", strength=0.3)]

        llm_output = LLMOutput(
            entropy=0.5, confidence=0.9, temperature=0.7, num_tokens=100
        )

        result = engine.validate(purposes, llm_output)

        self.assertEqual(result.verdict, Verdict.REJECT)

    def test_validate_reject_low_confidence(self):
        """Rechazo por baja confianza."""
        engine = SemanticValidationEngine(
            knowledge_graph=create_default_knowledge_graph()
        )

        purposes = [BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.9)]

        llm_output = LLMOutput(
            entropy=5.0, confidence=0.3, temperature=1.5, num_tokens=100
        )

        result = engine.validate(purposes, llm_output)

        self.assertEqual(result.verdict, Verdict.REJECT)

    def test_explain_verdict(self):
        """Explicación de veredicto."""
        engine = SemanticValidationEngine()

        purposes = [BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.9)]

        llm_output = LLMOutput(
            entropy=0.5, confidence=0.9, temperature=0.7, num_tokens=100
        )

        result = engine.validate(purposes, llm_output)

        explanation = engine.explain_verdict(result)

        self.assertIn("Verdict:", explanation)
        self.assertIn("Overall Score:", explanation)
        self.assertIn("Signal Breakdown:", explanation)


# ========================================================================================
# TESTS: Casos Extremos
# ========================================================================================


class TestEdgeCases(TestBase):
    """Tests de casos extremos."""

    def test_empty_function(self):
        """Función vacía."""
        code = """
def empty():
    pass
"""

        dataflow, complexity = ASTStaticAnalyzer.analyze_code(code)

        self.assertEqual(complexity.num_functions, 1)
        self.assertEqual(complexity.cyclomatic_complexity, 1)

    def test_only_comments(self):
        """Solo comentarios."""
        code = """
# This is a comment
# Another comment
"""

        dataflow, complexity = ASTStaticAnalyzer.analyze_code(code)

        # Sin funciones ni complejidad significativa
        self.assertEqual(complexity.num_functions, 0)

    def test_unicode_identifiers(self):
        """Identificadores Unicode."""
        code = """
café = 42
résultat = café + 1
"""

        dataflow, complexity = ASTStaticAnalyzer.analyze_code(code)

        self.assertIn("café", dataflow.writes)
        self.assertIn("résultat", dataflow.writes)

    def test_complex_boolean_expression(self):
        """Expresión booleana compleja."""
        code = """
if (a and b) or (c and d) or (e and f):
    pass
"""

        dataflow, complexity = ASTStaticAnalyzer.analyze_code(code)

        # Debe contar todos los operadores
        self.assertGreater(complexity.cyclomatic_complexity, 3)

    def test_nested_comprehensions(self):
        """Comprensiones anidadas."""
        code = """
result = [[x*y for x in range(10)] for y in range(10)]
"""

        dataflow, complexity = ASTStaticAnalyzer.analyze_code(code)

        # Dos for → dos incrementos de complejidad
        self.assertGreaterEqual(complexity.cyclomatic_complexity, 3)


# ========================================================================================
# TESTS: Performance
# ========================================================================================


class TestPerformance(TestBase):
    """Tests de performance."""

    def test_large_function(self):
        """Función grande."""
        # Crear función con muchas líneas
        lines = ["def large_function():"]
        for i in range(100):
            lines.append(f"    x{i} = {i}")
        lines.append("    return x99")

        code = "\n".join(lines)

        # Debe completar sin timeout
        dataflow, complexity = ASTStaticAnalyzer.analyze_code(code)

        self.assertEqual(complexity.num_functions, 1)
        # Muchas variables
        self.assertGreater(len(dataflow.writes), 50)

    def test_many_functions(self):
        """Muchas funciones."""
        lines = []
        for i in range(50):
            lines.append(f"def func{i}():")
            lines.append(f"    return {i}")

        code = "\n".join(lines)

        dataflow, complexity = ASTStaticAnalyzer.analyze_code(code)

        self.assertEqual(complexity.num_functions, 50)


# ========================================================================================
# TESTS: Integración
# ========================================================================================


class TestIntegration(TestBase):
    """Tests de integración end-to-end."""

    def test_full_pipeline_simple_tool(self):
        """Pipeline completo: herramienta simple."""
        code = """
def cache_result(key, value):
    cache[key] = value
    return value
"""

        # 1. Análisis de código
        dataflow, complexity = ASTStaticAnalyzer.analyze_code(code)

        # 2. Validación semántica
        engine = SemanticValidationEngine(
            knowledge_graph=create_default_knowledge_graph(),
            risk_profile=RiskProfile(risk_tolerance=0.6),
        )

        purposes = [BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.9)]

        llm_output = LLMOutput(
            entropy=0.5, confidence=0.9, temperature=0.7, num_tokens=50
        )

        code_metrics = {
            "cyclomatic": complexity.cyclomatic_complexity,
            "depth": complexity.max_nesting_depth,
            "loc": 3,
        }

        result = engine.validate(purposes, llm_output, code_metrics)

        # Debe ser viable
        self.assertTrue(result.verdict.is_accepted)

    def test_full_pipeline_complex_tool(self):
        """Pipeline completo: herramienta compleja."""
        code = """
def complex_processor(data):
    results = []
    for item in data:
        if item.type == 'A':
            if item.value > 100:
                results.append(process_a(item))
            else:
                results.append(default_value())
        elif item.type == 'B':
            results.append(process_b(item))
        elif item.type == 'C':
            results.append(process_c(item))
        else:
            raise ValueError("Unknown type")
    return results
"""

        dataflow, complexity = ASTStaticAnalyzer.analyze_code(code)

        # Complejidad alta
        self.assertGreater(complexity.cyclomatic_complexity, 5)

        # Validación con perfil conservador
        engine = SemanticValidationEngine(risk_profile=RiskProfile(risk_tolerance=0.2))

        purposes = [
            BusinessPurpose("data_validation", "DATA_QUALITY_ENHANCEMENT", strength=0.8)
        ]

        llm_output = LLMOutput(
            entropy=0.8, confidence=0.85, temperature=0.8, num_tokens=150
        )

        code_metrics = {
            "cyclomatic": complexity.cyclomatic_complexity,
            "depth": complexity.max_nesting_depth,
            "loc": 12,
        }

        result = engine.validate(purposes, llm_output, code_metrics)

        # Puede ser condicional o warning debido a complejidad
        self.assertIn(
            result.verdict, [Verdict.VIABLE, Verdict.CONDITIONAL, Verdict.WARNING]
        )


# ========================================================================================
# RUNNER DE TESTS
# ========================================================================================


def run_test_suite():
    """Ejecuta la suite completa de tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Agregar todos los tests
    suite.addTests(loader.loadTestsFromTestCase(TestDataFlowCoordinates))
    suite.addTests(loader.loadTestsFromTestCase(TestComplexityProfile))
    suite.addTests(loader.loadTestsFromTestCase(TestDataFlowAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestJSONStructureValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestTabularNormalizer))
    suite.addTests(loader.loadTestsFromTestCase(TestASTStaticAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestBusinessPurpose))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskProfile))
    suite.addTests(loader.loadTestsFromTestCase(TestPurposeValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestConfidenceFilter))
    suite.addTests(loader.loadTestsFromTestCase(TestConstraintMapper))
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticValidationEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Ejecutar
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN DE TESTS")
    print("=" * 80)
    print(f"Tests ejecutados:  {result.testsRun}")
    print(
        f"Éxitos:           {result.testsRun - len(result.failures) - len(result.errors)}"
    )
    print(f"Fallos:           {len(result.failures)}")
    print(f"Errores:          {len(result.errors)}")
    print(f"Omitidos:         {len(result.skipped)}")
    print("=" * 80)

    return result.wasSuccessful()


# ========================================================================================
# PUNTO DE ENTRADA
# ========================================================================================

if __name__ == "__main__":
    import sys

    success = run_test_suite()
    sys.exit(0 if success else 1)
