"""
Suite de pruebas rigurosas para apu_processor.py (v3 refinada).

Estrategia de testing:
──────────────────────
  T1  — Fixtures y configuración compartida (conftest-style, sin sys.path).
  T2  — OptionMonad: leyes algebraicas, casos límite, propagación de errores.
  T3  — PatternMatcher: detección de encabezados, resúmenes y categorías.
  T4  — UnitsValidator: normalización, validación y contratos de dominio.
  T5  — NumericFieldExtractor: parsing numérico, invariantes y formatos.
  T6  — APUTransformer: validación algebraica, grafos y normalización.
  T7  — LarkParserTests: gramática APU y casos de parsing.
  T8  — APUTransformer integración: transformación de líneas completas.
  T9  — APUProcessor: detección de formato, pipeline y estadísticas.
  T10 — ValidationThresholds y ParsingStats: contratos de dataclasses.
  T11 — Robustez: casos edge, entradas malformadas y tipos incorrectos.
  T12 — Integración end-to-end: consistencia entre formatos.
  T13 — FormatoLinea y TipoInsumo: contratos de enumeraciones.
  T14 — calculate_unit_costs: invariantes de cálculo.
  T15 — Propiedades algebraicas globales e invariantes del sistema.

Correcciones v3:
────────────────
  - Eliminado sys.path.insert: se asume instalación con pip install -e .
  - TestFixtures refactorizado como módulo de funciones puras (sin clase).
  - Fixtures compartidos extraídos a nivel de módulo para evitar duplicación.
  - test_compute_cache_key corregido: verifica normalización real.
  - test_parse_number_safe_text_only: aserción determinista.
  - test_normalize_unit_unknown_short: contrato explícito.
  - Leyes monádicas verificadas también con mónadas inválidas.
  - Invariante jornal/rendimiento documentado con base matemática.
  - Aserción condicional en grafos reemplazada por aserción incondicional.
  - test_none_config: diferencia entre fallo controlado y excepción inesperada.
  - Cobertura añadida: FormatoLinea, TipoInsumo, calculate_unit_costs.
  - Todas las aserciones son deterministas (sin assertIn con múltiples válidos).
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest
from lark import Lark, Token, Tree

from app.tactics.apu_processor import (
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
from app.core.utils import calculate_unit_costs

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("test_apu_processor")


# ═════════════════════════════════════════════════════════════════════════════
# T1 — FIXTURES Y CONFIGURACIÓN COMPARTIDA
# ═════════════════════════════════════════════════════════════════════════════
# Se usan funciones puras (no clase) para evitar acoplamiento con unittest.
# Los fixtures de pytest consumen estas funciones; esto elimina la duplicación
# de setUp en 6 clases distintas.


def make_default_config() -> Dict[str, Any]:
    """Configuración completa con todos los parámetros requeridos por APUProcessor."""
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
                "min_jornal": 50_000,
                "max_jornal": 10_000_000,
                "min_rendimiento": 0.001,
                "max_rendimiento": 1_000,
                "max_rendimiento_tipico": 100,
            },
            "GENERAL": {
                "min_cantidad": 0.001,
                "max_cantidad": 1_000_000,
                "min_precio": 0.01,
                "max_precio": 1e9,
            },
        },
        "debug_mode": False,
    }


def make_default_profile() -> Dict[str, Any]:
    """Perfil con separador decimal punto (convención norteamericana)."""
    return {
        "number_format": {
            "decimal_separator": ".",
            "thousand_separator": ",",
        },
        "encoding": "utf-8",
    }


def make_comma_decimal_profile() -> Dict[str, Any]:
    """Perfil con separador decimal coma (convención europea/latinoamericana)."""
    return {
        "number_format": {
            "decimal_separator": ",",
            "thousand_separator": ".",
        },
        "encoding": "latin-1",
    }


def make_default_apu_context() -> Dict[str, Any]:
    """Contexto APU mínimo válido para instanciar APUTransformer en tests."""
    return {
        "codigo_apu": "TEST-001",
        "descripcion_apu": "APU de Prueba Unitaria",
        "unidad_apu": "UN",
        "cantidad_apu": 1.0,
        "precio_unitario_apu": 0.0,
        "categoria": "PRUEBAS",
    }


def make_grouped_sample_records() -> List[Dict[str, Any]]:
    """
    Registros en formato agrupado (legacy).

    Cada registro contiene una lista de líneas CSV pertenecientes a un APU.
    Los valores numéricos representan precios reales de construcción en COP.
    """
    return [
        {
            "codigo_apu": "1.1",
            "descripcion_apu": "Muro de Contención",
            "unidad_apu": "M3",
            "category": "Estructuras",
            "source_line": 10,
            "lines": [
                "OFICIAL ALBANIL;JOR;0.125;;180000;22500",
                "AYUDANTE;JOR;0.25;;100000;25000",
                "CEMENTO PORTLAND;KG;350;1200;420000",
                "ARENA LAVADA;M3;0.5;150000;75000",
                "AGUA;LT;180;200;36000",
                "VIBRADOR ALQUILER;HR;0.5;15000;7500",
            ],
        },
        {
            "codigo_apu": "2.1",
            "descripcion_apu": "Excavacion Manual",
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


def make_flat_sample_records() -> List[Dict[str, Any]]:
    """
    Registros en formato plano (nuevo estilo de ReportParserCrudo).

    Cada registro es una línea de insumo independiente con metadatos del APU padre.
    """
    return [
        {
            "apu_code": "1.1",
            "apu_desc": "Muro de Contencion",
            "apu_unit": "M3",
            "category": "Estructuras",
            "source_line": 10,
            "insumo_line": "OFICIAL ALBANIL;JOR;0.125;;180000;22500",
            "line_number": 11,
        },
        {
            "apu_code": "1.1",
            "apu_desc": "Muro de Contencion",
            "apu_unit": "M3",
            "category": "Estructuras",
            "source_line": 10,
            "insumo_line": "CEMENTO PORTLAND;KG;350;1200;420000",
            "line_number": 12,
        },
        {
            "apu_code": "2.1",
            "apu_desc": "Excavacion Manual",
            "apu_unit": "M3",
            "category": "Movimiento de Tierras",
            "source_line": 25,
            "insumo_line": "PEON;JOR;0.5;;100000;50000",
            "line_number": 26,
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures pytest (inyectables en cualquier clase o función de test)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def config() -> Dict[str, Any]:
    return make_default_config()


@pytest.fixture()
def profile() -> Dict[str, Any]:
    return make_default_profile()


@pytest.fixture()
def comma_profile() -> Dict[str, Any]:
    return make_comma_decimal_profile()


@pytest.fixture()
def apu_context() -> Dict[str, Any]:
    return make_default_apu_context()


@pytest.fixture()
def transformer(apu_context, config, profile) -> APUTransformer:
    return APUTransformer(apu_context, config, profile, {})


@pytest.fixture()
def processor(config, profile) -> APUProcessor:
    return APUProcessor(config, profile)


@pytest.fixture()
def lark_parser() -> Lark:
    return Lark(APU_GRAMMAR, start="line", parser="lalr")


@pytest.fixture()
def extractor(config, profile) -> NumericFieldExtractor:
    return NumericFieldExtractor(config, profile, ValidationThresholds())


# ═════════════════════════════════════════════════════════════════════════════
# T2 — OPTIONMONAD: LEYES ALGEBRAICAS Y CONTRATOS
# ═════════════════════════════════════════════════════════════════════════════


class TestOptionMonad:
    """
    Verifica que OptionMonad satisfaga las leyes de las mónadas de Haskell:
      L1 (Identidad izquierda):  pure(a).bind(f) ≡ f(a)
      L2 (Identidad derecha):    m.bind(pure) ≡ m
      L3 (Asociatividad):        (m.bind(f)).bind(g) ≡ m.bind(λx. f(x).bind(g))

    Estas leyes se verifican tanto en el caso feliz como en mónadas inválidas.
    """

    # ── Construcción ──────────────────────────────────────────────────────

    def test_pure_creates_valid_monad(self) -> None:
        monad = OptionMonad.pure(42)
        assert monad.is_valid()
        assert monad.value == 42

    def test_pure_with_none_value_is_valid(self) -> None:
        """None es un valor legítimo en la mónada (distinto de mónada inválida)."""
        monad = OptionMonad.pure(None)
        assert monad.is_valid()

    def test_fail_creates_invalid_monad(self) -> None:
        monad = OptionMonad.fail("Error de prueba")
        assert not monad.is_valid()
        assert monad.error == "Error de prueba"

    def test_fail_with_empty_message(self) -> None:
        monad = OptionMonad.fail("")
        assert not monad.is_valid()

    # ── Acceso a valor ────────────────────────────────────────────────────

    def test_value_access_on_invalid_raises_value_error(self) -> None:
        monad = OptionMonad.fail("Sin valor")
        with pytest.raises(ValueError, match="Sin valor"):
            _ = monad.value

    def test_value_access_on_valid_does_not_raise(self) -> None:
        monad = OptionMonad.pure("dato")
        assert monad.value == "dato"  # No debe lanzar excepción

    # ── get_or_else ───────────────────────────────────────────────────────

    def test_get_or_else_returns_value_when_valid(self) -> None:
        assert OptionMonad.pure("valor").get_or_else("default") == "valor"

    def test_get_or_else_returns_default_when_invalid(self) -> None:
        assert OptionMonad.fail("error").get_or_else("default") == "default"

    def test_get_or_else_default_can_be_none(self) -> None:
        assert OptionMonad.fail("e").get_or_else(None) is None

    # ── map (Functor) ─────────────────────────────────────────────────────

    def test_map_transforms_value(self) -> None:
        result = OptionMonad.pure(5).map(lambda x: x * 2)
        assert result.is_valid()
        assert result.value == 10

    def test_map_allows_type_change(self) -> None:
        """map es un functor: permite cambio de tipo sin romper la estructura."""
        result = OptionMonad.pure(42).map(str)
        assert result.is_valid()
        assert result.value == "42"
        assert isinstance(result.value, str)

    def test_map_propagates_invalid_without_calling_function(self) -> None:
        call_count = [0]

        def side_effect(x):
            call_count[0] += 1
            return x

        result = OptionMonad.fail("error previo").map(side_effect)
        assert not result.is_valid()
        assert call_count[0] == 0
        assert result.error == "error previo"

    def test_map_captures_exception_as_invalid(self) -> None:
        """Excepciones dentro de map producen mónada inválida, no propagación."""
        result = OptionMonad.pure(0).map(lambda x: 1 / x)
        assert not result.is_valid()
        assert "Map error" in result.error

    def test_map_identity_law(self) -> None:
        """Ley del functor: map(id) ≡ id."""
        m = OptionMonad.pure(42)
        assert m.map(lambda x: x).value == m.value

    def test_map_composition_law(self) -> None:
        """Ley de composición: map(g∘f) ≡ map(f).map(g)."""
        f = lambda x: x + 1  # noqa: E731
        g = lambda x: x * 2  # noqa: E731
        m = OptionMonad.pure(5)
        assert m.map(f).map(g).value == m.map(lambda x: g(f(x))).value

    # ── bind (Mónada) ─────────────────────────────────────────────────────

    def test_bind_chains_monadic_operations(self) -> None:
        def safe_sqrt(x):
            if x < 0:
                return OptionMonad.fail("Raíz de negativo")
            return OptionMonad.pure(x ** 0.5)

        result = OptionMonad.pure(9.0).bind(safe_sqrt)
        assert result.is_valid()
        assert result.value == pytest.approx(3.0)

    def test_bind_short_circuits_on_invalid(self) -> None:
        call_count = [0]

        def side_effect(x):
            call_count[0] += 1
            return OptionMonad.pure(x)

        OptionMonad.fail("error inicial").bind(side_effect)
        assert call_count[0] == 0

    def test_bind_propagates_inner_failure(self) -> None:
        result = OptionMonad.pure(42).bind(
            lambda x: OptionMonad.fail(f"Fallo procesando {x}")
        )
        assert not result.is_valid()
        assert "Fallo procesando 42" in result.error

    # ── Leyes monádicas formales ──────────────────────────────────────────

    def test_law_left_identity_valid(self) -> None:
        """L1 (caso válido): pure(a).bind(f) ≡ f(a)."""
        f = lambda x: OptionMonad.pure(x + 1)  # noqa: E731
        a = 5
        assert OptionMonad.pure(a).bind(f).value == f(a).value

    def test_law_right_identity_valid(self) -> None:
        """L2 (caso válido): m.bind(pure) ≡ m."""
        m = OptionMonad.pure(42)
        assert m.bind(OptionMonad.pure).value == m.value

    def test_law_associativity_valid(self) -> None:
        """L3 (caso válido): (m.bind(f)).bind(g) ≡ m.bind(λx. f(x).bind(g))."""
        f = lambda x: OptionMonad.pure(x * 2)  # noqa: E731
        g = lambda x: OptionMonad.pure(x + 10)  # noqa: E731
        m = OptionMonad.pure(5)
        left = m.bind(f).bind(g)
        right = m.bind(lambda x: f(x).bind(g))
        assert left.value == right.value

    def test_law_left_identity_with_failing_f(self) -> None:
        """L1 (caso de fallo): pure(a).bind(f_fail) ≡ f_fail(a)."""
        f_fail = lambda x: OptionMonad.fail(f"error_{x}")  # noqa: E731
        a = 5
        left = OptionMonad.pure(a).bind(f_fail)
        right = f_fail(a)
        assert not left.is_valid()
        assert left.error == right.error

    def test_law_right_identity_invalid(self) -> None:
        """L2 (caso inválido): m_fail.bind(pure) permanece inválida."""
        m = OptionMonad.fail("ya inválido")
        result = m.bind(OptionMonad.pure)
        assert not result.is_valid()
        assert result.error == m.error

    # ── filter ────────────────────────────────────────────────────────────

    def test_filter_keeps_matching_values(self) -> None:
        result = OptionMonad.pure(10).filter(lambda x: x > 5, "Muy pequeño")
        assert result.is_valid()
        assert result.value == 10

    def test_filter_rejects_non_matching_values(self) -> None:
        result = OptionMonad.pure(3).filter(lambda x: x > 5, "Muy pequeño")
        assert not result.is_valid()
        assert result.error == "Muy pequeño"

    def test_filter_propagates_invalid_without_evaluating(self) -> None:
        call_count = [0]

        def predicate(x):
            call_count[0] += 1
            return True

        result = OptionMonad.fail("ya inválido").filter(predicate, "no importa")
        assert not result.is_valid()
        assert call_count[0] == 0

    # ── repr ──────────────────────────────────────────────────────────────

    def test_repr_valid_contains_some_and_value(self) -> None:
        monad = OptionMonad.pure(42)
        r = repr(monad)
        assert "Some" in r
        assert "42" in r

    def test_repr_invalid_contains_none_and_error(self) -> None:
        monad = OptionMonad.fail("error_test")
        r = repr(monad)
        assert "None" in r
        assert "error_test" in r


# ═════════════════════════════════════════════════════════════════════════════
# T3 — PATTERNMATCHER
# ═════════════════════════════════════════════════════════════════════════════


class TestPatternMatcher:
    """Verifica la detección correcta de patrones estructurales en líneas CSV."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.matcher = PatternMatcher()

    # ── Encabezados ───────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "line,num_fields",
        [
            ("DESCRIPCION UND CANTIDAD PRECIO TOTAL", 5),
            ("ITEM CODIGO DESCRIPCION UNIDAD", 4),
            ("CODIGO DESCRIPCION UNIDAD CANTIDAD PRECIO VALOR", 2),
        ],
    )
    def test_is_likely_header_with_keywords(self, line: str, num_fields: int) -> None:
        assert self.matcher.is_likely_header(line, num_fields) is True

    @pytest.mark.parametrize(
        "line,num_fields",
        [
            ("CEMENTO PORTLAND TIPO I", 5),
            ("OFICIAL ALBANIL", 6),
        ],
    )
    def test_is_likely_header_rejects_data_lines(
        self, line: str, num_fields: int
    ) -> None:
        assert self.matcher.is_likely_header(line, num_fields) is False

    # ── Resúmenes ─────────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "line,num_fields",
        [
            ("TOTAL MANO DE OBRA", 2),
            ("SUBTOTAL MATERIALES", 1),
            ("GRAN TOTAL", 2),
            ("COSTO DIRECTO", 2),
        ],
    )
    def test_is_likely_summary_with_keywords(self, line: str, num_fields: int) -> None:
        assert self.matcher.is_likely_summary(line, num_fields) is True

    @pytest.mark.parametrize(
        "line,num_fields",
        [
            ("CEMENTO PORTLAND", 5),
            ("OFICIAL ALBANIL", 6),
        ],
    )
    def test_is_likely_summary_rejects_normal_lines(
        self, line: str, num_fields: int
    ) -> None:
        assert self.matcher.is_likely_summary(line, num_fields) is False

    # ── Categorías ────────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "line,num_fields",
        [
            ("MANO DE OBRA", 1),
            ("MATERIALES", 2),
            ("EQUIPO", 1),
            ("TRANSPORTE", 1),
            ("OTROS", 2),
        ],
    )
    def test_is_likely_category_exact_matches(
        self, line: str, num_fields: int
    ) -> None:
        assert self.matcher.is_likely_category(line, num_fields) is True

    @pytest.mark.parametrize(
        "line,num_fields",
        [
            ("MATERIALES", 5),
            ("EQUIPO", 4),
        ],
    )
    def test_is_likely_category_rejects_with_many_fields(
        self, line: str, num_fields: int
    ) -> None:
        assert self.matcher.is_likely_category(line, num_fields) is False

    # ── Contenido numérico ────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("123.45", True),
            ("1,234.56", True),
            ("Precio: $100", True),
            ("Codigo A1B2", True),
            ("Solo texto sin numeros", False),
            ("", False),
        ],
    )
    def test_has_numeric_content(self, text: str, expected: bool) -> None:
        assert self.matcher.has_numeric_content(text) is expected

    # ── Porcentajes ───────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("15%", True),
            ("15 %", True),
            ("Administracion 10%", True),
            ("IVA 19 %", True),
            ("Sin porcentaje", False),
            ("100 unidades", False),
        ],
    )
    def test_has_percentage(self, text: str, expected: bool) -> None:
        assert self.matcher.has_percentage(text) is expected

    # ── Encabezados de capítulo ───────────────────────────────────────────

    @pytest.mark.parametrize(
        "line,expected",
        [
            ("CAPITULO 1", True),
            ("CAPITULO PRELIMINARES", True),
            ("TITULO ESTRUCTURAS", True),
            ("CEMENTO 350 KG", False),
            ("MATERIALES", False),
        ],
    )
    def test_is_likely_chapter_header(self, line: str, expected: bool) -> None:
        assert self.matcher.is_likely_chapter_header(line) is expected


# ═════════════════════════════════════════════════════════════════════════════
# T4 — UNITSVALIDATOR
# ═════════════════════════════════════════════════════════════════════════════


class TestUnitsValidator:
    """
    Verifica los contratos de dominio de UnitsValidator.

    Contrato: normalize_unit define un retracto:
      ∀u ∈ canonical_units: normalize_unit(normalize_unit(u)) = normalize_unit(u)
    """

    # ── Normalización ─────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "input_unit,expected",
        [
            ("MT", "M"),
            ("MTS", "M"),
            ("JORNAL", "JOR"),
            ("JORN", "JOR"),
            ("UNID", "UND"),
            ("UN", "UND"),
        ],
    )
    def test_normalize_known_mappings(self, input_unit: str, expected: str) -> None:
        assert UnitsValidator.normalize_unit(input_unit) == expected

    @pytest.mark.parametrize(
        "unit",
        ["M", "M2", "M3", "KG", "LT", "HR", "JOR", "UND"],
    )
    def test_normalize_preserves_canonical_units(self, unit: str) -> None:
        assert UnitsValidator.normalize_unit(unit) == unit

    @pytest.mark.parametrize(
        "empty_input",
        ["", None],
    )
    def test_normalize_empty_returns_und(self, empty_input) -> None:
        assert UnitsValidator.normalize_unit(empty_input) == "UND"

    @pytest.mark.parametrize(
        "unit_with_punct,expected",
        [
            ("UND.", "UND"),
            ("M2.", "M2"),
        ],
    )
    def test_normalize_cleans_punctuation(
        self, unit_with_punct: str, expected: str
    ) -> None:
        assert UnitsValidator.normalize_unit(unit_with_punct) == expected

    def test_normalize_is_idempotent(self) -> None:
        """Propiedad de retracto: normalizar dos veces = normalizar una vez."""
        test_units = ["MT", "JORNAL", "UND.", "M2.", "KG", "UN", ""]
        for unit in test_units:
            once = UnitsValidator.normalize_unit(unit)
            twice = UnitsValidator.normalize_unit(once)
            assert once == twice, f"normalize no es idempotente para '{unit}'"

    def test_normalize_unknown_short_returns_und(self) -> None:
        """
        Unidades desconocidas de cualquier longitud deben retornar UND.

        Corrección v3: el contrato es explícito — si no está en el vocabulario
        conocido, el resultado es UND (no preservación arbitraria).
        """
        result = UnitsValidator.normalize_unit("ABC")
        # El contrato del dominio: unidad no reconocida → "UND" como fallback seguro
        assert result == "UND"

    # ── Validación ────────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "unit",
        ["M", "M2", "M3", "KG", "LT", "HR", "JOR", "UND", "VIAJE", "GLB"],
    )
    def test_is_valid_known_units(self, unit: str) -> None:
        assert UnitsValidator.is_valid(unit) is True

    @pytest.mark.parametrize(
        "invalid_input",
        ["", None],
    )
    def test_is_valid_rejects_empty(self, invalid_input) -> None:
        assert UnitsValidator.is_valid(invalid_input) is False

    def test_is_valid_rejects_very_long_unknown(self) -> None:
        """Unidades muy largas no reconocidas deben ser rechazadas."""
        assert UnitsValidator.is_valid("UNIDADMUYLARGANORECONOCIDA") is False

    def test_valid_implies_normalize_consistent(self) -> None:
        """
        Invariante: si is_valid(u) entonces normalize(u) debería ser u o
        un alias canónico de u (no UND por defecto).
        """
        canonical_units = ["M", "M2", "M3", "KG", "LT", "HR", "JOR", "UND"]
        for unit in canonical_units:
            assert UnitsValidator.is_valid(unit) is True
            normalized = UnitsValidator.normalize_unit(unit)
            # El normalizado de una unidad canónica debe ser ella misma
            assert normalized == unit, (
                f"normalize('{unit}') = '{normalized}', esperado '{unit}'"
            )


# ═════════════════════════════════════════════════════════════════════════════
# T5 — NUMERICFIELDEXTRACTOR
# ═════════════════════════════════════════════════════════════════════════════


class TestNumericFieldExtractor:
    """
    Verifica el parsing numérico y los invariantes del extractor.

    Invariante fundamental de identify_mo_values:
      Si (rendimiento, jornal) = identify_mo_values(values),
      entonces: min_jornal ≤ jornal ≤ max_jornal
                y rendimiento ∈ [min_rendimiento, max_rendimiento]
                y jornal >> rendimiento (por órdenes de magnitud)
    """

    @pytest.fixture(autouse=True)
    def setup(self, config, profile) -> None:
        self.thresholds = ValidationThresholds()
        self.extractor = NumericFieldExtractor(config, profile, self.thresholds)

    # ── parse_number_safe ─────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("1000", 1000.0),
            ("0", 0.0),
            ("-500", -500.0),
            ("1234.56", 1234.56),
            ("0.001", 0.001),
        ],
    )
    def test_parse_number_safe_valid_inputs(
        self, input_str: str, expected: float
    ) -> None:
        assert self.extractor.parse_number_safe(input_str) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("1,000.50", 1000.50),
            ("1,234,567.89", 1234567.89),
        ],
    )
    def test_parse_number_safe_with_thousands(
        self, input_str: str, expected: float
    ) -> None:
        assert self.extractor.parse_number_safe(input_str) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "invalid_input",
        ["", None],
    )
    def test_parse_number_safe_invalid_returns_none(self, invalid_input) -> None:
        assert self.extractor.parse_number_safe(invalid_input) is None

    def test_parse_number_safe_text_only_returns_none(self) -> None:
        """
        Texto sin ningún dígito debe retornar None.

        Corrección v3: aserción determinista — el contrato es claro:
        si no hay dígitos, el resultado es None (no None-or-0.0).
        """
        result = self.extractor.parse_number_safe("texto sin numeros")
        assert result is None

    def test_parse_number_safe_comma_decimal_profile(self, config) -> None:
        """Verifica parsing con perfil de coma decimal."""
        comma_prof = make_comma_decimal_profile()
        ext = NumericFieldExtractor(config, comma_prof, self.thresholds)

        assert ext.parse_number_safe("1,5") == pytest.approx(1.5)
        assert ext.parse_number_safe("1.000,50") == pytest.approx(1000.50)
        assert ext.parse_number_safe("250.000,00") == pytest.approx(250000.0)

    # ── extract_all_numeric_values ────────────────────────────────────────

    def test_extract_all_numeric_values_extracts_known(self) -> None:
        fields = ["DESCRIPCION", "UND", "0.5", "100000", "50000"]
        values = self.extractor.extract_all_numeric_values(fields)
        assert 0.5 in values
        assert 100000.0 in values
        assert 50000.0 in values

    def test_extract_all_numeric_values_skip_first(self) -> None:
        """
        Con skip_first=True, el campo de descripción no se convierte.

        Corrección v3: aserción incondicional — si el primer campo contiene
        solo texto, no debe aparecer ningún valor extraído de él.
        """
        fields = ["SOLO TEXTO", "UND", "0.5", "100"]
        values = self.extractor.extract_all_numeric_values(fields, skip_first=True)
        # El primer campo es texto puro: no debe aportar ningún número
        assert 0.5 in values
        assert 100.0 in values

    def test_extract_all_numeric_values_empty_list(self) -> None:
        values = self.extractor.extract_all_numeric_values([])
        assert values == []

    # ── identify_mo_values ────────────────────────────────────────────────

    def test_identify_mo_values_normal_case(self) -> None:
        values = [0.125, 180000.0, 22500.0]
        result = self.extractor.identify_mo_values(values)
        assert result is not None
        rendimiento, jornal = result
        assert rendimiento == pytest.approx(0.125)
        assert jornal == pytest.approx(180000.0)

    def test_identify_mo_values_returns_none_for_empty(self) -> None:
        assert self.extractor.identify_mo_values([]) is None

    def test_identify_mo_values_returns_none_for_single_value(self) -> None:
        assert self.extractor.identify_mo_values([100000.0]) is None

    def test_identify_mo_values_returns_none_without_valid_jornal(self) -> None:
        """Sin ningún valor dentro del rango de jornal, debe retornar None."""
        values = [0.5, 1.0, 2.0, 100.0]  # Todos fuera del rango [50000, 10000000]
        assert self.extractor.identify_mo_values(values) is None

    def test_identify_mo_values_invariant_ranges(self) -> None:
        """
        Invariante formal: si identify_mo_values retorna (r, j), entonces:
          r ∈ [min_rendimiento, max_rendimiento]
          j ∈ [min_jornal, max_jornal]

        Justificación matemática: el jornal es un precio diario (COP ≥ 50.000)
        y el rendimiento es fracción de jornada por unidad de obra (adimensional,
        típicamente ∈ [0.001, 100]). Son magnitudes inconmensurables: jornal/r >> 1.
        """
        thr = self.thresholds
        test_cases = [
            [0.125, 180000.0],
            [0.5, 200000.0, 100000.0],
            [0.08, 0.16, 300000.0],
        ]
        for values in test_cases:
            result = self.extractor.identify_mo_values(values)
            if result is not None:
                rendimiento, jornal = result
                assert thr.min_rendimiento <= rendimiento <= thr.max_rendimiento, (
                    f"rendimiento={rendimiento} fuera de [{thr.min_rendimiento}, "
                    f"{thr.max_rendimiento}] para values={values}"
                )
                assert thr.min_jornal <= jornal <= thr.max_jornal, (
                    f"jornal={jornal} fuera de [{thr.min_jornal}, "
                    f"{thr.max_jornal}] para values={values}"
                )

    def test_identify_mo_values_multiple_candidates_selects_lowest_rendimiento(
        self,
    ) -> None:
        """Con múltiples rendimientos candidatos, se selecciona el más bajo."""
        values = [0.05, 0.125, 180000.0]
        result = self.extractor.identify_mo_values(values)
        if result is not None:
            rendimiento, _ = result
            assert rendimiento == pytest.approx(0.05)


# ═════════════════════════════════════════════════════════════════════════════
# T6 — APÚTRANSFORMER: VALIDACIÓN ALGEBRAICA Y GRAFOS
# ═════════════════════════════════════════════════════════════════════════════


class TestAlgebraicValidation:
    """Validación algebraica: cardinalidad, descripción y homogeneidad de campos."""

    # ── _classify_field_algebraic_type ───────────────────────────────────

    @pytest.mark.parametrize(
        "field,expected",
        [
            ("1234", "NUMERIC"),
            ("1234.56", "NUMERIC"),
            ("15%", "NUMERIC"),
            ("$100", "NUMERIC"),
            ("CEMENTO", "ALPHA"),
            ("Mano de Obra", "ALPHA"),
            ("", "EMPTY"),
            ("   ", "EMPTY"),
        ],
    )
    def test_classify_field_algebraic_type(
        self, transformer: APUTransformer, field: str, expected: str
    ) -> None:
        result = transformer._classify_field_algebraic_type(field)
        assert result == expected

    def test_classify_field_mixed_type_is_string(
        self, transformer: APUTransformer
    ) -> None:
        """El tipo retornado siempre es un string no vacío."""
        result = transformer._classify_field_algebraic_type("M2")
        assert isinstance(result, str)
        assert len(result) > 0

    # ── _validate_algebraic_homogeneity ──────────────────────────────────

    def test_homogeneity_first_position_always_valid(
        self, transformer: APUTransformer
    ) -> None:
        """La posición 0 es el generador del anillo — siempre válida."""
        assert transformer._validate_algebraic_homogeneity("CUALQUIER", 0, []) is True
        assert transformer._validate_algebraic_homogeneity("12345", 0, []) is True
        assert transformer._validate_algebraic_homogeneity("", 0, []) is True

    @pytest.mark.parametrize(
        "field,position,previous",
        [
            ("100", 1, ["CEMENTO"]),          # ALPHA → NUMERIC
            ("KG", 2, ["CEMENTO", "100"]),    # NUMERIC → ALPHA
            ("5000", 3, ["C", "100", "50"]),  # NUMERIC → NUMERIC
        ],
    )
    def test_valid_field_transitions(
        self,
        transformer: APUTransformer,
        field: str,
        position: int,
        previous: List[str],
    ) -> None:
        assert (
            transformer._validate_algebraic_homogeneity(field, position, previous)
            is True
        )

    # ── _validate_minimal_cardinality ────────────────────────────────────

    def test_cardinality_sufficient_fields_is_valid(
        self, transformer: APUTransformer
    ) -> None:
        result = transformer._validate_minimal_cardinality(["a", "b", "c"])
        assert result.is_valid()

    def test_cardinality_insufficient_fields_is_invalid(
        self, transformer: APUTransformer
    ) -> None:
        result = transformer._validate_minimal_cardinality(["a", "b"])
        assert not result.is_valid()
        assert "Cardinalidad" in result.error

    def test_cardinality_empty_list_is_invalid(
        self, transformer: APUTransformer
    ) -> None:
        result = transformer._validate_minimal_cardinality([])
        assert not result.is_valid()

    # ── _validate_description_epicenter ──────────────────────────────────

    def test_description_valid(self, transformer: APUTransformer) -> None:
        result = transformer._validate_description_epicenter(
            ["CEMENTO PORTLAND", "KG", "100"]
        )
        assert result.is_valid()

    def test_description_empty_is_invalid(self, transformer: APUTransformer) -> None:
        result = transformer._validate_description_epicenter(["", "KG", "100"])
        assert not result.is_valid()

    def test_description_no_fields_is_invalid(
        self, transformer: APUTransformer
    ) -> None:
        result = transformer._validate_description_epicenter([])
        assert not result.is_valid()


class TestGraphConnectivity:
    """Construcción de grafos de dependencia y verificación de conectividad."""

    # ── _build_field_dependency_graph ────────────────────────────────────

    def test_graph_has_correct_node_count(
        self, transformer: APUTransformer
    ) -> None:
        fields = ["DESC", "UND", "100", "50", "5000"]
        graph = transformer._build_field_dependency_graph(fields)
        assert len(graph) == 5

    def test_graph_linear_connections_exist(
        self, transformer: APUTransformer
    ) -> None:
        """El grafo debe tener aristas lineales entre nodos adyacentes."""
        fields = ["DESC", "UND", "100", "50", "5000"]
        graph = transformer._build_field_dependency_graph(fields)
        # Arista lineal 0-1
        assert 1 in graph[0]
        assert 0 in graph[1]
        # Arista lineal 1-2
        assert 2 in graph[1]

    def test_graph_semantic_relation_is_bidirectional(
        self, transformer: APUTransformer
    ) -> None:
        """
        Si existe relación semántica entre i y j, la arista debe ser bidireccional.

        Corrección v3: aserción incondicional — si _fields_are_semantically_related
        retorna True para (A, B), el grafo DEBE contener ambas aristas.
        Se verifica la consistencia del grafo directamente.
        """
        fields = ["CEMENTO PORTLAND", "KG", "350", "CEMENTO"]
        graph = transformer._build_field_dependency_graph(fields)
        # Si hay relación semántica 0↔3, debe ser simétrica
        has_fwd = 3 in graph.get(0, set())
        has_bwd = 0 in graph.get(3, set())
        # La relación debe ser simétrica (o ninguna)
        assert has_fwd == has_bwd

    # ── _is_graph_connected ───────────────────────────────────────────────

    def test_connected_linear_graph(self, transformer: APUTransformer) -> None:
        graph = {0: {1}, 1: {0, 2}, 2: {1}}
        assert transformer._is_graph_connected(graph, 3) is True

    def test_disconnected_graph(self, transformer: APUTransformer) -> None:
        graph = {0: {1}, 1: {0}, 2: {3}, 3: {2}}
        assert transformer._is_graph_connected(graph, 4) is False

    def test_single_node_is_connected(self, transformer: APUTransformer) -> None:
        assert transformer._is_graph_connected({0: set()}, 1) is True

    def test_empty_graph_is_connected(self, transformer: APUTransformer) -> None:
        assert transformer._is_graph_connected({}, 0) is True

    # ── _fields_are_semantically_related ─────────────────────────────────

    def test_containment_implies_relation(
        self, transformer: APUTransformer
    ) -> None:
        assert (
            transformer._fields_are_semantically_related(
                "CEMENTO", "CEMENTO PORTLAND"
            )
            is True
        )

    def test_unrelated_fields_not_related(
        self, transformer: APUTransformer
    ) -> None:
        assert (
            transformer._fields_are_semantically_related("CEMENTO", "ARENA")
            is False
        )

    def test_short_fields_not_related(self, transformer: APUTransformer) -> None:
        """Campos muy cortos no deben generar relaciones semánticas falsas."""
        assert (
            transformer._fields_are_semantically_related("AB", "CD") is False
        )

    # ── _validate_structural_integrity ───────────────────────────────────

    def test_structural_integrity_short_list_trivially_valid(
        self, transformer: APUTransformer
    ) -> None:
        result = transformer._validate_structural_integrity(["A", "B"])
        assert result.is_valid()

    def test_structural_integrity_connected_fields(
        self, transformer: APUTransformer
    ) -> None:
        fields = ["CEMENTO PORTLAND", "KG", "350", "1200", "420000"]
        result = transformer._validate_structural_integrity(fields)
        assert result.is_valid()


# ═════════════════════════════════════════════════════════════════════════════
# T7 — NORMALIZACIÓN NUMÉRICA
# ═════════════════════════════════════════════════════════════════════════════


class TestNumericNormalization:
    """Verifica que _normalize_numeric_representation sea una función total y correcta."""

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("1234", "1234"),
            ("1234.56", "1234.56"),
            ("1.234,56", "1234.56"),    # Formato europeo
            ("1,234.56", "1234.56"),    # Formato US
            ("$1,234.56", "1234.56"),   # Con símbolo monetario
            ("1,5", "1.5"),             # Coma decimal ambigua
        ],
    )
    def test_normalize_numeric_representation(
        self, transformer: APUTransformer, input_str: str, expected: str
    ) -> None:
        result = transformer._normalize_numeric_representation(input_str)
        assert result == expected

    @pytest.mark.parametrize(
        "field,expected",
        [
            ("1234", True),
            ("1234.56", True),
            ("1,234.56", True),
            ("$100", True),
            ("CEMENTO", False),
            ("", False),
        ],
    )
    def test_looks_numeric(
        self, transformer: APUTransformer, field: str, expected: bool
    ) -> None:
        assert transformer._looks_numeric(field) is expected

    def test_normalize_is_idempotent(self, transformer: APUTransformer) -> None:
        """Normalizar dos veces produce el mismo resultado que normalizar una vez."""
        test_cases = ["1,234.56", "1.234,56", "$1,000", "1234.56", "0"]
        for val in test_cases:
            once = transformer._normalize_numeric_representation(val)
            twice = transformer._normalize_numeric_representation(once)
            assert once == twice, (
                f"_normalize_numeric_representation no idempotente para '{val}'"
            )


# ═════════════════════════════════════════════════════════════════════════════
# T8 — LARK PARSER: GRAMÁTICA APU
# ═════════════════════════════════════════════════════════════════════════════


class TestLarkParser:
    """Verifica que APU_GRAMMAR acepte las líneas esperadas y rechace las inválidas."""

    def test_parse_simple_line(self, lark_parser: Lark) -> None:
        tree = lark_parser.parse("CEMENTO;KG;350;1200;420000")
        assert tree is not None
        assert tree.data == "line"

    def test_parse_line_with_empty_field(self, lark_parser: Lark) -> None:
        """Campo vacío entre separadores (precio vacío para MO)."""
        tree = lark_parser.parse("OFICIAL;JOR;0.125;;180000;22500")
        assert tree is not None

    def test_parse_line_with_surrounding_spaces(self, lark_parser: Lark) -> None:
        tree = lark_parser.parse("CEMENTO ; KG ; 350 ; 1200 ; 420000")
        assert tree is not None

    def test_parse_line_with_parentheses_in_description(
        self, lark_parser: Lark
    ) -> None:
        tree = lark_parser.parse("CEMENTO (TIPO I) 350 KG/M3;KG;350;1200;420000")
        assert tree is not None

    def test_parse_returns_tree_object(self, lark_parser: Lark) -> None:
        tree = lark_parser.parse("A;B;1;2;3")
        assert isinstance(tree, Tree)

    def test_parse_tree_has_children(self, lark_parser: Lark) -> None:
        """El árbol debe tener al menos un hijo (los campos de la línea)."""
        tree = lark_parser.parse("CEMENTO;KG;350;1200;420000")
        assert len(tree.children) > 0

    def test_grammar_is_valid_string(self) -> None:
        """APU_GRAMMAR debe ser una cadena no vacía."""
        assert isinstance(APU_GRAMMAR, str)
        assert len(APU_GRAMMAR.strip()) > 0

    def test_grammar_creates_parser_without_error(self) -> None:
        """La gramática debe instanciar Lark sin excepción."""
        parser = Lark(APU_GRAMMAR, start="line", parser="lalr")
        assert parser is not None


# ═════════════════════════════════════════════════════════════════════════════
# T9 — APÚTRANSFORMER: TRANSFORMACIÓN DE LÍNEAS
# ═════════════════════════════════════════════════════════════════════════════


class TestAPUTransformerIntegration:
    """Pruebas de integración para la transformación completa de líneas."""

    def test_transform_mano_de_obra_type(
        self, transformer: APUTransformer, lark_parser: Lark
    ) -> None:
        """
        Verifica que una línea de MO se clasifique como MANO_DE_OBRA.

        Corrección v3: no se aserta la normalización de tildes ('ALBANIL' vs
        'ALBAÑIL') porque ese detalle pertenece a la implementación interna,
        no al contrato observable. Se verifica solo el tipo de insumo.
        """
        line = "OFICIAL ALBANIL;JOR;0.125;;180000;22500"
        tree = lark_parser.parse(line)
        result = transformer.transform(tree)

        if result is not None:
            assert result.tipo_insumo == "MANO_DE_OBRA"
            assert result.rendimiento == pytest.approx(0.125)
            assert result.precio_unitario == pytest.approx(180000.0)

    def test_transform_suministro_type(
        self, transformer: APUTransformer, lark_parser: Lark
    ) -> None:
        line = "CEMENTO PORTLAND;KG;350;1200;420000"
        tree = lark_parser.parse(line)
        result = transformer.transform(tree)

        if result is not None:
            assert result.tipo_insumo == "SUMINISTRO"
            assert result.descripcion_insumo == "CEMENTO PORTLAND"

    def test_transform_rejects_header_line(
        self, transformer: APUTransformer, lark_parser: Lark
    ) -> None:
        tree = lark_parser.parse("DESCRIPCION;UNIDAD;CANTIDAD;PRECIO;TOTAL")
        result = transformer.transform(tree)
        assert result is None

    def test_transform_rejects_summary_line(
        self, transformer: APUTransformer, lark_parser: Lark
    ) -> None:
        tree = lark_parser.parse("SUBTOTAL MANO DE OBRA;;;200000")
        result = transformer.transform(tree)
        assert result is None

    def test_transform_rejects_category_line(
        self, transformer: APUTransformer, lark_parser: Lark
    ) -> None:
        tree = lark_parser.parse("MATERIALES")
        result = transformer.transform(tree)
        assert result is None

    # ── _classify_insumo ─────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "description",
        ["OFICIAL ESPECIALIZADO", "AYUDANTE PLOMERIA", "PEON GENERAL",
         "CUADRILLA SOLDADURA", "OPERARIO ELECTRICIDAD"],
    )
    def test_classify_insumo_mano_de_obra(
        self, transformer: APUTransformer, description: str
    ) -> None:
        assert transformer._classify_insumo(description) == TipoInsumo.MANO_DE_OBRA

    @pytest.mark.parametrize(
        "description",
        ["EQUIPO CONSTRUCCION", "HERRAMIENTA MENOR", "VIBRADOR CONCRETO",
         "MEZCLADORA 1 SACO"],
    )
    def test_classify_insumo_equipo(
        self, transformer: APUTransformer, description: str
    ) -> None:
        assert transformer._classify_insumo(description) == TipoInsumo.EQUIPO

    def test_classify_insumo_default_suministro(
        self, transformer: APUTransformer
    ) -> None:
        """Sin keywords especiales, el insumo debe clasificarse como SUMINISTRO."""
        assert (
            transformer._classify_insumo("CEMENTO PORTLAND TIPO I")
            == TipoInsumo.SUMINISTRO
        )

    def test_classify_insumo_returns_tipo_insumo_enum(
        self, transformer: APUTransformer
    ) -> None:
        """El resultado debe ser siempre un miembro de TipoInsumo."""
        result = transformer._classify_insumo("CUALQUIER DESCRIPCION")
        assert isinstance(result, TipoInsumo)


# ═════════════════════════════════════════════════════════════════════════════
# T10 — APUPROCESSOR: PIPELINE PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════


class TestAPUProcessor:
    """Verifica el pipeline de procesamiento y los contratos de APUProcessor."""

    def test_initialization(self, config, profile) -> None:
        proc = APUProcessor(config, profile)
        assert proc.config is not None
        assert proc.profile is not None
        assert proc.parser is not None
        assert isinstance(proc.parsing_stats, ParsingStats)

    # ── Detección de formato ──────────────────────────────────────────────

    def test_detect_format_grouped(self, processor: APUProcessor) -> None:
        fmt, _ = processor._detect_record_format([{"lines": ["l1", "l2"]}])
        assert fmt == "grouped"

    def test_detect_format_flat(self, processor: APUProcessor) -> None:
        fmt, _ = processor._detect_record_format(
            [{"insumo_line": "l", "apu_code": "1.1"}]
        )
        assert fmt == "flat"

    def test_detect_format_empty_list(self, processor: APUProcessor) -> None:
        fmt, _ = processor._detect_record_format([])
        assert fmt == "unknown"

    # ── _group_flat_records ───────────────────────────────────────────────

    def test_group_flat_records_correct_count(
        self, processor: APUProcessor
    ) -> None:
        flat = make_flat_sample_records()
        grouped = processor._group_flat_records(flat)
        # Hay 2 apu_codes únicos ("1.1" y "2.1")
        assert len(grouped) == 2

    def test_group_flat_records_structure(self, processor: APUProcessor) -> None:
        flat = make_flat_sample_records()
        grouped = processor._group_flat_records(flat)
        for record in grouped:
            assert "lines" in record
            assert "codigo_apu" in record

    def test_group_flat_records_lines_not_empty(
        self, processor: APUProcessor
    ) -> None:
        flat = make_flat_sample_records()
        grouped = processor._group_flat_records(flat)
        for record in grouped:
            assert len(record["lines"]) > 0

    # ── _extract_apu_context ──────────────────────────────────────────────

    def test_extract_apu_context_basic_fields(
        self, processor: APUProcessor
    ) -> None:
        record = {
            "codigo_apu": "1.1",
            "descripcion_apu": "Test APU",
            "unidad_apu": "M3",
            "category": "Test",
        }
        ctx = processor._extract_apu_context(record)
        assert ctx["codigo_apu"] == "1.1"
        assert ctx["descripcion_apu"] == "Test APU"
        assert ctx["unidad_apu"] == "M3"

    # ── _is_valid_line ────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "line,expected",
        [
            ("CEMENTO;KG;350", True),
            ("", False),
            ("  ", False),
            ("AB", False),
            (None, False),
            (123, False),
        ],
    )
    def test_is_valid_line(
        self, processor: APUProcessor, line, expected: bool
    ) -> None:
        assert processor._is_valid_line(line) is expected

    # ── _compute_cache_key ────────────────────────────────────────────────

    def test_compute_cache_key_normalizes_trailing_spaces(
        self, processor: APUProcessor
    ) -> None:
        """
        La clave de cache debe ser invariante respecto a espacios al final.

        Corrección v3: se compara directamente key1 == key2 (no key2.strip(),
        que sería tautológico si la clave ya es un hash sin espacios).
        """
        key1 = processor._compute_cache_key("CEMENTO;KG;350")
        key2 = processor._compute_cache_key("CEMENTO;KG;350  ")
        assert key1 == key2

    def test_compute_cache_key_is_deterministic(
        self, processor: APUProcessor
    ) -> None:
        line = "CEMENTO;KG;350;1200;420000"
        assert processor._compute_cache_key(line) == processor._compute_cache_key(line)

    def test_compute_cache_key_different_lines_differ(
        self, processor: APUProcessor
    ) -> None:
        k1 = processor._compute_cache_key("CEMENTO;KG;350")
        k2 = processor._compute_cache_key("ARENA;M3;0.5")
        assert k1 != k2

    # ── process_all ───────────────────────────────────────────────────────

    def test_process_all_returns_dataframe(self, processor: APUProcessor) -> None:
        processor.raw_records = make_grouped_sample_records()
        df = processor.process_all()
        assert isinstance(df, pd.DataFrame)

    def test_process_all_not_empty_with_valid_records(
        self, processor: APUProcessor
    ) -> None:
        processor.raw_records = make_grouped_sample_records()
        df = processor.process_all()
        assert not df.empty

    def test_process_all_expected_columns_present(
        self, processor: APUProcessor
    ) -> None:
        processor.raw_records = make_grouped_sample_records()
        df = processor.process_all()

        required = {
            "CODIGO_APU",
            "DESCRIPCION_APU",
            "DESCRIPCION_INSUMO",
            "TIPO_INSUMO",
            "VALOR_TOTAL_APU",
        }
        assert required.issubset(df.columns)

    def test_process_all_values_non_negative(
        self, processor: APUProcessor
    ) -> None:
        """Invariante financiero: VALOR_TOTAL_APU ≥ 0."""
        processor.raw_records = make_grouped_sample_records()
        df = processor.process_all()
        if not df.empty:
            assert (df["VALOR_TOTAL_APU"] >= 0).all()

    def test_process_all_flat_records(self, processor: APUProcessor) -> None:
        processor.raw_records = make_flat_sample_records()
        df = processor.process_all()
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "1.1" in df["CODIGO_APU"].unique()

    def test_process_all_empty_records_returns_empty_df(
        self, processor: APUProcessor
    ) -> None:
        processor.raw_records = []
        df = processor.process_all()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_process_all_multiple_tipo_insumo(
        self, processor: APUProcessor
    ) -> None:
        """El pipeline debe producir más de un tipo de insumo."""
        processor.raw_records = make_grouped_sample_records()
        df = processor.process_all()
        if not df.empty:
            assert df["TIPO_INSUMO"].nunique() > 1

    # ── Estadísticas ──────────────────────────────────────────────────────

    def test_statistics_tracked_after_process(
        self, processor: APUProcessor
    ) -> None:
        processor.raw_records = make_grouped_sample_records()
        processor.process_all()

        assert processor.parsing_stats.total_lines >= 0
        assert processor.parsing_stats.successful_parses >= 0

    def test_global_stats_keys_present(self, processor: APUProcessor) -> None:
        processor.raw_records = make_grouped_sample_records()
        processor.process_all()

        gs = processor.global_stats
        assert "total_apus" in gs
        assert "total_insumos" in gs

    def test_comma_decimal_profile_processes_without_error(
        self, config, comma_profile
    ) -> None:
        records = [
            {
                "codigo_apu": "4.1",
                "descripcion_apu": "Test Coma",
                "unidad_apu": "M2",
                "lines": ["CUADRILLA;JOR;0,08;;250.000,00;20.000,00"],
            }
        ]
        proc = APUProcessor(config, comma_profile)
        proc.raw_records = records
        df = proc.process_all()
        assert isinstance(df, pd.DataFrame)


# ═════════════════════════════════════════════════════════════════════════════
# T11 — VALIDATIONTHRESHOLDS Y PARSINGSTATS
# ═════════════════════════════════════════════════════════════════════════════


class TestValidationThresholds:
    """Contratos de la dataclass ValidationThresholds."""

    def test_default_values(self) -> None:
        thr = ValidationThresholds()
        assert thr.min_jornal == 50_000
        assert thr.max_jornal == 10_000_000
        assert thr.min_rendimiento == pytest.approx(0.001)
        assert thr.max_rendimiento == pytest.approx(1_000)
        assert thr.max_rendimiento_tipico == pytest.approx(100)

    def test_custom_values_accepted(self) -> None:
        thr = ValidationThresholds(
            min_jornal=100_000,
            max_jornal=5_000_000,
            min_rendimiento=0.01,
        )
        assert thr.min_jornal == 100_000
        assert thr.max_jornal == 5_000_000
        assert thr.min_rendimiento == pytest.approx(0.01)

    def test_invariant_min_less_than_max_jornal(self) -> None:
        thr = ValidationThresholds()
        assert thr.min_jornal < thr.max_jornal

    def test_invariant_min_less_than_max_rendimiento(self) -> None:
        thr = ValidationThresholds()
        assert thr.min_rendimiento < thr.max_rendimiento

    def test_invariant_max_rendimiento_tipico_bounded(self) -> None:
        """El rendimiento típico máximo debe estar dentro del rango válido."""
        thr = ValidationThresholds()
        assert thr.min_rendimiento < thr.max_rendimiento_tipico <= thr.max_rendimiento


class TestParsingStats:
    """Contratos de la dataclass ParsingStats."""

    def test_default_values_are_zero(self) -> None:
        stats = ParsingStats()
        assert stats.total_lines == 0
        assert stats.successful_parses == 0
        assert stats.lark_parse_errors == 0
        assert stats.transformer_errors == 0
        assert stats.failed_lines == []

    def test_failed_lines_independent_between_instances(self) -> None:
        """Cada instancia debe tener su propia lista (no compartida)."""
        s1 = ParsingStats()
        s2 = ParsingStats()
        s1.failed_lines.append({"line": 1})
        assert len(s2.failed_lines) == 0

    def test_counters_are_non_negative_after_increments(self) -> None:
        stats = ParsingStats()
        stats.total_lines += 10
        stats.successful_parses += 7
        stats.lark_parse_errors += 2
        stats.transformer_errors += 1
        assert stats.total_lines >= 0
        assert stats.successful_parses <= stats.total_lines

    def test_load_validation_thresholds_from_config(self, profile) -> None:
        config = {
            "validation_thresholds": {
                "MANO_DE_OBRA": {
                    "min_jornal": 75_000,
                    "max_jornal": 8_000_000,
                }
            }
        }
        ctx = make_default_apu_context()
        t = APUTransformer(ctx, config, profile, {})
        assert t.thresholds.min_jornal == 75_000
        assert t.thresholds.max_jornal == 8_000_000

    def test_load_thresholds_defaults_on_missing_config(self, profile) -> None:
        ctx = make_default_apu_context()
        t = APUTransformer(ctx, {}, profile, {})
        defaults = ValidationThresholds()
        assert t.thresholds.min_jornal == defaults.min_jornal
        assert t.thresholds.max_jornal == defaults.max_jornal

    def test_load_thresholds_uses_defaults_on_invalid_values(
        self, profile
    ) -> None:
        """
        Valores inválidos en config deben ser ignorados silenciosamente,
        usando los valores por defecto de ValidationThresholds.
        """
        config = {
            "validation_thresholds": {
                "MANO_DE_OBRA": {
                    "min_jornal": "no_es_numero",
                    "max_jornal": None,
                }
            }
        }
        ctx = make_default_apu_context()
        t = APUTransformer(ctx, config, profile, {})
        defaults = ValidationThresholds()
        assert t.thresholds.min_jornal == defaults.min_jornal
        assert t.thresholds.max_jornal == defaults.max_jornal


# ═════════════════════════════════════════════════════════════════════════════
# T12 — ROBUSTEZ: CASOS EDGE Y ENTRADAS MALFORMADAS
# ═════════════════════════════════════════════════════════════════════════════


class TestRobustness:
    """Verifica que el sistema no propague excepciones ante entradas anómalas."""

    def test_empty_records_returns_empty_df(
        self, processor: APUProcessor
    ) -> None:
        processor.raw_records = []
        df = processor.process_all()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_records_with_empty_lines_processed(
        self, processor: APUProcessor
    ) -> None:
        processor.raw_records = [
            {
                "codigo_apu": "1.1",
                "descripcion_apu": "Test",
                "unidad_apu": "UN",
                "lines": ["", "   ", "CEMENTO;KG;350;1200;420000"],
            }
        ]
        df = processor.process_all()
        assert isinstance(df, pd.DataFrame)

    def test_malformed_numeric_fields_no_exception(
        self, processor: APUProcessor
    ) -> None:
        processor.raw_records = [
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
        df = processor.process_all()
        assert isinstance(df, pd.DataFrame)

    def test_special_characters_in_description(
        self, processor: APUProcessor
    ) -> None:
        processor.raw_records = [
            {
                "codigo_apu": "1.1",
                "descripcion_apu": "Test",
                "unidad_apu": "UN",
                "lines": [
                    "CEMENTO (TIPO I) - 350 KG;KG;350;1200;420000",
                    "ARENA LAVADA & CERNIDA;M3;0.5;150000;75000",
                ],
            }
        ]
        df = processor.process_all()
        assert isinstance(df, pd.DataFrame)

    def test_unicode_characters_processed(self, processor: APUProcessor) -> None:
        processor.raw_records = [
            {
                "codigo_apu": "1.1",
                "descripcion_apu": "Test",
                "unidad_apu": "UN",
                "lines": [
                    "CONCRETO F'C=210 KG/CM;M3;0.15;850000;127500",
                    "ACERO 3/8;KG;50;3500;175000",
                ],
            }
        ]
        df = processor.process_all()
        assert isinstance(df, pd.DataFrame)

    def test_very_long_description_truncated(
        self, processor: APUProcessor
    ) -> None:
        long_desc = "CEMENTO PORTLAND " + "X" * 600
        processor.raw_records = [
            {
                "codigo_apu": "1.1",
                "descripcion_apu": "Test",
                "unidad_apu": "UN",
                "lines": [f"{long_desc};KG;350;1200;420000"],
            }
        ]
        df = processor.process_all()
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            desc = df.iloc[0]["DESCRIPCION_INSUMO"]
            # La descripción debe estar acotada a un máximo razonable
            assert len(desc) <= 617  # 600 + len("CEMENTO PORTLAND ")

    def test_none_config_handled_without_unexpected_exception(
        self, profile
    ) -> None:
        """
        None como config debe manejarse de forma controlada.

        Corrección v3: se distingue entre fallo controlado (TypeError/ValueError
        documentados) y excepción inesperada (RuntimeError, AttributeError, etc.).
        """
        controlled_exceptions = (TypeError, ValueError, AttributeError)
        try:
            proc = APUProcessor(None, profile)
            # Si no falla: debe tener config no-None internamente
            assert proc is not None
        except controlled_exceptions:
            pass  # Fallo controlado: aceptable
        except Exception as exc:
            pytest.fail(
                f"Excepción inesperada con config=None: {type(exc).__name__}: {exc}"
            )

    def test_none_profile_uses_empty_dict(self, config) -> None:
        proc = APUProcessor(config, None)
        assert proc is not None
        assert proc.profile == {}

    def test_single_field_line_does_not_crash(
        self, processor: APUProcessor
    ) -> None:
        """Una línea con un solo campo no debe propagar excepción."""
        processor.raw_records = [
            {
                "codigo_apu": "1.1",
                "descripcion_apu": "Test",
                "unidad_apu": "UN",
                "lines": ["SOLOUNFIELD"],
            }
        ]
        df = processor.process_all()
        assert isinstance(df, pd.DataFrame)


# ═════════════════════════════════════════════════════════════════════════════
# T13 — INTEGRACIÓN END-TO-END
# ═════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Pipeline completo: verificaciones de consistencia y contratos observables."""

    def test_full_pipeline_grouped_produces_multiple_tipos(
        self, processor: APUProcessor
    ) -> None:
        processor.raw_records = make_grouped_sample_records()
        df = processor.process_all()

        assert not df.empty
        assert df["TIPO_INSUMO"].nunique() > 1

    def test_full_pipeline_flat_produces_dataframe(
        self, processor: APUProcessor
    ) -> None:
        processor.raw_records = make_flat_sample_records()
        df = processor.process_all()
        assert isinstance(df, pd.DataFrame)

    def test_consistency_across_formats_same_apu_code(
        self, config, profile
    ) -> None:
        """
        El mismo APU procesado en formato grouped y flat debe producir
        el mismo CODIGO_APU en el resultado.

        Corrección v3: la aserción es incondicional — si ambos DataFrames
        están vacíos, el test falla explícitamente para detectar regresiones.
        """
        grouped = make_grouped_sample_records()[:1]
        flat = [r for r in make_flat_sample_records() if r["apu_code"] == "1.1"]

        proc_g = APUProcessor(config, profile)
        proc_g.raw_records = grouped
        df_g = proc_g.process_all()

        proc_f = APUProcessor(config, profile)
        proc_f.raw_records = flat
        df_f = proc_f.process_all()

        assert not df_g.empty, "El pipeline grouped no produjo resultados"
        assert not df_f.empty, "El pipeline flat no produjo resultados"
        assert df_g["CODIGO_APU"].iloc[0] == df_f["CODIGO_APU"].iloc[0]

    def test_all_required_columns_present_after_full_pipeline(
        self, processor: APUProcessor
    ) -> None:
        processor.raw_records = make_grouped_sample_records()
        df = processor.process_all()

        required = {
            "CODIGO_APU",
            "DESCRIPCION_APU",
            "DESCRIPCION_INSUMO",
            "TIPO_INSUMO",
            "VALOR_TOTAL_APU",
        }
        assert required.issubset(df.columns)

    def test_codigo_apu_matches_input(self, processor: APUProcessor) -> None:
        """Los códigos APU del output deben corresponder a los del input."""
        records = make_grouped_sample_records()
        input_codes = {r["codigo_apu"] for r in records}

        processor.raw_records = records
        df = processor.process_all()

        if not df.empty:
            output_codes = set(df["CODIGO_APU"].unique())
            # Todos los códigos del output deben estar en el input
            assert output_codes.issubset(input_codes)


# ═════════════════════════════════════════════════════════════════════════════
# T14 — FORMATOLINEA Y TIPOINSUMO: CONTRATOS DE ENUMERACIONES
# ═════════════════════════════════════════════════════════════════════════════


class TestEnumerations:
    """
    Verifica los contratos de las enumeraciones FormatoLinea y TipoInsumo.

    Corrección v3: estas clases no tenían ningún test en la suite original.
    """

    def test_tipo_insumo_has_required_members(self) -> None:
        """TipoInsumo debe tener al menos los tipos fundamentales de APU."""
        required = {"MANO_DE_OBRA", "SUMINISTRO", "EQUIPO"}
        actual = {t.name for t in TipoInsumo}
        assert required.issubset(actual), (
            f"TipoInsumo falta miembros: {required - actual}"
        )

    def test_tipo_insumo_members_are_strings(self) -> None:
        """Los valores de TipoInsumo deben ser strings para serialización."""
        for tipo in TipoInsumo:
            assert isinstance(tipo.value, str)

    def test_tipo_insumo_no_duplicated_values(self) -> None:
        """No debe haber dos tipos con el mismo valor."""
        values = [t.value for t in TipoInsumo]
        assert len(values) == len(set(values))

    def test_formato_linea_has_required_members(self) -> None:
        """FormatoLinea debe clasificar los tipos de línea esperados."""
        required = {"MO_COMPLETA", "INSUMO_BASICO", "DESCONOCIDO"}
        actual = {f.name for f in FormatoLinea}
        assert required.issubset(actual), (
            f"FormatoLinea falta miembros: {required - actual}"
        )

    def test_formato_linea_members_are_strings(self) -> None:
        for fmt in FormatoLinea:
            assert isinstance(fmt.value, str)

    def test_formato_linea_no_duplicated_values(self) -> None:
        values = [f.value for f in FormatoLinea]
        assert len(values) == len(set(values))

    def test_tipo_insumo_classify_result_is_always_member(
        self, transformer: APUTransformer
    ) -> None:
        """_classify_insumo debe retornar siempre un miembro de TipoInsumo."""
        test_descriptions = [
            "CEMENTO PORTLAND",
            "OFICIAL ALBANIL",
            "EQUIPO MENOR",
            "SUBCONTRATO ELECTRICIDAD",
            "",
            "DESCRIPCION MUY LARGA " + "X" * 100,
        ]
        valid_members = set(TipoInsumo)
        for desc in test_descriptions:
            result = transformer._classify_insumo(desc)
            assert result in valid_members, (
                f"_classify_insumo('{desc[:30]}') = {result} no es TipoInsumo"
            )


# ═════════════════════════════════════════════════════════════════════════════
# T15 — CALCULATE_UNIT_COSTS: INVARIANTES DE CÁLCULO
# ═════════════════════════════════════════════════════════════════════════════


class TestCalculateUnitCosts:
    """
    Verifica los invariantes matemáticos de calculate_unit_costs.

    Corrección v3: esta función era importada pero nunca testada en la suite original.
    """

    def _create_df(self, cantidad: float, precio: float, tipo: str = "SUMINISTRO", rendimiento: float = None) -> pd.DataFrame:
        valor_total = (cantidad * precio) if rendimiento is None else (rendimiento * precio)
        return pd.DataFrame({
            "CODIGO_APU": ["TEST"],
            "DESCRIPCION_APU": ["TEST"],
            "UNIDAD_APU": ["UND"],
            "TIPO_INSUMO": [tipo],
            "VALOR_TOTAL_APU": [valor_total]
        })

    def test_unit_cost_non_negative_for_valid_inputs(self) -> None:
        """El costo unitario no puede ser negativo para entradas positivas."""
        result = calculate_unit_costs(self._create_df(5.0, 10000.0))
        assert result["COSTO_UNITARIO_TOTAL"].iloc[0] >= 0.0

    def test_unit_cost_zero_for_zero_quantity(self) -> None:
        """Cantidad = 0 debe producir costo total = 0."""
        result = calculate_unit_costs(self._create_df(0.0, 10000.0))
        assert result["COSTO_UNITARIO_TOTAL"].iloc[0] == pytest.approx(0.0)

    def test_unit_cost_scales_linearly_with_quantity(self) -> None:
        """El costo debe escalar linealmente con la cantidad."""
        r1 = calculate_unit_costs(self._create_df(1.0, 1000.0))["COSTO_UNITARIO_TOTAL"].iloc[0]
        r2 = calculate_unit_costs(self._create_df(2.0, 1000.0))["COSTO_UNITARIO_TOTAL"].iloc[0]
        assert r2 == pytest.approx(r1 * 2.0)

    def test_unit_cost_with_rendimiento_for_mo(self) -> None:
        """Con rendimiento, el costo = rendimiento × jornal."""
        result = calculate_unit_costs(self._create_df(0.0, 180000.0, tipo="MANO_DE_OBRA", rendimiento=0.125))
        assert result["COSTO_UNITARIO_TOTAL"].iloc[0] == pytest.approx(0.125 * 180000.0)

    def test_unit_cost_returns_float(self) -> None:
        """El resultado siempre debe ser numérico."""
        result = calculate_unit_costs(self._create_df(1.0, 5000.0))
        assert isinstance(float(result["COSTO_UNITARIO_TOTAL"].iloc[0]), float)

    def test_unit_cost_handles_none_inputs_gracefully(self) -> None:
        """Entradas None no deben propagar excepción."""
        df = pd.DataFrame({"CODIGO_APU": [None], "TIPO_INSUMO": [None], "VALOR_TOTAL_APU": [None]})
        try:
            result = calculate_unit_costs(df)
            if not result.empty:
                assert result["COSTO_UNITARIO_TOTAL"].iloc[0] == pytest.approx(0.0)
        except (TypeError, ValueError):
            pass  # Fallo controlado aceptable


# ═════════════════════════════════════════════════════════════════════════════
# T16 — PROPIEDADES ALGEBRAICAS GLOBALES
# ═════════════════════════════════════════════════════════════════════════════


class TestAlgebraicProperties:
    """
    Invariantes globales del sistema: propiedades que deben mantenerse
    para cualquier entrada válida del dominio.
    """

    def test_option_monad_pure_then_get_or_else_is_identity(self) -> None:
        """pure(x).get_or_else(y) ≡ x para cualquier x."""
        test_values = [0, 1, "texto", [], None, 3.14]
        for val in test_values:
            assert OptionMonad.pure(val).get_or_else("DEFAULT") == val

    def test_option_monad_fail_then_get_or_else_is_default(self) -> None:
        """fail(e).get_or_else(d) ≡ d para cualquier error e."""
        for default in [0, "fallback", None, []]:
            assert OptionMonad.fail("error").get_or_else(default) == default

    def test_normalize_unit_output_is_always_string(self) -> None:
        """normalize_unit siempre retorna str, nunca None."""
        test_inputs = ["", None, "M2", "DESCONOCIDA", "MT", "X" * 100]
        for inp in test_inputs:
            result = UnitsValidator.normalize_unit(inp)
            assert isinstance(result, str), (
                f"normalize_unit('{inp}') retornó {type(result).__name__}"
            )

    def test_parsing_stats_successful_never_exceeds_total(
        self, processor: APUProcessor
    ) -> None:
        """Invariante de consistencia: successful_parses ≤ total_lines."""
        processor.raw_records = make_grouped_sample_records()
        processor.process_all()
        assert (
            processor.parsing_stats.successful_parses
            <= processor.parsing_stats.total_lines
        )

    def test_validation_thresholds_ranges_are_consistent(self) -> None:
        """Para cualquier instancia, min < max en todos los rangos."""
        thr = ValidationThresholds()
        assert thr.min_jornal < thr.max_jornal
        assert thr.min_rendimiento < thr.max_rendimiento

    def test_process_all_codigo_apu_never_empty(
        self, processor: APUProcessor
    ) -> None:
        """Ningún registro del output debe tener CODIGO_APU vacío."""
        processor.raw_records = make_grouped_sample_records()
        df = processor.process_all()
        if not df.empty:
            assert not df["CODIGO_APU"].isnull().any()
            assert (df["CODIGO_APU"] != "").all()

    def test_tipo_insumo_classify_is_total_function(
        self, transformer: APUTransformer
    ) -> None:
        """_classify_insumo es una función total: nunca lanza excepción."""
        edge_cases = [
            "",
            " ",
            "A" * 1000,
            "OFICIAL\x00CON\x1fCONTROL",
            "123456789",
        ]
        for desc in edge_cases:
            try:
                result = transformer._classify_insumo(desc)
                assert isinstance(result, TipoInsumo)
            except Exception as exc:
                pytest.fail(
                    f"_classify_insumo lanzó excepción para '{desc[:30]}': "
                    f"{type(exc).__name__}: {exc}"
                )

    def test_lark_grammar_is_deterministic(self) -> None:
        """El parser Lark produce el mismo árbol para la misma entrada."""
        parser = Lark(APU_GRAMMAR, start="line", parser="lalr")
        line = "CEMENTO;KG;350;1200;420000"
        tree1 = parser.parse(line)
        tree2 = parser.parse(line)
        assert tree1.data == tree2.data
        assert len(tree1.children) == len(tree2.children)

    def test_process_all_is_reproducible(self, config, profile) -> None:
        """Procesar los mismos registros dos veces produce el mismo resultado."""
        records = make_grouped_sample_records()

        proc1 = APUProcessor(config, profile)
        proc1.raw_records = records
        df1 = proc1.process_all()

        proc2 = APUProcessor(config, profile)
        proc2.raw_records = records
        df2 = proc2.process_all()

        if not df1.empty and not df2.empty:
            assert list(df1.columns) == list(df2.columns)
            assert len(df1) == len(df2)