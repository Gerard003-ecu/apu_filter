"""
Suite de pruebas exhaustiva para el módulo de Constitución de Datos.

Organización por capas:
    1. Funciones de normalización (puras)
    2. Validadores numéricos y de cadenas
    3. Enumeraciones y constantes
    4. InsumoProcesado y subclases (ManoDeObra, Equipo, Suministro, etc.)
    5. APUStructure y métricas topológicas
    6. Factory functions y pipeline completo
    7. Casos límite y regresiones

Convenciones:
    - Cada test tiene un nombre descriptivo en español.
    - Los fixtures proveen datos base reutilizables.
    - Se usan marcadores pytest para categorizar tests.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict
from unittest.mock import patch

import pytest

# ============================================================================
# IMPORTACIONES DEL MÓDULO BAJO PRUEBA
# ============================================================================

from schema import (
    APUStructure,
    Equipo,
    INSUMO_CLASS_MAP,
    InsumoDataError,
    InsumoProcesado,
    InvalidTipoInsumoError,
    ManoDeObra,
    MAX_CODIGO_LENGTH,
    MAX_DESCRIPCION_LENGTH,
    MIN_CANTIDAD,
    MAX_CANTIDAD,
    MIN_PRECIO,
    MAX_PRECIO,
    MIN_RENDIMIENTO,
    MAX_RENDIMIENTO,
    NumericValidator,
    Otro,
    SchemaError,
    Stratum,
    StringValidator,
    Suministro,
    TipoInsumo,
    Transporte,
    UNIDADES_TIEMPO,
    UNIDADES_MASA,
    UNIDADES_GENERICAS,
    UnitNormalizationError,
    VALOR_TOTAL_ERROR_TOLERANCE,
    VALOR_TOTAL_WARNING_TOLERANCE,
    ValidationError,
    create_insumo,
    create_insumo_from_raw,
    get_all_tipo_insumo_values,
    get_tipo_insumo_class,
    is_valid_tipo_insumo,
    normalize_codigo,
    normalize_description,
    normalize_unit,
    validate_insumo_data,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def datos_insumo_base() -> Dict[str, Any]:
    """Diccionario base válido para crear cualquier insumo."""
    return {
        "codigo_apu": "APU-001",
        "descripcion_apu": "Excavación manual",
        "unidad_apu": "M3",
        "descripcion_insumo": "Obrero raso",
        "unidad_insumo": "HORA",
        "cantidad": 2.0,
        "precio_unitario": 15000.0,
        "valor_total": 30000.0,
        "tipo_insumo": "MANO_DE_OBRA",
        "rendimiento": 0.5,
    }


@pytest.fixture
def datos_equipo() -> Dict[str, Any]:
    """Datos válidos para un insumo tipo Equipo."""
    return {
        "codigo_apu": "APU-002",
        "descripcion_apu": "Compactación mecánica",
        "unidad_apu": "M2",
        "descripcion_insumo": "Vibrocompactador",
        "unidad_insumo": "HORA",
        "cantidad": 0.5,
        "precio_unitario": 80000.0,
        "valor_total": 40000.0,
        "tipo_insumo": "EQUIPO",
    }


@pytest.fixture
def datos_suministro() -> Dict[str, Any]:
    """Datos válidos para un insumo tipo Suministro."""
    return {
        "codigo_apu": "APU-003",
        "descripcion_apu": "Concreto 3000 PSI",
        "unidad_apu": "M3",
        "descripcion_insumo": "Cemento Portland Tipo I",
        "unidad_insumo": "KG",
        "cantidad": 350.0,
        "precio_unitario": 450.0,
        "valor_total": 157500.0,
        "tipo_insumo": "SUMINISTRO",
    }


@pytest.fixture
def datos_transporte() -> Dict[str, Any]:
    """Datos válidos para un insumo tipo Transporte."""
    return {
        "codigo_apu": "APU-004",
        "descripcion_apu": "Transporte de materiales",
        "unidad_apu": "M3",
        "descripcion_insumo": "Volqueta 10m3",
        "unidad_insumo": "VIAJE",
        "cantidad": 3.0,
        "precio_unitario": 120000.0,
        "valor_total": 360000.0,
        "tipo_insumo": "TRANSPORTE",
    }


@pytest.fixture
def datos_otro() -> Dict[str, Any]:
    """Datos válidos para un insumo tipo Otro."""
    return {
        "codigo_apu": "APU-005",
        "descripcion_apu": "Administración",
        "unidad_apu": "UNIDAD",
        "descripcion_insumo": "Costos indirectos",
        "unidad_insumo": "%",
        "cantidad": 1.0,
        "precio_unitario": 50000.0,
        "valor_total": 50000.0,
        "tipo_insumo": "OTRO",
    }


@pytest.fixture
def insumo_mano_obra(datos_insumo_base) -> ManoDeObra:
    """Instancia válida de ManoDeObra."""
    return create_insumo(**datos_insumo_base)


@pytest.fixture
def insumo_equipo(datos_equipo) -> Equipo:
    """Instancia válida de Equipo."""
    return create_insumo(**datos_equipo)


@pytest.fixture
def insumo_suministro(datos_suministro) -> Suministro:
    """Instancia válida de Suministro."""
    return create_insumo(**datos_suministro)


@pytest.fixture
def insumo_transporte(datos_transporte) -> Transporte:
    """Instancia válida de Transporte."""
    return create_insumo(**datos_transporte)


@pytest.fixture
def insumo_otro(datos_otro) -> Otro:
    """Instancia válida de Otro."""
    return create_insumo(**datos_otro)


@pytest.fixture
def apu_completo(
    insumo_mano_obra, insumo_equipo, insumo_suministro
) -> APUStructure:
    """APU con tres tipos de recursos distintos."""
    apu = APUStructure(
        id="APU-COMP-001",
        description="APU completo de prueba",
        unit="M3",
        quantity=10.0,
    )
    apu.add_resource(insumo_mano_obra)
    apu.add_resource(insumo_equipo)
    apu.add_resource(insumo_suministro)
    return apu


# ============================================================================
# 1. FUNCIONES DE NORMALIZACIÓN
# ============================================================================

class TestNormalizeUnit:
    """Pruebas para normalize_unit."""

    @pytest.mark.parametrize("entrada,esperado", [
        ("hora", "HORA"),
        ("Hr", "HORA"),
        ("HRS", "HORA"),
        ("h", "HORA"),
        ("kg", "KG"),
        ("kilogramo", "KG"),
        ("Kilogramos", "KG"),
        ("m3", "M3"),
        ("m³", "M3"),
        ("M2", "M2"),
        ("m²", "M2"),
        ("metro", "M"),
        ("metros", "M"),
        ("Mts", "M"),
        ("litro", "L"),
        ("lt", "L"),
        ("und", "UNIDAD"),
        ("unidad", "UNIDAD"),
        ("viaje", "VIAJE"),
        ("ton-km", "TON-KM"),
        ("jornal", "JOR"),
    ])
    def test_normalizacion_conocida(self, entrada: str, esperado: str):
        """Unidades conocidas se mapean correctamente."""
        assert normalize_unit(entrada) == esperado

    def test_unidad_desconocida_pasa_a_mayusculas(self):
        """Unidades no mapeadas se convierten a mayúsculas."""
        assert normalize_unit("barril") == "BARRIL"

    @pytest.mark.parametrize("entrada", [None, "", "   "])
    def test_entrada_vacia_retorna_unidad(self, entrada):
        """Entradas vacías o nulas retornan 'UNIDAD'."""
        assert normalize_unit(entrada) == "UNIDAD"

    def test_idempotencia(self):
        """normalize_unit(normalize_unit(x)) == normalize_unit(x)."""
        for entrada in ["hora", "kg", "m3", "und", "UNIDAD", "HORA"]:
            primera = normalize_unit(entrada)
            segunda = normalize_unit(primera)
            assert primera == segunda, f"No idempotente para '{entrada}'"

    def test_con_espacios(self):
        """Se eliminan espacios antes de buscar."""
        assert normalize_unit("  kg  ") == "KG"

    def test_tipo_no_string(self):
        """Entrada no-string retorna 'UNIDAD'."""
        assert normalize_unit(123) == "UNIDAD"  # type: ignore[arg-type]


class TestNormalizeDescription:
    """Pruebas para normalize_description."""

    def test_elimina_acentos(self):
        """Los diacríticos se eliminan."""
        resultado = normalize_description("Válvula de presión")
        assert "A" in resultado  # á → A
        assert "O" in resultado  # ó → O
        assert resultado == "VALVULA DE PRESION"

    def test_elimina_caracteres_especiales(self):
        """Caracteres no alfanuméricos prohibidos se eliminan."""
        resultado = normalize_description("Tubo @#$ PVC 1/2\"")
        assert "@" not in resultado
        assert "#" not in resultado
        assert "$" not in resultado

    def test_mantiene_caracteres_permitidos(self):
        """Guiones, puntos, paréntesis y barras se preservan."""
        resultado = normalize_description("Tubo PVC (1/2) - Tipo A")
        assert "/" in resultado
        assert "(" in resultado
        assert ")" in resultado
        assert "-" in resultado

    def test_colapsa_espacios(self):
        """Múltiples espacios se colapsan a uno."""
        resultado = normalize_description("  Cemento    Portland   ")
        assert "  " not in resultado
        assert resultado == "CEMENTO PORTLAND"

    def test_truncamiento(self):
        """Descripciones largas se truncan a MAX_DESCRIPCION_LENGTH."""
        largo = "A" * (MAX_DESCRIPCION_LENGTH + 100)
        resultado = normalize_description(largo)
        assert len(resultado) == MAX_DESCRIPCION_LENGTH

    @pytest.mark.parametrize("entrada", [None, "", 0])
    def test_entrada_invalida_retorna_vacio(self, entrada):
        """Entradas nulas, vacías o no-string retornan cadena vacía."""
        assert normalize_description(entrada) == ""

    def test_idempotencia(self):
        """Aplicar dos veces produce el mismo resultado."""
        texto = "Mézcladora de concreto 350 lt"
        assert normalize_description(texto) == normalize_description(
            normalize_description(texto)
        )


class TestNormalizeCodigo:
    """Pruebas para normalize_codigo."""

    def test_codigo_valido(self):
        """Códigos alfanuméricos con guiones y puntos son válidos."""
        assert normalize_codigo("APU-001.A") == "APU-001.A"

    def test_convierte_a_mayusculas(self):
        """El código se convierte a mayúsculas."""
        assert normalize_codigo("apu-001") == "APU-001"

    def test_elimina_caracteres_invalidos(self):
        """Espacios y caracteres especiales se eliminan."""
        assert normalize_codigo("APU 001 @#") == "APU001"

    def test_codigo_vacio_lanza_error(self):
        """Código vacío lanza ValidationError."""
        with pytest.raises(ValidationError):
            normalize_codigo("")

    def test_codigo_none_lanza_error(self):
        """Código None lanza ValidationError."""
        with pytest.raises(ValidationError):
            normalize_codigo(None)

    def test_codigo_solo_caracteres_invalidos(self):
        """Código que queda vacío tras limpieza lanza ValidationError."""
        with pytest.raises(ValidationError, match="vacío tras normalización"):
            normalize_codigo("@#$%")

    def test_codigo_excede_longitud(self):
        """Código que excede MAX_CODIGO_LENGTH lanza ValidationError."""
        largo = "A" * (MAX_CODIGO_LENGTH + 1)
        with pytest.raises(ValidationError, match="excede límite"):
            normalize_codigo(largo)

    def test_idempotencia(self):
        """Doble normalización es idempotente."""
        codigo = "apu-test.123"
        assert normalize_codigo(codigo) == normalize_codigo(
            normalize_codigo(codigo)
        )


# ============================================================================
# 2. VALIDADORES
# ============================================================================

class TestNumericValidator:
    """Pruebas para NumericValidator."""

    # --- validate_non_negative ---

    def test_valor_valido(self):
        result = NumericValidator.validate_non_negative(10.5, "test")
        assert result == 10.5

    def test_valor_cero(self):
        result = NumericValidator.validate_non_negative(0.0, "test")
        assert result == 0.0

    def test_valor_decimal(self):
        from decimal import Decimal
        result = NumericValidator.validate_non_negative(Decimal("3.14"), "test")
        assert abs(result - 3.14) < 1e-10

    def test_valor_entero(self):
        result = NumericValidator.validate_non_negative(42, "test")
        assert result == 42.0

    def test_valor_negativo_lanza_error(self):
        with pytest.raises(ValidationError, match="mínimo"):
            NumericValidator.validate_non_negative(-1.0, "test")

    def test_valor_infinito_lanza_error(self):
        with pytest.raises(ValidationError, match="finito"):
            NumericValidator.validate_non_negative(float("inf"), "test")

    def test_valor_nan_lanza_error(self):
        with pytest.raises(ValidationError, match="finito"):
            NumericValidator.validate_non_negative(float("nan"), "test")

    def test_tipo_invalido_lanza_error(self):
        with pytest.raises(ValidationError, match="numérico"):
            NumericValidator.validate_non_negative("abc", "test")  # type: ignore

    def test_rango_maximo(self):
        with pytest.raises(ValidationError, match="máximo"):
            NumericValidator.validate_non_negative(1000.0, "test", max_value=100.0)

    def test_rango_minimo_custom(self):
        with pytest.raises(ValidationError, match="mínimo"):
            NumericValidator.validate_non_negative(0.5, "test", min_value=1.0)

    def test_valor_en_limite_superior(self):
        """El límite superior es inclusivo."""
        result = NumericValidator.validate_non_negative(100.0, "test", max_value=100.0)
        assert result == 100.0

    # --- relative_difference ---

    def test_diferencia_cero(self):
        assert NumericValidator.relative_difference(10.0, 10.0) == 0.0

    def test_diferencia_simetrica(self):
        """La diferencia relativa es simétrica."""
        d1 = NumericValidator.relative_difference(100.0, 105.0)
        d2 = NumericValidator.relative_difference(105.0, 100.0)
        assert abs(d1 - d2) < 1e-10

    def test_diferencia_valores_cercanos_a_cero(self):
        """Estabilidad numérica cerca de cero."""
        result = NumericValidator.relative_difference(1e-15, 2e-15)
        assert math.isfinite(result)

    def test_diferencia_conocida(self):
        """5% de diferencia → Δ ≈ 0.05."""
        result = NumericValidator.relative_difference(100.0, 105.0)
        assert abs(result - 5.0 / 105.0) < 1e-10

    # --- values_consistent ---

    def test_consistente_exacto(self):
        ok, diff = NumericValidator.values_consistent(100.0, 100.0)
        assert ok is True
        assert diff == 0.0

    def test_consistente_dentro_tolerancia_absoluta(self):
        ok, diff = NumericValidator.values_consistent(100.0, 100.005, abs_tolerance=0.01)
        assert ok is True

    def test_inconsistente_fuera_tolerancia(self):
        ok, diff = NumericValidator.values_consistent(
            100.0, 200.0, rel_tolerance=0.01, abs_tolerance=0.01
        )
        assert ok is False
        assert diff > 0.01

    def test_consistente_dentro_tolerancia_relativa(self):
        ok, diff = NumericValidator.values_consistent(
            100.0, 100.5, rel_tolerance=0.01, abs_tolerance=0.001
        )
        assert ok is True


class TestStringValidator:
    """Pruebas para StringValidator."""

    def test_cadena_valida(self):
        result = StringValidator.validate_non_empty("Hola", "test")
        assert result == "Hola"

    def test_cadena_con_espacios_se_limpia(self):
        result = StringValidator.validate_non_empty("  Hola  ", "test")
        assert result == "Hola"

    def test_cadena_vacia_lanza_error(self):
        with pytest.raises(ValidationError, match="vacío"):
            StringValidator.validate_non_empty("", "test")

    def test_cadena_solo_espacios_lanza_error(self):
        with pytest.raises(ValidationError, match="vacío"):
            StringValidator.validate_non_empty("   ", "test")

    def test_tipo_no_string_lanza_error(self):
        with pytest.raises(ValidationError, match="str"):
            StringValidator.validate_non_empty(123, "test")  # type: ignore

    def test_longitud_maxima_excedida(self):
        with pytest.raises(ValidationError, match="longitud máxima"):
            StringValidator.validate_non_empty("A" * 100, "test", max_length=10)

    def test_longitud_maxima_exacta(self):
        """Longitud exacta al máximo es válida."""
        result = StringValidator.validate_non_empty("A" * 10, "test", max_length=10)
        assert len(result) == 10

    def test_sin_strip(self):
        """Con strip=False se preservan espacios."""
        result = StringValidator.validate_non_empty(
            "  Hola  ", "test", strip=False
        )
        assert result == "  Hola  "


# ============================================================================
# 3. ENUMERACIONES
# ============================================================================

class TestTipoInsumo:
    """Pruebas para la enumeración TipoInsumo."""

    @pytest.mark.parametrize("entrada,esperado", [
        ("MANO_DE_OBRA", TipoInsumo.MANO_DE_OBRA),
        ("EQUIPO", TipoInsumo.EQUIPO),
        ("SUMINISTRO", TipoInsumo.SUMINISTRO),
        ("TRANSPORTE", TipoInsumo.TRANSPORTE),
        ("OTRO", TipoInsumo.OTRO),
    ])
    def test_conversion_directa(self, entrada: str, esperado: TipoInsumo):
        assert TipoInsumo.from_string(entrada) == esperado

    @pytest.mark.parametrize("entrada", [
        "mano de obra",
        "Mano De Obra",
        "MANO DE OBRA",
        "mano-de-obra",
        "MANO-DE-OBRA",
    ])
    def test_normalizacion_variantes(self, entrada: str):
        """Distintas formas textuales convergen al mismo enum."""
        assert TipoInsumo.from_string(entrada) == TipoInsumo.MANO_DE_OBRA

    def test_identidad_enum(self):
        """Pasar un enum devuelve el mismo enum."""
        original = TipoInsumo.EQUIPO
        assert TipoInsumo.from_string(original) is original

    def test_tipo_invalido_lanza_error(self):
        with pytest.raises(InvalidTipoInsumoError, match="inválido"):
            TipoInsumo.from_string("INEXISTENTE")

    def test_tipo_no_string_lanza_error(self):
        with pytest.raises(InvalidTipoInsumoError, match="str"):
            TipoInsumo.from_string(42)  # type: ignore

    def test_valid_values(self):
        """valid_values() retorna todos los valores como frozenset."""
        valores = TipoInsumo.valid_values()
        assert isinstance(valores, frozenset)
        assert len(valores) == 5
        assert "MANO_DE_OBRA" in valores
        assert "EQUIPO" in valores

    def test_cache_funciona(self):
        """Llamadas repetidas usan la caché."""
        r1 = TipoInsumo.from_string("equipo")
        r2 = TipoInsumo.from_string("equipo")
        assert r1 is r2

    def test_con_espacios_extra(self):
        """Espacios al inicio/fin se eliminan."""
        assert TipoInsumo.from_string("  EQUIPO  ") == TipoInsumo.EQUIPO


class TestStratum:
    """Pruebas para la enumeración Stratum."""

    def test_orden_jerarquico(self):
        """Los niveles mantienen orden numérico."""
        assert Stratum.WISDOM < Stratum.STRATEGY < Stratum.TACTICS < Stratum.PHYSICS

    def test_valores_numericos(self):
        assert Stratum.WISDOM == 0
        assert Stratum.PHYSICS == 3

    def test_es_int_enum(self):
        """Stratum es comparable como entero."""
        assert Stratum.TACTICS + 1 == Stratum.PHYSICS


# ============================================================================
# 4. INSUMO PROCESADO Y SUBCLASES
# ============================================================================

class TestInsumoProcesado:
    """Pruebas para InsumoProcesado (clase base)."""

    def test_creacion_valida(self, datos_insumo_base):
        """Un insumo con datos válidos se crea sin errores."""
        insumo = create_insumo(**datos_insumo_base)
        assert insumo.is_valid is True

    def test_stratum_es_physics(self, insumo_mano_obra):
        """El stratum de un insumo siempre es PHYSICS."""
        assert insumo_mano_obra.stratum == Stratum.PHYSICS

    def test_id_determinista(self, datos_insumo_base):
        """El mismo dato produce el mismo ID."""
        i1 = create_insumo(**datos_insumo_base)
        i2 = create_insumo(**datos_insumo_base)
        assert i1.id == i2.id

    def test_id_diferente_para_datos_diferentes(self, datos_insumo_base):
        """Datos distintos producen IDs distintos."""
        i1 = create_insumo(**datos_insumo_base)
        datos_insumo_base["descripcion_insumo"] = "Otro recurso"
        i2 = create_insumo(**datos_insumo_base)
        assert i1.id != i2.id

    def test_total_cost_calculado(self, insumo_mano_obra):
        """total_cost = cantidad × precio_unitario."""
        expected = insumo_mano_obra.cantidad * insumo_mano_obra.precio_unitario
        assert abs(insumo_mano_obra.total_cost - expected) < 1e-10

    def test_normalizacion_aplicada(self, datos_insumo_base):
        """Los campos de texto se normalizan."""
        datos_insumo_base["descripcion_insumo"] = "  obrero raso  "
        datos_insumo_base["unidad_insumo"] = "hr"
        insumo = create_insumo(**datos_insumo_base)
        assert insumo.descripcion_insumo == "OBRERO RASO"
        assert insumo.unidad_insumo == "HORA"

    def test_to_dict_excluye_campos_internos(self, insumo_mano_obra):
        """to_dict() no incluye campos con prefijo '_'."""
        d = insumo_mano_obra.to_dict()
        assert "_validated" not in d
        assert "codigo_apu" in d
        assert "tipo_insumo" in d

    def test_formato_origen_invalido_se_normaliza(self, datos_insumo_base):
        """Formatos desconocidos se convierten a 'GENERIC'."""
        datos_insumo_base["formato_origen"] = "FORMATO_RARO"
        insumo = create_insumo(**datos_insumo_base)
        assert insumo.formato_origen == "GENERIC"

    def test_formato_origen_valido(self, datos_insumo_base):
        """Formatos conocidos se preservan."""
        datos_insumo_base["formato_origen"] = "FORMATO_A"
        insumo = create_insumo(**datos_insumo_base)
        assert insumo.formato_origen == "FORMATO_A"


class TestValidacionesInsumo:
    """Pruebas de validación de invariantes en InsumoProcesado."""

    def test_codigo_vacio_lanza_error(self, datos_insumo_base):
        """Código vacío causa ValidationError."""
        datos_insumo_base["codigo_apu"] = ""
        with pytest.raises(ValidationError):
            create_insumo(**datos_insumo_base)

    def test_descripcion_vacia_lanza_error(self, datos_insumo_base):
        """Descripción de insumo vacía causa ValidationError."""
        datos_insumo_base["descripcion_insumo"] = ""
        with pytest.raises(ValidationError):
            create_insumo(**datos_insumo_base)

    def test_cantidad_negativa_lanza_error(self, datos_insumo_base):
        """Cantidad negativa causa ValidationError."""
        datos_insumo_base["cantidad"] = -1.0
        datos_insumo_base["valor_total"] = -15000.0
        with pytest.raises(ValidationError):
            create_insumo(**datos_insumo_base)

    def test_precio_negativo_lanza_error(self, datos_insumo_base):
        """Precio negativo causa ValidationError."""
        datos_insumo_base["precio_unitario"] = -100.0
        datos_insumo_base["valor_total"] = -200.0
        with pytest.raises(ValidationError):
            create_insumo(**datos_insumo_base)

    def test_cantidad_infinita_lanza_error(self, datos_insumo_base):
        """Cantidad infinita causa ValidationError."""
        datos_insumo_base["cantidad"] = float("inf")
        with pytest.raises(ValidationError, match="finito"):
            create_insumo(**datos_insumo_base)

    def test_cantidad_nan_lanza_error(self, datos_insumo_base):
        """Cantidad NaN causa ValidationError."""
        datos_insumo_base["cantidad"] = float("nan")
        with pytest.raises(ValidationError, match="finito"):
            create_insumo(**datos_insumo_base)

    def test_cantidad_excede_maximo(self, datos_insumo_base):
        """Cantidad superior al máximo causa ValidationError."""
        datos_insumo_base["cantidad"] = MAX_CANTIDAD + 1
        with pytest.raises(ValidationError, match="máximo"):
            create_insumo(**datos_insumo_base)

    def test_precio_excede_maximo(self, datos_insumo_base):
        """Precio superior al máximo causa ValidationError."""
        datos_insumo_base["precio_unitario"] = MAX_PRECIO + 1
        with pytest.raises(ValidationError):
            create_insumo(**datos_insumo_base)

    def test_tipo_insumo_invalido(self, datos_insumo_base):
        """Tipo de insumo desconocido lanza InvalidTipoInsumoError."""
        datos_insumo_base["tipo_insumo"] = "ALIEN"
        with pytest.raises(InvalidTipoInsumoError):
            create_insumo(**datos_insumo_base)


class TestConsistenciaValorTotal:
    """Pruebas para la validación valor_total ≈ cantidad × precio."""

    def test_consistencia_exacta(self, datos_insumo_base):
        """Valores exactamente consistentes no generan error."""
        datos_insumo_base["cantidad"] = 10.0
        datos_insumo_base["precio_unitario"] = 100.0
        datos_insumo_base["valor_total"] = 1000.0
        insumo = create_insumo(**datos_insumo_base)
        assert insumo.is_valid

    def test_inconsistencia_grave_lanza_error(self, datos_insumo_base):
        """Diferencia > 5% lanza ValidationError."""
        datos_insumo_base["cantidad"] = 10.0
        datos_insumo_base["precio_unitario"] = 100.0
        datos_insumo_base["valor_total"] = 2000.0  # 100% de diferencia
        with pytest.raises(ValidationError, match="Inconsistencia grave"):
            create_insumo(**datos_insumo_base)

    def test_divergencia_leve_emite_warning(self, datos_insumo_base):
        """Diferencia entre 1% y 5% emite UserWarning."""
        datos_insumo_base["cantidad"] = 10.0
        datos_insumo_base["precio_unitario"] = 100.0
        # 2% de diferencia
        datos_insumo_base["valor_total"] = 1020.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            insumo = create_insumo(**datos_insumo_base)
            divergencia_warnings = [
                x for x in w if "Divergencia" in str(x.message)
            ]
            assert len(divergencia_warnings) >= 1

    def test_cero_por_cero_es_valido(self, datos_insumo_base):
        """cantidad=0, precio=0, valor_total=0 es válido."""
        datos_insumo_base["cantidad"] = 0.0
        datos_insumo_base["precio_unitario"] = 0.0
        datos_insumo_base["valor_total"] = 0.0
        insumo = create_insumo(**datos_insumo_base)
        assert insumo.is_valid

    def test_producto_cero_valor_positivo_lanza_error(self, datos_insumo_base):
        """cantidad×precio=0 pero valor_total>0 lanza error."""
        datos_insumo_base["cantidad"] = 0.0
        datos_insumo_base["precio_unitario"] = 100.0
        datos_insumo_base["valor_total"] = 500.0
        with pytest.raises(ValidationError, match="cantidad×precio=0"):
            create_insumo(**datos_insumo_base)

    def test_tolerancia_absoluta_pequena(self, datos_insumo_base):
        """Diferencia absoluta <= 0.01 se acepta sin warning."""
        datos_insumo_base["cantidad"] = 1.0
        datos_insumo_base["precio_unitario"] = 100.0
        datos_insumo_base["valor_total"] = 100.005
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            insumo = create_insumo(**datos_insumo_base)
            divergencia_warnings = [
                x for x in w if "Divergencia" in str(x.message)
            ]
            assert len(divergencia_warnings) == 0


class TestManoDeObra:
    """Pruebas específicas para ManoDeObra."""

    def test_instancia_correcta(self, insumo_mano_obra):
        assert isinstance(insumo_mano_obra, ManoDeObra)
        assert isinstance(insumo_mano_obra, InsumoProcesado)

    def test_tipo_insumo(self, insumo_mano_obra):
        assert insumo_mano_obra.tipo_insumo == "MANO_DE_OBRA"

    def test_requires_rendimiento(self):
        assert ManoDeObra.REQUIRES_RENDIMIENTO is True

    def test_rendimiento_cero_emite_warning(self, datos_insumo_base):
        """Rendimiento=0 en ManoDeObra emite warning."""
        datos_insumo_base["rendimiento"] = 0.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            create_insumo(**datos_insumo_base)
            rend_warnings = [
                x for x in w if "Rendimiento" in str(x.message)
            ]
            assert len(rend_warnings) >= 1

    def test_discrepancia_rendimiento_cantidad_emite_warning(
        self, datos_insumo_base
    ):
        """
        Si rendimiento > 0 y cantidad ≠ 1/rendimiento, emite warning.
        """
        datos_insumo_base["rendimiento"] = 10.0  # esperada: 0.1
        datos_insumo_base["cantidad"] = 0.5      # discrepancia
        datos_insumo_base["valor_total"] = 0.5 * 15000.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            create_insumo(**datos_insumo_base)
            disc_warnings = [
                x for x in w if "Discrepancia" in str(x.message)
            ]
            assert len(disc_warnings) >= 1

    def test_rendimiento_consistente_sin_warning(self, datos_insumo_base):
        """Cuando cantidad ≈ 1/rendimiento, no hay warning de discrepancia."""
        datos_insumo_base["rendimiento"] = 2.0
        datos_insumo_base["cantidad"] = 0.5
        datos_insumo_base["valor_total"] = 0.5 * 15000.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            create_insumo(**datos_insumo_base)
            disc_warnings = [
                x for x in w if "Discrepancia Rendimiento" in str(x.message)
            ]
            assert len(disc_warnings) == 0

    def test_jornal_valido(self, datos_insumo_base):
        """El campo jornal se valida correctamente."""
        datos_insumo_base["jornal"] = 50000.0
        insumo = create_insumo(**datos_insumo_base)
        assert insumo.jornal == 50000.0

    def test_jornal_negativo_lanza_error(self, datos_insumo_base):
        """Jornal negativo causa error."""
        datos_insumo_base["jornal"] = -1000.0
        with pytest.raises(ValidationError):
            create_insumo(**datos_insumo_base)


class TestEquipo:
    """Pruebas para Equipo."""

    def test_instancia_correcta(self, insumo_equipo):
        assert isinstance(insumo_equipo, Equipo)
        assert insumo_equipo.tipo_insumo == "EQUIPO"

    def test_no_requiere_rendimiento(self):
        assert Equipo.REQUIRES_RENDIMIENTO is False

    def test_unidades_esperadas_son_tiempo(self):
        assert Equipo.EXPECTED_UNITS == UNIDADES_TIEMPO


class TestSuministro:
    """Pruebas para Suministro."""

    def test_instancia_correcta(self, insumo_suministro):
        assert isinstance(insumo_suministro, Suministro)
        assert insumo_suministro.tipo_insumo == "SUMINISTRO"

    def test_cantidad_cero_emite_warning(self, datos_suministro):
        """Suministro con cantidad=0 emite warning."""
        datos_suministro["cantidad"] = 0.0
        datos_suministro["valor_total"] = 0.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            create_insumo(**datos_suministro)
            qty_warnings = [
                x for x in w if "cantidad=0" in str(x.message)
            ]
            assert len(qty_warnings) >= 1

    def test_unidades_esperadas_amplias(self):
        """Suministro acepta múltiples categorías de unidades."""
        assert "KG" in Suministro.EXPECTED_UNITS
        assert "M3" in Suministro.EXPECTED_UNITS
        assert "UNIDAD" in Suministro.EXPECTED_UNITS
        assert "M2" in Suministro.EXPECTED_UNITS


class TestTransporte:
    """Pruebas para Transporte."""

    def test_instancia_correcta(self, insumo_transporte):
        assert isinstance(insumo_transporte, Transporte)
        assert insumo_transporte.tipo_insumo == "TRANSPORTE"

    def test_unidades_transporte(self):
        assert "VIAJE" in Transporte.EXPECTED_UNITS
        assert "TON-KM" in Transporte.EXPECTED_UNITS


class TestOtro:
    """Pruebas para Otro."""

    def test_instancia_correcta(self, insumo_otro):
        assert isinstance(insumo_otro, Otro)
        assert insumo_otro.tipo_insumo == "OTRO"

    def test_sin_restriccion_unidades(self):
        """Otro no restringe unidades."""
        assert len(Otro.EXPECTED_UNITS) == 0


# ============================================================================
# 5. APU STRUCTURE Y MÉTRICAS TOPOLÓGICAS
# ============================================================================

class TestAPUStructure:
    """Pruebas para APUStructure."""

    def test_stratum_es_tactics(self, apu_completo):
        assert apu_completo.stratum == Stratum.TACTICS

    def test_support_base_width(self, apu_completo):
        assert apu_completo.support_base_width == 3

    def test_apu_vacio(self):
        apu = APUStructure(id="EMPTY", description="Vacío")
        assert apu.support_base_width == 0
        assert apu.total_cost == 0.0

    def test_add_resource(self, insumo_mano_obra):
        apu = APUStructure(id="TEST", description="Test")
        apu.add_resource(insumo_mano_obra)
        assert apu.support_base_width == 1

    def test_add_resource_tipo_invalido(self):
        """Solo se aceptan InsumoProcesado."""
        apu = APUStructure(id="TEST", description="Test")
        with pytest.raises(TypeError, match="InsumoProcesado"):
            apu.add_resource("no es un insumo")  # type: ignore

    def test_add_resource_stratum_invalido(self):
        """Solo se aceptan nodos PHYSICS."""
        apu_child = APUStructure(id="CHILD", description="Hijo")
        apu_parent = APUStructure(id="PARENT", description="Padre")
        with pytest.raises(TypeError, match="PHYSICS"):
            apu_parent.add_resource(apu_child)  # type: ignore

    def test_total_cost(self, apu_completo):
        """El costo total es la suma de valor_total de recursos."""
        expected = sum(r.valor_total for r in apu_completo.resources)
        assert abs(apu_completo.total_cost - expected) < 1e-10

    def test_is_inverted_pyramid_false(self, apu_completo):
        """Un APU con 3 recursos y quantity=10 no es pirámide invertida."""
        assert apu_completo.is_inverted_pyramid is False

    def test_is_inverted_pyramid_true(self, insumo_mano_obra):
        """APU con quantity>1000 y un solo recurso es pirámide invertida."""
        apu = APUStructure(
            id="INV", description="Invertido", quantity=5000.0
        )
        apu.add_resource(insumo_mano_obra)
        assert apu.is_inverted_pyramid is True


class TestCostBreakdown:
    """Pruebas para get_cost_breakdown y get_cost_fractions."""

    def test_breakdown_por_tipo(self, apu_completo):
        breakdown = apu_completo.get_cost_breakdown()
        assert "MANO_DE_OBRA" in breakdown
        assert "EQUIPO" in breakdown
        assert "SUMINISTRO" in breakdown

    def test_breakdown_suma_correcta(self, apu_completo):
        """La suma del breakdown es igual al total_cost."""
        breakdown = apu_completo.get_cost_breakdown()
        assert abs(sum(breakdown.values()) - apu_completo.total_cost) < 1e-10

    def test_fractions_suman_uno(self, apu_completo):
        """Las fracciones de costo suman ≈ 1.0."""
        fractions = apu_completo.get_cost_fractions()
        assert abs(sum(fractions.values()) - 1.0) < 1e-10

    def test_fractions_apu_vacio(self):
        """APU vacío retorna diccionario vacío."""
        apu = APUStructure(id="EMPTY", description="Vacío")
        assert apu.get_cost_fractions() == {}

    def test_fractions_valores_cero(self, datos_insumo_base):
        """Si todos los valores son 0, las fracciones son 0."""
        datos_insumo_base["cantidad"] = 0.0
        datos_insumo_base["precio_unitario"] = 0.0
        datos_insumo_base["valor_total"] = 0.0
        insumo = create_insumo(**datos_insumo_base)
        apu = APUStructure(id="ZERO", description="Zero cost")
        apu.add_resource(insumo)
        fractions = apu.get_cost_fractions()
        assert all(v == 0.0 for v in fractions.values())


class TestTopologicalStabilityIndex:
    """Pruebas para el Índice de Estabilidad Topológica (Ψ)."""

    def test_apu_vacio_estabilidad_cero(self):
        """APU sin recursos tiene Ψ = 0."""
        apu = APUStructure(id="EMPTY", description="Vacío")
        assert apu.topological_stability_index() == 0.0

    def test_un_recurso_estabilidad_cero(self, insumo_mano_obra):
        """Un solo recurso → entropía = 0 → Ψ = 0."""
        apu = APUStructure(id="SINGLE", description="Un recurso")
        apu.add_resource(insumo_mano_obra)
        psi = apu.topological_stability_index()
        assert psi == 0.0

    def test_estabilidad_en_rango(self, apu_completo):
        """Ψ ∈ [0, 1]."""
        psi = apu_completo.topological_stability_index()
        assert 0.0 <= psi <= 1.0

    def test_mayor_diversidad_mayor_estabilidad(
        self, datos_insumo_base, datos_equipo, datos_suministro
    ):
        """
        Un APU con 3 tipos distintos es más estable que uno con 2 tipos iguales
        (a igual distribución de costos).
        """
        # APU mono-tipo (3 recursos del mismo tipo)
        apu_mono = APUStructure(id="MONO", description="Mono-tipo")
        for i in range(3):
            d = datos_insumo_base.copy()
            d["descripcion_insumo"] = f"Recurso {i}"
            apu_mono.add_resource(create_insumo(**d))

        # APU diverso (3 tipos distintos)
        apu_diverso = APUStructure(id="DIV", description="Diverso")
        apu_diverso.add_resource(create_insumo(**datos_insumo_base))
        apu_diverso.add_resource(create_insumo(**datos_equipo))
        apu_diverso.add_resource(create_insumo(**datos_suministro))

        assert (
            apu_diverso.topological_stability_index()
            > apu_mono.topological_stability_index()
        )

    def test_distribucion_uniforme_maxima_entropia(
        self, datos_insumo_base, datos_equipo, datos_suministro
    ):
        """
        Distribución uniforme de costos maximiza la componente de entropía.
        """
        valor_uniforme = 10000.0
        apu = APUStructure(id="UNIF", description="Uniforme")

        for datos in [datos_insumo_base, datos_equipo, datos_suministro]:
            d = datos.copy()
            d["cantidad"] = 1.0
            d["precio_unitario"] = valor_uniforme
            d["valor_total"] = valor_uniforme
            apu.add_resource(create_insumo(**d))

        psi = apu.topological_stability_index()
        # Con 3 tipos y distribución uniforme, Ψ debería ser alto
        assert psi > 0.5

    def test_recursos_con_valor_cero(self, datos_insumo_base):
        """APU donde todos los costos son 0 tiene Ψ = 0."""
        apu = APUStructure(id="ZEROS", description="Todo cero")
        for i in range(3):
            d = datos_insumo_base.copy()
            d["cantidad"] = 0.0
            d["precio_unitario"] = 0.0
            d["valor_total"] = 0.0
            d["descripcion_insumo"] = f"Zero {i}"
            apu.add_resource(create_insumo(**d))

        assert apu.topological_stability_index() == 0.0

    def test_estabilidad_es_determinista(self, apu_completo):
        """Múltiples llamadas producen el mismo resultado."""
        psi1 = apu_completo.topological_stability_index()
        psi2 = apu_completo.topological_stability_index()
        assert psi1 == psi2

    def test_dos_recursos_mismo_tipo_valor_diferente(self, datos_insumo_base):
        """
        Dos recursos del mismo tipo con valores desiguales tienen entropía < 1.
        """
        apu = APUStructure(id="SKEW", description="Sesgado")

        d1 = datos_insumo_base.copy()
        d1["cantidad"] = 1.0
        d1["precio_unitario"] = 100.0
        d1["valor_total"] = 100.0
        d1["descripcion_insumo"] = "Recurso A"
        apu.add_resource(create_insumo(**d1))

        d2 = datos_insumo_base.copy()
        d2["cantidad"] = 1.0
        d2["precio_unitario"] = 900.0
        d2["valor_total"] = 900.0
        d2["descripcion_insumo"] = "Recurso B"
        apu.add_resource(create_insumo(**d2))

        psi = apu.topological_stability_index()
        assert 0.0 < psi < 1.0

    def test_formula_geometrica_ambos_factores_necesarios(
        self, datos_insumo_base
    ):
        """
        La media geométrica ponderada D^0.4 · H^0.6 garantiza que
        si uno de los factores es 0, el índice completo es 0.
        
        Con un solo recurso, H=0 → Ψ=0 independientemente de D.
        """
        apu = APUStructure(id="GEO", description="Geométrica")
        apu.add_resource(create_insumo(**datos_insumo_base))
        # n=1 → H=0 → 0^0.6 = 0 → Ψ = D^0.4 * 0 = 0
        assert apu.topological_stability_index() == 0.0


# ============================================================================
# 6. FACTORY FUNCTIONS
# ============================================================================

class TestCreateInsumo:
    """Pruebas para la factory create_insumo."""

    def test_crea_mano_de_obra(self, datos_insumo_base):
        insumo = create_insumo(**datos_insumo_base)
        assert isinstance(insumo, ManoDeObra)

    def test_crea_equipo(self, datos_equipo):
        insumo = create_insumo(**datos_equipo)
        assert isinstance(insumo, Equipo)

    def test_crea_suministro(self, datos_suministro):
        insumo = create_insumo(**datos_suministro)
        assert isinstance(insumo, Suministro)

    def test_crea_transporte(self, datos_transporte):
        insumo = create_insumo(**datos_transporte)
        assert isinstance(insumo, Transporte)

    def test_crea_otro(self, datos_otro):
        insumo = create_insumo(**datos_otro)
        assert isinstance(insumo, Otro)

    def test_sin_tipo_insumo_lanza_error(self, datos_insumo_base):
        del datos_insumo_base["tipo_insumo"]
        with pytest.raises(ValidationError, match="tipo_insumo"):
            create_insumo(**datos_insumo_base)

    def test_filtra_kwargs_invalidos(self, datos_insumo_base):
        """Argumentos no reconocidos se filtran sin error."""
        datos_insumo_base["campo_inexistente"] = "valor"
        insumo = create_insumo(**datos_insumo_base)
        assert insumo.is_valid

    def test_tipo_normalizado_en_resultado(self, datos_insumo_base):
        datos_insumo_base["tipo_insumo"] = "mano de obra"
        insumo = create_insumo(**datos_insumo_base)
        assert insumo.tipo_insumo == "MANO_DE_OBRA"

    @pytest.mark.parametrize("tipo,clase", [
        ("MANO_DE_OBRA", ManoDeObra),
        ("EQUIPO", Equipo),
        ("SUMINISTRO", Suministro),
        ("TRANSPORTE", Transporte),
        ("OTRO", Otro),
    ])
    def test_mapeo_tipo_clase(self, tipo: str, clase: type, datos_insumo_base):
        """Cada tipo se mapea a su clase correcta."""
        datos_insumo_base["tipo_insumo"] = tipo
        if tipo == "TRANSPORTE":
            datos_insumo_base["unidad_insumo"] = "VIAJE"
        elif tipo == "SUMINISTRO":
            datos_insumo_base["unidad_insumo"] = "KG"
        elif tipo == "OTRO":
            datos_insumo_base["unidad_insumo"] = "%"
        insumo = create_insumo(**datos_insumo_base)
        assert isinstance(insumo, clase)


class TestValidateInsumoData:
    """Pruebas para validate_insumo_data."""

    def test_datos_validos(self, datos_insumo_base):
        result = validate_insumo_data(datos_insumo_base)
        assert isinstance(result, dict)
        assert result["tipo_insumo"] == "MANO_DE_OBRA"

    def test_input_no_dict_lanza_error(self):
        with pytest.raises(ValidationError, match="dict"):
            validate_insumo_data("no soy un dict")  # type: ignore

    def test_campos_faltantes(self):
        with pytest.raises(ValidationError, match="obligatorios"):
            validate_insumo_data({"codigo_apu": "TEST"})

    def test_campo_none_obligatorio(self, datos_insumo_base):
        datos_insumo_base["codigo_apu"] = None
        with pytest.raises(ValidationError, match="obligatorios"):
            validate_insumo_data(datos_insumo_base)

    def test_campo_numerico_none_usa_default(self, datos_insumo_base):
        """Campos numéricos con None toman valor por defecto."""
        datos_insumo_base["cantidad"] = None
        result = validate_insumo_data(datos_insumo_base)
        assert result["cantidad"] == 0.0

    def test_campo_numerico_string_se_convierte(self, datos_insumo_base):
        """Strings numéricos se convierten a float."""
        datos_insumo_base["cantidad"] = "25.5"
        result = validate_insumo_data(datos_insumo_base)
        assert result["cantidad"] == 25.5

    def test_campo_numerico_invalido_lanza_error(self, datos_insumo_base):
        datos_insumo_base["cantidad"] = "no_soy_numero"
        with pytest.raises(ValidationError, match="numérico"):
            validate_insumo_data(datos_insumo_base)

    def test_campo_numerico_negativo_lanza_error(self, datos_insumo_base):
        datos_insumo_base["cantidad"] = -5.0
        with pytest.raises(ValidationError, match="negativo"):
            validate_insumo_data(datos_insumo_base)

    def test_campo_numerico_infinito_lanza_error(self, datos_insumo_base):
        datos_insumo_base["precio_unitario"] = float("inf")
        with pytest.raises(ValidationError, match="finito"):
            validate_insumo_data(datos_insumo_base)

    def test_valor_total_calculado_si_cero(self, datos_insumo_base):
        """Si valor_total=0, se calcula como cantidad × precio."""
        datos_insumo_base["valor_total"] = 0.0
        datos_insumo_base["cantidad"] = 5.0
        datos_insumo_base["precio_unitario"] = 200.0
        result = validate_insumo_data(datos_insumo_base)
        assert result["valor_total"] == 1000.0

    def test_defaults_aplicados(self, datos_insumo_base):
        """Campos opcionales ausentes reciben defaults."""
        for key in ["capitulo", "rendimiento", "jornal"]:
            datos_insumo_base.pop(key, None)
        result = validate_insumo_data(datos_insumo_base)
        assert result["capitulo"] == "GENERAL"
        assert result["rendimiento"] == 0.0
        assert result["jornal"] == 0.0

    def test_tipo_insumo_normalizado(self, datos_insumo_base):
        datos_insumo_base["tipo_insumo"] = "mano de obra"
        result = validate_insumo_data(datos_insumo_base)
        assert result["tipo_insumo"] == "MANO_DE_OBRA"
        assert result["categoria"] == "MANO_DE_OBRA"


class TestCreateInsumoFromRaw:
    """Pruebas para el pipeline completo create_insumo_from_raw."""

    def test_pipeline_completo(self, datos_insumo_base):
        insumo = create_insumo_from_raw(datos_insumo_base)
        assert isinstance(insumo, ManoDeObra)
        assert insumo.is_valid

    def test_con_datos_parciales(self):
        """Datos con campos numéricos faltantes se completan."""
        raw = {
            "codigo_apu": "APU-RAW-001",
            "descripcion_apu": "Test raw",
            "unidad_apu": "M3",
            "descripcion_insumo": "Material genérico",
            "unidad_insumo": "KG",
            "tipo_insumo": "SUMINISTRO",
            "cantidad": 10.0,
            "precio_unitario": 100.0,
        }
        insumo = create_insumo_from_raw(raw)
        assert insumo.valor_total == 1000.0  # Calculado automáticamente

    def test_con_strings_numericos(self):
        """Strings numéricos se procesan correctamente."""
        raw = {
            "codigo_apu": "APU-STR-001",
            "descripcion_apu": "Test strings",
            "unidad_apu": "M2",
            "descripcion_insumo": "Material X",
            "unidad_insumo": "UND",
            "tipo_insumo": "OTRO",
            "cantidad": "5",
            "precio_unitario": "200.50",
        }
        insumo = create_insumo_from_raw(raw)
        assert insumo.cantidad == 5.0
        assert insumo.precio_unitario == 200.50

    def test_datos_invalidos_lanza_error(self):
        """Datos irrecuperables lanzan error."""
        with pytest.raises(ValidationError):
            create_insumo_from_raw({"codigo_apu": "X"})


# ============================================================================
# 7. FUNCIONES DE UTILIDAD
# ============================================================================

class TestUtilidades:
    """Pruebas para funciones de compatibilidad."""

    def test_get_tipo_insumo_class(self):
        assert get_tipo_insumo_class("EQUIPO") is Equipo
        assert get_tipo_insumo_class(TipoInsumo.MANO_DE_OBRA) is ManoDeObra

    def test_get_tipo_insumo_class_invalido(self):
        with pytest.raises(InvalidTipoInsumoError):
            get_tipo_insumo_class("FANTASMA")

    def test_get_all_tipo_insumo_values(self):
        values = get_all_tipo_insumo_values()
        assert isinstance(values, frozenset)
        assert len(values) == 5

    def test_is_valid_tipo_insumo(self):
        assert is_valid_tipo_insumo("EQUIPO") is True
        assert is_valid_tipo_insumo("FANTASMA") is False
        assert is_valid_tipo_insumo("mano de obra") is True

    def test_insumo_class_map_completo(self):
        """El mapa cubre todos los tipos de TipoInsumo."""
        for tipo in TipoInsumo:
            assert tipo.value in INSUMO_CLASS_MAP


# ============================================================================
# 8. CASOS LÍMITE Y REGRESIONES
# ============================================================================

class TestCasosLimite:
    """Pruebas de borde y regresiones."""

    def test_valores_en_limites_exactos(self, datos_insumo_base):
        """Valores exactamente en los límites son válidos."""
        datos_insumo_base["cantidad"] = MAX_CANTIDAD
        datos_insumo_base["precio_unitario"] = 1.0
        datos_insumo_base["valor_total"] = MAX_CANTIDAD
        insumo = create_insumo(**datos_insumo_base)
        assert insumo.is_valid

    def test_cantidad_cero_es_valida(self, datos_insumo_base):
        """Cantidad = 0 es válida (no negativa)."""
        datos_insumo_base["cantidad"] = 0.0
        datos_insumo_base["valor_total"] = 0.0
        insumo = create_insumo(**datos_insumo_base)
        assert insumo.cantidad == 0.0

    def test_rendimiento_cero_es_valido(self, datos_insumo_base):
        """Rendimiento = 0 es válido (emite warning en ManoDeObra)."""
        datos_insumo_base["rendimiento"] = 0.0
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            insumo = create_insumo(**datos_insumo_base)
            assert insumo.rendimiento == 0.0

    def test_descripcion_con_unicode_complejo(self, datos_insumo_base):
        """Caracteres Unicode complejos se procesan sin error."""
        datos_insumo_base["descripcion_insumo"] = "Señal vía — «túnel» ñandú"
        insumo = create_insumo(**datos_insumo_base)
        assert "SENAL" in insumo.descripcion_insumo
        assert "NANDU" in insumo.descripcion_insumo

    def test_codigo_con_puntos_y_guiones(self, datos_insumo_base):
        """Puntos y guiones son válidos en códigos."""
        datos_insumo_base["codigo_apu"] = "CAP.01-ITEM.02.A"
        insumo = create_insumo(**datos_insumo_base)
        assert insumo.codigo_apu == "CAP.01-ITEM.02.A"

    def test_multiples_insumos_mismo_apu(self, datos_insumo_base):
        """Se pueden crear múltiples insumos con el mismo código APU."""
        insumos = []
        for i in range(5):
            d = datos_insumo_base.copy()
            d["descripcion_insumo"] = f"Recurso {i}"
            insumos.append(create_insumo(**d))
        assert all(ins.is_valid for ins in insumos)
        assert len({ins.id for ins in insumos}) == 5  # IDs únicos

    def test_precision_numerica_flotante(self, datos_insumo_base):
        """Valores con alta precisión decimal no causan falsos positivos."""
        datos_insumo_base["cantidad"] = 1.0 / 3.0
        datos_insumo_base["precio_unitario"] = 99999.99
        datos_insumo_base["valor_total"] = (1.0 / 3.0) * 99999.99
        insumo = create_insumo(**datos_insumo_base)
        assert insumo.is_valid

    def test_apu_con_cinco_tipos(
        self,
        datos_insumo_base,
        datos_equipo,
        datos_suministro,
        datos_transporte,
        datos_otro,
    ):
        """APU con los 5 tipos de insumo tiene diversidad máxima."""
        apu = APUStructure(id="FULL", description="Todos los tipos")

        for datos in [
            datos_insumo_base,
            datos_equipo,
            datos_suministro,
            datos_transporte,
            datos_otro,
        ]:
            apu.add_resource(create_insumo(**datos))

        assert apu.support_base_width == 5
        psi = apu.topological_stability_index()
        assert psi > 0.7  # Alta estabilidad con diversidad máxima

    def test_herencia_completa(self, insumo_mano_obra):
        """ManoDeObra hereda de InsumoProcesado → TopologicalNode."""
        assert isinstance(insumo_mano_obra, TopologicalNode)
        assert isinstance(insumo_mano_obra, InsumoProcesado)
        assert isinstance(insumo_mano_obra, ManoDeObra)

    def test_jerarquia_excepciones(self):
        """Las excepciones mantienen la jerarquía correcta."""
        assert issubclass(ValidationError, SchemaError)
        assert issubclass(InvalidTipoInsumoError, ValidationError)
        assert issubclass(InsumoDataError, ValidationError)
        assert issubclass(UnitNormalizationError, SchemaError)

    def test_error_fatal_en_factory_lanza_insumo_data_error(
        self, datos_insumo_base
    ):
        """
        Errores inesperados en la creación se envuelven en InsumoDataError.
        """
        with patch.object(
            ManoDeObra, "__post_init__", side_effect=RuntimeError("boom")
        ):
            with pytest.raises(InsumoDataError, match="fatal"):
                create_insumo(**datos_insumo_base)


class TestValidacionUnidadesCategoria:
    """Pruebas para la validación de unidades contra categoría esperada."""

    def test_unidad_esperada_sin_warning(self, datos_insumo_base):
        """Unidad correcta para ManoDeObra (HORA) no emite warning de unidad."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            create_insumo(**datos_insumo_base)
            unit_warnings = [
                x for x in w if "unidades esperadas" in str(x.message).lower()
            ]
            assert len(unit_warnings) == 0

    def test_unidad_inesperada_emite_warning(self, datos_insumo_base):
        """Unidad 'KG' para ManoDeObra emite warning."""
        datos_insumo_base["unidad_insumo"] = "KG"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            insumo = create_insumo(**datos_insumo_base)
            unit_warnings = [
                x for x in w
                if "no está en las unidades esperadas" in str(x.message)
            ]
            assert len(unit_warnings) >= 1
            # Pero el insumo se crea igualmente
            assert insumo.is_valid

    def test_otro_no_emite_warning_unidad(self, datos_otro):
        """Tipo 'Otro' no restringe unidades."""
        datos_otro["unidad_insumo"] = "BARRIL"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            create_insumo(**datos_otro)
            unit_warnings = [
                x for x in w
                if "unidades esperadas" in str(x.message).lower()
            ]
            assert len(unit_warnings) == 0


class TestDeterministicHash:
    """Pruebas para la función de hash determinista."""

    def test_determinista(self, datos_insumo_base):
        """Mismos datos → mismo ID en ejecuciones distintas."""
        i1 = create_insumo(**datos_insumo_base)
        i2 = create_insumo(**datos_insumo_base)
        assert i1.id == i2.id

    def test_sensibilidad(self, datos_insumo_base):
        """Cambio mínimo en datos → ID diferente."""
        i1 = create_insumo(**datos_insumo_base)
        datos_insumo_base["descripcion_insumo"] = "Obrero raso X"
        i2 = create_insumo(**datos_insumo_base)
        assert i1.id != i2.id

    def test_id_contiene_codigo_apu(self, insumo_mano_obra):
        """El ID incluye el código APU como prefijo."""
        assert insumo_mano_obra.id.startswith("APU-001")


class TestInsumoClassMap:
    """Pruebas de integridad del registro de clases."""

    def test_cobertura_completa(self):
        """Todos los TipoInsumo tienen clase registrada."""
        for tipo in TipoInsumo:
            assert tipo.value in INSUMO_CLASS_MAP, (
                f"Tipo {tipo.value} sin clase registrada"
            )

    def test_clases_son_subclases(self):
        """Todas las clases registradas son subclases de InsumoProcesado."""
        for tipo, cls in INSUMO_CLASS_MAP.items():
            assert issubclass(cls, InsumoProcesado), (
                f"{cls.__name__} no es subclase de InsumoProcesado"
            )

    def test_no_hay_duplicados(self):
        """Cada tipo mapea a una clase distinta."""
        clases = list(INSUMO_CLASS_MAP.values())
        assert len(clases) == len(set(clases))


class TestEdgeCasesEntropy:
    """Casos límite para el cálculo de entropía."""

    def test_un_recurso_domina(self, datos_insumo_base, datos_equipo):
        """
        Cuando un recurso domina el 99% del costo,
        la entropía es baja y Ψ refleja eso.
        """
        apu = APUStructure(id="DOM", description="Dominante")

        d_grande = datos_insumo_base.copy()
        d_grande["cantidad"] = 1.0
        d_grande["precio_unitario"] = 1_000_000.0
        d_grande["valor_total"] = 1_000_000.0
        apu.add_resource(create_insumo(**d_grande))

        d_pequeno = datos_equipo.copy()
        d_pequeno["cantidad"] = 1.0
        d_pequeno["precio_unitario"] = 100.0
        d_pequeno["valor_total"] = 100.0
        apu.add_resource(create_insumo(**d_pequeno))

        psi = apu.topological_stability_index()
        # Baja entropía y solo 2 tipos → Ψ bajo
        assert psi < 0.5

    def test_distribucion_perfectamente_uniforme(self, datos_insumo_base):
        """
        n recursos con el mismo costo → H_norm = 1.0.
        """
        apu = APUStructure(id="PERF", description="Perfecto")
        n = 4
        for i in range(n):
            d = datos_insumo_base.copy()
            d["cantidad"] = 1.0
            d["precio_unitario"] = 1000.0
            d["valor_total"] = 1000.0
            d["descripcion_insumo"] = f"Recurso uniforme {i}"
            apu.add_resource(create_insumo(**d))

        psi = apu.topological_stability_index()
        # Solo 1 tipo (MANO_DE_OBRA) → diversidad baja
        # Pero la entropía es máxima
        assert psi > 0.0

    def test_entropia_con_valores_muy_pequenos(self, datos_insumo_base):
        """Valores muy pequeños (pero positivos) no causan problemas numéricos."""
        apu = APUStructure(id="TINY", description="Tiny values")
        for i in range(3):
            d = datos_insumo_base.copy()
            d["cantidad"] = 0.0001
            d["precio_unitario"] = 0.001
            d["valor_total"] = 0.0001 * 0.001
            d["descripcion_insumo"] = f"Tiny {i}"
            apu.add_resource(create_insumo(**d))

        psi = apu.topological_stability_index()
        assert math.isfinite(psi)
        assert 0.0 <= psi <= 1.0


# ============================================================================
# 9. INTEGRACIÓN
# ============================================================================

class TestIntegracion:
    """Pruebas de integración que verifican el flujo completo."""

    def test_pipeline_raw_a_apu(self):
        """
        Flujo completo: datos crudos → insumos → APU → métricas.
        """
        raw_insumos = [
            {
                "codigo_apu": "INT-001",
                "descripcion_apu": "Muro en ladrillo",
                "unidad_apu": "M2",
                "descripcion_insumo": "Oficial de construcción",
                "unidad_insumo": "hora",
                "cantidad": 1.5,
                "precio_unitario": 12000,
                "valor_total": 18000,
                "tipo_insumo": "mano de obra",
                "rendimiento": 8.0,
            },
            {
                "codigo_apu": "INT-001",
                "descripcion_apu": "Muro en ladrillo",
                "unidad_apu": "M2",
                "descripcion_insumo": "Ladrillo tolete",
                "unidad_insumo": "und",
                "cantidad": 35,
                "precio_unitario": 800,
                "valor_total": 28000,
                "tipo_insumo": "suministro",
            },
            {
                "codigo_apu": "INT-001",
                "descripcion_apu": "Muro en ladrillo",
                "unidad_apu": "M2",
                "descripcion_insumo": "Andamio metálico",
                "unidad_insumo": "hora",
                "cantidad": 0.5,
                "precio_unitario": 5000,
                "valor_total": 2500,
                "tipo_insumo": "equipo",
            },
        ]

        # Crear insumos
        insumos = [create_insumo_from_raw(r) for r in raw_insumos]
        assert len(insumos) == 3
        assert all(i.is_valid for i in insumos)

        # Verificar tipos correctos
        assert isinstance(insumos[0], ManoDeObra)
        assert isinstance(insumos[1], Suministro)
        assert isinstance(insumos[2], Equipo)

        # Crear APU
        apu = APUStructure(
            id="INT-001",
            description="Muro en ladrillo",
            unit="M2",
            quantity=50.0,
        )
        for insumo in insumos:
            apu.add_resource(insumo)

        # Verificar métricas
        assert apu.support_base_width == 3
        assert apu.total_cost == 18000 + 28000 + 2500
        assert not apu.is_inverted_pyramid

        # Estabilidad topológica
        psi = apu.topological_stability_index()
        assert 0.0 < psi <= 1.0

        # Breakdown de costos
        breakdown = apu.get_cost_breakdown()
        assert breakdown["MANO_DE_OBRA"] == 18000.0
        assert breakdown["SUMINISTRO"] == 28000.0
        assert breakdown["EQUIPO"] == 2500.0

        # Fracciones
        fractions = apu.get_cost_fractions()
        assert abs(sum(fractions.values()) - 1.0) < 1e-10

    def test_multiples_apus_independientes(self, datos_insumo_base, datos_equipo):
        """Crear múltiples APUs independientes no causa interferencia."""
        apu1 = APUStructure(id="A1", description="APU 1")
        apu2 = APUStructure(id="A2", description="APU 2")

        apu1.add_resource(create_insumo(**datos_insumo_base))
        apu2.add_resource(create_insumo(**datos_equipo))

        assert apu1.support_base_width == 1
        assert apu2.support_base_width == 1
        assert apu1.resources[0].tipo_insumo != apu2.resources[0].tipo_insumo

    def test_serialization_roundtrip(self, datos_insumo_base):
        """to_dict() produce un diccionario reutilizable."""
        insumo = create_insumo(**datos_insumo_base)
        d = insumo.to_dict()

        assert isinstance(d, dict)
        assert d["codigo_apu"] == insumo.codigo_apu
        assert d["tipo_insumo"] == insumo.tipo_insumo
        assert d["cantidad"] == insumo.cantidad
        assert d["precio_unitario"] == insumo.precio_unitario