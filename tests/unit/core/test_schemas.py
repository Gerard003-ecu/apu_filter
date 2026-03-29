"""
Suite de pruebas rigurosas para ``app.core.schemas``.

Cobertura:
    [T1] Enumeraciones: orden, dependencias, métodos.
    [T2] Normalización: idempotencia, casos límite.
    [T3] Validadores: numéricos, strings, híbridos.
    [T4] Invariantes: (I1-I6) formales.
    [T5] InsumoProcesado: construcción, validación, serialización.
    [T6] Subclases: especialización y comportamiento.
    [T7] APUStructure: topología, entropía, estabilidad.
    [T8] Factory: resolución de tipos, pipelines.
    [T9] Edge cases: degeneración, overflow, casos límite.
    [T10] Integración end-to-end.

Ejecutar::

    pytest tests/unit/core/test_schemas.py -v --tb=short
    pytest tests/unit/core/test_schemas.py -v --cov=app.core.schemas --cov-report=html
"""

from __future__ import annotations

import math
import warnings
import sys
from decimal import Decimal
from typing import Dict, List, Set, Tuple

import pytest

from app.core.schemas import (
    # Enumeraciones
    Stratum,
    TipoInsumo,
    # Excepciones
    SchemaError,
    ValidationError,
    InvalidTipoInsumoError,
    InsumoDataError,
    InvariantError,
    # Estructuras
    TopologicalNode,
    InsumoProcesado,
    APUStructure,
    ManoDeObra,
    Equipo,
    Suministro,
    Transporte,
    Otro,
    # Factories
    create_insumo,
    create_insumo_from_raw,
    validate_insumo_data,
    # Normalización
    normalize_unit,
    normalize_description,
    normalize_codigo,
    # Validadores
    NumericValidator,
    StringValidator,
    NumericalTolerances,
    PhysicalBounds,
    # Constantes
    UNIDADES_TIEMPO,
    UNIDADES_MASA,
    UNIDADES_VOLUMEN,
    UNIDADES_AREA,
    UNIDADES_LONGITUD,
    UNIDADES_TRANSPORTE,
    UNIDADES_GENERICAS,
    INSUMO_CLASS_MAP,
    # Utilidades
    get_tipo_insumo_class,
    get_all_tipo_insumo_values,
    is_valid_tipo_insumo,
    _deterministic_short_hash,
    _TOLERANCES,
    _BOUNDS,
)


# =============================================================================
# TEST SUITE 1: ENUMERACIONES
# =============================================================================


@pytest.mark.unit
class TestStratum:
    """Verifica propiedades de Stratum."""

    def test_stratum_values_defined(self) -> None:
        """Todos los valores están definidos."""
        assert Stratum.WISDOM.value == 0
        assert Stratum.ALPHA.value == 1
        assert Stratum.OMEGA.value == 2
        assert Stratum.STRATEGY.value == 3
        assert Stratum.TACTICS.value == 4
        assert Stratum.PHYSICS.value == 5

    def test_stratum_apex_and_base(self) -> None:
        """apex_stratum() = WISDOM, base_stratum() = PHYSICS."""
        assert Stratum.apex_stratum() == Stratum.WISDOM
        assert Stratum.base_stratum() == Stratum.PHYSICS

    def test_stratum_requires_order(self) -> None:
        """requires() retorna estratos con value > this.value."""
        # TACTICS (4) requiere PHYSICS (5)
        assert Stratum.PHYSICS in Stratum.TACTICS.requires()
        assert Stratum.TACTICS not in Stratum.TACTICS.requires()

        # WISDOM (0) requiere todos excepto él mismo
        wisdom_deps = Stratum.WISDOM.requires()
        assert len(wisdom_deps) == 5
        assert Stratum.PHYSICS in wisdom_deps
        assert Stratum.WISDOM not in wisdom_deps

    def test_stratum_ordered_bottom_up(self) -> None:
        """ordered_bottom_up() retorna [PHYSICS, ..., WISDOM]."""
        ordered = Stratum.ordered_bottom_up()
        assert ordered[0] == Stratum.PHYSICS
        assert ordered[-1] == Stratum.WISDOM

        # Verificar orden descendente de value
        for i in range(len(ordered) - 1):
            assert ordered[i].value > ordered[i+1].value

    def test_stratum_ordered_top_down(self) -> None:
        """ordered_top_down() retorna [WISDOM, ..., PHYSICS]."""
        ordered = Stratum.ordered_top_down()
        assert ordered[0] == Stratum.WISDOM
        assert ordered[-1] == Stratum.PHYSICS

        # Verificar orden ascendente de value
        for i in range(len(ordered) - 1):
            assert ordered[i].value < ordered[i+1].value


@pytest.mark.unit
class TestTipoInsumo:
    """Verifica enumeración TipoInsumo."""

    def test_tipo_insumo_members(self) -> None:
        """Todos los 5 miembros están definidos."""
        assert TipoInsumo.MANO_DE_OBRA.value == "MANO_DE_OBRA"
        assert TipoInsumo.EQUIPO.value == "EQUIPO"
        assert TipoInsumo.SUMINISTRO.value == "SUMINISTRO"
        assert TipoInsumo.TRANSPORTE.value == "TRANSPORTE"
        assert TipoInsumo.OTRO.value == "OTRO"

    def test_tipo_insumo_valid_values(self) -> None:
        """valid_values() retorna 5 valores."""
        values = TipoInsumo.valid_values()
        assert len(values) == 5
        assert "MANO_DE_OBRA" in values

    def test_tipo_insumo_from_string_exact(self) -> None:
        """from_string() con valores exactos."""
        assert TipoInsumo.from_string("MANO_DE_OBRA") == TipoInsumo.MANO_DE_OBRA
        assert TipoInsumo.from_string("EQUIPO") == TipoInsumo.EQUIPO

    def test_tipo_insumo_from_string_normalized(self) -> None:
        """from_string() normaliza caso y espacios."""
        assert TipoInsumo.from_string("mano_de_obra") == TipoInsumo.MANO_DE_OBRA
        assert TipoInsumo.from_string("mano de obra") == TipoInsumo.MANO_DE_OBRA
        assert TipoInsumo.from_string("MANO-DE-OBRA") == TipoInsumo.MANO_DE_OBRA

    def test_tipo_insumo_from_string_idempotent(self) -> None:
        """Aplicar from_string dos veces produce mismo resultado."""
        tipo = TipoInsumo.from_string("equipo")
        assert TipoInsumo.from_string(tipo) == tipo

    def test_tipo_insumo_from_string_invalid(self) -> None:
        """from_string() rechaza valores inválidos."""
        with pytest.raises(InvalidTipoInsumoError):
            TipoInsumo.from_string("INVALIDO")

        with pytest.raises(InvalidTipoInsumoError):
            TipoInsumo.from_string(123)

    def test_tipo_insumo_inheritance(self) -> None:
        """TipoInsumo hereda de str y Enum."""
        assert isinstance(TipoInsumo.EQUIPO, str)
        assert TipoInsumo.EQUIPO == "EQUIPO"


# =============================================================================
# TEST SUITE 2: NORMALIZACIÓN (IDEMPOTENCIA)
# =============================================================================


@pytest.mark.unit
class TestNormalizationIdempotence:
    """
    Verifica propiedad: f(f(x)) = f(x) para normalizadores.

    Teorema: Una función es idempotente si aplicarla dos veces
    produce el mismo resultado que aplicarla una vez.
    """

    def test_normalize_unit_idempotent_known(self) -> None:
        """normalize_unit es idempotente para unidades conocidas."""
        test_cases = ["HORA", "DIA", "KG", "M3", "UNIDAD"]
        for unit in test_cases:
            f1 = normalize_unit(unit)
            f2 = normalize_unit(f1)
            assert f1 == f2, f"normalize_unit no idempotente: {unit}"

    def test_normalize_unit_idempotent_unknown(self) -> None:
        """normalize_unit es idempotente para unidades desconocidas."""
        test_cases = ["XYZABC", "custom_unit", "hora"]
        for unit in test_cases:
            f1 = normalize_unit(unit)
            f2 = normalize_unit(f1)
            assert f1 == f2

    def test_normalize_unit_idempotent_edge(self) -> None:
        """normalize_unit es idempotente para casos límite."""
        for unit in [None, "", "   ", "hora", "HORA"]:
            f1 = normalize_unit(unit)
            f2 = normalize_unit(f1)
            assert f1 == f2

    def test_normalize_description_idempotent(self) -> None:
        """normalize_description es idempotente."""
        test_cases = [
            "Acero estructural",
            "HORMIGÓN ARMADO",
            "Pintura—Blanca.",
            "Café® con leche",
            None,
            "",
        ]
        for desc in test_cases:
            f1 = normalize_description(desc)
            f2 = normalize_description(f1)
            assert f1 == f2, f"normalize_description no idempotente: {desc}"

    def test_normalize_codigo_idempotent(self) -> None:
        """normalize_codigo es idempotente."""
        test_cases = [
            "APU-001.a",
            "apu_001_a",
            "APU/001#a",  # / y # se removerán
        ]
        for codigo in test_cases:
            f1 = normalize_codigo(codigo)
            f2 = normalize_codigo(f1)
            assert f1 == f2

    def test_normalize_codigo_rejects_empty(self) -> None:
        """normalize_codigo rechaza vacío tras normalización."""
        with pytest.raises(ValidationError):
            normalize_codigo("///###")  # Se vuelve vacío


# =============================================================================
# TEST SUITE 3: NORMALIZACIÓN (MAPPEO Y CONVENCIONES)
# =============================================================================


@pytest.mark.unit
class TestNormalizationMappings:
    """Verifica mapeos y convenciones de normalización."""

    def test_normalize_unit_tiempo_mappings(self) -> None:
        """Mapeos de unidades de tiempo."""
        assert normalize_unit("hora") == "HORA"
        assert normalize_unit("h") == "HORA"
        assert normalize_unit("dia") == "DIA"
        assert normalize_unit("d") == "DIA"

    def test_normalize_unit_masa_mappings(self) -> None:
        """Mapeos de unidades de masa."""
        assert normalize_unit("kg") == "KG"
        assert normalize_unit("kilogramo") == "KG"
        assert normalize_unit("ton") == "TON"
        assert normalize_unit("t") == "TON"

    def test_normalize_description_removes_diacritics(self) -> None:
        """Elimina diacríticos (NFD descomposición)."""
        # "café" con acento → "CAFE" sin acento
        result = normalize_description("café")
        assert "Á" not in result and "á" not in result
        assert "CAFE" in result or "CAF" in result

    def test_normalize_description_uppercase(self) -> None:
        """Convierte a mayúsculas."""
        result = normalize_description("Hormigón armado")
        assert result == result.upper()

    def test_normalize_codigo_keeps_allowed_chars(self) -> None:
        """Mantiene [\\w\\-.] (alfanuméricos, guión, punto)."""
        code = "APU-001.2-B"
        result = normalize_codigo(code)
        assert "-" in result
        assert "." in result

    def test_normalize_codigo_removes_invalid_chars(self) -> None:
        """Remueve caracteres inválidos."""
        code = "APU/001#@!"
        result = normalize_codigo(code)
        assert "/" not in result
        assert "#" not in result
        assert "@" not in result


# =============================================================================
# TEST SUITE 4: VALIDADOR NUMÉRICO
# =============================================================================


@pytest.mark.unit
class TestNumericValidator:
    """Verifica lógica de validación numérica."""

    def test_validate_non_negative_valid_float(self) -> None:
        """Acepta floats válidos."""
        result = NumericValidator.validate_non_negative(
            42.5, "test_field", 0.0, 100.0
        )
        assert result == 42.5

    def test_validate_non_negative_valid_int(self) -> None:
        """Acepta ints (convertidos a float)."""
        result = NumericValidator.validate_non_negative(
            42, "test_field", 0.0, 100.0
        )
        assert result == 42.0
        assert isinstance(result, float)

    def test_validate_non_negative_valid_decimal(self) -> None:
        """Acepta Decimal (convertidos a float)."""
        result = NumericValidator.validate_non_negative(
            Decimal("42.5"), "test_field"
        )
        assert result == 42.5

    def test_validate_non_negative_zero_valid(self) -> None:
        """Acepta 0.0."""
        result = NumericValidator.validate_non_negative(0.0, "test", 0.0)
        assert result == 0.0

    def test_validate_non_negative_below_min(self) -> None:
        """Rechaza valores < min_value."""
        with pytest.raises(ValidationError, match="mínimo"):
            NumericValidator.validate_non_negative(
                -1.0, "test", min_value=0.0
            )

    def test_validate_non_negative_above_max(self) -> None:
        """Rechaza valores > max_value."""
        with pytest.raises(ValidationError, match="máximo"):
            NumericValidator.validate_non_negative(
                101.0, "test", min_value=0.0, max_value=100.0
            )

    def test_validate_non_negative_inf(self) -> None:
        """Rechaza infinito."""
        with pytest.raises(ValidationError, match="finito"):
            NumericValidator.validate_non_negative(float('inf'), "test")

    def test_validate_non_negative_nan(self) -> None:
        """Rechaza NaN."""
        with pytest.raises(ValidationError, match="finito"):
            NumericValidator.validate_non_negative(float('nan'), "test")

    def test_validate_non_negative_non_numeric(self) -> None:
        """Rechaza strings y otros tipos."""
        with pytest.raises(ValidationError, match="numérico"):
            NumericValidator.validate_non_negative("42", "test")

        with pytest.raises(ValidationError, match="numérico"):
            NumericValidator.validate_non_negative(None, "test")

    def test_relative_difference_zero(self) -> None:
        """Diferencia relativa de valores iguales = 0."""
        assert NumericValidator.relative_difference(5.0, 5.0) == 0.0

    def test_relative_difference_symmetric(self) -> None:
        """Diferencia relativa es simétrica."""
        d1 = NumericValidator.relative_difference(1.0, 2.0)
        d2 = NumericValidator.relative_difference(2.0, 1.0)
        assert d1 == d2

    def test_relative_difference_epsilon_handling(self) -> None:
        """Maneja epsilon para valores cercanos a cero."""
        # Ambos ~0: should use epsilon como denominador
        epsilon = 1e-15
        d = NumericValidator.relative_difference(1e-20, 1e-20, epsilon=epsilon)
        assert d == 0.0

    def test_values_consistent_absolute_tolerance(self) -> None:
        """Modo tolerancia absoluta: |actual - expected| ≤ abs_tol."""
        ok, diff = NumericValidator.values_consistent(
            actual=0.0001,
            expected=0.0,
            rel_tolerance=0.1,
            abs_tolerance=0.001
        )
        assert ok

    def test_values_consistent_relative_tolerance(self) -> None:
        """Modo tolerancia relativa: Δ_rel ≤ rel_tol."""
        ok, diff = NumericValidator.values_consistent(
            actual=1.05,
            expected=1.0,
            rel_tolerance=0.1,  # 10%
            abs_tolerance=0.001
        )
        assert ok

    def test_values_consistent_both_fail(self) -> None:
        """Falla si ambas tolerancias se exceden."""
        ok, diff = NumericValidator.values_consistent(
            actual=10.0,
            expected=1.0,
            rel_tolerance=0.01,  # 1%
            abs_tolerance=0.1
        )
        assert not ok


# =============================================================================
# TEST SUITE 5: VALIDADOR STRING
# =============================================================================


@pytest.mark.unit
class TestStringValidator:
    """Verifica validación de strings."""

    def test_validate_non_empty_valid(self) -> None:
        """Acepta strings no vacíos."""
        result = StringValidator.validate_non_empty(
            "hello", "field"
        )
        assert result == "hello"

    def test_validate_non_empty_strips(self) -> None:
        """Strip por defecto."""
        result = StringValidator.validate_non_empty(
            "  hello  ", "field", strip=True
        )
        assert result == "hello"

    def test_validate_non_empty_no_strip(self) -> None:
        """Con strip=False preserva espacios."""
        result = StringValidator.validate_non_empty(
            "  hello  ", "field", strip=False
        )
        assert result == "  hello  "

    def test_validate_non_empty_empty_raises(self) -> None:
        """Rechaza strings vacíos."""
        with pytest.raises(ValidationError, match="vacío"):
            StringValidator.validate_non_empty("", "field")

        with pytest.raises(ValidationError, match="vacío"):
            StringValidator.validate_non_empty("   ", "field")

    def test_validate_non_empty_non_string_raises(self) -> None:
        """Rechaza no-strings."""
        with pytest.raises(ValidationError, match="str"):
            StringValidator.validate_non_empty(123, "field")

        with pytest.raises(ValidationError, match="str"):
            StringValidator.validate_non_empty(None, "field")

    def test_validate_non_empty_max_length(self) -> None:
        """Verifica longitud máxima."""
        result = StringValidator.validate_non_empty(
            "hello", "field", max_length=5
        )
        assert result == "hello"

        with pytest.raises(ValidationError, match="excede"):
            StringValidator.validate_non_empty(
                "hello world", "field", max_length=5
            )


# =============================================================================
# TEST SUITE 6: INVARIANTES FORMALES (I1-I6)
# =============================================================================


@pytest.mark.unit
class TestInvariants:
    """
    Verifica invariantes de dominio (I1-I6).
    """

    def test_invariant_i1_conservation_valid(self) -> None:
        """(I1) ley de conservación: valor_total ≈ cantidad × precio."""
        insumo = Suministro(
            codigo_apu="APU-001",
            descripcion_apu="Hormigón",
            unidad_apu="M3",
            descripcion_insumo="Acero",
            unidad_insumo="KG",
            cantidad=100.0,
            precio_unitario=50.0,
            valor_total=5000.0,
            tipo_insumo="SUMINISTRO",
        )

        assert insumo.is_valid
        invariants = insumo.check_invariants()
        assert any(bool(result) for inv, result in invariants if "Conservation" in inv)

    def test_invariant_i1_conservation_violated(self) -> None:
        """(I1) rechaza valor_total inconsistente."""
        with pytest.raises(InvariantError, match="conservación"):
            Suministro(
                codigo_apu="APU-001",
                descripcion_apu="Hormigón",
                unidad_apu="M3",
                descripcion_insumo="Acero",
                unidad_insumo="KG",
                cantidad=100.0,
                precio_unitario=50.0,
                valor_total=1000.0,  # ≠ 100 × 50
                tipo_insumo="SUMINISTRO",
            )

    def test_invariant_i2_non_negativity(self) -> None:
        """(I2) rechaza valores negativos."""
        with pytest.raises(ValidationError):
            Suministro(
                codigo_apu="APU-001",
                descripcion_apu="Hormigón",
                unidad_apu="M3",
                descripcion_insumo="Acero",
                unidad_insumo="KG",
                cantidad=-10.0,  # Negativo
                precio_unitario=50.0,
                valor_total=0.0,
                tipo_insumo="SUMINISTRO",
            )

    def test_invariant_i3_bounds_cantidad(self) -> None:
        """(I3) acotación: cantidad ∈ [0, 10^6]."""
        # Válido: cantidad = 10^6
        insumo = Suministro(
            codigo_apu="APU-001",
            descripcion_apu="Test",
            unidad_apu="UNIDAD",
            descripcion_insumo="Test",
            unidad_insumo="UNIDAD",
            cantidad=1_000_000.0,
            precio_unitario=1.0,
            valor_total=1_000_000.0,
            tipo_insumo="SUMINISTRO",
        )
        assert insumo.is_valid

        # Inválido: cantidad > 10^6
        with pytest.raises(ValidationError):
            Suministro(
                codigo_apu="APU-001",
                descripcion_apu="Test",
                unidad_apu="UNIDAD",
                descripcion_insumo="Test",
                unidad_insumo="UNIDAD",
                cantidad=1_000_001.0,
                precio_unitario=1.0,
                valor_total=1_000_001.0,
                tipo_insumo="SUMINISTRO",
            )

    def test_invariant_i5_typology(self) -> None:
        """(I5) tipo_insumo ∈ {5 tipos válidos}."""
        # Normalización automática a tipo válido
        insumo = Suministro(
            codigo_apu="APU-001",
            descripcion_apu="Test",
            unidad_apu="UNIDAD",
            descripcion_insumo="Test",
            unidad_insumo="UNIDAD",
            cantidad=1.0,
            precio_unitario=1.0,
            valor_total=1.0,
            tipo_insumo="suministro",  # Lowercase → normalizado
        )
        assert insumo.tipo_insumo == "SUMINISTRO"


# =============================================================================
# TEST SUITE 7: INSUMO PROCESADO
# =============================================================================


@pytest.mark.unit
class TestInsumoProcesado:
    """Verifica comportamiento de InsumoProcesado."""

    def test_insumo_valid_construction(self) -> None:
        """Construcción válida."""
        insumo = Suministro(
            codigo_apu="APU-001",
            descripcion_apu="Estructural",
            unidad_apu="M3",
            descripcion_insumo="Acero A36",
            unidad_insumo="KG",
            cantidad=500.0,
            precio_unitario=10.0,
            valor_total=5000.0,
            tipo_insumo="SUMINISTRO",
        )

        assert insumo.is_valid
        assert insumo.stratum == Stratum.PHYSICS
        assert len(insumo.id) > 0
        assert insumo.total_cost == 5000.0

    def test_insumo_normalization(self) -> None:
        """Los campos se normalizan automáticamente."""
        insumo = Suministro(
            codigo_apu="apu-001",  # Será normalizado
            descripcion_apu="Hormigón Armado",
            unidad_apu="m3",  # Será normalizado
            descripcion_insumo="Acero Corrugado",
            unidad_insumo="kg",  # Será normalizado
            cantidad=100.0,
            precio_unitario=50.0,
            valor_total=5000.0,
            tipo_insumo="suministro",  # Será normalizado
        )

        # Verificar normalización
        assert insumo.codigo_apu == normalize_codigo("apu-001")
        assert insumo.tipo_insumo == "SUMINISTRO"

    def test_insumo_total_cost_property(self) -> None:
        """Propiedad total_cost calcula cantidad × precio."""
        insumo = Suministro(
            codigo_apu="APU-001",
            descripcion_apu="Test",
            unidad_apu="UNIDAD",
            descripcion_insumo="Test",
            unidad_insumo="UNIDAD",
            cantidad=100.0,
            precio_unitario=25.0,
            valor_total=2500.0,
            tipo_insumo="SUMINISTRO",
        )

        assert insumo.total_cost == 2500.0
        assert insumo.total_cost == insumo.cantidad * insumo.precio_unitario

    def test_insumo_to_dict_excludes_internal(self) -> None:
        """to_dict() excluye campos internos (_*)."""
        insumo = Suministro(
            codigo_apu="APU-001",
            descripcion_apu="Test",
            unidad_apu="UNIDAD",
            descripcion_insumo="Test",
            unidad_insumo="UNIDAD",
            cantidad=1.0,
            precio_unitario=1.0,
            valor_total=1.0,
            tipo_insumo="SUMINISTRO",
        )

        d = insumo.to_dict()
        assert "_validated" not in d
        assert "_validation_errors" not in d
        assert "codigo_apu" in d

    def test_insumo_deterministic_id(self) -> None:
        """El ID es determinista (mismo contenido → mismo ID)."""
        data = {
            "codigo_apu": "APU-001",
            "descripcion_apu": "Test",
            "unidad_apu": "UNIDAD",
            "descripcion_insumo": "TestInsume",
            "unidad_insumo": "UNIDAD",
            "cantidad": 1.0,
            "precio_unitario": 1.0,
            "valor_total": 1.0,
            "tipo_insumo": "SUMINISTRO",
        }

        insumo1 = Suministro(**data)
        insumo2 = Suministro(**data)

        # Mismo contenido → mismo ID
        assert insumo1.id == insumo2.id


# =============================================================================
# TEST SUITE 8: SUBCLASES ESPECIALIZADAS
# =============================================================================


@pytest.mark.unit
class TestInsumoSubclasses:
    """Verifica comportamiento especializado de subclases."""

    def test_mano_de_obra_requires_rendimiento(self) -> None:
        """ManoDeObra tiene REQUIRES_RENDIMIENTO=True."""
        assert ManoDeObra.REQUIRES_RENDIMIENTO is True

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            insumo = ManoDeObra(
                codigo_apu="APU-001",
                descripcion_apu="Excavación",
                unidad_apu="HORA",
                descripcion_insumo="Obrero",
                unidad_insumo="HORA",
                cantidad=8.0,
                precio_unitario=25.0,
                valor_total=200.0,
                tipo_insumo="MANO_DE_OBRA",
                rendimiento=0.0,  # Debería generar warning
            )
            assert len(w) > 0

    def test_equipo_expected_units_tiempo(self) -> None:
        """Equipo espera unidades de tiempo."""
        assert Equipo.EXPECTED_UNITS == UNIDADES_TIEMPO

    def test_suministro_warning_cantidad_cero(self) -> None:
        """Suministro advierte si cantidad=0."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            insumo = Suministro(
                codigo_apu="APU-001",
                descripcion_apu="Test",
                unidad_apu="UNIDAD",
                descripcion_insumo="Test",
                unidad_insumo="UNIDAD",
                cantidad=0.0,  # Warning esperado
                precio_unitario=100.0,
                valor_total=0.0,
                tipo_insumo="SUMINISTRO",
            )
            assert len(w) > 0

    def test_transporte_expected_units(self) -> None:
        """Transporte espera unidades de transporte."""
        assert Transporte.EXPECTED_UNITS == UNIDADES_TRANSPORTE

    def test_otro_no_unit_restriction(self) -> None:
        """Otro acepta cualquier unidad."""
        assert Otro.EXPECTED_UNITS == frozenset()


# =============================================================================
# TEST SUITE 9: APU STRUCTURE
# =============================================================================


@pytest.mark.integration
class TestAPUStructure:
    """Verifica estructura y cálculos de APUStructure."""

    def test_apu_creation(self) -> None:
        """Creación básica de APU."""
        apu = APUStructure(
            id="APU-001",
            description="Excavación",
            unit="M3",
            quantity=100.0,
        )

        assert apu.stratum == Stratum.TACTICS
        assert apu.support_base_width == 0
        assert apu.total_cost == 0.0

    def test_apu_add_resource(self) -> None:
        """Agregar recursos a APU."""
        apu = APUStructure(description="Excavación")

        insumo = Suministro(
            codigo_apu="APU-001",
            descripcion_apu="Excavación",
            unidad_apu="M3",
            descripcion_insumo="Acero",
            unidad_insumo="KG",
            cantidad=100.0,
            precio_unitario=10.0,
            valor_total=1000.0,
            tipo_insumo="SUMINISTRO",
        )

        apu.add_resource(insumo)
        assert apu.support_base_width == 1
        assert apu.total_cost == 1000.0

    def test_apu_add_resource_wrong_stratum(self) -> None:
        """Rechaza nodos que no son PHYSICS."""
        apu = APUStructure(description="Test")

        # Intentar agregar otro APU (TACTICS)
        other_apu = APUStructure(description="Child")

        with pytest.raises(TypeError):
            apu.add_resource(other_apu)  # type: ignore

    def test_apu_is_inverted_pyramid(self) -> None:
        """Detecta pirámide invertida: cantidad > 1000, width = 1."""
        apu = APUStructure(
            description="Large", quantity=5000.0
        )

        insumo = Suministro(
            codigo_apu="APU-001",
            descripcion_apu="Test",
            unidad_apu="UNIDAD",
            descripcion_insumo="Single",
            unidad_insumo="UNIDAD",
            cantidad=1.0,
            precio_unitario=1.0,
            valor_total=1.0,
            tipo_insumo="SUMINISTRO",
        )

        apu.add_resource(insumo)
        assert apu.is_inverted_pyramid

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_apu_cost_breakdown(self) -> None:
        """Distribución de costos por tipo."""
        apu = APUStructure(description="Mixto")

        # Suministro: $1000
        s1 = Suministro(
            codigo_apu="APU-001", descripcion_apu="Test",
            unidad_apu="UNIDAD", descripcion_insumo="A",
            unidad_insumo="UNIDAD", cantidad=100.0,
            precio_unitario=10.0, valor_total=1000.0,
            tipo_insumo="SUMINISTRO"
        )

        # ManoDeObra: $2000
        m1 = ManoDeObra(
            codigo_apu="APU-001", descripcion_apu="Test",
            unidad_apu="HORA", descripcion_insumo="Obrero",
            unidad_insumo="HORA", cantidad=40.0,
            precio_unitario=50.0, valor_total=2000.0,
            tipo_insumo="MANO_DE_OBRA",
            rendimiento=1.0/40.0
        )

        apu.add_resource(s1)
        apu.add_resource(m1)

        breakdown = apu.get_cost_breakdown()
        assert breakdown["SUMINISTRO"] == 1000.0
        assert breakdown["MANO_DE_OBRA"] == 2000.0

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_apu_cost_fractions(self) -> None:
        """Fracciones de costo (suma ≈ 1.0)."""
        apu = APUStructure(description="Test")

        s1 = Suministro(
            codigo_apu="APU-001", descripcion_apu="Test",
            unidad_apu="UNIDAD", descripcion_insumo="A",
            unidad_insumo="UNIDAD", cantidad=100.0,
            precio_unitario=10.0, valor_total=1000.0,
            tipo_insumo="SUMINISTRO"
        )

        m1 = ManoDeObra(
            codigo_apu="APU-001", descripcion_apu="Test",
            unidad_apu="HORA", descripcion_insumo="Obrero",
            unidad_insumo="HORA", cantidad=40.0,
            precio_unitario=25.0, valor_total=1000.0,
            tipo_insumo="MANO_DE_OBRA",
            rendimiento=1.0/40.0
        )

        apu.add_resource(s1)
        apu.add_resource(m1)

        fractions = apu.get_cost_fractions()
        total_fraction = sum(fractions.values())
        assert abs(total_fraction - 1.0) < 1e-9


# =============================================================================
# TEST SUITE 10: TOPOLOGICAL STABILITY INDEX
# =============================================================================


@pytest.mark.integration
class TestTopologicalStabilityIndex:
    """
    Verifica cálculo del índice de estabilidad topológica Ψ.

    Teorema: Ψ = D^0.4 · H_norm^0.6

    donde:
        D = |tipos únicos| / 5  ∈ [0, 1]
        H_norm = H / ln(n)  ∈ [0, 1]
        H = -Σ p ln(p)  (entropía de Shannon)
    """

    def test_psi_zero_no_resources(self) -> None:
        """Ψ = 0 cuando no hay recursos (n=0)."""
        apu = APUStructure(description="Empty")
        assert apu.topological_stability_index() == 0.0

    def test_psi_zero_single_resource(self) -> None:
        """Ψ = epsilon cuando hay un solo recurso (n=1)."""
        apu = APUStructure(description="Single")

        insumo = Suministro(
            codigo_apu="APU-001", descripcion_apu="Test",
            unidad_apu="UNIDAD", descripcion_insumo="One",
            unidad_insumo="UNIDAD", cantidad=1.0,
            precio_unitario=1.0, valor_total=1.0,
            tipo_insumo="SUMINISTRO"
        )

        apu.add_resource(insumo)
        assert apu.topological_stability_index() == sys.float_info.epsilon

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_psi_zero_total_value_zero(self) -> None:
        """Ψ = epsilon cuando valor_total = 0, n=1."""
        apu = APUStructure(description="Zero")

        insumo = Suministro(
            codigo_apu="APU-001", descripcion_apu="Test",
            unidad_apu="UNIDAD", descripcion_insumo="Zero",
            unidad_insumo="UNIDAD", cantidad=0.0,
            precio_unitario=100.0, valor_total=0.0,
            tipo_insumo="SUMINISTRO"
        )

        apu.add_resource(insumo)
        # n=1
        psi = apu.topological_stability_index()
        assert psi == sys.float_info.epsilon

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_psi_increases_with_diversity(self) -> None:
        """Ψ aumenta con diversidad categórica."""
        # APU con un solo tipo
        apu1 = APUStructure(description="Mono")
        for i in range(5):
            s = Suministro(
                codigo_apu="APU-001", descripcion_apu="Test",
                unidad_apu="UNIDAD", descripcion_insumo=f"Item{i}",
                unidad_insumo="UNIDAD", cantidad=1.0,
                precio_unitario=1.0, valor_total=1.0,
                tipo_insumo="SUMINISTRO"
            )
            apu1.add_resource(s)
        
        psi1 = apu1.topological_stability_index()

        # APU con múltiples tipos
        apu2 = APUStructure(description="Diverse")
        tipos = ["SUMINISTRO", "MANO_DE_OBRA", "EQUIPO", "TRANSPORTE", "OTRO"]
        for i, tipo in enumerate(tipos):
            cls = INSUMO_CLASS_MAP[tipo]
            insumo = cls(
                codigo_apu="APU-002", descripcion_apu="Test",
                unidad_apu="UNIDAD", descripcion_insumo=f"Type{i}",
                unidad_insumo="UNIDAD", cantidad=1.0,
                precio_unitario=1.0, valor_total=1.0,
                tipo_insumo=tipo,
                rendimiento=1.0 if tipo == "MANO_DE_OBRA" else 0.0
            )
            apu2.add_resource(insumo)

        psi2 = apu2.topological_stability_index()

        # Mayor diversidad → mayor Ψ
        assert psi2 > psi1

    def test_psi_increases_with_entropy(self) -> None:
        """Ψ aumenta cuando costos están más distribuidos."""
        # Costos concentrados en un tipo
        apu1 = APUStructure(description="Concentrated")
        for i in range(5):
            s = Suministro(
                codigo_apu="APU-001", descripcion_apu="Test",
                unidad_apu="UNIDAD", descripcion_insumo=f"Item{i}",
                unidad_insumo="UNIDAD", cantidad=1.0 if i == 0 else 0.001,
                precio_unitario=1000.0 if i == 0 else 1.0,
                valor_total=1000.0 if i == 0 else 0.001,
                tipo_insumo="SUMINISTRO"
            )
            apu1.add_resource(s)

        psi1 = apu1.topological_stability_index()

        # Costos distribuidos equitativamente
        apu2 = APUStructure(description="Distributed")
        for i in range(5):
            s = Suministro(
                codigo_apu="APU-002", descripcion_apu="Test",
                unidad_apu="UNIDAD", descripcion_insumo=f"Item{i}",
                unidad_insumo="UNIDAD", cantidad=1.0,
                precio_unitario=200.0,
                valor_total=200.0,
                tipo_insumo="SUMINISTRO"
            )
            apu2.add_resource(s)

        psi2 = apu2.topological_stability_index()

        # Mayor distribución → mayor Ψ
        assert psi2 > psi1

    def test_psi_range(self) -> None:
        """Ψ ∈ [0, 1] siempre."""
        apu = APUStructure(description="Test")

        for i in range(10):
            s = Suministro(
                codigo_apu="APU-001", descripcion_apu="Test",
                unidad_apu="UNIDAD", descripcion_insumo=f"Item{i}",
                unidad_insumo="UNIDAD", cantidad=float(i + 1),
                precio_unitario=float(i + 1),
                valor_total=float((i + 1) ** 2),
                tipo_insumo="SUMINISTRO"
            )
            apu.add_resource(s)

        psi = apu.topological_stability_index()
        assert 0.0 <= psi <= 1.0


# =============================================================================
# TEST SUITE 11: FACTORY FUNCTIONS
# =============================================================================


@pytest.mark.unit
class TestFactoryFunctions:
    """Verifica factories de creación de insumos."""

    def test_validate_insumo_data_complete(self) -> None:
        """Valida datos completos."""
        raw_data = {
            "codigo_apu": "APU-001",
            "descripcion_apu": "Excavación",
            "unidad_apu": "m3",
            "descripcion_insumo": "Acero A36",
            "unidad_insumo": "kg",
            "cantidad": 500.0,
            "precio_unitario": 10.0,
            "valor_total": 5000.0,
            "tipo_insumo": "suministro",
        }

        cleaned = validate_insumo_data(raw_data)
        assert cleaned["codigo_apu"] == "APU-001"
        assert cleaned["tipo_insumo"] == "SUMINISTRO"
        assert cleaned["cantidad"] == 500.0

    def test_validate_insumo_data_missing_required(self) -> None:
        """Rechaza datos incompletos."""
        raw_data = {
            "codigo_apu": "APU-001",
            # Falta descripcion_apu y otros campos obligatorios
        }

        with pytest.raises(ValidationError, match="faltantes"):
            validate_insumo_data(raw_data)

    def test_validate_insumo_data_not_dict(self) -> None:
        """Rechaza si no es diccionario."""
        with pytest.raises(ValidationError, match="dict"):
            validate_insumo_data([1, 2, 3])  # type: ignore

    def test_validate_insumo_data_calculates_valor_total(self) -> None:
        """Calcula valor_total si falta."""
        raw_data = {
            "codigo_apu": "APU-001",
            "descripcion_apu": "Test",
            "unidad_apu": "UNIDAD",
            "descripcion_insumo": "Test",
            "unidad_insumo": "UNIDAD",
            "cantidad": 100.0,
            "precio_unitario": 50.0,
            # valor_total omitido
            "tipo_insumo": "SUMINISTRO",
        }

        cleaned = validate_insumo_data(raw_data)
        assert cleaned["valor_total"] == 5000.0

    def test_create_insumo_suministro(self) -> None:
        """Crea Suministro vía factory."""
        insumo = create_insumo(
            codigo_apu="APU-001",
            descripcion_apu="Test",
            unidad_apu="UNIDAD",
            descripcion_insumo="Test",
            unidad_insumo="UNIDAD",
            cantidad=1.0,
            precio_unitario=1.0,
            valor_total=1.0,
            tipo_insumo="SUMINISTRO",
        )

        assert isinstance(insumo, Suministro)
        assert insumo.is_valid

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_create_insumo_mano_de_obra(self) -> None:
        """Crea ManoDeObra vía factory."""
        insumo = create_insumo(
            codigo_apu="APU-001",
            descripcion_apu="Test",
            unidad_apu="HORA",
            descripcion_insumo="Test",
            unidad_insumo="HORA",
            cantidad=8.0,
            precio_unitario=25.0,
            valor_total=200.0,
            tipo_insumo="MANO_DE_OBRA",
            rendimiento=1.0/8.0
        )

        assert isinstance(insumo, ManoDeObra)

    def test_create_insumo_filters_kwargs(self) -> None:
        """Factory filtra kwargs a parámetros válidos."""
        insumo = create_insumo(
            codigo_apu="APU-001",
            descripcion_apu="Test",
            unidad_apu="UNIDAD",
            descripcion_insumo="Test",
            unidad_insumo="UNIDAD",
            cantidad=1.0,
            precio_unitario=1.0,
            valor_total=1.0,
            tipo_insumo="SUMINISTRO",
            invalid_param="should_be_ignored",  # Será ignorado
        )

        assert insumo.is_valid

    def test_create_insumo_from_raw(self) -> None:
        """Pipeline completo: raw → validar → crear."""
        raw_data = {
            "codigo_apu": "APU-001",
            "descripcion_apu": "Excavación",
            "unidad_apu": "m3",
            "descripcion_insumo": "Acero",
            "unidad_insumo": "kg",
            "cantidad": 500.0,
            "precio_unitario": 10.0,
            "valor_total": 5000.0,
            "tipo_insumo": "suministro",
        }

        insumo = create_insumo_from_raw(raw_data)
        assert isinstance(insumo, Suministro)
        assert insumo.is_valid


# =============================================================================
# TEST SUITE 12: UTILIDADES
# =============================================================================


@pytest.mark.unit
class TestUtilityFunctions:
    """Verifica funciones de utilidad."""

    def test_get_tipo_insumo_class_suministro(self) -> None:
        """Resuelve clase Suministro."""
        cls = get_tipo_insumo_class("SUMINISTRO")
        assert cls is Suministro

    def test_get_tipo_insumo_class_mano_de_obra(self) -> None:
        """Resuelve clase ManoDeObra."""
        cls = get_tipo_insumo_class("mano_de_obra")
        assert cls is ManoDeObra

    def test_get_tipo_insumo_class_invalid(self) -> None:
        """Rechaza tipo inválido."""
        with pytest.raises(InvalidTipoInsumoError):
            get_tipo_insumo_class("INVALIDO")

    def test_get_all_tipo_insumo_values(self) -> None:
        """Retorna 5 valores."""
        values = get_all_tipo_insumo_values()
        assert len(values) == 5
        assert "SUMINISTRO" in values

    def test_is_valid_tipo_insumo_valid(self) -> None:
        """Retorna True para tipos válidos."""
        assert is_valid_tipo_insumo("SUMINISTRO")
        assert is_valid_tipo_insumo("suministro")

    def test_is_valid_tipo_insumo_invalid(self) -> None:
        """Retorna False para tipos inválidos."""
        assert not is_valid_tipo_insumo("INVALIDO")


# =============================================================================
# TEST SUITE 13: EDGE CASES Y DEGENERACIÓN
# =============================================================================


@pytest.mark.unit
class TestEdgeCasesAndDegeneracy:
    """Verifica comportamiento en casos límite."""

    def test_zero_precio_valid(self) -> None:
        """Precio = 0 es válido (donación)."""
        insumo = Suministro(
            codigo_apu="APU-001",
            descripcion_apu="Test",
            unidad_apu="UNIDAD",
            descripcion_insumo="Free",
            unidad_insumo="UNIDAD",
            cantidad=100.0,
            precio_unitario=0.0,
            valor_total=0.0,
            tipo_insumo="SUMINISTRO",
        )
        assert insumo.is_valid

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_zero_cantidad_valid(self) -> None:
        """Cantidad = 0 es válida."""
        insumo = Suministro(
            codigo_apu="APU-001",
            descripcion_apu="Test",
            unidad_apu="UNIDAD",
            descripcion_insumo="None",
            unidad_insumo="UNIDAD",
            cantidad=0.0,
            precio_unitario=100.0,
            valor_total=0.0,
            tipo_insumo="SUMINISTRO",
        )
        assert insumo.is_valid

    def test_very_large_values(self) -> None:
        """Valores máximos permitidos."""
        insumo = Suministro(
            codigo_apu="APU-001",
            descripcion_apu="Test",
            unidad_apu="UNIDAD",
            descripcion_insumo="Max",
            unidad_insumo="UNIDAD",
            cantidad=_BOUNDS.cantidad_max,
            precio_unitario=_BOUNDS.precio_max,
            valor_total=_BOUNDS.valor_total_max,
            tipo_insumo="SUMINISTRO",
        )
        assert insumo.is_valid

    def test_very_small_nonzero_values(self) -> None:
        """Valores muy pequeños (pero > 0)."""
        insumo = Suministro(
            codigo_apu="APU-001",
            descripcion_apu="Test",
            unidad_apu="UNIDAD",
            descripcion_insumo="Tiny",
            unidad_insumo="UNIDAD",
            cantidad=1e-10,
            precio_unitario=1e-10,
            valor_total=1e-20,
            tipo_insumo="SUMINISTRO",
        )
        assert insumo.is_valid

    def test_hash_length_consistent(self) -> None:
        """Hash tiene longitud fija."""
        h1 = _deterministic_short_hash("test1")
        h2 = _deterministic_short_hash("test2")
        assert len(h1) == 10
        assert len(h2) == 10

    def test_hash_deterministic(self) -> None:
        """Hash es determinista."""
        h1 = _deterministic_short_hash("same_text")
        h2 = _deterministic_short_hash("same_text")
        assert h1 == h2

    def test_hash_different_inputs(self) -> None:
        """Inputs diferentes → hashes diferentes."""
        h1 = _deterministic_short_hash("text1")
        h2 = _deterministic_short_hash("text2")
        assert h1 != h2


# =============================================================================
# TEST SUITE 14: TOLERANCIAS Y CONSERVACIÓN
# =============================================================================


@pytest.mark.unit
class TestTolerancesAndConservation:
    """Verifica tolerancias híbridas y ley de conservación."""

    def test_conservation_within_absolute_tolerance(self) -> None:
        """Valores cercanos a cero son consistentes por tolerancia absoluta."""
        # Ambos valores << tolerance absoluta → OK
        ok, _ = NumericValidator.values_consistent(
            actual=1e-12,
            expected=1e-12,
            abs_tolerance=_TOLERANCES.conservation_absolute,
            rel_tolerance=_TOLERANCES.conservation_relative,
        )
        assert ok

    def test_conservation_within_relative_tolerance(self) -> None:
        """Valores grandes son consistentes por tolerancia relativa."""
        expected = 1e9
        actual = expected * (1 + 1e-7)  # 0.00001% de error relativo

        ok, rel_diff = NumericValidator.values_consistent(
            actual=actual,
            expected=expected,
            abs_tolerance=_TOLERANCES.conservation_absolute,
            rel_tolerance=_TOLERANCES.conservation_relative,
        )
        assert ok

    def test_conservation_exceeds_both_tolerances(self) -> None:
        """Falla si excede ambas tolerancias."""
        ok, _ = NumericValidator.values_consistent(
            actual=1.0,
            expected=100.0,
            abs_tolerance=0.01,
            rel_tolerance=0.01,  # 1%
        )
        assert not ok


# =============================================================================
# TEST SUITE 15: INTEGRACIÓN END-TO-END
# =============================================================================


@pytest.mark.integration
class TestEndToEndIntegration:
    """Tests de integración completa."""

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_complete_presupuesto_creation(self) -> None:
        """
        Crea un presupuesto completo:
        - Múltiples APUs
        - Múltiples tipos de insumos
        - Verifica invariantes
        - Calcula estabilidad
        """
        # Crear insumos crudos
        raw_insumos = [
            {
                "codigo_apu": "APU-001",
                "descripcion_apu": "Cimentación",
                "unidad_apu": "M3",
                "descripcion_insumo": "Hormigón",
                "unidad_insumo": "M3",
                "cantidad": 50.0,
                "precio_unitario": 500.0,
                "tipo_insumo": "SUMINISTRO",
            },
            {
                "codigo_apu": "APU-001",
                "descripcion_apu": "Cimentación",
                "unidad_apu": "M3",
                "descripcion_insumo": "Obrero",
                "unidad_insumo": "HORA",
                "cantidad": 200.0,
                "precio_unitario": 25.0,
                "tipo_insumo": "MANO_DE_OBRA",
                "rendimiento": 1.0/200.0
            },
            {
                "codigo_apu": "APU-002",
                "descripcion_apu": "Estructura",
                "unidad_apu": "M2",
                "descripcion_insumo": "Acero A36",
                "unidad_insumo": "KG",
                "cantidad": 1000.0,
                "precio_unitario": 10.0,
                "tipo_insumo": "SUMINISTRO",
            },
        ]

        # Crear insumos
        insumos = []
        for raw in raw_insumos:
            insumo = create_insumo_from_raw(raw)
            insumos.append(insumo)
            assert insumo.is_valid

        # Agrupar en APUs
        apu_001 = APUStructure(description="Cimentación", quantity=50.0)
        apu_002 = APUStructure(description="Estructura", quantity=200.0)

        for insumo in insumos[:2]:
            apu_001.add_resource(insumo)

        for insumo in insumos[2:]:
            apu_002.add_resource(insumo)

        # Verificar
        assert apu_001.support_base_width == 2
        assert apu_002.support_base_width == 1
        assert apu_001.total_cost == 50*500 + 200*25  # 27500
        assert apu_002.total_cost == 1000*10  # 10000

        # Estabilidad
        psi_001 = apu_001.topological_stability_index()
        psi_002 = apu_002.topological_stability_index()

        assert 0 <= psi_001 <= 1
        assert 0 <= psi_002 <= 1
        assert psi_001 > psi_002  # Más diversidad → más estabilidad


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])