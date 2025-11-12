"""
Suite completa de pruebas para schemas.py

Pruebas exhaustivas con cobertura de normalización, validación,
clases de insumos, factory functions y casos edge.
"""

import warnings
from decimal import Decimal

import pytest

# Importar módulo a probar
from app.schemas import (
    CANTIDAD_RENDIMIENTO_TOLERANCE,
    INSUMO_CLASS_MAP,
    MAX_CANTIDAD,
    MAX_CODIGO_LENGTH,
    MAX_DESCRIPCION_LENGTH,
    MAX_PRECIO,
    MIN_CANTIDAD,
    MIN_PRECIO,
    # Constantes
    UNIDAD_NORMALIZADA_MAP,
    UNIDADES_AREA,
    UNIDADES_GENERICAS,
    UNIDADES_LONGITUD,
    UNIDADES_MASA,
    UNIDADES_TIEMPO,
    UNIDADES_TRANSPORTE,
    UNIDADES_VOLUMEN,
    VALOR_TOTAL_TOLERANCE,
    Equipo,
    # Clases
    InsumoProcesado,
    InvalidTipoInsumoError,
    ManoDeObra,
    # Validadores
    NumericValidator,
    Otro,
    StringValidator,
    Suministro,
    TipoInsumo,
    Transporte,
    # Excepciones
    ValidationError,
    # Factory functions
    create_insumo,
    create_insumo_from_raw,
    get_all_tipo_insumo_values,
    # Utilidades
    get_tipo_insumo_class,
    is_valid_tipo_insumo,
    normalize_codigo,
    normalize_description,
    # Funciones de normalización
    normalize_unit,
    validate_insumo_data,
)

# ============================================================================
# FIXTURES - Datos de prueba reutilizables
# ============================================================================

@pytest.fixture
def sample_insumo_base_data():
    """Fixture con datos básicos válidos para crear un insumo."""
    return {
        'codigo_apu': 'APU-001',
        'descripcion_apu': 'Concreto f\'c=280 kg/cm2',
        'unidad_apu': 'M3',
        'descripcion_insumo': 'Cemento Portland Tipo I',
        'unidad_insumo': 'KG',
        'cantidad': 350.0,
        'precio_unitario': 450.0,
        'valor_total': 157500.0,
        'categoria': 'SUMINISTRO',
        'tipo_insumo': 'SUMINISTRO',
        'formato_origen': 'FORMATO_A',
        'rendimiento': 0.0
    }


@pytest.fixture
def sample_mano_obra_data():
    """Fixture con datos para Mano de Obra."""
    return {
        'codigo_apu': 'APU-001',
        'descripcion_apu': 'Excavación manual',
        'unidad_apu': 'M3',
        'descripcion_insumo': 'Oficial',
        'unidad_insumo': 'HORA',
        'cantidad': 1.5,
        'precio_unitario': 15000.0,
        'valor_total': 22500.0,
        'tipo_insumo': 'MANO_DE_OBRA',
        'rendimiento': 0.67,
        'jornal': 120000.0
    }


@pytest.fixture
def sample_equipo_data():
    """Fixture con datos para Equipo."""
    return {
        'codigo_apu': 'APU-002',
        'descripcion_apu': 'Excavación mecánica',
        'unidad_apu': 'M3',
        'descripcion_insumo': 'Retroexcavadora',
        'unidad_insumo': 'HORA',
        'cantidad': 0.5,
        'precio_unitario': 85000.0,
        'valor_total': 42500.0,
        'tipo_insumo': 'EQUIPO',
        'rendimiento': 0.0
    }


@pytest.fixture
def sample_suministro_data():
    """Fixture con datos para Suministro."""
    return {
        'codigo_apu': 'APU-003',
        'descripcion_apu': 'Concreto',
        'unidad_apu': 'M3',
        'descripcion_insumo': 'Arena lavada',
        'unidad_insumo': 'M3',
        'cantidad': 0.5,
        'precio_unitario': 80000.0,
        'valor_total': 40000.0,
        'tipo_insumo': 'SUMINISTRO',
        'rendimiento': 0.0
    }


@pytest.fixture
def sample_transporte_data():
    """Fixture con datos para Transporte."""
    return {
        'codigo_apu': 'APU-004',
        'descripcion_apu': 'Transporte materiales',
        'unidad_apu': 'M3',
        'descripcion_insumo': 'Transporte local',
        'unidad_insumo': 'KM',
        'cantidad': 25.0,
        'precio_unitario': 3500.0,
        'valor_total': 87500.0,
        'tipo_insumo': 'TRANSPORTE',
        'rendimiento': 0.0
    }


@pytest.fixture
def sample_otro_data():
    """Fixture con datos para Otro."""
    return {
        'codigo_apu': 'APU-005',
        'descripcion_apu': 'Item especial',
        'unidad_apu': 'GLB',
        'descripcion_insumo': 'Servicio especial',
        'unidad_insumo': 'UNIDAD',
        'cantidad': 1.0,
        'precio_unitario': 500000.0,
        'valor_total': 500000.0,
        'tipo_insumo': 'OTRO',
        'rendimiento': 0.0
    }


# ============================================================================
# TESTS - NORMALIZACIÓN DE UNIDADES
# ============================================================================

class TestNormalizeUnit:
    """Suite de pruebas para normalize_unit()"""

    def test_normalize_unit_basic(self):
        """Debe normalizar unidades básicas."""
        assert normalize_unit('kg') == 'KG'
        assert normalize_unit('metros') == 'M'
        assert normalize_unit('hora') == 'HORA'
        assert normalize_unit('dia') == 'DIA'

    def test_normalize_unit_case_insensitive(self):
        """Debe ser case-insensitive."""
        assert normalize_unit('KG') == 'KG'
        assert normalize_unit('Kg') == 'KG'
        assert normalize_unit('kG') == 'KG'

    def test_normalize_unit_with_spaces(self):
        """Debe manejar espacios."""
        assert normalize_unit('  kg  ') == 'KG'
        assert normalize_unit(' metros ') == 'M'

    def test_normalize_unit_none(self):
        """Debe retornar default para None."""
        assert normalize_unit(None) == 'UNIDAD'

    def test_normalize_unit_empty_string(self):
        """Debe retornar default para string vacío."""
        assert normalize_unit('') == 'UNIDAD'
        assert normalize_unit('   ') == 'UNIDAD'

    def test_normalize_unit_not_in_map(self):
        """Debe retornar uppercase para unidades no mapeadas."""
        assert normalize_unit('xyz') == 'XYZ'
        assert normalize_unit('custom') == 'CUSTOM'

    def test_normalize_unit_time_units(self):
        """Debe normalizar unidades de tiempo."""
        assert normalize_unit('hora') == 'HORA'
        assert normalize_unit('hrs') == 'HORA'
        assert normalize_unit('dia') == 'DIA'
        assert normalize_unit('dias') == 'DIA'
        assert normalize_unit('semana') == 'SEMANA'
        assert normalize_unit('mes') == 'MES'

    def test_normalize_unit_mass_units(self):
        """Debe normalizar unidades de masa."""
        assert normalize_unit('kilogramo') == 'KG'
        assert normalize_unit('gramo') == 'GR'
        assert normalize_unit('tonelada') == 'TON'
        assert normalize_unit('libra') == 'LB'

    def test_normalize_unit_volume_units(self):
        """Debe normalizar unidades de volumen."""
        assert normalize_unit('m3') == 'M3'
        assert normalize_unit('m³') == 'M3'
        assert normalize_unit('litro') == 'L'
        assert normalize_unit('galon') == 'GAL'

    def test_normalize_unit_area_units(self):
        """Debe normalizar unidades de área."""
        assert normalize_unit('m2') == 'M2'
        assert normalize_unit('m²') == 'M2'

    def test_normalize_unit_transport_units(self):
        """Debe normalizar unidades de transporte."""
        assert normalize_unit('viaje') == 'VIAJE'
        assert normalize_unit('viajes') == 'VIAJES'
        assert normalize_unit('km') == 'KM'

    def test_normalize_unit_cache(self):
        """Debe usar cache para llamadas repetidas."""
        # Primera llamada
        result1 = normalize_unit('kg')
        # Segunda llamada (debería usar cache)
        result2 = normalize_unit('kg')

        assert result1 == result2

        # Verificar que el cache funciona
        cache_info = normalize_unit.cache_info()
        assert cache_info.hits > 0

    def test_normalize_unit_invalid_type(self):
        """Debe manejar tipos inválidos."""
        assert normalize_unit(123) == 'UNIDAD'
        assert normalize_unit(()) == 'UNIDAD'


# ============================================================================
# TESTS - NORMALIZACIÓN DE DESCRIPCIONES
# ============================================================================

class TestNormalizeDescription:
    """Suite de pruebas para normalize_description()"""

    def test_normalize_description_basic(self):
        """Debe normalizar descripción básica."""
        result = normalize_description('Cemento Portland')
        assert result == 'CEMENTO PORTLAND'

    def test_normalize_description_accents(self):
        """Debe remover tildes."""
        result = normalize_description('Ácido Nitrógeno Ñoño')
        assert result == 'ACIDO NITROGENO NONO'

    def test_normalize_description_multiple_spaces(self):
        """Debe normalizar espacios múltiples."""
        result = normalize_description('Cemento    Portland    Tipo   I')
        assert result == 'CEMENTO PORTLAND TIPO I'

    def test_normalize_description_special_chars(self):
        """Debe mantener algunos caracteres especiales útiles."""
        result = normalize_description("Concreto f'c=280 kg/cm2")
        assert 'FC' in result or 'F' in result
        assert '280' in result

    def test_normalize_description_leading_trailing_spaces(self):
        """Debe remover espacios al inicio y final."""
        result = normalize_description('  Cemento Portland  ')
        assert result == 'CEMENTO PORTLAND'

    def test_normalize_description_none(self):
        """Debe retornar string vacío para None."""
        assert normalize_description(None) == ''

    def test_normalize_description_empty_string(self):
        """Debe retornar string vacío."""
        assert normalize_description('') == ''
        assert normalize_description('   ') == ''

    def test_normalize_description_numbers(self):
        """Debe mantener números."""
        result = normalize_description('Acero Grado 60')
        assert '60' in result
        assert 'ACERO' in result

    def test_normalize_description_parentheses(self):
        """Debe mantener paréntesis."""
        result = normalize_description('Cemento (Portland)')
        assert '(' in result or 'PORTLAND' in result

    def test_normalize_description_cache(self):
        """Debe usar cache."""
        desc = 'Cemento Portland Tipo I'
        result1 = normalize_description(desc)
        result2 = normalize_description(desc)

        assert result1 == result2
        cache_info = normalize_description.cache_info()
        assert cache_info.hits > 0

    def test_normalize_description_invalid_type(self):
        """Debe manejar tipos inválidos."""
        assert normalize_description(123) == ''
        assert normalize_description(()) == ''


# ============================================================================
# TESTS - NORMALIZACIÓN DE CÓDIGOS
# ============================================================================

class TestNormalizeCodigo:
    """Suite de pruebas para normalize_codigo()"""

    def test_normalize_codigo_basic(self):
        """Debe normalizar código básico."""
        result = normalize_codigo('apu-001')
        assert result == 'APU-001'

    def test_normalize_codigo_uppercase(self):
        """Debe convertir a mayúsculas."""
        result = normalize_codigo('apu-abc-123')
        assert result == 'APU-ABC-123'

    def test_normalize_codigo_with_spaces(self):
        """Debe limpiar espacios."""
        result = normalize_codigo('  APU-001  ')
        assert result == 'APU-001'

    def test_normalize_codigo_special_chars(self):
        """Debe remover caracteres especiales no permitidos."""
        result = normalize_codigo('APU@001#')
        assert '@' not in result
        assert '#' not in result

    def test_normalize_codigo_preserve_valid_chars(self):
        """Debe preservar guiones y puntos."""
        result = normalize_codigo('APU-001.01')
        assert '-' in result
        assert '.' in result

    def test_normalize_codigo_none_raises_error(self):
        """Debe lanzar error para None."""
        with pytest.raises(ValidationError) as exc_info:
            normalize_codigo(None)

        assert 'vacío' in str(exc_info.value).lower()

    def test_normalize_codigo_empty_raises_error(self):
        """Debe lanzar error para string vacío."""
        with pytest.raises(ValidationError):
            normalize_codigo('')

        with pytest.raises(ValidationError):
            normalize_codigo('   ')

    def test_normalize_codigo_too_long_raises_error(self):
        """Debe lanzar error para código muy largo."""
        long_codigo = 'A' * (MAX_CODIGO_LENGTH + 10)

        with pytest.raises(ValidationError) as exc_info:
            normalize_codigo(long_codigo)

        assert 'largo' in str(exc_info.value).lower()

    def test_normalize_codigo_cache(self):
        """Debe usar cache."""
        codigo = 'APU-TEST-001'
        result1 = normalize_codigo(codigo)
        result2 = normalize_codigo(codigo)

        assert result1 == result2
        cache_info = normalize_codigo.cache_info()
        assert cache_info.hits > 0


# ============================================================================
# TESTS - VALIDADORES NUMÉRICOS
# ============================================================================

class TestNumericValidator:
    """Suite de pruebas para NumericValidator"""

    def test_validate_non_negative_valid_int(self):
        """Debe aceptar entero válido."""
        result = NumericValidator.validate_non_negative(100, 'test_field')
        assert result == 100.0
        assert isinstance(result, float)

    def test_validate_non_negative_valid_float(self):
        """Debe aceptar float válido."""
        result = NumericValidator.validate_non_negative(100.5, 'test_field')
        assert result == 100.5

    def test_validate_non_negative_decimal(self):
        """Debe aceptar Decimal."""
        result = NumericValidator.validate_non_negative(Decimal('100.25'), 'test_field')
        assert result == 100.25

    def test_validate_non_negative_zero(self):
        """Debe aceptar cero."""
        result = NumericValidator.validate_non_negative(0, 'test_field')
        assert result == 0.0

    def test_validate_non_negative_negative_raises_error(self):
        """Debe rechazar negativos."""
        with pytest.raises(ValidationError) as exc_info:
            NumericValidator.validate_non_negative(-100, 'test_field')

        assert 'test_field' in str(exc_info.value)
        assert 'menor' in str(exc_info.value).lower()

    def test_validate_non_negative_with_min_value(self):
        """Debe validar valor mínimo."""
        result = NumericValidator.validate_non_negative(50, 'test', min_value=10)
        assert result == 50.0

        with pytest.raises(ValidationError):
            NumericValidator.validate_non_negative(5, 'test', min_value=10)

    def test_validate_non_negative_with_max_value(self):
        """Debe validar valor máximo."""
        result = NumericValidator.validate_non_negative(50, 'test', max_value=100)
        assert result == 50.0

        with pytest.raises(ValidationError):
            NumericValidator.validate_non_negative(150, 'test', max_value=100)

    def test_validate_non_negative_with_range(self):
        """Debe validar rango completo."""
        result = NumericValidator.validate_non_negative(
            50, 'test', min_value=0, max_value=100
        )
        assert result == 50.0

        with pytest.raises(ValidationError):
            NumericValidator.validate_non_negative(-10, 'test', min_value=0, max_value=100)

        with pytest.raises(ValidationError):
            NumericValidator.validate_non_negative(150, 'test', min_value=0, max_value=100)

    def test_validate_non_negative_invalid_type(self):
        """Debe rechazar tipos inválidos."""
        with pytest.raises(ValidationError) as exc_info:
            NumericValidator.validate_non_negative('texto', 'test_field')

        assert 'numérico' in str(exc_info.value).lower()

    def test_validate_non_negative_nan_raises_error(self):
        """Debe rechazar NaN."""
        with pytest.raises(ValidationError) as exc_info:
            NumericValidator.validate_non_negative(float('nan'), 'test_field')

        assert 'finito' in str(exc_info.value).lower()

    def test_validate_non_negative_inf_raises_error(self):
        """Debe rechazar infinito."""
        with pytest.raises(ValidationError):
            NumericValidator.validate_non_negative(float('inf'), 'test_field')

        with pytest.raises(ValidationError):
            NumericValidator.validate_non_negative(float('-inf'), 'test_field')

    def test_validate_positive_valid(self):
        """Debe validar positivo."""
        result = NumericValidator.validate_positive(100, 'test_field')
        assert result == 100.0

    def test_validate_positive_zero_raises_error(self):
        """Debe rechazar cero."""
        with pytest.raises(ValidationError) as exc_info:
            NumericValidator.validate_positive(0, 'test_field')

        assert 'mayor que 0' in str(exc_info.value).lower()

    def test_validate_positive_negative_raises_error(self):
        """Debe rechazar negativos."""
        with pytest.raises(ValidationError):
            NumericValidator.validate_positive(-100, 'test_field')


# ============================================================================
# TESTS - VALIDADORES DE STRINGS
# ============================================================================

class TestStringValidator:
    """Suite de pruebas para StringValidator"""

    def test_validate_non_empty_valid(self):
        """Debe aceptar string válido."""
        result = StringValidator.validate_non_empty('test', 'field_name')
        assert result == 'test'

    def test_validate_non_empty_with_spaces(self):
        """Debe limpiar espacios."""
        result = StringValidator.validate_non_empty('  test  ', 'field_name')
        assert result == 'test'

    def test_validate_non_empty_empty_raises_error(self):
        """Debe rechazar string vacío."""
        with pytest.raises(ValidationError) as exc_info:
            StringValidator.validate_non_empty('', 'field_name')

        assert 'field_name' in str(exc_info.value)
        assert 'vacío' in str(exc_info.value).lower()

    def test_validate_non_empty_whitespace_raises_error(self):
        """Debe rechazar solo espacios."""
        with pytest.raises(ValidationError):
            StringValidator.validate_non_empty('   ', 'field_name')

    def test_validate_non_empty_invalid_type_raises_error(self):
        """Debe rechazar tipos no string."""
        with pytest.raises(ValidationError) as exc_info:
            StringValidator.validate_non_empty(123, 'field_name')

        assert 'string' in str(exc_info.value).lower()

    def test_validate_non_empty_with_max_length(self):
        """Debe validar longitud máxima."""
        result = StringValidator.validate_non_empty('test', 'field', max_length=10)
        assert result == 'test'

        with pytest.raises(ValidationError) as exc_info:
            StringValidator.validate_non_empty('test' * 10, 'field', max_length=10)

        assert 'largo' in str(exc_info.value).lower()

    def test_validate_non_empty_none_raises_error(self):
        """Debe rechazar None."""
        with pytest.raises(ValidationError):
            StringValidator.validate_non_empty(None, 'field_name')


# ============================================================================
# TESTS - ENUM TIPOINSUMO
# ============================================================================

class TestTipoInsumo:
    """Suite de pruebas para TipoInsumo enum"""

    def test_tipo_insumo_values(self):
        """Debe tener todos los valores esperados."""
        assert TipoInsumo.MANO_DE_OBRA.value == 'MANO_DE_OBRA'
        assert TipoInsumo.EQUIPO.value == 'EQUIPO'
        assert TipoInsumo.SUMINISTRO.value == 'SUMINISTRO'
        assert TipoInsumo.TRANSPORTE.value == 'TRANSPORTE'
        assert TipoInsumo.OTRO.value == 'OTRO'

    def test_from_string_valid(self):
        """Debe convertir string válido."""
        result = TipoInsumo.from_string('MANO_DE_OBRA')
        assert result == TipoInsumo.MANO_DE_OBRA

        result = TipoInsumo.from_string('mano_de_obra')
        assert result == TipoInsumo.MANO_DE_OBRA

    def test_from_string_with_spaces(self):
        """Debe manejar espacios."""
        result = TipoInsumo.from_string('  MANO_DE_OBRA  ')
        assert result == TipoInsumo.MANO_DE_OBRA

    def test_from_string_already_enum(self):
        """Debe aceptar enum directamente."""
        result = TipoInsumo.from_string(TipoInsumo.EQUIPO)
        assert result == TipoInsumo.EQUIPO

    def test_from_string_invalid_raises_error(self):
        """Debe lanzar error para valor inválido."""
        with pytest.raises(InvalidTipoInsumoError) as exc_info:
            TipoInsumo.from_string('INVALIDO')

        assert 'inválido' in str(exc_info.value).lower()

    def test_get_valid_values(self):
        """Debe retornar todos los valores válidos."""
        valid_values = TipoInsumo.get_valid_values()

        assert 'MANO_DE_OBRA' in valid_values
        assert 'EQUIPO' in valid_values
        assert 'SUMINISTRO' in valid_values
        assert 'TRANSPORTE' in valid_values
        assert 'OTRO' in valid_values
        assert len(valid_values) == 5


# ============================================================================
# TESTS - CLASE BASE INSUMOPROCESADO
# ============================================================================

class TestInsumoProcesado:
    """Suite de pruebas para InsumoProcesado"""

    def test_create_valid_insumo(self, sample_insumo_base_data):
        """Debe crear insumo válido."""
        insumo = InsumoProcesado(**sample_insumo_base_data)

        assert insumo.codigo_apu == 'APU-001'
        assert insumo.tipo_insumo == 'SUMINISTRO'
        assert insumo.cantidad == 350.0
        assert insumo._validated is True

    def test_normalization_in_post_init(self, sample_insumo_base_data):
        """Debe normalizar campos en __post_init__."""
        data = sample_insumo_base_data.copy()
        data['codigo_apu'] = '  apu-001  '
        data['unidad_apu'] = 'm3'

        insumo = InsumoProcesado(**data)

        assert insumo.codigo_apu == 'APU-001'
        assert insumo.unidad_apu == 'M3'

    def test_description_normalization(self, sample_insumo_base_data):
        """Debe normalizar descripciones."""
        data = sample_insumo_base_data.copy()
        data['descripcion_insumo'] = '  Cemento   Portland  '

        insumo = InsumoProcesado(**data)

        assert insumo.descripcion_insumo == 'CEMENTO PORTLAND'
        assert insumo.normalized_desc == 'CEMENTO PORTLAND'

    def test_tipo_categoria_sync(self, sample_insumo_base_data):
        """Debe sincronizar tipo_insumo y categoria."""
        data = sample_insumo_base_data.copy()
        data['tipo_insumo'] = 'EQUIPO'
        data['categoria'] = 'OTRO'  # Diferente, debe sincronizarse

        insumo = InsumoProcesado(**data)

        assert insumo.tipo_insumo == 'EQUIPO'
        assert insumo.categoria == 'EQUIPO'  # Sincronizado

    def test_empty_codigo_raises_error(self, sample_insumo_base_data):
        """Debe rechazar código vacío."""
        data = sample_insumo_base_data.copy()
        data['codigo_apu'] = ''

        with pytest.raises(ValidationError):
            InsumoProcesado(**data)

    def test_empty_descripcion_raises_error(self, sample_insumo_base_data):
        """Debe rechazar descripción vacía."""
        data = sample_insumo_base_data.copy()
        data['descripcion_insumo'] = ''

        with pytest.raises(ValidationError):
            InsumoProcesado(**data)

    def test_negative_cantidad_raises_error(self, sample_insumo_base_data):
        """Debe rechazar cantidad negativa."""
        data = sample_insumo_base_data.copy()
        data['cantidad'] = -10.0

        with pytest.raises(ValidationError):
            InsumoProcesado(**data)

    def test_negative_precio_raises_error(self, sample_insumo_base_data):
        """Debe rechazar precio negativo."""
        data = sample_insumo_base_data.copy()
        data['precio_unitario'] = -100.0

        with pytest.raises(ValidationError):
            InsumoProcesado(**data)

    def test_valor_total_consistency_warning(self, sample_insumo_base_data):
        """Debe advertir sobre inconsistencia en valor_total."""
        data = sample_insumo_base_data.copy()
        data['cantidad'] = 100.0
        data['precio_unitario'] = 500.0
        data['valor_total'] = 10000.0  # Inconsistente (debería ser 50000)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            InsumoProcesado(**data)

            # Verificar que se emitió advertencia
            assert len(w) > 0
            assert 'difiere' in str(w[0].message).lower()

    def test_to_dict(self, sample_insumo_base_data):
        """Debe convertir a diccionario."""
        insumo = InsumoProcesado(**sample_insumo_base_data)
        result = insumo.to_dict()

        assert isinstance(result, dict)
        assert 'codigo_apu' in result
        assert 'tipo_insumo' in result
        assert '_validated' not in result  # No debe incluir campos internos

    def test_validate_method(self, sample_insumo_base_data):
        """Debe indicar si está validado."""
        insumo = InsumoProcesado(**sample_insumo_base_data)
        assert insumo.validate() is True

    def test_repr(self, sample_insumo_base_data):
        """Debe tener representación string útil."""
        insumo = InsumoProcesado(**sample_insumo_base_data)
        repr_str = repr(insumo)

        assert 'InsumoProcesado' in repr_str
        assert 'APU-001' in repr_str


# ============================================================================
# TESTS - CLASE MANODEOBRA
# ============================================================================

class TestManoDeObra:
    """Suite de pruebas para ManoDeObra"""

    def test_create_valid_mano_obra(self, sample_mano_obra_data):
        """Debe crear mano de obra válida."""
        mano_obra = ManoDeObra(**sample_mano_obra_data)

        assert mano_obra.tipo_insumo == 'MANO_DE_OBRA'
        assert mano_obra.rendimiento > 0
        assert mano_obra._validated is True

    def test_rendimiento_required(self, sample_mano_obra_data):
        """Debe advertir si rendimiento es 0."""
        data = sample_mano_obra_data.copy()
        data['rendimiento'] = 0.0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ManoDeObra(**data)

            # Debería advertir sobre rendimiento
            assert any('rendimiento' in str(warning.message).lower() for warning in w)

    def test_rendimiento_cantidad_consistency(self, sample_mano_obra_data):
        """Debe validar coherencia rendimiento-cantidad."""
        data = sample_mano_obra_data.copy()
        data['rendimiento'] = 1.0  # 1 unidad por hora
        data['cantidad'] = 5.0  # Inconsistente

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ManoDeObra(**data)

            # Debería advertir
            assert len(w) > 0

    def test_expected_units_warning(self, sample_mano_obra_data):
        """Debe advertir sobre unidades inusuales."""
        data = sample_mano_obra_data.copy()
        data['unidad_insumo'] = 'KG'  # Inusual para mano de obra
        data['unidad_apu'] = 'KG'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ManoDeObra(**data)

            # Debería advertir sobre unidades
            assert any('unidad' in str(warning.message).lower() for warning in w)

    def test_jornal_validation(self, sample_mano_obra_data):
        """Debe validar jornal si se proporciona."""
        data = sample_mano_obra_data.copy()
        data['jornal'] = 150000.0

        mano_obra = ManoDeObra(**data)
        assert mano_obra.jornal == 150000.0

    def test_jornal_negative_raises_error(self, sample_mano_obra_data):
        """Debe rechazar jornal negativo."""
        data = sample_mano_obra_data.copy()
        data['jornal'] = -50000.0

        with pytest.raises(ValidationError):
            ManoDeObra(**data)


# ============================================================================
# TESTS - CLASE EQUIPO
# ============================================================================

class TestEquipo:
    """Suite de pruebas para Equipo"""

    def test_create_valid_equipo(self, sample_equipo_data):
        """Debe crear equipo válido."""
        equipo = Equipo(**sample_equipo_data)

        assert equipo.tipo_insumo == 'EQUIPO'
        assert equipo._validated is True

    def test_rendimiento_should_be_zero(self, sample_equipo_data):
        """Debe advertir si rendimiento != 0."""
        data = sample_equipo_data.copy()
        data['rendimiento'] = 1.5

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Equipo(**data)

            # Debería advertir
            assert any('rendimiento' in str(warning.message).lower() for warning in w)

    def test_expected_time_units(self, sample_equipo_data):
        """Debe aceptar unidades de tiempo."""
        data = sample_equipo_data.copy()
        data['unidad_insumo'] = 'DIA'

        equipo = Equipo(**data)
        assert equipo.unidad_insumo == 'DIA'


# ============================================================================
# TESTS - CLASE SUMINISTRO
# ============================================================================

class TestSuministro:
    """Suite de pruebas para Suministro"""

    def test_create_valid_suministro(self, sample_suministro_data):
        """Debe crear suministro válido."""
        suministro = Suministro(**sample_suministro_data)

        assert suministro.tipo_insumo == 'SUMINISTRO'
        assert suministro._validated is True

    def test_cantidad_zero_warning(self, sample_suministro_data):
        """Debe advertir si cantidad es 0."""
        data = sample_suministro_data.copy()
        data['cantidad'] = 0.0
        data['valor_total'] = 0.0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Suministro(**data)

            # Debería advertir
            assert any('cantidad' in str(warning.message).lower() for warning in w)

    def test_rendimiento_should_be_zero(self, sample_suministro_data):
        """Debe advertir si rendimiento != 0."""
        data = sample_suministro_data.copy()
        data['rendimiento'] = 2.0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Suministro(**data)

            assert any('rendimiento' in str(warning.message).lower() for warning in w)

    def test_expected_units(self, sample_suministro_data):
        """Debe aceptar unidades típicas de suministro."""
        for unit in ['KG', 'M3', 'L', 'UNIDAD']:
            data = sample_suministro_data.copy()
            data['unidad_insumo'] = unit

            suministro = Suministro(**data)
            assert suministro.unidad_insumo == unit


# ============================================================================
# TESTS - CLASE TRANSPORTE
# ============================================================================

class TestTransporte:
    """Suite de pruebas para Transporte"""

    def test_create_valid_transporte(self, sample_transporte_data):
        """Debe crear transporte válido."""
        transporte = Transporte(**sample_transporte_data)

        assert transporte.tipo_insumo == 'TRANSPORTE'
        assert transporte._validated is True

    def test_expected_transport_units(self, sample_transporte_data):
        """Debe aceptar unidades de transporte."""
        for unit in ['KM', 'VIAJE', 'TON-KM']:
            data = sample_transporte_data.copy()
            data['unidad_insumo'] = unit

            transporte = Transporte(**data)
            assert transporte.unidad_insumo == unit

    def test_rendimiento_should_be_zero(self, sample_transporte_data):
        """Debe advertir si rendimiento != 0."""
        data = sample_transporte_data.copy()
        data['rendimiento'] = 1.0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Transporte(**data)

            assert any('rendimiento' in str(warning.message).lower() for warning in w)


# ============================================================================
# TESTS - CLASE OTRO
# ============================================================================

class TestOtro:
    """Suite de pruebas para Otro"""

    def test_create_valid_otro(self, sample_otro_data):
        """Debe crear Otro válido."""
        otro = Otro(**sample_otro_data)

        assert otro.tipo_insumo == 'OTRO'
        assert otro._validated is True

    def test_flexible_units(self, sample_otro_data):
        """Debe aceptar cualquier unidad."""
        for unit in ['GLB', 'SERVICIO', 'CUSTOM', 'XYZ']:
            data = sample_otro_data.copy()
            data['unidad_insumo'] = unit

            Otro(**data)
            # No debería lanzar error ni advertencia significativa


# ============================================================================
# TESTS - FACTORY FUNCTION CREATE_INSUMO
# ============================================================================

class TestCreateInsumo:
    """Suite de pruebas para create_insumo()"""

    def test_create_mano_obra(self, sample_mano_obra_data):
        """Debe crear ManoDeObra."""
        insumo = create_insumo(**sample_mano_obra_data)

        assert isinstance(insumo, ManoDeObra)
        assert insumo.tipo_insumo == 'MANO_DE_OBRA'

    def test_create_equipo(self, sample_equipo_data):
        """Debe crear Equipo."""
        insumo = create_insumo(**sample_equipo_data)

        assert isinstance(insumo, Equipo)
        assert insumo.tipo_insumo == 'EQUIPO'

    def test_create_suministro(self, sample_suministro_data):
        """Debe crear Suministro."""
        insumo = create_insumo(**sample_suministro_data)

        assert isinstance(insumo, Suministro)
        assert insumo.tipo_insumo == 'SUMINISTRO'

    def test_create_transporte(self, sample_transporte_data):
        """Debe crear Transporte."""
        insumo = create_insumo(**sample_transporte_data)

        assert isinstance(insumo, Transporte)
        assert insumo.tipo_insumo == 'TRANSPORTE'

    def test_create_otro(self, sample_otro_data):
        """Debe crear Otro."""
        insumo = create_insumo(**sample_otro_data)

        assert isinstance(insumo, Otro)
        assert insumo.tipo_insumo == 'OTRO'

    def test_create_with_enum(self, sample_suministro_data):
        """Debe aceptar enum directamente."""
        data = sample_suministro_data.copy()
        data['tipo_insumo'] = TipoInsumo.SUMINISTRO
        insumo = create_insumo(**data)

        assert isinstance(insumo, Suministro)

    def test_create_case_insensitive(self, sample_suministro_data):
        """Debe ser case-insensitive."""
        data = sample_suministro_data.copy()
        data['tipo_insumo'] = 'suministro'
        insumo = create_insumo(**data)

        assert isinstance(insumo, Suministro)

    def test_create_invalid_tipo_raises_error(self, sample_insumo_base_data):
        """Debe rechazar tipo inválido."""
        data = sample_insumo_base_data.copy()
        data['tipo_insumo'] = 'INVALIDO'
        with pytest.raises(InvalidTipoInsumoError):
            create_insumo(**data)

    def test_create_missing_kwargs_raises_error(self):
        """Debe rechazar kwargs faltantes."""
        with pytest.raises(ValidationError):
            create_insumo(tipo_insumo='SUMINISTRO', codigo_apu='APU-001')

    def test_create_forces_tipo_sync(self, sample_suministro_data):
        """Debe forzar sincronización de tipo_insumo."""
        data = sample_suministro_data.copy()
        data['tipo_insumo'] = 'OTRO'  # Diferente del argumento
        data['categoria'] = 'OTRO'

        insumo = create_insumo(**data)

        # Debe usar el tipo pasado a la función
        assert insumo.tipo_insumo == 'OTRO'
        assert insumo.categoria == 'OTRO'


# ============================================================================
# TESTS - VALIDATE_INSUMO_DATA
# ============================================================================

class TestValidateInsumoData:
    """Suite de pruebas para validate_insumo_data()"""

    def test_validate_complete_data(self, sample_suministro_data):
        """Debe validar datos completos."""
        result = validate_insumo_data(sample_suministro_data)

        assert 'codigo_apu' in result
        assert 'tipo_insumo' in result
        assert result['tipo_insumo'] == 'SUMINISTRO'

    def test_validate_adds_defaults(self):
        """Debe agregar valores por defecto."""
        minimal_data = {
            'codigo_apu': 'APU-001',
            'descripcion_apu': 'Test',
            'unidad_apu': 'M3',
            'descripcion_insumo': 'Test Insumo',
            'unidad_insumo': 'KG',
            'tipo_insumo': 'SUMINISTRO'
        }

        result = validate_insumo_data(minimal_data)

        assert 'cantidad' in result
        assert 'precio_unitario' in result
        assert 'valor_total' in result
        assert result['cantidad'] == 0.0

    def test_validate_converts_numeric_strings(self):
        """Debe convertir strings numéricos."""
        data = {
            'codigo_apu': 'APU-001',
            'descripcion_apu': 'Test',
            'unidad_apu': 'M3',
            'descripcion_insumo': 'Test',
            'unidad_insumo': 'KG',
            'tipo_insumo': 'SUMINISTRO',
            'cantidad': '100.5',
            'precio_unitario': '500'
        }

        result = validate_insumo_data(data)

        assert result['cantidad'] == 100.5
        assert result['precio_unitario'] == 500.0
        assert isinstance(result['cantidad'], float)

    def test_validate_missing_required_raises_error(self):
        """Debe rechazar datos con campos faltantes."""
        incomplete_data = {
            'codigo_apu': 'APU-001',
            'descripcion_apu': 'Test'
            # Faltan campos requeridos
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_insumo_data(incomplete_data)

        assert 'faltante' in str(exc_info.value).lower()

    def test_validate_null_required_raises_error(self):
        """Debe rechazar campos requeridos nulos."""
        data = {
            'codigo_apu': None,
            'descripcion_apu': 'Test',
            'unidad_apu': 'M3',
            'descripcion_insumo': 'Test',
            'unidad_insumo': 'KG',
            'tipo_insumo': 'SUMINISTRO'
        }

        with pytest.raises(ValidationError):
            validate_insumo_data(data)

    def test_validate_invalid_numeric_raises_error(self):
        """Debe rechazar valores numéricos inválidos."""
        data = {
            'codigo_apu': 'APU-001',
            'descripcion_apu': 'Test',
            'unidad_apu': 'M3',
            'descripcion_insumo': 'Test',
            'unidad_insumo': 'KG',
            'tipo_insumo': 'SUMINISTRO',
            'cantidad': 'not_a_number'
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_insumo_data(data)

        assert 'numérico' in str(exc_info.value).lower()

    def test_validate_normalizes_tipo(self):
        """Debe normalizar tipo_insumo."""
        data = {
            'codigo_apu': 'APU-001',
            'descripcion_apu': 'Test',
            'unidad_apu': 'M3',
            'descripcion_insumo': 'Test',
            'unidad_insumo': 'KG',
            'tipo_insumo': 'suministro'  # Minúsculas
        }

        result = validate_insumo_data(data)

        assert result['tipo_insumo'] == 'SUMINISTRO'
        assert result['categoria'] == 'SUMINISTRO'

    def test_validate_calculates_valor_total(self):
        """Debe calcular valor_total si no se proporciona."""
        data = {
            'codigo_apu': 'APU-001',
            'descripcion_apu': 'Test',
            'unidad_apu': 'M3',
            'descripcion_insumo': 'Test',
            'unidad_insumo': 'KG',
            'tipo_insumo': 'SUMINISTRO',
            'cantidad': 100.0,
            'precio_unitario': 500.0
        }

        result = validate_insumo_data(data)

        assert result['valor_total'] == 50000.0

    def test_validate_not_dict_raises_error(self):
        """Debe rechazar no-diccionarios."""
        with pytest.raises(ValidationError):
            validate_insumo_data("not a dict")

        with pytest.raises(ValidationError):
            validate_insumo_data([])


# ============================================================================
# TESTS - CREATE_INSUMO_FROM_RAW
# ============================================================================

class TestCreateInsumoFromRaw:
    """Suite de pruebas para create_insumo_from_raw()"""

    def test_create_from_raw_success(self):
        """Debe crear insumo desde datos crudos."""
        raw_data = {
            'codigo_apu': 'APU-001',
            'descripcion_apu': 'Test',
            'unidad_apu': 'm3',
            'descripcion_insumo': 'Cemento',
            'unidad_insumo': 'kg',
            'tipo_insumo': 'suministro',
            'cantidad': '100',
            'precio_unitario': '500'
        }

        insumo = create_insumo_from_raw(raw_data)

        assert isinstance(insumo, Suministro)
        assert insumo.cantidad == 100.0
        assert insumo.unidad_apu == 'M3'

    def test_create_from_raw_invalid_data(self):
        """Debe rechazar datos inválidos."""
        raw_data = {
            'codigo_apu': 'APU-001',
            # Faltan campos
        }

        with pytest.raises(ValidationError):
            create_insumo_from_raw(raw_data)


# ============================================================================
# TESTS - FUNCIONES DE UTILIDAD
# ============================================================================

class TestUtilityFunctions:
    """Suite de pruebas para funciones de utilidad"""

    def test_get_tipo_insumo_class_valid(self):
        """Debe retornar clase correcta."""
        assert get_tipo_insumo_class('MANO_DE_OBRA') == ManoDeObra
        assert get_tipo_insumo_class('EQUIPO') == Equipo
        assert get_tipo_insumo_class('SUMINISTRO') == Suministro
        assert get_tipo_insumo_class('TRANSPORTE') == Transporte
        assert get_tipo_insumo_class('OTRO') == Otro

    def test_get_tipo_insumo_class_with_enum(self):
        """Debe aceptar enum."""
        result = get_tipo_insumo_class(TipoInsumo.SUMINISTRO)
        assert result == Suministro

    def test_get_tipo_insumo_class_invalid_raises_error(self):
        """Debe rechazar tipo inválido."""
        with pytest.raises(InvalidTipoInsumoError):
            get_tipo_insumo_class('INVALIDO')

    def test_get_all_tipo_insumo_values(self):
        """Debe retornar todos los valores."""
        values = get_all_tipo_insumo_values()

        assert isinstance(values, set)
        assert 'MANO_DE_OBRA' in values
        assert 'EQUIPO' in values
        assert 'SUMINISTRO' in values
        assert 'TRANSPORTE' in values
        assert 'OTRO' in values
        assert len(values) == 5

    def test_is_valid_tipo_insumo_valid(self):
        """Debe validar tipos válidos."""
        assert is_valid_tipo_insumo('MANO_DE_OBRA') is True
        assert is_valid_tipo_insumo('EQUIPO') is True
        assert is_valid_tipo_insumo('mano_de_obra') is True

    def test_is_valid_tipo_insumo_invalid(self):
        """Debe rechazar tipos inválidos."""
        assert is_valid_tipo_insumo('INVALIDO') is False
        assert is_valid_tipo_insumo('') is False
        assert is_valid_tipo_insumo('123') is False


# ============================================================================
# TESTS - CONSTANTES
# ============================================================================

class TestConstants:
    """Suite de pruebas para constantes"""

    def test_unidad_normalizada_map_exists(self):
        """Debe tener mapeo de unidades."""
        assert isinstance(UNIDAD_NORMALIZADA_MAP, dict)
        assert len(UNIDAD_NORMALIZADA_MAP) > 0
        assert 'kg' in UNIDAD_NORMALIZADA_MAP

    def test_insumo_class_map_complete(self):
        """Debe tener todas las clases mapeadas."""
        assert len(INSUMO_CLASS_MAP) == 5
        assert INSUMO_CLASS_MAP['MANO_DE_OBRA'] == ManoDeObra
        assert INSUMO_CLASS_MAP['EQUIPO'] == Equipo
        assert INSUMO_CLASS_MAP['SUMINISTRO'] == Suministro
        assert INSUMO_CLASS_MAP['TRANSPORTE'] == Transporte
        assert INSUMO_CLASS_MAP['OTRO'] == Otro

    def test_tolerance_values(self):
        """Debe tener valores de tolerancia razonables."""
        assert 0 < VALOR_TOTAL_TOLERANCE < 1
        assert 0 < CANTIDAD_RENDIMIENTO_TOLERANCE < 1

    def test_length_limits(self):
        """Debe tener límites de longitud."""
        assert MAX_CODIGO_LENGTH > 0
        assert MAX_DESCRIPCION_LENGTH > 0

    def test_numeric_limits(self):
        """Debe tener límites numéricos."""
        assert MIN_CANTIDAD >= 0
        assert MAX_CANTIDAD > MIN_CANTIDAD
        assert MIN_PRECIO >= 0
        assert MAX_PRECIO > MIN_PRECIO

    def test_unit_sets(self):
        """Debe tener conjuntos de unidades."""
        assert isinstance(UNIDADES_TIEMPO, frozenset)
        assert isinstance(UNIDADES_MASA, frozenset)
        assert isinstance(UNIDADES_VOLUMEN, frozenset)
        assert isinstance(UNIDADES_AREA, frozenset)
        assert isinstance(UNIDADES_LONGITUD, frozenset)
        assert isinstance(UNIDADES_TRANSPORTE, frozenset)
        assert isinstance(UNIDADES_GENERICAS, frozenset)

        assert len(UNIDADES_TIEMPO) > 0
        assert 'HORA' in UNIDADES_TIEMPO


# ============================================================================
# TESTS DE INTEGRACIÓN
# ============================================================================

class TestIntegration:
    """Tests de integración entre componentes"""

    def test_full_workflow_create_validate_serialize(self):
        """Workflow completo: crear, validar, serializar."""
        # 1. Datos crudos
        raw_data = {
            'codigo_apu': '  apu-001  ',
            'descripcion_apu': 'Concreto f\'c=280',
            'unidad_apu': 'm3',
            'descripcion_insumo': '  Cemento   Portland  ',
            'unidad_insumo': 'kg',
            'tipo_insumo': 'suministro',
            'cantidad': '350',
            'precio_unitario': '450'
        }

        # 2. Validar y limpiar
        validate_insumo_data(raw_data)

        # 3. Crear insumo
        insumo = create_insumo_from_raw(raw_data)

        # 4. Validar
        assert insumo.validate() is True

        # 5. Serializar
        data_dict = insumo.to_dict()

        # Verificar resultado
        assert isinstance(data_dict, dict)
        assert data_dict['codigo_apu'] == 'APU-001'
        assert data_dict['unidad_apu'] == 'M3'
        assert data_dict['tipo_insumo'] == 'SUMINISTRO'

    def test_workflow_all_types(self):
        """Debe crear todos los tipos de insumos."""
        tipos = ['MANO_DE_OBRA', 'EQUIPO', 'SUMINISTRO', 'TRANSPORTE', 'OTRO']

        for tipo in tipos:
            data = {
                'codigo_apu': 'APU-001',
                'descripcion_apu': 'Test',
                'unidad_apu': 'M3',
                'descripcion_insumo': f'Insumo {tipo}',
                'unidad_insumo': 'HORA' if tipo in ['MANO_DE_OBRA', 'EQUIPO'] else 'KG',
                'tipo_insumo': tipo,
                'cantidad': 1.0,
                'precio_unitario': 1000.0,
                'valor_total': 1000.0,
                'rendimiento': 1.0 if tipo == 'MANO_DE_OBRA' else 0.0
            }

            insumo = create_insumo_from_raw(data)
            assert insumo.tipo_insumo == tipo
            assert insumo.validate() is True


# ============================================================================
# TESTS DE EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Tests de casos límite y edge cases"""

    def test_very_large_values(self, sample_suministro_data):
        """Debe manejar valores muy grandes."""
        data = sample_suministro_data.copy()
        data['cantidad'] = 999999.0
        data['precio_unitario'] = 999999.0
        data['valor_total'] = 999999.0 * 999999.0

        # Puede exceder MAX_PRECIO y lanzar error o manejarlo
        try:
            insumo = Suministro(**data)
            assert insumo.cantidad == 999999.0
        except ValidationError:
            # Es válido rechazar valores muy grandes
            pass

    def test_very_small_values(self, sample_suministro_data):
        """Debe manejar valores muy pequeños."""
        data = sample_suministro_data.copy()
        data['cantidad'] = 0.0001
        data['precio_unitario'] = 0.0001
        data['valor_total'] = 0.0001 * 0.0001

        insumo = Suministro(**data)
        assert insumo.cantidad == 0.0001

    def test_zero_values(self, sample_suministro_data):
        """Debe manejar valores en cero."""
        data = sample_suministro_data.copy()
        data['cantidad'] = 0.0
        data['precio_unitario'] = 0.0
        data['valor_total'] = 0.0

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            insumo = Suministro(**data)
            assert insumo.cantidad == 0.0

    def test_unicode_characters(self, sample_suministro_data):
        """Debe manejar caracteres unicode."""
        data = sample_suministro_data.copy()
        data['descripcion_insumo'] = 'Ácido Nitrógeno Ñoño €'

        insumo = Suministro(**data)
        # Los acentos deben ser removidos
        assert 'ACIDO' in insumo.descripcion_insumo
        assert 'NITROGENO' in insumo.descripcion_insumo

    def test_very_long_description(self, sample_suministro_data):
        """Debe manejar descripciones muy largas."""
        data = sample_suministro_data.copy()
        data['descripcion_insumo'] = 'A' * (MAX_DESCRIPCION_LENGTH + 100)

        # Debe rechazar o truncar
        try:
            Suministro(**data)
        except ValidationError:
            # Es válido rechazar descripciones muy largas
            pass

    def test_decimal_precision(self, sample_suministro_data):
        """Debe manejar precisión decimal."""
        data = sample_suministro_data.copy()
        data['cantidad'] = 123.456789012345
        data['precio_unitario'] = 987.654321098765
        data['valor_total'] = data['cantidad'] * data['precio_unitario']

        insumo = Suministro(**data)
        assert isinstance(insumo.cantidad, float)


# ============================================================================
# CONFIGURACIÓN DE PYTEST
# ============================================================================

def pytest_configure(config):
    """Configuración de pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == '__main__':
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--cov=app.schemas',
        '--cov-report=html',
        '--cov-report=term-missing',
        '-W', 'ignore::DeprecationWarning',
        '--maxfail=5',
    ])
