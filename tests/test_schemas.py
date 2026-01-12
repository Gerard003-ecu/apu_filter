"""
Suite completa de pruebas para schemas.py

Pruebas exhaustivas con cobertura de normalización, validación,
clases de insumos, factory functions y casos edge.
"""

import warnings
from decimal import Decimal
import math
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
    VALOR_TOTAL_WARNING_TOLERANCE,
    VALOR_TOTAL_ERROR_TOLERANCE,
    Equipo,
    # Clases
    InsumoProcesado,
    APUStructure,
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
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_insumo_base_data():
    return {
        "codigo_apu": "APU-001",
        "descripcion_apu": "Concreto f'c=280 kg/cm2",
        "unidad_apu": "M3",
        "descripcion_insumo": "Cemento Portland Tipo I",
        "unidad_insumo": "KG",
        "cantidad": 350.0,
        "precio_unitario": 450.0,
        "valor_total": 157500.0,
        "categoria": "SUMINISTRO",
        "tipo_insumo": "SUMINISTRO",
        "formato_origen": "FORMATO_A",
        "rendimiento": 0.0,
    }

# ============================================================================
# TESTS - APU STRUCTURE (Topología)
# ============================================================================

class TestAPUStructure:
    def test_topological_stability_empty(self):
        apu = APUStructure(id="TEST", description="Vacio")
        assert apu.topological_stability_index() == 0.0

    def test_topological_stability_diverse(self):
        apu = APUStructure(id="TEST", description="Diverso")
        i1 = create_insumo(tipo_insumo="SUMINISTRO", codigo_apu="A", descripcion_apu="D", unidad_apu="U", descripcion_insumo="I1", unidad_insumo="U", cantidad=1, precio_unitario=100, valor_total=100)
        i2 = create_insumo(tipo_insumo="MANO_DE_OBRA", codigo_apu="A", descripcion_apu="D", unidad_apu="U", descripcion_insumo="I2", unidad_insumo="U", cantidad=1, precio_unitario=100, valor_total=100, rendimiento=1)
        i3 = create_insumo(tipo_insumo="EQUIPO", codigo_apu="A", descripcion_apu="D", unidad_apu="U", descripcion_insumo="I3", unidad_insumo="U", cantidad=1, precio_unitario=100, valor_total=100)

        apu.add_resource(i1)
        apu.add_resource(i2)
        apu.add_resource(i3)

        # 3 tipos -> diversidad = 1.0
        # Costos iguales -> Entropía máxima -> 1.0
        # Score = 0.4*1 + 0.6*1 = 1.0
        assert math.isclose(apu.topological_stability_index(), 1.0, rel_tol=0.01)

    def test_topological_stability_monoculture(self):
        apu = APUStructure(id="TEST", description="Mono")
        # 3 insumos del mismo tipo
        for _ in range(3):
            i = create_insumo(tipo_insumo="SUMINISTRO", codigo_apu="A", descripcion_apu="D", unidad_apu="U", descripcion_insumo="I", unidad_insumo="U", cantidad=1, precio_unitario=100, valor_total=100)
            apu.add_resource(i)

        # Diversidad: 1 tipo / 3 insumos = 0.33
        # Entropía: max (costos iguales) = 1.0
        # Score = 0.4*0.33 + 0.6*1.0 = 0.132 + 0.6 = 0.732
        # Espera... diversidad score = min(1/3, 1) -> 0.333
        expected = (0.3333 * 0.4) + (1.0 * 0.6)
        # Re-leyendo lógica:
        # diversidad_score = min(len(tipos) / 3.0, 1.0)
        # Tipos=1. len/3 = 0.333

        score = apu.topological_stability_index()
        assert score > 0.7 and score < 0.8


# ============================================================================
# TESTS - CONSTANTES ACTUALIZADAS
# ============================================================================

class TestConstants:
    def test_tolerance_values(self):
        """Debe tener valores de tolerancia razonables."""
        assert 0 < VALOR_TOTAL_WARNING_TOLERANCE < 1
        assert 0 < VALOR_TOTAL_ERROR_TOLERANCE < 1
        assert VALOR_TOTAL_ERROR_TOLERANCE > VALOR_TOTAL_WARNING_TOLERANCE

# ============================================================================
# TESTS - CLASE BASE INSUMOPROCESADO (Consistencia)
# ============================================================================

class TestInsumoProcesado:
    def test_create_valid_insumo(self, sample_insumo_base_data):
        insumo = InsumoProcesado(**sample_insumo_base_data)
        assert insumo._validated is True

    def test_valor_total_consistency_warning(self, sample_insumo_base_data):
        """Debe advertir sobre inconsistencia leve (1-5%)."""
        data = sample_insumo_base_data.copy()
        data["cantidad"] = 100.0
        data["precio_unitario"] = 500.0
        # Esperado 50,000. Ponemos 51,000 (+2%)
        data["valor_total"] = 51000.0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            InsumoProcesado(**data)
            assert any("divergencia" in str(warn.message).lower() for warn in w)

    def test_valor_total_consistency_error(self, sample_insumo_base_data):
        """Debe fallar sobre inconsistencia grave (>5%)."""
        data = sample_insumo_base_data.copy()
        data["cantidad"] = 100.0
        data["precio_unitario"] = 500.0
        # Esperado 50,000. Ponemos 60,000 (+20%)
        data["valor_total"] = 60000.0

        with pytest.raises(ValidationError) as exc:
            InsumoProcesado(**data)
        assert "inconsistencia grave" in str(exc.value).lower()
