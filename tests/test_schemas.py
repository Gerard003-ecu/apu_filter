import pytest

from app.schemas import (
    Equipo,
    InsumoProcesado,
    ManoDeObra,
    Otro,
    Suministro,
    Transporte,
)


@pytest.fixture
def common_args():
    """Fixture con argumentos comunes para crear un InsumoProcesado."""
    return {
        "codigo_apu": "123",
        "descripcion_apu": "Test APU",
        "unidad_apu": "M2",
        "descripcion_insumo": "Test Insumo",
        "unidad_insumo": "UN",
        "cantidad": 1.0,
        "precio_unitario": 100.0,
        "valor_total": 100.0,
        "categoria": "TEST",
        "formato_origen": "TEST_FORMAT",
        "normalized_desc": "test insumo",
    }

def test_insumo_procesado(common_args):
    """
    Prueba la creación de una instancia de InsumoProcesado.

    Verifica que la clase base se pueda instanciar correctamente y que los
    atributos se asignen como se espera.
    """
    insumo = InsumoProcesado(tipo_insumo="BASE", **common_args)
    assert insumo.codigo_apu == "123"
    assert insumo.tipo_insumo == "BASE"

def test_mano_de_obra(common_args):
    """
    Prueba la creación de una instancia de ManoDeObra.

    Asegura que la subclase ManoDeObra se cree correctamente, incluyendo
    su atributo específico 'rendimiento'.
    """
    insumo = ManoDeObra(
        rendimiento=0.5, tipo_insumo="MANO_DE_OBRA", **common_args
    )
    assert insumo.rendimiento == 0.5
    assert insumo.tipo_insumo == "MANO_DE_OBRA"
    assert insumo.codigo_apu == "123"

def test_equipo(common_args):
    """
    Prueba la creación de una instancia de Equipo.

    Verifica que la subclase Equipo se pueda instanciar y que el
    'tipo_insumo' se asigne correctamente.
    """
    insumo = Equipo(tipo_insumo="EQUIPO", **common_args)
    assert insumo.tipo_insumo == "EQUIPO"
    assert insumo.codigo_apu == "123"

def test_suministro(common_args):
    """
    Prueba la creación de una instancia de Suministro.

    Verifica que la subclase Suministro se pueda instanciar y que el
    'tipo_insumo' se asigne correctamente.
    """
    insumo = Suministro(tipo_insumo="SUMINISTRO", **common_args)
    assert insumo.tipo_insumo == "SUMINISTRO"
    assert insumo.codigo_apu == "123"

def test_transporte(common_args):
    """
    Prueba la creación de una instancia de Transporte.

    Verifica que la subclase Transporte se pueda instanciar y que el
    'tipo_insumo' se asigne correctamente.
    """
    insumo = Transporte(tipo_insumo="TRANSPORTE", **common_args)
    assert insumo.tipo_insumo == "TRANSPORTE"
    assert insumo.codigo_apu == "123"

def test_otro(common_args):
    """
    Prueba la creación de una instancia de Otro.

    Verifica que la subclase para insumos no clasificados se pueda
    instanciar y que el 'tipo_insumo' se asigne correctamente.
    """
    insumo = Otro(tipo_insumo="OTRO", **common_args)
    assert insumo.tipo_insumo == "OTRO"
    assert insumo.codigo_apu == "123"
