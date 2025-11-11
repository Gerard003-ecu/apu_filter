
import pytest

from app.schemas import (
    Equipo,
    InsumoProcesado,
    ManoDeObra,
    Suministro,
    Transporte,
    create_insumo,
    normalize_description,
    normalize_tipo_insumo,
    normalize_unit,
    validate_insumo_data,
)


# --- Fixtures mejoradas ---
@pytest.fixture
def base_insumo_data():
    """Fixture con datos mínimos válidos para crear un insumo."""
    return {
        "codigo_apu": "APU-001",
        "descripcion_apu": "Construcción de muro de concreto",
        "unidad_apu": "m2",
        "descripcion_insumo": "Cemento portland 50kg",
        "unidad_insumo": "bolsa",
        "cantidad": 10.5,
        "precio_unitario": 25.75,
        "valor_total": 270.375,  # 10.5 * 25.75
        "categoria": "SUMINISTRO",
        "formato_origen": "EXCEL",
        "tipo_insumo": "SUMINISTRO",
    }


@pytest.fixture
def mano_obra_data():
    """Fixture para Mano de Obra con rendimiento coherente."""
    return {
        "codigo_apu": "APU-002",
        "descripcion_apu": "Instalación de tuberías",
        "unidad_apu": "hora",
        "descripcion_insumo": "Operario especializado",
        "unidad_insumo": "persona",
        "cantidad": 2.0,  # 1 / 0.5
        "precio_unitario": 150.0,
        "valor_total": 300.0,
        "categoria": "MANO_DE_OBRA",
        "formato_origen": "PDF",
        "tipo_insumo": "MANO_DE_OBRA",
        "rendimiento": 0.5,
    }


@pytest.fixture
def equipo_data():
    """Fixture para Equipo con unidades típicas."""
    return {
        "codigo_apu": "APU-003",
        "descripcion_apu": "Excavación con retroexcavadora",
        "unidad_apu": "dia",
        "descripcion_insumo": "Retroexcavadora CAT 320",
        "unidad_insumo": "unidad",
        "cantidad": 1.0,
        "precio_unitario": 800.0,
        "valor_total": 800.0,
        "categoria": "EQUIPO",
        "formato_origen": "CSV",
        "tipo_insumo": "EQUIPO",
        "rendimiento": 0.0,
    }


@pytest.fixture
def transporte_data():
    """Fixture para Transporte con unidades típicas."""
    return {
        "codigo_apu": "APU-004",
        "descripcion_apu": "Transporte de arena",
        "unidad_apu": "km",
        "descripcion_insumo": "Camión de 10 toneladas",
        "unidad_insumo": "viaje",
        "cantidad": 5.0,
        "precio_unitario": 45.0,
        "valor_total": 225.0,
        "categoria": "TRANSPORTE",
        "formato_origen": "TXT",
        "tipo_insumo": "TRANSPORTE",
        "rendimiento": 0.0,
    }


# --- Pruebas de normalización de campos ---
def test_normalize_unit():
    """Prueba que normalize_unit normaliza correctamente las unidades."""
    assert normalize_unit("m2") == "M2"
    assert normalize_unit("KM") == "KM"
    assert normalize_unit("hora") == "HORA"
    assert normalize_unit("Viaje") == "VIAJE"
    assert normalize_unit("ton-km") == "TON-KM"
    assert normalize_unit("xyz") == "XYZ"  # No reconocida → se mantiene
    assert normalize_unit(None) == "UNIDAD"
    assert normalize_unit("") == "UNIDAD"
    assert normalize_unit(123) == "UNIDAD"  # No string → fallback


def test_normalize_description():
    """Prueba que normalize_description limpia y normaliza descripciones."""
    assert normalize_description("  Cemento Portland  ") == "CEMENTO PORTLAND"
    assert normalize_description("Cemento con tilde: ácido") == "CEMENTO CON TILDE: ACIDO"
    assert normalize_description("") == ""
    assert normalize_description(None) == ""
    assert normalize_description("  \t\n  ") == ""


def test_normalize_tipo_insumo():
    """Prueba que normalize_tipo_insumo valida y normaliza el tipo."""
    assert normalize_tipo_insumo("mano_de_obra") == "MANO_DE_OBRA"
    assert normalize_tipo_insumo("EQUIPO") == "EQUIPO"
    assert normalize_tipo_insumo("suministro") == "SUMINISTRO"
    with pytest.raises(ValueError, match="Tipo de insumo inválido"):
        normalize_tipo_insumo("INVALID_TYPE")


# --- Pruebas de InsumoProcesado (clase base) ---
def test_insumo_procesado_creacion_correcta(base_insumo_data):
    """Prueba que InsumoProcesado se crea correctamente y normaliza campos."""
    insumo = InsumoProcesado(**base_insumo_data)
    assert insumo.codigo_apu == "APU-001"
    assert insumo.descripcion_apu == "CONSTRUCCION DE MURO DE CONCRETO"
    assert insumo.descripcion_insumo == "CEMENTO PORTLAND 50KG"
    assert insumo.unidad_apu == "M2"
    assert insumo.unidad_insumo == "BOLSA"
    assert insumo.tipo_insumo == "SUMINISTRO"
    assert insumo.categoria == "SUMINISTRO"
    assert insumo.normalized_desc == "CEMENTO PORTLAND 50KG"
    assert insumo.formato_origen == "EXCEL"


def test_insumo_procesado_valida_valor_total(base_insumo_data):
    """Prueba que InsumoProcesado lanza advertencia si valor_total es inconsistente."""
    base_insumo_data["valor_total"] = 9999.99  # Inconsistente con 10.5 * 25.75
    with pytest.warns(UserWarning, match="valor_total.*no coincide"):
        insumo = InsumoProcesado(**base_insumo_data)
        assert insumo.valor_total == 9999.99


def test_insumo_procesado_lanza_error_si_codigo_vacio(base_insumo_data):
    """Prueba que InsumoProcesado lanza error si codigo_apu está vacío."""
    base_insumo_data["codigo_apu"] = ""
    with pytest.raises(ValueError, match="codigo_apu no puede estar vacío"):
        InsumoProcesado(**base_insumo_data)


def test_insumo_procesado_lanza_error_si_descripcion_insumo_vacia(base_insumo_data):
    """Prueba que InsumoProcesado lanza error si descripcion_insumo está vacía."""
    base_insumo_data["descripcion_insumo"] = ""
    with pytest.raises(ValueError, match="descripcion_insumo no puede estar vacío"):
        InsumoProcesado(**base_insumo_data)


def test_insumo_procesado_lanza_error_si_cantidad_negativa(base_insumo_data):
    """Prueba que InsumoProcesado lanza error si cantidad es negativa."""
    base_insumo_data["cantidad"] = -1.0
    with pytest.raises(ValueError, match="cantidad debe ser un número no negativo"):
        InsumoProcesado(**base_insumo_data)


def test_insumo_procesado_lanza_error_si_precio_unitario_negativo(base_insumo_data):
    """Prueba que InsumoProcesado lanza error si precio_unitario es negativo."""
    base_insumo_data["precio_unitario"] = -10.0
    with pytest.raises(ValueError, match="precio_unitario debe ser un número no negativo"):
        InsumoProcesado(**base_insumo_data)


def test_insumo_procesado_lanza_error_si_valor_total_negativo(base_insumo_data):
    """Prueba que InsumoProcesado lanza error si valor_total es negativo."""
    base_insumo_data["valor_total"] = -5.0
    with pytest.raises(ValueError, match="valor_total debe ser un número no negativo"):
        InsumoProcesado(**base_insumo_data)


def test_insumo_procesado_lanza_error_si_tipo_insumo_invalido(base_insumo_data):
    """Prueba que InsumoProcesado lanza error si tipo_insumo es inválido."""
    base_insumo_data["tipo_insumo"] = "INVALID"
    with pytest.raises(ValueError, match="Tipo de insumo inválido"):
        InsumoProcesado(**base_insumo_data)


def test_insumo_procesado_lanza_error_si_cantidad_no_numerica(base_insumo_data):
    """Prueba que InsumoProcesado lanza error si cantidad no es numérica."""
    base_insumo_data["cantidad"] = "abc"
    with pytest.raises(ValueError, match="cantidad debe ser un número no negativo"):
        InsumoProcesado(**base_insumo_data)


# --- Pruebas de subclases específicas ---
def test_mano_de_obra_rendimiento_valido(mano_obra_data):
    """Prueba que ManoDeObra acepta rendimiento válido."""
    insumo = ManoDeObra(**mano_obra_data)
    assert insumo.rendimiento == 0.5
    assert insumo.tipo_insumo == "MANO_DE_OBRA"
    assert insumo.categoria == "MANO_DE_OBRA"


def test_mano_de_obra_rendimiento_negativo(mano_obra_data):
    """Prueba que ManoDeObra rechaza rendimiento negativo."""
    mano_obra_data["rendimiento"] = -0.1
    with pytest.raises(ValueError, match="rendimiento debe ser un número no negativo"):
        ManoDeObra(**mano_obra_data)


def test_mano_de_obra_inconsistencia_rendimiento_cantidad(mano_obra_data):
    """Prueba que ManoDeObra emite advertencia si cantidad no coincide con rendimiento."""
    mano_obra_data["rendimiento"] = 0.5
    mano_obra_data["cantidad"] = 5.0  # Debería ser 2.0 → 1/0.5
    with pytest.warns(UserWarning, match="rendimiento=0.5 sugiere cantidad≈2.0000, pero cantidad=5.0"):
        insumo = ManoDeObra(**mano_obra_data)
        assert insumo.cantidad == 5.0


def test_mano_de_obra_unidad_no_tipica(mano_obra_data):
    """Prueba que ManoDeObra emite advertencia si unidad no es típica."""
    mano_obra_data["unidad_apu"] = "KG"
    with pytest.warns(UserWarning, match="unidades 'KG' o 'PERSONA' no son típicas"):
        ManoDeObra(**mano_obra_data)


def test_equipo_rendimiento_no_cero(equipo_data):
    """Prueba que Equipo emite advertencia si rendimiento != 0."""
    equipo_data["rendimiento"] = 0.1
    with pytest.warns(UserWarning, match="rendimiento=0.1 no es relevante"):
        Equipo(**equipo_data)


def test_suministro_cantidad_cero(suministro_data=None):
    """Prueba que Suministro emite advertencia si cantidad == 0."""
    data = {
        "codigo_apu": "APU-005",
        "descripcion_apu": "Test",
        "unidad_apu": "m2",
        "descripcion_insumo": "Test",
        "unidad_insumo": "und",
        "cantidad": 0.0,
        "precio_unitario": 10.0,
        "valor_total": 0.0,
        "categoria": "SUMINISTRO",
        "formato_origen": "TEST",
        "tipo_insumo": "SUMINISTRO",
        "rendimiento": 0.0,
    }
    with pytest.warns(UserWarning, match="cantidad=0.0. ¿Es intencional?"):
        Suministro(**data)


def test_transporte_unidad_no_tipica(transporte_data):
    """Prueba que Transporte emite advertencia si unidad no es típica."""
    transporte_data["unidad_apu"] = "KG"
    with pytest.warns(UserWarning, match="unidad 'KG' o 'VIAJE' inusual"):
        Transporte(**transporte_data)


# --- Pruebas de create_insumo y validate_insumo_data ---
def test_create_insumo_exitoso(base_insumo_data):
    """Prueba que create_insumo crea correctamente una instancia."""
    insumo = create_insumo(**base_insumo_data)
    assert isinstance(insumo, Suministro)
    assert insumo.tipo_insumo == "SUMINISTRO"
    assert insumo.categoria == "SUMINISTRO"


def test_create_insumo_tipo_invalido(base_insumo_data):
    """Prueba que create_insumo lanza error si tipo es inválido."""
    base_insumo_data["tipo_insumo"] = "INVALID"
    with pytest.raises(ValueError, match="Tipo de insumo inválido"):
        create_insumo(**base_insumo_data)


def test_create_insumo_argumentos_incorrectos(base_insumo_data):
    """Prueba que create_insumo lanza error si faltan argumentos requeridos."""
    base_insumo_data.pop("codigo_apu")
    with pytest.raises(ValueError, match="Error creando insumo tipo SUMINISTRO"):
        create_insumo(**base_insumo_data)


def test_validate_insumo_data_exitoso(base_insumo_data):
    """Prueba que validate_insumo_data limpia y normaliza correctamente."""
    cleaned = validate_insumo_data(base_insumo_data)
    assert cleaned["unidad_apu"] == "M2"
    assert cleaned["unidad_insumo"] == "BOLSA"
    assert cleaned["tipo_insumo"] == "SUMINISTRO"
    assert cleaned["categoria"] == "SUMINISTRO"
    assert cleaned["cantidad"] == 10.5
    assert cleaned["normalized_desc"] == "CEMENTO PORTLAND 50KG"


def test_validate_insumo_data_falta_campo_obligatorio(base_insumo_data):
    """Prueba que validate_insumo_data lanza error si falta campo obligatorio."""
    base_insumo_data.pop("codigo_apu")
    with pytest.raises(ValueError, match="Campo requerido faltante o nulo: codigo_apu"):
        validate_insumo_data(base_insumo_data)


def test_validate_insumo_data_tipo_invalido(base_insumo_data):
    """Prueba que validate_insumo_data lanza error si tipo_insumo es inválido."""
    base_insumo_data["tipo_insumo"] = "INVALID"
    with pytest.raises(ValueError, match="Invalid tipo_insumo: Tipo de insumo inválido"):
        validate_insumo_data(base_insumo_data)


def test_validate_insumo_data_cantidad_no_numerica(base_insumo_data):
    """Prueba que validate_insumo_data lanza error si cantidad no es numérica."""
    base_insumo_data["cantidad"] = "abc"
    with pytest.raises(ValueError, match="Campo cantidad debe ser numérico"):
        validate_insumo_data(base_insumo_data)


def test_validate_insumo_data_normaliza_tipos(base_insumo_data):
    """Prueba que validate_insumo_data convierte tipos automáticamente."""
    base_insumo_data["cantidad"] = "10"
    base_insumo_data["precio_unitario"] = "25.75"
    base_insumo_data["valor_total"] = "270.375"
    base_insumo_data["rendimiento"] = "0.5"

    cleaned = validate_insumo_data(base_insumo_data)
    assert isinstance(cleaned["cantidad"], float)
    assert isinstance(cleaned["precio_unitario"], float)
    assert isinstance(cleaned["valor_total"], float)
    assert isinstance(cleaned["rendimiento"], float)
    assert cleaned["cantidad"] == 10.0
    assert cleaned["precio_unitario"] == 25.75


# --- Pruebas de edge cases y comportamientos no documentados ---
def test_insumo_procesado_con_normalized_desc_manual(base_insumo_data):
    """Prueba que si se pasa normalized_desc, se ignora y se genera de nuevo."""
    base_insumo_data["descripcion_insumo"] = "Cemento"
    base_insumo_data["normalized_desc"] = "DESCRIPCION_MANUAL"
    insumo = InsumoProcesado(**base_insumo_data)
    assert insumo.normalized_desc == "CEMENTO"


def test_insumo_procesado_con_categoria_diferente(base_insumo_data):
    """Prueba que categoria se sobrescribe por tipo_insumo."""
    base_insumo_data["categoria"] = "OTRO"
    base_insumo_data["tipo_insumo"] = "SUMINISTRO"
    insumo = InsumoProcesado(**base_insumo_data)
    assert insumo.categoria == "SUMINISTRO"


def test_to_dict_serializable(base_insumo_data):
    """Prueba que to_dict() devuelve un diccionario serializable y completo."""
    insumo = InsumoProcesado(**base_insumo_data)
    d = insumo.to_dict()
    assert isinstance(d, dict)
    assert len(d) == 13
    assert d["codigo_apu"] == "APU-001"
    assert d["normalized_desc"] == "CEMENTO PORTLAND 50KG"
    assert d["unidad_apu"] == "M2"
    assert d["tipo_insumo"] == "SUMINISTRO"
