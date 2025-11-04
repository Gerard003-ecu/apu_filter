# app/schemas.py
from dataclasses import dataclass


@dataclass
class InsumoProcesado:
    """Estructura base para cualquier insumo de APU."""
    codigo_apu: str
    descripcion_apu: str
    unidad_apu: str
    descripcion_insumo: str
    unidad_insumo: str
    cantidad: float
    precio_unitario: float
    valor_total: float
    categoria: str
    formato_origen: str
    tipo_insumo: str
    normalized_desc: str

@dataclass
class ManoDeObra(InsumoProcesado):
    """Esquema específico para Mano de Obra."""
    rendimiento: float

@dataclass
class Equipo(InsumoProcesado):
    """
    Representa un insumo de tipo 'Equipo' en un APU.

    Hereda todos los atributos de InsumoProcesado y se utiliza para
    identificar y agrupar los costos asociados a la maquinaria y
    herramientas utilizadas en el análisis de precios unitarios.
    """
    pass

@dataclass
class Suministro(InsumoProcesado):
    """
    Representa un insumo de tipo 'Suministro' en un APU.

    Este esquema se utiliza para los materiales y productos consumibles
    que se integran directamente en la obra o el producto final.
    """
    pass

@dataclass
class Transporte(InsumoProcesado):
    """
    Representa un insumo de tipo 'Transporte' en un APU.

    Define los costos asociados al transporte de materiales, equipo o
    personal, heradando la estructura base de InsumoProcesado.
    """
    pass

@dataclass
class Otro(InsumoProcesado):
    """
    Representa un insumo no clasificado dentro de las categorías estándar.

    Este esquema actúa como un contenedor para cualquier insumo que no
    corresponda a 'Mano de Obra', 'Equipo', 'Suministro' o 'Transporte',
    asegurando que todos los datos sean capturados.
    """
    pass
