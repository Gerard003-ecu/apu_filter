# app/schemas.py
from dataclasses import dataclass, field

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
    """Esquema específico para Equipo."""
    pass

@dataclass
class Suministro(InsumoProcesado):
    """Esquema específico para Suministro."""
    pass

@dataclass
class Transporte(InsumoProcesado):
    """Esquema específico para Transporte."""
    pass

@dataclass
class Otro(InsumoProcesado):
    """Esquema para insumos no clasificados."""
    pass
