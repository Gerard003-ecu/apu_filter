# app/schemas.py
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InsumoProcesado:
    """
    Estructura base para cualquier insumo de APU.
    
    Campos añadidos para robustecer la compatibilidad:
    - normalized_desc: Descripción normalizada para comparaciones
    - rendimiento: Campo opcional para compatibilidad con ManoDeObra
    """
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
    normalized_desc: str = ""  # Campo añadido para compatibilidad
    rendimiento: float = 0.0   # Campo añadido para compatibilidad con ManoDeObra

    def __post_init__(self):
        """
        Validación básica después de la inicialización.
        """
        self._validate_basic_fields()
        
    def _validate_basic_fields(self):
        """Valida campos básicos requeridos."""
        if not self.codigo_apu or not isinstance(self.codigo_apu, str):
            raise ValueError("codigo_apu es requerido y debe ser string")
            
        if not self.descripcion_insumo or not isinstance(self.descripcion_insumo, str):
            raise ValueError("descripcion_insumo es requerido y debe ser string")
            
        if self.cantidad < 0:
            raise ValueError("cantidad no puede ser negativa")
            
        if self.precio_unitario < 0:
            raise ValueError("precio_unitario no puede ser negativo")
            
        if self.valor_total < 0:
            raise ValueError("valor_total no puede ser negativo")

    def to_dict(self) -> dict:
        """Convierte el objeto a diccionario para serialización."""
        return {
            'codigo_apu': self.codigo_apu,
            'descripcion_apu': self.descripcion_apu,
            'unidad_apu': self.unidad_apu,
            'descripcion_insumo': self.descripcion_insumo,
            'unidad_insumo': self.unidad_insumo,
            'cantidad': self.cantidad,
            'precio_unitario': self.precio_unitario,
            'valor_total': self.valor_total,
            'categoria': self.categoria,
            'formato_origen': self.formato_origen,
            'tipo_insumo': self.tipo_insumo,
            'normalized_desc': self.normalized_desc,
            'rendimiento': self.rendimiento
        }


@dataclass
class ManoDeObra(InsumoProcesado):
    """Esquema específico para Mano de Obra."""
    
    def __post_init__(self):
        """
        Validaciones específicas para Mano de Obra.
        """
        super()._validate_basic_fields()
        self._validate_mo_specific()
        
    def _validate_mo_specific(self):
        """Validaciones específicas de Mano de Obra."""
        if self.rendimiento < 0:
            raise ValueError("rendimiento no puede ser negativo")
            
        # Para MO, si hay rendimiento > 0, validar coherencia con cantidad
        if self.rendimiento > 0 and self.cantidad > 0:
            calculated_quantity = 1.0 / self.rendimiento
            tolerance = 0.0001
            if abs(self.cantidad - calculated_quantity) > tolerance:
                # No lanzamos error pero registramos la inconsistencia
                pass


@dataclass
class Equipo(InsumoProcesado):
    """
    Representa un insumo de tipo 'Equipo' en un APU.

    Hereda todos los atributos de InsumoProcesado y se utiliza para
    identificar y agrupar los costos asociados a la maquinaria y
    herramientas utilizadas en el análisis de precios unitarios.
    """
    
    def __post_init__(self):
        """Validaciones específicas para Equipo."""
        super()._validate_basic_fields()
        self._validate_equipo_specific()
        
    def _validate_equipo_specific(self):
        """Validaciones específicas para equipos."""
        # Los equipos típicamente se miden en tiempo (DIA, HORA)
        time_units = ['DIA', 'HORA', 'HR', 'SEMANA', 'MES']
        if self.unidad_apu not in time_units and self.unidad_insumo not in time_units:
            # No es error, pero puede ser una advertencia
            pass


@dataclass
class Suministro(InsumoProcesado):
    """
    Representa un insumo de tipo 'Suministro' en un APU.

    Este esquema se utiliza para los materiales y productos consumibles
    que se integran directamente en la obra o el producto final.
    """
    
    def __post_init__(self):
        """Validaciones específicas para Suministro."""
        super()._validate_basic_fields()
        self._validate_suministro_specific()
        
    def _validate_suministro_specific(self):
        """Validaciones específicas para suministros."""
        # Los suministros típicamente tienen cantidad > 0
        if self.cantidad <= 0:
            # No lanzamos error porque puede ser un suministro con cantidad cero
            # pero registramos para análisis
            pass


@dataclass
class Transporte(InsumoProcesado):
    """
    Representa un insumo de tipo 'Transporte' en un APU.

    Define los costos asociados al transporte de materiales, equipo o
    personal, heredando la estructura base de InsumoProcesado.
    """
    
    def __post_init__(self):
        """Validaciones específicas para Transporte."""
        super()._validate_basic_fields()
        self._validate_transporte_specific()
        
    def _validate_transporte_specific(self):
        """Validaciones específicas para transporte."""
        # El transporte típicamente se mide en VIAJE, KM, etc.
        transport_units = ['VIAJE', 'KM', 'MILLA', 'TON-KM']
        if (self.unidad_apu not in transport_units and 
            self.unidad_insumo not in transport_units):
            # No es error, pero puede ser una advertencia
            pass


@dataclass
class Otro(InsumoProcesado):
    """
    Representa un insumo no clasificado dentro de las categorías estándar.

    Este esquema actúa como un contenedor para cualquier insumo que no
    corresponda a 'Mano de Obra', 'Equipo', 'Suministro' o 'Transporte',
    asegurando que todos los datos sean capturados.
    """
    
    def __post_init__(self):
        """Validaciones básicas para Otros."""
        super()._validate_basic_fields()


# Diccionario de mapeo para crear instancias dinámicamente
INSUMO_CLASS_MAP = {
    "MANO_DE_OBRA": ManoDeObra,
    "EQUIPO": Equipo,
    "SUMINISTRO": Suministro,
    "TRANSPORTE": Transporte,
    "OTRO": Otro
}


def create_insumo(tipo_insumo: str, **kwargs) -> InsumoProcesado:
    """
    Factory function para crear instancias de insumos dinámicamente.
    
    Args:
        tipo_insumo: Tipo de insumo a crear
        **kwargs: Argumentos para la clase específica
        
    Returns:
        Instancia de la clase de insumo correspondiente
        
    Raises:
        ValueError: Si el tipo_insumo no es válido
    """
    if tipo_insumo not in INSUMO_CLASS_MAP:
        raise ValueError(f"Tipo de insumo no válido: {tipo_insumo}")
    
    insumo_class = INSUMO_CLASS_MAP[tipo_insumo]
    
    try:
        return insumo_class(**kwargs)
    except TypeError as e:
        raise ValueError(f"Error creando insumo tipo {tipo_insumo}: {e}")


def validate_insumo_data(insumo_data: dict) -> dict:
    """
    Valida y limpia datos para crear un insumo.
    
    Args:
        insumo_data: Diccionario con datos del insumo
        
    Returns:
        Diccionario validado y limpiado
    """
    required_fields = [
        'codigo_apu', 'descripcion_apu', 'unidad_apu', 
        'descripcion_insumo', 'unidad_insumo', 'tipo_insumo'
    ]
    
    # Verificar campos requeridos
    for field in required_fields:
        if field not in insumo_data:
            raise ValueError(f"Campo requerido faltante: {field}")
    
    # Asegurar valores por defecto
    defaults = {
        'cantidad': 0.0,
        'precio_unitario': 0.0,
        'valor_total': 0.0,
        'categoria': 'OTRO',
        'formato_origen': 'GENERIC',
        'normalized_desc': '',
        'rendimiento': 0.0
    }
    
    cleaned_data = insumo_data.copy()
    for field, default_value in defaults.items():
        if field not in cleaned_data or cleaned_data[field] is None:
            cleaned_data[field] = default_value
    
    return cleaned_data