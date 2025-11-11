# app/schemas.py
import logging
import re
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

# Configuración básica de logging
logger = logging.getLogger(__name__)

# --- Normalización de unidades ---
UNIDAD_NORMALIZADA_MAP = {
    # Mano de Obra
    'hora': 'HORA', 'hr': 'HORA', 'hrs': 'HORA', 'h': 'HORA',
    'dia': 'DIA', 'd': 'DIA',
    'semana': 'SEMANA', 'sem': 'SEMANA',
    'mes': 'MES',
    # Transporte
    'km': 'KM', 'kilometro': 'KM', 'kilómetro': 'KM',
    'milla': 'MILLA', 'millas': 'MILLA',
    'viaje': 'VIAJE', 'viajes': 'VIAJE',
    'ton-km': 'TON-KM', 'ton km': 'TON-KM',
    # Suministros y otros
    'unidad': 'UNIDAD', 'und': 'UNIDAD', 'u': 'UNIDAD',
    'kg': 'KG', 'gramo': 'GR', 'gr': 'GR',
    'm3': 'M3', 'm²': 'M2', 'm2': 'M2',
    'l': 'L', 'litro': 'L',
}

def normalize_unit(unit: Optional[str]) -> str:
    """Normaliza una unidad de medida a su forma estándar."""
    if not isinstance(unit, str):
        return 'UNIDAD'  # default fallback
    normalized = unit.strip().lower()
    return UNIDAD_NORMALIZADA_MAP.get(normalized, unit.upper())


# --- Tipos de insumo válidos y mapeo ---
VALID_TIPOS_INSUMO = {
    "MANO_DE_OBRA", "EQUIPO", "SUMINISTRO", "TRANSPORTE", "OTRO"
}

def normalize_tipo_insumo(tipo: str) -> str:
    """Normaliza y valida el tipo de insumo."""
    # Manejar enums y strings
    if hasattr(tipo, 'value'):
        tipo = tipo.value

    normalized = str(tipo).strip().upper()
    if normalized not in VALID_TIPOS_INSUMO:
        raise ValueError(f"Tipo de insumo inválido: {tipo}. Valores válidos: {VALID_TIPOS_INSUMO}")
    return normalized


# --- Normalización de descripción ---
def normalize_description(desc: Optional[str]) -> str:
    """Normaliza descripción: elimina espacios extra, convierte a mayúsculas, elimina tildes."""
    if not isinstance(desc, str):
        return ""
    # Eliminar tildes
    import unicodedata
    desc = unicodedata.normalize('NFKD', desc).encode('ASCII', 'ignore').decode('ASCII')
    # Limpiar espacios múltiples
    desc = re.sub(r'\s+', ' ', desc.strip())
    return desc.upper()


@dataclass
class InsumoProcesado:
    """
    Estructura base para cualquier insumo de APU.
    
    Campos añadidos para robustecer la compatibilidad:
    - normalized_desc: Descripción normalizada para comparaciones (generada automáticamente)
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
    normalized_desc: str = ""  # Se genera automáticamente
    rendimiento: float = 0.0   # Se valida en subclases

    def __post_init__(self):
        """Validación inicial y normalización automática."""
        self._normalize_fields()
        self._validate_required_fields()
        self._validate_consistency()

    def _normalize_fields(self):
        """Normaliza campos clave y asegura consistencia."""
        self.codigo_apu = self.codigo_apu.strip() if isinstance(self.codigo_apu, str) else ""
        self.descripcion_apu = normalize_description(self.descripcion_apu)
        self.descripcion_insumo = normalize_description(self.descripcion_insumo)
        self.unidad_apu = normalize_unit(self.unidad_apu)
        self.unidad_insumo = normalize_unit(self.unidad_insumo)
        self.tipo_insumo = normalize_tipo_insumo(self.tipo_insumo)
        self.formato_origen = self.formato_origen.strip().upper() if isinstance(self.formato_origen, str) else "GENERIC"

        # Forzar sincronización de categoria y regeneración de normalized_desc
        self.categoria = self.tipo_insumo
        self.normalized_desc = normalize_description(self.descripcion_insumo)

    def _validate_required_fields(self):
        """Valida campos obligatorios y tipos."""
        if not self.codigo_apu:
            raise ValueError("codigo_apu no puede estar vacío")
        if not isinstance(self.codigo_apu, str):
            raise TypeError("codigo_apu debe ser string")

        if not self.descripcion_insumo:
            raise ValueError("descripcion_insumo no puede estar vacío")
        if not isinstance(self.descripcion_insumo, str):
            raise TypeError("descripcion_insumo debe ser string")

        if not isinstance(self.cantidad, (int, float)) or self.cantidad < 0:
            raise ValueError("cantidad debe ser un número no negativo")

        if not isinstance(self.precio_unitario, (int, float)) or self.precio_unitario < 0:
            raise ValueError("precio_unitario debe ser un número no negativo")

        if not isinstance(self.valor_total, (int, float)) or self.valor_total < 0:
            raise ValueError("valor_total debe ser un número no negativo")

        # Validar coherencia entre valor_total y cálculo esperado
        expected_total = self.cantidad * self.precio_unitario
        tolerance = 0.01  # 1% de tolerancia por redondeo
        if abs(self.valor_total - expected_total) > tolerance:
            warnings.warn(
                f"valor_total ({self.valor_total}) no coincide con cantidad ({self.cantidad}) * precio_unitario ({self.precio_unitario}). "
                f"Valor esperado: {expected_total:.4f}. Considerar ajuste.",
                UserWarning,
                stacklevel=2
            )

    def _validate_consistency(self):
        """Valida coherencia lógica entre campos (sobreescrita por subclases)."""
        # Forzar coherencia entre tipo_insumo y categoria
        if self.categoria != self.tipo_insumo:
            warnings.warn(
                f"Inconsistencia: tipo_insumo='{self.tipo_insumo}' pero categoria='{self.categoria}'. "
                f"Se recomienda alinear ambos campos.",
                UserWarning,
                stacklevel=2
            )

    def to_dict(self) -> dict:
        """Convierte el objeto a diccionario para serialización (automático)."""
        return asdict(self)


@dataclass
class ManoDeObra(InsumoProcesado):
    """Esquema específico para Mano de Obra."""
    jornal: float = 0.0
    def _validate_consistency(self):
        """Validaciones específicas de Mano de Obra."""
        super()._validate_consistency()

        # Validar rendimiento
        if not isinstance(self.rendimiento, (int, float)) or self.rendimiento < 0:
            raise ValueError("rendimiento debe ser un número no negativo")

        # Si rendimiento > 0, validar coherencia con cantidad
        if self.rendimiento > 0:
            calculated_quantity = 1.0 / self.rendimiento
            tolerance = 0.0001
            if abs(self.cantidad - calculated_quantity) > tolerance:
                warnings.warn(
                    f"Mano de Obra: rendimiento={self.rendimiento} sugiere cantidad≈{calculated_quantity:.4f}, "
                    f"pero cantidad={self.cantidad}. Posible error de entrada.",
                    UserWarning,
                    stacklevel=2
                )

        # Validar unidades típicas
        expected_units = {'HORA', 'DIA', 'SEMANA', 'MES'}
        if self.unidad_apu not in expected_units and self.unidad_insumo not in expected_units:
            warnings.warn(
                f"Mano de Obra: unidades '{self.unidad_apu}' o '{self.unidad_insumo}' no son típicas. "
                f"Unidades esperadas: {expected_units}",
                UserWarning,
                stacklevel=2
            )


@dataclass
class Equipo(InsumoProcesado):
    """
    Representa un insumo de tipo 'Equipo' en un APU.
    """

    def _validate_consistency(self):
        """Validaciones específicas para equipos."""
        super()._validate_consistency()

        # Equipos típicamente se miden en tiempo
        time_units = {'DIA', 'HORA', 'SEMANA', 'MES'}
        if self.unidad_apu not in time_units and self.unidad_insumo not in time_units:
            warnings.warn(
                f"Equipo: unidades '{self.unidad_apu}' o '{self.unidad_insumo}' no son típicas para equipos. "
                f"Se recomienda usar: {time_units}",
                UserWarning,
                stacklevel=2
            )

        # Validar que rendimiento sea 0 (no aplica)
        if self.rendimiento != 0:
            warnings.warn(
                f"Equipo: rendimiento={self.rendimiento} no es relevante. Se recomienda dejar en 0.0",
                UserWarning,
                stacklevel=2
            )


@dataclass
class Suministro(InsumoProcesado):
    """
    Representa un insumo de tipo 'Suministro' en un APU.
    """

    def _validate_consistency(self):
        """Validaciones específicas para suministros."""
        super()._validate_consistency()

        # Suministros deben tener cantidad > 0 (salvo excepciones documentadas)
        if self.cantidad == 0:
            warnings.warn(
                "Suministro: cantidad=0.0. ¿Es intencional? Los suministros suelen tener cantidad > 0.",
                UserWarning,
                stacklevel=2
            )

        # Unidades típicas
        common_units = {'KG', 'M2', 'M3', 'L', 'UNIDAD', 'GR', 'TON'}
        if self.unidad_apu not in common_units and self.unidad_insumo not in common_units:
            warnings.warn(
                f"Suministro: unidad '{self.unidad_apu}' o '{self.unidad_insumo}' inusual. "
                f"Unidades comunes: {common_units}",
                UserWarning,
                stacklevel=2
            )

        # Rendimiento no aplica
        if self.rendimiento != 0:
            warnings.warn(
                f"Suministro: rendimiento={self.rendimiento} no es relevante. Se recomienda dejar en 0.0",
                UserWarning,
                stacklevel=2
            )


@dataclass
class Transporte(InsumoProcesado):
    """
    Representa un insumo de tipo 'Transporte' en un APU.
    """

    def _validate_consistency(self):
        """Validaciones específicas para transporte."""
        super()._validate_consistency()

        # Unidades típicas
        transport_units = {'KM', 'VIAJE', 'MILLA', 'TON-KM'}
        if self.unidad_apu not in transport_units and self.unidad_insumo not in transport_units:
            warnings.warn(
                f"Transporte: unidad '{self.unidad_apu}' o '{self.unidad_insumo}' inusual. "
                f"Unidades esperadas: {transport_units}",
                UserWarning,
                stacklevel=2
            )

        # Rendimiento no aplica
        if self.rendimiento != 0:
            warnings.warn(
                f"Transporte: rendimiento={self.rendimiento} no es relevante. Se recomienda dejar en 0.0",
                UserWarning,
                stacklevel=2
            )


@dataclass
class Otro(InsumoProcesado):
    """
    Representa un insumo no clasificado dentro de las categorías estándar.
    """

    def _validate_consistency(self):
        """Validaciones básicas para Otros."""
        super()._validate_consistency()


# --- Diccionario de mapeo para crear instancias dinámicamente ---
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
        ValueError: Si el tipo_insumo no es válido o los argumentos son inválidos
        TypeError: Si los tipos de datos no coinciden
    """
    tipo_normalizado = normalize_tipo_insumo(tipo_insumo)
    kwargs['tipo_insumo'] = tipo_normalizado
    kwargs['categoria'] = tipo_normalizado  # Forzar coherencia

    insumo_class = INSUMO_CLASS_MAP.get(tipo_normalizado)
    if not insumo_class:
        raise ValueError(f"Tipo de insumo no válido: {tipo_insumo}")

    try:
        return insumo_class(**kwargs)
    except TypeError as e:
        raise ValueError(f"Error creando insumo tipo {tipo_normalizado}: {e}. "
                         f"Argumentos recibidos: {list(kwargs.keys())}") from e


def validate_insumo_data(insumo_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida y limpia datos para crear un insumo.
    Normaliza campos, asigna defaults, y valida existencia de campos críticos.
    
    Args:
        insumo_data: Diccionario con datos del insumo
        
    Returns:
        Diccionario validado y limpiado listo para pasar a create_insumo()
        
    Raises:
        ValueError: Si faltan campos obligatorios o tienen tipos inválidos
    """
    required_fields = [
        'codigo_apu', 'descripcion_apu', 'unidad_apu',
        'descripcion_insumo', 'unidad_insumo', 'tipo_insumo'
    ]

    # Verificar campos requeridos
    for field in required_fields:
        if field not in insumo_data or insumo_data[field] is None:
            raise ValueError(f"Campo requerido faltante o nulo: {field}")

    # Asegurar valores por defecto y normalizar tipos
    defaults = {
        'cantidad': 0.0,
        'precio_unitario': 0.0,
        'valor_total': 0.0,
        'categoria': 'OTRO',
        'formato_origen': 'GENERIC',
        'normalized_desc': '',
        'rendimiento': 0.0
    }

    cleaned_data = {}
    for key in insumo_data:
        if key not in defaults and key not in required_fields:
            warnings.warn(f"Campo inesperado en datos de insumo: {key}", UserWarning, stacklevel=2)
        cleaned_data[key] = insumo_data[key]

    for field, default_value in defaults.items():
        value = cleaned_data.get(field)
        if value is None:
            cleaned_data[field] = default_value
        elif field in ['cantidad', 'precio_unitario', 'valor_total', 'rendimiento']:
            try:
                cleaned_data[field] = float(value)
            except (ValueError, TypeError):
                raise ValueError(f"Campo {field} debe ser numérico. Valor recibido: {value}")

    # Validar tipo_insumo aquí también (por si viene mal)
    if 'tipo_insumo' in cleaned_data:
        try:
            cleaned_data['tipo_insumo'] = normalize_tipo_insumo(cleaned_data['tipo_insumo'])
        except ValueError as e:
            raise ValueError(f"Invalid tipo_insumo: {e}")

    return cleaned_data
