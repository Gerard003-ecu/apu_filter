"""
Módulo: MIC Agent (Ecualizador Categórico)
==========================================

Implementación de un adaptador de impedancia formalizado como un funtor
entre la categoría de salidas probabilísticas (LLM) y la categoría
determinista estratificada (MIC).

Marco Teórico
=============

Definimos dos categorías:

Categoría LLM (L):
------------------
- Objetos: Respuestas del modelo de lenguaje como Mappings JSON
- Morfismos: Transformaciones de prompt/contexto
- No tiene estructura estratificada inherente

Categoría MIC (M):
------------------
- Objetos: CategoricalState con estratos DIKW certificados
- Morfismos: Vectores tipados por estrato (ver mic_algebra.py)
- Estructura: Poset de estratos con clausura transitiva

Funtor de Impedancia F: L → M:
------------------------------
El MICAgent implementa un funtor F que:
1. Asigna objetos: F_obj(llm_output) = CategoricalState
2. Preserva estructura: F respeta las restricciones de estrato

La adaptación de impedancia consiste en:
1. Sensado de estrato objetivo
2. Selección de contrato (Silo A) y cartucho TOON (Silo B)
3. Validación de clausura transitiva DIKW
4. Encapsulación monádica del resultado
5. Proyección hacia la MIC

Protocolo TOON (Tabular Object-Oriented Notation):
--------------------------------------------------
Formato de compresión determinista para contexto:
    --- INICIO {cartridge_id} ---
    {header_template}
    {key}|{json_value}
    ...
    --- FIN TOON ---

La compresión es idempotente y tiene inversa definida.

Invariantes:
------------
1. Compatibilidad de estrato: cod(F(x)) ∈ stratum_objetivo
2. Clausura transitiva: validated_strata ⊇ stratum.requires()
3. Validación contractual: payload ⊨ schema
4. Monadicidad del error: error se propaga sin transformación
5. Trazabilidad: cada operación genera CategoricalEqualizerSeed

Referencias:
- Mac Lane, S. (1971). Categories for the Working Mathematician
- Baez, J. & Stay, M. (2011). Physics, Topology, Logic and Computation
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    ClassVar,
    Deque,
    Dict,
    Final,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

from app.core.schemas import Stratum
from app.core.mic_algebra import (
    CategoricalState,
    _canonicalize,
)
from app.adapters.tools_interface import MICRegistry



def _stable_hash(data: Any) -> str:
    """Genera un hash SHA-256 estable a partir de cualquier estructura de datos canonicalizable."""
    serialized = json.dumps(
        _canonicalize(data),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

logger = logging.getLogger("MIC.Agent.CategoricalEqualizer")


# =============================================================================
# CONSTANTES
# =============================================================================

# Límite de entradas en la traza de auditoría
MAX_AUDIT_TRAIL_SIZE: Final[int] = 10_000

# Prefijos del protocolo TOON
TOON_START_MARKER: Final[str] = "--- INICIO"
TOON_END_MARKER: Final[str] = "--- FIN TOON ---"
TOON_FIELD_SEPARATOR: Final[str] = "|"

# Versión del protocolo de encapsulación
ENCAPSULATION_PROTOCOL_VERSION: Final[str] = "1.0.0"


# =============================================================================
# EXCEPCIONES
# =============================================================================

class MICAgentError(Exception):
    """Excepción base para errores del MIC Agent."""


class StratumResolutionError(MICAgentError):
    """Error al resolver el estrato de un vector."""


class ContractValidationError(MICAgentError):
    """Error en validación de contrato JSON."""


class ClosureViolationError(MICAgentError):
    """Violación de clausura transitiva DIKW."""


class AlgebraicVetoError(MICAgentError):
    """Veto por violación de invariantes algebraicos."""


class TOONCompressionError(MICAgentError):
    """Error en compresión/descompresión TOON."""


class SiloAccessError(MICAgentError):
    """Error accediendo a silos de contratos/cartuchos."""


class ProjectionError(MICAgentError):
    """Error en proyección hacia la MIC."""


# =============================================================================
# ENUMERACIONES
# =============================================================================

class ImpedanceMatchStatus(str, Enum):
    """
    Veredicto del adaptador de impedancia.
    
    Taxonomía:
    - LAMINAR_PROJECTION: Adaptación exitosa, flujo laminar hacia MIC
    - STRATUM_MISMATCH_REJECTED: Clausura transitiva violada
    - TOON_COMPRESSION_ERROR: Fallo en protocolo TOON
    - ALGEBRAIC_VETO: Violación de invariante duro
    - SCHEMA_VALIDATION_ERROR: Payload no satisface contrato
    - MIC_RESOLUTION_ERROR: Error en resolución de vector MIC
    - INPUT_TYPE_ERROR: Tipo de entrada inválido
    """
    LAMINAR_PROJECTION = "LAMINAR_PROJECTION"
    STRATUM_MISMATCH_REJECTED = "STRATUM_MISMATCH_REJECTED"
    TOON_COMPRESSION_ERROR = "TOON_COMPRESSION_ERROR"
    ALGEBRAIC_VETO = "ALGEBRAIC_VETO"
    SCHEMA_VALIDATION_ERROR = "SCHEMA_VALIDATION_ERROR"
    MIC_RESOLUTION_ERROR = "MIC_RESOLUTION_ERROR"
    INPUT_TYPE_ERROR = "INPUT_TYPE_ERROR"


class ValidationSeverity(Enum):
    """Severidad de errores de validación."""
    ERROR = auto()
    WARNING = auto()
    INFO = auto()


# =============================================================================
# TIPOS
# =============================================================================

T = TypeVar("T")
JSONValue = Union[None, bool, int, float, str, List["JSONValue"], Dict[str, "JSONValue"]]
JSONSchema = Dict[str, Any]
PayloadType = Mapping[str, Any]

# Validador de invariante algebraico
AlgebraicValidator = Callable[[Stratum, PayloadType], Optional[str]]


# =============================================================================
# PROTOCOLOS
# =============================================================================

@runtime_checkable
class VectorInfoProvider(Protocol):
    """Protocolo para obtener información de vectores MIC."""
    def get_vector_info(self, vector_name: str) -> Optional[Dict[str, Any]]: ...


@runtime_checkable
class ProjectionTarget(Protocol):
    """Protocolo para proyección hacia MIC."""
    def project_intent(
        self,
        target_basis_vector: str,
        stratum_target: int,
        validated_subspaces: List[str],
        orthogonality_guarantee: float,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]: ...


# =============================================================================
# DATACLASSES DE AUDITORÍA
# =============================================================================

@dataclass(frozen=True, slots=True)
class SchemaValidationResult:
    """
    Resultado de validación de schema JSON.
    
    Attributes:
        is_valid: Si la validación fue exitosa
        errors: Lista de errores encontrados
        warnings: Lista de advertencias
        path: Ruta JSON donde ocurrió el error (si aplica)
    """
    is_valid: bool
    errors: Tuple[str, ...] = field(default_factory=tuple)
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    path: str = "$"
    
    @classmethod
    def success(cls) -> "SchemaValidationResult":
        """Factory para resultado exitoso."""
        return cls(is_valid=True)
    
    @classmethod
    def failure(
        cls, 
        error: str, 
        path: str = "$"
    ) -> "SchemaValidationResult":
        """Factory para resultado fallido."""
        return cls(is_valid=False, errors=(error,), path=path)
    
    @classmethod
    def merge(
        cls, 
        results: Iterable["SchemaValidationResult"]
    ) -> "SchemaValidationResult":
        """Combina múltiples resultados."""
        all_errors: List[str] = []
        all_warnings: List[str] = []
        
        for r in results:
            all_errors.extend(r.errors)
            all_warnings.extend(r.warnings)
        
        return cls(
            is_valid=len(all_errors) == 0,
            errors=tuple(all_errors),
            warnings=tuple(all_warnings),
        )

    @property
    def error(self) -> Optional[str]:
        """Primer error, o None si es válido."""
        return self.errors[0] if self.errors else None


@dataclass(frozen=True, slots=True)
class CategoricalEqualizerSeed:
    """
    Traza inmutable de auditoría Zero-Trust.
    
    Registra cada operación del ecualizador categórico para
    reproducibilidad y auditoría.
    
    Invariantes:
    - Todos los campos son deterministas
    - Los hashes son calculables independientemente
    - El timestamp es el único campo no determinista
    """
    target_vector: str
    target_stratum: Stratum
    silo_a_contract_id: str
    silo_b_cartridge_id: str
    impedance_match_status: ImpedanceMatchStatus
    token_compression_ratio: float = 0.0
    raw_telemetry_hash: str = ""
    llm_output_hash: str = ""
    validation_errors: Tuple[str, ...] = field(default_factory=tuple)
    protocol_version: str = ENCAPSULATION_PROTOCOL_VERSION
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "target_vector": self.target_vector,
            "target_stratum": self.target_stratum.name,
            "silo_a_contract_id": self.silo_a_contract_id,
            "silo_b_cartridge_id": self.silo_b_cartridge_id,
            "impedance_match_status": self.impedance_match_status.value,
            "token_compression_ratio": self.token_compression_ratio,
            "raw_telemetry_hash": self.raw_telemetry_hash,
            "llm_output_hash": self.llm_output_hash,
            "validation_errors": list(self.validation_errors),
            "protocol_version": self.protocol_version,
            "timestamp": self.timestamp,
        }
    
    def compute_hash(self) -> str:
        """Calcula hash determinista de la semilla."""
        # Excluir timestamp del hash para determinismo
        data = {
            k: v for k, v in self.to_dict().items()
            if k != "timestamp"
        }
        return _stable_hash(data)


@dataclass(frozen=True)
class TOONDocument:
    """
    Documento TOON (Tabular Object-Oriented Notation).
    
    Representa la forma comprimida de telemetría con
    metadatos de cartucho.
    """
    cartridge_id: str
    header_template: str
    records: Tuple[Tuple[str, str], ...]  # (key, json_value)
    
    def render(self) -> str:
        """Renderiza el documento TOON como string."""
        lines = [
            f"{TOON_START_MARKER} {self.cartridge_id} ---",
            self.header_template,
        ]
        for key, value in self.records:
            lines.append(f"{key}{TOON_FIELD_SEPARATOR}{value}")
        lines.append(TOON_END_MARKER)
        return "\n".join(lines)
    
    @classmethod
    def parse(cls, content: str) -> "TOONDocument":
        """
        Parsea un documento TOON desde string.
        
        Raises:
            TOONCompressionError: Si el formato es inválido
        """
        lines = content.strip().split("\n")
        
        if len(lines) < 3:
            raise TOONCompressionError("Documento TOON demasiado corto")
        
        # Parsear header
        header_line = lines[0]
        if not header_line.startswith(TOON_START_MARKER):
            raise TOONCompressionError(
                f"Marcador de inicio inválido: {header_line[:50]}"
            )
        
        # Extraer cartridge_id
        try:
            cartridge_id = header_line.split(TOON_START_MARKER)[1].strip().rstrip("-").strip()
        except IndexError:
            raise TOONCompressionError("No se pudo extraer cartridge_id")
        
        # Verificar footer
        if lines[-1].strip() != TOON_END_MARKER:
            raise TOONCompressionError("Marcador de fin faltante o inválido")
        
        # Parsear template y records
        header_lines = []
        records: List[Tuple[str, str]] = []
        
        # La convención es que el header finaliza en la primera línea que contenga
        # valores escapados (ej: "value") o una estructura JSON.
        # Alternativamente, si estamos procesando la última línea del bloque,
        # es un record si tiene el separador.
        # Una manera más rigurosa: el primer campo de un record no suele ser 'key' o 'col1'.
        # Y sabemos que TOON_FIELD_SEPARATOR se usa tanto en el template como en los records.
        # Sin embargo, los records son `key|json_value`.

        in_records = False
        for line in lines[1:-1]:
            if not in_records:
                # Determinar si la línea es un record.
                # Heurística robusta: si tiene separador y la segunda parte empieza por un carácter JSON
                # válido (", {, [, número, true, false, null) y no es simplemente 'value' o 'col2'.
                if TOON_FIELD_SEPARATOR in line:
                    parts = line.split(TOON_FIELD_SEPARATOR, 1)
                    val = parts[1].strip()
                    if val.startswith('"') or val.startswith('{') or val.startswith('[') or val in ('true', 'false', 'null') or val.lstrip('-').replace('.', '', 1).isdigit():
                        in_records = True

            if in_records:
                if TOON_FIELD_SEPARATOR in line:
                    parts = line.split(TOON_FIELD_SEPARATOR, 1)
                    records.append((parts[0], parts[1]))
            else:
                header_lines.append(line)

        header_template = "\n".join(header_lines)
        
        return cls(
            cartridge_id=cartridge_id,
            header_template=header_template,
            records=tuple(records),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Reconstruye el diccionario original."""
        result: Dict[str, Any] = {}
        for key, json_value in self.records:
            try:
                result[key] = json.loads(json_value)
            except json.JSONDecodeError:
                result[key] = json_value
        return result


# =============================================================================
# CONTRATOS Y CARTUCHOS
# =============================================================================

@dataclass(frozen=True)
class SiloAContract:
    """
    Contrato JSON Schema del Silo A.
    
    Define la estructura esperada para un estrato específico.
    """
    contract_id: str
    stratum: Stratum
    schema: JSONSchema
    description: str = ""
    version: str = "1.0.0"
    
    def validate_schema_integrity(self) -> bool:
        """Verifica integridad básica del schema."""
        if not isinstance(self.schema, dict):
            return False
        if "type" not in self.schema:
            return False
        return True


@dataclass(frozen=True)
class SiloBCartridge:
    """
    Cartucho TOON del Silo B.
    
    Define el formato de compresión para un estrato.
    """
    cartridge_id: str
    stratum: Stratum
    header_template: str
    field_definitions: Tuple[str, ...] = field(default_factory=tuple)
    description: str = ""
    version: str = "1.0.0"


# =============================================================================
# UTILIDADES
# =============================================================================

def normalize_stratum(value: Any) -> Stratum:
    """
    Normaliza un valor a Stratum con validación estricta.
    
    Args:
        value: Valor a normalizar (Stratum, int, o str)
    
    Returns:
        Stratum normalizado
    
    Raises:
        StratumResolutionError: Si el valor no puede normalizarse
    """
    if isinstance(value, Stratum):
        return value
    
    if isinstance(value, int):
        try:
            return Stratum(value)
        except ValueError as e:
            raise StratumResolutionError(
                f"Valor entero inválido para estrato: {value}"
            ) from e
    
    if isinstance(value, str):
        # Intentar por nombre
        try:
            return Stratum[value.upper()]
        except KeyError:
            pass
        
        # Intentar como entero en string
        try:
            return Stratum(int(value))
        except (ValueError, KeyError) as e:
            raise StratumResolutionError(
                f"String inválido para estrato: '{value}'"
            ) from e
    
    raise StratumResolutionError(
        f"Tipo no soportado para estrato: {type(value).__name__}"
    )


def python_type_matches(expected_type: str, value: Any) -> bool:
    """
    Verifica si un valor Python corresponde a un tipo JSON Schema.
    
    Args:
        expected_type: Tipo JSON Schema esperado
        value: Valor a verificar
    
    Returns:
        True si el tipo coincide
    """
    type_mapping: Dict[str, Callable[[Any], bool]] = {
        "null": lambda v: v is None,
        "boolean": lambda v: isinstance(v, bool),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "string": lambda v: isinstance(v, str),
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, Mapping),
    }
    
    checker = type_mapping.get(expected_type)
    if checker is None:
        # Tipo desconocido, aceptar
        return True
    
    return checker(value)


def compute_json_path(base: str, key: Union[str, int]) -> str:
    """Construye ruta JSON para mensajes de error."""
    if isinstance(key, int):
        return f"{base}[{key}]"
    return f"{base}.{key}" if base != "$" else f"$.{key}"


# =============================================================================
# VALIDADOR DE SCHEMA
# =============================================================================

class SchemaValidator:
    """
    Validador JSON Schema determinista.
    
    Soporta un subconjunto de JSON Schema Draft-07:
    - type (null, boolean, integer, number, string, array, object)
    - required
    - properties
    - items
    - minimum, maximum
    - minLength, maxLength
    - enum
    - const
    - pattern (básico)
    
    No soporta (deliberadamente para determinismo):
    - $ref (referencias externas)
    - allOf, anyOf, oneOf
    - if/then/else
    - format con validación externa
    """

    def __init__(self) -> None:
        self._validators: Dict[str, Callable[..., SchemaValidationResult]] = {
            "type": self._validate_type,
            "required": self._validate_required,
            "properties": self._validate_properties,
            "items": self._validate_items,
            "minimum": self._validate_minimum,
            "maximum": self._validate_maximum,
            "exclusiveMinimum": self._validate_exclusive_minimum,
            "exclusiveMaximum": self._validate_exclusive_maximum,
            "minLength": self._validate_min_length,
            "maxLength": self._validate_max_length,
            "enum": self._validate_enum,
            "const": self._validate_const,
            "minItems": self._validate_min_items,
            "maxItems": self._validate_max_items,
        }

    def validate(
        self,
        schema: JSONSchema,
        payload: Any,
        path: str = "$",
    ) -> SchemaValidationResult:
        """
        Valida un payload contra un JSON Schema.
        
        Args:
            schema: JSON Schema a validar contra
            payload: Datos a validar
            path: Ruta JSON actual (para mensajes de error)
        
        Returns:
            Resultado de validación con errores detallados
        """
        results: List[SchemaValidationResult] = []
        
        for keyword, constraint in schema.items():
            validator = self._validators.get(keyword)
            if validator is not None:
                result = validator(constraint, payload, schema, path)
                results.append(result)
        
        return SchemaValidationResult.merge(results)

    def _validate_type(
        self,
        expected_type: Union[str, List[str]],
        value: Any,
        schema: JSONSchema,
        path: str,
    ) -> SchemaValidationResult:
        """Valida restricción 'type'."""
        types = [expected_type] if isinstance(expected_type, str) else expected_type
        
        for t in types:
            if python_type_matches(t, value):
                return SchemaValidationResult.success()
        
        actual_type = type(value).__name__
        return SchemaValidationResult.failure(
            f"Tipo inválido en '{path}': esperado {types}, recibido '{actual_type}'",
            path,
        )

    def _validate_required(
        self,
        required_keys: List[str],
        value: Any,
        schema: JSONSchema,
        path: str,
    ) -> SchemaValidationResult:
        """Valida restricción 'required'."""
        if not isinstance(value, Mapping):
            return SchemaValidationResult.success()  # Se valida en 'type'
        
        missing = [k for k in required_keys if k not in value]
        if missing:
            return SchemaValidationResult.failure(
                f"Claves requeridas faltantes en '{path}': {missing}",
                path,
            )
        
        return SchemaValidationResult.success()

    def _validate_properties(
        self,
        properties: Dict[str, JSONSchema],
        value: Any,
        schema: JSONSchema,
        path: str,
    ) -> SchemaValidationResult:
        """Valida restricción 'properties' recursivamente."""
        if not isinstance(value, Mapping):
            return SchemaValidationResult.success()
        
        results: List[SchemaValidationResult] = []
        
        for prop_name, prop_schema in properties.items():
            if prop_name in value:
                prop_path = compute_json_path(path, prop_name)
                result = self.validate(prop_schema, value[prop_name], prop_path)
                results.append(result)
        
        return SchemaValidationResult.merge(results)

    def _validate_items(
        self,
        items_schema: JSONSchema,
        value: Any,
        schema: JSONSchema,
        path: str,
    ) -> SchemaValidationResult:
        """Valida restricción 'items' para arrays."""
        if not isinstance(value, list):
            return SchemaValidationResult.success()
        
        results: List[SchemaValidationResult] = []
        
        for i, item in enumerate(value):
            item_path = compute_json_path(path, i)
            result = self.validate(items_schema, item, item_path)
            results.append(result)
        
        return SchemaValidationResult.merge(results)

    def _validate_minimum(
        self,
        minimum: Union[int, float],
        value: Any,
        schema: JSONSchema,
        path: str,
    ) -> SchemaValidationResult:
        """Valida restricción 'minimum'."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return SchemaValidationResult.success()
        
        if value < minimum:
            return SchemaValidationResult.failure(
                f"Valor en '{path}' ({value}) menor que mínimo ({minimum})",
                path,
            )
        
        return SchemaValidationResult.success()

    def _validate_maximum(
        self,
        maximum: Union[int, float],
        value: Any,
        schema: JSONSchema,
        path: str,
    ) -> SchemaValidationResult:
        """Valida restricción 'maximum'."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return SchemaValidationResult.success()
        
        if value > maximum:
            return SchemaValidationResult.failure(
                f"Valor en '{path}' ({value}) mayor que máximo ({maximum})",
                path,
            )
        
        return SchemaValidationResult.success()

    def _validate_exclusive_minimum(
        self,
        minimum: Union[int, float],
        value: Any,
        schema: JSONSchema,
        path: str,
    ) -> SchemaValidationResult:
        """Valida restricción 'exclusiveMinimum'."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return SchemaValidationResult.success()
        
        if value <= minimum:
            return SchemaValidationResult.failure(
                f"Valor en '{path}' ({value}) no mayor estricto que {minimum}",
                path,
            )
        
        return SchemaValidationResult.success()

    def _validate_exclusive_maximum(
        self,
        maximum: Union[int, float],
        value: Any,
        schema: JSONSchema,
        path: str,
    ) -> SchemaValidationResult:
        """Valida restricción 'exclusiveMaximum'."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return SchemaValidationResult.success()
        
        if value >= maximum:
            return SchemaValidationResult.failure(
                f"Valor en '{path}' ({value}) no menor estricto que {maximum}",
                path,
            )
        
        return SchemaValidationResult.success()

    def _validate_min_length(
        self,
        min_length: int,
        value: Any,
        schema: JSONSchema,
        path: str,
    ) -> SchemaValidationResult:
        """Valida restricción 'minLength' para strings."""
        if not isinstance(value, str):
            return SchemaValidationResult.success()
        
        if len(value) < min_length:
            return SchemaValidationResult.failure(
                f"String en '{path}' (len={len(value)}) menor que minLength ({min_length})",
                path,
            )
        
        return SchemaValidationResult.success()

    def _validate_max_length(
        self,
        max_length: int,
        value: Any,
        schema: JSONSchema,
        path: str,
    ) -> SchemaValidationResult:
        """Valida restricción 'maxLength' para strings."""
        if not isinstance(value, str):
            return SchemaValidationResult.success()
        
        if len(value) > max_length:
            return SchemaValidationResult.failure(
                f"String en '{path}' (len={len(value)}) mayor que maxLength ({max_length})",
                path,
            )
        
        return SchemaValidationResult.success()

    def _validate_enum(
        self,
        allowed_values: List[Any],
        value: Any,
        schema: JSONSchema,
        path: str,
    ) -> SchemaValidationResult:
        """Valida restricción 'enum'."""
        if value not in allowed_values:
            return SchemaValidationResult.failure(
                f"Valor en '{path}' ({value!r}) no está en enum: {allowed_values}",
                path,
            )
        
        return SchemaValidationResult.success()

    def _validate_const(
        self,
        const_value: Any,
        value: Any,
        schema: JSONSchema,
        path: str,
    ) -> SchemaValidationResult:
        """Valida restricción 'const'."""
        if value != const_value:
            return SchemaValidationResult.failure(
                f"Valor en '{path}' ({value!r}) no es constante esperada ({const_value!r})",
                path,
            )
        
        return SchemaValidationResult.success()

    def _validate_min_items(
        self,
        min_items: int,
        value: Any,
        schema: JSONSchema,
        path: str,
    ) -> SchemaValidationResult:
        """Valida restricción 'minItems' para arrays."""
        if not isinstance(value, list):
            return SchemaValidationResult.success()
        
        if len(value) < min_items:
            return SchemaValidationResult.failure(
                f"Array en '{path}' (len={len(value)}) menor que minItems ({min_items})",
                path,
            )
        
        return SchemaValidationResult.success()

    def _validate_max_items(
        self,
        max_items: int,
        value: Any,
        schema: JSONSchema,
        path: str,
    ) -> SchemaValidationResult:
        """Valida restricción 'maxItems' para arrays."""
        if not isinstance(value, list):
            return SchemaValidationResult.success()
        
        if len(value) > max_items:
            return SchemaValidationResult.failure(
                f"Array en '{path}' (len={len(value)}) mayor que maxItems ({max_items})",
                path,
            )
        
        return SchemaValidationResult.success()


# =============================================================================
# VALIDADORES ALGEBRAICOS
# =============================================================================

class AlgebraicVetoRegistry:
    """
    Registro de validadores de invariantes algebraicos por estrato.
    
    Los vetos algebraicos son restricciones físicas/lógicas que no pueden
    violarse independientemente del output del LLM.
    """

    def __init__(self) -> None:
        self._validators: Dict[Stratum, List[AlgebraicValidator]] = {
            s: [] for s in Stratum
        }
        self._register_default_validators()

    def _register_default_validators(self) -> None:
        """Registra validadores algebraicos por defecto."""
        
        # PHYSICS: Leyes de conservación
        def physics_conservation(stratum: Stratum, payload: PayloadType) -> Optional[str]:
            # Segunda ley de la termodinámica
            dissipated = payload.get("dissipated_power")
            if dissipated is not None and isinstance(dissipated, (int, float)):
                if dissipated < 0:
                    return "Violación termodinámica: dissipated_power < 0"
            
            # Conservación de energía
            energy_in = payload.get("energy_input", 0)
            energy_out = payload.get("energy_output", 0)
            if isinstance(energy_in, (int, float)) and isinstance(energy_out, (int, float)):
                if energy_out > energy_in * 1.001:  # Tolerancia numérica
                    return "Violación de conservación: energy_output > energy_input"
            
            return None
        
        self._validators[Stratum.PHYSICS].append(physics_conservation)
        
        # TACTICS: Restricciones de estabilidad
        def tactics_stability(stratum: Stratum, payload: PayloadType) -> Optional[str]:
            stability = payload.get("pyramid_stability_index")
            if stability is not None and isinstance(stability, (int, float)):
                if stability < 0 or stability > 1:
                    return f"Índice de estabilidad fuera de rango [0,1]: {stability}"
            return None
        
        self._validators[Stratum.TACTICS].append(tactics_stability)
        
        # STRATEGY: Restricciones de fricción territorial
        def strategy_friction(stratum: Stratum, payload: PayloadType) -> Optional[str]:
            friction = payload.get("territorial_friction")
            if friction is not None and isinstance(friction, (int, float)):
                if friction < 1.0:
                    return f"Fricción territorial debe ser >= 1.0: {friction}"
            return None
        
        self._validators[Stratum.STRATEGY].append(strategy_friction)
        
        # WISDOM: Veredictos válidos
        def wisdom_verdict(stratum: Stratum, payload: PayloadType) -> Optional[str]:
            verdict = payload.get("final_verdict")
            valid_verdicts = {"VIABLE", "PRECAUCION", "RECHAZAR"}
            if verdict is not None and verdict not in valid_verdicts:
                return f"Veredicto inválido '{verdict}', debe ser uno de {valid_verdicts}"
            return None
        
        self._validators[Stratum.WISDOM].append(wisdom_verdict)

    def register_validator(
        self,
        stratum: Stratum,
        validator: AlgebraicValidator,
    ) -> None:
        """Registra un validador algebraico adicional."""
        self._validators[stratum].append(validator)

    def validate(
        self,
        stratum: Stratum,
        payload: PayloadType,
    ) -> List[str]:
        """
        Ejecuta todos los validadores para un estrato.
        
        Returns:
            Lista de errores de veto (vacía si todo OK)
        """
        errors: List[str] = []
        
        for validator in self._validators.get(stratum, []):
            try:
                error = validator(stratum, payload)
                if error:
                    errors.append(error)
            except Exception as e:
                errors.append(f"Error en validador algebraico: {e}")
        
        return errors


# =============================================================================
# GESTIÓN DE SILOS
# =============================================================================

class SiloManager:
    """
    Administrador de Silos A (contratos JSON) y B (cartuchos TOON).
    
    Los silos están indexados por estrato DIKW y pueden tener
    múltiples contratos/cartuchos por estrato.
    """

    def __init__(self) -> None:
        self._silo_a: Dict[Stratum, Dict[str, SiloAContract]] = {}
        self._silo_b: Dict[Stratum, Dict[str, SiloBCartridge]] = {}
        self._default_contract_selector: Callable[[Dict[str, SiloAContract], str], str] = (
            lambda contracts, vector: next(iter(contracts), "Generic_Contract")
        )
        self._default_cartridge_selector: Callable[[Dict[str, SiloBCartridge], str], str] = (
            lambda cartridges, vector: next(iter(cartridges), "Generic_Cartridge")
        )
        
        self._initialize_default_silos()

    def _initialize_default_silos(self) -> None:
        """Inicializa silos con contratos y cartuchos por defecto."""
        
        # Silo A: Contratos JSON por estrato
        self._register_contract(SiloAContract(
            contract_id="PHS_Conservation_Seed",
            stratum=Stratum.PHYSICS,
            schema={
                "type": "object",
                "required": ["dissipated_power"],
                "properties": {
                    "dissipated_power": {"type": "number", "minimum": 0},
                    "energy_input": {"type": "number", "minimum": 0},
                    "energy_output": {"type": "number", "minimum": 0},
                },
            },
            description="Contrato de conservación termodinámica",
        ))
        
        self._register_contract(SiloAContract(
            contract_id="Logistical_Topology_Seed",
            stratum=Stratum.TACTICS,
            schema={
                "type": "object",
                "required": ["pyramid_stability_index"],
                "properties": {
                    "pyramid_stability_index": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "flow_efficiency": {"type": "number"},
                },
            },
            description="Contrato de topología logística",
        ))
        
        self._register_contract(SiloAContract(
            contract_id="Riemannian_Friction_Contract",
            stratum=Stratum.STRATEGY,
            schema={
                "type": "object",
                "required": ["territorial_friction"],
                "properties": {
                    "territorial_friction": {"type": "number", "minimum": 1.0},
                    "risk_coupling": {"type": "number"},
                },
            },
            description="Contrato de fricción Riemanniana",
        ))
        
        self._register_contract(SiloAContract(
            contract_id="Acta_Deliberacion_Seed",
            stratum=Stratum.WISDOM,
            schema={
                "type": "object",
                "required": ["final_verdict"],
                "properties": {
                    "final_verdict": {
                        "type": "string",
                        "enum": ["VIABLE", "PRECAUCION", "RECHAZAR"],
                    },
                    "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "rationale": {"type": "string"},
                },
            },
            description="Contrato de acta de deliberación",
        ))
        
        # Silo B: Cartuchos TOON por estrato
        self._register_cartridge(SiloBCartridge(
            cartridge_id="Maxwell_FDTD_TOON_Cartridge",
            stratum=Stratum.PHYSICS,
            header_template="Malla_Yee_Leapfrog\nkey|value|unit|confidence",
            field_definitions=("key", "value", "unit", "confidence"),
            description="Cartucho FDTD para física electromagnética",
        ))
        
        self._register_cartridge(SiloBCartridge(
            cartridge_id="Persistence_Barcode_TOON_Cartridge",
            stratum=Stratum.TACTICS,
            header_template="Diagrama_Persistencia_API\nkey|value|window|entropy",
            field_definitions=("key", "value", "window", "entropy"),
            description="Cartucho de persistencia topológica",
        ))
        
        self._register_cartridge(SiloBCartridge(
            cartridge_id="Riemannian_TOON_Cartridge",
            stratum=Stratum.STRATEGY,
            header_template="Tensor_Covarianza_Riesgos_Acoplados\nkey|value|coupling",
            field_definitions=("key", "value", "coupling"),
            description="Cartucho Riemanniano para estrategia",
        ))
        
        self._register_cartridge(SiloBCartridge(
            cartridge_id="Telemetry_Passport_TOON_Cartridge",
            stratum=Stratum.WISDOM,
            header_template="Pasaporte_Digital_Transaccional\nkey|value|semantic_role",
            field_definitions=("key", "value", "semantic_role"),
            description="Cartucho de pasaporte de telemetría",
        ))

    def _register_contract(self, contract: SiloAContract) -> None:
        """Registra un contrato en Silo A."""
        if contract.stratum not in self._silo_a:
            self._silo_a[contract.stratum] = {}
        self._silo_a[contract.stratum][contract.contract_id] = contract

    def _register_cartridge(self, cartridge: SiloBCartridge) -> None:
        """Registra un cartucho en Silo B."""
        if cartridge.stratum not in self._silo_b:
            self._silo_b[cartridge.stratum] = {}
        self._silo_b[cartridge.stratum][cartridge.cartridge_id] = cartridge

    def fetch_contract(
        self,
        stratum: Stratum,
        target_vector: str,
    ) -> Tuple[str, JSONSchema]:
        """
        Obtiene contrato JSON para un estrato y vector.
        
        Args:
            stratum: Estrato DIKW objetivo
            target_vector: Nombre del vector MIC
        
        Returns:
            Tupla (contract_id, schema)
        
        Raises:
            SiloAccessError: Si no hay contratos para el estrato
        """
        contracts = self._silo_a.get(stratum, {})
        
        if not contracts:
            # Retornar contrato genérico
            return "Generic_Contract", {"type": "object", "properties": {}}
        
        contract_id = self._default_contract_selector(contracts, target_vector)
        contract = contracts.get(contract_id)
        
        if contract is None:
            raise SiloAccessError(
                f"Contrato '{contract_id}' no encontrado para estrato {stratum.name}"
            )
        
        return contract.contract_id, contract.schema

    def fetch_cartridge(
        self,
        stratum: Stratum,
        target_vector: str,
    ) -> Tuple[str, str]:
        """
        Obtiene cartucho TOON para un estrato y vector.
        
        Args:
            stratum: Estrato DIKW objetivo
            target_vector: Nombre del vector MIC
        
        Returns:
            Tupla (cartridge_id, header_template)
        
        Raises:
            SiloAccessError: Si no hay cartuchos para el estrato
        """
        cartridges = self._silo_b.get(stratum, {})
        
        if not cartridges:
            return "Generic_Cartridge", "Tabla_Generica\nkey|value"
        
        cartridge_id = self._default_cartridge_selector(cartridges, target_vector)
        cartridge = cartridges.get(cartridge_id)
        
        if cartridge is None:
            raise SiloAccessError(
                f"Cartucho '{cartridge_id}' no encontrado para estrato {stratum.name}"
            )
        
        return cartridge.cartridge_id, cartridge.header_template

    def list_contracts(self, stratum: Optional[Stratum] = None) -> List[str]:
        """Lista IDs de contratos, opcionalmente filtrados por estrato."""
        if stratum is not None:
            return list(self._silo_a.get(stratum, {}).keys())
        
        result: List[str] = []
        for contracts in self._silo_a.values():
            result.extend(contracts.keys())
        return result

    def list_cartridges(self, stratum: Optional[Stratum] = None) -> List[str]:
        """Lista IDs de cartuchos, opcionalmente filtrados por estrato."""
        if stratum is not None:
            return list(self._silo_b.get(stratum, {}).keys())
        
        result: List[str] = []
        for cartridges in self._silo_b.values():
            result.extend(cartridges.keys())
        return result


# =============================================================================
# COMPRESOR TOON
# =============================================================================

class TOONCompressor:
    """
    Compresor determinista de telemetría a formato TOON.
    
    Propiedades:
    - Determinista: misma entrada → misma salida
    - Reversible: parse(render(x)) ≈ x
    - Estable: ordenamiento consistente de claves
    """

    def compress(
        self,
        telemetry: PayloadType,
        cartridge_id: str,
        header_template: str,
    ) -> TOONDocument:
        """
        Comprime telemetría a documento TOON.
        
        Args:
            telemetry: Datos de telemetría
            cartridge_id: ID del cartucho a usar
            header_template: Template del header
        
        Returns:
            Documento TOON comprimido
        """
        records: List[Tuple[str, str]] = []
        
        for key in sorted(telemetry.keys()):
            value = telemetry[key]
            json_value = json.dumps(value, ensure_ascii=False, default=str)
            records.append((str(key), json_value))
        
        return TOONDocument(
            cartridge_id=cartridge_id,
            header_template=header_template,
            records=tuple(records),
        )

    def decompress(self, document: TOONDocument) -> Dict[str, Any]:
        """
        Descomprime documento TOON a diccionario.
        
        Args:
            document: Documento TOON
        
        Returns:
            Diccionario reconstruido
        """
        return document.to_dict()

    def compute_ratio(
        self,
        original: PayloadType,
        compressed: str,
    ) -> float:
        """
        Calcula ratio de compresión.
        
        Ratio < 1: compresión efectiva
        Ratio > 1: expansión
        Ratio = 1: sin cambio
        """
        original_str = json.dumps(
            original, sort_keys=True, ensure_ascii=False, default=str
        )
        original_size = max(len(original_str), 1)
        compressed_size = max(len(compressed), 1)
        
        return compressed_size / original_size


# =============================================================================
# TRAZA DE AUDITORÍA THREAD-SAFE
# =============================================================================

class AuditTrail:
    """
    Traza de auditoría thread-safe con límite de tamaño.
    
    Implementa un buffer circular para evitar crecimiento ilimitado
    mientras mantiene las entradas más recientes.
    """

    def __init__(self, max_size: int = MAX_AUDIT_TRAIL_SIZE) -> None:
        self._buffer: Deque[CategoricalEqualizerSeed] = deque(maxlen=max_size)
        self._lock = threading.RLock()
        self._total_count = 0

    def append(self, seed: CategoricalEqualizerSeed) -> None:
        """Agrega una entrada a la traza."""
        with self._lock:
            self._buffer.append(seed)
            self._total_count += 1

    def get_all(self) -> List[CategoricalEqualizerSeed]:
        """Retorna copia de todas las entradas."""
        with self._lock:
            return list(self._buffer)

    def get_recent(self, n: int) -> List[CategoricalEqualizerSeed]:
        """Retorna las n entradas más recientes."""
        with self._lock:
            return list(self._buffer)[-n:]

    def get_by_status(
        self,
        status: ImpedanceMatchStatus,
    ) -> List[CategoricalEqualizerSeed]:
        """Filtra entradas por status."""
        with self._lock:
            return [
                s for s in self._buffer
                if s.impedance_match_status == status
            ]

    def get_by_stratum(
        self,
        stratum: Stratum,
    ) -> List[CategoricalEqualizerSeed]:
        """Filtra entradas por estrato."""
        with self._lock:
            return [
                s for s in self._buffer
                if s.target_stratum == stratum
            ]

    def clear(self) -> None:
        """Limpia la traza."""
        with self._lock:
            self._buffer.clear()

    @property
    def size(self) -> int:
        """Número de entradas actuales."""
        with self._lock:
            return len(self._buffer)

    @property
    def total_count(self) -> int:
        """Total histórico de entradas (incluyendo descartadas)."""
        with self._lock:
            return self._total_count

    def get_statistics(self) -> Dict[str, Any]:
        """Calcula estadísticas de la traza."""
        with self._lock:
            if not self._buffer:
                return {
                    "total_entries": 0,
                    "current_size": 0,
                    "status_distribution": {},
                    "stratum_distribution": {},
                }
            
            status_counts: Dict[str, int] = {}
            stratum_counts: Dict[str, int] = {}
            
            for seed in self._buffer:
                status_name = seed.impedance_match_status.value
                stratum_name = seed.target_stratum.name
                
                status_counts[status_name] = status_counts.get(status_name, 0) + 1
                stratum_counts[stratum_name] = stratum_counts.get(stratum_name, 0) + 1
            
            return {
                "total_entries": self._total_count,
                "current_size": len(self._buffer),
                "status_distribution": status_counts,
                "stratum_distribution": stratum_counts,
            }


# =============================================================================
# MIC AGENT
# =============================================================================

class MICAgent:
    """
    Adaptador de impedancia categórico entre LLM y MIC.
    
    Implementa un funtor F: L → M donde:
    - L: Categoría de outputs del LLM (Mappings JSON)
    - M: Categoría MIC (CategoricalState estratificados)
    
    El adaptador garantiza:
    1. Sensado correcto del estrato objetivo
    2. Selección de contrato y cartucho apropiados
    3. Validación de clausura transitiva DIKW
    4. Compresión TOON determinista
    5. Encapsulación monádica del resultado
    6. Proyección auditada hacia la MIC
    
    Thread-Safety:
    - La traza de auditoría es thread-safe
    - Los silos son inmutables después de inicialización
    - Cada invocación es independiente
    """

    def __init__(
        self,
        mic_registry: MICRegistry,
        silo_manager: Optional[SiloManager] = None,
        schema_validator: Optional[SchemaValidator] = None,
        algebraic_veto_registry: Optional[AlgebraicVetoRegistry] = None,
        toon_compressor: Optional[TOONCompressor] = None,
        audit_trail_size: int = MAX_AUDIT_TRAIL_SIZE,
    ) -> None:
        """
        Inicializa el MIC Agent.
        
        Args:
            mic_registry: Registro MIC para proyección
            silo_manager: Gestor de silos (opcional, usa default)
            schema_validator: Validador de schema (opcional, usa default)
            algebraic_veto_registry: Registro de vetos (opcional, usa default)
            toon_compressor: Compresor TOON (opcional, usa default)
            audit_trail_size: Tamaño máximo de traza de auditoría
        """
        self._mic = mic_registry
        self._silo_manager = silo_manager or SiloManager()
        self._schema_validator = schema_validator or SchemaValidator()
        self._algebraic_vetos = algebraic_veto_registry or AlgebraicVetoRegistry()
        self._toon_compressor = toon_compressor or TOONCompressor()
        self._audit_trail = AuditTrail(max_size=audit_trail_size)

    # -------------------------------------------------------------------------
    # PROPIEDADES
    # -------------------------------------------------------------------------

    @property
    def audit_trail(self) -> AuditTrail:
        """Acceso a la traza de auditoría."""
        return self._audit_trail

    @property
    def silo_manager(self) -> SiloManager:
        """Acceso al gestor de silos."""
        return self._silo_manager

    # -------------------------------------------------------------------------
    # INTROSPECCIÓN DE ESTRATO
    # -------------------------------------------------------------------------

    def sense_stratum(self, target_vector: str) -> Stratum:
        """
        Detecta el estrato asociado a un vector MIC.
        
        Args:
            target_vector: Nombre del vector en el espacio MIC
        
        Returns:
            Estrato DIKW del vector
        
        Raises:
            StratumResolutionError: Si el vector no existe o no tiene estrato
        """
        info = self._mic.get_vector_info(target_vector)
        
        if info is None:
            raise StratumResolutionError(
                f"Vector '{target_vector}' no existe en el espacio MIC"
            )
        
        if "stratum" not in info:
            raise StratumResolutionError(
                f"Vector '{target_vector}' no reporta estrato asociado"
            )
        
        return normalize_stratum(info["stratum"])

    # -------------------------------------------------------------------------
    # VALIDACIÓN DE CLAUSURA
    # -------------------------------------------------------------------------

    def validate_closure(
        self,
        target_stratum: Stratum,
        validated_strata: FrozenSet[Stratum],
    ) -> Optional[str]:
        """
        Valida clausura transitiva DIKW.
        
        Para que un estrato σ sea alcanzable, todos los estratos en
        σ.requires() deben estar validados.
        
        Args:
            target_stratum: Estrato objetivo
            validated_strata: Estratos ya validados
        
        Returns:
            Mensaje de error si hay violación, None si OK
        """
        required = target_stratum.requires()
        missing = required - validated_strata
        
        if missing:
            return (
                f"Clausura transitiva incumplida para {target_stratum.name}: "
                f"faltan estratos {sorted(s.name for s in missing)}"
            )
        
        return None

    # -------------------------------------------------------------------------
    # COMPRESIÓN TOON
    # -------------------------------------------------------------------------

    def compress_telemetry(
        self,
        target_vector: str,
        telemetry: PayloadType,
    ) -> Tuple[str, TOONDocument]:
        """
        Comprime telemetría usando el cartucho apropiado.
        
        Args:
            target_vector: Vector objetivo
            telemetry: Datos de telemetría
        
        Returns:
            Tupla (cartridge_id, documento_TOON)
        
        Raises:
            TOONCompressionError: Si la compresión falla
        """
        stratum = self.sense_stratum(target_vector)
        cartridge_id, header_template = self._silo_manager.fetch_cartridge(
            stratum, target_vector
        )
        
        try:
            document = self._toon_compressor.compress(
                telemetry, cartridge_id, header_template
            )
            return cartridge_id, document
        except Exception as e:
            raise TOONCompressionError(
                f"Error comprimiendo telemetría: {e}"
            ) from e

    def inject_functorial_context(
        self,
        target_vector: str,
        raw_telemetry: PayloadType,
    ) -> str:
        """
        Comprime telemetría a representación TOON string.
        
        Método de conveniencia para obtener el string directamente.
        
        Args:
            target_vector: Vector objetivo
            raw_telemetry: Telemetría cruda
        
        Returns:
            String TOON comprimido
        """
        _, document = self.compress_telemetry(target_vector, raw_telemetry)
        
        compressed = document.render()
        
        stratum = self.sense_stratum(target_vector)
        logger.info(
            "Contexto comprimido: vector=%s estrato=%s ratio=%.2f",
            target_vector,
            stratum.name,
            self._toon_compressor.compute_ratio(raw_telemetry, compressed),
        )
        
        return compressed

    # -------------------------------------------------------------------------
    # ENCAPSULACIÓN MONÁDICA
    # -------------------------------------------------------------------------

    def encapsulate_monad(
        self,
        target_vector: str,
        llm_output: Any,
        validated_strata: FrozenSet[Stratum],
        context_hashes: Optional[FrozenSet[str]] = None,
        raw_telemetry: Optional[PayloadType] = None,
        force_override: bool = False,
    ) -> CategoricalState:
        """
        Encapsula output del LLM en un CategoricalState.
        
        Esta es la operación central del funtor F: L → M.
        
        Pipeline de encapsulación:
        1. Validar tipo de entrada
        2. Sensar estrato objetivo
        3. Seleccionar contrato y cartucho
        4. Validar clausura (a menos que force_override)
        5. Validar contra schema JSON
        6. Aplicar vetos algebraicos
        7. Comprimir telemetría (si se proporciona)
        8. Generar semilla de auditoría
        9. Construir CategoricalState
        
        Args:
            target_vector: Vector MIC objetivo
            llm_output: Salida del LLM (debe ser Mapping)
            validated_strata: Estratos ya validados
            context_hashes: Hashes de contexto (opcional)
            raw_telemetry: Telemetría para compresión (opcional)
            force_override: Si True, ignora validación de clausura
        
        Returns:
            CategoricalState encapsulado (puede estar en error)
        """
        # 1. Validar tipo de entrada
        if not isinstance(llm_output, Mapping):
            return self._create_error_state(
                target_vector="unknown",
                status=ImpedanceMatchStatus.INPUT_TYPE_ERROR,
                error_msg=f"LLM output debe ser Mapping, recibido: {type(llm_output).__name__}",
                validated_strata=validated_strata,
            )
        
        # 2. Sensar estrato
        try:
            stratum = self.sense_stratum(target_vector)
        except StratumResolutionError as e:
            return self._create_error_state(
                target_vector=target_vector,
                status=ImpedanceMatchStatus.STRATUM_MISMATCH_REJECTED,
                error_msg=str(e),
                validated_strata=validated_strata,
            )
        
        # 3. Seleccionar contrato y cartucho
        try:
            contract_id, schema = self._silo_manager.fetch_contract(stratum, target_vector)
            cartridge_id, _ = self._silo_manager.fetch_cartridge(stratum, target_vector)
        except SiloAccessError as e:
            return self._create_error_state(
                target_vector=target_vector,
                status=ImpedanceMatchStatus.SCHEMA_VALIDATION_ERROR,
                error_msg=str(e),
                validated_strata=validated_strata,
                stratum=stratum,
            )
        
        # Inicializar estado y errores
        status = ImpedanceMatchStatus.LAMINAR_PROJECTION
        error_msg: Optional[str] = None
        validation_errors: List[str] = []
        
        # 4. Validar clausura
        if not force_override:
            closure_error = self.validate_closure(stratum, validated_strata)
            if closure_error:
                status = ImpedanceMatchStatus.STRATUM_MISMATCH_REJECTED
                error_msg = closure_error
                validation_errors.append(closure_error)
        
        # 5. Validar contra schema
        if status == ImpedanceMatchStatus.LAMINAR_PROJECTION:
            schema_result = self._schema_validator.validate(schema, llm_output)
            if not schema_result.is_valid:
                status = ImpedanceMatchStatus.SCHEMA_VALIDATION_ERROR
                error_msg = schema_result.error
                validation_errors.extend(schema_result.errors)
        
        # 6. Aplicar vetos algebraicos
        if status in [ImpedanceMatchStatus.LAMINAR_PROJECTION, ImpedanceMatchStatus.SCHEMA_VALIDATION_ERROR]:
            veto_errors = self._algebraic_vetos.validate(stratum, llm_output)
            if veto_errors:
                status = ImpedanceMatchStatus.ALGEBRAIC_VETO
                error_msg = veto_errors[0]
                validation_errors.extend(veto_errors)
        
        # 7. Comprimir telemetría
        compressed_context = ""
        compression_ratio = 0.0
        
        if raw_telemetry is not None:
            try:
                compressed_context = self.inject_functorial_context(
                    target_vector, raw_telemetry
                )
                compression_ratio = self._toon_compressor.compute_ratio(
                    raw_telemetry, compressed_context
                )
            except TOONCompressionError as e:
                if status == ImpedanceMatchStatus.LAMINAR_PROJECTION:
                    status = ImpedanceMatchStatus.TOON_COMPRESSION_ERROR
                    error_msg = str(e)
                validation_errors.append(str(e))
        
        # 8. Generar semilla de auditoría
        audit_seed = CategoricalEqualizerSeed(
            target_vector=target_vector,
            target_stratum=stratum,
            silo_a_contract_id=contract_id,
            silo_b_cartridge_id=cartridge_id,
            impedance_match_status=status,
            token_compression_ratio=compression_ratio,
            raw_telemetry_hash=_stable_hash(raw_telemetry or {}),
            llm_output_hash=_stable_hash(dict(llm_output)),
            validation_errors=tuple(validation_errors),
        )
        self._audit_trail.append(audit_seed)
        
        # 9. Construir contexto
        context = {
            "target_vector": target_vector,
            "target_stratum": stratum.name,
            "contract_id": contract_id,
            "cartridge_id": cartridge_id,
            "context_hashes": sorted(context_hashes or frozenset()),
            "compression_ratio": compression_ratio,
            "audit_seed_hash": audit_seed.compute_hash(),
            "protocol_version": ENCAPSULATION_PROTOCOL_VERSION,
        }
        
        if compressed_context:
            context["compressed_context"] = compressed_context
        
        # 10. Retornar estado
        if status != ImpedanceMatchStatus.LAMINAR_PROJECTION:
            logger.warning(
                "Encapsulación vetada: vector=%s estrato=%s status=%s error=%s",
                target_vector,
                stratum.name,
                status.value,
                error_msg,
            )
            return CategoricalState(
                payload={},
                context=context,
                validated_strata=validated_strata,
                error=status.value,
                error_details={
                    "reason": error_msg,
                    "contract_failed": contract_id,
                    "validation_errors": validation_errors,
                },
            )
        
        # Éxito: agregar estrato al conjunto validado
        new_validated = validated_strata | frozenset([stratum])
        
        return CategoricalState(
            payload=dict(llm_output),
            context=context,
            validated_strata=new_validated,
            error=None,
            error_details=None,
        )

    def _create_error_state(
        self,
        target_vector: str,
        status: ImpedanceMatchStatus,
        error_msg: str,
        validated_strata: FrozenSet[Stratum],
        stratum: Optional[Stratum] = None,
    ) -> CategoricalState:
        """Helper para crear estado de error."""
        context = {
            "target_vector": target_vector,
            "impedance_status": status.value,
            "protocol_version": ENCAPSULATION_PROTOCOL_VERSION,
        }
        if stratum:
            context["target_stratum"] = stratum.name
        
        return CategoricalState(
            payload={},
            context=context,
            validated_strata=validated_strata,
            error=status.value,
            error_details={"reason": error_msg},
        )

    # -------------------------------------------------------------------------
    # PROYECCIÓN HACIA MIC
    # -------------------------------------------------------------------------

    def execute_projection(
        self,
        target_vector: str,
        llm_output: Any,
        validated_strata: FrozenSet[Stratum],
        context_hashes: Optional[FrozenSet[str]] = None,
        raw_telemetry: Optional[PayloadType] = None,
        force_override: bool = False,
    ) -> Dict[str, Any]:
        """
        Orquesta el ciclo completo de adaptación y proyección.
        
        Este método:
        1. Encapsula el output del LLM
        2. Si hay error, retorna VETO
        3. Si OK, proyecta hacia la MIC
        4. Retorna resultado estructurado
        
        Args:
            target_vector: Vector MIC objetivo
            llm_output: Salida del LLM
            validated_strata: Estratos ya validados
            context_hashes: Hashes de contexto
            raw_telemetry: Telemetría para compresión
            force_override: Si True, ignora clausura
        
        Returns:
            Diccionario con resultado de proyección
        """
        # Encapsular
        categorical_state = self.encapsulate_monad(
            target_vector=target_vector,
            llm_output=llm_output,
            validated_strata=validated_strata,
            context_hashes=context_hashes,
            raw_telemetry=raw_telemetry,
            force_override=force_override,
        )
        
        # Verificar errores de encapsulación
        if categorical_state.is_failed:
            logger.warning(
                "Proyección abortada: %s",
                categorical_state.error,
            )
            return {
                "status": "VETO",
                "impedance_status": categorical_state.error,
                "reason": (
                    categorical_state.error_details.get("reason")
                    if categorical_state.error_details else None
                ),
                "details": categorical_state.error_details,
                "context": categorical_state.context,
            }
        
        # Proyectar hacia MIC
        try:
            stratum = self.sense_stratum(target_vector)
            
            logger.info(
                "Proyectando: vector=%s estrato=%s",
                target_vector,
                stratum.name,
            )
            
            mic_result = self._mic.project_intent(
                target_basis_vector=target_vector,
                stratum_target=stratum.value,
                validated_subspaces=[
                    s.name for s in categorical_state.validated_strata
                ],
                orthogonality_guarantee=0.0,
                payload=categorical_state.payload,
            )
            
            return {
                "status": "OK",
                "impedance_status": ImpedanceMatchStatus.LAMINAR_PROJECTION.value,
                "target_vector": target_vector,
                "target_stratum": stratum.name,
                "categorical_state_hash": categorical_state.compute_hash(),
                "validated_strata": sorted(
                    s.name for s in categorical_state.validated_strata
                ),
                "mic_result": mic_result,
                "audit_context": categorical_state.context,
            }
        
        except Exception as e:
            logger.exception("Error en proyección MIC")
            
            # Registrar error en auditoría
            error_seed = CategoricalEqualizerSeed(
                target_vector=target_vector,
                target_stratum=self.sense_stratum(target_vector) if target_vector else Stratum.PHYSICS,
                silo_a_contract_id="unknown",
                silo_b_cartridge_id="unknown",
                impedance_match_status=ImpedanceMatchStatus.MIC_RESOLUTION_ERROR,
                validation_errors=(str(e),),
            )
            self._audit_trail.append(error_seed)
            
            return {
                "status": "ERROR",
                "impedance_status": ImpedanceMatchStatus.MIC_RESOLUTION_ERROR.value,
                "reason": str(e),
                "target_vector": target_vector,
                "exception_type": type(e).__name__,
            }

    # -------------------------------------------------------------------------
    # MÉTODOS DE CONVENIENCIA
    # -------------------------------------------------------------------------

    def get_audit_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de la traza de auditoría."""
        return self._audit_trail.get_statistics()

    def get_recent_audits(self, n: int = 10) -> List[Dict[str, Any]]:
        """Retorna las n auditorías más recientes como dicts."""
        return [seed.to_dict() for seed in self._audit_trail.get_recent(n)]

    def clear_audit_trail(self) -> None:
        """Limpia la traza de auditoría."""
        self._audit_trail.clear()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Excepciones
    "MICAgentError",
    "StratumResolutionError",
    "ContractValidationError",
    "ClosureViolationError",
    "AlgebraicVetoError",
    "TOONCompressionError",
    "SiloAccessError",
    "ProjectionError",
    
    # Enums
    "ImpedanceMatchStatus",
    "ValidationSeverity",
    
    # Dataclasses
    "SchemaValidationResult",
    "CategoricalEqualizerSeed",
    "TOONDocument",
    "SiloAContract",
    "SiloBCartridge",
    
    # Clases principales
    "SchemaValidator",
    "AlgebraicVetoRegistry",
    "SiloManager",
    "TOONCompressor",
    "AuditTrail",
    "MICAgent",
    
    # Utilidades
    "normalize_stratum",
    "python_type_matches",
    
    # Constantes
    "MAX_AUDIT_TRAIL_SIZE",
    "TOON_START_MARKER",
    "TOON_END_MARKER",
    "ENCAPSULATION_PROTOCOL_VERSION",
]