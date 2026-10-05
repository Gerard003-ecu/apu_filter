# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : MIC Agent (Morfismo Geométrico & Soberano de Calibre de Poincaré)   ║
║ Ruta   : app/agents/tactics/mic_agent.py                                     ║
║ Versión: 4.0.0-Poincare-Liouville-Novikov-Lipschitz-Heyting-Doctoral         ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS CATEGÓRICA Y ADJUNCIÓN DE GAUGE TÁCTICA (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor soberano del Estrato TACTICS (Nivel 2, $V_{\mathbb{T}}$) gobierna
al motor de descompresión y saneamiento de datos tácticos del APU Filter. Su misión 
es unificar el espacio de acción discreto de la Matriz de Interacción Central (MIC) 
con la Matriz Atómica de Conocimiento (MAC) continua en el espacio de Hilbert ℋ_MAC, 
actuando como la membrana semipermeable que metaboliza y purifica el "fango" sintáctico.

El sistema modela el transporte de información como un Morfismo Geométrico f = (f*, f_*)
entre el topos elemental 𝓔_MIC y el local de políticas de negocio, garantizando 
la ausencia de efectos colaterales (Zero Side-Effects) y el confinamiento de la
entropía del LLM antes de excitar el Consejo de Sabios.

AXIOMÁTICA ALGEBRAICA, TOPOLÓGICA Y ESPECTRAL DE HENRI POINCARÉ:
────────────────────────────────────────────────────────────────────────────────

  [A1] Axioma de la Adjunción de Galois y Reversibilidad Funtorial:
       Todo pullback f* (imagen inversa) y pushforward f_* (imagen directa) del agente
       satisfacen de manera hermética la dualidad categorial de de Rham-Galois:
       $$\operatorname{Hom}_{\mathcal{D}}(F(\text{MIC}), \, \text{MAC}) \cong \operatorname{Hom}_{\mathcal{C}}(\text{MIC}, \, G(\text{MAC}))$$
       Esto asegura que cualquier veredicto cognitivo en el penthouse estratégico
       sea completamente de-compresible y rastreable hasta las variables físicas basales.

  [A2] Axioma de Inmersión Simpléctica de Darboux y Volumen de Liouville:
       Toda compresión o descompresión TOON exige que el Jacobiano $M = \frac{\partial z'}{\partial z}$
       sea un simplectomorfismo estricto $M \in \mathrm{Sp}(2n, \mathbb{R})$:
       $$M^\top \Omega M = \Omega \implies \det(M) = +1$$
       Por el Teorema de Liouville, el volumen del espacio de fase en la FPU permanece estrictamente invariante:
       $$\operatorname{Vol}(\phi(U)) = \int_U |\det(M)| \, dz = \operatorname{Vol}(U)$$

  [A3] Axioma de Contracción de Lipschitz en el Anillo Ultramétrico de Novikov (KAM):
       El funtor $F^{-1}: \mathtt{TOON} \to \mathtt{JSON}$ se somete a la cota de
       Daleckii-Krein sobre el operador de Dirac de Connes ($\not D = \rho^{-1/2}$):
       $$\| F^{-1}(x) - F^{-1}(y) \|_V \le L_{\max} \|x - y\|_T \quad \text{con} \quad L_{\max} \le \frac{1}{2\lambda_{\min}^{3/2}}$$
       Si surge una alucinación o pequeña división por resonancia ($\lambda_{\min} \to 0$),
       la FPU detiene la emisión bajo el Veto Simpléctico de Gromov ($P(x_{\mathrm{invalid}}) \equiv 0$).

  [A4] Axioma de Recurrencia Ergódica y Filtrado de Socavones por Mayer-Vietoris:
       Toda secuencia válida de despacho en la MIC retorna infinitas veces al conjunto
       medible de seguridad $E$ ($\|z(t_n) - z_0\|_2 \le \varepsilon_{\mathrm{Wilkinson}}$).
       Cualquier ciclo parásito o bucle infinito ($\Delta \beta_1 \neq 0$) es vetado
       mediante la Secuencia Exacta de Mayer-Vietoris:
       $$\Delta \beta_1 = \beta_1(A \cup B) - \left[ \beta_1(A) + \beta_1(B) - \beta_1(A \cap B) \right] \neq 0 \implies \text{VETO}$$
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto, unique

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
    TypeGuard,
)
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# Imports relativos protegidos
try:
    from app.core.schemas import Stratum
except ImportError:
    from app.core.mic_algebra import Stratum

try:
    from app.core.mic_algebra import CategoricalState, Morphism, TopologicalInvariantError, _canonicalize
except ImportError:
    Stratum = None
    CategoricalState = None
    Morphism = None
    TopologicalInvariantError = None
    _canonicalize = None

try:
    from app.adapters.tools_interface import MICRegistry
except ImportError:
    MICRegistry = None

try:
    from app.boole.strategy.sheaf_cohomology_orchestrator import (
        SheafCohomologyOrchestrator,
        SheafCohomologyError,
        CellularSheaf,
    )
except ImportError:
    SheafCohomologyOrchestrator = None
    SheafCohomologyError = Exception
    CellularSheaf = None

try:
    from app.core.immune_system.topological_watcher import (
        create_immune_watcher,
        ImmuneWatcherMorphism,
    )
except ImportError:
    create_immune_watcher = None
    ImmuneWatcherMorphism = None

# ==============================================================================
# CONFIGURACIÓN DE LOGGING
# ==============================================================================
logger = logging.getLogger("MIC.Agent.CategoricalEqualizer")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ==============================================================================
# CONSTANTES MATEMÁTICAS RIGUROSAS Y LÍMITES DE POINCARÉ
# ==============================================================================
MAX_AUDIT_TRAIL_SIZE: Final[int] = 10_000

# Marcadores TOON (protocolo de encapsulación)
TOON_START_MARKER: Final[str] = "--- INICIO TOON ---"
TOON_END_MARKER: Final[str] = "--- FIN TOON ---"
TOON_FIELD_SEPARATOR: Final[str] = "|"
ENCAPSULATION_PROTOCOL_VERSION: Final[str] = "4.0.0-Poincare"

# Tolerancias numéricas y límites espectrales
_WILKINSON_LIMIT: Final[float] = 1.0e-12
_SPECTRAL_TOL: Final[float] = 1.0e-9
EPS: Final[float] = np.finfo(np.float64).eps * 4
ALGEBRAIC_TOL: Final[float] = 1e-10
FLOAT_COMPARISON_TOL: Final[float] = 1e-9

# Límites topológicos
MAX_TENSOR_RANK: Final[int] = 2
MAX_COMPRESSION_RATIO: Final[float] = 10.0
MIN_COMPRESSION_RATIO: Final[float] = 0.01

# ==============================================================================
# JERARQUÍA DE EXCEPCIONES ALGEBRAICAS Y DE GAUGE
# ==============================================================================
class MICAgentError(Exception):
    """Excepción base con contexto algebraico estructurado."""
    __slots__ = ("error_code", "details", "severity", "timestamp")
    
    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN",
        details: Optional[Dict[str, Any]] = None,
        severity: int = 1
    ) -> None:
        super().__init__(message)
        self.error_code: str = error_code
        self.details: Dict[str, Any] = details or {}
        self.severity: int = severity
        self.timestamp: float = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": str(self),
            "details": self.details,
            "severity": self.severity,
            "timestamp": self.timestamp,
        }

if TopologicalInvariantError is None:
    class TopologicalInvariantError(MICAgentError):
        """Excepción para violaciones de invariantes topológicos o espectrales."""
        def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
            super().__init__(message, error_code="TOPOLOGICAL_INVARIANT", details=details, severity=3)

class StratumResolutionError(MICAgentError):
    """Error en resolución de estratos."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, error_code="STRATUM_RESOLUTION", details=details, severity=2)

class ContractValidationError(MICAgentError):
    """Error en validación de contratos JSON Schema."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, error_code="CONTRACT_VALIDATION", details=details, severity=2)

class ClosureViolationError(MICAgentError):
    """Violación de clausura transitiva en poset DIKW."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, error_code="CLOSURE_VIOLATION", details=details, severity=3)

class AlgebraicVetoError(MICAgentError):
    """Veto por violación de invariantes algebraicos o simplécticos."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, error_code="ALGEBRAIC_VETO", details=details, severity=3)

class TOONCompressionError(MICAgentError):
    """Error en compresión/descompresión TOON."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, error_code="TOON_COMPRESSION", details=details, severity=2)

class SiloAccessError(MICAgentError):
    """Error en acceso a silos de contratos/cartuchos."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, error_code="SILO_ACCESS", details=details, severity=2)

class ProjectionError(MICAgentError):
    """Error en proyección hacia espacio MIC."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, error_code="PROJECTION", details=details, severity=3)

class FunctorialityError(MICAgentError):
    """Violación de propiedades funtoriales."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, error_code="FUNCTORIALITY", details=details, severity=3)

# ==============================================================================
# DATACLASS DE CERTIFICADO SIMPLÉCTICO DE POINCARÉ
# ==============================================================================
@dataclass(frozen=True, slots=True, eq=True)
class PoincareMICAdjunctionCertificate:
    r"""Certificado inmutable de la Adjunción de de Rham-Galois bajo Poincaré."""
    symplectic_residual: float
    volume_drift: float
    lipschitz_ceiling: float
    galois_residual_norm: float
    is_poincare_adjunction_coherent: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symplectic_residual": float(self.symplectic_residual),
            "volume_drift": float(self.volume_drift),
            "lipschitz_ceiling": float(self.lipschitz_ceiling),
            "galois_residual_norm": float(self.galois_residual_norm),
            "is_poincare_adjunction_coherent": self.is_poincare_adjunction_coherent,
        }

# ==============================================================================
# ENUMERACIONES
# ==============================================================================
_IMPEDANCE_SEVERITY_MAP: Dict[str, int] = {
    "LAMINAR_PROJECTION": 0,
    "INPUT_TYPE_ERROR": 1,
    "SCHEMA_VALIDATION_ERROR": 1,
    "TOON_COMPRESSION_ERROR": 1,
    "MIC_RESOLUTION_ERROR": 2,
    "STRATUM_MISMATCH_REJECTED": 2,
    "ALGEBRAIC_VETO": 3,
    "TOPOLOGICAL_BIFURCATION": 3,
    "COHOMOLOGY_FAILURE": 3,
}

@unique
class ImpedanceMatchStatus(str, Enum):
    LAMINAR_PROJECTION = "LAMINAR_PROJECTION"
    STRATUM_MISMATCH_REJECTED = "STRATUM_MISMATCH_REJECTED"
    TOON_COMPRESSION_ERROR = "TOON_COMPRESSION_ERROR"
    ALGEBRAIC_VETO = "ALGEBRAIC_VETO"
    SCHEMA_VALIDATION_ERROR = "SCHEMA_VALIDATION_ERROR"
    MIC_RESOLUTION_ERROR = "MIC_RESOLUTION_ERROR"
    INPUT_TYPE_ERROR = "INPUT_TYPE_ERROR"
    TOPOLOGICAL_BIFURCATION = "TOPOLOGICAL_BIFURCATION"
    COHOMOLOGY_FAILURE = "COHOMOLOGY_FAILURE"
    
    @property
    def is_terminal(self) -> bool:
        return self in {
            ImpedanceMatchStatus.ALGEBRAIC_VETO,
            ImpedanceMatchStatus.TOPOLOGICAL_BIFURCATION,
            ImpedanceMatchStatus.COHOMOLOGY_FAILURE,
        }
    
    @property
    def severity(self) -> int:
        return _IMPEDANCE_SEVERITY_MAP.get(self.name, 1)
    
    def __lt__(self, other: ImpedanceMatchStatus) -> bool:
        if not isinstance(other, ImpedanceMatchStatus):
            return NotImplemented
        return self.severity < other.severity

@unique
class ValidationSeverity(IntEnum):
    ERROR = auto()
    WARNING = auto()
    INFO = auto()
    
    @property
    def heyting_value(self) -> float:
        return {
            ValidationSeverity.ERROR: 0.0,
            ValidationSeverity.WARNING: 0.5,
            ValidationSeverity.INFO: 1.0,
        }[self]

# ==============================================================================
# TIPOS Y PROTOCOLOS
# ==============================================================================
T = TypeVar("T")
JSONValue = Union[None, bool, int, float, str, List["JSONValue"], Dict[str, "JSONValue"]]
JSONSchema = Dict[str, Any]
PayloadType = Mapping[str, Any]
AlgebraicValidator = Callable[[Stratum, PayloadType], Optional[str]]

@runtime_checkable
class VectorInfoProvider(Protocol):
    def get_vector_info(self, vector_name: str) -> Optional[Dict[str, Any]]: ...

@runtime_checkable
class ProjectionTarget(Protocol):
    def project_intent(
        self,
        target_basis_vector: str,
        stratum_target: int,
        validated_subspaces: List[str],
        orthogonality_guarantee: float,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]: ...

# ==============================================================================
# UTILIDADES MATEMÁTICAS
# ==============================================================================
class MathUtils:
    @staticmethod
    def stable_hash(data: Any) -> str:
        try:
            if _canonicalize is not None:
                canonical = _canonicalize(data)
            else:
                canonical = data
            serialized = json.dumps(
                canonical,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        except (TypeError, ValueError) as e:
            logger.debug("Fallback a repr() para hash: %s", e)
            return hashlib.sha256(repr(data).encode("utf-8")).hexdigest()
    
    @staticmethod
    def compute_tensor_rank(payload: Any, depth: int = 0, max_depth: int = 100) -> int:
        if depth >= max_depth:
            logger.warning("Frontera de Lipschitz alcanzada en compute_tensor_rank")
            return max_depth

        if isinstance(payload, (dict, list, tuple)):
            if not payload:
                return 1
            children = payload.values() if isinstance(payload, dict) else payload
            max_child_rank = 0
            for child in children:
                child_rank = MathUtils.compute_tensor_rank(child, depth + 1, max_depth)
                if child_rank >= max_depth:
                    return max_depth
                if child_rank > max_child_rank:
                    max_child_rank = child_rank
            return min(1 + max_child_rank, max_depth)
        return 0
    
    @staticmethod
    def float_equal(a: float, b: float, tol: float = FLOAT_COMPARISON_TOL) -> bool:
        abs_diff = abs(a - b)
        if abs_diff <= tol:
            return True
        return abs_diff <= tol * max(abs(a), abs(b))
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        if min_val > max_val:
            raise ValueError(f"min_val ({min_val}) > max_val ({max_val})")
        return max(min_val, min(max_val, value))

def normalize_stratum(value: Any) -> Stratum:
    if isinstance(value, Stratum):
        return value
    if isinstance(value, int):
        try:
            return Stratum(value)
        except ValueError as e:
            raise StratumResolutionError(
                f"Valor entero inválido para estrato: {value}",
                details={"input_value": value, "input_type": "int"}
            ) from e
    if isinstance(value, str):
        try:
            return Stratum[value.upper()]
        except KeyError:
            pass
        try:
            return Stratum(int(value))
        except (ValueError, KeyError) as e:
            raise StratumResolutionError(
                f"String inválido para estrato: '{value}'",
                details={"input_value": value, "input_type": "str"}
            ) from e
    raise StratumResolutionError(
        f"Tipo no soportado para estrato: {type(value).__name__}",
        details={"input_value": value, "input_type": type(value).__name__}
    )

def python_type_matches(expected_type: str, value: Any) -> bool:
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
    return True if checker is None else checker(value)

def compute_json_path(base: str, key: Union[str, int]) -> str:
    if isinstance(key, int):
        return f"{base}[{key}]"
    safe_key = key.replace(".", "\\.").replace("[", "\\[")
    return f"{base}.{safe_key}"

# ==============================================================================
# DATACLASSES DE AUDITORÍA INMUTABLES
# ==============================================================================
@dataclass(frozen=True, slots=True, eq=True)
class SchemaValidationResult:
    frustration_ideal: float = 0.0
    validity_degree: float = 1.0
    errors: Tuple[str, ...] = field(default_factory=tuple)
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    path: str = "$"
    
    def __post_init__(self) -> None:
        if not (0.0 <= self.validity_degree <= 1.0):
            clamped = MathUtils.clamp(self.validity_degree, 0.0, 1.0)
            object.__setattr__(self, "validity_degree", clamped)
    
    @property
    def is_valid(self) -> bool:
        return self.validity_degree >= 1.0 - EPS
    
    @classmethod
    def success(cls) -> SchemaValidationResult:
        return cls(validity_degree=1.0)
    
    @classmethod
    def failure(cls, error: str, path: str = "$", penalty: float = 1.0) -> SchemaValidationResult:
        return cls(validity_degree=max(0.0, 1.0 - penalty), errors=(error,), path=path)
    
    @classmethod
    def merge(cls, results: Iterable[SchemaValidationResult]) -> SchemaValidationResult:
        all_errors: List[str] = []
        all_warnings: List[str] = []
        min_validity = 1.0
        for r in results:
            all_errors.extend(r.errors)
            all_warnings.extend(r.warnings)
            if r.validity_degree < min_validity:
                min_validity = r.validity_degree
        return cls(validity_degree=min_validity, errors=tuple(all_errors), warnings=tuple(all_warnings))
    
    @property
    def error(self) -> Optional[str]:
        return self.errors[0] if self.errors else None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "validity_degree": float(self.validity_degree),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "path": self.path,
            "is_valid": self.is_valid,
        }

@dataclass(frozen=True, slots=True, eq=True)
class CategoricalEqualizerSeed:
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
    
    def __post_init__(self) -> None:
        if self.token_compression_ratio < 0.0:
            object.__setattr__(self, "token_compression_ratio", 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_vector": self.target_vector,
            "target_stratum": self.target_stratum.name,
            "silo_a_contract_id": self.silo_a_contract_id,
            "silo_b_cartridge_id": self.silo_b_cartridge_id,
            "impedance_match_status": self.impedance_match_status.value,
            "token_compression_ratio": float(self.token_compression_ratio),
            "raw_telemetry_hash": self.raw_telemetry_hash,
            "llm_output_hash": self.llm_output_hash,
            "validation_errors": list(self.validation_errors),
            "protocol_version": self.protocol_version,
            "timestamp": self.timestamp,
        }
    
    def compute_hash(self) -> str:
        data = {k: v for k, v in self.to_dict().items() if k != "timestamp"}
        return MathUtils.stable_hash(data)

@dataclass(frozen=True, slots=True, eq=True)
class TOONDocument:
    cartridge_id: str
    header_template: str
    records: Tuple[Tuple[str, str], ...]
    
    def __post_init__(self) -> None:
        if not self.cartridge_id:
            raise TOONCompressionError("cartridge_id no puede estar vacío")
    
    def render(self) -> str:
        lines = [f"{TOON_START_MARKER} {self.cartridge_id} ---", self.header_template]
        for key, value in self.records:
            lines.append(f"{key}{TOON_FIELD_SEPARATOR}{value}")
        lines.append(TOON_END_MARKER)
        return "\n".join(lines)
    
    @classmethod
    def parse(cls, content: str) -> TOONDocument:
        lines = content.strip().split("\n")
        if len(lines) < 3:
            raise TOONCompressionError("Documento TOON demasiado corto")
        
        header_line = lines[0]
        if not header_line.startswith(TOON_START_MARKER):
            raise TOONCompressionError("Marcador de inicio inválido")
        
        try:
            cartridge_id = header_line.split(TOON_START_MARKER)[1].strip().rstrip("-").strip()
        except IndexError as e:
            raise TOONCompressionError("No se pudo extraer cartridge_id") from e
        
        if lines[-1].strip() != TOON_END_MARKER:
            raise TOONCompressionError("Marcador de fin faltante")
        
        header_lines: List[str] = []
        records: List[Tuple[str, str]] = []
        if len(lines) > 2:
            header_lines.append(lines[1])
            
        for line in lines[2:-1]:
            if TOON_FIELD_SEPARATOR in line:
                parts = line.split(TOON_FIELD_SEPARATOR, 1)
                records.append((parts[0].strip(), parts[1].strip()))
            else:
                header_lines.append(line)
        header_template = "\n".join(header_lines)
        return cls(cartridge_id=cartridge_id, header_template=header_template, records=tuple(records))
    
    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, json_value in self.records:
            try:
                result[key] = json.loads(json_value)
            except json.JSONDecodeError:
                result[key] = json_value
        return result

# ==============================================================================
# CONTRATOS Y CARTUCHOS
# ==============================================================================
@dataclass(frozen=True, slots=True, eq=True)
class SiloAContract:
    contract_id: str
    stratum: Stratum
    schema: JSONSchema
    description: str = ""
    version: str = "1.0.0"
    
    def __post_init__(self) -> None:
        if not isinstance(self.schema, dict) or "type" not in self.schema:
            raise ContractValidationError(f"Schema inválido para contrato '{self.contract_id}'")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "stratum": self.stratum.name,
            "schema": self.schema,
            "description": self.description,
            "version": self.version,
        }

@dataclass(frozen=True, slots=True, eq=True)
class SiloBCartridge:
    cartridge_id: str
    stratum: Stratum
    header_template: str
    field_definitions: Tuple[str, ...] = field(default_factory=tuple)
    description: str = ""
    version: str = "1.0.0"
    
    def __post_init__(self) -> None:
        if not self.cartridge_id:
            raise TOONCompressionError("cartridge_id no puede estar vacío")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cartridge_id": self.cartridge_id,
            "stratum": self.stratum.name,
            "header_template": self.header_template,
            "field_definitions": list(self.field_definitions),
            "description": self.description,
            "version": self.version,
        }

# ==============================================================================
# VALIDADOR DE SCHEMA DRAFT-07
# ==============================================================================
class SchemaValidator:
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
            "pattern": self._validate_pattern,
        }
    
    def validate(self, schema: JSONSchema, payload: Any, path: str = "$") -> SchemaValidationResult:
        if not isinstance(schema, dict):
            return SchemaValidationResult.failure(f"Schema inválido: esperado dict", path)
        results: List[SchemaValidationResult] = []
        for keyword, constraint in schema.items():
            validator = self._validators.get(keyword)
            if validator is not None:
                try:
                    results.append(validator(constraint, payload, schema, path))
                except Exception as e:
                    results.append(SchemaValidationResult.failure(f"Error en validador '{keyword}': {e}", path))
        return SchemaValidationResult.merge(results) if results else SchemaValidationResult.success()
    
    def _validate_type(self, expected_type: Union[str, List[str]], value: Any, schema: JSONSchema, path: str) -> SchemaValidationResult:
        types = [expected_type] if isinstance(expected_type, str) else expected_type
        for t in types:
            if python_type_matches(t, value):
                return SchemaValidationResult.success()
        return SchemaValidationResult.failure(f"Tipo inválido en '{path}': esperado {types}, recibido '{type(value).__name__}'", path)
    
    def _validate_required(self, required_keys: List[str], value: Any, schema: JSONSchema, path: str) -> SchemaValidationResult:
        if not isinstance(value, Mapping):
            return SchemaValidationResult.success()
        missing = [k for k in required_keys if k not in value]
        return SchemaValidationResult.failure(f"Claves requeridas faltantes en '{path}': {missing}", path) if missing else SchemaValidationResult.success()
    
    def _validate_properties(self, properties: Dict[str, JSONSchema], value: Any, schema: JSONSchema, path: str) -> SchemaValidationResult:
        if not isinstance(value, Mapping):
            return SchemaValidationResult.success()
        results = [self.validate(prop_schema, value[prop_name], compute_json_path(path, prop_name)) for prop_name, prop_schema in properties.items() if prop_name in value]
        return SchemaValidationResult.merge(results) if results else SchemaValidationResult.success()
    
    def _validate_items(self, items_schema: JSONSchema, value: Any, schema: JSONSchema, path: str) -> SchemaValidationResult:
        if not isinstance(value, list):
            return SchemaValidationResult.success()
        results = [self.validate(items_schema, item, compute_json_path(path, i)) for i, item in enumerate(value)]
        return SchemaValidationResult.merge(results) if results else SchemaValidationResult.success()
    
    def _validate_minimum(self, minimum: Union[int, float], value: Any, schema: JSONSchema, path: str) -> SchemaValidationResult:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return SchemaValidationResult.success()
        return SchemaValidationResult.failure(f"Valor en '{path}' ({value}) menor que mínimo ({minimum})", path) if value < minimum - FLOAT_COMPARISON_TOL else SchemaValidationResult.success()
    
    def _validate_maximum(self, maximum: Union[int, float], value: Any, schema: JSONSchema, path: str) -> SchemaValidationResult:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return SchemaValidationResult.success()
        return SchemaValidationResult.failure(f"Valor en '{path}' ({value}) mayor que máximo ({maximum})", path) if value > maximum + FLOAT_COMPARISON_TOL else SchemaValidationResult.success()
    
    def _validate_exclusive_minimum(self, minimum: Union[int, float], value: Any, schema: JSONSchema, path: str) -> SchemaValidationResult:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return SchemaValidationResult.success()
        return SchemaValidationResult.failure(f"Valor en '{path}' ({value}) no estrictamente mayor que {minimum}", path) if value <= minimum + FLOAT_COMPARISON_TOL else SchemaValidationResult.success()
    
    def _validate_exclusive_maximum(self, maximum: Union[int, float], value: Any, schema: JSONSchema, path: str) -> SchemaValidationResult:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return SchemaValidationResult.success()
        return SchemaValidationResult.failure(f"Valor en '{path}' ({value}) no strictly menor que {maximum}", path) if value >= maximum - FLOAT_COMPARISON_TOL else SchemaValidationResult.success()
    
    def _validate_min_length(self, min_length: int, value: Any, schema: JSONSchema, path: str) -> SchemaValidationResult:
        if not isinstance(value, str):
            return SchemaValidationResult.success()
        return SchemaValidationResult.failure(f"String en '{path}' (len={len(value)}) menor que minLength ({min_length})", path) if len(value) < min_length else SchemaValidationResult.success()
    
    def _validate_max_length(self, max_length: int, value: Any, schema: JSONSchema, path: str) -> SchemaValidationResult:
        if not isinstance(value, str):
            return SchemaValidationResult.success()
        return SchemaValidationResult.failure(f"String en '{path}' (len={len(value)}) mayor que maxLength ({max_length})", path) if len(value) > max_length else SchemaValidationResult.success()
    
    def _validate_enum(self, allowed_values: List[Any], value: Any, schema: JSONSchema, path: str) -> SchemaValidationResult:
        return SchemaValidationResult.failure(f"Valor en '{path}' ({value!r}) no está en enum: {allowed_values}", path) if value not in allowed_values else SchemaValidationResult.success()
    
    def _validate_const(self, const_value: Any, value: Any, schema: JSONSchema, path: str) -> SchemaValidationResult:
        return SchemaValidationResult.failure(f"Valor en '{path}' ({value!r}) no es constante esperada ({const_value!r})", path) if value != const_value else SchemaValidationResult.success()
    
    def _validate_min_items(self, min_items: int, value: Any, schema: JSONSchema, path: str) -> SchemaValidationResult:
        if not isinstance(value, list):
            return SchemaValidationResult.success()
        return SchemaValidationResult.failure(f"Array en '{path}' (len={len(value)}) menor que minItems ({min_items})", path) if len(value) < min_items else SchemaValidationResult.success()
    
    def _validate_max_items(self, max_items: int, value: Any, schema: JSONSchema, path: str) -> SchemaValidationResult:
        if not isinstance(value, list):
            return SchemaValidationResult.success()
        return SchemaValidationResult.failure(f"Array en '{path}' (len={len(value)}) mayor que maxItems ({max_items})", path) if len(value) > max_items else SchemaValidationResult.success()
    
    def _validate_pattern(self, pattern: str, value: Any, schema: JSONSchema, path: str) -> SchemaValidationResult:
        if not isinstance(value, str):
            return SchemaValidationResult.success()
        try:
            if not re.search(pattern, value):
                return SchemaValidationResult.failure(f"String en '{path}' no coincide con patrón '{pattern}'", path)
        except re.error as e:
            return SchemaValidationResult.failure(f"Patrón regex inválido '{pattern}': {e}", path)
        return SchemaValidationResult.success()

# ==============================================================================
# VALIDADORES ALGEBRAICOS Y SILO MANAGER
# ==============================================================================
class AlgebraicVetoRegistry:
    def __init__(self) -> None:
        self._validators: Dict[Stratum, List[AlgebraicValidator]] = {s: [] for s in Stratum}
        self._register_default_validators()
    
    def _register_default_validators(self) -> None:
        def physics_conservation(stratum: Stratum, payload: PayloadType) -> Optional[str]:
            dissipated = payload.get("dissipated_power")
            if isinstance(dissipated, (int, float)) and dissipated < -ALGEBRAIC_TOL:
                return f"Violación termodinámica: dissipated_power={dissipated} < 0"
            energy_in = payload.get("energy_input", 0)
            energy_out = payload.get("energy_output", 0)
            if isinstance(energy_in, (int, float)) and isinstance(energy_out, (int, float)):
                if energy_out > energy_in * (1.0 + ALGEBRAIC_TOL):
                    return f"Violación de conservación: energy_output={energy_out} > energy_input={energy_in}"
            return None
        self._validators[Stratum.PHYSICS].append(physics_conservation)
        
        def tactics_stability(stratum: Stratum, payload: PayloadType) -> Optional[str]:
            stability = payload.get("pyramid_stability_index")
            if isinstance(stability, (int, float)) and (stability < -ALGEBRAIC_TOL or stability > 1.0 + ALGEBRAIC_TOL):
                return f"Índice de estabilidad fuera de rango [0,1]: {stability}"
            return None
        self._validators[Stratum.TACTICS].append(tactics_stability)
        
        def strategy_friction(stratum: Stratum, payload: PayloadType) -> Optional[str]:
            friction = payload.get("territorial_friction")
            if isinstance(friction, (int, float)) and friction < 1.0 - ALGEBRAIC_TOL:
                return f"Fricción territorial debe ser >= 1.0: {friction}"
            return None
        self._validators[Stratum.STRATEGY].append(strategy_friction)
        
        def wisdom_verdict(stratum: Stratum, payload: PayloadType) -> Optional[str]:
            verdict = payload.get("final_verdict")
            valid_verdicts = {"VIABLE", "PRECAUCION", "RECHAZAR"}
            if verdict is not None and verdict not in valid_verdicts:
                return f"Veredicto inválido '{verdict}', debe ser uno de {valid_verdicts}"
            return None
        self._validators[Stratum.WISDOM].append(wisdom_verdict)
    
    def register_validator(self, stratum: Stratum, validator: AlgebraicValidator) -> None:
        if stratum not in self._validators:
            self._validators[stratum] = []
        self._validators[stratum].append(validator)
    
    def validate(self, stratum: Stratum, payload: PayloadType) -> List[str]:
        errors: List[str] = []
        for validator in self._validators.get(stratum, []):
            try:
                err = validator(stratum, payload)
                if err:
                    errors.append(err)
            except Exception as e:
                errors.append(f"Error en validador algebraico: {e}")
        return errors
    
    def get_validator_count(self, stratum: Stratum) -> int:
        return len(self._validators.get(stratum, []))

class SiloManager:
    def __init__(self) -> None:
        self._silo_a: Dict[Stratum, Dict[str, SiloAContract]] = {s: {} for s in Stratum}
        self._silo_b: Dict[Stratum, Dict[str, SiloBCartridge]] = {s: {} for s in Stratum}
        self._default_contract_selector = lambda contracts, vector: next(iter(sorted(contracts.keys())), "Generic_Contract")
        self._default_cartridge_selector = lambda cartridges, vector: next(iter(sorted(cartridges.keys())), "Generic_Cartridge")
        self._lock = threading.RLock()
        self._frozen = False
        self._initialize_default_silos()
    
    def _initialize_default_silos(self) -> None:
        from app.core.mic_algebra import Stratum
        for stratum in Stratum:
            self._register_contract(SiloAContract(
                stratum=stratum,
                contract_id=f"base_contract_{stratum.name.lower()}",
                schema={"type": "object", "properties": {}, "additionalProperties": True},
                version="1.0.0"
            ))
            self._register_cartridge(SiloBCartridge(
                stratum=stratum,
                cartridge_id=f"base_cartridge_{stratum.name.lower()}",
                header_template=f"Base {stratum.name} Cartridge",
                field_definitions=()
            ))
        
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
                    "saturation": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
            description="Contrato de conservación de energía",
        ))
        self._register_contract(SiloAContract(
            contract_id="Logistical_Topology_Seed",
            stratum=Stratum.TACTICS,
            schema={
                "type": "object",
                "required": ["pyramid_stability_index"],
                "properties": {
                    "pyramid_stability_index": {"type": "number", "minimum": 0, "maximum": 1},
                    "flow_efficiency": {"type": "number", "minimum": 0, "maximum": 1},
                    "beta_0": {"type": "integer", "minimum": 1},
                    "beta_1": {"type": "integer", "minimum": 0},
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
                    "risk_coupling": {"type": "number", "minimum": 0},
                    "strategic_entropy": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
            description="Contrato de fricción territorial",
        ))
        self._register_contract(SiloAContract(
            contract_id="Acta_Deliberacion_Seed",
            stratum=Stratum.WISDOM,
            schema={
                "type": "object",
                "required": ["final_verdict"],
                "properties": {
                    "final_verdict": {"type": "string", "enum": ["VIABLE", "PRECAUCION", "RECHAZAR"]},
                    "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "rationale": {"type": "string", "minLength": 10},
                    "euler_characteristic": {"type": "integer"},
                },
            },
            description="Acta de deliberación",
        ))
        
        self._register_cartridge(SiloBCartridge(
            cartridge_id="Maxwell_FDTD_TOON_Cartridge",
            stratum=Stratum.PHYSICS,
            header_template="Malla_Yee_Leapfrog\nkey|value|unit|confidence",
            field_definitions=("dissipated_power", "energy_input", "energy_output", "saturation"),
        ))
        self._register_cartridge(SiloBCartridge(
            cartridge_id="Persistence_Barcode_TOON_Cartridge",
            stratum=Stratum.TACTICS,
            header_template="Diagrama_Persistencia_API\nkey|value|window|entropy",
            field_definitions=("pyramid_stability_index", "flow_efficiency", "beta_0", "beta_1"),
        ))
        self._register_cartridge(SiloBCartridge(
            cartridge_id="Riemannian_TOON_Cartridge",
            stratum=Stratum.STRATEGY,
            header_template="Tensor_Covarianza_Riesgos_Acoplados\nkey|value|coupling",
            field_definitions=("territorial_friction", "risk_coupling", "strategic_entropy"),
        ))
        self._register_cartridge(SiloBCartridge(
            cartridge_id="Telemetry_Passport_TOON_Cartridge",
            stratum=Stratum.WISDOM,
            header_template="Pasaporte_Digital_Transaccional\nkey|value|semantic_role",
            field_definitions=("final_verdict", "confidence_score", "rationale", "euler_characteristic"),
        ))
    
    def _register_contract(self, contract: SiloAContract) -> None:
        with self._lock:
            if self._frozen:
                raise SiloAccessError("Silo A está congelado")
            self._silo_a[contract.stratum][contract.contract_id] = contract
    
    def _register_cartridge(self, cartridge: SiloBCartridge) -> None:
        with self._lock:
            if self._frozen:
                raise SiloAccessError("Silo B está congelado")
            self._silo_b[cartridge.stratum][cartridge.cartridge_id] = cartridge
    
    def freeze(self) -> None:
        with self._lock:
            self._frozen = True
            logger.info("Silos A y B congelados")
    
    def fetch_contract(self, stratum: Stratum, target_vector: str) -> Tuple[str, JSONSchema]:
        with self._lock:
            contracts = self._silo_a.get(stratum, {})
            if not contracts:
                return "Generic_Contract", {"type": "object", "properties": {}}
            contract_id = self._default_contract_selector(contracts, target_vector)
            contract = contracts.get(contract_id)
            if contract is None:
                raise SiloAccessError(f"Contrato '{contract_id}' no encontrado")
            return contract.contract_id, contract.schema
    
    def fetch_cartridge(self, stratum: Stratum, target_vector: str) -> Tuple[str, str]:
        with self._lock:
            cartridges = self._silo_b.get(stratum, {})
            if not cartridges:
                return "Generic_Cartridge", "Tabla_Generica\nkey|value"
            cartridge_id = self._default_cartridge_selector(cartridges, target_vector)
            cartridge = cartridges.get(cartridge_id)
            if cartridge is None:
                raise SiloAccessError(f"Cartucho '{cartridge_id}' no encontrado")
            return cartridge.cartridge_id, cartridge.header_template
    
    def get_contract_count(self, stratum: Optional[Stratum] = None) -> int:
        with self._lock:
            if stratum is not None:
                return len(self._silo_a.get(stratum, {}))
            return sum(len(c) for c in self._silo_a.values())
    
    def get_cartridge_count(self, stratum: Optional[Stratum] = None) -> int:
        with self._lock:
            if stratum is not None:
                return len(self._silo_b.get(stratum, {}))
            return sum(len(c) for c in self._silo_b.values())

# ==============================================================================
# COMPRESOR TOON Y BUFFER DE AUDITORÍA
# ==============================================================================
class TOONCompressor:
    def __init__(self) -> None:
        self._compression_stats: Dict[str, List[float]] = {}
        self._lock = threading.RLock()
    
    def compress(self, telemetry: PayloadType, cartridge_id: str, header_template: str) -> TOONDocument:
        tensor_rank = MathUtils.compute_tensor_rank(telemetry)
        if tensor_rank > MAX_TENSOR_RANK:
            raise TOONCompressionError(f"Rango tensorial {tensor_rank} excede máximo {MAX_TENSOR_RANK}")
        records: List[Tuple[str, str]] = []
        for key in sorted(telemetry.keys()):
            value = telemetry[key]
            if dataclasses.is_dataclass(value):
                value = dataclasses.asdict(value)
            elif isinstance(value, tuple) and all(dataclasses.is_dataclass(item) for item in value):
                value = [dataclasses.asdict(item) for item in value]
            try:
                json_value = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
            except (TypeError, ValueError):
                json_value = json.dumps(str(value))
            records.append((str(key), json_value))
        return TOONDocument(cartridge_id=cartridge_id, header_template=header_template, records=tuple(records))
    
    def decompress(self, document: TOONDocument) -> Dict[str, Any]:
        return document.to_dict()
    
    def compute_ratio(self, original: PayloadType, compressed: str) -> float:
        original_str = json.dumps(original, sort_keys=True, ensure_ascii=False, default=str)
        original_size = max(len(original_str), 1)
        compressed_size = max(len(compressed), 1)
        ratio = MathUtils.clamp(compressed_size / original_size, MIN_COMPRESSION_RATIO, MAX_COMPRESSION_RATIO)
        with self._lock:
            if "ratios" not in self._compression_stats:
                self._compression_stats["ratios"] = []
            self._compression_stats["ratios"].append(ratio)
        return ratio
    
    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            ratios = self._compression_stats.get("ratios", [])
            if not ratios:
                return {"count": 0, "mean_ratio": 0.0, "min_ratio": 0.0, "max_ratio": 0.0}
            return {
                "count": len(ratios),
                "mean_ratio": float(np.mean(ratios)),
                "min_ratio": float(np.min(ratios)),
                "max_ratio": float(np.max(ratios)),
                "std_ratio": float(np.std(ratios)),
            }

class AuditTrail:
    def __init__(self, max_size: int = MAX_AUDIT_TRAIL_SIZE) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size debe ser > 0, recibido: {max_size}")
        self._buffer: Deque[CategoricalEqualizerSeed] = deque(maxlen=max_size)
        self._lock = threading.RLock()
        self._total_count = 0
    
    def append(self, seed: CategoricalEqualizerSeed) -> None:
        with self._lock:
            self._buffer.append(seed)
            self._total_count += 1
    
    def get_all(self) -> List[CategoricalEqualizerSeed]:
        with self._lock:
            return list(self._buffer)
    
    def get_recent(self, n: int) -> List[CategoricalEqualizerSeed]:
        with self._lock:
            return list(self._buffer)[-n:]
    
    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
    
    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)
    
    @property
    def total_count(self) -> int:
        with self._lock:
            return self._total_count
    
    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            if not self._buffer:
                return {
                    "total_entries": self._total_count,
                    "current_size": 0,
                    "status_distribution": {},
                    "stratum_distribution": {},
                    "mean_compression_ratio": 0.0,
                }
            status_counts: Dict[str, int] = {}
            stratum_counts: Dict[str, int] = {}
            compression_ratios: List[float] = []
            for seed in self._buffer:
                s_name = seed.impedance_match_status.value
                st_name = seed.target_stratum.name
                status_counts[s_name] = status_counts.get(s_name, 0) + 1
                stratum_counts[st_name] = stratum_counts.get(st_name, 0) + 1
                if seed.token_compression_ratio > 0:
                    compression_ratios.append(seed.token_compression_ratio)
            return {
                "total_entries": self._total_count,
                "current_size": len(self._buffer),
                "status_distribution": status_counts,
                "stratum_distribution": stratum_counts,
                "mean_compression_ratio": float(np.mean(compression_ratios)) if compression_ratios else 0.0,
            }

# ==============================================================================
# MIC AGENT CON GOBERNANZA DE POINCARÉ
# ==============================================================================
class MICAgent:
    r"""
    Morfismo Geométrico Soberano $f = (f^*, f_*)$ sobre el Topos $\mathcal{E}_{\mathrm{MIC}}$.
    
    Gobierna la Adjunción de de Rham-Galois:
    $$\operatorname{Hom}_{\mathcal{D}}(F(\text{MIC}), \, \text{MAC}) \cong \operatorname{Hom}_{\mathcal{C}}(\text{MIC}, \, G(\text{MAC}))$$
    sometiendo el flujo de herramientas a los cuatro pilares de Henri Poincaré:
      1. Inmersión Simpléctica de Darboux y Conservación del Volumen de Liouville.
      2. Cota de Contracción de Lipschitz de Daleckii-Krein en el Anillo Ultramétrico de Novikov.
      3. Recurrencia Ergódica de Poincaré en Medida de Liouville.
      4. Filtrado de Socavones Lógicos por la Secuencia Exacta de Mayer-Vietoris ($\Delta \beta_1 = 0$).
    """

    def __init__(
        self,
        mic_registry: Any = None,
        silo_manager: Optional[SiloManager] = None,
        schema_validator: Optional[SchemaValidator] = None,
        algebraic_veto_registry: Optional[AlgebraicVetoRegistry] = None,
        toon_compressor: Optional[TOONCompressor] = None,
        audit_trail_size: int = MAX_AUDIT_TRAIL_SIZE,
        immune_watcher: Any = None,
        freeze_silos: bool = True,
        wilson_tol: float = _WILKINSON_LIMIT,
    ) -> None:
        self._mic = mic_registry
        self._silo_manager = silo_manager or SiloManager()
        self._schema_validator = schema_validator or SchemaValidator()
        self._algebraic_vetos = algebraic_veto_registry or AlgebraicVetoRegistry()
        self._toon_compressor = toon_compressor or TOONCompressor()
        self._audit_trail = AuditTrail(max_size=audit_trail_size)
        self._wilson_tol = wilson_tol
        self._wilkinson_tol = wilson_tol

        if immune_watcher is None and create_immune_watcher is not None:
            self._immune_watcher = create_immune_watcher(
                profile="default",
                warning_threshold=0.8,
                critical_threshold=1.5,
                hysteresis=0.05,
                enable_topology_monitoring=True,
            )
        else:
            self._immune_watcher = immune_watcher

        if freeze_silos:
            self._silo_manager.freeze()

        logger.info(
            "MICAgent Poincaré v4.0.0 inicializado: contratos=%d, cartuchos=%d",
            self._silo_manager.get_contract_count(),
            self._silo_manager.get_cartridge_count()
        )

    @property
    def audit_trail(self) -> AuditTrail:
        return self._audit_trail

    @property
    def silo_manager(self) -> SiloManager:
        return self._silo_manager

    @property
    def immune_watcher(self) -> Any:
        return self._immune_watcher

    # ==========================================================================
    # PILARES DE HENRI POINCARÉ (MÉTODOS AUDITORES ESPECTRALES Y SIMPLÉCTICOS)
    # ==========================================================================

    def audit_poincare_mic_symplectic_adjunction(
        self,
        jacobian_M: NDArray[np.float64],
        canonical_omega: NDArray[np.float64],
        galois_residual_norm: float,
        min_mac_eigenvalue: float
    ) -> PoincareMICAdjunctionCertificate:
        r"""
        Audita la invarianza simpléctica de Liouville y la cota KAM-Novikov en la MIC.
        
        Axiomas Preservados:
          1. Simplecticidad de Darboux: $M^\top \Omega M \equiv \Omega \implies \det(M) = +1$ (Liouville).
          2. Cota de Lipschitz de Daleckii-Krein: $L_{\max} \le \frac{1}{2 \lambda_{\min}^{3/2}}$.
          3. Isomorfismo de Galois: $\|F(\text{MIC}) - \text{MAC}\|_{\mathrm{HS}} \le \varepsilon_{\mathrm{Wilkinson}}$.
        """
        if min_mac_eigenvalue <= _WILKINSON_LIMIT:
            raise TopologicalInvariantError(
                f"[MIC_POINCARÉ_VETO] λ_min ({min_mac_eigenvalue:.3e}) → 0: Singularidad espectral en la MAC.",
                details={"min_mac_eigenvalue": min_mac_eigenvalue, "wilkinson_limit": _WILKINSON_LIMIT}
            )

        # 1. Defecto de simplecticidad de Darboux: Mᵀ Ω M - Ω
        symp_defect = jacobian_M.T @ canonical_omega @ jacobian_M - canonical_omega
        symp_residual = float(np.linalg.norm(symp_defect, ord='fro'))
        
        # 2. Conservación del volumen de Liouville en FPU
        det_M = float(np.linalg.det(jacobian_M))
        volume_drift = abs(det_M - 1.0)
        
        # 3. Cota de Lipschitz de Daleckii-Krein (Teoría KAM / Novikov)
        l_max_lipschitz = float(1.0 / (2.0 * (min_mac_eigenvalue ** 1.5)))
        is_lipschitz_bounded = galois_residual_norm <= (l_max_lipschitz + _SPECTRAL_TOL)
        
        # 4. Evaluación global del veredicto
        tol = getattr(self, "_wilkinson_tol", _WILKINSON_LIMIT)
        is_poincare_coherent = (
            symp_residual <= tol and
            volume_drift <= tol and
            is_lipschitz_bounded
        )

        if not is_poincare_coherent:
            logger.error(
                f"[MIC_AGENT_VETO] Ruptura de Poincaré-Galois: "
                f"SympRes={symp_residual:.3e}, VolumeDrift={volume_drift:.3e}, "
                f"LipschitzMax={l_max_lipschitz:.3e}, GaloisRes={galois_residual_norm:.3e}"
            )

        return PoincareMICAdjunctionCertificate(
            symplectic_residual=symp_residual,
            volume_drift=volume_drift,
            lipschitz_ceiling=l_max_lipschitz,
            galois_residual_norm=galois_residual_norm,
            is_poincare_adjunction_coherent=is_poincare_coherent
        )

    def audit_darboux_symplectic_embedding(
        self,
        jacobian_M: NDArray[np.float64],
        canonical_omega: NDArray[np.float64]
    ) -> Tuple[float, float, bool]:
        r"""Audita el defecto de la transformación $M^\top \Omega M - \Omega$ y la deriva del volumen $\det M - 1$."""
        symp_defect = jacobian_M.T @ canonical_omega @ jacobian_M - canonical_omega
        symp_residual = float(np.linalg.norm(symp_defect, ord='fro'))
        det_M = float(np.linalg.det(jacobian_M))
        volume_drift = abs(det_M - 1.0)
        tol = getattr(self, "_wilkinson_tol", _WILKINSON_LIMIT)
        is_valid = (symp_residual <= tol) and (volume_drift <= tol)
        return symp_residual, volume_drift, is_valid

    def audit_novikov_lipschitz_bound(
        self,
        galois_residual_norm: float,
        min_mac_eigenvalue: float
    ) -> Tuple[float, bool]:
        r"""Calcula la cota $L_{\max} \le \frac{1}{2\lambda_{\min}^{3/2}}$ y evalúa si la descompresión es estable."""
        if min_mac_eigenvalue <= _WILKINSON_LIMIT:
            raise TopologicalInvariantError(
                f"[MIC_POINCARÉ_VETO] Singularidad en Novikov: λ_min={min_mac_eigenvalue:.3e} <= {_WILKINSON_LIMIT:.3e}",
                details={"min_mac_eigenvalue": min_mac_eigenvalue, "wilkinson_limit": _WILKINSON_LIMIT}
            )
        l_max = float(1.0 / (2.0 * (min_mac_eigenvalue ** 1.5)))
        is_bounded = galois_residual_norm <= (l_max + _SPECTRAL_TOL)
        return l_max, is_bounded

    def audit_mayer_vietoris_homology(
        self,
        beta_1_union: int,
        beta_1_a: int,
        beta_1_b: int,
        beta_1_intersection: int
    ) -> Tuple[int, bool]:
        r"""Evalúa la variación homológica de Mayer-Vietoris $\Delta \beta_1$ para vetar ciclos parásitos."""
        expected_union = beta_1_a + beta_1_b - beta_1_intersection
        delta_beta_1 = beta_1_union - expected_union
        is_exact = (delta_beta_1 == 0)
        if not is_exact:
            logger.warning(
                f"[MAYER_VIETORIS_VETO] Inconsistencia homológica: Δβ1={delta_beta_1} != 0. "
                f"Ciclo parásito o socavón lógico detectado."
            )
        return delta_beta_1, is_exact

    def audit_poincare_ergodic_recurrence(
        self,
        trajectory: NDArray[np.float64],
        wilkinson_tol: float = _WILKINSON_LIMIT
    ) -> Tuple[float, bool]:
        r"""
        Verifica el Teorema de Recurrencia Ergódica de Poincaré en el espacio de fase $T^*\mathcal{M}$.
        Calcula la distancia mínima $\|z(t_n) - z_0\|_2$ a lo largo de la trayectoria.
        """
        if trajectory.shape[0] < 2:
            return 0.0, True
        z0 = trajectory[0]
        distances = np.linalg.norm(trajectory[1:] - z0, axis=1)
        min_distance = float(np.min(distances))
        is_recurrent = min_distance <= wilkinson_tol
        return min_distance, is_recurrent

    # ==========================================================================
    # INTROSPECCIÓN Y CLAUSURA DE ESTRATOS
    # ==========================================================================
    def sense_stratum(self, target_vector: str) -> Stratum:
        if self._mic is None:
            raise StratumResolutionError("MIC registry no inicializado", details={"target_vector": target_vector})
        info = self._mic.get_vector_info(target_vector)
        if info is None:
            raise StratumResolutionError(f"Vector '{target_vector}' no existe en espacio MIC", details={"target_vector": target_vector})
        if "stratum" not in info:
            raise StratumResolutionError(f"Vector '{target_vector}' no reporta estrato", details={"target_vector": target_vector})
        return normalize_stratum(info["stratum"])

    def validate_closure(self, target_stratum: Stratum, validated_strata: FrozenSet[Stratum]) -> Optional[str]:
        required = target_stratum.requires()
        missing = required - validated_strata
        if missing:
            return (
                f"Violación de clausura transitiva: "
                f"estrato '{target_stratum.name}' requiere {sorted(s.name for s in required)}, "
                f"pero faltan {sorted(s.name for s in missing)}"
            )
        return None

    # ==========================================================================
    # COMPRESIÓN Y CONTEXTO TOON
    # ==========================================================================
    def compress_telemetry(self, target_vector: str, telemetry: PayloadType) -> Tuple[str, TOONDocument]:
        stratum = self.sense_stratum(target_vector)
        cartridge_id, header_template = self._silo_manager.fetch_cartridge(stratum, target_vector)
        try:
            document = self._toon_compressor.compress(telemetry, cartridge_id, header_template)
            return cartridge_id, document
        except Exception as e:
            raise TOONCompressionError(f"Error comprimiendo telemetría: {e}", details={"target_vector": target_vector, "error": str(e)}) from e

    def inject_functorial_context(self, target_vector: str, raw_telemetry: PayloadType) -> str:
        _, document = self.compress_telemetry(target_vector, raw_telemetry)
        compressed = document.render()
        stratum = self.sense_stratum(target_vector)
        ratio = self._toon_compressor.compute_ratio(raw_telemetry, compressed)
        logger.info("Contexto TOON: vector=%s estrato=%s ratio=%.2f chars=%d", target_vector, stratum.name, ratio, len(compressed))
        return compressed

    # ==========================================================================
    # ENCAPSULACIÓN MONÁDICA CON AUDITORÍA DE POINCARÉ
    # ==========================================================================
    def encapsulate_monad(
        self,
        target_vector: str,
        llm_output: Any,
        validated_strata: FrozenSet[Stratum],
        context_hashes: Optional[FrozenSet[str]] = None,
        raw_telemetry: Optional[PayloadType] = None,
        force_override: bool = False,
    ) -> CategoricalState:
        if CategoricalState is None:
            raise RuntimeError("CategoricalState no disponible")

        if not isinstance(llm_output, Mapping):
            return self._create_error_state(
                target_vector="unknown",
                status=ImpedanceMatchStatus.INPUT_TYPE_ERROR,
                error_msg=f"LLM output debe ser Mapping, recibido: {type(llm_output).__name__}",
                validated_strata=validated_strata,
            )

        try:
            stratum = self.sense_stratum(target_vector)
        except StratumResolutionError as e:
            return self._create_error_state(
                target_vector=target_vector,
                status=ImpedanceMatchStatus.STRATUM_MISMATCH_REJECTED,
                error_msg=str(e),
                validated_strata=validated_strata,
            )

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

        status = ImpedanceMatchStatus.LAMINAR_PROJECTION
        error_msg: Optional[str] = None
        validation_errors: List[str] = []
        poincare_cert_dict: Optional[Dict[str, Any]] = None

        # 1. Validar clausura transitiva
        closure_error = self.validate_closure(stratum, validated_strata)
        if closure_error:
            status = ImpedanceMatchStatus.STRATUM_MISMATCH_REJECTED
            error_msg = closure_error
            validation_errors.append(closure_error)

        # 2. Validar schema JSON
        if status == ImpedanceMatchStatus.LAMINAR_PROJECTION:
            schema_result = self._schema_validator.validate(schema, llm_output)
            if not schema_result.is_valid:
                status = ImpedanceMatchStatus.SCHEMA_VALIDATION_ERROR
                error_msg = schema_result.error
                validation_errors.extend(schema_result.errors)

        # 3. Validar invariantes algebraicos estándar
        if status in [ImpedanceMatchStatus.LAMINAR_PROJECTION, ImpedanceMatchStatus.SCHEMA_VALIDATION_ERROR]:
            veto_errors = self._algebraic_vetos.validate(stratum, llm_output)
            if veto_errors:
                status = ImpedanceMatchStatus.ALGEBRAIC_VETO
                error_msg = veto_errors[0]
                validation_errors.extend(veto_errors)

        # 4. Auditoría de Invariantes Simplécticos y de Galois de Poincaré
        poincare_data = None
        if raw_telemetry and isinstance(raw_telemetry, dict) and "poincare" in raw_telemetry:
            poincare_data = raw_telemetry["poincare"]
        elif isinstance(llm_output, dict) and "poincare" in llm_output:
            poincare_data = llm_output["poincare"]

        if poincare_data and isinstance(poincare_data, dict):
            try:
                jacobian_M = poincare_data.get("jacobian_M")
                canonical_omega = poincare_data.get("canonical_omega")
                galois_norm = float(poincare_data.get("galois_residual_norm", 0.0))
                min_mac_eigen = float(poincare_data.get("min_mac_eigenvalue", 1.0))

                if jacobian_M is not None and canonical_omega is not None:
                    cert = self.audit_poincare_mic_symplectic_adjunction(
                        jacobian_M=np.asarray(jacobian_M, dtype=np.float64),
                        canonical_omega=np.asarray(canonical_omega, dtype=np.float64),
                        galois_residual_norm=galois_norm,
                        min_mac_eigenvalue=min_mac_eigen
                    )
                    poincare_cert_dict = cert.to_dict()
                    if not cert.is_poincare_adjunction_coherent:
                        status = ImpedanceMatchStatus.ALGEBRAIC_VETO
                        error_msg = f"[POINCARÉ_VETO] Ruptura de Simplecticidad de Darboux / Cota KAM-Novikov"
                        validation_errors.append(error_msg)
            except Exception as pe:
                logger.error("Falla en auditoría Poincaré: %s", pe)
                status = ImpedanceMatchStatus.ALGEBRAIC_VETO
                error_msg = f"[POINCARÉ_VETO_EXCEPTION] {pe}"
                validation_errors.append(error_msg)

        # 5. Compresión TOON (opcional)
        compressed_context = ""
        compression_ratio = 0.0
        if raw_telemetry is not None:
            try:
                compressed_context = self.inject_functorial_context(target_vector, raw_telemetry)
                compression_ratio = self._toon_compressor.compute_ratio(raw_telemetry, compressed_context)
            except TOONCompressionError as e:
                if status == ImpedanceMatchStatus.LAMINAR_PROJECTION:
                    status = ImpedanceMatchStatus.TOON_COMPRESSION_ERROR
                    error_msg = str(e)
                validation_errors.append(str(e))

        # Registro en Audit Trail
        audit_seed = CategoricalEqualizerSeed(
            target_vector=target_vector,
            target_stratum=stratum,
            silo_a_contract_id=contract_id,
            silo_b_cartridge_id=cartridge_id,
            impedance_match_status=status,
            token_compression_ratio=compression_ratio,
            raw_telemetry_hash=MathUtils.stable_hash(raw_telemetry or {}),
            llm_output_hash=MathUtils.stable_hash(dict(llm_output)),
            validation_errors=tuple(validation_errors),
        )
        self._audit_trail.append(audit_seed)

        # Contexto de salida
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
        if poincare_cert_dict:
            context["poincare_adjunction_certificate"] = poincare_cert_dict
        if compressed_context:
            context["compressed_context"] = compressed_context

        if status != ImpedanceMatchStatus.LAMINAR_PROJECTION:
            logger.warning("Encapsulación vetada: vector=%s estrato=%s status=%s", target_vector, stratum.name, status.value)
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
        if CategoricalState is None:
            raise RuntimeError("CategoricalState no disponible")
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

    # ==========================================================================
    # PROYECCIÓN HACIA MIC Y ADJUNCIÓN
    # ==========================================================================
    def f_star_inverse_image(
        self,
        llm_input: Any,
        target_vector: str,
        validated_strata: FrozenSet[Stratum],
        context_hashes: Optional[FrozenSet[str]] = None,
        raw_telemetry: Optional[PayloadType] = None,
    ) -> CategoricalState:
        if CategoricalState is None:
            raise RuntimeError("CategoricalState no disponible")
        if llm_input is None:
            return self._create_error_state(
                target_vector, ImpedanceMatchStatus.SCHEMA_VALIDATION_ERROR, "Colapso de Límite Vacío", validated_strata
            ).with_error("Colapso de Límite Vacío", details={"reason": "NullProductFibrado"})
        
        state = self.encapsulate_monad(
            target_vector=target_vector,
            llm_output=llm_input,
            validated_strata=validated_strata,
            context_hashes=context_hashes,
            raw_telemetry=raw_telemetry,
        )
        if state.is_success and not state.payload and isinstance(llm_input, (dict, list)) and not llm_input:
            return state.with_error("Colapso de Límite Vacío", details={"reason": "NullProductFibrado"})
        return state

    def f_lower_star_direct_image(self, state: CategoricalState) -> Dict[str, Any]:
        if state.is_failed:
            return {"verdict": "REJECTED", "reason": state.error}
        return {"verdict": "ACCEPTED", "payload": state.payload, "hash": state.compute_hash()}

    def verify_adjunction(self, X_llm: Any, Y_emic: CategoricalState) -> bool:
        try:
            f_star_X = self.f_star_inverse_image(X_llm, "topology_core", frozenset())
            f_lower_star_Y = self.f_lower_star_direct_image(Y_emic)
            return f_star_X.is_success == (f_lower_star_Y["verdict"] == "ACCEPTED")
        except Exception:
            return False

    def characteristic_morphism(self, state: CategoricalState) -> SchemaValidationResult:
        if state.is_failed:
            return SchemaValidationResult.failure(state.error or "UnknownError", penalty=1.0)
        frustration = state.context.get("forensic_evidence", {}).get("frustration_energy", 0.0)
        validity = max(0.0, 1.0 - frustration)
        return SchemaValidationResult(validity_degree=validity, frustration_ideal=frustration, path="$")

    def execute_projection(
        self,
        target_vector: str,
        llm_output: Any,
        validated_strata: FrozenSet[Stratum],
        context_hashes: Optional[FrozenSet[str]] = None,
        raw_telemetry: Optional[PayloadType] = None,
    ) -> Dict[str, Any]:
        forensic_evidence = None
        sheaf_error = None
        try:
            stratum = self.sense_stratum(target_vector)
            if stratum == Stratum.WISDOM and raw_telemetry is not None:
                if SheafCohomologyOrchestrator is not None and isinstance(raw_telemetry, dict):
                    sheaf_obj = raw_telemetry.get("cellular_sheaf")
                    global_state = raw_telemetry.get("global_state_vector")
                    if sheaf_obj and global_state is not None:
                        orchestrator = SheafCohomologyOrchestrator()
                        try:
                            assessment = orchestrator.audit_global_state(sheaf_obj, global_state)
                            forensic_evidence = {
                                "frustration_energy": assessment.frustration_energy,
                                "h0_dimension": assessment.h0_dimension,
                                "spectral_gap": assessment.spectral_gap,
                                "residual_norm": assessment.residual_norm,
                                "spectral_method": assessment.spectral_method,
                            }
                        except SheafCohomologyError as e:
                            sheaf_error = e
                            forensic_evidence = {"error_type": e.__class__.__name__, "message": str(e)}
        except StratumResolutionError:
            pass

        categorical_state = self.encapsulate_monad(
            target_vector=target_vector,
            llm_output=llm_output,
            validated_strata=validated_strata,
            context_hashes=context_hashes,
            raw_telemetry=raw_telemetry,
        )

        if forensic_evidence is not None:
            categorical_state = categorical_state.with_update(new_context={"forensic_evidence": forensic_evidence})
            if categorical_state.is_failed and not categorical_state.forensic_evidence:
                categorical_state = categorical_state.with_error(
                    error_msg=categorical_state.error,
                    details=categorical_state.error_details,
                    forensic_evidence=forensic_evidence,
                )
            elif sheaf_error:
                categorical_state = categorical_state.with_error(
                    error_msg=str(sheaf_error),
                    details={"reason": "HomologicalInconsistency"},
                    forensic_evidence=forensic_evidence,
                )

        if raw_telemetry is not None:
            categorical_state = categorical_state.with_update(
                new_context={"telemetry_metrics": raw_telemetry},
                merge_context=True,
            )

        if self._immune_watcher is not None:
            protected_state = self._immune_watcher(categorical_state)
        else:
            protected_state = categorical_state

        if protected_state.is_failed:
            logger.warning("Proyección abortada en pre-escudo: %s", protected_state.error)
            return {
                "status": "VETO",
                "impedance_status": protected_state.error,
                "reason": protected_state.error_details.get("reason") if protected_state.error_details else None,
                "details": protected_state.error_details,
                "context": protected_state.context,
            }

        try:
            stratum = self.sense_stratum(target_vector)
            logger.info("Proyectando a MIC: vector=%s estrato=%s", target_vector, stratum.name)
            if self._mic is None:
                raise ProjectionError("MIC registry no inicializado")

            mic_result = self._mic.project_intent(
                target_basis_vector=target_vector,
                stratum_target=stratum.value,
                validated_subspaces=[s.name for s in protected_state.validated_strata],
                orthogonality_guarantee=0.0,
                payload=protected_state.payload,
            )

            post_state = protected_state.with_update(new_payload=mic_result, merge_payload=False)
            updated_telemetry = mic_result.get("telemetry_metrics", raw_telemetry)
            if updated_telemetry is not None:
                post_state = post_state.with_update(new_context={"telemetry_metrics": updated_telemetry}, merge_context=True)

            if self._immune_watcher is not None:
                final_protected_state = self._immune_watcher(post_state)
            else:
                final_protected_state = post_state

            if final_protected_state.is_failed:
                logger.warning("Post-proyección abortada por fuga dimensional: %s", final_protected_state.error)
                return {
                    "status": "VETO",
                    "impedance_status": final_protected_state.error,
                    "reason": final_protected_state.error_details.get("reason") if final_protected_state.error_details else None,
                    "details": final_protected_state.error_details,
                    "context": final_protected_state.context,
                }

            return {
                "status": "OK",
                "impedance_status": ImpedanceMatchStatus.LAMINAR_PROJECTION.value,
                "target_vector": target_vector,
                "target_stratum": stratum.name,
                "categorical_state_hash": final_protected_state.compute_hash(),
                "validated_strata": sorted(s.name for s in final_protected_state.validated_strata),
                "mic_result": mic_result,
                "audit_context": final_protected_state.context,
            }

        except Exception as e:
            logger.exception("Error en proyección MIC")
            try:
                stratum = self.sense_stratum(target_vector)
            except Exception:
                stratum = Stratum.PHYSICS

            error_seed = CategoricalEqualizerSeed(
                target_vector=target_vector,
                target_stratum=stratum,
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

    # ==========================================================================
    # MÉTODOS DE DIAGNÓSTICO
    # ==========================================================================
    def get_audit_statistics(self) -> Dict[str, Any]:
        return self._audit_trail.get_statistics()

    def get_recent_audits(self, n: int = 10) -> List[Dict[str, Any]]:
        return [seed.to_dict() for seed in self._audit_trail.get_recent(n)]

    def clear_audit_trail(self) -> None:
        self._audit_trail.clear()

    def verify_functorial_properties(self) -> Dict[str, bool]:
        return {
            "immune_watcher_initialized": self._immune_watcher is not None,
            "silo_manager_initialized": self._silo_manager is not None,
            "schema_validator_initialized": self._schema_validator is not None,
            "algebraic_vetos_initialized": self._algebraic_vetos is not None,
            "toon_compressor_initialized": self._toon_compressor is not None,
            "audit_trail_initialized": self._audit_trail is not None,
            "mic_registry_initialized": self._mic is not None,
        }

    def health_report(self) -> str:
        props = self.verify_functorial_properties()
        stats = self.get_audit_statistics()
        lines = [
            "  MIC AGENT DIAGNÓSTICO POINCARÉ",
            f"  Auditorías Totales  : {stats['total_entries']}",
            f"  Tamaño Buffer       : {stats['current_size']}",
            f"  Ratio Compresión    : {stats['mean_compression_ratio']:.2f}",
            "  COMPONENTES:",
        ]
        for prop, ok in props.items():
            lines.append(f"    [{'✓' if ok else '✗'}] {prop}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MICAgent("
            f"contratos={self._silo_manager.get_contract_count()}, "
            f"cartuchos={self._silo_manager.get_cartridge_count()}, "
            f"auditorías={self._audit_trail.total_count})"
        )

# ==============================================================================
# EXPORTACIÓN PÚBLICA
# ==============================================================================
__all__ = [
    # Excepciones y Certificados
    "MICAgentError",
    "TopologicalInvariantError",
    "StratumResolutionError",
    "ContractValidationError",
    "ClosureViolationError",
    "AlgebraicVetoError",
    "TOONCompressionError",
    "SiloAccessError",
    "ProjectionError",
    "FunctorialityError",
    "PoincareMICAdjunctionCertificate",
    
    # Enumeraciones
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
    "MathUtils",
    "normalize_stratum",
    "python_type_matches",
    "compute_json_path",
    
    # Constantes
    "MAX_AUDIT_TRAIL_SIZE",
    "TOON_START_MARKER",
    "TOON_END_MARKER",
    "TOON_FIELD_SEPARATOR",
    "ENCAPSULATION_PROTOCOL_VERSION",
    "EPS",
    "ALGEBRAIC_TOL",
    "FLOAT_COMPARISON_TOL",
    "MAX_TENSOR_RANK",
    "MAX_COMPRESSION_RATIO",
    "MIN_COMPRESSION_RATIO",
]
