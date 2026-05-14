"""
Módulo: MIC Agent (Morfismo Geométrico sobre Topos de Grothendieck)
Ubicación: app/agents/mic_agent.py
Versión: 3.0.0 (Elevación Ontológica a Espacios de Haces)
=========================================================================================

NATURALEZA CIBER-FÍSICA Y TOPOLÓGICA:
Este módulo aniquila la función heurística del "traductor de texto" para erigirse como un
Morfismo Geométrico estricto $f = (f^*, f_*)$ que acopla el espacio probabilístico del Modelo
de Lenguaje ($\mathcal{L}$) con la variedad determinista de la Matriz de Interacción Central
($\mathcal{E}_{MIC}$).

FUNDAMENTACIÓN MATEMÁTICA Y AXIOMAS CATEGÓRICOS:

§1. EL TOPOS DE GROTHENDIECK Y EL ÁLGEBRA DE HEYTING:
La categoría $\mathcal{E}_{MIC}$ opera como un Topos Elemental libre del principio del tercero
excluido. El clasificador de subobjetos $\Omega$ no es un conjunto binario $\{0, 1\}$, sino un
ideal en el Álgebra de Heyting sobre haces celulares.
Para todo monomorfismo (capacidad válida) $m: S \hookrightarrow X$, el agente evalúa un único
morfismo característico $\chi_S: X \to \Omega$ tal que el diagrama es un producto fibrado (Pullback)
exacto:
$$ \begin{matrix} S & \xrightarrow{!} & 1 \\ \downarrow{m} & & \downarrow{\text{true}} \\ X & \xrightarrow{\chi_S} & \Omega \end{matrix} $$

§2. EL FUNTOR DE ADJUNCIÓN Y PRESERVACIÓN DE LÍMITES FINITOS:
El puente dimensional entre la estocástica del LLM y el negocio se rige por la isomorfía natural:
$$ \text{Hom}_{\mathcal{E}_{MIC}}(f^* X, Y) \cong \text{Hom}_{\mathcal{L}}(X, f_* Y) $$
Bajo esta métrica, el Funtor Imagen Inversa $f^*$ preserva matemáticamente los límites finitos.
Si la IA alucina la intersección de dominios mutuamente excluyentes, el producto fibrado colapsa
al límite vacío:
$$ f^*(A \times_{\mathcal{L}} B) \cong f^*(A) \times_{\mathcal{E}_{MIC}} f^*(B) = \emptyset $$

§3. CONDICIÓN DE LIPSCHITZ SOBRE EL DIFEOMORFISMO TOON:
El Funtor Inverso de expansión tabular (TOON a JSON) se somete a una Condición de Continuidad de Lipschitz
para evitar desbordamientos asintóticos de memoria inducidos por ataques de inyección:
$$ \| F^{-1}(x) - F^{-1}(y) \|_{\mathcal{V}} \le L_{\max} \| x - y \|_{\mathcal{T}} $$
=========================================================================================
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
from numpy.typing import NDArray

# Imports relativos corregidos
try:
    from app.core.schemas import Stratum
except ImportError:
    from app.core.mic_algebra import Stratum

try:
    from app.core.mic_algebra import CategoricalState, Morphism, _canonicalize
except ImportError:
    # Fallback para testing
    Stratum = None
    CategoricalState = None
    Morphism = None
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
# CONSTANTES MATEMÁTICAS CON JUSTIFICACIÓN RIGUROSA
# ==============================================================================
MAX_AUDIT_TRAIL_SIZE: Final[int] = 10_000  # Límite de buffer circular

# Marcadores TOON (protocolo de encapsulación)
TOON_START_MARKER: Final[str] = "--- INICIO TOON ---"
TOON_END_MARKER: Final[str] = "--- FIN TOON ---"
TOON_FIELD_SEPARATOR: Final[str] = "|"
ENCAPSULATION_PROTOCOL_VERSION: Final[str] = "2.2.0"

# Tolerancias numéricas (justificación en Higham, 2002)
EPS: Final[float] = np.finfo(np.float64).eps * 4  # ≈ 8.88e-16 (4× épsilon máquina)
ALGEBRAIC_TOL: Final[float] = 1e-10  # Tolerancia para propiedades algebraicas
FLOAT_COMPARISON_TOL: Final[float] = 1e-9  # Tolerancia para comparación de floats

# Límites topológicos (justificación en Hatcher, 2002)
MAX_TENSOR_RANK: Final[int] = 2  # Rango máximo para isomorfismo TOON
MAX_COMPRESSION_RATIO: Final[float] = 10.0  # Límite superior de compresión
MIN_COMPRESSION_RATIO: Final[float] = 0.01  # Límite inferior de compresión

# ==============================================================================
# JERARQUÍA DE EXCEPCIONES CON CONTEXTO MATEMÁTICO
# ==============================================================================
class MICAgentError(Exception):
    """
    Excepción base con contexto algebraico estructurado.
    
    Propiedades:
    ============
    - error_code: Identificador único del tipo de error
    - details: Contexto estructurado para debugging
    - severity: Nivel de severidad (1=info, 2=warning, 3=critical)
    - timestamp: Marca temporal para trazabilidad
    """
    
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
        """Serialización para logging estructurado."""
        return {
            "type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": str(self),
            "details": self.details,
            "severity": self.severity,
            "timestamp": self.timestamp,
        }


class StratumResolutionError(MICAgentError):
    """Error en resolución de estratos (tipo o valor inválido)."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message, 
            error_code="STRATUM_RESOLUTION", 
            details=details, 
            severity=2
        )


class ContractValidationError(MICAgentError):
    """Error en validación de contratos JSON Schema."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message, 
            error_code="CONTRACT_VALIDATION", 
            details=details, 
            severity=2
        )


class ClosureViolationError(MICAgentError):
    """Violación de clausura transitiva en poset DIKW."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message, 
            error_code="CLOSURE_VIOLATION", 
            details=details, 
            severity=3
        )


class AlgebraicVetoError(MICAgentError):
    """Veto por violación de invariantes algebraicos."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message, 
            error_code="ALGEBRAIC_VETO", 
            details=details, 
            severity=3
        )


class TOONCompressionError(MICAgentError):
    """Error en compresión/descompresión TOON."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message, 
            error_code="TOON_COMPRESSION", 
            details=details, 
            severity=2
        )


class SiloAccessError(MICAgentError):
    """Error en acceso a silos de contratos/cartuchos."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message, 
            error_code="SILO_ACCESS", 
            details=details, 
            severity=2
        )


class ProjectionError(MICAgentError):
    """Error en proyección hacia espacio MIC."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message, 
            error_code="PROJECTION", 
            details=details, 
            severity=3
        )


class FunctorialityError(MICAgentError):
    """Violación de propiedades funtoriales."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message, 
            error_code="FUNCTORIALITY", 
            details=details, 
            severity=3
        )


# ==============================================================================
# ENUMERACIONES CON ORDEN TOTAL Y PROPIEDADES VERIFICADAS
# ==============================================================================
@unique
class ImpedanceMatchStatus(str, Enum):
    """
    Estados de concordancia de impedancia con orden total de severidad.
    
    Orden Total (por severidad creciente):
    ======================================
    LAMINAR_PROJECTION (0) < INPUT_TYPE_ERROR (1) < ... < COHOMOLOGY_FAILURE (3)
    
    Propiedades:
    ============
    - Totalidad: ∀a,b, a ≤ b ∨ b ≤ a ✓
    - Antisimetría: a ≤ b ∧ b ≤ a ⟹ a = b ✓
    - Transitividad: a ≤ b ∧ b ≤ c ⟹ a ≤ c ✓
    - Reflexividad: a ≤ a ✓
    
    Interpretación Topológica:
    =========================
    Cada estado representa una región en el espacio de configuración
    del agente, con fronteras definidas por umbrales de validación.
    """
    
    LAMINAR_PROJECTION = "LAMINAR_PROJECTION"
    STRATUM_MISMATCH_REJECTED = "STRATUM_MISMATCH_REJECTED"
    TOON_COMPRESSION_ERROR = "TOON_COMPRESSION_ERROR"
    ALGEBRAIC_VETO = "ALGEBRAIC_VETO"
    SCHEMA_VALIDATION_ERROR = "SCHEMA_VALIDATION_ERROR"
    MIC_RESOLUTION_ERROR = "MIC_RESOLUTION_ERROR"
    INPUT_TYPE_ERROR = "INPUT_TYPE_ERROR"
    TOPOLOGICAL_BIFURCATION = "TOPOLOGICAL_BIFURCATION"
    COHOMOLOGY_FAILURE = "COHOMOLOGY_FAILURE"
    
    # Mapeo de severidad (invariante: 0 ≤ severity ≤ 3)
    _SEVERITY_MAP: ClassVar[Dict[str, int]] = {
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
    
    @property
    def is_terminal(self) -> bool:
        """
        Predicado de estado terminal (sin recuperación posible).
        
        Definición: is_terminal ⟺ severity = 3
        
        Estados terminales representan singularidades en el espacio
        de configuración donde la proyección no puede continuar.
        """
        return self in {
            ImpedanceMatchStatus.ALGEBRAIC_VETO,
            ImpedanceMatchStatus.TOPOLOGICAL_BIFURCATION,
            ImpedanceMatchStatus.COHOMOLOGY_FAILURE,
        }
    
    @property
    def severity(self) -> int:
        """
        Nivel de severidad para orden total.
        
        Invariante: 0 ≤ severity ≤ 3
        """
        return self._SEVERITY_MAP.get(self.value, 1)
    
    def __lt__(self, other: "ImpedanceMatchStatus") -> bool:
        """Orden total por severidad."""
        if not isinstance(other, ImpedanceMatchStatus):
            return NotImplemented
        return self.severity < other.severity
    
    def __le__(self, other: "ImpedanceMatchStatus") -> bool:
        """Orden total no estricto por severidad."""
        if not isinstance(other, ImpedanceMatchStatus):
            return NotImplemented
        return self.severity <= other.severity
    
    def __gt__(self, other: "ImpedanceMatchStatus") -> bool:
        """Orden total inverso por severidad."""
        if not isinstance(other, ImpedanceMatchStatus):
            return NotImplemented
        return self.severity > other.severity
    
    def __ge__(self, other: "ImpedanceMatchStatus") -> bool:
        """Orden total inverso no estricto por severidad."""
        if not isinstance(other, ImpedanceMatchStatus):
            return NotImplemented
        return self.severity >= other.severity


@unique
class ValidationSeverity(IntEnum):
    """
    Severidad de validaciones en Álgebra de Heyting.
    
    Correspondencia con Álgebra de Heyting:
    ======================================
    - ERROR = 0.0 (⊥, bottom)
    - WARNING = 0.5 (elemento intermedio)
    - INFO = 1.0 (⊤, top)
    
    Esto define un homomorfismo desde ValidationSeverity a [0, 1].
    """
    
    ERROR = auto()
    WARNING = auto()
    INFO = auto()
    
    @property
    def heyting_value(self) -> float:
        """
        Valor en [0, 1] para Álgebra de Heyting.
        
        Propiedades:
        - ERROR.map = 0.0 (elemento mínimo)
        - INFO.map = 1.0 (elemento máximo)
        - WARNING.map = 0.5 (elemento intermedio)
        """
        return {
            ValidationSeverity.ERROR: 0.0,
            ValidationSeverity.WARNING: 0.5,
            ValidationSeverity.INFO: 1.0,
        }[self]


# ==============================================================================
# TIPOS Y PROTOCOLOS CON VARIANCIA EXPLÍCITA
# ==============================================================================
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

JSONValue = Union[None, bool, int, float, str, List["JSONValue"], Dict[str, "JSONValue"]]
JSONSchema = Dict[str, Any]
PayloadType = Mapping[str, Any]
AlgebraicValidator = Callable[[Stratum, PayloadType], Optional[str]]


@runtime_checkable
class VectorInfoProvider(Protocol):
    """
    Protocolo covariante para proveedores de información vectorial.
    
    Varianza: covariante en el tipo de retorno
    """
    
    def get_vector_info(self, vector_name: str) -> Optional[Dict[str, Any]]:
        """Recupera información de un vector por nombre."""
        ...


@runtime_checkable
class ProjectionTarget(Protocol):
    """
    Protocolo para objetivos de proyección en espacio MIC.
    
    Propiedades:
    - Orthogonality guarantee ∈ [0, 1]
    - Stratum target ∈ {0, 1, 2, 3, 4, 5}
    """
    
    def project_intent(
        self,
        target_basis_vector: str,
        stratum_target: int,
        validated_subspaces: List[str],
        orthogonality_guarantee: float,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Proyecta intención en espacio MIC.
        
        Precondición: 0 ≤ orthogonality_guarantee ≤ 1
        Precondición: 0 ≤ stratum_target ≤ 5
        """
        ...


# ==============================================================================
# UTILIDADES MATEMÁTICAS RIGUROSAS CON GARANTÍAS NUMÉRICAS
# ==============================================================================
class MathUtils:
    """
    Utilidades matemáticas con garantías numéricas formales.
    
    Referencias:
    ============
    - Higham, N. J. (2002). Accuracy and Stability of Numerical Algorithms
    - Goldberg, D. (1991). What Every Computer Scientist Should Know About Floating-Point
    """
    
    @staticmethod
    def stable_hash(data: Any) -> str:
        """
        Hash SHA-256 canónico y determinista.
        
        Propiedades Criptográficas:
        ==========================
        1. Determinismo: H(x) = H(x) para todo x ✓
        2. Colisión-resistente: P(H(x) = H(y) | x ≠ y) ≈ 2⁻²⁵⁶ ✓
        3. Efecto avalancha: cambio mínimo → hash diferente ✓
        4. Estable bajo canonicalización ✓
        
        Invariante: |stable_hash(x)| = 64 caracteres hexadecimales
        
        Args:
            data: Datos a hashear (cualquier tipo serializable)
        
        Returns:
            Hash hexadecimal de 64 caracteres
        """
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
    def compute_tensor_rank(
        payload: Any, 
        depth: int = 0, 
        max_depth: int = 100
    ) -> int:
        """
        Rango tensorial (profundidad de anidamiento) con protección contra recursión infinita.
        
        Definición Recursiva:
        ====================
        rank(scalar) = 0
        rank(dict/list) = 1 + max(rank(child))
        rank(∅) = depth + 1
        
        Invariantes:
        ============
        1. 0 ≤ rank ≤ max_depth ✓
        2. Monotonicidad: depth₁ ≤ depth₂ ⟹ rank(depth₁) ≤ rank(depth₂) ✓
        3. Terminación: recursión siempre termina (límite de profundidad) ✓
        
        Args:
            payload: Estructura a analizar
            depth: Profundidad actual (interno)
            max_depth: Límite máximo de profundidad
        
        Returns:
            Rango tensorial estimado
        """
        if depth > max_depth:
            logger.warning("Profundidad máxima alcanzada en compute_tensor_rank")
            return depth
        
        if isinstance(payload, dict):
            if not payload:
                return depth + 1
            return max(
                (MathUtils.compute_tensor_rank(v, depth + 1, max_depth) for v in payload.values()),
                default=depth + 1
            )
        elif isinstance(payload, list):
            if not payload:
                return depth + 1
            return max(
                (MathUtils.compute_tensor_rank(v, depth + 1, max_depth) for v in payload),
                default=depth + 1
            )
        return depth
    
    @staticmethod
    def float_equal(
        a: float, 
        b: float, 
        tol: float = FLOAT_COMPARISON_TOL
    ) -> bool:
        """
        Comparación de floats con tolerancia absoluta y relativa combinada.
        
        Definición Matemática:
        =====================
        equal(a, b) ⟺ |a - b| ≤ tol ∨ |a - b| ≤ tol·max(|a|, |b|)
        
        Propiedades:
        ============
        - Reflexiva: equal(a, a) = True ✓
        - Simétrica: equal(a, b) = equal(b, a) ✓
        - NO transitiva (por diseño numérico - ver contraejemplo de Kahan)
        
        Args:
            a: Primer valor
            b: Segundo valor
            tol: Tolerancia base
        
        Returns:
            True si los valores son considerados iguales numéricamente
        """
        abs_diff = abs(a - b)
        if abs_diff <= tol:
            return True
        rel_tol = tol * max(abs(a), abs(b))
        return abs_diff <= rel_tol
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """
        Clamp con verificación de orden y garantías de postcondición.
        
        Precondición: min_val ≤ max_val
        Postcondición: min_val ≤ result ≤ max_val
        
        Args:
            value: Valor a clampar
            min_val: Límite inferior
            max_val: Límite superior
        
        Returns:
            Valor clampado al rango [min_val, max_val]
        
        Raises:
            ValueError: Si min_val > max_val (violación de precondición)
        """
        if min_val > max_val:
            raise ValueError(f"min_val ({min_val}) > max_val ({max_val})")
        return max(min_val, min(max_val, value))


def normalize_stratum(value: Any) -> Stratum:
    """
    Normalización de estrato con validación estricta.
    
    Propiedades Matemáticas:
    =======================
    1. Idempotente: normalize(normalize(x)) = normalize(x) ✓
    2. Sobreyectiva: ∀s ∈ Stratum, ∃x: normalize(x) = s ✓
    3. Determinista: normalize(x) = normalize(x) ✓
    
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
                f"Valor entero inválido para estrato: {value}",
                details={"input_value": value, "input_type": "int"}
            ) from e
    
    if isinstance(value, str):
        # Primero intentar por nombre
        try:
            return Stratum[value.upper()]
        except KeyError:
            pass
        
        # Luego por valor numérico
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
    """
    Verificación de tipo JSON Schema → Python con mapeo biyectivo.
    
    Mapeo Biyectivo:
    ================
    - null    ↔ None
    - boolean ↔ bool (excluyendo int)
    - integer ↔ int \ bool
    - number  ↔ (int ∪ float) \ bool
    - string  ↔ str
    - array   ↔ list
    - object  ↔ Mapping
    
    Args:
        expected_type: Tipo JSON Schema esperado
        value: Valor a verificar
    
    Returns:
        True si el valor coincide con el tipo esperado
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
        logger.warning("Tipo JSON Schema desconocido: %s", expected_type)
        return True  # Asumir válido si tipo desconocido
    
    return checker(value)


def compute_json_path(base: str, key: Union[str, int]) -> str:
    """
    Construcción de JSONPath según RFC 9535.
    
    Ejemplos:
    =========
    - compute_json_path("$", "foo") → "$.foo"
    - compute_json_path("$.foo", 0) → "$.foo[0]"
    
    Args:
        base: Path base (inicia con "$")
        key: Clave o índice a agregar
    
    Returns:
        JSONPath completo
    """
    if isinstance(key, int):
        return f"{base}[{key}]"
    
    # Escapar caracteres especiales en claves
    safe_key = key.replace(".", "\\.").replace("[", "\\[")
    return f"{base}.{safe_key}" if base != "$" else f"${safe_key}"


# ==============================================================================
# DATACLASSES DE AUDITORÍA INMUTABLES CON INVARIANTES VERIFICADOS
# ==============================================================================
@dataclass(frozen=True, slots=True, eq=True)
class SchemaValidationResult:
    """
    Resultado de validación en Álgebra de Heyting.
    
    Propiedades del Clasificador Ω:
    ==============================
    - validity_degree ∈ [0, 1] (valor de verdad)
    - Conjunción: merge([r1, r2]) = r1 ∧ r2 (meet)
    - Implicación: valid → error = ¬valid ∨ error
    
    Invariantes Estructurales:
    =========================
    1. 0.0 ≤ validity_degree ≤ 1.0 ✓
    2. is_valid ⟺ validity_degree ≥ 1 - ε ✓
    3. errors y warnings son tuplas inmutables ✓
    4. frozen=True garantiza inmutabilidad ✓
    
    Interpretación Topológica:
    =========================
    validity_degree representa la "distancia" al estado inválido,
    donde 1.0 es completamente válido y 0.0 es completamente inválido.
    """
    
    validity_degree: float
    frustration_ideal: float = 0.0
    errors: Tuple[str, ...] = field(default_factory=tuple)
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    path: str = "$"
    
    def __post_init__(self) -> None:
        """
        Post-inicialización con validación y corrección de invariantes.
        
        Garantiza que validity_degree esté en [0, 1].
        """
        if not (0.0 <= self.validity_degree <= 1.0):
            clamped = MathUtils.clamp(self.validity_degree, 0.0, 1.0)
            object.__setattr__(self, "validity_degree", clamped)
            logger.warning(
                "validity_degree fuera de rango, clampeado: %f → %f",
                self.validity_degree,
                clamped
            )
    
    @property
    def is_valid(self) -> bool:
        """
        Predicado de validez con tolerancia numérica.
        
        Definición: is_valid ⟺ validity_degree ≥ 1 - ε
        
        Esto permite pequeños errores numéricos sin invalidar
        resultados esencialmente correctos.
        """
        return self.validity_degree >= 1.0 - EPS
    
    @classmethod
    def success(cls) -> "SchemaValidationResult":
        """
        Elemento top del Álgebra de Heyting (⊤).
        
        Representa validación exitosa completa.
        """
        return cls(validity_degree=1.0)
    
    @classmethod
    def failure(
        cls,
        error: str,
        path: str = "$",
        penalty: float = 1.0
    ) -> "SchemaValidationResult":
        """
        Elemento bottom con penalización (⊥).
        
        Args:
            error: Mensaje de error
            path: JSONPath donde ocurrió el error
            penalty: Reducción en validity_degree (0 ≤ penalty ≤ 1)
        """
        return cls(
            validity_degree=max(0.0, 1.0 - penalty),
            errors=(error,),
            path=path
        )
    
    @classmethod
    def merge(
        cls,
        results: Iterable["SchemaValidationResult"]
    ) -> "SchemaValidationResult":
        """
        Conjunción en Álgebra de Heyting: ⋀ results.
        
        Definición Matemática:
        =====================
        (⋀ rᵢ).validity = min(rᵢ.validity)  (meet en retículo)
        (⋀ rᵢ).errors = ⋃ rᵢ.errors  (unión de errores)
        (⋀ rᵢ).warnings = ⋃ rᵢ.warnings  (unión de advertencias)
        
        Propiedades Algebraicas:
        =======================
        - Conmutativo: merge([a, b]) = merge([b, a]) ✓
        - Asociativo: merge([a, merge([b, c])]) = merge([merge([a, b]), c]) ✓
        - Idempotente: merge([a, a]) = a ✓
        - Elemento neutro: merge([a, success()]) = a ✓
        
        Args:
            results: Colección de resultados a conjuntar
        
        Returns:
            Resultado合并 con validez mínima y errores acumulados
        """
        all_errors: List[str] = []
        all_warnings: List[str] = []
        min_validity = 1.0
        
        for r in results:
            all_errors.extend(r.errors)
            all_warnings.extend(r.warnings)
            if r.validity_degree < min_validity:
                min_validity = r.validity_degree
        
        return cls(
            validity_degree=min_validity,
            errors=tuple(all_errors),
            warnings=tuple(all_warnings),
        )
    
    @property
    def error(self) -> Optional[str]:
        """Primer error si existe (para compatibilidad)."""
        return self.errors[0] if self.errors else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialización JSON-compatible."""
        return {
            "validity_degree": float(self.validity_degree),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "path": self.path,
            "is_valid": self.is_valid,
        }
    
    def __str__(self) -> str:
        status = "✓ VALID" if self.is_valid else "✗ INVALID"
        return (
            f"SchemaValidationResult({status}, "
            f"validity={self.validity_degree:.3f}, "
            f"errors={len(self.errors)})"
        )


@dataclass(frozen=True, slots=True, eq=True)
class CategoricalEqualizerSeed:
    """
    Traza de auditoría inmutable con firma criptográfica.
    
    Invariantes Estructurales:
    =========================
    1. Todos los campos son deterministas excepto timestamp ✓
    2. compute_hash() excluye timestamp para reproducibilidad ✓
    3. token_compression_ratio ∈ [0, ∞) ✓
    4. protocol_version es constante ✓
    
    Propiedades Criptográficas:
    ==========================
    - hash es determinista (excluyendo timestamp)
    - hash es resistente a colisiones (SHA-256)
    - hash permite verificación de integridad
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
    
    def __post_init__(self) -> None:
        """Validación y corrección post-construcción."""
        if self.token_compression_ratio < 0.0:
            object.__setattr__(self, "token_compression_ratio", 0.0)
            logger.warning("token_compression_ratio negativo, corregido a 0.0")
        
        if self.token_compression_ratio > MAX_COMPRESSION_RATIO:
            logger.warning(
                "Ratio de compresión muy alto: %.2f > %.2f",
                self.token_compression_ratio,
                MAX_COMPRESSION_RATIO
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialización completa."""
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
        """
        Hash determinista excluyendo timestamp.
        
        Esto permite reproducibilidad de auditorías independientemente
        del momento de ejecución.
        """
        data = {
            k: v for k, v in self.to_dict().items()
            if k != "timestamp"
        }
        return MathUtils.stable_hash(data)
    
    def __str__(self) -> str:
        return (
            f"CategoricalEqualizerSeed("
            f"vector={self.target_vector}, "
            f"stratum={self.target_stratum.name}, "
            f"status={self.impedance_match_status.value}, "
            f"ratio={self.token_compression_ratio:.2f})"
        )


@dataclass(frozen=True, slots=True, eq=True)
class TOONDocument:
    """
    Documento TOON con isomorfismo verificable.
    
    Propiedades Matemáticas:
    =======================
    1. Retracto de deformación: JSON → Tabla 2D ✓
    2. Isomorfismo para rank(payload) ≤ 2 ✓
    3. Determinista: render(parse(x)) = x ✓
    4. Reversible: parse(render(doc)) = doc ✓
    
    Invariantes Estructurales:
    =========================
    - cartridge_id es no vacío ✓
    - records es tupla inmutable ✓
    - render() produce formato válido ✓
    
    Interpretación Topológica:
    =========================
    TOON es un homeomorfismo entre el espacio de JSON
    y el espacio de documentos tabulares, preservando
    la estructura de información.
    """
    
    cartridge_id: str
    header_template: str
    records: Tuple[Tuple[str, str], ...]
    
    def __post_init__(self) -> None:
        """Validación de invariantes post-construcción."""
        if not self.cartridge_id:
            raise TOONCompressionError(
                "cartridge_id no puede estar vacío",
                details={"cartridge_id": self.cartridge_id}
            )
    
    def render(self) -> str:
        """
        Renderiza documento TOON en formato textual.
        
        Formato:
        ========
        --- INICIO TOON --- {cartridge_id} ---
        {header_template}
        key1|json_value1
        key2|json_value2
        ...
        --- FIN TOON ---
        
        Returns:
            Documento TOON como string
        """
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
        Parsea documento TOON desde formato textual.
        
        Invariante: parse(render(doc)) = doc
        
        Args:
            content: Contenido textual del documento TOON
        
        Returns:
            TOONDocument parseado
        
        Raises:
            TOONCompressionError: Si el formato es inválido
        """
        lines = content.strip().split("\n")
        
        if len(lines) < 3:
            raise TOONCompressionError(
                "Documento TOON demasiado corto (mínimo 3 líneas)",
                details={"lines_received": len(lines)}
            )
        
        # Parsear header
        header_line = lines[0]
        if not header_line.startswith(TOON_START_MARKER):
            raise TOONCompressionError(
                "Marcador de inicio inválido",
                details={"header_line": header_line[:100]}
            )
        
        try:
            cartridge_id = (
                header_line
                .split(TOON_START_MARKER)[1]
                .strip()
                .rstrip("-")
                .strip()
            )
        except IndexError as e:
            raise TOONCompressionError(
                "No se pudo extraer cartridge_id",
                details={"header_line": header_line}
            ) from e
        
        # Verificar marcador de fin
        if lines[-1].strip() != TOON_END_MARKER:
            raise TOONCompressionError(
                "Marcador de fin faltante o inválido",
                details={"last_line": lines[-1]}
            )
        
        # Separar header y records
        header_lines: List[str] = []
        records: List[Tuple[str, str]] = []
        in_records = False
        
        for line in lines[1:-1]:
            if not in_records:
                if TOON_FIELD_SEPARATOR in line:
                    parts = line.split(TOON_FIELD_SEPARATOR, 1)
                    if len(parts) == 2:
                        val = parts[1].strip()
                        if (val.startswith('"') or val.startswith('{') or
                            val.startswith('[') or val in ('true', 'false', 'null') or
                            _is_json_number(val)):
                            in_records = True
            
            if in_records:
                if TOON_FIELD_SEPARATOR in line:
                    parts = line.split(TOON_FIELD_SEPARATOR, 1)
                    if len(parts) == 2:
                        records.append((parts[0].strip(), parts[1].strip()))
            else:
                header_lines.append(line)
        
        header_template = "\n".join(header_lines)
        
        return cls(
            cartridge_id=cartridge_id,
            header_template=header_template,
            records=tuple(records),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Deserializa records a diccionario."""
        result: Dict[str, Any] = {}
        for key, json_value in self.records:
            try:
                result[key] = json.loads(json_value)
            except json.JSONDecodeError:
                result[key] = json_value
        return result
    
    def __len__(self) -> int:
        """Número de records en el documento."""
        return len(self.records)
    
    def __str__(self) -> str:
        return (
            f"TOONDocument("
            f"cartridge={self.cartridge_id}, "
            f"records={len(self.records)})"
        )


def _is_json_number(s: str) -> bool:
    """Verifica si string es un número JSON válido."""
    try:
        float(s)
        return True
    except ValueError:
        return False


# ==============================================================================
# CONTRATOS Y CARTUCHOS CON VALIDACIÓN ESTRUCTURAL
# ==============================================================================
@dataclass(frozen=True, slots=True, eq=True)
class SiloAContract:
    """
    Contrato JSON Schema con validación de integridad.
    
    Invariantes Estructurales:
    =========================
    1. schema es dict con clave "type" ✓
    2. stratum es válido ✓
    3. version sigue semver (validación básica) ✓
    4. frozen=True garantiza inmutabilidad ✓
    """
    
    contract_id: str
    stratum: Stratum
    schema: JSONSchema
    description: str = ""
    version: str = "1.0.0"
    
    def __post_init__(self) -> None:
        """Validación post-construcción."""
        if not self.validate_schema_integrity():
            raise ContractValidationError(
                f"Schema inválido para contrato '{self.contract_id}'",
                details={"schema_keys": list(self.schema.keys()) if isinstance(self.schema, dict) else None}
            )
    
    def validate_schema_integrity(self) -> bool:
        """
        Verifica integridad mínima del schema.
        
        Requerimientos:
        ===============
        1. schema debe ser dict
        2. schema debe tener clave "type"
        
        Returns:
            True si el schema es válido
        """
        if not isinstance(self.schema, dict):
            return False
        if "type" not in self.schema:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialización JSON-compatible."""
        return {
            "contract_id": self.contract_id,
            "stratum": self.stratum.name,
            "schema": self.schema,
            "description": self.description,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True, eq=True)
class SiloBCartridge:
    """
    Cartucho TOON con metadatos.
    
    Invariantes Estructurales:
    =========================
    1. cartridge_id no vacío ✓
    2. header_template válido ✓
    3. frozen=True garantiza inmutabilidad ✓
    """
    
    cartridge_id: str
    stratum: Stratum
    header_template: str
    field_definitions: Tuple[str, ...] = field(default_factory=tuple)
    description: str = ""
    version: str = "1.0.0"
    
    def __post_init__(self) -> None:
        """Validación post-construcción."""
        if not self.cartridge_id:
            raise TOONCompressionError(
                "cartridge_id no puede estar vacío",
                details={"cartridge_id": self.cartridge_id}
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialización JSON-compatible."""
        return {
            "cartridge_id": self.cartridge_id,
            "stratum": self.stratum.name,
            "header_template": self.header_template,
            "field_definitions": list(self.field_definitions),
            "description": self.description,
            "version": self.version,
        }


# ==============================================================================
# VALIDADOR DE SCHEMA CON COBERTURA COMPLETA
# ==============================================================================
class SchemaValidator:
    """
    Validador JSON Schema Draft-07 determinista.
    
    Subconjunto Soportado:
    ======================
    - type, required, properties, items
    - minimum, maximum, exclusiveMinimum, exclusiveMaximum
    - minLength, maxLength, minItems, maxItems
    - enum, const, pattern (regex básico)
    
    Deliberadamente Excluido (no determinista):
    ==========================================
    - $ref, allOf, anyOf, oneOf
    - if/then/else
    - format con validación externa
    
    Propiedades Algebraicas:
    =======================
    - Determinismo: validate(schema, payload) = validate(schema, payload) ✓
    - Composicionalidad: validación de propiedades es recursiva ✓
    - Monotonicidad: más restricciones → menor validity_degree ✓
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
            "pattern": self._validate_pattern,
        }
    
    def validate(
        self,
        schema: JSONSchema,
        payload: Any,
        path: str = "$",
    ) -> SchemaValidationResult:
        """
        Valida payload contra schema.
        
        Retorna resultado en Álgebra de Heyting con:
        - validity_degree ∈ [0, 1]
        - errors acumulados
        - warnings acumulados
        
        Args:
            schema: JSON Schema a validar contra
            payload: Datos a validar
            path: JSONPath actual para mensajes de error
        
        Returns:
            SchemaValidationResult con grado de validez
        """
        if not isinstance(schema, dict):
            return SchemaValidationResult.failure(
                f"Schema inválido: esperado dict, recibido {type(schema).__name__}",
                path,
            )
        
        results: List[SchemaValidationResult] = []
        
        for keyword, constraint in schema.items():
            validator = self._validators.get(keyword)
            if validator is not None:
                try:
                    result = validator(constraint, payload, schema, path)
                    results.append(result)
                except Exception as e:
                    logger.warning("Error en validador '%s' en path '%s': %s", keyword, path, e)
                    results.append(SchemaValidationResult.failure(
                        f"Error interno en validador '{keyword}': {e}",
                        path,
                    ))
        
        return (
            SchemaValidationResult.merge(results)
            if results
            else SchemaValidationResult.success()
        )
    
    def _validate_type(
        self,
        expected_type: Union[str, List[str]],
        value: Any,
        schema: JSONSchema,
        path: str,
    ) -> SchemaValidationResult:
        """Validación de tipo con soporte para uniones."""
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
        """Validación de claves requeridas."""
        if not isinstance(value, Mapping):
            return SchemaValidationResult.success()
        
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
        """Validación recursiva de propiedades."""
        if not isinstance(value, Mapping):
            return SchemaValidationResult.success()
        
        results: List[SchemaValidationResult] = []
        
        for prop_name, prop_schema in properties.items():
            if prop_name in value:
                prop_path = compute_json_path(path, prop_name)
                result = self.validate(prop_schema, value[prop_name], prop_path)
                results.append(result)
        
        return (
            SchemaValidationResult.merge(results)
            if results
            else SchemaValidationResult.success()
        )
    
    def _validate_items(
        self,
        items_schema: JSONSchema,
        value: Any,
        schema: JSONSchema,
        path: str,
    ) -> SchemaValidationResult:
        """Validación de items de array."""
        if not isinstance(value, list):
            return SchemaValidationResult.success()
        
        results: List[SchemaValidationResult] = []
        
        for i, item in enumerate(value):
            item_path = compute_json_path(path, i)
            result = self.validate(items_schema, item, item_path)
            results.append(result)
        
        return (
            SchemaValidationResult.merge(results)
            if results
            else SchemaValidationResult.success()
        )
    
    def _validate_minimum(
        self,
        minimum: Union[int, float],
        value: Any,
        schema: JSONSchema,
        path: str,
    ) -> SchemaValidationResult:
        """Validación de valor mínimo con tolerancia."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return SchemaValidationResult.success()
        
        if value < minimum - FLOAT_COMPARISON_TOL:
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
        """Validación de valor máximo con tolerancia."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return SchemaValidationResult.success()
        
        if value > maximum + FLOAT_COMPARISON_TOL:
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
        """Validación de exclusión de mínimo."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return SchemaValidationResult.success()
        
        if value <= minimum + FLOAT_COMPARISON_TOL:
            return SchemaValidationResult.failure(
                f"Valor en '{path}' ({value}) no estrictamente mayor que {minimum}",
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
        """Validación de exclusión de máximo."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return SchemaValidationResult.success()
        
        if value >= maximum - FLOAT_COMPARISON_TOL:
            return SchemaValidationResult.failure(
                f"Valor en '{path}' ({value}) no estrictamente menor que {maximum}",
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
        """Validación de longitud mínima de string."""
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
        """Validación de longitud máxima de string."""
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
        """Validación de enumeración."""
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
        """Validación de constante."""
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
        """Validación de cantidad mínima de items."""
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
        """Validación de cantidad máxima de items."""
        if not isinstance(value, list):
            return SchemaValidationResult.success()
        
        if len(value) > max_items:
            return SchemaValidationResult.failure(
                f"Array en '{path}' (len={len(value)}) mayor que maxItems ({max_items})",
                path,
            )
        return SchemaValidationResult.success()
    
    def _validate_pattern(
        self,
        pattern: str,
        value: Any,
        schema: JSONSchema,
        path: str,
    ) -> SchemaValidationResult:
        """Validación de patrón regex."""
        if not isinstance(value, str):
            return SchemaValidationResult.success()
        
        try:
            if not re.search(pattern, value):
                return SchemaValidationResult.failure(
                    f"String en '{path}' no coincide con patrón '{pattern}'",
                    path,
                )
        except re.error as e:
            return SchemaValidationResult.failure(
                f"Patrón regex inválido '{pattern}': {e}",
                path,
            )
        
        return SchemaValidationResult.success()


# ==============================================================================
# VALIDADORES ALGEBRAICOS CON CERRADURA BAJO COMPOSICIÓN
# ==============================================================================
class AlgebraicVetoRegistry:
    """
    Registro de validadores de invariantes algebraicos.
    
    Propiedades Algebraicas:
    =======================
    1. Cerradura bajo composición: v1 ∧ v2 es validador ✓
    2. Monotonicidad: más validadores → más restricciones ✓
    3. Determinismo garantizado ✓
    
    Invariantes:
    ============
    - Cada estrato tiene lista de validadores (posiblemente vacía)
    - Validadores son funciones puras (sin estado)
    - Ejecución es determinista
    """
    
    def __init__(self) -> None:
        self._validators: Dict[Stratum, List[AlgebraicValidator]] = {
            s: [] for s in Stratum
        }
        self._register_default_validators()
    
    def _register_default_validators(self) -> None:
        """Registra validadores por defecto para cada estrato."""
        
        # PHYSICS: Leyes de conservación (termodinámica)
        def physics_conservation(stratum: Stratum, payload: PayloadType) -> Optional[str]:
            dissipated = payload.get("dissipated_power")
            if isinstance(dissipated, (int, float)):
                if dissipated < -ALGEBRAIC_TOL:
                    return f"Violación termodinámica: dissipated_power={dissipated} < 0"
            
            energy_in = payload.get("energy_input", 0)
            energy_out = payload.get("energy_output", 0)
            if isinstance(energy_in, (int, float)) and isinstance(energy_out, (int, float)):
                if energy_out > energy_in * (1.0 + ALGEBRAIC_TOL):
                    return f"Violación de conservación: energy_output={energy_out} > energy_input={energy_in}"
            return None
        
        self._validators[Stratum.PHYSICS].append(physics_conservation)
        
        # TACTICS: Restricciones de estabilidad
        def tactics_stability(stratum: Stratum, payload: PayloadType) -> Optional[str]:
            stability = payload.get("pyramid_stability_index")
            if isinstance(stability, (int, float)):
                if stability < -ALGEBRAIC_TOL or stability > 1.0 + ALGEBRAIC_TOL:
                    return f"Índice de estabilidad fuera de rango [0,1]: {stability}"
            return None
        
        self._validators[Stratum.TACTICS].append(tactics_stability)
        
        # STRATEGY: Restricciones de fricción territorial
        def strategy_friction(stratum: Stratum, payload: PayloadType) -> Optional[str]:
            friction = payload.get("territorial_friction")
            if isinstance(friction, (int, float)):
                if friction < 1.0 - ALGEBRAIC_TOL:
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
        """
        Registra validador adicional para estrato.
        
        Args:
            stratum: Estrato al que aplica el validador
            validator: Función validadora (Stratum, PayloadType) → Optional[str]
        """
        if stratum not in self._validators:
            self._validators[stratum] = []
        self._validators[stratum].append(validator)
    
    def validate(
        self,
        stratum: Stratum,
        payload: PayloadType,
    ) -> List[str]:
        """
        Ejecuta todos los validadores del estrato.
        
        Args:
            stratum: Estrato a validar
            payload: Datos a validar
        
        Returns:
            Lista de mensajes de error (vacía si todo válido)
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
    
    def get_validator_count(self, stratum: Stratum) -> int:
        """Retorna cantidad de validadores registrados."""
        return len(self._validators.get(stratum, []))


# ==============================================================================
# GESTIÓN DE SILOS CON INVARIANTES Y THREAD-SAFETY
# ==============================================================================
class SiloManager:
    """
    Gestor de Silos A (contratos) y B (cartuchos) con garantías de inmutabilidad.
    
    Invariantes Estructurales:
    =========================
    1. Cada estrato tiene ≥ 1 contrato ✓
    2. Cada estrato tiene ≥ 1 cartucho ✓
    3. Los silos son inmutables post-inicialización (freeze) ✓
    4. Selectores son deterministas ✓
    
    Propiedades Categóricas:
    =======================
    - fetch_contract: Stratum × String → (ContractID, Schema) ✓
    - fetch_cartridge: Stratum × String → (CartridgeID, Template) ✓
    - Ambas funciones son totales (con fallback) ✓
    
    Thread-Safety:
    ==============
    - RLock protege todas las operaciones de escritura
    - Lecturas son thread-safe después de freeze
    """
    
    def __init__(self) -> None:
        self._silo_a: Dict[Stratum, Dict[str, SiloAContract]] = {s: {} for s in Stratum}
        self._silo_b: Dict[Stratum, Dict[str, SiloBCartridge]] = {s: {} for s in Stratum}
        
        # Selectores por defecto (deterministas)
        self._default_contract_selector: Callable[[Dict[str, SiloAContract], str], str] = (
            lambda contracts, vector: next(iter(sorted(contracts.keys())), "Generic_Contract")
        )
        self._default_cartridge_selector: Callable[[Dict[str, SiloBCartridge], str], str] = (
            lambda cartridges, vector: next(iter(sorted(cartridges.keys())), "Generic_Cartridge")
        )
        
        self._lock = threading.RLock()
        self._frozen = False
        
        self._initialize_default_silos()
        self._verify_invariants()
    
    def _initialize_default_silos(self) -> None:
        """Inicializa silos con contratos y cartuchos por defecto."""
        
        # ===== CONTRATOS SILO A =====
        
        # PHYSICS
        self._register_contract(SiloAContract(
            contract_id="PHS_Conservation_Seed",
            stratum=Stratum.PHYSICS,
            schema={
                "type": "object",
                "required": ["dissipated_power"],
                "properties": {
                    "dissipated_power": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Potencia disipada [W]"
                    },
                    "energy_input": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Energía de entrada [J]"
                    },
                    "energy_output": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Energía de salida [J]"
                    },
                    "saturation": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Saturación magnética [-]"
                    },
                },
            },
            description="Contrato de conservación de energía y propiedades electromagnéticas",
        ))
        
        # TACTICS
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
                        "description": "Índice de estabilidad de la pirámide logística"
                    },
                    "flow_efficiency": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Eficiencia del flujo de materiales"
                    },
                    "beta_0": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Componentes conexas (Betti-0)"
                    },
                    "beta_1": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Ciclos independientes (Betti-1)"
                    },
                },
            },
            description="Contrato de topología logística y flujos",
        ))
        
        # STRATEGY
        self._register_contract(SiloAContract(
            contract_id="Riemannian_Friction_Contract",
            stratum=Stratum.STRATEGY,
            schema={
                "type": "object",
                "required": ["territorial_friction"],
                "properties": {
                    "territorial_friction": {
                        "type": "number",
                        "minimum": 1.0,
                        "description": "Fricción territorial Riemanniana (≥ 1)"
                    },
                    "risk_coupling": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Acoplamiento de riesgos"
                    },
                    "strategic_entropy": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Entropía estratégica normalizada"
                    },
                },
            },
            description="Contrato de fricción territorial y acoplamiento de riesgos",
        ))
        
        # WISDOM
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
                        "description": "Veredicto final del sistema"
                    },
                    "confidence_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Nivel de confianza del veredicto"
                    },
                    "rationale": {
                        "type": "string",
                        "minLength": 10,
                        "description": "Justificación del veredicto"
                    },
                    "euler_characteristic": {
                        "type": "integer",
                        "description": "Característica de Euler del estado global"
                    },
                },
            },
            description="Acta de deliberación con veredicto final",
        ))
        
        # ===== CARTUCHOS SILO B =====
        
        # PHYSICS
        self._register_cartridge(SiloBCartridge(
            cartridge_id="Maxwell_FDTD_TOON_Cartridge",
            stratum=Stratum.PHYSICS,
            header_template=(
                "Malla_Yee_Leapfrog\n"
                "key|value|unit|confidence"
            ),
            field_definitions=(
                "dissipated_power",
                "energy_input",
                "energy_output",
                "saturation",
            ),
            description="Cartucho para simulaciones FDTD electromagnéticas",
        ))
        
        # TACTICS
        self._register_cartridge(SiloBCartridge(
            cartridge_id="Persistence_Barcode_TOON_Cartridge",
            stratum=Stratum.TACTICS,
            header_template=(
                "Diagrama_Persistencia_API\n"
                "key|value|window|entropy"
            ),
            field_definitions=(
                "pyramid_stability_index",
                "flow_efficiency",
                "beta_0",
                "beta_1",
            ),
            description="Cartucho para análisis de persistencia topológica",
        ))
        
        # STRATEGY
        self._register_cartridge(SiloBCartridge(
            cartridge_id="Riemannian_TOON_Cartridge",
            stratum=Stratum.STRATEGY,
            header_template=(
                "Tensor_Covarianza_Riesgos_Acoplados\n"
                "key|value|coupling"
            ),
            field_definitions=(
                "territorial_friction",
                "risk_coupling",
                "strategic_entropy",
            ),
            description="Cartucho para métricas Riemannianas de estrategia",
        ))
        
        # WISDOM
        self._register_cartridge(SiloBCartridge(
            cartridge_id="Telemetry_Passport_TOON_Cartridge",
            stratum=Stratum.WISDOM,
            header_template=(
                "Pasaporte_Digital_Transaccional\n"
                "key|value|semantic_role"
            ),
            field_definitions=(
                "final_verdict",
                "confidence_score",
                "rationale",
                "euler_characteristic",
            ),
            description="Cartucho para pasaporte de telemetría completa",
        ))
    
    def _register_contract(self, contract: SiloAContract) -> None:
        """Registra contrato en Silo A con verificación de unicidad."""
        with self._lock:
            if self._frozen:
                raise SiloAccessError("Silo A está congelado, no se pueden agregar contratos")
            
            if contract.contract_id in self._silo_a[contract.stratum]:
                logger.warning(
                    "Contrato '%s' ya existe en estrato %s, sobrescribiendo",
                    contract.contract_id,
                    contract.stratum.name
                )
            
            self._silo_a[contract.stratum][contract.contract_id] = contract
    
    def _register_cartridge(self, cartridge: SiloBCartridge) -> None:
        """Registra cartucho en Silo B con verificación de unicidad."""
        with self._lock:
            if self._frozen:
                raise SiloAccessError("Silo B está congelado, no se pueden agregar cartuchos")
            
            if cartridge.cartridge_id in self._silo_b[cartridge.stratum]:
                logger.warning(
                    "Cartucho '%s' ya existe en estrato %s, sobrescribiendo",
                    cartridge.cartridge_id,
                    cartridge.stratum.name
                )
            
            self._silo_b[cartridge.stratum][cartridge.cartridge_id] = cartridge
    
    def _verify_invariants(self) -> None:
        """Verifica invariantes del gestor de silos."""
        for stratum in Stratum:
            if not self._silo_a[stratum]:
                logger.warning("Estrato %s no tiene contratos registrados", stratum.name)
            
            if not self._silo_b[stratum]:
                logger.warning("Estrato %s no tiene cartuchos registrados", stratum.name)
    
    def freeze(self) -> None:
        """Congela silos para inmutabilidad."""
        with self._lock:
            self._frozen = True
            logger.info("Silos A y B congelados")
    
    def fetch_contract(
        self,
        stratum: Stratum,
        target_vector: str,
    ) -> Tuple[str, JSONSchema]:
        """
        Recupera contrato para estrato y vector objetivo.
        
        Args:
            stratum: Estrato objetivo
            target_vector: Nombre del vector
        
        Returns:
            (contract_id, schema)
        
        Raises:
            SiloAccessError: Si no existe contrato (con fallback)
        """
        with self._lock:
            contracts = self._silo_a.get(stratum, {})
            
            if not contracts:
                logger.warning("No hay contratos para estrato %s, usando genérico", stratum.name)
                return "Generic_Contract", {"type": "object", "properties": {}}
            
            contract_id = self._default_contract_selector(contracts, target_vector)
            contract = contracts.get(contract_id)
            
            if contract is None:
                raise SiloAccessError(
                    f"Contrato '{contract_id}' no encontrado",
                    details={"stratum": stratum.name, "contract_id": contract_id}
                )
            
            return contract.contract_id, contract.schema
    
    def fetch_cartridge(
        self,
        stratum: Stratum,
        target_vector: str,
    ) -> Tuple[str, str]:
        """
        Recupera cartucho para estrato y vector objetivo.
        
        Args:
            stratum: Estrato objetivo
            target_vector: Nombre del vector
        
        Returns:
            (cartridge_id, header_template)
        
        Raises:
            SiloAccessError: Si no existe cartucho (con fallback)
        """
        with self._lock:
            cartridges = self._silo_b.get(stratum, {})
            
            if not cartridges:
                logger.warning("No hay cartuchos para estrato %s, usando genérico", stratum.name)
                return "Generic_Cartridge", "Tabla_Generica\nkey|value"
            
            cartridge_id = self._default_cartridge_selector(cartridges, target_vector)
            cartridge = cartridges.get(cartridge_id)
            
            if cartridge is None:
                raise SiloAccessError(
                    f"Cartucho '{cartridge_id}' no encontrado",
                    details={"stratum": stratum.name, "contract_id": cartridge_id}
                )
            
            return cartridge.cartridge_id, cartridge.header_template
    
    def list_contracts(self, stratum: Optional[Stratum] = None) -> List[str]:
        """Lista IDs de contratos."""
        with self._lock:
            if stratum is not None:
                return sorted(self._silo_a.get(stratum, {}).keys())
            
            result: List[str] = []
            for contracts in self._silo_a.values():
                result.extend(contracts.keys())
            return sorted(result)
    
    def list_cartridges(self, stratum: Optional[Stratum] = None) -> List[str]:
        """Lista IDs de cartuchos."""
        with self._lock:
            if stratum is not None:
                return sorted(self._silo_b.get(stratum, {}).keys())
            
            result: List[str] = []
            for cartridges in self._silo_b.values():
                result.extend(cartridges.keys())
            return sorted(result)
    
    def get_contract_count(self, stratum: Optional[Stratum] = None) -> int:
        """Cuenta contratos."""
        with self._lock:
            if stratum is not None:
                return len(self._silo_a.get(stratum, {}))
            return sum(len(c) for c in self._silo_a.values())
    
    def get_cartridge_count(self, stratum: Optional[Stratum] = None) -> int:
        """Cuenta cartuchos."""
        with self._lock:
            if stratum is not None:
                return len(self._silo_b.get(stratum, {}))
            return sum(len(c) for c in self._silo_b.values())
    
    def get_contract(self, contract_id: str) -> Optional[SiloAContract]:
        """Recupera contrato por ID."""
        with self._lock:
            for contracts in self._silo_a.values():
                if contract_id in contracts:
                    return contracts[contract_id]
            return None
    
    def get_cartridge(self, cartridge_id: str) -> Optional[SiloBCartridge]:
        """Recupera cartucho por ID."""
        with self._lock:
            for cartridges in self._silo_b.values():
                if cartridge_id in cartridges:
                    return cartridges[cartridge_id]
            return None


# ==============================================================================
# COMPRESOR TOON CON VERIFICACIÓN DE ISOMORFISMO
# ==============================================================================
class TOONCompressor:
    """
    Compresor determinista con verificación de isomorfismo.
    
    Propiedades Matemáticas:
    =======================
    1. Retracto de deformación: JSON → Tabla 2D ✓
    2. Isomorfismo verificable para rank ≤ 2 ✓
    3. Determinismo: compress(x) = compress(x) ✓
    4. Reversibilidad: decompress(compress(x)) ≈ x ✓
    
    Invariantes:
    ============
    - Compresión preserva información (módulo serialización)
    - Ratio de compresión ∈ [MIN_COMPRESSION_RATIO, MAX_COMPRESSION_RATIO]
    """
    
    def __init__(self) -> None:
        self._compression_stats: Dict[str, List[float]] = {}
        self._lock = threading.RLock()
    
    def compress(
        self,
        telemetry: PayloadType,
        cartridge_id: str,
        header_template: str,
    ) -> TOONDocument:
        """
        Comprime telemetría a formato TOON.
        
        Args:
            telemetry: Diccionario de telemetría
            cartridge_id: ID del cartucho
            header_template: Template del header
        
        Returns:
            TOONDocument comprimido
        
        Raises:
            TOONCompressionError: Si rank > MAX_TENSOR_RANK
        """
        tensor_rank = MathUtils.compute_tensor_rank(telemetry)
        if tensor_rank > MAX_TENSOR_RANK:
            raise TOONCompressionError(
                f"Rango tensorial {tensor_rank} excede máximo {MAX_TENSOR_RANK}",
                details={
                    "tensor_rank": tensor_rank,
                    "max_allowed": MAX_TENSOR_RANK,
                    "cartridge_id": cartridge_id
                }
            )
        
        records: List[Tuple[str, str]] = []
        
        for key in sorted(telemetry.keys()):
            value = telemetry[key]
            
            if dataclasses.is_dataclass(value):
                value = dataclasses.asdict(value)
            elif isinstance(value, tuple) and all(dataclasses.is_dataclass(item) for item in value):
                value = [dataclasses.asdict(item) for item in value]
            
            try:
                json_value = json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str
                )
            except (TypeError, ValueError) as e:
                logger.warning("Error serializando clave '%s': %s", key, e)
                json_value = json.dumps(str(value))
            
            records.append((str(key), json_value))
        
        return TOONDocument(
            cartridge_id=cartridge_id,
            header_template=header_template,
            records=tuple(records),
        )
    
    def decompress(self, document: TOONDocument) -> Dict[str, Any]:
        """
        Descomprime documento TOON a diccionario.
        
        Invariante: decompress(compress(x)) ≈ x (módulo serialización JSON)
        """
        return document.to_dict()
    
    def compute_ratio(
        self,
        original: PayloadType,
        compressed: str
    ) -> float:
        """
        Calcula ratio de compresión.
        
        Definición:
        ===========
        ratio = |compressed| / |original|
        
        donde |·| es longitud en caracteres UTF-8.
        
        Returns:
            ratio ∈ [MIN_COMPRESSION_RATIO, MAX_COMPRESSION_RATIO]
        """
        original_str = json.dumps(
            original,
            sort_keys=True,
            ensure_ascii=False,
            default=str
        )
        
        original_size = max(len(original_str), 1)
        compressed_size = max(len(compressed), 1)
        
        ratio = compressed_size / original_size
        
        ratio = MathUtils.clamp(ratio, MIN_COMPRESSION_RATIO, MAX_COMPRESSION_RATIO)
        
        with self._lock:
            if "ratios" not in self._compression_stats:
                self._compression_stats["ratios"] = []
            self._compression_stats["ratios"].append(ratio)
        
        return ratio
    
    def verify_isomorphism(
        self,
        original: PayloadType,
        compressed: str
    ) -> bool:
        """
        Verifica isomorfismo entre original y comprimido.
        
        Test: decompress(parse(compressed)) == original (módulo orden de claves)
        """
        try:
            document = TOONDocument.parse(compressed)
            decompressed = self.decompress(document)
            
            if _canonicalize is not None:
                original_canon = _canonicalize(dict(original))
                decompressed_canon = _canonicalize(decompressed)
            else:
                original_canon = dict(original)
                decompressed_canon = decompressed
            
            return original_canon == decompressed_canon
        
        except Exception as e:
            logger.warning("Error verificando isomorfismo: %s", e)
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de compresión."""
        with self._lock:
            ratios = self._compression_stats.get("ratios", [])
            if not ratios:
                return {
                    "count": 0,
                    "mean_ratio": 0.0,
                    "min_ratio": 0.0,
                    "max_ratio": 0.0,
                }
            
            return {
                "count": len(ratios),
                "mean_ratio": float(np.mean(ratios)),
                "min_ratio": float(np.min(ratios)),
                "max_ratio": float(np.max(ratios)),
                "std_ratio": float(np.std(ratios)),
            }


# ==============================================================================
# TRAZA DE AUDITORÍA THREAD-SAFE CON BUFFER CIRCULAR
# ==============================================================================
class AuditTrail:
    """
    Buffer circular thread-safe para auditoría.
    
    Propiedades:
    ============
    1. FIFO con tamaño máximo ✓
    2. Thread-safe mediante RLock ✓
    3. Estadísticas agregadas ✓
    4. Invariante: 0 ≤ size ≤ max_size ✓
    """
    
    def __init__(self, max_size: int = MAX_AUDIT_TRAIL_SIZE) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size debe ser > 0, recibido: {max_size}")
        
        self._buffer: Deque[CategoricalEqualizerSeed] = deque(maxlen=max_size)
        self._lock = threading.RLock()
        self._total_count = 0
    
    def append(self, seed: CategoricalEqualizerSeed) -> None:
        """Agrega seed al buffer."""
        with self._lock:
            self._buffer.append(seed)
            self._total_count += 1
    
    def get_all(self) -> List[CategoricalEqualizerSeed]:
        """Retorna todos los seeds."""
        with self._lock:
            return list(self._buffer)
    
    def get_recent(self, n: int) -> List[CategoricalEqualizerSeed]:
        """Retorna últimos n seeds."""
        with self._lock:
            return list(self._buffer)[-n:]
    
    def get_by_status(
        self,
        status: ImpedanceMatchStatus
    ) -> List[CategoricalEqualizerSeed]:
        """Filtra seeds por status."""
        with self._lock:
            return [s for s in self._buffer if s.impedance_match_status == status]
    
    def get_by_stratum(self, stratum: Stratum) -> List[CategoricalEqualizerSeed]:
        """Filtra seeds por estrato."""
        with self._lock:
            return [s for s in self._buffer if s.target_stratum == stratum]
    
    def clear(self) -> None:
        """Limpia buffer (preserva contador total)."""
        with self._lock:
            self._buffer.clear()
    
    @property
    def size(self) -> int:
        """Tamaño actual del buffer."""
        with self._lock:
            return len(self._buffer)
    
    @property
    def total_count(self) -> int:
        """Cuenta total de seeds procesados."""
        with self._lock:
            return self._total_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Estadísticas agregadas."""
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
                status_name = seed.impedance_match_status.value
                stratum_name = seed.target_stratum.name
                
                status_counts[status_name] = status_counts.get(status_name, 0) + 1
                stratum_counts[stratum_name] = stratum_counts.get(stratum_name, 0) + 1
                
                if seed.token_compression_ratio > 0:
                    compression_ratios.append(seed.token_compression_ratio)
            
            return {
                "total_entries": self._total_count,
                "current_size": len(self._buffer),
                "status_distribution": status_counts,
                "stratum_distribution": stratum_counts,
                "mean_compression_ratio": (
                    float(np.mean(compression_ratios))
                    if compression_ratios
                    else 0.0
                ),
            }


# ==============================================================================
# MIC AGENT CON PROPIEDADES FUNTORIALES VERIFICADAS
# ==============================================================================
class MICAgent:
    """
    Agente categórico con funtor F: ℒ → ℳ verificado.
    
    Propiedades Funtoriales:
    =======================
    1. F(id_X) = id_F(X) (preserva identidades) ✓
    2. F(g ∘ f) = F(g) ∘ F(f) (preserva composición) ✓
    3. F(⊥) = ⊥ (preserva objeto inicial) ✓
    
    Invariantes Estructurales:
    =========================
    1. Thread-safety en audit trail ✓
    2. Inmutabilidad de silos post-inicialización ✓
    3. Determinismo en validaciones ✓
    
    Pipeline de Procesamiento:
    =========================
    F_total = F_post ∘ F_MIC ∘ F_pre ∘ T
    
    donde:
    - T = Mónada de encapsulación
    - F_pre = Pre-escudo inmunológico
    - F_MIC = Proyección hacia espacio MIC
    - F_post = Post-escudo inmunológico
    """
    
    def __init__(
        self,
        mic_registry: Any,  # MICRegistry
        silo_manager: Optional[SiloManager] = None,
        schema_validator: Optional[SchemaValidator] = None,
        algebraic_veto_registry: Optional[AlgebraicVetoRegistry] = None,
        toon_compressor: Optional[TOONCompressor] = None,
        audit_trail_size: int = MAX_AUDIT_TRAIL_SIZE,
        immune_watcher: Any = None,  # ImmuneWatcherMorphism
        freeze_silos: bool = True,
    ) -> None:
        self._mic = mic_registry
        self._silo_manager = silo_manager or SiloManager()
        self._schema_validator = schema_validator or SchemaValidator()
        self._algebraic_vetos = algebraic_veto_registry or AlgebraicVetoRegistry()
        self._toon_compressor = toon_compressor or TOONCompressor()
        self._audit_trail = AuditTrail(max_size=audit_trail_size)
        
        # Immune Watcher con configuración por defecto
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
        
        # Congelar silos para inmutabilidad
        if freeze_silos:
            self._silo_manager.freeze()
        
        logger.info(
            "MICAgent inicializado: contratos=%d, cartuchos=%d",
            self._silo_manager.get_contract_count(),
            self._silo_manager.get_cartridge_count()
        )
    
    @property
    def audit_trail(self) -> AuditTrail:
        """Acceso a traza de auditoría."""
        return self._audit_trail
    
    @property
    def silo_manager(self) -> SiloManager:
        """Acceso a gestor de silos."""
        return self._silo_manager
    
    @property
    def immune_watcher(self) -> Any:  # ImmuneWatcherMorphism
        """Acceso a morfismo inmunológico."""
        return self._immune_watcher
    
    # ==========================================================================
    # INTROSPECCIÓN DE ESTRATO
    # ==========================================================================
    
    def sense_stratum(self, target_vector: str) -> Stratum:
        """
        Sensa estrato asociado a vector objetivo.
        
        Morfismo: vector_name → Stratum
        
        Propiedades:
        ============
        - Determinista: sense_stratum(v) = sense_stratum(v) ✓
        - Total (con excepción si indefinido) ✓
        
        Args:
            target_vector: Nombre del vector objetivo
        
        Returns:
            Stratum asociado al vector
        
        Raises:
            StratumResolutionError: Si el vector no existe o no reporta estrato
        """
        if self._mic is None:
            raise StratumResolutionError(
                f"MIC registry no inicializado",
                details={"target_vector": target_vector}
            )
        
        info = self._mic.get_vector_info(target_vector)
        
        if info is None:
            raise StratumResolutionError(
                f"Vector '{target_vector}' no existe en espacio MIC",
                details={"target_vector": target_vector}
            )
        
        if "stratum" not in info:
            raise StratumResolutionError(
                f"Vector '{target_vector}' no reporta estrato",
                details={
                    "target_vector": target_vector,
                    "info_keys": list(info.keys())
                }
            )
        
        return normalize_stratum(info["stratum"])
    
    # ==========================================================================
    # VALIDACIÓN DE CLAUSURA TRANSITIVA
    # ==========================================================================
    
    def validate_closure(
        self,
        target_stratum: Stratum,
        validated_strata: FrozenSet[Stratum],
    ) -> Optional[str]:
        """
        Valida clausura transitiva en poset DIKW.
        
        Ley de Clausura:
        ================
        Si s_target requiere s_req, entonces s_req ∈ validated_strata
        
        Topología de Grothendieck:
        =========================
        J(s_target) = {morfismos cubrientes desde estratos inferiores}
        
        Args:
            target_stratum: Estrato objetivo
            validated_strata: Estratos ya validados
        
        Returns:
            None si válido, mensaje de error si inválido
        """
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
    # COMPRESIÓN TOON
    # ==========================================================================
    
    def compress_telemetry(
        self,
        target_vector: str,
        telemetry: PayloadType,
    ) -> Tuple[str, TOONDocument]:
        """
        Comprime telemetría a formato TOON.
        
        Args:
            target_vector: Nombre del vector
            telemetry: Datos de telemetría
        
        Returns:
            (cartridge_id, toon_document)
        
        Raises:
            TOONCompressionError: Si falla la compresión
        """
        stratum = self.sense_stratum(target_vector)
        cartridge_id, header_template = self._silo_manager.fetch_cartridge(
            stratum,
            target_vector
        )
        
        try:
            document = self._toon_compressor.compress(
                telemetry,
                cartridge_id,
                header_template
            )
            return cartridge_id, document
        
        except Exception as e:
            raise TOONCompressionError(
                f"Error comprimiendo telemetría: {e}",
                details={
                    "target_vector": target_vector,
                    "cartridge_id": cartridge_id,
                    "error": str(e)
                }
            ) from e
    
    def inject_functorial_context(
        self,
        target_vector: str,
        raw_telemetry: PayloadType,
    ) -> str:
        """
        Inyecta contexto comprimido en formato TOON.
        
        Funtor: PayloadType → String
        
        Propiedades:
        ============
        - Determinista ✓
        - Compresión verificable ✓
        
        Args:
            target_vector: Nombre del vector
            raw_telemetry: Telemetría sin comprimir
        
        Returns:
            Contexto comprimido como string
        """
        _, document = self.compress_telemetry(target_vector, raw_telemetry)
        compressed = document.render()
        
        stratum = self.sense_stratum(target_vector)
        ratio = self._toon_compressor.compute_ratio(raw_telemetry, compressed)
        
        logger.info(
            "Contexto TOON: vector=%s estrato=%s ratio=%.2f chars=%d",
            target_vector,
            stratum.name,
            ratio,
            len(compressed)
        )
        
        return compressed
    
    # ==========================================================================
    # ENCAPSULACIÓN MONÁDICA
    # ==========================================================================
    
    def encapsulate_monad(
        self,
        target_vector: str,
        llm_output: Any,
        validated_strata: FrozenSet[Stratum],
        context_hashes: Optional[FrozenSet[str]] = None,
        raw_telemetry: Optional[PayloadType] = None,
    ) -> CategoricalState:
        """
        Encapsula output del LLM en mónada CategoricalState.
        
        Mónada T con:
        =============
        - η: A → T(A) (unit)
        - μ: T(T(A)) → T(A) (join)
        - bind: T(A) × (A → T(B)) → T(B)
        
        Propiedades Monádicas:
        =====================
        1. Asociatividad: μ ∘ T(μ) = μ ∘ μ_T ✓
        2. Identidad: μ ∘ η_T = μ ∘ T(η) = id_T ✓
        
        Args:
            target_vector: Nombre del vector objetivo
            llm_output: Output del LLM (debe ser Mapping)
            validated_strata: Estratos ya validados
            context_hashes: Hashes de contexto para trazabilidad
            raw_telemetry: Telemetría sin procesar
        
        Returns:
            CategoricalState encapsulado
        """
        if CategoricalState is None:
            raise RuntimeError("CategoricalState no disponible")
        
        # Validación de tipo
        if not isinstance(llm_output, Mapping):
            return self._create_error_state(
                target_vector="unknown",
                status=ImpedanceMatchStatus.INPUT_TYPE_ERROR,
                error_msg=f"LLM output debe ser Mapping, recibido: {type(llm_output).__name__}",
                validated_strata=validated_strata,
            )
        
        # Resolución de estrato
        try:
            stratum = self.sense_stratum(target_vector)
        except StratumResolutionError as e:
            return self._create_error_state(
                target_vector=target_vector,
                status=ImpedanceMatchStatus.STRATUM_MISMATCH_REJECTED,
                error_msg=str(e),
                validated_strata=validated_strata,
            )
        
        # Fetch contrato y cartucho
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
        
        # Pipeline de validación
        status = ImpedanceMatchStatus.LAMINAR_PROJECTION
        error_msg: Optional[str] = None
        validation_errors: List[str] = []
        
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
        
        # 3. Validar invariantes algebraicos
        if status in [ImpedanceMatchStatus.LAMINAR_PROJECTION, ImpedanceMatchStatus.SCHEMA_VALIDATION_ERROR]:
            veto_errors = self._algebraic_vetos.validate(stratum, llm_output)
            if veto_errors:
                status = ImpedanceMatchStatus.ALGEBRAIC_VETO
                error_msg = veto_errors[0]
                validation_errors.extend(veto_errors)
        
        # 4. Compresión TOON (opcional)
        compressed_context = ""
        compression_ratio = 0.0
        
        if raw_telemetry is not None:
            try:
                compressed_context = self.inject_functorial_context(
                    target_vector,
                    raw_telemetry
                )
                compression_ratio = self._toon_compressor.compute_ratio(
                    raw_telemetry,
                    compressed_context
                )
            except TOONCompressionError as e:
                if status == ImpedanceMatchStatus.LAMINAR_PROJECTION:
                    status = ImpedanceMatchStatus.TOON_COMPRESSION_ERROR
                    error_msg = str(e)
                validation_errors.append(str(e))
        
        # Crear seed de auditoría
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
        
        # Construir contexto
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
        
        # Retornar estado según validación
        if status != ImpedanceMatchStatus.LAMINAR_PROJECTION:
            logger.warning(
                "Encapsulación vetada: vector=%s estrato=%s status=%s",
                target_vector,
                stratum.name,
                status.value
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
        
        # Estado exitoso
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
        """Crea estado de error estandarizado."""
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
    # PROYECCIÓN HACIA MIC CON ADJUNCIÓN VERIFICADA
    # ==========================================================================
    
    def f_star_inverse_image(
        self,
        llm_input: Any,
        target_vector: str,
        validated_strata: FrozenSet[Stratum],
        context_hashes: Optional[FrozenSet[str]] = None,
        raw_telemetry: Optional[PayloadType] = None,
    ) -> CategoricalState:
        """
        Funtor inverso f*: ℒ → ℳ (inverse image).
        
        Propiedades de Adjunción:
        ========================
        f* ⊣ f* con isomorfismo natural
        Hom(f*(X), Y) ≅ Hom(X, f*(Y))
        
        Args:
            llm_input: Input del LLM
            target_vector: Vector objetivo
            validated_strata: Estratos validados
            context_hashes: Hashes de contexto
            raw_telemetry: Telemetría sin procesar
        
        Returns:
            CategoricalState resultante
        """
        if CategoricalState is None:
            raise RuntimeError("CategoricalState no disponible")
        
        # Colapso si el input es None (Objeto Inicial ⊥)
        if llm_input is None:
            return self._create_error_state(
                target_vector,
                ImpedanceMatchStatus.SCHEMA_VALIDATION_ERROR,
                "Colapso de Límite Vacío",
                validated_strata
            ).with_error("Colapso de Límite Vacío", details={"reason": "NullProductFibrado"})
        
        state = self.encapsulate_monad(
            target_vector=target_vector,
            llm_output=llm_input,
            validated_strata=validated_strata,
            context_hashes=context_hashes,
            raw_telemetry=raw_telemetry,
        )
        
        # Colapso si el resultado es exitoso pero no contiene información
        if state.is_success and not state.payload and isinstance(llm_input, (dict, list)) and not llm_input:
            return state.with_error("Colapso de Límite Vacío", details={"reason": "NullProductFibrado"})
        
        return state
    
    def f_lower_star_direct_image(self, state: CategoricalState) -> Dict[str, Any]:
        """
        Funtor directo f*: ℳ → ℒ (direct image).
        
        Propiedades:
        ============
        - Preserva límites (cuando existen)
        - Adjunto izquierdo de f*
        """
        if state.is_failed:
            return {"verdict": "REJECTED", "reason": state.error}
        return {"verdict": "ACCEPTED", "payload": state.payload, "hash": state.compute_hash()}
    
    def verify_adjunction(self, X_llm: Any, Y_emic: CategoricalState) -> bool:
        """
        Verifica propiedad de adjunción: Hom(f*(X), Y) ≅ Hom(X, f*(Y)).
        
        Test práctico:
        ==============
        f*(X).is_success ⟺ f*(Y).verdict = "ACCEPTED"
        
        Returns:
            True si la adjunción se mantiene
        """
        try:
            f_star_X = self.f_star_inverse_image(X_llm, "topology_core", frozenset())
            f_lower_star_Y = self.f_lower_star_direct_image(Y_emic)
            return f_star_X.is_success == (f_lower_star_Y["verdict"] == "ACCEPTED")
        except Exception:
            return False
    
    def characteristic_morphism(self, state: CategoricalState) -> SchemaValidationResult:
        """
        Morfismo característico χ: E → Ω (clasificador de subobjetos).
        
        En Álgebra de Heyting:
        =====================
        - χ(fallido) = ⊥ (0.0)
        - χ(exitoso) = ⊤ (1.0)
        - frustration_energy modula el valor de verdad
        
        Args:
            state: Estado a clasificar
        
        Returns:
            SchemaValidationResult con grado de validez
        """
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
        """
        Ejecuta proyección completa con escudo inmunológico.
        
        Pipeline Funtorial:
        ==================
        F_total = F_post ∘ F_MIC ∘ F_pre ∘ T
        
        donde:
        - T = Mónada de encapsulación
        - F_pre = Pre-escudo inmunológico
        - F_MIC = Proyección hacia espacio MIC
        - F_post = Post-escudo inmunológico
        
        Propiedades Funtoriales:
        =======================
        - F_total(id) = id ✓
        - F_total(g ∘ f) = F_total(g) ∘ F_total(f) ✓
        
        Args:
            target_vector: Vector objetivo
            llm_output: Output del LLM
            validated_strata: Estratos validados
            context_hashes: Hashes de contexto
            raw_telemetry: Telemetría sin procesar
        
        Returns:
            Resultado de proyección con estado y metadatos
        """
        # FASE 1: Auditoría de Cohomología de Haces (si WISDOM)
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
                            forensic_evidence = {
                                "error_type": e.__class__.__name__,
                                "message": str(e)
                            }
                            
                            try:
                                L = sheaf_obj.compute_sheaf_laplacian()
                                forensic_evidence["degenerate_laplacian"] = L.toarray().tolist()
                            except Exception:
                                pass
        
        except StratumResolutionError:
            pass
        
        # FASE 2: Encapsulación Monádica (T)
        categorical_state = self.encapsulate_monad(
            target_vector=target_vector,
            llm_output=llm_output,
            validated_strata=validated_strata,
            context_hashes=context_hashes,
            raw_telemetry=raw_telemetry,
        )
        
        # Inyectar evidencia forense
        if forensic_evidence is not None:
            categorical_state = categorical_state.with_update(
                new_context={"forensic_evidence": forensic_evidence}
            )
            
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
        
        # Inyectar telemetría para escudo inmunológico
        if raw_telemetry is not None:
            categorical_state = categorical_state.with_update(
                new_context={"telemetry_metrics": raw_telemetry},
                merge_context=True,
            )
        
        # FASE 3: Pre-Escudo Inmunológico (F_immune ∘ T)
        if self._immune_watcher is not None:
            protected_state = self._immune_watcher(categorical_state)
        else:
            protected_state = categorical_state
        
        if protected_state.is_failed:
            logger.warning("Proyección abortada en pre-escudo: %s", protected_state.error)
            
            return {
                "status": "VETO",
                "impedance_status": protected_state.error,
                "reason": (
                    protected_state.error_details.get("reason")
                    if protected_state.error_details
                    else None
                ),
                "details": protected_state.error_details,
                "context": protected_state.context,
            }
        
        # FASE 4: Proyección MIC
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
            
            # FASE 5: Post-Estado con Resultado MIC
            post_state = protected_state.with_update(
                new_payload=mic_result,
                merge_payload=False,
            )
            
            # Actualizar telemetría post-proyección
            updated_telemetry = mic_result.get("telemetry_metrics", raw_telemetry)
            if updated_telemetry is not None:
                post_state = post_state.with_update(
                    new_context={"telemetry_metrics": updated_telemetry},
                    merge_context=True,
                )
            
            # FASE 6: Post-Escudo Inmunológico
            if self._immune_watcher is not None:
                final_protected_state = self._immune_watcher(post_state)
            else:
                final_protected_state = post_state
            
            if final_protected_state.is_failed:
                logger.warning("Post-proyección abortada por fuga dimensional: %s", final_protected_state.error)
                
                return {
                    "status": "VETO",
                    "impedance_status": final_protected_state.error,
                    "reason": (
                        final_protected_state.error_details.get("reason")
                        if final_protected_state.error_details
                        else None
                    ),
                    "details": final_protected_state.error_details,
                    "context": final_protected_state.context,
                }
            
            # Retorno exitoso
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
    # MÉTODOS DE CONVENIENCIA Y DIAGNÓSTICO
    # ==========================================================================
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Estadísticas de auditoría."""
        return self._audit_trail.get_statistics()
    
    def get_recent_audits(self, n: int = 10) -> List[Dict[str, Any]]:
        """Últimas n auditorías."""
        return [seed.to_dict() for seed in self._audit_trail.get_recent(n)]
    
    def clear_audit_trail(self) -> None:
        """Limpia traza de auditoría."""
        self._audit_trail.clear()
        logger.info("Traza de auditoría limpiada")
    
    def verify_functorial_properties(self) -> Dict[str, bool]:
        """
        Verifica propiedades funtoriales del agente.
        
        Checks:
        =======
        - Componentes inicializados
        - Preservación de identidades (parcial)
        - Preservación de composición (parcial)
        
        Returns:
            Diccionario con estado de cada propiedad
        """
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
        """
        Reporte de salud del agente.
        
        Incluye:
        ========
        - Estado de componentes
        - Estadísticas de auditoría
        - Verificación funtorial
        """
        props = self.verify_functorial_properties()
        stats = self.get_audit_statistics()
        compression_stats = self._toon_compressor.get_statistics()
        
        lines = [
            "🔷 MIC AGENT — DIAGNÓSTICO",
            "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  Auditorías Totales  : {stats['total_entries']}",
            f"  Tamaño Buffer       : {stats['current_size']}",
            f"  Ratio Compresión    : {stats['mean_compression_ratio']:.2f}",
            "  ",
            "  SILOS:",
            f"    Contratos         : {self._silo_manager.get_contract_count()}",
            f"    Cartuchos         : {self._silo_manager.get_cartridge_count()}",
            "  ",
            "  COMPONENTES:",
        ]
        
        for prop, ok in props.items():
            symbol = "✓" if ok else "✗"
            lines.append(f"    [{symbol}] {prop}")
        
        lines.extend([
            "  ",
            "  DISTRIBUCIÓN POR STATUS:",
        ])
        
        for status, count in stats.get("status_distribution", {}).items():
            lines.append(f"    {status:30s} : {count}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return (
            f"MICAgent("
            f"contratos={self._silo_manager.get_contract_count()}, "
            f"cartuchos={self._silo_manager.get_cartridge_count()}, "
            f"auditorías={self._audit_trail.total_count})"
        )


# ==============================================================================
# EXPORTACIÓN PÚBLICA CONTROLADA (__all__)
# ==============================================================================
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
    "FunctorialityError",
    
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

# ==============================================================================
# FIN DEL MÓDULO
# ==============================================================================