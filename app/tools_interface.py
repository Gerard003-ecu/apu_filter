"""
Módulo: Matriz de Interacción Central (MIC) - El Espacio Vectorial de Intenciones
================================================================================

Este componente implementa el núcleo de gobernanza y despacho del sistema, actuando
como un **Espacio Vectorial Jerarquizado con Filtración Topológica**.

Fundamentos Matemáticos Formalizados:
─────────────────────────────────────

1. Álgebra Lineal de la Acción (Base Canónica):
   - Cada servicio s_i es un vector base e_i ∈ ℝⁿ
   - La MIC define el espacio V = span{e₁, e₂, ..., eₙ}
   - Ortogonalidad Funcional: ⟨eᵢ, eⱼ⟩ = δᵢⱼ (Kronecker)

2. Filtración de Subespacios (Pirámide DIKW):
   V_PHYSICS ⊂ V_TACTICS ⊂ V_STRATEGY ⊂ V_WISDOM
   
   Con la restricción de Clausura Transitiva:
   π_k(v) válido ⟺ ∀j > k: validated(V_j) = True

3. Persistencia Homológica:
   - H₀: Componentes conexos (clusters de issues)
   - H₁: Ciclos (dependencias circulares)
   - Entropía de persistencia para cuantificar complejidad

4. Gatekeeper como Proyección Condicional:
   G(v, ctx) = v si valid(ctx) else 0⃗
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum
from functools import lru_cache, wraps
from pathlib import Path
from typing import (
    Any, Callable, Dict, Final, FrozenSet, Generic, List, 
    Optional, Protocol, Sequence, Set, Tuple, Type, TypeVar, Union,
    runtime_checkable, Iterator
)

import numpy as np
import pandas as pd

# =============================================================================
# IMPORTACIONES CON FALLBACK ROBUSTO Y LOGGING
# =============================================================================

logger = logging.getLogger("MIC")

def _safe_import(module_path: str, class_name: str) -> Optional[Type]:
    """Importación segura con logging de diagnóstico."""
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name, None)
    except ImportError as e:
        logger.debug(f"Optional import failed: {module_path}.{class_name} - {e}")
        return None

CSVCleaner = _safe_import("scripts.clean_csv", "CSVCleaner")
APUFileDiagnostic = _safe_import("scripts.diagnose_apus_file", "APUFileDiagnostic")
InsumosFileDiagnostic = _safe_import("scripts.diagnose_insumos_file", "InsumosFileDiagnostic")
PresupuestoFileDiagnostic = _safe_import("scripts.diagnose_presupuesto_file", "PresupuestoFileDiagnostic")
FinancialConfig = _safe_import(".financial_engine", "FinancialConfig")
FinancialEngine = _safe_import(".financial_engine", "FinancialEngine")

# Fallback para Stratum
try:
    from .schemas import Stratum
except ImportError:
    class Stratum(IntEnum):
        """
        Representación jerárquica del modelo DIKW.
        
        La ordenación numérica (WISDOM=0 < PHYSICS=3) refleja la
        precedencia conceptual, mientras que la validación sigue
        el orden inverso (PHYSICS debe validarse primero).
        """
        WISDOM = 0      # Síntesis estratégica (vértice)
        STRATEGY = 1    # Planificación financiera  
        TACTICS = 2     # Estructura operativa
        PHYSICS = 3     # Datos físicos (base de la pirámide)
        
        @classmethod
        def base_stratum(cls) -> 'Stratum':
            """Retorna el estrato base de la filtración."""
            return cls.PHYSICS
        
        @classmethod
        def apex_stratum(cls) -> 'Stratum':
            """Retorna el estrato superior de la filtración."""
            return cls.WISDOM
        
        def requires(self) -> FrozenSet['Stratum']:
            """Retorna los estratos prerrequisito (clausura transitiva)."""
            return frozenset(s for s in Stratum if s.value > self.value)

# Vectores mock para testing standalone
try:
    from app.adapters.mic_vectors import (
        vector_stabilize_flux,
        vector_parse_raw_structure,
        vector_structure_logic,
        vector_lateral_pivot,
        vector_audit_homological_fusion
    )
except ImportError:
    def _mock_vector(**kwargs) -> Dict[str, Any]:
        return {"success": True, "mock": True, **kwargs}
    
    vector_stabilize_flux = _mock_vector
    vector_parse_raw_structure = _mock_vector
    vector_structure_logic = _mock_vector
    vector_lateral_pivot = _mock_vector
    vector_audit_homological_fusion = _mock_vector


# =============================================================================
# CONSTANTES MATEMÁTICAS Y DE CONFIGURACIÓN
# =============================================================================

# Límites de archivos
MAX_FILE_SIZE_BYTES: Final[int] = 100 * 1024 * 1024  # 100 MB

# Codificaciones soportadas (conjunto canónico)
SUPPORTED_ENCODINGS: Final[FrozenSet[str]] = frozenset({
    "utf-8", "utf-8-sig", "latin-1", "iso-8859-1",
    "cp1252", "ascii", "utf-16", "utf-16-le", "utf-16-be",
})

_ENCODING_ALIASES: Final[Dict[str, str]] = {
    "utf8": "utf-8", "latin1": "latin-1", 
    "iso88591": "iso-8859-1", "cp65001": "utf-8",
}

VALID_DELIMITERS: Final[FrozenSet[str]] = frozenset({",", ";", "\t", "|", ":"})
VALID_EXTENSIONS: Final[FrozenSet[str]] = frozenset({".csv", ".txt", ".tsv"})

# Constantes numéricas con significado matemático
_EPSILON: Final[float] = 1e-10  # Tolerancia para comparaciones flotantes
_PHI: Final[float] = (1 + math.sqrt(5)) / 2  # Proporción áurea (para heurísticas)
_DEFAULT_RANDOM_SEED: Final[int] = 42
_MAX_SAMPLE_ROWS: Final[int] = 1000
_PERSISTENCE_THRESHOLD: Final[float] = 0.01  # Umbral de significancia topológica
_CACHE_TTL_SECONDS: Final[float] = 300.0  # TTL del cache (5 minutos)


# =============================================================================
# TIPOS GENÉRICOS Y PROTOCOLOS
# =============================================================================

T = TypeVar('T')
R = TypeVar('R')


@runtime_checkable
class TelemetryContextProtocol(Protocol):
    """Protocolo para contextos de telemetría observacional."""
    def get_business_report(self) -> Dict[str, Any]: ...


@runtime_checkable
class DiagnosticProtocol(Protocol):
    """Protocolo para clases diagnósticas de archivos."""
    def diagnose(self) -> None: ...
    def to_dict(self) -> Dict[str, Any]: ...


@runtime_checkable
class VectorHandler(Protocol):
    """Protocolo para handlers de vectores en la MIC."""
    def __call__(self, **kwargs: Any) -> Dict[str, Any]: ...


# =============================================================================
# JERARQUÍA DE EXCEPCIONES (Álgebra de Errores)
# =============================================================================

class MICException(Exception):
    """
    Clase base para excepciones de la MIC.
    
    Implementa un patrón de "error estructurado" donde cada excepción
    porta su contexto algebraico (detalles, categoría, severidad).
    """
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        category: str = "mic_error"
    ) -> None:
        super().__init__(message)
        self.details: Dict[str, Any] = details or {}
        self.category: str = category
        self.timestamp: float = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa la excepción a diccionario para respuestas API."""
        return {
            "error": str(self),
            "error_type": type(self).__name__,
            "error_category": self.category,
            "error_details": self.details,
            "timestamp": self.timestamp
        }


class FileNotFoundDiagnosticError(MICException):
    """Archivo no encontrado durante diagnóstico."""
    def __init__(self, path: Union[str, Path], **kwargs):
        super().__init__(
            f"File not found: {path}",
            details={"path": str(path), **kwargs},
            category="validation"
        )


class UnsupportedFileTypeError(MICException):
    """Tipo de archivo no soportado por el sistema."""
    def __init__(self, file_type: str, available: List[str]):
        super().__init__(
            f"Unsupported file type: {file_type}",
            details={"file_type": file_type, "available_types": available},
            category="validation"
        )


class FileValidationError(MICException):
    """Error de validación de archivo (extensión, tamaño, etc.)."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, details=kwargs, category="validation")


class CleaningError(MICException):
    """Error durante proceso de limpieza de archivos."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, details=kwargs, category="cleaning")


class MICHierarchyViolationError(MICException):
    """
    Violación de la Clausura Transitiva en la Filtración de Estratos.
    
    Ocurre cuando se intenta proyectar un vector a un estrato k
    sin haber validado todos los estratos base j > k.
    """
    
    def __init__(
        self,
        target_stratum: Stratum,
        missing_strata: Set[Stratum],
        validated_strata: Set[Stratum]
    ) -> None:
        missing_names = sorted([s.name for s in missing_strata], key=lambda n: Stratum[n].value, reverse=True)
        validated_names = sorted([s.name for s in validated_strata], key=lambda n: Stratum[n].value, reverse=True)
        
        message = (
            f"Clausura Transitiva Violada: No se puede proyectar a {target_stratum.name}. "
            f"Estratos faltantes: {', '.join(missing_names)}. "
            f"Valide primero las operaciones de {Stratum.base_stratum().name}."
        )
        
        super().__init__(
            message,
            details={
                "target_stratum": target_stratum.name,
                "missing_strata": missing_names,
                "validated_strata": validated_names
            },
            category="hierarchy_violation"
        )
        
        self.target_stratum = target_stratum
        self.missing_strata = missing_strata
        self.validated_strata = validated_strata


# =============================================================================
# ENUMERACIONES
# =============================================================================

class FileType(str, IntEnum if False else type('FileTypeBase', (str,), {})):
    """
    Tipos de archivo soportados para diagnóstico.
    
    Nota: Usamos herencia dual str+Enum para serialización JSON nativa.
    """
    APUS = "apus"
    INSUMOS = "insumos"
    PRESUPUESTO = "presupuesto"

    @classmethod
    def values(cls) -> List[str]:
        """Retorna lista de valores válidos."""
        return [member.value for member in cls]

    @classmethod
    def from_string(cls, value: str) -> 'FileType':
        """Parsea string a FileType con normalización robusta."""
        if not isinstance(value, str):
            raise TypeError(f"Expected string, got {type(value).__name__}")

        normalized = value.strip().lower()

        for member in cls:
            if member.value == normalized:
                return member

        raise ValueError(
            f"'{value}' no es válido. Opciones: {', '.join(cls.values())}"
        )


# Reimplementar FileType como Enum real
from enum import Enum

class FileType(str, Enum):
    """Tipos de archivo soportados para diagnóstico."""
    APUS = "apus"
    INSUMOS = "insumos"
    PRESUPUESTO = "presupuesto"

    @classmethod
    def values(cls) -> List[str]:
        return [member.value for member in cls]

    @classmethod
    def from_string(cls, value: str) -> 'FileType':
        if not isinstance(value, str):
            raise TypeError(f"Expected string, got {type(value).__name__}")
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"'{value}' no es válido. Opciones: {', '.join(cls.values())}")


# =============================================================================
# ESTRUCTURAS TOPOLÓGICAS INMUTABLES
# =============================================================================

@dataclass(frozen=True, slots=True)
class PersistenceInterval:
    """
    Intervalo de persistencia [birth, death) en el diagrama de persistencia.
    
    Representa el "tiempo de vida" de una característica topológica (componente
    conexo, ciclo, cavidad) durante la filtración del complejo simplicial.
    
    Invariantes:
        - birth ≥ 0
        - death ≥ birth (death = ∞ indica característica esencial)
        - dimension ∈ {0, 1, 2, ...}
    
    Attributes:
        birth: Tiempo de nacimiento de la característica
        death: Tiempo de muerte (float('inf') si es esencial)
        dimension: Dimensión homológica (0=componentes, 1=ciclos, 2=cavidades)
    """
    birth: float
    death: float
    dimension: int = 0

    def __post_init__(self) -> None:
        # Validación de invariantes (usando object.__setattr__ solo si es necesario)
        if self.birth < 0:
            raise ValueError(f"birth debe ser ≥ 0, recibido: {self.birth}")
        if self.death < self.birth:
            raise ValueError(f"death ({self.death}) debe ser ≥ birth ({self.birth})")
        if self.dimension < 0:
            raise ValueError(f"dimension debe ser ≥ 0, recibido: {self.dimension}")

    @property
    def persistence(self) -> float:
        """Tiempo de vida de la característica: death - birth."""
        return self.death - self.birth if not self.is_essential else float('inf')

    @property
    def is_essential(self) -> bool:
        """True si la característica nunca muere (death = ∞)."""
        return math.isinf(self.death)
    
    @property
    def midpoint(self) -> float:
        """Punto medio del intervalo (útil para visualización)."""
        if self.is_essential:
            return self.birth
        return (self.birth + self.death) / 2

    def __lt__(self, other: 'PersistenceInterval') -> bool:
        """Ordenamiento por persistencia descendente (mayor persistencia primero)."""
        if not isinstance(other, PersistenceInterval):
            return NotImplemented
        # Intervalos esenciales van primero
        if self.is_essential != other.is_essential:
            return self.is_essential
        return self.persistence > other.persistence
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa el intervalo a diccionario."""
        return {
            "birth": self.birth,
            "death": self.death if not self.is_essential else "inf",
            "persistence": self.persistence if not self.is_essential else "inf",
            "dimension": self.dimension,
            "is_essential": self.is_essential
        }


@dataclass(frozen=True, slots=True)
class BettiNumbers:
    """
    Números de Betti de un espacio topológico.
    
    Los números de Betti β_k cuentan el número de "agujeros" k-dimensionales:
        - β₀: Componentes conexos
        - β₁: Ciclos independientes (agujeros 1D)
        - β₂: Cavidades (agujeros 2D)
    
    La Característica de Euler χ = β₀ - β₁ + β₂ es un invariante topológico.
    """
    beta_0: int  # Componentes conexos
    beta_1: int  # Ciclos independientes
    beta_2: int  # Cavidades
    
    def __post_init__(self) -> None:
        for name, val in [("beta_0", self.beta_0), ("beta_1", self.beta_1), ("beta_2", self.beta_2)]:
            if not isinstance(val, int) or val < 0:
                raise ValueError(f"{name} debe ser entero no negativo, recibido: {val}")
    
    @property
    def euler_characteristic(self) -> int:
        """Característica de Euler χ = β₀ - β₁ + β₂."""
        return self.beta_0 - self.beta_1 + self.beta_2
    
    @property
    def total_rank(self) -> int:
        """Rango total de homología Σβ_k."""
        return self.beta_0 + self.beta_1 + self.beta_2
    
    @property
    def is_connected(self) -> bool:
        """True si el espacio es conexo (β₀ = 1)."""
        return self.beta_0 == 1
    
    @property
    def has_cycles(self) -> bool:
        """True si existen ciclos no triviales (β₁ > 0)."""
        return self.beta_1 > 0
    
    @classmethod
    def zero(cls) -> 'BettiNumbers':
        """Crea números de Betti nulos (espacio vacío)."""
        return cls(beta_0=0, beta_1=0, beta_2=0)
    
    @classmethod
    def point(cls) -> 'BettiNumbers':
        """Números de Betti de un punto."""
        return cls(beta_0=1, beta_1=0, beta_2=0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "beta_0": self.beta_0,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "betti_numbers": [self.beta_0, self.beta_1, self.beta_2],
            "euler_characteristic": self.euler_characteristic,
            "total_rank": self.total_rank,
            "is_connected": self.is_connected,
            "has_cycles": self.has_cycles
        }


@dataclass(frozen=True, slots=True)
class TopologicalSummary:
    """
    Resumen completo de características topológicas de un dataset.
    
    Encapsula números de Betti, entropía estructural y métricas derivadas
    para proporcionar una "firma topológica" del espacio de datos.
    """
    betti: BettiNumbers
    structural_entropy: float
    persistence_entropy: float
    intrinsic_dimension: int = 1

    @classmethod
    def empty(cls) -> 'TopologicalSummary':
        """Crea un resumen vacío para casos de error o datos vacíos."""
        return cls(
            betti=BettiNumbers.zero(),
            structural_entropy=0.0,
            persistence_entropy=0.0,
            intrinsic_dimension=0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            **self.betti.to_dict(),
            "structural_entropy": round(self.structural_entropy, 6),
            "persistence_entropy": round(self.persistence_entropy, 6),
            "intrinsic_dimension": self.intrinsic_dimension
        }


# =============================================================================
# VECTOR DE INTENCIÓN (FORMALIZACIÓN ALGEBRAICA)
# =============================================================================

@dataclass(frozen=True, slots=True)
class IntentVector:
    """
    Vector de intención inmutable proyectado sobre la MIC.
    
    Formalización: Un IntentVector es un elemento del espacio vectorial V
    definido sobre el anillo de servicios S, con coordenadas en el contexto C.
    
    v = (service_name, payload, context) ∈ S × P × C
    
    La norma ||v|| representa la "magnitud" de la intención, útil para
    priorización en colas de despacho.
    """
    service_name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        if not self.service_name or not self.service_name.strip():
            raise ValueError("service_name no puede estar vacío")
    
    @property
    def payload_hash(self) -> str:
        """Hash del payload para cache y deduplicación."""
        content = str(sorted(self.payload.items()))
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    @property
    def norm(self) -> float:
        """
        Norma del vector de intención.
        
        Heurística: ||v|| = √(|payload| + |context|)
        """
        return math.sqrt(len(self.payload) + len(self.context))
    
    def with_context(self, **additional_context) -> 'IntentVector':
        """Crea un nuevo vector con contexto extendido (inmutabilidad)."""
        new_context = {**self.context, **additional_context}
        return IntentVector(
            service_name=self.service_name,
            payload=self.payload,
            context=new_context
        )


# =============================================================================
# CACHE CON TTL (Thread-Safe)
# =============================================================================

@dataclass
class CacheEntry(Generic[T]):
    """Entrada de cache con timestamp para TTL."""
    value: T
    timestamp: float
    
    def is_expired(self, ttl_seconds: float) -> bool:
        """Verifica si la entrada ha expirado."""
        return (time.time() - self.timestamp) > ttl_seconds


class TTLCache(Generic[T]):
    """
    Cache thread-safe con Time-To-Live.
    
    Implementa un patrón de cache con expiración automática para
    resultados de operaciones costosas (análisis topológico, etc.).
    """
    
    def __init__(self, ttl_seconds: float = _CACHE_TTL_SECONDS, max_size: int = 100):
        self._data: Dict[str, CacheEntry[T]] = {}
        self._lock = threading.RLock()
        self._ttl = ttl_seconds
        self._max_size = max_size
    
    def get(self, key: str) -> Optional[T]:
        """Obtiene valor del cache si existe y no ha expirado."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if entry.is_expired(self._ttl):
                del self._data[key]
                return None
            return entry.value
    
    def set(self, key: str, value: T) -> None:
        """Almacena valor en cache."""
        with self._lock:
            # Evicción LRU simple si excede tamaño máximo
            if len(self._data) >= self._max_size:
                oldest_key = min(self._data.keys(), key=lambda k: self._data[k].timestamp)
                del self._data[oldest_key]
            
            self._data[key] = CacheEntry(value=value, timestamp=time.time())
    
    def clear(self) -> int:
        """Limpia el cache y retorna número de entradas eliminadas."""
        with self._lock:
            count = len(self._data)
            self._data.clear()
            return count
    
    def prune_expired(self) -> int:
        """Elimina entradas expiradas y retorna cantidad eliminada."""
        with self._lock:
            expired_keys = [k for k, v in self._data.items() if v.is_expired(self._ttl)]
            for key in expired_keys:
                del self._data[key]
            return len(expired_keys)
    
    @property
    def size(self) -> int:
        """Número de entradas en el cache."""
        with self._lock:
            return len(self._data)


# =============================================================================
# MATRIZ DE INTERACCIÓN CENTRAL (MIC) - NÚCLEO DEL SISTEMA
# =============================================================================

class MICRegistry:
    """
    Matriz de Interacción Central (MIC).
    
    Implementa un Espacio Vectorial Jerárquico donde:
    
    1. **Base Canónica**: Cada servicio registrado es un vector base eᵢ
       en el espacio de acción ℝⁿ.
    
    2. **Filtración de Estratos**: Define la cadena de inclusiones
       V_PHYSICS ⊂ V_TACTICS ⊂ V_STRATEGY ⊂ V_WISDOM
    
    3. **Gatekeeper Algebraico**: Operador de proyección condicional
       G(v, ctx) = v si valid(ctx) else 0⃗
    
    4. **Clausura Transitiva**: Para proyectar a estrato k, todos los
       estratos base j > k deben estar validados.
    
    Thread Safety:
        Todas las operaciones de registro y proyección están protegidas
        por un RLock para garantizar consistencia en entornos concurrentes.
    
    Invariantes:
        - Un servicio registrado no puede cambiar de estrato sin re-registro
        - La proyección fallida siempre retorna respuesta estructurada
        - El cache respeta el TTL configurado
    """
    
    __slots__ = ('_vectors', '_lock', '_cache', '_logger', '_metrics')
    
    def __init__(self, cache_ttl: float = _CACHE_TTL_SECONDS) -> None:
        """
        Inicializa la MIC vacía.
        
        Args:
            cache_ttl: Time-to-live del cache en segundos
        """
        self._vectors: Dict[str, Tuple[Stratum, VectorHandler]] = {}
        self._lock = threading.RLock()
        self._cache: TTLCache[Dict[str, Any]] = TTLCache(ttl_seconds=cache_ttl)
        self._logger = logging.getLogger("MIC.Registry")
        self._metrics: Dict[str, int] = {"projections": 0, "cache_hits": 0, "violations": 0}
    
    # ─────────────────────────────────────────────────────────────────────────
    # PROPIEDADES DE INTROSPECCIÓN
    # ─────────────────────────────────────────────────────────────────────────
    
    @property
    def registered_services(self) -> List[str]:
        """Lista de servicios registrados (vectores base)."""
        with self._lock:
            return list(self._vectors.keys())
    
    @property
    def dimension(self) -> int:
        """Dimensión del espacio vectorial (número de vectores base)."""
        with self._lock:
            return len(self._vectors)
    
    @property
    def metrics(self) -> Dict[str, int]:
        """Métricas de uso de la MIC."""
        with self._lock:
            return self._metrics.copy()
    
    def is_registered(self, service_name: str) -> bool:
        """Verifica si un servicio está registrado como vector base."""
        with self._lock:
            return service_name in self._vectors
    
    def get_stratum(self, service_name: str) -> Optional[Stratum]:
        """Retorna el estrato de un servicio registrado."""
        with self._lock:
            entry = self._vectors.get(service_name)
            return entry[0] if entry else None
    
    def get_services_by_stratum(self, stratum: Stratum) -> List[str]:
        """Retorna todos los servicios en un estrato dado."""
        with self._lock:
            return [name for name, (s, _) in self._vectors.items() if s == stratum]
    
    # ─────────────────────────────────────────────────────────────────────────
    # REGISTRO DE VECTORES
    # ─────────────────────────────────────────────────────────────────────────
    
    def register_vector(
        self,
        service_name: str,
        stratum: Stratum,
        handler: VectorHandler
    ) -> None:
        """
        Registra un microservicio como vector base en la MIC.
        
        Args:
            service_name: Identificador único del servicio
            stratum: Estrato jerárquico al que pertenece
            handler: Función que ejecuta la lógica del servicio
        
        Raises:
            ValueError: Si service_name está vacío
            TypeError: Si handler no es callable
        
        Notes:
            Si el servicio ya existe, se sobrescribe con warning.
        """
        with self._lock:
            # Validación de parámetros
            if not service_name or not service_name.strip():
                raise ValueError("service_name no puede estar vacío")
            
            if not callable(handler):
                raise TypeError(f"handler debe ser callable, recibido: {type(handler).__name__}")
            
            # Warning si sobrescribimos
            if service_name in self._vectors:
                old_stratum = self._vectors[service_name][0]
                self._logger.warning(
                    f"Sobrescribiendo vector '{service_name}': "
                    f"{old_stratum.name} → {stratum.name}"
                )
            
            self._vectors[service_name] = (stratum, handler)
            self._logger.info(f"Vector registrado: {service_name} [{stratum.name}]")
    
    def unregister_vector(self, service_name: str) -> bool:
        """
        Elimina un servicio de la MIC.
        
        Returns:
            True si se eliminó, False si no existía
        """
        with self._lock:
            if service_name in self._vectors:
                del self._vectors[service_name]
                self._logger.info(f"Vector eliminado: {service_name}")
                return True
            return False
    
    # ─────────────────────────────────────────────────────────────────────────
    # VALIDACIÓN JERÁRQUICA (GATEKEEPER)
    # ─────────────────────────────────────────────────────────────────────────
    
    def _normalize_validated_strata(self, raw: Any) -> Set[Stratum]:
        """
        Normaliza el conjunto de estratos validados desde el contexto.
        
        Acepta múltiples formatos de entrada:
        - Set[Stratum], List[Stratum], FrozenSet[Stratum]
        - List[str] con nombres de estratos
        - List[int] con valores numéricos
        
        Returns:
            Set[Stratum] normalizado y validado
        """
        if raw is None:
            return set()
        
        if not isinstance(raw, (set, list, tuple, frozenset)):
            self._logger.warning(f"validated_strata tipo inválido: {type(raw).__name__}")
            return set()
        
        normalized: Set[Stratum] = set()
        
        for item in raw:
            try:
                if isinstance(item, Stratum):
                    normalized.add(item)
                elif isinstance(item, int):
                    normalized.add(Stratum(item))
                elif isinstance(item, str):
                    normalized.add(Stratum[item.upper().strip()])
            except (ValueError, KeyError):
                self._logger.debug(f"Ignorando estrato inválido: {item!r}")
        
        return normalized
    
    def _validate_hierarchy(
        self,
        target: Stratum,
        validated: Set[Stratum],
        force_override: bool = False
    ) -> Tuple[bool, Set[Stratum]]:
        """
        Valida la clausura transitiva de prerrequisitos.
        
        Args:
            target: Estrato objetivo de la proyección
            validated: Estratos ya validados en el contexto
            force_override: Si True, bypasea la validación (con warning)
        
        Returns:
            Tuple (es_válido, estratos_faltantes)
        """
        if force_override:
            self._logger.warning(
                f"⚠️ Validación jerárquica bypaseada para {target.name} via force_override"
            )
            return True, set()
        
        required = target.requires()
        missing = required - validated
        
        return len(missing) == 0, missing
    
    # ─────────────────────────────────────────────────────────────────────────
    # PROYECCIÓN DE INTENCIONES (NÚCLEO DE DESPACHO)
    # ─────────────────────────────────────────────────────────────────────────
    
    def project_intent(
        self,
        service_name: str,
        payload: Dict[str, Any],
        context: Dict[str, Any],
        *,
        use_cache: bool = False
    ) -> Dict[str, Any]:
        """
        Proyecta una intención sobre el espacio vectorial de la MIC.
        
        Este es el método central de despacho. Implementa el flujo:
        1. Resolución: Verifica que el servicio exista
        2. Normalización: Procesa el contexto y estratos validados
        3. Validación: Verifica clausura transitiva (Gatekeeper)
        4. Ejecución: Invoca el handler con el payload
        5. Propagación: Marca el estrato como validado en la respuesta
        
        Args:
            service_name: Nombre del servicio/vector a proyectar
            payload: Argumentos para el handler
            context: Contexto de ejecución (debe incluir validated_strata)
            use_cache: Si True, intenta usar cache para resultados idempotentes
        
        Returns:
            Dict con resultado de la ejecución o error estructurado
        
        Raises:
            No lanza excepciones directamente; todos los errores se encapsulan
            en la respuesta con success=False.
        """
        # ─── Fase 0: Cache Check ───
        if use_cache:
            cache_key = f"{service_name}:{hashlib.sha256(str(sorted(payload.items())).encode()).hexdigest()[:16]}"
            cached = self._cache.get(cache_key)
            if cached is not None:
                with self._lock:
                    self._metrics["cache_hits"] += 1
                self._logger.debug(f"Cache hit: {service_name}")
                return cached
        
        # ─── Fase 1: Resolución del Vector ───
        with self._lock:
            self._metrics["projections"] += 1
            
            if service_name not in self._vectors:
                available = list(self._vectors.keys())
                error_msg = (
                    f"Vector desconocido: '{service_name}'. "
                    f"Disponibles: {available if available else 'ninguno registrado'}"
                )
                return self._create_error_response(
                    ValueError(error_msg),
                    error_category="resolution_error"
                )
            
            target_stratum, handler = self._vectors[service_name]
        
        # ─── Fase 2: Normalización del Contexto ───
        raw_validated = context.get("validated_strata", set())
        validated_strata = self._normalize_validated_strata(raw_validated)
        force_override = bool(context.get("force_physics_override", False))
        
        # ─── Fase 3: Validación Jerárquica (Gatekeeper) ───
        is_valid, missing = self._validate_hierarchy(
            target_stratum, validated_strata, force_override
        )
        
        if not is_valid:
            with self._lock:
                self._metrics["violations"] += 1
            
            error = MICHierarchyViolationError(
                target_stratum=target_stratum,
                missing_strata=missing,
                validated_strata=validated_strata
            )
            self._logger.error(str(error))
            return self._create_error_response(error)
        
        # ─── Fase 4: Ejecución del Handler ───
        try:
            result = handler(**payload)
            
            # Normalizar resultado a dict
            if not isinstance(result, dict):
                result = {"success": True, "result": result}
            
            # ─── Fase 5: Propagación de Validación ───
            if result.get("success", False):
                result["_mic_validation_update"] = target_stratum.value
                result["_mic_stratum"] = target_stratum.name
                result["_mic_validated_strata"] = [s.name for s in validated_strata | {target_stratum}]
            
            # Cache si es exitoso y está habilitado
            if use_cache and result.get("success", False):
                self._cache.set(cache_key, result)
            
            return result
        
        except TypeError as e:
            # Error de firma: payload no coincide con handler
            self._logger.error(f"Firma de handler incorrecta para '{service_name}': {e}")
            return self._create_error_response(
                e,
                error_category="handler_signature_error",
                service_name=service_name,
                hint="Verifique que las claves del payload coincidan con los parámetros del handler"
            )
        
        except Exception as e:
            self._logger.exception(f"Error ejecutando vector '{service_name}'")
            return self._create_error_response(
                e,
                error_category="execution_error",
                service_name=service_name
            )
    
    # ─────────────────────────────────────────────────────────────────────────
    # UTILIDADES Y RESPUESTAS
    # ─────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _create_error_response(
        error: Union[Exception, str],
        error_category: str = "error",
        **extra
    ) -> Dict[str, Any]:
        """Crea respuesta de error estructurada y consistente."""
        if isinstance(error, MICException):
            return {"success": False, **error.to_dict(), **extra}
        
        msg = str(error)
        details = getattr(error, "details", {}) if isinstance(error, Exception) else {}
        
        return {
            "success": False,
            "error": msg,
            "error_type": type(error).__name__ if isinstance(error, Exception) else "Error",
            "error_category": error_category,
            "error_details": details,
            **extra
        }
    
    def clear_cache(self) -> int:
        """Limpia el cache de resultados. Retorna número de entradas eliminadas."""
        count = self._cache.clear()
        self._logger.info(f"Cache limpiado: {count} entradas eliminadas")
        return count
    
    def get_stratum_hierarchy(self) -> Dict[str, List[str]]:
        """
        Retorna la estructura jerárquica de la MIC.
        
        Returns:
            Dict mapeando cada estrato a sus servicios
        """
        hierarchy: Dict[str, List[str]] = {s.name: [] for s in Stratum}
        
        with self._lock:
            for name, (stratum, _) in self._vectors.items():
                hierarchy[stratum.name].append(name)
        
        return hierarchy


# =============================================================================
# FUNCIONES DE ENTROPÍA Y PROBABILIDAD
# =============================================================================

def compute_shannon_entropy(
    probabilities: Sequence[float],
    base: float = 2.0
) -> float:
    """
    Calcula la entropía de Shannon con estabilidad numérica.
    
    H(X) = -Σ p(x) · log_b(p(x))
    
    Args:
        probabilities: Distribución de probabilidad (debe sumar ~1.0)
        base: Base del logaritmo (2.0 para bits, e para nats)
    
    Returns:
        Entropía en la unidad correspondiente a la base
    
    Raises:
        ValueError: Si alguna probabilidad es negativa
    
    Notes:
        - Probabilidades cercanas a 0 se manejan con _EPSILON
        - La distribución se normaliza si no suma exactamente 1.0
    """
    if not probabilities:
        return 0.0

    probs = np.asarray(probabilities, dtype=np.float64)

    if np.any(probs < 0):
        raise ValueError("Las probabilidades no pueden ser negativas")

    total = np.sum(probs)
    if total < _EPSILON:
        return 0.0

    # Normalizar si es necesario
    if not np.isclose(total, 1.0, rtol=1e-5):
        probs = probs / total

    # Filtrar valores muy pequeños para estabilidad
    mask = probs > _EPSILON
    nonzero_probs = probs[mask]

    if len(nonzero_probs) == 0:
        return 0.0

    log_probs = np.log(nonzero_probs) / np.log(base)
    entropy = -np.sum(nonzero_probs * log_probs)

    return float(max(0.0, entropy))


def distribution_from_counts(counts: Union[Dict[Any, int], Counter]) -> List[float]:
    """Convierte conteos a distribución de probabilidad normalizada."""
    if not counts:
        return []
    
    values = list(counts.values())
    total = sum(values)
    
    if total == 0:
        return []
    
    return [v / total for v in values]


def compute_persistence_entropy(intervals: Sequence[PersistenceInterval]) -> float:
    """
    Calcula la entropía del diagrama de persistencia.
    
    La entropía de persistencia cuantifica la "complejidad topológica"
    del espacio: valores altos indican muchas características de
    persistencias similares (más complejo), valores bajos indican
    dominancia de pocas características (más simple).
    
    Args:
        intervals: Secuencia de intervalos de persistencia
    
    Returns:
        Entropía normalizada en [0, 1]
    """
    if not intervals:
        return 0.0
    
    # Filtrar intervalos finitos (esenciales tienen persistencia infinita)
    finite_intervals = [iv for iv in intervals if not iv.is_essential]
    
    if not finite_intervals:
        return 0.0
    
    persistences = [iv.persistence for iv in finite_intervals]
    total_persistence = sum(persistences)
    
    if total_persistence < _EPSILON:
        return 0.0
    
    probs = [p / total_persistence for p in persistences]
    raw_entropy = compute_shannon_entropy(probs)
    
    # Normalizar por entropía máxima posible
    max_entropy = math.log2(len(probs)) if len(probs) > 1 else 1.0
    
    return raw_entropy / max_entropy if max_entropy > 0 else 0.0


# =============================================================================
# ANÁLISIS TOPOLÓGICO
# =============================================================================

def detect_cyclic_patterns(lines: List[str], max_period: int = 50) -> int:
    """
    Detecta patrones cíclicos en una secuencia de líneas.
    
    Utiliza autocorrelación discreta para encontrar periodicidades.
    
    Args:
        lines: Lista de líneas a analizar
        max_period: Período máximo a buscar
    
    Returns:
        Número de patrones cíclicos significativos detectados (proxy para β₁)
    """
    if len(lines) < 3:
        return 0
    
    cycles_found = 0
    
    for period in range(1, min(max_period, len(lines) // 2)):
        matches = sum(1 for i in range(len(lines) - period) if lines[i] == lines[i + period])
        comparisons = len(lines) - period
        
        if comparisons > 0 and matches / comparisons > 0.8:
            cycles_found += 1
    
    return cycles_found


def estimate_intrinsic_dimension(lines: List[str]) -> int:
    """
    Estima la dimensión intrínseca del espacio de datos.
    
    Para archivos tabulares, esto corresponde aproximadamente al
    número de columnas (grados de libertad).
    """
    if not lines:
        return 0
    
    sample = lines[0]
    
    for delimiter in [',', ';', '\t', '|']:
        if delimiter in sample:
            col_counts = [len(line.split(delimiter)) for line in lines[:100]]
            if col_counts:
                return max(set(col_counts), key=col_counts.count)
    
    return 1


def analyze_topological_features(file_path: Path) -> TopologicalSummary:
    """
    Analiza características topológicas de un archivo.
    
    Computa una aproximación a los números de Betti basada en:
    - β₀: Número de patrones únicos (componentes conexos)
    - β₁: Patrones cíclicos detectados
    - Entropía estructural de la distribución de líneas
    
    Args:
        file_path: Ruta al archivo a analizar
    
    Returns:
        TopologicalSummary con métricas calculadas
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = [line.rstrip('\n\r') for line in f.readlines()[:_MAX_SAMPLE_ROWS]]
        
        if not lines:
            return TopologicalSummary.empty()
        
        # Conteo de patrones únicos
        line_counts = Counter(lines)
        num_unique = len(line_counts)
        
        # Números de Betti aproximados
        beta_0 = num_unique  # Componentes conexos
        beta_1 = detect_cyclic_patterns(lines)  # Ciclos
        
        # Dimensión intrínseca
        dimension = estimate_intrinsic_dimension(lines)
        
        # Entropía estructural
        distribution = distribution_from_counts(line_counts)
        structural_entropy = compute_shannon_entropy(distribution)
        
        # Crear resumen
        betti = BettiNumbers(beta_0=beta_0, beta_1=beta_1, beta_2=0)
        
        return TopologicalSummary(
            betti=betti,
            structural_entropy=structural_entropy,
            persistence_entropy=0.0,  # Se calcula con diagrama de persistencia
            intrinsic_dimension=dimension
        )
    
    except Exception as e:
        logger.warning(f"Análisis topológico falló: {e}")
        return TopologicalSummary.empty()


def compute_homology_from_diagnostic(diagnostic_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calcula grupos de homología a partir de datos diagnósticos.
    
    Mapea el espacio de issues a un espacio topológico:
    - Tipos de issues → Componentes conexos (β₀)
    - Referencias circulares → Ciclos (β₁)
    """
    issues = diagnostic_data.get("issues", [])
    warnings = diagnostic_data.get("warnings", [])
    
    # Tipos de issues como componentes
    issue_types: Set[str] = set()
    for issue in issues:
        if isinstance(issue, dict):
            issue_types.add(issue.get("type", issue.get("code", "unknown")))
        else:
            issue_types.add(str(type(issue).__name__))
    
    beta_0 = max(1, len(issue_types))
    
    # Detectar referencias circulares
    circular_keywords = {"circular", "cycle", "loop", "recursive", "dependency"}
    
    def has_circular(item: Any) -> bool:
        return any(kw in str(item).lower() for kw in circular_keywords)
    
    beta_1 = sum(1 for w in warnings if has_circular(w))
    beta_1 += sum(1 for i in issues if has_circular(i))
    
    betti = BettiNumbers(beta_0=beta_0, beta_1=beta_1, beta_2=0)
    
    return {
        "H_0": f"ℤ^{beta_0}",
        "H_1": f"ℤ^{beta_1}",
        **betti.to_dict()
    }


def compute_persistence_diagram(diagnostic_data: Dict[str, Any]) -> List[PersistenceInterval]:
    """
    Calcula diagrama de persistencia para issues diagnósticos.
    
    Cada issue genera un intervalo cuya persistencia es proporcional
    a su severidad.
    """
    issues = diagnostic_data.get("issues", [])
    if not issues:
        return []
    
    severity_map = {"CRITICAL": 1.0, "HIGH": 0.8, "MEDIUM": 0.5, "LOW": 0.2, "INFO": 0.1}
    intervals: List[PersistenceInterval] = []
    
    for idx, issue in enumerate(issues):
        if isinstance(issue, dict):
            raw_severity = issue.get("severity", "MEDIUM")
            severity = severity_map.get(str(raw_severity).upper(), 0.5)
        else:
            severity = 0.5
        
        birth = idx * 0.1
        death = birth + severity
        
        try:
            intervals.append(PersistenceInterval(birth=birth, death=death, dimension=0))
        except ValueError:
            continue
    
    # Filtrar por umbral de significancia y ordenar
    significant = [iv for iv in intervals if iv.persistence >= _PERSISTENCE_THRESHOLD]
    significant.sort()
    
    return significant


def compute_diagnostic_magnitude(diagnostic_data: Dict[str, Any]) -> float:
    """
    Calcula la magnitud normalizada del vector diagnóstico.
    
    Usa norma ponderada: ||v|| = √(w_e·|errors|² + w_i·|issues|² + w_w·|warnings|²)
    normalizada por tanh para acotar en [0, 1].
    """
    n_errors = len(diagnostic_data.get("errors", []))
    n_issues = len(diagnostic_data.get("issues", []))
    n_warnings = len(diagnostic_data.get("warnings", []))
    
    # Pesos por severidad
    raw_magnitude = math.sqrt(
        3.0 * (n_errors ** 2) +
        2.0 * (n_issues ** 2) +
        1.0 * (n_warnings ** 2)
    )
    
    return round(math.tanh(raw_magnitude / 10.0), 4)


# =============================================================================
# VALIDACIÓN DE ARCHIVOS
# =============================================================================

def normalize_path(path: Union[str, Path]) -> Path:
    """Normaliza ruta a Path absoluto y resuelto."""
    if path is None:
        raise ValueError("Path no puede ser None")
    if not str(path).strip():
        raise ValueError("Path no puede estar vacío")
    return Path(path).expanduser().resolve()


def validate_file_exists(path: Path) -> None:
    """Valida existencia y tipo del archivo."""
    if not path.exists():
        raise FileNotFoundDiagnosticError(path)
    if not path.is_file():
        raise FileValidationError(f"La ruta no es un archivo: {path}")


def validate_file_extension(path: Path) -> str:
    """Valida extensión del archivo."""
    ext = path.suffix.lower()
    if ext not in VALID_EXTENSIONS:
        raise FileValidationError(
            f"Extensión inválida: {ext}",
            expected=list(VALID_EXTENSIONS)
        )
    return ext


def validate_file_size(path: Path, max_size: int = MAX_FILE_SIZE_BYTES) -> Tuple[int, bool]:
    """
    Valida tamaño del archivo.
    
    Returns:
        Tuple (tamaño_bytes, está_vacío)
    """
    size = path.stat().st_size
    if size > max_size:
        raise FileValidationError(
            f"Archivo excede límite: {size:,} > {max_size:,} bytes",
            actual_size=size,
            max_size=max_size
        )
    return size, size == 0


def normalize_encoding(encoding: str) -> str:
    """Normaliza nombre de codificación con fallback seguro."""
    if not encoding:
        return "utf-8"
    
    norm = encoding.lower().replace("_", "-").replace(" ", "")
    
    # Buscar en aliases
    for alias, standard in _ENCODING_ALIASES.items():
        if norm == alias.lower().replace("_", "").replace("-", ""):
            return standard
    
    # Verificar si es soportado directamente
    if encoding.lower() in SUPPORTED_ENCODINGS:
        return encoding.lower()
    
    return "utf-8"


def normalize_file_type(file_type: Union[str, FileType]) -> FileType:
    """Normaliza tipo de archivo."""
    if isinstance(file_type, FileType):
        return file_type
    return FileType.from_string(file_type)


# =============================================================================
# REGISTRO DE DIAGNÓSTICOS
# =============================================================================

_DIAGNOSTIC_REGISTRY: Dict[FileType, Optional[Type[DiagnosticProtocol]]] = {
    FileType.APUS: APUFileDiagnostic,
    FileType.INSUMOS: InsumosFileDiagnostic,
    FileType.PRESUPUESTO: PresupuestoFileDiagnostic,
}


def get_diagnostic_class(file_type: FileType) -> Type[DiagnosticProtocol]:
    """Obtiene la clase diagnóstica para un tipo de archivo."""
    diagnostic_class = _DIAGNOSTIC_REGISTRY.get(file_type)
    
    if diagnostic_class is None:
        raise UnsupportedFileTypeError(
            file_type.value,
            FileType.values()
        )
    
    return diagnostic_class


# =============================================================================
# HANDLERS DE LA MIC (VECTORES DE SERVICIO)
# =============================================================================

def diagnose_file(
    file_path: Union[str, Path],
    file_type: Union[str, FileType],
    *,
    validate_extension: bool = True,
    max_file_size: Optional[int] = None,
    topological_analysis: bool = False
) -> Dict[str, Any]:
    """
    Vector de Diagnóstico con análisis topológico opcional.
    
    Estrato: PHYSICS (Nivel 3 - Base de la pirámide)
    Prerrequisitos: Ninguno
    
    Args:
        file_path: Ruta al archivo a diagnosticar
        file_type: Tipo de archivo (apus, insumos, presupuesto)
        validate_extension: Si validar extensión del archivo
        max_file_size: Límite de tamaño personalizado
        topological_analysis: Si incluir análisis topológico
    
    Returns:
        Dict con resultados del diagnóstico
    """
    path_str = str(file_path)
    
    try:
        # ─── Validación ───
        path = normalize_path(file_path)
        normalized_type = normalize_file_type(file_type)
        
        logger.info(f"Iniciando diagnóstico: {path} [{normalized_type.value}]")
        
        validate_file_exists(path)
        
        if validate_extension:
            validate_file_extension(path)
        
        effective_max = max_file_size or MAX_FILE_SIZE_BYTES
        size, is_empty = validate_file_size(path, effective_max)
        
        if is_empty:
            logger.warning(f"Archivo vacío: {path}")
            return {
                "success": True,
                "diagnostic_completed": True,
                "is_empty": True,
                "file_type": normalized_type.value,
                "file_path": str(path),
                "file_size_bytes": 0
            }
        
        # ─── Diagnóstico Base ───
        diagnostic_class = get_diagnostic_class(normalized_type)
        diagnostic = diagnostic_class(str(path))
        diagnostic.diagnose()
        
        result_data = diagnostic.to_dict()
        result_data["diagnostic_completed"] = True
        
        # ─── Análisis Topológico ───
        if topological_analysis:
            logger.debug("Ejecutando análisis topológico")
            
            topo_summary = analyze_topological_features(path)
            result_data["topological_features"] = topo_summary.to_dict()
            
            homology = compute_homology_from_diagnostic(result_data)
            result_data["homology"] = homology
            
            intervals = compute_persistence_diagram(result_data)
            result_data["persistence_diagram"] = [iv.to_dict() for iv in intervals]
            result_data["persistence_entropy"] = compute_persistence_entropy(intervals)
        
        magnitude = compute_diagnostic_magnitude(result_data)
        
        logger.info(f"Diagnóstico completado: {path} [magnitud={magnitude}]")
        
        return {
            "success": True,
            **result_data,
            "file_type": normalized_type.value,
            "file_path": str(path),
            "file_size_bytes": size,
            "diagnostic_magnitude": magnitude,
            "has_topological_analysis": topological_analysis
        }
    
    except MICException as e:
        logger.warning(f"Error de validación: {e}")
        return {"success": False, **e.to_dict()}
    
    except Exception as e:
        logger.exception(f"Error inesperado en diagnóstico: {e}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "error_category": "unexpected",
            "file_path": path_str
        }


# =============================================================================
# BOOTSTRAP Y REGISTRO DE VECTORES
# =============================================================================

def register_core_vectors(
    mic: MICRegistry,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Registra los vectores fundamentales del sistema en la MIC.
    
    Establece la base vectorial canónica {e₁, e₂, ..., eₙ} del espacio
    de operaciones, organizada por estratos DIKW.
    
    Args:
        mic: Instancia de MICRegistry donde registrar
        config: Configuración opcional del sistema
    """
    # ─── Vectores PHYSICS (Base de la pirámide) ───
    mic.register_vector("stabilize_flux", Stratum.PHYSICS, vector_stabilize_flux)
    mic.register_vector("parse_raw", Stratum.PHYSICS, vector_parse_raw_structure)
    
    # ─── Vectores TACTICS ───
    mic.register_vector("structure_logic", Stratum.TACTICS, vector_structure_logic)
    mic.register_vector("audit_fusion_homology", Stratum.TACTICS, vector_audit_homological_fusion)
    
    # ─── Vectores STRATEGY ───
    mic.register_vector("lateral_thinking_pivot", Stratum.STRATEGY, vector_lateral_pivot)
    
    # ─── Vectores con dependencias opcionales ───
    if config:
        try:
            from app.semantic_estimator import SemanticEstimatorService
            service = SemanticEstimatorService(config)
            service.register_in_mic(mic)
            logger.info("✅ Vectores semánticos registrados")
        except Exception as e:
            logger.warning(f"⚠️ Vectores semánticos no disponibles: {e}")
    
    try:
        from app.semantic_dictionary import SemanticDictionaryService
        semantic_dict = SemanticDictionaryService()
        semantic_dict.register_in_mic(mic)
        logger.info("✅ Diccionario semántico registrado")
    except Exception as e:
        logger.warning(f"⚠️ Diccionario semántico no disponible: {e}")
    
    logger.info(f"✅ MIC inicializada con {mic.dimension} vectores base")


# =============================================================================
# API PÚBLICA
# =============================================================================

def get_supported_file_types() -> List[str]:
    """Retorna tipos de archivo soportados."""
    return FileType.values()


def get_supported_delimiters() -> List[str]:
    """Retorna delimitadores CSV soportados."""
    return list(VALID_DELIMITERS)


def get_supported_encodings() -> List[str]:
    """Retorna codificaciones soportadas."""
    return list(SUPPORTED_ENCODINGS)


def validate_file_for_processing(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Valida completamente un archivo para procesamiento.
    
    Returns:
        Dict con {"valid": bool, "size": int, "extension": str, ...}
    """
    try:
        p = normalize_path(path)
        validate_file_exists(p)
        ext = validate_file_extension(p)
        size, is_empty = validate_file_size(p)
        return {"valid": True, "size": size, "extension": ext, "is_empty": is_empty}
    except MICException as e:
        return {"valid": False, "errors": [str(e)], **e.to_dict()}
    except Exception as e:
        return {"valid": False, "errors": [str(e)]}


# =============================================================================
# SINGLETON DE LA MIC (OPCIONAL)
# =============================================================================

_global_mic: Optional[MICRegistry] = None
_mic_lock = threading.Lock()


def get_global_mic() -> MICRegistry:
    """
    Obtiene la instancia global de la MIC (patrón Singleton thread-safe).
    
    Útil para aplicaciones que requieren una única MIC compartida.
    """
    global _global_mic
    
    if _global_mic is None:
        with _mic_lock:
            if _global_mic is None:
                _global_mic = MICRegistry()
                register_core_vectors(_global_mic)
    
    return _global_mic