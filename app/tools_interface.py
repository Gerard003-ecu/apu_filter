"""
Módulo MIC Refinado - Espacio Vectorial Jerárquico con Clausura Transitiva.

Este módulo implementa la Matriz de Interacción Central (MIC) fusionando:
1. Estructura Base (Propuesta 1): MICRegistry, IntentVector, Gatekeeper.
2. Inteligencia Matemática (Propuesta 2): Métodos refinados con topología algebraica y entropía.

Modelo Matemático:
------------------
Sea V un espacio vectorial sobre R con base {e_1, ..., e_n} donde cada e_i
representa un microservicio. Definimos una filtración:

    V = V_0 ⊃ V_1 ⊃ V_2 ⊃ V_3

donde V_k corresponde al estrato k (WISDOM=0, STRATEGY=1, TACTICS=2, PHYSICS=3).

La proyección de intención π: I → V satisface la restricción:
    π(intent) ∈ V_k ⟹ ∀j > k: V_j está validado

Esto garantiza que las operaciones de alto nivel solo se ejecutan
cuando la base física está correctamente establecida.

Funciones Auxiliares de Topología Algebraica y Análisis Estructural.

Marco Matemático Adicional:
---------------------------
1. Homología Simplicial: H_k(X) calculada via complejos de cadenas
2. Persistencia: Filtración F_0 ⊆ F_1 ⊆ ... ⊆ F_n con tracking de nacimiento/muerte
3. Entropía de Shannon: H(X) = -Σ p(x)log(p(x)) con regularización
4. Métricas de Riesgo: Basadas en teoría de portafolios de Markowitz
"""

import codecs
import logging
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)

import numpy as np
import pandas as pd
import networkx as nx

# Integración del Motor Financiero y Scripts Existentes
from scripts.clean_csv import CSVCleaner
from scripts.diagnose_apus_file import APUFileDiagnostic
from scripts.diagnose_insumos_file import InsumosFileDiagnostic
from scripts.diagnose_presupuesto_file import PresupuestoFileDiagnostic
from .financial_engine import FinancialConfig, FinancialEngine
from .schemas import Stratum

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES Y CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

MAX_FILE_SIZE_BYTES: Final[int] = 100 * 1024 * 1024  # 100 MB

SUPPORTED_ENCODINGS: Final[FrozenSet[str]] = frozenset({
    "utf-8", "utf-8-sig", "latin-1", "iso-8859-1",
    "cp1252", "ascii", "utf-16", "utf-16-le", "utf-16-be",
})

# Mapeo de aliases comunes para normalización
_ENCODING_ALIASES: Final[Dict[str, str]] = {
    "utf8": "utf-8",
    "latin1": "latin-1",
    "iso88591": "iso-8859-1",
    "cp65001": "utf-8",
}

VALID_DELIMITERS: Final[FrozenSet[str]] = frozenset({",", ";", "\t", "|", ":"})
VALID_EXTENSIONS: Final[FrozenSet[str]] = frozenset({".csv", ".txt", ".tsv"})

# Constantes Matemáticas de la Propuesta
_EPSILON: Final[float] = 1e-10  # Para estabilidad numérica
_DEFAULT_RANDOM_SEED: Final[int] = 42
_MAX_SAMPLE_ROWS: Final[int] = 1000  # Límite para análisis topológico
_PERSISTENCE_THRESHOLD: Final[float] = 0.01  # Filtrar características efímeras


# ══════════════════════════════════════════════════════════════════════════════
# PROTOCOLOS
# ══════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class TelemetryContextProtocol(Protocol):
    """Protocolo para contextos de telemetría."""
    def get_business_report(self) -> Dict[str, Any]: ...


@runtime_checkable
class DiagnosticProtocol(Protocol):
    """Protocolo para clases diagnósticas."""
    def diagnose(self) -> None: ...
    def to_dict(self) -> Dict[str, Any]: ...


# ══════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES JERÁRQUICAS
# ══════════════════════════════════════════════════════════════════════════════

class DiagnosticError(Exception):
    """Base para errores de diagnóstico."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.details: Dict[str, Any] = details or {}


class FileNotFoundDiagnosticError(DiagnosticError):
    """Archivo no encontrado durante diagnóstico."""
    pass


class UnsupportedFileTypeError(DiagnosticError):
    """Tipo de archivo no soportado."""
    pass


class FileValidationError(DiagnosticError):
    """Error de validación de archivo."""
    pass


class CleaningError(Exception):
    """Error durante limpieza de archivos."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.details: Dict[str, Any] = details or {}


class MICHierarchyViolationError(Exception):
    """Violación de la jerarquía de estratos en la MIC."""
    def __init__(
        self,
        message: str,
        target_stratum: 'Stratum',
        missing_strata: Set['Stratum']
    ) -> None:
        super().__init__(message)
        self.target_stratum = target_stratum
        self.missing_strata = missing_strata


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════════════════════

class FileType(str, Enum):
    """Tipos de archivo soportados para diagnóstico."""
    APUS = "apus"
    INSUMOS = "insumos"
    PRESUPUESTO = "presupuesto"

    @classmethod
    def values(cls) -> List[str]:
        """Retorna lista de valores válidos."""
        return [member.value for member in cls]

    @classmethod
    def from_string(cls, value: str) -> "FileType":
        """
        Parsea string a FileType con normalización.
        """
        if not isinstance(value, str):
            raise ValueError(f"Expected string, got {type(value).__name__}")

        normalized = value.strip().lower()

        for member in cls:
            if member.value == normalized:
                return member

        valid_options = ", ".join(cls.values())
        raise ValueError(f"'{value}' is not valid. Options: {valid_options}")


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS TOPOLÓGICAS (Propuesta 2)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PersistenceInterval:
    """
    Intervalo de persistencia [birth, death) en el diagrama.

    Representa el tiempo de vida de una característica topológica
    durante la filtración del complejo simplicial.
    """
    birth: float
    death: float
    dimension: int = 0  # 0 = componentes, 1 = ciclos, 2 = cavidades

    def __post_init__(self):
        if self.birth < 0:
            raise ValueError(f"Birth time must be non-negative, got {self.birth}")
        if self.death < self.birth:
            raise ValueError(f"Death ({self.death}) must be >= birth ({self.birth})")
        if self.dimension < 0:
            raise ValueError(f"Dimension must be non-negative, got {self.dimension}")

    @property
    def persistence(self) -> float:
        """Tiempo de vida de la característica."""
        return self.death - self.birth

    @property
    def is_essential(self) -> bool:
        """True si la característica nunca muere (death = inf)."""
        return math.isinf(self.death)

    def __lt__(self, other: 'PersistenceInterval') -> bool:
        """Ordenamiento por persistencia descendente."""
        return self.persistence > other.persistence


@dataclass(frozen=True)
class TopologicalSummary:
    """
    Resumen de características topológicas de un dataset.

    Encapsula números de Betti, entropía estructural y métricas derivadas.
    """
    betti_0: int  # Componentes conexos
    betti_1: int  # Ciclos independientes
    betti_2: int  # Cavidades
    euler_characteristic: int
    structural_entropy: float
    persistence_entropy: float

    @classmethod
    def empty(cls) -> 'TopologicalSummary':
        """Crea un resumen vacío para casos de error."""
        return cls(
            betti_0=0, betti_1=0, betti_2=0,
            euler_characteristic=0,
            structural_entropy=0.0,
            persistence_entropy=0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "beta_0": self.betti_0,
            "beta_1": self.betti_1,
            "beta_2": self.betti_2,
            "betti_numbers": [self.betti_0, self.betti_1, self.betti_2],
            "euler_characteristic": self.euler_characteristic,
            "structural_entropy": round(self.structural_entropy, 6),
            "persistence_entropy": round(self.persistence_entropy, 6),
        }


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE LA MIC (Propuesta 1)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class IntentVector:
    """
    Vector de intención inmutable proyectado sobre la MIC.

    Representa una solicitud del Agente para ejecutar un servicio,
    encapsulando el contexto necesario para la validación jerárquica.
    """
    service_name: str
    payload: Dict[str, Any]
    context: Dict[str, Any]

    def __post_init__(self):
        # Validación en construcción
        if not self.service_name or not self.service_name.strip():
            raise ValueError("service_name cannot be empty")


class MICRegistry:
    """
    Matriz de Interacción Central (MIC).

    Implementa un espacio vectorial jerárquico donde:
    - Cada servicio es un vector base e_i
    - Los estratos definen una filtración descendente
    - El Gatekeeper asegura la clausura transitiva de validaciones

    Invariante Topológico:
        Para proyectar en estrato k, todos los estratos j > k deben estar validados.
        Esto forma una precondición de tipo "fibración" donde PHYSICS es la base.
    """

    def __init__(self) -> None:
        self._vectors: Dict[str, Tuple[Stratum, Callable[..., Dict[str, Any]]]] = {}
        self._logger = logging.getLogger("MIC")

    @property
    def registered_services(self) -> List[str]:
        """Lista de servicios registrados."""
        return list(self._vectors.keys())

    def is_registered(self, vector_name: str) -> bool:
        """Verifica si un vector está registrado."""
        return vector_name in self._vectors

    def get_vector_stratum(self, vector_name: str) -> Optional[Stratum]:
        """Retorna el estrato de un vector."""
        if vector_name in self._vectors:
            return self._vectors[vector_name][0]
        return None

    def get_required_strata(self, stratum: Stratum) -> Set[Stratum]:
        """Retorna los estratos requeridos para ejecutar en el estrato dado."""
        return self._compute_required_strata(stratum)

    def register_vector(
        self,
        service_name: str,
        stratum: Stratum,
        handler: Callable[..., Dict[str, Any]]
    ) -> None:
        """
        Registra un microservicio como vector base en la MIC.
        """
        if not service_name or not service_name.strip():
            raise ValueError("service_name cannot be empty")

        if not callable(handler):
            raise TypeError("handler must be callable")

        if service_name in self._vectors:
            self._logger.warning(f"Overwriting existing vector: {service_name}")

        self._vectors[service_name] = (stratum, handler)
        self._logger.info(f"Vector registered: {service_name} [{stratum.name}]")

    def _compute_required_strata(self, target: Stratum) -> Set[Stratum]:
        """
        Calcula la clausura transitiva de prerrequisitos.
        """
        return {s for s in Stratum if s.value > target.value}

    def _normalize_validated_strata(
        self,
        raw_validated: Any
    ) -> Set[Stratum]:
        """
        Normaliza el conjunto de estratos validados.
        """
        if raw_validated is None:
            return set()

        if not isinstance(raw_validated, (set, list, tuple, frozenset)):
            self._logger.warning(
                f"Invalid validated_strata type: {type(raw_validated).__name__}, "
                f"expected iterable"
            )
            return set()

        normalized: Set[Stratum] = set()

        for item in raw_validated:
            try:
                if isinstance(item, Stratum):
                    normalized.add(item)
                elif isinstance(item, int):
                    normalized.add(Stratum(item))
                elif isinstance(item, str):
                    normalized.add(Stratum[item.upper().strip()])
                else:
                    self._logger.debug(f"Skipping invalid stratum item: {item!r}")
            except (ValueError, KeyError) as e:
                self._logger.debug(f"Could not normalize stratum item {item!r}: {e}")

        return normalized

    def _validate_hierarchy(
        self,
        target: Stratum,
        validated: Set[Stratum],
        force_override: bool = False
    ) -> Tuple[bool, Set[Stratum]]:
        """
        Valida los prerrequisitos jerárquicos (Gatekeeper).
        """
        if force_override:
            self._logger.warning(
                f"Hierarchy validation bypassed for {target.name} via force_override"
            )
            return True, set()

        required = self._compute_required_strata(target)
        missing = required - validated

        return len(missing) == 0, missing

    def project_intent(
        self,
        service_name: str,
        payload: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Proyecta una intención sobre el espacio vectorial.
        """
        # ─── Fase 1: Resolución ───
        if service_name not in self._vectors:
            available = self.registered_services
            error_msg = (
                f"Unknown vector: '{service_name}'. "
                f"Available: {available if available else 'none registered'}"
            )
            raise ValueError(error_msg)

        target_stratum, handler = self._vectors[service_name]

        # ─── Fase 2: Normalización ───
        raw_validated = context.get("validated_strata", set())
        validated_strata = self._normalize_validated_strata(raw_validated)
        force_override = bool(context.get("force_physics_override", False))

        # ─── Fase 3: Validación Jerárquica ───
        is_valid, missing = self._validate_hierarchy(
            target_stratum, validated_strata, force_override
        )

        if not is_valid:
            missing_names = sorted(
                [s.name for s in missing],
                key=lambda name: Stratum[name].value,
                reverse=True  # PHYSICS primero
            )
            error_msg = (
                f"MIC Hierarchy Violation: Cannot project to {target_stratum.name}. "
                f"Missing prerequisite strata: {', '.join(missing_names)}. "
                f"Validate PHYSICS operations first."
            )
            self._logger.error(error_msg)

            return _create_error_response(
                PermissionError(error_msg),
                error_category="mic_hierarchy_violation",
                target_stratum=target_stratum.name,
                missing_strata=missing_names,
                validated_strata=[s.name for s in validated_strata]
            )

        # ─── Fase 4: Ejecución ───
        try:
            result = handler(**payload)

            # Asegurar que result es dict
            if not isinstance(result, dict):
                result = {"success": True, "result": result}

            # ─── Fase 5: Propagación de Validación ───
            if result.get("success", False):
                result["_mic_validation_update"] = target_stratum
                result["_mic_stratum"] = target_stratum.name

            return result

        except TypeError as e:
            # Error común: firma del handler no coincide con payload
            self._logger.error(
                f"Handler signature mismatch for '{service_name}': {e}",
                exc_info=True
            )
            return _create_error_response(
                e,
                error_category="mic_handler_signature_error",
                service_name=service_name,
                hint="Verify that payload keys match handler parameter names"
            )
        except Exception as e:
            self._logger.error(
                f"Error executing vector '{service_name}': {e}",
                exc_info=True
            )
            return _create_error_response(
                e,
                error_category="mic_execution_error",
                service_name=service_name
            )


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRO DE DIAGNÓSTICOS
# ══════════════════════════════════════════════════════════════════════════════

_DIAGNOSTIC_REGISTRY: Final[Dict[FileType, Type[DiagnosticProtocol]]] = {
    FileType.APUS: APUFileDiagnostic,
    FileType.INSUMOS: InsumosFileDiagnostic,
    FileType.PRESUPUESTO: PresupuestoFileDiagnostic,
}


def _get_diagnostic_class(file_type: FileType) -> Type[DiagnosticProtocol]:
    """Obtiene la clase diagnóstica para un tipo de archivo."""
    diagnostic_class = _DIAGNOSTIC_REGISTRY.get(file_type)

    if diagnostic_class is None:
        raise UnsupportedFileTypeError(
            f"No diagnostic registered for type: {file_type.value}",
            details={
                "file_type": file_type.value,
                "available_types": FileType.values()
            }
        )

    return diagnostic_class


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE ENTROPÍA Y PROBABILIDAD (Propuesta 2)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_shannon_entropy(
    probabilities: Sequence[float],
    base: float = 2.0
) -> float:
    """
    Calcula la entropía de Shannon con estabilidad numérica.

    H(X) = -Σ p(x) * log_b(p(x))
    """
    if not probabilities:
        return 0.0

    probs = np.asarray(probabilities, dtype=np.float64)

    if np.any(probs < 0):
        raise ValueError("Probabilities cannot be negative")

    total = np.sum(probs)
    if total < _EPSILON:
        return 0.0

    if not np.isclose(total, 1.0, rtol=1e-5):
        probs = probs / total

    nonzero_probs = probs[probs > _EPSILON]

    if len(nonzero_probs) == 0:
        return 0.0

    log_probs = np.log(nonzero_probs) / np.log(base)
    entropy = -np.sum(nonzero_probs * log_probs)

    return float(max(0.0, entropy))


def _compute_distribution_from_counts(
    counts: Union[Dict[Any, int], Counter]
) -> List[float]:
    """Convierte conteos a distribución de probabilidad."""
    if not counts:
        return []

    values = list(counts.values())
    total = sum(values)

    if total == 0:
        return []

    return [v / total for v in values]


def _compute_persistence_entropy(
    intervals: Sequence[PersistenceInterval]
) -> float:
    """Calcula la entropía del diagrama de persistencia."""
    if not intervals:
        return 0.0

    finite_intervals = [iv for iv in intervals if not iv.is_essential]

    if not finite_intervals:
        return 0.0

    persistences = [iv.persistence for iv in finite_intervals]
    total_persistence = sum(persistences)

    if total_persistence < _EPSILON:
        return 0.0

    probs = [p / total_persistence for p in persistences]

    raw_entropy = _compute_shannon_entropy(probs)
    max_entropy = math.log2(len(probs)) if len(probs) > 1 else 1.0

    return raw_entropy / max_entropy if max_entropy > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE TOPOLOGÍA ALGEBRAICA (Propuesta 2)
# ══════════════════════════════════════════════════════════════════════════════

def _detect_cyclic_patterns(lines: List[str], max_period: int = 50) -> int:
    """
    Detecta patrones cíclicos en una secuencia de líneas.
    """
    if len(lines) < 3:
        return 0

    cycles_found = 0

    for period in range(1, min(max_period, len(lines) // 2)):
        matches = 0
        comparisons = 0

        for i in range(len(lines) - period):
            comparisons += 1
            if lines[i] == lines[i + period]:
                matches += 1

        if comparisons > 0:
            match_ratio = matches / comparisons
            if match_ratio > 0.8:
                cycles_found += 1

    return cycles_found


def _estimate_intrinsic_dimension(lines: List[str]) -> int:
    """
    Estima la dimensión intrínseca del espacio de datos.
    """
    if not lines:
        return 0

    sample = lines[0] if lines else ""

    for delimiter in [',', ';', '\t', '|']:
        if delimiter in sample:
            col_counts = [len(line.split(delimiter)) for line in lines[:100]]
            if col_counts:
                return max(set(col_counts), key=col_counts.count)

    return 1


def _analyze_topological_features(file_path: Path) -> Dict[str, Any]:
    """
    Analiza características topológicas de un archivo.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = [line.rstrip('\n\r') for line in f.readlines()[:_MAX_SAMPLE_ROWS]]

        if not lines:
            return TopologicalSummary.empty().to_dict()

        line_counts = Counter(lines)
        num_unique_patterns = len(line_counts)

        beta_0 = num_unique_patterns
        beta_1 = _detect_cyclic_patterns(lines)
        dimension = _estimate_intrinsic_dimension(lines)
        euler_char = beta_0 - beta_1

        distribution = _compute_distribution_from_counts(line_counts)
        structural_entropy = _compute_shannon_entropy(distribution)

        return {
            "beta_0": beta_0,
            "beta_1": beta_1,
            "beta_2": 0,
            "euler_characteristic": euler_char,
            "intrinsic_dimension": dimension,
            "structural_entropy": round(structural_entropy, 6),
            "num_lines": len(lines),
            "num_unique_patterns": num_unique_patterns,
            "repetition_ratio": 1.0 - (num_unique_patterns / len(lines)) if lines else 0.0
        }

    except Exception as e:
        logger.warning(f"Topological analysis failed: {e}")
        return {
            **TopologicalSummary.empty().to_dict(),
            "analysis_error": str(e)
        }


def _compute_homology_groups(diagnostic_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calcula grupos de homología a partir de datos diagnósticos.
    """
    issues = diagnostic_data.get("issues", [])
    warnings = diagnostic_data.get("warnings", [])
    errors = diagnostic_data.get("errors", [])

    issue_types: Set[str] = set()
    for issue in issues:
        if isinstance(issue, dict):
            issue_types.add(issue.get("type", issue.get("code", "unknown")))
        else:
            issue_types.add(str(type(issue).__name__))

    beta_0 = max(1, len(issue_types))

    circular_keywords = {"circular", "cycle", "loop", "recursive", "dependency"}

    def has_circular_reference(item: Any) -> bool:
        text = str(item).lower()
        return any(kw in text for kw in circular_keywords)

    circular_in_warnings = sum(1 for w in warnings if has_circular_reference(w))
    circular_in_issues = sum(1 for i in issues if has_circular_reference(i))

    beta_1 = circular_in_warnings + circular_in_issues
    euler_char = beta_0 - beta_1
    total_rank = beta_0 + beta_1

    return {
        "H_0": f"ℤ^{beta_0}",
        "H_1": f"ℤ^{beta_1}",
        "beta_0": beta_0,
        "beta_1": beta_1,
        "betti_numbers": [beta_0, beta_1, 0],
        "euler_characteristic": euler_char,
        "total_rank": total_rank,
        "is_connected": beta_0 == 1,
        "has_cycles": beta_1 > 0
    }


def _compute_persistence_diagram(
    diagnostic_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Calcula diagrama de persistencia para características topológicas.
    """
    issues = diagnostic_data.get("issues", [])

    if not issues:
        return []

    severity_map: Dict[str, float] = {
        "CRITICAL": 1.0,
        "HIGH": 0.8,
        "MEDIUM": 0.5,
        "LOW": 0.2,
        "INFO": 0.1,
    }

    intervals: List[PersistenceInterval] = []

    for idx, issue in enumerate(issues):
        if isinstance(issue, dict):
            raw_severity = issue.get("severity", "MEDIUM")
            if isinstance(raw_severity, str):
                severity = severity_map.get(raw_severity.upper(), 0.5)
            elif isinstance(raw_severity, (int, float)):
                severity = float(raw_severity)
            else:
                severity = 0.5
        else:
            severity = 0.5

        birth = idx * 0.1
        death = birth + severity

        try:
            interval = PersistenceInterval(
                birth=birth,
                death=death,
                dimension=0
            )
            intervals.append(interval)
        except ValueError:
            continue

    significant_intervals = [
        iv for iv in intervals
        if iv.persistence >= _PERSISTENCE_THRESHOLD
    ]

    significant_intervals.sort()

    return [
        {
            "birth": round(iv.birth, 4),
            "death": round(iv.death, 4),
            "persistence": round(iv.persistence, 4),
            "dimension": iv.dimension,
            "is_significant": iv.persistence >= 0.5
        }
        for iv in significant_intervals
    ]


def _compute_diagnostic_magnitude(diagnostic_data: Dict[str, Any]) -> float:
    """Calcula la magnitud normalizada del vector diagnóstico."""
    n_issues = len(diagnostic_data.get("issues", []))
    n_warnings = len(diagnostic_data.get("warnings", []))
    n_errors = len(diagnostic_data.get("errors", []))

    w_errors = 3.0
    w_issues = 2.0
    w_warnings = 1.0

    raw_magnitude = math.sqrt(
        w_errors * (n_errors ** 2) +
        w_issues * (n_issues ** 2) +
        w_warnings * (n_warnings ** 2)
    )

    normalization_factor = 10.0
    normalized = math.tanh(raw_magnitude / normalization_factor)

    return round(normalized, 4)


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE ANÁLISIS CSV (Propuesta 2)
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_effective_rank(df: pd.DataFrame, threshold: float = 0.95) -> int:
    """Estima el rango efectivo de un DataFrame usando SVD."""
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty or numeric_df.shape[1] < 2:
        return numeric_df.shape[1] if not numeric_df.empty else 0

    try:
        filled = numeric_df.fillna(numeric_df.mean())
        std = filled.std()
        std[std == 0] = 1
        normalized = (filled - filled.mean()) / std

        _, singular_values, _ = np.linalg.svd(normalized.values, full_matrices=False)

        variance_explained = singular_values ** 2
        total_variance = variance_explained.sum()

        if total_variance < _EPSILON:
            return 0

        cumulative_variance = np.cumsum(variance_explained) / total_variance
        effective_rank = int(np.searchsorted(cumulative_variance, threshold) + 1)

        return min(effective_rank, len(singular_values))

    except Exception:
        return numeric_df.shape[1]


def _analyze_csv_topology(
    path: Path,
    delimiter: str,
    encoding: str
) -> Dict[str, Any]:
    """Analiza la topología estructural de un archivo CSV."""
    try:
        df = pd.read_csv(
            path,
            sep=delimiter,
            encoding=encoding,
            nrows=_MAX_SAMPLE_ROWS,
            on_bad_lines='skip'
        )

        if df.empty:
            return {
                "rows": 0,
                "columns": 0,
                "density": 0.0,
                "null_entropy": 0.0,
                "effective_rank": 0,
                "is_empty": True
            }

        n_rows, n_cols = df.shape
        total_cells = n_rows * n_cols
        non_null_cells = df.notna().sum().sum()
        density = non_null_cells / total_cells if total_cells > 0 else 0.0

        null_ratios = df.isnull().mean().values
        null_ratios_safe = np.clip(null_ratios, _EPSILON, 1.0 - _EPSILON)
        binary_entropies = -(
            null_ratios_safe * np.log2(null_ratios_safe) +
            (1 - null_ratios_safe) * np.log2(1 - null_ratios_safe)
        )
        null_entropy = float(np.mean(binary_entropies))

        effective_rank = _estimate_effective_rank(df)

        dtype_counts = Counter(str(dtype) for dtype in df.dtypes)
        dtype_probs = _compute_distribution_from_counts(dtype_counts)
        type_entropy = _compute_shannon_entropy(dtype_probs)

        return {
            "rows": n_rows,
            "columns": n_cols,
            "density": round(density, 4),
            "null_entropy": round(null_entropy, 4),
            "effective_rank": effective_rank,
            "type_entropy": round(type_entropy, 4),
            "dimensionality": n_cols,
            "sparsity": round(1.0 - density, 4),
            "is_empty": False
        }

    except pd.errors.EmptyDataError:
        return {"rows": 0, "columns": 0, "is_empty": True, "error": "Empty file"}
    except Exception as e:
        logger.warning(f"CSV topology analysis failed: {e}")
        return {"error": str(e), "is_empty": None}


def _compute_topological_preservation(
    initial: Dict[str, Any],
    final: Dict[str, Any]
) -> Dict[str, Any]:
    """Compara topologías inicial y final para medir preservación."""
    if initial.get("error") or final.get("error"):
        return {
            "preservation_rate": 0.0,
            "is_valid": False,
            "error": initial.get("error") or final.get("error")
        }

    if initial.get("is_empty") or final.get("is_empty"):
        return {
            "preservation_rate": 0.0 if final.get("is_empty") else 1.0,
            "is_valid": True,
            "note": "Empty file involved"
        }

    init_rows = initial.get("rows", 0)
    final_rows = final.get("rows", 0)

    if init_rows == 0:
        row_preservation = 1.0 if final_rows == 0 else 0.0
    else:
        row_preservation = min(final_rows / init_rows, 1.0)

    init_cols = initial.get("columns", 0)
    final_cols = final.get("columns", 0)

    if init_cols == 0:
        col_preservation = 1.0 if final_cols == 0 else 0.0
    else:
        col_preservation = min(final_cols / init_cols, 1.0)

    init_density = initial.get("density", 0.0)
    final_density = final.get("density", 0.0)
    density_delta = final_density - init_density

    init_entropy = initial.get("null_entropy", 0.0)
    final_entropy = final.get("null_entropy", 0.0)
    entropy_delta = final_entropy - init_entropy

    preservation_rate = 0.6 * row_preservation + 0.4 * col_preservation

    if density_delta > 0:
        preservation_rate = min(1.0, preservation_rate + 0.1 * density_delta)

    return {
        "preservation_rate": round(preservation_rate, 4),
        "row_preservation": round(row_preservation, 4),
        "column_preservation": round(col_preservation, 4),
        "density_delta": round(density_delta, 4),
        "entropy_delta": round(entropy_delta, 4),
        "is_valid": True,
        "improved_density": density_delta > 0,
        "reduced_entropy": entropy_delta < 0
    }


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES FINANCIERAS CON ANÁLISIS TOPOLÓGICO (Propuesta 2)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_topological_cash_flows(
    amount: float,
    time_years: int,
    std_dev: float,
    *,
    random_seed: Optional[int] = None,
    decay_rate: float = 0.02
) -> List[float]:
    """Genera flujos de caja con dinámica temporal realista."""
    rng = np.random.default_rng(random_seed or _DEFAULT_RANDOM_SEED)

    if amount <= 0 or time_years <= 0:
        return []

    base_return = 0.20
    flows: List[float] = []

    for t in range(time_years):
        effective_rate = base_return * math.exp(-decay_rate * t)
        base_flow = amount * effective_rate

        if std_dev > 0:
            noise_factor = rng.lognormal(mean=0, sigma=std_dev / amount)
            noise_factor = np.clip(noise_factor, 0.5, 2.0)
        else:
            noise_factor = 1.0

        flow = base_flow * noise_factor
        flows.append(float(flow))

    return flows


def _analyze_risk_manifold(
    amount: float,
    std_dev: float,
    time_years: int,
    flows: List[float]
) -> Dict[str, Any]:
    """Construye una variedad de riesgo para análisis topológico."""
    if not flows or amount <= 0:
        return {
            "dimension": 0,
            "curvature": 0.0,
            "volatility_surface": 0.0,
            "flow_stability": 0.0,
            "is_degenerate": True,
            "local_extrema": 0, # Added for tests
            "mean_flow": 0.0 # Added for tests
        }

    flows_array = np.array(flows)
    realized_std = np.std(flows_array)
    volatility_surface = realized_std / amount if amount > 0 else 0.0

    mean_flow = np.mean(flows_array)
    if mean_flow > _EPSILON:
        cv = realized_std / mean_flow
        stability = 1.0 / (1.0 + cv)
    else:
        stability = 0.0

    if len(flows) >= 3:
        second_diff = np.diff(flows_array, n=2)
        curvature = float(np.mean(np.abs(second_diff))) / (amount + _EPSILON)
    else:
        curvature = 0.0

    if len(set(flows)) == 1:
        effective_dimension = 1
    else:
        effective_dimension = 2

    local_extrema = 0
    for i in range(1, len(flows) - 1):
        if (flows[i] > flows[i-1] and flows[i] > flows[i+1]) or \
           (flows[i] < flows[i-1] and flows[i] < flows[i+1]):
            local_extrema += 1

    return {
        "dimension": effective_dimension,
        "curvature": round(curvature, 6),
        "volatility_surface": round(volatility_surface, 6),
        "flow_stability": round(stability, 4),
        "local_extrema": local_extrema,
        "is_degenerate": effective_dimension < 2,
        "mean_flow": round(mean_flow, 2),
        "realized_volatility": round(realized_std, 2)
    }


def _compute_risk_homology(manifold: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula invariantes homológicos del espacio de riesgo."""
    if manifold.get("is_degenerate", True):
        return {
            "risk_holes_beta_1": 0,
            "risk_regimes_beta_0": 1,
            "euler_characteristic": 1,
            "interpretation": "Degenerate risk space - insufficient data"
        }

    stability = manifold.get("flow_stability", 0.5)
    local_extrema = manifold.get("local_extrema", 0)
    volatility = manifold.get("volatility_surface", 0.0)

    instability = 1.0 - stability
    risk_holes = int(instability * 5 + local_extrema * 0.5)

    risk_regimes = 1 + int(volatility * 10)
    risk_regimes = min(risk_regimes, 5)

    euler = risk_regimes - risk_holes

    if risk_holes == 0:
        interpretation = "Solid risk surface - well-protected"
    elif risk_holes <= 2:
        interpretation = "Minor risk exposures - manageable"
    elif risk_holes <= 4:
        interpretation = "Significant risk gaps - requires hedging"
    else:
        interpretation = "Critical risk topology - high exposure"

    return {
        "risk_holes_beta_1": risk_holes,
        "risk_regimes_beta_0": risk_regimes,
        "euler_characteristic": euler,
        "interpretation": interpretation,
        "raw_instability": round(instability, 4)
    }


def _compute_opportunity_persistence(manifold: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Calcula la persistencia de oportunidades en el espacio financiero."""
    mean_flow = manifold.get("mean_flow", 0.0)
    stability = manifold.get("flow_stability", 0.5)
    volatility = manifold.get("volatility_surface", 0.1)
    local_extrema = manifold.get("local_extrema", 0)

    intervals = []

    if mean_flow > 0:
        death_time = 1.0 / (1.0 + volatility) if volatility > 0 else float('inf')
        intervals.append({
            "birth": 0.0,
            "death": round(death_time, 4) if not math.isinf(death_time) else "inf",
            "persistence": round(death_time, 4) if not math.isinf(death_time) else "inf",
            "type": "primary_opportunity",
            "strength": round(mean_flow, 2)
        })

    for i in range(local_extrema):
        birth = 0.1 * (i + 1)
        death = birth + 0.3 * stability
        intervals.append({
            "birth": round(birth, 4),
            "death": round(death, 4),
            "persistence": round(death - birth, 4),
            "type": "secondary_opportunity",
            "index": i
        })

    intervals.sort(key=lambda x: -(x["persistence"] if isinstance(x["persistence"], (int, float)) else float('inf')))

    return intervals


def _compute_risk_adjusted_return(
    analysis: Dict[str, Any],
    risk_tolerance: float
) -> float:
    """Calcula el retorno ajustado al riesgo usando Sharpe Ratio modificado."""
    npv = analysis.get("npv", 0.0)

    if npv is None:
        return 0.0

    risk_free_rate = 0.03
    volatility = analysis.get("volatility", 0.30)
    time_years = analysis.get("project_life_years", 5)

    risk_aversion = max(0.01, 1.0 - risk_tolerance)
    denominator = volatility * math.sqrt(time_years) * (1 + risk_aversion)

    if denominator < _EPSILON:
        return float(npv)

    excess_return = npv - (risk_free_rate * npv)
    risk_adjusted = excess_return / denominator

    return round(risk_adjusted, 4)


def _compute_topological_efficiency(
    analysis: Dict[str, Any],
    manifold: Dict[str, Any]
) -> float:
    """Calcula la eficiencia topológica del proyecto."""
    npv = analysis.get("npv", 0.0) or 0.0
    curvature = manifold.get("curvature", 0.0)
    stability = manifold.get("flow_stability", 0.5)

    instability = 1.0 - stability
    roughness = curvature + instability

    normalized_npv = 1.0 / (1.0 + math.exp(-npv / 10000)) if npv != 0 else 0.5

    efficiency = normalized_npv / (1.0 + roughness)

    return round(efficiency, 4)


# ══════════════════════════════════════════════════════════════════════════════
# HANDLERS DE LA MIC REFINADOS (Propuesta 2)
# ══════════════════════════════════════════════════════════════════════════════

def diagnose_file(
    file_path: Union[str, Path],
    file_type: Union[str, FileType],
    *,
    validate_extension: bool = True,
    max_file_size: Optional[int] = None,
    topological_analysis: bool = False
) -> Dict[str, Any]:
    """
    Vector de Diagnóstico con análisis topológico opcional (Nivel 3 - PHYSICS).
    """
    path_str = str(file_path)

    try:
        # ─── Fase 1: Validación ───
        path = _normalize_path(file_path)
        normalized_type = _normalize_file_type(file_type)

        logger.info(f"Initiating diagnosis: path={path}, type={normalized_type.value}")

        _validate_file_exists(path)

        if validate_extension:
            _validate_file_extension(path)

        effective_max = max_file_size or MAX_FILE_SIZE_BYTES
        size, is_empty = _validate_file_size(path, effective_max)

        if is_empty:
            logger.warning(f"File is empty: {path}")
            return _create_success_response(
                {"diagnostic_completed": True, "is_empty": True},
                file_type=normalized_type.value,
                file_path=str(path),
                file_size_bytes=0,
                warning="File is empty"
            )

        # ─── Fase 2: Diagnóstico Base ───
        diagnostic_class = _get_diagnostic_class(normalized_type)
        diagnostic = diagnostic_class(str(path))
        diagnostic.diagnose()

        result_data = _extract_diagnostic_result(diagnostic)

        # ─── Fase 3: Análisis Topológico Opcional ───
        if topological_analysis:
            logger.debug("Executing topological analysis")

            # Características topológicas del archivo
            topo_features = _analyze_topological_features(path)
            result_data["topological_features"] = topo_features

            # Homología basada en issues encontrados
            homology = _compute_homology_groups(result_data)
            result_data["homology"] = homology

            # Diagrama de persistencia
            persistence = _compute_persistence_diagram(result_data)
            result_data["persistence_diagram"] = persistence

            # Entropía de persistencia
            if persistence:
                intervals = [
                    PersistenceInterval(
                        birth=p["birth"],
                        death=p["death"],
                        dimension=p.get("dimension", 0)
                    )
                    for p in persistence
                ]
                result_data["persistence_entropy"] = _compute_persistence_entropy(intervals)

        # ─── Fase 4: Métricas Globales ───
        magnitude = _compute_diagnostic_magnitude(result_data)

        logger.info(f"Diagnosis completed: path={path}, magnitude={magnitude}")

        return _create_success_response(
            result_data,
            file_type=normalized_type.value,
            file_path=str(path),
            file_size_bytes=size,
            diagnostic_magnitude=magnitude,
            has_topological_analysis=topological_analysis
        )

    except FileNotFoundDiagnosticError as e:
        logger.warning(f"File not found: {path_str}")
        return _create_error_response(e, error_category="validation")

    except (UnsupportedFileTypeError, FileValidationError) as e:
        logger.warning(f"Validation error: {e}")
        return _create_error_response(e, error_category="validation")

    except (ValueError, TypeError) as e:
        logger.warning(f"Parameter error: {e}")
        return _create_error_response(e, error_category="validation")

    except IOError as e:
        logger.error(f"IO error: {e}")
        return _create_error_response(e, error_category="io_error")

    except Exception as e:
        logger.exception(f"Unexpected error in diagnosis: {e}")
        return _create_error_response(e, error_category="unexpected")


def clean_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    *,
    delimiter: str = ";",
    encoding: str = "utf-8",
    overwrite: bool = True,
    validate_extension: bool = True,
    max_file_size: Optional[int] = None,
    preserve_topology: bool = True,
    compression_ratio: Optional[float] = None
) -> Dict[str, Any]:
    """
    Vector de Limpieza con preservación topológica (Nivel 3 - PHYSICS).
    """
    input_str = str(input_path)

    try:
        # ─── Fase 1: Validación de Entrada ───
        input_p = _normalize_path(input_path)

        logger.info(f"Initiating cleaning: {input_p}")

        _validate_file_exists(input_p)

        if validate_extension:
            _validate_file_extension(input_p)

        validated_delimiter, validated_encoding = _validate_csv_parameters(
            delimiter, encoding
        )

        # ─── Fase 2: Análisis Topológico Pre-Limpieza ───
        initial_topology: Dict[str, Any] = {}
        if preserve_topology:
            initial_topology = _analyze_csv_topology(
                input_p, validated_delimiter, validated_encoding
            )
            logger.debug(f"Initial topology: {initial_topology}")

        # ─── Fase 3: Configuración de Salida ───
        if output_path is None:
            output_p = _generate_output_path(input_p)
        else:
            output_p = _normalize_path(output_path)

        # Validar que input != output
        try:
            if input_p.resolve() == output_p.resolve():
                raise ValueError(
                    "Output path cannot be the same as input path. "
                    "Use a different filename or directory."
                )
        except OSError:
            # Si resolve() falla, comparar strings
            if str(input_p) == str(output_p):
                raise ValueError("Output path cannot be the same as input path")

        # Verificar sobrescritura
        if output_p.exists() and not overwrite:
            raise ValueError(
                f"Output file exists and overwrite=False: {output_p}"
            )

        # Crear directorio de salida si no existe
        output_p.parent.mkdir(parents=True, exist_ok=True)

        # ─── Fase 4: Ejecución de Limpieza ───
        cleaner = CSVCleaner(
            input_path=str(input_p),
            output_path=str(output_p),
            delimiter=validated_delimiter,
            encoding=validated_encoding,
            overwrite=overwrite
        )

        stats = cleaner.clean()

        # Normalizar estadísticas
        if isinstance(stats, dict):
            stats_dict = stats
        elif hasattr(stats, "to_dict") and callable(stats.to_dict):
            stats_dict = stats.to_dict()
        else:
            stats_dict = {"raw_result": str(stats)}

        # ─── Fase 5: Análisis Topológico Post-Limpieza ───
        if preserve_topology and not initial_topology.get("error"):
            final_topology = _analyze_csv_topology(
                output_p, validated_delimiter, validated_encoding
            )

            preservation = _compute_topological_preservation(
                initial_topology, final_topology
            )

            stats_dict["topological_preservation"] = preservation
            stats_dict["initial_topology"] = {
                "rows": initial_topology.get("rows"),
                "columns": initial_topology.get("columns"),
                "density": initial_topology.get("density")
            }
            stats_dict["final_topology"] = {
                "rows": final_topology.get("rows"),
                "columns": final_topology.get("columns"),
                "density": final_topology.get("density")
            }

            # Advertir si la preservación es baja
            pres_rate = preservation.get("preservation_rate", 1.0)
            if pres_rate < 0.8:
                logger.warning(
                    f"Low topological preservation ({pres_rate:.2%}). "
                    f"Significant data loss may have occurred."
                )

        logger.info(f"Cleaning completed: {output_p}")

        return _create_success_response(
            stats_dict,
            input_path=str(input_p),
            output_path=str(output_p),
            preserved_topology=preserve_topology
        )

    except FileNotFoundDiagnosticError as e:
        logger.warning(f"File not found: {input_str}")
        return _create_error_response(e, error_category="validation")

    except (FileValidationError, ValueError) as e:
        logger.warning(f"Validation error: {e}")
        return _create_error_response(e, error_category="validation")

    except CleaningError as e:
        logger.error(f"Cleaning error: {e}")
        return _create_error_response(e, error_category="cleaning")

    except IOError as e:
        logger.error(f"IO error during cleaning: {e}")
        wrapped = CleaningError(
            f"IO error during cleaning: {e}",
            details={"original_error": str(e), "input_path": input_str}
        )
        return _create_error_response(wrapped, error_category="cleaning")

    except Exception as e:
        logger.exception(f"Unexpected error in cleaning: {e}")
        return _create_error_response(e, error_category="unexpected")


def analyze_financial_viability(
    amount: float,
    std_dev: float,
    time_years: int,
    *,
    risk_tolerance: float = 0.05,
    market_volatility: Optional[float] = None,
    topological_risk_analysis: bool = False,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Vector de Análisis Financiero con riesgo topológico (Nivel 1 - STRATEGY).
    """
    try:
        # ─── Fase 1: Validación de Parámetros ───
        if not isinstance(amount, (int, float)):
            raise TypeError(f"amount must be numeric, got {type(amount).__name__}")
        if amount <= 0:
            raise ValueError("Amount must be positive")

        if not isinstance(std_dev, (int, float)):
            raise TypeError(f"std_dev must be numeric, got {type(std_dev).__name__}")
        if std_dev < 0:
            raise ValueError("Standard deviation cannot be negative")

        if not isinstance(time_years, int):
            raise TypeError(f"time_years must be integer, got {type(time_years).__name__}")
        if time_years <= 0:
            raise ValueError("Time horizon must be positive")

        if not 0 <= risk_tolerance <= 1:
            raise ValueError("risk_tolerance must be in [0, 1]")

        logger.info(
            f"Analyzing financial viability: amount={amount}, "
            f"std_dev={std_dev}, years={time_years}"
        )

        # ─── Fase 2: Generación de Flujos ───
        cash_flows = _generate_topological_cash_flows(
            amount=amount,
            time_years=time_years,
            std_dev=std_dev,
            random_seed=random_seed
        )

        # ─── Fase 3: Análisis Base ───
        config = FinancialConfig(project_life_years=time_years)
        engine = FinancialEngine(config)

        analysis = engine.analyze_project(
            initial_investment=amount,
            expected_cash_flows=cash_flows,
            cost_std_dev=std_dev,
            project_volatility=market_volatility or 0.30
        )

        # ─── Fase 4: Extracción de Resultados ───
        performance = analysis.get("performance", {})

        results: Dict[str, Any] = {
            "wacc": analysis.get("wacc"),
            "npv": analysis.get("npv"),
            "irr": analysis.get("irr"),
            "payback_years": analysis.get("payback_years"),
            "recommendation": performance.get("recommendation"),
            "risk_level": performance.get("risk_level"),
            "cash_flows_summary": {
                "mean": round(np.mean(cash_flows), 2) if cash_flows else 0,
                "std": round(np.std(cash_flows), 2) if cash_flows else 0,
                "min": round(min(cash_flows), 2) if cash_flows else 0,
                "max": round(max(cash_flows), 2) if cash_flows else 0,
            }
        }

        # ─── Fase 5: Métricas de Riesgo Ajustado ───
        results["risk_adjusted_return"] = _compute_risk_adjusted_return(
            analysis, risk_tolerance
        )

        # ─── Fase 6: Análisis Topológico Opcional ───
        if topological_risk_analysis:
            logger.debug("Executing topological risk analysis")

            # Variedad de riesgo
            risk_manifold = _analyze_risk_manifold(
                amount, std_dev, time_years, cash_flows
            )
            results["risk_manifold"] = risk_manifold

            # Homología del riesgo
            risk_homology = _compute_risk_homology(risk_manifold)
            results["topological_risk"] = risk_homology

            # Persistencia de oportunidades
            opportunity_persistence = _compute_opportunity_persistence(risk_manifold)
            results["opportunity_persistence"] = opportunity_persistence

            # Eficiencia topológica
            results["topological_efficiency"] = _compute_topological_efficiency(
                analysis, risk_manifold
            )

        logger.info(
            f"Financial analysis complete: NPV={results.get('npv')}, "
            f"recommendation={results.get('recommendation')}"
        )

        return {
            "success": True,
            "results": results,
            "parameters": {
                "initial_investment": amount,
                "std_dev": std_dev,
                "time_years": time_years,
                "risk_tolerance": risk_tolerance,
                "random_seed": random_seed or _DEFAULT_RANDOM_SEED
            }
        }

    except (ValueError, TypeError) as e:
        logger.warning(f"Validation error in financial analysis: {e}")
        return _create_error_response(e, error_category="validation")

    except Exception as e:
        logger.exception(f"Error in financial analysis: {e}")
        return _create_error_response(e, error_category="financial_analysis")


def get_telemetry_status(
    telemetry_context: Optional[TelemetryContextProtocol] = None
) -> Dict[str, Any]:
    """
    Vector de Telemetría del Sistema (Nivel 3 - PHYSICS/Observability).
    """
    # ─── Caso 1: Sin Contexto Activo ───
    if telemetry_context is None:
        return {
            "success": True,
            "status": "IDLE",
            "system_health": "UNKNOWN",
            "message": "No active processing context",
            "has_active_context": False
        }

    # ─── Caso 2: Contexto Inválido ───
    if not isinstance(telemetry_context, TelemetryContextProtocol):
        context_type = type(telemetry_context).__name__
        logger.warning(f"Invalid telemetry context type: {context_type}")
        return {
            "success": False,
            "status": "ERROR",
            "system_health": "DEGRADED",
            "error": f"Invalid context type: {context_type}",
            "message": (
                f"Telemetry context must implement TelemetryContextProtocol, "
                f"got {context_type}"
            ),
            "has_active_context": False
        }

    # ─── Caso 3: Contexto Válido ───
    try:
        report = telemetry_context.get_business_report()

        # Manejar reporte nulo
        if report is None:
            return {
                "success": True,
                "status": "ACTIVE",
                "system_health": "UNKNOWN",
                "message": "Context returned null report",
                "has_active_context": True
            }

        # Manejar reporte no-diccionario
        if not isinstance(report, dict):
            return {
                "success": True,
                "status": "ACTIVE",
                "system_health": "UNKNOWN",
                "has_active_context": True,
                "raw_report": report
            }

        # Construir respuesta con valores del reporte
        return {
            "success": True,
            "status": report.get("status", "ACTIVE"),
            "system_health": report.get("system_health", "HEALTHY"),
            "has_active_context": True,
            **{k: v for k, v in report.items() if k not in ("status", "system_health")}
        }

    except Exception as e:
        logger.exception(f"Telemetry error: {e}")
        return {
            "success": False,
            "status": "ERROR",
            "system_health": "DEGRADED",
            "error": str(e),
            "message": f"Error retrieving telemetry report: {e}",
            "has_active_context": True
        }

# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE VALIDACIÓN Y UTILIDADES (Restauradas)
# ══════════════════════════════════════════════════════════════════════════════

def _validate_path_not_empty(path: Union[str, Path]) -> None:
    if not path:
        raise ValueError("Path cannot be empty")
    if isinstance(path, str) and not path.strip():
        raise ValueError("Path cannot be empty")

def _normalize_path(path: Union[str, Path]) -> Path:
    if path is None:
        raise ValueError("Path cannot be None")
    _validate_path_not_empty(path)
    return Path(path).expanduser().resolve()

def _validate_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundDiagnosticError(f"File not found: {path}")
    if not path.is_file():
        raise FileValidationError(f"Path is not a file: {path}")

def _validate_file_extension(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in VALID_EXTENSIONS:
        raise FileValidationError(f"Invalid extension: {ext}. Expected: {VALID_EXTENSIONS}")
    return ext

def _validate_file_size(path: Path, max_size: int) -> Tuple[int, bool]:
    size = path.stat().st_size
    if size > max_size:
        raise FileValidationError(f"File size {size} exceeds limit {max_size}")
    return size, size == 0

def _normalize_encoding(encoding: str) -> str:
    norm = encoding.lower().replace("-", "").replace("_", "")
    for alias, standard in _ENCODING_ALIASES.items():
        if norm == alias.replace("-", ""):
            return standard
    if encoding.lower() in SUPPORTED_ENCODINGS:
        return encoding.lower()
    return "utf-8" # Default fallback

def _validate_csv_parameters(delimiter: str, encoding: str) -> Tuple[str, str]:
    if delimiter not in VALID_DELIMITERS:
        raise ValueError(f"Invalid delimiter: {delimiter}")
    norm_enc = _normalize_encoding(encoding)
    return delimiter, norm_enc

def _normalize_file_type(file_type: Union[str, FileType]) -> FileType:
    if isinstance(file_type, FileType):
        return file_type
    return FileType.from_string(file_type)

def _generate_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_clean{input_path.suffix}")

def _create_success_response(data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    return {"success": True, **data, **kwargs}

def _create_error_response(error: Union[Exception, str], error_category: str = "error", **kwargs) -> Dict[str, Any]:
    msg = str(error)
    details = getattr(error, "details", {}) if isinstance(error, Exception) else {}
    return {
        "success": False,
        "error": msg,
        "error_category": error_category,
        "error_details": details,
        "error_type": type(error).__name__ if isinstance(error, Exception) else "Error",
        **kwargs
    }

def _extract_diagnostic_result(diagnostic: DiagnosticProtocol) -> Dict[str, Any]:
    result = diagnostic.to_dict()
    result["diagnostic_completed"] = True
    return result

# Public Utilities

def get_supported_file_types() -> List[str]:
    return FileType.values()

def get_supported_delimiters() -> List[str]:
    return list(VALID_DELIMITERS)

def get_supported_encodings() -> List[str]:
    return list(SUPPORTED_ENCODINGS)

def is_valid_file_type(file_type: Union[str, FileType]) -> bool:
    try:
        _normalize_file_type(file_type)
        return True
    except ValueError:
        return False

def validate_file_for_processing(path: Path) -> Dict[str, Any]:
    try:
        p = _normalize_path(path)
        _validate_file_exists(p)
        ext = _validate_file_extension(p)
        size, empty = _validate_file_size(p, MAX_FILE_SIZE_BYTES)
        return {"valid": True, "size": size, "extension": ext, "is_empty": empty}
    except Exception as e:
        return {"valid": False, "errors": [str(e)]}
