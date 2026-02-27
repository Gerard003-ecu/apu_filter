"""
Módulo: Governance Engine (El Auditor de Datos)
================================================

Este componente implementa el motor de Gobernanza Computacional que asegura que
los "Productos de Datos" (presupuestos, APUs) cumplan con las leyes del sistema
antes de ser aceptados. Transforma la validación pasiva en una auditoría activa
basada en ontologías y reglas de negocio.

Arquitectura de Validación:
---------------------------

1. Política como Código (Policy as Code):
   Evalúa reglas declarativas para determinar cumplimiento de estándares.
   Genera un `ComplianceReport` con veredicto de severidad (PASS, WARNING, FAIL).

2. Validación Semántica (Ontology Check):
   Utiliza un Grafo de Conocimiento (Ontología) para verificar que los insumos
   pertenezcan al dominio correcto del APU. Infiere contexto semántico para
   detectar anomalías invisibles a la validación sintáctica.

3. Sistema de Penalización (Scorecard):
   Calcula un puntaje de gobernanza aplicando penalizaciones ponderadas.
   Puntaje bajo activa protocolos de rechazo automático o auditoría manual.

4. Gestión de Contratos de Datos:
   Verifica que estructura y contenido respeten contratos definidos
   (campos obligatorios, tipos de datos, restricciones de dominio).

Invariantes del Sistema:
------------------------
- Score ∈ [0.0, 100.0]
- Score < fail_threshold ⟹ status = FAIL
- error_count > 0 ⟹ status = FAIL
- ComplianceReport es inmutable después de construcción
"""

from __future__ import annotations

import json
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from functools import cached_property, lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    FrozenSet,
    Iterator,
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

import pandas as pd

from app.constants import ColumnNames


logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTES GLOBALES
# ============================================================================

# Tolerancia numérica
EPSILON: Final[float] = 1e-9

# Score máximo y mínimo
MAX_SCORE: Final[float] = 100.0
MIN_SCORE: Final[float] = 0.0

# Directorio de configuración por defecto
DEFAULT_CONFIG_DIR: Final[str] = "config"

# Archivos de configuración
ONTOLOGY_FILENAME: Final[str] = "ontology.json"
DATA_CONTRACT_FILENAME: Final[str] = "data_contract.yaml"


# ============================================================================
# EXCEPCIONES DEL DOMINIO
# ============================================================================


class GovernanceError(Exception):
    """Excepción base del motor de gobernanza."""
    pass


class ConfigurationError(GovernanceError):
    """Error de configuración del motor."""
    pass


class OntologyError(GovernanceError):
    """Error relacionado con la ontología."""
    pass


class OntologyLoadError(OntologyError):
    """Error al cargar la ontología."""
    
    def __init__(self, message: str, path: Optional[Path] = None):
        self.path = path
        super().__init__(message)


class OntologyValidationError(OntologyError):
    """Error de validación de estructura de ontología."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        self.field = field
        super().__init__(message)


class PolicyError(GovernanceError):
    """Error relacionado con políticas."""
    pass


class ValidationError(GovernanceError):
    """Error durante la validación de datos."""
    pass


class SchemaError(ValidationError):
    """Error de schema/estructura de datos."""
    
    def __init__(self, message: str, missing_columns: Optional[List[str]] = None):
        self.missing_columns = missing_columns or []
        super().__init__(message)


# ============================================================================
# ENUMERACIONES
# ============================================================================


class Severity(str, Enum):
    """
    Niveles de severidad para violaciones de gobernanza.
    
    Forma un orden total por criticidad:
    INFO < WARNING < ERROR
    
    Propiedades:
    - penalty: Penalización asociada al score
    - is_blocking: Si bloquea el procesamiento
    - emoji: Representación visual
    """

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    
    @property
    def penalty(self) -> float:
        """Penalización asociada a este nivel de severidad."""
        return _SEVERITY_PENALTIES.get(self, 0.0)
    
    @property
    def is_blocking(self) -> bool:
        """Indica si esta severidad bloquea el procesamiento."""
        return self == Severity.ERROR
    
    @property
    def priority(self) -> int:
        """Prioridad numérica (mayor = más severo)."""
        return {
            Severity.INFO: 0,
            Severity.WARNING: 1,
            Severity.ERROR: 2,
        }[self]
    
    @property
    def emoji(self) -> str:
        """Representación visual."""
        return {
            Severity.INFO: "ℹ️",
            Severity.WARNING: "⚠️",
            Severity.ERROR: "❌",
        }[self]
    
    def __lt__(self, other: Severity) -> bool:
        return self.priority < other.priority
    
    def __le__(self, other: Severity) -> bool:
        return self.priority <= other.priority
    
    @classmethod
    def from_string(cls, value: str) -> Severity:
        """
        Parsea severidad desde string con normalización.
        
        Args:
            value: String a parsear.
            
        Returns:
            Severidad correspondiente, ERROR si es inválido.
        """
        if not value:
            return cls.ERROR
        
        normalized = value.upper().strip()
        try:
            return cls(normalized)
        except ValueError:
            logger.warning(f"Severidad inválida '{value}', usando ERROR")
            return cls.ERROR
    
    @classmethod
    def max_severity(cls, *severities: Severity) -> Severity:
        """Retorna la severidad máxima del conjunto."""
        if not severities:
            return cls.INFO
        return max(severities, key=lambda s: s.priority)


class ComplianceStatus(str, Enum):
    """
    Estados posibles del reporte de cumplimiento.
    
    Forma un orden total por severidad:
    PASS < WARNING < FAIL
    """

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    
    @property
    def is_successful(self) -> bool:
        """Indica si el estado representa éxito."""
        return self == ComplianceStatus.PASS
    
    @property
    def requires_attention(self) -> bool:
        """Indica si el estado requiere atención."""
        return self != ComplianceStatus.PASS
    
    @property
    def is_blocking(self) -> bool:
        """Indica si el estado bloquea el procesamiento."""
        return self == ComplianceStatus.FAIL
    
    @property
    def emoji(self) -> str:
        """Representación visual."""
        return {
            ComplianceStatus.PASS: "✅",
            ComplianceStatus.WARNING: "⚠️",
            ComplianceStatus.FAIL: "❌",
        }[self]
    
    @property
    def priority(self) -> int:
        """Prioridad numérica (mayor = más severo)."""
        return {
            ComplianceStatus.PASS: 0,
            ComplianceStatus.WARNING: 1,
            ComplianceStatus.FAIL: 2,
        }[self]
    
    def __lt__(self, other: ComplianceStatus) -> bool:
        return self.priority < other.priority
    
    @classmethod
    def from_score_and_errors(
        cls,
        score: float,
        error_count: int,
        warning_count: int,
        thresholds: ScoreThresholds,
    ) -> ComplianceStatus:
        """
        Determina el status basado en score y contadores.
        
        Args:
            score: Score actual.
            error_count: Número de errores.
            warning_count: Número de advertencias.
            thresholds: Umbrales de score.
            
        Returns:
            Status correspondiente.
        """
        if error_count > 0 or score < thresholds.fail_threshold:
            return cls.FAIL
        if warning_count > 0 or score < thresholds.warning_threshold:
            return cls.WARNING
        return cls.PASS


class ViolationType(str, Enum):
    """Tipos de violaciones de gobernanza."""
    
    SEMANTIC_INCONSISTENCY = "SEMANTIC_INCONSISTENCY"
    SEMANTIC_INCOMPLETENESS = "SEMANTIC_INCOMPLETENESS"
    SCHEMA_ERROR = "SCHEMA_ERROR"
    TYPE_ERROR = "TYPE_ERROR"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    ONTOLOGY_MISMATCH = "ONTOLOGY_MISMATCH"
    CUSTOM = "CUSTOM"
    
    @classmethod
    def from_string(cls, value: str) -> ViolationType:
        """Parsea desde string."""
        try:
            return cls(value.upper().strip())
        except ValueError:
            return cls.CUSTOM


# ============================================================================
# CONFIGURACIÓN
# ============================================================================


@dataclass(frozen=True)
class ScoreThresholds:
    """
    Umbrales para determinar el status basado en score.
    
    Invariantes:
    - 0 ≤ fail_threshold < warning_threshold ≤ MAX_SCORE
    """
    
    fail_threshold: float = 70.0
    warning_threshold: float = 90.0
    
    def __post_init__(self) -> None:
        """Valida invariantes."""
        if not (MIN_SCORE <= self.fail_threshold < self.warning_threshold <= MAX_SCORE):
            raise ConfigurationError(
                f"Umbrales inválidos: debe cumplir "
                f"{MIN_SCORE} ≤ fail({self.fail_threshold}) < "
                f"warning({self.warning_threshold}) ≤ {MAX_SCORE}"
            )
    
    def classify_score(self, score: float) -> ComplianceStatus:
        """Clasifica un score en status."""
        if score < self.fail_threshold:
            return ComplianceStatus.FAIL
        if score < self.warning_threshold:
            return ComplianceStatus.WARNING
        return ComplianceStatus.PASS


@dataclass(frozen=True)
class SeverityPenalties:
    """
    Penalizaciones por nivel de severidad.
    
    Inmutable para garantizar consistencia.
    """
    
    error_penalty: float = 5.0
    warning_penalty: float = 1.0
    info_penalty: float = 0.0
    
    def __post_init__(self) -> None:
        """Valida que las penalizaciones sean no negativas."""
        for name, value in [
            ("error", self.error_penalty),
            ("warning", self.warning_penalty),
            ("info", self.info_penalty),
        ]:
            if value < 0:
                raise ConfigurationError(
                    f"Penalización de {name} no puede ser negativa: {value}"
                )
    
    def get_penalty(self, severity: Severity) -> float:
        """Obtiene la penalización para una severidad."""
        return {
            Severity.ERROR: self.error_penalty,
            Severity.WARNING: self.warning_penalty,
            Severity.INFO: self.info_penalty,
        }.get(severity, 0.0)
    
    def to_dict(self) -> Dict[Severity, float]:
        """Convierte a diccionario."""
        return {
            Severity.ERROR: self.error_penalty,
            Severity.WARNING: self.warning_penalty,
            Severity.INFO: self.info_penalty,
        }


# Configuración global por defecto (inmutable)
_DEFAULT_PENALTIES: Final[SeverityPenalties] = SeverityPenalties()
_DEFAULT_THRESHOLDS: Final[ScoreThresholds] = ScoreThresholds()

# Diccionario de penalizaciones para acceso rápido en Severity.penalty
_SEVERITY_PENALTIES: Final[Dict[Severity, float]] = _DEFAULT_PENALTIES.to_dict()


@dataclass(frozen=True)
class SemanticPolicyConfig:
    """Configuración de política semántica."""
    
    enable_ontology_check: bool = True
    strict_mode: bool = False
    min_required_matches: int = 1
    max_violations_per_apu: int = 10
    
    @classmethod
    def default(cls) -> SemanticPolicyConfig:
        """Retorna configuración por defecto."""
        return cls()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SemanticPolicyConfig:
        """Construye desde diccionario."""
        return cls(
            enable_ontology_check=bool(data.get("enable_ontology_check", True)),
            strict_mode=bool(data.get("strict_mode", False)),
            min_required_matches=int(data.get("min_required_matches", 1)),
            max_violations_per_apu=int(data.get("max_violations_per_apu", 10)),
        )


@dataclass(frozen=True)
class GovernanceConfig:
    """Configuración consolidada del motor de gobernanza."""
    
    config_dir: Path = field(default_factory=lambda: Path(DEFAULT_CONFIG_DIR))
    thresholds: ScoreThresholds = field(default_factory=ScoreThresholds)
    penalties: SeverityPenalties = field(default_factory=SeverityPenalties)
    semantic_policy: SemanticPolicyConfig = field(
        default_factory=SemanticPolicyConfig.default
    )
    
    # Archivos de configuración
    ontology_filename: str = ONTOLOGY_FILENAME
    data_contract_filename: str = DATA_CONTRACT_FILENAME
    
    @property
    def ontology_path(self) -> Path:
        """Ruta al archivo de ontología."""
        return self.config_dir / self.ontology_filename
    
    @property
    def data_contract_path(self) -> Path:
        """Ruta al archivo de contrato de datos."""
        return self.config_dir / self.data_contract_filename


# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================


@dataclass(frozen=True)
class Violation:
    """
    Representa una violación de gobernanza inmutable.
    
    Attributes:
        type: Tipo de violación.
        message: Descripción detallada.
        severity: Nivel de severidad.
        context: Datos adicionales para debugging.
        timestamp: Momento de la detección.
    """
    
    type: str
    message: str
    severity: Severity
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def create(
        cls,
        type_: str,
        message: str,
        severity: Union[str, Severity] = Severity.ERROR,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Violation]:
        """
        Factory method con validación.
        
        Args:
            type_: Tipo de violación.
            message: Mensaje descriptivo.
            severity: Nivel de severidad.
            context: Contexto adicional.
            
        Returns:
            Violation si los parámetros son válidos, None en caso contrario.
        """
        # Validar tipo
        if not type_ or not isinstance(type_, str):
            logger.error("Violation.create: 'type_' debe ser string no vacío")
            return None
        
        # Validar mensaje
        if not message or not isinstance(message, str):
            logger.error("Violation.create: 'message' debe ser string no vacío")
            return None
        
        # Normalizar severidad
        if isinstance(severity, str):
            severity = Severity.from_string(severity)
        
        # Validar contexto
        safe_context = {}
        if context and isinstance(context, dict):
            safe_context = dict(context)
        
        return cls(
            type=type_.strip(),
            message=message.strip(),
            severity=severity,
            context=safe_context,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        result = {
            "type": self.type,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.context:
            result["context"] = self.context
        return result
    
    @property
    def penalty(self) -> float:
        """Penalización asociada a esta violación."""
        return self.severity.penalty
    
    @property
    def is_blocking(self) -> bool:
        """Indica si esta violación bloquea el procesamiento."""
        return self.severity.is_blocking


@dataclass
class ComplianceReport:
    """
    Reporte de cumplimiento de gobernanza.
    
    Acumula violaciones y calcula score/status automáticamente.
    
    Attributes:
        initial_score: Score inicial antes de penalizaciones.
        violations: Lista de violaciones detectadas.
        semantic_alerts: Alertas semánticas (legacy).
        thresholds: Umbrales de clasificación.
        penalties: Penalizaciones por severidad.
        timestamp: Momento de creación del reporte.
    """

    initial_score: float = MAX_SCORE
    violations: List[Violation] = field(default_factory=list)
    semantic_alerts: List[Dict[str, Any]] = field(default_factory=list)
    thresholds: ScoreThresholds = field(default_factory=ScoreThresholds)
    penalties: SeverityPenalties = field(default_factory=SeverityPenalties)
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Campos computados (se actualizan en cada violación)
    _score: float = field(default=MAX_SCORE, repr=False)
    _error_count: int = field(default=0, repr=False)
    _warning_count: int = field(default=0, repr=False)
    _info_count: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """Inicializa score desde initial_score."""
        self._score = self.initial_score

    @property
    def score(self) -> float:
        """Score actual después de penalizaciones."""
        return self._score
    
    @property
    def error_count(self) -> int:
        """Número de violaciones de tipo ERROR."""
        return self._error_count
    
    @property
    def warning_count(self) -> int:
        """Número de violaciones de tipo WARNING."""
        return self._warning_count
    
    @property
    def info_count(self) -> int:
        """Número de violaciones de tipo INFO."""
        return self._info_count
    
    @property
    def status(self) -> ComplianceStatus:
        """Estado del reporte basado en score y errores."""
        return ComplianceStatus.from_score_and_errors(
            self._score,
            self._error_count,
            self._warning_count,
            self.thresholds,
        )
    
    @property
    def is_compliant(self) -> bool:
        """Indica si el reporte es compatible (PASS o WARNING)."""
        return not self.status.is_blocking
    
    @property
    def total_violations(self) -> int:
        """Total de violaciones registradas."""
        return len(self.violations)
    
    @property
    def total_penalty(self) -> float:
        """Penalización total aplicada."""
        return self.initial_score - self._score

    def add_violation(
        self,
        type_: str,
        message: str,
        severity: Union[str, Severity] = "ERROR",
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Registra una violación de reglas.
        
        Args:
            type_: Tipo/categoría de la violación.
            message: Descripción detallada.
            severity: Nivel de severidad.
            context: Datos adicionales para debugging.
            
        Returns:
            True si se registró correctamente, False en caso contrario.
        """
        violation = Violation.create(type_, message, severity, context)
        if violation is None:
            return False
        
        self.violations.append(violation)
        
        # Aplicar penalización
        penalty = self.penalties.get_penalty(violation.severity)
        self._score = max(MIN_SCORE, self._score - penalty)
        
        # Actualizar contadores
        if violation.severity == Severity.ERROR:
            self._error_count += 1
        elif violation.severity == Severity.WARNING:
            self._warning_count += 1
        else:
            self._info_count += 1
        
        return True
    
    def add_violations(self, violations: Sequence[Violation]) -> int:
        """
        Registra múltiples violaciones.
        
        Args:
            violations: Secuencia de violaciones a agregar.
            
        Returns:
            Número de violaciones agregadas exitosamente.
        """
        count = 0
        for v in violations:
            if self.add_violation(v.type, v.message, v.severity, v.context):
                count += 1
        return count
    
    def merge(self, other: ComplianceReport) -> None:
        """
        Fusiona otro reporte en este.
        
        Args:
            other: Reporte a fusionar.
        """
        for violation in other.violations:
            self.add_violation(
                violation.type,
                violation.message,
                violation.severity,
                violation.context,
            )
        self.semantic_alerts.extend(other.semantic_alerts)

    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumen del reporte."""
        return {
            "status": self.status.value,
            "status_emoji": self.status.emoji,
            "score": round(self._score, 2),
            "total_violations": self.total_violations,
            "errors": self._error_count,
            "warnings": self._warning_count,
            "infos": self._info_count,
            "is_compliant": self.is_compliant,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario completo."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "score": round(self._score, 2),
            "initial_score": self.initial_score,
            "total_penalty": round(self.total_penalty, 2),
            "is_compliant": self.is_compliant,
            "counts": {
                "total": self.total_violations,
                "errors": self._error_count,
                "warnings": self._warning_count,
                "infos": self._info_count,
            },
            "violations": [v.to_dict() for v in self.violations],
            "semantic_alerts": self.semantic_alerts,
            "thresholds": {
                "fail": self.thresholds.fail_threshold,
                "warning": self.thresholds.warning_threshold,
            },
        }
    
    def get_violations_by_severity(
        self,
        severity: Severity,
    ) -> List[Violation]:
        """Filtra violaciones por severidad."""
        return [v for v in self.violations if v.severity == severity]
    
    def get_violations_by_type(self, type_: str) -> List[Violation]:
        """Filtra violaciones por tipo."""
        return [v for v in self.violations if v.type == type_]


# ============================================================================
# ONTOLOGÍA
# ============================================================================


@dataclass(frozen=True)
class DomainRule:
    """
    Reglas de un dominio de ontología.
    
    Inmutable después de construcción.
    """
    
    name: str
    identifying_keywords: FrozenSet[str]
    forbidden_keywords: FrozenSet[str]
    required_keywords: FrozenSet[str]
    min_required_matches: int = 1
    
    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> DomainRule:
        """
        Construye desde diccionario.
        
        Args:
            name: Nombre del dominio.
            data: Datos del dominio.
            
        Returns:
            DomainRule construido.
        """
        def normalize_keywords(keywords: Any) -> FrozenSet[str]:
            if not isinstance(keywords, list):
                return frozenset()
            return frozenset(
                kw.strip().upper()
                for kw in keywords
                if isinstance(kw, str) and kw.strip()
            )
        
        identifying = data.get("identifying_keywords", [name])
        
        return cls(
            name=name.upper(),
            identifying_keywords=normalize_keywords(identifying),
            forbidden_keywords=normalize_keywords(data.get("forbidden_keywords", [])),
            required_keywords=normalize_keywords(data.get("required_keywords", [])),
            min_required_matches=int(data.get("min_required_matches", 1)),
        )
    
    def matches_description(self, description: str) -> int:
        """
        Cuenta cuántas keywords identificadoras coinciden.
        
        Args:
            description: Texto a verificar (ya normalizado).
            
        Returns:
            Número de coincidencias.
        """
        if not description or not self.identifying_keywords:
            return 0
        return sum(1 for kw in self.identifying_keywords if kw in description)
    
    def has_forbidden(self, text: str) -> List[str]:
        """
        Encuentra keywords prohibidas en el texto.
        
        Args:
            text: Texto a verificar (ya normalizado).
            
        Returns:
            Lista de keywords prohibidas encontradas.
        """
        if not text or not self.forbidden_keywords:
            return []
        return [kw for kw in self.forbidden_keywords if kw in text]
    
    def count_required(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Cuenta keywords requeridas presentes y ausentes.
        
        Args:
            text: Texto a verificar (ya normalizado).
            
        Returns:
            Tupla (encontradas, faltantes).
        """
        if not self.required_keywords:
            return [], []
        
        found = [kw for kw in self.required_keywords if kw in text]
        missing = [kw for kw in self.required_keywords if kw not in text]
        
        return found, missing


class Ontology:
    """
    Representa una ontología de dominios para validación semántica.
    
    Proporciona caché para búsquedas eficientes.
    """
    
    def __init__(self, domains: Optional[Dict[str, DomainRule]] = None):
        """
        Inicializa la ontología.
        
        Args:
            domains: Diccionario de dominios (nombre -> reglas).
        """
        self._domains: Dict[str, DomainRule] = domains or {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Ontology:
        """
        Construye desde diccionario de JSON.
        
        Args:
            data: Datos cargados del JSON.
            
        Returns:
            Ontology construida.
        """
        domains_data = data.get("domains", {})
        if not isinstance(domains_data, dict):
            raise OntologyValidationError("'domains' debe ser un diccionario")
        
        domains = {}
        for name, rules in domains_data.items():
            if not isinstance(rules, dict):
                raise OntologyValidationError(
                    f"Dominio '{name}' debe ser un diccionario",
                    field=name,
                )
            domains[name.upper()] = DomainRule.from_dict(name, rules)
        
        return cls(domains=domains)
    
    @classmethod
    def load_from_file(cls, path: Path) -> Ontology:
        """
        Carga ontología desde archivo JSON.
        
        Args:
            path: Ruta al archivo.
            
        Returns:
            Ontology cargada.
            
        Raises:
            OntologyLoadError: Si hay error de carga.
            OntologyValidationError: Si hay error de validación.
        """
        if not path.exists():
            raise OntologyLoadError(f"Archivo no encontrado: {path}", path)
        
        if not path.is_file():
            raise OntologyLoadError(f"La ruta no es un archivo: {path}", path)
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except PermissionError:
            raise OntologyLoadError(f"Sin permisos para leer: {path}", path)
        except Exception as e:
            raise OntologyLoadError(f"Error leyendo archivo: {e}", path)
        
        if not content:
            raise OntologyLoadError(f"Archivo vacío: {path}", path)
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise OntologyLoadError(
                f"Error de sintaxis JSON en línea {e.lineno}: {e.msg}",
                path,
            )
        
        if not isinstance(data, dict):
            raise OntologyValidationError("La ontología debe ser un objeto JSON")
        
        return cls.from_dict(data)
    
    @property
    def domain_count(self) -> int:
        """Número de dominios definidos."""
        return len(self._domains)
    
    @property
    def domain_names(self) -> List[str]:
        """Lista de nombres de dominios."""
        return list(self._domains.keys())
    
    @property
    def is_empty(self) -> bool:
        """Indica si la ontología está vacía."""
        return len(self._domains) == 0
    
    def get_domain(self, name: str) -> Optional[DomainRule]:
        """Obtiene un dominio por nombre."""
        return self._domains.get(name.upper())
    
    def infer_domain(self, description: str) -> Optional[DomainRule]:
        """
        Infiere el dominio más probable para una descripción.
        
        Args:
            description: Descripción a analizar (ya normalizada).
            
        Returns:
            DomainRule del mejor match, None si no hay match.
        """
        if not description or not self._domains:
            return None
        
        best_match: Optional[DomainRule] = None
        best_count = 0
        
        for domain in self._domains.values():
            match_count = domain.matches_description(description)
            if match_count > best_count:
                best_count = match_count
                best_match = domain
        
        return best_match
    
    def __iter__(self) -> Iterator[DomainRule]:
        """Itera sobre los dominios."""
        return iter(self._domains.values())


# ============================================================================
# PROTOCOLOS
# ============================================================================


@runtime_checkable
class GovernanceValidator(Protocol):
    """
    Protocolo para validadores de gobernanza.
    
    Define la interfaz común para diferentes tipos de validación.
    """
    
    def validate(
        self,
        data: pd.DataFrame,
        report: ComplianceReport,
    ) -> ComplianceReport:
        """
        Ejecuta la validación sobre los datos.
        
        Args:
            data: DataFrame a validar.
            report: Reporte donde registrar violaciones.
            
        Returns:
            Reporte actualizado.
        """
        ...


# ============================================================================
# VALIDADORES
# ============================================================================


class SemanticValidator:
    """
    Validador de coherencia semántica basado en ontología.
    
    Verifica que los insumos de cada APU pertenezcan al dominio correcto.
    """
    
    def __init__(
        self,
        ontology: Ontology,
        config: SemanticPolicyConfig,
    ):
        """
        Inicializa el validador.
        
        Args:
            ontology: Ontología para validación.
            config: Configuración de política.
        """
        self._ontology = ontology
        self._config = config
    
    def validate(
        self,
        dataframe: pd.DataFrame,
        report: ComplianceReport,
    ) -> ComplianceReport:
        """
        Ejecuta la validación semántica.
        
        Args:
            dataframe: DataFrame con APUs e insumos.
            report: Reporte donde registrar violaciones.
            
        Returns:
            Reporte actualizado.
        """
        if not self._should_validate(dataframe, report):
            return report
        
        logger.info("🧠 Iniciando Validación Semántica de APUs...")
        
        if self._ontology.is_empty:
            logger.warning("⚠️ Ontología vacía o sin dominios definidos")
            return report
        
        # Procesar cada APU
        try:
            grouped = dataframe.groupby(ColumnNames.CODIGO_APU)
        except Exception as e:
            report.add_violation(
                ViolationType.PROCESSING_ERROR.value,
                f"Error agrupando datos: {e}",
                Severity.ERROR,
            )
            return report
        
        processed_count = 0
        for apu_code, group in grouped:
            if group.empty:
                continue
            
            self._validate_apu(apu_code, group, report)
            processed_count += 1
        
        # Log resumen
        summary = report.get_summary()
        logger.info(
            f"✅ Validación Semántica completada | "
            f"APUs: {processed_count} | "
            f"Status: {summary['status']} | "
            f"Score: {summary['score']} | "
            f"Violaciones: {summary['total_violations']}"
        )
        
        return report
    
    def _should_validate(
        self,
        dataframe: pd.DataFrame,
        report: ComplianceReport,
    ) -> bool:
        """Verifica precondiciones para validación."""
        if not self._config.enable_ontology_check:
            logger.info("ℹ️ Validación semántica desactivada por política")
            return False
        
        if dataframe is None:
            logger.warning("⚠️ DataFrame es None")
            return False
        
        if not isinstance(dataframe, pd.DataFrame):
            report.add_violation(
                ViolationType.TYPE_ERROR.value,
                f"Tipo de datos inválido: {type(dataframe).__name__}",
                Severity.ERROR,
            )
            return False
        
        if dataframe.empty:
            logger.warning("⚠️ DataFrame vacío")
            return False
        
        # Verificar columnas requeridas
        required = [
            ColumnNames.CODIGO_APU,
            ColumnNames.DESCRIPCION_APU,
            ColumnNames.DESCRIPCION_INSUMO,
        ]
        missing = [col for col in required if col not in dataframe.columns]
        
        if missing:
            report.add_violation(
                ViolationType.SCHEMA_ERROR.value,
                f"Faltan columnas: {missing}",
                Severity.ERROR,
                {"missing_columns": missing},
            )
            return False
        
        return True
    
    def _validate_apu(
        self,
        apu_code: Any,
        group: pd.DataFrame,
        report: ComplianceReport,
    ) -> None:
        """Valida un APU individual."""
        # Obtener descripción del APU
        try:
            desc_raw = group[ColumnNames.DESCRIPCION_APU].iloc[0]
            desc_apu = self._normalize_text(desc_raw)
        except (IndexError, KeyError):
            return
        
        # Inferir dominio
        domain = self._ontology.infer_domain(desc_apu)
        if domain is None:
            logger.debug(f"No se pudo inferir dominio para APU: {apu_code}")
            return
        
        # Obtener insumos normalizados
        insumos = group[ColumnNames.DESCRIPCION_INSUMO].apply(self._normalize_text)
        
        # Verificar keywords prohibidas
        self._check_forbidden(apu_code, domain, insumos, report)
        
        # Verificar keywords requeridas
        self._check_required(apu_code, domain, insumos, report)
    
    def _check_forbidden(
        self,
        apu_code: Any,
        domain: DomainRule,
        insumos: pd.Series,
        report: ComplianceReport,
    ) -> None:
        """Verifica keywords prohibidas."""
        if not domain.forbidden_keywords:
            return
        
        violation_count = 0
        
        for keyword in domain.forbidden_keywords:
            if violation_count >= self._config.max_violations_per_apu:
                break
            
            try:
                mask = insumos.str.contains(keyword, regex=False, na=False)
                if mask.any():
                    violating = insumos[mask].unique().tolist()
                    
                    sample_size = 3
                    sample = violating[:sample_size]
                    suffix = (
                        f" (y {len(violating) - sample_size} más)"
                        if len(violating) > sample_size
                        else ""
                    )
                    
                    report.add_violation(
                        ViolationType.SEMANTIC_INCONSISTENCY.value,
                        f"APU '{apu_code}' ({domain.name}): "
                        f"Insumos con término prohibido '{keyword}': {sample}{suffix}",
                        Severity.WARNING,
                        {
                            "apu_code": str(apu_code),
                            "domain": domain.name,
                            "forbidden_keyword": keyword,
                            "violation_count": len(violating),
                        },
                    )
                    violation_count += 1
                    
            except Exception as e:
                logger.debug(f"Error verificando keyword '{keyword}': {e}")
    
    def _check_required(
        self,
        apu_code: Any,
        domain: DomainRule,
        insumos: pd.Series,
        report: ComplianceReport,
    ) -> None:
        """Verifica keywords requeridas."""
        if not domain.required_keywords:
            return
        
        # Concatenar todos los insumos
        try:
            all_text = " ".join(insumos.dropna().astype(str))
        except Exception:
            return
        
        if not all_text.strip():
            return
        
        found, missing = domain.count_required(all_text)
        
        if len(found) < domain.min_required_matches:
            report.add_violation(
                ViolationType.SEMANTIC_INCOMPLETENESS.value,
                f"APU '{apu_code}' ({domain.name}): "
                f"Insumos insuficientes. "
                f"Encontrados: {len(found)}/{domain.min_required_matches}. "
                f"Faltantes: {missing[:5]}",
                Severity.WARNING,
                {
                    "apu_code": str(apu_code),
                    "domain": domain.name,
                    "found": found,
                    "missing": missing,
                    "min_required": domain.min_required_matches,
                },
            )
    
    @staticmethod
    def _normalize_text(value: Any) -> str:
        """Normaliza texto a mayúsculas."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        return str(value).strip().upper()


# ============================================================================
# MOTOR DE GOBERNANZA
# ============================================================================


class GovernanceEngine:
    """
    Motor de reglas para validar integridad semántica y estructural.
    
    Coordina múltiples validadores y genera reportes de cumplimiento.
    """

    def __init__(
        self,
        config: Optional[GovernanceConfig] = None,
        ontology: Optional[Ontology] = None,
    ):
        """
        Inicializa el motor.
        
        Args:
            config: Configuración del motor.
            ontology: Ontología pre-cargada (opcional).
        """
        self._config = config or GovernanceConfig()
        self._ontology = ontology
        self._semantic_policy = self._config.semantic_policy
        self._ontology_loaded = ontology is not None
        self._config_loaded = False
        
        # Cargar configuración si no se proporcionó ontología
        if ontology is None:
            self._load_configuration()
        else:
            self._config_loaded = True
        
        # Inicializar validadores
        self._validators: List[GovernanceValidator] = []
        self._initialize_validators()
        
        logger.debug(
            f"GovernanceEngine inicializado | "
            f"Config: {self._config.config_dir} | "
            f"Ontología: {'cargada' if self._ontology_loaded else 'no cargada'}"
        )

    def _load_configuration(self) -> None:
        """Carga configuración desde archivos."""
        ontology_loaded = self._load_ontology()
        policy_loaded = self._load_semantic_policy()
        
        self._config_loaded = ontology_loaded or policy_loaded
        
        if not self._config_loaded:
            logger.warning(
                "⚠️ No se cargó configuración. Usando valores por defecto."
            )

    def _load_ontology(self) -> bool:
        """Carga la ontología desde archivo."""
        path = self._config.ontology_path
        
        if not path.exists():
            logger.warning(f"⚠️ Ontología no encontrada: {path}")
            return False
        
        try:
            self._ontology = Ontology.load_from_file(path)
            self._ontology_loaded = True
            logger.info(f"✅ Ontología cargada: {path}")
            return True
            
        except OntologyLoadError as e:
            logger.error(f"❌ Error cargando ontología: {e}")
            return False
        except OntologyValidationError as e:
            logger.error(f"❌ Ontología inválida: {e}")
            return False

    def _load_semantic_policy(self) -> bool:
        """Carga políticas semánticas desde data_contract.yaml."""
        path = self._config.data_contract_path
        
        if not path.exists():
            logger.warning(f"⚠️ Data contract no encontrado: {path}")
            return False
        
        # Verificar disponibilidad de PyYAML
        try:
            import yaml
        except ImportError:
            logger.warning("⚠️ PyYAML no instalado. pip install pyyaml")
            return False
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            
            if not content:
                logger.warning(f"⚠️ Archivo vacío: {path}")
                return False
            
            contract = yaml.safe_load(content)
            
            if contract is None or not isinstance(contract, dict):
                logger.warning(f"⚠️ Contenido inválido en {path}")
                return False
            
            policy_data = contract.get("semantic_policy", {})
            if isinstance(policy_data, dict):
                self._semantic_policy = SemanticPolicyConfig.from_dict(policy_data)
            
            logger.info(f"✅ Políticas cargadas: {path}")
            return True
            
        except yaml.YAMLError as e:
            logger.error(f"❌ Error YAML: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error cargando políticas: {type(e).__name__}: {e}")
            return False

    def _initialize_validators(self) -> None:
        """Inicializa los validadores disponibles."""
        if self._ontology and self._ontology_loaded:
            semantic_validator = SemanticValidator(
                ontology=self._ontology,
                config=self._semantic_policy,
            )
            self._validators.append(semantic_validator)

    # =========================================================================
    # API PÚBLICA
    # =========================================================================

    @property
    def ontology(self) -> Dict[str, Any]:
        """
        Acceso al diccionario de ontología (compatibilidad legacy).
        """
        if self._ontology is None:
            return {}
        # Reconstruir diccionario para compatibilidad
        return {
            "domains": {
                domain.name: {
                    "identifying_keywords": list(domain.identifying_keywords),
                    "forbidden_keywords": list(domain.forbidden_keywords),
                    "required_keywords": list(domain.required_keywords),
                    "min_required_matches": domain.min_required_matches,
                }
                for domain in self._ontology
            }
        }

    @property
    def semantic_policy(self) -> Dict[str, Any]:
        """Acceso a la política semántica como dict (compatibilidad legacy)."""
        return {
            "enable_ontology_check": self._semantic_policy.enable_ontology_check,
            "strict_mode": self._semantic_policy.strict_mode,
            "min_required_matches": self._semantic_policy.min_required_matches,
            "max_violations_per_apu": self._semantic_policy.max_violations_per_apu,
        }

    @property
    def is_ontology_loaded(self) -> bool:
        """Indica si la ontología está cargada."""
        return self._ontology_loaded

    @property
    def is_configured(self) -> bool:
        """Indica si el motor está configurado."""
        return self._config_loaded

    def load_ontology(self, path: str) -> bool:
        """
        Carga una ontología desde una ruta específica.
        
        Args:
            path: Ruta al archivo JSON de ontología.
            
        Returns:
            True si se cargó correctamente.
        """
        if not path:
            logger.error("❌ Ruta de ontología vacía")
            return False
        
        ontology_path = Path(path)
        
        try:
            self._ontology = Ontology.load_from_file(ontology_path)
            self._ontology_loaded = True
            
            # Reinicializar validadores
            self._validators = []
            self._initialize_validators()
            
            logger.info(f"✅ Ontología recargada: {path}")
            return True
            
        except (OntologyLoadError, OntologyValidationError) as e:
            logger.error(f"❌ Error: {e}")
            return False

    def check_semantic_coherence(
        self,
        dataframe: pd.DataFrame,
    ) -> ComplianceReport:
        """
        Verifica la coherencia semántica de los APUs y sus insumos.
        
        Args:
            dataframe: DataFrame conteniendo APUs e insumos.
            
        Returns:
            ComplianceReport con las violaciones detectadas.
        """
        report = self._create_report()
        
        # Verificar precondiciones
        if not self._semantic_policy.enable_ontology_check:
            logger.info("ℹ️ Validación semántica desactivada")
            return report
        
        if not self._ontology_loaded or self._ontology is None:
            logger.warning("⚠️ Ontología no cargada")
            return report
        
        # Usar validador semántico
        for validator in self._validators:
            if isinstance(validator, SemanticValidator):
                validator.validate(dataframe, report)
                break
        
        return report

    def validate_all(self, dataframe: pd.DataFrame) -> ComplianceReport:
        """
        Ejecuta todas las validaciones disponibles.
        
        Args:
            dataframe: DataFrame a validar.
            
        Returns:
            Reporte consolidado de todas las validaciones.
        """
        report = self._create_report()
        
        for validator in self._validators:
            try:
                validator.validate(dataframe, report)
            except Exception as e:
                report.add_violation(
                    ViolationType.PROCESSING_ERROR.value,
                    f"Error en validador {type(validator).__name__}: {e}",
                    Severity.ERROR,
                )
        
        return report

    def _create_report(self) -> ComplianceReport:
        """Crea un nuevo reporte con la configuración actual."""
        return ComplianceReport(
            thresholds=self._config.thresholds,
            penalties=self._config.penalties,
        )

    def get_status(self) -> Dict[str, Any]:
        """Retorna estado del motor."""
        return {
            "config_loaded": self._config_loaded,
            "ontology_loaded": self._ontology_loaded,
            "domain_count": self._ontology.domain_count if self._ontology else 0,
            "domains": self._ontology.domain_names if self._ontology else [],
            "validators_count": len(self._validators),
            "semantic_policy": self.semantic_policy,
        }


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def create_engine(
    config_dir: Optional[str] = None,
    thresholds: Optional[ScoreThresholds] = None,
    penalties: Optional[SeverityPenalties] = None,
) -> GovernanceEngine:
    """
    Factory function para crear un motor de gobernanza configurado.
    
    Args:
        config_dir: Directorio de configuración.
        thresholds: Umbrales de score.
        penalties: Penalizaciones por severidad.
        
    Returns:
        Motor configurado.
    """
    config = GovernanceConfig(
        config_dir=Path(config_dir) if config_dir else Path(DEFAULT_CONFIG_DIR),
        thresholds=thresholds or ScoreThresholds(),
        penalties=penalties or SeverityPenalties(),
    )
    return GovernanceEngine(config=config)


def create_engine_with_ontology(
    ontology_path: str,
    enable_semantic_check: bool = True,
    strict_mode: bool = False,
) -> GovernanceEngine:
    """
    Crea un motor con ontología específica.
    
    Args:
        ontology_path: Ruta a la ontología.
        enable_semantic_check: Si habilitar validación semántica.
        strict_mode: Si usar modo estricto.
        
    Returns:
        Motor configurado.
        
    Raises:
        OntologyLoadError: Si no se puede cargar la ontología.
    """
    ontology = Ontology.load_from_file(Path(ontology_path))
    
    config = GovernanceConfig(
        semantic_policy=SemanticPolicyConfig(
            enable_ontology_check=enable_semantic_check,
            strict_mode=strict_mode,
        ),
    )
    
    return GovernanceEngine(config=config, ontology=ontology)


def create_strict_engine(config_dir: Optional[str] = None) -> GovernanceEngine:
    """
    Crea un motor en modo estricto (penalizaciones más altas).
    
    Args:
        config_dir: Directorio de configuración.
        
    Returns:
        Motor en modo estricto.
    """
    config = GovernanceConfig(
        config_dir=Path(config_dir) if config_dir else Path(DEFAULT_CONFIG_DIR),
        thresholds=ScoreThresholds(fail_threshold=80.0, warning_threshold=95.0),
        penalties=SeverityPenalties(
            error_penalty=10.0,
            warning_penalty=3.0,
            info_penalty=0.5,
        ),
        semantic_policy=SemanticPolicyConfig(strict_mode=True),
    )
    return GovernanceEngine(config=config)


def validate_dataframe(
    dataframe: pd.DataFrame,
    ontology_path: Optional[str] = None,
    config_dir: Optional[str] = None,
) -> ComplianceReport:
    """
    Función de conveniencia para validación rápida.
    
    Args:
        dataframe: DataFrame a validar.
        ontology_path: Ruta a ontología (opcional).
        config_dir: Directorio de configuración (opcional).
        
    Returns:
        Reporte de cumplimiento.
    """
    if ontology_path:
        engine = create_engine_with_ontology(ontology_path)
    else:
        engine = create_engine(config_dir=config_dir)
    
    return engine.validate_all(dataframe)


# ============================================================================
# COMPATIBILIDAD LEGACY
# ============================================================================

# Mantener constantes globales para compatibilidad con código existente
SEVERITY_PENALTIES: Dict[Severity, float] = _SEVERITY_PENALTIES
SCORE_THRESHOLDS: Dict[str, float] = {
    "fail": _DEFAULT_THRESHOLDS.fail_threshold,
    "warning": _DEFAULT_THRESHOLDS.warning_threshold,
}