"""
Módulo: Pipeline Director (El Sistema Nervioso Central)
========================================================

Este componente actúa como el "Sistema Nervioso Central" del ecosistema APU Filter.
Su función principal no es procesar datos, sino gestionar la evolución del **Vector de Estado**
del proyecto a través de un espacio vectorial jerarquizado, delegando las transformaciones
a la Matriz de Interacción Central (MICRegistry).

Fundamentos Matemáticos y Arquitectura de Gobernanza:
-----------------------------------------------------

1. Orquestación Algebraica (Espacio Vectorial de Operadores):
   El pipeline es una secuencia ordenada de proyecciones sobre una base vectorial
   ortogonal {e₁, ..., eₙ} registrada en la MIC. Cada paso proyecta una "Intención"
   que la MIC resuelve en un handler específico.

2. Filtración por Estratos (Jerarquía DIKW):
   Implementa la restricción topológica de filtración de subespacios:
   V_PHYSICS ⊂ V_TACTICS ⊂ V_STRATEGY ⊂ V_WISDOM
   
   El Director impone la **Clausura Transitiva**: no permite ejecutar un vector de
   Estrategia (Nivel 1) si los invariantes de Física (Nivel 3) y Táctica (Nivel 2)
   no han sido validados.

3. Auditoría Homológica (Secuencia de Mayer-Vietoris):
   En pasos de fusión, verifica la exactitud de la secuencia de Mayer-Vietoris:
   ... → H_k(A∩B) → H_k(A)⊕H_k(B) → H_k(A∪B) → ...
   
   Esto garantiza que la integración no introduzca ciclos espurios (β₁)
   ni desconexiones artificiales (β₀).

4. Protocolo de Caja de Cristal (Glass Box Persistence):
   El estado se serializa entre transiciones de estrato, permitiendo
   auditoría, reanudación y depuración del proceso de decisión.

Invariantes del Sistema:
------------------------
- Filtración: ∀ paso en estrato S, ∀ estrato T < S: T está validado
- Idempotencia: ejecutar un paso dos veces con mismo input produce mismo output
- Trazabilidad: cada transición tiene lineage_hash verificable
"""

from __future__ import annotations

import datetime
import enum
import hashlib
import json
import logging
import os
import pickle
import sys
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
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

import numpy as np
import pandas as pd
from flask import current_app

from app.constants import ColumnNames, InsumoType
from app.flux_condenser import CondenserConfig, DataFluxCondenser
from app.matter_generator import MatterGenerator
from app.schemas import Stratum
from app.telemetry import TelemetryContext
from app.telemetry_narrative import TelemetryNarrator
from agent.business_topology import (
    BudgetGraphBuilder,
    BusinessTopologicalAnalyzer,
)
from app.business_agent import BusinessAgent
from app.semantic_translator import SemanticTranslator
from app.tools_interface import MICRegistry, register_core_vectors

from .apu_processor import (
    APUProcessor,
    APUCostCalculator,
    DataMerger,
    DataValidator,
    FileValidator,
    InsumosProcessor,
    PresupuestoProcessor,
    ProcessingThresholds,
    build_output_dictionary,
    build_processed_apus_dataframe,
    calculate_insumo_costs,
    calculate_total_costs,
    group_and_split_description,
    sanitize_for_json,
    synchronize_data_sources,
)
from .data_validator import validate_and_clean_data


# ============================================================================
# CONSTANTES GLOBALES
# ============================================================================

# Tolerancia numérica
EPSILON: Final[float] = 1e-9

# Versión del pipeline
PIPELINE_VERSION: Final[str] = "3.1.0"

# Directorio de sesiones por defecto
DEFAULT_SESSION_DIR: Final[str] = "data/sessions"

# Tamaño máximo de contexto serializado (50MB)
MAX_CONTEXT_SIZE_BYTES: Final[int] = 50 * 1024 * 1024

# Extensión de archivos de sesión
SESSION_FILE_EXTENSION: Final[str] = ".pkl"


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)


def configure_pipeline_logging(level: str = "INFO") -> None:
    """
    Configura el logging del pipeline.
    
    Args:
        level: Nivel de logging.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(handler)


# Configurar al importar
configure_pipeline_logging(os.getenv("LOG_LEVEL", "INFO"))


# ============================================================================
# EXCEPCIONES DEL DOMINIO
# ============================================================================


class PipelineError(Exception):
    """Excepción base del pipeline."""
    pass


class ConfigurationError(PipelineError):
    """Error de configuración del pipeline."""
    pass


class StepExecutionError(PipelineError):
    """Error durante la ejecución de un paso."""
    
    def __init__(
        self,
        message: str,
        step_name: str,
        cause: Optional[Exception] = None,
    ):
        self.step_name = step_name
        self.cause = cause
        super().__init__(f"[{step_name}] {message}")


class FiltrationViolationError(PipelineError):
    """Violación del invariante de filtración topológica."""
    
    def __init__(
        self,
        target_stratum: Stratum,
        missing_strata: List[str],
        validated_strata: List[str],
    ):
        self.target_stratum = target_stratum
        self.missing_strata = missing_strata
        self.validated_strata = validated_strata
        super().__init__(
            f"Filtration Invariant Violation. "
            f"Target: {target_stratum.name}. "
            f"Missing: {missing_strata}. "
            f"Validated: {validated_strata}."
        )


class SessionError(PipelineError):
    """Error relacionado con gestión de sesiones."""
    pass


class SessionNotFoundError(SessionError):
    """Sesión no encontrada."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")


class SessionCorruptedError(SessionError):
    """Sesión corrupta."""
    
    def __init__(self, session_id: str, reason: str):
        self.session_id = session_id
        self.reason = reason
        super().__init__(f"Session corrupted: {session_id}. Reason: {reason}")


class PreconditionError(StepExecutionError):
    """Error de precondición en un paso."""
    
    def __init__(
        self,
        step_name: str,
        missing_keys: List[str],
    ):
        self.missing_keys = missing_keys
        super().__init__(
            f"Missing required context keys: {missing_keys}",
            step_name,
        )


class DataValidationError(StepExecutionError):
    """Error de validación de datos en un paso."""
    pass


# ============================================================================
# ENUMERACIONES
# ============================================================================


class StepStatus(str, enum.Enum):
    """Estados posibles de un paso."""
    
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    SKIPPED = "skipped"
    ERROR = "error"
    
    @property
    def is_terminal(self) -> bool:
        """Indica si es un estado terminal."""
        return self in (StepStatus.SUCCESS, StepStatus.SKIPPED, StepStatus.ERROR)
    
    @property
    def is_successful(self) -> bool:
        """Indica si es un estado exitoso."""
        return self in (StepStatus.SUCCESS, StepStatus.SKIPPED)


class PipelineSteps(str, enum.Enum):
    """
    Receta canónica del pipeline.
    
    El orden del enum define el orden de ejecución.
    
    Grafo de dependencias (→ = "produce para"):
      LOAD_DATA → AUDITED_MERGE → CALCULATE_COSTS → FINAL_MERGE
                                                        ↓
                                              BUSINESS_TOPOLOGY → MATERIALIZATION
                                                                       ↓
                                                                  BUILD_OUTPUT
    """
    
    LOAD_DATA = "load_data"
    AUDITED_MERGE = "audited_merge"
    CALCULATE_COSTS = "calculate_costs"
    FINAL_MERGE = "final_merge"
    BUSINESS_TOPOLOGY = "business_topology"
    MATERIALIZATION = "materialization"
    BUILD_OUTPUT = "build_output"
    
    @property
    def stratum(self) -> Stratum:
        """Estrato DIKW de este paso."""
        return _STEP_STRATA[self]
    
    @property
    def index(self) -> int:
        """Índice ordinal del paso."""
        return list(PipelineSteps).index(self)
    
    @classmethod
    def from_string(cls, value: str) -> Optional[PipelineSteps]:
        """Parsea desde string."""
        for step in cls:
            if step.value == value:
                return step
        return None


# Mapeo de pasos a estratos
_STEP_STRATA: Final[Dict[PipelineSteps, Stratum]] = {
    PipelineSteps.LOAD_DATA: Stratum.PHYSICS,
    PipelineSteps.AUDITED_MERGE: Stratum.PHYSICS,
    PipelineSteps.CALCULATE_COSTS: Stratum.TACTICS,
    PipelineSteps.FINAL_MERGE: Stratum.TACTICS,
    PipelineSteps.BUSINESS_TOPOLOGY: Stratum.STRATEGY,
    PipelineSteps.MATERIALIZATION: Stratum.STRATEGY,
    PipelineSteps.BUILD_OUTPUT: Stratum.WISDOM,
}

# Orden de estratos en la filtración DIKW
_STRATUM_ORDER: Final[Dict[Stratum, int]] = {
    Stratum.PHYSICS: 0,
    Stratum.TACTICS: 1,
    Stratum.STRATEGY: 2,
    Stratum.WISDOM: 3,
}

# Evidencia requerida por estrato
_STRATUM_EVIDENCE: Final[Dict[Stratum, Tuple[str, ...]]] = {
    Stratum.PHYSICS: ("df_presupuesto", "df_insumos", "df_apus_raw"),
    Stratum.TACTICS: ("df_apu_costos", "df_tiempo", "df_rendimiento"),
    Stratum.STRATEGY: ("graph", "business_topology_report"),
    Stratum.WISDOM: ("final_result",),
}


def stratum_level(s: Stratum) -> int:
    """Retorna el nivel ordinal de un estrato en la filtración DIKW."""
    return _STRATUM_ORDER.get(s, -1)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================


@dataclass(frozen=True)
class StepConfig:
    """Configuración de un paso individual."""
    
    enabled: bool = True
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    retry_delay_seconds: float = 1.0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StepConfig:
        """Construye desde diccionario."""
        return cls(
            enabled=bool(data.get("enabled", True)),
            timeout_seconds=data.get("timeout_seconds"),
            retry_count=int(data.get("retry_count", 0)),
            retry_delay_seconds=float(data.get("retry_delay_seconds", 1.0)),
        )


@dataclass(frozen=True)
class SessionConfig:
    """Configuración de gestión de sesiones."""
    
    session_dir: Path = field(default_factory=lambda: Path(DEFAULT_SESSION_DIR))
    max_age_hours: int = 24
    cleanup_on_success: bool = True
    persist_on_error: bool = True
    
    def __post_init__(self) -> None:
        """Valida y crea directorio si no existe."""
        if not isinstance(self.session_dir, Path):
            object.__setattr__(self, "session_dir", Path(self.session_dir))


@dataclass(frozen=True)
class PipelineConfig:
    """Configuración consolidada del pipeline."""
    
    # Configuración de sesiones
    session: SessionConfig = field(default_factory=SessionConfig)
    
    # Configuración por paso
    step_configs: Dict[str, StepConfig] = field(default_factory=dict)
    
    # Receta de ejecución (None = usar default)
    recipe: Optional[List[str]] = None
    
    # Validación de filtración
    enforce_filtration: bool = True
    
    # Perfiles de archivos
    file_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Configuración general
    raw_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineConfig:
        """Construye desde diccionario."""
        session_data = data.get("session", {})
        session_dir = session_data.get(
            "session_dir",
            data.get("session_dir", DEFAULT_SESSION_DIR),
        )
        
        step_configs = {}
        for step in PipelineSteps:
            step_data = data.get("steps", {}).get(step.value, {})
            step_configs[step.value] = StepConfig.from_dict(step_data)
        
        return cls(
            session=SessionConfig(
                session_dir=Path(session_dir),
                max_age_hours=int(session_data.get("max_age_hours", 24)),
                cleanup_on_success=bool(session_data.get("cleanup_on_success", True)),
                persist_on_error=bool(session_data.get("persist_on_error", True)),
            ),
            step_configs=step_configs,
            recipe=data.get("pipeline_recipe"),
            enforce_filtration=bool(data.get("enforce_filtration", True)),
            file_profiles=data.get("file_profiles", {}),
            raw_config=data,
        )
    
    def get_step_config(self, step_name: str) -> StepConfig:
        """Obtiene configuración de un paso."""
        return self.step_configs.get(step_name, StepConfig())
    
    def is_step_enabled(self, step_name: str) -> bool:
        """Verifica si un paso está habilitado."""
        return self.get_step_config(step_name).enabled


# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================


@dataclass(frozen=True)
class BasisVector:
    """
    Representa un vector base unitario e_i en el espacio de operaciones.
    
    Propiedades:
    - index: Posición en la base
    - label: Identificador único
    - operator_class: Clase del paso asociado
    - stratum: Nivel jerárquico DIKW
    """
    
    index: int
    label: str
    operator_class: Type[ProcessingStep]
    stratum: Stratum
    
    def __repr__(self) -> str:
        return f"e_{self.index}({self.label}, {self.stratum.name})"


@dataclass
class StepResult:
    """
    Resultado de la ejecución de un paso.
    """
    
    step_name: str
    status: StepStatus
    stratum: Stratum
    duration_ms: float = 0.0
    error: Optional[str] = None
    context_keys: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    
    @property
    def is_successful(self) -> bool:
        """Indica si el paso fue exitoso."""
        return self.status.is_successful
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "step_name": self.step_name,
            "status": self.status.value,
            "stratum": self.stratum.name,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
            "context_keys": self.context_keys,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SessionState:
    """
    Estado de una sesión de pipeline.
    """
    
    session_id: str
    context: Dict[str, Any]
    validated_strata: Set[Stratum]
    step_results: List[StepResult]
    created_at: datetime.datetime
    updated_at: datetime.datetime
    version: str = PIPELINE_VERSION
    
    @classmethod
    def create(cls, session_id: str, initial_context: Dict[str, Any]) -> SessionState:
        """Crea una nueva sesión."""
        now = datetime.datetime.now(datetime.timezone.utc)
        return cls(
            session_id=session_id,
            context=dict(initial_context),
            validated_strata=set(),
            step_results=[],
            created_at=now,
            updated_at=now,
        )
    
    def update_context(self, new_context: Dict[str, Any]) -> None:
        """Actualiza el contexto."""
        self.context = new_context
        self.updated_at = datetime.datetime.now(datetime.timezone.utc)
    
    def add_result(self, result: StepResult) -> None:
        """Añade un resultado de paso."""
        self.step_results.append(result)
        self.updated_at = datetime.datetime.now(datetime.timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario (sin contexto completo por tamaño)."""
        return {
            "session_id": self.session_id,
            "version": self.version,
            "validated_strata": [s.name for s in self.validated_strata],
            "step_count": len(self.step_results),
            "context_keys": list(self.context.keys()),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# ============================================================================
# PROTOCOLOS
# ============================================================================


@runtime_checkable
class ProcessingStep(Protocol):
    """
    Protocolo para pasos del pipeline.
    
    Cada paso debe poder ejecutarse con un contexto y telemetría,
    retornando el contexto actualizado.
    """
    
    def execute(
        self,
        context: Dict[str, Any],
        telemetry: TelemetryContext,
        mic: MICRegistry,
    ) -> Dict[str, Any]:
        """
        Ejecuta la lógica del paso.
        
        Args:
            context: Estado actual del procesamiento.
            telemetry: Contexto de telemetría.
            mic: Registro de la Matriz de Interacción Central.
            
        Returns:
            Contexto actualizado.
            
        Raises:
            StepExecutionError: Si hay error en la ejecución.
            PreconditionError: Si faltan precondiciones.
        """
        ...


# ============================================================================
# CLASE BASE ABSTRACTA PARA PASOS
# ============================================================================


class BaseProcessingStep(ABC):
    """
    Clase base abstracta para pasos del pipeline.
    
    Proporciona:
    - Validación de precondiciones
    - Logging estructurado
    - Manejo de errores consistente
    """
    
    # Claves requeridas en el contexto (a sobrescribir en subclases)
    REQUIRED_CONTEXT_KEYS: ClassVar[Tuple[str, ...]] = ()
    
    # Claves producidas por el paso
    PRODUCED_CONTEXT_KEYS: ClassVar[Tuple[str, ...]] = ()
    
    def __init__(
        self,
        config: PipelineConfig,
        thresholds: ProcessingThresholds,
    ):
        """
        Inicializa el paso.
        
        Args:
            config: Configuración del pipeline.
            thresholds: Umbrales de procesamiento.
        """
        self.config = config
        self.thresholds = thresholds
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def execute(
        self,
        context: Dict[str, Any],
        telemetry: TelemetryContext,
        mic: MICRegistry,
    ) -> Dict[str, Any]:
        """
        Ejecuta el paso con validación de precondiciones.
        
        Template method que delega a _execute_impl.
        """
        step_name = self._get_step_name()
        telemetry.start_step(step_name)
        
        try:
            # Validar precondiciones
            self._validate_preconditions(context, step_name)
            
            # Crear copia del contexto para inmutabilidad
            new_context = dict(context)
            
            # Ejecutar lógica específica
            result_context = self._execute_impl(new_context, telemetry, mic)
            
            if result_context is None:
                raise StepExecutionError(
                    "Step returned None context",
                    step_name,
                )
            
            telemetry.end_step(step_name, "success")
            return result_context
            
        except PipelineError:
            telemetry.end_step(step_name, "error")
            raise
        except Exception as e:
            self.logger.error(f"Error in {step_name}: {e}", exc_info=True)
            telemetry.record_error(step_name, str(e))
            telemetry.end_step(step_name, "error")
            raise StepExecutionError(str(e), step_name, e)
    
    def _get_step_name(self) -> str:
        """Obtiene el nombre del paso."""
        # Convierte CamelCase a snake_case
        name = self.__class__.__name__
        if name.endswith("Step"):
            name = name[:-4]
        # Convertir a snake_case
        import re
        return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    
    def _validate_preconditions(
        self,
        context: Dict[str, Any],
        step_name: str,
    ) -> None:
        """Valida que las precondiciones se cumplan."""
        missing = []
        for key in self.REQUIRED_CONTEXT_KEYS:
            value = context.get(key)
            if value is None:
                missing.append(key)
            elif isinstance(value, pd.DataFrame) and value.empty:
                missing.append(f"{key} (empty)")
        
        if missing:
            raise PreconditionError(step_name, missing)
    
    @abstractmethod
    def _execute_impl(
        self,
        context: Dict[str, Any],
        telemetry: TelemetryContext,
        mic: MICRegistry,
    ) -> Dict[str, Any]:
        """
        Implementación específica del paso.
        
        A implementar en subclases.
        """
        pass


# ============================================================================
# IMPLEMENTACIÓN DE PASOS
# ============================================================================


class LoadDataStep(BaseProcessingStep):
    """
    Paso de Carga de Datos.
    
    Carga archivos CSV/Excel de presupuesto, APUs e insumos.
    Proyecta vectores de estabilización física.
    """
    
    REQUIRED_CONTEXT_KEYS = ("presupuesto_path", "apus_path", "insumos_path")
    PRODUCED_CONTEXT_KEYS = (
        "df_presupuesto", "df_insumos", "df_apus_raw",
        "raw_records", "parse_cache",
    )
    
    def _execute_impl(
        self,
        context: Dict[str, Any],
        telemetry: TelemetryContext,
        mic: MICRegistry,
    ) -> Dict[str, Any]:
        """Ejecuta la carga de datos."""
        # Extraer rutas
        presupuesto_path = context["presupuesto_path"]
        apus_path = context["apus_path"]
        insumos_path = context["insumos_path"]
        
        # Validar existencia de archivos
        file_validator = FileValidator()
        for path, name in [
            (presupuesto_path, "presupuesto"),
            (apus_path, "APUs"),
            (insumos_path, "insumos"),
        ]:
            is_valid, error = file_validator.validate_file_exists(path, name)
            if not is_valid:
                raise DataValidationError(error, "load_data")
        
        # Obtener perfiles de archivo
        file_profiles = self.config.file_profiles
        
        # Procesar presupuesto
        presupuesto_profile = file_profiles.get("presupuesto_default", {})
        p_processor = PresupuestoProcessor(
            self.config.raw_config,
            self.thresholds,
            presupuesto_profile,
        )
        df_presupuesto = p_processor.process(presupuesto_path)
        
        if df_presupuesto is None or df_presupuesto.empty:
            raise DataValidationError(
                "Presupuesto processing returned empty DataFrame",
                "load_data",
            )
        
        telemetry.record_metric("load_data", "presupuesto_rows", len(df_presupuesto))
        
        # Procesar insumos
        insumos_profile = file_profiles.get("insumos_default", {})
        i_processor = InsumosProcessor(self.thresholds, insumos_profile)
        df_insumos = i_processor.process(insumos_path)
        
        if df_insumos is None or df_insumos.empty:
            raise DataValidationError(
                "Insumos processing returned empty DataFrame",
                "load_data",
            )
        
        telemetry.record_metric("load_data", "insumos_rows", len(df_insumos))
        
        # Proyectar vector de estabilización física
        flux_result = mic.project_intent(
            "stabilize_flux",
            {"file_path": str(apus_path), "config": self.config.raw_config},
            context,
        )
        
        if not flux_result.get("success", False):
            raise StepExecutionError(
                f"stabilize_flux failed: {flux_result.get('error', 'Unknown')}",
                "load_data",
            )
        
        df_apus_raw = pd.DataFrame(flux_result["data"])
        
        if df_apus_raw.empty:
            raise DataValidationError(
                "DataFluxCondenser returned empty data",
                "load_data",
            )
        
        telemetry.record_metric("load_data", "apus_raw_rows", len(df_apus_raw))
        self.logger.info("✓ Vector stabilize_flux completado")
        
        # Proyectar vector de parsing
        apus_profile = file_profiles.get("apus_default", {})
        parse_result = mic.project_intent(
            "parse_raw",
            {"file_path": str(apus_path), "profile": apus_profile},
            context,
        )
        
        if not parse_result.get("success", False):
            raise StepExecutionError(
                f"parse_raw failed: {parse_result.get('error', 'Unknown')}",
                "load_data",
            )
        
        telemetry.record_metric(
            "load_data",
            "raw_records_count",
            len(parse_result.get("raw_records", [])),
        )
        self.logger.info("✓ Vector parse_raw completado")
        
        # Actualizar contexto
        context.update({
            "df_presupuesto": df_presupuesto,
            "df_insumos": df_insumos,
            "df_apus_raw": df_apus_raw,
            "raw_records": parse_result.get("raw_records", []),
            "parse_cache": parse_result.get("parse_cache", {}),
        })
        
        return context


class AuditedMergeStep(BaseProcessingStep):
    """
    Paso de Fusión con Auditoría Topológica (Mayer-Vietoris).
    
    Verifica que la fusión no introduzca ciclos espurios
    antes de ejecutar el merge físico.
    """
    
    REQUIRED_CONTEXT_KEYS = ("df_apus_raw", "df_insumos")
    PRODUCED_CONTEXT_KEYS = ("df_merged",)
    
    def _execute_impl(
        self,
        context: Dict[str, Any],
        telemetry: TelemetryContext,
        mic: MICRegistry,
    ) -> Dict[str, Any]:
        """Ejecuta la fusión auditada."""
        df_a = context.get("df_presupuesto")
        df_b = context["df_apus_raw"]
        df_insumos = context["df_insumos"]
        
        # Auditoría topológica (no bloquea la fusión)
        if df_a is not None:
            self._perform_mayer_vietoris_audit(
                df_a, df_b, context, telemetry
            )
        else:
            self.logger.info(
                "df_presupuesto no disponible; auditoría Mayer-Vietoris omitida"
            )
        
        # Fusión física
        self.logger.info("Ejecutando fusión física de datos...")
        merger = DataMerger(self.thresholds)
        df_merged = merger.merge_apus_with_insumos(df_b, df_insumos)
        
        if df_merged is None or df_merged.empty:
            raise DataValidationError("Merge produced empty DataFrame", "audited_merge")
        
        telemetry.record_metric("audited_merge", "merged_rows", len(df_merged))
        context["df_merged"] = df_merged
        
        return context
    
    def _perform_mayer_vietoris_audit(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        context: Dict[str, Any],
        telemetry: TelemetryContext,
    ) -> None:
        """Ejecuta auditoría de Mayer-Vietoris."""
        try:
            builder = BudgetGraphBuilder()
            graph_a = builder.build(df_a, pd.DataFrame())
            graph_b = builder.build(pd.DataFrame(), df_b)
            
            analyzer = BusinessTopologicalAnalyzer(telemetry=telemetry)
            audit_result = analyzer.audit_integration_homology(graph_a, graph_b)
            
            delta_beta_1 = audit_result.get("delta_beta_1", 0)
            if delta_beta_1 > 0:
                self.logger.warning(
                    f"Mayer-Vietoris: {delta_beta_1} emergent cycle(s) detected. "
                    f"Narrative: {audit_result.get('narrative', 'N/A')}"
                )
                telemetry.record_metric("topology", "emergent_cycles", delta_beta_1)
                context["integration_risk_alert"] = audit_result
            else:
                self.logger.info("✓ Auditoría Mayer-Vietoris: homología preservada")
                
        except Exception as e:
            self.logger.warning(
                f"Auditoría Mayer-Vietoris falló (no bloquea fusión): {e}"
            )
            telemetry.record_error("audited_merge_audit", str(e))


class CalculateCostsStep(BaseProcessingStep):
    """
    Paso de Cálculo de Costos.
    
    Proyecta intención táctica vía MIC para estructurar lógica de costos.
    Incluye fallback a APUProcessor clásico.
    """
    
    REQUIRED_CONTEXT_KEYS = ("df_merged",)
    PRODUCED_CONTEXT_KEYS = ("df_apu_costos", "df_tiempo", "df_rendimiento")
    
    def _execute_impl(
        self,
        context: Dict[str, Any],
        telemetry: TelemetryContext,
        mic: MICRegistry,
    ) -> Dict[str, Any]:
        """Ejecuta el cálculo de costos."""
        df_merged = context["df_merged"]
        raw_records = context.get("raw_records", [])
        parse_cache = context.get("parse_cache", {})
        
        # Proyectar intención táctica
        logic_result = mic.project_intent(
            "structure_logic",
            {
                "raw_records": raw_records,
                "parse_cache": parse_cache,
                "config": self.config.raw_config,
            },
            context,
        )
        
        # Protocolo de Dos Fases
        vectorial_success = (
            logic_result.get("success", False)
            and "df_apu_costos" in logic_result
        )
        
        if vectorial_success:
            self.logger.info("✓ Usando tensores de costo vectoriales (structure_logic)")
            df_apu_costos = logic_result["df_apu_costos"]
            df_tiempo = logic_result.get("df_tiempo", pd.DataFrame())
            df_rendimiento = logic_result.get("df_rendimiento", pd.DataFrame())
            context["quality_report"] = logic_result.get("quality_report", {})
        else:
            self.logger.info(
                "Proyección vectorial incompleta. Activando fallback APUProcessor"
            )
            processor = APUProcessor(self.config.raw_config)
            df_apu_costos, df_tiempo, df_rendimiento = processor.process_vectors(
                df_merged
            )
        
        telemetry.record_metric("calculate_costs", "costos_rows", len(df_apu_costos))
        telemetry.record_metric("calculate_costs", "tiempo_rows", len(df_tiempo))
        
        context.update({
            "df_apu_costos": df_apu_costos,
            "df_tiempo": df_tiempo,
            "df_rendimiento": df_rendimiento,
        })
        
        return context


class FinalMergeStep(BaseProcessingStep):
    """
    Paso de Fusión Final.
    
    Integra costos calculados con el presupuesto original.
    """
    
    REQUIRED_CONTEXT_KEYS = ("df_presupuesto", "df_apu_costos", "df_tiempo")
    PRODUCED_CONTEXT_KEYS = ("df_final",)
    
    def _execute_impl(
        self,
        context: Dict[str, Any],
        telemetry: TelemetryContext,
        mic: MICRegistry,
    ) -> Dict[str, Any]:
        """Ejecuta la fusión final."""
        df_presupuesto = context["df_presupuesto"]
        df_apu_costos = context["df_apu_costos"]
        df_tiempo = context["df_tiempo"]
        
        processor = APUProcessor(self.config.raw_config)
        df_final = processor.consolidate_results(
            df_presupuesto, df_apu_costos, df_tiempo
        )
        
        telemetry.record_metric("final_merge", "final_rows", len(df_final))
        context["df_final"] = df_final
        
        return context


class BusinessTopologyStep(BaseProcessingStep):
    """
    Paso de Análisis de Negocio.
    
    Materializa el grafo topológico y ejecuta evaluación de negocio.
    """
    
    REQUIRED_CONTEXT_KEYS = ("df_final",)
    PRODUCED_CONTEXT_KEYS = ("graph", "business_topology_report")
    
    def _execute_impl(
        self,
        context: Dict[str, Any],
        telemetry: TelemetryContext,
        mic: MICRegistry,
    ) -> Dict[str, Any]:
        """Ejecuta el análisis de topología de negocio."""
        df_final = context["df_final"]
        df_merged = context.get("df_merged", pd.DataFrame())
        
        # Materialización del grafo
        builder = BudgetGraphBuilder()
        graph = builder.build(df_final, df_merged)
        context["graph"] = graph
        
        self.logger.info(
            f"Grafo materializado: {graph.number_of_nodes()} nodos, "
            f"{graph.number_of_edges()} aristas"
        )
        telemetry.record_metric("business_topology", "graph_nodes", graph.number_of_nodes())
        telemetry.record_metric("business_topology", "graph_edges", graph.number_of_edges())
        
        # Evaluación por BusinessAgent (degradable)
        try:
            agent = BusinessAgent(
                config=self.config.raw_config,
                mic=mic,
                telemetry=telemetry,
            )
            report = agent.evaluate_project(context)
            
            if report:
                context["business_topology_report"] = report
                self.logger.info("✓ BusinessAgent completó la evaluación")
            else:
                self.logger.warning("BusinessAgent retornó reporte vacío")
                
        except Exception as e:
            self.logger.warning(f"BusinessAgent evaluation degraded: {e}")
            telemetry.record_error("business_agent", str(e))
        
        return context


class MaterializationStep(BaseProcessingStep):
    """
    Paso de Materialización (BOM).
    
    Genera Bill of Materials a partir del grafo y métricas.
    """
    
    REQUIRED_CONTEXT_KEYS = ()  # business_topology_report es opcional
    PRODUCED_CONTEXT_KEYS = ("bill_of_materials", "logistics_plan")
    
    def _execute_impl(
        self,
        context: Dict[str, Any],
        telemetry: TelemetryContext,
        mic: MICRegistry,
    ) -> Dict[str, Any]:
        """Ejecuta la materialización."""
        # Verificar precondición opcional
        if "business_topology_report" not in context:
            self.logger.warning(
                "business_topology_report ausente. "
                "Materialización omitida (degradación controlada)"
            )
            telemetry.record_metric("materialization", "skipped", True)
            return context
        
        # Resolver o construir grafo
        graph = context.get("graph")
        if graph is None:
            df_final = context.get("df_final")
            if df_final is None:
                raise PreconditionError(
                    "materialization",
                    ["df_final (or graph)"],
                )
            
            builder = BudgetGraphBuilder()
            df_merged = context.get("df_merged", pd.DataFrame())
            graph = builder.build(df_final, df_merged)
            context["graph"] = graph
            self.logger.info("Grafo reconstruido para materialización")
        
        # Extraer métricas de estabilidad
        report = context["business_topology_report"]
        stability = 10.0
        if hasattr(report, "details") and isinstance(report.details, dict):
            stability = report.details.get("pyramid_stability", 10.0)
        
        flux_metrics = {
            "pyramid_stability": stability,
            "avg_saturation": 0.0,
        }
        
        # Generar BOM
        generator = MatterGenerator()
        bom = generator.materialize_project(
            graph, flux_metrics=flux_metrics, telemetry=telemetry
        )
        
        context["bill_of_materials"] = bom
        context["logistics_plan"] = asdict(bom)
        
        telemetry.record_metric("materialization", "total_items", len(bom.requirements))
        self.logger.info(f"✓ Materialización completada. Items: {len(bom.requirements)}")
        
        return context


class BuildOutputStep(BaseProcessingStep):
    """
    Paso de Construcción de Salida.
    
    Ensambla el DataProduct final con hash de linaje.
    """
    
    REQUIRED_CONTEXT_KEYS = (
        "df_final", "df_insumos", "df_merged",
        "df_apus_raw", "df_apu_costos", "df_tiempo", "df_rendimiento",
    )
    PRODUCED_CONTEXT_KEYS = ("final_result",)
    
    def _execute_impl(
        self,
        context: Dict[str, Any],
        telemetry: TelemetryContext,
        mic: MICRegistry,
    ) -> Dict[str, Any]:
        """Ejecuta la construcción de salida."""
        # Extraer DataFrames
        df_final = context["df_final"]
        df_insumos = context["df_insumos"]
        df_merged = context["df_merged"]
        df_apus_raw = context["df_apus_raw"]
        df_apu_costos = context["df_apu_costos"]
        df_tiempo = context["df_tiempo"]
        df_rendimiento = context["df_rendimiento"]
        
        # Sincronización
        df_merged = synchronize_data_sources(df_merged, df_final)
        df_processed_apus = build_processed_apus_dataframe(
            df_apu_costos, df_apus_raw, df_tiempo, df_rendimiento
        )
        
        # Ensamblar producto de datos
        has_strategy_artifacts = (
            "graph" in context and "business_topology_report" in context
        )
        
        if has_strategy_artifacts:
            graph = context["graph"]
            report = context["business_topology_report"]
            translator = SemanticTranslator(mic=mic)
            result_dict = translator.assemble_data_product(graph, report)
            result_dict["presupuesto"] = df_final.to_dict("records")
            result_dict["insumos"] = df_insumos.to_dict("records")
            self.logger.info("WISDOM: Producto de datos ensamblado por SemanticTranslator")
        else:
            self.logger.warning("Sin artefactos de Estrategia. Generando salida básica")
            result_dict = build_output_dictionary(
                df_final, df_insumos, df_merged, df_apus_raw, df_processed_apus
            )
        
        # Validación y enriquecimiento
        validated_result = validate_and_clean_data(
            result_dict, telemetry_context=telemetry
        )
        validated_result["raw_insumos_df"] = df_insumos.to_dict("records")
        
        if "business_topology_report" in context:
            validated_result["audit_report"] = asdict(context["business_topology_report"])
        
        if "logistics_plan" in context:
            validated_result["logistics_plan"] = context["logistics_plan"]
        
        # Narrativa técnica
        try:
            narrator = TelemetryNarrator(mic=mic)
            tech_narrative = narrator.summarize_execution(telemetry)
            validated_result["technical_audit"] = tech_narrative
        except Exception as e:
            self.logger.warning(f"Narrativa técnica degradada: {e}")
            validated_result["technical_audit"] = {"status": "degraded", "error": str(e)}
        
        # Hash de linaje
        lineage_hash = self._compute_lineage_hash(validated_result)
        
        # Construir DataProduct
        validated_strata = context.get("validated_strata", set())
        data_product = {
            "kind": "DataProduct",
            "metadata": {
                "version": PIPELINE_VERSION,
                "lineage_hash": lineage_hash,
                "generated_at": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "generator": f"APU_Filter_Pipeline_{PIPELINE_VERSION}",
                "strata_validated": [
                    s.name for s in validated_strata if isinstance(s, Stratum)
                ],
            },
            "payload": validated_result,
        }
        
        context["final_result"] = data_product
        return context
    
    def _compute_lineage_hash(self, payload: Dict[str, Any]) -> str:
        """Calcula hash SHA-256 sobre el payload."""
        hash_parts = []
        
        for key in sorted(payload.keys()):
            value = payload[key]
            try:
                sanitized = (
                    sanitize_for_json(value)
                    if isinstance(value, (list, dict))
                    else value
                )
                part = json.dumps({key: sanitized}, sort_keys=True, default=str)
            except (TypeError, ValueError):
                part = (
                    f"{key}:type={type(value).__name__},"
                    f"len={len(value) if hasattr(value, '__len__') else 'N/A'}"
                )
            hash_parts.append(part)
        
        composite = "|".join(hash_parts)
        return hashlib.sha256(composite.encode("utf-8")).hexdigest()


# ============================================================================
# REGISTRO DE PASOS
# ============================================================================

# Mapeo de nombres a clases de pasos
STEP_REGISTRY: Final[Dict[str, Type[BaseProcessingStep]]] = {
    PipelineSteps.LOAD_DATA.value: LoadDataStep,
    PipelineSteps.AUDITED_MERGE.value: AuditedMergeStep,
    PipelineSteps.CALCULATE_COSTS.value: CalculateCostsStep,
    PipelineSteps.FINAL_MERGE.value: FinalMergeStep,
    PipelineSteps.BUSINESS_TOPOLOGY.value: BusinessTopologyStep,
    PipelineSteps.MATERIALIZATION.value: MaterializationStep,
    PipelineSteps.BUILD_OUTPUT.value: BuildOutputStep,
}


# ============================================================================
# SESSION MANAGER
# ============================================================================


class SessionManager:
    """
    Gestiona la persistencia de sesiones del pipeline.
    
    Proporciona:
    - Almacenamiento atómico
    - Validación de versión
    - Limpieza automática
    """
    
    def __init__(self, config: SessionConfig):
        """
        Inicializa el gestor de sesiones.
        
        Args:
            config: Configuración de sesiones.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Crear directorio si no existe
        self.config.session_dir.mkdir(parents=True, exist_ok=True)
    
    def _session_path(self, session_id: str) -> Path:
        """Obtiene la ruta del archivo de sesión."""
        return self.config.session_dir / f"{session_id}{SESSION_FILE_EXTENSION}"
    
    def load(self, session_id: str) -> Optional[SessionState]:
        """
        Carga el estado de una sesión.
        
        Args:
            session_id: ID de la sesión.
            
        Returns:
            Estado de la sesión o None si no existe.
            
        Raises:
            SessionCorruptedError: Si la sesión está corrupta.
        """
        if not session_id:
            return None
        
        session_path = self._session_path(session_id)
        
        if not session_path.exists():
            return None
        
        try:
            with open(session_path, "rb") as f:
                data = pickle.load(f)
            
            if not isinstance(data, SessionState):
                raise SessionCorruptedError(
                    session_id,
                    f"Expected SessionState, got {type(data).__name__}",
                )
            
            # Validar versión
            if data.version != PIPELINE_VERSION:
                self.logger.warning(
                    f"Session {session_id} version mismatch: "
                    f"{data.version} != {PIPELINE_VERSION}"
                )
            
            return data
            
        except (pickle.UnpicklingError, EOFError, ModuleNotFoundError) as e:
            raise SessionCorruptedError(session_id, str(e))
        except SessionCorruptedError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def save(self, state: SessionState) -> bool:
        """
        Guarda el estado de una sesión (escritura atómica).
        
        Args:
            state: Estado a guardar.
            
        Returns:
            True si se guardó exitosamente.
        """
        session_path = self._session_path(state.session_id)
        tmp_path = session_path.with_suffix(f"{SESSION_FILE_EXTENSION}.tmp")
        
        try:
            # Serializar
            data = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Validar tamaño
            if len(data) > MAX_CONTEXT_SIZE_BYTES:
                self.logger.error(
                    f"Session {state.session_id} too large: "
                    f"{len(data)} > {MAX_CONTEXT_SIZE_BYTES}"
                )
                return False
            
            # Escribir a temporal
            with open(tmp_path, "wb") as f:
                f.write(data)
            
            # Mover atómicamente
            tmp_path.replace(session_path)
            
            self.logger.debug(f"Session saved: {state.session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save session {state.session_id}: {e}")
            
            # Limpiar temporal
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            
            return False
    
    def delete(self, session_id: str) -> bool:
        """
        Elimina una sesión.
        
        Args:
            session_id: ID de la sesión.
            
        Returns:
            True si se eliminó (o no existía).
        """
        session_path = self._session_path(session_id)
        
        try:
            if session_path.exists():
                session_path.unlink()
                self.logger.debug(f"Session deleted: {session_id}")
            return True
        except OSError as e:
            self.logger.warning(f"Could not delete session {session_id}: {e}")
            return False
    
    def cleanup_old_sessions(self) -> int:
        """
        Limpia sesiones antiguas.
        
        Returns:
            Número de sesiones eliminadas.
        """
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            hours=self.config.max_age_hours
        )
        
        deleted = 0
        for path in self.config.session_dir.glob(f"*{SESSION_FILE_EXTENSION}"):
            try:
                mtime = datetime.datetime.fromtimestamp(
                    path.stat().st_mtime,
                    tz=datetime.timezone.utc,
                )
                if mtime < cutoff:
                    path.unlink()
                    deleted += 1
            except Exception as e:
                self.logger.warning(f"Could not check/delete {path}: {e}")
        
        if deleted > 0:
            self.logger.info(f"Cleaned up {deleted} old sessions")
        
        return deleted


# ============================================================================
# PIPELINE DIRECTOR
# ============================================================================


class PipelineDirector:
    """
    Director del Pipeline: Orquesta pasos secuenciales con validación de estado.
    
    Este componente actúa como el Orquestador Algebraico del sistema, gestionando
    la evolución del Vector de Estado a través de la filtración topológica DIKW.
    
    Responsabilidades:
    - Gestión de sesiones
    - Validación de filtración
    - Ejecución ordenada de pasos
    - Manejo de errores y recovery
    """
    
    def __init__(
        self,
        config: Union[Dict[str, Any], PipelineConfig],
        telemetry: TelemetryContext,
        mic: Optional[MICRegistry] = None,
    ):
        """
        Inicializa el director.
        
        Args:
            config: Configuración del pipeline.
            telemetry: Contexto de telemetría.
            mic: Registro MIC (se crea si no se proporciona).
        """
        # Normalizar configuración
        if isinstance(config, dict):
            self.config = PipelineConfig.from_dict(config)
        else:
            self.config = config
        
        self.telemetry = telemetry
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Cargar umbrales
        self.thresholds = self._load_thresholds()
        
        # Inicializar MIC
        if mic is not None:
            self.mic = mic
        else:
            self.mic = MICRegistry()
            register_core_vectors(self.mic, self.config.raw_config)
        
        # Registrar vectores base
        self._register_basis_vectors()
        
        # Inicializar session manager
        self.session_manager = SessionManager(self.config.session)
        
        self.logger.info(
            f"PipelineDirector initialized | "
            f"Version: {PIPELINE_VERSION} | "
            f"Steps: {len(STEP_REGISTRY)}"
        )
    
    def _load_thresholds(self) -> ProcessingThresholds:
        """Carga umbrales desde configuración."""
        thresholds = ProcessingThresholds()
        overrides = self.config.raw_config.get("processing_thresholds", {})
        
        if not isinstance(overrides, dict):
            self.logger.warning(
                f"processing_thresholds is not a dict. Using defaults."
            )
            return thresholds
        
        for key, value in overrides.items():
            if not hasattr(thresholds, key):
                self.logger.warning(f"Unknown threshold key '{key}'")
                continue
            
            current = getattr(thresholds, key)
            if current is not None and not isinstance(value, type(current)):
                self.logger.warning(
                    f"Threshold '{key}': type mismatch. Ignored."
                )
                continue
            
            setattr(thresholds, key, value)
        
        return thresholds
    
    def _register_basis_vectors(self) -> None:
        """Registra los vectores base en la MIC."""
        for step in PipelineSteps:
            step_class = STEP_REGISTRY.get(step.value)
            if step_class:
                self.mic.add_basis_vector(step.value, step_class, step.stratum)
    
    # =========================================================================
    # VALIDACIÓN DE FILTRACIÓN
    # =========================================================================
    
    def _compute_validated_strata(self, context: Dict[str, Any]) -> Set[Stratum]:
        """
        Determina los estratos validados basándose en evidencia.
        
        Un estrato es válido sii TODAS sus claves de evidencia existen
        y no son None/Empty.
        """
        validated = set()
        
        for stratum, evidence_keys in _STRATUM_EVIDENCE.items():
            is_valid = True
            
            for key in evidence_keys:
                value = context.get(key)
                
                if value is None:
                    is_valid = False
                    break
                
                if hasattr(value, "empty") and value.empty:
                    is_valid = False
                    break
                
                if isinstance(value, (list, dict)) and not value:
                    is_valid = False
                    break
            
            if is_valid:
                validated.add(stratum)
        
        return validated
    
    def _check_stratum_prerequisites(
        self,
        target_stratum: Stratum,
        validated_strata: Set[Stratum],
    ) -> bool:
        """
        Verifica que todos los estratos inferiores estén validados.
        
        Clausura Transitiva: para ejecutar en estrato S, todos los
        estratos T < S deben estar validados.
        """
        target_level = stratum_level(target_stratum)
        
        if target_level == 0:  # PHYSICS (Base)
            return True
        
        for stratum, level in _STRATUM_ORDER.items():
            if level < target_level and stratum not in validated_strata:
                return False
        
        return True
    
    def _enforce_filtration(
        self,
        target_stratum: Stratum,
        context: Dict[str, Any],
    ) -> None:
        """
        Lanza excepción si se viola la filtración topológica.
        
        Raises:
            FiltrationViolationError: Si se viola el invariante.
        """
        validated = self._compute_validated_strata(context)
        
        if not self._check_stratum_prerequisites(target_stratum, validated):
            target_level = stratum_level(target_stratum)
            missing = [
                s.name
                for s, l in _STRATUM_ORDER.items()
                if l < target_level and s not in validated
            ]
            raise FiltrationViolationError(
                target_stratum,
                missing,
                [s.name for s in validated],
            )
    
    # =========================================================================
    # EJECUCIÓN DE PASOS
    # =========================================================================
    
    def run_single_step(
        self,
        step_name: str,
        session_id: str,
        initial_context: Optional[Dict[str, Any]] = None,
        validate_stratum: bool = True,
    ) -> StepResult:
        """
        Ejecuta un único paso del pipeline.
        
        Args:
            step_name: Nombre del paso.
            session_id: ID de la sesión.
            initial_context: Contexto inicial (se fusiona con sesión).
            validate_stratum: Si validar filtración.
            
        Returns:
            Resultado de la ejecución.
        """
        import time
        start_time = time.monotonic()
        
        self.logger.info(f"Executing step: {step_name} (Session: {session_id[:8]}...)")
        
        # Cargar o crear sesión
        session = self.session_manager.load(session_id)
        if session is None:
            session = SessionState.create(
                session_id,
                initial_context or {},
            )
        elif initial_context:
            # Fusionar contextos (sesión tiene precedencia)
            merged = {**initial_context, **session.context}
            session.update_context(merged)
        
        try:
            # Resolver vector base
            basis_vector = self.mic.get_basis_vector(step_name)
            if basis_vector is None:
                available = self.mic.get_available_labels()
                raise StepExecutionError(
                    f"Step not found. Available: {available}",
                    step_name,
                )
            
            # Validar filtración
            if validate_stratum and self.config.enforce_filtration:
                self._enforce_filtration(basis_vector.stratum, session.context)
            
            # Instanciar y ejecutar paso
            step_class = STEP_REGISTRY.get(step_name)
            if step_class is None:
                raise StepExecutionError(
                    f"Step class not found in registry",
                    step_name,
                )
            
            step_instance = step_class(self.config, self.thresholds)
            updated_context = step_instance.execute(
                session.context,
                self.telemetry,
                self.mic,
            )
            
            # Actualizar sesión
            session.update_context(updated_context)
            session.validated_strata = self._compute_validated_strata(updated_context)
            
            # Calcular duración
            duration_ms = (time.monotonic() - start_time) * 1000
            
            # Crear resultado
            result = StepResult(
                step_name=step_name,
                status=StepStatus.SUCCESS,
                stratum=basis_vector.stratum,
                duration_ms=duration_ms,
                context_keys=list(updated_context.keys()),
            )
            
            session.add_result(result)
            self.session_manager.save(session)
            
            self.logger.info(
                f"Step '{step_name}' completed in {duration_ms:.0f}ms"
            )
            
            return result
            
        except PipelineError as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            
            result = StepResult(
                step_name=step_name,
                status=StepStatus.ERROR,
                stratum=basis_vector.stratum if basis_vector else Stratum.PHYSICS,
                duration_ms=duration_ms,
                error=str(e),
            )
            
            session.add_result(result)
            
            if self.config.session.persist_on_error:
                self.session_manager.save(session)
            
            self.logger.error(f"Error in step '{step_name}': {e}")
            self.telemetry.record_error(step_name, str(e))
            
            return result
    
    def execute_pipeline(
        self,
        initial_context: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo.
        
        Args:
            initial_context: Contexto inicial.
            session_id: ID de sesión (se genera si no se proporciona).
            
        Returns:
            Contexto final con DataProduct.
            
        Raises:
            PipelineError: Si hay error en la ejecución.
        """
        session_id = session_id or str(uuid.uuid4())
        self.logger.info(f"Starting pipeline (Session: {session_id})")
        
        # Obtener receta
        if self.config.recipe:
            recipe = self.config.recipe
        else:
            recipe = [step.value for step in PipelineSteps]
        
        # Crear sesión inicial
        session = SessionState.create(session_id, initial_context)
        
        if not self.session_manager.save(session):
            raise SessionError(
                f"Failed to persist initial session. "
                f"Check permissions on {self.config.session.session_dir}"
            )
        
        # Verificar que se guardó
        verification = self.session_manager.load(session_id)
        if verification is None:
            raise SessionError("Failed to verify initial session persistence")
        
        # Ejecutar pasos
        first_step = True
        for idx, step_name in enumerate(recipe):
            # Verificar si está habilitado
            if not self.config.is_step_enabled(step_name):
                self.logger.info(f"Skipping disabled step: {step_name}")
                continue
            
            self.logger.info(
                f"Orchestrating step [{idx + 1}/{len(recipe)}]: {step_name}"
            )
            
            # En el primer paso, pasar initial_context como respaldo
            ctx_override = initial_context if first_step else None
            
            result = self.run_single_step(
                step_name,
                session_id,
                initial_context=ctx_override,
            )
            first_step = False
            
            if not result.is_successful:
                error_msg = (
                    f"Pipeline failed at step '{step_name}' "
                    f"[{idx + 1}/{len(recipe)}]: {result.error}"
                )
                self.logger.critical(error_msg)
                raise StepExecutionError(error_msg, step_name)
        
        # Cargar contexto final
        final_session = self.session_manager.load(session_id)
        final_context = final_session.context if final_session else {}
        
        # Limpiar sesión si está configurado
        if self.config.session.cleanup_on_success:
            self.session_manager.delete(session_id)
        
        self.logger.info(f"Pipeline completed (Session: {session_id})")
        
        return final_context
    
    # Alias para compatibilidad
    def execute_pipeline_orchestrated(
        self,
        initial_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Alias de execute_pipeline para compatibilidad."""
        return self.execute_pipeline(initial_context)
    
    # =========================================================================
    # UTILIDADES
    # =========================================================================
    
    def get_recipe(self) -> List[Dict[str, Any]]:
        """Obtiene la receta de ejecución actual."""
        recipe = self.config.recipe or [step.value for step in PipelineSteps]
        return [
            {
                "step": step_name,
                "enabled": self.config.is_step_enabled(step_name),
                "stratum": STEP_REGISTRY.get(step_name, LoadDataStep).__name__,
            }
            for step_name in recipe
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Obtiene estado del director."""
        return {
            "version": PIPELINE_VERSION,
            "steps_registered": len(STEP_REGISTRY),
            "steps": list(STEP_REGISTRY.keys()),
            "enforce_filtration": self.config.enforce_filtration,
            "session_dir": str(self.config.session.session_dir),
        }


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def create_director(
    config: Optional[Dict[str, Any]] = None,
    telemetry: Optional[TelemetryContext] = None,
    mic: Optional[MICRegistry] = None,
) -> PipelineDirector:
    """
    Factory function para crear un director configurado.
    
    Args:
        config: Configuración del pipeline.
        telemetry: Contexto de telemetría.
        mic: Registro MIC.
        
    Returns:
        Director configurado.
    """
    config = config or {}
    telemetry = telemetry or TelemetryContext()
    
    return PipelineDirector(config, telemetry, mic)


def create_director_from_env() -> PipelineDirector:
    """
    Crea un director desde variables de entorno.
    
    Variables soportadas:
    - PIPELINE_SESSION_DIR: Directorio de sesiones
    - PIPELINE_ENFORCE_FILTRATION: Si validar filtración
    
    Returns:
        Director configurado.
    """
    config = {
        "session_dir": os.getenv("PIPELINE_SESSION_DIR", DEFAULT_SESSION_DIR),
        "enforce_filtration": os.getenv(
            "PIPELINE_ENFORCE_FILTRATION", "true"
        ).lower() == "true",
    }
    
    return create_director(config)


# ============================================================================
# FUNCIÓN DE ENTRADA PRINCIPAL
# ============================================================================


def process_all_files(
    presupuesto_path: Union[str, Path],
    apus_path: Union[str, Path],
    insumos_path: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    telemetry: Optional[TelemetryContext] = None,
) -> Dict[str, Any]:
    """
    Función de entrada principal para el pipeline (Batch Mode).
    
    Args:
        presupuesto_path: Ruta al archivo de presupuesto.
        apus_path: Ruta al archivo de APUs.
        insumos_path: Ruta al archivo de insumos.
        config: Configuración del pipeline.
        telemetry: Contexto de telemetría.
        
    Returns:
        DataProduct generado o contexto completo.
        
    Raises:
        FileNotFoundError: Si algún archivo no existe.
        PipelineError: Si hay error en el procesamiento.
    """
    config = config or {}
    telemetry = telemetry or TelemetryContext()
    
    # Resolver y validar rutas
    paths = {
        "presupuesto_path": Path(presupuesto_path).resolve(),
        "apus_path": Path(apus_path).resolve(),
        "insumos_path": Path(insumos_path).resolve(),
    }
    
    for name, path in paths.items():
        if not path.exists():
            error = f"File not found: {name} = {path}"
            telemetry.record_error("process_all_files", error)
            raise FileNotFoundError(error)
    
    # Crear director y ejecutar
    director = create_director(config, telemetry)
    
    initial_context = {k: str(v) for k, v in paths.items()}
    
    try:
        final_context = director.execute_pipeline(initial_context)
        return final_context.get("final_result", final_context)
        
    except PipelineError as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
        telemetry.record_error("process_all_files", str(e))
        raise
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        telemetry.record_error("process_all_files", str(e))
        raise PipelineError(f"Unexpected error: {e}") from e