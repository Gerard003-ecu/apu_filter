"""
Módulo: Business Agent (El Cerebro Ejecutivo del Consejo)
=========================================================

Este componente actúa como el nodo de síntesis superior en la jerarquía DIKW.
Su función es integrar los hallazgos del Arquitecto (Topología) y el Oráculo
(Finanzas) para emitir un "Veredicto Holístico" sobre la viabilidad del proyecto.

Opera bajo el principio de **"No hay Estrategia sin Física"**, negándose a emitir
juicios financieros si la estabilidad estructural no ha sido validada por la MIC.

Fundamentos Teóricos:
─────────────────────

1. Síntesis Topológico-Financiera (El Funtor de Decisión):
   F: (T × Φ) → D
   donde T = espacio topológico, Φ = espacio financiero, D = espacio de decisión

2. Protocolo Challenger (Auditoría Adversarial):
   Reglas de veto lógico con pensamiento lateral vía MIC

3. Termodinámica del Valor:
   Evaluación usando T_sys (volatilidad), exergía (trabajo útil) y entropía

4. Cliente de la MIC (Gobernanza Algebraica):
   Proyección de vectores de intención con validación de clausura transitiva

Arquitectura:
─────────────
- `BusinessAgent`: Fachada principal del pipeline de evaluación
- `RiskChallenger`: Motor de auditoría adversarial con pivotes estratégicos
- `DataFrameValidator`: Validación estructural y semántica de datos
- `TopologyBuilder`: Construcción del modelo topológico
- `DecisionAlgebra`: Operaciones algebraicas para síntesis multicriterio
- `ReportComposer`: Generación de reportes ejecutivos
"""

from __future__ import annotations

import copy
import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from functools import cached_property, lru_cache
from typing import (
    Any, Callable, Dict, Final, FrozenSet, Generic, Iterator,
    List, Mapping, NamedTuple, Optional, Protocol, Sequence, Set,
    Tuple, Type, TypeVar, Union, cast, runtime_checkable
)

import numpy as np
import pandas as pd

# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# IMPORTS CON FALLBACK
# =============================================================================

def _safe_import(module_path: str, class_name: str) -> Optional[Type]:
    """Importación segura con logging."""
    try:
        parts = module_path.split(".")
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name, None)
    except ImportError as e:
        logger.debug(f"Import opcional fallido: {module_path}.{class_name} - {e}")
        return None


# NetworkX con validación
try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:
    nx = None  # type: ignore
    _HAS_NETWORKX = False
    logger.warning("NetworkX no disponible - funcionalidad topológica limitada")


# Componentes del sistema
try:
    from agent.business_topology import (
        BudgetGraphBuilder,
        BusinessTopologicalAnalyzer,
        ConstructionRiskReport,
    )
except ImportError:
    BudgetGraphBuilder = None  # type: ignore
    BusinessTopologicalAnalyzer = None  # type: ignore
    ConstructionRiskReport = None  # type: ignore
    logger.warning("business_topology no disponible")

try:
    from app.constants import ColumnNames
except ImportError:
    # Fallback con columnas estándar
    class ColumnNames:
        CODIGO_APU = "codigo_apu"
        DESCRIPCION_APU = "descripcion_apu"
        VALOR_TOTAL = "valor_total"
        DESCRIPCION_INSUMO = "descripcion_insumo"
        CANTIDAD_APU = "cantidad_apu"
        COSTO_INSUMO_EN_APU = "costo_insumo_en_apu"

try:
    from app.financial_engine import FinancialConfig, FinancialEngine
except ImportError:
    FinancialConfig = None  # type: ignore
    FinancialEngine = None  # type: ignore

try:
    from app.schemas import Stratum
except ImportError:
    from enum import IntEnum
    class Stratum(IntEnum):
        WISDOM = 0
        STRATEGY = 1
        TACTICS = 2
        PHYSICS = 3

try:
    from app.semantic_translator import SemanticTranslator
except ImportError:
    SemanticTranslator = None  # type: ignore

try:
    from app.telemetry import TelemetryContext
except ImportError:
    @dataclass
    class TelemetryContext:
        def record_error(self, category: str, message: str) -> None:
            logger.error(f"[{category}] {message}")

try:
    from app.tools_interface import MICRegistry
except ImportError:
    MICRegistry = None  # type: ignore


# =============================================================================
# CONSTANTES Y CONFIGURACIÓN
# =============================================================================

@dataclass(frozen=True, slots=True)
class RiskChallengerThresholds:
    """
    Umbrales para el Risk Challenger.
    
    Todos los umbrales están en el rango [0, 1] y representan
    puntos de decisión para vetos y alertas.
    """
    critical_stability: float = 0.70      # Ψ < threshold → Veto crítico
    warning_stability: float = 0.85       # Ψ < threshold → Advertencia
    coherence_minimum: float = 0.60       # Coherencia mínima aceptable
    cycle_density_limit: float = 0.33     # β₁/n máximo aceptable
    integrity_penalty_veto: float = 0.30  # Penalización por veto crítico
    integrity_penalty_warn: float = 0.15  # Penalización por advertencia
    
    def __post_init__(self) -> None:
        """Valida que todos los umbrales estén en [0, 1]."""
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if not (0 <= value <= 1):
                raise ValueError(f"Umbral {field_name} fuera de rango [0, 1]: {value}")


@dataclass(frozen=True, slots=True)
class DecisionWeights:
    """
    Pesos para la combinación convexa de dimensiones de decisión.
    
    Los pesos representan la importancia relativa de cada dimensión
    en el vector de decisión final. Deben sumar 1.0 (o serán normalizados).
    """
    topology: float = 0.40       # Peso de dimensión topológica
    finance: float = 0.40        # Peso de dimensión financiera
    thermodynamics: float = 0.20 # Peso de dimensión termodinámica
    
    def __post_init__(self) -> None:
        """Valida que los pesos sean no negativos."""
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"Peso {field_name} no puede ser negativo: {value}")
    
    @property
    def normalized(self) -> 'DecisionWeights':
        """Retorna pesos normalizados a suma 1."""
        total = self.topology + self.finance + self.thermodynamics
        if total <= 0:
            return DecisionWeights(1/3, 1/3, 1/3)
        return DecisionWeights(
            topology=self.topology / total,
            finance=self.finance / total,
            thermodynamics=self.thermodynamics / total
        )
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Retorna como tupla (α, β, γ)."""
        return (self.topology, self.finance, self.thermodynamics)


class FinancialDefaults:
    """Valores por defecto para parámetros financieros."""
    INITIAL_INVESTMENT: Final[float] = 1_000_000.0
    CASH_FLOW_RATIO: Final[float] = 0.30
    CASH_FLOW_PERIODS: Final[int] = 5
    COST_STD_DEV_RATIO: Final[float] = 0.15
    PROJECT_VOLATILITY: Final[float] = 0.20


class NumericalConstants:
    """Constantes numéricas para estabilidad de cálculos."""
    EPSILON: Final[float] = 1e-10
    LOG_MIN: Final[float] = 1e-30
    SCORE_MAX: Final[float] = 100.0
    DECAY_RATE_FRAGMENTATION: Final[float] = 0.693  # ln(2)


# =============================================================================
# EXCEPCIONES
# =============================================================================

class BusinessAgentError(Exception):
    """Clase base para excepciones del Business Agent."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": str(self),
            "error_type": type(self).__name__,
            "details": self.details,
            "timestamp": self.timestamp
        }


class ConfigurationError(BusinessAgentError):
    """Error en configuración del agente."""
    pass


class ValidationError(BusinessAgentError):
    """Error en validación de datos de entrada."""
    pass


class TopologicalAnomalyError(BusinessAgentError):
    """Anomalía detectada en estructura topológica."""
    pass


class MICHierarchyViolationError(BusinessAgentError):
    """Violación de jerarquía de estratos en la MIC."""
    pass


class FinancialProjectionError(BusinessAgentError):
    """Error en proyección financiera."""
    pass


class SynthesisError(BusinessAgentError):
    """Error en síntesis de reporte estratégico."""
    pass


# =============================================================================
# ENUMERACIONES
# =============================================================================

class RiskClassification(Enum):
    """Clasificación de riesgo financiero."""
    SAFE = "SAFE"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"
    
    @classmethod
    def from_string(cls, value: str) -> 'RiskClassification':
        """Parsea string a clasificación normalizada."""
        if not isinstance(value, str):
            return cls.UNKNOWN
        
        normalized = value.upper().strip()
        
        safe_kw = {"LOW", "BAJO", "SAFE", "SEGURO", "MINIMAL", "MÍNIMO"}
        moderate_kw = {"MODERATE", "MODERADO", "MEDIUM", "MEDIO"}
        high_kw = {"HIGH", "ALTO"}
        critical_kw = {"CRITICAL", "CRÍTICO", "SEVERE", "SEVERO"}
        
        if any(kw in normalized for kw in safe_kw):
            return cls.SAFE
        elif any(kw in normalized for kw in moderate_kw):
            return cls.MODERATE
        elif any(kw in normalized for kw in critical_kw):
            return cls.CRITICAL
        elif any(kw in normalized for kw in high_kw):
            return cls.HIGH
        return cls.UNKNOWN


class VetoSeverity(Enum):
    """Severidad del veto emitido."""
    CRITICO = "CRÍTICO"
    SEVERO = "SEVERO"
    MODERADO = "MODERADO"
    LEVE = "LEVE"


class PivotType(Enum):
    """Tipos de pivote lateral para excepciones estratégicas."""
    MONOPOLIO_COBERTURADO = "MONOPOLIO_COBERTURADO"
    OPCION_ESPERA = "OPCION_ESPERA"
    CUARENTENA_TOPOLOGICA = "CUARENTENA_TOPOLOGICA"


# =============================================================================
# ESTRUCTURAS DE DATOS
# =============================================================================

@dataclass(frozen=True, slots=True)
class FinancialParameters:
    """
    Parámetros financieros validados para análisis del proyecto.
    
    Inmutable para garantizar consistencia durante el pipeline.
    
    Attributes:
        initial_investment: Inversión inicial (debe ser > 0)
        cash_flows: Flujos de caja proyectados (inmutable)
        cost_std_dev: Desviación estándar de costos
        project_volatility: Volatilidad estimada [0, 1]
    """
    initial_investment: float
    cash_flows: Tuple[float, ...]
    cost_std_dev: float
    project_volatility: float
    
    def __post_init__(self) -> None:
        """Valida invariantes financieros."""
        if self.initial_investment <= 0:
            raise ValueError("La inversión inicial debe ser positiva")
        if self.cost_std_dev < 0:
            raise ValueError("La desviación estándar no puede ser negativa")
        if not (0 <= self.project_volatility <= 1):
            raise ValueError("La volatilidad debe estar en [0, 1]")
    
    @property
    def periods(self) -> int:
        """Número de períodos de flujo de caja."""
        return len(self.cash_flows)
    
    @property
    def total_cash_flow(self) -> float:
        """Suma total de flujos de caja."""
        return sum(self.cash_flows)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "initial_investment": self.initial_investment,
            "cash_flows": list(self.cash_flows),
            "cost_std_dev": self.cost_std_dev,
            "project_volatility": self.project_volatility,
            "periods": self.periods,
            "total_cash_flow": self.total_cash_flow,
        }


@dataclass(frozen=True, slots=True)
class BettiNumbers:
    """
    Números de Betti del complejo simplicial.
    
    β₀: Componentes conexas
    β₁: Ciclos independientes
    β₂: Cavidades (generalmente 0 para grafos)
    """
    beta_0: int = 1
    beta_1: int = 0
    beta_2: int = 0
    
    @property
    def euler_characteristic(self) -> int:
        """Característica de Euler χ = β₀ - β₁ + β₂."""
        return self.beta_0 - self.beta_1 + self.beta_2
    
    @property
    def is_connected(self) -> bool:
        """True si el espacio es conexo (β₀ = 1)."""
        return self.beta_0 == 1
    
    @property
    def has_cycles(self) -> bool:
        """True si existen ciclos (β₁ > 0)."""
        return self.beta_1 > 0
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BettiNumbers':
        """Crea desde diccionario."""
        return cls(
            beta_0=int(d.get("beta_0", 1)),
            beta_1=int(d.get("beta_1", 0)),
            beta_2=int(d.get("beta_2", 0))
        )
    
    def to_dict(self) -> Dict[str, int]:
        return {"beta_0": self.beta_0, "beta_1": self.beta_1, "beta_2": self.beta_2}


@dataclass(frozen=True, slots=True)
class TopologicalMetricsBundle:
    """
    Conjunto cohesivo de métricas topológicas del presupuesto.
    
    Agrupa los invariantes topológicos para facilitar su transporte
    entre componentes del pipeline. Inmutable para thread-safety.
    
    Attributes:
        betti: Números de Betti
        pyramid_stability: Índice de estabilidad piramidal Ψ ∈ [0, 1]
        n_nodes: Número de nodos en el grafo
        n_edges: Número de aristas
        is_connected: Si el grafo es conexo
        persistence_diagram: Diagrama de persistencia (opcional)
    """
    betti: BettiNumbers
    pyramid_stability: float
    n_nodes: int
    n_edges: int
    is_connected: bool
    persistence_diagram: Optional[Tuple[Tuple[float, float], ...]] = None
    
    @property
    def structural_coherence(self) -> float:
        """
        Índice de coherencia estructural mediante invariantes topológicos.
        
        Fórmula:
            C(K) = exp(-λ₀·max(0, β₀-1)) × exp(-λ₁·β₁/√n) × Ψ
        
        donde:
        - β₀: Componentes conexas (ideal: β₀ = 1)
        - β₁: Ciclos independientes
        - n: Número de vértices (normalización)
        - Ψ: Estabilidad piramidal
        - λ₀, λ₁: Tasas de decaimiento
        
        Returns:
            Índice ∈ [0, 1], donde 1 = máxima coherencia
        """
        nc = NumericalConstants()
        
        n_vertices = max(self.n_nodes, 1)
        
        # Tasas de decaimiento
        lambda_0 = nc.DECAY_RATE_FRAGMENTATION  # ln(2) ≈ 0.693
        lambda_1 = nc.DECAY_RATE_FRAGMENTATION / max(1, math.sqrt(n_vertices))
        
        # Penalización por fragmentación (β₀ > 1)
        excess_components = max(0, self.betti.beta_0 - 1)
        fragmentation_factor = math.exp(-lambda_0 * excess_components)
        
        # Penalización por ciclos
        cycle_factor = math.exp(-lambda_1 * self.betti.beta_1)
        
        # Composición multiplicativa
        raw_coherence = fragmentation_factor * cycle_factor * self.pyramid_stability
        
        return max(0.0, min(1.0, raw_coherence))
    
    @property
    def cycle_density(self) -> float:
        """Densidad de ciclos: β₁/n."""
        if self.n_nodes <= 0:
            return 0.0
        return self.betti.beta_1 / self.n_nodes
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "betti_numbers": self.betti.to_dict(),
            "pyramid_stability": self.pyramid_stability,
            "structural_coherence": self.structural_coherence,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "is_connected": self.is_connected,
            "cycle_density": self.cycle_density,
            "euler_characteristic": self.betti.euler_characteristic,
        }


class ValidationResult(NamedTuple):
    """Resultado de validación de DataFrames."""
    is_valid: bool
    message: str
    diagnostics: Dict[str, Any]
    
    @classmethod
    def success(cls, diagnostics: Optional[Dict[str, Any]] = None) -> 'ValidationResult':
        return cls(True, "Validación exitosa", diagnostics or {})
    
    @classmethod
    def failure(cls, message: str, diagnostics: Optional[Dict[str, Any]] = None) -> 'ValidationResult':
        return cls(False, message, diagnostics or {})


@dataclass
class VetoRecord:
    """Registro de veto emitido por el Risk Challenger."""
    veto_type: str
    severity: VetoSeverity
    stability_at_veto: float
    financial_class: RiskClassification
    original_integrity: float
    penalty_applied: float
    reason: str
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.veto_type,
            "severity": self.severity.value,
            "stability_at_veto": self.stability_at_veto,
            "financial_class_at_veto": self.financial_class.value,
            "original_integrity": self.original_integrity,
            "penalty_applied": self.penalty_applied,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


@dataclass
class LateralExceptionRecord:
    """Registro de excepción por pensamiento lateral."""
    exception_type: str
    pivot_type: PivotType
    penalty_relief: float
    reason: str
    approved_by_mic: bool
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "exception_type": self.exception_type,
            "pivot_type": self.pivot_type.value,
            "penalty_relief": self.penalty_relief,
            "reason": self.reason,
            "approved_by_mic": self.approved_by_mic,
            "timestamp": self.timestamp,
        }


# =============================================================================
# PROTOCOLOS
# =============================================================================

@runtime_checkable
class GraphProtocol(Protocol):
    """Protocolo para objetos de grafo (NetworkX-compatible)."""
    def number_of_nodes(self) -> int: ...
    def number_of_edges(self) -> int: ...
    def to_undirected(self) -> Any: ...


@runtime_checkable
class TopologicalAnalyzerProtocol(Protocol):
    """Protocolo para analizadores topológicos."""
    def calculate_betti_numbers(self, graph: Any) -> Any: ...
    def calculate_pyramid_stability(self, graph: Any) -> float: ...
    def generate_executive_report(self, graph: Any, financial_metrics: Dict) -> Any: ...
    def analyze_thermal_flow(self, graph: Any) -> Dict[str, Any]: ...


@runtime_checkable
class GraphBuilderProtocol(Protocol):
    """Protocolo para constructores de grafos."""
    def build(self, df_budget: pd.DataFrame, df_detail: pd.DataFrame) -> Any: ...


# =============================================================================
# VALIDADOR DE DATAFRAMES
# =============================================================================

class DataFrameValidator:
    """
    Validador estructural y semántico de DataFrames de entrada.
    
    Implementa verificación en cuatro niveles:
    1. Existencia: DataFrames no nulos y no vacíos
    2. Esquema: Columnas requeridas con tipos correctos
    3. Consistencia Referencial: Integridad de claves foráneas
    4. Distribución: Detección de anomalías estadísticas
    """
    
    # Esquemas de validación
    BUDGET_SCHEMA: Dict[str, Dict[str, Any]] = {
        ColumnNames.CODIGO_APU: {"type": "categorical", "required": True},
        ColumnNames.DESCRIPCION_APU: {"type": "string", "required": True},
        ColumnNames.VALOR_TOTAL: {"type": "numeric", "required": False, "min": 0},
    }
    
    DETAIL_SCHEMA: Dict[str, Dict[str, Any]] = {
        ColumnNames.CODIGO_APU: {"type": "categorical", "required": True},
        ColumnNames.DESCRIPCION_INSUMO: {"type": "string", "required": True},
        ColumnNames.CANTIDAD_APU: {"type": "numeric", "required": True, "min": 0},
        ColumnNames.COSTO_INSUMO_EN_APU: {"type": "numeric", "required": True, "min": 0},
    }
    
    # Mapeos legacy
    LEGACY_MAPPINGS: Dict[str, str] = {
        "item": ColumnNames.CODIGO_APU,
        "descripcion": ColumnNames.DESCRIPCION_APU,
        "total": ColumnNames.VALOR_TOTAL,
        "codigo": ColumnNames.CODIGO_APU,
        "desc_insumo": ColumnNames.DESCRIPCION_INSUMO,
        "cantidad": ColumnNames.CANTIDAD_APU,
        "costo": ColumnNames.COSTO_INSUMO_EN_APU,
    }
    
    def __init__(self, outlier_threshold: float = 0.10):
        """
        Args:
            outlier_threshold: Proporción máxima de outliers aceptable
        """
        self._outlier_threshold = outlier_threshold
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def validate(
        self,
        df_presupuesto: Optional[pd.DataFrame],
        df_apus_detail: Optional[pd.DataFrame]
    ) -> ValidationResult:
        """
        Valida los DataFrames de entrada.
        
        Returns:
            ValidationResult con estado, mensaje y diagnósticos
        """
        diagnostics = self._init_diagnostics()
        
        # Nivel 1: Existencia
        existence_result = self._validate_existence(df_presupuesto, df_apus_detail, diagnostics)
        if not existence_result.is_valid:
            return existence_result
        
        # Nivel 2: Esquema
        schema_result = self._validate_schema(df_presupuesto, df_apus_detail, diagnostics)
        if not schema_result.is_valid:
            return schema_result
        
        # Nivel 3: Consistencia referencial
        self._validate_referential_integrity(df_presupuesto, df_apus_detail, diagnostics)
        
        # Nivel 4: Distribución
        self._validate_distribution(df_presupuesto, diagnostics)
        
        # Log warnings
        for warning in diagnostics.get("warnings", []):
            self._logger.warning(f"⚠️ Validación: {warning}")
        
        return ValidationResult.success(diagnostics)
    
    def _init_diagnostics(self) -> Dict[str, Any]:
        """Inicializa estructura de diagnósticos."""
        return {
            "validation_timestamp": pd.Timestamp.now().isoformat(),
            "warnings": [],
            "schema_compatibility": {},
            "column_check": {"presupuesto": "OK", "detalle": "OK"},
            "missing_columns": {"presupuesto": [], "detalle": []},
            "null_analysis": {"presupuesto": {"total_nulls": 0}, "detalle": {"total_nulls": 0}},
            "duplicate_analysis": {"duplicated_codes": []},
            "value_range_analysis": {"negative_monetary_values": 0},
            "distribution_analysis": {},
            "referential_integrity": {},
        }
    
    def _validate_existence(
        self,
        df_budget: Optional[pd.DataFrame],
        df_detail: Optional[pd.DataFrame],
        diagnostics: Dict[str, Any]
    ) -> ValidationResult:
        """Valida existencia y no-vacuidad de DataFrames."""
        if df_budget is None:
            return ValidationResult.failure("DataFrame 'df_presupuesto' es None", diagnostics)
        if df_detail is None:
            return ValidationResult.failure("DataFrame 'df_merged' es None", diagnostics)
        if df_budget.empty:
            return ValidationResult.failure("DataFrame 'df_presupuesto' está vacío", diagnostics)
        if df_detail.empty:
            return ValidationResult.failure("DataFrame 'df_merged' está vacío", diagnostics)
        
        diagnostics["row_counts"] = {
            "presupuesto": len(df_budget),
            "apus_detail": len(df_detail),
            "detalle": len(df_detail),
        }
        
        return ValidationResult.success(diagnostics)
    
    def _find_column(self, df: pd.DataFrame, target: str) -> Optional[str]:
        """Busca columna por nombre moderno o legacy."""
        if target in df.columns:
            return target
        for legacy, modern in self.LEGACY_MAPPINGS.items():
            if modern == target and legacy in df.columns:
                return legacy
        return None
    
    def _validate_schema(
        self,
        df_budget: pd.DataFrame,
        df_detail: pd.DataFrame,
        diagnostics: Dict[str, Any]
    ) -> ValidationResult:
        """Valida esquemas de ambos DataFrames."""
        all_errors: List[str] = []
        
        budget_errors = self._validate_single_schema(
            df_budget, self.BUDGET_SCHEMA, "Presupuesto", "presupuesto", diagnostics
        )
        detail_errors = self._validate_single_schema(
            df_detail, self.DETAIL_SCHEMA, "APUs Detail", "detalle", diagnostics
        )
        
        all_errors.extend(budget_errors)
        all_errors.extend(detail_errors)
        
        if all_errors:
            return ValidationResult.failure("; ".join(all_errors), diagnostics)
        
        return ValidationResult.success(diagnostics)
    
    def _validate_single_schema(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Dict[str, Any]],
        df_name: str,
        diag_key: str,
        diagnostics: Dict[str, Any]
    ) -> List[str]:
        """Valida un DataFrame contra su esquema."""
        errors: List[str] = []
        
        # Análisis de nulos
        null_count = df.isnull().sum().sum()
        diagnostics["null_analysis"][diag_key]["total_nulls"] = int(null_count)
        
        for col_name, spec in schema.items():
            actual_col = self._find_column(df, col_name)
            
            if actual_col is None:
                if spec["required"]:
                    errors.append(f"{df_name}: Columna requerida '{col_name}' no encontrada")
                    diagnostics["missing_columns"][diag_key].append(col_name)
                    diagnostics["column_check"][diag_key] = "FAIL"
                continue
            
            # Registrar mapeo
            if actual_col != col_name:
                diagnostics["schema_compatibility"][actual_col] = col_name
            
            # Validar tipo
            if spec["type"] == "numeric":
                if not pd.api.types.is_numeric_dtype(df[actual_col]):
                    errors.append(
                        f"{df_name}: Columna '{actual_col}' debe ser numérica, es {df[actual_col].dtype}"
                    )
                elif "min" in spec:
                    invalid_count = (df[actual_col] < spec["min"]).sum()
                    if invalid_count > 0:
                        errors.append(
                            f"{df_name}: '{actual_col}' tiene {invalid_count} valores < {spec['min']}"
                        )
                        if spec["min"] == 0:
                            diagnostics["value_range_analysis"]["negative_monetary_values"] += int(invalid_count)
        
        return errors
    
    def _validate_referential_integrity(
        self,
        df_budget: pd.DataFrame,
        df_detail: pd.DataFrame,
        diagnostics: Dict[str, Any]
    ) -> None:
        """Valida integridad referencial entre DataFrames."""
        budget_apu_col = self._find_column(df_budget, ColumnNames.CODIGO_APU)
        detail_apu_col = self._find_column(df_detail, ColumnNames.CODIGO_APU)
        
        if budget_apu_col:
            duplicates = df_budget[budget_apu_col].duplicated()
            if duplicates.any():
                diagnostics["duplicate_analysis"]["duplicated_codes"] = (
                    df_budget.loc[duplicates, budget_apu_col].unique().tolist()
                )
        
        if budget_apu_col and detail_apu_col:
            budget_codes = set(df_budget[budget_apu_col].dropna().unique())
            detail_codes = set(df_detail[detail_apu_col].dropna().unique())
            
            orphan_details = detail_codes - budget_codes
            missing_details = budget_codes - detail_codes
            
            if orphan_details:
                diagnostics["warnings"].append(
                    f"APUs en detalle sin referencia en presupuesto: {len(orphan_details)}"
                )
            if missing_details:
                diagnostics["warnings"].append(
                    f"APUs en presupuesto sin detalle: {len(missing_details)}"
                )
            
            diagnostics["referential_integrity"] = {
                "budget_codes": len(budget_codes),
                "detail_codes": len(detail_codes),
                "orphan_codes": list(orphan_details),
                "coverage_ratio": len(budget_codes & detail_codes) / max(len(budget_codes), 1),
            }
    
    def _validate_distribution(
        self,
        df_budget: pd.DataFrame,
        diagnostics: Dict[str, Any]
    ) -> None:
        """Valida distribución estadística de valores."""
        valor_col = self._find_column(df_budget, ColumnNames.VALOR_TOTAL)
        
        if valor_col is None or len(df_budget) < 10:
            return
        
        values = df_budget[valor_col].dropna()
        if len(values) == 0:
            return
        
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (values < lower_bound) | (values > upper_bound)
        outliers = values[outlier_mask]
        outlier_ratio = len(outliers) / len(values)
        
        diagnostics["distribution_analysis"] = {
            "total_values": len(values),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "outlier_count": len(outliers),
            "outlier_indices": values.index[outlier_mask].tolist(),
            "outlier_ratio": float(outlier_ratio),
        }
        
        if outlier_ratio > self._outlier_threshold:
            diagnostics["warnings"].append(
                f"Alta proporción de outliers: {outlier_ratio:.1%} ({len(outliers)} valores)"
            )


# =============================================================================
# CONSTRUCTOR DE MODELO TOPOLÓGICO
# =============================================================================

class TopologyBuilder:
    """
    Constructor del modelo topológico del presupuesto.
    
    Transforma DataFrames de presupuesto en un complejo simplicial
    (representado como grafo dirigido) y calcula sus invariantes.
    """
    
    def __init__(
        self,
        graph_builder: Optional[GraphBuilderProtocol] = None,
        analyzer: Optional[TopologicalAnalyzerProtocol] = None,
        telemetry: Optional[TelemetryContext] = None
    ):
        self._graph_builder = graph_builder
        self._analyzer = analyzer
        self._telemetry = telemetry or TelemetryContext()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def build(
        self,
        df_presupuesto: pd.DataFrame,
        df_apus_detail: pd.DataFrame
    ) -> Tuple[Any, TopologicalMetricsBundle]:
        """
        Construye modelo topológico y calcula métricas.
        
        Returns:
            Tupla (grafo, métricas)
        
        Raises:
            TopologicalAnomalyError: Si la estructura viola restricciones
            RuntimeError: Si la construcción falla
        """
        self._logger.info("🏗️ Construyendo topología del presupuesto...")
        
        if self._graph_builder is None:
            raise RuntimeError("GraphBuilder no configurado")
        
        try:
            # Construcción del grafo
            graph = self._graph_builder.build(df_presupuesto, df_apus_detail)
            
            if graph is None:
                raise RuntimeError("El constructor de grafos retornó None")
            
            n_nodes = graph.number_of_nodes()
            n_edges = graph.number_of_edges()
            
            if n_nodes == 0:
                raise TopologicalAnomalyError(
                    "El grafo construido no tiene vértices",
                    details={"df_presupuesto_rows": len(df_presupuesto)}
                )
            
            self._logger.debug(f"Grafo construido: |V|={n_nodes}, |E|={n_edges}")
            
            # Análisis de conectividad
            if _HAS_NETWORKX:
                undirected = graph.to_undirected()
                is_connected = nx.is_connected(undirected)
                n_components = nx.number_connected_components(undirected)
                
                if not is_connected:
                    self._logger.warning(
                        f"⚠️ Grafo no conexo: {n_components} componentes"
                    )
            else:
                is_connected = True
                n_components = 1
            
            # Cálculo de invariantes
            if self._analyzer is None:
                raise RuntimeError("TopologicalAnalyzer no configurado")
            
            betti_raw = self._analyzer.calculate_betti_numbers(graph)
            
            # Normalizar a BettiNumbers
            if hasattr(betti_raw, '__dataclass_fields__'):
                betti_dict = asdict(betti_raw)
            elif isinstance(betti_raw, dict):
                betti_dict = betti_raw
            else:
                betti_dict = {"beta_0": n_components, "beta_1": 0, "beta_2": 0}
            
            betti = BettiNumbers.from_dict(betti_dict)
            pyramid_stability = self._analyzer.calculate_pyramid_stability(graph)
            
            # Verificar cotas de viabilidad
            self._verify_viability_bounds(betti, n_nodes)
            
            # Homología persistente (opcional)
            persistence = self._calculate_persistence(graph)
            
            self._logger.info(
                f"Métricas topológicas: β₀={betti.beta_0}, β₁={betti.beta_1}, "
                f"Ψ={pyramid_stability:.3f}, Conexo={is_connected}"
            )
            
            bundle = TopologicalMetricsBundle(
                betti=betti,
                pyramid_stability=pyramid_stability,
                n_nodes=n_nodes,
                n_edges=n_edges,
                is_connected=is_connected,
                persistence_diagram=persistence
            )
            
            return graph, bundle
        
        except TopologicalAnomalyError:
            raise
        except Exception as e:
            self._telemetry.record_error("topology_build", str(e))
            raise RuntimeError(f"Error construyendo modelo topológico: {e}") from e
    
    def _verify_viability_bounds(self, betti: BettiNumbers, n_nodes: int) -> None:
        """Verifica cotas de viabilidad topológica."""
        beta_1 = betti.beta_1
        
        # Cota empírica: β₁ ≤ √n
        cycle_bound = math.ceil(math.sqrt(n_nodes))
        
        if beta_1 > cycle_bound:
            self._logger.warning(
                f"⚠️ Alto número de ciclos: β₁={beta_1} > √n≈{cycle_bound}"
            )
        
        # Cota dura: β₁ > n es patología severa
        if beta_1 > n_nodes:
            raise TopologicalAnomalyError(
                f"Patología crítica: β₁={beta_1} > |V|={n_nodes}",
                details={"beta_1": beta_1, "n_nodes": n_nodes}
            )
    
    def _calculate_persistence(self, graph: Any) -> Optional[Tuple[Tuple[float, float], ...]]:
        """Calcula homología persistente si está disponible."""
        if self._analyzer is None:
            return None
        
        if not hasattr(self._analyzer, "calculate_persistence"):
            return None
        
        try:
            raw_persistence = self._analyzer.calculate_persistence(graph)
            if not raw_persistence:
                return None
            
            persistence: List[Tuple[float, float]] = []
            for item in raw_persistence:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    birth, death = float(item[0]), float(item[1])
                    if not math.isfinite(death):
                        death = birth + 10.0
                    if math.isfinite(birth):
                        persistence.append((birth, death))
            
            return tuple(persistence) if persistence else None
        
        except Exception as e:
            self._logger.debug(f"Homología persistente no disponible: {e}")
            return None


# =============================================================================
# ÁLGEBRA DE DECISIONES
# =============================================================================

class DecisionAlgebra:
    """
    Operaciones algebraicas para síntesis multicriterio.
    
    Implementa el marco matemático para combinar dimensiones
    topológicas, financieras y termodinámicas en un vector de decisión.
    """
    
    @staticmethod
    def normalize_to_sphere(
        vector: np.ndarray,
        epsilon: float = NumericalConstants.EPSILON
    ) -> np.ndarray:
        """
        Proyecta vector a la esfera unitaria S^(n-1).
        
        Si ‖v‖ < ε, retorna vector uniforme.
        """
        norm = np.linalg.norm(vector)
        if norm < epsilon:
            n = len(vector)
            return np.ones(n) / np.sqrt(n)
        return vector / norm
    
    @staticmethod
    def weighted_geometric_mean(
        factors: Sequence[float],
        weights: Optional[Sequence[float]] = None,
        epsilon: float = NumericalConstants.EPSILON
    ) -> float:
        """
        Media geométrica ponderada: (∏ᵢ xᵢ^wᵢ)^(1/Σwᵢ)
        
        Robusta ante factores no positivos.
        """
        if not factors:
            return 0.0
        
        n = len(factors)
        if weights is None:
            weights = [1.0 / n] * n
        
        if len(weights) != n:
            raise ValueError("Dimensiones de factores y pesos no coinciden")
        
        # Sanitizar
        clean_factors = [max(f, epsilon) for f in factors]
        clean_weights = [max(w, 0.0) for w in weights]
        
        weight_sum = sum(clean_weights)
        if weight_sum < epsilon:
            return 0.0
        
        # Si hay factor cero con peso positivo, resultado es 0
        for f, w in zip(factors, clean_weights):
            if f <= 0 and w > 0:
                return 0.0
        
        # Calcular en log-space
        log_sum = sum(w * math.log(f) for f, w in zip(clean_factors, clean_weights))
        return float(math.exp(log_sum / weight_sum))
    
    @staticmethod
    def convex_combination(
        vectors: Sequence[np.ndarray],
        weights: DecisionWeights
    ) -> np.ndarray:
        """
        Combinación convexa de vectores: d = Σᵢ αᵢ·vᵢ
        
        Args:
            vectors: Lista [topo_vec, finance_vec, thermo_vec]
            weights: Pesos para cada dimensión
        
        Returns:
            Vector de decisión combinado
        """
        if len(vectors) != 3:
            raise ValueError("Se esperan exactamente 3 vectores")
        
        normalized = weights.normalized
        alpha, beta, gamma = normalized.to_tuple()
        
        return (
            alpha * vectors[0] +
            beta * vectors[1] +
            gamma * vectors[2]
        )
    
    @classmethod
    def compute_quality_factors(
        cls,
        topo_bundle: TopologicalMetricsBundle,
        financial_metrics: Dict[str, Any],
        entropy: float,
        exergy: float,
        initial_investment: float = 1e6
    ) -> Tuple[float, float, float]:
        """
        Calcula factores de calidad para cada dimensión.
        
        Returns:
            Tupla (topo_quality, finance_quality, thermo_quality) ∈ [0, 1]³
        """
        # Calidad topológica
        topo_quality = math.sqrt(
            topo_bundle.structural_coherence * topo_bundle.pyramid_stability
        )
        
        # Calidad financiera
        npv = financial_metrics.get("npv", 0.0)
        if not isinstance(npv, (int, float)) or not math.isfinite(npv):
            npv = 0.0
        
        inv = max(abs(initial_investment), 1e6)
        npv_normalized = npv / inv
        finance_quality = (math.tanh(npv_normalized) + 1.0) / 2.0
        
        # Calidad termodinámica
        thermo_quality = ((1.0 - max(0, min(1, entropy))) + max(0, min(1, exergy))) / 2.0
        
        return (
            max(0.0, min(1.0, topo_quality)),
            max(0.0, min(1.0, finance_quality)),
            max(0.0, min(1.0, thermo_quality))
        )


# =============================================================================
# ESTRATEGIAS DE PIVOTE (PATTERN STRATEGY)
# =============================================================================

class PivotStrategy(ABC):
    """Estrategia base para evaluación de pivotes laterales."""
    
    @property
    @abstractmethod
    def pivot_type(self) -> PivotType:
        """Tipo de pivote que implementa esta estrategia."""
        pass
    
    @abstractmethod
    def evaluate(
        self,
        stability: float,
        financial_class: RiskClassification,
        thermal_metrics: Dict[str, Any],
        financial_metrics: Dict[str, Any],
        synergy_risk: Dict[str, Any],
        beta_1: int
    ) -> Tuple[bool, str]:
        """
        Evalúa si el pivote aplica.
        
        Returns:
            Tupla (aplica, razón)
        """
        pass


class MonopolioCoberturadoStrategy(PivotStrategy):
    """
    Pivote: Monopolio Coberturado (Topología vs Termodinámica).
    
    Condición: base estrecha (Ψ < 0.70) + sistema frío (T < 15°C) +
               inercia financiera alta (capacidad > umbral)
    """
    
    @property
    def pivot_type(self) -> PivotType:
        return PivotType.MONOPOLIO_COBERTURADO
    
    def __init__(
        self,
        stability_threshold: float = 0.70,
        temp_threshold: float = 15.0,
        inertia_threshold: float = 0.70
    ):
        self._stability_threshold = stability_threshold
        self._temp_threshold = temp_threshold
        self._inertia_threshold = inertia_threshold
    
    def evaluate(
        self,
        stability: float,
        financial_class: RiskClassification,
        thermal_metrics: Dict[str, Any],
        financial_metrics: Dict[str, Any],
        synergy_risk: Dict[str, Any],
        beta_1: int
    ) -> Tuple[bool, str]:
        system_temp = thermal_metrics.get("system_temperature", 25.0)
        heat_capacity = thermal_metrics.get("heat_capacity", 0.5)
        
        if (
            stability < self._stability_threshold and
            system_temp < self._temp_threshold and
            heat_capacity > self._inertia_threshold
        ):
            return True, (
                "Riesgo logístico neutralizado por alta inercia térmica financiera."
            )
        
        return False, (
            "Condiciones termodinámicas insuficientes para cobertura de monopolio."
        )


class OpcionEsperaStrategy(PivotStrategy):
    """
    Pivote: Opción de Espera (Opciones Reales).
    
    Condición: riesgo alto + valor de opción de espera > VPN × k
    """
    
    @property
    def pivot_type(self) -> PivotType:
        return PivotType.OPCION_ESPERA
    
    def __init__(self, npv_multiplier: float = 1.5):
        self._npv_multiplier = npv_multiplier
    
    def evaluate(
        self,
        stability: float,
        financial_class: RiskClassification,
        thermal_metrics: Dict[str, Any],
        financial_metrics: Dict[str, Any],
        synergy_risk: Dict[str, Any],
        beta_1: int
    ) -> Tuple[bool, str]:
        if financial_class != RiskClassification.HIGH:
            return False, "El riesgo financiero no es HIGH"
        
        real_options = financial_metrics.get("real_options", {})
        wait_value = float(real_options.get("wait_option_value", 0.0))
        npv = float(financial_metrics.get("npv", 0.0))
        
        threshold = max(npv, 0.0) * self._npv_multiplier
        
        if wait_value > threshold:
            return True, (
                f"Valor de opción de espera ({wait_value:.4f}) > "
                f"umbral VPN × {self._npv_multiplier} = {threshold:.4f}"
            )
        
        return False, "El valor de la opción de retraso no justifica la inactividad."


class CuarentenaTopologicaStrategy(PivotStrategy):
    """
    Pivote: Cuarentena Topológica (Aislamiento de Ciclos).
    
    Condición: ciclos presentes (β₁ > 0) SIN sinergia multiplicativa
    """
    
    @property
    def pivot_type(self) -> PivotType:
        return PivotType.CUARENTENA_TOPOLOGICA
    
    def evaluate(
        self,
        stability: float,
        financial_class: RiskClassification,
        thermal_metrics: Dict[str, Any],
        financial_metrics: Dict[str, Any],
        synergy_risk: Dict[str, Any],
        beta_1: int
    ) -> Tuple[bool, str]:
        has_synergy = bool(synergy_risk.get("synergy_detected", False))
        
        if beta_1 > 0 and not has_synergy:
            return True, (
                "Ciclos detectados pero confinados. Se aprueba ejecución "
                "exceptuando el subgrafo aislado."
            )
        
        if has_synergy:
            return False, (
                "Los ciclos topológicos presentan sinergia multiplicativa. "
                "Cuarentena imposible."
            )
        
        return False, "No hay ciclos que requieran cuarentena."


# =============================================================================
# RISK CHALLENGER
# =============================================================================

class RiskChallenger:
    """
    Motor de Auditoría Adversarial con Pensamiento Lateral.
    
    Formula "Intenciones" y las proyecta sobre la MIC.
    Si la MIC aprueba (validando clausura transitiva), emite excepción estratégica.
    
    Atributos:
        thresholds: Umbrales configurables para vetos
        mic: Matriz de Interacción Central (opcional)
        strategies: Estrategias de pivote registradas
    """
    
    def __init__(
        self,
        thresholds: Optional[RiskChallengerThresholds] = None,
        mic: Optional[Any] = None,
        strategies: Optional[Sequence[PivotStrategy]] = None
    ):
        """
        Args:
            thresholds: Umbrales para vetos y alertas
            mic: MICRegistry para proyección de pivotes
            strategies: Estrategias de pivote personalizadas
        """
        self._thresholds = thresholds or RiskChallengerThresholds()
        self._mic = mic
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Registrar estrategias
        self._strategies: Dict[PivotType, PivotStrategy] = {}
        
        if strategies:
            for strategy in strategies:
                self._strategies[strategy.pivot_type] = strategy
        else:
            # Estrategias por defecto
            self._strategies = {
                PivotType.MONOPOLIO_COBERTURADO: MonopolioCoberturadoStrategy(
                    stability_threshold=self._thresholds.critical_stability
                ),
                PivotType.OPCION_ESPERA: OpcionEsperaStrategy(),
                PivotType.CUARENTENA_TOPOLOGICA: CuarentenaTopologicaStrategy(),
            }
    
    def challenge_verdict(
        self,
        report: 'ConstructionRiskReport',
        session_context: Optional[Dict[str, Any]] = None
    ) -> 'ConstructionRiskReport':
        """
        Ejecuta auditoría adversarial proyectando vectores a la MIC.
        
        Args:
            report: Reporte base a auditar
            session_context: Contexto de sesión con validated_strata
        
        Returns:
            Reporte modificado (potencialmente con vetos o excepciones)
        """
        self._logger.info("⚖️ Risk Challenger: Iniciando auditoría...")
        
        details = report.details or {}
        session_context = session_context or {}
        
        # Extraer métricas
        stability, coherence, beta_1, n_nodes = self._extract_metrics(details)
        financial_class = self._classify_risk(report.financial_risk_level)
        
        thermal = details.get("thermal_metrics", {})
        financial = details.get("financial_metrics", {})
        synergy = details.get("synergy_risk", {})
        
        current_report = report
        
        # Contexto MIC
        mic_context = self._build_mic_context(session_context)
        
        # Construir payload base
        base_payload = {
            "report_state": {
                "stability": stability,
                "beta_1": beta_1,
                "financial_class": financial_class.value
            },
            "thermal_metrics": thermal,
            "financial_metrics": financial,
            "synergy_risk": synergy
        }
        
        # ═══ REGLA 1: Estabilidad Crítica ═══
        if stability is not None and stability < self._thresholds.critical_stability:
            current_report = self._apply_pivot_or_veto(
                report=current_report,
                pivot_type=PivotType.MONOPOLIO_COBERTURADO,
                stability=stability,
                financial_class=financial_class,
                base_payload=base_payload,
                mic_context=mic_context,
                thermal=thermal,
                financial=financial,
                synergy=synergy,
                beta_1=beta_1 or 0
            )
        
        # ═══ REGLA 2: Estabilidad Subóptima ═══
        elif stability is not None and stability < self._thresholds.warning_stability:
            if financial_class in (RiskClassification.SAFE, RiskClassification.MODERATE, RiskClassification.HIGH):
                current_report = self._emit_veto(
                    report=current_report,
                    veto_type="ALERTA_STRUCTURAL_WARNING",
                    severity=VetoSeverity.SEVERO,
                    stability=stability,
                    financial_class=financial_class,
                    penalty=self._thresholds.integrity_penalty_warn,
                    reason=(
                        f"Estabilidad piramidal Ψ={stability:.3f} es subóptima "
                        f"(umbral: {self._thresholds.warning_stability:.2f})"
                    )
                )
        
        # ═══ REGLA 3: Opción de Espera (Riesgo Alto) ═══
        if financial_class == RiskClassification.HIGH:
            current_report = self._try_pivot(
                report=current_report,
                pivot_type=PivotType.OPCION_ESPERA,
                stability=stability or 1.0,
                financial_class=financial_class,
                base_payload=base_payload,
                mic_context=mic_context,
                thermal=thermal,
                financial=financial,
                synergy=synergy,
                beta_1=beta_1 or 0
            )
        
        # ═══ REGLA 4: Cuarentena Topológica ═══
        if beta_1 is not None and beta_1 > 0:
            current_report = self._apply_cycle_handling(
                report=current_report,
                stability=stability or 1.0,
                financial_class=financial_class,
                base_payload=base_payload,
                mic_context=mic_context,
                thermal=thermal,
                financial=financial,
                synergy=synergy,
                beta_1=beta_1,
                n_nodes=n_nodes or 1
            )
        
        return current_report
    
    def _extract_metrics(
        self,
        details: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
        """Extrae métricas de estabilidad del reporte."""
        stability = details.get("pyramid_stability")
        coherence = details.get("structural_coherence")
        beta_1: Optional[int] = None
        n_nodes: Optional[int] = None
        
        topo_inv = details.get("topological_invariants", {})
        if stability is None:
            stability = topo_inv.get("pyramid_stability")
        if coherence is None:
            coherence = topo_inv.get("structural_coherence")
        
        betti = topo_inv.get("betti_numbers", {})
        beta_1 = betti.get("beta_1")
        
        if "graph_order" in details:
            n_nodes = details["graph_order"]
        elif "n_nodes" in topo_inv:
            n_nodes = topo_inv["n_nodes"]
        
        return stability, coherence, beta_1, n_nodes
    
    def _classify_risk(self, risk_level: Any) -> RiskClassification:
        """Normaliza nivel de riesgo."""
        return RiskClassification.from_string(str(risk_level))
    
    def _build_mic_context(self, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """Construye contexto para proyección MIC."""
        validated = session_context.get("validated_strata", {Stratum.PHYSICS, Stratum.TACTICS})
        
        return {
            "validated_strata": validated,
            "telemetry_context": session_context.get("telemetry_context")
        }
    
    def _try_pivot(
        self,
        report: 'ConstructionRiskReport',
        pivot_type: PivotType,
        stability: float,
        financial_class: RiskClassification,
        base_payload: Dict[str, Any],
        mic_context: Dict[str, Any],
        thermal: Dict[str, Any],
        financial: Dict[str, Any],
        synergy: Dict[str, Any],
        beta_1: int
    ) -> 'ConstructionRiskReport':
        """Intenta aplicar un pivote vía MIC."""
        strategy = self._strategies.get(pivot_type)
        if strategy is None:
            return report
        
        # Evaluar localmente primero
        applies, reason = strategy.evaluate(
            stability=stability,
            financial_class=financial_class,
            thermal_metrics=thermal,
            financial_metrics=financial,
            synergy_risk=synergy,
            beta_1=beta_1
        )
        
        if not applies:
            return report
        
        # Proyectar a MIC
        payload = {**base_payload, "pivot_type": pivot_type.value}
        projection = self._project_to_mic("lateral_thinking_pivot", payload, mic_context)
        
        if projection.get("success"):
            self._logger.info(f"🧠 MIC Aprobó: {pivot_type.value}")
            return self._emit_lateral_exception(
                report=report,
                pivot_type=pivot_type,
                exception_type=f"EXCEPCIÓN_{pivot_type.value}",
                penalty_relief=projection.get("payload", {}).get("penalty_relief", 0.0),
                reason=projection.get("payload", {}).get("reasoning", reason)
            )
        
        return report
    
    def _apply_pivot_or_veto(
        self,
        report: 'ConstructionRiskReport',
        pivot_type: PivotType,
        stability: float,
        financial_class: RiskClassification,
        base_payload: Dict[str, Any],
        mic_context: Dict[str, Any],
        thermal: Dict[str, Any],
        financial: Dict[str, Any],
        synergy: Dict[str, Any],
        beta_1: int
    ) -> 'ConstructionRiskReport':
        """Intenta pivote; si falla, emite veto."""
        modified = self._try_pivot(
            report=report,
            pivot_type=pivot_type,
            stability=stability,
            financial_class=financial_class,
            base_payload=base_payload,
            mic_context=mic_context,
            thermal=thermal,
            financial=financial,
            synergy=synergy,
            beta_1=beta_1
        )
        
        # Si el reporte no cambió y hay riesgo financiero, emitir veto
        if modified is report and financial_class in (
            RiskClassification.SAFE, RiskClassification.MODERATE, RiskClassification.HIGH
        ):
            return self._emit_veto(
                report=report,
                veto_type="VETO_CRITICAL_INSTABILITY",
                severity=VetoSeverity.CRITICO,
                stability=stability,
                financial_class=financial_class,
                penalty=self._thresholds.integrity_penalty_veto,
                reason="La Cimentación Logística es angosta y no hay inercia que la cubra."
            )
        
        return modified
    
    def _apply_cycle_handling(
        self,
        report: 'ConstructionRiskReport',
        stability: float,
        financial_class: RiskClassification,
        base_payload: Dict[str, Any],
        mic_context: Dict[str, Any],
        thermal: Dict[str, Any],
        financial: Dict[str, Any],
        synergy: Dict[str, Any],
        beta_1: int,
        n_nodes: int
    ) -> 'ConstructionRiskReport':
        """Maneja ciclos topológicos."""
        # Intentar cuarentena
        modified = self._try_pivot(
            report=report,
            pivot_type=PivotType.CUARENTENA_TOPOLOGICA,
            stability=stability,
            financial_class=financial_class,
            base_payload=base_payload,
            mic_context=mic_context,
            thermal=thermal,
            financial=financial,
            synergy=synergy,
            beta_1=beta_1
        )
        
        if modified is not report:
            return modified
        
        # Aplicar penalización estándar por ciclos
        cycle_density = beta_1 / max(n_nodes, 1)
        
        if cycle_density > self._thresholds.cycle_density_limit:
            self._logger.warning(
                f"⚠️ Densidad de ciclos β₁/n = {cycle_density:.3f} > "
                f"{self._thresholds.cycle_density_limit:.2f}"
            )
            
            new_details = {**(report.details or {})}
            new_details["challenger_cycle_warning"] = {
                "beta_1": beta_1,
                "n_nodes": n_nodes,
                "cycle_density": cycle_density,
                "threshold": self._thresholds.cycle_density_limit,
            }
            new_details.setdefault("penalties_applied", []).append("cycle_penalty")
            
            return self._create_modified_report(
                report,
                integrity_score=report.integrity_score * 0.95,
                details=new_details
            )
        
        return report
    
    def _project_to_mic(
        self,
        service_name: str,
        payload: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Proyecta intención a la MIC."""
        if self._mic is None:
            return {"success": False, "error": "MIC no configurada"}
        
        try:
            return self._mic.project_intent(service_name, payload, context)
        except Exception as e:
            self._logger.warning(f"Proyección MIC falló: {e}")
            return {"success": False, "error": str(e)}
    
    def _emit_veto(
        self,
        report: 'ConstructionRiskReport',
        veto_type: str,
        severity: VetoSeverity,
        stability: float,
        financial_class: RiskClassification,
        penalty: float,
        reason: str
    ) -> 'ConstructionRiskReport':
        """Emite veto estructurado."""
        self._logger.warning(f"🚨 Risk Challenger: {veto_type} - {reason}")
        
        original_integrity = report.integrity_score
        new_integrity = max(0.0, original_integrity * (1.0 - penalty))
        
        # Acta de deliberación
        debate_log = self._generate_veto_narrative(
            veto_type=veto_type,
            severity=severity,
            stability=stability,
            financial_class=financial_class,
            original_integrity=original_integrity,
            new_integrity=new_integrity,
            reason=reason
        )
        
        new_details = {**(report.details or {})}
        new_details["challenger_verdict"] = VetoRecord(
            veto_type=veto_type,
            severity=severity,
            stability_at_veto=stability,
            financial_class=financial_class,
            original_integrity=original_integrity,
            penalty_applied=penalty,
            reason=reason
        ).to_dict()
        
        if severity == VetoSeverity.CRITICO:
            new_details["challenger_applied"] = True
        else:
            new_details["challenger_warning"] = True
        
        new_details.setdefault("penalties_applied", []).append(veto_type)
        
        return self._create_modified_report(
            report,
            integrity_score=new_integrity,
            details=new_details,
            strategic_narrative=f"{debate_log}\n\n{report.strategic_narrative}",
            financial_risk_level=f"RIESGO ESTRUCTURAL ({severity.value})"
        )
    
    def _emit_lateral_exception(
        self,
        report: 'ConstructionRiskReport',
        pivot_type: PivotType,
        exception_type: str,
        penalty_relief: float,
        reason: str
    ) -> 'ConstructionRiskReport':
        """Emite excepción por pensamiento lateral."""
        original_integrity = report.integrity_score
        new_integrity = min(100.0, original_integrity * (1.0 + penalty_relief))
        
        debate_log = self._generate_exception_narrative(
            exception_type=exception_type,
            reason=reason
        )
        
        new_details = {**(report.details or {})}
        new_details["lateral_thinking_applied"] = exception_type
        new_details["challenger_applied"] = True
        new_details["lateral_exception"] = LateralExceptionRecord(
            exception_type=exception_type,
            pivot_type=pivot_type,
            penalty_relief=penalty_relief,
            reason=reason,
            approved_by_mic=True
        ).to_dict()
        
        return self._create_modified_report(
            report,
            integrity_score=new_integrity,
            details=new_details,
            strategic_narrative=f"{debate_log}\n\n{report.strategic_narrative}",
            financial_risk_level="ESTRATEGIA MODIFICADA (PENSAMIENTO LATERAL)"
        )
    
    def _generate_veto_narrative(
        self,
        veto_type: str,
        severity: VetoSeverity,
        stability: float,
        financial_class: RiskClassification,
        original_integrity: float,
        new_integrity: float,
        reason: str
    ) -> str:
        """Genera narrativa de veto."""
        return (
            "━" * 60 + "\n"
            "🏛️ **ACTA DE DELIBERACIÓN DEL CONSEJO DE RIESGO**\n"
            "━" * 60 + "\n\n"
            f"📋 **Tipo de Veto:** {veto_type}\n"
            f"⚠️ **Severidad:** {severity.value}\n\n"
            "**Posiciones de los Agentes:**\n\n"
            f"1. 🤵 **Gestor Financiero:** «El proyecto es financieramente {financial_class.value}. "
            "Los indicadores de rentabilidad son favorables.»\n\n"
            f"2. 👷 **Ingeniero Estructural:** «OBJECIÓN. {reason}»\n\n"
            f"3. ⚖️ **Fiscal de Riesgos:** «Se detecta contradicción lógica entre "
            f"viabilidad financiera (Φ={financial_class.value}) y estabilidad estructural "
            f"(Ψ={stability:.3f}).»\n\n"
            "**VEREDICTO FINAL:**\n"
            f"Se emite **{veto_type}**. La integridad del proyecto se degrada de "
            f"{original_integrity:.1f} a {new_integrity:.1f} puntos.\n\n"
            "━" * 60
        )
    
    def _generate_exception_narrative(
        self,
        exception_type: str,
        reason: str
    ) -> str:
        """Genera narrativa de excepción lateral."""
        return (
            "━" * 60 + "\n"
            "🏛️ **ACTA DEL CONSEJO: EXCEPCIÓN POR PENSAMIENTO LATERAL**\n"
            "━" * 60 + "\n\n"
            f"⚖️ **Resolución de la MIC:** {exception_type}\n\n"
            f"**Fiscal de Riesgos:** «{reason} Se levanta el veto estructural "
            "o se modifica la estrategia base.»\n\n"
            "━" * 60
        )
    
    def _create_modified_report(
        self,
        original: 'ConstructionRiskReport',
        integrity_score: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
        strategic_narrative: Optional[str] = None,
        financial_risk_level: Optional[str] = None
    ) -> 'ConstructionRiskReport':
        """Crea copia modificada del reporte."""
        return ConstructionRiskReport(
            integrity_score=integrity_score if integrity_score is not None else original.integrity_score,
            waste_alerts=original.waste_alerts,
            circular_risks=original.circular_risks,
            complexity_level=original.complexity_level,
            financial_risk_level=financial_risk_level if financial_risk_level else original.financial_risk_level,
            details=details if details is not None else original.details,
            strategic_narrative=strategic_narrative if strategic_narrative else original.strategic_narrative
        )


# =============================================================================
# COMPOSITOR DE REPORTES
# =============================================================================

class ReportComposer:
    """
    Compositor de reportes ejecutivos mediante álgebra de decisiones.
    
    Integra dimensiones topológicas, financieras y termodinámicas
    en un reporte estratégico unificado.
    """
    
    def __init__(
        self,
        analyzer: Optional[TopologicalAnalyzerProtocol] = None,
        translator: Optional[Any] = None,
        weights: Optional[DecisionWeights] = None,
        telemetry: Optional[TelemetryContext] = None
    ):
        self._analyzer = analyzer
        self._translator = translator
        self._weights = weights or DecisionWeights()
        self._telemetry = telemetry or TelemetryContext()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def compose(
        self,
        graph: Any,
        topo_bundle: TopologicalMetricsBundle,
        financial_metrics: Dict[str, Any],
        thermal_metrics: Dict[str, Any],
        entropy: float = 0.5,
        exergy: float = 0.6
    ) -> 'ConstructionRiskReport':
        """
        Genera reporte ejecutivo con álgebra multicriterio.
        
        Args:
            graph: Grafo del presupuesto
            topo_bundle: Bundle de métricas topológicas
            financial_metrics: Métricas financieras
            thermal_metrics: Métricas termodinámicas
            entropy: Entropía del sistema [0, 1]
            exergy: Exergía [0, 1]
        
        Returns:
            ConstructionRiskReport con análisis completo
        """
        self._logger.info("🧠 Componiendo reporte con álgebra multicriterio...")
        
        # Reporte base del analizador topológico
        if self._analyzer is None:
            raise SynthesisError("TopologicalAnalyzer no configurado")
        
        base_report = self._analyzer.generate_executive_report(graph, financial_metrics)
        
        if base_report is None:
            raise SynthesisError("El analizador generó reporte nulo")
        
        # Calcular vectores de decisión
        weights = self._weights.normalized
        
        topo_quality, finance_quality, thermo_quality = DecisionAlgebra.compute_quality_factors(
            topo_bundle=topo_bundle,
            financial_metrics=financial_metrics,
            entropy=entropy,
            exergy=exergy,
            initial_investment=financial_metrics.get("initial_investment", 1e6)
        )
        
        # Score integrado (media geométrica ponderada)
        integrated_score = DecisionAlgebra.weighted_geometric_mean(
            factors=[topo_quality, finance_quality, thermo_quality],
            weights=list(weights.to_tuple())
        )
        
        integrated_score_100 = float(np.clip(integrated_score * 100.0, 0.0, 100.0))
        
        # Construir resumen de álgebra de decisiones
        decision_summary = {
            "weights": {"alpha": weights.topology, "beta": weights.finance, "gamma": weights.thermodynamics},
            "quality_factors": {
                "topology": float(topo_quality),
                "finance": float(finance_quality),
                "thermodynamics": float(thermo_quality),
            },
            "integrated_score": integrated_score,
        }
        
        # Generar narrativa
        narrative = self._generate_narrative(
            topo_bundle=topo_bundle,
            financial_metrics=financial_metrics,
            thermal_metrics=thermal_metrics,
            decision_summary=decision_summary,
            base_narrative=getattr(base_report, 'strategic_narrative', '')
        )
        
        # Construir detalles enriquecidos
        enriched_details = {
            **(base_report.details or {}),
            "strategic_narrative": narrative,
            "financial_metrics": financial_metrics,
            "thermal_metrics": thermal_metrics,
            "thermodynamics": {
                "entropy": float(entropy),
                "exergy": float(exergy),
                "negentropy": float(1.0 - entropy),
            },
            "topological_invariants": topo_bundle.to_dict(),
            "decision_algebra": decision_summary,
        }
        
        return ConstructionRiskReport(
            integrity_score=integrated_score_100,
            waste_alerts=base_report.waste_alerts,
            circular_risks=base_report.circular_risks,
            complexity_level=base_report.complexity_level,
            financial_risk_level=base_report.financial_risk_level,
            details=enriched_details,
            strategic_narrative=narrative
        )
    
    def _generate_narrative(
        self,
        topo_bundle: TopologicalMetricsBundle,
        financial_metrics: Dict[str, Any],
        thermal_metrics: Dict[str, Any],
        decision_summary: Dict[str, Any],
        base_narrative: str
    ) -> str:
        """Genera narrativa estratégica."""
        if self._translator is None:
            # Fallback a narrativa básica
            quality = decision_summary.get("quality_factors", {})
            return (
                f"Reporte con score de integridad {decision_summary['integrated_score']*100:.1f}/100. "
                f"Coherencia topológica: {quality.get('topology', 0):.2%}. "
                f"Salud financiera: {quality.get('finance', 0):.2%}. "
                f"Calidad termodinámica: {quality.get('thermodynamics', 0):.2%}."
            )
        
        try:
            strategic_report = self._translator.compose_strategic_narrative(
                topological_metrics=topo_bundle.betti.to_dict(),
                financial_metrics=financial_metrics,
                stability=topo_bundle.pyramid_stability,
                thermal_metrics=thermal_metrics,
                decision_algebra=decision_summary,
            )
            return getattr(strategic_report, "raw_narrative", str(strategic_report))
        except Exception as e:
            self._logger.warning(f"⚠️ Generación de narrativa falló: {e}")
            return base_narrative


# =============================================================================
# BUSINESS AGENT - FACHADA PRINCIPAL
# =============================================================================

class BusinessAgent:
    """
    Orquesta la inteligencia de negocio para evaluar proyectos.
    
    Combina análisis topológico (complejo simplicial del presupuesto)
    con análisis financiero (VPN, TIR, VaR) para producir evaluación holística.
    
    Principio: "No hay Estrategia sin Física"
    
    Ejemplo de uso:
        ```python
        agent = BusinessAgent(config, mic, telemetry)
        
        report = agent.evaluate_project({
            "df_presupuesto": df_budget,
            "df_merged": df_detail,
            "initial_investment": 1_000_000,
            "validated_strata": {"PHYSICS", "TACTICS"}
        })
        
        if report and report.integrity_score > 70:
            print("Proyecto viable")
        ```
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        mic: Any,
        telemetry: Optional[TelemetryContext] = None,
        *,
        graph_builder: Optional[GraphBuilderProtocol] = None,
        topology_analyzer: Optional[TopologicalAnalyzerProtocol] = None,
        semantic_translator: Optional[Any] = None,
        validator: Optional[DataFrameValidator] = None
    ):
        """
        Inicializa el agente con inyección de dependencias.
        
        Args:
            config: Configuración global
            mic: Matriz de Interacción Central
            telemetry: Contexto de telemetría
            graph_builder: Constructor de grafos (inyectable para testing)
            topology_analyzer: Analizador topológico
            semantic_translator: Traductor semántico
            validator: Validador de DataFrames
        """
        self._validate_config(config)
        self._config = config
        self._mic = mic
        self._telemetry = telemetry or TelemetryContext()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Componentes (con inyección o defaults)
        self._validator = validator or DataFrameValidator()
        self._graph_builder = graph_builder or (BudgetGraphBuilder() if BudgetGraphBuilder else None)
        self._topology_analyzer = topology_analyzer or (
            BusinessTopologicalAnalyzer(self._telemetry) if BusinessTopologicalAnalyzer else None
        )
        self._translator = semantic_translator or (
            SemanticTranslator(mic=mic) if SemanticTranslator else None
        )
        
        # Builders compuestos
        self._topology_builder = TopologyBuilder(
            graph_builder=self._graph_builder,
            analyzer=self._topology_analyzer,
            telemetry=self._telemetry
        )
        
        # Risk Challenger
        challenger_thresholds = self._build_challenger_thresholds(config)
        self._risk_challenger = RiskChallenger(
            thresholds=challenger_thresholds,
            mic=mic
        )
        
        # Compositor de reportes
        decision_weights = self._build_decision_weights(config)
        self._report_composer = ReportComposer(
            analyzer=self._topology_analyzer,
            translator=self._translator,
            weights=decision_weights,
            telemetry=self._telemetry
        )
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Valida configuración."""
        if not isinstance(config, dict):
            raise ConfigurationError("La configuración debe ser un diccionario")
        
        financial_cfg = config.get("financial_config", {})
        
        for field in ["risk_free_rate", "discount_rate", "market_return"]:
            if field in financial_cfg:
                value = financial_cfg[field]
                if not isinstance(value, (int, float)) or value < 0:
                    raise ConfigurationError(
                        f"'{field}' debe ser número no negativo: {value}"
                    )
    
    def _build_challenger_thresholds(self, config: Dict[str, Any]) -> RiskChallengerThresholds:
        """Construye umbrales del challenger desde config."""
        challenger_cfg = config.get("risk_challenger_config", {})
        
        if not challenger_cfg:
            return RiskChallengerThresholds()
        
        return RiskChallengerThresholds(
            critical_stability=challenger_cfg.get("critical_stability", 0.70),
            warning_stability=challenger_cfg.get("warning_stability", 0.85),
            coherence_minimum=challenger_cfg.get("coherence_minimum", 0.60),
            cycle_density_limit=challenger_cfg.get("cycle_density_limit", 0.33),
            integrity_penalty_veto=challenger_cfg.get("integrity_penalty_veto", 0.30),
            integrity_penalty_warn=challenger_cfg.get("integrity_penalty_warn", 0.15),
        )
    
    def _build_decision_weights(self, config: Dict[str, Any]) -> DecisionWeights:
        """Construye pesos de decisión desde config."""
        weights_cfg = config.get("decision_weights", {})
        
        if not weights_cfg:
            return DecisionWeights()
        
        return DecisionWeights(
            topology=weights_cfg.get("topology", 0.40),
            finance=weights_cfg.get("finance", 0.40),
            thermodynamics=weights_cfg.get("thermodynamics", 0.20)
        )
    
    def evaluate_project(self, context: Dict[str, Any]) -> Optional['ConstructionRiskReport']:
        """
        Ejecuta evaluación completa del proyecto.
        
        Pipeline:
        1. Validación de datos
        2. Análisis topológico
        3. Análisis termodinámico
        4. Análisis financiero (con inyección causal)
        5. Síntesis y composición
        6. Auditoría adversarial
        
        Args:
            context: Contexto con DataFrames y parámetros
        
        Returns:
            ConstructionRiskReport o None si falla
        """
        self._logger.info("🤖 Iniciando evaluación de proyecto...")
        
        # ═══ Fase 0: Validación ═══
        df_presupuesto = context.get("df_final") or context.get("df_presupuesto")
        df_apus_detail = context.get("df_merged")
        
        validation = self._validator.validate(df_presupuesto, df_apus_detail)
        
        if not validation.is_valid:
            self._logger.warning(f"Validación fallida: {validation.message}")
            self._telemetry.record_error("validation", validation.message)
            
            # Reporte vacío estructurado para DataFrames vacíos
            if df_presupuesto is not None and df_presupuesto.empty:
                return ConstructionRiskReport(
                    integrity_score=0.0,
                    waste_alerts=[],
                    circular_risks=[],
                    complexity_level="Desconocida",
                    financial_risk_level="Desconocido",
                    details=validation.diagnostics,
                    strategic_narrative="Datos insuficientes para análisis."
                )
            return None
        
        # ═══ Fase 1: Topología ═══
        try:
            graph, topo_bundle = self._topology_builder.build(df_presupuesto, df_apus_detail)
            context["graph"] = graph
        except (TopologicalAnomalyError, RuntimeError) as e:
            self._logger.error(f"❌ Fase topológica: {e}")
            self._telemetry.record_error("topology", str(e))
            return None
        
        # ═══ Fase 2: Termodinámica ═══
        thermal_metrics, entropy, exergy = self._analyze_thermodynamics(graph, context)
        
        # ═══ Fase 3: Finanzas ═══
        try:
            financial_params = self._extract_financial_params(context)
            financial_metrics = self._perform_financial_analysis(
                params=financial_params,
                session_context=context,
                topo_bundle=topo_bundle,
                thermal_metrics=thermal_metrics
            )
        except (MICHierarchyViolationError, FinancialProjectionError) as e:
            self._logger.error(f"❌ Fase financiera: {e}")
            self._telemetry.record_error("financial", str(e))
            return None
        
        # ═══ Fase 4 & 5: Síntesis y Auditoría ═══
        try:
            report = self._report_composer.compose(
                graph=graph,
                topo_bundle=topo_bundle,
                financial_metrics=financial_metrics,
                thermal_metrics=thermal_metrics,
                entropy=entropy,
                exergy=exergy
            )
            
            # Auditoría adversarial
            audited_report = self._risk_challenger.challenge_verdict(report, session_context=context)
            
            # Verificación de integridad numérica
            if not np.isfinite(audited_report.integrity_score):
                self._logger.error(f"❌ Score no finito: {audited_report.integrity_score}")
                audited_report = self._create_fallback_report(audited_report)
            
            self._logger.info("✅ Evaluación completada con éxito")
            return audited_report
            
        except SynthesisError as e:
            self._logger.error(f"❌ Fase de síntesis: {e}")
            self._telemetry.record_error("synthesis", str(e))
            return None
    
    def _analyze_thermodynamics(
        self,
        graph: Any,
        context: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, float]:
        """Analiza métricas termodinámicas."""
        try:
            if self._topology_analyzer is not None:
                thermal = self._topology_analyzer.analyze_thermal_flow(graph)
            else:
                thermal = {"system_temperature": 0.0}
            
            entropy = context.get("system_entropy", 0.5)
            exergy = context.get("budget_exergy", 0.6)
            
            return thermal, entropy, exergy
            
        except Exception as e:
            self._logger.warning(f"⚠️ Fallo parcial en termodinámica: {e}")
            return {"system_temperature": 0.0}, 0.5, 0.5
    
    def _extract_financial_params(self, context: Dict[str, Any]) -> FinancialParameters:
        """Extrae parámetros financieros del contexto."""
        defaults = self._config.get("default_financial_params", {})
        
        initial_investment = context.get(
            "initial_investment",
            defaults.get("initial_investment", FinancialDefaults.INITIAL_INVESTMENT)
        )
        
        if "cash_flows" in context:
            cash_flows = tuple(context["cash_flows"])
        else:
            ratio = defaults.get("cash_flow_ratio", FinancialDefaults.CASH_FLOW_RATIO)
            periods = defaults.get("cash_flow_periods", FinancialDefaults.CASH_FLOW_PERIODS)
            cash_flows = tuple(initial_investment * ratio for _ in range(periods))
        
        cost_std_dev = context.get(
            "cost_std_dev",
            initial_investment * defaults.get("cost_std_dev_ratio", FinancialDefaults.COST_STD_DEV_RATIO)
        )
        
        volatility = context.get(
            "project_volatility",
            defaults.get("project_volatility", FinancialDefaults.PROJECT_VOLATILITY)
        )
        
        return FinancialParameters(
            initial_investment=initial_investment,
            cash_flows=cash_flows,
            cost_std_dev=cost_std_dev,
            project_volatility=volatility
        )
    
    def _perform_financial_analysis(
        self,
        params: FinancialParameters,
        session_context: Dict[str, Any],
        topo_bundle: TopologicalMetricsBundle,
        thermal_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ejecuta análisis financiero con inyección causal."""
        self._logger.info("🤖 Proyectando vector financiero...")
        
        # Validar jerarquía MIC
        validated_strata = session_context.get("validated_strata", set())
        if isinstance(validated_strata, list):
            validated_strata = set(validated_strata)
        
        required = {"PHYSICS", "TACTICS"}
        missing = required - validated_strata
        
        if missing:
            raise MICHierarchyViolationError(
                f"Estratos {missing} no validados",
                details={"required": list(required), "validated": list(validated_strata)}
            )
        
        # Construir payload con inyección causal
        payload = {
            "amount": params.initial_investment,
            "std_dev": params.cost_std_dev,
            "time": params.periods,
            "cash_flows": list(params.cash_flows),
            "topological_conditioning": {
                "structural_coherence": topo_bundle.structural_coherence,
                "beta_1_penalty": topo_bundle.betti.beta_1 * 0.1,
                "is_connected": topo_bundle.is_connected
            },
            "thermal_adjustment": {
                "system_temperature": thermal_metrics.get("system_temperature", 0.0),
                "volatility_multiplier": 1.0 + thermal_metrics.get("system_temperature", 0.0) * 0.5
            }
        }
        
        mic_context = {
            "validated_strata": validated_strata,
            "session_id": session_context.get("session_id", "unknown"),
            "causal_injection": True
        }
        
        # Proyección a MIC
        try:
            response = self._mic.project_intent("financial_analysis", payload, mic_context)
            
            if not response.get("success"):
                error = response.get("error", "Error MIC desconocido")
                raise FinancialProjectionError(f"Proyección fallida: {error}")
            
            results = copy.deepcopy(response.get("results", {}))
            
            # Enriquecer con factores estructurales
            if "npv" in results:
                structural_factor = topo_bundle.structural_coherence
                results["npv_adjusted"] = results["npv"] * structural_factor
                results["structural_discount"] = 1.0 - structural_factor
            
            self._logger.info(f"✅ Proyección financiera: VPN={results.get('npv', 'N/A')}")
            return results
            
        except MICHierarchyViolationError:
            raise
        except Exception as e:
            raise FinancialProjectionError(f"Fallo en proyección: {e}") from e
    
    def _create_fallback_report(self, original: 'ConstructionRiskReport') -> 'ConstructionRiskReport':
        """Crea reporte fallback para errores numéricos."""
        return ConstructionRiskReport(
            integrity_score=0.0,
            waste_alerts=original.waste_alerts,
            circular_risks=original.circular_risks,
            complexity_level=original.complexity_level,
            financial_risk_level="ERROR NUMÉRICO",
            details=original.details,
            strategic_narrative=original.strategic_narrative
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Propiedades de acceso (para tests y compatibilidad)
    # ─────────────────────────────────────────────────────────────────────────
    
    @property
    def config(self) -> Dict[str, Any]:
        return self._config
    
    @property
    def mic(self) -> Any:
        return self._mic
    
    @property
    def telemetry(self) -> TelemetryContext:
        return self._telemetry
    
    @property
    def risk_challenger(self) -> RiskChallenger:
        return self._risk_challenger
    
    @property
    def graph_builder(self) -> Optional[GraphBuilderProtocol]:
        return self._graph_builder
    
    @property
    def topological_analyzer(self) -> Optional[TopologicalAnalyzerProtocol]:
        return self._topology_analyzer
    
    @property
    def translator(self) -> Optional[Any]:
        return self._translator


# =============================================================================
# FUNCIONES DE CONVENIENCIA
# =============================================================================

def create_business_agent(
    config: Dict[str, Any],
    mic: Any,
    telemetry: Optional[TelemetryContext] = None
) -> BusinessAgent:
    """
    Factory function para crear BusinessAgent.
    
    Args:
        config: Configuración del agente
        mic: Matriz de Interacción Central
        telemetry: Contexto de telemetría
    
    Returns:
        Instancia configurada de BusinessAgent
    """
    return BusinessAgent(config=config, mic=mic, telemetry=telemetry)


def evaluate_project(
    context: Dict[str, Any],
    config: Dict[str, Any],
    mic: Any
) -> Optional['ConstructionRiskReport']:
    """
    Función de conveniencia para evaluar un proyecto.
    
    Crea un BusinessAgent temporal y ejecuta la evaluación.
    
    Args:
        context: Contexto con DataFrames y parámetros
        config: Configuración
        mic: MICRegistry
    
    Returns:
        ConstructionRiskReport o None
    """
    agent = create_business_agent(config, mic)
    return agent.evaluate_project(context)