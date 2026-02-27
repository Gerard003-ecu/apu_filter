"""
Módulo: Business Topological Analyzer (El Arquitecto Estructural)
=================================================================

Este componente actúa como el "Sabio de la Estructura" dentro del Consejo. Su función
es modelar el presupuesto no como una lista contable, sino como un **Complejo Simplicial
Abstracto**. Aplica teoremas matemáticos para diagnosticar riesgos que son invisibles
para el análisis financiero tradicional.

Fundamentos Matemáticos y Diagnósticos:
---------------------------------------

1. Homología Computacional (Los Invariantes βₙ):
   Calcula los Números de Betti para caracterizar la conectividad del negocio:
   - β₀ (Componentes Conexas): Detecta fragmentación. β₀ > 1 revela "Islas de Datos".
   - β₁ (Ciclos): Detecta "Socavones Lógicos". β₁ > 0 indica dependencias circulares.
   - β₂ (Cavidades): Para grafos (1-complejos simpliciales), β₂ = 0 siempre.
   
   Invariante de Euler-Poincaré: χ = β₀ - β₁ + β₂ = V - E + F

2. Física del Equilibrio (Índice Ψ):
   Implementa la métrica de **Estabilidad Piramidal**:
   
   Ψ = w_r·f_ratio + w_d·f_density + w_a·f_acyclic + w_c·f_connectivity
   
   Donde:
   - f_ratio: log₁₀(1 + insumos/APUs) / log₁₀(11) — Amplitud de base
   - f_density: Penalización por densidad extrema
   - f_acyclic: 1.0 si DAG, penalizado por SCCs
   - f_connectivity: 1/num_components

3. Análisis Espectral (El Valor de Fiedler):
   Analiza el espectro de la Matriz Laplaciana (L = D - A):
   - Valor de Fiedler (λ₂): Conectividad algebraica. λ₂ ≈ 0 implica fractura.
   - Gap Espectral: λ₂/λ_max — Capacidad de sincronización.
   - Riesgo de Resonancia: CV(eigenvalores) < 0.15 indica vulnerabilidad.

4. Termodinámica Estructural (Convección Inflacionaria):
   Simula volatilidad de precios como "Calor" que se difunde desde insumos (hojas)
   hacia el proyecto total (raíz). Estructuras mal conectadas atrapan calor.

5. Auditoría de Fusión (Mayer-Vietoris):
   Verifica la secuencia exacta de homología para detectar ciclos espurios
   introducidos por la integración de datos.
"""

from __future__ import annotations

import logging
import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, IntEnum, auto
from functools import lru_cache
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
    TypeVar,
    Union,
    runtime_checkable,
)

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.linalg

from app.constants import ColumnNames
from app.telemetry import TelemetryContext
from app.telemetry_schemas import TopologicalMetrics

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTES GLOBALES
# ============================================================================

# Tolerancias numéricas
EPSILON: Final[float] = 1e-9
EIGENVALUE_TOLERANCE: Final[float] = 1e-6

# Límites del Laplaciano normalizado
LAPLACIAN_MAX_EIGENVALUE: Final[float] = 2.0

# Nodo raíz por defecto
DEFAULT_ROOT_NODE: Final[str] = "PROYECTO_TOTAL"


# ============================================================================
# EXCEPCIONES DEL DOMINIO
# ============================================================================


class TopologyAnalyzerError(Exception):
    """Excepción base del analizador topológico."""
    pass


class GraphConstructionError(TopologyAnalyzerError):
    """Error durante la construcción del grafo."""
    pass


class TopologyInvariantError(TopologyAnalyzerError):
    """Violación de invariante topológico."""
    
    def __init__(
        self,
        message: str,
        beta_0: int,
        beta_1: int,
        beta_2: int,
        euler_observed: int,
    ):
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.euler_observed = euler_observed
        euler_expected = beta_0 - beta_1 + beta_2
        super().__init__(
            f"{message}. χ_observed={euler_observed}, χ_expected={euler_expected} "
            f"(β₀={beta_0}, β₁={beta_1}, β₂={beta_2})"
        )


class SpectralAnalysisError(TopologyAnalyzerError):
    """Error durante el análisis espectral."""
    pass


class InsufficientDataError(TopologyAnalyzerError):
    """Datos insuficientes para el análisis."""
    pass


# ============================================================================
# ENUMERACIONES
# ============================================================================


class NodeType(str, Enum):
    """Tipos de nodos en el grafo de presupuesto."""
    
    ROOT = "ROOT"
    CAPITULO = "CAPITULO"
    APU = "APU"
    INSUMO = "INSUMO"
    
    @property
    def level(self) -> int:
        """Nivel jerárquico del tipo de nodo."""
        return {
            NodeType.ROOT: 0,
            NodeType.CAPITULO: 1,
            NodeType.APU: 2,
            NodeType.INSUMO: 3,
        }[self]


class RiskLevel(str, Enum):
    """Niveles de riesgo estandarizados."""
    
    NONE = "NINGUNO"
    LOW = "BAJO"
    MEDIUM = "MEDIO"
    HIGH = "ALTO"
    CRITICAL = "CRÍTICO"
    
    @classmethod
    def from_score(cls, score: float) -> RiskLevel:
        """Determina nivel de riesgo desde un score [0, 1]."""
        if score <= 0.0:
            return cls.NONE
        if score <= 0.1:
            return cls.LOW
        if score <= 0.3:
            return cls.MEDIUM
        if score <= 0.6:
            return cls.HIGH
        return cls.CRITICAL
    
    @classmethod
    def from_temperature(cls, temp: float) -> RiskLevel:
        """Determina nivel de riesgo desde temperatura."""
        if temp <= 35.0:
            return cls.LOW
        if temp <= 50.0:
            return cls.MEDIUM
        if temp <= 70.0:
            return cls.HIGH
        return cls.CRITICAL


class ComplexityLevel(str, Enum):
    """Niveles de complejidad del proyecto."""
    
    LOW = "Baja"
    MEDIUM = "Media"
    HIGH = "Alta"
    CRITICAL = "CRÍTICA"
    
    @classmethod
    def from_integrity_score(cls, score: float) -> ComplexityLevel:
        """Determina complejidad desde score de integridad."""
        if score >= 85.0:
            return cls.LOW
        if score >= 70.0:
            return cls.MEDIUM
        if score >= 40.0:
            return cls.HIGH
        return cls.CRITICAL


class SpectralStatus(str, Enum):
    """Estados del análisis espectral."""
    
    SUCCESS = "success"
    INSUFFICIENT_NODES = "insufficient_nodes"
    DEGENERATE = "degenerate_after_isolation_removal"
    INSUFFICIENT_EIGENVALUES = "insufficient_eigenvalues"
    CONVERGENCE_FAILURE = "convergence_failure"
    ERROR = "error"


class MergeVerdict(str, Enum):
    """Veredictos de la auditoría de fusión Mayer-Vietoris."""
    
    CLEAN_MERGE = "CLEAN_MERGE"
    TOPOLOGY_SIMPLIFIED = "TOPOLOGY_SIMPLIFIED"
    INTEGRATION_CONFLICT = "INTEGRATION_CONFLICT"
    INCONSISTENT_TOPOLOGY = "INCONSISTENT_TOPOLOGY"


class MergeSeverity(str, Enum):
    """Severidad del resultado de fusión."""
    
    SUCCESS = "success"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# ============================================================================
# CONFIGURACIÓN CENTRALIZADA
# ============================================================================


@dataclass(frozen=True)
class ThermalConfig:
    """
    Configuración para el modelo termodinámico.
    
    Las temperaturas representan volatilidad esperada de precios.
    """
    
    # Temperaturas base por tipo de insumo (°C metafórico)
    temperature_combustible: float = 95.0
    temperature_acero: float = 85.0
    temperature_transporte: float = 75.0
    temperature_cemento: float = 60.0
    temperature_mano_obra: float = 25.0
    temperature_default: float = 30.0
    temperature_ambient: float = 25.0
    
    # Umbrales de riesgo térmico
    threshold_medium: float = 35.0
    threshold_high: float = 50.0
    threshold_critical: float = 70.0
    
    # Umbral para hotspots
    hotspot_threshold: float = 50.0
    
    def get_base_temperature(self, description: str, tipo_insumo: str) -> float:
        """
        Obtiene temperatura base según descripción y tipo.
        
        Args:
            description: Descripción del insumo.
            tipo_insumo: Tipo de insumo.
            
        Returns:
            Temperatura base en °C metafórico.
        """
        text = f"{description} {tipo_insumo}".upper()
        
        # Mapeo de keywords a temperaturas
        keyword_temps = {
            "COMBUSTIBLE": self.temperature_combustible,
            "GASOLINA": self.temperature_combustible,
            "DIESEL": self.temperature_combustible,
            "ACERO": self.temperature_acero,
            "HIERRO": self.temperature_acero,
            "METAL": self.temperature_acero,
            "TRANSPORTE": self.temperature_transporte,
            "FLETE": self.temperature_transporte,
            "CEMENTO": self.temperature_cemento,
            "CONCRETO": self.temperature_cemento,
            "HORMIGÓN": self.temperature_cemento,
            "MANO DE OBRA": self.temperature_mano_obra,
            "PERSONAL": self.temperature_mano_obra,
            "LABOR": self.temperature_mano_obra,
        }
        
        max_temp = self.temperature_default
        for keyword, temp in keyword_temps.items():
            if keyword in text:
                max_temp = max(max_temp, temp)
        
        return max_temp


@dataclass(frozen=True)
class SpectralConfig:
    """Configuración para análisis espectral."""
    
    # Umbral para considerar grafo desconectado
    fiedler_disconnected_threshold: float = 1e-9
    
    # Coeficiente de variación para riesgo de resonancia
    resonance_cv_threshold: float = 0.15
    
    # Número máximo de eigenvalores a calcular en modo sparse
    max_eigenvalues_sparse: int = 6
    
    # Umbral para modo denso vs sparse
    dense_threshold_nodes: int = 50
    
    # Tolerancia para ARPACK
    arpack_tolerance: float = 1e-6
    
    # Factor para maxiter de ARPACK
    arpack_maxiter_factor: int = 100


@dataclass(frozen=True)
class StabilityConfig:
    """Configuración para cálculo de estabilidad piramidal."""
    
    # Pesos para componentes de estabilidad
    weight_ratio: float = 0.35
    weight_density: float = 0.20
    weight_acyclic: float = 0.30
    weight_connectivity: float = 0.15
    
    # Umbrales de densidad
    density_sparse_threshold: float = 0.001
    density_dense_threshold: float = 0.5
    
    # Penalización por SCC
    scc_penalty_factor: float = 0.1
    scc_max_penalty: float = 0.5
    
    def __post_init__(self) -> None:
        """Valida que los pesos sumen 1.0."""
        total = (
            self.weight_ratio
            + self.weight_density
            + self.weight_acyclic
            + self.weight_connectivity
        )
        if abs(total - 1.0) > EPSILON:
            raise ValueError(
                f"Stability weights must sum to 1.0, got {total:.4f}"
            )


@dataclass(frozen=True)
class SynergyConfig:
    """Configuración para detección de sinergia de riesgo."""
    
    # Mínimo de nodos compartidos para considerar sinergia
    min_shared_nodes: int = 2
    
    # Percentil para determinar nodos críticos
    criticality_percentile: float = 75.0
    
    # Máximo de nodos puente a reportar
    max_bridge_nodes: int = 10
    
    # Factor para cálculo de score
    score_scaling_factor: float = 2.0


@dataclass(frozen=True)
class ScoringConfig:
    """Configuración para el scoring de integridad."""
    
    # Pesos para componentes del score base
    weight_euler: float = 0.30
    weight_stability: float = 0.25
    weight_density: float = 0.20
    weight_base: float = 0.25
    
    # Penalizaciones
    penalty_per_cycle: float = 5.0
    penalty_max_cycles: float = 25.0
    penalty_synergy_factor: float = 30.0
    penalty_max_synergy: float = 20.0
    penalty_resonance: float = 10.0
    
    def __post_init__(self) -> None:
        """Valida que los pesos sumen 1.0."""
        total = (
            self.weight_euler
            + self.weight_stability
            + self.weight_density
            + self.weight_base
        )
        if abs(total - 1.0) > EPSILON:
            raise ValueError(
                f"Scoring weights must sum to 1.0, got {total:.4f}"
            )


@dataclass(frozen=True)
class TopologyAnalyzerConfig:
    """Configuración consolidada del analizador topológico."""
    
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    synergy: SynergyConfig = field(default_factory=SynergyConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    
    # Límite de ciclos a detectar
    max_cycles: int = 100
    
    # Longitud máxima de ciclo a reportar
    max_cycle_length: int = 10
    
    # Recursos críticos a reportar
    top_critical_resources: int = 5
    
    # Validación estricta de invariantes topológicos
    strict_euler_validation: bool = False


# ============================================================================
# ESTRUCTURAS DE DATOS VALIDADAS
# ============================================================================


@dataclass(frozen=True)
class ValidatedBettiNumbers:
    """
    Números de Betti validados con invariante de Euler.
    
    Para grafos (1-complejos simpliciales):
    - β₀: Componentes conexas
    - β₁: Ciclos independientes (genus)
    - β₂: Cavidades (siempre 0 para grafos)
    
    Invariante: χ = β₀ - β₁ + β₂
    """
    
    beta_0: int
    beta_1: int
    beta_2: int
    euler_characteristic: int
    
    def __post_init__(self) -> None:
        """Valida invariantes."""
        if self.beta_0 < 0 or self.beta_1 < 0 or self.beta_2 < 0:
            raise ValueError("Betti numbers must be non-negative")
        
        expected_chi = self.beta_0 - self.beta_1 + self.beta_2
        if self.euler_characteristic != expected_chi:
            raise TopologyInvariantError(
                "Euler characteristic inconsistency",
                self.beta_0,
                self.beta_1,
                self.beta_2,
                self.euler_characteristic,
            )
    
    @classmethod
    def from_graph(cls, graph: nx.DiGraph) -> ValidatedBettiNumbers:
        """
        Calcula números de Betti desde un grafo dirigido.
        
        Utiliza MultiGraph para preservar aristas múltiples.
        """
        if graph.number_of_nodes() == 0:
            return cls(beta_0=0, beta_1=0, beta_2=0, euler_characteristic=0)
        
        # Convertir a MultiGraph para análisis correcto
        undirected = nx.MultiGraph()
        undirected.add_nodes_from(graph.nodes(data=True))
        undirected.add_edges_from(graph.edges(data=True))
        
        beta_0 = nx.number_connected_components(undirected)
        n_edges = undirected.number_of_edges()
        n_nodes = undirected.number_of_nodes()
        
        # Fórmula de Euler para 1-complejos: χ = V - E = β₀ - β₁
        # Despejando: β₁ = E - V + β₀
        beta_1 = max(0, n_edges - n_nodes + beta_0)
        beta_2 = 0  # Siempre 0 para grafos
        euler_char = beta_0 - beta_1 + beta_2
        
        return cls(
            beta_0=beta_0,
            beta_1=beta_1,
            beta_2=beta_2,
            euler_characteristic=euler_char,
        )
    
    def to_metrics(self) -> TopologicalMetrics:
        """Convierte a TopologicalMetrics para compatibilidad."""
        return TopologicalMetrics(
            beta_0=self.beta_0,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            euler_characteristic=self.euler_characteristic,
        )
    
    @property
    def is_connected(self) -> bool:
        """Verifica si el grafo es conexo."""
        return self.beta_0 == 1
    
    @property
    def is_acyclic(self) -> bool:
        """Verifica si no hay ciclos."""
        return self.beta_1 == 0
    
    @property
    def genus(self) -> int:
        """Género topológico (número de 'agujeros')."""
        return self.beta_1


@dataclass
class SpectralMetrics:
    """
    Métricas del análisis espectral del Laplaciano.
    
    Atributos:
        fiedler_value: λ₂ (conectividad algebraica)
        spectral_gap: λ₂/λ_max (capacidad de sincronización)
        spectral_energy: ||L||_F² (dispersión estructural)
        wavelength: 1/λ_max (escala de difusión)
        lambda_max: Máximo eigenvalor
        resonance_risk: True si hay riesgo de resonancia
        algebraic_connectivity: Alias de fiedler_value
        eigenvalues: Lista de eigenvalores principales
        isolated_nodes_removed: Nodos aislados removidos para análisis
        status: Estado del análisis
    """
    
    fiedler_value: float
    spectral_gap: float
    spectral_energy: float
    wavelength: float
    lambda_max: float
    resonance_risk: bool
    algebraic_connectivity: float
    eigenvalues: List[float]
    isolated_nodes_removed: int
    status: SpectralStatus
    
    @classmethod
    def empty(cls, status: SpectralStatus = SpectralStatus.INSUFFICIENT_NODES) -> SpectralMetrics:
        """Crea métricas vacías para casos degenerados."""
        return cls(
            fiedler_value=0.0,
            spectral_gap=0.0,
            spectral_energy=0.0,
            wavelength=float("inf"),
            lambda_max=0.0,
            resonance_risk=False,
            algebraic_connectivity=0.0,
            eigenvalues=[],
            isolated_nodes_removed=0,
            status=status,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "fiedler_value": round(self.fiedler_value, 6),
            "spectral_gap": round(self.spectral_gap, 6),
            "spectral_energy": round(self.spectral_energy, 4),
            "wavelength": round(self.wavelength, 6) if self.wavelength != float("inf") else "inf",
            "lambda_max": round(self.lambda_max, 6),
            "resonance_risk": self.resonance_risk,
            "algebraic_connectivity": round(self.algebraic_connectivity, 6),
            "eigenvalues": [round(e, 6) for e in self.eigenvalues[:8]],
            "isolated_nodes_removed": self.isolated_nodes_removed,
            "status": self.status.value,
        }


@dataclass
class ThermalMetrics:
    """
    Métricas del análisis de flujo térmico.
    
    Modela la propagación de volatilidad (calor) desde insumos hacia el proyecto.
    """
    
    system_temperature: float
    thermal_risk_level: RiskLevel
    hotspots: List[Dict[str, Any]]
    node_temperatures: Dict[str, float]
    
    @classmethod
    def empty(cls) -> ThermalMetrics:
        """Crea métricas vacías."""
        return cls(
            system_temperature=25.0,
            thermal_risk_level=RiskLevel.LOW,
            hotspots=[],
            node_temperatures={},
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "system_temperature": round(self.system_temperature, 2),
            "thermal_risk_level": self.thermal_risk_level.value,
            "hotspots": self.hotspots[:10],  # Limitar para serialización
            # node_temperatures omitido por tamaño
        }


@dataclass
class SynergyMetrics:
    """
    Métricas de sinergia de riesgo (efecto dominó).
    """
    
    synergy_detected: bool
    synergy_score: float
    synergy_strength: float  # Alias para compatibilidad
    risk_level: RiskLevel
    bridge_nodes: List[Dict[str, Any]]
    intersecting_cycles_count: int
    intersecting_cycles: List[List[str]] = field(default_factory=list)
    
    @classmethod
    def none_detected(cls) -> SynergyMetrics:
        """Crea métricas para caso sin sinergia."""
        return cls(
            synergy_detected=False,
            synergy_score=0.0,
            synergy_strength=0.0,
            risk_level=RiskLevel.NONE,
            bridge_nodes=[],
            intersecting_cycles_count=0,
            intersecting_cycles=[],
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "synergy_detected": self.synergy_detected,
            "synergy_score": round(self.synergy_score, 4),
            "synergy_strength": round(self.synergy_strength, 4),
            "risk_level": self.risk_level.value,
            "bridge_nodes": self.bridge_nodes[:10],
            "intersecting_cycles_count": self.intersecting_cycles_count,
        }


@dataclass
class MayerVietorisResult:
    """
    Resultado de la auditoría de fusión Mayer-Vietoris.
    
    Verifica la secuencia exacta:
    ... → H_k(A∩B) → H_k(A)⊕H_k(B) → H_k(A∪B) → ...
    """
    
    status: MergeVerdict
    severity: MergeSeverity
    delta_beta_1: int
    beta_1_observed: int
    beta_1_theoretical: int
    discrepancy: float
    boundary_nodes: List[str]
    details: Dict[str, int]
    narrative: str
    
    @property
    def is_clean(self) -> bool:
        """Verifica si la fusión es limpia."""
        return self.status == MergeVerdict.CLEAN_MERGE
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "status": self.status.value,
            "severity": self.severity.value,
            "delta_beta_1": self.delta_beta_1,
            "beta_1_observed": self.beta_1_observed,
            "beta_1_theoretical": self.beta_1_theoretical,
            "discrepancy": round(self.discrepancy, 2),
            "boundary_nodes": self.boundary_nodes[:10],
            "details": self.details,
            "narrative": self.narrative,
        }


@dataclass
class AnomalyClassification:
    """Clasificación de nodos anómalos."""
    
    isolated_nodes: List[Dict[str, Any]]
    orphan_insumos: List[Dict[str, Any]]
    empty_apus: List[Dict[str, Any]]
    
    @property
    def total_anomalies(self) -> int:
        """Total de anomalías detectadas."""
        return len(self.isolated_nodes) + len(self.orphan_insumos) + len(self.empty_apus)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "isolated_nodes": self.isolated_nodes,
            "orphan_insumos": self.orphan_insumos,
            "empty_apus": self.empty_apus,
        }


@dataclass
class ConstructionRiskReport:
    """
    Reporte Ejecutivo de Riesgos de Construcción.

    Consolida el análisis topológico, financiero y estructural en un
    informe unificado para la toma de decisiones estratégicas.
    """

    integrity_score: float
    waste_alerts: List[str]
    circular_risks: List[str]
    complexity_level: ComplexityLevel
    details: Dict[str, Any] = field(default_factory=dict)
    financial_risk_level: Optional[str] = None
    strategic_narrative: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def __post_init__(self) -> None:
        """Valida campos."""
        if not (0.0 <= self.integrity_score <= 100.0):
            raise ValueError(
                f"integrity_score must be in [0, 100], got {self.integrity_score}"
            )
        # Convertir string a enum si es necesario
        if isinstance(self.complexity_level, str):
            try:
                object.__setattr__(
                    self,
                    "complexity_level",
                    ComplexityLevel(self.complexity_level),
                )
            except ValueError:
                object.__setattr__(self, "complexity_level", ComplexityLevel.MEDIUM)
    
    @property
    def is_critical(self) -> bool:
        """Indica si el proyecto está en estado crítico."""
        return self.complexity_level == ComplexityLevel.CRITICAL
    
    @property
    def has_cycles(self) -> bool:
        """Indica si hay ciclos detectados."""
        return len(self.circular_risks) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario para API/persistencia."""
        return {
            "integrity_score": round(self.integrity_score, 1),
            "waste_alerts": self.waste_alerts,
            "circular_risks": self.circular_risks,
            "complexity_level": self.complexity_level.value,
            "financial_risk_level": self.financial_risk_level,
            "strategic_narrative": self.strategic_narrative,
            "timestamp": self.timestamp,
            "is_critical": self.is_critical,
            "has_cycles": self.has_cycles,
            "details": self.details,
        }


# ============================================================================
# UTILIDADES DE PARSING
# ============================================================================


class NumericParser:
    """
    Parser robusto para valores numéricos con soporte de localización.
    
    Maneja formatos:
    - 1,234,567.89 (inglés)
    - 1.234.567,89 (europeo)
    - 1234567.89 (sin separador de miles)
    """
    
    @staticmethod
    def parse(value: Any, default: float = 0.0) -> float:
        """
        Convierte un valor a float de manera segura.
        
        Args:
            value: Valor a convertir.
            default: Valor por defecto si falla la conversión.
            
        Returns:
            Valor numérico flotante.
        """
        if pd.isna(value) or value is None:
            return default
        
        if isinstance(value, (int, float)):
            result = float(value)
            return default if math.isnan(result) or math.isinf(result) else result
        
        try:
            # Limpiar string
            str_value = (
                str(value)
                .strip()
                .replace("\xa0", "")
                .replace(" ", "")
                .replace("$", "")
                .replace("€", "")
                .replace("£", "")
            )
            
            if not str_value or str_value in ("-", "N/A", "NA", "null", "None"):
                return default
            
            has_comma = "," in str_value
            has_dot = "." in str_value
            
            if has_comma and has_dot:
                # Determinar separador decimal por posición
                last_comma = str_value.rfind(",")
                last_dot = str_value.rfind(".")
                
                if last_dot > last_comma:
                    # Formato inglés: 1,234,567.89
                    str_value = str_value.replace(",", "")
                else:
                    # Formato europeo: 1.234.567,89
                    str_value = str_value.replace(".", "").replace(",", ".")
            
            elif has_comma and not has_dot:
                parts = str_value.split(",")
                # Heurística: última parte con 1-3 dígitos es decimal
                if (
                    len(parts) == 2
                    and 1 <= len(parts[1]) <= 3
                    and parts[1].isdigit()
                ):
                    str_value = str_value.replace(",", ".")
                else:
                    # Es separador de miles
                    str_value = str_value.replace(",", "")
            
            result = float(str_value)
            return default if math.isnan(result) or math.isinf(result) else result
            
        except (ValueError, TypeError, AttributeError):
            return default
    
    @staticmethod
    def sanitize_code(value: Any) -> str:
        """
        Sanitiza un código o identificador.
        
        Args:
            value: Valor a sanitizar.
            
        Returns:
            String limpio y normalizado.
        """
        if pd.isna(value) or value is None:
            return ""
        sanitized = str(value).strip()
        # Colapsar espacios múltiples
        sanitized = " ".join(sanitized.split())
        return sanitized


# ============================================================================
# CONSTRUCTOR DEL GRAFO DE PRESUPUESTO
# ============================================================================


class BudgetGraphBuilder:
    """
    Constructor del Grafo de Presupuesto (Topología de Negocio).

    Implementa la construcción jerárquica de la estructura piramidal:
    - Nivel 0: Raíz del Proyecto (Apex)
    - Nivel 1: Capítulos (Pilares Estructurales)
    - Nivel 2: APUs (Cuerpo Táctico)
    - Nivel 3: Insumos (Cimentación de Recursos)
    """

    def __init__(self, root_node: str = DEFAULT_ROOT_NODE):
        """
        Inicializa el constructor.
        
        Args:
            root_node: Nombre del nodo raíz.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ROOT_NODE = root_node
        self._parser = NumericParser()

    def _create_node_attributes(
        self,
        node_type: NodeType,
        source: str = "generated",
        idx: int = -1,
        inferred: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Crea diccionario de atributos estandarizado para un nodo.
        
        Args:
            node_type: Tipo de nodo.
            source: Origen del dato (presupuesto, detail, generated).
            idx: Índice original en DataFrame.
            inferred: Si el nodo fue inferido.
            **kwargs: Atributos adicionales.
            
        Returns:
            Diccionario de atributos.
        """
        attrs = {
            "type": node_type.value,
            "level": node_type.level,
            "source": source,
            "original_index": idx,
            "inferred": inferred,
        }
        attrs.update(kwargs)
        return attrs

    def _create_apu_attributes(
        self,
        row: pd.Series,
        source: str,
        idx: int,
        inferred: bool,
    ) -> Dict[str, Any]:
        """Genera atributos específicos para nodos APU."""
        attrs = self._create_node_attributes(
            node_type=NodeType.APU,
            source=source,
            idx=idx,
            inferred=inferred,
        )
        if not inferred:
            attrs["description"] = self._parser.sanitize_code(
                row.get(ColumnNames.DESCRIPCION_APU)
            )
            attrs["quantity"] = self._parser.parse(
                row.get(ColumnNames.CANTIDAD_PRESUPUESTO)
            )
        return attrs

    def _create_insumo_attributes(
        self,
        row: pd.Series,
        insumo_desc: str,
        source: str,
        idx: int,
    ) -> Dict[str, Any]:
        """Genera atributos específicos para nodos INSUMO."""
        return self._create_node_attributes(
            node_type=NodeType.INSUMO,
            source=source,
            idx=idx,
            description=insumo_desc,
            tipo_insumo=self._parser.sanitize_code(row.get(ColumnNames.TIPO_INSUMO)),
            unit_cost=self._parser.parse(row.get(ColumnNames.COSTO_INSUMO_EN_APU)),
        )

    def _upsert_edge(
        self,
        G: nx.DiGraph,
        u: str,
        v: str,
        unit_cost: float,
        quantity: float,
        idx: int,
    ) -> bool:
        """
        Inserta o actualiza una arista con agregación de costos.
        
        Permite múltiples ocurrencias del mismo insumo en un APU.
        
        Args:
            G: Grafo.
            u: Nodo origen.
            v: Nodo destino.
            unit_cost: Costo unitario.
            quantity: Cantidad.
            idx: Índice original.
            
        Returns:
            True si se creó nueva arista, False si se actualizó.
        """
        total_cost = unit_cost * quantity

        if G.has_edge(u, v):
            edge = G[u][v]
            edge["quantity"] += quantity
            edge["total_cost"] += total_cost
            edge["occurrence_count"] += 1
            edge.setdefault("original_indices", []).append(idx)
            return False

        G.add_edge(
            u,
            v,
            quantity=quantity,
            unit_cost=unit_cost,
            total_cost=total_cost,
            occurrence_count=1,
            original_indices=[idx],
        )
        return True

    def _compute_graph_statistics(self, G: nx.DiGraph) -> Dict[str, int]:
        """Calcula estadísticas básicas del grafo construido."""
        stats = {
            "chapter_count": 0,
            "apu_count": 0,
            "insumo_count": 0,
            "inferred_count": 0,
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
        }

        type_counters = {
            "CAPITULO": "chapter_count",
            "APU": "apu_count",
            "INSUMO": "insumo_count",
        }

        for _, data in G.nodes(data=True):
            node_type = data.get("type")
            if node_type in type_counters:
                stats[type_counters[node_type]] += 1
                if node_type == "APU" and data.get("inferred", False):
                    stats["inferred_count"] += 1

        return stats

    def _process_presupuesto_row(
        self,
        G: nx.DiGraph,
        row: pd.Series,
        idx: int,
        chapter_cols: List[str],
    ) -> None:
        """
        Procesa una fila del presupuesto para construir estructura superior.
        
        Niveles: Raíz -> Capítulo -> APU
        """
        apu_code = self._parser.sanitize_code(row.get(ColumnNames.CODIGO_APU))
        if not apu_code:
            return

        # Calcular costo total del APU
        total_cost = self._parser.parse(row.get(ColumnNames.VALOR_TOTAL_APU))
        if total_cost == 0.0:
            # Fallback: Cantidad * Precio Unitario
            qty = self._parser.parse(row.get(ColumnNames.CANTIDAD_PRESUPUESTO))
            price = self._parser.parse(row.get(ColumnNames.PRECIO_UNIT_APU))
            total_cost = qty * price

        # Crear nodo APU
        attrs = self._create_apu_attributes(
            row, source="presupuesto", idx=idx, inferred=False
        )
        G.add_node(apu_code, **attrs)

        # Buscar capítulo
        chapter_name = None
        for col in chapter_cols:
            val = self._parser.sanitize_code(row.get(col))
            if val:
                chapter_name = val
                break

        if chapter_name:
            if chapter_name not in G:
                G.add_node(
                    chapter_name,
                    **self._create_node_attributes(
                        NodeType.CAPITULO,
                        description=f"Capítulo: {chapter_name}",
                    ),
                )
                G.add_edge(
                    self.ROOT_NODE,
                    chapter_name,
                    relation="CONTAINS",
                    weight=0.0,
                    total_cost=0.0,
                )

            # Acumular costo en arista Root -> Capítulo
            if G.has_edge(self.ROOT_NODE, chapter_name):
                edge = G[self.ROOT_NODE][chapter_name]
                edge["weight"] = edge.get("weight", 0.0) + total_cost
                edge["total_cost"] = edge.get("total_cost", 0.0) + total_cost

            # Arista Capítulo -> APU
            G.add_edge(
                chapter_name,
                apu_code,
                relation="CONTAINS",
                weight=total_cost,
                total_cost=total_cost,
            )
        else:
            # APU huérfano de capítulo
            G.add_edge(
                self.ROOT_NODE,
                apu_code,
                relation="CONTAINS",
                weight=total_cost,
                total_cost=total_cost,
            )

    def _process_apu_detail_row(
        self,
        G: nx.DiGraph,
        row: pd.Series,
        idx: int,
    ) -> None:
        """
        Procesa una fila de detalle de APU para construir la base.
        
        Niveles: APU -> Insumo
        """
        apu_code = self._parser.sanitize_code(row.get(ColumnNames.CODIGO_APU))
        insumo_desc = self._parser.sanitize_code(row.get(ColumnNames.DESCRIPCION_INSUMO))

        if not apu_code or not insumo_desc:
            return

        # Inferir APU si no existe
        if apu_code not in G:
            attrs = self._create_apu_attributes(
                row, source="detail", idx=idx, inferred=True
            )
            G.add_node(apu_code, **attrs)
            G.add_edge(self.ROOT_NODE, apu_code, relation="CONTAINS_INFERRED")

        # Crear o reutilizar nodo Insumo
        if insumo_desc not in G:
            attrs = self._create_insumo_attributes(
                row, insumo_desc, source="detail", idx=idx
            )
            G.add_node(insumo_desc, **attrs)

        # Establecer relación APU -> Insumo
        qty = self._parser.parse(row.get(ColumnNames.CANTIDAD_APU))
        cost = self._parser.parse(row.get(ColumnNames.COSTO_INSUMO_EN_APU))
        self._upsert_edge(G, apu_code, insumo_desc, cost, qty, idx)

    def build(
        self,
        presupuesto_df: Optional[pd.DataFrame],
        apus_detail_df: Optional[pd.DataFrame] = None,
    ) -> nx.DiGraph:
        """
        Construye el Grafo Piramidal de Presupuesto.
        
        Args:
            presupuesto_df: DataFrame del presupuesto general.
            apus_detail_df: DataFrame con detalle de APUs.
            
        Returns:
            Grafo dirigido representando la topología del proyecto.
            
        Raises:
            GraphConstructionError: Si hay errores críticos durante la construcción.
        """
        G = nx.DiGraph(name="BudgetTopology")
        self.logger.info("Iniciando construcción del Grafo Piramidal...")

        # Nodo Raíz (Ápice)
        G.add_node(
            self.ROOT_NODE,
            **self._create_node_attributes(
                NodeType.ROOT,
                description="Proyecto Completo",
            ),
        )

        # Columnas candidatas para identificar capítulos
        chapter_cols = ["CAPITULO", "CATEGORIA", "TITULO"]

        # Procesar Presupuesto (Niveles 0-2)
        if presupuesto_df is not None and not presupuesto_df.empty:
            available_cols = [c for c in chapter_cols if c in presupuesto_df.columns]
            for idx, row in presupuesto_df.iterrows():
                try:
                    self._process_presupuesto_row(G, row, idx, available_cols)
                except Exception as e:
                    self.logger.warning(f"Error procesando fila {idx} de presupuesto: {e}")

        # Procesar Detalle de APUs (Nivel 3)
        if apus_detail_df is not None and not apus_detail_df.empty:
            for idx, row in apus_detail_df.iterrows():
                try:
                    self._process_apu_detail_row(G, row, idx)
                except Exception as e:
                    self.logger.warning(f"Error procesando fila {idx} de detalle: {e}")

        stats = self._compute_graph_statistics(G)
        self.logger.info(f"Grafo Piramidal construido: {stats}")
        
        return G


# ============================================================================
# ANALIZADOR TOPOLÓGICO PRINCIPAL
# ============================================================================


class BusinessTopologicalAnalyzer:
    """
    Analizador de Topología de Negocio V2.

    Fusiona el rigor matemático (Topología Algebraica, Análisis Espectral)
    con narrativa de ingeniería civil ("La Voz del Consejo").

    Responsabilidades:
    - Calcular invariantes topológicos (Betti numbers)
    - Analizar estabilidad espectral y resonancia
    - Detectar ciclos y sinergia de riesgos
    - Modelar flujo térmico (volatilidad)
    - Generar reportes estratégicos
    """

    def __init__(
        self,
        config: Optional[TopologyAnalyzerConfig] = None,
        telemetry: Optional[TelemetryContext] = None,
    ):
        """
        Inicializa el analizador.
        
        Args:
            config: Configuración del analizador.
            telemetry: Contexto de telemetría opcional.
        """
        self.config = config or TopologyAnalyzerConfig()
        self.telemetry = telemetry
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Caché para cálculos costosos
        self._cycle_cache: Dict[int, Tuple[List[List[str]], bool]] = {}
        self._spectral_cache: Dict[int, SpectralMetrics] = {}
    
    @property
    def max_cycles(self) -> int:
        """Límite máximo de ciclos (compatibilidad legacy)."""
        return self.config.max_cycles
    
    def _clear_caches(self) -> None:
        """Limpia los cachés internos."""
        self._cycle_cache.clear()
        self._spectral_cache.clear()

    # ========================================================================
    # ANÁLISIS ESPECTRAL
    # ========================================================================

    def analyze_spectral_stability(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Calcula la Estabilidad Espectral y métricas del Laplaciano.
        
        Fundamentos Matemáticos:
        - Fiedler Value (λ₂): Conectividad algebraica
        - Gap Espectral: λ₂/λ_max (capacidad de sincronización)
        - Energía Espectral: ||L||_F² (dispersión estructural)
        
        Args:
            graph: Grafo a analizar.
            
        Returns:
            Dict con métricas espectrales.
        """
        metrics = self._analyze_spectral_stability_internal(graph)
        return metrics.to_dict()
    
    def _analyze_spectral_stability_internal(self, graph: nx.DiGraph) -> SpectralMetrics:
        """Implementación interna del análisis espectral."""
        n_nodes = graph.number_of_nodes()
        
        if n_nodes < 2:
            return SpectralMetrics.empty(SpectralStatus.INSUFFICIENT_NODES)
        
        # Verificar caché
        graph_hash = hash(frozenset(graph.edges()))
        if graph_hash in self._spectral_cache:
            return self._spectral_cache[graph_hash]
        
        cfg = self.config.spectral
        
        try:
            # Convertir a no dirigido para análisis simétrico
            ud_graph = graph.to_undirected()
            
            # Remover nodos aislados
            isolated = list(nx.isolates(ud_graph))
            if isolated:
                ud_graph = ud_graph.copy()
                ud_graph.remove_nodes_from(isolated)
            
            n_effective = ud_graph.number_of_nodes()
            if n_effective < 2:
                return SpectralMetrics.empty(SpectralStatus.DEGENERATE)
            
            # Laplaciano Normalizado
            L = nx.normalized_laplacian_matrix(ud_graph).astype(np.float64)
            
            # Energía Espectral
            if scipy.sparse.issparse(L):
                spectral_energy = float(scipy.sparse.linalg.norm(L, "fro") ** 2)
            else:
                spectral_energy = float(np.linalg.norm(L, "fro") ** 2)
            
            # Cálculo de Eigenvalores
            if n_effective <= cfg.dense_threshold_nodes:
                eigenvalues = self._compute_eigenvalues_dense(L)
            else:
                eigenvalues = self._compute_eigenvalues_sparse(L, n_effective, cfg)
            
            if len(eigenvalues) < 2:
                return SpectralMetrics.empty(SpectralStatus.INSUFFICIENT_EIGENVALUES)
            
            # Procesar eigenvalores
            fiedler_value = float(eigenvalues[1])
            if fiedler_value < cfg.fiedler_disconnected_threshold:
                fiedler_value = 0.0
            
            lambda_max = min(float(eigenvalues[-1]), LAPLACIAN_MAX_EIGENVALUE)
            spectral_gap = fiedler_value / lambda_max if lambda_max > EPSILON else 0.0
            wavelength = 1.0 / lambda_max if lambda_max > EPSILON else float("inf")
            
            # Riesgo de Resonancia
            resonance_risk = self._compute_resonance_risk(eigenvalues, cfg)
            
            metrics = SpectralMetrics(
                fiedler_value=fiedler_value,
                spectral_gap=spectral_gap,
                spectral_energy=spectral_energy,
                wavelength=wavelength,
                lambda_max=lambda_max,
                resonance_risk=resonance_risk,
                algebraic_connectivity=fiedler_value,
                eigenvalues=[float(e) for e in eigenvalues],
                isolated_nodes_removed=len(isolated),
                status=SpectralStatus.SUCCESS,
            )
            
            # Cachear resultado
            self._spectral_cache[graph_hash] = metrics
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error en análisis espectral: {e}", exc_info=True)
            return SpectralMetrics.empty(SpectralStatus.ERROR)
    
    def _compute_eigenvalues_dense(self, L: Any) -> np.ndarray:
        """Calcula eigenvalores usando método denso."""
        eigenvalues = np.linalg.eigvalsh(L.toarray() if scipy.sparse.issparse(L) else L)
        return np.sort(np.real(eigenvalues))
    
    def _compute_eigenvalues_sparse(
        self,
        L: Any,
        n_nodes: int,
        cfg: SpectralConfig,
    ) -> np.ndarray:
        """Calcula eigenvalores usando método sparse."""
        k_small = min(n_nodes - 1, cfg.max_eigenvalues_sparse)
        maxiter = n_nodes * cfg.arpack_maxiter_factor
        
        try:
            eigenvalues_small = scipy.sparse.linalg.eigsh(
                L,
                k=k_small,
                which="SM",
                return_eigenvectors=False,
                tol=cfg.arpack_tolerance,
                maxiter=maxiter,
            )
        except scipy.sparse.linalg.ArpackNoConvergence as e:
            eigenvalues_small = e.eigenvalues
            self.logger.warning(f"ARPACK no convergió para eigenvalores pequeños")
        
        try:
            eigenvalues_large = scipy.sparse.linalg.eigsh(
                L, k=1, which="LM", return_eigenvectors=False
            )
        except scipy.sparse.linalg.ArpackNoConvergence as e:
            eigenvalues_large = e.eigenvalues if len(e.eigenvalues) > 0 else np.array([2.0])
        
        return np.sort(np.concatenate([eigenvalues_small, eigenvalues_large]))
    
    def _compute_resonance_risk(
        self,
        eigenvalues: np.ndarray,
        cfg: SpectralConfig,
    ) -> bool:
        """Calcula riesgo de resonancia basado en CV de eigenvalores."""
        if len(eigenvalues) <= 2:
            return True
        
        eigen_std = np.std(eigenvalues)
        eigen_mean = np.mean(eigenvalues)
        cv = eigen_std / eigen_mean if eigen_mean > EPSILON else 0.0
        
        return cv < cfg.resonance_cv_threshold

    # ========================================================================
    # EFICIENCIA DE EULER
    # ========================================================================

    def calculate_euler_efficiency(
        self,
        graph: nx.DiGraph,
        weighted: bool = False,
    ) -> float:
        """
        Calcula la Eficiencia de Euler mediante decaimiento exponencial.
        
        Eficiencia = exp(-exceso_aristas / n_nodos)
        
        donde exceso_aristas = E - (V - 1)
        
        Args:
            graph: Grafo a analizar.
            weighted: Si aplica penalización adicional por aristas pesadas.
            
        Returns:
            Eficiencia normalizada [0.0, 1.0].
        """
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        
        if n_nodes == 0:
            return 1.0
        
        # Un grafo desconexo tiene eficiencia 0
        if n_nodes > 1 and not nx.is_weakly_connected(graph):
            return 0.0
        
        # Mínimo de aristas para un árbol
        min_edges = max(0, n_nodes - 1)
        excess_edges = max(0, n_edges - min_edges)
        
        # Eficiencia base
        efficiency = math.exp(-excess_edges / n_nodes)
        
        if weighted:
            # Penalización por aristas con peso significativo
            heavy_weight_threshold = 1.0
            has_heavy_edges = any(
                d.get("weight", 0.0) > heavy_weight_threshold
                for _, _, d in graph.edges(data=True)
            )
            efficiency *= 0.8 if has_heavy_edges else 0.95
        
        return min(1.0, max(0.0, efficiency))

    # ========================================================================
    # NÚMEROS DE BETTI
    # ========================================================================

    def calculate_betti_numbers(self, graph: nx.DiGraph) -> TopologicalMetrics:
        """
        Calcula los Números de Betti (Invariantes Topológicos).
        
        Args:
            graph: Grafo a analizar.
            
        Returns:
            TopologicalMetrics con β₀, β₁, β₂ y χ.
        """
        betti = ValidatedBettiNumbers.from_graph(graph)
        return betti.to_metrics()

    # ========================================================================
    # ESTABILIDAD PIRAMIDAL
    # ========================================================================

    def calculate_pyramid_stability(self, graph: nx.DiGraph) -> float:
        """
        Calcula el Índice de Estabilidad Piramidal (Ψ).
        
        Modelo: Pirámide estable = base ancha (insumos) + cúspide estrecha (APUs).
        
        Ψ = w_r·f_ratio + w_d·f_density + w_a·f_acyclic + w_c·f_connectivity
        
        Args:
            graph: Grafo a analizar.
            
        Returns:
            Índice de estabilidad normalizado [0, 1].
        """
        if graph.number_of_nodes() == 0:
            return 0.0
        
        cfg = self.config.stability
        
        # Contar tipos de nodos
        type_counts = {t.value: 0 for t in NodeType}
        for _, data in graph.nodes(data=True):
            node_type = data.get("type", "")
            if node_type in type_counts:
                type_counts[node_type] += 1
        
        num_apus = type_counts[NodeType.APU.value]
        num_insumos = type_counts[NodeType.INSUMO.value]
        
        # Factor 1: Ratio de Base (escalamiento logarítmico)
        if num_apus == 0:
            ratio_score = 0.0 if num_insumos == 0 else 0.5
        elif num_insumos == 0:
            ratio_score = 0.3
        else:
            base_ratio = num_insumos / num_apus
            # log₁₀(1 + ratio) / log₁₀(11) para normalizar
            ratio_score = min(1.0, math.log10(1 + base_ratio) / math.log10(11))
        
        # Factor 2: Densidad
        density = nx.density(graph)
        if density < cfg.density_sparse_threshold:
            density_score = 0.5  # Muy disperso
        elif density > cfg.density_dense_threshold:
            density_score = 0.3  # Muy denso
        else:
            density_score = 1.0 - min(0.7, density)
        
        # Factor 3: Aciclicidad
        if nx.is_directed_acyclic_graph(graph):
            acyclic_score = 1.0
        else:
            # Penalización por SCCs no triviales
            sccs = [c for c in nx.strongly_connected_components(graph) if len(c) > 1]
            cycle_penalty = min(cfg.scc_max_penalty, len(sccs) * cfg.scc_penalty_factor)
            acyclic_score = max(0.0, 0.5 - cycle_penalty)
        
        # Factor 4: Conectividad
        if nx.is_weakly_connected(graph):
            connectivity_score = 1.0
        else:
            num_components = nx.number_weakly_connected_components(graph)
            connectivity_score = 1.0 / num_components
        
        # Combinación ponderada
        stability = (
            cfg.weight_ratio * ratio_score
            + cfg.weight_density * density_score
            + cfg.weight_acyclic * acyclic_score
            + cfg.weight_connectivity * connectivity_score
        )
        
        return round(max(0.0, min(1.0, stability)), 4)

    # ========================================================================
    # AUDITORÍA MAYER-VIETORIS
    # ========================================================================

    def audit_integration_homology(
        self,
        graph_a: nx.DiGraph,
        graph_b: nx.DiGraph,
    ) -> Dict[str, Any]:
        """
        Ejecuta el Test de Mayer-Vietoris para auditoría de fusión.
        
        Evalúa si A ∪ B introduce complejidad cíclica no presente en A, B.
        
        Args:
            graph_a: Primer grafo (ej. Presupuesto).
            graph_b: Segundo grafo (ej. Detalle APUs).
            
        Returns:
            Diagnóstico de la fusión.
        """
        result = self._audit_integration_homology_internal(graph_a, graph_b)
        return result.to_dict()
    
    def _audit_integration_homology_internal(
        self,
        graph_a: nx.DiGraph,
        graph_b: nx.DiGraph,
    ) -> MayerVietorisResult:
        """Implementación interna de la auditoría."""
        # Calcular Betti numbers para cada grafo
        betti_a = ValidatedBettiNumbers.from_graph(graph_a)
        betti_b = ValidatedBettiNumbers.from_graph(graph_b)
        
        # Unión
        graph_union = nx.compose(graph_a, graph_b)
        betti_union = ValidatedBettiNumbers.from_graph(graph_union)
        
        # Intersección
        nodes_a = set(graph_a.nodes())
        nodes_b = set(graph_b.nodes())
        common_nodes = nodes_a & nodes_b
        
        graph_intersection = nx.DiGraph()
        if common_nodes:
            graph_intersection.add_nodes_from(
                (n, graph_a.nodes[n]) for n in common_nodes
            )
            for u, v in graph_a.edges():
                if u in common_nodes and v in common_nodes and graph_b.has_edge(u, v):
                    graph_intersection.add_edge(u, v)
        
        betti_intersection = ValidatedBettiNumbers.from_graph(graph_intersection)
        
        # Cálculo de Mayer-Vietoris
        delta_connectivity = (
            betti_intersection.beta_0
            - betti_a.beta_0
            - betti_b.beta_0
            + betti_union.beta_0
        )
        
        beta1_theoretical = (
            betti_a.beta_1
            + betti_b.beta_1
            - betti_intersection.beta_1
            + max(0, delta_connectivity)
        )
        
        emergent_observed = betti_union.beta_1 - (betti_a.beta_1 + betti_b.beta_1)
        discrepancy = abs(betti_union.beta_1 - beta1_theoretical)
        
        # Determinar veredicto
        if discrepancy <= 1:
            if emergent_observed > 0:
                verdict = MergeVerdict.INTEGRATION_CONFLICT
                severity = MergeSeverity.WARNING
            elif emergent_observed < 0:
                verdict = MergeVerdict.TOPOLOGY_SIMPLIFIED
                severity = MergeSeverity.INFO
            else:
                verdict = MergeVerdict.CLEAN_MERGE
                severity = MergeSeverity.SUCCESS
        else:
            verdict = MergeVerdict.INCONSISTENT_TOPOLOGY
            severity = MergeSeverity.ERROR
        
        # Identificar nodos frontera
        boundary_nodes = []
        for node in common_nodes:
            neighbors_a = set(graph_a.successors(node)) | set(graph_a.predecessors(node))
            neighbors_b = set(graph_b.successors(node)) | set(graph_b.predecessors(node))
            if (neighbors_a - nodes_b) | (neighbors_b - nodes_a):
                boundary_nodes.append(node)
        
        narrative = self._generate_mayer_vietoris_narrative(
            emergent_observed, discrepancy, len(boundary_nodes)
        )
        
        return MayerVietorisResult(
            status=verdict,
            severity=severity,
            delta_beta_1=emergent_observed,
            beta_1_observed=betti_union.beta_1,
            beta_1_theoretical=beta1_theoretical,
            discrepancy=discrepancy,
            boundary_nodes=boundary_nodes,
            details={
                "beta_1_A": betti_a.beta_1,
                "beta_1_B": betti_b.beta_1,
                "beta_1_intersection": betti_intersection.beta_1,
                "beta_1_union": betti_union.beta_1,
                "common_nodes_count": len(common_nodes),
            },
            narrative=narrative,
        )

    def _generate_mayer_vietoris_narrative(
        self,
        observed: int,
        discrepancy: float,
        boundary_count: int,
    ) -> str:
        """Genera narrativa del análisis Mayer-Vietoris."""
        parts = []
        
        if discrepancy > 2:
            parts.append(
                f"⚠️ ANOMALÍA TOPOLÓGICA: Discrepancia significativa (Δ={discrepancy:.1f}). "
                f"Revisar coherencia en {boundary_count} nodos de frontera."
            )
        elif discrepancy > 1:
            parts.append(f"⚠️ Discrepancia menor (Δ={discrepancy:.1f}). Posible redundancia.")
        
        if observed > 0:
            parts.append(
                f"🚨 CONFLICTO DE INTEGRACIÓN: {observed} nuevo(s) ciclo(s) emergentes."
            )
        elif observed < 0:
            parts.append(
                f"✅ SIMPLIFICACIÓN TOPOLÓGICA: Eliminación de {abs(observed)} ciclo(s)."
            )
        elif discrepancy <= 1:
            parts.append("✅ FUSIÓN LIMPIA: Integración topológicamente neutral.")
        
        return " ".join(parts) if parts else "Análisis completado."

    # ========================================================================
    # DETECCIÓN DE CICLOS
    # ========================================================================

    def _get_raw_cycles(
        self,
        graph: nx.DiGraph,
        use_cache: bool = True,
    ) -> Tuple[List[List[str]], bool]:
        """
        Obtiene ciclos del grafo.
        
        Args:
            graph: Grafo a analizar.
            use_cache: Si usar caché.
            
        Returns:
            Tupla (lista de ciclos, flag de truncamiento).
        """
        graph_hash = hash(frozenset(graph.edges()))
        
        if use_cache and graph_hash in self._cycle_cache:
            return self._cycle_cache[graph_hash]
        
        cycles: List[List[str]] = []
        truncated = False
        max_length = self.config.max_cycle_length
        
        try:
            cycle_generator = nx.simple_cycles(graph)
            for count, cycle in enumerate(cycle_generator):
                if count >= self.config.max_cycles:
                    truncated = True
                    self.logger.warning(
                        f"Ciclos truncados en {self.config.max_cycles}"
                    )
                    break
                if len(cycle) <= max_length:
                    cycles.append([str(n) for n in cycle])
        except Exception as e:
            self.logger.error(f"Error en detección de ciclos: {e}")
        
        cycles.sort(key=len)
        
        if use_cache:
            self._cycle_cache[graph_hash] = (cycles, truncated)
        
        return cycles, truncated

    # ========================================================================
    # SINERGIA DE RIESGO
    # ========================================================================

    def detect_risk_synergy(
        self,
        graph: nx.DiGraph,
        raw_cycles: Optional[List[List[str]]] = None,
        weighted: bool = False,
    ) -> Dict[str, Any]:
        """
        Detecta Sinergia de Riesgo (Efecto Dominó).
        
        Identifica si múltiples ciclos comparten nodos críticos (puentes).
        
        Args:
            graph: Grafo a analizar.
            raw_cycles: Ciclos pre-calculados.
            weighted: Si considera peso de nodos puente.
            
        Returns:
            Análisis de sinergia.
        """
        metrics = self._detect_risk_synergy_internal(graph, raw_cycles, weighted)
        return metrics.to_dict()
    
    def _detect_risk_synergy_internal(
        self,
        graph: nx.DiGraph,
        raw_cycles: Optional[List[List[str]]] = None,
        weighted: bool = False,
    ) -> SynergyMetrics:
        """Implementación interna de detección de sinergia."""
        if raw_cycles is None:
            raw_cycles, _ = self._get_raw_cycles(graph)
        
        if len(raw_cycles) < 2:
            return SynergyMetrics.none_detected()
        
        cfg = self.config.synergy
        
        # Centralidad de intermediación
        try:
            k = min(100, graph.number_of_nodes())
            betweenness = nx.betweenness_centrality(graph, normalized=True, k=k)
        except Exception:
            betweenness = {n: 0.0 for n in graph.nodes()}
        
        # Umbral adaptativo para nodos críticos
        vals = list(betweenness.values())
        if len(vals) >= 4:
            threshold = np.percentile(vals, cfg.criticality_percentile)
        else:
            threshold = np.mean(vals) if vals else 0.0
        
        critical_nodes = {n for n, c in betweenness.items() if c >= threshold and c > 0}
        
        # Encontrar pares de ciclos que comparten nodos
        synergy_pairs: List[Tuple[int, int]] = []
        bridge_occ: Dict[str, List[Tuple[int, int]]] = {}
        intersecting_cycles: List[List[str]] = []
        
        cycle_sets = [set(c) for c in raw_cycles]
        n_cycles = len(cycle_sets)
        
        for i in range(n_cycles):
            for j in range(i + 1, n_cycles):
                inter = cycle_sets[i] & cycle_sets[j]
                # Requiere compartir al menos min_shared_nodes
                has_synergy = (
                    len(inter) >= cfg.min_shared_nodes
                    or (weighted and bool(inter & critical_nodes))
                )
                if has_synergy:
                    synergy_pairs.append((i, j))
                    if not intersecting_cycles:
                        intersecting_cycles.append(raw_cycles[i])
                    for node in inter:
                        bridge_occ.setdefault(node, []).append((i, j))
        
        if not synergy_pairs:
            return SynergyMetrics.none_detected()
        
        # Ranking de nodos puente
        bridge_nodes = sorted(
            [
                {
                    "id": n,
                    "cycles_connected": len(lst),
                    "betweenness": round(betweenness.get(n, 0), 4),
                }
                for n, lst in bridge_occ.items()
            ],
            key=lambda x: (x["cycles_connected"], x["betweenness"]),
            reverse=True,
        )
        
        # Score de sinergia
        total_pairs = n_cycles * (n_cycles - 1) / 2
        pair_ratio = len(synergy_pairs) / total_pairs if total_pairs > 0 else 0
        synergy_score = min(1.0, pair_ratio * cfg.score_scaling_factor)
        
        if weighted:
            synergy_score = min(1.0, synergy_score * 1.1)
        
        risk_level = RiskLevel.from_score(synergy_score)
        
        return SynergyMetrics(
            synergy_detected=True,
            synergy_score=round(synergy_score, 4),
            synergy_strength=round(synergy_score, 4),
            risk_level=risk_level,
            bridge_nodes=bridge_nodes[:cfg.max_bridge_nodes],
            intersecting_cycles_count=len(synergy_pairs),
            intersecting_cycles=intersecting_cycles,
        )

    # ========================================================================
    # FLUJO TÉRMICO
    # ========================================================================

    def analyze_thermal_flow(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Calcula el Flujo Térmico Estructural (Modelo de Difusión).
        
        Simula cómo la volatilidad de precios (temperatura) se propaga
        desde insumos (hojas) hacia el proyecto total (raíz).
        
        Args:
            graph: Grafo del presupuesto.
            
        Returns:
            Mapa térmico del proyecto.
        """
        metrics = self._analyze_thermal_flow_internal(graph)
        return metrics.to_dict() | {"node_temperatures": metrics.node_temperatures}
    
    def _analyze_thermal_flow_internal(self, graph: nx.DiGraph) -> ThermalMetrics:
        """Implementación interna del análisis térmico."""
        if graph.number_of_nodes() == 0:
            return ThermalMetrics.empty()
        
        cfg = self.config.thermal
        node_temps: Dict[str, float] = {}
        node_costs: Dict[str, float] = {}
        
        # 1. Asignar temperatura a hojas (Insumos)
        for node, data in graph.nodes(data=True):
            if data.get("type") == NodeType.INSUMO.value:
                desc = str(data.get("description", ""))
                tipo = str(data.get("tipo_insumo", ""))
                node_temps[node] = cfg.get_base_temperature(desc, tipo)
                
                # Costo acumulado
                total = sum(
                    graph[pred][node].get("total_cost", 0.0)
                    for pred in graph.predecessors(node)
                )
                node_costs[node] = max(total, data.get("unit_cost", 0.0))
            else:
                node_temps[node] = 0.0
                node_costs[node] = 0.0
        
        # 2. Propagación Bottom-Up
        try:
            topo_order = list(reversed(list(nx.topological_sort(graph))))
        except nx.NetworkXUnfeasible:
            # Fallback para grafos con ciclos
            topo_order = sorted(
                graph.nodes(),
                key=lambda n: graph.nodes[n].get("level", 0),
                reverse=True,
            )
        
        for node in topo_order:
            if graph.nodes[node].get("type") == NodeType.INSUMO.value:
                continue
            
            children = list(graph.successors(node))
            if not children:
                node_temps[node] = cfg.temperature_default * 0.5
                continue
            
            weighted_sum = 0.0
            cost_sum = 0.0
            
            for child in children:
                edge = graph[node][child]
                w = edge.get("total_cost", edge.get("weight", 0.0))
                if w == 0.0:
                    w = node_costs.get(child, 1.0)
                
                t = node_temps.get(child, cfg.temperature_default)
                weighted_sum += t * w
                cost_sum += w
            
            if cost_sum > 0:
                node_temps[node] = weighted_sum / cost_sum
                node_costs[node] = cost_sum
            else:
                node_temps[node] = cfg.temperature_default * 0.5
        
        # 3. Temperatura del Sistema (ponderada por costo)
        total_project_cost = 0.0
        weighted_temp_sum = 0.0
        
        for node in graph.nodes():
            if graph.nodes[node].get("type") == NodeType.APU.value:
                cost = node_costs.get(node, 0.0)
                temp = node_temps.get(node, 0.0)
                weighted_temp_sum += cost * temp
                total_project_cost += cost
        
        if total_project_cost > 0:
            system_temp = weighted_temp_sum / total_project_cost
        else:
            system_temp = cfg.temperature_ambient
        
        # Clasificación y hotspots
        risk_level = RiskLevel.from_temperature(system_temp)
        
        hotspots = sorted(
            [
                {"id": n, "temp": round(t, 2)}
                for n, t in node_temps.items()
                if t > cfg.hotspot_threshold
            ],
            key=lambda x: x["temp"],
            reverse=True,
        )
        
        return ThermalMetrics(
            system_temperature=round(system_temp, 2),
            thermal_risk_level=risk_level,
            hotspots=hotspots,
            node_temperatures=node_temps,
        )

    # ========================================================================
    # CONVECCIÓN INFLACIONARIA
    # ========================================================================

    def analyze_inflationary_convection(
        self,
        graph: nx.DiGraph,
        fluid_nodes: List[str],
    ) -> Dict[str, Any]:
        """
        Analiza el contagio inflacionario por convección.
        
        Args:
            graph: Grafo del presupuesto.
            fluid_nodes: Nodos que actúan como fluido (ej. Transporte).
            
        Returns:
            Impacto convectivo.
        """
        convection_impact: Dict[str, float] = {}
        affected_nodes: List[str] = []
        fluid_set = set(fluid_nodes)
        
        for node in graph.nodes():
            if graph.nodes[node].get("type") != NodeType.APU.value:
                continue
            
            total_cost = 0.0
            fluid_cost = 0.0
            
            for succ in graph.successors(node):
                edge = graph[node][succ]
                cost = edge.get("total_cost", edge.get("weight", 0.0))
                total_cost += cost
                
                if succ in fluid_set:
                    fluid_cost += cost
            
            if total_cost > 0:
                impact = fluid_cost / total_cost
                if impact > 0:
                    convection_impact[node] = impact
                    if impact > 0.2:  # Umbral de alto riesgo
                        affected_nodes.append(node)
        
        return {
            "affected_nodes_count": len(affected_nodes),
            "high_risk_nodes": affected_nodes,
            "convection_impact": convection_impact,
        }

    # ========================================================================
    # CLASIFICACIÓN DE ANOMALÍAS
    # ========================================================================

    def _classify_anomalous_nodes(self, graph: nx.DiGraph) -> AnomalyClassification:
        """Clasifica nodos anómalos."""
        isolated: List[Dict[str, Any]] = []
        orphan_insumos: List[Dict[str, Any]] = []
        empty_apus: List[Dict[str, Any]] = []
        
        for n, d in graph.nodes(data=True):
            if d.get("type") == NodeType.ROOT.value:
                continue
            
            in_deg = graph.in_degree(n)
            out_deg = graph.out_degree(n)
            node_type = d.get("type")
            
            info = dict(d)
            info.update({"id": n, "in_degree": in_deg, "out_degree": out_deg})
            
            if in_deg == 0 and out_deg == 0:
                isolated.append(info)
                if node_type == NodeType.INSUMO.value:
                    orphan_insumos.append(info)
            elif node_type == NodeType.INSUMO.value and in_deg == 0:
                orphan_insumos.append(info)
            elif node_type == NodeType.APU.value and out_deg == 0:
                empty_apus.append(info)
        
        return AnomalyClassification(
            isolated_nodes=isolated,
            orphan_insumos=orphan_insumos,
            empty_apus=empty_apus,
        )

    # ========================================================================
    # RECURSOS CRÍTICOS
    # ========================================================================

    def _identify_critical_resources(
        self,
        graph: nx.DiGraph,
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Identifica recursos críticos por grado de entrada."""
        if top_n is None:
            top_n = self.config.top_critical_resources
        
        resources = []
        for n, d in graph.nodes(data=True):
            if d.get("type") == NodeType.INSUMO.value:
                in_degree = graph.in_degree(n)
                if in_degree > 0:
                    resources.append({
                        "id": n,
                        "in_degree": in_degree,
                        "description": d.get("description", ""),
                    })
        
        resources.sort(key=lambda x: x["in_degree"], reverse=True)
        return resources[:top_n]

    # ========================================================================
    # REPORTE EJECUTIVO
    # ========================================================================

    def generate_executive_report(
        self,
        graph: nx.DiGraph,
        financial_metrics: Optional[Dict[str, Any]] = None,
    ) -> ConstructionRiskReport:
        """
        Genera el Reporte Ejecutivo con Scoring Bayesiano.
        
        Integra métricas topológicas, espectrales, térmicas y financieras.
        
        Args:
            graph: Grafo del proyecto.
            financial_metrics: Métricas financieras externas.
            
        Returns:
            Informe consolidado.
        """
        cfg = self.config.scoring
        
        # Calcular todas las métricas
        betti = ValidatedBettiNumbers.from_graph(graph)
        metrics = betti.to_metrics()
        raw_cycles, truncated = self._get_raw_cycles(graph)
        synergy = self._detect_risk_synergy_internal(graph, raw_cycles)
        pyramid_stability = self.calculate_pyramid_stability(graph)
        spectral = self._analyze_spectral_stability_internal(graph)
        thermal = self._analyze_thermal_flow_internal(graph)
        anomalies = self._classify_anomalous_nodes(graph)
        critical_resources = self._identify_critical_resources(graph)
        
        # Calcular score
        euler_factor = self.calculate_euler_efficiency(graph)
        density = nx.density(graph) if graph.number_of_nodes() > 0 else 0
        density_factor = 1.0 - min(0.5, density)
        
        base_score = 100.0 * (
            cfg.weight_euler * euler_factor
            + cfg.weight_stability * pyramid_stability
            + cfg.weight_density * density_factor
            + cfg.weight_base
        )
        
        # Penalizaciones
        penalty = 0.0
        if betti.beta_1 > 0:
            penalty += min(cfg.penalty_max_cycles, betti.beta_1 * cfg.penalty_per_cycle)
        if synergy.synergy_detected:
            penalty += min(cfg.penalty_max_synergy, synergy.synergy_score * cfg.penalty_synergy_factor)
        if spectral.resonance_risk:
            penalty += cfg.penalty_resonance
        
        integrity_score = max(0.0, min(100.0, base_score - penalty))
        complexity_level = ComplexityLevel.from_integrity_score(integrity_score)
        
        # Alertas y riesgos
        waste_alerts = []
        if anomalies.isolated_nodes:
            waste_alerts.append(
                f"🚨 {len(anomalies.isolated_nodes)} nodo(s) aislado(s)."
            )
        
        circular_risks = []
        if betti.beta_1 > 0:
            circular_risks.append(f"🚨 {betti.beta_1} ciclo(s) detectado(s).")
        if synergy.synergy_detected:
            circular_risks.append(f"☣️ Riesgo Sistémico: {synergy.risk_level.value}")
        
        # Narrativa
        narrative = self._generate_strategic_narrative(
            metrics, synergy, pyramid_stability, thermal, spectral
        )
        
        # Conectividad
        is_dag = nx.is_directed_acyclic_graph(graph)
        
        return ConstructionRiskReport(
            integrity_score=round(integrity_score, 1),
            waste_alerts=waste_alerts,
            circular_risks=circular_risks,
            complexity_level=complexity_level,
            financial_risk_level="Pendiente",
            strategic_narrative=narrative,
            details={
                "metrics": asdict(metrics),
                "euler_efficiency": euler_factor,
                "pyramid_stability": pyramid_stability,
                "synergy_risk": synergy.to_dict(),
                "anomalies": anomalies.to_dict(),
                "thermal": thermal.to_dict(),
                "spectral": spectral.to_dict(),
                "spectral_analysis": spectral.to_dict(),  # Alias V3.0
                "density": density,
                "connectivity": {"is_dag": is_dag},
                "cycles": [" -> ".join(c) for c in raw_cycles[:5]],
                "raw_cycles": raw_cycles,
                "critical_resources": critical_resources,
                "convection_risk": {"high_risk_nodes": []},
            },
        )

    def _generate_strategic_narrative(
        self,
        metrics: TopologicalMetrics,
        synergy: SynergyMetrics,
        stability: float,
        thermal: ThermalMetrics,
        spectral: SpectralMetrics,
    ) -> str:
        """Genera la narrativa estratégica."""
        sections = []
        
        # 1. Diagnóstico Estructural
        if stability > 0.7:
            sections.append("🏗️ **ESTRUCTURA SÓLIDA**: Base robusta.")
        elif stability > 0.4:
            sections.append("⚠️ **ESTRUCTURA MODERADA**: Oportunidades de mejora.")
        else:
            sections.append("🚨 **PIRÁMIDE INVERTIDA**: Cimentación insuficiente.")
        
        # 2. Integridad Lógica
        if metrics.beta_1 > 0:
            sections.append(f"🔄 **COMPLEJIDAD CÍCLICA**: {metrics.beta_1} ciclo(s).")
        else:
            sections.append("✅ **TRAZABILIDAD LIMPIA**: Flujo acíclico.")
        
        # 3. Sinergia de Riesgo
        if synergy.synergy_detected:
            sections.append(
                f"☣️ **SINERGIA DE RIESGO**: Nivel {synergy.risk_level.value}."
            )
        
        # 4. Riesgo Térmico
        if thermal.thermal_risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            sections.append(
                f"🌡️ **FIEBRE INFLACIONARIA**: {thermal.system_temperature}°."
            )
        
        # 5. Resonancia
        if spectral.resonance_risk:
            sections.append("📡 **RIESGO DE RESONANCIA**: Vulnerabilidad sistémica.")
        
        return " ".join(sections)

    # ========================================================================
    # MÉTODOS DE COMPATIBILIDAD LEGACY
    # ========================================================================

    def analyze_structural_integrity(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Analiza la integridad estructural (compatibilidad V2).
        
        Wrapper que retorna estructura plana de métricas.
        """
        report = self.generate_executive_report(graph)
        metrics = report.details["metrics"]
        
        result = {
            "business.betti_b0": metrics["beta_0"],
            "business.betti_b1": metrics["beta_1"],
            "business.euler_characteristic": metrics["euler_characteristic"],
            "business.cycles_count": metrics["beta_1"],
            "business.is_dag": 1 if report.details["connectivity"]["is_dag"] else 0,
            "business.isolated_count": len(report.details["anomalies"]["isolated_nodes"]),
            "business.empty_apus_count": len(report.details["anomalies"]["empty_apus"]),
            "integrity_score": report.integrity_score,
            "details": report.details,
        }
        
        # Estructura adicional para tests
        result["details"]["critical_resources"] = self._identify_critical_resources(graph)
        result["details"]["graph_summary"] = {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
        }
        result["details"]["topology"] = {"betti_numbers": metrics}
        
        if self.telemetry:
            self.telemetry.record_metric("business_topology.analysis_complete", 1)
        
        return result

    def get_audit_report(self, analysis_result: Dict[str, Any]) -> List[str]:
        """
        Genera reporte de auditoría en formato lista (Legacy).
        """
        lines = [
            "=== AUDITORIA ESTRUCTURAL ===",
            f"Integridad: {analysis_result.get('integrity_score', 'N/A')}",
            f"Ciclos de Costo: {analysis_result.get('business.betti_b1', 0)}",
            f"Eficiencia de Euler: {analysis_result.get('details', {}).get('euler_efficiency', 0.0):.4f}",
        ]
        
        if analysis_result.get("business.betti_b1", 0) > 0:
            lines.append("ALERTA: Estructura Circular Detectada (CRITICAS)")
        
        if analysis_result.get("business.isolated_count", 0) > 0:
            lines.append("ADVERTENCIA: Nodos aislados detectados (desperdicio potencial)")
        
        lines.append("Puntuacion de Integridad: Calculada")
        
        return lines

    def _detect_cycles(self, graph: nx.DiGraph) -> Tuple[List[str], bool]:
        """Legacy wrapper para detección de ciclos."""
        raw, truncated = self._get_raw_cycles(graph)
        formatted = [" -> ".join(map(str, c + [c[0]])) for c in raw]
        return formatted, truncated

    def _compute_connectivity_analysis(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Análisis de conectividad (Legacy)."""
        non_trivial_scc = [
            list(c) for c in nx.strongly_connected_components(graph) if len(c) > 1
        ]
        return {
            "is_dag": nx.is_directed_acyclic_graph(graph),
            "num_wcc": nx.number_weakly_connected_components(graph),
            "is_weakly_connected": nx.is_weakly_connected(graph),
            "num_non_trivial_scc": len(non_trivial_scc),
            "non_trivial_scc": non_trivial_scc,
        }

    def _interpret_topology(self, metrics: TopologicalMetrics) -> Dict[str, str]:
        """Interpreta métricas topológicas (Legacy)."""
        return {
            "beta_0": (
                f"{metrics.beta_0} componente(s) conexo(s)"
                if metrics.beta_0 == 1
                else f"{metrics.beta_0} componentes desconexas"
            ),
            "beta_1": (
                f"{metrics.beta_1} ciclo(s)"
                if metrics.beta_1 > 0
                else "Acíclico (Sin ciclos)"
            ),
        }

    def materialize_structure(
        self,
        df_presupuesto: pd.DataFrame,
        df_apus_detail: Optional[pd.DataFrame] = None,
    ) -> nx.DiGraph:
        """
        Materializa la estructura topológica del negocio.
        
        Args:
            df_presupuesto: Estructura del presupuesto.
            df_apus_detail: Detalle de recursos/insumos.
            
        Returns:
            Grafo del negocio.
        """
        self.logger.info("🧠 STRATEGY: Materializando estructura topológica...")
        builder = BudgetGraphBuilder()
        return builder.build(df_presupuesto, apus_detail_df=df_apus_detail)


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================


def create_analyzer(
    config: Optional[TopologyAnalyzerConfig] = None,
    telemetry: Optional[TelemetryContext] = None,
) -> BusinessTopologicalAnalyzer:
    """Factory function para crear un analizador configurado."""
    return BusinessTopologicalAnalyzer(config=config, telemetry=telemetry)


def analyze_budget_topology(
    presupuesto_df: pd.DataFrame,
    apus_detail_df: Optional[pd.DataFrame] = None,
    config: Optional[TopologyAnalyzerConfig] = None,
) -> ConstructionRiskReport:
    """
    Función de conveniencia para análisis rápido.
    
    Args:
        presupuesto_df: DataFrame del presupuesto.
        apus_detail_df: DataFrame de detalle de APUs.
        config: Configuración opcional.
        
    Returns:
        Reporte de riesgos de construcción.
    """
    analyzer = create_analyzer(config=config)
    graph = analyzer.materialize_structure(presupuesto_df, apus_detail_df)
    return analyzer.generate_executive_report(graph)


def verify_topology_invariants(graph: nx.DiGraph) -> Dict[str, bool]:
    """
    Verifica invariantes topológicos del grafo.
    
    Returns:
        Diccionario con resultados de verificación.
    """
    results = {}
    
    try:
        betti = ValidatedBettiNumbers.from_graph(graph)
        results["euler_consistent"] = True
        results["betti_non_negative"] = (
            betti.beta_0 >= 0 and betti.beta_1 >= 0 and betti.beta_2 >= 0
        )
        results["connected_if_beta0_one"] = (
            betti.beta_0 != 1 or nx.is_weakly_connected(graph)
        )
    except TopologyInvariantError:
        results["euler_consistent"] = False
    except Exception as e:
        results["error"] = str(e)
    
    return results