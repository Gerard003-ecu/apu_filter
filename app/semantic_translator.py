"""
Módulo: Semantic Translator (El Intérprete Diplomático del Consejo)
===================================================================

Este componente actúa como el "Puente Cognitivo" entre la Matemática Profunda
(Topología, Finanzas, Termodinámica) y la Toma de Decisiones Ejecutiva.
Su función no es reportar datos, sino emitir "Veredictos" y narrativas de riesgo
basadas en la evidencia técnica recolectada por el resto del Consejo.

Fundamentos Teóricos y Arquitectura Algebraica:
-----------------------------------------------

1. Retículo de Veredictos (Lattice Theory):
   Implementa una estructura algebraica de orden $(Verdict, \le, \sqcup, \sqcap)$ donde:
   - $VIABLE (\bot)$ < $REVISAR$ < $INVIABLE (\top)$.
   - La operación de síntesis es el Supremo ($\sqcup$ o Join), adoptando siempre el
     criterio más conservador (Worst-Case Scenario) entre los agentes.
   - Propiedades verificadas: Idempotencia, Conmutatividad, Asociatividad, Absorción.

2. Invariantes Topológicos:
   - Característica de Euler: χ = β₀ - β₁ + β₂ (verificado en construcción)
   - Conectividad algebraica: λ₂ (valor de Fiedler) > 0 implica grafo conexo
   - Índice de estabilidad piramidal: Ψ = Σweights(base) / Σweights(apex)

3. Síntesis DIKW (Data → Wisdom):
   Eleva los datos crudos a Sabiduría mediante la integración de contextos:
   - Traduce β₁ > 0 (Topología) → "Socavón Lógico" (Bloqueo administrativo).
   - Traduce Ψ < 1.0 (Física) → "Pirámide Invertida" (Riesgo de colapso logístico).
   - Traduce T_sys > 50°C (Termodinámica) → "Fiebre Inflacionaria".

4. Narrativa Generativa Causal (GraphRAG):
   Utiliza Retrieval-Augmented Generation sobre Grafos para explicar la causalidad.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, Enum, auto
from functools import lru_cache, total_ordering
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    FrozenSet,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

import networkx as nx

# Importaciones del proyecto
from app.telemetry_schemas import (
    PhysicsMetrics,
    TopologicalMetrics,
    ControlMetrics,
    ThermodynamicMetrics,
)
from app.tools_interface import MICRegistry, register_core_vectors

try:
    from app.schemas import Stratum
except ImportError:
    from enum import IntEnum as StratumBase

    class Stratum(StratumBase):
        WISDOM = 0
        STRATEGY = 1
        TACTICS = 2
        PHYSICS = 3


logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES Y TIPOS
# ============================================================================

T = TypeVar("T")

# Constantes físicas
ABSOLUTE_ZERO_CELSIUS: Final[float] = -273.15
KELVIN_OFFSET: Final[float] = 273.15

# Tolerancias numéricas
EPSILON: Final[float] = 1e-9


# ============================================================================
# SISTEMA DE UNIDADES PARA TEMPERATURA
# ============================================================================


@dataclass(frozen=True, order=True)
class Temperature:
    """
    Representa temperatura con conversión automática entre unidades.
    
    Almacena internamente en Kelvin para evitar valores negativos absolutos.
    Implementa orden total para comparaciones.
    
    Invariante: kelvin >= 0 (por definición termodinámica)
    """
    
    kelvin: float
    
    def __post_init__(self) -> None:
        if self.kelvin < 0:
            raise ValueError(
                f"Temperature cannot be below absolute zero: {self.kelvin}K"
            )
    
    @classmethod
    def from_celsius(cls, celsius: float) -> Temperature:
        """Construye desde grados Celsius."""
        return cls(kelvin=celsius + KELVIN_OFFSET)
    
    @classmethod
    def from_kelvin(cls, kelvin: float) -> Temperature:
        """Construye desde Kelvin."""
        return cls(kelvin=kelvin)
    
    @property
    def celsius(self) -> float:
        """Convierte a Celsius."""
        return self.kelvin - KELVIN_OFFSET
    
    @property
    def is_absolute_zero(self) -> bool:
        """Verifica si es cero absoluto."""
        return abs(self.kelvin) < EPSILON
    
    def __str__(self) -> str:
        return f"{self.celsius:.1f}°C ({self.kelvin:.1f}K)"


# ============================================================================
# EXCEPCIONES ESPECÍFICAS DEL DOMINIO
# ============================================================================


class SemanticTranslatorError(Exception):
    """Excepción base del traductor semántico."""
    pass


class TopologyInvariantViolation(SemanticTranslatorError):
    """Violación de invariante topológico."""
    
    def __init__(self, message: str, beta_0: int, beta_1: int, beta_2: int, chi: int):
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.chi = chi
        expected_chi = beta_0 - beta_1 + beta_2
        super().__init__(
            f"{message}. χ={chi}, pero β₀-β₁+β₂={expected_chi} "
            f"(β₀={beta_0}, β₁={beta_1}, β₂={beta_2})"
        )


class LatticeViolation(SemanticTranslatorError):
    """Violación de propiedad del lattice."""
    pass


class MetricsValidationError(SemanticTranslatorError):
    """Error en validación de métricas."""
    pass


# ============================================================================
# CONFIGURACIÓN CENTRALIZADA CON VALIDACIÓN
# ============================================================================


@dataclass(frozen=True)
class StabilityThresholds:
    """
    Umbrales para interpretación del índice de estabilidad piramidal (Ψ).

    Fundamentación topológica:
    - Ψ < critical: Pirámide Invertida (Cimentación insuficiente)
    - Ψ ∈ [critical, warning): Estructura Isostática (Equilibrio precario)
    - Ψ ∈ [warning, solid): Estabilidad moderada
    - Ψ ≥ solid: Estructura Antisísmica (Base robusta con redundancia)

    Invariante de orden: 0 ≤ critical < warning < solid
    """

    critical: float = 1.0
    warning: float = 3.0
    solid: float = 10.0

    def __post_init__(self) -> None:
        """Valida invariantes de los umbrales."""
        if self.critical < 0:
            raise ValueError("critical threshold must be non-negative")
        if not (self.critical < self.warning < self.solid):
            raise ValueError(
                f"Thresholds must satisfy: critical < warning < solid. "
                f"Got: {self.critical} < {self.warning} < {self.solid}"
            )

    def classify(self, stability: float) -> str:
        """
        Clasifica un valor de estabilidad.
        
        Returns:
            Clasificación: 'invalid', 'critical', 'warning', 'stable', 'robust'
        """
        if stability < 0 or not math.isfinite(stability):
            return "invalid"
        if stability < self.critical:
            return "critical"
        if stability < self.warning:
            return "warning"
        if stability < self.solid:
            return "stable"
        return "robust"
    
    def severity_score(self, stability: float) -> float:
        """
        Calcula un score de severidad normalizado [0, 1].
        
        0 = robusto, 1 = crítico
        """
        if stability < 0:
            return 1.0
        if stability >= self.solid:
            return 0.0
        # Interpolación lineal inversa
        return 1.0 - min(1.0, stability / self.solid)


@dataclass(frozen=True)
class TopologicalThresholds:
    """
    Umbrales para interpretación de números de Betti.

    Fundamentación Algebraica:
    - β₀ (componentes conexos): Mide fragmentación del grafo
    - β₁ (ciclos independientes): Mide "agujeros" lógicos (genus)
    - χ (característica de Euler): Invariante topológico χ = β₀ - β₁ + β₂

    Invariantes:
    - β₀ ≥ 1 para proyectos no vacíos
    - β₁ ≥ 0 (ciclos no pueden ser negativos)
    - cycles_optimal ≤ cycles_warning ≤ cycles_critical
    """

    connected_components_optimal: int = 1
    cycles_optimal: int = 0
    cycles_warning: int = 1
    cycles_critical: int = 3
    max_fragmentation: int = 5
    
    # Umbrales espectrales
    fiedler_connected_threshold: float = 0.01  # λ₂ > ε implica conexo
    fiedler_robust_threshold: float = 0.5      # Alta conectividad algebraica

    def __post_init__(self) -> None:
        """Valida invariantes."""
        if self.connected_components_optimal < 1:
            raise ValueError("optimal connected components must be >= 1")
        if self.cycles_optimal < 0:
            raise ValueError("optimal cycles must be non-negative")
        if not (self.cycles_optimal <= self.cycles_warning <= self.cycles_critical):
            raise ValueError(
                f"cycle thresholds must be ordered: "
                f"{self.cycles_optimal} ≤ {self.cycles_warning} ≤ {self.cycles_critical}"
            )
        if self.fiedler_connected_threshold < 0:
            raise ValueError("Fiedler threshold must be non-negative")

    def classify_connectivity(self, beta_0: int) -> str:
        """Clasifica nivel de conectividad."""
        if beta_0 <= 0:
            return "empty"
        if beta_0 == self.connected_components_optimal:
            return "unified"
        if beta_0 <= self.max_fragmentation:
            return "fragmented"
        return "severely_fragmented"

    def classify_cycles(self, beta_1: int) -> str:
        """Clasifica nivel de ciclos."""
        if beta_1 < 0:
            return "invalid"
        if beta_1 <= self.cycles_optimal:
            return "clean"
        if beta_1 <= self.cycles_warning:
            return "minor"
        if beta_1 <= self.cycles_critical:
            return "moderate"
        return "critical"
    
    def classify_spectral_connectivity(self, fiedler_value: float) -> str:
        """Clasifica conectividad usando valor de Fiedler."""
        if fiedler_value < self.fiedler_connected_threshold:
            return "disconnected"
        if fiedler_value < self.fiedler_robust_threshold:
            return "weakly_connected"
        return "strongly_connected"

    def validate_euler_characteristic(
        self, beta_0: int, beta_1: int, beta_2: int, chi: int
    ) -> bool:
        """
        Valida el invariante de Euler: χ = β₀ - β₁ + β₂
        
        Para grafos (2-complejos simpliciales), β₂ suele ser 0.
        """
        expected_chi = beta_0 - beta_1 + beta_2
        return chi == expected_chi


@dataclass(frozen=True)
class ThermalThresholds:
    """
    Umbrales para métricas termodinámicas (en Celsius).

    Metáfora: El proyecto como sistema termodinámico
    - Temperatura: Volatilidad/Inflación de precios
    - Entropía: Desorden administrativo
    - Exergía: Eficiencia de inversión
    
    Invariantes:
    - temperature_cold < temperature_warm < temperature_hot < temperature_critical
    - 0 ≤ entropy_low < entropy_high ≤ 1
    - 0 ≤ exergy_poor < exergy_efficient ≤ 1
    """

    temperature_cold: float = 20.0
    temperature_warm: float = 35.0
    temperature_hot: float = 50.0
    temperature_critical: float = 75.0

    entropy_low: float = 0.3
    entropy_high: float = 0.7
    entropy_death: float = 0.95  # Umbral de muerte térmica

    exergy_efficient: float = 0.7
    exergy_poor: float = 0.3
    
    # Capacidad calorífica mínima aceptable
    heat_capacity_minimum: float = 0.2

    def __post_init__(self) -> None:
        """Valida invariantes."""
        temps = [
            self.temperature_cold,
            self.temperature_warm,
            self.temperature_hot,
            self.temperature_critical,
        ]
        if temps != sorted(temps):
            raise ValueError("Temperature thresholds must be strictly increasing")
        
        if not (0 <= self.entropy_low < self.entropy_high <= 1):
            raise ValueError("Entropy thresholds must satisfy: 0 ≤ low < high ≤ 1")
        
        if not (0 <= self.exergy_poor < self.exergy_efficient <= 1):
            raise ValueError("Exergy thresholds must satisfy: 0 ≤ poor < efficient ≤ 1")

    def classify_temperature(self, temp_celsius: float) -> str:
        """Clasifica temperatura del sistema."""
        if not math.isfinite(temp_celsius):
            return "invalid"
        if temp_celsius < ABSOLUTE_ZERO_CELSIUS:
            return "invalid"
        if temp_celsius <= self.temperature_cold:
            return "cold"
        if temp_celsius <= self.temperature_warm:
            return "stable"
        if temp_celsius <= self.temperature_hot:
            return "warm"
        if temp_celsius <= self.temperature_critical:
            return "hot"
        return "critical"
    
    def classify_entropy(self, entropy: float) -> str:
        """Clasifica nivel de entropía."""
        if not (0 <= entropy <= 1):
            return "invalid"
        if entropy >= self.entropy_death:
            return "death"
        if entropy >= self.entropy_high:
            return "high"
        if entropy <= self.entropy_low:
            return "low"
        return "moderate"
    
    def classify_exergy(self, exergy: float) -> str:
        """Clasifica eficiencia exergética."""
        if not (0 <= exergy <= 1):
            return "invalid"
        if exergy >= self.exergy_efficient:
            return "efficient"
        if exergy <= self.exergy_poor:
            return "poor"
        return "moderate"


@dataclass(frozen=True)
class FinancialThresholds:
    """
    Umbrales para métricas financieras.
    
    WACC: Costo de Capital Ponderado
    PI: Índice de Rentabilidad (Profitability Index)
    """

    wacc_low: float = 0.05
    wacc_moderate: float = 0.10
    wacc_high: float = 0.15

    profitability_excellent: float = 1.5
    profitability_good: float = 1.2
    profitability_marginal: float = 1.0

    contingency_minimal: float = 0.05
    contingency_standard: float = 0.10
    contingency_high: float = 0.20
    
    def classify_profitability(self, pi: float) -> str:
        """Clasifica índice de rentabilidad."""
        if pi >= self.profitability_excellent:
            return "excellent"
        if pi >= self.profitability_good:
            return "good"
        if pi >= self.profitability_marginal:
            return "marginal"
        return "poor"


@dataclass(frozen=True)
class TranslatorConfig:
    """
    Configuración consolidada del traductor semántico.

    Agrupa todos los umbrales y configuraciones en una única estructura.
    """

    stability: StabilityThresholds = field(default_factory=StabilityThresholds)
    topology: TopologicalThresholds = field(default_factory=TopologicalThresholds)
    thermal: ThermalThresholds = field(default_factory=ThermalThresholds)
    financial: FinancialThresholds = field(default_factory=FinancialThresholds)

    # Límites de procesamiento
    max_cycle_path_display: int = 5
    max_stress_points_display: int = 3
    max_narrative_length: int = 10000

    # Configuración de determinismo
    deterministic_market: bool = True
    default_market_index: int = 0
    
    # Validación de invariantes topológicos
    strict_euler_validation: bool = True


# ============================================================================
# LATTICE DE VEREDICTOS (ESTRUCTURA ALGEBRAICA)
# ============================================================================


class VerdictLevel(IntEnum):
    """
    Lattice de veredictos con orden total.

    Estructura algebraica: (VerdictLevel, ≤, ⊔, ⊓)
    
    Propiedades del Lattice (verificadas en tests):
    1. Idempotencia: a ⊔ a = a, a ⊓ a = a
    2. Conmutatividad: a ⊔ b = b ⊔ a, a ⊓ b = b ⊓ a
    3. Asociatividad: (a ⊔ b) ⊔ c = a ⊔ (b ⊔ c)
    4. Absorción: a ⊔ (a ⊓ b) = a, a ⊓ (a ⊔ b) = a
    5. Identidad: a ⊔ ⊥ = a, a ⊓ ⊤ = a

    Semántica del orden (menor es mejor):
    - VIABLE (⊥): Proyecto puede proceder sin restricciones
    - CONDICIONAL: Viable con modificaciones menores
    - REVISAR: Requiere análisis adicional antes de decidir
    - PRECAUCION: Riesgos significativos identificados
    - RECHAZAR (⊤): No viable en estado actual
    """

    VIABLE = 0          # ⊥ - Bottom (mejor caso)
    CONDICIONAL = 1     # Viable con condiciones
    REVISAR = 2         # Necesita más análisis
    PRECAUCION = 3      # Advertencia significativa
    RECHAZAR = 4        # ⊤ - Top (peor caso)

    @classmethod
    def bottom(cls) -> VerdictLevel:
        """Elemento mínimo del lattice (⊥)."""
        return cls.VIABLE

    @classmethod
    def top(cls) -> VerdictLevel:
        """Elemento máximo del lattice (⊤)."""
        return cls.RECHAZAR

    @classmethod
    def supremum(cls, *levels: VerdictLevel) -> VerdictLevel:
        """
        Operación JOIN (⊔): supremo del conjunto.
        
        Semántica: toma el peor caso (criterio conservador).
        
        Propiedad: ⊔∅ = ⊥ (el supremo del vacío es el bottom)
        """
        if not levels:
            return cls.bottom()
        return cls(max(level.value for level in levels))

    @classmethod
    def infimum(cls, *levels: VerdictLevel) -> VerdictLevel:
        """
        Operación MEET (⊓): ínfimo del conjunto.
        
        Semántica: toma el mejor caso.
        
        Propiedad: ⊓∅ = ⊤ (el ínfimo del vacío es el top)
        """
        if not levels:
            return cls.top()
        return cls(min(level.value for level in levels))

    def join(self, other: VerdictLevel) -> VerdictLevel:
        """Join binario: self ⊔ other."""
        return VerdictLevel.supremum(self, other)

    def meet(self, other: VerdictLevel) -> VerdictLevel:
        """Meet binario: self ⊓ other."""
        return VerdictLevel.infimum(self, other)

    def __or__(self, other: VerdictLevel) -> VerdictLevel:
        """Sintaxis: a | b = a ⊔ b (join)."""
        return self.join(other)

    def __and__(self, other: VerdictLevel) -> VerdictLevel:
        """Sintaxis: a & b = a ⊓ b (meet)."""
        return self.meet(other)
    
    def __le__(self, other: VerdictLevel) -> bool:
        """Orden del lattice: a ≤ b sii a ⊔ b = b."""
        return self.value <= other.value

    @property
    def emoji(self) -> str:
        """Representación visual del veredicto."""
        return {
            VerdictLevel.VIABLE: "✅",
            VerdictLevel.CONDICIONAL: "🔵",
            VerdictLevel.REVISAR: "🔍",
            VerdictLevel.PRECAUCION: "⚠️",
            VerdictLevel.RECHAZAR: "🛑",
        }[self]
    
    @property
    def description(self) -> str:
        """Descripción textual del veredicto."""
        return {
            VerdictLevel.VIABLE: "Proyecto viable sin restricciones",
            VerdictLevel.CONDICIONAL: "Viable con condiciones",
            VerdictLevel.REVISAR: "Requiere análisis adicional",
            VerdictLevel.PRECAUCION: "Riesgos significativos",
            VerdictLevel.RECHAZAR: "No viable",
        }[self]

    @property
    def is_positive(self) -> bool:
        """Indica si el veredicto permite proceder."""
        return self.value <= VerdictLevel.CONDICIONAL.value

    @property
    def is_negative(self) -> bool:
        """Indica si el veredicto bloquea el proyecto."""
        return self == VerdictLevel.RECHAZAR
    
    @property
    def requires_attention(self) -> bool:
        """Indica si requiere atención inmediata."""
        return self.value >= VerdictLevel.PRECAUCION.value
    
    @classmethod
    def verify_lattice_laws(cls) -> Dict[str, bool]:
        """
        Verifica las leyes del lattice para todos los elementos.
        
        Útil para testing y debugging.
        """
        all_elements = list(cls)
        results = {}
        
        # Idempotencia
        results["idempotent_join"] = all(a | a == a for a in all_elements)
        results["idempotent_meet"] = all(a & a == a for a in all_elements)
        
        # Conmutatividad
        results["commutative_join"] = all(
            a | b == b | a for a in all_elements for b in all_elements
        )
        results["commutative_meet"] = all(
            a & b == b & a for a in all_elements for b in all_elements
        )
        
        # Asociatividad
        results["associative_join"] = all(
            (a | b) | c == a | (b | c)
            for a in all_elements for b in all_elements for c in all_elements
        )
        results["associative_meet"] = all(
            (a & b) & c == a & (b & c)
            for a in all_elements for b in all_elements for c in all_elements
        )
        
        # Absorción
        results["absorption_join"] = all(
            a | (a & b) == a for a in all_elements for b in all_elements
        )
        results["absorption_meet"] = all(
            a & (a | b) == a for a in all_elements for b in all_elements
        )
        
        # Identidad
        results["identity_join_bottom"] = all(a | cls.bottom() == a for a in all_elements)
        results["identity_meet_top"] = all(a & cls.top() == a for a in all_elements)
        
        return results


class FinancialVerdict(Enum):
    """
    Veredictos financieros específicos.

    Mapea a VerdictLevel para integración con el lattice.
    """

    ACCEPT = "ACEPTAR"
    CONDITIONAL = "CONDICIONAL"
    REVIEW = "REVISAR"
    REJECT = "RECHAZAR"

    def to_verdict_level(self) -> VerdictLevel:
        """Homomorfismo al lattice de veredictos."""
        mapping = {
            FinancialVerdict.ACCEPT: VerdictLevel.VIABLE,
            FinancialVerdict.CONDITIONAL: VerdictLevel.CONDICIONAL,
            FinancialVerdict.REVIEW: VerdictLevel.REVISAR,
            FinancialVerdict.REJECT: VerdictLevel.RECHAZAR,
        }
        return mapping.get(self, VerdictLevel.REVISAR)

    @classmethod
    def from_string(cls, value: str) -> FinancialVerdict:
        """Parsea desde string con normalización."""
        if not value:
            return cls.REVIEW
        normalized = value.upper().strip()
        for verdict in cls:
            if verdict.value.upper() == normalized or verdict.name == normalized:
                return verdict
        return cls.REVIEW


# ============================================================================
# ESTRUCTURAS DE DATOS VALIDADAS
# ============================================================================


@runtime_checkable
class HasBettiNumbers(Protocol):
    """Protocolo para objetos con números de Betti."""

    @property
    def beta_0(self) -> int: ...

    @property
    def beta_1(self) -> int: ...


@dataclass(frozen=True)
class ValidatedTopology:
    """
    Métricas topológicas validadas con invariante de Euler.
    
    Invariante: euler_characteristic == beta_0 - beta_1 + beta_2
    """
    
    beta_0: int
    beta_1: int
    beta_2: int
    euler_characteristic: int
    fiedler_value: float
    spectral_gap: float
    pyramid_stability: float
    structural_entropy: float
    
    def __post_init__(self) -> None:
        """Valida invariantes topológicos."""
        if self.beta_0 < 0 or self.beta_1 < 0 or self.beta_2 < 0:
            raise ValueError("Betti numbers must be non-negative")
        
        expected_chi = self.beta_0 - self.beta_1 + self.beta_2
        if self.euler_characteristic != expected_chi:
            raise TopologyInvariantViolation(
                "Euler characteristic inconsistency",
                self.beta_0,
                self.beta_1,
                self.beta_2,
                self.euler_characteristic,
            )
    
    @classmethod
    def from_metrics(
        cls,
        metrics: Union[TopologicalMetrics, Dict[str, Any]],
        strict: bool = False,
    ) -> ValidatedTopology:
        """
        Construye desde métricas, con validación opcional.
        
        Si strict=False, corrige la característica de Euler automáticamente.
        """
        if isinstance(metrics, dict):
            beta_0 = int(metrics.get("beta_0", 1))
            beta_1 = int(metrics.get("beta_1", 0))
            beta_2 = int(metrics.get("beta_2", 0))
            chi = int(metrics.get("euler_characteristic", beta_0 - beta_1 + beta_2))
            fiedler = float(metrics.get("fiedler_value", 1.0))
            gap = float(metrics.get("spectral_gap", 0.0))
            stability = float(metrics.get("pyramid_stability", 1.0))
            entropy = float(metrics.get("structural_entropy", 0.0))
        else:
            beta_0 = metrics.beta_0
            beta_1 = metrics.beta_1
            beta_2 = metrics.beta_2
            chi = metrics.euler_characteristic
            fiedler = metrics.fiedler_value
            gap = metrics.spectral_gap
            stability = metrics.pyramid_stability
            entropy = metrics.structural_entropy
        
        # Corregir Euler si no es estricto
        expected_chi = beta_0 - beta_1 + beta_2
        if not strict and chi != expected_chi:
            logger.warning(
                f"Correcting Euler characteristic: {chi} → {expected_chi}"
            )
            chi = expected_chi
        
        return cls(
            beta_0=beta_0,
            beta_1=beta_1,
            beta_2=beta_2,
            euler_characteristic=chi,
            fiedler_value=fiedler,
            spectral_gap=gap,
            pyramid_stability=stability,
            structural_entropy=entropy,
        )
    
    @property
    def is_connected(self) -> bool:
        """Verifica si el grafo es conexo (β₀ = 1)."""
        return self.beta_0 == 1
    
    @property
    def has_cycles(self) -> bool:
        """Verifica si hay ciclos (β₁ > 0)."""
        return self.beta_1 > 0
    
    @property
    def genus(self) -> int:
        """Género topológico (número de 'agujeros')."""
        return self.beta_1


@dataclass
class StratumAnalysisResult:
    """
    Resultado del análisis de un estrato DIKW.

    Integra con la jerarquía:
    - PHYSICS (Datos)
    - TACTICS (Información)
    - STRATEGY (Conocimiento)
    - WISDOM (Sabiduría)
    """

    stratum: Stratum
    verdict: VerdictLevel
    narrative: str
    metrics_summary: Dict[str, Any]
    issues: List[str] = field(default_factory=list)
    confidence: float = 1.0  # Nivel de confianza en el análisis

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "stratum": self.stratum.name,
            "verdict": self.verdict.name,
            "verdict_emoji": self.verdict.emoji,
            "narrative": self.narrative,
            "metrics": self.metrics_summary,
            "issues": self.issues,
            "confidence": self.confidence,
        }
    
    @property
    def severity_score(self) -> float:
        """Score de severidad normalizado [0, 1]."""
        return self.verdict.value / VerdictLevel.RECHAZAR.value


@dataclass
class StrategicReport:
    """
    Reporte estratégico completo.

    Representa el juicio final integrado del Consejo.
    """

    title: str
    verdict: VerdictLevel
    executive_summary: str
    strata_analysis: Dict[Stratum, StratumAnalysisResult]
    recommendations: List[str]
    raw_narrative: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    confidence: float = 1.0

    @property
    def is_viable(self) -> bool:
        """Indica si el proyecto es viable (veredicto positivo)."""
        return self.verdict.is_positive
    
    @property
    def requires_immediate_action(self) -> bool:
        """Indica si se requiere acción inmediata."""
        return self.verdict.requires_attention

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario para API/persistencia."""
        return {
            "title": self.title,
            "verdict": self.verdict.name,
            "verdict_emoji": self.verdict.emoji,
            "verdict_description": self.verdict.description,
            "is_viable": self.is_viable,
            "requires_action": self.requires_immediate_action,
            "executive_summary": self.executive_summary,
            "strata_analysis": {
                s.name: a.to_dict()
                for s, a in self.strata_analysis.items()
            },
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
        }


# ============================================================================
# CACHÉ DE NARRATIVAS
# ============================================================================


class NarrativeCache:
    """
    Caché thread-safe para narrativas generadas.
    
    Evita regenerar narrativas idénticas.
    """
    
    def __init__(self, maxsize: int = 256):
        self._maxsize = maxsize
        self._cache: Dict[str, str] = {}
        self._access_order: List[str] = []
    
    def _make_key(self, domain: str, classification: str, params: Dict[str, Any]) -> str:
        """Genera clave única para la narrativa."""
        # Ordenar params para consistencia
        sorted_params = tuple(sorted(params.items())) if params else ()
        return f"{domain}:{classification}:{hash(sorted_params)}"
    
    def get(
        self,
        domain: str,
        classification: str,
        params: Dict[str, Any],
    ) -> Optional[str]:
        """Obtiene narrativa del caché si existe."""
        key = self._make_key(domain, classification, params)
        if key in self._cache:
            # Mover al final (LRU)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def put(
        self,
        domain: str,
        classification: str,
        params: Dict[str, Any],
        narrative: str,
    ) -> None:
        """Almacena narrativa en caché."""
        key = self._make_key(domain, classification, params)
        
        # Evicción LRU si es necesario
        while len(self._cache) >= self._maxsize:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        
        self._cache[key] = narrative
        self._access_order.append(key)
    
    def clear(self) -> None:
        """Limpia el caché."""
        self._cache.clear()
        self._access_order.clear()


# ============================================================================
# TRADUCTOR SEMÁNTICO PRINCIPAL
# ============================================================================


class SemanticTranslator:
    """
    Traductor semántico que convierte métricas técnicas en narrativa de ingeniería.

    Interpreta el presupuesto como una estructura física donde:
    - Insumos = Cimentación de Recursos (Nivel 3 - PHYSICS)
    - APUs = Cuerpo Táctico (Nivel 2 - TACTICS)
    - Capítulos = Pilares Estructurales (Nivel 1 - STRATEGY)
    - Proyecto = Ápice / Objetivo Final (Nivel 0 - WISDOM)

    El traductor implementa un Funtor F: MetricSpace → NarrativeSpace
    que preserva el orden de severidad.
    """

    def __init__(
        self,
        config: Optional[TranslatorConfig] = None,
        market_provider: Optional[Callable[[], str]] = None,
        mic: Optional[MICRegistry] = None,
        enable_cache: bool = True,
    ) -> None:
        """
        Inicializa el traductor.

        Args:
            config: Configuración consolidada de umbrales
            market_provider: Proveedor de contexto de mercado (inyección de dependencia)
            mic: Registro MIC para obtener narrativas
            enable_cache: Habilitar caché de narrativas
        """
        self.config = config or TranslatorConfig()
        self._market_provider = market_provider
        self._cache = NarrativeCache() if enable_cache else None

        if mic:
            self.mic = mic
        else:
            self.mic = MICRegistry()
            register_core_vectors(self.mic, config={})

        logger.debug(
            f"SemanticTranslator initialized | "
            f"Ψ_critical={self.config.stability.critical:.2f}, "
            f"deterministic={self.config.deterministic_market}, "
            f"cache_enabled={enable_cache}"
        )

    # ========================================================================
    # HELPERS INTERNOS
    # ========================================================================

    def _fetch_narrative(
        self,
        domain: str,
        classification: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Obtiene narrativa del MIC con caché opcional.
        
        Args:
            domain: Dominio de la narrativa (TOPOLOGY_CYCLES, THERMAL_TEMPERATURE, etc.)
            classification: Clasificación específica (clean, critical, etc.)
            params: Parámetros para interpolación
            
        Returns:
            Narrativa formateada
        """
        params = params or {}
        
        # Intentar caché primero
        if self._cache:
            cached = self._cache.get(domain, classification, params)
            if cached:
                return cached
        
        # Obtener del MIC
        response = self.mic.project_intent(
            "fetch_narrative",
            {
                "domain": domain,
                "classification": classification,
                "params": params,
            },
            {"force_physics_override": True},
        )
        
        narrative = response.get("narrative", f"[{domain}.{classification}]")
        
        # Almacenar en caché
        if self._cache:
            self._cache.put(domain, classification, params, narrative)
        
        return narrative

    def _normalize_temperature(
        self,
        value: float,
        assume_kelvin_if_high: bool = True,
    ) -> Temperature:
        """
        Normaliza un valor de temperatura a objeto Temperature.
        
        Heurística: Si el valor > 100, probablemente es Kelvin.
        """
        if assume_kelvin_if_high and value > 100:
            return Temperature.from_kelvin(value)
        return Temperature.from_celsius(value)

    def _safe_extract_numeric(
        self,
        data: Dict[str, Any],
        key: str,
        default: float = 0.0,
    ) -> float:
        """Extrae valor numérico de forma segura."""
        value = data.get(key)
        if value is None:
            return default
        if isinstance(value, (int, float)):
            if math.isnan(value) or math.isinf(value):
                return default
            return float(value)
        return default

    def _safe_extract_nested(
        self,
        data: Dict[str, Any],
        path: List[str],
        default: float = 0.0,
    ) -> float:
        """Extrae valor numérico de path anidado."""
        current: Any = data
        for key in path:
            if not isinstance(current, dict):
                return default
            current = current.get(key)
            if current is None:
                return default
        if isinstance(current, (int, float)):
            if math.isnan(current) or math.isinf(current):
                return default
            return float(current)
        return default

    # ========================================================================
    # GRAPHRAG: EXPLICACIÓN CAUSAL
    # ========================================================================

    def explain_cycle_path(self, cycle_nodes: List[str]) -> str:
        """
        Genera narrativa explicando la ruta del ciclo (GraphRAG).
        
        El ciclo se presenta con cierre explícito: A → B → ... → A
        """
        if not cycle_nodes:
            return ""

        max_display = self.config.max_cycle_path_display
        
        # Preparar nodos para mostrar
        display_nodes = list(cycle_nodes[:max_display])
        
        # Indicar truncamiento si aplica
        if len(cycle_nodes) > max_display:
            remaining = len(cycle_nodes) - max_display
            display_nodes.append(f"... (+{remaining} nodos)")
        
        # Cerrar el ciclo si no está cerrado
        if cycle_nodes and display_nodes and display_nodes[-1] != cycle_nodes[0]:
            if not display_nodes[-1].startswith("..."):
                display_nodes.append(cycle_nodes[0])

        payload = {
            "anomaly_type": "CYCLE",
            "path_nodes": display_nodes,
            "total_length": len(cycle_nodes),
        }

        response = self.mic.project_intent(
            "project_graph_narrative",
            payload,
            {"force_physics_override": True},
        )

        return response.get("narrative", "") if response.get("success") else ""

    def explain_stress_point(
        self,
        node_id: str,
        degree: Union[int, str],
        stratum: Stratum = Stratum.PHYSICS,
    ) -> str:
        """
        Explica por qué un nodo es punto de estrés crítico (GraphRAG).
        
        Args:
            node_id: Identificador del nodo
            degree: Grado del nodo (puede ser numérico o descriptivo)
            stratum: Estrato donde se encuentra el nodo
        """
        # Normalizar grado
        if isinstance(degree, int):
            safe_degree = degree
        elif isinstance(degree, str):
            if degree.isdigit():
                safe_degree = int(degree)
            else:
                # Descriptores cualitativos → valor alto
                safe_degree = 10
        else:
            safe_degree = 0

        payload = {
            "anomaly_type": "STRESS",
            "vector": {
                "node_id": node_id,
                "node_type": "INSUMO",
                "stratum": stratum.value,
                "in_degree": safe_degree,
                "out_degree": 0,
            },
        }
        
        response = self.mic.project_intent(
            "project_graph_narrative",
            payload,
            {"force_physics_override": True},
        )
        
        return response.get("narrative", "") if response.get("success") else ""

    # ========================================================================
    # TRADUCCIÓN DE TOPOLOGÍA
    # ========================================================================

    def translate_topology(
        self,
        metrics: Union[TopologicalMetrics, Dict[str, Any], ValidatedTopology],
        stability: float = 0.0,
        synergy_risk: Optional[Dict[str, Any]] = None,
        spectral: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, VerdictLevel]:
        """
        Traduce métricas topológicas a narrativa de ingeniería.
        
        Análisis:
        1. β₁ (ciclos): Genus estructural / "socavones"
        2. Sinergia de riesgo: Ciclos que comparten recursos críticos
        3. Espectral: Conectividad algebraica (Fiedler)
        4. β₀ (componentes): Coherencia de obra
        5. Ψ (estabilidad): Solidez de cimentación
        
        Returns:
            Tupla (narrativa, veredicto)
        """
        # Normalizar a ValidatedTopology
        if isinstance(metrics, ValidatedTopology):
            topo = metrics
        else:
            topo = ValidatedTopology.from_metrics(
                metrics,
                strict=self.config.strict_euler_validation,
            )

        synergy = synergy_risk or {}
        spec = spectral or {}

        # Usar stability de topo si no se pasa explícitamente
        eff_stability = stability if stability != 0.0 else topo.pyramid_stability
        
        # Validar estabilidad
        if eff_stability < 0:
            raise MetricsValidationError("Stability Ψ must be non-negative")

        narrative_parts: List[str] = []
        verdicts: List[VerdictLevel] = []

        # 1. β₁: Ciclos / Genus Estructural
        cycle_narrative, cycle_verdict = self._translate_cycles(topo.beta_1)
        narrative_parts.append(cycle_narrative)
        verdicts.append(cycle_verdict)

        # 2. Sinergia de Riesgo (Producto Cup)
        synergy_detected = bool(synergy.get("synergy_detected", False))
        if synergy_detected:
            synergy_narrative = self._translate_synergy(synergy)
            narrative_parts.append(synergy_narrative)
            verdicts.append(VerdictLevel.RECHAZAR)

            # GraphRAG: Explicar nodos puente
            bridge_nodes = synergy.get("bridge_nodes", [])
            if bridge_nodes:
                first_bridge = bridge_nodes[0]
                node_id = (
                    first_bridge.get("id")
                    if isinstance(first_bridge, dict)
                    else str(first_bridge)
                )
                explanation = self.explain_stress_point(node_id, "múltiples")
                if explanation:
                    narrative_parts.append(explanation)

        # 3. Análisis Espectral
        fiedler = topo.fiedler_value
        resonance_risk = bool(spec.get("resonance_risk", False))
        wavelength = float(spec.get("wavelength", 0.0))

        if fiedler > 0 or resonance_risk:
            spec_narrative = self._translate_spectral(fiedler, wavelength, resonance_risk)
            narrative_parts.append(spec_narrative)
            if resonance_risk:
                verdicts.append(VerdictLevel.PRECAUCION)

        # 4. β₀: Coherencia / Conectividad
        conn_narrative, conn_verdict = self._translate_connectivity(topo.beta_0)
        narrative_parts.append(conn_narrative)
        verdicts.append(conn_verdict)

        # 5. Ψ: Solidez de Cimentación
        stab_narrative, stab_verdict = self._translate_stability(eff_stability)
        narrative_parts.append(stab_narrative)
        verdicts.append(stab_verdict)

        # Veredicto final: supremum (peor caso)
        final_verdict = VerdictLevel.supremum(*verdicts)

        return "\n".join(narrative_parts), final_verdict

    def _translate_cycles(self, beta_1: int) -> Tuple[str, VerdictLevel]:
        """Traduce β₁ (ciclos) a narrativa."""
        classification = self.config.topology.classify_cycles(beta_1)
        
        if classification == "invalid":
            return "⚠️ **Valor de ciclos inválido**", VerdictLevel.REVISAR
        
        narrative = self._fetch_narrative(
            "TOPOLOGY_CYCLES",
            classification,
            {"beta_1": beta_1},
        )

        verdict_map = {
            "clean": VerdictLevel.VIABLE,
            "minor": VerdictLevel.CONDICIONAL,
            "moderate": VerdictLevel.PRECAUCION,
            "critical": VerdictLevel.RECHAZAR,
        }

        return narrative, verdict_map.get(classification, VerdictLevel.REVISAR)

    def _translate_connectivity(self, beta_0: int) -> Tuple[str, VerdictLevel]:
        """Traduce β₀ (componentes conexos) a narrativa."""
        classification = self.config.topology.classify_connectivity(beta_0)
        narrative = self._fetch_narrative(
            "TOPOLOGY_CONNECTIVITY",
            classification,
            {"beta_0": beta_0},
        )

        verdict_map = {
            "empty": VerdictLevel.RECHAZAR,
            "unified": VerdictLevel.VIABLE,
            "fragmented": VerdictLevel.CONDICIONAL,
            "severely_fragmented": VerdictLevel.PRECAUCION,
        }

        return narrative, verdict_map.get(classification, VerdictLevel.REVISAR)

    def _translate_stability(self, stability: float) -> Tuple[str, VerdictLevel]:
        """Traduce Ψ (índice de estabilidad piramidal) a narrativa."""
        classification = self.config.stability.classify(stability)

        if classification == "invalid":
            return "⚠️ **Valor de estabilidad inválido**", VerdictLevel.REVISAR

        narrative = self._fetch_narrative(
            "STABILITY",
            classification,
            {"stability": round(stability, 2)},
        )

        verdict_map = {
            "critical": VerdictLevel.RECHAZAR,
            "warning": VerdictLevel.PRECAUCION,
            "stable": VerdictLevel.CONDICIONAL,
            "robust": VerdictLevel.VIABLE,
        }

        return narrative, verdict_map.get(classification, VerdictLevel.REVISAR)

    def _translate_synergy(self, synergy: Dict[str, Any]) -> str:
        """Traduce sinergia de riesgo a narrativa."""
        count = int(synergy.get("intersecting_cycles_count", 0))
        return self._fetch_narrative("MISC", "SYNERGY", {"count": count})

    def _translate_spectral(
        self,
        fiedler: float,
        wavelength: float,
        resonance_risk: bool,
    ) -> str:
        """Traduce métricas espectrales a narrativa."""
        parts = []

        # Cohesión (valor de Fiedler = conectividad algebraica)
        cohesion_class = self.config.topology.classify_spectral_connectivity(fiedler)
        cohesion_map = {
            "strongly_connected": "high",
            "weakly_connected": "standard",
            "disconnected": "low",
        }
        cohesion_type = cohesion_map.get(cohesion_class, "standard")
        
        parts.append(
            self._fetch_narrative(
                "SPECTRAL_COHESION",
                cohesion_type,
                {"fiedler": round(fiedler, 4)},
            )
        )

        # Resonancia
        resonance_type = "risk" if resonance_risk else "safe"
        parts.append(
            self._fetch_narrative(
                "SPECTRAL_RESONANCE",
                resonance_type,
                {"wavelength": round(wavelength, 4)},
            )
        )

        return " ".join(parts)

    # ========================================================================
    # TRADUCCIÓN DE TERMODINÁMICA
    # ========================================================================

    def translate_thermodynamics(
        self,
        metrics: Union[ThermodynamicMetrics, Dict[str, Any]],
    ) -> Tuple[str, VerdictLevel]:
        """
        Traduce métricas termodinámicas a narrativa.
        
        Metáforas:
        - Temperatura → Volatilidad/Inflación
        - Entropía → Desorden administrativo
        - Exergía → Eficiencia de inversión
        - Capacidad calorífica → Inercia financiera
        """
        # Normalizar inputs
        if isinstance(metrics, dict):
            temp_raw = float(metrics.get(
                "system_temperature",
                metrics.get("temperature", 298.15),  # Default: 25°C en Kelvin
            ))
            thermo = ThermodynamicMetrics(
                system_temperature=temp_raw,
                entropy=float(metrics.get("entropy", 0.0)),
                heat_capacity=float(metrics.get("heat_capacity", 0.5)),
            )
        elif isinstance(metrics, ThermodynamicMetrics):
            thermo = metrics
        else:
            thermo = ThermodynamicMetrics()

        # Normalizar temperatura
        temp = self._normalize_temperature(thermo.system_temperature)
        temp_celsius = temp.celsius

        # Normalizar valores a [0, 1]
        entropy = max(0.0, min(1.0, thermo.entropy))
        exergy = max(0.0, min(1.0, thermo.exergetic_efficiency))

        parts = []
        verdicts = []

        # 1. Eficiencia Exergética
        exergy_pct = exergy * 100.0
        exergy_class = self.config.thermal.classify_exergy(exergy)
        parts.append(f"⚡ **Eficiencia Exergética del {exergy_pct:.1f}%**.")

        exergy_verdict_map = {
            "efficient": VerdictLevel.VIABLE,
            "moderate": VerdictLevel.CONDICIONAL,
            "poor": VerdictLevel.PRECAUCION,
            "invalid": VerdictLevel.REVISAR,
        }
        verdicts.append(exergy_verdict_map.get(exergy_class, VerdictLevel.CONDICIONAL))

        # 2. Entropía
        entropy_class = self.config.thermal.classify_entropy(entropy)
        if entropy_class == "death":
            parts.append(self._fetch_narrative("MISC", "THERMAL_DEATH"))
            verdicts.append(VerdictLevel.RECHAZAR)
        elif entropy_class == "high":
            parts.append(
                self._fetch_narrative(
                    "THERMAL_ENTROPY",
                    "high",
                    {"entropy": round(entropy, 2)},
                )
            )
            verdicts.append(VerdictLevel.PRECAUCION)
        elif entropy_class == "low":
            parts.append(
                self._fetch_narrative(
                    "THERMAL_ENTROPY",
                    "low",
                    {"entropy": round(entropy, 2)},
                )
            )

        # 3. Temperatura
        temp_class = self.config.thermal.classify_temperature(temp_celsius)
        parts.append(
            self._fetch_narrative(
                "THERMAL_TEMPERATURE",
                temp_class,
                {"temperature": round(temp_celsius, 1)},
            )
        )

        temp_verdict_map = {
            "cold": VerdictLevel.VIABLE,
            "stable": VerdictLevel.VIABLE,
            "warm": VerdictLevel.CONDICIONAL,
            "hot": VerdictLevel.PRECAUCION,
            "critical": VerdictLevel.RECHAZAR,
            "invalid": VerdictLevel.REVISAR,
        }
        verdicts.append(temp_verdict_map.get(temp_class, VerdictLevel.REVISAR))

        # Receta para fiebre
        if temp_class in ("hot", "critical"):
            parts.append(
                "💊 **Receta**: Se recomienda enfriar mediante contratos de futuros "
                "o stock preventivo."
            )

        # 4. Inercia (Capacidad Calorífica)
        if thermo.heat_capacity < self.config.thermal.heat_capacity_minimum:
            parts.append(
                f"🍂 **Hoja al Viento**: Baja inercia financiera "
                f"(C_v={thermo.heat_capacity:.2f}). Riesgo de volatilidad extrema."
            )
            verdicts.append(VerdictLevel.PRECAUCION)

        final_verdict = VerdictLevel.supremum(*verdicts)
        return " ".join(parts), final_verdict

    # ========================================================================
    # TRADUCCIÓN FINANCIERA
    # ========================================================================

    def translate_financial(
        self,
        metrics: Dict[str, Any],
    ) -> Tuple[str, VerdictLevel, FinancialVerdict]:
        """
        Traduce métricas financieras a narrativa.
        
        Returns:
            Tupla (narrativa, veredicto_general, veredicto_financiero)
        """
        validated = self._validate_financial_metrics(metrics)
        parts = []

        # WACC
        parts.append(
            self._fetch_narrative(
                "MISC",
                "WACC",
                {"wacc": validated["wacc"]},
            )
        )

        # Contingencia
        parts.append(
            self._fetch_narrative(
                "MISC",
                "CONTINGENCY",
                {"contingency": validated["contingency_recommended"]},
            )
        )

        # Veredicto financiero
        fin_verdict = validated["recommendation"]
        pi = validated["profitability_index"]

        verdict_key = fin_verdict.name.lower()
        parts.append(
            self._fetch_narrative(
                "FINANCIAL_VERDICT",
                verdict_key,
                {"pi": round(pi, 3)},
            )
        )

        # Mapear a VerdictLevel (homomorfismo)
        general_verdict = fin_verdict.to_verdict_level()

        return "\n".join(parts), general_verdict, fin_verdict

    def _validate_financial_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Valida y normaliza métricas financieras."""
        if not isinstance(metrics, dict):
            raise MetricsValidationError(
                f"Expected dict for financial metrics, got {type(metrics).__name__}"
            )

        def extract_verdict(data: Dict) -> FinancialVerdict:
            performance = data.get("performance", {})
            if not isinstance(performance, dict):
                return FinancialVerdict.REVIEW
            rec = performance.get("recommendation", "REVISAR")
            return FinancialVerdict.from_string(str(rec))

        return {
            "wacc": self._safe_extract_numeric(metrics, "wacc", 0.0),
            "contingency_recommended": self._safe_extract_nested(
                metrics,
                ["contingency", "recommended"],
                0.0,
            ),
            "recommendation": extract_verdict(metrics),
            "profitability_index": self._safe_extract_nested(
                metrics,
                ["performance", "profitability_index"],
                0.0,
            ),
        }

    # ========================================================================
    # CONTEXTO DE MERCADO
    # ========================================================================

    def get_market_context(self) -> str:
        """
        Obtiene inteligencia de mercado desde el proveedor inyectado o MIC.
        """
        if self._market_provider:
            try:
                context = self._market_provider()
                return f"🌍 **Suelo de Mercado**: {context}"
            except Exception as e:
                logger.warning(f"Market provider failed: {e}")
                return "🌍 **Suelo de Mercado**: No disponible."

        # Fallback a MIC
        params = {
            "deterministic": self.config.deterministic_market,
            "index": self.config.default_market_index,
        }
        response = self.mic.project_intent(
            "fetch_narrative",
            {"domain": "MARKET_CONTEXT", "params": params},
            {},
        )
        context = response.get("narrative", "Datos de mercado no disponibles.")

        return f"🌍 **Suelo de Mercado**: {context}"

    # ========================================================================
    # ANÁLISIS POR ESTRATO DIKW
    # ========================================================================

    def _analyze_physics_stratum(
        self,
        thermal: ThermodynamicMetrics,
        stability: float,
        physics: Optional[PhysicsMetrics] = None,
        control: Optional[ControlMetrics] = None,
    ) -> StratumAnalysisResult:
        """
        Analiza el estrato PHYSICS (datos, flujo, temperatura).
        
        Incluye:
        - Giroscopía (FluxCondenser)
        - Oráculo de Laplace (Control)
        - Análisis térmico
        - Dinámica de bombeo
        """
        issues = []
        verdicts = []
        narrative_parts = []

        # 1. Giroscopía
        if physics:
            if physics.gyroscopic_stability < 0.3:
                issues.append("Nutación crítica detectada")
                narrative_parts.append(
                    self._fetch_narrative("GYROSCOPIC_STABILITY", "nutation")
                )
                verdicts.append(VerdictLevel.RECHAZAR)
            elif physics.gyroscopic_stability < 0.7:
                issues.append("Precesión detectada")
                narrative_parts.append(
                    self._fetch_narrative("GYROSCOPIC_STABILITY", "precession")
                )
                verdicts.append(VerdictLevel.PRECAUCION)
            else:
                narrative_parts.append(
                    self._fetch_narrative("GYROSCOPIC_STABILITY", "stable")
                )
                verdicts.append(VerdictLevel.VIABLE)

        # 2. Oráculo de Laplace (Control)
        if control:
            if not control.is_stable:
                issues.append("Divergencia Matemática (RHP)")
                narrative_parts.append(
                    self._fetch_narrative("LAPLACE_CONTROL", "unstable")
                )
                verdicts.append(VerdictLevel.RECHAZAR)
            elif control.phase_margin_deg < 30:
                issues.append("Estabilidad Marginal (Resonancia)")
                narrative_parts.append(
                    self._fetch_narrative("LAPLACE_CONTROL", "marginal")
                )
                verdicts.append(VerdictLevel.PRECAUCION)
            else:
                verdicts.append(VerdictLevel.VIABLE)

        # 3. Análisis térmico
        temp = self._normalize_temperature(thermal.system_temperature)
        temp_class = self.config.thermal.classify_temperature(temp.celsius)
        
        if temp_class == "critical":
            issues.append(f"Temperatura crítica: {temp}")
            verdicts.append(VerdictLevel.RECHAZAR)
        elif temp_class == "hot":
            issues.append(f"Temperatura elevada: {temp}")
            verdicts.append(VerdictLevel.PRECAUCION)
        else:
            verdicts.append(VerdictLevel.VIABLE)

        # 4. Entropía (Muerte Térmica)
        entropy_class = self.config.thermal.classify_entropy(thermal.entropy)
        if entropy_class == "death":
            issues.append("Muerte Térmica del Sistema")
            narrative_parts.append(self._fetch_narrative("MISC", "THERMAL_DEATH"))
            verdicts.append(VerdictLevel.RECHAZAR)
        elif entropy_class == "high":
            issues.append(f"Alta entropía: {thermal.entropy:.2f}")
            verdicts.append(VerdictLevel.PRECAUCION)

        # 5. Dinámica de Bombeo
        if physics:
            if physics.pressure > 0.7:
                issues.append(f"Inestabilidad de Tubería (P={physics.pressure:.2f})")
                narrative_parts.append(
                    self._fetch_narrative(
                        "PUMP_DYNAMICS",
                        "water_hammer",
                        {"pressure": physics.pressure},
                    )
                )
                verdicts.append(VerdictLevel.PRECAUCION)

            narrative_parts.append(
                self._fetch_narrative(
                    "PUMP_DYNAMICS",
                    "accumulator_pressure",
                    {"pressure": physics.saturation * 100.0},
                )
            )

        verdict = VerdictLevel.supremum(*verdicts) if verdicts else VerdictLevel.VIABLE

        # Narrativa base según veredicto
        if verdict == VerdictLevel.VIABLE:
            base_narrative = "Base física estable. Flujo de datos sin turbulencia."
        elif verdict == VerdictLevel.RECHAZAR:
            base_narrative = "Inestabilidad física crítica. Datos no confiables."
        else:
            base_narrative = "Señales de inestabilidad en la capa física."

        full_narrative = f"{base_narrative} {' '.join(narrative_parts)}".strip()

        metrics_summary: Dict[str, Any] = {
            "temperature": thermal.system_temperature,
            "temperature_celsius": temp.celsius,
            "entropy": thermal.entropy,
            "exergy": thermal.exergy,
            "stability": stability,
        }
        if physics:
            metrics_summary["gyroscopic_stability"] = physics.gyroscopic_stability
            metrics_summary["pressure"] = physics.pressure
        if control:
            metrics_summary["phase_margin"] = control.phase_margin_deg
            metrics_summary["is_stable"] = control.is_stable

        return StratumAnalysisResult(
            stratum=Stratum.PHYSICS,
            verdict=verdict,
            narrative=full_narrative,
            metrics_summary=metrics_summary,
            issues=issues,
        )

    def _analyze_tactics_stratum(
        self,
        topo: ValidatedTopology,
        synergy: Dict[str, Any],
        spectral: Dict[str, Any],
        stability: float,
    ) -> StratumAnalysisResult:
        """Analiza el estrato TACTICS (estructura topológica)."""
        issues = []
        verdicts = []

        # Ciclos
        cycle_class = self.config.topology.classify_cycles(topo.beta_1)
        if cycle_class != "clean":
            issues.append(f"Ciclos detectados (β₁={topo.beta_1})")
            verdict_map = {
                "critical": VerdictLevel.RECHAZAR,
                "moderate": VerdictLevel.PRECAUCION,
                "minor": VerdictLevel.CONDICIONAL,
            }
            verdicts.append(verdict_map.get(cycle_class, VerdictLevel.CONDICIONAL))
        else:
            verdicts.append(VerdictLevel.VIABLE)

        # Conectividad
        conn_class = self.config.topology.classify_connectivity(topo.beta_0)
        if conn_class != "unified":
            issues.append(f"Fragmentación (β₀={topo.beta_0})")
            verdicts.append(VerdictLevel.CONDICIONAL)

        # Sinergia de riesgo
        synergy_detected = bool(synergy.get("synergy_detected", False))
        if synergy_detected:
            issues.append("Sinergia de riesgo (Efecto Dominó)")
            verdicts.append(VerdictLevel.RECHAZAR)

        # Estabilidad piramidal
        stab_class = self.config.stability.classify(stability)
        if stab_class == "critical":
            issues.append(f"Pirámide Invertida (Ψ={stability:.2f})")
            verdicts.append(VerdictLevel.RECHAZAR)
        elif stab_class == "warning":
            issues.append(f"Estabilidad marginal (Ψ={stability:.2f})")
            verdicts.append(VerdictLevel.PRECAUCION)

        verdict = VerdictLevel.supremum(*verdicts) if verdicts else VerdictLevel.VIABLE

        if verdict == VerdictLevel.VIABLE:
            narrative = "Estructura topológicamente sólida y conexa."
        elif verdict == VerdictLevel.RECHAZAR:
            narrative = "Estructura comprometida. Reparaciones necesarias antes de proceder."
        else:
            narrative = "Estructura con defectos menores a corregir."

        return StratumAnalysisResult(
            stratum=Stratum.TACTICS,
            verdict=verdict,
            narrative=narrative,
            metrics_summary={
                "beta_0": topo.beta_0,
                "beta_1": topo.beta_1,
                "beta_2": topo.beta_2,
                "euler_characteristic": topo.euler_characteristic,
                "fiedler_value": topo.fiedler_value,
                "stability": stability,
                "synergy_risk": synergy_detected,
            },
            issues=issues,
        )

    def _analyze_strategy_stratum(
        self,
        financial_metrics: Dict[str, Any],
    ) -> StratumAnalysisResult:
        """Analiza el estrato STRATEGY (viabilidad financiera)."""
        try:
            validated = self._validate_financial_metrics(financial_metrics)
            fin_verdict = validated["recommendation"]
            verdict = fin_verdict.to_verdict_level()

            issues = []
            if verdict.is_negative:
                issues.append("Proyecto financieramente inviable")
            elif verdict == VerdictLevel.REVISAR:
                issues.append("Métricas financieras inconclusas")

            if verdict.is_positive:
                narrative = "Modelo financiero viable y robusto."
            elif verdict == VerdictLevel.REVISAR:
                narrative = "Viabilidad financiera indeterminada. Requiere análisis adicional."
            else:
                narrative = "Proyecto no es financieramente viable en condiciones actuales."

            return StratumAnalysisResult(
                stratum=Stratum.STRATEGY,
                verdict=verdict,
                narrative=narrative,
                metrics_summary={
                    "wacc": validated["wacc"],
                    "profitability_index": validated["profitability_index"],
                    "recommendation": fin_verdict.value,
                },
                issues=issues,
            )

        except Exception as e:
            logger.error(f"Error analyzing strategy stratum: {e}")
            return StratumAnalysisResult(
                stratum=Stratum.STRATEGY,
                verdict=VerdictLevel.REVISAR,
                narrative=f"Error en análisis financiero: {e}",
                metrics_summary={},
                issues=[str(e)],
                confidence=0.5,
            )

    def _analyze_wisdom_stratum(
        self,
        topo: ValidatedTopology,
        financial_metrics: Dict[str, Any],
        stability: float,
        synergy: Dict[str, Any],
        final_verdict: VerdictLevel,
        has_errors: bool,
    ) -> StratumAnalysisResult:
        """Genera el análisis del estrato WISDOM (veredicto integrado)."""
        narrative = self._generate_final_advice(
            topo,
            financial_metrics,
            stability,
            synergy,
            has_errors,
        )

        confidence = 0.7 if has_errors else 1.0

        return StratumAnalysisResult(
            stratum=Stratum.WISDOM,
            verdict=final_verdict,
            narrative=narrative,
            metrics_summary={
                "final_verdict": final_verdict.name,
                "is_viable": final_verdict.is_positive,
                "requires_action": final_verdict.requires_attention,
            },
            issues=["Errores en análisis previo"] if has_errors else [],
            confidence=confidence,
        )

    # ========================================================================
    # GENERACIÓN DE NARRATIVAS COMPUESTAS
    # ========================================================================

    def _generate_report_header(self) -> str:
        """Genera el encabezado del reporte."""
        return (
            "# 🏗️ INFORME DE INGENIERÍA ESTRATÉGICA\n\n"
            f"*Análisis de Coherencia Fractal | "
            f"Estabilidad Crítica: Ψ < {self.config.stability.critical}*\n"
        )

    def _generate_integrated_diagnosis(
        self,
        stability: float,
        temperature_celsius: float,
        physics_verdict: VerdictLevel,
    ) -> str:
        """Genera diagnóstico integrado (edificio + calor)."""
        parts = []

        # Evaluación de la base
        stab_class = self.config.stability.classify(stability)
        
        if stab_class in ("critical", "warning"):
            parts.append(
                f"El edificio se apoya sobre una **base inestable** (Ψ={stability:.2f}). "
                "La cimentación de recursos es insuficiente para la carga táctica."
            )
        else:
            parts.append(
                f"El edificio tiene una **cimentación sólida** (Ψ={stability:.2f}). "
                "La base de recursos es amplia y redundante."
            )

        # Evaluación del clima
        temp_class = self.config.thermal.classify_temperature(temperature_celsius)
        
        if temp_class in ("hot", "critical"):
            parts.append(
                f"Sin embargo, el entorno es hostil. Detectamos una "
                f"**Fiebre Inflacionaria** de {temperature_celsius:.1f}°C "
                "entrando por los insumos (volatilidad de precios)."
            )

            # Interacción base + calor (disipación)
            if stab_class == "robust":
                parts.append(
                    "✅ Gracias a la base ancha, la estructura actúa como un "
                    "**disipador de calor** eficiente. El riesgo de sobrecostos "
                    "se diluye en la red de proveedores resiliente."
                )
            elif stab_class in ("critical", "warning"):
                parts.append(
                    "🚨 Debido a la base estrecha (Pirámide Invertida), "
                    "**el calor no se disipa**. Se concentra en los pocos puntos "
                    "de apoyo, creando un riesgo crítico de fractura financiera."
                )
            else:
                parts.append(
                    "⚠️ La estructura tiene capacidad moderada de disipación, "
                    "pero una ola de calor sostenida podría comprometer "
                    "la integridad financiera."
                )
        else:
            parts.append(
                f"El entorno es térmicamente estable ({temperature_celsius:.1f}°C). "
                "El riesgo inflacionario externo es bajo."
            )

        return " ".join(parts)

    def _generate_final_advice(
        self,
        topo: ValidatedTopology,
        financial_metrics: Dict[str, Any],
        stability: float,
        synergy: Dict[str, Any],
        has_errors: bool,
    ) -> str:
        """Genera el dictamen final del Ingeniero Jefe."""
        if has_errors:
            return self._fetch_narrative("FINAL_VERDICTS", "analysis_failed")

        # Sinergia de riesgo (máxima prioridad)
        synergy_detected = bool(synergy.get("synergy_detected", False))
        if synergy_detected:
            result = self._fetch_narrative("FINAL_VERDICTS", "synergy_risk")
            intersecting_cycles = synergy.get("intersecting_cycles", [])
            if intersecting_cycles:
                cycle_explanation = self.explain_cycle_path(intersecting_cycles[0])
                if cycle_explanation:
                    result += f"\n\n{cycle_explanation}"
            return result

        # Pirámide Invertida
        stab_class = self.config.stability.classify(stability)
        is_inverted = stab_class == "critical"

        try:
            fin_verdict = self._validate_financial_metrics(financial_metrics)[
                "recommendation"
            ]
        except Exception:
            fin_verdict = FinancialVerdict.REVIEW

        if is_inverted:
            if fin_verdict == FinancialVerdict.ACCEPT:
                return self._fetch_narrative(
                    "FINAL_VERDICTS",
                    "inverted_pyramid_viable",
                    {"stability": round(stability, 2)},
                )
            else:
                return self._fetch_narrative("FINAL_VERDICTS", "inverted_pyramid_reject")

        # Agujeros topológicos
        if topo.has_cycles:
            return self._fetch_narrative(
                "FINAL_VERDICTS",
                "has_holes",
                {"beta_1": topo.beta_1},
            )

        # Caso ideal
        if fin_verdict == FinancialVerdict.ACCEPT:
            return self._fetch_narrative("FINAL_VERDICTS", "certified")

        # Fallback
        return self._fetch_narrative("FINAL_VERDICTS", "review_required")

    def _generate_executive_summary(
        self,
        final_verdict: VerdictLevel,
        strata: Dict[Stratum, StratumAnalysisResult],
        errors: List[str],
    ) -> str:
        """Genera resumen ejecutivo del reporte."""
        parts = []

        parts.append(
            f"**Veredicto: {final_verdict.emoji} {final_verdict.name}**\n"
            f"*{final_verdict.description}*\n"
        )

        # Resumen por estrato
        for stratum in [Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY]:
            if stratum in strata:
                analysis = strata[stratum]
                parts.append(
                    f"- **{stratum.name}**: {analysis.verdict.emoji} {analysis.verdict.name}"
                )

        if errors:
            parts.append(
                f"\n⚠️ Se detectaron {len(errors)} errores durante el análisis."
            )

        return "\n".join(parts)

    def _generate_recommendations(
        self,
        final_verdict: VerdictLevel,
        strata: Dict[Stratum, StratumAnalysisResult],
        synergy: Dict[str, Any],
    ) -> List[str]:
        """Genera recomendaciones accionables basadas en el análisis."""
        recommendations = []

        if final_verdict == VerdictLevel.VIABLE:
            recommendations.append("✅ Proceder con la ejecución del proyecto.")
            return recommendations

        # Recomendaciones por estrato comprometido
        for stratum, analysis in strata.items():
            if analysis.verdict.requires_attention or analysis.verdict.is_negative:
                if stratum == Stratum.PHYSICS:
                    recommendations.append(
                        "🔧 Revisar fuentes de datos y estabilizar flujo de información."
                    )
                elif stratum == Stratum.TACTICS:
                    if synergy.get("synergy_detected", False):
                        recommendations.append(
                            "⚡ Desacoplar ciclos de dependencia que comparten recursos críticos."
                        )
                    if analysis.metrics_summary.get("beta_1", 0) > 0:
                        recommendations.append(
                            "🔄 Sanear topología eliminando ciclos de dependencia."
                        )
                    if analysis.metrics_summary.get("beta_0", 1) > 1:
                        recommendations.append(
                            "🔗 Conectar componentes aislados del proyecto."
                        )
                elif stratum == Stratum.STRATEGY:
                    recommendations.append(
                        "💰 Revisar modelo financiero y ajustar parámetros de viabilidad."
                    )

        if not recommendations:
            recommendations.append(
                "🔍 Realizar análisis adicional para determinar siguiente paso."
            )

        return recommendations[:5]  # Máximo 5 recomendaciones

    # ========================================================================
    # COMPOSICIÓN DE REPORTE ESTRATÉGICO
    # ========================================================================

    def compose_strategic_narrative(
        self,
        topological_metrics: Union[TopologicalMetrics, Dict[str, Any], ValidatedTopology],
        financial_metrics: Dict[str, Any],
        stability: float = 0.0,
        synergy_risk: Optional[Dict[str, Any]] = None,
        spectral: Optional[Dict[str, Any]] = None,
        thermal_metrics: Optional[Union[ThermodynamicMetrics, Dict[str, Any]]] = None,
        physics_metrics: Optional[Union[PhysicsMetrics, Dict[str, Any]]] = None,
        control_metrics: Optional[Union[ControlMetrics, Dict[str, Any]]] = None,
        critical_resources: Optional[List[Dict[str, Any]]] = None,
        raw_cycles: Optional[List[List[str]]] = None,
        **kwargs: Any,
    ) -> StrategicReport:
        """
        Compone el reporte ejecutivo completo integrando todos los estratos DIKW.
        
        Args:
            topological_metrics: Métricas topológicas (Betti, Fiedler, etc.)
            financial_metrics: Métricas financieras (WACC, PI, etc.)
            stability: Índice de estabilidad piramidal Ψ
            synergy_risk: Información de sinergia de riesgo
            spectral: Métricas espectrales adicionales
            thermal_metrics: Métricas termodinámicas
            physics_metrics: Métricas físicas (giroscopía, presión)
            control_metrics: Métricas de control (Laplace)
            critical_resources: Recursos críticos para GraphRAG
            raw_cycles: Ciclos crudos para GraphRAG
            
        Returns:
            StrategicReport completo con veredicto integrado
        """
        # Normalizar topología
        if isinstance(topological_metrics, ValidatedTopology):
            topo = topological_metrics
        else:
            topo = ValidatedTopology.from_metrics(
                topological_metrics,
                strict=self.config.strict_euler_validation,
            )

        # Normalizar termodinámica
        tm = thermal_metrics or {}
        if isinstance(tm, ThermodynamicMetrics):
            thermal = tm
        else:
            thermal = ThermodynamicMetrics(
                **{k: v for k, v in tm.items() if k in ThermodynamicMetrics.__annotations__}
            )

        # Normalizar física
        pm = physics_metrics or {}
        if isinstance(pm, PhysicsMetrics):
            physics = pm
        else:
            physics = PhysicsMetrics(
                **{k: v for k, v in pm.items() if k in PhysicsMetrics.__annotations__}
            ) if pm else None

        # Normalizar control
        cm = control_metrics or kwargs.get("control") or {}
        if isinstance(cm, ControlMetrics):
            control = cm
        else:
            # Fallback legacy desde physics_metrics
            if not cm and isinstance(pm, dict):
                cm = {
                    "is_stable": pm.get("is_stable_lhp", True),
                    "phase_margin_deg": pm.get("phase_margin_deg", 45.0),
                }
            control = ControlMetrics(
                **{k: v for k, v in cm.items() if k in ControlMetrics.__annotations__}
            ) if cm else None

        synergy = synergy_risk or {}
        spec = spectral or {}

        # Usar stability del topo si no se pasa explícitamente
        eff_stability = stability if stability != 0.0 else topo.pyramid_stability

        # Acumuladores
        strata_analysis: Dict[Stratum, StratumAnalysisResult] = {}
        section_narratives: List[str] = []
        all_verdicts: List[VerdictLevel] = []
        errors: List[str] = []

        # ====== HEADER ======
        section_narratives.append(self._generate_report_header())

        # ====== PHYSICS: Base Térmica y Dinámica ======
        physics_result = self._analyze_physics_stratum(
            thermal,
            eff_stability,
            physics,
            control,
        )
        strata_analysis[Stratum.PHYSICS] = physics_result
        all_verdicts.append(physics_result.verdict)

        # ====== TACTICS: Estructura Topológica ======
        tactics_result = self._analyze_tactics_stratum(
            topo,
            synergy,
            spec,
            eff_stability,
        )
        strata_analysis[Stratum.TACTICS] = tactics_result
        all_verdicts.append(tactics_result.verdict)

        # ====== STRATEGY: Viabilidad Financiera ======
        strategy_result = self._analyze_strategy_stratum(financial_metrics)
        strata_analysis[Stratum.STRATEGY] = strategy_result
        all_verdicts.append(strategy_result.verdict)

        # ====== SECCIÓN 1: Diagnóstico Integrado ======
        section_narratives.append("## 🏗️ Diagnóstico del Edificio Vivo\n")

        temp = self._normalize_temperature(thermal.system_temperature)
        integrated_diagnosis = self._generate_integrated_diagnosis(
            eff_stability,
            temp.celsius,
            physics_result.verdict,
        )
        section_narratives.append(integrated_diagnosis)
        section_narratives.append("")

        # ====== SECCIÓN 1.5: Dinámica de Bombeo ======
        section_narratives.append("### 0. Dinámica de Bombeo (Física)\n")
        section_narratives.append(physics_result.narrative)
        section_narratives.append("")

        # ====== SECCIÓN 2: Detalles Topológicos ======
        section_narratives.append("### 1. Auditoría de Integridad Estructural\n")
        try:
            topo_narrative, topo_verdict = self.translate_topology(
                topo,
                eff_stability,
                synergy,
                spec,
            )
            section_narratives.append(topo_narrative)
            all_verdicts.append(topo_verdict)

            # GraphRAG: Recursos críticos
            if critical_resources:
                limit = self.config.max_stress_points_display
                for resource in critical_resources[:limit]:
                    node_id = resource.get("id")
                    degree = resource.get("in_degree", 0)
                    if node_id:
                        explanation = self.explain_stress_point(node_id, degree)
                        if explanation:
                            section_narratives.append(f"- {explanation}")

            # GraphRAG: Ciclos
            if raw_cycles:
                limit_cycles = self.config.max_cycle_path_display
                for cycle_path in raw_cycles[:limit_cycles]:
                    explanation = self.explain_cycle_path(cycle_path)
                    if explanation:
                        section_narratives.append(f"- {explanation}")

        except Exception as e:
            error_msg = f"Error analizando estructura: {e}"
            section_narratives.append(f"❌ {error_msg}")
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
        section_narratives.append("")

        # ====== SECCIÓN 3: Detalles Financieros ======
        section_narratives.append("### 2. Análisis de Cargas Financieras\n")
        try:
            fin_narrative, fin_verdict, _ = self.translate_financial(financial_metrics)
            section_narratives.append(fin_narrative)
            all_verdicts.append(fin_verdict)
        except Exception as e:
            error_msg = f"Error analizando finanzas: {e}"
            section_narratives.append(f"❌ {error_msg}")
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
        section_narratives.append("")

        # ====== SECCIÓN 4: Mercado ======
        section_narratives.append("### 3. Geotecnia de Mercado\n")
        section_narratives.append(self.get_market_context())
        section_narratives.append("")

        # ====== WISDOM: Veredicto Final ======
        final_verdict = VerdictLevel.supremum(*all_verdicts)

        # Elevar severidad si hay errores (clausura)
        if errors:
            final_verdict = final_verdict | VerdictLevel.REVISAR

        wisdom_result = self._analyze_wisdom_stratum(
            topo,
            financial_metrics,
            eff_stability,
            synergy,
            final_verdict,
            bool(errors),
        )
        strata_analysis[Stratum.WISDOM] = wisdom_result

        # ====== SECCIÓN 5: Dictamen Final ======
        section_narratives.append("### 💡 Dictamen del Ingeniero Jefe\n")
        section_narratives.append(wisdom_result.narrative)

        # Construir executive summary
        executive_summary = self._generate_executive_summary(
            final_verdict,
            strata_analysis,
            errors,
        )

        # Generar recomendaciones
        recommendations = self._generate_recommendations(
            final_verdict,
            strata_analysis,
            synergy,
        )

        # Calcular confianza global
        confidences = [a.confidence for a in strata_analysis.values()]
        global_confidence = sum(confidences) / len(confidences) if confidences else 1.0

        return StrategicReport(
            title="INFORME DE INGENIERÍA ESTRATÉGICA",
            verdict=final_verdict,
            executive_summary=executive_summary,
            strata_analysis=strata_analysis,
            recommendations=recommendations,
            raw_narrative="\n".join(section_narratives),
            confidence=global_confidence,
        )

    def assemble_data_product(
        self,
        graph: nx.DiGraph,
        report: Any,
    ) -> Dict[str, Any]:
        """
        Ensambla el Producto de Datos Final (Sabiduría).

        Integra la estructura (Grafo) y el análisis (Reporte) en un artefacto
        consumible por el negocio.
        """
        narrative = getattr(report, "strategic_narrative", "")
        verdict = getattr(report, "integrity_score", 0.0)

        return {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "verdict_score": verdict,
                "complexity_level": getattr(report, "complexity_level", "Unknown"),
            },
            "narrative": {
                "executive_summary": narrative,
                "alerts": (
                    getattr(report, "waste_alerts", [])
                    + getattr(report, "circular_risks", [])
                ),
            },
            "topology": {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "details": getattr(report, "details", {}),
            },
        }

    # ========================================================================
    # MÉTODOS LEGACY DE COMPATIBILIDAD
    # ========================================================================

    def compose_strategic_narrative_legacy(
        self,
        topological_metrics: Any,
        financial_metrics: Dict[str, Any],
        stability: float = 0.0,
        synergy_risk: Optional[Dict[str, Any]] = None,
        spectral: Optional[Dict[str, Any]] = None,
        thermal_metrics: Optional[Dict[str, Any]] = None,
        physics_metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Versión legacy que retorna solo el string de narrativa.
        
        Mantiene compatibilidad con código existente.
        """
        report = self.compose_strategic_narrative(
            topological_metrics=topological_metrics,
            financial_metrics=financial_metrics,
            stability=stability,
            synergy_risk=synergy_risk,
            spectral=spectral,
            thermal_metrics=thermal_metrics,
            physics_metrics=physics_metrics,
        )
        return report.raw_narrative


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================


def create_translator(
    config: Optional[TranslatorConfig] = None,
    market_provider: Optional[Callable[[], str]] = None,
    mic: Optional[MICRegistry] = None,
    enable_cache: bool = True,
) -> SemanticTranslator:
    """Factory function para crear un traductor configurado."""
    return SemanticTranslator(
        config=config,
        market_provider=market_provider,
        mic=mic,
        enable_cache=enable_cache,
    )


def translate_metrics_to_narrative(
    topological_metrics: Any,
    financial_metrics: Dict[str, Any],
    stability: float = 0.0,
    **kwargs: Any,
) -> str:
    """Función de conveniencia para traducción rápida."""
    translator = SemanticTranslator()
    report = translator.compose_strategic_narrative(
        topological_metrics=topological_metrics,
        financial_metrics=financial_metrics,
        stability=stability,
        **kwargs,
    )
    return report.raw_narrative


def verify_verdict_lattice() -> bool:
    """
    Verifica que VerdictLevel cumple las leyes del lattice.
    
    Útil para testing y debugging.
    
    Returns:
        True si todas las propiedades se cumplen
    """
    results = VerdictLevel.verify_lattice_laws()
    all_pass = all(results.values())
    
    if not all_pass:
        failed = [k for k, v in results.items() if not v]
        logger.error(f"Lattice laws violated: {failed}")
    
    return all_pass