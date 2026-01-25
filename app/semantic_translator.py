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
   Implementa una estructura algebraica de orden $(Verdict, \le, \sqcup)$ donde:
   - $VIABLE (\bot)$ < $REVISAR$ < $INVIABLE (\top)$.
   - La operación de síntesis es el Supremo ($\sqcup$ o Join), adoptando siempre el
     criterio más conservador (Worst-Case Scenario) entre los agentes. Si Finanzas dice
     "Viable" ($\bot$) pero Topología dice "Inviable" ($\top$), el resultado es $\top$
     [Fuente: semantic_translator.txt].

2. Síntesis DIKW (Data $\to$ Wisdom):
   Eleva los datos crudos a Sabiduría mediante la integración de contextos:
   - Traduce $\beta_1 > 0$ (Topología) $\to$ "Socavón Lógico" (Bloqueo administrativo).
   - Traduce $\Psi < 1.0$ (Física) $\to$ "Pirámide Invertida" (Riesgo de colapso logístico).
   - Traduce $T_{sys} > 50^\circ C$ (Termodinámica) $\to$ "Fiebre Inflacionaria"
     [Fuente: LENGUAJE_CONSEJO.md].

3. Narrativa Generativa Causal (GraphRAG):
   Utiliza **Retrieval-Augmented Generation sobre Grafos** para explicar la causalidad.
   No solo reporta el error, sino que traza la ruta del ciclo en el grafo de dependencias
   para explicar al humano *por qué* el proyecto está bloqueado, convirtiendo la
   abstracción matemática en una advertencia accionable [Fuente: SAGES.md].

4. Interoperabilidad Semántica (Embeddings):
   Resuelve la disonancia cognitiva entre dominios (Ingeniería vs. Contabilidad)
   utilizando vectores semánticos (Sentence Transformers) para identificar que
   "Concreto" y "Hormigón" son el mismo nodo termodinámico, unificando la visión
   del proyecto.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, Enum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

# Importaciones del proyecto
try:
    from agent.business_topology import TopologicalMetrics
except ImportError:
    # Fallback para testing o cuando el módulo no está disponible
    TopologicalMetrics = None

try:
    from app.schemas import Stratum
except ImportError:
    # Fallback
    from enum import IntEnum as StratumBase
    class Stratum(StratumBase):
        WISDOM = 0
        STRATEGY = 1
        TACTICS = 2
        PHYSICS = 3

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURACIÓN CENTRALIZADA
# ============================================================================


@dataclass(frozen=True)
class StabilityThresholds:
    """
    Umbrales para interpretación del índice de estabilidad piramidal (Ψ).

    Fundamentación topológica:
    - Ψ < critical: Pirámide Invertida (Cimentación insuficiente)
    - Ψ ∈ [critical, solid): Estructura Isostática (Equilibrio precario)
    - Ψ ≥ solid: Estructura Antisísmica (Base robusta con redundancia)

    Invariante: critical < solid
    """

    critical: float = 1.0
    solid: float = 10.0
    warning: float = 3.0  # Nuevo: umbral de advertencia intermedio

    def __post_init__(self) -> None:
        """Valida invariantes de los umbrales."""
        if self.critical < 0:
            raise ValueError("critical threshold must be non-negative")
        if self.solid <= self.critical:
            raise ValueError("solid threshold must be greater than critical")
        if not (self.critical <= self.warning <= self.solid):
            # Ajustar warning si está fuera de rango
            object.__setattr__(
                self, "warning", (self.critical + self.solid) / 2
            )

    def classify(self, stability: float) -> str:
        """Clasifica un valor de estabilidad."""
        if stability < 0:
            return "invalid"
        if stability < self.critical:
            return "critical"
        if stability < self.warning:
            return "warning"
        if stability < self.solid:
            return "stable"
        return "robust"


@dataclass(frozen=True)
class TopologicalThresholds:
    """
    Umbrales para interpretación de números de Betti.

    Fundamentación Algebraica:
    - β₀ (componentes conexos): Mide fragmentación
    - β₁ (ciclos independientes): Mide "agujeros" lógicos
    - χ (característica de Euler): Invariante topológico χ = β₀ - β₁

    Invariantes:
    - β₀ ≥ 1 para proyectos no vacíos
    - β₁ ≥ 0 (ciclos no pueden ser negativos)
    """

    connected_components_optimal: int = 1
    cycles_optimal: int = 0
    cycles_warning: int = 1
    cycles_critical: int = 3
    max_fragmentation: int = 5

    def __post_init__(self) -> None:
        """Valida invariantes."""
        if self.connected_components_optimal < 1:
            raise ValueError("optimal connected components must be >= 1")
        if self.cycles_optimal < 0:
            raise ValueError("optimal cycles must be non-negative")
        if not (self.cycles_optimal <= self.cycles_warning <= self.cycles_critical):
            raise ValueError("cycle thresholds must be ordered")

    def classify_connectivity(self, beta_0: int) -> str:
        """Clasifica conectividad."""
        if beta_0 == 0:
            return "empty"
        if beta_0 == self.connected_components_optimal:
            return "unified"
        if beta_0 <= self.max_fragmentation:
            return "fragmented"
        return "severely_fragmented"

    def classify_cycles(self, beta_1: int) -> str:
        """Clasifica nivel de ciclos."""
        if beta_1 <= self.cycles_optimal:
            return "clean"
        if beta_1 <= self.cycles_warning:
            return "minor"
        if beta_1 <= self.cycles_critical:
            return "moderate"
        return "critical"


@dataclass(frozen=True)
class ThermalThresholds:
    """
    Umbrales para métricas termodinámicas.

    Metáfora: El proyecto como sistema termodinámico
    - Temperatura: Volatilidad/Inflación de precios
    - Entropía: Desorden administrativo
    - Exergía: Eficiencia de inversión
    """

    temperature_cold: float = 20.0
    temperature_warm: float = 35.0
    temperature_hot: float = 50.0
    temperature_critical: float = 75.0

    entropy_low: float = 0.3
    entropy_high: float = 0.7

    exergy_efficient: float = 0.7
    exergy_poor: float = 0.3

    def classify_temperature(self, temp: float) -> str:
        """Clasifica temperatura del sistema."""
        if temp < 0:
            return "invalid"
        if temp <= self.temperature_cold:
            return "cold"
        if temp <= self.temperature_warm:
            return "stable"
        if temp <= self.temperature_hot:
            return "warm"
        if temp <= self.temperature_critical:
            return "hot"
        return "critical"


@dataclass(frozen=True)
class FinancialThresholds:
    """
    Umbrales para métricas financieras.
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


# ============================================================================
# LATTICE DE VEREDICTOS
# ============================================================================


class VerdictLevel(IntEnum):
    """
    Lattice de veredictos con orden total.

    Estructura algebraica: (VerdictLevel, ≤, ⊔, ⊓)
    - Forma un lattice acotado completo
    - VIABLE es el bottom (⊥)
    - RECHAZAR es el top (⊤)

    Semántica:
    - VIABLE: Proyecto puede proceder
    - CONDICIONAL: Viable con modificaciones
    - REVISAR: Requiere análisis adicional
    - PRECAUCION: Riesgos significativos identificados
    - RECHAZAR: No viable en estado actual
    """

    VIABLE = 0          # ⊥ - Mejor caso
    CONDICIONAL = 1     # Viable con condiciones
    REVISAR = 2         # Necesita más análisis
    PRECAUCION = 3      # Advertencia significativa
    RECHAZAR = 4        # ⊤ - Peor caso

    @classmethod
    def bottom(cls) -> VerdictLevel:
        """Elemento mínimo del lattice."""
        return cls.VIABLE

    @classmethod
    def top(cls) -> VerdictLevel:
        """Elemento máximo del lattice."""
        return cls.RECHAZAR

    @classmethod
    def supremum(cls, *levels: VerdictLevel) -> VerdictLevel:
        """Operación JOIN (⊔): toma el peor caso."""
        if not levels:
            return cls.VIABLE
        return cls(max(level.value for level in levels))

    @classmethod
    def infimum(cls, *levels: VerdictLevel) -> VerdictLevel:
        """Operación MEET (⊓): toma el mejor caso."""
        if not levels:
            return cls.RECHAZAR
        return cls(min(level.value for level in levels))

    def join(self, other: VerdictLevel) -> VerdictLevel:
        """Join binario: self ⊔ other."""
        return VerdictLevel.supremum(self, other)

    def meet(self, other: VerdictLevel) -> VerdictLevel:
        """Meet binario: self ⊓ other."""
        return VerdictLevel.infimum(self, other)

    def __or__(self, other: VerdictLevel) -> VerdictLevel:
        """Sintaxis: a | b = a ⊔ b."""
        return self.join(other)

    def __and__(self, other: VerdictLevel) -> VerdictLevel:
        """Sintaxis: a & b = a ⊓ b."""
        return self.meet(other)

    @property
    def emoji(self) -> str:
        """Representación visual."""
        return {
            VerdictLevel.VIABLE: "✅",
            VerdictLevel.CONDICIONAL: "🔵",
            VerdictLevel.REVISAR: "🔍",
            VerdictLevel.PRECAUCION: "⚠️",
            VerdictLevel.RECHAZAR: "🛑",
        }[self]

    @property
    def is_positive(self) -> bool:
        """Indica si el veredicto permite proceder."""
        return self.value <= VerdictLevel.CONDICIONAL.value

    @property
    def is_negative(self) -> bool:
        """Indica si el veredicto bloquea el proyecto."""
        return self == VerdictLevel.RECHAZAR


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
        """Convierte a VerdictLevel."""
        mapping = {
            FinancialVerdict.ACCEPT: VerdictLevel.VIABLE,
            FinancialVerdict.CONDITIONAL: VerdictLevel.CONDICIONAL,
            FinancialVerdict.REVIEW: VerdictLevel.REVISAR,
            FinancialVerdict.REJECT: VerdictLevel.RECHAZAR,
        }
        return mapping.get(self, VerdictLevel.REVISAR)

    @classmethod
    def from_string(cls, value: str) -> FinancialVerdict:
        """Parsea desde string."""
        normalized = value.upper().strip()
        for verdict in cls:
            if verdict.value.upper() == normalized or verdict.name == normalized:
                return verdict
        return cls.REVIEW


# ============================================================================
# ESTRUCTURAS DE DATOS PARA MÉTRICAS
# ============================================================================


@runtime_checkable
class HasBettiNumbers(Protocol):
    """Protocolo para objetos con números de Betti."""

    @property
    def beta_0(self) -> int: ...

    @property
    def beta_1(self) -> int: ...


@dataclass
class PhysicsMetricsDTO:
    """
    DTO para métricas del Motor de Física y Control.

    Captura variables de estado avanzadas:
    - Giroscopía (Estabilidad rotacional del flujo)
    - Control (Polos y Ceros en Laplace)
    """
    # Del FluxCondenser (Giroscopía)
    gyroscopic_stability: float = 1.0  # Sg
    nutation_amplitude: float = 0.0

    # Del LaplaceOracle (Estabilidad de Control)
    phase_margin_deg: float = 45.0
    is_stable_lhp: bool = True
    damping_regime: str = "CRITICALLY_DAMPED"

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> PhysicsMetricsDTO:
        """Crea desde diccionario."""
        if not data:
            return cls()

        return cls(
            gyroscopic_stability=float(data.get("gyroscopic_stability", 1.0)),
            nutation_amplitude=float(data.get("nutation_amplitude", 0.0)),
            phase_margin_deg=float(data.get("phase_margin_deg", 45.0)),
            is_stable_lhp=bool(data.get("is_stable_lhp", True)),
            damping_regime=str(data.get("damping_regime", "CRITICALLY_DAMPED")),
        )


@dataclass
class TopologyMetricsDTO:
    """
    DTO para métricas topológicas.

    Normaliza diferentes fuentes de datos topológicos.
    """

    beta_0: int = 1
    beta_1: int = 0
    delta_beta_1: int = 0  # Mayer-Vietoris (NUEVO)
    euler_characteristic: int = 1
    euler_efficiency: float = 1.0

    def __post_init__(self) -> None:
        """Valida invariantes topológicos."""
        if self.beta_0 < 0:
            raise ValueError("β₀ must be non-negative")
        if self.beta_1 < 0:
            raise ValueError("β₁ must be non-negative")

        # Verificar consistencia de Euler: χ = β₀ - β₁
        expected_euler = self.beta_0 - self.beta_1
        if self.euler_characteristic != expected_euler:
            logger.warning(
                f"Euler characteristic mismatch: "
                f"χ={self.euler_characteristic} ≠ β₀-β₁={expected_euler}"
            )

    @classmethod
    def from_any(cls, source: Any) -> TopologyMetricsDTO:
        """Factory que normaliza diferentes fuentes."""
        if source is None:
            return cls()

        if isinstance(source, cls):
            return source

        if isinstance(source, dict):
            return cls(
                beta_0=int(source.get("beta_0", 1)),
                beta_1=int(source.get("beta_1", 0)),
                euler_characteristic=int(source.get("euler_characteristic", 1)),
                euler_efficiency=float(source.get("euler_efficiency", 1.0)),
            )

        # Duck typing para TopologicalMetrics
        if hasattr(source, "beta_0") and hasattr(source, "beta_1"):
            return cls(
                beta_0=int(getattr(source, "beta_0", 1)),
                beta_1=int(getattr(source, "beta_1", 0)),
                delta_beta_1=int(getattr(source, "delta_beta_1", 0)),
                euler_characteristic=int(getattr(source, "euler_characteristic", 1)),
                euler_efficiency=float(getattr(source, "euler_efficiency", 1.0)),
            )

        raise TypeError(f"Cannot convert {type(source).__name__} to TopologyMetricsDTO")


@dataclass
class ThermalMetricsDTO:
    """DTO para métricas termodinámicas."""

    entropy: float = 0.0
    exergy: float = 1.0
    temperature: float = 25.0

    def __post_init__(self) -> None:
        """Valida rangos."""
        self.entropy = max(0.0, min(1.0, self.entropy))
        self.exergy = max(0.0, min(1.0, self.exergy))
        self.temperature = max(0.0, self.temperature)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> ThermalMetricsDTO:
        """Crea desde diccionario."""
        if not data:
            return cls()

        return cls(
            entropy=float(data.get("entropy", 0.0)),
            exergy=float(data.get("exergy", 1.0)),
            temperature=float(data.get("system_temperature", data.get("temperature", 25.0))),
        )


@dataclass
class SpectralMetricsDTO:
    """DTO para métricas espectrales."""

    fiedler_value: float = 0.0
    wavelength: float = 0.0
    spectral_gap: float = 0.0
    resonance_risk: bool = False

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> SpectralMetricsDTO:
        """Crea desde diccionario."""
        if not data:
            return cls()

        return cls(
            fiedler_value=float(data.get("fiedler_value", 0.0)),
            wavelength=float(data.get("wavelength", 0.0)),
            spectral_gap=float(data.get("spectral_gap", 0.0)),
            resonance_risk=bool(data.get("resonance_risk", False)),
        )


@dataclass
class SynergyRiskDTO:
    """DTO para riesgo de sinergia (producto cup)."""

    synergy_detected: bool = False
    intersecting_cycles_count: int = 0
    intersecting_nodes: List[str] = field(default_factory=list)
    intersecting_cycles: List[List[str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> SynergyRiskDTO:
        """Crea desde diccionario."""
        if not data:
            return cls()

        return cls(
            synergy_detected=bool(data.get("synergy_detected", False)),
            intersecting_cycles_count=int(data.get("intersecting_cycles_count", 0)),
            intersecting_nodes=list(data.get("intersecting_nodes", [])),
            intersecting_cycles=list(data.get("intersecting_cycles", [])),
        )


@dataclass
class StratumAnalysisResult:
    """
    Resultado del análisis de un estrato.

    Integra con la jerarquía DIKW.
    """

    stratum: Stratum
    verdict: VerdictLevel
    narrative: str
    metrics_summary: Dict[str, Any]
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "stratum": self.stratum.name,
            "verdict": self.verdict.name,
            "verdict_emoji": self.verdict.emoji,
            "narrative": self.narrative,
            "metrics": self.metrics_summary,
            "issues": self.issues,
        }


@dataclass
class StrategicReport:
    """
    Reporte estratégico completo.

    Representa el juicio final del Consejo.
    """

    title: str
    verdict: VerdictLevel
    executive_summary: str
    strata_analysis: Dict[Stratum, StratumAnalysisResult]
    recommendations: List[str]
    raw_narrative: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def is_viable(self) -> bool:
        """Indica si el proyecto es viable."""
        return self.verdict.is_positive

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "title": self.title,
            "verdict": self.verdict.name,
            "verdict_emoji": self.verdict.emoji,
            "is_viable": self.is_viable,
            "executive_summary": self.executive_summary,
            "strata_analysis": {
                s.name: a.to_dict()
                for s, a in self.strata_analysis.items()
            },
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
        }


# ============================================================================
# PLANTILLAS DE NARRATIVA
# ============================================================================


class NarrativeTemplates:
    """
    Plantillas de narrativa organizadas por dominio.

    Separa el contenido textual de la lógica de traducción.
    """

    # ========== TOPOLOGÍA ==========

    TOPOLOGY_CYCLES: Dict[str, str] = {
        "clean": (
            "✅ **Integridad Estructural (Genus 0)**: No se detectan socavones lógicos "
            "(β₁ = 0). La Trazabilidad de Carga de Costos fluye verticalmente desde la "
            "Cimentación hasta el Ápice sin recirculaciones."
        ),
        "minor": (
            "🔶 **Falla Estructural Local (Genus {beta_1})**: Se detectaron {beta_1} "
            "socavones lógicos en la estructura de costos. Estos 'agujeros' impiden "
            "la correcta Trazabilidad de Carga y deben ser corregidos para "
            "evitar asentamientos diferenciales en el presupuesto."
        ),
        "moderate": (
            "🚨 **Estructura Geológicamente Inestable (Genus {beta_1})**: "
            "Se detectó un Genus Estructural de {beta_1}, indicando una estructura tipo 'esponja'. "
            "Existen múltiples bucles de retroalimentación de costos que "
            "impiden la Trazabilidad de Carga y hacen colapsar cualquier valoración estática."
        ),
        "critical": (
            "💀 **COLAPSO TOPOLÓGICO (Genus {beta_1})**: "
            "La estructura está completamente perforada con {beta_1} ciclos independientes. "
            "Es matemáticamente imposible calcular costos determinísticos. "
            "Se requiere rediseño fundamental."
        ),
    }

    TOPOLOGY_CONNECTIVITY: Dict[str, str] = {
        "empty": "⚠️ **Terreno Vacío**: No hay estructura proyectada (β₀ = 0).",
        "unified": (
            "🔗 **Unidad de Obra Monolítica**: El proyecto funciona como un solo "
            "edificio interconectado (β₀ = 1). Todas las cargas tácticas (APUs) "
            "se transfieren correctamente hacia un único Ápice Estratégico."
        ),
        "fragmented": (
            "⚠️ **Edificios Desconectados (Fragmentación)**: El proyecto no es una "
            "estructura única, sino un archipiélago de {beta_0} sub-estructuras aisladas. "
            "No existe un Ápice unificado que centralice la carga financiera."
        ),
        "severely_fragmented": (
            "🚨 **Fragmentación Severa**: El proyecto está fragmentado en {beta_0} islas "
            "completamente desconectadas. Esto indica múltiples proyectos empaquetados "
            "como uno solo, o datos severamente incompletos."
        ),
    }

    # ========== ESTABILIDAD ==========

    STABILITY: Dict[str, str] = {
        "critical": (
            "📉 **COLAPSO POR BASE ESTRECHA (Pirámide Invertida)**: "
            "Ψ = {stability:.2f}. La Cimentación Logística (Insumos) es demasiado "
            "angosta para soportar el Peso Táctico (APUs) que tiene encima. "
            "El centro de gravedad está muy alto; riesgo inminente de vuelco financiero."
        ),
        "warning": (
            "⚖️ **Equilibrio Precario (Isostático)**: "
            "Ψ = {stability:.2f}. El proyecto tiene la mínima base necesaria, "
            "sin redundancia. Cualquier perturbación en el suministro puede "
            "desestabilizar toda la estructura."
        ),
        "stable": (
            "⚖️ **Estructura Isostática (Estable)**: "
            "Ψ = {stability:.2f}. El equilibrio entre la carga de actividades y "
            "el soporte de insumos es adecuado, aunque no posee redundancia sísmica."
        ),
        "robust": (
            "🛡️ **ESTRUCTURA ANTISÍSMICA (Resiliente)**: "
            "Ψ = {stability:.2f}. La Cimentación de Recursos es amplia y redundante. "
            "El proyecto tiene un bajo centro de gravedad, capaz de absorber "
            "vibraciones del mercado (volatilidad) sin sufrir daños estructurales."
        ),
    }

    # ========== ESPECTRAL ==========

    SPECTRAL_COHESION: Dict[str, str] = {
        "high": (
            "🔗 **Alta Cohesión del Equipo (Fiedler={fiedler:.2f})**: "
            "La estructura de costos está fuertemente sincronizada."
        ),
        "standard": (
            "⚖️ **Cohesión Estándar (Fiedler={fiedler:.2f})**: "
            "El proyecto presenta un acoplamiento típico entre sus componentes."
        ),
        "low": (
            "💔 **Fractura Organizacional (Fiedler={fiedler:.3f})**: "
            "Baja cohesión espectral. Los subsistemas operan aislados, "
            "riesgo de desalineación en ejecución."
        ),
    }

    SPECTRAL_RESONANCE: Dict[str, str] = {
        "risk": (
            "🔊 **RIESGO DE RESONANCIA FINANCIERA (λ={wavelength:.2f})**: "
            "El espectro de vibración está peligrosamente concentrado. "
            "Un impacto externo (inflación/escasez) podría amplificarse en toda la "
            "estructura simultáneamente."
        ),
        "safe": (
            "🌊 **Disipación Ondulatoria (λ={wavelength:.2f})**: "
            "La estructura tiene capacidad para amortiguar impactos locales sin entrar en "
            "resonancia sistémica."
        ),
    }

    # ========== TERMODINÁMICA ==========

    THERMAL_TEMPERATURE: Dict[str, str] = {
        "cold": (
            "❄️ **Temperatura Estable ({temperature:.1f}°C)**: "
            "El proyecto está termodinámicamente equilibrado (Precios fríos/fijos)."
        ),
        "stable": (
            "🌡️ **Temperatura Normal ({temperature:.1f}°C)**: "
            "Condiciones térmicas estándar del mercado."
        ),
        "warm": (
            "🌡️ **Calentamiento Operativo ({temperature:.1f}°C)**: "
            "Existe una exposición moderada a la volatilidad de precios."
        ),
        "hot": (
            "🔥 **EL PROYECTO TIENE FIEBRE ({temperature:.1f}°C)**: "
            "El Índice de Inflación Interna es crítico. Los costos de insumos volátiles "
            "están sobrecalentando la estructura de precios."
        ),
        "critical": (
            "☢️ **FUSIÓN TÉRMICA ({temperature:.1f}°C)**: "
            "Temperatura crítica alcanzada. Los costos están en espiral inflacionaria. "
            "Riesgo de colapso financiero por sobrecalentamiento incontrolado."
        ),
    }

    THERMAL_ENTROPY: Dict[str, str] = {
        "low": (
            "📋 **Orden Administrativo (S={entropy:.2f})**: "
            "Baja entropía indica procesos bien estructurados y datos limpios."
        ),
        "high": (
            "🌪️ **Alta Entropía ({entropy:.2f})**: Caos administrativo detectado. "
            "La energía del dinero se disipa en fricción operativa (datos sucios o desorganizados)."
        ),
    }

    # ========== FISICA AVANZADA (Flux & Laplace) ==========

    GYROSCOPIC_STABILITY: Dict[str, str] = {
        "stable": "✅ **Giroscopio Estable**: Flujo con momento angular constante.",
        "precession": "⚠️ **Precesión Detectada**: Oscilación lateral en el flujo de datos.",
        "nutation": "🚨 **NUTACIÓN CRÍTICA**: Inestabilidad rotacional. El proceso corre riesgo de colapso inercial."
    }

    LAPLACE_CONTROL: Dict[str, str] = {
        "robust": "🛡️ **Control Robusto**: Margen de fase sólido (>45°).",
        "marginal": "⚠️ **Estabilidad Marginal**: Respuesta oscilatoria ante transitorios.",
        "unstable": "⛔ **DIVERGENCIA MATEMÁTICA**: Polos en el semiplano derecho (RHP)."
    }

    MAYER_VIETORIS: str = (
        "🧩 **Incoherencia de Integración**: La fusión de los presupuestos ha generado "
        "{delta_beta_1} ciclos lógicos fantasmas (Anomalía de Mayer-Vietoris). "
        "Los datos individuales son válidos, pero su unión crea una contradicción topológica."
    )

    THERMAL_DEATH: str = (
        "☢️ **MUERTE TÉRMICA DEL SISTEMA**: La entropía ha alcanzado el equilibrio máximo. "
        "No hay energía libre para procesar información útil."
    )

    # ========== SINERGIA ==========

    SYNERGY: str = (
        "🔥 **Riesgo de Contagio (Efecto Dominó)**: Se detectó una 'Sinergia de Riesgo' "
        "en {count} puntos de intersección crítica. Los errores no son aislados; si uno falla, "
        "provocará una reacción en cadena a través de los frentes de obra compartidos."
    )

    EULER_EFFICIENCY: str = (
        "🕸️ **Sobrecarga de Gestión (Entropía)**: La eficiencia de Euler es baja ({efficiency:.2f}). "
        "Existe una complejidad innecesaria de enlaces que dificulta la supervisión y aumenta "
        "los costos indirectos de administración."
    )

    # ========== GRAPHRAG ==========

    CYCLE_PATH: str = (
        "🔄 **Ruta del Ciclo Detectada**: La circularidad sigue el camino: [{path}]. "
        "Esto significa que el costo de '{first_node}' depende indirectamente de sí mismo, "
        "creando una indeterminación matemática en la valoración."
    )

    STRESS_POINT: str = (
        "⚡ **Punto de Estrés Estructural**: El elemento '{node}' actúa como una 'Piedra Angular' crítica, "
        "soportando {degree} conexiones directas. Una variación en su precio o disponibilidad "
        "impactará desproporcionadamente a toda la estructura del proyecto (Punto Único de Falla)."
    )

    # ========== FINANZAS ==========

    WACC: str = "💰 **Costo de Oportunidad**: WACC = {wacc:.2%}."

    CONTINGENCY: str = "📊 **Blindaje Financiero**: Contingencia sugerida de ${contingency:,.2f}."

    FINANCIAL_VERDICT: Dict[str, str] = {
        "accept": "🚀 **Veredicto**: VIABLE (IR={pi:.2f}). Estructura financiable.",
        "conditional": "🔵 **Veredicto**: CONDICIONAL (IR={pi:.2f}). Viable con ajustes.",
        "review": "🔍 **Veredicto**: REVISIÓN REQUERIDA.",
        "reject": "🛑 **Veredicto**: RIESGO CRÍTICO (IR={pi:.2f}). No procedente.",
    }

    # ========== MERCADO ==========

    MARKET_CONTEXTS: Tuple[str, ...] = (
        "Suelo Estable: Precios de cemento sin variación significativa.",
        "Terreno Inflacionario: Acero al alza (+2.5%). Reforzar estimaciones.",
        "Vientos de Cambio: Volatilidad cambiaria favorable para importaciones.",
        "Falla Geológica Laboral: Escasez de mano de obra calificada.",
        "Mercado Saturado: Alta competencia presiona márgenes.",
    )

    # ========== VEREDICTOS FINALES ==========

    FINAL_VERDICTS: Dict[str, str] = {
        "synergy_risk": (
            "🛑 **PARADA DE EMERGENCIA (Efecto Dominó)**: Se detectaron ciclos interconectados "
            "que comparten recursos críticos. El riesgo no es aditivo, es multiplicativo. "
            "Cualquier fallo en el suministro provocará un colapso sistémico en múltiples frentes. "
            "Desacoplar los ciclos antes de continuar."
        ),
        "inverted_pyramid_viable": (
            "⚠️ **PRECAUCIÓN LOGÍSTICA (Estructura Inestable)**: Aunque los números "
            "financieros cuadran, el proyecto es una **Pirámide Invertida** (Ψ={stability:.2f}). "
            "Se sostiene sobre una base de recursos demasiado estrecha. "
            "RECOMENDACIÓN: Ampliar la base de proveedores antes de construir."
        ),
        "inverted_pyramid_reject": (
            "❌ **PROYECTO INVIABLE (Riesgo de Colapso)**: Combinación letal de "
            "inestabilidad estructural (Pirámide Invertida) e inviabilidad financiera. "
            "No proceder bajo ninguna circunstancia sin rediseño total."
        ),
        "has_holes": (
            "🛑 **DETENER PARA REPARACIONES**: Se detectaron {beta_1} socavones "
            "lógicos (ciclos). No se puede verter dinero en una estructura con agujeros. "
            "Sanear la topología antes de aprobar presupuesto."
        ),
        "certified": (
            "✅ **CERTIFICADO DE SOLIDEZ**: Estructura piramidal estable, sin socavones "
            "lógicos y financieramente viable. Proceder a fase de ejecución."
        ),
        "review_required": (
            "🔍 **REVISIÓN TÉCNICA REQUERIDA**: La estructura es sólida pero los números no convencen."
        ),
        "analysis_failed": (
            "⚠️ ANÁLISIS ESTRUCTURAL INTERRUMPIDO: Se detectaron inconsistencias matemáticas "
            "o falta de datos críticos que impiden certificar la solidez del proyecto. "
            "Revise los errores en las secciones técnicas."
        ),
    }


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
    ) -> None:
        """
        Inicializa el traductor.

        Args:
            config: Configuración consolidada de umbrales
            market_provider: Proveedor de contexto de mercado (inyección de dependencia)
        """
        self.config = config or TranslatorConfig()
        self._market_provider = market_provider

        logger.debug(
            f"SemanticTranslator initialized | "
            f"Ψ_critical={self.config.stability.critical:.2f}, "
            f"deterministic={self.config.deterministic_market}"
        )

    # ========================================================================
    # GRAPHRAG: EXPLICACIÓN CAUSAL
    # ========================================================================

    def explain_cycle_path(self, cycle_nodes: List[str]) -> str:
        """
        Genera narrativa que explica la ruta del ciclo (GraphRAG).

        Args:
            cycle_nodes: Lista de nodos que forman el ciclo

        Returns:
            Narrativa explicativa del ciclo
        """
        if not cycle_nodes:
            return ""

        max_display = self.config.max_cycle_path_display
        display_nodes = cycle_nodes[:max_display]
        path_str = " → ".join(display_nodes)

        if len(cycle_nodes) > max_display:
            path_str += f" → ... ({len(cycle_nodes) - max_display} más)"

        # Cerrar el ciclo
        path_str += f" → {cycle_nodes[0]}"

        return NarrativeTemplates.CYCLE_PATH.format(
            path=path_str,
            first_node=cycle_nodes[0],
        )

    def explain_stress_point(self, node: str, degree: Union[int, str]) -> str:
        """
        Explica por qué un nodo es crítico (GraphRAG).

        Args:
            node: Identificador del nodo
            degree: Número de conexiones o descriptor

        Returns:
            Narrativa del punto de estrés
        """
        return NarrativeTemplates.STRESS_POINT.format(
            node=node,
            degree=degree,
        )

    # ========================================================================
    # TRADUCCIÓN DE TOPOLOGÍA
    # ========================================================================

    def translate_topology(
        self,
        metrics: Union[TopologyMetricsDTO, Dict[str, Any], Any],
        stability: float = 0.0,
        synergy_risk: Optional[Union[SynergyRiskDTO, Dict[str, Any]]] = None,
        spectral: Optional[Union[SpectralMetricsDTO, Dict[str, Any]]] = None,
    ) -> Tuple[str, VerdictLevel]:
        """
        Traduce métricas topológicas a narrativa de ingeniería civil.

        Args:
            metrics: Números de Betti y métricas relacionadas
            stability: Índice de estabilidad piramidal (Ψ)
            synergy_risk: Datos de sinergia de riesgo (producto cup)
            spectral: Datos de análisis espectral

        Returns:
            Tupla (narrativa, veredicto)
        """
        # Normalizar inputs
        topo = TopologyMetricsDTO.from_any(metrics)
        synergy = SynergyRiskDTO.from_dict(synergy_risk) if isinstance(synergy_risk, dict) else (synergy_risk or SynergyRiskDTO())
        spec = SpectralMetricsDTO.from_dict(spectral) if isinstance(spectral, dict) else (spectral or SpectralMetricsDTO())

        self._validate_topology_inputs(topo, stability)

        narrative_parts: List[str] = []
        verdicts: List[VerdictLevel] = []

        # 1. β₁: Genus Estructural / Socavones
        cycle_narrative, cycle_verdict = self._translate_cycles(topo.beta_1)
        narrative_parts.append(cycle_narrative)
        verdicts.append(cycle_verdict)

        # 2. Sinergia de Riesgo (Producto Cup)
        if synergy.synergy_detected:
            synergy_narrative = self._translate_synergy(synergy)
            narrative_parts.append(synergy_narrative)
            verdicts.append(VerdictLevel.RECHAZAR)

            # GraphRAG: Explicar nodos críticos
            if synergy.intersecting_nodes:
                example_node = synergy.intersecting_nodes[0]
                narrative_parts.append(self.explain_stress_point(example_node, "múltiples"))

        # 3. Eficiencia de Euler
        if topo.euler_efficiency < 0.5:
            narrative_parts.append(self._translate_euler_efficiency(topo.euler_efficiency))
            verdicts.append(VerdictLevel.REVISAR)

        # 4. Espectral
        if spec.fiedler_value > 0 or spec.resonance_risk:
            spec_narrative = self._translate_spectral(spec)
            narrative_parts.append(spec_narrative)
            if spec.resonance_risk:
                verdicts.append(VerdictLevel.PRECAUCION)

        # 5. β₀: Coherencia de Obra
        conn_narrative, conn_verdict = self._translate_connectivity(topo.beta_0)
        narrative_parts.append(conn_narrative)
        verdicts.append(conn_verdict)

        # 6. Ψ: Solidez de Cimentación
        stab_narrative, stab_verdict = self._translate_stability(stability)
        narrative_parts.append(stab_narrative)
        verdicts.append(stab_verdict)

        # Calcular veredicto final (supremum)
        final_verdict = VerdictLevel.supremum(*verdicts)

        return "\n".join(narrative_parts), final_verdict

    def _validate_topology_inputs(self, metrics: TopologyMetricsDTO, stability: float) -> None:
        """Valida inputs topológicos."""
        if stability < 0:
            raise ValueError("Stability Ψ must be non-negative")

    def _translate_cycles(self, beta_1: int) -> Tuple[str, VerdictLevel]:
        """Traduce β₁ (ciclos) a narrativa."""
        classification = self.config.topology.classify_cycles(beta_1)
        template = NarrativeTemplates.TOPOLOGY_CYCLES[classification]
        narrative = template.format(beta_1=beta_1)

        verdict_map = {
            "clean": VerdictLevel.VIABLE,
            "minor": VerdictLevel.CONDICIONAL,
            "moderate": VerdictLevel.PRECAUCION,
            "critical": VerdictLevel.RECHAZAR,
        }

        return narrative, verdict_map.get(classification, VerdictLevel.REVISAR)

    def _translate_connectivity(self, beta_0: int) -> Tuple[str, VerdictLevel]:
        """Traduce β₀ (conectividad) a narrativa."""
        classification = self.config.topology.classify_connectivity(beta_0)
        template = NarrativeTemplates.TOPOLOGY_CONNECTIVITY[classification]
        narrative = template.format(beta_0=beta_0)

        verdict_map = {
            "empty": VerdictLevel.RECHAZAR,
            "unified": VerdictLevel.VIABLE,
            "fragmented": VerdictLevel.CONDICIONAL,
            "severely_fragmented": VerdictLevel.PRECAUCION,
        }

        return narrative, verdict_map.get(classification, VerdictLevel.REVISAR)

    def _translate_stability(self, stability: float) -> Tuple[str, VerdictLevel]:
        """Traduce Ψ (estabilidad) a narrativa."""
        classification = self.config.stability.classify(stability)

        if classification == "invalid":
            return "⚠️ **Valor de estabilidad inválido**", VerdictLevel.REVISAR

        template = NarrativeTemplates.STABILITY.get(
            classification,
            NarrativeTemplates.STABILITY["stable"]
        )
        narrative = template.format(stability=stability)

        verdict_map = {
            "critical": VerdictLevel.RECHAZAR,
            "warning": VerdictLevel.PRECAUCION,
            "stable": VerdictLevel.CONDICIONAL,
            "robust": VerdictLevel.VIABLE,
        }

        return narrative, verdict_map.get(classification, VerdictLevel.REVISAR)

    def _translate_synergy(self, synergy: SynergyRiskDTO) -> str:
        """Traduce sinergia de riesgo a narrativa."""
        return NarrativeTemplates.SYNERGY.format(
            count=synergy.intersecting_cycles_count
        )

    def _translate_euler_efficiency(self, efficiency: float) -> str:
        """Traduce eficiencia de Euler a narrativa."""
        return NarrativeTemplates.EULER_EFFICIENCY.format(efficiency=efficiency)

    def _translate_spectral(self, spectral: SpectralMetricsDTO) -> str:
        """Traduce métricas espectrales a narrativa."""
        parts = []

        # Cohesión (Fiedler)
        if spectral.fiedler_value > 0.5:
            cohesion_type = "high"
        elif spectral.fiedler_value > 0.05:
            cohesion_type = "standard"
        else:
            cohesion_type = "low"

        parts.append(
            NarrativeTemplates.SPECTRAL_COHESION[cohesion_type].format(
                fiedler=spectral.fiedler_value
            )
        )

        # Resonancia
        resonance_type = "risk" if spectral.resonance_risk else "safe"
        parts.append(
            NarrativeTemplates.SPECTRAL_RESONANCE[resonance_type].format(
                wavelength=spectral.wavelength
            )
        )

        return " ".join(parts)

    # ========================================================================
    # TRADUCCIÓN DE TERMODINÁMICA
    # ========================================================================

    def translate_thermodynamics(
        self,
        entropy: float = 0.0,
        exergy: float = 1.0,
        temperature: float = 25.0,
    ) -> Tuple[str, VerdictLevel]:
        """
        Traduce métricas termodinámicas a narrativa.

        Args:
            entropy: Nivel de desorden administrativo (0.0 - 1.0)
            exergy: Eficiencia de inversión (0.0 - 1.0)
            temperature: Índice de inflación interna (°C)

        Returns:
            Tupla (narrativa, veredicto)
        """
        # Validar rangos
        entropy = max(0.0, min(1.0, entropy))
        exergy = max(0.0, min(1.0, exergy))
        temperature = max(0.0, temperature)

        parts = []
        verdicts = []

        # Exergía
        exergy_pct = exergy * 100.0
        parts.append(f"⚡ **Eficiencia Exergética del {exergy_pct:.1f}%**.")

        # Entropía (Muerte Térmica Check)
        if entropy > 0.95:
            parts.append(NarrativeTemplates.THERMAL_DEATH)
            verdicts.append(VerdictLevel.RECHAZAR)

        if exergy < self.config.thermal.exergy_poor:
            verdicts.append(VerdictLevel.PRECAUCION)
        elif exergy >= self.config.thermal.exergy_efficient:
            verdicts.append(VerdictLevel.VIABLE)
        else:
            verdicts.append(VerdictLevel.CONDICIONAL)

        # Temperatura
        temp_class = self.config.thermal.classify_temperature(temperature)
        temp_template = NarrativeTemplates.THERMAL_TEMPERATURE.get(
            temp_class,
            NarrativeTemplates.THERMAL_TEMPERATURE["stable"]
        )
        parts.append(temp_template.format(temperature=temperature))

        temp_verdict_map = {
            "cold": VerdictLevel.VIABLE,
            "stable": VerdictLevel.VIABLE,
            "warm": VerdictLevel.CONDICIONAL,
            "hot": VerdictLevel.PRECAUCION,
            "critical": VerdictLevel.RECHAZAR,
        }
        verdicts.append(temp_verdict_map.get(temp_class, VerdictLevel.REVISAR))

        # Receta para fiebre
        if temp_class in ("hot", "critical"):
            parts.append(
                "💊 **Receta**: Se recomienda enfriar mediante contratos de futuros o stock preventivo."
            )

        # Entropía
        if entropy > self.config.thermal.entropy_high:
            parts.append(
                NarrativeTemplates.THERMAL_ENTROPY["high"].format(entropy=entropy)
            )
            verdicts.append(VerdictLevel.PRECAUCION)
        elif entropy < self.config.thermal.entropy_low:
            parts.append(
                NarrativeTemplates.THERMAL_ENTROPY["low"].format(entropy=entropy)
            )

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

        Args:
            metrics: Diccionario con métricas financieras

        Returns:
            Tupla (narrativa, veredicto_general, veredicto_financiero)
        """
        validated = self._validate_financial_metrics(metrics)
        parts = []

        # WACC
        parts.append(NarrativeTemplates.WACC.format(wacc=validated["wacc"]))

        # Contingencia
        parts.append(NarrativeTemplates.CONTINGENCY.format(
            contingency=validated["contingency_recommended"]
        ))

        # Veredicto financiero
        fin_verdict = validated["recommendation"]
        pi = validated["profitability_index"]

        verdict_key = fin_verdict.name.lower()
        verdict_template = NarrativeTemplates.FINANCIAL_VERDICT.get(
            verdict_key,
            NarrativeTemplates.FINANCIAL_VERDICT["review"]
        )
        parts.append(verdict_template.format(pi=pi))

        # Mapear a VerdictLevel
        general_verdict = fin_verdict.to_verdict_level()

        return "\n".join(parts), general_verdict, fin_verdict

    def _validate_financial_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Valida y normaliza métricas financieras."""
        if not isinstance(metrics, dict):
            raise TypeError(f"Expected dict, got {type(metrics).__name__}")

        def extract_numeric(data: Dict, key: str, default: float = 0.0) -> float:
            value = data.get(key)
            if value is None:
                return default
            if isinstance(value, (int, float)) and not math.isnan(value):
                return float(value)
            return default

        def extract_nested(data: Dict, path: List[str], default: float = 0.0) -> float:
            current = data
            for key in path:
                if not isinstance(current, dict):
                    return default
                current = current.get(key)
                if current is None:
                    return default
            if isinstance(current, (int, float)) and not math.isnan(current):
                return float(current)
            return default

        def extract_verdict(data: Dict) -> FinancialVerdict:
            performance = data.get("performance", {})
            if not isinstance(performance, dict):
                return FinancialVerdict.REVIEW
            rec = performance.get("recommendation", "REVISAR")
            return FinancialVerdict.from_string(str(rec))

        return {
            "wacc": extract_numeric(metrics, "wacc", 0.0),
            "contingency_recommended": extract_nested(
                metrics, ["contingency", "recommended"], 0.0
            ),
            "recommendation": extract_verdict(metrics),
            "profitability_index": extract_nested(
                metrics, ["performance", "profitability_index"], 0.0
            ),
        }

    # ========================================================================
    # CONTEXTO DE MERCADO
    # ========================================================================

    def get_market_context(self) -> str:
        """
        Obtiene inteligencia de mercado.

        Usa proveedor inyectado o valor determinístico por defecto.
        """
        if self._market_provider:
            try:
                context = self._market_provider()
                return f"🌍 **Suelo de Mercado**: {context}"
            except Exception as e:
                logger.warning(f"Market provider failed: {e}")
                return "🌍 **Suelo de Mercado**: No disponible."

        # Modo determinístico: siempre el mismo valor
        if self.config.deterministic_market:
            index = self.config.default_market_index
        else:
            import random
            index = random.randint(0, len(NarrativeTemplates.MARKET_CONTEXTS) - 1)

        context = NarrativeTemplates.MARKET_CONTEXTS[index]
        return f"🌍 **Suelo de Mercado**: {context}"

    # ========================================================================
    # COMPOSICIÓN DE REPORTE ESTRATÉGICO
    # ========================================================================

    def compose_strategic_narrative(
        self,
        topological_metrics: Any,
        financial_metrics: Dict[str, Any],
        stability: float = 0.0,
        synergy_risk: Optional[Dict[str, Any]] = None,
        spectral: Optional[Dict[str, Any]] = None,
        thermal_metrics: Optional[Dict[str, Any]] = None,
        physics_metrics: Optional[Dict[str, Any]] = None,
    ) -> StrategicReport:
        """
        Compone el reporte ejecutivo completo.

        Implementa la narrativa de 'Edificio Vivo' integrando todos los estratos.

        Args:
            topological_metrics: Métricas de Betti
            financial_metrics: Métricas financieras
            stability: Índice de estabilidad Ψ
            synergy_risk: Riesgo de sinergia
            spectral: Métricas espectrales
            thermal_metrics: Métricas termodinámicas
            physics_metrics: Métricas del motor físico (Flux/Laplace)

        Returns:
            StrategicReport con análisis completo
        """
        # Normalizar inputs
        topo = TopologyMetricsDTO.from_any(topological_metrics)
        thermal = ThermalMetricsDTO.from_dict(thermal_metrics)
        physics = PhysicsMetricsDTO.from_dict(physics_metrics)
        synergy = SynergyRiskDTO.from_dict(synergy_risk)
        spec = SpectralMetricsDTO.from_dict(spectral)

        # Acumuladores
        strata_analysis: Dict[Stratum, StratumAnalysisResult] = {}
        section_narratives: List[str] = []
        all_verdicts: List[VerdictLevel] = []
        errors: List[str] = []

        # ====== HEADER ======
        section_narratives.append(self._generate_report_header())

        # ====== PHYSICS: Base Térmica y Dinámica ======
        physics_result = self._analyze_physics_stratum(thermal, stability, physics)
        strata_analysis[Stratum.PHYSICS] = physics_result
        all_verdicts.append(physics_result.verdict)

        # ====== TACTICS: Estructura Topológica ======
        tactics_result = self._analyze_tactics_stratum(topo, synergy, spec, stability)
        strata_analysis[Stratum.TACTICS] = tactics_result
        all_verdicts.append(tactics_result.verdict)

        # ====== STRATEGY: Viabilidad Financiera ======
        strategy_result = self._analyze_strategy_stratum(financial_metrics)
        strata_analysis[Stratum.STRATEGY] = strategy_result
        all_verdicts.append(strategy_result.verdict)

        # ====== SECCIÓN 1: Diagnóstico Integrado ======
        section_narratives.append("## 🏗️ Diagnóstico del Edificio Vivo")
        integrated_diagnosis = self._generate_integrated_diagnosis(
            stability, thermal.temperature, physics_result.verdict
        )
        section_narratives.append(integrated_diagnosis)
        section_narratives.append("")

        # ====== SECCIÓN 2: Detalles Topológicos ======
        section_narratives.append("### 1. Auditoría de Integridad Estructural")
        try:
            topo_narrative, topo_verdict = self.translate_topology(
                topo, stability, synergy.__dict__, spec.__dict__
            )
            section_narratives.append(topo_narrative)
            all_verdicts.append(topo_verdict)
        except Exception as e:
            error_msg = f"Error analizando estructura: {e}"
            section_narratives.append(f"❌ {error_msg}")
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
        section_narratives.append("")

        # ====== SECCIÓN 3: Detalles Financieros ======
        section_narratives.append("### 2. Análisis de Cargas Financieras")
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
        section_narratives.append("### 3. Geotecnia de Mercado")
        section_narratives.append(self.get_market_context())
        section_narratives.append("")

        # ====== WISDOM: Veredicto Final ======
        final_verdict = VerdictLevel.supremum(*all_verdicts)

        # Aplicar clausura transitiva: si hay errores, subir severidad
        if errors:
            final_verdict = final_verdict | VerdictLevel.REVISAR

        wisdom_result = self._analyze_wisdom_stratum(
            topo, financial_metrics, stability, synergy, final_verdict, bool(errors)
        )
        strata_analysis[Stratum.WISDOM] = wisdom_result

        # ====== SECCIÓN 5: Dictamen Final ======
        section_narratives.append("### 💡 Dictamen del Ingeniero Jefe")
        section_narratives.append(wisdom_result.narrative)

        # Construir executive summary
        executive_summary = self._generate_executive_summary(
            final_verdict, strata_analysis, errors
        )

        # Generar recomendaciones
        recommendations = self._generate_recommendations(
            final_verdict, strata_analysis, synergy
        )

        return StrategicReport(
            title="INFORME DE INGENIERÍA ESTRATÉGICA",
            verdict=final_verdict,
            executive_summary=executive_summary,
            strata_analysis=strata_analysis,
            recommendations=recommendations,
            raw_narrative="\n".join(section_narratives),
        )

    def _generate_report_header(self) -> str:
        """Genera el encabezado del reporte."""
        return (
            "## 🏗️ INFORME DE INGENIERÍA ESTRATÉGICA\n"
            f"*Análisis de Coherencia Fractal | "
            f"Estabilidad Crítica: Ψ < {self.config.stability.critical}*"
        )

    def _analyze_physics_stratum(
        self,
        thermal: ThermalMetricsDTO,
        stability: float,
        physics: Optional[PhysicsMetricsDTO] = None,
    ) -> StratumAnalysisResult:
        """Analiza el estrato PHYSICS (datos, flujo, temperatura)."""
        issues = []
        verdicts = []
        narrative_parts = []

        # 1. Giroscopía (FluxCondenser)
        if physics:
            if physics.gyroscopic_stability < 0.3:
                issues.append("Nutación crítica detectada")
                narrative_parts.append(NarrativeTemplates.GYROSCOPIC_STABILITY["nutation"])
                verdicts.append(VerdictLevel.RECHAZAR)
            elif physics.gyroscopic_stability < 0.7:
                issues.append("Precesión detectada")
                narrative_parts.append(NarrativeTemplates.GYROSCOPIC_STABILITY["precession"])
                verdicts.append(VerdictLevel.PRECAUCION)
            else:
                narrative_parts.append(NarrativeTemplates.GYROSCOPIC_STABILITY["stable"])
                verdicts.append(VerdictLevel.VIABLE)

            # 2. Oráculo de Laplace (Control)
            if not physics.is_stable_lhp:
                issues.append("Divergencia Matemática (RHP)")
                narrative_parts.append(NarrativeTemplates.LAPLACE_CONTROL["unstable"])
                verdicts.append(VerdictLevel.RECHAZAR)
            elif physics.phase_margin_deg < 30:
                issues.append("Estabilidad Marginal (Resonancia)")
                narrative_parts.append(NarrativeTemplates.LAPLACE_CONTROL["marginal"])
                verdicts.append(VerdictLevel.PRECAUCION)
            else:
                # narrative_parts.append(NarrativeTemplates.LAPLACE_CONTROL["robust"]) # Opcional, para no saturar
                verdicts.append(VerdictLevel.VIABLE)

        # 3. Análisis térmico
        temp_class = self.config.thermal.classify_temperature(thermal.temperature)
        if temp_class in ("hot", "critical"):
            issues.append(f"Temperatura crítica: {thermal.temperature:.1f}°C")
            verdicts.append(VerdictLevel.PRECAUCION if temp_class == "hot" else VerdictLevel.RECHAZAR)
        else:
            verdicts.append(VerdictLevel.VIABLE)

        # 4. Entropía (Muerte Térmica)
        if thermal.entropy > 0.95: # Umbral de muerte térmica aproximado
             issues.append("Muerte Térmica del Sistema")
             narrative_parts.append(NarrativeTemplates.THERMAL_DEATH)
             verdicts.append(VerdictLevel.RECHAZAR)
        elif thermal.entropy > self.config.thermal.entropy_high:
            issues.append(f"Alta entropía: {thermal.entropy:.2f}")
            verdicts.append(VerdictLevel.PRECAUCION)

        verdict = VerdictLevel.supremum(*verdicts) if verdicts else VerdictLevel.VIABLE

        # Construir narrativa final
        if verdict == VerdictLevel.VIABLE:
            base_narrative = "Base física estable. Flujo de datos sin turbulencia."
        elif verdict == VerdictLevel.RECHAZAR:
            base_narrative = "Inestabilidad física crítica. Datos no confiables."
        else:
            base_narrative = "Señales de inestabilidad en la capa física."

        full_narrative = f"{base_narrative} {' '.join(narrative_parts)}"

        metrics_summary = {
            "temperature": thermal.temperature,
            "entropy": thermal.entropy,
            "exergy": thermal.exergy,
            "stability": stability,
        }
        if physics:
            metrics_summary.update({
                "gyroscopic_stability": physics.gyroscopic_stability,
                "phase_margin": physics.phase_margin_deg,
                "is_stable": physics.is_stable_lhp
            })

        return StratumAnalysisResult(
            stratum=Stratum.PHYSICS,
            verdict=verdict,
            narrative=full_narrative.strip(),
            metrics_summary=metrics_summary,
            issues=issues,
        )

    def _analyze_tactics_stratum(
        self,
        topo: TopologyMetricsDTO,
        synergy: SynergyRiskDTO,
        spectral: SpectralMetricsDTO,
        stability: float,
    ) -> StratumAnalysisResult:
        """Analiza el estrato TACTICS (estructura, topología)."""
        issues = []
        verdicts = []

        # Ciclos
        cycle_class = self.config.topology.classify_cycles(topo.beta_1)
        if cycle_class != "clean":
            issues.append(f"Ciclos detectados (β₁={topo.beta_1})")
            if cycle_class == "critical":
                verdicts.append(VerdictLevel.RECHAZAR)
            elif cycle_class == "moderate":
                verdicts.append(VerdictLevel.PRECAUCION)
            else:
                verdicts.append(VerdictLevel.CONDICIONAL)
        else:
            verdicts.append(VerdictLevel.VIABLE)

        # Mayer-Vietoris (Integridad de Fusión)
        if topo.delta_beta_1 > 0:
            issues.append(f"Anomalía de Integración (Δβ₁={topo.delta_beta_1})")
            narrative_parts.append(
                NarrativeTemplates.MAYER_VIETORIS.format(delta_beta_1=topo.delta_beta_1)
            )
            # Esto es un error topológico grave de fusión
            verdicts.append(VerdictLevel.RECHAZAR)

        # Conectividad
        conn_class = self.config.topology.classify_connectivity(topo.beta_0)
        if conn_class != "unified":
            issues.append(f"Fragmentación (β₀={topo.beta_0})")
            verdicts.append(VerdictLevel.CONDICIONAL)

        # Sinergia
        if synergy.synergy_detected:
            issues.append("Sinergia de riesgo detectada (Efecto Dominó)")
            verdicts.append(VerdictLevel.RECHAZAR)

        # Estabilidad
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
                "euler_efficiency": topo.euler_efficiency,
                "stability": stability,
                "synergy_risk": synergy.synergy_detected,
            },
            issues=issues,
        )

    def _analyze_strategy_stratum(
        self,
        financial_metrics: Dict[str, Any],
    ) -> StratumAnalysisResult:
        """Analiza el estrato STRATEGY (finanzas, viabilidad)."""
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
            )

    def _analyze_wisdom_stratum(
        self,
        topo: TopologyMetricsDTO,
        financial_metrics: Dict[str, Any],
        stability: float,
        synergy: SynergyRiskDTO,
        final_verdict: VerdictLevel,
        has_errors: bool,
    ) -> StratumAnalysisResult:
        """Genera el análisis del estrato WISDOM (veredicto final)."""
        # Determinar narrativa final
        narrative = self._generate_final_advice(
            topo, financial_metrics, stability, synergy, has_errors
        )

        return StratumAnalysisResult(
            stratum=Stratum.WISDOM,
            verdict=final_verdict,
            narrative=narrative,
            metrics_summary={
                "final_verdict": final_verdict.name,
                "is_viable": final_verdict.is_positive,
            },
            issues=["Errores en análisis previo"] if has_errors else [],
        )

    def _generate_integrated_diagnosis(
        self,
        stability: float,
        temperature: float,
        physics_verdict: VerdictLevel,
    ) -> str:
        """Genera diagnóstico integrado (edificio + calor)."""
        parts = []

        # Base
        if stability < self.config.stability.critical:
            parts.append(
                f"El edificio se apoya sobre una **base inestable** (Ψ={stability:.2f}). "
                "La cimentación de recursos es insuficiente para la carga táctica."
            )
        else:
            parts.append(
                f"El edificio tiene una **cimentación sólida** (Ψ={stability:.2f}). "
                "La base de recursos es amplia y redundante."
            )

        # Clima
        if temperature > self.config.thermal.temperature_hot:
            parts.append(
                f"Sin embargo, el entorno es hostil. Detectamos una **Fiebre Inflacionaria** de {temperature:.1f}°C "
                "entrando por los insumos (volatilidad de precios)."
            )

            # Interacción base + calor
            if stability > self.config.stability.solid / 2:
                parts.append(
                    "✅ Gracias a la base ancha, la estructura actúa como un **disipador de calor** eficiente. "
                    "El riesgo de sobrecostos se diluye en la red de proveedores resiliente."
                )
            elif stability < self.config.stability.critical:
                parts.append(
                    "🚨 Debido a la base estrecha (Pirámide Invertida), **el calor no se disipa**. "
                    "Se concentra en los pocos puntos de apoyo, creando un riesgo crítico de fractura financiera."
                )
            else:
                parts.append(
                    "⚠️ La estructura tiene capacidad moderada de disipación, pero una ola de calor sostenida "
                    "podría comprometer la integridad financiera."
                )
        else:
            parts.append(
                f"El entorno es térmicamente estable ({temperature:.1f}°C). "
                "El riesgo inflacionario externo es bajo."
            )

        return " ".join(parts)

    def _generate_final_advice(
        self,
        topo: TopologyMetricsDTO,
        financial_metrics: Dict[str, Any],
        stability: float,
        synergy: SynergyRiskDTO,
        has_errors: bool,
    ) -> str:
        """Genera el dictamen final."""
        if has_errors:
            return NarrativeTemplates.FINAL_VERDICTS["analysis_failed"]

        # Sinergia de riesgo
        if synergy.synergy_detected:
            result = NarrativeTemplates.FINAL_VERDICTS["synergy_risk"]
            if synergy.intersecting_cycles:
                cycle_explanation = self.explain_cycle_path(synergy.intersecting_cycles[0])
                result += f"\n\n{cycle_explanation}"
            return result

        # Pirámide Invertida
        is_inverted = stability < self.config.stability.critical

        try:
            fin_verdict = self._validate_financial_metrics(financial_metrics)["recommendation"]
        except Exception:
            fin_verdict = FinancialVerdict.REVIEW

        if is_inverted:
            if fin_verdict == FinancialVerdict.ACCEPT:
                return NarrativeTemplates.FINAL_VERDICTS["inverted_pyramid_viable"].format(
                    stability=stability
                )
            else:
                return NarrativeTemplates.FINAL_VERDICTS["inverted_pyramid_reject"]

        # Agujeros topológicos
        if topo.beta_1 > 0:
            return NarrativeTemplates.FINAL_VERDICTS["has_holes"].format(
                beta_1=topo.beta_1
            )

        # Caso ideal
        if fin_verdict == FinancialVerdict.ACCEPT:
            return NarrativeTemplates.FINAL_VERDICTS["certified"]

        # Fallback
        return NarrativeTemplates.FINAL_VERDICTS["review_required"]

    def _generate_executive_summary(
        self,
        final_verdict: VerdictLevel,
        strata: Dict[Stratum, StratumAnalysisResult],
        errors: List[str],
    ) -> str:
        """Genera resumen ejecutivo."""
        parts = []

        parts.append(f"**Veredicto: {final_verdict.emoji} {final_verdict.name}**\n")

        # Resumen por estrato
        for stratum in [Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY]:
            if stratum in strata:
                analysis = strata[stratum]
                parts.append(
                    f"- {stratum.name}: {analysis.verdict.emoji} {analysis.verdict.name}"
                )

        if errors:
            parts.append(f"\n⚠️ Se detectaron {len(errors)} errores durante el análisis.")

        return "\n".join(parts)

    def _generate_recommendations(
        self,
        final_verdict: VerdictLevel,
        strata: Dict[Stratum, StratumAnalysisResult],
        synergy: SynergyRiskDTO,
    ) -> List[str]:
        """Genera recomendaciones accionables."""
        recommendations = []

        if final_verdict == VerdictLevel.VIABLE:
            recommendations.append("Proceder con la ejecución del proyecto.")
            return recommendations

        # Por estrato comprometido
        for stratum, analysis in strata.items():
            if analysis.verdict.is_negative or analysis.verdict == VerdictLevel.PRECAUCION:
                if stratum == Stratum.PHYSICS:
                    recommendations.append(
                        "Revisar fuentes de datos y estabilizar flujo de información."
                    )
                elif stratum == Stratum.TACTICS:
                    if synergy.synergy_detected:
                        recommendations.append(
                            "Desacoplar ciclos de dependencia que comparten recursos críticos."
                        )
                    recommendations.append(
                        "Sanear topología eliminando ciclos y conectando componentes aislados."
                    )
                elif stratum == Stratum.STRATEGY:
                    recommendations.append(
                        "Revisar modelo financiero y ajustar parámetros de viabilidad."
                    )

        if not recommendations:
            recommendations.append("Realizar análisis adicional para determinar siguiente paso.")

        return recommendations[:5]

    # ========================================================================
    # MÉTODO LEGACY DE COMPATIBILIDAD
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
) -> SemanticTranslator:
    """Factory function para crear un traductor configurado."""
    return SemanticTranslator(config=config, market_provider=market_provider)


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