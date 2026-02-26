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
    ) -> None:
        """
        Inicializa el traductor.

        Args:
            config: Configuración consolidada de umbrales
            market_provider: Proveedor de contexto de mercado (inyección de dependencia)
            mic: Registro MIC para obtener narrativas (opcional, se crea default si falta)
        """
        self.config = config or TranslatorConfig()
        self._market_provider = market_provider

        if mic:
            self.mic = mic
        else:
            # Fallback for legacy tests / standalone usage
            self.mic = MICRegistry()
            # Register core vectors to ensure SemanticDictionary is available
            register_core_vectors(self.mic, config={})

        logger.debug(
            f"SemanticTranslator initialized | "
            f"Ψ_critical={self.config.stability.critical:.2f}, "
            f"deterministic={self.config.deterministic_market}"
        )

    def _fetch_narrative(self, domain: str, classification: str, params: Dict[str, Any] = None) -> str:
        """Helper to fetch narrative from MIC."""
        # Use force_physics_override to access WISDOM layer (Dictionary) even if lower strata fail
        response = self.mic.project_intent(
            "fetch_narrative",
            {
                "domain": domain,
                "classification": classification,
                "params": params or {}
            },
            {"force_physics_override": True}
        )
        return response.get("narrative", f"[{domain}.{classification}]")

    # ========================================================================
    # GRAPHRAG: EXPLICACIÓN CAUSAL
    # ========================================================================

    def explain_cycle_path(self, cycle_nodes: List[str]) -> str:
        """
        Genera narrativa que explica la ruta del ciclo (GraphRAG).
        Delega la explicación causal al Diccionario vía MIC.
        """
        if not cycle_nodes:
            return ""

        max_display = self.config.max_cycle_path_display

        display_nodes = cycle_nodes[:max_display]
        if len(cycle_nodes) > max_display:
            display_nodes.append(f"... ({len(cycle_nodes) - max_display} más)")

        # Ensure cycle closure in display
        if cycle_nodes and display_nodes[-1] != cycle_nodes[0]:
             display_nodes.append(cycle_nodes[0])

        payload = {
            "anomaly_type": "CYCLE",
            "path_nodes": display_nodes
        }

        # Proyección a la MIC invocando GraphRAG
        response = self.mic.project_intent(
            "project_graph_narrative",
            payload,
            {"force_physics_override": True}
        )

        return response.get("narrative", "") if response.get("success") else ""

    def explain_stress_point(self, node_id: str, degree: Union[int, str], stratum: Stratum = Stratum.PHYSICS) -> str:
        """Explica por qué un nodo es crítico (GraphRAG) delegando a la MIC."""

        # Robust handling for degree if string is passed (e.g. "múltiples")
        safe_degree = 0
        if isinstance(degree, int):
            safe_degree = degree
        elif isinstance(degree, str):
            if degree.isdigit():
                safe_degree = int(degree)
            else:
                # "múltiples" or other qualitative descriptors imply high connectivity
                safe_degree = 10

        payload = {
            "anomaly_type": "STRESS",
            "vector": {
                "node_id": node_id,
                "node_type": "INSUMO", # Default as stress points usually in physics
                "stratum": stratum.value,
                "in_degree": safe_degree,
                "out_degree": 0
            }
        }
        response = self.mic.project_intent(
            "project_graph_narrative",
            payload,
            {"force_physics_override": True}
        )
        return response.get("narrative", "") if response.get("success") else ""

    # ========================================================================
    # TRADUCCIÓN DE TOPOLOGÍA
    # ========================================================================

    def translate_topology(
        self,
        metrics: Union[TopologicalMetrics, Dict[str, Any], Any],
        stability: float = 0.0,
        synergy_risk: Optional[Dict[str, Any]] = None,
        spectral: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, VerdictLevel]:
        """
        Traduce métricas topológicas a narrativa de ingeniería civil.
        """
        # Normalizar inputs
        if isinstance(metrics, dict):
            topo = TopologicalMetrics(
                beta_0=int(metrics.get("beta_0", 1)),
                beta_1=int(metrics.get("beta_1", 0)),
                beta_2=int(metrics.get("beta_2", 0)),
                euler_characteristic=int(metrics.get("euler_characteristic", 1)),
                fiedler_value=float(metrics.get("fiedler_value", 1.0)),
                spectral_gap=float(metrics.get("spectral_gap", 0.0)),
                pyramid_stability=float(metrics.get("pyramid_stability", 1.0)),
                structural_entropy=float(metrics.get("structural_entropy", 0.0)),
            )
        elif isinstance(metrics, TopologicalMetrics):
            topo = metrics
        else:
            # Fallback
            topo = TopologicalMetrics()

        synergy = synergy_risk or {}
        spec = spectral or {}

        # Validar estabilidad (usar pyramid_stability si no se pasa explícitamente)
        eff_stability = stability if stability != 0.0 else topo.pyramid_stability
        self._validate_topology_inputs(topo, eff_stability)

        narrative_parts: List[str] = []
        verdicts: List[VerdictLevel] = []

        # 1. β₁: Genus Estructural / Socavones
        cycle_narrative, cycle_verdict = self._translate_cycles(topo.beta_1)
        narrative_parts.append(cycle_narrative)
        verdicts.append(cycle_verdict)

        # 2. Sinergia de Riesgo (Producto Cup)
        synergy_detected = bool(synergy.get("synergy_detected", False))
        if synergy_detected:
            synergy_narrative = self._translate_synergy(synergy)
            narrative_parts.append(synergy_narrative)
            verdicts.append(VerdictLevel.RECHAZAR)

            # GraphRAG: Explicar nodos críticos
            bridge_nodes = synergy.get("bridge_nodes", [])
            if bridge_nodes:
                # bridge_nodes is list of dicts with 'id' key
                example_node = bridge_nodes[0].get("id") if isinstance(bridge_nodes[0], dict) else str(bridge_nodes[0])
                narrative_parts.append(self.explain_stress_point(example_node, "múltiples"))

        # 4. Espectral
        fiedler = topo.fiedler_value
        resonance_risk = bool(spec.get("resonance_risk", False))
        wavelength = float(spec.get("wavelength", 0.0))

        if fiedler > 0 or resonance_risk:
            spec_narrative = self._translate_spectral(fiedler, wavelength, resonance_risk)
            narrative_parts.append(spec_narrative)
            if resonance_risk:
                verdicts.append(VerdictLevel.PRECAUCION)

        # 5. β₀: Coherencia de Obra
        conn_narrative, conn_verdict = self._translate_connectivity(topo.beta_0)
        narrative_parts.append(conn_narrative)
        verdicts.append(conn_verdict)

        # 6. Ψ: Solidez de Cimentación
        stab_narrative, stab_verdict = self._translate_stability(eff_stability)
        narrative_parts.append(stab_narrative)
        verdicts.append(stab_verdict)

        # Calcular veredicto final (supremum)
        final_verdict = VerdictLevel.supremum(*verdicts)

        return "\n".join(narrative_parts), final_verdict

    def _validate_topology_inputs(self, metrics: TopologicalMetrics, stability: float) -> None:
        """Valida inputs topológicos."""
        if stability < 0:
            raise ValueError("Stability Ψ must be non-negative")

    def _translate_cycles(self, beta_1: int) -> Tuple[str, VerdictLevel]:
        """Traduce β₁ (ciclos) a narrativa."""
        classification = self.config.topology.classify_cycles(beta_1)
        narrative = self._fetch_narrative("TOPOLOGY_CYCLES", classification, {"beta_1": beta_1})

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
        narrative = self._fetch_narrative("TOPOLOGY_CONNECTIVITY", classification, {"beta_0": beta_0})

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

        narrative = self._fetch_narrative("STABILITY", classification, {"stability": stability})

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

    def _translate_euler_efficiency(self, efficiency: float) -> str:
        """Traduce eficiencia de Euler a narrativa."""
        return self._fetch_narrative("MISC", "EULER_EFFICIENCY", {"efficiency": efficiency})

    def _translate_spectral(self, fiedler: float, wavelength: float, resonance_risk: bool) -> str:
        """Traduce métricas espectrales a narrativa."""
        parts = []

        # Cohesión (Fiedler)
        if fiedler > 0.5:
            cohesion_type = "high"
        elif fiedler > 0.05:
            cohesion_type = "standard"
        else:
            cohesion_type = "low"

        parts.append(
            self._fetch_narrative("SPECTRAL_COHESION", cohesion_type, {"fiedler": fiedler})
        )

        # Resonancia
        resonance_type = "risk" if resonance_risk else "safe"
        parts.append(
            self._fetch_narrative("SPECTRAL_RESONANCE", resonance_type, {"wavelength": wavelength})
        )

        return " ".join(parts)

    # ========================================================================
    # TRADUCCIÓN DE TERMODINÁMICA
    # ========================================================================

    def translate_thermodynamics(
        self,
        metrics: Union[ThermodynamicMetrics, Dict[str, Any], Any],
    ) -> Tuple[str, VerdictLevel]:
        """
        Traduce métricas termodinámicas a narrativa.
        """
        # Normalizar inputs
        if isinstance(metrics, dict):
            temp = float(metrics.get("system_temperature", metrics.get("temperature", 25.0)))
            thermo = ThermodynamicMetrics(
                system_temperature=temp,
                entropy=float(metrics.get("entropy", 0.0)),
                heat_capacity=float(metrics.get("heat_capacity", 0.5)),
            )
        elif isinstance(metrics, ThermodynamicMetrics):
            thermo = metrics
        else:
            thermo = ThermodynamicMetrics()

        entropy = thermo.entropy
        exergy = thermo.exergetic_efficiency
        temperature_k = thermo.system_temperature

        temperature_c = temperature_k - 273.15 if temperature_k > 0 else 0.0

        entropy = max(0.0, min(1.0, entropy))
        exergy = max(0.0, min(1.0, exergy))
        temperature_c = max(-273.15, temperature_c)

        parts = []
        verdicts = []

        # Exergía
        exergy_pct = exergy * 100.0
        parts.append(f"⚡ **Eficiencia Exergética del {exergy_pct:.1f}%**.")

        # Entropía (Muerte Térmica Check)
        if entropy > 0.95:
            parts.append(self._fetch_narrative("MISC", "THERMAL_DEATH"))
            verdicts.append(VerdictLevel.RECHAZAR)

        if exergy < self.config.thermal.exergy_poor:
            verdicts.append(VerdictLevel.PRECAUCION)
        elif exergy >= self.config.thermal.exergy_efficient:
            verdicts.append(VerdictLevel.VIABLE)
        else:
            verdicts.append(VerdictLevel.CONDICIONAL)

        # Temperatura
        temp_class = self.config.thermal.classify_temperature(temperature_c)
        parts.append(
            self._fetch_narrative("THERMAL_TEMPERATURE", temp_class, {"temperature": temperature_c})
        )

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
                self._fetch_narrative("THERMAL_ENTROPY", "high", {"entropy": entropy})
            )
            verdicts.append(VerdictLevel.PRECAUCION)
        elif entropy < self.config.thermal.entropy_low:
            parts.append(
                self._fetch_narrative("THERMAL_ENTROPY", "low", {"entropy": entropy})
            )

        # Inercia (Capacidad Calorífica)
        if thermo.heat_capacity < 0.2:
            parts.append(
                "🍂 **Hoja al Viento**: Baja inercia financiera (C_v < 0.2). Riesgo de volatilidad extrema."
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
        """
        validated = self._validate_financial_metrics(metrics)
        parts = []

        # WACC
        parts.append(
            self._fetch_narrative("MISC", "WACC", {"wacc": validated["wacc"]})
        )

        # Contingencia
        parts.append(
            self._fetch_narrative("MISC", "CONTINGENCY", {"contingency": validated["contingency_recommended"]})
        )

        # Veredicto financiero
        fin_verdict = validated["recommendation"]
        pi = validated["profitability_index"]

        verdict_key = fin_verdict.name.lower()
        parts.append(
            self._fetch_narrative("FINANCIAL_VERDICT", verdict_key, {"pi": pi})
        )

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
        """
        if self._market_provider:
            try:
                context = self._market_provider()
                return f"🌍 **Suelo de Mercado**: {context}"
            except Exception as e:
                logger.warning(f"Market provider failed: {e}")
                return "🌍 **Suelo de Mercado**: No disponible."

        # Modo determinístico o random via MIC
        params = {"deterministic": self.config.deterministic_market, "index": self.config.default_market_index}
        response = self.mic.project_intent("fetch_narrative", {"domain": "MARKET_CONTEXT", "params": params}, {})
        context = response.get("narrative", "")

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
        control_metrics: Optional[Dict[str, Any]] = None,
        critical_resources: Optional[List[Dict[str, Any]]] = None,
        raw_cycles: Optional[List[List[str]]] = None,
        **kwargs: Any,
    ) -> StrategicReport:
        """
        Compone el reporte ejecutivo completo.
        """
        # Normalizar inputs
        # Topología
        if isinstance(topological_metrics, TopologicalMetrics):
            topo = topological_metrics
        elif isinstance(topological_metrics, dict):
            topo = TopologicalMetrics(**{k: v for k, v in topological_metrics.items() if k in TopologicalMetrics.__annotations__})
        else:
            topo = TopologicalMetrics()

        # Termodinámica
        tm = thermal_metrics or {}
        if isinstance(tm, ThermodynamicMetrics):
            thermal = tm
        else:
            thermal = ThermodynamicMetrics(**{k: v for k, v in tm.items() if k in ThermodynamicMetrics.__annotations__})

        # Física
        pm = physics_metrics or {}
        if isinstance(pm, PhysicsMetrics):
            physics = pm
        else:
            physics = PhysicsMetrics(**{k: v for k, v in pm.items() if k in PhysicsMetrics.__annotations__})

        # Control
        cm = control_metrics or kwargs.get("control") or {}
        if isinstance(cm, ControlMetrics):
            control = cm
        else:
            # Intentar extraer de physics_metrics legacy si existe
            if not cm and isinstance(pm, dict):
                cm = {
                    "is_stable": pm.get("is_stable_lhp", True),
                    "phase_margin_deg": pm.get("phase_margin_deg", 45.0),
                }
            control = ControlMetrics(**{k: v for k, v in cm.items() if k in ControlMetrics.__annotations__})

        synergy = synergy_risk or {}
        spec = spectral or {}

        # Acumuladores
        strata_analysis: Dict[Stratum, StratumAnalysisResult] = {}
        section_narratives: List[str] = []
        all_verdicts: List[VerdictLevel] = []
        errors: List[str] = []

        # ====== HEADER ======
        section_narratives.append(self._generate_report_header())

        # ====== PHYSICS: Base Térmica y Dinámica ======
        physics_result = self._analyze_physics_stratum(thermal, stability, physics, control)
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

        temp_k = thermal.system_temperature
        temp_c = temp_k - 273.15 if temp_k > 0 else 0.0

        integrated_diagnosis = self._generate_integrated_diagnosis(
            stability, temp_c, physics_result.verdict
        )
        section_narratives.append(integrated_diagnosis)
        section_narratives.append("")

        # ====== SECCIÓN 1.5: Dinámica de Bombeo (Física) ======
        section_narratives.append("### 0. Dinámica de Bombeo (Física)")
        section_narratives.append(physics_result.narrative)
        section_narratives.append("")

        # ====== SECCIÓN 2: Detalles Topológicos ======
        section_narratives.append("### 1. Auditoría de Integridad Estructural")
        try:
            topo_narrative, topo_verdict = self.translate_topology(
                topo, stability, synergy, spec
            )
            section_narratives.append(topo_narrative)
            all_verdicts.append(topo_verdict)

            # Phase 4: Iterate over critical resources (GraphRAG)
            if critical_resources:
                # We show up to config.max_stress_points_display resources
                limit = self.config.max_stress_points_display
                for i, resource in enumerate(critical_resources[:limit]):
                    node_id = resource.get("id")
                    degree = resource.get("in_degree", 0)
                    if node_id:
                        explanation = self.explain_stress_point(node_id, degree)
                        if explanation:
                            section_narratives.append(f"- {explanation}")

            # Phase 4: Iterate over cycles (GraphRAG)
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
        thermal: ThermodynamicMetrics,
        stability: float,
        physics: Optional[PhysicsMetrics] = None,
        control: Optional[ControlMetrics] = None,
    ) -> StratumAnalysisResult:
        """Analiza el estrato PHYSICS (datos, flujo, temperatura)."""
        issues = []
        verdicts = []
        narrative_parts = []

        # 1. Giroscopía (FluxCondenser)
        if physics:
            if physics.gyroscopic_stability < 0.3:
                issues.append("Nutación crítica detectada")
                narrative_parts.append(self._fetch_narrative("GYROSCOPIC_STABILITY", "nutation"))
                verdicts.append(VerdictLevel.RECHAZAR)
            elif physics.gyroscopic_stability < 0.7:
                issues.append("Precesión detectada")
                narrative_parts.append(self._fetch_narrative("GYROSCOPIC_STABILITY", "precession"))
                verdicts.append(VerdictLevel.PRECAUCION)
            else:
                narrative_parts.append(self._fetch_narrative("GYROSCOPIC_STABILITY", "stable"))
                verdicts.append(VerdictLevel.VIABLE)

        # 2. Oráculo de Laplace (Control)
        if control:
            if not control.is_stable:
                issues.append("Divergencia Matemática (RHP)")
                narrative_parts.append(self._fetch_narrative("LAPLACE_CONTROL", "unstable"))
                verdicts.append(VerdictLevel.RECHAZAR)
            elif control.phase_margin_deg < 30:
                issues.append("Estabilidad Marginal (Resonancia)")
                narrative_parts.append(self._fetch_narrative("LAPLACE_CONTROL", "marginal"))
                verdicts.append(VerdictLevel.PRECAUCION)
            else:
                verdicts.append(VerdictLevel.VIABLE)

        # 3. Análisis térmico
        temp_k = thermal.system_temperature
        temp_c = temp_k - 273.15 if temp_k > 0 else 0.0

        temp_class = self.config.thermal.classify_temperature(temp_c)
        if temp_class in ("hot", "critical"):
            issues.append(f"Temperatura crítica: {temp_c:.1f}°C")
            verdicts.append(VerdictLevel.PRECAUCION if temp_class == "hot" else VerdictLevel.RECHAZAR)
        else:
            verdicts.append(VerdictLevel.VIABLE)

        # 4. Entropía (Muerte Térmica)
        if thermal.entropy > 0.95:
             issues.append("Muerte Térmica del Sistema")
             narrative_parts.append(self._fetch_narrative("MISC", "THERMAL_DEATH"))
             verdicts.append(VerdictLevel.RECHAZAR)
        elif thermal.entropy > self.config.thermal.entropy_high:
            issues.append(f"Alta entropía: {thermal.entropy:.2f}")
            verdicts.append(VerdictLevel.PRECAUCION)

        # 5. Dinámica de Bombeo
        if physics:
            # water_hammer_pressure -> pressure (aproximación si no hay campo específico)
            if physics.pressure > 0.7:
                issues.append(f"Inestabilidad de Tubería (P={physics.pressure:.2f})")
                narrative_parts.append(
                    self._fetch_narrative("PUMP_DYNAMICS", "water_hammer", {"pressure": physics.pressure})
                )
                verdicts.append(VerdictLevel.PRECAUCION)

            # Presión del Acumulador
            narrative_parts.append(
                self._fetch_narrative("PUMP_DYNAMICS", "accumulator_pressure", {"pressure": physics.saturation * 100.0})
            )

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
            "temperature": thermal.system_temperature,
            "entropy": thermal.entropy,
            "exergy": thermal.exergy,
            "stability": stability,
        }
        if physics:
            metrics_summary.update({
                "gyroscopic_stability": physics.gyroscopic_stability,
            })
        if control:
            metrics_summary.update({
                "phase_margin": control.phase_margin_deg,
                "is_stable": control.is_stable
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
        topo: TopologicalMetrics,
        synergy: Dict[str, Any],
        spectral: Dict[str, Any],
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

        # Conectividad
        conn_class = self.config.topology.classify_connectivity(topo.beta_0)
        if conn_class != "unified":
            issues.append(f"Fragmentación (β₀={topo.beta_0})")
            verdicts.append(VerdictLevel.CONDICIONAL)

        # Sinergia
        synergy_detected = bool(synergy.get("synergy_detected", False))
        if synergy_detected:
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
                "euler_characteristic": topo.euler_characteristic,
                "stability": stability,
                "synergy_risk": synergy_detected,
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
        topo: TopologicalMetrics,
        financial_metrics: Dict[str, Any],
        stability: float,
        synergy: Dict[str, Any],
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
        topo: TopologicalMetrics,
        financial_metrics: Dict[str, Any],
        stability: float,
        synergy: Dict[str, Any],
        has_errors: bool,
    ) -> str:
        """Genera el dictamen final."""
        if has_errors:
            return self._fetch_narrative("FINAL_VERDICTS", "analysis_failed")

        # Sinergia de riesgo
        synergy_detected = bool(synergy.get("synergy_detected", False))
        if synergy_detected:
            result = self._fetch_narrative("FINAL_VERDICTS", "synergy_risk")
            intersecting_cycles = synergy.get("intersecting_cycles", [])
            if intersecting_cycles:
                cycle_explanation = self.explain_cycle_path(intersecting_cycles[0])
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
                return self._fetch_narrative("FINAL_VERDICTS", "inverted_pyramid_viable", {"stability": stability})
            else:
                return self._fetch_narrative("FINAL_VERDICTS", "inverted_pyramid_reject")

        # Agujeros topológicos
        if topo.beta_1 > 0:
            return self._fetch_narrative("FINAL_VERDICTS", "has_holes", {"beta_1": topo.beta_1})

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
        synergy: Dict[str, Any],
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
                    if synergy.get("synergy_detected", False):
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

    def assemble_data_product(self, graph: nx.DiGraph, report: Any) -> Dict[str, Any]:
        """
        Ensambla el Producto de Datos Final (Sabiduría).

        Integra la estructura (Grafo) y el análisis (Reporte) en un artefacto
        consumible por el negocio.

        Args:
            graph: Grafo del negocio.
            report: Reporte de riesgo (ConstructionRiskReport).

        Returns:
            Dict: Estructura JSON del producto de datos.
        """
        # Extraer narrativa y veredicto del reporte
        narrative = report.strategic_narrative if hasattr(report, "strategic_narrative") else ""
        verdict = report.integrity_score if hasattr(report, "integrity_score") else 0.0
        
        # Estructura del producto de datos
        product = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "verdict_score": verdict,
                "complexity_level": getattr(report, "complexity_level", "Unknown"),
            },
            "narrative": {
                "executive_summary": narrative,
                "alerts": getattr(report, "waste_alerts", []) + getattr(report, "circular_risks", []),
            },
            "topology": {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "details": getattr(report, "details", {}),
            },
            # "graph_data": nx.node_link_data(graph) # Opcional, si es serializable
        }
        return product

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
    mic: Optional[MICRegistry] = None,
) -> SemanticTranslator:
    """Factory function para crear un traductor configurado."""
    return SemanticTranslator(config=config, market_provider=market_provider, mic=mic)


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
