"""
Módulo de Narrativa de Telemetría Híbrido (Lattice + DIKW + Topología).

Extiende la lógica algebraica de severidad con una estructura ontológica piramidal.
Transforma la ejecución técnica en un 'Juicio del Consejo'.

Arquitectura Algebraica:
------------------------
1. Lattice de Severidad: (SeverityLevel, ≤, ⊔, ⊓) forma un lattice acotado completo
   - ⊥ (bottom) = OPTIMO
   - ⊤ (top) = CRITICO
   - ⊔ (join/supremum) = max severidad
   - ⊓ (meet/infimum) = min severidad

2. Filtración de Estratos: F₀ ⊂ F₁ ⊂ F₂ ⊂ F₃
   - WISDOM (0) ⊂ STRATEGY (1) ⊂ TACTICS (2) ⊂ PHYSICS (3)
   - Clausura Transitiva: Fallo en Fᵢ implica compromiso en Fⱼ para j < i

3. Grafo de Spans: Bosque con invariante χ = β₀

Referencias:
- LENGUAJE_CONSEJO.md [1]: Definición de la voz de los sabios.
- schemas.txt [2]: Definición del Enum Stratum.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from app.schemas import Stratum
from app.telemetry import StepStatus, TelemetryContext, TelemetrySpan
from app.tools_interface import MICRegistry, register_core_vectors

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================


class NarratorConfig:
    """Configuración centralizada del narrador."""

    # Límites de procesamiento
    MAX_RECURSION_DEPTH: int = 150
    MAX_FORENSIC_EVIDENCE: int = 10
    MAX_DISPLAYED_WARNINGS: int = 5
    MAX_ISSUES_PER_PHASE: int = 50
    MAX_PATH_LENGTH: int = 500

    # Umbrales de métricas anómalas
    ANOMALY_THRESHOLDS: Dict[str, float] = {
        "saturation": 0.9,
        "flyback_voltage": 0.5,
        "dissipated_power": 50.0,
        "error_rate": 0.1,
        "water_hammer_pressure": 0.7,
        "pump_work": 1000.0,  # Umbral arbitrario para alta inyección
    }

    # Tipos de issues que NO son críticos
    NON_CRITICAL_ISSUE_TYPES: FrozenSet[str] = frozenset({
        "Warning",
        "Info",
        "Metric",
        "MetricAnomaly",
        "Debug",
        "Trace",
    })


class StratumTopology:
    """
    Define la topología de la pirámide DIKW con orden explícito.
    
    Jerarquía (de base a cima):
    - PHYSICS (3): Base física, ingesta de datos
    - TACTICS (2): Estructura lógica, costos
    - STRATEGY (1): Valor financiero, riesgo
    - WISDOM (0): Síntesis final, respuesta
    
    Regla de Clausura: Fallo en nivel N invalida niveles < N
    """

    # Orden explícito: mayor número = más cerca de la base física
    HIERARCHY: Dict[Stratum, int] = {
        Stratum.PHYSICS: 3,   # Base de la pirámide
        Stratum.TACTICS: 2,
        Stratum.STRATEGY: 1,
        Stratum.WISDOM: 0,    # Cima de la pirámide
    }

    # Orden de evaluación: de base a cima
    EVALUATION_ORDER: Tuple[Stratum, ...] = (
        Stratum.PHYSICS,
        Stratum.TACTICS,
        Stratum.STRATEGY,
        Stratum.WISDOM,
    )

    # Mapeo de pasos a estratos (configurable)
    DEFAULT_STEP_MAPPING: Dict[str, Stratum] = {
        # PHYSICS - Ingesta y Estabilidad
        "load_data": Stratum.PHYSICS,
        "merge_data": Stratum.PHYSICS,
        "flux_condenser": Stratum.PHYSICS,
        "final_merge": Stratum.PHYSICS,
        "data_validation": Stratum.PHYSICS,
        "parse_input": Stratum.PHYSICS,

        # TACTICS - Estructura y Costos
        "calculate_costs": Stratum.TACTICS,
        "materialization": Stratum.TACTICS,
        "oracle_analyze": Stratum.TACTICS,
        "topology_analysis": Stratum.TACTICS,
        "dependency_graph": Stratum.TACTICS,

        # STRATEGY - Valor y Riesgo
        "business_topology": Stratum.STRATEGY,
        "financial_analysis": Stratum.STRATEGY,
        "risk_assessment": Stratum.STRATEGY,
        "npv_calculation": Stratum.STRATEGY,

        # WISDOM - Síntesis
        "build_output": Stratum.WISDOM,
        "response_preparation": Stratum.WISDOM,
        "narrative_generation": Stratum.WISDOM,
    }

    @classmethod
    def get_level(cls, stratum: Stratum) -> int:
        """Obtiene el nivel jerárquico (mayor = más base)."""
        return cls.HIERARCHY.get(stratum, 0)

    @classmethod
    def get_stratum_for_step(
        cls,
        step_name: str,
        custom_mapping: Optional[Dict[str, Stratum]] = None,
    ) -> Stratum:
        """
        Determina el estrato para un paso dado.
        
        Prioridad: custom_mapping > DEFAULT_STEP_MAPPING > PHYSICS (default)
        """
        if custom_mapping and step_name in custom_mapping:
            return custom_mapping[step_name]

        if step_name in cls.DEFAULT_STEP_MAPPING:
            return cls.DEFAULT_STEP_MAPPING[step_name]

        # Heurística: buscar prefijos conocidos
        step_lower = step_name.lower()
        if any(kw in step_lower for kw in ("load", "parse", "flux", "merge", "ingest")):
            return Stratum.PHYSICS
        if any(kw in step_lower for kw in ("cost", "topology", "graph", "material")):
            return Stratum.TACTICS
        if any(kw in step_lower for kw in ("financial", "risk", "npv", "business")):
            return Stratum.STRATEGY
        if any(kw in step_lower for kw in ("output", "response", "build", "narrative")):
            return Stratum.WISDOM

        return Stratum.PHYSICS  # Default conservador

    @classmethod
    def is_higher_than(cls, a: Stratum, b: Stratum) -> bool:
        """Retorna True si 'a' está más arriba en la pirámide que 'b'."""
        return cls.get_level(a) < cls.get_level(b)

    @classmethod
    def get_strata_above(cls, stratum: Stratum) -> List[Stratum]:
        """Retorna estratos superiores al dado."""
        level = cls.get_level(stratum)
        return [s for s, l in cls.HIERARCHY.items() if l < level]

    @classmethod
    def get_strata_below(cls, stratum: Stratum) -> List[Stratum]:
        """Retorna estratos inferiores (más base) al dado."""
        level = cls.get_level(stratum)
        return [s for s, l in cls.HIERARCHY.items() if l > level]


# ============================================================================
# LATTICE DE SEVERIDAD
# ============================================================================


class SeverityLevel(IntEnum):
    """
    Lattice de severidad con orden total.
    
    Estructura algebraica: (SeverityLevel, ≤, ⊔, ⊓)
    - Forma un lattice acotado completo
    - ⊥ (bottom) = OPTIMO (elemento mínimo)
    - ⊤ (top) = CRITICO (elemento máximo)
    
    Propiedades del Lattice:
    - Conmutatividad: a ⊔ b = b ⊔ a
    - Asociatividad: (a ⊔ b) ⊔ c = a ⊔ (b ⊔ c)
    - Idempotencia: a ⊔ a = a
    - Absorción: a ⊔ (a ⊓ b) = a
    """

    OPTIMO = 0       # ⊥ - Bottom element
    ADVERTENCIA = 1  # Elemento intermedio
    CRITICO = 2      # ⊤ - Top element

    @classmethod
    def bottom(cls) -> SeverityLevel:
        """Retorna el elemento mínimo del lattice (⊥)."""
        return cls.OPTIMO

    @classmethod
    def top(cls) -> SeverityLevel:
        """Retorna el elemento máximo del lattice (⊤)."""
        return cls.CRITICO

    @classmethod
    def from_step_status(cls, status: Union[StepStatus, str, None]) -> SeverityLevel:
        """
        Morfismo desde StepStatus al lattice de severidad.
        
        Preserva orden: SUCCESS < WARNING < FAILURE
        """
        if status is None:
            return cls.OPTIMO

        # Normalizar string a StepStatus
        if isinstance(status, str):
            try:
                status = StepStatus.from_string(status)
            except (ValueError, AttributeError, KeyError):
                logger.debug(f"Could not parse status string: {status}")
                return cls.OPTIMO

        # Mapeo explícito
        mapping: Dict[StepStatus, SeverityLevel] = {
            StepStatus.SUCCESS: cls.OPTIMO,
            StepStatus.IN_PROGRESS: cls.OPTIMO,
            StepStatus.SKIPPED: cls.OPTIMO,
            StepStatus.WARNING: cls.ADVERTENCIA,
            StepStatus.CANCELLED: cls.ADVERTENCIA,
            StepStatus.FAILURE: cls.CRITICO,
        }

        return mapping.get(status, cls.OPTIMO)

    @classmethod
    def from_error_count(cls, count: int, threshold: int = 1) -> SeverityLevel:
        """Deriva severidad desde conteo de errores."""
        if count <= 0:
            return cls.OPTIMO
        if count < threshold:
            return cls.ADVERTENCIA
        return cls.CRITICO

    @classmethod
    def supremum(cls, *levels: SeverityLevel) -> SeverityLevel:
        """
        Operación JOIN (⊔) del lattice.
        Retorna el supremo (máximo) de los niveles dados.
        
        Propiedad: a ⊔ ⊥ = a (OPTIMO es identidad)
        """
        if not levels:
            return cls.OPTIMO  # ⊥ es identidad del join

        try:
            max_value = max(level.value for level in levels)
            return cls(max_value)
        except (ValueError, TypeError):
            return cls.OPTIMO

    @classmethod
    def infimum(cls, *levels: SeverityLevel) -> SeverityLevel:
        """
        Operación MEET (⊓) del lattice.
        Retorna el ínfimo (mínimo) de los niveles dados.
        
        Propiedad: a ⊓ ⊤ = a (CRITICO es identidad)
        """
        if not levels:
            return cls.CRITICO  # ⊤ es identidad del meet

        try:
            min_value = min(level.value for level in levels)
            return cls(min_value)
        except (ValueError, TypeError):
            return cls.CRITICO

    def join(self, other: SeverityLevel) -> SeverityLevel:
        """Operación join binaria: self ⊔ other."""
        return SeverityLevel.supremum(self, other)

    def meet(self, other: SeverityLevel) -> SeverityLevel:
        """Operación meet binaria: self ⊓ other."""
        return SeverityLevel.infimum(self, other)

    def __or__(self, other: SeverityLevel) -> SeverityLevel:
        """Sintaxis: a | b = a ⊔ b (join)."""
        return self.join(other)

    def __and__(self, other: SeverityLevel) -> SeverityLevel:
        """Sintaxis: a & b = a ⊓ b (meet)."""
        return self.meet(other)

    @property
    def is_critical(self) -> bool:
        """Indica si es el elemento máximo."""
        return self == SeverityLevel.CRITICO

    @property
    def is_optimal(self) -> bool:
        """Indica si es el elemento mínimo."""
        return self == SeverityLevel.OPTIMO

    @property
    def emoji(self) -> str:
        """Representación visual."""
        return {
            SeverityLevel.OPTIMO: "✅",
            SeverityLevel.ADVERTENCIA: "⚠️",
            SeverityLevel.CRITICO: "❌",
        }[self]


# ============================================================================
# ESTRUCTURAS DE DATOS INMUTABLES
# ============================================================================


@dataclass(frozen=False)  # No frozen para permitir context mutable
class Issue:
    """
    Evidencia forense con localización topológica.
    
    Representa un problema detectado durante la ejecución,
    con información de su posición en el árbol de spans.
    """

    source: str
    message: str
    issue_type: str
    depth: int
    topological_path: Tuple[str, ...]  # Inmutable, estructurado
    timestamp: Optional[str] = None
    stratum: Optional[Stratum] = None
    severity: SeverityLevel = SeverityLevel.ADVERTENCIA
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validación y normalización post-inicialización."""
        # Asegurar que topological_path sea tupla
        if isinstance(self.topological_path, str):
            object.__setattr__(
                self,
                "topological_path",
                tuple(self.topological_path.split(" → "))
            )
        elif isinstance(self.topological_path, list):
            object.__setattr__(
                self,
                "topological_path",
                tuple(self.topological_path)
            )

        # Truncar mensaje si es muy largo
        if len(self.message) > NarratorConfig.MAX_PATH_LENGTH:
            object.__setattr__(
                self,
                "message",
                self.message[:NarratorConfig.MAX_PATH_LENGTH] + "..."
            )

        # Derivar severidad del tipo si no está explícita
        if self.severity == SeverityLevel.ADVERTENCIA:
            derived = self._derive_severity_from_type()
            object.__setattr__(self, "severity", derived)

    def _derive_severity_from_type(self) -> SeverityLevel:
        """Deriva severidad basada en el tipo de issue."""
        if self.issue_type in NarratorConfig.NON_CRITICAL_ISSUE_TYPES:
            return SeverityLevel.ADVERTENCIA
        return SeverityLevel.CRITICO

    @property
    def is_critical(self) -> bool:
        """Indica si el issue es crítico."""
        return self.severity == SeverityLevel.CRITICO

    @property
    def path_string(self) -> str:
        """Retorna el path como string legible."""
        return " → ".join(self.topological_path)

    @property
    def path_depth(self) -> int:
        """Profundidad calculada desde el path."""
        return len(self.topological_path) - 1

    def with_stratum(self, stratum: Stratum) -> Issue:
        """Retorna copia con estrato asignado."""
        new_context = dict(self.context)
        new_context["stratum"] = stratum.name
        return Issue(
            source=self.source,
            message=self.message,
            issue_type=self.issue_type,
            depth=self.depth,
            topological_path=self.topological_path,
            timestamp=self.timestamp,
            stratum=stratum,
            severity=self.severity,
            context=new_context,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        result = {
            "source": self.source,
            "message": self.message,
            "type": self.issue_type,
            "severity": self.severity.name,
            "depth": self.depth,
            "topological_path": self.path_string,
        }

        if self.timestamp is not None:
            result["timestamp"] = self.timestamp

        if self.stratum is not None:
            result["stratum"] = self.stratum.name

        if self.context:
            result["context"] = self.context

        return result


@dataclass
class PhaseAnalysis:
    """
    Resultado del análisis de una fase (span raíz).
    
    Representa el estado agregado de un subárbol completo de spans.
    """

    name: str
    stratum: Stratum
    severity: SeverityLevel
    duration_seconds: float
    issues: List[Issue]
    warning_count: int
    child_count: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validación post-inicialización."""
        if self.duration_seconds < 0:
            self.duration_seconds = 0.0

        # Limitar issues almacenados
        if len(self.issues) > NarratorConfig.MAX_ISSUES_PER_PHASE:
            self.issues = self.issues[:NarratorConfig.MAX_ISSUES_PER_PHASE]

    @property
    def critical_issues(self) -> List[Issue]:
        """Filtra issues críticos."""
        return [i for i in self.issues if i.is_critical]

    @property
    def warnings(self) -> List[Issue]:
        """Filtra warnings (no críticos)."""
        return [i for i in self.issues if not i.is_critical]

    @property
    def has_failures(self) -> bool:
        """Indica si hay fallos críticos."""
        return self.severity == SeverityLevel.CRITICO

    @property
    def is_clean(self) -> bool:
        """Indica si no hay issues de ningún tipo."""
        return len(self.issues) == 0 and self.severity == SeverityLevel.OPTIMO

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        displayed_warnings = self.warnings[:NarratorConfig.MAX_DISPLAYED_WARNINGS]

        return {
            "name": self.name,
            "stratum": self.stratum.name,
            "status": self.severity.name,
            "status_emoji": self.severity.emoji,
            "duration": f"{self.duration_seconds:.3f}s",
            "duration_seconds": round(self.duration_seconds, 6),
            "critical_issues": [i.to_dict() for i in self.critical_issues],
            "critical_count": len(self.critical_issues),
            "warning_count": self.warning_count,
            "warnings": [i.to_dict() for i in displayed_warnings],
            "child_count": self.child_count,
            "metrics": self.metrics,
        }


@dataclass
class StratumAnalysis:
    """
    Análisis consolidado de un Estrato de la Pirámide.
    
    Representa la salud agregada de una capa completa de la arquitectura.
    Aplica el operador supremum sobre todas las fases del estrato.
    """

    stratum: Stratum
    severity: SeverityLevel
    narrative: str
    phases: List[PhaseAnalysis]
    issues: List[Issue]
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Recalcula severidad si es necesario."""
        if self.phases and self.severity == SeverityLevel.OPTIMO:
            # Recalcular como supremum de las fases
            phase_severities = [p.severity for p in self.phases]
            self.severity = SeverityLevel.supremum(*phase_severities)

    @property
    def is_compromised(self) -> bool:
        """Indica si el estrato está en estado crítico."""
        return self.severity == SeverityLevel.CRITICO

    @property
    def is_healthy(self) -> bool:
        """Indica si el estrato está en estado óptimo."""
        return self.severity == SeverityLevel.OPTIMO

    @property
    def phase_count(self) -> int:
        """Número de fases en este estrato."""
        return len(self.phases)

    @property
    def total_duration(self) -> float:
        """Duración total de todas las fases."""
        return sum(p.duration_seconds for p in self.phases)

    @property
    def critical_issue_count(self) -> int:
        """Conteo de issues críticos."""
        return sum(1 for i in self.issues if i.is_critical)

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "stratum": self.stratum.name,
            "level": StratumTopology.get_level(self.stratum),
            "severity": self.severity.name,
            "severity_emoji": self.severity.emoji,
            "is_compromised": self.is_compromised,
            "narrative": self.narrative,
            "phase_count": self.phase_count,
            "total_duration_seconds": round(self.total_duration, 6),
            "critical_issues": self.critical_issue_count,
            "total_issues": len(self.issues),
            "metrics": self.metrics,
            "phases": [p.name for p in self.phases],
        }


@dataclass
class PyramidalReport:
    """
    Reporte final estructurado jerárquicamente (DIKW).
    
    Contiene el juicio del Consejo de Sabios con trazabilidad completa.
    """

    verdict: str
    verdict_code: str
    executive_summary: str
    global_severity: SeverityLevel
    strata_analysis: Dict[Stratum, StratumAnalysis]
    forensic_evidence: List[Issue]
    phases: List[PhaseAnalysis]
    causality_chain: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def is_approved(self) -> bool:
        """Indica si el proyecto fue aprobado."""
        return self.global_severity == SeverityLevel.OPTIMO

    @property
    def failed_strata(self) -> List[Stratum]:
        """Lista de estratos que fallaron."""
        return [s for s, a in self.strata_analysis.items() if a.is_compromised]

    @property
    def root_cause_stratum(self) -> Optional[Stratum]:
        """
        Retorna el estrato de causa raíz (el más base que falló).
        Siguiendo la Clausura Transitiva.
        """
        for stratum in StratumTopology.EVALUATION_ORDER:
            if stratum in self.strata_analysis:
                if self.strata_analysis[stratum].is_compromised:
                    return stratum
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario con compatibilidad."""
        return {
            # Campos principales
            "verdict": self.verdict,
            "verdict_code": self.verdict_code,
            "executive_summary": self.executive_summary,
            "global_severity": self.global_severity.name,
            "is_approved": self.is_approved,
            "timestamp": self.timestamp,

            # Alias de compatibilidad
            "narrative": self.executive_summary,
            "status": self.global_severity.name,

            # Análisis detallado
            "strata_analysis": {
                s.name: a.to_dict()
                for s, a in self.strata_analysis.items()
            },

            # Evidencia forense
            "forensic_evidence": [
                i.to_dict()
                for i in self.forensic_evidence[:NarratorConfig.MAX_FORENSIC_EVIDENCE]
            ],

            # Cadena causal
            "causality_chain": self.causality_chain,
            "root_cause_stratum": (
                self.root_cause_stratum.name if self.root_cause_stratum else None
            ),

            # Recomendaciones
            "recommendations": self.recommendations,

            # Fases (compatibilidad con UI)
            "phases": [p.to_dict() for p in self.phases],
        }


# ============================================================================
# NARRADOR PRINCIPAL
# ============================================================================


class TelemetryNarrator:
    """
    Narrador que implementa la lógica del 'Consejo de Sabios'.

    Organiza la evidencia forense según la jerarquía DIKW
    y aplica Clausura Transitiva para determinar el veredicto.
    
    Flujo de Procesamiento:
    1. Análisis de Fases (spans individuales)
    2. Agrupación por Estratos (consolidación)
    3. Síntesis de Sabiduría (veredicto final)
    4. Generación de Reporte (serialización)
    """

    def __init__(
        self,
        step_mapping: Optional[Dict[str, Stratum]] = None,
        config: Optional[NarratorConfig] = None,
        mic: Optional[MICRegistry] = None,
    ):
        """
        Inicializa el narrador.

        Args:
            step_mapping: Mapeo personalizado de pasos a estratos
            config: Configuración personalizada
            mic: MIC Registry for fetching narratives.
        """
        self.step_mapping = step_mapping or {}
        self.config = config or NarratorConfig()

        if mic:
            self.mic = mic
        else:
            # Fallback for legacy usage
            self.mic = MICRegistry()
            register_core_vectors(self.mic, config={})

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

    def get_stratum_narrative(
        self,
        stratum: Stratum,
        severity: SeverityLevel,
        issues: List[Issue],
    ) -> str:
        """Genera narrativa apropiada para el estrato y severidad."""

        # Mapping severity to domain
        if severity == SeverityLevel.OPTIMO:
            domain = "TELEMETRY_SUCCESS"
            classification = stratum.name
        elif severity == SeverityLevel.ADVERTENCIA:
            domain = "TELEMETRY_WARNINGS"
            classification = stratum.name
        else: # CRITICO
            domain = f"TELEMETRY_FAILURES_{stratum.name}"
            classification = self._detect_failure_type(stratum, issues)

        return self._fetch_narrative(domain, classification)

    def _detect_failure_type(self, stratum: Stratum, issues: List[Issue]) -> str:
        """Detecta el tipo específico de fallo basado en los issues."""
        issue_messages = " ".join(i.message.lower() for i in issues)

        if stratum == Stratum.PHYSICS:
            if "nutación" in issue_messages or "nutation" in issue_messages:
                return "nutation"
            if "muerte térmica" in issue_messages or "thermal death" in issue_messages:
                return "thermal_death"
            if "divergencia" in issue_messages or "unstable" in issue_messages or "rhp" in issue_messages:
                return "laplace_unstable"
            if "saturación" in issue_messages or "saturation" in issue_messages:
                return "saturation"
            if "golpe de ariete" in issue_messages or "water_hammer" in issue_messages or "tubería" in issue_messages:
                return "water_hammer"
            if "esfuerzo de inyección" in issue_messages or "pump_work" in issue_messages:
                return "high_injection_work"
            if "corrupt" in issue_messages or "invalid" in issue_messages:
                return "corruption"

        elif stratum == Stratum.TACTICS:
            if "mayer-vietoris" in issue_messages or "integración" in issue_messages:
                return "mayer_vietoris"
            if "ciclo" in issue_messages or "cycle" in issue_messages or "β₁" in issue_messages:
                return "cycles"
            if "desconect" in issue_messages or "disconnect" in issue_messages:
                return "disconnected"

        elif stratum == Stratum.STRATEGY:
            if "var" in issue_messages or "volatil" in issue_messages:
                return "high_var"
            if "npv" in issue_messages and "negativ" in issue_messages:
                return "negative_npv"

        return "default"

    def _get_verdict_info(self, verdict_code: str) -> Tuple[str, str]:
        """Obtiene título y descripción del veredicto desde MIC."""
        full_text = self._fetch_narrative("TELEMETRY_VERDICTS", verdict_code)
        # Parse based on assuming a newline separator if present, or just split.
        # Format in Dictionary: "Title\nDescription"
        if "\n" in full_text:
            parts = full_text.split("\n", 1)
            return parts[0], parts[1]
        return full_text, ""

    def get_root_cause_stratum(self, context: TelemetryContext) -> Optional[Stratum]:
        """
        Identifica el estrato de causa raíz de un fallo.

        Retorna el Stratum más bajo (mayor valor numérico) donde se originó el primer fallo.
        Esto permite identificar si el problema es de "base" (PHYSICS) o de "cúspide" (WISDOM).
        """
        if not context or not context.root_spans:
            return None

        deepest_failure_stratum = None
        deepest_level = -1

        # Recorrido BFS para encontrar todos los fallos y determinar el más profundo (base)
        queue = list(context.root_spans)
        while queue:
            span = queue.pop(0)

            if span.status == StepStatus.FAILURE:
                # Determinar estrato del span
                stratum = span.stratum

                # Heurística: Si es default (PHYSICS) y tenemos mapping, intentar refinar
                if stratum == Stratum.PHYSICS:
                    stratum = StratumTopology.get_stratum_for_step(span.name, self.step_mapping)

                level = StratumTopology.get_level(stratum)

                # Buscamos el nivel más alto (mayor entero = más base = PHYSICS=3)
                if level > deepest_level:
                    deepest_level = level
                    deepest_failure_stratum = stratum

            queue.extend(span.children)

        return deepest_failure_stratum

    def summarize_execution(self, context: TelemetryContext) -> Dict[str, Any]:
        """
        Punto de entrada principal.
        
        Ejecuta la síntesis piramidal y retorna el reporte estructurado.
        """
        if context is None:
            logger.warning("TelemetryNarrator received None context")
            return self._generate_empty_report().to_dict()

        root_spans = getattr(context, "root_spans", [])

        # Modo legacy: sin spans pero con steps/errors
        if not root_spans:
            if context.steps or context.errors:
                return self._summarize_legacy(context)

            # Si no hay spans, ni steps/errors, ni métricas registradas, es un contexto vacío
            if not context.metrics:
                return self._generate_empty_report().to_dict()

            # Fallthrough to main flow to check typed metrics (Global Analysis)

        try:
            # 1. Análisis por Fases
            phases_analysis = self._analyze_all_phases(root_spans)

            # 2. Agrupación por Estratos
            strata_results = self._group_by_stratum(phases_analysis)

            # 2.5 Análisis Global (Typed State Vectors)
            # Inyecta problemas detectados en los objetos tipados globales (physics, topology, etc.)
            global_issues = self._analyze_global_context(context)
            self._inject_global_issues(strata_results, global_issues)

            # 3. Síntesis de Sabiduría
            verdict_code, global_severity = self._determine_verdict(strata_results)
            executive_summary, causality = self._synthesize_narrative(
                strata_results, verdict_code
            )

            # 4. Extracción de Evidencia
            forensic_evidence = self._extract_forensic_evidence(
                strata_results, verdict_code
            )

            # 5. Generación de Recomendaciones
            recommendations = self._generate_recommendations(
                strata_results, verdict_code
            )

            # 6. Construcción del Reporte
            report = PyramidalReport(
                verdict=self._get_verdict_info(verdict_code)[0],
                verdict_code=verdict_code,
                executive_summary=executive_summary,
                global_severity=global_severity,
                strata_analysis=strata_results,
                forensic_evidence=forensic_evidence,
                phases=phases_analysis,
                causality_chain=causality,
                recommendations=recommendations,
            )

            return report.to_dict()

        except Exception as e:
            logger.error(f"Error in summarize_execution: {e}", exc_info=True)
            return self._generate_error_report(str(e)).to_dict()

    def _analyze_global_context(self, context: TelemetryContext) -> Dict[Stratum, List[Issue]]:
        """
        Analiza los vectores de estado globales buscando anomalías.
        Prioriza los objetos tipados sobre el diccionario de métricas.
        """
        issues_by_stratum = {s: [] for s in Stratum}

        # 1. PHYSICS
        if hasattr(context, "physics"):
            p = context.physics
            # Saturation
            if p.saturation > self.config.ANOMALY_THRESHOLDS.get("saturation", 0.9):
                issues_by_stratum[Stratum.PHYSICS].append(Issue(
                    source="FluxCondenser (Global)",
                    message=f"Saturación global crítica: {p.saturation:.1%}",
                    issue_type="MetricAnomaly",
                    depth=0,
                    topological_path=("global", "physics"),
                    stratum=Stratum.PHYSICS,
                    severity=SeverityLevel.ADVERTENCIA
                ))
            # Flyback
            if p.flyback_voltage > self.config.ANOMALY_THRESHOLDS.get("flyback_voltage", 0.5):
                issues_by_stratum[Stratum.PHYSICS].append(Issue(
                    source="FluxCondenser (Global)",
                    message=f"Voltaje Flyback excesivo: {p.flyback_voltage:.2f}V",
                    issue_type="MetricAnomaly",
                    depth=0,
                    topological_path=("global", "physics"),
                    stratum=Stratum.PHYSICS,
                    severity=SeverityLevel.CRITICO
                ))
            # Dissipated Power
            if p.dissipated_power > self.config.ANOMALY_THRESHOLDS.get("dissipated_power", 50.0):
                issues_by_stratum[Stratum.PHYSICS].append(Issue(
                    source="FluxCondenser (Global)",
                    message=f"Potencia disipada alta: {p.dissipated_power:.2f}W",
                    issue_type="MetricAnomaly",
                    depth=0,
                    topological_path=("global", "physics"),
                    stratum=Stratum.PHYSICS,
                    severity=SeverityLevel.ADVERTENCIA
                ))
            # Hamiltonian Excess
            if p.hamiltonian_excess >= 0.01:
                issues_by_stratum[Stratum.PHYSICS].append(Issue(
                    source="FluxCondenser (Global)",
                    message=f"Violación de Conservación de Energía (H_excess={p.hamiltonian_excess:.3f})",
                    issue_type="HamiltonianViolation",
                    depth=0,
                    topological_path=("global", "physics"),
                    stratum=Stratum.PHYSICS,
                    severity=SeverityLevel.CRITICO
                ))

        # 2. CONTROL (Physics Stratum)
        if hasattr(context, "control"):
            c = context.control
            if not c.is_stable:
                issues_by_stratum[Stratum.PHYSICS].append(Issue(
                    source="LaplaceOracle (Global)",
                    message="Sistema inestable (Polos en RHP)",
                    issue_type="StabilityError",
                    depth=0,
                    topological_path=("global", "control"),
                    stratum=Stratum.PHYSICS,
                    severity=SeverityLevel.CRITICO
                ))

        # 3. THERMODYNAMICS (Physics Stratum for Temp)
        if hasattr(context, "thermodynamics"):
            t = context.thermodynamics
            # Convertir a Celsius para comparar con umbral de 75°C
            # O usar umbral Kelvin: 75°C approx 348.15 K
            if t.system_temperature > 348.15: # Critical (> 75°C)
                temp_c = t.system_temperature - 273.15
                issues_by_stratum[Stratum.PHYSICS].append(Issue(
                    source="Thermodynamics (Global)",
                    message=f"Temperatura del sistema crítica: {temp_c:.1f}°C",
                    issue_type="ThermalAnomaly",
                    depth=0,
                    topological_path=("global", "thermodynamics"),
                    stratum=Stratum.PHYSICS,
                    severity=SeverityLevel.CRITICO
                ))
            # Low Inertia (Hoja al Viento)
            if t.heat_capacity < 0.2:
                issues_by_stratum[Stratum.STRATEGY].append(Issue(
                    source="Thermodynamics (Global)",
                    message=f"Baja Inercia Financiera (Hoja al Viento): C_v={t.heat_capacity:.2f}",
                    issue_type="LowInertia",
                    depth=0,
                    topological_path=("global", "thermodynamics"),
                    stratum=Stratum.STRATEGY,
                    severity=SeverityLevel.ADVERTENCIA
                ))

        # 4. TOPOLOGY (Tactics Stratum)
        if hasattr(context, "topology"):
            topo = context.topology
            if topo.beta_1 > 0:
                issues_by_stratum[Stratum.TACTICS].append(Issue(
                    source="Topology (Global)",
                    message=f"Ciclos lógicos detectados: β₁={topo.beta_1}",
                    issue_type="TopologicalDefect",
                    depth=0,
                    topological_path=("global", "topology"),
                    stratum=Stratum.TACTICS,
                    severity=SeverityLevel.CRITICO
                ))
            if topo.pyramid_stability < 1.0:
                issues_by_stratum[Stratum.TACTICS].append(Issue(
                    source="Topology (Global)",
                    message=f"Pirámide Invertida (Inestabilidad): Ψ={topo.pyramid_stability:.2f}",
                    issue_type="StructuralInstability",
                    depth=0,
                    topological_path=("global", "topology"),
                    stratum=Stratum.TACTICS,
                    severity=SeverityLevel.CRITICO
                ))

        return issues_by_stratum

    def _inject_global_issues(
        self,
        strata_results: Dict[Stratum, StratumAnalysis],
        global_issues: Dict[Stratum, List[Issue]]
    ) -> None:
        """Inyecta issues globales en los resultados de estrato."""
        for stratum, issues in global_issues.items():
            if not issues:
                continue

            if stratum in strata_results:
                analysis = strata_results[stratum]
                analysis.issues.extend(issues)

                # Recalcular severidad
                global_severity = SeverityLevel.OPTIMO
                for issue in issues:
                    global_severity = global_severity | issue.severity

                analysis.severity = analysis.severity | global_severity

                # Actualizar narrativa si empeoró
                if global_severity.is_critical:
                    analysis.narrative = f"Fallo Global detectado: {issues[0].message}"

    # ========================================================================
    # ANÁLISIS DE FASES
    # ========================================================================

    def _analyze_all_phases(self, spans: List[TelemetrySpan]) -> List[PhaseAnalysis]:
        """Analiza todas las fases (spans raíz)."""
        return [self._analyze_phase(span) for span in spans]

    def _analyze_phase(self, span: TelemetrySpan) -> PhaseAnalysis:
        """
        Analiza un span y su subárbol completo.
        
        Aplica el supremum sobre:
        - Severidad directa del span
        - Severidad inducida por issues
        - Severidad heredada de hijos
        """
        # Recolectar issues recursivamente
        issues = list(self._collect_issues_recursive(
            span,
            depth=0,
            path_prefix=(),
        ))

        # Separar críticos y warnings
        criticals = [i for i in issues if i.is_critical]
        warnings = [i for i in issues if not i.is_critical]

        # Calcular severidades componentes
        direct_severity = SeverityLevel.from_step_status(span.status)
        induced_severity = self._compute_induced_severity(criticals, warnings)
        hierarchy_severity = self._compute_hierarchy_severity(span)

        # Supremum de todas las severidades
        final_severity = SeverityLevel.supremum(
            direct_severity,
            induced_severity,
            hierarchy_severity,
        )

        # Determinar estrato
        # Heurística híbrida: Preferir stratum explícito del span si no es PHYSICS (default)
        # Si es PHYSICS, consultar el mapeo por nombre para auto-clasificación
        if span.stratum != Stratum.PHYSICS:
            stratum = span.stratum
        else:
            stratum = StratumTopology.get_stratum_for_step(
                span.name,
                self.step_mapping,
            )

        # Asignar estrato a issues
        issues_with_stratum = [i.with_stratum(stratum) for i in issues]

        duration = span.duration if span.duration is not None else 0.0

        return PhaseAnalysis(
            name=span.name,
            stratum=stratum,
            severity=final_severity,
            duration_seconds=duration,
            issues=issues_with_stratum,
            warning_count=len(warnings),
            child_count=len(span.children),
            metrics=dict(span.metrics) if span.metrics else {},
        )

    def _collect_issues_recursive(
        self,
        span: TelemetrySpan,
        depth: int,
        path_prefix: Tuple[str, ...],
    ) -> Iterator[Issue]:
        """
        Recolecta issues del span y sus descendientes.
        
        Implementa DFS con límite de profundidad.
        """
        current_path = path_prefix + (span.name,)

        # Verificar límite de recursión
        if depth > self.config.MAX_RECURSION_DEPTH:
            yield Issue(
                source=span.name,
                message=f"Profundidad máxima excedida ({self.config.MAX_RECURSION_DEPTH})",
                issue_type="RecursionLimit",
                depth=depth,
                topological_path=current_path,
                severity=SeverityLevel.ADVERTENCIA,
            )
            return

        # Extraer errores explícitos
        yield from self._extract_explicit_errors(span, depth, current_path)

        # Detectar fallos silenciosos
        if self._is_silent_failure(span):
            yield Issue(
                source=span.name,
                message="Fallo estructural sin traza explícita (Silent Failure)",
                issue_type="SilentFailure",
                depth=depth,
                topological_path=current_path,
                severity=SeverityLevel.CRITICO,
            )

        # Detectar warnings implícitos
        if self._is_implicit_warning(span):
            yield Issue(
                source=span.name,
                message="Advertencia de fase (Estado WARNING)",
                issue_type="Warning",
                depth=depth,
                topological_path=current_path,
                severity=SeverityLevel.ADVERTENCIA,
            )

        # Extraer métricas anómalas
        yield from self._extract_anomalous_metrics(span, depth, current_path)

        # Recursión en hijos
        for child in span.children:
            yield from self._collect_issues_recursive(
                child,
                depth + 1,
                current_path,
            )

    def _extract_explicit_errors(
        self,
        span: TelemetrySpan,
        depth: int,
        path: Tuple[str, ...],
    ) -> Iterator[Issue]:
        """Extrae errores explícitos del span."""
        if not hasattr(span, "errors") or not span.errors:
            return

        for error in span.errors:
            if not isinstance(error, dict):
                continue

            yield Issue(
                source=span.name,
                message=error.get("message", "Error sin mensaje"),
                issue_type=error.get("type", "Error"),
                depth=depth,
                topological_path=path,
                timestamp=error.get("timestamp"),
                severity=SeverityLevel.CRITICO,
                context={
                    "traceback": error.get("traceback", "")[:500],
                } if error.get("traceback") else {},
            )

    def _is_silent_failure(self, span: TelemetrySpan) -> bool:
        """Detecta fallo sin errores explícitos."""
        return (
            span.status == StepStatus.FAILURE
            and (not hasattr(span, "errors") or not span.errors)
        )

    def _is_implicit_warning(self, span: TelemetrySpan) -> bool:
        """Detecta warning sin errores explícitos."""
        return (
            span.status == StepStatus.WARNING
            and (not hasattr(span, "errors") or not span.errors)
        )

    def _extract_anomalous_metrics(
        self,
        span: TelemetrySpan,
        depth: int,
        path: Tuple[str, ...],
    ) -> Iterator[Issue]:
        """Extrae métricas que indican anomalías."""
        metrics = getattr(span, "metrics", None)
        if not metrics or not isinstance(metrics, dict):
            return

        for metric_name, metric_data in metrics.items():
            anomaly_info = self._check_metric_anomaly(metric_name, metric_data)
            if anomaly_info:
                yield Issue(
                    source=span.name,
                    message=anomaly_info["message"],
                    issue_type="MetricAnomaly",
                    depth=depth,
                    topological_path=path,
                    severity=SeverityLevel.ADVERTENCIA,
                    context={"metric": metric_name, "value": anomaly_info["value"]},
                )

    def _check_metric_anomaly(
        self,
        name: str,
        data: Any,
    ) -> Optional[Dict[str, Any]]:
        """Verifica si una métrica es anómala."""
        # Estructura dict con flag explícito
        if isinstance(data, dict):
            if data.get("anomalous", False):
                return {
                    "message": f"Métrica anómala: {name}={data.get('value', data)}",
                    "value": data.get("value", data),
                }

        # Verificar contra umbrales conocidos
        if isinstance(data, (int, float)):
            for threshold_name, threshold_value in self.config.ANOMALY_THRESHOLDS.items():
                if threshold_name in name.lower():
                    if data > threshold_value:
                        return {
                            "message": f"Métrica {name}={data} excede umbral ({threshold_value})",
                            "value": data,
                        }

        return None

    def _compute_induced_severity(
        self,
        criticals: List[Issue],
        warnings: List[Issue],
    ) -> SeverityLevel:
        """Calcula severidad inducida por issues."""
        if criticals:
            return SeverityLevel.CRITICO
        if warnings:
            return SeverityLevel.ADVERTENCIA
        return SeverityLevel.OPTIMO

    def _compute_hierarchy_severity(
        self,
        span: TelemetrySpan,
        depth: int = 0,
        visited: Optional[set] = None,
    ) -> SeverityLevel:
        """
        Calcula severidad considerando toda la jerarquía (DFS).
        
        Aplica supremum sobre el span y todos sus descendientes.
        Maneja ciclos y profundidad máxima.
        """
        if visited is None:
            visited = set()

        # Evitar ciclos y profundidad excesiva
        span_id = id(span)
        if span_id in visited or depth > self.config.MAX_RECURSION_DEPTH:
            return SeverityLevel.from_step_status(span.status)

        visited.add(span_id)
        severity = SeverityLevel.from_step_status(span.status)

        for child in span.children:
            child_severity = self._compute_hierarchy_severity(child, depth + 1, visited)
            severity = severity | child_severity  # join

        visited.remove(span_id)  # Backtracking (opcional, pero limpio)
        return severity

    # ========================================================================
    # AGRUPACIÓN POR ESTRATOS
    # ========================================================================

    def _group_by_stratum(
        self,
        phases: List[PhaseAnalysis],
    ) -> Dict[Stratum, StratumAnalysis]:
        """
        Agrupa análisis de fases por estrato.
        
        Para cada estrato:
        - Calcula severidad como supremum de sus fases
        - Genera narrativa apropiada
        - Agrega métricas
        """
        # Inicializar grupos vacíos
        grouped: Dict[Stratum, List[PhaseAnalysis]] = {s: [] for s in Stratum}

        # Clasificar fases
        for phase in phases:
            grouped[phase.stratum].append(phase)

        # Construir análisis por estrato
        results: Dict[Stratum, StratumAnalysis] = {}

        for stratum in StratumTopology.EVALUATION_ORDER:
            stratum_phases = grouped[stratum]

            # Calcular severidad del estrato
            if stratum_phases:
                severity = SeverityLevel.supremum(
                    *(p.severity for p in stratum_phases)
                )
            else:
                severity = SeverityLevel.OPTIMO

            # Recolectar todos los issues
            all_issues = list(chain.from_iterable(
                p.issues for p in stratum_phases
            ))

            # Generar narrativa
            narrative = self.get_stratum_narrative(
                stratum,
                severity,
                all_issues,
            )

            # Agregar métricas
            metrics = self._aggregate_stratum_metrics(stratum_phases)

            results[stratum] = StratumAnalysis(
                stratum=stratum,
                severity=severity,
                narrative=narrative,
                phases=stratum_phases,
                issues=all_issues,
                metrics=metrics,
            )

        return results

    def _aggregate_stratum_metrics(
        self,
        phases: List[PhaseAnalysis],
    ) -> Dict[str, Any]:
        """Agrega métricas de las fases de un estrato."""
        if not phases:
            return {"duration_total": 0.0, "warnings": 0, "phases": 0}

        return {
            "duration_total": round(sum(p.duration_seconds for p in phases), 6),
            "warnings": sum(p.warning_count for p in phases),
            "phases": len(phases),
            "critical_count": sum(len(p.critical_issues) for p in phases),
        }

    # ========================================================================
    # SÍNTESIS DE SABIDURÍA
    # ========================================================================

    def _determine_verdict(
        self,
        strata: Dict[Stratum, StratumAnalysis],
    ) -> Tuple[str, SeverityLevel]:
        """
        Determina el veredicto final aplicando Clausura Transitiva.
        
        Regla: Fallo en estrato N invalida todos los estratos superiores.
        Evaluación: PHYSICS → TACTICS → STRATEGY → WISDOM
        """
        # Evaluar de base a cima (siguiendo el orden de filtración)
        for stratum in StratumTopology.EVALUATION_ORDER:
            analysis = strata.get(stratum)
            if analysis and analysis.is_compromised:
                verdict_code = f"REJECTED_{stratum.name}"
                return verdict_code, SeverityLevel.CRITICO

        # Verificar warnings
        has_warnings = any(
            strata[s].severity == SeverityLevel.ADVERTENCIA
            for s in Stratum
        )

        if has_warnings:
            return "APPROVED", SeverityLevel.ADVERTENCIA

        return "APPROVED", SeverityLevel.OPTIMO

    def _synthesize_narrative(
        self,
        strata: Dict[Stratum, StratumAnalysis],
        verdict_code: str,
    ) -> Tuple[str, List[str]]:
        """
        Sintetiza la narrativa ejecutiva.
        
        Retorna (narrativa, cadena_causal)
        """
        title, base_message = self._get_verdict_info(verdict_code)
        causality_chain = []

        # Si hay rechazo, construir cadena causal
        if verdict_code.startswith("REJECTED_"):
            failed_stratum_name = verdict_code.replace("REJECTED_", "")
            try:
                failed_stratum = Stratum[failed_stratum_name]
                analysis = strata.get(failed_stratum)
                if analysis:
                    causality_chain.append(f"Fallo detectado en: {failed_stratum.name}")
                    causality_chain.append(f"Diagnóstico: {analysis.narrative}")

                    # Agregar estratos invalidados
                    invalidated = StratumTopology.get_strata_above(failed_stratum)
                    if invalidated:
                        causality_chain.append(
                            f"Estratos invalidados por clausura: "
                            f"{', '.join(s.name for s in invalidated)}"
                        )
            except (KeyError, ValueError):
                pass

        # Construir narrativa completa
        narrative_parts = [f"{title}\n", base_message]

        if causality_chain:
            narrative_parts.append("\n\n**Cadena Causal:**")
            for item in causality_chain:
                narrative_parts.append(f"\n> {item}")

        return "".join(narrative_parts), causality_chain

    # ========================================================================
    # EVIDENCIA FORENSE
    # ========================================================================

    def _extract_forensic_evidence(
        self,
        strata: Dict[Stratum, StratumAnalysis],
        verdict_code: str,
    ) -> List[Issue]:
        """
        Extrae evidencia forense relevante.
        
        Prioriza issues del estrato de causa raíz.
        """
        evidence: List[Issue] = []

        # Determinar estrato de causa raíz
        root_cause_stratum: Optional[Stratum] = None

        for stratum in StratumTopology.EVALUATION_ORDER:
            analysis = strata.get(stratum)
            if analysis and analysis.is_compromised:
                root_cause_stratum = stratum
                break

        if root_cause_stratum:
            # Extraer issues críticos del estrato que falló
            target_analysis = strata[root_cause_stratum]
            for issue in target_analysis.issues:
                if issue.is_critical:
                    evidence.append(issue.with_stratum(root_cause_stratum))
        else:
            # Sin fallos críticos: recolectar warnings relevantes
            for stratum in StratumTopology.EVALUATION_ORDER:
                analysis = strata.get(stratum)
                if analysis:
                    for issue in analysis.issues:
                        if issue.severity == SeverityLevel.ADVERTENCIA:
                            evidence.append(issue.with_stratum(stratum))

        # Ordenar por profundidad (issues más superficiales primero)
        evidence.sort(key=lambda i: (i.depth, i.source))

        return evidence[:self.config.MAX_FORENSIC_EVIDENCE]

    # ========================================================================
    # RECOMENDACIONES
    # ========================================================================

    def _generate_recommendations(
        self,
        strata: Dict[Stratum, StratumAnalysis],
        verdict_code: str,
    ) -> List[str]:
        """Genera recomendaciones accionables."""
        recommendations: List[str] = []

        if verdict_code == "APPROVED":
            recommendations.append(
                "Sistema operando correctamente. Mantener monitoreo regular."
            )
            return recommendations

        # Recomendaciones por tipo de rechazo
        if "PHYSICS" in verdict_code:
            recommendations.extend([
                "Verificar la integridad de los datos de entrada.",
                "Revisar conexiones y fuentes de datos.",
                "Analizar logs del FluxCondenser para identificar puntos de fallo.",
            ])

        elif "TACTICS" in verdict_code:
            recommendations.extend([
                "Revisar el grafo de dependencias del proyecto.",
                "Verificar que no existan referencias circulares.",
                "Validar la estructura jerárquica de los datos.",
            ])

        elif "STRATEGY" in verdict_code:
            recommendations.extend([
                "Revisar parámetros del modelo financiero.",
                "Considerar análisis de sensibilidad.",
                "Evaluar escenarios alternativos.",
            ])

        elif "WISDOM" in verdict_code:
            recommendations.extend([
                "Revisar la configuración del generador de respuestas.",
                "Verificar que los datos de entrada sean completos.",
            ])

        return recommendations[:5]

    # ========================================================================
    # MODOS ESPECIALES
    # ========================================================================

    def _summarize_legacy(self, context: TelemetryContext) -> Dict[str, Any]:
        """
        Modo compatibilidad para contextos sin spans.
        
        Trata todos los errores como fallos en PHYSICS.
        """
        has_errors = len(context.errors) > 0

        # Construir análisis de estratos vacíos
        strata_analysis: Dict[Stratum, StratumAnalysis] = {}

        for stratum in Stratum:
            is_physics = (stratum == Stratum.PHYSICS)
            severity = (
                SeverityLevel.CRITICO if (is_physics and has_errors)
                else SeverityLevel.OPTIMO
            )
            narrative = (
                "Fallo legacy detectado." if (is_physics and has_errors)
                else "Nivel inactivo."
            )

            strata_analysis[stratum] = StratumAnalysis(
                stratum=stratum,
                severity=severity,
                narrative=narrative,
                phases=[],
                issues=[],
                metrics={"duration_total": 0.0, "warnings": 0, "phases": 0},
            )

        # Agregar issues de errores legacy
        if has_errors:
            legacy_issues = []
            for err in context.errors:
                legacy_issues.append(Issue(
                    source=err.get("step", "legacy"),
                    message=err.get("message", "Unknown error"),
                    issue_type=err.get("type", "Error"),
                    depth=0,
                    topological_path=("legacy",),
                    stratum=Stratum.PHYSICS,
                    severity=SeverityLevel.CRITICO,
                ))
            strata_analysis[Stratum.PHYSICS].issues = legacy_issues

        verdict_code = "REJECTED_PHYSICS" if has_errors else "APPROVED"
        global_severity = SeverityLevel.CRITICO if has_errors else SeverityLevel.OPTIMO

        title, description = self._get_verdict_info(verdict_code)

        report = PyramidalReport(
            verdict=title,
            verdict_code=verdict_code,
            executive_summary=f"{title}\n{description}",
            global_severity=global_severity,
            strata_analysis=strata_analysis,
            forensic_evidence=strata_analysis[Stratum.PHYSICS].issues,
            phases=[],
            causality_chain=["Modo legacy: sin telemetría jerárquica"],
            recommendations=["Actualizar a telemetría con spans para mejor diagnóstico."],
        )

        return report.to_dict()

    def _generate_empty_report(self) -> PyramidalReport:
        """Genera reporte vacío para contextos sin datos."""
        empty_strata: Dict[Stratum, StratumAnalysis] = {}

        for stratum in Stratum:
            empty_strata[stratum] = StratumAnalysis(
                stratum=stratum,
                severity=SeverityLevel.OPTIMO,
                narrative="Nivel inactivo.",
                phases=[],
                issues=[],
                metrics={"duration_total": 0.0, "warnings": 0, "phases": 0},
            )

        return PyramidalReport(
            verdict="✅ **SIN ACTIVIDAD**",
            verdict_code="EMPTY",
            executive_summary="Sin telemetría registrada. Contexto vacío.",
            global_severity=SeverityLevel.OPTIMO,
            strata_analysis=empty_strata,
            forensic_evidence=[],
            phases=[],
            causality_chain=[],
            recommendations=["Verificar que el proceso se ejecutó correctamente."],
        )

    def _generate_error_report(self, error_message: str) -> PyramidalReport:
        """Genera reporte de error interno del narrador."""
        error_strata: Dict[Stratum, StratumAnalysis] = {}

        for stratum in Stratum:
            error_strata[stratum] = StratumAnalysis(
                stratum=stratum,
                severity=SeverityLevel.CRITICO if stratum == Stratum.WISDOM else SeverityLevel.OPTIMO,
                narrative="Error en generación de reporte." if stratum == Stratum.WISDOM else "Estado desconocido.",
                phases=[],
                issues=[],
                metrics={},
            )

        # Agregar issue del error
        error_strata[Stratum.WISDOM].issues = [
            Issue(
                source="TelemetryNarrator",
                message=f"Error interno: {error_message}",
                issue_type="NarratorError",
                depth=0,
                topological_path=("narrator", "summarize_execution"),
                stratum=Stratum.WISDOM,
                severity=SeverityLevel.CRITICO,
            )
        ]

        return PyramidalReport(
            verdict="⚠️ **ERROR EN NARRADOR**",
            verdict_code="NARRATOR_ERROR",
            executive_summary=f"Error generando el reporte: {error_message}",
            global_severity=SeverityLevel.CRITICO,
            strata_analysis=error_strata,
            forensic_evidence=error_strata[Stratum.WISDOM].issues,
            phases=[],
            causality_chain=[f"Error interno: {error_message}"],
            recommendations=[
                "Revisar logs del narrador.",
                "Verificar integridad del TelemetryContext.",
            ],
        )


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================


def create_narrator(
    step_mapping: Optional[Dict[str, Stratum]] = None,
    mic: Optional[MICRegistry] = None,
) -> TelemetryNarrator:
    """Factory function para crear un narrador configurado."""
    return TelemetryNarrator(step_mapping=step_mapping, mic=mic)


def summarize_context(context: TelemetryContext) -> Dict[str, Any]:
    """Función de conveniencia para resumir un contexto."""
    narrator = TelemetryNarrator()
    return narrator.summarize_execution(context)
