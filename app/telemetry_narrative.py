"""
Módulo de Narrativa de Telemetría Híbrido (Lattice + DIKW + Topología).

Extiende la lógica algebraica de severidad con una estructura ontológica piramidal.
Transforma la ejecución técnica en un 'Juicio del Consejo'.

Referencias:
- LENGUAJE_CONSEJO.md [1]: Definición de la voz de los sabios.
- schemas.txt [2]: Definición del Enum Stratum.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from itertools import chain
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from app.telemetry import StepStatus, TelemetryContext, TelemetrySpan
from app.schemas import Stratum

# --- Estructuras Algebraicas (Lattice Logic) ---

class SeverityLevel(IntEnum):
    """
    Lattice de severidad con orden total.
    Estructura algebraica: (SeverityLevel, ≤, ⊔, ⊓) forma un lattice acotado completo.
    """
    OPTIMO = 0
    ADVERTENCIA = 1
    CRITICO = 2

    @classmethod
    def from_step_status(cls, status: StepStatus) -> "SeverityLevel":
        mapping = {
            StepStatus.SUCCESS: cls.OPTIMO,
            StepStatus.WARNING: cls.ADVERTENCIA,
            StepStatus.FAILURE: cls.CRITICO,
        }
        if isinstance(status, str):
            try:
                status = StepStatus.from_string(status)
            except (ValueError, AttributeError, KeyError):
                return cls.OPTIMO
        return mapping.get(status, cls.OPTIMO)

    @classmethod
    def supremum(cls, *levels: "SeverityLevel") -> "SeverityLevel":
        if not levels:
            return cls.OPTIMO
        return cls(max(level.value for level in levels))

    @classmethod
    def infimum(cls, *levels: "SeverityLevel") -> "SeverityLevel":
        if not levels:
            return cls.CRITICO
        return cls(min(level.value for level in levels))


# --- Estructuras de Datos (DIKW & Topología) ---

@dataclass
class Issue:
    """Evidencia forense inmutable con profundidad topológica."""
    source: str
    message: str
    issue_type: str
    depth: int
    topological_path: str
    timestamp: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    _NON_CRITICAL_TYPES: frozenset = frozenset({"Warning", "Info", "Metric"})

    @property
    def is_critical(self) -> bool:
        return self.issue_type not in self._NON_CRITICAL_TYPES

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "source": self.source,
            "message": self.message,
            "type": self.issue_type,
            "depth": self.depth,
            "topological_path": self.topological_path,
        }
        if self.timestamp is not None:
            result["timestamp"] = self.timestamp
        if self.context:
            result["context"] = self.context
        return result

@dataclass
class PhaseAnalysis:
    """Resultado del análisis de una fase (span raíz)."""
    name: str
    severity: SeverityLevel
    duration_seconds: float
    issues: List[Issue]
    warning_count: int

    MAX_DISPLAYED_WARNINGS: int = 5

    @property
    def critical_issues(self) -> List[Issue]:
        return [i for i in self.issues if i.is_critical]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if not i.is_critical]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.severity.name,
            "duration": f"{self.duration_seconds:.2f}s",
            "critical_issues": [i.to_dict() for i in self.critical_issues],
            "warning_count": self.warning_count,
            "warnings": [i.to_dict() for i in self.warnings[: self.MAX_DISPLAYED_WARNINGS]],
        }

@dataclass
class StratumAnalysis:
    """
    Análisis consolidado de un Estrato de la Pirámide.
    Representa la salud de una capa completa de realidad (Física, Táctica, Estrategia).
    """
    stratum: Stratum
    severity: SeverityLevel
    narrative: str
    metrics: Dict[str, Any]
    issues: List[Issue]

    @property
    def is_compromised(self) -> bool:
        return self.severity == SeverityLevel.CRITICO

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stratum": self.stratum.name,
            "severity": self.severity.name,
            "narrative": self.narrative,
            "metrics": self.metrics,
            "issues_count": len(self.issues)
        }

@dataclass
class PyramidalReport:
    """
    Reporte final estructurado jerárquicamente (DIKW).
    Sustituye al NarrativeReport plano.
    """
    verdict: str                  # El Juicio Final (Wisdom)
    executive_summary: str        # Narrativa causal (Top-down)
    strata_analysis: Dict[str, Dict[str, Any]] # Detalle por nivel (Bottom-up)
    forensic_evidence: List[Dict[str, Any]]    # Trazabilidad de errores
    phases: List[Dict[str, Any]] = field(default_factory=list) # Compatibilidad

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "executive_summary": self.executive_summary,
            "narrative": self.executive_summary, # Alias de compatibilidad
            "strata_analysis": self.strata_analysis,
            "forensic_evidence": self.forensic_evidence,
            "phases": self.phases # Mantener compatibilidad con UI existente si es necesario
        }


# --- Narrador Piramidal ---

class TelemetryNarrator:
    """
    Narrador que implementa la lógica del 'Consejo de Sabios'.
    Organiza la evidencia forense según la jerarquía DIKW y aplica Clausura Transitiva.
    """

    MAX_RECURSION_DEPTH: int = 150
    MAX_FORENSIC_EVIDENCE: int = 10

    # Mapeo de Pasos del Pipeline a Estratos (Ontología del Sistema)
    STEP_TO_STRATUM = {
        # Nivel 3: FÍSICA (El Guardián) - Ingesta y Estabilidad
        "load_data": Stratum.PHYSICS,
        "merge_data": Stratum.PHYSICS,
        "flux_condenser": Stratum.PHYSICS,
        "final_merge": Stratum.PHYSICS,

        # Nivel 2: TÁCTICA (El Arquitecto) - Estructura y Costos
        "calculate_costs": Stratum.TACTICS,
        "materialization": Stratum.TACTICS,
        "oracle_analyze": Stratum.TACTICS, # Análisis de polos/ceros

        # Nivel 1: ESTRATEGIA (El Oráculo) - Valor y Riesgo
        "business_topology": Stratum.STRATEGY, # Genera el reporte de riesgo
        "financial_analysis": Stratum.STRATEGY,

        # Nivel 0: SABIDURÍA (El Agente) - Síntesis y Salida
        "build_output": Stratum.WISDOM,
        "response_preparation": Stratum.WISDOM
    }

    def summarize_execution(self, context: TelemetryContext) -> Dict[str, Any]:
        """
        Punto de entrada principal.
        Ejecuta la síntesis piramidal.
        """
        root_spans = context.root_spans

        if not root_spans:
            if context.steps or context.errors:
                return self._summarize_legacy(context)
            return self._generate_empty_report()

        # 1. Análisis por Fases (Lógica Algebraica existente)
        phases_analysis = [self._analyze_phase(span) for span in root_spans]

        # 2. Agrupación por Estratos (Lógica Topológica)
        strata_results = self._group_by_stratum(phases_analysis)

        # 3. Construcción de Narrativa Causal (Lógica DIKW)
        narrative, verdict = self._synthesize_wisdom(strata_results)

        # 4. Generación del Reporte
        critical_evidence = self._extract_critical_evidence(strata_results)

        report = PyramidalReport(
            verdict=verdict,
            executive_summary=narrative,
            strata_analysis={k.name: v.to_dict() for k, v in strata_results.items()},
            forensic_evidence=[i.to_dict() for i in critical_evidence[:self.MAX_FORENSIC_EVIDENCE]],
            phases=[p.to_dict() for p in phases_analysis]
        )

        return report.to_dict()

    def _group_by_stratum(self, phases: List[PhaseAnalysis]) -> Dict[Stratum, StratumAnalysis]:
        """Agrupa los análisis de fases en estratos coherentes."""
        grouped = {s: [] for s in Stratum}

        for phase in phases:
            # Identificar estrato del paso
            stratum = self.STEP_TO_STRATUM.get(phase.name, Stratum.PHYSICS) # Default a Physics si desconocido
            grouped[stratum].append(phase)

        results = {}
        # Iterar en orden inverso numérico (PHYSICS=3 -> WISDOM=0) para procesar desde la base
        # Aunque el orden de computación en el dict no importa, la lógica de severidad es independiente
        for stratum in Stratum:
            phases_in_stratum = grouped[stratum]

            # Calcular severidad del estrato (Supremo del lattice)
            if phases_in_stratum:
                severity = SeverityLevel.supremum(*(p.severity for p in phases_in_stratum))
            else:
                severity = SeverityLevel.OPTIMO

            # Generar narrativa específica del nivel
            narrative = self._narrate_stratum(stratum, phases_in_stratum, severity)

            # Agregar métricas (mock simple o agregación real)
            metrics = self._aggregate_metrics(phases_in_stratum)

            results[stratum] = StratumAnalysis(
                stratum=stratum,
                severity=severity,
                narrative=narrative,
                metrics=metrics,
                issues=list(chain.from_iterable(p.issues for p in phases_in_stratum))
            )

        return results

    def _narrate_stratum(self, stratum: Stratum, phases: List[PhaseAnalysis], severity: SeverityLevel) -> str:
        """
        Genera la voz del 'Sabio' correspondiente al nivel.
        Ref: LENGUAJE_CONSEJO.md
        """
        if not phases:
            return "Nivel inactivo."

        if severity == SeverityLevel.OPTIMO:
            if stratum == Stratum.PHYSICS:
                return "✅ **Cimentación Estable**: Flujo laminar de datos confirmado. Sin turbulencia (Flyback)."
            elif stratum == Stratum.TACTICS:
                return "✅ **Estructura Coherente**: Topología conexa (β₀=1) y acíclica (β₁=0)."
            elif stratum == Stratum.STRATEGY:
                return "✅ **Viabilidad Confirmada**: El modelo financiero es robusto ante la volatilidad."
            elif stratum == Stratum.WISDOM:
                return "✅ **Síntesis Completa**: Respuesta generada exitosamente."

        # Narrativa de Fallo
        if severity >= SeverityLevel.ADVERTENCIA:
            if stratum == Stratum.PHYSICS:
                return "🔥 **Falla en Cimentación**: Se detectó inestabilidad física (Saturación/Flyback). Los datos no son confiables."
            elif stratum == Stratum.TACTICS:
                # Buscar evidencia específica de ciclos
                has_cycles = any("ciclo" in i.message.lower() for p in phases for i in p.issues)
                if has_cycles:
                    return "🔄 **Socavón Lógico Detectado**: La estructura contiene bucles infinitos (β₁ > 0). El costo es incalculable."
                return "🏗️ **Fragmentación**: El grafo del proyecto está desconectado (Islas de datos)."
            elif stratum == Stratum.STRATEGY:
                return "📉 **Riesgo Sistémico**: Aunque la estructura es válida, la simulación financiera proyecta pérdidas (VaR alto)."
            elif stratum == Stratum.WISDOM:
                return "⚠️ **Síntesis Comprometida**: Hubo problemas generando la respuesta final."

        return "Ejecución completada con advertencias menores."

    def _synthesize_wisdom(self, strata: Dict[Stratum, StratumAnalysis]) -> Tuple[str, str]:
        """
        La fase de Sabiduría: Determina la causalidad del estado final.
        Aplica la regla de Clausura Transitiva: Fallo abajo implica fallo arriba.
        """
        # 1. Chequeo de Física (Nivel 3 - Base)
        physics = strata[Stratum.PHYSICS]
        if physics.is_compromised:
            return (
                "⛔ **PROCESO ABORTADO POR INESTABILIDAD FÍSICA**\n"
                "El Guardián detectó que el flujo de datos es turbulento o corrupto. "
                "No tiene sentido analizar la estrategia financiera de datos que no existen físicamente.\n"
                f"> Diagnóstico: {physics.narrative}",
                "RECHAZADO_TECNICO"
            )

        # 2. Chequeo de Táctica (Nivel 2 - Estructura)
        tactics = strata[Stratum.TACTICS]
        if tactics.is_compromised:
            return (
                "🚧 **VETO ESTRUCTURAL DEL ARQUITECTO**\n"
                "Los datos son legibles, pero forman una estructura imposible (Pirámide Invertida o Ciclos). "
                "Cualquier cálculo financiero sobre esta base sería una alucinación.\n"
                f"> Diagnóstico: {tactics.narrative}",
                "VETO_ESTRUCTURAL"
            )

        # 3. Chequeo de Estrategia (Nivel 1 - Valor)
        strategy = strata[Stratum.STRATEGY]
        if strategy.is_compromised:
            return (
                "📉 **ALERTA FINANCIERA DEL ORÁCULO**\n"
                "La estructura es sólida, pero el mercado es hostil o el proyecto no es rentable.\n"
                f"> Diagnóstico: {strategy.narrative}",
                "RIESGO_FINANCIERO"
            )

        # 4. Sabiduría (Éxito)
        return (
            "🏛️ **CERTIFICADO DE SOLIDEZ INTEGRAL**\n"
            "El Consejo valida el proyecto en todas sus dimensiones: "
            "Físicamente estable, Topológicamente conexo y Financieramente viable.",
            "APROBADO"
        )

    def _extract_critical_evidence(self, strata: Dict[Stratum, StratumAnalysis]) -> List[Issue]:
        """Extrae solo la evidencia relevante para el nivel de fallo más bajo."""
        # Detectar el nivel más bajo (número más alto) que falló
        failed_stratum = None
        # Iterar PHYSICS(3) -> TACTICS(2) -> STRATEGY(1) -> WISDOM(0)
        for stratum in [Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY, Stratum.WISDOM]:
            if strata[stratum].is_compromised:
                failed_stratum = stratum
                break

        all_evidence = []

        if failed_stratum:
            # Si hay un fallo, priorizamos la evidencia de ese estrato y los inferiores (aunque aquí inferior es numero mayor)
            # La lógica de clausura dice: si falla Physics, solo importa Physics.
            # Si falla Tactics, importa Tactics (y quizás Physics tuvo warnings pero no fallo crítico).
            # Entonces mostramos evidencia del estrato que falló.
            target_analysis = strata[failed_stratum]
            for issue in target_analysis.issues:
                if issue.is_critical:
                    issue.context["stratum"] = failed_stratum.name
                    all_evidence.append(issue)
        else:
            # Si no hay fallos críticos, recolectamos warnings relevantes de todos los estratos
            for analysis in strata.values():
                for issue in analysis.issues:
                    if issue.is_critical: # Debería ser vacío si no hay fallo crítico, pero por seguridad
                         issue.context["stratum"] = analysis.stratum.name
                         all_evidence.append(issue)

        return all_evidence

    def _aggregate_metrics(self, phases: List[PhaseAnalysis]) -> Dict[str, Any]:
        """Agrega métricas básicas de una lista de fases."""
        return {
            "duration_total": sum(p.duration_seconds for p in phases),
            "warnings": sum(p.warning_count for p in phases)
        }

    # --- Métodos de Análisis de Fase (Heredados/Adaptados de la lógica original) ---

    def _analyze_phase(self, span: TelemetrySpan) -> PhaseAnalysis:
        issues = list(self._collect_issues_recursive(span, depth=0, path_prefix=""))
        criticals = [i for i in issues if i.is_critical]
        warnings = [i for i in issues if not i.is_critical]

        direct_severity = SeverityLevel.from_step_status(span.status)
        induced_severity = self._compute_induced_severity(criticals, warnings)
        hierarchy_severity = self._compute_hierarchy_severity(span)

        final_severity = SeverityLevel.supremum(direct_severity, induced_severity, hierarchy_severity)
        duration = span.duration if span.duration is not None else 0.0

        return PhaseAnalysis(
            name=span.name,
            severity=final_severity,
            duration_seconds=duration,
            issues=issues,
            warning_count=len(warnings),
        )

    def _collect_issues_recursive(self, span: TelemetrySpan, depth: int, path_prefix: str) -> Iterator[Issue]:
        current_path = f"{path_prefix} → {span.name}" if path_prefix else span.name

        if depth > self.MAX_RECURSION_DEPTH:
            yield Issue(
                source=span.name,
                message=f"Profundidad máxima excedida ({self.MAX_RECURSION_DEPTH})",
                issue_type="RecursionLimit",
                depth=depth,
                topological_path=current_path,
            )
            return

        yield from self._extract_explicit_errors(span, depth, current_path)

        if self._is_silent_failure(span):
            yield Issue(
                source=span.name,
                message="Fallo estructural sin traza explícita (Silent Failure)",
                issue_type="SilentFailure",
                depth=depth,
                topological_path=current_path,
            )

        if self._is_implicit_warning(span):
            yield Issue(
                source=span.name,
                message="Advertencia de fase (Estado WARNING)",
                issue_type="Warning",
                depth=depth,
                topological_path=current_path,
            )

        yield from self._extract_anomalous_metrics(span, depth, current_path)

        for child in span.children:
            yield from self._collect_issues_recursive(child, depth + 1, current_path)

    def _extract_explicit_errors(self, span: TelemetrySpan, depth: int, path: str) -> Iterator[Issue]:
        for error in span.errors:
            yield Issue(
                source=span.name,
                message=error.get("message", "Error sin mensaje"),
                issue_type=error.get("type", "Error"),
                depth=depth,
                timestamp=error.get("timestamp"),
                topological_path=path,
            )

    def _is_silent_failure(self, span: TelemetrySpan) -> bool:
        return span.status == StepStatus.FAILURE and not span.errors

    def _is_implicit_warning(self, span: TelemetrySpan) -> bool:
        return span.status == StepStatus.WARNING and not span.errors

    def _extract_anomalous_metrics(self, span: TelemetrySpan, depth: int, path: str) -> Iterator[Issue]:
        metrics = getattr(span, "metrics", None)
        if not metrics or not isinstance(metrics, dict):
            return

        for metric_name, metric_data in metrics.items():
            is_anomalous = False
            display_value = metric_data

            if isinstance(metric_data, dict):
                is_anomalous = bool(metric_data.get("anomalous", False))
                display_value = metric_data.get("value", metric_data)

            if is_anomalous:
                yield Issue(
                    source=span.name,
                    message=f"Métrica anómala: {metric_name}={display_value}",
                    issue_type="Warning",
                    depth=depth,
                    topological_path=path,
                )

    def _compute_induced_severity(self, criticals: List[Issue], warnings: List[Issue]) -> SeverityLevel:
        if criticals:
            return SeverityLevel.CRITICO
        if warnings:
            return SeverityLevel.ADVERTENCIA
        return SeverityLevel.OPTIMO

    def _compute_hierarchy_severity(self, span: TelemetrySpan) -> SeverityLevel:
        severity = SeverityLevel.from_step_status(span.status)
        for child in span.children:
            child_severity = self._compute_hierarchy_severity(child)
            severity = SeverityLevel.supremum(severity, child_severity)
        return severity

    def _summarize_legacy(self, context: TelemetryContext) -> Dict[str, Any]:
        """
        Modo compatibilidad para contextos planos (sin jerarquía de spans).
        Asume que cualquier error es un fallo técnico crítico (PHYSICS).
        """
        has_errors = len(context.errors) > 0

        # Generar análisis de estratos dummy
        empty_strata = {}
        for s in Stratum:
            is_physics = (s == Stratum.PHYSICS)
            severity = SeverityLevel.CRITICO.name if (is_physics and has_errors) else SeverityLevel.OPTIMO.name
            narrative = "Fallo legacy detectado." if (is_physics and has_errors) else "Nivel inactivo."

            empty_strata[s.name] = {
                "stratum": s.name,
                "severity": severity,
                "narrative": narrative,
                "metrics": {"duration_total": 0.0, "warnings": 0},
                "issues_count": len(context.errors) if is_physics else 0
            }

        verdict = "RECHAZADO_TECNICO" if has_errors else "APROBADO"
        summary = "Fallo detectado en modo compatibilidad legacy." if has_errors else "Ejecución legacy exitosa."

        # Convertir errores legacy a Issues
        evidence = []
        for err in context.errors:
            evidence.append({
                "source": err.get("step", "legacy"),
                "message": err.get("message", "Unknown error"),
                "type": err.get("type", "Error"),
                "depth": 0,
                "topological_path": "legacy",
                "context": {"stratum": "PHYSICS"}
            })

        return {
            "verdict": verdict,
            "executive_summary": summary,
            "narrative": summary, # Alias de compatibilidad
            "strata_analysis": empty_strata,
            "forensic_evidence": evidence,
            "phases": []
        }

    def _generate_empty_report(self) -> Dict[str, Any]:
        empty_strata = {}
        for s in Stratum:
            empty_strata[s.name] = {
                "stratum": s.name,
                "severity": SeverityLevel.OPTIMO.name,
                "narrative": "Nivel inactivo.",
                "metrics": {"duration_total": 0.0, "warnings": 0},
                "issues_count": 0
            }

        return {
            "verdict": "OPTIMO",
            "executive_summary": "Sin telemetría registrada. Contexto vacío.",
            "strata_analysis": empty_strata,
            "forensic_evidence": [],
            "phases": []
        }
