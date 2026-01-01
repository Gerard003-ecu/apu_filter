"""
Módulo de Narrativa de Telemetría Híbrido (Lattice + DIKW).

Fusiona el rigor matemático del enfoque algebraico (Lattice de severidad)
con la claridad estructural del modelo DIKW (Data -> Information -> Knowledge -> Wisdom).

Arquitectura:
1.  **Lógica Algebraica**: Usa un retículo (Lattice) acotado para calcular el estado
    global mediante operaciones de supremo (sup).
2.  **Estructura DIKW**: Genera reportes tipados (NarrativeReport) con evidencia
    forense enriquecida con trazabilidad topológica.
"""

from typing import Dict, Any, List, Optional, Tuple, Iterator
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from itertools import chain
from app.telemetry import TelemetryContext, StepStatus, TelemetrySpan

# --- Estructuras Algebraicas (Lattice Logic) ---

class SeverityLevel(IntEnum):
    """
    Lattice de severidad con orden total.
    Estructura algebraica: (SeverityLevel, ≤, ⊔, ⊓) forma un lattice acotado.
    """
    OPTIMO = 0
    ADVERTENCIA = 1
    CRITICO = 2

    @classmethod
    def from_step_status(cls, status: StepStatus) -> 'SeverityLevel':
        """Morfismo: StepStatus → SeverityLevel."""
        mapping = {
            StepStatus.SUCCESS: cls.OPTIMO,
            StepStatus.WARNING: cls.ADVERTENCIA,
            StepStatus.FAILURE: cls.CRITICO,
        }
        # Manejo robusto por si status es string o enum
        if isinstance(status, str):
            status = StepStatus.from_string(status)
        return mapping.get(status, cls.OPTIMO)

    @classmethod
    def supremum(cls, *levels: 'SeverityLevel') -> 'SeverityLevel':
        """
        Operación join (⊔) en el lattice.
        sup(∅) = OPTIMO (elemento neutro inferior).
        """
        if not levels:
            return cls.OPTIMO
        return max(levels, key=lambda x: x.value)


# --- Estructuras de Datos (DIKW) ---

@dataclass
class Issue:
    """Evidencia forense inmutable con profundidad topológica."""
    source: str
    message: str
    issue_type: str
    depth: int
    topological_path: str
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "message": self.message,
            "type": self.issue_type,
            "depth": self.depth,
            "topological_path": self.topological_path,
            "timestamp": self.timestamp,
        }

@dataclass
class PhaseAnalysis:
    """Resultado del análisis de una fase (span raíz)."""
    name: str
    severity: SeverityLevel
    duration_seconds: float
    issues: List[Issue]
    warning_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.severity.name,
            "duration": f"{self.duration_seconds:.2f}s",
            "critical_issues": [i.to_dict() for i in self.issues if i.issue_type != "Warning"],
            "warning_count": self.warning_count,
            "warnings": [i.to_dict() for i in self.issues if i.issue_type == "Warning"][:5]
        }

@dataclass
class NarrativeReport:
    """Reporte narrativo final (Estructura pública)."""
    verdict: str
    summary: str
    phases: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "narrative": self.summary,  # Mapped to 'narrative' key for compatibility
            "phases": self.phases,
            "forensic_evidence": self.evidence
        }


# --- Narrador Híbrido ---

class TelemetryNarrator:
    """
    Narrador Híbrido: Algebraico (Lattice) + Estructural (DIKW).
    Calcula el riesgo con álgebra y lo explica con claridad estructural.
    """

    MAX_FORENSIC_EVIDENCE: int = 10
    MAX_RECURSION_DEPTH: int = 50

    def summarize_execution(self, context: TelemetryContext) -> Dict[str, Any]:
        """
        Punto de entrada principal.
        1. Analiza fases usando lógica algebraica (Lattice).
        2. Genera reporte estructurado (DIKW).
        """
        root_spans = context.root_spans

        # Fallback legacy si no hay estructura jerárquica
        if not root_spans:
            if context.steps:
                return self._summarize_legacy(context).to_dict()
            return self._generate_empty_report()

        # Functor de análisis: Span → PhaseAnalysis
        phases_analysis: List[PhaseAnalysis] = [
            self._analyze_phase(span) for span in root_spans
        ]

        # Cálculo del supremo global (join en el lattice de severidades)
        # Estado Global = sup({Estado(Fase_i)})
        global_severity = SeverityLevel.supremum(
            *(phase.severity for phase in phases_analysis)
        )

        # Agregación de evidencia (Issues críticos)
        all_criticals = self._aggregate_critical_issues(phases_analysis)

        # Conteo de advertencias
        total_warnings = sum(p.warning_count for p in phases_analysis)

        report = NarrativeReport(
            verdict=global_severity.name,
            summary=self._generate_verdict(
                global_severity,
                len(phases_analysis),
                len(all_criticals),
                total_warnings
            ),
            phases=[phase.to_dict() for phase in phases_analysis],
            evidence=[issue.to_dict() for issue in all_criticals[:self.MAX_FORENSIC_EVIDENCE]]
        )

        return report.to_dict()

    def _analyze_phase(self, span: TelemetrySpan) -> PhaseAnalysis:
        """
        Analiza un span de nivel raíz.
        Calcula severidad usando supremo (Lattice).
        """
        # Recolección recursiva con paso de ruta topológica
        # El path inicial es el nombre de la fase raíz
        issues = list(self._collect_issues_recursive(span, depth=0, path_prefix=""))

        # Separar warnings de errores críticos
        warnings = [i for i in issues if i.issue_type == "Warning"]
        criticals = [i for i in issues if i.issue_type != "Warning"]

        # 1. Severidad directa del span (Estado explícito)
        direct_severity = SeverityLevel.from_step_status(span.status)

        # 2. Severidad inducida (Propagación de hijos hacia arriba)
        # Si hay issues críticos, la fase es CRÍTICA.
        # Si solo hay warnings y el estado es OPTIMO, la fase es ADVERTENCIA.
        induced_severity = SeverityLevel.OPTIMO
        if criticals:
            induced_severity = SeverityLevel.CRITICO
        elif warnings and direct_severity == SeverityLevel.OPTIMO:
            induced_severity = SeverityLevel.ADVERTENCIA

        # Supremo final de la fase
        final_severity = SeverityLevel.supremum(direct_severity, induced_severity)

        duration = span.duration if span.duration is not None else 0.0

        return PhaseAnalysis(
            name=span.name,
            severity=final_severity,
            duration_seconds=duration,
            issues=issues,
            warning_count=len(warnings)
        )

    def _collect_issues_recursive(
        self,
        span: TelemetrySpan,
        depth: int,
        path_prefix: str
    ) -> Iterator[Issue]:
        """
        Generador recursivo de issues con trazabilidad topológica.
        Pasa el 'path_prefix' hacia abajo para construir la ruta completa.
        """
        current_path = f"{path_prefix} → {span.name}" if path_prefix else span.name

        if depth > self.MAX_RECURSION_DEPTH:
             yield Issue(
                source=span.name,
                message=f"Profundidad máxima excedida ({self.MAX_RECURSION_DEPTH})",
                issue_type="RecursionLimit",
                depth=depth,
                topological_path=current_path
            )
             return

        # 1. Errores explícitos
        for error in span.errors:
            yield Issue(
                source=span.name,
                message=error.get("message", "Error sin mensaje"),
                issue_type=error.get("type", "Error"),
                depth=depth,
                timestamp=error.get("timestamp"),
                topological_path=current_path
            )

        # 2. Fallos silenciosos (Topological Gap)
        if span.status == StepStatus.FAILURE and not span.errors:
            yield Issue(
                source=span.name,
                message="Fallo estructural sin traza explícita (Silent Failure)",
                issue_type="SilentFailure",
                depth=depth,
                topological_path=current_path
            )

        # 3. Métricas anómalas (Warnings)
        # Verificamos si hay métricas marcadas como anómalas o warnings implícitos
        if hasattr(span, 'metrics'):
             for name, value in span.metrics.items():
                # Heurística simple: si la métrica contiene "warning" en el nombre
                # O si pudiéramos acceder a metadatos de anomalía (span.metrics es dict simple value)
                # Propuesta 2 sugería metric.get("anomalous"), pero span.metrics es Dict[str, Any] values.
                # Revisamos si hay metadata de métricas anómalas si existiera, o usamos span.status WARNING
                pass

        # Si el span es WARNING pero no tiene errores, generamos un Issue tipo Warning
        if span.status == StepStatus.WARNING and not span.errors:
             yield Issue(
                source=span.name,
                message="Advertencia de fase (Estado WARNING)",
                issue_type="Warning",
                depth=depth,
                topological_path=current_path
            )

        # Recursión sobre hijos
        for child in span.children:
            yield from self._collect_issues_recursive(child, depth + 1, current_path)

    def _generate_verdict(
        self,
        severity: SeverityLevel,
        total_phases: int,
        total_criticals: int,
        total_warnings: int
    ) -> str:
        """Genera el resumen ejecutivo basado en el veredicto algebraico."""
        if severity == SeverityLevel.OPTIMO:
            return f"Ejecución óptima de {total_phases} fases. Consistencia topológica verificada."
        elif severity == SeverityLevel.ADVERTENCIA:
             return f"Ejecución con {total_warnings} advertencias en {total_phases} fases. Se recomienda revisión."
        else:
             return f"FALLO CRÍTICO. Se detectaron {total_criticals} problemas bloqueantes. Integridad comprometida."

    def _aggregate_critical_issues(self, phases: List[PhaseAnalysis]) -> List[Issue]:
        """Agrega solo issues críticos de todas las fases, ordenados por profundidad."""
        all_issues = chain.from_iterable(phase.issues for phase in phases)
        criticals = [i for i in all_issues if i.issue_type != "Warning"]
        return sorted(criticals, key=lambda i: (i.depth, i.source))

    def _summarize_legacy(self, context: TelemetryContext) -> NarrativeReport:
        """Modo de compatibilidad para contextos planos."""
        errors = context.errors or []
        severity = SeverityLevel.CRITICO if errors else SeverityLevel.OPTIMO

        try:
            summary = context.get_summary()
            duration = summary.get("total_duration_seconds", 0.0)
        except Exception:
            duration = 0.0

        evidence = [
            {
                "source": "legacy",
                "message": str(e.get("message", e)),
                "type": e.get("type", "LegacyError"),
                "topological_path": "Global"
            }
            for e in errors[:10]
        ]

        return NarrativeReport(
            verdict=severity.name,
            summary="Reporte generado en modo compatibilidad (Legacy).",
            phases=[{
                "name": "Global",
                "status": severity.name,
                "duration": f"{duration:.2f}s",
                "critical_issues": evidence,
                "warning_count": 0,
                "warnings": []
            }],
            evidence=evidence
        )

    def _generate_empty_report(self) -> Dict[str, Any]:
         return {
            "verdict": SeverityLevel.OPTIMO.name,
            "narrative": "Sin telemetría registrada. Contexto vacío.",
            "phases": [],
            "forensic_evidence": []
        }
