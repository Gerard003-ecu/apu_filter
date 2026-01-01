"""
Módulo de Narrativa de Telemetría (DIKW).

Transforma los datos crudos de telemetría (spans, logs, métricas) en una
narrativa estructurada siguiendo el modelo DIKW (Data -> Information -> Knowledge -> Wisdom).

Arquitectura Fractal:
1. Sabiduría (Veredicto): Estado global del proceso.
2. Conocimiento (Diagnóstico): Análisis por fases/ramas.
3. Información (Evidencia): Errores específicos y cuellos de botella.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from app.telemetry import TelemetryContext, StepStatus, TelemetrySpan

@dataclass
class NarrativeReport:
    """Reporte narrativo final."""
    verdict: str  # OPTIMO, ADVERTENCIA, CRITICO
    summary: str  # Resumen ejecutivo
    phases: List[Dict[str, Any]]  # Diagnóstico por fases
    evidence: List[Dict[str, Any]]  # Errores raíz / Anomalías

class TelemetryNarrator:
    """
    Narrador que convierte un árbol de telemetría en una historia legible.
    """

    def summarize_execution(self, context: TelemetryContext) -> Dict[str, Any]:
        """
        Punto de entrada principal. Genera el reporte DIKW.
        """

        # 1. Analizar estructura (Traverse Tree)
        root_spans = context.root_spans

        # Si no hay spans jerárquicos, intentamos inferir de los pasos planos (Legacy fallback)
        if not root_spans and context.steps:
             return self._summarize_legacy(context)

        # 2. Construir diagnóstico por fases (Nivel 1: Ramas principales)
        phases_diagnosis = []
        global_status = "OPTIMO"
        critical_errors = []

        for span in root_spans:
            phase_report = self._analyze_phase(span)
            phases_diagnosis.append(phase_report)

            # Acumular errores críticos
            critical_errors.extend(phase_report["critical_issues"])

            # Determinar peor estado
            if phase_report["status"] == "CRITICO":
                global_status = "CRITICO"
            elif phase_report["status"] == "ADVERTENCIA" and global_status != "CRITICO":
                global_status = "ADVERTENCIA"

        # 3. Generar Veredicto (Cúspide)
        verdict_summary = self._generate_verdict(global_status, len(phases_diagnosis), len(critical_errors))

        return {
            "verdict": global_status,
            "narrative": verdict_summary,
            "phases": phases_diagnosis,
            "forensic_evidence": critical_errors[:10] # Top 10 critical issues
        }

    def _analyze_phase(self, span: TelemetrySpan) -> Dict[str, Any]:
        """Analiza un span de nivel raíz (Fase)."""

        issues = []

        # Recolectar errores recursivamente
        self._collect_issues(span, issues)

        status = "OPTIMO"
        if span.status == StepStatus.FAILURE:
            status = "CRITICO"
        elif span.status == StepStatus.WARNING:
            status = "ADVERTENCIA"
        elif issues:
             # Si el span dice Success pero tiene hijos con errores
             status = "ADVERTENCIA"

        return {
            "name": span.name,
            "status": status,
            "duration": f"{span.duration:.2f}s",
            "critical_issues": issues
        }

    def _collect_issues(self, span: TelemetrySpan, collector: List[Dict[str, Any]]):
        """Busca recursivamente nodos con error."""
        # Si este nodo tiene error directo
        for error in span.errors:
            collector.append({
                "source": span.name,
                "message": error.get("message"),
                "type": error.get("type", "Error"),
                "timestamp": error.get("timestamp")
            })

        # Si el estado es fallido pero no hay error registrado explícitamente
        if span.status == StepStatus.FAILURE and not span.errors:
             collector.append({
                "source": span.name,
                "message": "Fallo silencioso o timeout",
                "type": "SilentFailure"
            })

        for child in span.children:
            self._collect_issues(child, collector)

    def _generate_verdict(self, status: str, total_phases: int, total_errors: int) -> str:
        if status == "OPTIMO":
            return f"Ejecución exitosa de {total_phases} fases. Estabilidad estructural confirmada."
        elif status == "ADVERTENCIA":
            return f"Ejecución completada con {total_errors} advertencias. Se recomienda revisión de fases afectadas."
        else:
            return f"FALLO CRÍTICO. Se detectaron {total_errors} problemas bloqueantes que comprometen la integridad del proceso."

    def _summarize_legacy(self, context: TelemetryContext) -> Dict[str, Any]:
        """Fallback para cuando no se usan spans."""
        errors = context.errors
        status = "OPTIMO"
        if errors:
            status = "CRITICO" if len(errors) > 0 else "ADVERTENCIA" # Simplificación

        # En el resumen legacy, la clave es 'total_duration_seconds' en to_dict(),
        # pero get_summary() devuelve 'total_duration_seconds' en la nueva implementación robusta.
        # Ajustamos para usar la clave correcta del resumen.
        summary = context.get_summary()
        duration = summary.get("total_duration_seconds", 0.0)

        return {
            "verdict": status,
            "narrative": "Reporte generado en modo compatibilidad (sin jerarquía).",
            "phases": [{"name": "Global", "status": status, "duration": f"{duration:.2f}s"}],
            "forensic_evidence": errors[:10]
        }
