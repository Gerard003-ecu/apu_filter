"""
Módulo de Narrativa de Telemetría Híbrido (Lattice + DIKW).

Fusiona el rigor matemático del enfoque algebraico (Lattice de severidad)
con la claridad estructural del modelo DIKW (Data -> Information -> Knowledge -> Wisdom).

Arquitectura:
1.  **Lógica Algebraica**: Usa un retículo (Lattice) acotado para calcular el estado
    global mediante operaciones de supremo (sup) e ínfimo (inf).
2.  **Estructura DIKW**: Genera reportes tipados (NarrativeReport) con evidencia
    forense enriquecida con trazabilidad topológica.
"""

from dataclasses import dataclass
from enum import IntEnum
from itertools import chain
from typing import Any, Dict, Iterator, List, Optional

from app.telemetry import StepStatus, TelemetryContext, TelemetrySpan

# --- Estructuras Algebraicas (Lattice Logic) ---


class SeverityLevel(IntEnum):
    """
    Lattice de severidad con orden total.
    Estructura algebraica: (SeverityLevel, ≤, ⊔, ⊓) forma un lattice acotado completo.

    Propiedades verificadas:
    - ⊥ = OPTIMO (elemento mínimo)
    - ⊤ = CRITICO (elemento máximo)
    - ∀a,b: a ⊔ b = max(a,b), a ⊓ b = min(a,b)
    """

    OPTIMO = 0
    ADVERTENCIA = 1
    CRITICO = 2

    @classmethod
    def from_step_status(cls, status: StepStatus) -> "SeverityLevel":
        """
        Morfismo: StepStatus → SeverityLevel.
        Preserva estructura de orden entre dominios.
        """
        mapping = {
            StepStatus.SUCCESS: cls.OPTIMO,
            StepStatus.WARNING: cls.ADVERTENCIA,
            StepStatus.FAILURE: cls.CRITICO,
        }

        # Normalización robusta de entrada heterogénea
        if isinstance(status, str):
            try:
                status = StepStatus.from_string(status)
            except (ValueError, AttributeError, KeyError):
                return cls.OPTIMO

        return mapping.get(status, cls.OPTIMO)

    @classmethod
    def supremum(cls, *levels: "SeverityLevel") -> "SeverityLevel":
        """
        Operación join (⊔) en el lattice.

        Propiedades algebraicas:
        - sup(∅) = ⊥ = OPTIMO (identidad del join)
        - Asociatividad: sup(a, sup(b, c)) = sup(sup(a, b), c)
        - Conmutatividad: sup(a, b) = sup(b, a)
        - Idempotencia: sup(a, a) = a
        """
        if not levels:
            return cls.OPTIMO
        return cls(max(level.value for level in levels))

    @classmethod
    def infimum(cls, *levels: "SeverityLevel") -> "SeverityLevel":
        """
        Operación meet (⊓) en el lattice.

        inf(∅) = ⊤ = CRITICO (identidad del meet).
        Dual del supremum bajo la relación de orden.
        """
        if not levels:
            return cls.CRITICO
        return cls(min(level.value for level in levels))


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

    # Tipos que no bloquean ejecución
    _NON_CRITICAL_TYPES: frozenset = frozenset({"Warning", "Info", "Metric"})

    @property
    def is_critical(self) -> bool:
        """Predicado: π(issue) → {True, False} para clasificación bloqueante."""
        return self.issue_type not in self._NON_CRITICAL_TYPES

    def to_dict(self) -> Dict[str, Any]:
        """Serialización con exclusión de campos nulos (sparse representation)."""
        result = {
            "source": self.source,
            "message": self.message,
            "type": self.issue_type,
            "depth": self.depth,
            "topological_path": self.topological_path,
        }
        if self.timestamp is not None:
            result["timestamp"] = self.timestamp
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
        """Proyección lazy: π_crítico(Issues)."""
        return [i for i in self.issues if i.is_critical]

    @property
    def warnings(self) -> List[Issue]:
        """Proyección lazy: π_warning(Issues)."""
        return [i for i in self.issues if not i.is_critical]

    def to_dict(self) -> Dict[str, Any]:
        """Serialización estructurada con proyecciones cacheables."""
        return {
            "name": self.name,
            "status": self.severity.name,
            "duration": f"{self.duration_seconds:.2f}s",
            "critical_issues": [i.to_dict() for i in self.critical_issues],
            "warning_count": self.warning_count,
            "warnings": [i.to_dict() for i in self.warnings[: self.MAX_DISPLAYED_WARNINGS]],
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
            "narrative": self.summary,
            "phases": self.phases,
            "forensic_evidence": self.evidence,
        }


# --- Narrador Híbrido ---


class TelemetryNarrator:
    """
    Narrador Híbrido: Algebraico (Lattice) + Estructural (DIKW).

    Composición de funtores:
    - F: TelemetryContext → List[PhaseAnalysis]  (análisis)
    - G: List[PhaseAnalysis] → NarrativeReport   (síntesis)
    - Narrador = G ∘ F
    """

    MAX_FORENSIC_EVIDENCE: int = 10
    MAX_RECURSION_DEPTH: int = 50

    def summarize_execution(self, context: TelemetryContext) -> Dict[str, Any]:
        """
        Punto de entrada principal.
        Composición: Estructuración DIKW ∘ Análisis Algebraico.
        """
        root_spans = context.root_spans

        if not root_spans:
            if context.steps:
                return self._summarize_legacy(context).to_dict()
            return self._generate_empty_report()

        # Functor F: Span → PhaseAnalysis
        phases_analysis: List[PhaseAnalysis] = [
            self._analyze_phase(span) for span in root_spans
        ]

        # Cálculo del supremo global: ⊔{σ(φ) | φ ∈ Fases}
        global_severity = SeverityLevel.supremum(
            *(phase.severity for phase in phases_analysis)
        )

        all_criticals = self._aggregate_critical_issues(phases_analysis)
        total_warnings = sum(p.warning_count for p in phases_analysis)

        report = NarrativeReport(
            verdict=global_severity.name,
            summary=self._generate_verdict(
                global_severity, len(phases_analysis), len(all_criticals), total_warnings
            ),
            phases=[phase.to_dict() for phase in phases_analysis],
            evidence=[
                issue.to_dict() for issue in all_criticals[: self.MAX_FORENSIC_EVIDENCE]
            ],
        )

        return report.to_dict()

    def _analyze_phase(self, span: TelemetrySpan) -> PhaseAnalysis:
        """
        Análisis de span raíz mediante propagación de severidad.

        Teorema de composición:
        σ(Fase) = sup(σ_directa(span), σ_inducida(issues))

        donde σ_inducida = sup({σ(i) | i ∈ Issues(Fase)})
        """
        issues = list(self._collect_issues_recursive(span, depth=0, path_prefix=""))

        # Partición: Issues = Críticos ⊔ Warnings
        criticals = [i for i in issues if i.is_critical]
        warnings = [i for i in issues if not i.is_critical]

        # σ_directa: morfismo desde estado del span
        direct_severity = SeverityLevel.from_step_status(span.status)

        # σ_inducida: propagación bottom-up desde issues
        induced_severity = self._compute_induced_severity(criticals, warnings)

        # Supremo: propiedad fundamental del lattice
        final_severity = SeverityLevel.supremum(direct_severity, induced_severity)

        duration = span.duration if span.duration is not None else 0.0

        return PhaseAnalysis(
            name=span.name,
            severity=final_severity,
            duration_seconds=duration,
            issues=issues,
            warning_count=len(warnings),
        )

    def _compute_induced_severity(
        self, criticals: List[Issue], warnings: List[Issue]
    ) -> SeverityLevel:
        """
        Calcula severidad inducida por presencia de issues.

        Función característica:
        σ_inducida = CRITICO      si |Críticos| > 0
                   = ADVERTENCIA  si |Warnings| > 0 ∧ |Críticos| = 0
                   = OPTIMO       si |Issues| = 0
        """
        if criticals:
            return SeverityLevel.CRITICO
        if warnings:
            return SeverityLevel.ADVERTENCIA
        return SeverityLevel.OPTIMO

    def _collect_issues_recursive(
        self, span: TelemetrySpan, depth: int, path_prefix: str
    ) -> Iterator[Issue]:
        """
        Recorrido DFS con trazabilidad topológica.

        Invariantes:
        - path_prefix: camino desde raíz hasta padre
        - depth: distancia geodésica desde raíz
        - Cada Issue contiene coordenadas exactas en el árbol
        """
        current_path = f"{path_prefix} → {span.name}" if path_prefix else span.name

        # Guard contra ciclos o profundidad excesiva
        if depth > self.MAX_RECURSION_DEPTH:
            yield Issue(
                source=span.name,
                message=f"Profundidad máxima excedida ({self.MAX_RECURSION_DEPTH})",
                issue_type="RecursionLimit",
                depth=depth,
                topological_path=current_path,
            )
            return

        # 1. Extracción de errores explícitos
        yield from self._extract_explicit_errors(span, depth, current_path)

        # 2. Detección de fallos silenciosos (gap topológico)
        if self._is_silent_failure(span):
            yield Issue(
                source=span.name,
                message="Fallo estructural sin traza explícita (Silent Failure)",
                issue_type="SilentFailure",
                depth=depth,
                topological_path=current_path,
            )

        # 3. Detección de warnings implícitos
        if self._is_implicit_warning(span):
            yield Issue(
                source=span.name,
                message="Advertencia de fase (Estado WARNING)",
                issue_type="Warning",
                depth=depth,
                topological_path=current_path,
            )

        # 4. Métricas anómalas
        yield from self._extract_anomalous_metrics(span, depth, current_path)

        # Recursión sobre hijos (inducción estructural)
        for child in span.children:
            yield from self._collect_issues_recursive(child, depth + 1, current_path)

    def _extract_explicit_errors(
        self, span: TelemetrySpan, depth: int, path: str
    ) -> Iterator[Issue]:
        """Proyección de errores explícitos registrados en el span."""
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
        """
        Detecta gap topológico: fallo sin evidencia explícita.
        Predicado: status = FAILURE ∧ errors = ∅
        """
        return span.status == StepStatus.FAILURE and not span.errors

    def _is_implicit_warning(self, span: TelemetrySpan) -> bool:
        """
        Detecta warning sin error asociado.
        Predicado: status = WARNING ∧ errors = ∅
        """
        return span.status == StepStatus.WARNING and not span.errors

    def _extract_anomalous_metrics(
        self, span: TelemetrySpan, depth: int, path: str
    ) -> Iterator[Issue]:
        """
        Extrae issues de métricas anómalas.

        Soporta formatos:
        1. Dict[str, Numeric] - valores simples (sin detección)
        2. Dict[str, Dict] - estructurado: {'value': x, 'anomalous': bool}
        """
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

    def _generate_verdict(
        self,
        severity: SeverityLevel,
        total_phases: int,
        total_criticals: int,
        total_warnings: int,
    ) -> str:
        """
        Morfismo de narrativa: SeverityLevel → String.
        Transforma veredicto algebraico en texto ejecutivo.
        """
        templates = {
            SeverityLevel.OPTIMO: (
                f"Ejecución óptima de {total_phases} fase(s). "
                "Consistencia topológica verificada."
            ),
            SeverityLevel.ADVERTENCIA: (
                f"Ejecución con {total_warnings} advertencia(s) en {total_phases} fase(s). "
                "Se recomienda revisión."
            ),
            SeverityLevel.CRITICO: (
                f"FALLO CRÍTICO: {total_criticals} problema(s) bloqueante(s) detectado(s). "
                "Integridad comprometida."
            ),
        }
        return templates.get(severity, "Estado indeterminado.")

    def _aggregate_critical_issues(self, phases: List[PhaseAnalysis]) -> List[Issue]:
        """
        Agregación de issues críticos con ordenamiento topológico.
        Orden: (depth, source) - BFS lexicográfico.
        """
        all_issues = chain.from_iterable(phase.issues for phase in phases)
        criticals = [issue for issue in all_issues if issue.is_critical]
        return sorted(criticals, key=lambda i: (i.depth, i.source))

    def _summarize_legacy(self, context: TelemetryContext) -> NarrativeReport:
        """
        Modo compatibilidad para contextos planos (sin jerarquía).
        Preserva contrato de salida con proyección a estructura Global.
        """
        errors = context.errors or []
        severity = SeverityLevel.CRITICO if errors else SeverityLevel.OPTIMO
        duration = self._safe_extract_duration(context)

        evidence = [
            {
                "source": "legacy",
                "message": self._safe_error_message(e),
                "type": self._safe_error_type(e),
                "depth": 0,
                "topological_path": "Global",
            }
            for e in errors[: self.MAX_FORENSIC_EVIDENCE]
        ]

        return NarrativeReport(
            verdict=severity.name,
            summary="Reporte generado en modo compatibilidad (Legacy).",
            phases=[
                {
                    "name": "Global",
                    "status": severity.name,
                    "duration": f"{duration:.2f}s",
                    "critical_issues": evidence,
                    "warning_count": 0,
                    "warnings": [],
                }
            ],
            evidence=evidence,
        )

    def _safe_extract_duration(self, context: TelemetryContext) -> float:
        """Extracción defensiva de duración."""
        try:
            summary = context.get_summary()
            raw_duration = summary.get("total_duration_seconds", 0.0)
            return float(raw_duration) if raw_duration is not None else 0.0
        except (AttributeError, TypeError, ValueError, KeyError):
            return 0.0

    def _safe_error_message(self, error: Any) -> str:
        """Extracción defensiva de mensaje de error."""
        if isinstance(error, dict):
            return str(error.get("message", error))
        return str(error)

    def _safe_error_type(self, error: Any) -> str:
        """Extracción defensiva de tipo de error."""
        if isinstance(error, dict):
            return str(error.get("type", "LegacyError"))
        return "LegacyError"

    def _generate_empty_report(self) -> Dict[str, Any]:
        """
        Genera reporte para contexto vacío.
        Elemento neutro del espacio de reportes.
        """
        return {
            "verdict": SeverityLevel.OPTIMO.name,
            "narrative": "Sin telemetría registrada. Contexto vacío.",
            "phases": [],
            "forensic_evidence": [],
        }
