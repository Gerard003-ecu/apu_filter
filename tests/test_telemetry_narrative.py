"""
Suite de Pruebas para el Narrador Piramidal (PyramidalNarrator)
===============================================================

Verifica la implementación de la lógica DIKW y Clausura Transitiva.
Ref: app/propuesta_telemetria_narrativa_piramidal.txt
"""

import pytest
from app.telemetry import StepStatus, TelemetryContext
from app.telemetry_narrative import TelemetryNarrator, SeverityLevel, Stratum

@pytest.fixture
def narrator() -> TelemetryNarrator:
    return TelemetryNarrator()

@pytest.fixture
def context() -> TelemetryContext:
    return TelemetryContext()

class TestPyramidalLogic:
    """Pruebas de la lógica de clausura transitiva y estratificación."""

    def test_physics_failure_blocks_everything(self, narrator, context):
        """
        Escenario: Falla en 'load_data' (PHYSICS).
        Resultado esperado: RECHAZADO_TECNICO.
        La estrategia (aunque no se ejecute o falle) es irrelevante.
        """
        # Simular fallo en capa física
        with context.span("load_data") as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Corrupted CSV", "type": "DataError"})

        # Simular paso de estrategia (que no debería importar)
        with context.span("business_topology"):
            pass

        report = narrator.summarize_execution(context)

        assert report["verdict"] == "RECHAZADO_TECNICO"
        assert "PROCESO ABORTADO POR INESTABILIDAD FÍSICA" in report["executive_summary"]
        
        # Verificar que la evidencia apunta a Physics
        evidence = report["forensic_evidence"]
        assert len(evidence) > 0
        assert evidence[0]["context"]["stratum"] == "PHYSICS"

    def test_tactics_failure_veto(self, narrator, context):
        """
        Escenario: Física OK, pero falla 'calculate_costs' (TACTICS).
        Resultado esperado: VETO_ESTRUCTURAL.
        """
        # Física OK
        with context.span("load_data"):
            pass
        
        # Táctica Falla
        with context.span("calculate_costs") as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Cyclic Dependency", "type": "TopologyError"})

        report = narrator.summarize_execution(context)

        assert report["verdict"] == "VETO_ESTRUCTURAL"
        assert "VETO ESTRUCTURAL DEL ARQUITECTO" in report["executive_summary"]
        assert report["strata_analysis"]["PHYSICS"]["severity"] == "OPTIMO"
        assert report["strata_analysis"]["TACTICS"]["severity"] == "CRITICO"

    def test_strategy_risk_alert(self, narrator, context):
        """
        Escenario: Física y Táctica OK, falla 'financial_analysis' (STRATEGY).
        Resultado esperado: RIESGO_FINANCIERO.
        """
        with context.span("load_data"): pass
        with context.span("calculate_costs"): pass
        
        with context.span("financial_analysis") as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "VaR Exceeded", "type": "FinancialError"})

        report = narrator.summarize_execution(context)

        assert report["verdict"] == "RIESGO_FINANCIERO"
        assert "ALERTA FINANCIERA DEL ORÁCULO" in report["executive_summary"]

    def test_success_certificate(self, narrator, context):
        """
        Escenario: Todo OK.
        Resultado esperado: APROBADO.
        """
        with context.span("load_data"): pass
        with context.span("calculate_costs"): pass
        with context.span("financial_analysis"): pass
        with context.span("build_output"): pass

        report = narrator.summarize_execution(context)

        assert report["verdict"] == "APROBADO"
        assert "CERTIFICADO DE SOLIDEZ INTEGRAL" in report["executive_summary"]

    def test_warning_narrative_enrichment(self, narrator, context):
        """
        Verifica que los warnings generen narrativa rica específica del estrato.
        """
        # Physics Warning
        with context.span("load_data") as span:
            span.status = StepStatus.WARNING
            # Implicit warning (no error message)

        report = narrator.summarize_execution(context)
        
        physics_narrative = report["strata_analysis"]["PHYSICS"]["narrative"]
        assert "Falla en Cimentación" in physics_narrative or "inestabilidad física" in physics_narrative

    def test_specific_cycle_narrative(self, narrator, context):
        """
        Verifica la detección de palabras clave en mensajes de error (e.g., 'ciclo').
        """
        with context.span("calculate_costs") as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Detectado ciclo infinito en grafo", "type": "LogicError"})

        report = narrator.summarize_execution(context)
        
        tactics_narrative = report["strata_analysis"]["TACTICS"]["narrative"]
        assert "Socavón Lógico" in tactics_narrative

class TestStructureAndLegacy:
    """Pruebas de estructura de reporte y compatibilidad."""

    def test_report_structure_keys(self, narrator, context):
        report = narrator.summarize_execution(context)
        required = ["verdict", "executive_summary", "strata_analysis", "forensic_evidence", "phases"]
        for key in required:
            assert key in report

    def test_strata_analysis_keys(self, narrator, context):
        report = narrator.summarize_execution(context)
        # Check all stratums are present even if empty
        for stratum in ["WISDOM", "STRATEGY", "TACTICS", "PHYSICS"]:
            assert stratum in report["strata_analysis"]

    def test_unknown_step_defaults_to_physics(self, narrator, context):
        """Pasos desconocidos deben ir a PHYSICS por seguridad (default)."""
        with context.span("unknown_step") as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Unknown error", "type": "Error"})

        report = narrator.summarize_execution(context)
        
        # Debe aparecer como fallo en PHYSICS
        assert report["strata_analysis"]["PHYSICS"]["severity"] == "CRITICO"
        assert report["verdict"] == "RECHAZADO_TECNICO"

    def test_extract_critical_evidence_filtering(self, narrator, context):
        """
        La evidencia solo debe venir del estrato más bajo fallido.
        Si falla Physics y Strategy, solo mostrar Physics.
        """
        # Physics Fail
        with context.span("load_data") as s1:
            s1.status = StepStatus.FAILURE
            s1.errors.append({"message": "Physics Fail", "type": "Error"})

        # Strategy Fail
        with context.span("financial_analysis") as s2:
            s2.status = StepStatus.FAILURE
            s2.errors.append({"message": "Strategy Fail", "type": "Error"})
            
        report = narrator.summarize_execution(context)
        
        evidence_messages = [e["message"] for e in report["forensic_evidence"]]
        assert "Physics Fail" in evidence_messages
        assert "Strategy Fail" not in evidence_messages

