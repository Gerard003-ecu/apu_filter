"""
Suite de Integración Piramidal: Validación de Clausura Transitiva DIKW.

Objetivo: Garantizar que la incoherencia en un nivel inferior (N+1) 
vete o degrade automáticamente la validez del nivel superior (N).

Jerarquía de Prueba:
1. PHYSICS (Nivel 3): Estabilidad de Flujo (FluxCondenser)
2. TACTICS (Nivel 2): Coherencia Topológica (BusinessTopology)
3. STRATEGY (Nivel 1): Viabilidad Financiera (FinancialEngine)
4. WISDOM (Nivel 0): Veredicto Final (SemanticTranslator)
"""

import pytest
from app.schemas import Stratum
from app.telemetry import TelemetryContext, StepStatus
from app.telemetry_narrative import TelemetryNarrator, SeverityLevel
from app.semantic_translator import SemanticTranslator, VerdictLevel, TopologyMetricsDTO
from app.business_agent import RiskChallenger, ConstructionRiskReport

# ============================================================================
# ESCENARIO 1: EL VETO FÍSICO (Sin Física no hay Economía)
# ============================================================================
def test_physics_veto_propagates_to_wisdom():
    """
    Prueba que una inestabilidad en el flujo de datos (PHYSICS) 
    invalida cualquier análisis financiero posterior.
    Ref: flux_condenser.txt, telemetry_narrative.txt
    """
    # 1. Setup: Contexto con fallo en capa física (Flyback Voltage alto)
    context = TelemetryContext()
    
    # Simulamos el paso del FluxCondenser fallando
    with context.span("flux_condenser", stratum=Stratum.PHYSICS) as span:
        span.status = StepStatus.FAILURE
        # Inyectamos métrica de inestabilidad física
        context.record_metric("flux_condenser", "max_flyback_voltage", 50.0) 
        span.errors.append({
            "message": "Voltaje Flyback Crítico: Ruptura de inercia de datos",
            "type": "PhysicalInstability"
        })

    # Simulamos que, absurdamente, el paso financiero corrió y dio ganancia
    with context.span("financial_analysis", stratum=Stratum.STRATEGY) as span:
        context.record_metric("financial", "roi", 0.50) # 50% ROI (Alucinación)

    # 2. Ejecución: Narrador Piramidal
    narrator = TelemetryNarrator()
    report = narrator.summarize_execution(context)

    # 3. Validación: El veredicto debe ser RECHAZO TÉCNICO, ignorando el ROI
    print(f"\nNarrativa Generada: {report['verdict']}")
    
    assert report["global_severity"] == SeverityLevel.CRITICO
    assert "PHYSICS" in report["root_cause_stratum"].name
    assert "Inestabilidad" in report["executive_summary"]
    
    # La narrativa NO debe celebrar el ROI si la física falló
    assert "Rentabilidad" not in report["executive_summary"]

# ============================================================================
# ESCENARIO 2: LA PARADOJA ESTRUCTURAL (Socavones Lógicos)
# ============================================================================
def test_tactical_veto_on_strategy():
    """
    Prueba que una topología imposible (Ciclos en TACTICS) 
    impide la certificación financiera (STRATEGY), aunque los números sumen.
    Ref: business_topology.txt, metodos.md
    """
    # 1. Setup: Topología con Betti_1 > 0 (Ciclos)
    translator = SemanticTranslator()
    
    # Datos topológicos corruptos (Ciclo A->B->A)
    topology_metrics = TopologyMetricsDTO(
        beta_0=1, 
        beta_1=5, # 5 Ciclos detectados!
        euler_efficiency=0.2
    )
    
    # Datos financieros excelentes (Trampa)
    financial_metrics = {
        "performance": {
            "recommendation": "ACEPTAR",
            "profitability_index": 2.5 # Retorno masivo
        },
        "wacc": 0.08
    }

    # 2. Ejecución: Traducción Semántica
    report = translator.compose_strategic_narrative(
        topological_metrics=topology_metrics,
        financial_metrics=financial_metrics,
        stability=10.0 # Base estable, pero estructura con ciclos
    )

    # 3. Validación: El sabio debe detectar la imposibilidad matemática
    print(f"\nVeredicto Semántico: {report.verdict.name}")
    
    # Debe rechazar o poner en precaución severa
    assert report.verdict >= VerdictLevel.PRECAUCION
    
    # La narrativa debe mencionar los ciclos explícitamente
    assert "Socavón Lógico" in report.raw_narrative or "Ciclos" in report.raw_narrative
    assert "Genus 5" in report.raw_narrative

# ============================================================================
# ESCENARIO 3: LA AUDITORÍA ADVERSARIAL (El Challenger)
# ============================================================================
def test_risk_challenger_detects_inverted_pyramid():
    """
    Prueba específica del RiskChallenger: Detectar la 'Pirámide Invertida'.
    Si Finanzas dice 'Bajo Riesgo' pero Estabilidad (Psi) < 1.0, 
    el Challenger debe intervenir.
    Ref: business_agent.txt [3, 4]
    """
    challenger = RiskChallenger()

    # 1. Crear un reporte preliminar "optimista" pero frágil
    naive_report = ConstructionRiskReport(
        integrity_score=95.0, # Score alto injustificado
        financial_risk_level="BAJO", # El financiero no vio el riesgo estructural
        details={
            "topological_invariants": {
                "pyramid_stability": 0.45 # Ψ < 1.0 (CRÍTICO)
            }
        },
        waste_alerts=[],
        circular_risks=[],
        complexity_level="Baja"
    )

    # 2. Ejecución: El Challenger audita el reporte
    audited_report = challenger.challenge_verdict(naive_report)

    # 3. Validación: Degradación del reporte
    print(f"\nRiesgo Original: {naive_report.financial_risk_level}")
    print(f"Riesgo Auditado: {audited_report.financial_risk_level}")

    assert audited_report.financial_risk_level != "BAJO"
    assert "ESTRUCTURAL" in audited_report.financial_risk_level.upper() or "OCULTO" in audited_report.financial_risk_level.upper()
    
    # El score de integridad debe haber bajado
    assert audited_report.integrity_score < 95.0
    
    # Debe haber evidencia del debate en la narrativa
    assert "CONTRADICCIÓN DETECTADA" in audited_report.strategic_narrative or "ACTA DE DELIBERACIÓN" in audited_report.strategic_narrative
