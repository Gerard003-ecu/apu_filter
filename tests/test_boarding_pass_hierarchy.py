"""
Suite de Integración: Jerarquía del Pasabordo (The Boarding Pass Hierarchy)
===========================================================================

Este test evalúa la integridad del "Vector de Estado" a medida que viaja por
los estratos de la Malla Agéntica. Valida la regla de oro:
"No hay Estrategia sin Física, ni Sabiduría sin Táctica".

Componentes Evaluados:
----------------------
1. TelemetryContext (El Pasaporte)
2. TelemetrySchemas (La Constitución - Invariantes)
3. TelemetryNarrator (El Juez - Clausura Transitiva)
4. SemanticTranslator (El Orador - Síntesis)
"""

import pytest
from app.telemetry import TelemetryContext
from app.telemetry_narrative import TelemetryNarrator
from app.semantic_translator import SemanticTranslator, VerdictLevel
from app.telemetry_schemas import (
    PhysicsMetrics, 
    TopologicalMetrics, 
    ThermodynamicMetrics,
    ControlMetrics
)
from app.schemas import Stratum

class TestBoardingPassHierarchy:

    @pytest.fixture
    def narrator(self):
        return TelemetryNarrator()

    @pytest.fixture
    def translator(self):
        return SemanticTranslator()

    def test_physics_veto_overrides_financial_success(self, narrator, translator):
        """
        ESCENARIO: El 'Espejismo Financiero'.
        
        Situación:
        - El Oráculo Financiero dice que el proyecto es una mina de oro (ROI alto).
        - Pero el Guardián Físico detecta una violación de conservación de energía (Hamiltoniano).
        
        Resultado Esperado:
        - El sistema debe emitir un VETO FÍSICO.
        - La narrativa debe hablar de "Inestabilidad Numérica", no de ganancias.
        """
        # 1. Crear el Pasabordo (Contexto)
        ctx = TelemetryContext()
        
        # 2. Inyectar Falla Física (Hamiltonian Excess > Tolerancia)
        # Esto simula que el FluxCondenser detectó datos corruptos que crean energía.
        ctx.update_physics(
            hamiltonian_excess=0.05,  # 5% de error (CRÍTICO)
            kinetic_energy=100.0
        )
        
        # 3. Inyectar Éxito Financiero (Estrategia)
        # Simulamos que, ignorando la física, los números financieros se ven bien.
        ctx.record_metric("financial_analysis", "roi", 0.25) # 25% ROI
        ctx.record_metric("financial_analysis", "npv", 1_000_000)
        
        # 4. Ejecutar el Juicio del Narrador
        report = narrator.summarize_execution(ctx)
        
        # VERIFICACIÓN DEL VETO
        assert report["verdict_code"] == "REJECTED_PHYSICS", \
            "El fallo físico debe prevalecer sobre el éxito financiero."
        
        assert "Violación de Conservación de Energía" in report["executive_summary"], \
            "La narrativa debe identificar la causa raíz física."

    def test_tactical_paradox_mayer_vietoris(self, narrator, translator):
        """
        ESCENARIO: La 'Paradoja Topológica'.
        
        Situación:
        - La física es estable.
        - Pero la fusión de presupuestos crea ciclos fantasmas (Mayer-Vietoris).
        
        Resultado Esperado:
        - Veto Táctico.
        - El Traductor debe explicar la incoherencia de integración.
        """
        ctx = TelemetryContext()
        
        # 1. Física Sana
        ctx.update_physics(hamiltonian_excess=0.00001) # Despreciable
        
        # 2. Falla Táctica (Discrepancia Homológica)
        ctx.update_topology(
            beta_1=1,                # Hay 1 ciclo
            mayer_vietoris_delta=1   # Y ese ciclo nació de la fusión (Incoherencia)
        )
        
        # 3. Traducción Semántica
        # Simulamos la extracción de DTOs que haría el pipeline
        topo_metrics = ctx.topology
        
        # El traductor genera la narrativa final
        narrative, verdict = translator.translate_topology(topo_metrics)
        
        # VERIFICACIÓN
        assert verdict == VerdictLevel.RECHAZAR
        assert "Incoherencia de Integración" in narrative
        assert "ciclos lógicos fantasmas" in narrative

    def test_thermodynamic_fragility(self, narrator, translator):
        """
        ESCENARIO: La 'Hoja al Viento'.
        
        Situación:
        - Física y Topología perfectas.
        - Pero la Inercia Financiera es muy baja (Estructura frágil ante volatilidad).
        
        Resultado Esperado:
        - Aprobación Condicional (Precaución).
        - Advertencia sobre volatilidad.
        """
        ctx = TelemetryContext()
        
        # 1. Base Sólida
        ctx.update_physics(hamiltonian_excess=0.0)
        ctx.update_topology(beta_1=0, pyramid_stability=1.2)
        
        # 2. Fragilidad Térmica
        ctx.thermodynamics = ThermodynamicMetrics(
            financial_inertia=0.1,       # Muy baja inercia (Hoja al viento)
            system_temperature=30.0
        )
        
        # 3. Análisis del Traductor
        narrative, verdict = translator.translate_thermodynamics(ctx.thermodynamics)
        
        # VERIFICACIÓN
        assert verdict == VerdictLevel.PRECAUCION
        assert "Hoja al Viento" in narrative
        assert "contratos de futuros" in narrative # Consejo específico

    def test_schema_invariant_enforcement(self):
        """
        ESCENARIO: Violación Constitucional.
        
        Situación:
        - Se intenta inyectar datos físicamente imposibles (Energía creada de la nada).
        
        Resultado Esperado:
        - El esquema (TelemetrySchema) debe lanzar una excepción inmediata.
        - Esto valida que los microservicios no puedan comunicarse con datos corruptos.
        """
        with pytest.raises(ValueError) as excinfo:
            # Intentar crear una métrica física con ganancia de energía mágica (negativa excess)
            # Nota: Hamiltonian Excess negativo grande implica creación de energía (imposible).
            PhysicsMetrics(hamiltonian_excess=-100.0)
        
        assert "Violación Termodinámica" in str(excinfo.value)

    def test_full_dikw_flow(self, narrator):
        """
        ESCENARIO: El Camino Dorado (Golden Path).
        
        Situación:
        - Proyecto perfecto en todos los niveles.
        
        Resultado Esperado:
        - Certificado de Solidez Integral.
        """
        ctx = TelemetryContext()
        
        # PHYSICS: Estable
        ctx.update_physics(saturation=0.5, flyback_voltage=0.1)
        
        # TACTICS: Conexo y Acíclico
        ctx.update_topology(beta_0=1, beta_1=0, mayer_vietoris_delta=0)
        
        # STRATEGY: Estable
        ctx.set_control_state(ControlMetrics(is_stable=True, phase_margin_deg=60))
        
        # WISDOM: Narrativa
        report = narrator.summarize_execution(ctx)
        
        assert report["verdict_code"] == "APPROVED"
        assert "CERTIFICADO DE SOLIDEZ INTEGRAL" in report["executive_summary"]
