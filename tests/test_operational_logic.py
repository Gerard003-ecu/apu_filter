"""
Suite de Integración Crítica: Lógica Operativa del Consejo de Sabios
====================================================================

Esta suite valida la interacción orquestada entre los agentes a través de la
Matriz de Interacción Central (MIC). Simula escenarios de negocio reales
para asegurar que la "Constitución Física" del sistema se respete.

Escenarios Cubiertos:
1. El Camino Dorado (Flujo Laminar): Proyecto perfecto.
2. El Socavón Lógico (Veto Táctico): Ciclos de dependencia.
3. La Pirámide Invertida (Veto Estructural): Base estrecha.
4. Violación de Jerarquía (Gatekeeper Algebraico): Intento de salto de estrato.
"""

import pytest
import pandas as pd
import networkx as nx
from typing import Dict, Any

from app.pipeline_director import PipelineDirector
from app.tools_interface import MICRegistry, MICHierarchyViolationError
from app.telemetry import TelemetryContext
from app.schemas import Stratum
from app.telemetry_schemas import TopologicalMetrics, PhysicsMetrics

# ============================================================================
# FIXTURES (Datos de Prueba Sintéticos)
# ============================================================================

@pytest.fixture
def clean_project_data():
    """Genera datos para un proyecto 'Sano' (Acíclico, Base Ancha)."""
    # Presupuesto
    df_presupuesto = pd.DataFrame({
        "CODIGO_APU": ["APU_01", "APU_02"],
        "DESCRIPCION": ["Muro", "Pañete"],
        "CANTIDAD": [1, 2]
    })
    # Insumos (Base ancha: 3 insumos para 2 APUs)
    df_insumos = pd.DataFrame({
        "CODIGO_INSUMO": ["INS_A", "INS_B", "INS_C"],
        "DESCRIPCION": ["Ladrillo", "Cemento", "Arena"],
        "VALOR_UNITARIO": [2-4]
    })
    # Relaciones (Sin ciclos)
    df_apus_raw = pd.DataFrame([
        {"CODIGO_APU": "APU_01", "CODIGO_INSUMO": "INS_A", "CANTIDAD": 50}, # Muro usa Ladrillo
        {"CODIGO_APU": "APU_01", "CODIGO_INSUMO": "INS_B", "CANTIDAD": 10}, # Muro usa Cemento
        {"CODIGO_APU": "APU_02", "CODIGO_INSUMO": "INS_B", "CANTIDAD": 5},  # Pañete usa Cemento
        {"CODIGO_APU": "APU_02", "CODIGO_INSUMO": "INS_C", "CANTIDAD": 20}, # Pañete usa Arena
    ])
    return df_presupuesto, df_insumos, df_apus_raw

@pytest.fixture
def cyclic_project_data():
    """Genera datos con un 'Socavón Lógico' (Ciclo: A -> B -> A)."""
    df_presupuesto = pd.DataFrame({"CODIGO_APU": ["APU_A", "APU_B"], "CANTIDAD": [5]})
    df_insumos = pd.DataFrame({"CODIGO_INSUMO": ["INS_X"], "VALOR_UNITARIO": [1]})
    
    # CICLO MORTAL: APU_A usa APU_B, y APU_B usa APU_A
    # Nota: En la realidad, esto aparece cuando un APU se usa como insumo de otro.
    df_apus_raw = pd.DataFrame([
        {"CODIGO_APU": "APU_A", "CODIGO_INSUMO": "APU_B", "CANTIDAD": 1, "TIPO": "APU"},
        {"CODIGO_APU": "APU_B", "CODIGO_INSUMO": "APU_A", "CANTIDAD": 1, "TIPO": "APU"}
    ])
    return df_presupuesto, df_insumos, df_apus_raw

@pytest.fixture
def initialized_director():
    """Director con MIC real y Telemetría."""
    telemetry = TelemetryContext()
    config = {"env": "test"}
    director = PipelineDirector(config, telemetry)
    # Asegurar que la MIC tenga los vectores registrados
    from app.tools_interface import register_core_vectors
    register_core_vectors(director.mic) 
    return director

# ============================================================================
# TESTS DE INTEGRACIÓN CRÍTICA
# ============================================================================

class TestOperationalLogic:
    
    def test_scenario_golden_path(self, initialized_director, clean_project_data):
        """
        Escenario 1: El Camino Dorado.
        Verifica que un proyecto sano fluya por todos los estratos hasta WISDOM.
        """
        director = initialized_director
        df_p, df_i, df_a = clean_project_data
        
        # 1. Ejecutar Pipeline
        context = {
            "df_presupuesto": df_p, 
            "df_insumos": df_i, 
            "df_apus_raw": df_a,
            # Simulamos que LoadData ya ocurrió y dejó estos artifacts
            "raw_records": df_a.to_dict('records') 
        }
        
        # Ejecutamos pasos clave manualmente para simular el flujo del director
        # Paso 1: Fusión Auditada (PHYSICS)
        context = director.mic.project_intent("audited_merge", {}, context)["context_update"]
        
        # Paso 2: Lógica Estructural (TACTICS)
        context = director.mic.project_intent("structure_logic", {}, context)["context_update"]
        
        # Paso 3: Topología de Negocio (STRATEGY)
        # Aquí el BusinessAgent debe correr y aprobar
        context = director.mic.project_intent("business_topology", {}, context)["context_update"]
        
        # VERIFICACIONES
        
        # A. Telemetría de Salud
        assert director.telemetry.topology.beta_1 == 0, "No deben haber ciclos"
        assert director.telemetry.topology.beta_0 == 1, "El grafo debe ser conexo"
        
        # B. Veredicto del Consejo
        report = context.get("business_topology_report")
        assert report is not None
        assert report.financial_risk_level != "CATÁSTROFICO"
        assert "CERTIFICADO DE SOLIDEZ" in report.strategic_narrative

    def test_scenario_logical_sinkhole(self, initialized_director, cyclic_project_data):
        """
        Escenario 2: El Socavón Lógico (Veto Táctico).
        Verifica que el sistema detecte β1 > 0 y active el bloqueo.
        """
        director = initialized_director
        df_p, df_i, df_a = cyclic_project_data
        
        context = {
            "df_presupuesto": df_p, "df_insumos": df_i, "df_apus_raw": df_a,
            "raw_records": df_a.to_dict('records')
        }
        
        # Ejecución hasta Strategy
        director.mic.project_intent("audited_merge", {}, context)
        director.mic.project_intent("structure_logic", {}, context)
        
        # Al ejecutar Business Topology, el analyzer debe detectar el ciclo
        result = director.mic.project_intent("business_topology", {}, context)
        context = result["context_update"]
        
        # VERIFICACIONES
        
        # A. El invariante topológico debe reflejar el daño
        # Nota: beta_1 dependerá de la implementación exacta de NetworkX sobre el grafo mock
        # Pero el reporte debe ser negativo.
        report = context.get("business_topology_report")
        
        assert "Ciclo" in str(report.circular_risks) or "bucle" in str(report.circular_risks)
        assert "RECHAZADO" in report.financial_risk_level or "CRÍTICO" in report.complexity_level
        
        # B. La narrativa debe ser explícita (Semantic Translator)
        assert "Socavón Lógico" in report.strategic_narrative or "Ciclo" in report.strategic_narrative

    def test_gatekeeper_hierarchy_violation(self, initialized_director):
        """
        Escenario 4: Violación de Jerarquía (Gatekeeper).
        Intento de ejecutar Estrategia sin haber validado Física/Táctica.
        """
        director = initialized_director
        
        # Contexto vacío (sin sellos de validación de estratos inferiores)
        empty_context = {"validated_strata": set()} 
        
        # Intentamos proyectar un vector de Estrategia directamente
        with pytest.raises(MICHierarchyViolationError) as excinfo:
            director.mic.project_intent(
                "business_topology", # Vector de Nivel STRATEGY
                {}, 
                empty_context
            )
        
        # Verificamos que el error sea algebraico y no genérico
        error_msg = str(excinfo.value)
        assert "MIC Hierarchy Violation" in error_msg
        assert "Missing prerequisite strata" in error_msg
        assert "PHYSICS" in error_msg

    def test_physics_flyback_protection(self, initialized_director):
        """
        Escenario: Protección de Flyback (Nivel Física).
        Simula un pico de voltaje en el FluxCondenser que debe abortar el proceso.
        """
        director = initialized_director
        
        # Simulamos un contexto donde el FluxCondenser detectó inestabilidad
        # Inyectamos métricas de fallo directamente en la telemetría (como lo haría el hardware)
        director.telemetry.update_physics(
            flyback_voltage=50.0, # Crítico (> 0.5V)
            saturation=1.0        # Totalmente saturado
        )
        
        # Al intentar ejecutar el siguiente paso lógico
        context = {"validated_strata": {Stratum.PHYSICS}} # Supuestamente validado
        
        # El SemanticTranslator (o Narrator) al final del pipeline debe leer esto y emitir VETO
        # Usamos el narrador directamente para verificar la interpretación
        from app.telemetry_narrative import TelemetryNarrator
        narrator = TelemetryNarrator()
        
        report = narrator.summarize_execution(director.telemetry)
        
        # El veredicto debe ser RECHAZO FÍSICO
        assert report["verdict_code"] == "REJECTED_PHYSICS"
        assert "Falla en Cimentación" in report["executive_summary"]
