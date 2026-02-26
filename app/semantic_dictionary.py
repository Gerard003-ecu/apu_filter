"""
Microservicio: Semantic Dictionary (El Guardián de la Ontología)
Estrato DIKW: WISDOM (Nivel 0)

Responsabilidad:
Aloja todas las plantillas narrativas y diccionarios de traducción.
Recibe métricas crudas (telemetry_schemas) y las transforma en "Empatía Táctica"
(prescripciones estratégicas) para el usuario final.
"""

import logging
from typing import Any, Dict, List, Tuple
import random

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

from app.tools_interface import MICRegistry

logger = logging.getLogger("SemanticDictionary")

class SemanticDictionaryService:
    def __init__(self):
        # Aquí se alojan las plantillas que antes vivían en semantic_translator y telemetry_narrative
        self.templates = {
            # ========== SEMANTIC TRANSLATOR ==========
            "TOPOLOGY_CYCLES": {
                "clean": (
                    "✅ **Integridad Estructural (Genus 0)**: No se detectan socavones lógicos "
                    "(β₁ = 0). La Trazabilidad de Carga de Costos fluye verticalmente desde la "
                    "Cimentación hasta el Ápice sin recirculaciones."
                ),
                "minor": (
                    "🔶 **Falla Estructural Local (Genus {beta_1})**: Se detectaron {beta_1} "
                    "socavones lógicos en la estructura de costos. Estos 'agujeros' impiden "
                    "la correcta Trazabilidad de Carga y deben ser corregidos para "
                    "evitar asentamientos diferenciales en el presupuesto."
                ),
                "moderate": (
                    "🚨 **Estructura Geológicamente Inestable (Genus {beta_1})**: "
                    "Se detectó un Genus Estructural de {beta_1}, indicando una estructura tipo 'esponja'. "
                    "Existen múltiples bucles de retroalimentación de costos que "
                    "impiden la Trazabilidad de Carga y hacen colapsar cualquier valoración estática."
                ),
                "critical": (
                    "💀 **COLAPSO TOPOLÓGICO (Genus {beta_1})**: "
                    "La estructura está completamente perforada con {beta_1} ciclos independientes. "
                    "Es matemáticamente imposible calcular costos determinísticos. "
                    "Se requiere rediseño fundamental."
                ),
            },
            "TOPOLOGY_CONNECTIVITY": {
                "empty": "⚠️ **Terreno Vacío**: No hay estructura proyectada (β₀ = 0).",
                "unified": (
                    "🔗 **Unidad de Obra Monolítica**: El proyecto funciona como un solo "
                    "edificio interconectado (β₀ = 1). Todas las cargas tácticas (APUs) "
                    "se transfieren correctamente hacia un único Ápice Estratégico."
                ),
                "fragmented": (
                    "⚠️ **Edificios Desconectados (Fragmentación)**: El proyecto no es una "
                    "estructura única, sino un archipiélago de {beta_0} sub-estructuras aisladas. "
                    "No existe un Ápice unificado que centralice la carga financiera."
                ),
                "severely_fragmented": (
                    "🚨 **Fragmentación Severa**: El proyecto está fragmentado en {beta_0} islas "
                    "completamente desconectadas. Esto indica múltiples proyectos empaquetados "
                    "como uno solo, o datos severamente incompletos."
                ),
            },
            "STABILITY": {
                "critical": (
                    "📉 **COLAPSO POR BASE ESTRECHA (Pirámide Invertida)**: "
                    "Ψ = {stability:.2f}. La Cimentación Logística (Insumos) es demasiado "
                    "angosta para soportar el Peso Táctico (APUs) que tiene encima. "
                    "El centro de gravedad está muy alto; riesgo inminente de vuelco financiero."
                ),
                "warning": (
                    "⚖️ **Equilibrio Precario (Isostático)**: "
                    "Ψ = {stability:.2f}. El proyecto tiene la mínima base necesaria, "
                    "sin redundancia. Cualquier perturbación en el suministro puede "
                    "desestabilizar toda la estructura."
                ),
                "stable": (
                    "⚖️ **Estructura Isostática (Estable)**: "
                    "Ψ = {stability:.2f}. El equilibrio entre la carga de actividades y "
                    "el soporte de insumos es adecuado, aunque no posee redundancia sísmica."
                ),
                "robust": (
                    "🛡️ **ESTRUCTURA ANTISÍSMICA (Resiliente)**: "
                    "Ψ = {stability:.2f}. La Cimentación de Recursos es amplia y redundante. "
                    "El proyecto tiene un bajo centro de gravedad, capaz de absorber "
                    "vibraciones del mercado (volatilidad) sin sufrir daños estructurales."
                ),
            },
            "SPECTRAL_COHESION": {
                "high": (
                    "🔗 **Alta Cohesión del Equipo (Fiedler={fiedler:.2f})**: "
                    "La estructura de costos está fuertemente sincronizada."
                ),
                "standard": (
                    "⚖️ **Cohesión Estándar (Fiedler={fiedler:.2f})**: "
                    "El proyecto presenta un acoplamiento típico entre sus componentes."
                ),
                "low": (
                    "💔 **Fractura Organizacional (Fiedler={fiedler:.3f})**: "
                    "Baja cohesión espectral. Los subsistemas operan aislados, "
                    "riesgo de desalineación en ejecución."
                ),
            },
            "SPECTRAL_RESONANCE": {
                "risk": (
                    "🔊 **RIESGO DE RESONANCIA FINANCIERA (λ={wavelength:.2f})**: "
                    "El espectro de vibración está peligrosamente concentrado. "
                    "Un impacto externo (inflación/escasez) podría amplificarse en toda la "
                    "estructura simultáneamente."
                ),
                "safe": (
                    "🌊 **Disipación Ondulatoria (λ={wavelength:.2f})**: "
                    "La estructura tiene capacidad para amortiguar impactos locales sin entrar en "
                    "resonancia sistémica."
                ),
            },
            "THERMAL_TEMPERATURE": {
                "cold": (
                    "❄️ **Temperatura Estable ({temperature:.1f}°C)**: "
                    "El proyecto está termodinámicamente equilibrado (Precios fríos/fijos)."
                ),
                "stable": (
                    "🌡️ **Temperatura Normal ({temperature:.1f}°C)**: "
                    "Condiciones térmicas estándar del mercado."
                ),
                "warm": (
                    "🌡️ **Calentamiento Operativo ({temperature:.1f}°C)**: "
                    "Existe una exposición moderada a la volatilidad de precios."
                ),
                "hot": (
                    "🔥 **EL PROYECTO TIENE FIEBRE ({temperature:.1f}°C)**: "
                    "El Índice de Inflación Interna es crítico. Los costos de insumos volátiles "
                    "están sobrecalentando la estructura de precios."
                ),
                "critical": (
                    "☢️ **FUSIÓN TÉRMICA ({temperature:.1f}°C)**: "
                    "Temperatura crítica alcanzada. Los costos están en espiral inflacionaria. "
                    "Riesgo de colapso financiero por sobrecalentamiento incontrolado."
                ),
            },
            "THERMAL_ENTROPY": {
                "low": (
                    "📋 **Orden Administrativo (S={entropy:.2f})**: "
                    "Baja entropía indica procesos bien estructurados y datos limpios."
                ),
                "high": (
                    "🌪️ **Alta Entropía ({entropy:.2f})**: Caos administrativo detectado. "
                    "La energía del dinero se disipa en fricción operativa (datos sucios o desorganizados)."
                ),
            },
            "GYROSCOPIC_STABILITY": {
                "stable": "✅ **Giroscopio Estable**: Flujo con momento angular constante.",
                "precession": "⚠️ **Precesión Detectada**: Oscilación lateral en el flujo de datos.",
                "nutation": "🚨 **NUTACIÓN CRÍTICA**: Inestabilidad rotacional. El proceso corre riesgo de colapso inercial."
            },
            "LAPLACE_CONTROL": {
                "robust": "🛡️ **Control Robusto**: Margen de fase sólido (>45°).",
                "marginal": "⚠️ **Estabilidad Marginal**: Respuesta oscilatoria ante transitorios.",
                "unstable": "⛔ **DIVERGENCIA MATEMÁTICA**: Polos en el semiplano derecho (RHP)."
            },
            "PUMP_DYNAMICS": {
                "efficiency_high": (
                    " Eficiencia de Inyección: **ALTA**. "
                    "El costo administrativo de procesar esta información es {joules_per_record:.2e} Joules por registro."
                ),
                "efficiency_low": (
                    " Eficiencia de Inyección: **BAJA**. "
                    "El costo administrativo de procesar esta información es {joules_per_record:.2e} Joules por registro."
                ),
                "water_hammer": (
                    "💥 **Inestabilidad de Tubería**: Se detectaron golpes de ariete (Presión={pressure:.2f}). "
                    "El flujo se detiene bruscamente, causando ondas de choque."
                ),
                "accumulator_pressure": (
                    "🔋 **Presión del Acumulador**: {pressure:.1f}%. Capacidad de amortiguamiento disponible."
                )
            },
            "FINANCIAL_VERDICT": {
                "accept": "🚀 **Veredicto**: VIABLE (IR={pi:.2f}). Estructura financiable.",
                "conditional": "🔵 **Veredicto**: CONDICIONAL (IR={pi:.2f}). Viable con ajustes.",
                "review": "🔍 **Veredicto**: REVISIÓN REQUERIDA.",
                "reject": "🛑 **Veredicto**: RIESGO CRÍTICO (IR={pi:.2f}). No procedente.",
            },
            "FINAL_VERDICTS": {
                "synergy_risk": (
                    "🛑 **PARADA DE EMERGENCIA (Efecto Dominó)**: Se detectaron ciclos interconectados "
                    "que comparten recursos críticos. El riesgo no es aditivo, es multiplicativo. "
                    "Cualquier fallo en el suministro provocará un colapso sistémico en múltiples frentes. "
                    "Desacoplar los ciclos antes de continuar."
                ),
                "inverted_pyramid_viable": (
                    "⚠️ **PRECAUCIÓN LOGÍSTICA (Estructura Inestable)**: Aunque los números "
                    "financieros cuadran, el proyecto es una **Pirámide Invertida** (Ψ={stability:.2f}). "
                    "Se sostiene sobre una base de recursos demasiado estrecha. "
                    "RECOMENDACIÓN: Ampliar la base de proveedores antes de construir."
                ),
                "inverted_pyramid_reject": (
                    "❌ **PROYECTO INVIABLE (Riesgo de Colapso)**: Combinación letal de "
                    "inestabilidad estructural (Pirámide Invertida) e inviabilidad financiera. "
                    "No proceder bajo ninguna circunstancia sin rediseño total."
                ),
                "has_holes": (
                    "🛑 **DETENER PARA REPARACIONES**: Se detectaron {beta_1} socavones "
                    "lógicos (ciclos). No se puede verter dinero en una estructura con agujeros. "
                    "Sanear la topología antes de aprobar presupuesto."
                ),
                "certified": (
                    "✅ **CERTIFICADO DE SOLIDEZ**: Estructura piramidal estable, sin socavones "
                    "lógicos y financieramente viable. Proceder a fase de ejecución."
                ),
                "review_required": (
                    "🔍 **REVISIÓN TÉCNICA REQUERIDA**: La estructura es sólida pero los números no convencen."
                ),
                "analysis_failed": (
                    "⚠️ ANÁLISIS ESTRUCTURAL INTERRUMPIDO: Se detectaron inconsistencias matemáticas "
                    "o falta de datos críticos que impiden certificar la solidez del proyecto. "
                    "Revise los errores en las secciones técnicas."
                ),
            },
            "MISC": {
                "MAYER_VIETORIS": (
                    "🧩 **Incoherencia de Integración**: La fusión de los presupuestos ha generado "
                    "{delta_beta_1} ciclos lógicos fantasmas (Anomalía de Mayer-Vietoris). "
                    "Los datos individuales son válidos, pero su unión crea una contradicción topológica."
                ),
                "THERMAL_DEATH": (
                    "☢️ **MUERTE TÉRMICA DEL SISTEMA**: La entropía ha alcanzado el equilibrio máximo. "
                    "No hay energía libre para procesar información útil."
                ),
                "SYNERGY": (
                    "🔥 **Riesgo de Contagio (Efecto Dominó)**: Se detectó una 'Sinergia de Riesgo' "
                    "en {count} puntos de intersección crítica. Los errores no son aislados; si uno falla, "
                    "provocará una reacción en cadena a través de los frentes de obra compartidos."
                ),
                "EULER_EFFICIENCY": (
                    "🕸️ **Sobrecarga de Gestión (Entropía)**: La eficiencia de Euler es baja ({efficiency:.2f}). "
                    "Existe una complejidad innecesaria de enlaces que dificulta la supervisión y aumenta "
                    "los costos indirectos de administración."
                ),
                "CYCLE_PATH": (
                    "🔄 **Ruta del Ciclo Detectada**: La circularidad sigue el camino: [{path}]. "
                    "Esto significa que el costo de '{first_node}' depende indirectamente de sí mismo, "
                    "creando una indeterminación matemática en la valoración."
                ),
                "STRESS_POINT": (
                    "⚡ **Punto de Estrés Estructural**: El elemento '{node}' actúa como una 'Piedra Angular' crítica, "
                    "soportando {degree} conexiones directas. Una variación en su precio o disponibilidad "
                    "impactará desproporcionadamente a toda la estructura del proyecto (Punto Único de Falla)."
                ),
                "WACC": "💰 **Costo de Oportunidad**: WACC = {wacc:.2%}.",
                "CONTINGENCY": "📊 **Blindaje Financiero**: Contingencia sugerida de ${contingency:,.2f}.",
            },

            # ========== TELEMETRY NARRATIVE ==========
            "TELEMETRY_SUCCESS": {
                "PHYSICS": (
                    "✅ **Cimentación Estable**: "
                    "Flujo laminar de datos confirmado. Sin turbulencia (Flyback). "
                    "La base física del proyecto es sólida."
                ),
                "TACTICS": (
                    "✅ **Estructura Coherente**: "
                    "Topología conexa (β₀=1) y acíclica (β₁=0). "
                    "El grafo de dependencias es válido."
                ),
                "STRATEGY": (
                    "✅ **Viabilidad Confirmada**: "
                    "El modelo financiero es robusto ante la volatilidad. "
                    "Los indicadores de riesgo están dentro de umbrales aceptables."
                ),
                "WISDOM": (
                    "✅ **Síntesis Completa**: "
                    "Respuesta generada exitosamente. "
                    "Todas las capas del análisis convergen."
                ),
            },
            "TELEMETRY_WARNINGS": {
                "PHYSICS": (
                    "⚠️ **Señales de Turbulencia**: "
                    "Se detectaron fluctuaciones en el flujo de datos. "
                    "Monitorear la situación."
                ),
                "TACTICS": (
                    "⚠️ **Estructura Subóptima**: "
                    "El grafo presenta redundancias o complejidad excesiva. "
                    "Considerar simplificación."
                ),
                "STRATEGY": (
                    "⚠️ **Sensibilidad Alta**: "
                    "El modelo financiero es sensible a variaciones. "
                    "Realizar análisis de escenarios."
                ),
                "WISDOM": (
                    "⚠️ **Síntesis Parcial**: "
                    "La respuesta se generó con algunas limitaciones. "
                    "Revisar calidad de inputs."
                ),
            },
            "TELEMETRY_FAILURES_PHYSICS": {
                "default": (
                    "🔥 **Falla en Cimentación**: "
                    "Se detectó inestabilidad física (Saturación/Flyback). "
                    "Los datos no son confiables."
                ),
                "saturation": (
                    "⚡ **Sobrecarga Detectada**: "
                    "El sistema alcanzó saturación crítica. "
                    "Reducir carga o escalar recursos."
                ),
                "corruption": (
                    "💥 **Datos Corruptos**: "
                    "La integridad de los datos de entrada está comprometida. "
                    "Verificar fuentes."
                ),
                "nutation": (
                    "🚨 **NUTACIÓN CRÍTICA**: "
                    "Inestabilidad rotacional detectada. El proceso corre riesgo de colapso inercial "
                    "por oscilaciones no amortiguadas."
                ),
                "thermal_death": (
                    "☢️ **MUERTE TÉRMICA DEL SISTEMA**: "
                    "La entropía ha alcanzado el equilibrio máximo. "
                    "No hay energía libre para procesar información útil."
                ),
                "laplace_unstable": (
                    "⛔ **DIVERGENCIA MATEMÁTICA**: "
                    "Polos en el semiplano derecho (RHP). El sistema es intrínsecamente explosivo "
                    "ante variaciones de entrada."
                ),
                "water_hammer": (
                    "🌊 **GOLPE DE ARIETE DETECTADO**: "
                    "Ondas de choque en la tubería de datos (Presión > 0.7). "
                    "Riesgo de ruptura en la persistencia."
                ),
                "high_injection_work": (
                    "💪 **Fase de Ingesta (Sobrecarga)**: "
                    "Alto esfuerzo de inyección detectado. La fricción de los datos "
                    "está consumiendo energía crítica."
                ),
            },
            "TELEMETRY_FAILURES_TACTICS": {
                "default": (
                    "🏗️ **Fragmentación Estructural**: "
                    "El grafo del proyecto está desconectado. "
                    "Existen islas de datos sin conexión."
                ),
                "cycles": (
                    "🔄 **Socavón Lógico Detectado**: "
                    "La estructura contiene bucles infinitos (β₁ > 0). "
                    "El costo es incalculable."
                ),
                "disconnected": (
                    "🧩 **Componentes Aislados**: "
                    "β₀ > 1 indica múltiples componentes desconectados. "
                    "Revisar enlaces entre módulos."
                ),
                "mayer_vietoris": (
                    "🧩 **ANOMALÍA DE INTEGRACIÓN (Mayer-Vietoris)**: "
                    "La fusión de datasets ha generado ciclos lógicos que no existían "
                    "en las fuentes originales. Inconsistencia topológica."
                ),
            },
            "TELEMETRY_FAILURES_STRATEGY": {
                "default": (
                    "📉 **Riesgo Sistémico**: "
                    "Aunque la estructura es válida, "
                    "la simulación financiera proyecta pérdidas."
                ),
                "high_var": (
                    "🎲 **Alta Volatilidad**: "
                    "El VaR excede umbrales aceptables. "
                    "Considerar coberturas o reducir exposición."
                ),
                "negative_npv": (
                    "💸 **Destrucción de Valor**: "
                    "El NPV proyectado es negativo. "
                    "El proyecto no genera valor económico."
                ),
            },
            "TELEMETRY_FAILURES_WISDOM": {
                "default": (
                    "⚠️ **Síntesis Comprometida**: "
                    "Hubo problemas generando la respuesta final. "
                    "Revisar pasos anteriores."
                ),
            },
            "TELEMETRY_VERDICTS": {
                "APPROVED": (
                    "🏛️ **CERTIFICADO DE SOLIDEZ INTEGRAL**\n"
                    "El Consejo valida el proyecto en todas sus dimensiones: "
                    "Físicamente estable, Topológicamente conexo y Financieramente viable."
                ),
                "REJECTED_PHYSICS": (
                    "⛔ **PROCESO ABORTADO POR INESTABILIDAD FÍSICA**\n"
                    "El Guardián detectó que el flujo de datos es turbulento o corrupto. "
                    "No tiene sentido analizar la estrategia financiera de datos que no existen físicamente."
                ),
                "REJECTED_TACTICS": (
                    "🚧 **VETO ESTRUCTURAL DEL ARQUITECTO**\n"
                    "Los datos son legibles, pero forman una estructura imposible. "
                    "Cualquier cálculo financiero sobre esta base sería una alucinación."
                ),
                "REJECTED_STRATEGY": (
                    "📉 **ALERTA FINANCIERA DEL ORÁCULO**\n"
                    "La estructura es sólida, pero el mercado es hostil o el proyecto no es rentable."
                ),
                "REJECTED_WISDOM": (
                    "⚠️ **FALLO EN SÍNTESIS FINAL**\n"
                    "Todas las capas base son válidas, pero hubo un error generando la respuesta."
                ),
            }
        }

        self.market_contexts = [
            "Suelo Estable: Precios de cemento sin variación significativa.",
            "Terreno Inflacionario: Acero al alza (+2.5%). Reforzar estimaciones.",
            "Vientos de Cambio: Volatilidad cambiaria favorable para importaciones.",
            "Falla Geológica Laboral: Escasez de mano de obra calificada.",
            "Mercado Saturado: Alta competencia presiona márgenes.",
        ]

    def fetch_narrative(self, domain: str, classification: str = None, params: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        [Vector de la MIC] Construye la narrativa basada en el dominio y la clasificación.
        Adapts to MICRegistry calling convention handler(**payload).
        """
        params = params or {}

        # Special handling for MARKET_CONTEXTS (random selection)
        if domain == "MARKET_CONTEXT":
            deterministic = params.get("deterministic", False)
            if deterministic:
                index = params.get("index", 0)
                narrative = self.market_contexts[index % len(self.market_contexts)]
            else:
                narrative = random.choice(self.market_contexts)
            return {
                "success": True,
                "narrative": narrative,
                "stratum": Stratum.WISDOM.name
            }

        try:
            template_group = self.templates.get(domain)
            if not template_group:
                return {"success": False, "error": f"Domain '{domain}' not found"}

            if isinstance(template_group, str):
                # Caso MISC donde el dominio apunta directamente al string si está flat
                # pero aquí usamos MISC para agrupar strings sueltos.
                narrative = template_group.format(**params)
            elif isinstance(template_group, dict):
                template = template_group.get(classification, "⚠️ Estado desconocido.")
                narrative = template.format(**params)
            else:
                narrative = str(template_group)

            return {
                "success": True,
                "narrative": narrative,
                "stratum": Stratum.WISDOM.name
            }
        except Exception as e:
            logger.error(f"Error generando narrativa para {domain}.{classification}: {e}")
            return {"success": False, "error": str(e)}

    def register_in_mic(self, mic: MICRegistry) -> None:
        """Registra el diccionario en la MIC."""
        mic.register_vector(
            service_name="fetch_narrative",
            stratum=Stratum.WISDOM,
            handler=self.fetch_narrative
        )
        logger.info("✅ Diccionario Semántico registrado en la MIC.")
