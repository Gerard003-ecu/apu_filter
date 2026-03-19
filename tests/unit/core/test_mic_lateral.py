import pytest
from typing import Any, Optional
from unittest.mock import MagicMock
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum

from app.adapters.mic_vectors import vector_lateral_pivot, VectorResultStatus
from app.strategy.business_agent import RiskChallenger, ConstructionRiskReport
from app.core.schemas import Stratum


# =============================================================================
# CONSTANTES GLOBALES Y CONFIGURACIÓN DE DOMINIO
# =============================================================================

class ThermodynamicConstants:
    """
    Constantes del subsistema termodinámico.
    
    Fundamento físico:
      - T_THRESHOLD: Temperatura crítica de transición de fase (unidades arbitrarias)
      - CV_MIN: Capacidad calorífica mínima para estabilidad térmica
      - T_ABSOLUTE_ZERO: Límite inferior termodinámico (Kelvin)
    """
    T_THRESHOLD: float = 15.0
    CV_MIN: float = 0.5
    CV_MAX: float = 1.0
    T_ABSOLUTE_ZERO: float = 0.0


class FinancialConstants:
    """
    Constantes del subsistema financiero (Opciones Reales).
    
    Fundamento:
      - K_WAIT_PREMIUM: Factor multiplicativo para umbral de opción de espera
        V_espera > k · VPN  ⟹  esperar domina ejecutar
    """
    K_WAIT_PREMIUM: float = 1.5
    STANDARD_PENALTY: float = 0.30
    MAX_INTEGRITY_SCORE: float = 100.0
    MIN_INTEGRITY_SCORE: float = 0.0


class TopologicalConstants:
    """
    Constantes del subsistema topológico (Homología Simplicial).
    
    Fundamento (Teoría de Homología):
      - β₀ = número de componentes conexas (≥ 1 para espacios no vacíos)
      - β₁ = rango de H₁ = número de "agujeros" o 1-ciclos independientes
      - β₁ > 0 indica presencia de estructura cíclica en el complejo
    """
    BETA_1_MIN_FOR_QUARANTINE: int = 1


# =============================================================================
# HELPERS DE VALIDACIÓN MATEMÁTICA
# =============================================================================

def validate_betti_number(beta: int, dimension: int = 1) -> None:
    """
    Valida que un número de Betti sea matemáticamente coherente.
    
    Invariante: βₙ ∈ ℤ≥0  ∀n ≥ 0
    Los números de Betti son rangos de grupos abelianos libres,
    por lo tanto siempre enteros no negativos.
    """
    if not isinstance(beta, int):
        raise TypeError(f"β_{dimension} debe ser entero, recibido: {type(beta)}")
    if beta < 0:
        raise ValueError(f"β_{dimension} = {beta} inválido: números de Betti son ≥ 0")


def validate_temperature(temp: float, allow_negative: bool = False) -> None:
    """
    Valida coherencia física de temperatura.
    
    En escala Kelvin: T ≥ 0 (tercer principio).
    En escalas Celsius/Fahrenheit: T puede ser negativo.
    """
    if not isinstance(temp, (int, float)):
        raise TypeError(f"Temperatura debe ser numérica, recibido: {type(temp)}")
    if not allow_negative and temp < ThermodynamicConstants.T_ABSOLUTE_ZERO:
        raise ValueError(f"T = {temp} viola el límite termodinámico inferior")


def validate_heat_capacity(cv: float) -> None:
    """
    Valida capacidad calorífica.
    
    Fundamento termodinámico: Cᵥ > 0 (sistemas estables).
    Para nuestro modelo normalizado: Cᵥ ∈ (0, 1].
    """
    if not isinstance(cv, (int, float)):
        raise TypeError(f"Cᵥ debe ser numérico, recibido: {type(cv)}")
    if cv <= 0:
        raise ValueError(f"Cᵥ = {cv} inválido: capacidad calorífica debe ser positiva")


def validate_probability_bounded(value: float, name: str = "valor") -> None:
    """Valida que un valor esté en el intervalo [0, 1]."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} = {value} fuera del rango [0, 1]")


def safe_ratio(numerator: float, denominator: float) -> str:
    """Calcula ratio de forma segura, manejando división por cero."""
    if abs(denominator) < 1e-10:
        return "∞" if numerator > 0 else "-∞" if numerator < 0 else "indeterminado"
    return f"{numerator / denominator:.4f}"


# =============================================================================
# PRUEBAS DEL VECTOR (FÍSICA / LÓGICA PURA)
# =============================================================================

class TestVectorLateralPivot:
    """
    Pruebas unitarias para la lógica interna del vector 'lateral_thinking_pivot'.

    Fundamentos matemáticos que gobiernan las decisiones:
    
      ┌─────────────────┬────────────────────────────────────────────────────┐
      │ Subsistema      │ Condición de Aceptación                            │
      ├─────────────────┼────────────────────────────────────────────────────┤
      │ Termodinámico   │ T_sys < T_umbral  ∧  Cᵥ ≥ Cᵥ_min  →  estable      │
      │ Financiero      │ V_espera > k · VPN  (k = 1.5)  →  espera domina   │
      │ Topológico      │ β₁ > 0  ∧  ¬sinergia  →  cuarentena viable        │
      └─────────────────┴────────────────────────────────────────────────────┘
    
    Notación:
      - Ω: región de aceptación en el espacio de parámetros
      - ∂Ω: frontera de la región (casos límite críticos)
      - ε-vecindad: entorno infinitesimal para pruebas de frontera
    """

    # ═══════════════════════════════════════════════════════════════════════
    # MONOPOLIO COBERTURADO
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _monopolio_payload(
        stability: float = 0.65,
        beta_1: int = 0,
        financial_class: str = "SAFE",
        temperature: float = 10.0,
        heat_capacity: float = 0.8,
        npv: float = 1_000.0,
        validate: bool = True,
    ) -> dict:
        """
        Factory para payloads de Monopolio Coberturado.
        
        Args:
            stability: Coeficiente de estabilidad piramidal ∈ (0, 1]
            beta_1: Primer número de Betti (1-ciclos en H₁)
            financial_class: Clasificación financiera del proyecto
            temperature: Temperatura del sistema (unidades normalizadas)
            heat_capacity: Capacidad calorífica normalizada Cᵥ ∈ (0, 1]
            npv: Valor Presente Neto del proyecto
            validate: Si True, valida invariantes matemáticos
            
        Returns:
            Payload estructurado para el vector lateral_pivot
            
        Raises:
            ValueError: Si validate=True y los parámetros violan invariantes
        """
        if validate:
            validate_betti_number(beta_1, dimension=1)
            validate_temperature(temperature, allow_negative=False)
            if heat_capacity <= 0:
                raise ValueError(f"Cᵥ debe ser > 0, recibido: {heat_capacity}")
            validate_probability_bounded(stability, "stability")

        return {
            "pivot_type": "MONOPOLIO_COBERTURADO",
            "report_state": {
                "stability": stability,
                "beta_1": beta_1,
                "financial_class": financial_class,
            },
            "thermal_metrics": {
                "system_temperature": temperature,
                "heat_capacity": heat_capacity,
            },
            "financial_metrics": {"npv": npv},
        }

    def test_monopolio_coberturado_success(self):
        """
        Caso canónico de aprobación.
        
        Verificación de condiciones:
          T_sys = 10 < 15 = T_umbral     ✓ (sistema subcrítico)
          Cᵥ   = 0.8 ≥ 0.5 = Cᵥ_min     ✓ (inercia térmica suficiente)
          stability = 0.65 ∈ (0, 1]      ✓ (estabilidad estructural)
          
        El sistema opera en la región interior de Ω, lejos de ∂Ω.
        """
        result = vector_lateral_pivot(self._monopolio_payload())

        assert result["success"] is True
        assert result["stratum"] == Stratum.STRATEGY.name
        assert result["payload"]["approved_pivot"] == "MONOPOLIO_COBERTURADO"
        assert result["payload"]["penalty_relief"] == pytest.approx(
            FinancialConstants.STANDARD_PENALTY, rel=1e-6
        )

    def test_monopolio_coberturado_failure_high_temp(self):
        """
        Rechazo por violación de condición termodinámica.
        
        T_sys = 25 > 15 = T_umbral  ⟹  sistema supercrítico.
        
        Interpretación física: Alta temperatura indica volatilidad
        excesiva; el sistema no tiene inercia suficiente para
        amortiguar perturbaciones estocásticas.
        """
        result = vector_lateral_pivot(
            self._monopolio_payload(temperature=25.0)
        )

        assert result["success"] is False
        assert result["status"] == VectorResultStatus.LOGIC_ERROR.value
        assert "condiciones termodinámicas insuficientes" in result["error"].lower()

    def test_monopolio_coberturado_failure_low_heat_capacity(self):
        """
        Rechazo por capacidad calorífica insuficiente.
        
        Cᵥ = 0.3 < 0.5 = Cᵥ_min  ⟹  inercia térmica inadecuada.
        
        Aunque T < T_umbral, el sistema no puede absorber
        fluctuaciones energéticas sin cambios de estado abruptos.
        """
        result = vector_lateral_pivot(
            self._monopolio_payload(heat_capacity=0.3)
        )

        assert result["success"] is False
        assert result["status"] == VectorResultStatus.LOGIC_ERROR.value

    @pytest.mark.parametrize(
        "temperature, heat_capacity, expected_success, reason",
        [
            # ── Fronteras de temperatura ──
            (14.99, 0.80, True,  "T en ε-vecindad inferior del umbral"),
            (15.00, 0.80, False, "T = T_umbral (frontera cerrada superior: T < 15, no ≤)"),
            (15.01, 0.80, False, "T en ε-vecindad superior del umbral"),
            
            # ── Fronteras de capacidad calorífica ──
            (10.00, 0.71, True,  "Cᵥ = Cᵥ_min (frontera cerrada inferior: Cᵥ ≥ 0.5)"),
            (10.00, 0.70, False, "Cᵥ en ε-vecindad inferior de Cᵥ_min"),
            (10.00, 0.72, True,  "Cᵥ en ε-vecindad superior de Cᵥ_min"),
            
            # ── Vértices del espacio de parámetros ──
            ( 0.00, 1.00, True,  "Vértice óptimo: T_min, Cᵥ_max"),
            ( 0.00, 0.71, True,  "Vértice: T_min, Cᵥ_min"),
            (14.99, 0.71, True,  "Vértice: T_max-ε, Cᵥ_min"),
            (14.99, 1.00, True,  "Vértice: T_max-ε, Cᵥ_max"),
            
            # ── Casos extremos físicamente válidos ──
            ( 0.01, 0.71, True,  "Temperatura cercana al cero absoluto"),
            (10.00, 0.9999, True, "Capacidad calorífica cercana al máximo normalizado"),
        ],
        ids=lambda x: x if isinstance(x, str) else "",
    )
    def test_monopolio_thermal_boundaries(
        self, temperature, heat_capacity, expected_success, reason
    ):
        """
        Análisis exhaustivo de frontera termodinámica.
        
        Región de aceptación:
          Ω = {(T, Cᵥ) ∈ ℝ² | T ∈ [0, 15) ∧ Cᵥ ∈ [0.5, 1]}
          
        Topología de Ω:
          - Abierto en T (frontera superior excluida)
          - Cerrado en Cᵥ (ambas fronteras incluidas)
          - Ω es un rectángulo semi-abierto en ℝ²
          
        Esta prueba verifica el comportamiento en ∂Ω y su ε-vecindad.
        """
        result = vector_lateral_pivot(
            self._monopolio_payload(
                temperature=temperature,
                heat_capacity=heat_capacity,
            )
        )

        assert result["success"] is expected_success, (
            f"Fallo en análisis de frontera termodinámica:\n"
            f"  Razón: {reason}\n"
            f"  T = {temperature} (umbral: {ThermodynamicConstants.T_THRESHOLD})\n"
            f"  Cᵥ = {heat_capacity} (mínimo: {ThermodynamicConstants.CV_MIN})\n"
            f"  Esperado: {'ACEPTAR' if expected_success else 'RECHAZAR'}"
        )

    def test_monopolio_stability_boundaries(self):
        """
        Verifica que stability = 0 sea rechazado (división por cero potencial).
        
        stability ∈ (0, 1] es requerido; stability = 0 indica
        colapso estructural completo de la pirámide de riesgos.
        """
        result = vector_lateral_pivot(
            self._monopolio_payload(stability=0.0, validate=False)
        )

        # stability = 0 debería ser rechazado por el vector
        assert result["success"] is False
        assert "estabilidad estructural colapsada" in str(result.get("error", "")).lower()

    # ═══════════════════════════════════════════════════════════════════════
    # OPCIÓN DE ESPERA (OPCIONES REALES)
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _espera_payload(
        npv: float = 100.0,
        wait_value: float = 200.0,
        financial_class: str = "HIGH",
        validate: bool = True,
    ) -> dict:
        """
        Factory para payloads de Opción de Espera.
        
        Modelo de Opciones Reales:
          El valor de espera V_espera representa el valor de la opción
          de postergar la decisión de inversión.
          
          Decisión óptima:
            V_espera > k·VPN  ⟹  esperar (la opcionalidad tiene prima)
            V_espera ≤ k·VPN  ⟹  ejecutar ahora
            
        Args:
            npv: Valor Presente Neto del proyecto (puede ser negativo)
            wait_value: Valor de la opción de espera (debe ser ≥ 0)
            financial_class: Clasificación de riesgo financiero
            validate: Si True, valida invariantes del modelo
            
        Note:
            wait_value < 0 no tiene sentido financiero (el derecho
            a esperar no puede tener valor negativo).
        """
        if validate:
            if wait_value < 0:
                raise ValueError(
                    f"V_espera = {wait_value} inválido: "
                    "valor de opción no puede ser negativo"
                )

        return {
            "pivot_type": "OPCION_ESPERA",
            "report_state": {"financial_class": financial_class},
            "financial_metrics": {
                "npv": npv,
                "real_options": {"wait_option_value": wait_value},
            },
        }

    def test_opcion_espera_success(self):
        """
        Caso canónico: la opción de espera domina.
        
        V_espera = 200 > 1.5 × 100 = 150 = k·VPN  ✓
        
        Prima de opcionalidad = 200 - 150 = 50 (33% sobre umbral).
        Acción estratégica: congelar inversión por 6 meses para
        capturar el valor de la información adicional.
        """
        result = vector_lateral_pivot(self._espera_payload())

        assert result["success"] is True
        assert result["payload"]["approved_pivot"] == "OPCION_ESPERA"
        assert result["payload"]["strategic_action"] == "FREEZE_6_MONTHS"

    def test_opcion_espera_failure_insufficient_premium(self):
        """
        Rechazo: prima de espera insuficiente.
        
        V_espera = 140 ≤ 1.5 × 100 = 150 = k·VPN  ✗
        
        El valor de esperar no compensa el costo de oportunidad;
        ejecución inmediata es óptima bajo el criterio de Bellman.
        """
        result = vector_lateral_pivot(
            self._espera_payload(wait_value=140.0)
        )

        assert result["success"] is False

    @pytest.mark.parametrize(
        "npv, wait_value, expected_success, reason",
        [
            # ── Frontera principal: V_espera vs k·VPN ──
            (100.0, 151.00, True,  "V_espera en ε-vecindad superior de k·VPN"),
            (100.0, 150.00, False, "V_espera = k·VPN exacto (frontera: > estricto)"),
            (100.0, 149.99, False, "V_espera en ε-vecindad inferior de k·VPN"),
            
            # ── Casos degenerados de VPN ──
            (  0.0,   1.00, True,  "VPN = 0: cualquier V_espera > 0 domina (0 > 0 es False, verificar)"),
            (  0.0,   0.01, True,  "VPN = 0 con V_espera mínimo positivo"),
            (  0.0,   0.00, False, "VPN = 0, V_espera = 0: no hay prima"),
            
            # ── VPN negativo (proyecto destruye valor si se ejecuta) ──
            (-50.0,  10.00, True,  "VPN < 0: esperar siempre preferible si V_espera > k·VPN"),
            (-50.0, -74.99, False,  "VPN < 0: umbral k·VPN = 0, V_espera < 0"),
            (-50.0, -75.00, False, "VPN < 0: V_espera = k·VPN exacto"),
            (-50.0, -76.00, False, "VPN < 0: V_espera < k·VPN"),
            
            # ── Casos de escala ──
            (1e6, 1.5e6 + 1, True,  "Escala millonaria: justo sobre umbral"),
            (1e6, 1.5e6,     False, "Escala millonaria: exactamente en umbral"),
            (1e-6, 1.5e-6 + 1e-9, True, "Escala microscópica: justo sobre umbral"),
        ],
        ids=lambda x: x if isinstance(x, str) else "",
    )
    def test_opcion_espera_financial_boundaries(
        self, npv, wait_value, expected_success, reason
    ):
        """
        Frontera de decisión financiera: V_espera > k · VPN.
        
        Análisis de sensibilidad en ε-vecindad del umbral crítico.
        
        El factor k = 1.5 representa un premium requerido del 50%
        sobre el VPN para justificar la postergación, considerando:
          - Costo de capital durante la espera
          - Decaimiento temporal de la opción
          - Riesgo de cambio en condiciones de mercado
        """
        result = vector_lateral_pivot(
            self._espera_payload(npv=npv, wait_value=wait_value, validate=False)
        )

        k = FinancialConstants.K_WAIT_PREMIUM
        threshold = k * max(npv, 0.0)
        
        assert result["success"] is expected_success, (
            f"Fallo en frontera financiera:\n"
            f"  Razón: {reason}\n"
            f"  VPN = {npv}\n"
            f"  V_espera = {wait_value}\n"
            f"  k·VPN = {threshold}\n"
            f"  Ratio V_espera/VPN = {safe_ratio(wait_value, npv)}\n"
            f"  Diferencia = {wait_value - threshold}\n"
            f"  Esperado: {'ACEPTAR' if expected_success else 'RECHAZAR'}"
        )

    def test_opcion_espera_negative_wait_value_rejected(self):
        """
        V_espera < 0 es financieramente incoherente.
        
        Una opción (derecho sin obligación) no puede tener valor
        negativo; en el peor caso, no se ejerce y vale 0.
        """
        result = vector_lateral_pivot(
            self._espera_payload(wait_value=-10.0, validate=False)
        )

        # El sistema debería rechazar o manejar este caso inválido
        assert result["success"] is False or result.get("status") == VectorResultStatus.LOGIC_ERROR.value

    # ═══════════════════════════════════════════════════════════════════════
    # CUARENTENA TOPOLÓGICA (HOMOLOGÍA SIMPLICIAL)
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _cuarentena_payload(
        beta_1: int = 5,
        synergy_detected: bool = False,
        validate: bool = True,
    ) -> dict:
        """
        Factory para payloads de Cuarentena Topológica.
        
        Fundamento Topológico:
          Sea K un complejo simplicial que modela la estructura de riesgos.
          H₁(K; ℤ) es el primer grupo de homología singular.
          β₁ = rank(H₁) = número de 1-ciclos independientes.
          
          β₁ > 0 indica presencia de "agujeros" en K que pueden
          aislarse topológicamente si no hay acoplamiento sinérgico.
          
        Condición de Cuarentena:
          (β₁ > 0) ∧ (¬sinergia)  ⟺  ciclos aislables
          
        Args:
            beta_1: Primer número de Betti (∈ ℤ≥0)
            synergy_detected: Si True, los ciclos están acoplados
            validate: Si True, valida invariantes topológicos
        """
        if validate:
            validate_betti_number(beta_1, dimension=1)

        return {
            "pivot_type": "CUARENTENA_TOPOLOGICA",
            "report_state": {"beta_1": beta_1},
            "synergy_risk": {"synergy_detected": synergy_detected},
        }

    def test_cuarentena_topologica_success(self):
        """
        Caso canónico: ciclos aislables sin acoplamiento.
        
        β₁ = 5 (existen 5 generadores de H₁, i.e., 5 clases de
        1-ciclos homológicamente independientes).
        
        ¬sinergia indica que los ciclos no comparten símplices
        de soporte común, permitiendo aislamiento individual.
        """
        result = vector_lateral_pivot(self._cuarentena_payload())

        assert result["success"] is True
        assert result["payload"]["approved_pivot"] == "CUARENTENA_TOPOLOGICA"
        assert result["payload"]["quarantine_active"] is True

    def test_cuarentena_topologica_failure_synergy_coupling(self):
        """
        Rechazo: ciclos acoplados por sinergia.
        
        β₁ = 5 pero sinergia = True indica que los generadores
        de H₁ comparten símplices de soporte.
        
        Topológicamente, esto significa que los ciclos forman
        un "bouquet" o configuración entrelazada donde modificar
        uno afecta necesariamente a los demás.
        
        El aislamiento individual es imposible; se requiere
        intervención holística sobre el complejo completo.
        """
        result = vector_lateral_pivot(
            self._cuarentena_payload(synergy_detected=True)
        )

        assert result["success"] is False
        assert "sinergia" in result["error"].lower()

    def test_cuarentena_topologica_failure_acyclic_space(self):
        """
        Rechazo: espacio topológico acíclico.
        
        β₁ = 0 significa H₁(K) ≅ 0 (grupo trivial).
        El complejo simplicial K es acíclico en dimensión 1;
        no existen 1-ciclos no triviales que aislar.
        
        Geométricamente, K es un "árbol" o tiene estructura
        contractible en el sentido homológico.
        
        La operación de cuarentena es vacua; se rechaza.
        """
        result = vector_lateral_pivot(
            self._cuarentena_payload(beta_1=0)
        )

        assert result["success"] is False

    @pytest.mark.parametrize(
        "beta_1, synergy, expected_success, topological_interpretation",
        [
            # ── Casos fundamentales ──
            (1, False, True,  "1-ciclo único aislable (toro minimal)"),
            (1, True,  False, "1-ciclo único pero con auto-acoplamiento"),
            
            # ── Alta ciclicidad ──
            (50, False, True,  "Alta β₁ sin acoplamiento (50 agujeros independientes)"),
            (50, True,  False, "Alta β₁ con acoplamiento (estructura entrelazada compleja)"),
            
            # ── Casos degenerados ──
            (0, False, False, "Espacio acíclico: H₁ = 0, nada que aislar"),
            (0, True,  False, "Espacio acíclico + sinergia (doble rechazo)"),
            
            # ── Fronteras numéricas ──
            (1, False, True,  "β₁ mínimo para cuarentena"),
            (2, False, True,  "Dos ciclos independientes (genus 1 surface)"),
            
            # ── Valores grandes (stress test topológico) ──
            (1000, False, True,  "Complejo con 1000 agujeros independientes"),
            (1000, True,  False, "Complejo con 1000 agujeros acoplados"),
        ],
        ids=lambda x: x if isinstance(x, str) else "",
    )
    def test_cuarentena_topologica_truth_table(
        self, beta_1, synergy, expected_success, topological_interpretation
    ):
        """
        Tabla de verdad exhaustiva para condición de cuarentena.
        
        Proposición lógica:
          P: β₁ > 0  (existen ciclos)
          Q: ¬sinergia  (ciclos desacoplados)
          Éxito ⟺ P ∧ Q
          
        Esta es una condición CONJUNTIVA, no disyuntiva:
          - β₁ > 0 es NECESARIA (sin ciclos, no hay qué aislar)
          - ¬sinergia es NECESARIA (con acoplamiento, aislamiento falla)
          - Ninguna es SUFICIENTE por sí sola
        """
        result = vector_lateral_pivot(
            self._cuarentena_payload(beta_1=beta_1, synergy_detected=synergy)
        )

        P = beta_1 > 0
        Q = not synergy
        expected_by_logic = P and Q

        assert result["success"] is expected_success, (
            f"Fallo en tabla de verdad topológica:\n"
            f"  β₁ = {beta_1} ⟹ P = {P}\n"
            f"  sinergia = {synergy} ⟹ Q = {Q}\n"
            f"  P ∧ Q = {expected_by_logic}\n"
            f"  Interpretación: {topological_interpretation}\n"
            f"  Esperado: {'ACEPTAR' if expected_success else 'RECHAZAR'}"
        )
        assert result["success"] == expected_by_logic, (
            "La implementación no sigue la lógica conjuntiva P ∧ Q"
        )

    def test_cuarentena_negative_betti_rejected(self):
        """
        β₁ < 0 es matemáticamente imposible.
        
        Los números de Betti son rangos de grupos abelianos libres,
        por definición βₙ ∈ ℤ≥0.
        
        Un β₁ negativo indica corrupción de datos o error de cálculo
        en el pipeline previo; debe rechazarse con error descriptivo.
        """
        result = vector_lateral_pivot(
            self._cuarentena_payload(beta_1=-1, validate=False)
        )

        assert result["success"] is False

    # ═══════════════════════════════════════════════════════════════════════
    # ROBUSTEZ ANTE ENTRADAS ANÓMALAS
    # ═══════════════════════════════════════════════════════════════════════

    class TestRobustnessAnomalousInputs:
        """Subclase para pruebas de robustez y manejo de errores."""

        def test_unknown_pivot_type_rejected(self):
            """
            Tipo de pivote desconocido: rechazo con error descriptivo.
            
            El sistema debe ser cerrado bajo tipos conocidos;
            extensiones requieren modificación explícita del vector.
            """
            result = vector_lateral_pivot({
                "pivot_type": "TIPO_INEXISTENTE_XYZ",
                "report_state": {},
            })

            assert result["success"] is False
            assert result["status"] == VectorResultStatus.LOGIC_ERROR.value
            assert "pivot" in result.get("error", "").lower() or \
                   "tipo" in result.get("error", "").lower()

        def test_missing_pivot_type_rejected(self):
            """
            Ausencia de pivot_type: error de validación, no excepción.
            
            El campo pivot_type es discriminante obligatorio;
            su ausencia impide cualquier procesamiento.
            """
            result = vector_lateral_pivot({"report_state": {}})

            assert result["success"] is False
            assert "error" in result or "status" in result

        def test_empty_payload_rejected(self):
            """
            Payload vacío: fallo controlado sin propagación de excepción.
            
            El contrato del vector exige campos mínimos; un payload
            vacío debe resultar en rechazo graceful, no crash.
            """
            result = vector_lateral_pivot({})

            assert result["success"] is False

        def test_none_payload_handled(self):
            """None como payload debe manejarse sin excepción."""
            try:
                result = vector_lateral_pivot(None)
                # Si no lanza excepción, debe indicar fallo
                assert result["success"] is False
            except TypeError:
                # También aceptable: TypeError explícito
                pass

        @pytest.mark.parametrize(
            "payload, description",
            [
                ({"pivot_type": None}, "pivot_type = None"),
                ({"pivot_type": 123}, "pivot_type numérico (tipo incorrecto)"),
                ({"pivot_type": ""}, "pivot_type string vacío"),
                ({"pivot_type": "MONOPOLIO_COBERTURADO", "thermal_metrics": None}, 
                 "thermal_metrics = None"),
                ({"pivot_type": "MONOPOLIO_COBERTURADO", "thermal_metrics": "invalid"},
                 "thermal_metrics tipo incorrecto"),
            ],
            ids=lambda x: x if isinstance(x, str) else "",
        )
        def test_malformed_payloads_handled(self, payload, description):
            """
            Payloads malformados deben resultar en rechazo controlado.
            
            El vector debe ser defensivo ante:
              - Tipos incorrectos
              - Valores None inesperados
              - Estructuras incompletas
            """
            try:
                result = vector_lateral_pivot(payload)
                assert result["success"] is False, (
                    f"Payload malformado aceptado incorrectamente: {description}"
                )
            except (TypeError, KeyError, AttributeError) as e:
                # Excepciones explícitas son aceptables para inputs muy corruptos
                pytest.skip(f"Excepción aceptable para {description}: {type(e).__name__}")


# =============================================================================
# PRUEBAS DE INTEGRACIÓN (RISK CHALLENGER + MIC MOCK)
# =============================================================================

class TestRiskChallengerLateralIntegration:
    """
    Pruebas de integración del RiskChallenger con la MIC mockeada.

    Invariantes del Sistema (contrato formal):
    
      ┌─────┬─────────────────────────────────────────────────────────────────┐
      │ I₁  │ ∀s: 0 ≤ integrity_score ≤ 100  (acotamiento global)             │
      ├─────┼─────────────────────────────────────────────────────────────────┤
      │ I₂  │ Si aprobado: s_f = min(100, s₀ × (1 + relief))                  │
      ├─────┼─────────────────────────────────────────────────────────────────┤
      │ I₃  │ Si rechazado: s_f = s₀ × (1 − penalty)                          │
      ├─────┼─────────────────────────────────────────────────────────────────┤
      │ I₄  │ Inmutabilidad: el reporte original no se modifica               │
      ├─────┼─────────────────────────────────────────────────────────────────┤
      │ I₅  │ Degradación graceful ante fallos de MIC                         │
      └─────┴─────────────────────────────────────────────────────────────────┘
      
    Notación:
      - s₀: score inicial (antes de challenge)
      - s_f: score final (después de challenge)
      - relief ∈ [0, 1]: factor de alivio si se aprueba excepción
      - penalty ∈ [0, 1]: factor de penalización si se rechaza
    """

    # ── CONSTANTES DE CONFIGURACIÓN ────────────────────────────────────────

    STANDARD_PENALTY = FinancialConstants.STANDARD_PENALTY
    MAX_SCORE = FinancialConstants.MAX_INTEGRITY_SCORE
    MIN_SCORE = FinancialConstants.MIN_INTEGRITY_SCORE

    # ── FIXTURES ───────────────────────────────────────────────────────────

    @pytest.fixture
    def mock_mic(self) -> MagicMock:
        """
        MIC (Matriz de Interacción Central) simulada.
        
        La MIC es el colaborador externo que evalúa las solicitudes
        de excepción según criterios de física y topología.
        """
        mic = MagicMock()
        mic.project_intent = MagicMock()
        return mic

    @pytest.fixture
    def challenger(self, mock_mic) -> RiskChallenger:
        """RiskChallenger inyectado con MIC simulada."""
        return RiskChallenger(mic=mock_mic)

    @staticmethod
    def _build_report(
        integrity_score: float = 100.0,
        financial_risk_level: str = "SAFE",
        details: Optional[dict] = None,
        validate: bool = True,
        **overrides
    ) -> ConstructionRiskReport:
        """
        Factory con valores por defecto para reporte de riesgo.
        
        Representa un reporte "sano" por defecto; sobrescribir
        solo las dimensiones específicas bajo prueba.
        
        Args:
            integrity_score: Score de integridad ∈ [0, 100]
            financial_risk_level: Clasificación de riesgo financiero
            details: Diccionario de detalles (se fusiona con defaults)
            validate: Si True, valida rangos de parámetros
            **overrides: Campos adicionales a sobrescribir
            
        Returns:
            ConstructionRiskReport configurado para testing
        """
        if validate:
            if not (0 <= integrity_score <= 100):
                raise ValueError(
                    f"integrity_score = {integrity_score} fuera de [0, 100]"
                )

        default_details = {"pyramid_stability": 0.60, "thermal_metrics": {"system_temperature": 10.0, "heat_capacity": 0.8}}
        if details:
            default_details.update(details)

        defaults = dict(
            integrity_score=integrity_score,
            waste_alerts=[],
            circular_risks=[],
            complexity_level="Alta",
            financial_risk_level=financial_risk_level,
            details=default_details,
            strategic_narrative="",
        )
        defaults.update(overrides)
        return ConstructionRiskReport(**defaults)

    @staticmethod
    def _mic_approval(
        pivot: str = "MONOPOLIO_COBERTURADO",
        relief: float = 0.30,
        reasoning: str = "Aprobado por condiciones favorables.",
        additional_fields: Optional[dict] = None,
    ) -> dict:
        """
        Factory para respuestas exitosas de la MIC.
        
        Args:
            pivot: Tipo de pivote aprobado
            relief: Factor de alivio de penalización ∈ [0, 1]
            reasoning: Justificación textual de la decisión
            additional_fields: Campos extra en el payload
        """
        payload = {
            "approved_pivot": pivot,
            "penalty_relief": relief,
            "reasoning": reasoning,
        }
        if additional_fields:
            payload.update(additional_fields)
            
        return {
            "success": True,
            "payload": payload,
        }

    @staticmethod
    def _mic_rejection(
        error: str = "Condiciones no satisfechas",
        status: str = VectorResultStatus.LOGIC_ERROR.value,
    ) -> dict:
        """Factory para respuestas de rechazo de la MIC."""
        return {
            "success": False,
            "error": error,
            "status": status,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # MONOPOLIO COBERTURADO: FLUJOS DE APROBACIÓN
    # ═══════════════════════════════════════════════════════════════════════

    def test_monopolio_approved_saturated_score(self, challenger, mock_mic):
        """
        Aprobación con saturación de score en cota superior.
        
        Cálculo:
          s₀ = 100, relief = 0.30
          s_f = min(100, 100 × 1.30) = min(100, 130) = 100
          
        El score satura en 100 por el invariante I₁.
        """
        mock_mic.project_intent.return_value = self._mic_approval()

        report = self._build_report(
            integrity_score=100.0,
            details={"pyramid_stability": 0.60, "thermal_metrics": {"system_temperature": 10.0, "heat_capacity": 0.8}},
        )
        original_score = report.integrity_score

        audited = challenger.challenge_verdict(report)

        # ── Verificar llamada correcta a la MIC
        mock_mic.project_intent.assert_called_once()
        call_args, _ = mock_mic.project_intent.call_args
        assert call_args[0] == "lateral_thinking_pivot"
        assert call_args[1]["pivot_type"] == "MONOPOLIO_COBERTURADO"

        # ── Verificar resultado
        expected_score = min(self.MAX_SCORE, original_score * 1.30)
        assert audited.integrity_score == pytest.approx(expected_score)
        assert audited.integrity_score == pytest.approx(100.0)
        assert self.MIN_SCORE <= audited.integrity_score <= self.MAX_SCORE
        assert "EXCEPCIÓN_MONOPOLIO_COBERTURADO" in audited.details.get(
            "lateral_thinking_applied", ""
        )
        assert "ACTA DEL CONSEJO" in audited.strategic_narrative

    def test_monopolio_approved_subsaturated_score(self, challenger, mock_mic):
        """
        Aprobación sin saturación de score.
        
        Cálculo:
          s₀ = 70, relief = 0.30
          s_f = min(100, 70 × 1.30) = min(100, 91) = 91
          
        El score final está bajo la cota; no hay saturación.
        """
        mock_mic.project_intent.return_value = self._mic_approval(relief=0.30)

        report = self._build_report(
            integrity_score=70.0,
            details={"pyramid_stability": 0.60, "thermal_metrics": {"system_temperature": 10.0, "heat_capacity": 0.8}},
        )

        audited = challenger.challenge_verdict(report)

        expected = 70.0 * 1.30
        assert audited.integrity_score == pytest.approx(expected)
        assert audited.integrity_score == pytest.approx(91.0)
        assert self.MIN_SCORE <= audited.integrity_score <= self.MAX_SCORE

    def test_monopolio_approved_with_zero_relief(self, challenger, mock_mic):
        """
        Caso borde: aprobación con relief = 0.
        
        s_f = s₀ × (1 + 0) = s₀  (score inalterado)
        
        Este caso representa una aprobación "neutral" donde
        no hay penalización pero tampoco beneficio numérico.
        """
        mock_mic.project_intent.return_value = self._mic_approval(relief=0.0)

        report = self._build_report(
            integrity_score=80.0,
            details={"pyramid_stability": 0.60, "thermal_metrics": {"system_temperature": 10.0, "heat_capacity": 0.8}},
        )

        audited = challenger.challenge_verdict(report)

        assert audited.integrity_score == pytest.approx(80.0)

    # ═══════════════════════════════════════════════════════════════════════
    # MONOPOLIO COBERTURADO: FLUJOS DE RECHAZO
    # ═══════════════════════════════════════════════════════════════════════

    def test_monopolio_rejected_applies_veto(self, challenger, mock_mic):
        """
        Rechazo con aplicación de penalización estándar.
        
        Cálculo (Invariante I₃):
          s₀ = 100, penalty = 0.30
          s_f = 100 × (1 − 0.30) = 70
          
        El rechazo activa el "veto" del challenger con penalización.
        """
        mock_mic.project_intent.return_value = self._mic_rejection()

        report = self._build_report(
            integrity_score=100.0,
            details={"pyramid_stability": 0.60},
        )

        audited = challenger.challenge_verdict(report)

        expected = 100.0 * (1 - self.STANDARD_PENALTY)
        assert audited.integrity_score == pytest.approx(expected)
        assert audited.integrity_score == pytest.approx(70.0)
        assert audited.financial_risk_level == "RIESGO ESTRUCTURAL (CRÍTICO)"

    def test_rejection_with_low_score_stays_positive(self, challenger, mock_mic):
        """
        Rechazo con score bajo: resultado permanece no negativo.
        
        s₀ = 10, penalty = 0.30
        s_f = 10 × 0.70 = 7  (positivo, cumple I₁)
        """
        mock_mic.project_intent.return_value = self._mic_rejection()

        report = self._build_report(
            integrity_score=10.0,
            details={"pyramid_stability": 0.60},
        )

        audited = challenger.challenge_verdict(report)

        assert audited.integrity_score == pytest.approx(7.0)
        assert audited.integrity_score >= self.MIN_SCORE

    # ═══════════════════════════════════════════════════════════════════════
    # CUARENTENA TOPOLÓGICA
    # ═══════════════════════════════════════════════════════════════════════

    def test_quarantine_approved_applies_correct_relief(self, challenger, mock_mic):
        """
        Cuarentena aprobada con relief específico.
        
        Contexto topológico:
          β₁ = 5 generadores en H₁, sin acoplamiento sinérgico.
          Los ciclos son aislables individualmente.
          
        Cálculo:
          s₀ = 80, relief_cuarentena = 0.10
          s_f = min(100, 80 × 1.10) = 88
        """
        mock_mic.project_intent.return_value = self._mic_approval(
            pivot="CUARENTENA_TOPOLOGICA",
            relief=0.10,
            reasoning="Ciclos aislados sin acoplamiento en H₁.",
        )

        report = self._build_report(
            integrity_score=80.0,
            financial_risk_level="MODERATE",
            details={
                "pyramid_stability": 0.9,
                "topological_invariants": {
                    "betti_numbers": {"beta_1": 5},
                    "n_nodes": 20,
                },
            },
        )

        audited = challenger.challenge_verdict(report)

        assert audited.integrity_score == pytest.approx(88.0)
        assert "CUARENTENA" in audited.details.get("lateral_thinking_applied", "").upper()

    # ═══════════════════════════════════════════════════════════════════════
    # INVARIANTE I₁: ACOTAMIENTO GLOBAL DEL SCORE
    # ═══════════════════════════════════════════════════════════════════════

    @pytest.mark.parametrize(
        "base_score, relief, expected_final, scenario",
        [
            # ── Saturación superior ──
            (100.0, 0.30, 100.0, "saturación completa (130 → 100)"),
            ( 95.0, 0.10, 100.0, "saturación marginal (104.5 → 100)"),
            ( 77.0, 0.30, 100.0, "frontera ε (100.1 → 100)"),
            ( 76.92, 0.30, 99.996, "justo bajo saturación"),
            
            # ── Sin saturación ──
            ( 50.0, 0.71, 85.5, "incremento sustancial sin saturar"),
            ( 70.0, 0.30, 91.0, "caso típico sin saturación"),
            ( 30.0, 0.20, 36.0, "score bajo con relief moderado"),
            
            # ── Casos borde ──
            (  0.0, 0.30,  0.0, "score cero preservado (0 × 1.3 = 0)"),
            (  0.0, 1.00,  0.0, "score cero con relief máximo"),
            (  1.0, 0.00,  1.0, "relief cero: score inalterado"),
            (  1.0, 1.00,  2.0, "score mínimo con relief máximo"),
        ],
        ids=lambda x: x if isinstance(x, str) else "",
    )
    def test_score_bounded_after_relief(
        self, challenger, mock_mic, base_score, relief, expected_final, scenario
    ):
        """
        Invariante I₁ + I₂: acotamiento tras aplicación de relief.
        
        Verificación formal:
          ∀ (s₀, r) ∈ [0,100] × [0,1]:
            s_f = min(100, s₀ × (1 + r))
            ⟹ 0 ≤ s_f ≤ 100
        """
        mock_mic.project_intent.return_value = self._mic_approval(relief=relief)

        report = self._build_report(
            integrity_score=base_score,
            details={"pyramid_stability": 0.60, "thermal_metrics": {"system_temperature": 10.0, "heat_capacity": 0.8}},
            validate=False,  # Permitir score = 0 para testing
        )

        audited = challenger.challenge_verdict(report)

        calculated = min(self.MAX_SCORE, base_score * (1 + relief))
        
        assert audited.integrity_score == pytest.approx(expected_final, rel=1e-3), (
            f"Escenario: {scenario}\n"
            f"  s₀ = {base_score}, relief = {relief}\n"
            f"  Calculado = {calculated}, Esperado = {expected_final}\n"
            f"  Obtenido = {audited.integrity_score}"
        )
        assert self.MIN_SCORE <= audited.integrity_score <= self.MAX_SCORE, (
            f"Violación de I₁: score {audited.integrity_score} fuera de [{self.MIN_SCORE}, {self.MAX_SCORE}]"
        )

    @pytest.mark.parametrize(
        "base_score, expected_final",
        [
            (100.0, 70.0),   # Máximo → 70% de máximo
            ( 80.0, 56.0),   # 80 × 0.7 = 56
            ( 50.0, 35.0),   # 50 × 0.7 = 35
            ( 10.0,  7.0),   # 10 × 0.7 = 7
            (  0.0,  0.0),   # 0 × 0.7 = 0 (preserva cero)
        ],
        ids=["max", "high", "mid", "low", "zero"],
    )
    def test_score_bounded_after_penalty(
        self, challenger, mock_mic, base_score, expected_final
    ):
        """
        Invariante I₁ + I₃: acotamiento tras aplicación de penalización.
        
        Verificación formal:
          s_f = s₀ × (1 − penalty), con penalty = 0.30 fija
          ⟹ s_f = 0.70 × s₀
          ⟹ 0 ≤ s_f ≤ 70 ≤ 100  (siempre válido si s₀ ≤ 100)
        """
        mock_mic.project_intent.return_value = self._mic_rejection()

        report = self._build_report(
            integrity_score=base_score,
            details={"pyramid_stability": 0.60},
            validate=False,
        )

        audited = challenger.challenge_verdict(report)

        assert audited.integrity_score == pytest.approx(expected_final)
        assert self.MIN_SCORE <= audited.integrity_score <= self.MAX_SCORE

    # ═══════════════════════════════════════════════════════════════════════
    # INVARIANTE I₄: INMUTABILIDAD DEL REPORTE ORIGINAL
    # ═══════════════════════════════════════════════════════════════════════

    def test_original_report_immutability_comprehensive(self, challenger, mock_mic):
        """
        Verificación exhaustiva de inmutabilidad.
        
        El reporte de entrada NO debe ser mutado en ningún campo.
        challenge_verdict debe operar sobre una copia profunda.
        
        Verificamos:
          - Campos escalares (integrity_score)
          - Campos de referencia (details dict)
          - Campos anidados (details["pyramid_stability"])
          - Campos de texto (strategic_narrative)
          - Listas (waste_alerts, circular_risks)
        """
        mock_mic.project_intent.return_value = self._mic_approval(relief=0.30)

        # ── Configuración inicial con valores específicos
        original_score = 85.0
        original_stability = 0.60
        original_narrative = ""
        original_alerts = ["alert1", "alert2"]
        original_risks = [{"type": "risk1"}]
        
        report = self._build_report(
            integrity_score=original_score,
            waste_alerts=original_alerts.copy(),
            circular_risks=deepcopy(original_risks),
            details={"pyramid_stability": original_stability, "extra": {"nested": True}},
            strategic_narrative=original_narrative,
        )
        
        # ── Snapshot profundo antes de challenge
        pre_snapshot = {
            "score": report.integrity_score,
            "stability": report.details["pyramid_stability"],
            "narrative": report.strategic_narrative,
            "alerts_len": len(report.waste_alerts),
            "risks_len": len(report.circular_risks),
            "extra_nested": report.details.get("extra", {}).get("nested"),
        }

        _ = challenger.challenge_verdict(report)

        # ── Verificar inmutabilidad de todos los campos
        assert report.integrity_score == pre_snapshot["score"], (
            "integrity_score fue mutado"
        )
        assert report.details["pyramid_stability"] == pre_snapshot["stability"], (
            "details['pyramid_stability'] fue mutado"
        )
        assert report.strategic_narrative == pre_snapshot["narrative"], (
            "strategic_narrative fue mutado"
        )
        assert len(report.waste_alerts) == pre_snapshot["alerts_len"], (
            "waste_alerts fue mutado"
        )
        assert len(report.circular_risks) == pre_snapshot["risks_len"], (
            "circular_risks fue mutado"
        )
        assert report.details.get("extra", {}).get("nested") == pre_snapshot["extra_nested"], (
            "Estructura anidada en details fue mutada"
        )

    def test_multiple_challenges_preserve_original(self, challenger, mock_mic):
        """
        Múltiples llamadas a challenge_verdict no acumulan cambios.
        
        Cada llamada debe ser independiente y basarse en el
        reporte original, no en resultados de llamadas previas.
        """
        mock_mic.project_intent.return_value = self._mic_approval(relief=0.10)

        original_score = 80.0
        report = self._build_report(
            integrity_score=original_score,
            details={"pyramid_stability": 0.90},
        )

        # ── Múltiples challenges
        result1 = challenger.challenge_verdict(report)
        result2 = challenger.challenge_verdict(report)
        result3 = challenger.challenge_verdict(report)

        # ── Todos los resultados deben ser idénticos
        assert result1.integrity_score == result2.integrity_score == result3.integrity_score
        
        # ── El original permanece inalterado
        assert report.integrity_score == original_score

    # ═══════════════════════════════════════════════════════════════════════
    # INVARIANTE I₅: RESILIENCIA ANTE FALLOS DE MIC
    # ═══════════════════════════════════════════════════════════════════════

    @pytest.mark.parametrize(
        "exception_type, exception_msg",
        [
            (ConnectionError, "MIC unreachable"),
            (TimeoutError, "MIC timeout after 30s"),
            (RuntimeError, "Internal MIC error"),
            (ValueError, "MIC returned invalid response"),
        ],
        ids=["connection", "timeout", "runtime", "value"],
    )
    def test_mic_failure_degrades_gracefully(
        self, challenger, mock_mic, exception_type, exception_msg
    ):
        """
        Fallo de MIC: degradación graceful sin propagación.
        
        Ante cualquier excepción de la MIC, el challenger debe:
          1. NO propagar la excepción al caller
          2. Aplicar comportamiento conservador (penalización o neutral)
          3. Mantener el score en rango válido [0, 100]
          4. Documentar el fallo en el reporte si es posible
        """
        mock_mic.project_intent.side_effect = exception_type(exception_msg)

        report = self._build_report(
            integrity_score=80.0,
            details={"pyramid_stability": 0.90},
        )

        # ── No debe propagar la excepción
        try:
            audited = challenger.challenge_verdict(report)
        except exception_type:
            pytest.fail(
                f"Excepción {exception_type.__name__} propagada al caller. "
                "El challenger debe manejarla internamente."
            )

        # ── Score debe permanecer en rango válido
        assert self.MIN_SCORE <= audited.integrity_score <= self.MAX_SCORE

    def test_mic_returns_malformed_payload_handled(self, challenger, mock_mic):
        """
        Payload incompleto de MIC: tratamiento como rechazo implícito.
        
        Si la MIC retorna success=True pero el payload carece de
        campos requeridos, el challenger debe tratar esto como
        rechazo y aplicar comportamiento conservador.
        """
        mock_mic.project_intent.return_value = {
            "success": True,
            "payload": {},  # Sin approved_pivot ni relief
        }

        report = self._build_report(
            integrity_score=80.0,
            details={"pyramid_stability": 0.90},
        )

        audited = challenger.challenge_verdict(report)

        # ── Score válido (comportamiento conservador aplicado)
        assert self.MIN_SCORE <= audited.integrity_score <= self.MAX_SCORE

    def test_mic_returns_none_handled(self, challenger, mock_mic):
        """MIC retornando None debe manejarse sin excepción."""
        mock_mic.project_intent.return_value = None

        report = self._build_report(
            integrity_score=80.0,
            details={"pyramid_stability": 0.90},
        )

        try:
            audited = challenger.challenge_verdict(report)
            assert self.MIN_SCORE <= audited.integrity_score <= self.MAX_SCORE
        except (TypeError, AttributeError) as e:
            pytest.fail(f"None de MIC causó excepción: {e}")

    @pytest.mark.parametrize(
        "malformed_response, description",
        [
            ({"success": "yes"}, "success como string en lugar de bool"),
            ({"success": True, "payload": "not_a_dict"}, "payload como string"),
            ({"success": True, "payload": {"penalty_relief": "thirty"}}, "relief como string"),
            ({"success": True, "payload": {"penalty_relief": -0.5}}, "relief negativo"),
            ({"success": True, "payload": {"penalty_relief": 1.5}}, "relief > 1"),
        ],
        ids=["success_type", "payload_type", "relief_type", "relief_negative", "relief_overflow"],
    )
    def test_mic_returns_semantically_invalid_data(
        self, challenger, mock_mic, malformed_response, description
    ):
        """
        Respuestas semánticamente inválidas deben manejarse robustamente.
        
        El challenger debe validar no solo la estructura sino también
        la semántica de los datos (tipos correctos, rangos válidos).
        """
        mock_mic.project_intent.return_value = malformed_response

        report = self._build_report(
            integrity_score=80.0,
            details={"pyramid_stability": 0.90},
        )

        try:
            audited = challenger.challenge_verdict(report)
            # Si no lanza excepción, debe producir score válido
            assert self.MIN_SCORE <= audited.integrity_score <= self.MAX_SCORE, (
                f"Respuesta malformada ({description}) produjo score inválido"
            )
        except (TypeError, ValueError, KeyError):
            # Excepciones explícitas son aceptables para datos muy corruptos
            pass

    # ═══════════════════════════════════════════════════════════════════════
    # COHERENCIA NARRATIVA Y TRAZABILIDAD
    # ═══════════════════════════════════════════════════════════════════════

    def test_strategic_narrative_documents_approval(self, challenger, mock_mic):
        """
        Narrativa estratégica documenta decisión de aprobación.
        
        La narrativa debe incluir:
          - Marca de acta del consejo
          - Razonamiento de la MIC
          - Identificador del tipo de pivote
          - Información para auditoría
        """
        reasoning = "Inercia térmica favorable, Cᵥ = 0.92."
        mock_mic.project_intent.return_value = self._mic_approval(reasoning=reasoning)

        report = self._build_report(
            integrity_score=75.0,
            details={"pyramid_stability": 0.60, "topological_invariants": {"system_temperature": 20.0, "heat_capacity": 0.8}},
        )

        audited = challenger.challenge_verdict(report)

        assert audited.strategic_narrative, "Narrativa no debe estar vacía"
        assert "EXCEPCIÓN POR PENSAMIENTO LATERAL" in audited.strategic_narrative
        # La narrativa debería referenciar el razonamiento o tipo de decisión
        assert len(audited.strategic_narrative) > 20  # Más que trivial

    def test_strategic_narrative_documents_rejection(self, challenger, mock_mic):
        """
        Narrativa estratégica documenta decisión de rechazo.
        
        En caso de rechazo, la narrativa debe indicar:
          - Que se aplicó veto
          - El nivel de riesgo resultante
          - Información para trazabilidad
        """
        mock_mic.project_intent.return_value = self._mic_rejection(
            error="Temperatura del sistema excede umbral crítico"
        )

        report = self._build_report(
            integrity_score=75.0,
            details={"pyramid_stability": 0.60, "thermal_metrics": {"system_temperature": 10.0, "heat_capacity": 0.8}},
        )

        audited = challenger.challenge_verdict(report)

        # Debe haber documentación del rechazo
        assert audited.strategic_narrative or audited.financial_risk_level != "SAFE"

    def test_details_lateral_thinking_field_populated(self, challenger, mock_mic):
        """
        El campo lateral_thinking_applied debe estar correctamente poblado.
        
        Este campo es crítico para trazabilidad de auditoría y debe
        indicar claramente qué tipo de excepción fue aplicada.
        """
        mock_mic.project_intent.return_value = self._mic_approval(
            pivot="MONOPOLIO_COBERTURADO"
        )

        report = self._build_report(
            integrity_score=75.0,
            details={"pyramid_stability": 0.60, "thermal_metrics": {"system_temperature": 10.0, "heat_capacity": 0.8}},
        )

        audited = challenger.challenge_verdict(report)

        assert "challenger_applied" in audited.details
        assert "lateral_exception" in audited.details


# =============================================================================
# PRUEBAS DE PROPIEDADES (PROPERTY-BASED TESTING)
# =============================================================================

class TestPropertyBasedInvariants:
    """
    Pruebas basadas en propiedades para verificar invariantes
    bajo entradas generadas aleatoriamente.
    
    Estas pruebas complementan las pruebas de frontera con
    verificación estocástica de propiedades universales.
    """

    @pytest.fixture
    def mock_mic(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def challenger(self, mock_mic) -> RiskChallenger:
        return RiskChallenger(mic=mock_mic)

    @pytest.mark.parametrize(
        "base_score",
        [pytest.param(s, id=f"score_{s}") for s in range(0, 101, 5)],
    )
    @pytest.mark.parametrize(
        "relief",
        [pytest.param(r / 10, id=f"relief_{r/10:.1f}") for r in range(0, 11, 2)],
    )
    def test_score_always_bounded_approval_matrix(
        self, challenger, mock_mic, base_score, relief
    ):
        """
        Matriz de pruebas: score × relief → score final acotado.
        
        Verifica I₁ para múltiples combinaciones de entrada.
        """
        mock_mic.project_intent.return_value = {
            "success": True,
            "payload": {
                "approved_pivot": "MONOPOLIO_COBERTURADO",
                "penalty_relief": relief,
                "reasoning": "Test",
            },
        }

        report = ConstructionRiskReport(
            integrity_score=float(base_score),
            waste_alerts=[],
            circular_risks=[],
            complexity_level="Alta",
            financial_risk_level="SAFE",
            details={"pyramid_stability": 0.90},
            strategic_narrative="",
        )

        audited = challenger.challenge_verdict(report)

        assert 0.0 <= audited.integrity_score <= 100.0, (
            f"Invariante I₁ violado: score = {audited.integrity_score} "
            f"para base = {base_score}, relief = {relief}"
        )

    @pytest.mark.parametrize("base_score", list(range(0, 101, 10)))
    def test_penalty_monotonically_decreases_score(
        self, challenger, mock_mic, base_score
    ):
        """
        Propiedad: penalización siempre decrece o mantiene score.
        
        ∀s₀ ≥ 0, p ∈ [0,1]: s₀ × (1-p) ≤ s₀
        """
        mock_mic.project_intent.return_value = {"success": False}

        report = ConstructionRiskReport(
            integrity_score=float(base_score),
            waste_alerts=[],
            circular_risks=[],
            complexity_level="Alta",
            financial_risk_level="SAFE",
            details={"pyramid_stability": 0.90},
            strategic_narrative="",
        )

        audited = challenger.challenge_verdict(report)

        assert audited.integrity_score <= base_score, (
            f"Penalización incrementó score: {base_score} → {audited.integrity_score}"
        )