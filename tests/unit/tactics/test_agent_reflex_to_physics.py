"""
Suite rigurosa de pruebas para el reflejo Agente → Física (ciclo OODA).

Contrato probado
----------------
Observe  : TelemetryData
Orient   : TelemetryData -> SystemStatus
Decide   : SystemStatus -> AgentDecision | None
Act      : fuera del alcance de esta suite

Hipótesis del dominio
---------------------
- Umbral crítico de flyback normalizado: θ = 0.8
- Región nominal: V_flyback ≤ θ
- Región crítica: V_flyback > θ
- Implicación obligatoria: CRITICO → ALERTA_CRITICA
- Para esta partición de estados, orient() depende del flyback y no de saturation

Propiedades matemáticas verificadas
-----------------------------------
1. Partición topológica: {NOMINAL, CRITICO} forma cobertura disjunta de ℝ⁺
2. Función característica: orient = χ_{(θ,∞)} (función de Heaviside trasladada)
3. Independencia funcional: ∂(orient)/∂(saturation) = 0
4. Determinismo: orient y decide son funciones puras (sin efectos laterales)
5. Grafo de alcanzabilidad: todo estado es alcanzable desde cualquier estado
"""

from __future__ import annotations

import math
import itertools
import sys
from enum import Enum
from typing import Final, Callable, TypeVar, Generator
from dataclasses import dataclass

import pytest

from app.tactics.apu_agent import (
    AgentDecision,
    AutonomousAgent,
    SystemStatus,
    TelemetryData,
)


# =============================================================================
# CONSTANTES DEL DOMINIO FÍSICO
# =============================================================================

# Umbral crítico normalizado del voltaje de flyback (sin unidades, ratio)
_FLYBACK_CRITICAL_THRESHOLD: Final[float] = 0.8

# Valores representativos para pruebas de partición
_SAFE_FLYBACK: Final[float] = 0.1
_BOUNDARY_FLYBACK: Final[float] = _FLYBACK_CRITICAL_THRESHOLD
_CRISIS_FLYBACK: Final[float] = 9.5

# Vecinos ULP (Unit in the Last Place) del umbral para pruebas de frontera
_JUST_BELOW_THRESHOLD: Final[float] = math.nextafter(
    _FLYBACK_CRITICAL_THRESHOLD,
    -math.inf,
)
_JUST_ABOVE_THRESHOLD: Final[float] = math.nextafter(
    _FLYBACK_CRITICAL_THRESHOLD,
    math.inf,
)

# Valores de saturación para pruebas de independencia
_NORMAL_SATURATION: Final[float] = 0.3
_CRISIS_SATURATION: Final[float] = 0.4

# Muestreo del dominio de saturación [0, 1]
# Note: we exclude 1.0 because a saturation > 0.95 triggers a critical alert
# by the CriticalSaturationEvaluator safety net.
_SATURATION_SAMPLES: Final[tuple[float, ...]] = (0.0, 0.25, 0.5, 0.75, 0.9)

# Muestreo monótono del dominio de flyback para verificar función escalón
_MONOTONE_FLYBACK_SAMPLES: Final[tuple[float, ...]] = (
    0.0,
    _SAFE_FLYBACK,
    _JUST_BELOW_THRESHOLD,
    _BOUNDARY_FLYBACK,
    _JUST_ABOVE_THRESHOLD,
    1.0,
    _CRISIS_FLYBACK,
)

# Valores patológicos para pruebas de robustez numérica
_SUBNORMAL_POSITIVE: Final[float] = sys.float_info.min * sys.float_info.epsilon
_MAX_FLOAT: Final[float] = sys.float_info.max
_MIN_POSITIVE_NORMAL: Final[float] = sys.float_info.min


# =============================================================================
# FUNCIONES AUXILIARES Y CONSTRUCTORES
# =============================================================================

def make_telemetry(
    flyback_voltage: float,
    saturation: float = _NORMAL_SATURATION,
) -> TelemetryData:
    """
    Constructor explícito para TelemetryData.
    
    Proporciona un punto único de construcción para mejorar la trazabilidad
    y facilitar cambios futuros en la estructura de datos.
    
    Args:
        flyback_voltage: Voltaje de flyback normalizado (sin unidades).
        saturation: Nivel de saturación del núcleo magnético [0, 1].
    
    Returns:
        Instancia de TelemetryData con los valores especificados.
    """
    return TelemetryData(
        flyback_voltage=flyback_voltage,
        saturation=saturation,
    )


def heaviside_classification(
    value: float,
    threshold: float,
) -> SystemStatus:
    """
    Función de Heaviside trasladada para clasificación binaria.
    
    Implementa la función característica χ_{(θ,∞)}(x):
        H(x - θ) = NOMINAL  si x ≤ θ
                   CRITICO  si x > θ
    
    Esta es la especificación matemática contra la cual se verifica orient().
    
    Args:
        value: Valor a clasificar.
        threshold: Umbral de transición.
    
    Returns:
        SystemStatus correspondiente según la función de Heaviside.
    """
    return SystemStatus.CRITICO if value > threshold else SystemStatus.NOMINAL


def generate_flyback_test_points(
    num_samples: int = 20,
) -> Generator[float, None, None]:
    """
    Genera puntos de prueba distribuidos estratégicamente en el dominio.
    
    La distribución incluye:
    - Puntos uniformes en [0, 2*θ]
    - Puntos concentrados cerca del umbral (refinamiento local)
    - Valores extremos
    
    Args:
        num_samples: Número de muestras uniformes base.
    
    Yields:
        Valores de flyback_voltage para pruebas.
    """
    # Muestreo uniforme del dominio principal
    for i in range(num_samples + 1):
        yield (2.0 * _FLYBACK_CRITICAL_THRESHOLD * i) / num_samples
    
    # Refinamiento local cerca del umbral (±10 ULPs)
    current = _FLYBACK_CRITICAL_THRESHOLD
    for _ in range(10):
        current = math.nextafter(current, -math.inf)
        yield current
    
    current = _FLYBACK_CRITICAL_THRESHOLD
    for _ in range(10):
        current = math.nextafter(current, math.inf)
        yield current


def is_valid_float_for_domain(value: float) -> bool:
    """
    Verifica si un valor flotante es válido para el dominio físico.
    
    El dominio físico de voltaje de flyback normalizado es [0, ∞),
    excluyendo NaN e infinitos.
    
    Args:
        value: Valor a verificar.
    
    Returns:
        True si el valor es válido para el dominio.
    """
    return math.isfinite(value) and value >= 0.0


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def agent() -> AutonomousAgent:
    """Instancia fresca por prueba para evitar contaminación entre casos."""
    return AutonomousAgent()


@pytest.fixture
def normal_telemetry() -> TelemetryData:
    """Telemetría representativa de operación nominal."""
    return make_telemetry(_SAFE_FLYBACK, _NORMAL_SATURATION)


@pytest.fixture
def crisis_telemetry() -> TelemetryData:
    """Telemetría representativa de condición crítica."""
    return make_telemetry(_CRISIS_FLYBACK, _CRISIS_SATURATION)


@pytest.fixture
def boundary_telemetry() -> TelemetryData:
    """Telemetría exactamente en el umbral."""
    return make_telemetry(_BOUNDARY_FLYBACK, _NORMAL_SATURATION)


# =============================================================================
# PRUEBAS DE LA FASE ORIENT
# =============================================================================

class TestOrientPhase:
    """
    Pruebas de la fase Orient: TelemetryData → SystemStatus.
    
    Verifica que orient() implementa correctamente la función característica
    χ_{(θ,∞)} donde θ = 0.8 es el umbral crítico de flyback.
    """

    # -------------------------------------------------------------------------
    # Pruebas de clasificación básica
    # -------------------------------------------------------------------------
    
    def test_normal_flyback_returns_nominal(
        self,
        agent: AutonomousAgent,
        normal_telemetry: TelemetryData,
    ) -> None:
        """Verifica clasificación NOMINAL para flyback seguro."""
        status = agent.orient(normal_telemetry)
        assert status == SystemStatus.NOMINAL, (
            f"V_flyback={_SAFE_FLYBACK} debe producir NOMINAL, "
            f"obtenido: {status}"
        )

    def test_crisis_flyback_returns_critico(
        self,
        agent: AutonomousAgent,
        crisis_telemetry: TelemetryData,
    ) -> None:
        """Verifica clasificación CRITICO para flyback en crisis."""
        status = agent.orient(crisis_telemetry)
        assert status == SystemStatus.CRITICO, (
            f"V_flyback={_CRISIS_FLYBACK} debe producir CRITICO, "
            f"obtenido: {status}"
        )

    def test_zero_flyback_returns_nominal(self, agent: AutonomousAgent) -> None:
        """Verifica clasificación del origen del dominio."""
        status = agent.orient(make_telemetry(0.0, 0.0))
        assert status == SystemStatus.NOMINAL, "V_flyback=0.0 debe producir NOMINAL"

    # -------------------------------------------------------------------------
    # Pruebas de frontera (boundary testing)
    # -------------------------------------------------------------------------

    def test_exact_threshold_returns_nominal(
        self,
        agent: AutonomousAgent,
        boundary_telemetry: TelemetryData,
    ) -> None:
        """
        Verifica comportamiento en el umbral exacto.
        
        El contrato especifica región crítica como V > θ (estrictamente mayor),
        por lo tanto V = θ debe clasificarse como NOMINAL.
        """
        status = agent.orient(boundary_telemetry)
        assert status == SystemStatus.NOMINAL, (
            f"En la frontera exacta V_flyback={_BOUNDARY_FLYBACK}, "
            "el contrato exige NOMINAL porque la región crítica es V > θ."
        )

    def test_just_below_threshold_returns_nominal(
        self,
        agent: AutonomousAgent,
    ) -> None:
        """Verifica clasificación un ULP por debajo del umbral."""
        status = agent.orient(make_telemetry(_JUST_BELOW_THRESHOLD))
        assert status == SystemStatus.NOMINAL, (
            f"V_flyback={_JUST_BELOW_THRESHOLD:.17g} debe permanecer en NOMINAL"
        )

    def test_just_above_threshold_returns_critico(
        self,
        agent: AutonomousAgent,
    ) -> None:
        """Verifica clasificación un ULP por encima del umbral."""
        status = agent.orient(make_telemetry(_JUST_ABOVE_THRESHOLD))
        assert status == SystemStatus.CRITICO, (
            f"V_flyback={_JUST_ABOVE_THRESHOLD:.17g} debe caer en CRITICO"
        )

    def test_boundary_neighborhood_continuity(
        self,
        agent: AutonomousAgent,
    ) -> None:
        """
        Verifica que no hay estados intermedios en la vecindad del umbral.
        
        La función de Heaviside tiene una discontinuidad de salto en θ;
        verificamos que orient() reproduce este comportamiento sin estados
        espurios.
        """
        # Generar vecindad del umbral (±100 ULPs)
        below_values = []
        above_values = []
        
        current = _BOUNDARY_FLYBACK
        for _ in range(100):
            current = math.nextafter(current, -math.inf)
            below_values.append(current)
        
        current = _BOUNDARY_FLYBACK
        for _ in range(100):
            current = math.nextafter(current, math.inf)
            above_values.append(current)
        
        below_statuses = {agent.orient(make_telemetry(v)) for v in below_values}
        above_statuses = {agent.orient(make_telemetry(v)) for v in above_values}
        
        assert below_statuses == {SystemStatus.NOMINAL}, (
            f"Todos los valores por debajo del umbral deben ser NOMINAL: {below_statuses}"
        )
        assert above_statuses == {SystemStatus.CRITICO}, (
            f"Todos los valores por encima del umbral deben ser CRITICO: {above_statuses}"
        )

    # -------------------------------------------------------------------------
    # Pruebas de propiedades algebraicas
    # -------------------------------------------------------------------------

    def test_orient_is_deterministic(
        self,
        agent: AutonomousAgent,
        crisis_telemetry: TelemetryData,
    ) -> None:
        """Verifica determinismo: misma entrada → misma salida."""
        results = [agent.orient(crisis_telemetry) for _ in range(5)]
        assert len(set(results)) == 1, (
            f"Orient no es determinístico: {results}"
        )

    def test_orient_is_pure_function(
        self,
        agent: AutonomousAgent,
    ) -> None:
        """
        Verifica que orient() es una función pura (sin efectos laterales).
        
        Una lectura de crisis no debe contaminar lecturas normales posteriores,
        y viceversa. Esto verifica la ausencia de estado interno persistente.
        """
        sequence = [
            (_SAFE_FLYBACK, SystemStatus.NOMINAL),
            (_CRISIS_FLYBACK, SystemStatus.CRITICO),
            (_SAFE_FLYBACK, SystemStatus.NOMINAL),
            (_CRISIS_FLYBACK, SystemStatus.CRITICO),
            (_BOUNDARY_FLYBACK, SystemStatus.NOMINAL),
            (_JUST_ABOVE_THRESHOLD, SystemStatus.CRITICO),
            (0.0, SystemStatus.NOMINAL),
        ]
        
        for flyback, expected in sequence:
            status = agent.orient(make_telemetry(flyback))
            assert status == expected, (
                f"orient(V_flyback={flyback}) = {status}, esperado: {expected}. "
                "Posible contaminación por estado interno."
            )

    def test_orient_does_not_contaminate_subsequent_normal_reading(
        self,
        agent: AutonomousAgent,
        crisis_telemetry: TelemetryData,
        normal_telemetry: TelemetryData,
    ) -> None:
        """Verifica ausencia de histéresis no deseada."""
        status_crisis = agent.orient(crisis_telemetry)
        assert status_crisis == SystemStatus.CRITICO

        status_recovery = agent.orient(normal_telemetry)
        assert status_recovery == SystemStatus.NOMINAL, (
            "Una lectura normal posterior a una crisis debe retornar NOMINAL"
        )

    # -------------------------------------------------------------------------
    # Pruebas de independencia funcional
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("flyback", "expected_status"),
        [
            (_SAFE_FLYBACK, SystemStatus.NOMINAL),
            (_BOUNDARY_FLYBACK, SystemStatus.NOMINAL),
            (_JUST_ABOVE_THRESHOLD, SystemStatus.CRITICO),
            (_CRISIS_FLYBACK, SystemStatus.CRITICO),
        ],
    )
    def test_orient_is_invariant_under_saturation(
        self,
        agent: AutonomousAgent,
        flyback: float,
        expected_status: SystemStatus,
    ) -> None:
        """
        Verifica independencia funcional: ∂(orient)/∂(saturation) = 0.
        
        Para la partición {NOMINAL, CRITICO}, el estado depende únicamente
        del flyback, no de la saturación.
        """
        statuses = {
            agent.orient(make_telemetry(flyback, saturation))
            for saturation in _SATURATION_SAMPLES
        }
        assert statuses == {expected_status}, (
            f"Para V_flyback={flyback}, saturation no debe cambiar el estado. "
            f"Obtenido: {statuses}"
        )

    def test_orient_saturation_independence_exhaustive(
        self,
        agent: AutonomousAgent,
    ) -> None:
        """
        Verifica independencia de saturación sobre muestreo exhaustivo.
        
        Para cada punto de flyback en el muestreo, verifica que la saturación
        no afecta el resultado.
        """
        for flyback in generate_flyback_test_points(num_samples=10):
            if not is_valid_float_for_domain(flyback):
                continue
                
            expected = heaviside_classification(flyback, _FLYBACK_CRITICAL_THRESHOLD)
            statuses = {
                agent.orient(make_telemetry(flyback, sat))
                for sat in _SATURATION_SAMPLES
            }
            
            assert statuses == {expected}, (
                f"Independencia de saturación violada para V_flyback={flyback}: "
                f"obtenido {statuses}, esperado {{{expected}}}"
            )

    # -------------------------------------------------------------------------
    # Pruebas de monotonía y función escalón
    # -------------------------------------------------------------------------

    def test_orient_is_monotone_in_flyback(self, agent: AutonomousAgent) -> None:
        """
        Verifica monotonía no decreciente en la clasificación.
        
        Si V₁ < V₂ y orient(V₁) = CRITICO, entonces orient(V₂) = CRITICO.
        Esto es equivalente a verificar que la función es un escalón.
        """
        statuses = [
            agent.orient(make_telemetry(flyback))
            for flyback in _MONOTONE_FLYBACK_SAMPLES
        ]

        valid_statuses = {SystemStatus.NOMINAL, SystemStatus.CRITICO}
        assert set(statuses).issubset(valid_statuses), (
            f"orient() devolvió estados fuera de la partición esperada: {statuses}"
        )

        critical_flags = [status == SystemStatus.CRITICO for status in statuses]
        assert critical_flags == sorted(critical_flags), (
            "La clasificación no es monótona respecto de V_flyback: "
            f"{list(zip(_MONOTONE_FLYBACK_SAMPLES, statuses, strict=True))}"
        )

    def test_orient_implements_heaviside_function(
        self,
        agent: AutonomousAgent,
    ) -> None:
        """
        Verifica que orient() implementa exactamente la función de Heaviside
        trasladada χ_{(θ,∞)}.
        """
        for flyback in generate_flyback_test_points():
            if not is_valid_float_for_domain(flyback):
                continue
                
            expected = heaviside_classification(flyback, _FLYBACK_CRITICAL_THRESHOLD)
            actual = agent.orient(make_telemetry(flyback))
            
            assert actual == expected, (
                f"orient(V_flyback={flyback:.17g}) = {actual}, "
                f"esperado por Heaviside: {expected}"
            )

    def test_orient_transition_point_uniqueness(
        self,
        agent: AutonomousAgent,
    ) -> None:
        """
        Verifica que existe exactamente un punto de transición.
        
        La función de Heaviside tiene una única discontinuidad; verificamos
        que no hay transiciones espurias en el dominio muestreado.
        """
        samples = sorted(generate_flyback_test_points(num_samples=50))
        statuses = [agent.orient(make_telemetry(v)) for v in samples]
        
        # Contar transiciones NOMINAL → CRITICO
        transitions = sum(
            1 for i in range(len(statuses) - 1)
            if statuses[i] == SystemStatus.NOMINAL 
            and statuses[i + 1] == SystemStatus.CRITICO
        )
        
        assert transitions == 1, (
            f"Se esperaba exactamente 1 transición, encontradas: {transitions}. "
            "Posible presencia de transiciones espurias."
        )
        
        # Verificar que no hay transiciones inversas (CRITICO → NOMINAL)
        reverse_transitions = sum(
            1 for i in range(len(statuses) - 1)
            if statuses[i] == SystemStatus.CRITICO 
            and statuses[i + 1] == SystemStatus.NOMINAL
        )
        
        assert reverse_transitions == 0, (
            f"Transiciones inversas detectadas: {reverse_transitions}. "
            "Violación de monotonía."
        )

    # -------------------------------------------------------------------------
    # Tabla de clasificación exhaustiva
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("flyback", "expected_status"),
        [
            (0.0, SystemStatus.NOMINAL),
            (_SAFE_FLYBACK, SystemStatus.NOMINAL),
            (_JUST_BELOW_THRESHOLD, SystemStatus.NOMINAL),
            (_BOUNDARY_FLYBACK, SystemStatus.NOMINAL),
            (_JUST_ABOVE_THRESHOLD, SystemStatus.CRITICO),
            (1.0, SystemStatus.CRITICO),
            (_CRISIS_FLYBACK, SystemStatus.CRITICO),
        ],
    )
    def test_orient_classification_table(
        self,
        agent: AutonomousAgent,
        flyback: float,
        expected_status: SystemStatus,
    ) -> None:
        """Verificación tabular de clasificaciones conocidas."""
        status = agent.orient(make_telemetry(flyback))
        assert status == expected_status, (
            f"orient(V_flyback={flyback:.17g}) = {status}, "
            f"esperado: {expected_status}"
        )


# =============================================================================
# PRUEBAS DE ROBUSTEZ NUMÉRICA
# =============================================================================

class TestNumericalRobustness:
    """
    Pruebas de robustez numérica para valores extremos y patológicos.
    
    Verifica el comportamiento correcto con:
    - Números subnormales
    - Valores cercanos a los límites de representación
    - Valores especiales IEEE 754
    """

    def test_subnormal_positive_classified_as_nominal(
        self,
        agent: AutonomousAgent,
    ) -> None:
        """Verifica clasificación de números subnormales positivos."""
        status = agent.orient(make_telemetry(_SUBNORMAL_POSITIVE))
        assert status == SystemStatus.NOMINAL, (
            f"Número subnormal {_SUBNORMAL_POSITIVE} debe ser NOMINAL"
        )

    def test_min_positive_normal_classified_as_nominal(
        self,
        agent: AutonomousAgent,
    ) -> None:
        """Verifica clasificación del mínimo positivo normal."""
        status = agent.orient(make_telemetry(_MIN_POSITIVE_NORMAL))
        assert status == SystemStatus.NOMINAL, (
            f"Mínimo positivo normal {_MIN_POSITIVE_NORMAL} debe ser NOMINAL"
        )

    def test_large_flyback_classified_as_critico(
        self,
        agent: AutonomousAgent,
    ) -> None:
        """Verifica clasificación de valores grandes pero finitos."""
        large_value = 1e10
        status = agent.orient(make_telemetry(large_value))
        assert status == SystemStatus.CRITICO, (
            f"Valor grande {large_value} debe ser CRITICO"
        )

    @pytest.mark.parametrize(
        "saturation",
        [0.0, _SUBNORMAL_POSITIVE, 0.5, 0.95],
    )
    def test_boundary_saturation_values(
        self,
        agent: AutonomousAgent,
        saturation: float,
    ) -> None:
        """Verifica que valores extremos de saturación no afectan orient()."""
        status_nominal = agent.orient(make_telemetry(_SAFE_FLYBACK, saturation))
        status_critico = agent.orient(make_telemetry(_CRISIS_FLYBACK, saturation))
        
        assert status_nominal == SystemStatus.NOMINAL
        assert status_critico == SystemStatus.CRITICO

    def test_floating_point_comparison_consistency(
        self,
        agent: AutonomousAgent,
    ) -> None:
        """
        Verifica consistencia en comparaciones de punto flotante.
        
        Asegura que no hay problemas con comparaciones que deberían ser
        determinísticas pero podrían variar por errores de redondeo.
        """
        # El umbral exacto debe comportarse consistentemente
        results = [
            agent.orient(make_telemetry(_BOUNDARY_FLYBACK))
            for _ in range(100)
        ]
        assert len(set(results)) == 1, (
            f"Resultados inconsistentes para umbral exacto: {set(results)}"
        )


# =============================================================================
# PRUEBAS DE LA FASE DECIDE
# =============================================================================

class TestDecidePhase:
    """
    Pruebas de la fase Decide: SystemStatus → AgentDecision | None.
    
    Verifica la implicación obligatoria: CRITICO → ALERTA_CRITICA.
    """

    def test_critico_produces_alerta_critica(self, agent: AutonomousAgent) -> None:
        """Verifica implicación CRITICO → ALERTA_CRITICA."""
        decision = agent.decide(SystemStatus.CRITICO)
        assert decision == AgentDecision.ALERTA_CRITICA, (
            f"CRITICO debe producir ALERTA_CRITICA, obtenido: {decision}"
        )

    def test_nominal_does_not_produce_alerta_critica(
        self,
        agent: AutonomousAgent,
    ) -> None:
        """Verifica que NOMINAL no produce alerta crítica."""
        decision = agent.decide(SystemStatus.NOMINAL)
        assert decision != AgentDecision.ALERTA_CRITICA, (
            "NOMINAL no debe producir ALERTA_CRITICA"
        )

    @pytest.mark.parametrize("status", tuple(SystemStatus))
    def test_decide_is_deterministic_for_each_status(
        self,
        agent: AutonomousAgent,
        status: SystemStatus,
    ) -> None:
        """Verifica determinismo de decide() para cada estado."""
        results = [agent.decide(status) for _ in range(5)]
        assert len(set(results)) == 1, (
            f"Decide no es determinístico para {status}: {results}"
        )

    @pytest.mark.parametrize("status", tuple(SystemStatus))
    def test_decide_returns_agent_decision_or_none(
        self,
        agent: AutonomousAgent,
        status: SystemStatus,
    ) -> None:
        """Verifica el tipo de retorno de decide()."""
        decision = agent.decide(status)
        assert decision is None or isinstance(decision, AgentDecision), (
            f"decide({status}) debe retornar AgentDecision o None, "
            f"obtenido: {type(decision)}"
        )

    def test_decide_is_pure_function(self, agent: AutonomousAgent) -> None:
        """Verifica que decide() no tiene efectos laterales."""
        sequence = [
            SystemStatus.NOMINAL,
            SystemStatus.CRITICO,
            SystemStatus.NOMINAL,
            SystemStatus.CRITICO,
        ]
        
        expected_decisions = [
            agent.decide(s) for s in sequence
        ]
        
        # Repetir la secuencia y verificar consistencia
        for status, expected in zip(sequence, expected_decisions, strict=True):
            actual = agent.decide(status)
            assert actual == expected, (
                f"decide({status}) inconsistente: {actual} vs {expected}"
            )

    def test_decide_idempotence(self, agent: AutonomousAgent) -> None:
        """
        Verifica idempotencia: llamar decide() múltiples veces con el mismo
        estado produce siempre el mismo resultado.
        """
        for status in SystemStatus:
            first_call = agent.decide(status)
            for _ in range(10):
                subsequent_call = agent.decide(status)
                assert subsequent_call == first_call, (
                    f"decide({status}) no es idempotente"
                )


# =============================================================================
# PRUEBAS DEL PIPELINE OODA
# =============================================================================

class TestOODAReflexLoop:
    """
    Pruebas del pipeline Orient → Decide.
    
    Verifica el comportamiento compuesto del ciclo de reflexión.
    """

    @pytest.mark.parametrize(
        ("flyback", "expected_status", "expect_critical_alert"),
        [
            (0.0, SystemStatus.NOMINAL, False),
            (_SAFE_FLYBACK, SystemStatus.NOMINAL, False),
            (_JUST_BELOW_THRESHOLD, SystemStatus.NOMINAL, False),
            (_BOUNDARY_FLYBACK, SystemStatus.NOMINAL, False),
            (_JUST_ABOVE_THRESHOLD, SystemStatus.CRITICO, True),
            (_CRISIS_FLYBACK, SystemStatus.CRITICO, True),
        ],
    )
    def test_pipeline_matches_domain_contract(
        self,
        agent: AutonomousAgent,
        flyback: float,
        expected_status: SystemStatus,
        expect_critical_alert: bool,
    ) -> None:
        """Verifica que el pipeline completo cumple el contrato del dominio."""
        saturation = (
            _CRISIS_SATURATION if flyback > _BOUNDARY_FLYBACK else _NORMAL_SATURATION
        )
        telemetry = make_telemetry(flyback, saturation)

        status = agent.orient(telemetry)
        decision = agent.decide(status)

        assert status == expected_status, (
            f"orient(V_flyback={flyback:.17g}) produjo {status}, "
            f"esperado: {expected_status}"
        )

        if expect_critical_alert:
            assert decision == AgentDecision.ALERTA_CRITICA, (
                f"Pipeline crítico debe emitir ALERTA_CRITICA, obtenido: {decision}"
            )
        else:
            assert decision != AgentDecision.ALERTA_CRITICA, (
                f"Pipeline nominal no debe emitir ALERTA_CRITICA, obtenido: {decision}"
            )

    def test_pipeline_normal_then_crisis_then_recovery(
        self,
        agent: AutonomousAgent,
        normal_telemetry: TelemetryData,
        crisis_telemetry: TelemetryData,
    ) -> None:
        """Verifica transiciones normal → crisis → recuperación."""
        # Fase normal
        status_1 = agent.orient(normal_telemetry)
        decision_1 = agent.decide(status_1)
        assert status_1 == SystemStatus.NOMINAL
        assert decision_1 != AgentDecision.ALERTA_CRITICA

        # Fase de crisis
        status_2 = agent.orient(crisis_telemetry)
        decision_2 = agent.decide(status_2)
        assert status_2 == SystemStatus.CRITICO
        assert decision_2 == AgentDecision.ALERTA_CRITICA

        # Recuperación
        status_3 = agent.orient(normal_telemetry)
        decision_3 = agent.decide(status_3)
        assert status_3 == SystemStatus.NOMINAL
        assert decision_3 != AgentDecision.ALERTA_CRITICA

    def test_pipeline_composition_is_associative(
        self,
        agent: AutonomousAgent,
    ) -> None:
        """
        Verifica que la composición del pipeline es asociativa.
        
        (orient ∘ id)(telemetry) seguido de decide debe producir el mismo
        resultado que orient seguido de (id ∘ decide).
        """
        test_cases = [
            make_telemetry(_SAFE_FLYBACK),
            make_telemetry(_CRISIS_FLYBACK),
            make_telemetry(_BOUNDARY_FLYBACK),
        ]
        
        for telemetry in test_cases:
            # Método 1: orient → decide
            status = agent.orient(telemetry)
            decision_1 = agent.decide(status)
            
            # Método 2: mismo proceso (verificando consistencia)
            status_2 = agent.orient(telemetry)
            decision_2 = agent.decide(status_2)
            
            assert decision_1 == decision_2, (
                f"Composición no consistente para {telemetry}"
            )

    def test_pipeline_exhaustive_state_coverage(
        self,
        agent: AutonomousAgent,
    ) -> None:
        """
        Verifica que el pipeline puede alcanzar todos los estados posibles.
        
        Esto verifica la propiedad de alcanzabilidad del grafo de estados.
        """
        reached_statuses: set[SystemStatus] = set()
        reached_decisions: set[AgentDecision | None] = set()
        
        for flyback in generate_flyback_test_points():
            if not is_valid_float_for_domain(flyback):
                continue
                
            telemetry = make_telemetry(flyback)
            status = agent.orient(telemetry)
            decision = agent.decide(status)
            
            reached_statuses.add(status)
            reached_decisions.add(decision)
        
        # Verificar cobertura de estados
        assert SystemStatus.NOMINAL in reached_statuses, (
            "Estado NOMINAL no alcanzado"
        )
        assert SystemStatus.CRITICO in reached_statuses, (
            "Estado CRITICO no alcanzado"
        )
        
        # Verificar cobertura de decisiones relevantes
        assert AgentDecision.ALERTA_CRITICA in reached_decisions, (
            "Decisión ALERTA_CRITICA no alcanzada"
        )


# =============================================================================
# PRUEBAS DE GRAFO DE TRANSICIONES
# =============================================================================

class TestStateTransitionGraph:
    """
    Pruebas del grafo de transiciones de estados.
    
    Verifica propiedades del autómata implícito definido por el ciclo OODA.
    """

    def test_all_states_reachable_from_nominal(
        self,
        agent: AutonomousAgent,
    ) -> None:
        """Verifica alcanzabilidad desde NOMINAL."""
        # Desde NOMINAL, verificar que podemos llegar a CRITICO
        nominal_telem = make_telemetry(_SAFE_FLYBACK)
        crisis_telem = make_telemetry(_CRISIS_FLYBACK)
        
        status_nominal = agent.orient(nominal_telem)
        assert status_nominal == SystemStatus.NOMINAL
        
        status_critico = agent.orient(crisis_telem)
        assert status_critico == SystemStatus.CRITICO

    def test_all_states_reachable_from_critico(
        self,
        agent: AutonomousAgent,
    ) -> None:
        """Verifica alcanzabilidad desde CRITICO."""
        # Desde CRITICO, verificar que podemos llegar a NOMINAL
        crisis_telem = make_telemetry(_CRISIS_FLYBACK)
        nominal_telem = make_telemetry(_SAFE_FLYBACK)
        
        status_critico = agent.orient(crisis_telem)
        assert status_critico == SystemStatus.CRITICO
        
        status_nominal = agent.orient(nominal_telem)
        assert status_nominal == SystemStatus.NOMINAL

    def test_state_graph_is_strongly_connected(
        self,
        agent: AutonomousAgent,
    ) -> None:
        """
        Verifica que el grafo de estados es fuertemente conexo.
        
        Todo par de estados (s₁, s₂) tiene un camino s₁ → s₂.
        """
        # Se requiere conectividad estricta de transición por telemetría.
        states = [SystemStatus.NOMINAL, SystemStatus.CRITICO]
        telemetry_for_state = {
            SystemStatus.NOMINAL: make_telemetry(_SAFE_FLYBACK),
            SystemStatus.CRITICO: make_telemetry(_CRISIS_FLYBACK),
        }
        
        for source in states:
            for target in states:
                # Podemos llegar a cualquier estado con una sola transición
                # (el sistema es sin memoria)
                target_telem = telemetry_for_state[target]
                reached = agent.orient(target_telem)
                assert reached == target, (
                    f"No se puede alcanzar {target} desde {source}"
                )

    def test_transition_determinism(
        self,
        agent: AutonomousAgent,
    ) -> None:
        """
        Verifica que las transiciones son determinísticas.
        
        Para un estado actual y una entrada dada, la transición es única.
        """
        test_telemetries = [
            make_telemetry(_SAFE_FLYBACK),
            make_telemetry(_CRISIS_FLYBACK),
            make_telemetry(_BOUNDARY_FLYBACK),
        ]
        
        for telemetry in test_telemetries:
            results = [agent.orient(telemetry) for _ in range(10)]
            assert len(set(results)) == 1, (
                f"Transición no determinística para {telemetry}"
            )


# =============================================================================
# INVARIANTES ALGEBRAICOS
# =============================================================================

class TestSystemStatusInvariants:
    """Invariantes algebraicos básicos del espacio de estados."""

    def test_nominal_and_critico_are_distinct(self) -> None:
        """Verifica distinción de estados (partición disjunta)."""
        assert SystemStatus.NOMINAL != SystemStatus.CRITICO

    def test_nominal_value_is_expected(self) -> None:
        """Verifica valor semántico de NOMINAL."""
        assert SystemStatus.NOMINAL.value == 1

    def test_critico_value_is_expected(self) -> None:
        """Verifica valor semántico de CRITICO."""
        assert SystemStatus.CRITICO.value == 5

    def test_system_status_is_enum(self) -> None:
        """Verifica que SystemStatus es una enumeración."""
        assert issubclass(SystemStatus, Enum)

    def test_agent_decision_is_enum(self) -> None:
        """Verifica que AgentDecision es una enumeración."""
        assert issubclass(AgentDecision, Enum)

    def test_alerta_critica_value_is_expected(self) -> None:
        """Verifica valor semántico de ALERTA_CRITICA."""
        assert AgentDecision.ALERTA_CRITICA.value == 4

    def test_system_status_exhaustive(self) -> None:
        """Verifica que conocemos todos los estados del sistema."""
        known_states = {
            SystemStatus.NOMINAL,
            SystemStatus.UNKNOWN,
            SystemStatus.INESTABLE,
            SystemStatus.SATURADO,
            SystemStatus.CRITICO,
            SystemStatus.DISCONNECTED,
        }
        actual_states = set(SystemStatus)
        
        # Advertir si hay estados desconocidos (no falla, pero informa)
        unknown = actual_states - known_states
        assert not unknown, (
            f"Estados no probados detectados: {unknown}. "
            "Actualizar la suite de pruebas."
        )

    def test_partition_is_exhaustive_and_disjoint(self) -> None:
        """
        Verifica que {NOMINAL, CRITICO} forma una partición válida.
        
        Una partición de un conjunto X es una colección de subconjuntos
        no vacíos, mutuamente disjuntos, cuya unión es X.
        """
        states = list(SystemStatus)
        
        # No vacíos
        assert len(states) >= 2, "Debe haber al menos 2 estados"
        
        # Mutuamente disjuntos (por ser Enum, esto es automático)
        assert len(states) == len(set(states)), "Estados duplicados detectados"
        
        # Para esta suite, verificamos la partición conocida
        assert SystemStatus.NOMINAL in states
        assert SystemStatus.CRITICO in states


# =============================================================================
# CONTRATOS DE TELEMETRY DATA
# =============================================================================

class TestTelemetryDataContracts:
    """Pruebas estructurales para TelemetryData."""

    def test_flyback_voltage_stored_correctly(self) -> None:
        """Verifica almacenamiento correcto de flyback_voltage."""
        td = make_telemetry(0.42, 0.3)
        assert td.flyback_voltage == pytest.approx(0.42)

    def test_saturation_stored_correctly(self) -> None:
        """Verifica almacenamiento correcto de saturation."""
        td = make_telemetry(0.1, 0.75)
        assert td.saturation == pytest.approx(0.75)

    def test_zero_fields_valid(self) -> None:
        """Verifica validez de campos en cero."""
        td = make_telemetry(0.0, 0.0)
        assert td.flyback_voltage == pytest.approx(0.0)
        assert td.saturation == pytest.approx(0.0)

    def test_crisis_fields_valid(self) -> None:
        """Verifica validez de campos en crisis con límite físico."""
        td = make_telemetry(_CRISIS_FLYBACK, _CRISIS_SATURATION)
        assert td.flyback_voltage == pytest.approx(1.0)
        assert td.saturation == pytest.approx(_CRISIS_SATURATION)
        assert td.was_clamped is True

    def test_boundary_neighbor_values_are_preserved(self) -> None:
        """Verifica preservación de orden en valores frontera."""
        td_below = make_telemetry(_JUST_BELOW_THRESHOLD)
        td_exact = make_telemetry(_BOUNDARY_FLYBACK)
        td_above = make_telemetry(_JUST_ABOVE_THRESHOLD)

        assert td_below.flyback_voltage < td_exact.flyback_voltage < td_above.flyback_voltage

    def test_telemetry_equality_reflexive(self) -> None:
        """Verifica reflexividad de igualdad."""
        td = make_telemetry(0.5, 0.5)
        assert td == td

    def test_telemetry_equality_symmetric(self) -> None:
        """Verifica simetría de igualdad."""
        td1 = make_telemetry(0.5, 0.5)
        td2 = make_telemetry(0.5, 0.5)
        assert td1 == td2
        assert td2 == td1

    def test_telemetry_inequality(self) -> None:
        """Verifica desigualdad para valores distintos."""
        td1 = make_telemetry(0.5, 0.5)
        td2 = make_telemetry(0.6, 0.5)
        td3 = make_telemetry(0.5, 0.6)
        
        assert td1 != td2
        assert td1 != td3
        assert td2 != td3


# =============================================================================
# PRUEBAS DE PROPIEDADES BASADAS EN HIPÓTESIS (PROPERTY-BASED TESTING)
# =============================================================================

class TestPropertyBasedOrient:
    """
    Pruebas basadas en propiedades para orient().
    
    Estas pruebas verifican invariantes que deben cumplirse para cualquier
    entrada válida del dominio.
    """

    @pytest.mark.parametrize(
        "flyback",
        [i * 0.1 for i in range(20)] + [_CRISIS_FLYBACK],
    )
    def test_orient_output_is_valid_status(
        self,
        agent: AutonomousAgent,
        flyback: float,
    ) -> None:
        """Propiedad: orient siempre retorna un SystemStatus válido."""
        status = agent.orient(make_telemetry(flyback))
        assert isinstance(status, SystemStatus)
        assert status in set(SystemStatus)

    @pytest.mark.parametrize(
        "flyback",
        [i * 0.1 for i in range(20)] + [_CRISIS_FLYBACK],
    )
    def test_orient_respects_threshold_contract(
        self,
        agent: AutonomousAgent,
        flyback: float,
    ) -> None:
        """
        Propiedad: orient respeta el contrato del umbral.
        
        ∀v ∈ ℝ⁺: v ≤ θ ⟹ orient(v) = NOMINAL
        ∀v ∈ ℝ⁺: v > θ ⟹ orient(v) = CRITICO
        """
        status = agent.orient(make_telemetry(flyback))
        
        if flyback <= _FLYBACK_CRITICAL_THRESHOLD:
            assert status == SystemStatus.NOMINAL, (
                f"V_flyback={flyback} ≤ θ={_FLYBACK_CRITICAL_THRESHOLD} "
                f"debe producir NOMINAL, obtenido: {status}"
            )
        else:
            assert status == SystemStatus.CRITICO, (
                f"V_flyback={flyback} > θ={_FLYBACK_CRITICAL_THRESHOLD} "
                f"debe producir CRITICO, obtenido: {status}"
            )

    @pytest.mark.parametrize(
        ("flyback1", "flyback2"),
        [
            (0.1, 0.2),
            (0.5, 0.7),
            (0.79, 0.8),
            (0.8, 0.81),
            (0.9, 1.5),
        ],
    )
    def test_orient_monotonicity_property(
        self,
        agent: AutonomousAgent,
        flyback1: float,
        flyback2: float,
    ) -> None:
        """
        Propiedad de monotonía: v₁ < v₂ ⟹ orient(v₁) ≤ orient(v₂).
        
        Donde el orden en SystemStatus es NOMINAL < CRITICO.
        """
        assert flyback1 < flyback2, "Precondición: flyback1 < flyback2"
        
        status1 = agent.orient(make_telemetry(flyback1))
        status2 = agent.orient(make_telemetry(flyback2))
        
        # Si status1 es CRITICO, status2 debe ser CRITICO (monotonía)
        if status1 == SystemStatus.CRITICO:
            assert status2 == SystemStatus.CRITICO, (
                f"Violación de monotonía: orient({flyback1})=CRITICO "
                f"pero orient({flyback2})={status2}"
            )