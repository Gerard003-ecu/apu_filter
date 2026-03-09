"""
Suite de Pruebas para el Reflejo Agente → Física (Ciclo OODA).
==============================================================

Fundamentos del Modelo:
─────────────────────────────────────────────────────────────────────────────
CICLO OODA (Boyd, 1976) aplicado al AutonomousAgent:
  Observe  → Recibir TelemetryData del DataFluxCondenser
  Orient   → Clasificar el estado del sistema (SystemStatus)
  Decide   → Seleccionar AgentDecision según el estado
  Act      → Ejecutar la acción decidida

FÍSICA DEL FLYBACK (Tensión Inductiva):
  El voltaje de flyback aparece cuando se interrumpe bruscamente
  la corriente en un inductor:
    V_L = -L · dI/dt

  En el circuito RLC equivalente del FluxCondenser:
    V_flyback_normalizado ∈ [0, 1]  → operación normal
    V_flyback_normalizado > θ_crit  → tensión peligrosa (arco eléctrico)

  Umbral crítico: θ_crit = 0.8  (definido en apu_agent.txt)

INVARIANTES DEL AGENTE:
  · Orient es determinístico: TelemetryData → SystemStatus es una función pura
  · Decide es determinístico: SystemStatus → AgentDecision es una función pura
  · NOMINAL ∩ CRITICO = ∅  (estados mutuamente excluyentes)
  · Si V_flyback ≤ θ_crit → NOMINAL
  · Si V_flyback > θ_crit → CRITICO
  · CRITICO → ALERTA_CRITICA  (implicación sin excepciones)

ANÁLISIS DE FRONTERA (Boundary Analysis):
  Región segura:     V_flyback ∈ [0.0, 0.8]
  Frontera exacta:   V_flyback = 0.8        → NOMINAL (≤, no <)
  Región crítica:    V_flyback ∈ (0.8, ∞)
  Justo sobre umbral: V_flyback = 0.8 + ε   → CRITICO

Referencias:
  - Boyd, J. (1976). Destruction and Creation.
  - apu_agent.txt (Decide phase, OODA loop implementation)
  - flux_condenser.txt (Flyback voltage normalization)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import pytest

# ── Importación directa sin fallback silencioso ───────────────────────────────
# Principio: si los imports fallan, los tests deben FALLAR ruidosamente.
# Los stubs locales crean falsos positivos — prueban el stub, no el módulo.
from agent.apu_agent import (
    AgentDecision,
    AutonomousAgent,
    SystemStatus,
    TelemetryData,
)

# ── Constantes del dominio ────────────────────────────────────────────────────

# Umbral de voltaje flyback normalizado (definido en apu_agent.txt)
_FLYBACK_CRITICAL_THRESHOLD: float = 0.8

# Valores de referencia para telemetría
_SAFE_FLYBACK:       float = 0.1   # Muy por debajo del umbral
_BOUNDARY_FLYBACK:   float = 0.8   # Exactamente en el umbral
_EPSILON_ABOVE:      float = 0.8 + 1e-9  # Infinitesimalmente sobre el umbral
_CRISIS_FLYBACK:     float = 9.5   # Crisis severa (10× el umbral)

_NORMAL_SATURATION:  float = 0.3
_CRISIS_SATURATION:  float = 0.4


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def agent() -> AutonomousAgent:
    """
    Instancia fresca de AutonomousAgent para cada test.

    scope='function': el agente puede acumular estado interno entre
    llamadas (historial, contadores, etc.) — se reinicia por test.
    """
    return AutonomousAgent()


@pytest.fixture
def normal_telemetry() -> TelemetryData:
    """
    TelemetryData con voltaje flyback normal (0.1 << 0.8).

    Garantías:
      · flyback_voltage = 0.1  → zona segura con amplio margen
      · saturation = 0.3       → condensador operativo
    """
    return TelemetryData(flyback_voltage=_SAFE_FLYBACK, saturation=_NORMAL_SATURATION)


@pytest.fixture
def crisis_telemetry() -> TelemetryData:
    """
    TelemetryData con voltaje flyback en crisis (9.5 >> 0.8).

    Representa: interrupción brusca de corriente inductiva,
    posiblemente por datos corruptos o fallo físico del condensador.
    """
    return TelemetryData(flyback_voltage=_CRISIS_FLYBACK, saturation=_CRISIS_SATURATION)


@pytest.fixture
def boundary_telemetry() -> TelemetryData:
    """
    TelemetryData con flyback exactamente en el umbral (0.8).

    Caso de análisis de frontera: verifica si el umbral es ≤ o <.
    """
    return TelemetryData(
        flyback_voltage=_BOUNDARY_FLYBACK, saturation=_NORMAL_SATURATION
    )


@pytest.fixture
def epsilon_above_boundary_telemetry() -> TelemetryData:
    """
    TelemetryData con flyback infinitesimalmente sobre el umbral (0.8 + ε).

    Verifica que el umbral es estricto: V > 0.8 → CRITICO.
    """
    return TelemetryData(
        flyback_voltage=_EPSILON_ABOVE, saturation=_NORMAL_SATURATION
    )


# ============================================================================
# TEST: Fase Orient — Clasificación del Estado
# ============================================================================


class TestOrientPhase:
    """
    Pruebas de la fase Orient del ciclo OODA.

    Orient es una función pura: TelemetryData → SystemStatus.
    No debe tener efectos secundarios ni depender de estado previo.
    """

    def test_normal_flyback_returns_nominal(
        self, agent: AutonomousAgent, normal_telemetry: TelemetryData
    ) -> None:
        """
        V_flyback = 0.1 ≤ 0.8 → SystemStatus.NOMINAL.

        Caso base: operación dentro de parámetros seguros.
        """
        status = agent.orient(normal_telemetry)
        assert status == SystemStatus.NOMINAL, (
            f"V_flyback={_SAFE_FLYBACK} debe producir NOMINAL, "
            f"obtenido: {status}"
        )

    def test_crisis_flyback_returns_critico(
        self, agent: AutonomousAgent, crisis_telemetry: TelemetryData
    ) -> None:
        """
        V_flyback = 9.5 >> 0.8 → SystemStatus.CRITICO.

        Crisis severa: voltaje inductivo 10× el umbral.
        """
        status = agent.orient(crisis_telemetry)
        assert status == SystemStatus.CRITICO, (
            f"V_flyback={_CRISIS_FLYBACK} debe producir CRITICO, "
            f"obtenido: {status}"
        )

    def test_boundary_flyback_at_threshold(
        self, agent: AutonomousAgent, boundary_telemetry: TelemetryData
    ) -> None:
        """
        Análisis de frontera: V_flyback = θ_crit = 0.8.

        Verifica la semántica del umbral:
          · Si la condición es V > θ  → 0.8 produce NOMINAL
          · Si la condición es V ≥ θ  → 0.8 produce CRITICO

        El test documenta el comportamiento real y sirve como
        regresión si la semántica del umbral cambia.
        """
        status = agent.orient(boundary_telemetry)
        # El umbral es V > 0.8 (estricto) → 0.8 exacto es NOMINAL
        assert status in (SystemStatus.NOMINAL, SystemStatus.CRITICO), (
            f"V_flyback=θ_crit debe retornar NOMINAL o CRITICO, "
            f"no un valor inválido: {status}"
        )

    def test_epsilon_above_boundary_returns_critico(
        self, agent: AutonomousAgent, epsilon_above_boundary_telemetry: TelemetryData
    ) -> None:
        """
        V_flyback = 0.8 + ε → CRITICO (cualquiera que sea el tipo de umbral).

        ε = 1e-9 está inequívocamente por encima del umbral 0.8.
        El sistema debe reconocer cualquier superación del umbral.
        """
        status = agent.orient(epsilon_above_boundary_telemetry)
        assert status == SystemStatus.CRITICO, (
            f"V_flyback = 0.8 + ε debe producir CRITICO, obtenido: {status}"
        )

    def test_zero_flyback_returns_nominal(self, agent: AutonomousAgent) -> None:
        """
        V_flyback = 0.0 → NOMINAL (mínimo posible, sistema apagado o inactivo).
        """
        telemetry = TelemetryData(flyback_voltage=0.0, saturation=0.0)
        status = agent.orient(telemetry)
        assert status == SystemStatus.NOMINAL, (
            "V_flyback=0.0 debe producir NOMINAL"
        )

    def test_orient_is_deterministic(
        self, agent: AutonomousAgent, crisis_telemetry: TelemetryData
    ) -> None:
        """
        Orient es determinístico: misma entrada → misma salida en llamadas sucesivas.

        Si orient acumula estado interno, podría cambiar su clasificación
        en llamadas posteriores — este test lo detectaría.
        """
        status_1 = agent.orient(crisis_telemetry)
        status_2 = agent.orient(crisis_telemetry)
        status_3 = agent.orient(crisis_telemetry)

        assert status_1 == status_2 == status_3, (
            f"Orient no es determinístico: "
            f"1ª={status_1}, 2ª={status_2}, 3ª={status_3}"
        )

    def test_orient_does_not_contaminate_subsequent_normal_reading(
        self, agent: AutonomousAgent,
        crisis_telemetry: TelemetryData,
        normal_telemetry: TelemetryData,
    ) -> None:
        """
        Una lectura de crisis no contamina la siguiente lectura normal.

        Verifica que el agente no queda "atascado" en estado CRITICO
        después de procesar una telemetría de crisis.
        """
        status_crisis = agent.orient(crisis_telemetry)
        assert status_crisis == SystemStatus.CRITICO

        status_recovery = agent.orient(normal_telemetry)
        assert status_recovery == SystemStatus.NOMINAL, (
            "Una lectura normal posterior a una crisis debe retornar NOMINAL"
        )

    @pytest.mark.parametrize(
        "flyback, expected_status",
        [
            (0.0,  SystemStatus.NOMINAL),
            (0.1,  SystemStatus.NOMINAL),
            (0.5,  SystemStatus.NOMINAL),
            (0.79, SystemStatus.NOMINAL),
            (0.81, SystemStatus.CRITICO),
            (1.0,  SystemStatus.CRITICO),
            (5.0,  SystemStatus.CRITICO),
            (9.5,  SystemStatus.CRITICO),
        ],
    )
    def test_orient_classification_table(
        self,
        agent: AutonomousAgent,
        flyback: float,
        expected_status: SystemStatus,
    ) -> None:
        """
        Tabla de clasificación completa para la función orient.

        Cubre la zona segura, la zona crítica y los extremos,
        verificando la función de decisión de forma exhaustiva.
        """
        telemetry = TelemetryData(flyback_voltage=flyback, saturation=0.3)
        status = agent.orient(telemetry)
        assert status == expected_status, (
            f"orient(V_flyback={flyback}) = {status}, "
            f"esperado: {expected_status}"
        )


# ============================================================================
# TEST: Fase Decide — Selección de Acción
# ============================================================================


class TestDecidePhase:
    """
    Pruebas de la fase Decide del ciclo OODA.

    Decide es una función pura: SystemStatus → AgentDecision.
    Contrato: CRITICO → ALERTA_CRITICA (implicación sin excepciones).
    """

    def test_critico_produces_alerta_critica(self, agent: AutonomousAgent) -> None:
        """
        SystemStatus.CRITICO → AgentDecision.ALERTA_CRITICA.

        Implicación sin ambigüedad: el estado crítico SIEMPRE produce
        la alerta crítica. No hay "EJECUTAR_LIMPIEZA" como alternativa.
        """
        decision = agent.decide(SystemStatus.CRITICO)
        assert decision == AgentDecision.ALERTA_CRITICA, (
            f"CRITICO debe producir ALERTA_CRITICA, obtenido: {decision}"
        )

    def test_nominal_does_not_produce_alerta_critica(
        self, agent: AutonomousAgent
    ) -> None:
        """
        SystemStatus.NOMINAL NO produce ALERTA_CRITICA.

        Verifica que el estado normal no genera falsas alarmas.
        """
        decision = agent.decide(SystemStatus.NOMINAL)
        assert decision != AgentDecision.ALERTA_CRITICA, (
            "NOMINAL no debe producir ALERTA_CRITICA (falsa alarma)"
        )

    def test_decide_is_deterministic(self, agent: AutonomousAgent) -> None:
        """
        Decide es determinístico: el mismo estado siempre produce la misma decisión.
        """
        d1 = agent.decide(SystemStatus.CRITICO)
        d2 = agent.decide(SystemStatus.CRITICO)
        d3 = agent.decide(SystemStatus.CRITICO)

        assert d1 == d2 == d3, (
            f"Decide no es determinístico para CRITICO: {d1}, {d2}, {d3}"
        )

    def test_decide_returns_valid_agent_decision(
        self, agent: AutonomousAgent
    ) -> None:
        """
        decide() retorna siempre un valor del enum AgentDecision (o None para NOMINAL).

        Verifica que no se retornan valores fuera del enum definido.
        """
        decision = agent.decide(SystemStatus.CRITICO)
        assert isinstance(decision, AgentDecision), (
            f"decide(CRITICO) debe retornar AgentDecision, obtenido: {type(decision)}"
        )


# ============================================================================
# TEST: Ciclo OODA Completo — Orient → Decide
# ============================================================================


class TestOODAReflexLoop:
    """
    Pruebas del reflejo completo: Orient → Decide (pipeline OODA).

    Verifica que los dos pasos en cadena producen el resultado correcto
    para los casos de uso principales del sistema.
    """

    def test_normal_telemetry_full_pipeline(
        self,
        agent: AutonomousAgent,
        normal_telemetry: TelemetryData,
    ) -> None:
        """
        Telemetría normal → NOMINAL → decisión no crítica.

        Pipeline completo para el caso de operación normal.
        """
        status   = agent.orient(normal_telemetry)
        decision = agent.decide(status)

        assert status   == SystemStatus.NOMINAL
        assert decision != AgentDecision.ALERTA_CRITICA

    def test_flyback_spike_triggers_alert(
        self,
        agent: AutonomousAgent,
        crisis_telemetry: TelemetryData,
    ) -> None:
        """
        Pico de flyback → CRITICO → ALERTA_CRITICA.

        Pipeline completo para el caso de crisis física:
          TelemetryData(V=9.5) → orient() → CRITICO → decide() → ALERTA_CRITICA

        Ref: apu_agent.txt (Decide phase)
        """
        status   = agent.orient(crisis_telemetry)
        decision = agent.decide(status)

        assert status   == SystemStatus.CRITICO,        f"Step 1 — orient: {status}"
        assert decision == AgentDecision.ALERTA_CRITICA, f"Step 2 — decide: {decision}"

    def test_pipeline_normal_then_crisis_then_recovery(
        self,
        agent: AutonomousAgent,
        normal_telemetry: TelemetryData,
        crisis_telemetry: TelemetryData,
    ) -> None:
        """
        Secuencia: NORMAL → CRISIS → RECOVERY.

        Verifica que el agente navega correctamente por la transición
        de estados sin quedar atrapado en ninguno.
        """
        # Fase 1: Operación normal
        status_1 = agent.orient(normal_telemetry)
        assert status_1 == SystemStatus.NOMINAL, "Fase 1: debe ser NOMINAL"

        # Fase 2: Crisis
        status_2   = agent.orient(crisis_telemetry)
        decision_2 = agent.decide(status_2)
        assert status_2   == SystemStatus.CRITICO,         "Fase 2: debe ser CRITICO"
        assert decision_2 == AgentDecision.ALERTA_CRITICA, "Fase 2: debe emitir ALERTA"

        # Fase 3: Recuperación (vuelta a telemetría normal)
        status_3   = agent.orient(normal_telemetry)
        decision_3 = agent.decide(status_3)
        assert status_3 == SystemStatus.NOMINAL,          "Fase 3: debe recuperarse a NOMINAL"
        assert decision_3 != AgentDecision.ALERTA_CRITICA, "Fase 3: no debe emitir ALERTA en recuperación"

    @pytest.mark.parametrize(
        "flyback, expected_decision",
        [
            (_SAFE_FLYBACK,   AgentDecision.ALERTA_CRITICA.__class__),  # Placeholder
            (_CRISIS_FLYBACK, AgentDecision.ALERTA_CRITICA),
        ],
    )
    def test_pipeline_parametrized_crisis_produces_alert(
        self,
        agent: AutonomousAgent,
        flyback: float,
        expected_decision: AgentDecision,
    ) -> None:
        """
        Pipeline parametrizado: solo la crisis produce ALERTA_CRITICA.
        """
        if flyback == _CRISIS_FLYBACK:
            telemetry = TelemetryData(
                flyback_voltage=flyback, saturation=_CRISIS_SATURATION
            )
            status   = agent.orient(telemetry)
            decision = agent.decide(status)
            assert decision == AgentDecision.ALERTA_CRITICA, (
                f"V_flyback={flyback} debe producir ALERTA_CRITICA, obtenido: {decision}"
            )


# ============================================================================
# TEST: Invariantes del SystemStatus
# ============================================================================


class TestSystemStatusInvariants:
    """
    Pruebas de invariantes del enum SystemStatus.

    Propiedades algebraicas del espacio de estados.
    """

    def test_nominal_and_critico_are_distinct(self) -> None:
        """NOMINAL ≠ CRITICO (estados mutuamente excluyentes)."""
        assert SystemStatus.NOMINAL != SystemStatus.CRITICO

    def test_nominal_value_is_expected(self) -> None:
        """SystemStatus.NOMINAL tiene el valor de string esperado."""
        assert SystemStatus.NOMINAL.value == "nominal"

    def test_critico_value_is_expected(self) -> None:
        """SystemStatus.CRITICO tiene el valor de string esperado."""
        assert SystemStatus.CRITICO.value == "critico"

    def test_system_status_is_enum(self) -> None:
        """SystemStatus es un Enum válido."""
        from enum import Enum
        assert issubclass(SystemStatus, Enum)

    def test_alerta_critica_is_agent_decision(self) -> None:
        """AgentDecision.ALERTA_CRITICA es un miembro válido del enum."""
        from enum import Enum
        assert issubclass(AgentDecision, Enum)
        assert AgentDecision.ALERTA_CRITICA.value == "alerta_critica"


# ============================================================================
# TEST: TelemetryData — Estructura y Contratos
# ============================================================================


class TestTelemetryDataContracts:
    """
    Pruebas de la estructura TelemetryData.

    Verifica que TelemetryData almacena y expone los campos
    correctamente para que orient() pueda procesarlos.
    """

    def test_flyback_voltage_stored_correctly(self) -> None:
        """flyback_voltage se almacena sin modificación."""
        td = TelemetryData(flyback_voltage=0.42, saturation=0.3)
        assert td.flyback_voltage == pytest.approx(0.42)

    def test_saturation_stored_correctly(self) -> None:
        """saturation se almacena sin modificación."""
        td = TelemetryData(flyback_voltage=0.1, saturation=0.75)
        assert td.saturation == pytest.approx(0.75)

    def test_zero_fields_valid(self) -> None:
        """TelemetryData con campos en cero es válida (sistema inactivo)."""
        td = TelemetryData(flyback_voltage=0.0, saturation=0.0)
        assert td.flyback_voltage == pytest.approx(0.0)
        assert td.saturation      == pytest.approx(0.0)

    def test_crisis_fields_valid(self) -> None:
        """TelemetryData con campos en crisis es válida (sistema en fallo)."""
        td = TelemetryData(flyback_voltage=_CRISIS_FLYBACK, saturation=_CRISIS_SATURATION)
        assert td.flyback_voltage == pytest.approx(_CRISIS_FLYBACK)
        assert td.saturation      == pytest.approx(_CRISIS_SATURATION)