"""
Suite rigurosa de pruebas para el reflejo Agente → Física (ciclo OODA).
═══════════════════════════════════════════════════════════════════════

Fundamentos Matemáticos
────────────────────────

Topología del Espacio de Estados
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sea Ω = ℝ⁺ × [0,1] el espacio de telemetría (V, S) donde:
  • V ∈ ℝ⁺ : voltaje de flyback normalizado
  • S ∈ [0,1] : saturación del inductor

Partición Topológica
~~~~~~~~~~~~~~~~~~~~
Definimos la partición disjunta exhaustiva:
  Ω = Ω_N ⊔ Ω_C  donde:
  • Ω_N = {(V,S) ∈ Ω : V ≤ θ}  (región nominal)
  • Ω_C = {(V,S) ∈ Ω : V > θ}   (región crítica)
  • θ = 0.8 (umbral crítico)

Álgebra de Boole
~~~~~~~~~~~~~~~~
El predicado de criticidad es la función característica:
  χ_C : Ω → 𝔹
  χ_C(V,S) = 𝟙_{(θ,∞)}(V) = { 1  si V > θ
                              { 0  si V ≤ θ

Función de Heaviside Trasladada
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
La función orient implementa H_θ : ℝ⁺ → {NOMINAL, CRITICO}:
  H_θ(V) = { NOMINAL   si V ≤ θ
           { CRITICO   si V > θ

Propiedades:
  1. H_θ es función escalón (discontinuidad de salto en θ)
  2. ∂H_θ/∂S = 0 (independencia funcional de saturación)
  3. H_θ es monótona no-decreciente
  4. H_θ es idempotente bajo composición con decide

Teoría de Grafos
~~~~~~~~~~~~~~~~
El autómata OODA define un grafo dirigido G = (V, E):
  • V = {NOMINAL, CRITICO} (vértices/estados)
  • E = V × V (grafo completo K₂)
  • Transiciones determinadas por χ_C

Propiedades del grafo:
  1. Fuertemente conexo (∀u,v ∈ V : ∃ camino u → v)
  2. Diámetro d(G) = 1 (alcanzabilidad en un paso)
  3. No posee ciclos con histéresis (función pura)

Álgebra Lineal
~~~~~~~~~~~~~~
La proyección π₁ : Ω → ℝ⁺ donde π₁(V,S) = V satisface:
  orient = H_θ ∘ π₁
  
El núcleo de la derivada parcial ker(∂/∂S) contiene a orient,
demostrando independencia funcional.

Contratos Verificados
────────────────────────
  • **Observe** : ℝ × [0,1] → TelemetryData
                  con clamping: V ↦ min(V, 1.0)
  • **Orient**  : TelemetryData → SystemStatus
                  implementa H_θ con θ = 0.8
  • **Decide**  : SystemStatus → AgentDecision ⊔ {⊥}
                  implicación: CRITICO → ALERTA_CRITICA

Invariantes de Dominio
──────────────────────
  • Umbral crítico: θ = 0.8 (constante física)
  • Cota superior: V_max = 1.0 (post-clamping)
  • Región nominal: [0, θ] ⊂ ℝ⁺
  • Región crítica: (θ, ∞) ∩ [0, 1]

Teoremas Verificados
────────────────────

**Teorema A (Partición Completa)**:
  ∀(V,S) ∈ Ω : χ_C(V,S) ⊕ χ_N(V,S) = 1
  donde χ_N = 1 - χ_C (complemento booleano)

**Teorema B (Heaviside y Contracción)**:
  ∀V ∈ ℝ⁺ : orient(make_telemetry(V)) = H_θ(min(V, 1.0))

**Teorema C (Independencia Funcional)**:
  ∀V ∈ ℝ⁺, ∀S₁,S₂ ∈ [0,1] :
    orient(V, S₁) = orient(V, S₂)

**Teorema D (Monotonía)**:
  ∀V₁,V₂ ∈ ℝ⁺ : V₁ ≤ V₂ ⟹ ord(orient(V₁)) ≤ ord(orient(V₂))
  donde ord(NOMINAL) = 0, ord(CRITICO) = 1

**Teorema E (Transitividad del Pipeline)**:
  (decide ∘ orient)(make_telemetry(V)) = {
    ALERTA_CRITICA  si V > θ
    ⊥ ∨ ¬ALERTA     si V ≤ θ
  }

**Teorema F (Alcanzabilidad Global)**:
  ∀s₁,s₂ ∈ SystemStatus : ∃V ∈ ℝ⁺ :
    orient(make_telemetry(V)) = s₁ ∧
    ∃V' ∈ ℝ⁺ : orient(make_telemetry(V')) = s₂

Cobertura de Pruebas
────────────────────
  ✓ Frontera del umbral (análisis ULP)
  ✓ Robustez IEEE 754 (subnormales, extremos)
  ✓ Independencia de saturación (producto cartesiano)
  ✓ Monotonía (ordenamiento total)
  ✓ Pureza funcional (ausencia de estado)
  ✓ Idempotencia del pipeline
  ✓ Alcanzabilidad del grafo de estados
  ✓ Clamping y contratos de datos
"""

from __future__ import annotations

import itertools
import math
import sys
from dataclasses import dataclass
from enum import Enum
from typing import (
    Callable,
    Final,
    Generator,
    Iterable,
    Optional,
    TypeVar,
    Union,
)

import pytest

from app.tactics.apu_agent import (
    AgentDecision,
    AutonomousAgent,
    SystemStatus,
    TelemetryData,
)


# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTES DEL DOMINIO FÍSICO
# ═════════════════════════════════════════════════════════════════════════════

# Parámetros topológicos del espacio de estados
# ──────────────────────────────────────────────

#: Umbral crítico θ ∈ ℝ⁺ para la función de Heaviside H_θ
THETA: Final[float] = 0.8

#: Cota superior del dominio post-clamping
V_MAX: Final[float] = 1.0

#: Epsilon de máquina para comparaciones numéricas
MACHINE_EPSILON: Final[float] = sys.float_info.epsilon

#: Valor subnormal positivo mínimo (testing de robustez)
SUBNORMAL_MIN: Final[float] = sys.float_info.min * MACHINE_EPSILON

#: Valor normal positivo mínimo
NORMAL_MIN: Final[float] = sys.float_info.min


# Puntos de prueba representativos
# ─────────────────────────────────

#: Valor nominal seguro: V << θ
V_SAFE: Final[float] = 0.1

#: Valor exactamente en el umbral: V = θ
V_BOUNDARY: Final[float] = THETA

#: Valor crítico que requiere clamping: V >> θ
V_CRISIS: Final[float] = 9.5

#: Vecindad inferior del umbral (un ULP por debajo)
V_THETA_MINUS_ULP: Final[float] = math.nextafter(THETA, -math.inf)

#: Vecindad superior del umbral (un ULP por encima)
V_THETA_PLUS_ULP: Final[float] = math.nextafter(THETA, math.inf)


# Valores de saturación para testing de independencia funcional
# ──────────────────────────────────────────────────────────────

#: Saturación nominal
S_NOMINAL: Final[float] = 0.3

#: Saturación en condición crítica
S_CRITICAL: Final[float] = 0.4

#: Muestreo del dominio [0,1] para verificar ∂orient/∂S = 0
SATURATION_DOMAIN_SAMPLE: Final[tuple[float, ...]] = (
    0.0,
    0.25,
    0.5,
    0.75,
    0.9,
)


# Muestreo monótono para verificación de H_θ
# ───────────────────────────────────────────

#: Secuencia estrictamente creciente que atraviesa θ
FLYBACK_MONOTONE_SEQUENCE: Final[tuple[float, ...]] = (
    0.0,
    V_SAFE,
    V_THETA_MINUS_ULP,
    V_BOUNDARY,
    V_THETA_PLUS_ULP,
    V_MAX,
    V_CRISIS,  # será clampado a V_MAX
)


# ═════════════════════════════════════════════════════════════════════════════
# UTILIDADES Y CONSTRUCTORES
# ═════════════════════════════════════════════════════════════════════════════


def make_telemetry(
    flyback_voltage: float,
    saturation: float = S_NOMINAL,
) -> TelemetryData:
    """
    Constructor canónico para TelemetryData.

    Proporciona un punto único de construcción que:
      • Centraliza la creación para trazabilidad
      • Facilita refactorizaciones futuras
      • Permite inyección de dependencias en tests

    Parámetros
    ──────────
    flyback_voltage : float
        Voltaje de flyback normalizado V ∈ ℝ⁺
    saturation : float, opcional
        Saturación del inductor S ∈ [0,1] (por defecto: S_NOMINAL)

    Retorna
    ───────
    TelemetryData
        Instancia con clamping aplicado si V > V_MAX

    Notas
    ─────
    El clamping V ↦ min(V, V_MAX) se realiza en el constructor
    de TelemetryData, manteniendo la separación de responsabilidades.
    """
    return TelemetryData(
        flyback_voltage=flyback_voltage,
        saturation=saturation,
    )


def heaviside_reference(value: float, threshold: float) -> SystemStatus:
    """
    Implementación de referencia de la función de Heaviside H_θ.

    Define la especificación matemática contra la cual se verifica
    la implementación de `orient()`.

    Parámetros
    ──────────
    value : float
        Valor a clasificar (típicamente voltaje de flyback)
    threshold : float
        Umbral θ de la función escalón

    Retorna
    ───────
    SystemStatus
        CRITICO si value > threshold, NOMINAL en caso contrario

    Propiedades Matemáticas
    ────────────────────────
    • Función escalón con salto en θ
    • Discontinuidad de primera especie
    • Límites laterales:
        lim_{x→θ⁻} H_θ(x) = NOMINAL
        lim_{x→θ⁺} H_θ(x) = CRITICO
    """
    return SystemStatus.CRITICO if value > threshold else SystemStatus.NOMINAL


def is_valid_domain_point(voltage: float, saturation: float) -> bool:
    """
    Verifica que (V, S) ∈ Ω = ℝ⁺ × [0,1].

    Parámetros
    ──────────
    voltage : float
        Componente de voltaje
    saturation : float
        Componente de saturación

    Retorna
    ───────
    bool
        True si el punto pertenece al dominio válido
    """
    return (
        math.isfinite(voltage)
        and voltage >= 0.0
        and math.isfinite(saturation)
        and 0.0 <= saturation <= 1.0
    )


def generate_ulp_neighborhood(
    center: float, radius: int = 10, direction: int = 0
) -> Generator[float, None, None]:
    """
    Genera vecindad de ULPs (Units in the Last Place) alrededor de un punto.

    Parámetros
    ──────────
    center : float
        Punto central
    radius : int, opcional
        Número de ULPs a explorar (por defecto: 10)
    direction : int, opcional
        Dirección: -1 (inferior), +1 (superior), 0 (ambas)

    Yields
    ──────
    float
        Valores en la vecindad ULP del centro

    Notas
    ─────
    Útil para probar comportamiento en fronteras de representación IEEE 754.
    """
    if direction <= 0:
        value = center
        for _ in range(radius):
            value = math.nextafter(value, -math.inf)
            yield value

    if direction >= 0:
        value = center
        for _ in range(radius):
            value = math.nextafter(value, math.inf)
            yield value


def generate_flyback_test_domain(
    num_uniform_samples: int = 20,
    num_ulp_samples: int = 10,
) -> Generator[float, None, None]:
    """
    Genera conjunto de prueba exhaustivo del dominio de flyback.

    La estrategia de muestreo combina:
      1. Muestreo uniforme en [0, 2θ] para cobertura global
      2. Refinamiento ULP denso alrededor de θ para análisis de frontera
      3. Valores extremos (subnormales, límites representables)

    Parámetros
    ──────────
    num_uniform_samples : int, opcional
        Número de muestras uniformes (por defecto: 20)
    num_ulp_samples : int, opcional
        Número de ULPs a cada lado de θ (por defecto: 10)

    Yields
    ──────
    float
        Valores de prueba en el dominio extendido [0, ∞)

    Propiedades del Muestreo
    ────────────────────────
    • Densidad variable (adaptativa cerca de θ)
    • Cobertura de ambas regiones topológicas (Ω_N, Ω_C)
    • Inclusión de valores patológicos para robustez
    """
    # Muestreo uniforme del dominio principal
    for i in range(num_uniform_samples + 1):
        yield (2.0 * THETA * i) / num_uniform_samples

    # Refinamiento ULP en la frontera
    yield from generate_ulp_neighborhood(THETA, num_ulp_samples, direction=0)

    # Valores extremos para robustez numérica
    yield SUBNORMAL_MIN
    yield NORMAL_MIN
    yield V_MAX
    yield V_CRISIS


def compute_transition_matrix(
    agent: AutonomousAgent,
    states: Iterable[SystemStatus],
    inputs: Iterable[float],
) -> dict[tuple[SystemStatus, SystemStatus], int]:
    """
    Calcula la matriz de transición del autómata de estados.

    Parámetros
    ──────────
    agent : AutonomousAgent
        Instancia del agente
    states : Iterable[SystemStatus]
        Conjunto de estados posibles
    inputs : Iterable[float]
        Conjunto de entradas (voltajes de flyback)

    Retorna
    ───────
    dict[tuple[SystemStatus, SystemStatus], int]
        Matriz de adyacencia del grafo de transiciones
        (src, dst) → número de transiciones observadas

    Notas
    ─────
    Para un sistema sin estado (función pura), la matriz
    debe ser invariante respecto al estado previo.
    """
    transitions: dict[tuple[SystemStatus, SystemStatus], int] = {}

    for src_state in states:
        for voltage in inputs:
            dst_state = agent.orient(make_telemetry(voltage))
            key = (src_state, dst_state)
            transitions[key] = transitions.get(key, 0) + 1

    return transitions


# ═════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def agent() -> AutonomousAgent:
    """
    Proporciona instancia aislada del agente autónomo.

    Cada test recibe una instancia fresca, garantizando:
      • Aislamiento entre pruebas
      • Ausencia de efectos secundarios
      • Reproducibilidad determinística
    """
    return AutonomousAgent()


@pytest.fixture
def telemetry_nominal() -> TelemetryData:
    """Telemetría representativa de operación nominal (V << θ)."""
    return make_telemetry(V_SAFE, S_NOMINAL)


@pytest.fixture
def telemetry_critical() -> TelemetryData:
    """Telemetría representativa de condición crítica (V >> θ)."""
    return make_telemetry(V_CRISIS, S_CRITICAL)


@pytest.fixture
def telemetry_boundary() -> TelemetryData:
    """Telemetría exactamente en el umbral (V = θ)."""
    return make_telemetry(V_BOUNDARY, S_NOMINAL)


@pytest.fixture
def telemetry_theta_minus_ulp() -> TelemetryData:
    """Telemetría un ULP por debajo del umbral."""
    return make_telemetry(V_THETA_MINUS_ULP, S_NOMINAL)


@pytest.fixture
def telemetry_theta_plus_ulp() -> TelemetryData:
    """Telemetría un ULP por encima del umbral."""
    return make_telemetry(V_THETA_PLUS_ULP, S_NOMINAL)


# ═════════════════════════════════════════════════════════════════════════════
# SUITE DE PRUEBAS: FASE ORIENT
# ═════════════════════════════════════════════════════════════════════════════


class TestOrientPhase:
    """
    Verificación de que orient implementa la función de Heaviside H_θ.

    Cobertura
    ─────────
      • Clasificación básica en ambas regiones topológicas
      • Análisis de frontera (umbral exacto y vecindad ULP)
      • Propiedades algebraicas (pureza, determinismo, monotonía)
      • Independencia funcional respecto a saturación
      • Conformidad con la especificación de referencia
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Clasificación Básica
    # ─────────────────────────────────────────────────────────────────────────

    def test_nominal_region_classification(
        self, agent: AutonomousAgent, telemetry_nominal: TelemetryData
    ) -> None:
        """
        Verifica que V << θ ⟹ orient(V) = NOMINAL.

        Propiedad: ∀V ∈ [0, θ] : H_θ(V) = NOMINAL
        """
        status = agent.orient(telemetry_nominal)
        assert status == SystemStatus.NOMINAL, (
            f"Fallo en región nominal: V={telemetry_nominal.flyback_voltage:.17g}, "
            f"esperado NOMINAL, obtenido {status}"
        )

    def test_critical_region_classification(
        self, agent: AutonomousAgent, telemetry_critical: TelemetryData
    ) -> None:
        """
        Verifica que V >> θ ⟹ orient(V) = CRITICO.

        Propiedad: ∀V ∈ (θ, ∞) : H_θ(V) = CRITICO
        """
        status = agent.orient(telemetry_critical)
        assert status == SystemStatus.CRITICO, (
            f"Fallo en región crítica: V={telemetry_critical.flyback_voltage:.17g}, "
            f"esperado CRITICO, obtenido {status}"
        )

    def test_zero_voltage_is_nominal(self, agent: AutonomousAgent) -> None:
        """
        Verifica condición de frontera inferior: V = 0 ⟹ NOMINAL.

        Propiedad: H_θ(0) = NOMINAL (continuidad por la derecha en 0)
        """
        status = agent.orient(make_telemetry(0.0, 0.0))
        assert status == SystemStatus.NOMINAL

    # ─────────────────────────────────────────────────────────────────────────
    # Análisis de Frontera en θ
    # ─────────────────────────────────────────────────────────────────────────

    def test_exact_threshold_is_nominal(
        self, agent: AutonomousAgent, telemetry_boundary: TelemetryData
    ) -> None:
        """
        Verifica que V = θ ⟹ orient(θ) = NOMINAL.

        Justificación topológica:
          La región crítica es el intervalo abierto (θ, ∞),
          luego θ ∉ Ω_C ⟹ θ ∈ Ω_N (por exhaustividad de la partición).

        Propiedad: lim_{x→θ⁻} H_θ(x) = H_θ(θ) (continuidad por la izquierda)
        """
        status = agent.orient(telemetry_boundary)
        assert status == SystemStatus.NOMINAL, (
            "El umbral exacto θ debe pertenecer a la región nominal (intervalo cerrado)"
        )

    def test_theta_minus_ulp_is_nominal(
        self, agent: AutonomousAgent, telemetry_theta_minus_ulp: TelemetryData
    ) -> None:
        """
        Verifica que nextafter(θ, -∞) ⟹ NOMINAL.

        Propiedad: ∀ε>0 pequeño : H_θ(θ - ε) = NOMINAL
        """
        status = agent.orient(telemetry_theta_minus_ulp)
        assert status == SystemStatus.NOMINAL, (
            f"Un ULP por debajo del umbral debe ser NOMINAL: "
            f"V={telemetry_theta_minus_ulp.flyback_voltage:.17g}"
        )

    def test_theta_plus_ulp_is_critical(
        self, agent: AutonomousAgent, telemetry_theta_plus_ulp: TelemetryData
    ) -> None:
        """
        Verifica que nextafter(θ, +∞) ⟹ CRITICO.

        Propiedad: lim_{x→θ⁺} H_θ(x) = CRITICO
        """
        status = agent.orient(telemetry_theta_plus_ulp)
        assert status == SystemStatus.CRITICO, (
            f"Un ULP por encima del umbral debe ser CRITICO: "
            f"V={telemetry_theta_plus_ulp.flyback_voltage:.17g}"
        )

    def test_ulp_neighborhood_partition(self, agent: AutonomousAgent) -> None:
        """
        Verifica que la vecindad ULP de θ respeta la partición topológica.

        Propiedad: La función escalón no presenta estados intermedios,
                   incluso en la resolución más fina representable (ULP).
        """
        # Muestreo denso por debajo de θ
        below_samples = list(generate_ulp_neighborhood(THETA, radius=100, direction=-1))
        # Muestreo denso por encima de θ
        above_samples = list(generate_ulp_neighborhood(THETA, radius=100, direction=1))

        # Verificación de clasificación consistente
        for v in below_samples:
            status = agent.orient(make_telemetry(v))
            assert status == SystemStatus.NOMINAL, (
                f"Violación de continuidad por la izquierda en V={v:.17g}"
            )

        for v in above_samples:
            status = agent.orient(make_telemetry(v))
            assert status == SystemStatus.CRITICO, (
                f"Violación de discontinuidad de salto en V={v:.17g}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Propiedades Algebraicas
    # ─────────────────────────────────────────────────────────────────────────

    def test_orient_is_deterministic(
        self, agent: AutonomousAgent, telemetry_critical: TelemetryData
    ) -> None:
        """
        Verifica determinismo: idéntica entrada ⟹ idéntica salida.

        Propiedad: H_θ es función (no relación multivaluada)
        """
        results = {agent.orient(telemetry_critical) for _ in range(100)}
        assert len(results) == 1, (
            f"La función orient no es determinística: {results}"
        )

    def test_orient_is_pure_function(self, agent: AutonomousAgent) -> None:
        """
        Verifica ausencia de estado interno (pureza funcional).

        Propiedad: orient no posee histéresis ni memoria de llamadas previas.
        """
        test_sequence = [
            (V_SAFE, SystemStatus.NOMINAL),
            (V_CRISIS, SystemStatus.CRITICO),
            (V_SAFE, SystemStatus.NOMINAL),
            (V_CRISIS, SystemStatus.CRITICO),
            (V_BOUNDARY, SystemStatus.NOMINAL),
            (V_THETA_PLUS_ULP, SystemStatus.CRITICO),
            (0.0, SystemStatus.NOMINAL),
            (V_THETA_MINUS_ULP, SystemStatus.NOMINAL),
        ]

        for voltage, expected_status in test_sequence:
            actual_status = agent.orient(make_telemetry(voltage))
            assert actual_status == expected_status, (
                f"Violación de pureza funcional: V={voltage:.17g}, "
                f"esperado {expected_status}, obtenido {actual_status}"
            )

    def test_orient_has_no_hysteresis(
        self,
        agent: AutonomousAgent,
        telemetry_critical: TelemetryData,
        telemetry_nominal: TelemetryData,
    ) -> None:
        """
        Verifica que no existe histéresis: estado actual no depende de estado previo.

        Propiedad: orient(x_{n+1}) es independiente de orient(x_n)
        """
        # Inducir estado crítico
        agent.orient(telemetry_critical)
        # Verificar que estado nominal posterior es correcto
        status_after_crisis = agent.orient(telemetry_nominal)

        assert status_after_crisis == SystemStatus.NOMINAL, (
            "Detectada histéresis: el estado previo CRITICO afecta clasificación NOMINAL"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Independencia Funcional (Teorema C)
    # ─────────────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "voltage,expected_status",
        [
            (V_SAFE, SystemStatus.NOMINAL),
            (V_BOUNDARY, SystemStatus.NOMINAL),
            (V_THETA_PLUS_ULP, SystemStatus.CRITICO),
            (V_CRISIS, SystemStatus.CRITICO),
        ],
    )
    def test_orient_invariant_under_saturation_parametric(
        self, agent: AutonomousAgent, voltage: float, expected_status: SystemStatus
    ) -> None:
        """
        Verifica ∂(orient)/∂S = 0 para puntos representativos.

        Propiedad: ∀S₁,S₂ ∈ [0,1] : orient(V, S₁) = orient(V, S₂)
        """
        observed_statuses = {
            agent.orient(make_telemetry(voltage, saturation))
            for saturation in SATURATION_DOMAIN_SAMPLE
        }

        assert observed_statuses == {expected_status}, (
            f"Violación de independencia funcional en V={voltage:.17g}: "
            f"esperado {{{expected_status}}}, obtenido {observed_statuses}"
        )

    def test_teorema_c_independencia_funcional_exhaustiva(
        self, agent: AutonomousAgent
    ) -> None:
        """
        Teorema C: Independencia Funcional Total.

        Enunciado Formal:
          ∀V ∈ ℝ⁺, ∀S ∈ [0,1] : orient(V, S) = H_θ(V)

        Demostración por Barrido:
          Para cada V en el muestreo del dominio, verifica que
          la variación de S no altera la clasificación.
        """
        for voltage in generate_flyback_test_domain(num_uniform_samples=15):
            if not is_valid_domain_point(voltage, 0.0):
                continue

            reference_status = heaviside_reference(voltage, THETA)
            observed_statuses = {
                agent.orient(make_telemetry(voltage, saturation))
                for saturation in SATURATION_DOMAIN_SAMPLE
            }

            assert observed_statuses == {reference_status}, (
                f"Teorema C violado en V={voltage:.17g}: "
                f"esperado {{{reference_status}}}, obtenido {observed_statuses}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Monotonía (Teorema D)
    # ─────────────────────────────────────────────────────────────────────────

    def test_orient_is_monotone_increasing(self, agent: AutonomousAgent) -> None:
        """
        Teorema D: Monotonía de H_θ.

        Enunciado:
          ∀V₁,V₂ ∈ ℝ⁺ : V₁ ≤ V₂ ⟹ ord(H_θ(V₁)) ≤ ord(H_θ(V₂))

        donde ord: SystemStatus → {0, 1} con:
          ord(NOMINAL) = 0
          ord(CRITICO) = 1

        Método de Verificación:
          Recorre secuencia estrictamente creciente y verifica que
          las transiciones NOMINAL → CRITICO no retroceden.
        """
        status_sequence = [
            agent.orient(make_telemetry(v)) for v in FLYBACK_MONOTONE_SEQUENCE
        ]

        # Mapeo a valores ordinales para verificar orden
        ordinal_sequence = [
            0 if s == SystemStatus.NOMINAL else 1 for s in status_sequence
        ]

        assert ordinal_sequence == sorted(ordinal_sequence), (
            f"Violación de monotonía: secuencia ordinal {ordinal_sequence} "
            f"no está ordenada"
        )

    def test_single_transition_point_existence(self, agent: AutonomousAgent) -> None:
        """
        Verifica que existe exactamente una transición NOMINAL → CRITICO.

        Propiedad: La función escalón posee un único punto de discontinuidad.
        """
        samples = sorted(set(generate_flyback_test_domain(num_uniform_samples=50)))
        samples = [v for v in samples if is_valid_domain_point(v, 0.0)]

        status_sequence = [agent.orient(make_telemetry(v)) for v in samples]

        # Contar transiciones hacia adelante (N → C)
        forward_transitions = sum(
            1
            for i in range(len(status_sequence) - 1)
            if status_sequence[i] == SystemStatus.NOMINAL
            and status_sequence[i + 1] == SystemStatus.CRITICO
        )

        # Contar transiciones hacia atrás (C → N) — debe ser cero
        backward_transitions = sum(
            1
            for i in range(len(status_sequence) - 1)
            if status_sequence[i] == SystemStatus.CRITICO
            and status_sequence[i + 1] == SystemStatus.NOMINAL
        )

        assert forward_transitions == 1, (
            f"Debe existir exactamente una transición N→C, encontradas: {forward_transitions}"
        )
        assert backward_transitions == 0, (
            f"No debe haber transiciones C→N (violación de monotonía), "
            f"encontradas: {backward_transitions}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Conformidad con Especificación de Referencia
    # ─────────────────────────────────────────────────────────────────────────

    def test_orient_matches_heaviside_reference(self, agent: AutonomousAgent) -> None:
        """
        Verifica que orient(V) = H_θ(V) para todo V en el dominio de prueba.

        Propiedad: La implementación es fiel a la especificación matemática.
        """
        for voltage in generate_flyback_test_domain():
            if not is_valid_domain_point(voltage, 0.0):
                continue

            expected = heaviside_reference(voltage, THETA)
            actual = agent.orient(make_telemetry(voltage))

            assert actual == expected, (
                f"Discrepancia con referencia en V={voltage:.17g}: "
                f"esperado {expected}, obtenido {actual}"
            )

    @pytest.mark.parametrize(
        "voltage,expected",
        [
            (0.0, SystemStatus.NOMINAL),
            (V_SAFE, SystemStatus.NOMINAL),
            (V_THETA_MINUS_ULP, SystemStatus.NOMINAL),
            (V_BOUNDARY, SystemStatus.NOMINAL),
            (V_THETA_PLUS_ULP, SystemStatus.CRITICO),
            (V_MAX, SystemStatus.CRITICO),
            (V_CRISIS, SystemStatus.CRITICO),
        ],
    )
    def test_classification_lookup_table(
        self, agent: AutonomousAgent, voltage: float, expected: SystemStatus
    ) -> None:
        """
        Tabla de verdad exhaustiva para puntos representativos.

        Garantiza cobertura de:
          • Frontera inferior (0)
          • Región nominal interior
          • Vecindad inferior de θ
          • Umbral exacto θ
          • Vecindad superior de θ
          • Región crítica interior
          • Frontera superior (post-clamping)
        """
        actual = agent.orient(make_telemetry(voltage))
        assert actual == expected, (
            f"Fallo en tabla de clasificación: V={voltage:.17g}, "
            f"esperado {expected}, obtenido {actual}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# SUITE DE PRUEBAS: ROBUSTEZ NUMÉRICA
# ═════════════════════════════════════════════════════════════════════════════


class TestNumericalRobustness:
    """
    Verificación de comportamiento robusto con valores extremos IEEE 754.

    Cobertura
    ─────────
      • Números subnormales (denormalizados)
      • Límites de representación (mínimos/máximos normales)
      • Valores grandes finitos
      • Estabilidad de comparaciones en coma flotante
    """

    def test_subnormal_positive_classification(self, agent: AutonomousAgent) -> None:
        """
        Verifica que números subnormales positivos se clasifican como NOMINAL.

        Propiedad: ∀x ∈ (0, min_normal) : x < θ ⟹ H_θ(x) = NOMINAL
        """
        status = agent.orient(make_telemetry(SUBNORMAL_MIN))
        assert status == SystemStatus.NOMINAL, (
            f"Fallo con número subnormal: V={SUBNORMAL_MIN:.17e}"
        )

    def test_min_positive_normal_classification(self, agent: AutonomousAgent) -> None:
        """
        Verifica clasificación del mínimo número normal positivo.

        Propiedad: min_normal < θ ⟹ H_θ(min_normal) = NOMINAL
        """
        status = agent.orient(make_telemetry(NORMAL_MIN))
        assert status == SystemStatus.NOMINAL, (
            f"Fallo con mínimo normal: V={NORMAL_MIN:.17e}"
        )

    def test_large_finite_values_are_critical(self, agent: AutonomousAgent) -> None:
        """
        Verifica que valores grandes finitos se clasifican como CRITICO.

        Propiedad: ∀x ≫ θ : H_θ(x) = CRITICO (antes del clamping)
        """
        large_values = [1e3, 1e6, 1e10, V_CRISIS]
        for v in large_values:
            status = agent.orient(make_telemetry(v))
            assert status == SystemStatus.CRITICO, (
                f"Valor grande no clasificado como CRITICO: V={v:.17e}"
            )

    @pytest.mark.parametrize(
        "saturation", [0.0, SUBNORMAL_MIN, 0.5, 0.95, 1.0 - MACHINE_EPSILON]
    )
    def test_extreme_saturation_independence(
        self, agent: AutonomousAgent, saturation: float
    ) -> None:
        """
        Verifica independencia funcional con valores extremos de saturación.

        Propiedad: ∂(orient)/∂S = 0 incluso para S extremos.
        """
        nominal_status = agent.orient(make_telemetry(V_SAFE, saturation))
        critical_status = agent.orient(make_telemetry(V_CRISIS, saturation))

        assert nominal_status == SystemStatus.NOMINAL, (
            f"Saturación extrema afectó clasificación nominal: S={saturation:.17e}"
        )
        assert critical_status == SystemStatus.CRITICO, (
            f"Saturación extrema afectó clasificación crítica: S={saturation:.17e}"
        )

    def test_threshold_comparison_stability(
        self, agent: AutonomousAgent, telemetry_boundary: TelemetryData
    ) -> None:
        """
        Verifica estabilidad de comparación en el umbral exacto.

        Propiedad: La evaluación de V = θ debe ser consistente y reproducible
                   (sin deriva por errores de redondeo acumulativos).
        """
        results = {agent.orient(telemetry_boundary) for _ in range(1000)}
        assert results == {SystemStatus.NOMINAL}, (
            f"Inestabilidad numérica detectada en θ: {results}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# SUITE DE PRUEBAS: FASE DECIDE
# ═════════════════════════════════════════════════════════════════════════════


class TestDecidePhase:
    """
    Verificación de la función decide: SystemStatus → AgentDecision ⊔ {⊥}.

    Cobertura
    ─────────
      • Implicación obligatoria: CRITICO → ALERTA_CRITICA
      • Prohibición: ¬CRITICO → ¬ALERTA_CRITICA
      • Determinismo y pureza
      • Idempotencia
    """

    def test_critical_status_implies_critical_alert(
        self, agent: AutonomousAgent
    ) -> None:
        """
        Verifica la implicación: CRITICO ⟹ ALERTA_CRITICA.

        Propiedad: decide(CRITICO) = ALERTA_CRITICA (obligatorio)
        """
        decision = agent.decide(SystemStatus.CRITICO)
        assert decision == AgentDecision.ALERTA_CRITICA, (
            f"Estado CRITICO debe generar ALERTA_CRITICA, obtenido: {decision}"
        )

    def test_nominal_status_excludes_critical_alert(
        self, agent: AutonomousAgent
    ) -> None:
        """
        Verifica que NOMINAL ⟹ ¬ALERTA_CRITICA.

        Propiedad: decide(NOMINAL) ≠ ALERTA_CRITICA
        """
        decision = agent.decide(SystemStatus.NOMINAL)
        assert decision != AgentDecision.ALERTA_CRITICA, (
            f"Estado NOMINAL no debe generar ALERTA_CRITICA, obtenido: {decision}"
        )

    def test_non_critical_states_exclude_critical_alert(
        self, agent: AutonomousAgent
    ) -> None:
        """
        Verifica que ∀s ≠ CRITICO : decide(s) ≠ ALERTA_CRITICA.

        Propiedad: ALERTA_CRITICA es exclusiva del estado CRITICO.
        """
        for status in SystemStatus:
            if status == SystemStatus.CRITICO:
                continue

            decision = agent.decide(status)
            assert decision != AgentDecision.ALERTA_CRITICA, (
                f"Estado {status} generó ALERTA_CRITICA inapropiadamente"
            )

    @pytest.mark.parametrize("status", tuple(SystemStatus))
    def test_decide_returns_valid_type(
        self, agent: AutonomousAgent, status: SystemStatus
    ) -> None:
        """
        Verifica que decide retorna un tipo válido.

        Propiedad: decide : SystemStatus → AgentDecision ⊔ {None}
        """
        decision = agent.decide(status)
        assert decision is None or isinstance(decision, AgentDecision), (
            f"Tipo de retorno inválido para {status}: {type(decision)}"
        )

    @pytest.mark.parametrize("status", tuple(SystemStatus))
    def test_decide_is_deterministic(
        self, agent: AutonomousAgent, status: SystemStatus
    ) -> None:
        """
        Verifica determinismo de decide.

        Propiedad: ∀s ∈ SystemStatus : decide(s) es único.
        """
        decisions = {agent.decide(status) for _ in range(100)}
        assert len(decisions) == 1, (
            f"La función decide no es determinística para {status}: {decisions}"
        )

    def test_decide_is_pure_function(self, agent: AutonomousAgent) -> None:
        """
        Verifica pureza funcional (ausencia de estado).

        Propiedad: decide no posee memoria de invocaciones previas.
        """
        test_sequence = [
            SystemStatus.NOMINAL,
            SystemStatus.CRITICO,
            SystemStatus.NOMINAL,
            SystemStatus.CRITICO,
            SystemStatus.NOMINAL,
        ]

        first_run = [agent.decide(s) for s in test_sequence]
        second_run = [agent.decide(s) for s in test_sequence]

        assert first_run == second_run, (
            "La función decide no es pura (resultados difieren entre ejecuciones)"
        )

    @pytest.mark.parametrize("status", tuple(SystemStatus))
    def test_decide_idempotence(
        self, agent: AutonomousAgent, status: SystemStatus
    ) -> None:
        """
        Verifica idempotencia: múltiples llamadas ⟹ idéntico resultado.

        Propiedad: ∀s : decide(s) = decide(decide(s)) (en el sentido funcional)
        """
        first_decision = agent.decide(status)
        subsequent_decisions = [agent.decide(status) for _ in range(50)]

        assert all(d == first_decision for d in subsequent_decisions), (
            f"Violación de idempotencia para {status}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# SUITE DE PRUEBAS: PIPELINE OODA COMPLETO
# ═════════════════════════════════════════════════════════════════════════════


class TestOODAReflexLoop:
    """
    Verificación del pipeline compuesto: decide ∘ orient.

    Cobertura
    ─────────
      • Conformidad del pipeline con el contrato completo
      • Ciclos de transición (nominal → crítico → nominal)
      • Cobertura exhaustiva de estados alcanzables
      • Transitividad de propiedades
    """

    @pytest.mark.parametrize(
        "voltage,expected_status,expects_critical_alert",
        [
            (0.0, SystemStatus.NOMINAL, False),
            (V_SAFE, SystemStatus.NOMINAL, False),
            (V_THETA_MINUS_ULP, SystemStatus.NOMINAL, False),
            (V_BOUNDARY, SystemStatus.NOMINAL, False),
            (V_THETA_PLUS_ULP, SystemStatus.CRITICO, True),
            (V_CRISIS, SystemStatus.CRITICO, True),
        ],
    )
    def test_pipeline_contract_compliance(
        self,
        agent: AutonomousAgent,
        voltage: float,
        expected_status: SystemStatus,
        expects_critical_alert: bool,
    ) -> None:
        """
        Teorema E: Transitividad del Pipeline OODA.

        Enunciado:
          (decide ∘ orient)(V) = {
            ALERTA_CRITICA   si V > θ
            ⊥ ∨ ¬ALERTA      si V ≤ θ
          }
        """
        telemetry = make_telemetry(voltage)
        status = agent.orient(telemetry)
        decision = agent.decide(status)

        assert status == expected_status, (
            f"Fase Orient falló: V={voltage:.17g}, "
            f"esperado {expected_status}, obtenido {status}"
        )

        if expects_critical_alert:
            assert decision == AgentDecision.ALERTA_CRITICA, (
                f"Fase Decide falló para condición crítica: V={voltage:.17g}, "
                f"obtenido {decision}"
            )
        else:
            assert decision != AgentDecision.ALERTA_CRITICA, (
                f"Fase Decide generó alerta crítica inapropiadamente: V={voltage:.17g}"
            )

    def test_nominal_critical_recovery_cycle(
        self,
        agent: AutonomousAgent,
        telemetry_nominal: TelemetryData,
        telemetry_critical: TelemetryData,
    ) -> None:
        """
        Verifica ciclo completo: NOMINAL → CRITICO → NOMINAL.

        Propiedad: El sistema puede transitar entre todos los estados
                   sin histéresis ni bloqueos.
        """
        # Estado inicial: NOMINAL
        status_initial = agent.orient(telemetry_nominal)
        decision_initial = agent.decide(status_initial)
        assert status_initial == SystemStatus.NOMINAL
        assert decision_initial != AgentDecision.ALERTA_CRITICA

        # Transición a CRITICO
        status_crisis = agent.orient(telemetry_critical)
        decision_crisis = agent.decide(status_crisis)
        assert status_crisis == SystemStatus.CRITICO
        assert decision_crisis == AgentDecision.ALERTA_CRITICA

        # Recuperación a NOMINAL
        status_recovery = agent.orient(telemetry_nominal)
        decision_recovery = agent.decide(status_recovery)
        assert status_recovery == SystemStatus.NOMINAL
        assert decision_recovery != AgentDecision.ALERTA_CRITICA

    def test_pipeline_exhaustive_state_coverage(
        self, agent: AutonomousAgent
    ) -> None:
        """
        Teorema F: Alcanzabilidad Global.

        Enunciado:
          ∀s₁,s₂ ∈ SystemStatus : ∃V ∈ ℝ⁺ tal que:
            orient(V) = s₁ ∧ ∃V' : orient(V') = s₂

        Método: Barrido exhaustivo del dominio y verificación de cobertura.
        """
        reached_statuses = set()
        reached_decisions = set()

        for voltage in generate_flyback_test_domain(num_uniform_samples=30):
            if not is_valid_domain_point(voltage, 0.0):
                continue

            status = agent.orient(make_telemetry(voltage))
            decision = agent.decide(status)

            reached_statuses.add(status)
            reached_decisions.add(decision)

        # Verificar cobertura de todos los estados relevantes
        assert SystemStatus.NOMINAL in reached_statuses, (
            "Estado NOMINAL no fue alcanzado en el barrido"
        )
        assert SystemStatus.CRITICO in reached_statuses, (
            "Estado CRITICO no fue alcanzado en el barrido"
        )
        assert AgentDecision.ALERTA_CRITICA in reached_decisions, (
            "Decisión ALERTA_CRITICA no fue alcanzada en el barrido"
        )

    def test_teorema_b_heaviside_y_pipeline(self, agent: AutonomousAgent) -> None:
        """
        Teorema B: Conformidad de Heaviside en el Pipeline.

        Enunciado:
          ∀V ∈ ℝ⁺ :
            V ≤ θ ⟹ orient(V) = NOMINAL ∧ decide(orient(V)) ≠ ALERTA_CRITICA
            V > θ ⟹ orient(V) = CRITICO ∧ decide(orient(V)) = ALERTA_CRITICA
        """
        # Caso: V = θ - 1 ULP (justo por debajo)
        telem_below = make_telemetry(V_THETA_MINUS_ULP)
        status_below = agent.orient(telem_below)
        decision_below = agent.decide(status_below)

        assert status_below == SystemStatus.NOMINAL, (
            f"Fallo en frontera inferior: V={V_THETA_MINUS_ULP:.17g}"
        )
        assert decision_below != AgentDecision.ALERTA_CRITICA, (
            "Alerta crítica generada inapropiadamente en frontera inferior"
        )

        # Caso: V = θ + 1 ULP (justo por encima)
        telem_above = make_telemetry(V_THETA_PLUS_ULP)
        status_above = agent.orient(telem_above)
        decision_above = agent.decide(status_above)

        assert status_above == SystemStatus.CRITICO, (
            f"Fallo en frontera superior: V={V_THETA_PLUS_ULP:.17g}"
        )
        assert decision_above == AgentDecision.ALERTA_CRITICA, (
            "Alerta crítica no generada en frontera superior"
        )


# ═════════════════════════════════════════════════════════════════════════════
# SUITE DE PRUEBAS: GRAFO DE TRANSICIONES DE ESTADOS
# ═════════════════════════════════════════════════════════════════════════════


class TestStateTransitionGraph:
    """
    Verificación de propiedades del grafo de estados implícito.

    Modelo Teórico
    ──────────────
    Grafo dirigido G = (V, E) donde:
      • V = {NOMINAL, CRITICO}
      • E = {(s, t) : ∃V ∈ ℝ⁺ tal que orient(V) = t}

    Para un sistema sin estado (función pura), E = V × V (grafo completo).

    Cobertura
    ─────────
      • Alcanzabilidad: todo estado es alcanzable desde cualquier estado
      • Conexión fuerte: ∀u,v ∈ V : ∃ camino u → v
      • Determinismo de transiciones
    """

    def test_all_states_reachable_from_nominal(
        self, agent: AutonomousAgent
    ) -> None:
        """
        Verifica que desde NOMINAL se puede alcanzar cualquier estado.

        Propiedad: ∀t ∈ V : ∃V tal que orient(V) = t
        """
        # Alcanzar NOMINAL
        status_nominal = agent.orient(make_telemetry(V_SAFE))
        assert status_nominal == SystemStatus.NOMINAL

        # Alcanzar CRITICO desde cualquier punto (sin histéresis)
        status_critical = agent.orient(make_telemetry(V_CRISIS))
        assert status_critical == SystemStatus.CRITICO

    def test_all_states_reachable_from_critical(
        self, agent: AutonomousAgent
    ) -> None:
        """
        Verifica que desde CRITICO se puede alcanzar cualquier estado.

        Propiedad: Simetría de alcanzabilidad (grafo no dirigido en funcional puro).
        """
        # Alcanzar CRITICO
        status_critical = agent.orient(make_telemetry(V_CRISIS))
        assert status_critical == SystemStatus.CRITICO

        # Alcanzar NOMINAL desde cualquier punto (sin histéresis)
        status_nominal = agent.orient(make_telemetry(V_SAFE))
        assert status_nominal == SystemStatus.NOMINAL

    def test_state_graph_is_strongly_connected(
        self, agent: AutonomousAgent
    ) -> None:
        """
        Verifica que el grafo de estados es fuertemente conexo.

        Propiedad: ∀u,v ∈ V : ∃ camino u → v con longitud ≤ 1.

        Para un sistema puro sin histéresis, la longitud es siempre 1.
        """
        telemetry_map = {
            SystemStatus.NOMINAL: make_telemetry(V_SAFE),
            SystemStatus.CRITICO: make_telemetry(V_CRISIS),
        }

        for source_status in SystemStatus:
            for target_status, target_telemetry in telemetry_map.items():
                reached_status = agent.orient(target_telemetry)
                assert reached_status == target_status, (
                    f"No se puede alcanzar {target_status} desde {source_status}"
                )

    def test_transition_determinism(self, agent: AutonomousAgent) -> None:
        """
        Verifica que las transiciones son determinísticas.

        Propiedad: ∀V ∈ ℝ⁺ : la transición inducida por V es única.
        """
        test_telemetries = [
            make_telemetry(V_SAFE),
            make_telemetry(V_CRISIS),
            make_telemetry(V_BOUNDARY),
            make_telemetry(V_THETA_MINUS_ULP),
            make_telemetry(V_THETA_PLUS_ULP),
        ]

        for telemetry in test_telemetries:
            transitions = {agent.orient(telemetry) for _ in range(100)}
            assert len(transitions) == 1, (
                f"Transición no determinística para V={telemetry.flyback_voltage:.17g}: "
                f"{transitions}"
            )

    def test_adjacency_matrix_is_complete(self, agent: AutonomousAgent) -> None:
        """
        Verifica que la matriz de adyacencia es completa (K₂).

        Propiedad: Para un sistema puro, todas las transiciones (u,v) son posibles.
        """
        states = list(SystemStatus)
        test_voltages = [V_SAFE, V_CRISIS]

        transition_matrix = compute_transition_matrix(agent, states, test_voltages)

        # Verificar que todas las entradas (u,v) existen
        for src in states:
            for dst in states:
                assert (src, dst) in transition_matrix or dst in {
                    agent.orient(make_telemetry(v)) for v in test_voltages
                }, f"Transición {src} → {dst} no observada"


# ═════════════════════════════════════════════════════════════════════════════
# SUITE DE PRUEBAS: INVARIANTES ALGEBRAICOS DE ENUMERACIONES
# ═════════════════════════════════════════════════════════════════════════════


class TestSystemStatusInvariants:
    """
    Verificación de propiedades algebraicas de SystemStatus y AgentDecision.

    Cobertura
    ─────────
      • Distinción de elementos (inyectividad de constructores)
      • Valores asignados (verificación de asignaciones explícitas)
      • Jerarquía de tipos (verificación de herencia)
      • Exhaustividad y disjunción de particiones
    """

    def test_nominal_and_critical_are_distinct(self) -> None:
        """
        Verifica que NOMINAL ≠ CRITICO.

        Propiedad: Inyectividad de constructores de enum.
        """
        assert SystemStatus.NOMINAL != SystemStatus.CRITICO, (
            "Los estados NOMINAL y CRITICO deben ser distintos"
        )

    def test_enum_value_assignments(self) -> None:
        """
        Verifica asignaciones explícitas de valores.

        Propiedad: Contratos de valores para interoperabilidad externa.
        """
        assert SystemStatus.NOMINAL.value == 1, (
            f"NOMINAL debe tener valor 1, obtenido: {SystemStatus.NOMINAL.value}"
        )
        assert SystemStatus.CRITICO.value == 5, (
            f"CRITICO debe tener valor 5, obtenido: {SystemStatus.CRITICO.value}"
        )
        assert AgentDecision.ALERTA_CRITICA.value == 4, (
            f"ALERTA_CRITICA debe tener valor 4, obtenido: {AgentDecision.ALERTA_CRITICA.value}"
        )

    def test_enums_are_proper_subclasses(self) -> None:
        """
        Verifica jerarquía de tipos.

        Propiedad: SystemStatus, AgentDecision <: Enum
        """
        assert issubclass(SystemStatus, Enum), (
            "SystemStatus debe ser subclase de Enum"
        )
        assert issubclass(AgentDecision, Enum), (
            "AgentDecision debe ser subclase de Enum"
        )

    def test_partition_is_exhaustive_and_disjoint(self) -> None:
        """
        Teorema A: Partición Completa.

        Enunciado:
          {NOMINAL, CRITICO} ⊆ SystemStatus
          NOMINAL ∩ CRITICO = ∅
          NOMINAL ∪ CRITICO cubre el dominio relevante

        Verificación: Los elementos de la partición están en el enum
                      y no hay duplicados.
        """
        status_set = set(SystemStatus)

        assert SystemStatus.NOMINAL in status_set, (
            "NOMINAL debe estar en SystemStatus"
        )
        assert SystemStatus.CRITICO in status_set, (
            "CRITICO debe estar en SystemStatus"
        )

        # Verificar que no hay duplicados (inyectividad)
        assert len(status_set) == len(tuple(SystemStatus)), (
            "Elementos duplicados detectados en SystemStatus"
        )


# ═════════════════════════════════════════════════════════════════════════════
# SUITE DE PRUEBAS: CONTRATOS DE TelemetryData
# ═════════════════════════════════════════════════════════════════════════════


class TestTelemetryDataContracts:
    """
    Verificación de contratos de la estructura de datos TelemetryData.

    Cobertura
    ─────────
      • Preservación de campos (flyback_voltage, saturation)
      • Clamping: V > V_MAX ⟹ V' = V_MAX ∧ was_clamped = True
      • Valores frontera (0, 1, θ)
      • Relaciones de orden (ULP preservation)
      • Igualdad estructural (reflexiva, simétrica, transitiva)
    """

    def test_flyback_voltage_storage(self) -> None:
        """
        Verifica almacenamiento del campo flyback_voltage.

        Propiedad: π₁(make_telemetry(V, S)) = min(V, V_MAX)
        """
        telemetry = make_telemetry(0.42, 0.3)
        assert telemetry.flyback_voltage == pytest.approx(0.42), (
            "El campo flyback_voltage no se almacena correctamente"
        )

    def test_saturation_storage(self) -> None:
        """
        Verifica almacenamiento del campo saturation.

        Propiedad: π₂(make_telemetry(V, S)) = S
        """
        telemetry = make_telemetry(0.1, 0.75)
        assert telemetry.saturation == pytest.approx(0.75), (
            "El campo saturation no se almacena correctamente"
        )

    def test_zero_values_handling(self) -> None:
        """
        Verifica manejo de valores cero (condición de frontera inferior).

        Propiedad: make_telemetry(0, 0) es válido y no requiere clamping.
        """
        telemetry = make_telemetry(0.0, 0.0)
        assert telemetry.flyback_voltage == 0.0
        assert telemetry.saturation == 0.0
        assert telemetry.was_clamped is False, (
            "El valor cero no debe activar clamping"
        )

    def test_high_flyback_clamping(self) -> None:
        """
        Verifica clamping para V > V_MAX.

        Contrato:
          ∀V > V_MAX :
            make_telemetry(V).flyback_voltage = V_MAX
            make_telemetry(V).was_clamped = True
        """
        telemetry = make_telemetry(V_CRISIS, S_CRITICAL)
        assert telemetry.flyback_voltage == pytest.approx(V_MAX), (
            f"Voltaje alto debe ser clampado a {V_MAX}, obtenido: {telemetry.flyback_voltage}"
        )
        assert telemetry.was_clamped is True, (
            "El flag was_clamped debe ser True para valores clampados"
        )

    def test_low_flyback_no_clamping(self) -> None:
        """
        Verifica ausencia de clamping para V ≤ V_MAX.

        Contrato:
          ∀V ≤ V_MAX :
            make_telemetry(V).flyback_voltage = V
            make_telemetry(V).was_clamped = False
        """
        test_voltages = [0.0, 0.5, V_MAX, V_THETA_MINUS_ULP, V_BOUNDARY]

        for voltage in test_voltages:
            telemetry = make_telemetry(voltage)
            assert telemetry.flyback_voltage == pytest.approx(voltage), (
                f"Voltaje válido no debe ser modificado: V={voltage:.17g}"
            )
            assert telemetry.was_clamped is False, (
                f"Flag was_clamped debe ser False para V={voltage:.17g}"
            )

    def test_boundary_value_ordering(self) -> None:
        """
        Verifica preservación del orden en valores frontera.

        Propiedad: V₁ < V₂ < V₃ ⟹ make_telemetry(V₁).flyback_voltage < ... < ...
        """
        telem_minus = make_telemetry(V_THETA_MINUS_ULP)
        telem_exact = make_telemetry(V_BOUNDARY)
        telem_plus = make_telemetry(V_THETA_PLUS_ULP)

        assert (
            telem_minus.flyback_voltage
            < telem_exact.flyback_voltage
            < telem_plus.flyback_voltage
        ), "El orden relativo de voltajes no se preserva"

    def test_equality_is_reflexive(self) -> None:
        """
        Verifica reflexividad de la igualdad.

        Propiedad: ∀x : x = x
        """
        telemetry = make_telemetry(0.5, 0.5)
        assert telemetry == telemetry, "La igualdad no es reflexiva"

    def test_equality_is_symmetric(self) -> None:
        """
        Verifica simetría de la igualdad.

        Propiedad: x = y ⟹ y = x
        """
        telem1 = make_telemetry(0.5, 0.5)
        telem2 = make_telemetry(0.5, 0.5)

        assert telem1 == telem2 and telem2 == telem1, (
            "La igualdad no es simétrica"
        )

    def test_inequality_detection(self) -> None:
        """
        Verifica detección de desigualdad estructural.

        Propiedad: Cambios en cualquier campo inducen desigualdad.
        """
        base = make_telemetry(0.5, 0.5)

        assert base != make_telemetry(0.6, 0.5), (
            "Cambio en flyback_voltage no detectado"
        )
        assert base != make_telemetry(0.5, 0.6), (
            "Cambio en saturation no detectado"
        )


# ═════════════════════════════════════════════════════════════════════════════
# SUITE DE PRUEBAS: PROPIEDADES UNIVERSALES (Property-Based Testing Manual)
# ═════════════════════════════════════════════════════════════════════════════


class TestUniversalProperties:
    """
    Verificación de propiedades universales sin librería externa.

    Método: Muestreo estratificado del dominio para maximizar cobertura.

    Cobertura
    ─────────
      • Validez de tipos de salida
      • Contrato de umbral
      • Monotonía relativa
      • Independencia funcional total
    """

    @pytest.mark.parametrize(
        "voltage",
        [i * 0.05 for i in range(25)] + [V_CRISIS, V_THETA_MINUS_ULP, V_THETA_PLUS_ULP],
    )
    def test_orient_output_validity(
        self, agent: AutonomousAgent, voltage: float
    ) -> None:
        """
        Propiedad Universal 1: Validez de Tipo de Salida.

        ∀V ∈ ℝ⁺ : orient(V) ∈ SystemStatus
        """
        status = agent.orient(make_telemetry(voltage))
        assert isinstance(status, SystemStatus), (
            f"Tipo de salida inválido para V={voltage:.17g}: {type(status)}"
        )
        assert status in tuple(SystemStatus), (
            f"Estado no reconocido: {status}"
        )

    @pytest.mark.parametrize(
        "voltage",
        [i * 0.05 for i in range(25)] + [V_CRISIS],
    )
    def test_orient_threshold_contract(
        self, agent: AutonomousAgent, voltage: float
    ) -> None:
        """
        Propiedad Universal 2: Contrato de Umbral.

        ∀V ∈ ℝ⁺ :
          V ≤ θ ⟹ orient(V) = NOMINAL
          V > θ ⟹ orient(V) = CRITICO
        """
        status = agent.orient(make_telemetry(voltage))

        # Aplicar clamping manualmente para la comparación
        clamped_voltage = min(voltage, V_MAX)

        if clamped_voltage <= THETA:
            assert status == SystemStatus.NOMINAL, (
                f"Violación del contrato: V={voltage:.17g} (clampado={clamped_voltage:.17g}) "
                f"debe ser NOMINAL, obtenido {status}"
            )
        else:
            assert status == SystemStatus.CRITICO, (
                f"Violación del contrato: V={voltage:.17g} (clampado={clamped_voltage:.17g}) "
                f"debe ser CRITICO, obtenido {status}"
            )

    @pytest.mark.parametrize(
        "v1,v2",
        [
            (0.1, 0.2),
            (0.5, 0.7),
            (V_THETA_MINUS_ULP, V_BOUNDARY),
            (V_BOUNDARY, V_THETA_PLUS_ULP),
            (0.9, V_CRISIS),
        ],
    )
    def test_orient_relative_monotonicity(
        self, agent: AutonomousAgent, v1: float, v2: float
    ) -> None:
        """
        Propiedad Universal 3: Monotonía Relativa.

        ∀V₁,V₂ : V₁ < V₂ ∧ orient(V₁) = CRITICO ⟹ orient(V₂) = CRITICO
        """
        assert v1 < v2, "Precondición violada: v1 debe ser menor que v2"

        status1 = agent.orient(make_telemetry(v1))
        status2 = agent.orient(make_telemetry(v2))

        if status1 == SystemStatus.CRITICO:
            assert status2 == SystemStatus.CRITICO, (
                f"Violación de monotonía: V₁={v1:.17g} → CRITICO, "
                f"pero V₂={v2:.17g} → {status2}"
            )

    def test_teorema_c_independence_comprehensive(
        self, agent: AutonomousAgent
    ) -> None:
        """
        Propiedad Universal 4: Independencia Funcional Comprehensiva.

        Teorema C (reformulado):
          ∀V ∈ ℝ⁺, ∀S₁,S₂ ∈ [0,1] : orient(V, S₁) = orient(V, S₂)

        Método: Barrido exhaustivo en producto cartesiano de dominios.
        """
        voltage_samples = [
            0.0,
            V_SAFE,
            V_THETA_MINUS_ULP,
            V_BOUNDARY,
            V_THETA_PLUS_ULP,
            V_CRISIS,
        ]

        for voltage in voltage_samples:
            # Clasificación de referencia con saturación arbitraria
            reference_status = agent.orient(make_telemetry(voltage, saturation=0.5))

            # Verificar invarianza para todas las saturaciones
            for saturation in SATURATION_DOMAIN_SAMPLE:
                actual_status = agent.orient(make_telemetry(voltage, saturation))
                assert actual_status == reference_status, (
                    f"Teorema C violado: V={voltage:.17g}, S={saturation:.17g}, "
                    f"esperado {reference_status}, obtenido {actual_status}"
                )


# ═════════════════════════════════════════════════════════════════════════════
# EJECUCIÓN Y METADATOS
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
            "--strict-markers",
            "-ra",
            "--color=yes",
        ]
    )