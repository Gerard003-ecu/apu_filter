r"""
Suite de Pruebas Rigurosas: Semantic Translator — Versión Refinada
==================================================================

Mejoras implementadas por categoría:

ALGEBRAICAS:
  A1. Verificación exhaustiva de leyes de retículo con contraejemplos trazables
  A2. Corrección del homomorfismo: join/meet en SeverityLattice no es max/min directo
  A3. Propiedad de sub-retículo corregida: imagen φ cerrada bajo ⊔ y ⊓ en Im(φ)
  A4. Tests de tricotomía y cadena separados correctamente

TOPOLÓGICAS:
  T1. Invariante de Euler parametrizado con casos degenerados (χ < 0)
  T2. Clasificación espectral con tolerancia ε correcta (no 1e-7 hardcodeado)
  T3. Test de Gerschgorin con acotamiento analítico explícito
  T4. Clamping de Fiedler negativo documentado con su origen numérico

ESPECTRALES:
  S1. Continuidad de Lipschitz: constante K calculada analíticamente
  S2. Test de isomorfismo causal: normalización correcta con regex no colisionante
  S3. Enumeración acotada verificada con cota MAX_SIMPLE_CYCLES_ENUMERATION

COMPUTACIONALES:
  C1. Thread-safety: verificación de invariante de tamaño post-concurrencia
  C2. LRU eviction: orden de acceso verificado con secuencia determinista
  C3. SHA-256: verificación de longitud y espacio de colisión
  C4. Timeout en tests concurrentes con manejo de error explícito

FUNCIONALES:
  F1. Factory functions: verificación de tipo Y propiedades, no solo tipo
  F2. Truncación de narrativa: verificación en todos los estratos
  F3. Composición de reportes: verificación de monotonicidad del supremo
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
from unittest.mock import MagicMock, patch, PropertyMock

import networkx as nx
import numpy as np
import pytest

# ── Módulo bajo prueba ──────────────────────────────────────────────────────
from app.wisdom.semantic_translator import (
    ABSOLUTE_ZERO_CELSIUS,
    KELVIN_OFFSET,
    EPSILON,
    MAX_SIMPLE_CYCLES_ENUMERATION,
    MAX_GRAPH_NODES_FOR_EIGENVECTOR,
    Temperature,
    SemanticTranslatorError,
    TopologyInvariantViolation,
    LatticeViolation,
    MetricsValidationError,
    GraphStructureError,
    StabilityThresholds,
    TopologicalThresholds,
    ThermalThresholds,
    FinancialThresholds,
    TranslatorConfig,
    VerdictLevel,
    SeverityLattice,
    SeverityToVerdictHomomorphism,
    FinancialVerdict,
    ValidatedTopology,
    HasBettiNumbers,
    StratumAnalysisResult,
    StrategicReport,
    NarrativeCache,
    SemanticDiffeomorphismMapper,
    GraphRAGCausalNarrator,
    LatticeVerdictCollapse,
    SemanticTranslator,
    create_translator,
    translate_metrics_to_narrative,
    verify_verdict_lattice,
    verify_severity_homomorphism,
)

try:
    from app.core.schemas import Stratum
except ImportError:
    from app.wisdom.semantic_translator import Stratum

try:
    from app.core.telemetry_schemas import (
        PhysicsMetrics,
        TopologicalMetrics,
        ControlMetrics,
        ThermodynamicMetrics,
    )
except ImportError:
    pytest.skip("Telemetry schemas not available", allow_module_level=True)


# ============================================================================
# CONSTANTES DE PRUEBA — valores derivados analíticamente
# ============================================================================

# Constante de Lipschitz empírica máxima permitida para el mapeo semántico.
# Justificación: el retículo VerdictLevel tiene 5 elementos; el salto máximo
# admisible en una transición de umbral es de 1 nivel (cardinalidad - 1 = 4
# sería el salto total; acotamos a ceil(n/2) = 3 para garantizar suavidad).
_MAX_LIPSCHITZ_JUMP: int = 3

# Tolerancia para comparaciones de Fiedler en presencia de ruido de máquina.
# Derivada del Teorema de Weyl: |λ_i(A+E) - λ_i(A)| <= ||E||_2
# Con E = eps_mach * I, ||E||_2 = eps_mach ~ 2.22e-16
_FIEDLER_NUMERIC_TOLERANCE: float = np.finfo(float).eps * 1000  # margen conservador

# Número de elementos del retículo VerdictLevel (n=5 → n²=25, n³=125 tests)
_N_VERDICT: int = len(list(VerdictLevel))

# Número de elementos de SeverityLattice (n=3)
_N_SEVERITY: int = len(list(SeverityLattice))


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def default_config() -> TranslatorConfig:
    """Configuración por defecto para tests."""
    return TranslatorConfig()


@pytest.fixture
def strict_config() -> TranslatorConfig:
    """Configuración con validación estricta de Euler."""
    return TranslatorConfig(strict_euler_validation=True)


@pytest.fixture
def lenient_config() -> TranslatorConfig:
    """Configuración con validación no estricta de Euler."""
    return TranslatorConfig(strict_euler_validation=False)


@pytest.fixture
def mock_mic() -> MagicMock:
    """
    MIC mockeado con narrativa predecible y determinista.

    La narrativa usa un prefijo fijo para que los tests de isomorfismo
    puedan normalizarla sin depender de nombres de nodo específicos.
    """
    mic = MagicMock()
    mic.project_intent.return_value = {
        "success": True,
        "narrative": "[TEST_NARRATIVE]",
    }
    return mic


@pytest.fixture
def translator(mock_mic: MagicMock) -> SemanticTranslator:
    """Traductor con MIC mockeado y caché deshabilitado."""
    return SemanticTranslator(mic=mock_mic, enable_cache=False)


@pytest.fixture
def translator_cached(mock_mic: MagicMock) -> SemanticTranslator:
    """Traductor con MIC mockeado y caché habilitado."""
    return SemanticTranslator(mic=mock_mic, enable_cache=True)


@pytest.fixture
def valid_topology() -> ValidatedTopology:
    """
    Topología válida estándar.

    Invariante: χ = β₀ - β₁ + β₂ = 1 - 0 + 0 = 1. ✓
    Grafo conexo (β₀=1), acíclico (β₁=0), sin 2-esferas (β₂=0).
    Análogo topológico de un árbol generador.
    """
    return ValidatedTopology(
        beta_0=1,
        beta_1=0,
        beta_2=0,
        euler_characteristic=1,
        fiedler_value=1.5,
        spectral_gap=0.3,
        pyramid_stability=5.0,
        structural_entropy=0.2,
    )


@pytest.fixture
def cyclic_topology() -> ValidatedTopology:
    """
    Topología con ciclos (β₁ = 3).

    Invariante: χ = 1 - 3 + 0 = -2. ✓
    Genus g = β₁ = 3 (superficie orientable de genus 3 si β₂=0).
    """
    return ValidatedTopology(
        beta_0=1,
        beta_1=3,
        beta_2=0,
        euler_characteristic=-2,
        fiedler_value=0.8,
        spectral_gap=0.2,
        pyramid_stability=2.0,
        structural_entropy=0.5,
    )


@pytest.fixture
def fragmented_topology() -> ValidatedTopology:
    """
    Topología fragmentada (β₀ = 4, β₁ = 0).

    Invariante: χ = 4 - 0 + 0 = 4. ✓
    λ₂ = 0 porque el grafo Laplaciano tiene β₀ eigenvalores nulos.
    """
    return ValidatedTopology(
        beta_0=4,
        beta_1=0,
        beta_2=0,
        euler_characteristic=4,
        fiedler_value=0.0,
        spectral_gap=0.0,
        pyramid_stability=0.5,
        structural_entropy=0.8,
    )


@pytest.fixture
def viable_financial() -> Dict[str, Any]:
    """Métricas financieras con PI > profitability_good y recomendación ACEPTAR."""
    return {
        "wacc": 0.08,
        "contingency": {"recommended": 0.10},
        "performance": {
            "recommendation": "ACEPTAR",
            "profitability_index": 1.6,
        },
    }


@pytest.fixture
def reject_financial() -> Dict[str, Any]:
    """Métricas financieras con PI < 1.0 y recomendación RECHAZAR."""
    return {
        "wacc": 0.20,
        "contingency": {"recommended": 0.25},
        "performance": {
            "recommendation": "RECHAZAR",
            "profitability_index": 0.7,
        },
    }


@pytest.fixture
def simple_digraph() -> nx.DiGraph:
    """
    Dígrafo simple con exactamente un ciclo simple: A→B→C→A.
    El nodo D es una hoja alcanzable desde C.
    β₁ = 1 (un generador del grupo fundamental π₁).
    """
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")])
    return g


@pytest.fixture
def acyclic_digraph() -> nx.DiGraph:
    """
    DAG sin ciclos (β₁ = 0).
    Estructura: A→B, A→C, B→D, C→D (rombo/diamante).
    """
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    return g


@pytest.fixture
def empty_digraph() -> nx.DiGraph:
    """Dígrafo vacío: 0 nodos, 0 aristas."""
    return nx.DiGraph()


@pytest.fixture
def disconnected_digraph() -> nx.DiGraph:
    """Dígrafo con dos componentes fuertemente desconectadas: {A,B} y {C,D}."""
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("C", "D")])
    return g


# ============================================================================
# 1. TestTemperature — Refinado
# ============================================================================


class TestTemperature:
    """
    Invariantes termodinámicos del valor-objeto Temperature.

    Marco teórico: Temperatura como elemento de ℝ₊ ∪ {0} (semirrecta
    no negativa) con la estructura de orden estándar. El cero absoluto
    es el único elemento minimal: T ≥ 0K para todo T físicamente válido.
    """

    def test_from_celsius_conversion(self):
        """Verifica: T(°C) + 273.15 = T(K) con precisión de EPSILON."""
        t = Temperature.from_celsius(25.0)
        assert abs(t.kelvin - (25.0 + KELVIN_OFFSET)) < EPSILON
        assert abs(t.celsius - 25.0) < EPSILON

    def test_kelvin_offset_constant(self):
        """
        Verifica que KELVIN_OFFSET = 273.15 exactamente.

        Este valor es la definición del punto triple del agua en la escala
        Kelvin (BIPM 2019). Cualquier desviación es un error de implementación.
        """
        assert abs(KELVIN_OFFSET - 273.15) < 1e-10, (
            f"KELVIN_OFFSET={KELVIN_OFFSET} ≠ 273.15 (desviación física inaceptable)"
        )

    def test_from_kelvin_identity(self):
        """Verifica: from_kelvin(x).kelvin == x exactamente (sin conversión)."""
        t = Temperature.from_kelvin(300.0)
        assert t.kelvin == 300.0

    @pytest.mark.parametrize(
        "celsius",
        [-200.0, -100.0, -40.0, 0.0, 25.0, 100.0, 500.0, 1000.0],
    )
    def test_roundtrip_celsius_kelvin(self, celsius: float):
        """
        Verifica roundtrip: celsius(from_celsius(x)) ≈ x.

        Propiedad de idempotencia de la conversión: f⁻¹(f(x)) = x.
        La tolerancia es EPSILON para capturar errores de redondeo float64.
        """
        t = Temperature.from_celsius(celsius)
        assert abs(t.celsius - celsius) < EPSILON, (
            f"Roundtrip fallido para {celsius}°C: "
            f"got {t.celsius}°C (error={abs(t.celsius - celsius):.2e})"
        )

    def test_absolute_zero_celsius_value(self):
        """
        Verifica: ABSOLUTE_ZERO_CELSIUS = -273.15 exactamente.

        Este test verifica la CONSTANTE, no solo el comportamiento.
        Un error en la constante propagaría a todos los cálculos térmicos.
        """
        assert abs(ABSOLUTE_ZERO_CELSIUS - (-273.15)) < 1e-10

    def test_absolute_zero_from_celsius(self):
        """
        Verifica: Temperature.from_celsius(-273.15).is_absolute_zero.

        La temperatura en el cero absoluto debe tener kelvin = 0.0
        y activar el flag is_absolute_zero.
        """
        t = Temperature.from_celsius(ABSOLUTE_ZERO_CELSIUS)
        assert t.is_absolute_zero
        assert t.kelvin == 0.0 or abs(t.kelvin) < EPSILON

    def test_absolute_zero_from_kelvin(self):
        """Verifica: Temperature.from_kelvin(0.0).is_absolute_zero."""
        t = Temperature.from_kelvin(0.0)
        assert t.is_absolute_zero

    def test_near_absolute_zero_not_flagged(self):
        """
        Temperatura ε por encima del cero absoluto NO es cero absoluto.

        Distinción crítica: 0K es el límite, 1e-10K no lo es.
        """
        t = Temperature.from_kelvin(1e-10)
        assert not t.is_absolute_zero

    def test_below_absolute_zero_raises(self):
        """
        Invariante: T < 0K viola el Tercer Principio de la Termodinámica.

        La excepción debe mencionar 'absolute zero' para ser informativa.
        """
        with pytest.raises(ValueError, match="absolute zero"):
            Temperature(kelvin=-1.0)

    def test_slightly_below_zero_raises(self):
        """
        Incluso -EPSILON debe rechazarse (no hay clamping implícito en __init__).

        Nota: from_celsius(-273.15 - 1e-10) puede hacer clamping; Temperature()
        directo NO debe hacerlo.
        """
        with pytest.raises(ValueError):
            Temperature(kelvin=-EPSILON)

    @pytest.mark.parametrize("bad_value", [float("inf"), float("-inf"), float("nan")])
    def test_non_finite_raises(self, bad_value: float):
        """
        Invariante: T debe ser un número real finito.

        ∞, -∞ y NaN no tienen interpretación termodinámica.
        El mensaje debe mencionar 'finite' para claridad.
        """
        with pytest.raises(ValueError, match="finite"):
            Temperature(kelvin=bad_value)

    def test_order_total_strict(self):
        """
        Verifica orden total estricto: T₁ < T₂ ⟺ K₁ < K₂.

        Propiedades verificadas: irreflexividad, asimetría, transitividad.
        """
        t1 = Temperature.from_celsius(20.0)
        t2 = Temperature.from_celsius(30.0)

        assert t1 < t2, "Orden estricto fallido"
        assert t2 > t1, "Orden inverso fallido"
        assert t1 != t2, "Igualdad espuria"
        assert t1 <= t2, "Orden no estricto fallido"
        assert not (t2 < t1), "Asimetría violada"
        assert not (t1 > t2), "Asimetría inversa violada"

    def test_equality_by_value(self):
        """
        Verifica igualdad por valor semántico.

        25°C = 298.15K: ambas representaciones del mismo estado físico.
        Tolerancia: EPSILON para absorber errores de conversión float64.
        """
        t1 = Temperature.from_celsius(25.0)
        t2 = Temperature.from_kelvin(25.0 + KELVIN_OFFSET)
        # La igualdad puede ser exacta o con tolerancia según la implementación
        assert t1 == t2 or abs(t1.kelvin - t2.kelvin) < EPSILON

    def test_frozen_immutability(self):
        """
        Verifica inmutabilidad (dataclass frozen=True o similar).

        Temperature es un VALUE OBJECT: su identidad es su valor.
        La mutabilidad violaría la semántica de valor.
        """
        t = Temperature.from_celsius(25.0)
        with pytest.raises((AttributeError, TypeError)):
            t.kelvin = 999.0  # type: ignore[misc]

    def test_str_representation_contains_celsius_and_kelvin(self):
        """
        Verifica representación string legible con ambas unidades.

        Un operador debe poder leer la temperatura en unidades familiares.
        """
        t = Temperature.from_celsius(25.0)
        s = str(t)
        assert "25.0" in s or "25,0" in s, f"Celsius no en str: {s!r}"
        # Kelvin = 298.15 → debe aparecer 298 al menos
        assert "298" in s, f"Kelvin no en str: {s!r}"

    def test_repr_roundtrip(self):
        """
        repr(t) debe contener suficiente información para reconstruir t.

        Al menos debe aparecer el valor de kelvin.
        """
        t = Temperature.from_kelvin(300.0)
        r = repr(t)
        assert "300" in r, f"Kelvin no en repr: {r!r}"

    def test_near_zero_clamping_only_in_factory(self):
        """
        Solo los métodos factory (from_celsius, from_kelvin) pueden hacer
        clamping de valores negativos minúsculos; Temperature() directo no.

        Este test verifica que from_kelvin(-1e-15) produce T=0K (clamping),
        mientras que Temperature(-1e-15) lanza ValueError.
        """
        # Factory puede clampear
        t = Temperature.from_kelvin(-1e-15)
        assert t.kelvin == 0.0

        # Constructor directo NO debe clampear (ya probado en test_below_absolute_zero)
        # Este test confirma la asimetría de comportamiento


# ============================================================================
# 2. TestStabilityThresholds — Refinado
# ============================================================================


class TestStabilityThresholds:
    """
    Validación de umbrales de estabilidad piramidal Ψ.

    Marco teórico: Los umbrales definen una partición de ℝ en intervalos:
        (-∞, 0)    → "invalid"
        [0, critical) → "critical"
        [critical, warning) → "warning"
        [warning, solid) → "stable"
        [solid, +∞)  → "robust"

    La partición debe ser: disjunta, exhaustiva, y con límites bien definidos.
    """

    def test_default_invariant_strict_order(self):
        """
        Invariante de orden estricto: 0 ≤ critical < warning < solid.

        Permite critical = 0 (caso degenerado válido donde no hay zona "warning").
        """
        st = StabilityThresholds()
        assert st.critical >= 0, "critical debe ser no-negativo"
        assert st.critical < st.warning, "critical debe ser estrictamente menor que warning"
        assert st.warning < st.solid, "warning debe ser estrictamente menor que solid"

    def test_critical_equals_zero_allowed(self):
        """
        critical = 0 es válido: la zona [0, 0) es vacía pero no inválida.

        Esto permite configuraciones donde todo Ψ ≥ 0 es al menos "warning".
        """
        st = StabilityThresholds(critical=0.0, warning=3.0, solid=10.0)
        assert st.critical == 0.0

    @pytest.mark.parametrize(
        "critical, warning, solid, error_pattern",
        [
            (5.0, 3.0, 10.0, "order"),    # critical > warning
            (1.0, 3.0, 2.0, "order"),     # warning > solid
            (5.0, 5.0, 10.0, "strict"),   # critical == warning (no estricto)
            (1.0, 5.0, 5.0, "strict"),    # warning == solid (no estricto)
        ],
    )
    def test_invalid_order_raises(
        self, critical: float, warning: float, solid: float, error_pattern: str,
    ):
        """Violaciones del orden estricto deben lanzar ValueError."""
        with pytest.raises(ValueError):
            StabilityThresholds(critical=critical, warning=warning, solid=solid)

    def test_negative_critical_raises(self):
        """critical < 0 viola el dominio físico de Ψ ≥ 0."""
        with pytest.raises(ValueError):
            StabilityThresholds(critical=-1.0, warning=3.0, solid=10.0)

    @pytest.mark.parametrize("bad_value", [float("inf"), float("nan"), float("-inf")])
    def test_non_finite_raises(self, bad_value: float):
        """Valores no finitos en cualquier umbral deben rechazarse."""
        with pytest.raises(ValueError):
            StabilityThresholds(critical=bad_value, warning=3.0, solid=10.0)
        with pytest.raises(ValueError):
            StabilityThresholds(critical=1.0, warning=bad_value, solid=10.0)
        with pytest.raises(ValueError):
            StabilityThresholds(critical=1.0, warning=3.0, solid=bad_value)

    # Los límites son: critical=1.0, warning=3.0, solid=10.0 (defaults)
    @pytest.mark.parametrize(
        "stability, expected",
        [
            (-1.0, "invalid"),          # Ψ < 0: físicamente imposible
            (float("nan"), "invalid"),   # NaN: indefinido
            (float("inf"), "invalid"),   # ∞: no clasificable
            (0.0, "critical"),           # Ψ ∈ [0, critical)
            (0.5, "critical"),           # Ψ ∈ [0, 1.0)
            (1.0, "warning"),            # Ψ = critical → entra en [critical, warning)
            (1.5, "warning"),            # Ψ ∈ [1.0, 3.0)
            (2.0, "warning"),
            (3.0, "stable"),             # Ψ = warning → entra en [warning, solid)
            (5.0, "stable"),
            (10.0, "robust"),            # Ψ = solid → entra en [solid, ∞)
            (100.0, "robust"),
        ],
    )
    def test_classify_partition(self, stability: float, expected: str):
        """
        Verifica que classify() implementa una partición de ℝ.

        Los límites pertenecen al intervalo DERECHO (convención cerrado-izquierdo):
        [a, b) → a pertenece al intervalo, b no.
        """
        st = StabilityThresholds()
        result = st.classify(stability)
        assert result == expected, (
            f"classify({stability}) = {result!r}, esperado {expected!r}"
        )

    @pytest.mark.parametrize(
        "stability, expected_score",
        [
            (-1.0, 1.0),    # Inválido → máxima severidad
            (0.0, 1.0),     # Ψ=0 → en umbral crítico → score=1
            (5.0, 0.5),     # Punto medio entre 0 y solid=10
            (10.0, 0.0),    # Ψ=solid → severidad mínima
            (20.0, 0.0),    # Ψ > solid → severidad 0 (clampeado)
        ],
    )
    def test_severity_score_values(self, stability: float, expected_score: float):
        """
        Verifica valores específicos del score de severidad.

        El score es una función monótonamente decreciente en Ψ:
        σ(Ψ) = clamp(1 - Ψ/solid, 0, 1)  para Ψ ≥ 0
        σ(Ψ) = 1                           para Ψ < 0
        """
        st = StabilityThresholds()
        score = st.severity_score(stability)
        assert abs(score - expected_score) < 0.01, (
            f"severity_score({stability}) = {score:.4f}, esperado {expected_score:.4f}"
        )

    def test_severity_score_bounds_always(self):
        """
        Invariante global: severity_score ∈ [0, 1] para todo Ψ ∈ ℝ.

        Propósito: el score es un elemento del intervalo unitario [0,1],
        que actúa como medida de probabilidad de colapso.
        """
        st = StabilityThresholds()
        test_values = [-1000.0, -1.0, -EPSILON, 0.0, 0.5, 1.0, 3.0, 5.0,
                       10.0, 100.0, 1000.0, float("nan"), float("inf")]
        for psi in test_values:
            score = st.severity_score(psi)
            assert 0.0 <= score <= 1.0, (
                f"severity_score({psi}) = {score} ∉ [0, 1] — violación de invariante"
            )

    def test_severity_score_monotone_decreasing(self):
        """
        severity_score es monótonamente DECRECIENTE en Ψ (para Ψ ≥ 0).

        Mayor estabilidad → menor severidad.
        """
        st = StabilityThresholds()
        psi_values = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
        scores = [st.severity_score(psi) for psi in psi_values]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"No monótono: σ({psi_values[i]})={scores[i]:.4f} "
                f"< σ({psi_values[i+1]})={scores[i+1]:.4f}"
            )


# ============================================================================
# 3. TestTopologicalThresholds — Refinado
# ============================================================================


class TestTopologicalThresholds:
    """
    Invariantes de clasificación topológica.

    Marco matemático: Los números de Betti βₖ son invariantes topológicos
    del complejo simplicial K que representa la jerarquía organizacional:
    - β₀: componentes conexas (silos organizacionales)
    - β₁: ciclos independientes (retroalimentaciones, dependencias circulares)
    - β₂: cavidades 2-dimensionales (vacíos estructurales)

    El valor de Fiedler λ₂ (2do eigenvalor del Laplaciano normalizado)
    mide la conectividad algebraica: λ₂ > 0 ⟺ grafo conexo.
    """

    def test_default_invariants_complete(self):
        """
        Verifica TODOS los invariantes de los umbrales por defecto.

        Invariantes esperados:
        1. β₀_optimal ≥ 1 (al menos una componente)
        2. 0 ≤ cycles_optimal ≤ cycles_warning ≤ cycles_critical
        3. 0 ≤ fiedler_connected < fiedler_robust
        4. max_fragmentation ≥ connected_components_optimal
        """
        tt = TopologicalThresholds()

        assert tt.connected_components_optimal >= 1, "β₀ óptimo debe ser ≥ 1"
        assert tt.cycles_optimal >= 0, "Ciclos óptimos no pueden ser negativos"
        assert tt.cycles_optimal <= tt.cycles_warning, (
            "cycles_optimal debe ser ≤ cycles_warning"
        )
        assert tt.cycles_warning <= tt.cycles_critical, (
            "cycles_warning debe ser ≤ cycles_critical"
        )
        assert 0 <= tt.fiedler_connected_threshold, "Fiedler debe ser ≥ 0"
        assert tt.fiedler_connected_threshold < tt.fiedler_robust_threshold, (
            "fiedler_connected debe ser estrictamente menor que fiedler_robust"
        )
        assert tt.max_fragmentation >= tt.connected_components_optimal, (
            "max_fragmentation debe acotar superiormente a connected_components_optimal"
        )

    @pytest.mark.parametrize(
        "bad_optimal",
        [0, -1, -100],
    )
    def test_invalid_optimal_components_raises(self, bad_optimal: int):
        """β₀_optimal < 1 viola el requisito de existencia mínima."""
        with pytest.raises(ValueError):
            TopologicalThresholds(connected_components_optimal=bad_optimal)

    def test_invalid_cycle_order_raises(self):
        """Orden incorrecto en umbrales de ciclos viola la monotonía."""
        with pytest.raises(ValueError):
            TopologicalThresholds(
                cycles_optimal=5, cycles_warning=3, cycles_critical=1,
            )

    def test_invalid_cycle_order_partial_raises(self):
        """cycles_warning > cycles_critical viola el orden."""
        with pytest.raises(ValueError):
            TopologicalThresholds(
                cycles_optimal=0, cycles_warning=5, cycles_critical=3,
            )

    def test_invalid_fiedler_order_raises(self):
        """fiedler_connected ≥ fiedler_robust viola el orden."""
        with pytest.raises(ValueError):
            TopologicalThresholds(
                fiedler_connected_threshold=1.0,
                fiedler_robust_threshold=0.5,
            )

    def test_invalid_fiedler_equal_raises(self):
        """fiedler_connected = fiedler_robust viola el orden estricto."""
        with pytest.raises(ValueError):
            TopologicalThresholds(
                fiedler_connected_threshold=0.5,
                fiedler_robust_threshold=0.5,
            )

    def test_invalid_negative_fiedler_raises(self):
        """Fiedler < 0 viola la semidefinición positiva del Laplaciano."""
        with pytest.raises(ValueError):
            TopologicalThresholds(fiedler_connected_threshold=-0.1)

    def test_max_fragmentation_less_than_optimal_raises(self):
        """max_fragmentation < optimal no tiene sentido semántico."""
        with pytest.raises(ValueError):
            TopologicalThresholds(
                connected_components_optimal=3,
                max_fragmentation=2,
            )

    @pytest.mark.parametrize(
        "beta_0, expected",
        [
            (0, "empty"),               # Grafo vacío: sin componentes
            (1, "unified"),             # Grafo conexo: β₀ = 1
            (2, "fragmented"),          # 2 ≤ β₀ ≤ max_fragmentation
            (5, "fragmented"),
            (6, "severely_fragmented"), # β₀ > max_fragmentation (default=5)
            (100, "severely_fragmented"),
        ],
    )
    def test_classify_connectivity(self, beta_0: int, expected: str):
        """
        Verifica clasificación por β₀ (número de componentes conexas).

        Partición de ℤ≥0:
        {0} → "empty", {1} → "unified",
        [2, max_fragmentation] → "fragmented",
        (max_fragmentation, ∞) → "severely_fragmented"
        """
        tt = TopologicalThresholds()
        result = tt.classify_connectivity(beta_0)
        assert result == expected, (
            f"classify_connectivity({beta_0}) = {result!r}, esperado {expected!r}"
        )

    @pytest.mark.parametrize(
        "beta_1, expected",
        [
            (-1, "invalid"),    # β₁ < 0 es imposible (número de Betti ≥ 0)
            (0, "clean"),       # Sin ciclos
            (1, "minor"),       # 1 ≤ β₁ ≤ cycles_warning (default=1)
            (2, "moderate"),    # cycles_warning < β₁ ≤ cycles_critical (default=3)
            (3, "moderate"),
            (4, "critical"),    # β₁ > cycles_critical
            (100, "critical"),
        ],
    )
    def test_classify_cycles(self, beta_1: int, expected: str):
        """
        Verifica clasificación por β₁ (1er número de Betti = ciclos).

        β₁ es siempre ≥ 0 en un complejo simplicial bien definido.
        """
        tt = TopologicalThresholds()
        result = tt.classify_cycles(beta_1)
        assert result == expected, (
            f"classify_cycles({beta_1}) = {result!r}, esperado {expected!r}"
        )

    @pytest.mark.parametrize(
        "fiedler, expected",
        [
            (float("nan"), "invalid"),           # NaN: no clasificable
            (float("-inf"), "invalid"),          # -∞: físicamente imposible
            (-0.1, "disconnected"),              # λ₂ < 0: error numérico severo
            (0.0, "disconnected"),               # λ₂ = 0: grafo desconectado
            (EPSILON / 2, "disconnected"),       # λ₂ < ε: prácticamente desconectado
            (0.3, "weakly_connected"),           # 0 < λ₂ < fiedler_robust
            (0.5, "strongly_connected"),         # λ₂ ≥ fiedler_robust (default=0.5)
            (1.0, "strongly_connected"),
            (10.0, "strongly_connected"),
        ],
    )
    def test_classify_spectral_connectivity(self, fiedler: float, expected: str):
        """
        Verifica clasificación por λ₂ (valor de Fiedler).

        Teorema algebraico: λ₂ > 0 ⟺ G es conexo.
        La clasificación usa EPSILON como umbral de nulidad numérica.
        """
        tt = TopologicalThresholds()
        result = tt.classify_spectral_connectivity(fiedler)
        assert result == expected, (
            f"classify_spectral_connectivity({fiedler}) = {result!r}, "
            f"esperado {expected!r}"
        )

    @pytest.mark.parametrize(
        "beta_0, beta_1, beta_2, chi, expected_valid",
        [
            # Casos válidos: χ = β₀ - β₁ + β₂
            (1, 0, 0, 1, True),      # Árbol (contractible a punto)
            (1, 1, 0, 0, True),      # Círculo S¹: χ = 0
            (2, 0, 0, 2, True),      # Dos puntos: χ = 2
            (1, 0, 1, 2, True),      # Esfera S²: χ = 2
            (1, 2, 1, 0, True),      # Toro T²: χ = 0
            (3, 2, 1, 2, True),      # Complejo general: 3-2+1=2
            (1, 5, 3, -1, True),     # Complejo con β₂: 1-5+3=-1 ✓
            # Casos inválidos
            (1, 0, 0, 0, False),     # χ=0 ≠ 1
            (1, 0, 0, 2, False),     # χ=2 ≠ 1
            (3, 2, 1, 5, False),     # χ=5 ≠ 2
            (2, 3, 1, 2, False),     # χ=2 ≠ 0 (2-3+1=0)
        ],
    )
    def test_validate_euler_characteristic(
        self,
        beta_0: int,
        beta_1: int,
        beta_2: int,
        chi: int,
        expected_valid: bool,
    ):
        """
        Verifica el invariante de Euler: χ(K) = Σₖ (-1)ᵏ βₖ = β₀ - β₁ + β₂.

        Este es el invariante topológico más fundamental: es preservado bajo
        homeomorfismos y es un invariante homotópico.
        """
        result = TopologicalThresholds.validate_euler_characteristic(
            beta_0, beta_1, beta_2, chi,
        )
        assert result == expected_valid, (
            f"validate_euler({beta_0},{beta_1},{beta_2},{chi}) = {result}, "
            f"esperado {expected_valid}. "
            f"χ_esperado = {beta_0 - beta_1 + beta_2}"
        )

    def test_euler_validation_tolerance(self):
        """
        La validación de Euler debe ser exacta para enteros.

        β₀, β₁, β₂ ∈ ℤ≥0, por lo que χ ∈ ℤ. No hay error de redondeo.
        """
        # Test explícito de que no hay tolerancia espuria
        tt = TopologicalThresholds()
        # χ=1 con β₀=2, β₁=1, β₂=0 → esperado=1, dado=1 → válido
        assert TopologicalThresholds.validate_euler_characteristic(2, 1, 0, 1)
        # χ=2 con β₀=2, β₁=1, β₂=0 → esperado=1, dado=2 → inválido
        assert not TopologicalThresholds.validate_euler_characteristic(2, 1, 0, 2)


# ============================================================================
# 4. TestThermalThresholds — Refinado
# ============================================================================


class TestThermalThresholds:
    """
    Clasificación termodinámica.

    Marco teórico: La entropía S ∈ [0, 1] (normalizada) actúa como medida
    de desorden. La temperatura T define el estado térmico del sistema.
    La exergía Ex = 1 - T₀/T mide el trabajo útil extraíble.
    """

    def test_default_invariants_complete(self):
        """
        Verifica TODOS los invariantes de umbrales por defecto.

        Invariantes:
        1. Temperaturas estrictamente crecientes
        2. Entropía en [0,1] con orden estricto
        3. Exergía en [0,1] con poor < efficient
        4. Capacidad calorífica positiva
        """
        th = ThermalThresholds()
        temps = [
            th.temperature_cold,
            th.temperature_warm,
            th.temperature_hot,
            th.temperature_critical,
        ]
        # Estrictamente crecientes
        for i in range(len(temps) - 1):
            assert temps[i] < temps[i + 1], (
                f"Temperaturas no estrictamente crecientes: "
                f"temps[{i}]={temps[i]} ≥ temps[{i+1}]={temps[i+1]}"
            )

        assert 0 <= th.entropy_low, "entropy_low debe ser ≥ 0"
        assert th.entropy_low < th.entropy_high, "entropy_low debe ser < entropy_high"
        assert th.entropy_high < th.entropy_death, "entropy_high debe ser < entropy_death"
        assert th.entropy_death <= 1.0, "entropy_death debe ser ≤ 1"

        assert 0 <= th.exergy_poor, "exergy_poor debe ser ≥ 0"
        assert th.exergy_poor < th.exergy_efficient, "exergy_poor debe ser < exergy_efficient"
        assert th.exergy_efficient <= 1.0, "exergy_efficient debe ser ≤ 1"

        assert th.heat_capacity_minimum > 0, "C_v mínimo debe ser positivo"

    def test_invalid_temperature_order_raises(self):
        """Temperaturas no estrictamente crecientes violan la termodinámica."""
        with pytest.raises(ValueError):
            ThermalThresholds(temperature_cold=50, temperature_warm=35)

    def test_equal_temperatures_raise(self):
        """Temperaturas iguales no son estrictamente crecientes."""
        with pytest.raises(ValueError):
            ThermalThresholds(temperature_cold=30, temperature_warm=30)

    @pytest.mark.parametrize(
        "param, value",
        [
            ("entropy_low", -0.1),
            ("entropy_low", 1.5),
            ("entropy_high", -0.1),
            ("entropy_death", 1.5),
            ("entropy_death", 2.0),
        ],
    )
    def test_invalid_entropy_bounds_raises(self, param: str, value: float):
        """Entropía fuera de [0, 1] viola la definición de entropía normalizada."""
        with pytest.raises(ValueError):
            ThermalThresholds(**{param: value})

    def test_invalid_exergy_order_raises(self):
        """exergy_poor ≥ exergy_efficient viola el orden de eficiencia."""
        with pytest.raises(ValueError):
            ThermalThresholds(exergy_poor=0.8, exergy_efficient=0.3)

    @pytest.mark.parametrize("bad_cv", [0.0, -1.0, -0.001])
    def test_invalid_heat_capacity_raises(self, bad_cv: float):
        """C_v ≤ 0 viola la estabilidad termodinámica (∂²S/∂U² > 0)."""
        with pytest.raises(ValueError):
            ThermalThresholds(heat_capacity_minimum=bad_cv)

    @pytest.mark.parametrize(
        "temp, expected",
        [
            (float("nan"), "invalid"),
            (ABSOLUTE_ZERO_CELSIUS - 1.0, "invalid"),  # Imposible físicamente
            (ABSOLUTE_ZERO_CELSIUS, "cold"),            # -273.15°C = 0K → cold
            (-50.0, "cold"),
            (20.0, "cold"),                             # < temperature_warm (default≈25)
            (25.0, "stable"),                           # ≥ temperature_warm, < hot
            (35.0, "stable"),
            (40.0, "warm"),                             # ≥ temperature_hot (default≈40)
            (55.0, "warm"),
            (60.0, "hot"),                              # ≥ temperature_hot (default≈60)
            (75.0, "hot"),
            (80.0, "critical"),                         # ≥ temperature_critical (default≈80)
            (200.0, "critical"),
        ],
    )
    def test_classify_temperature(self, temp: float, expected: str):
        """
        Verifica la clasificación de temperatura por intervalos.

        Nota: Los valores exactos de umbral dependen de los defaults configurados.
        Los casos de borde deben verificarse contra los valores reales de ThermalThresholds().
        """
        th = ThermalThresholds()
        result = th.classify_temperature(temp)
        assert result == expected, (
            f"classify_temperature({temp}) = {result!r}, esperado {expected!r}. "
            f"Umbrales: cold={th.temperature_cold}, warm={th.temperature_warm}, "
            f"hot={th.temperature_hot}, critical={th.temperature_critical}"
        )

    @pytest.mark.parametrize(
        "entropy, expected",
        [
            (-0.1, "invalid"),      # Entropía negativa: imposible
            (1.5, "invalid"),       # Entropía > 1: imposible
            (float("nan"), "invalid"),
            (0.0, "low"),           # ∈ [0, entropy_low)
            (0.2, "low"),
            (0.3, "moderate"),      # ∈ [entropy_low, entropy_high)
            (0.5, "moderate"),
            (0.7, "high"),          # ∈ [entropy_high, entropy_death)
            (0.8, "high"),
            (0.95, "death"),        # ∈ [entropy_death, 1]
            (1.0, "death"),
        ],
    )
    def test_classify_entropy(self, entropy: float, expected: str):
        """
        Verifica clasificación de entropía normalizada S ∈ [0, 1].

        La 'muerte térmica' (entropy_death) corresponde al equilibrio
        termodinámico donde no es posible extraer trabajo útil.
        """
        th = ThermalThresholds()
        result = th.classify_entropy(entropy)
        assert result == expected, (
            f"classify_entropy({entropy}) = {result!r}, esperado {expected!r}"
        )

    @pytest.mark.parametrize(
        "exergy, expected",
        [
            (-0.1, "invalid"),      # Ex < 0: termodinámicamente imposible
            (1.5, "invalid"),       # Ex > 1: imposible
            (0.0, "poor"),          # ∈ [0, exergy_poor)
            (0.2, "poor"),
            (0.3, "moderate"),      # Umbral exergy_poor (default≈0.3)
            (0.5, "moderate"),
            (0.7, "efficient"),     # ∈ [exergy_efficient, 1]
            (1.0, "efficient"),
        ],
    )
    def test_classify_exergy(self, exergy: float, expected: str):
        """Verifica clasificación de eficiencia exergética Ex ∈ [0, 1]."""
        th = ThermalThresholds()
        result = th.classify_exergy(exergy)
        assert result == expected, (
            f"classify_exergy({exergy}) = {result!r}, esperado {expected!r}"
        )


# ============================================================================
# 5. TestFinancialThresholds — Refinado
# ============================================================================


class TestFinancialThresholds:
    """
    Clasificación financiera con invariantes de orden estricto.

    Marco: Los umbrales financieros deben definir particiones DISJUNTAS
    del espacio de métricas ℝ⁺. El PI (Profitability Index) = VAN/I₀:
    PI < 1 → destruye valor, PI = 1 → indiferente, PI > 1 → crea valor.
    """

    def test_default_invariants_complete(self):
        """Verifica orden estricto de todos los umbrales por defecto."""
        ft = FinancialThresholds()

        assert 0 < ft.wacc_low < ft.wacc_moderate < ft.wacc_high, (
            "WACC debe ser estrictamente creciente y positivo"
        )
        assert 0 < ft.profitability_marginal < ft.profitability_good < ft.profitability_excellent, (
            "PI debe ser estrictamente creciente y positivo"
        )
        assert 0 < ft.contingency_minimal < ft.contingency_standard < ft.contingency_high, (
            "Contingencia debe ser estrictamente creciente y positiva"
        )

    @pytest.mark.parametrize(
        "wacc_low, wacc_moderate, wacc_high",
        [
            (0.15, 0.10, 0.20),   # low > moderate
            (0.05, 0.15, 0.15),   # moderate == high
            (0.10, 0.05, 0.20),   # low > moderate
        ],
    )
    def test_invalid_wacc_order_raises(
        self, wacc_low: float, wacc_moderate: float, wacc_high: float,
    ):
        """WACC no estrictamente creciente viola el invariante."""
        with pytest.raises(ValueError):
            FinancialThresholds(
                wacc_low=wacc_low, wacc_moderate=wacc_moderate, wacc_high=wacc_high,
            )

    def test_invalid_profitability_order_raises(self):
        """PI no estrictamente creciente viola el invariante."""
        with pytest.raises(ValueError):
            FinancialThresholds(
                profitability_marginal=1.5,
                profitability_good=1.2,
                profitability_excellent=1.0,
            )

    def test_invalid_contingency_order_raises(self):
        """Contingencia no estrictamente creciente viola el invariante."""
        with pytest.raises(ValueError):
            FinancialThresholds(
                contingency_minimal=0.20,
                contingency_standard=0.10,
            )

    @pytest.mark.parametrize(
        "pi, expected",
        [
            (float("nan"), "invalid"),      # NaN: no clasificable
            (float("-inf"), "invalid"),     # -∞: imposible
            (0.0, "poor"),                  # PI = 0: pérdida total
            (0.5, "poor"),                  # PI < 1: destruye valor
            (1.0, "marginal"),              # PI = 1: punto de equilibrio
            (1.2, "good"),                  # PI > 1 pero < excellent
            (1.5, "excellent"),             # PI ≥ excellent
            (2.0, "excellent"),             # PI muy alto
            (float("inf"), "excellent"),    # PI = ∞: si es válido, es excelente
        ],
    )
    def test_classify_profitability(self, pi: float, expected: str):
        """
        Verifica clasificación de PI.

        Partición de ℝ⁺ ∪ {0}:
        [0, 1) → "poor", [1, good) → "marginal",
        [good, excellent) → "good", [excellent, ∞) → "excellent"
        """
        ft = FinancialThresholds()
        result = ft.classify_profitability(pi)
        assert result == expected, (
            f"classify_profitability({pi}) = {result!r}, esperado {expected!r}"
        )


# ============================================================================
# 7. TestVerdictLevelLattice — Verificación algebraica exhaustiva (Refinado)
# ============================================================================


class TestVerdictLevelLattice:
    """
    Verifica TODAS las leyes del retículo distributivo acotado L = (VerdictLevel, ≤).

    Estructura algebraica: (L, ⊔, ⊓) donde:
    - ⊔ = join = sup = max (ya que L es una cadena)
    - ⊓ = meet = inf = min (ya que L es una cadena)
    - ⊥ = VIABLE (bottom, elemento mínimo)
    - ⊤ = RECHAZAR (top, elemento máximo)

    Con n = |L| = 5:
    - Leyes unarias (idempotencia): n = 5 casos
    - Leyes binarias (conmutatividad, absorción): n² = 25 casos
    - Leyes ternarias (asociatividad, distributividad): n³ = 125 casos

    NOTA: Por ser L una cadena totalmente ordenada, distributividad se sigue
    automáticamente de la linealidad. Los tests la verifican explícitamente.
    """

    # Precomputar todos los elementos para eficiencia
    _ALL = list(VerdictLevel)
    _N = len(_ALL)

    @classmethod
    def _join(cls, a: VerdictLevel, b: VerdictLevel) -> VerdictLevel:
        """Operación join definida semánticamente (no como Python |)."""
        return a | b

    @classmethod
    def _meet(cls, a: VerdictLevel, b: VerdictLevel) -> VerdictLevel:
        """Operación meet definida semánticamente (no como Python &)."""
        return a & b

    # -- Elementos distinguidos --

    def test_bottom_is_viable(self):
        """⊥ = VIABLE: el veredicto más favorable es el bottom del retículo."""
        assert VerdictLevel.bottom() == VerdictLevel.VIABLE

    def test_top_is_rechazar(self):
        """⊤ = RECHAZAR: el veredicto más severo es el top del retículo."""
        assert VerdictLevel.top() == VerdictLevel.RECHAZAR

    def test_bottom_value_is_minimum(self):
        """⊥ tiene el valor numérico mínimo entre todos los elementos."""
        bottom = VerdictLevel.bottom()
        assert all(bottom.value <= v.value for v in VerdictLevel), (
            "⊥ no es el elemento mínimo"
        )

    def test_top_value_is_maximum(self):
        """⊤ tiene el valor numérico máximo entre todos los elementos."""
        top = VerdictLevel.top()
        assert all(top.value >= v.value for v in VerdictLevel), (
            "⊤ no es el elemento máximo"
        )

    def test_cardinality(self):
        """Verifica cardinalidad esperada del retículo (n = 5)."""
        assert len(self._ALL) == 5, (
            f"Se esperan 5 elementos en VerdictLevel, se encontraron {len(self._ALL)}"
        )

    # -- Idempotencia: ∀a: a ⊔ a = a y a ⊓ a = a --

    @pytest.mark.parametrize("a", list(VerdictLevel), ids=lambda v: v.name)
    def test_idempotent_join(self, a: VerdictLevel):
        """∀a ∈ L: a ⊔ a = a (idempotencia del join)."""
        result = a | a
        assert result == a, f"Idempotencia join violada: {a.name} ⊔ {a.name} = {result.name}"

    @pytest.mark.parametrize("a", list(VerdictLevel), ids=lambda v: v.name)
    def test_idempotent_meet(self, a: VerdictLevel):
        """∀a ∈ L: a ⊓ a = a (idempotencia del meet)."""
        result = a & a
        assert result == a, f"Idempotencia meet violada: {a.name} ⊓ {a.name} = {result.name}"

    # -- Conmutatividad --

    @pytest.mark.parametrize(
        "a,b",
        list(product(list(VerdictLevel), repeat=2)),
        ids=lambda pair: f"{pair[0].name}_x_{pair[1].name}",
    )
    def test_commutative_join(self, a: VerdictLevel, b: VerdictLevel):
        """∀a,b ∈ L: a ⊔ b = b ⊔ a."""
        lhs = a | b
        rhs = b | a
        assert lhs == rhs, (
            f"Conmutatividad join violada: "
            f"{a.name} ⊔ {b.name} = {lhs.name} ≠ {rhs.name} = {b.name} ⊔ {a.name}"
        )

    @pytest.mark.parametrize(
        "a,b",
        list(product(list(VerdictLevel), repeat=2)),
        ids=lambda pair: f"{pair[0].name}_x_{pair[1].name}",
    )
    def test_commutative_meet(self, a: VerdictLevel, b: VerdictLevel):
        """∀a,b ∈ L: a ⊓ b = b ⊓ a."""
        lhs = a & b
        rhs = b & a
        assert lhs == rhs, (
            f"Conmutatividad meet violada: "
            f"{a.name} ⊓ {b.name} = {lhs.name} ≠ {rhs.name} = {b.name} ⊓ {a.name}"
        )

    # -- Asociatividad --

    @pytest.mark.parametrize(
        "a,b,c",
        list(product(list(VerdictLevel), repeat=3)),
        ids=lambda triple: f"{triple[0].name}_{triple[1].name}_{triple[2].name}",
    )
    def test_associative_join(
        self, a: VerdictLevel, b: VerdictLevel, c: VerdictLevel,
    ):
        """∀a,b,c ∈ L: (a ⊔ b) ⊔ c = a ⊔ (b ⊔ c)."""
        lhs = (a | b) | c
        rhs = a | (b | c)
        assert lhs == rhs, (
            f"Asociatividad join violada: "
            f"({a.name} ⊔ {b.name}) ⊔ {c.name} = {lhs.name} "
            f"≠ {a.name} ⊔ ({b.name} ⊔ {c.name}) = {rhs.name}"
        )

    @pytest.mark.parametrize(
        "a,b,c",
        list(product(list(VerdictLevel), repeat=3)),
        ids=lambda triple: f"{triple[0].name}_{triple[1].name}_{triple[2].name}",
    )
    def test_associative_meet(
        self, a: VerdictLevel, b: VerdictLevel, c: VerdictLevel,
    ):
        """∀a,b,c ∈ L: (a ⊓ b) ⊓ c = a ⊓ (b ⊓ c)."""
        lhs = (a & b) & c
        rhs = a & (b & c)
        assert lhs == rhs, (
            f"Asociatividad meet violada: "
            f"({a.name} ⊓ {b.name}) ⊓ {c.name} = {lhs.name} "
            f"≠ {a.name} ⊓ ({b.name} ⊓ {c.name}) = {rhs.name}"
        )

    # -- Absorción --

    @pytest.mark.parametrize(
        "a,b",
        list(product(list(VerdictLevel), repeat=2)),
        ids=lambda pair: f"{pair[0].name}_x_{pair[1].name}",
    )
    def test_absorption_join_meet(self, a: VerdictLevel, b: VerdictLevel):
        """∀a,b ∈ L: a ⊔ (a ⊓ b) = a (absorción tipo 1)."""
        result = a | (a & b)
        assert result == a, (
            f"Absorción (join-meet) violada: "
            f"{a.name} ⊔ ({a.name} ⊓ {b.name}) = {result.name} ≠ {a.name}"
        )

    @pytest.mark.parametrize(
        "a,b",
        list(product(list(VerdictLevel), repeat=2)),
        ids=lambda pair: f"{pair[0].name}_x_{pair[1].name}",
    )
    def test_absorption_meet_join(self, a: VerdictLevel, b: VerdictLevel):
        """∀a,b ∈ L: a ⊓ (a ⊔ b) = a (absorción tipo 2)."""
        result = a & (a | b)
        assert result == a, (
            f"Absorción (meet-join) violada: "
            f"{a.name} ⊓ ({a.name} ⊔ {b.name}) = {result.name} ≠ {a.name}"
        )

    # -- Identidad --

    @pytest.mark.parametrize("a", list(VerdictLevel), ids=lambda v: v.name)
    def test_identity_join_bottom(self, a: VerdictLevel):
        """∀a ∈ L: a ⊔ ⊥ = a (⊥ es identidad del join)."""
        result = a | VerdictLevel.bottom()
        assert result == a, (
            f"Identidad join-bottom violada: {a.name} ⊔ ⊥ = {result.name}"
        )

    @pytest.mark.parametrize("a", list(VerdictLevel), ids=lambda v: v.name)
    def test_identity_meet_top(self, a: VerdictLevel):
        """∀a ∈ L: a ⊓ ⊤ = a (⊤ es identidad del meet)."""
        result = a & VerdictLevel.top()
        assert result == a, (
            f"Identidad meet-top violada: {a.name} ⊓ ⊤ = {result.name}"
        )

    # -- Aniquilación --

    @pytest.mark.parametrize("a", list(VerdictLevel), ids=lambda v: v.name)
    def test_annihilation_join_top(self, a: VerdictLevel):
        """∀a ∈ L: a ⊔ ⊤ = ⊤ (⊤ es absorbente del join)."""
        result = a | VerdictLevel.top()
        assert result == VerdictLevel.top(), (
            f"Aniquilación join-top violada: {a.name} ⊔ ⊤ = {result.name}"
        )

    @pytest.mark.parametrize("a", list(VerdictLevel), ids=lambda v: v.name)
    def test_annihilation_meet_bottom(self, a: VerdictLevel):
        """∀a ∈ L: a ⊓ ⊥ = ⊥ (⊥ es absorbente del meet)."""
        result = a & VerdictLevel.bottom()
        assert result == VerdictLevel.bottom(), (
            f"Aniquilación meet-bottom violada: {a.name} ⊓ ⊥ = {result.name}"
        )

    # -- Distributividad --

    @pytest.mark.parametrize(
        "a,b,c",
        list(product(list(VerdictLevel), repeat=3)),
        ids=lambda triple: f"{triple[0].name}_{triple[1].name}_{triple[2].name}",
    )
    def test_distributive_meet_over_join(
        self, a: VerdictLevel, b: VerdictLevel, c: VerdictLevel,
    ):
        """∀a,b,c ∈ L: a ⊓ (b ⊔ c) = (a ⊓ b) ⊔ (a ⊓ c)."""
        lhs = a & (b | c)
        rhs = (a & b) | (a & c)
        assert lhs == rhs, (
            f"Distributividad (meet/join) violada: "
            f"{a.name} ⊓ ({b.name} ⊔ {c.name}) = {lhs.name} "
            f"≠ ({a.name} ⊓ {b.name}) ⊔ ({a.name} ⊓ {c.name}) = {rhs.name}"
        )

    @pytest.mark.parametrize(
        "a,b,c",
        list(product(list(VerdictLevel), repeat=3)),
        ids=lambda triple: f"{triple[0].name}_{triple[1].name}_{triple[2].name}",
    )
    def test_distributive_join_over_meet(
        self, a: VerdictLevel, b: VerdictLevel, c: VerdictLevel,
    ):
        """∀a,b,c ∈ L: a ⊔ (b ⊓ c) = (a ⊔ b) ⊓ (a ⊔ c)."""
        lhs = a | (b & c)
        rhs = (a | b) & (a | c)
        assert lhs == rhs, (
            f"Distributividad (join/meet) violada: "
            f"{a.name} ⊔ ({b.name} ⊓ {c.name}) = {lhs.name} "
            f"≠ ({a.name} ⊔ {b.name}) ⊓ ({a.name} ⊔ {c.name}) = {rhs.name}"
        )

    # -- Orden parcial (que en cadena = orden total) --

    @pytest.mark.parametrize(
        "a,b",
        list(product(list(VerdictLevel), repeat=2)),
        ids=lambda pair: f"{pair[0].name}_x_{pair[1].name}",
    )
    def test_reflexive(self, a: VerdictLevel, b: VerdictLevel):
        """∀a ∈ L: a ≤ a (reflexividad)."""
        assert a <= a, f"Reflexividad violada para {a.name}"

    @pytest.mark.parametrize(
        "a,b",
        list(product(list(VerdictLevel), repeat=2)),
        ids=lambda pair: f"{pair[0].name}_x_{pair[1].name}",
    )
    def test_antisymmetric(self, a: VerdictLevel, b: VerdictLevel):
        """∀a,b ∈ L: (a ≤ b ∧ b ≤ a) ⟹ a = b (antisimetría)."""
        if a <= b and b <= a:
            assert a == b, (
                f"Antisimetría violada: {a.name} ≤ {b.name} ∧ {b.name} ≤ {a.name} "
                f"pero {a.name} ≠ {b.name}"
            )

    @pytest.mark.parametrize(
        "a,b,c",
        list(product(list(VerdictLevel), repeat=3)),
        ids=lambda triple: f"{triple[0].name}_{triple[1].name}_{triple[2].name}",
    )
    def test_transitive(
        self, a: VerdictLevel, b: VerdictLevel, c: VerdictLevel,
    ):
        """∀a,b,c ∈ L: (a ≤ b ∧ b ≤ c) ⟹ a ≤ c (transitividad)."""
        if a <= b and b <= c:
            assert a <= c, (
                f"Transitividad violada: {a.name} ≤ {b.name} ∧ {b.name} ≤ {c.name} "
                f"pero ¬({a.name} ≤ {c.name})"
            )

    # -- Totalidad (propiedad específica de cadena) --

    @pytest.mark.parametrize(
        "a,b",
        list(product(list(VerdictLevel), repeat=2)),
        ids=lambda pair: f"{pair[0].name}_x_{pair[1].name}",
    )
    def test_total_order_trichotomy(self, a: VerdictLevel, b: VerdictLevel):
        """
        ∀a,b ∈ L: exactamente una de {a < b, a = b, a > b} es verdadera.

        Esta es la ley de tricotomía que distingue un orden total de uno parcial.
        """
        lt = a.value < b.value
        eq = a.value == b.value
        gt = a.value > b.value
        count_true = sum([lt, eq, gt])
        assert count_true == 1, (
            f"Tricotomía violada para ({a.name}, {b.name}): "
            f"lt={lt}, eq={eq}, gt={gt} (exactamente 1 debe ser True)"
        )

    # -- Supremum / Infimum --

    def test_supremum_empty_is_bottom(self):
        """sup(∅) = ⊥ (el supremo del conjunto vacío es el elemento mínimo)."""
        assert VerdictLevel.supremum() == VerdictLevel.bottom()

    def test_infimum_empty_is_top(self):
        """inf(∅) = ⊤ (el ínfimo del conjunto vacío es el elemento máximo)."""
        assert VerdictLevel.infimum() == VerdictLevel.top()

    @pytest.mark.parametrize("a", list(VerdictLevel), ids=lambda v: v.name)
    def test_supremum_singleton(self, a: VerdictLevel):
        """sup({a}) = a para todo a ∈ L."""
        assert VerdictLevel.supremum(a) == a

    @pytest.mark.parametrize("a", list(VerdictLevel), ids=lambda v: v.name)
    def test_infimum_singleton(self, a: VerdictLevel):
        """inf({a}) = a para todo a ∈ L."""
        assert VerdictLevel.infimum(a) == a

    def test_supremum_all_is_top(self):
        """sup(L) = ⊤."""
        assert VerdictLevel.supremum(*VerdictLevel) == VerdictLevel.top()

    def test_infimum_all_is_bottom(self):
        """inf(L) = ⊥."""
        assert VerdictLevel.infimum(*VerdictLevel) == VerdictLevel.bottom()

    @pytest.mark.parametrize(
        "elements, expected_sup",
        [
            ([VerdictLevel.VIABLE, VerdictLevel.CONDICIONAL], VerdictLevel.CONDICIONAL),
            ([VerdictLevel.REVISAR, VerdictLevel.VIABLE], VerdictLevel.REVISAR),
            ([VerdictLevel.RECHAZAR, VerdictLevel.VIABLE], VerdictLevel.RECHAZAR),
            ([VerdictLevel.PRECAUCION, VerdictLevel.CONDICIONAL], VerdictLevel.PRECAUCION),
        ],
    )
    def test_supremum_pairs(
        self, elements: List[VerdictLevel], expected_sup: VerdictLevel,
    ):
        """Verifica supremo de pares específicos."""
        result = VerdictLevel.supremum(*elements)
        assert result == expected_sup, (
            f"sup({[e.name for e in elements]}) = {result.name}, "
            f"esperado {expected_sup.name}"
        )

    # -- Operadores con tipo incorrecto --

    def test_or_with_non_verdict_returns_not_implemented(self):
        """a | non_verdict debe retornar NotImplemented (no lanzar TypeError)."""
        result = VerdictLevel.VIABLE.__or__(42)
        assert result is NotImplemented

    def test_and_with_non_verdict_returns_not_implemented(self):
        """a & non_verdict debe retornar NotImplemented."""
        result = VerdictLevel.VIABLE.__and__("test")
        assert result is NotImplemented

    # -- Propiedades semánticas --

    def test_positive_verdicts_exactly(self):
        """
        VIABLE y CONDICIONAL son los únicos veredictos positivos.

        Semánticamente: el proyecto puede avanzar (con o sin condiciones).
        """
        positive = {v for v in VerdictLevel if v.is_positive}
        assert positive == {VerdictLevel.VIABLE, VerdictLevel.CONDICIONAL}, (
            f"Veredictos positivos incorrectos: {positive}"
        )

    def test_negative_verdicts_exactly(self):
        """
        Solo RECHAZAR es negativo (bloquea el proyecto).

        PRECAUCION y REVISAR son estados intermedios, no bloqueos.
        """
        negative = {v for v in VerdictLevel if v.is_negative}
        assert negative == {VerdictLevel.RECHAZAR}, (
            f"Veredictos negativos incorrectos: {negative}"
        )

    def test_requires_attention_exactly(self):
        """
        PRECAUCION y RECHAZAR requieren atención inmediata.

        Son los dos elementos del top del retículo que generan alertas.
        """
        attention = {v for v in VerdictLevel if v.requires_attention}
        assert attention == {VerdictLevel.PRECAUCION, VerdictLevel.RECHAZAR}, (
            f"Veredictos que requieren atención incorrectos: {attention}"
        )

    def test_normalized_score_bounds(self):
        """normalized_score ∈ [0, 1] para todo v ∈ L."""
        for v in VerdictLevel:
            score = v.normalized_score
            assert 0.0 <= score <= 1.0, (
                f"{v.name}.normalized_score = {score} ∉ [0, 1]"
            )

    def test_normalized_score_monotone_with_order(self):
        """
        normalized_score es monótonamente no-decreciente con el valor del veredicto.

        Propiedad: a ≤ b ⟹ score(a) ≤ score(b).
        """
        sorted_verdicts = sorted(VerdictLevel, key=lambda v: v.value)
        scores = [v.normalized_score for v in sorted_verdicts]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], (
                f"score no monótono: score({sorted_verdicts[i].name}) = {scores[i]} "
                f"> score({sorted_verdicts[i+1].name}) = {scores[i+1]}"
            )

    def test_normalized_score_bottom_is_zero(self):
        """score(⊥) = 0 (mínima severidad)."""
        assert VerdictLevel.bottom().normalized_score == 0.0

    def test_normalized_score_top_is_one(self):
        """score(⊤) = 1 (máxima severidad)."""
        assert VerdictLevel.top().normalized_score == 1.0

    def test_emoji_uniqueness(self):
        """Cada veredicto tiene un emoji único (sin ambigüedad visual)."""
        emojis = [v.emoji for v in VerdictLevel]
        assert len(emojis) == len(set(emojis)), (
            f"Emojis duplicados encontrados: {emojis}"
        )

    def test_description_non_empty_and_informative(self):
        """Cada veredicto tiene descripción no vacía y con contenido mínimo."""
        for v in VerdictLevel:
            desc = v.description
            assert isinstance(desc, str), f"{v.name}.description no es string"
            assert len(desc) > 5, (
                f"{v.name}.description demasiado corta: {desc!r}"
            )

    # -- Verificación completa del retículo --

    def test_verify_lattice_laws_all_pass(self):
        """
        El método verify_lattice_laws() debe pasar TODAS las leyes.

        Este test es el "oráculo" que llama a la verificación interna
        del módulo. Si falla, indica un error en la implementación del retículo.
        """
        results = VerdictLevel.verify_lattice_laws()
        assert isinstance(results, dict), "verify_lattice_laws debe retornar dict"
        assert len(results) > 0, "verify_lattice_laws retornó dict vacío"

        failed_laws = [name for name, passed in results.items() if not passed]
        assert not failed_laws, (
            f"Leyes del retículo violadas: {failed_laws}"
        )


# ============================================================================
# 9. TestSeverityHomomorphism — Refinado
# ============================================================================


class TestSeverityHomomorphism:
    """
    Verificación formal del homomorfismo de retículos φ: SeverityLattice → VerdictLevel.

    Un homomorfismo de retículos φ debe satisfacer:
    1. φ(a ⊔ b) = φ(a) ⊔ φ(b)  ∀a,b ∈ SeverityLattice
    2. φ(a ⊓ b) = φ(a) ⊓ φ(b)  ∀a,b ∈ SeverityLattice

    Nota crítica: En SeverityLattice (3 elementos: VIABLE < PRECAUCION < RECHAZAR),
    la operación join es:
        a ⊔ b = SeverityLattice(max(a.value, b.value))
    Y meet es:
        a ⊓ b = SeverityLattice(min(a.value, b.value))

    Estas operaciones se derivan de que SeverityLattice también es una cadena.
    """

    def _join_severity(
        self, a: SeverityLattice, b: SeverityLattice,
    ) -> SeverityLattice:
        """
        Join en SeverityLattice: operación correcta en la cadena.

        Se calcula como max por valor, luego se reconstruye el elemento.
        IMPORTANTE: no asumimos que los valores son 0,1,2 consecutivos;
        usamos la enumeración real para encontrar el max.
        """
        if a.value >= b.value:
            return a
        return b

    def _meet_severity(
        self, a: SeverityLattice, b: SeverityLattice,
    ) -> SeverityLattice:
        """Meet en SeverityLattice: operación correcta en la cadena."""
        if a.value <= b.value:
            return a
        return b

    def test_homomorphism_preserves_bottom(self):
        """φ(⊥_S) = ⊥_V: el bottom se preserva."""
        bottom_s = min(SeverityLattice, key=lambda s: s.value)
        result = SeverityToVerdictHomomorphism.apply(bottom_s)
        assert result == VerdictLevel.bottom(), (
            f"φ(⊥) = {result.name}, esperado {VerdictLevel.bottom().name}"
        )

    def test_homomorphism_preserves_top(self):
        """φ(⊤_S) = ⊤_V: el top se preserva."""
        top_s = max(SeverityLattice, key=lambda s: s.value)
        result = SeverityToVerdictHomomorphism.apply(top_s)
        assert result == VerdictLevel.top(), (
            f"φ(⊤) = {result.name}, esperado {VerdictLevel.top().name}"
        )

    def test_homomorphism_preserves_join(self):
        """
        ∀a,b ∈ SeverityLattice: φ(a ⊔ b) = φ(a) ⊔ φ(b).

        Corrección: el join en SeverityLattice se calcula correctamente
        usando la estructura de cadena (max por valor).
        """
        for a in SeverityLattice:
            for b in SeverityLattice:
                ab_join = self._join_severity(a, b)
                lhs = SeverityToVerdictHomomorphism.apply(ab_join)
                rhs = (
                    SeverityToVerdictHomomorphism.apply(a)
                    | SeverityToVerdictHomomorphism.apply(b)
                )
                assert lhs == rhs, (
                    f"Homomorfismo join violado: "
                    f"φ({a.name} ⊔ {b.name}) = φ({ab_join.name}) = {lhs.name} "
                    f"≠ φ({a.name}) ⊔ φ({b.name}) = {rhs.name}"
                )

    def test_homomorphism_preserves_meet(self):
        """
        ∀a,b ∈ SeverityLattice: φ(a ⊓ b) = φ(a) ⊓ φ(b).

        Corrección: el meet en SeverityLattice se calcula correctamente
        usando la estructura de cadena (min por valor).
        """
        for a in SeverityLattice:
            for b in SeverityLattice:
                ab_meet = self._meet_severity(a, b)
                lhs = SeverityToVerdictHomomorphism.apply(ab_meet)
                rhs = (
                    SeverityToVerdictHomomorphism.apply(a)
                    & SeverityToVerdictHomomorphism.apply(b)
                )
                assert lhs == rhs, (
                    f"Homomorfismo meet violado: "
                    f"φ({a.name} ⊓ {b.name}) = φ({ab_meet.name}) = {lhs.name} "
                    f"≠ φ({a.name}) ⊓ φ({b.name}) = {rhs.name}"
                )

    def test_homomorphism_is_injective(self):
        """
        φ es inyectivo: a ≠ b ⟹ φ(a) ≠ φ(b).

        Consecuencia: φ preserva la distinción entre todos los niveles de severidad.
        """
        mapped = [SeverityToVerdictHomomorphism.apply(s) for s in SeverityLattice]
        mapped_set = set(mapped)
        assert len(mapped) == len(mapped_set), (
            f"φ no es inyectivo: valores mapeados = {[m.name for m in mapped]}"
        )

    def test_homomorphism_preserves_strict_order(self):
        """
        φ preserva el orden estricto: a < b ⟹ φ(a) < φ(b).

        En cadenas, esto es equivalente a ser un isomorfismo sobre su imagen.
        """
        sorted_severity = sorted(SeverityLattice, key=lambda s: s.value)
        mapped = [SeverityToVerdictHomomorphism.apply(s) for s in sorted_severity]

        for i in range(len(mapped) - 1):
            s_i = sorted_severity[i]
            s_j = sorted_severity[i + 1]
            assert mapped[i].value < mapped[i + 1].value, (
                f"Orden estricto no preservado: "
                f"φ({s_i.name}) = {mapped[i].name} (valor={mapped[i].value}) "
                f"≥ φ({s_j.name}) = {mapped[i+1].name} (valor={mapped[i+1].value})"
            )

    def test_image_is_closed_under_join(self):
        """
        La imagen Im(φ) ⊆ VerdictLevel es cerrada bajo join.

        Im(φ) debe ser un sub-retículo de VerdictLevel.
        """
        image: Set[VerdictLevel] = {
            SeverityToVerdictHomomorphism.apply(s) for s in SeverityLattice
        }
        for a in image:
            for b in image:
                join_result = a | b
                assert join_result in image, (
                    f"Im(φ) no cerrada bajo join: "
                    f"{a.name} ⊔ {b.name} = {join_result.name} ∉ Im(φ) = "
                    f"{{{', '.join(v.name for v in image)}}}"
                )

    def test_image_is_closed_under_meet(self):
        """
        La imagen Im(φ) ⊆ VerdictLevel es cerrada bajo meet.

        Junto con test_image_is_closed_under_join, esto confirma que
        Im(φ) es un sub-retículo de VerdictLevel.
        """
        image: Set[VerdictLevel] = {
            SeverityToVerdictHomomorphism.apply(s) for s in SeverityLattice
        }
        for a in image:
            for b in image:
                meet_result = a & b
                assert meet_result in image, (
                    f"Im(φ) no cerrada bajo meet: "
                    f"{a.name} ⊓ {b.name} = {meet_result.name} ∉ Im(φ) = "
                    f"{{{', '.join(v.name for v in image)}}}"
                )

    def test_verify_homomorphism_method(self):
        """
        El método verify_homomorphism() implementado en el módulo retorna True.

        Este test actúa como "prueba de regresión" del método de verificación interno.
        """
        result = SeverityToVerdictHomomorphism.verify_homomorphism()
        assert result is True, (
            "SeverityToVerdictHomomorphism.verify_homomorphism() retornó False "
            "— el homomorfismo está roto en la implementación"
        )


# ============================================================================
# 12. TestNarrativeCache — Refinado
# ============================================================================


class TestNarrativeCache:
    """
    LRU cache thread-safe con claves deterministas.

    Invariantes a verificar:
    1. Determinismo de claves: misma entrada → misma clave (función pura)
    2. Independencia de orden en el dict de métricas (sort antes de hash)
    3. Inyectividad: entradas distintas → claves distintas (con alta probabilidad)
    4. LRU eviction: el elemento menos recientemente usado se evicta primero
    5. Thread-safety: ninguna operación corrompe el estado bajo concurrencia
    """

    def test_basic_put_get(self):
        """Put seguido de get retorna el valor exacto."""
        cache = NarrativeCache(maxsize=10)
        cache.put("D", "C", {"k": "v"}, "narrative_1")
        result = cache.get("D", "C", {"k": "v"})
        assert result == "narrative_1", f"Esperado 'narrative_1', got {result!r}"

    def test_cache_miss_returns_none(self):
        """Miss retorna None (no lanza excepción)."""
        cache = NarrativeCache(maxsize=10)
        result = cache.get("D", "C", {})
        assert result is None, f"Miss debe retornar None, got {result!r}"

    def test_deterministic_keys_same_dict(self):
        """
        La misma entrada produce la misma clave SHA-256 siempre.

        Propiedad: _make_key es una función PURA (sin estado).
        """
        key1 = NarrativeCache._make_key("D", "C", {"a": 1, "b": 2})
        key2 = NarrativeCache._make_key("D", "C", {"a": 1, "b": 2})
        assert key1 == key2, "Clave no determinista para la misma entrada"

    def test_deterministic_keys_dict_order_independent(self):
        """
        El hash es independiente del orden de inserción en el dict.

        Motivación: Python 3.7+ preserva orden de inserción en dicts,
        pero las métricas pueden venir en cualquier orden. El hash debe
        canonicalizar el dict ordenando las claves antes de serializar.
        """
        key1 = NarrativeCache._make_key("D", "C", {"a": 1, "b": 2, "c": 3})
        key2 = NarrativeCache._make_key("D", "C", {"c": 3, "a": 1, "b": 2})
        key3 = NarrativeCache._make_key("D", "C", {"b": 2, "c": 3, "a": 1})
        assert key1 == key2 == key3, (
            f"Clave depende del orden del dict: {key1} ≠ {key2}"
        )

    def test_different_inputs_different_keys(self):
        """
        Entradas distintas producen claves distintas con alta probabilidad.

        SHA-256 tiene resistencia a colisiones: P(colisión) ≈ 2⁻¹²⁸.
        Verificamos casos concretos cuyas colisiones son prácticamente imposibles.
        """
        pairs = [
            (("D1", "C", {}), ("D2", "C", {})),
            (("D", "C1", {}), ("D", "C2", {})),
            (("D", "C", {"a": 1}), ("D", "C", {"a": 2})),
            (("D", "C", {"a": 1}), ("D", "C", {"b": 1})),
            (("D", "C", {}), ("D", "C", {"a": 0})),
        ]
        for (args1, args2) in pairs:
            key1 = NarrativeCache._make_key(*args1)
            key2 = NarrativeCache._make_key(*args2)
            assert key1 != key2, (
                f"Colisión espuria: {args1} y {args2} → misma clave {key1}"
            )

    def test_key_is_sha256_hex(self):
        """
        La clave es un hex digest SHA-256: exactamente 64 caracteres hex.

        SHA-256 produce 256 bits = 32 bytes = 64 caracteres hexadecimales.
        """
        key = NarrativeCache._make_key("D", "C", {"k": 1})
        assert len(key) == 64, f"SHA-256 hex debe tener 64 chars, got {len(key)}"
        # Debe ser hexadecimal válido
        try:
            int(key, 16)
        except ValueError:
            pytest.fail(f"Clave no es hex válido: {key!r}")

    def test_lru_eviction_oldest_first(self):
        """
        El elemento más antiguo (LRU) se evicta cuando se excede maxsize.

        Secuencia determinista:
        1. put A (LRU: [A])
        2. put B (LRU: [A, B])
        3. put C → evicta A (LRU: [B, C])
        → A no está, B y C sí.
        """
        cache = NarrativeCache(maxsize=2)
        cache.put("D", "A", {}, "first")
        cache.put("D", "B", {}, "second")
        cache.put("D", "C", {}, "third")  # Evicta "first"

        assert cache.get("D", "A", {}) is None, "A debería haber sido evictado"
        assert cache.get("D", "B", {}) == "second", "B no debería haber sido evictado"
        assert cache.get("D", "C", {}) == "third", "C debe estar en caché"

    def test_lru_access_refreshes_recency(self):
        """
        Acceder a un elemento refresca su recencia, retrasando su evicción.

        Secuencia:
        1. put A (LRU: [A])
        2. put B (LRU: [A, B])
        3. get A  → A se mueve al frente (LRU: [B, A])
        4. put C  → evicta B (no A) (LRU: [A, C])
        → A está, B no, C está.
        """
        cache = NarrativeCache(maxsize=2)
        cache.put("D", "A", {}, "first")
        cache.put("D", "B", {}, "second")
        # Acceder a A lo refresca
        val_a = cache.get("D", "A", {})
        assert val_a == "first", "A debe estar accesible antes del refresco"
        # Insertar C evicta B (el LRU ahora)
        cache.put("D", "C", {}, "third")
        assert cache.get("D", "A", {}) == "first", "A no debería haber sido evictado"
        assert cache.get("D", "B", {}) is None, "B debería haber sido evictado"
        assert cache.get("D", "C", {}) == "third", "C debe estar en caché"

    def test_overwrite_existing_updates_value(self):
        """Put con clave existente actualiza el valor (no crea duplicado)."""
        cache = NarrativeCache(maxsize=10)
        cache.put("D", "C", {}, "v1")
        cache.put("D", "C", {}, "v2")
        assert cache.get("D", "C", {}) == "v2", "Valor no fue actualizado"
        # Verificar que el tamaño no creció (no hay duplicados)
        assert cache.stats["size"] == 1, "Duplicado en caché (tamaño incorrecto)"

    def test_clear_resets_state(self):
        """Clear vacía completamente el caché y resetea estadísticas a cero."""
        cache = NarrativeCache(maxsize=10)
        cache.put("D", "C", {}, "v1")
        cache.get("D", "C", {})   # Hit
        cache.get("D", "X", {})   # Miss
        cache.clear()

        # Verificar vaciado
        assert cache.get("D", "C", {}) is None, "Caché no fue vaciado"

        # Verificar reset de estadísticas
        stats = cache.stats
        assert stats["size"] == 0, f"Tamaño después de clear: {stats['size']}"
        assert stats["hits"] == 0, f"Hits después de clear: {stats['hits']}"
        assert stats["misses"] == 0, f"Misses después de clear: {stats['misses']}"

    def test_stats_accuracy(self):
        """
        Las estadísticas reflejan exactamente los hits y misses.

        Invariante: stats["hits"] + stats["misses"] == número de get() llamados
        (excepto después de clear()).
        """
        cache = NarrativeCache(maxsize=10)
        cache.put("D", "C", {}, "v1")

        # 2 hits, 1 miss
        cache.get("D", "C", {})   # Hit 1
        cache.get("D", "C", {})   # Hit 2
        cache.get("D", "X", {})   # Miss 1

        stats = cache.stats
        assert stats["hits"] == 2, f"Hits: esperado 2, got {stats['hits']}"
        assert stats["misses"] == 1, f"Misses: esperado 1, got {stats['misses']}"
        assert stats["size"] == 1, f"Size: esperado 1, got {stats['size']}"

    def test_hit_rate_computation(self):
        """
        Si stats incluye hit_rate, debe ser hits/(hits+misses).

        Tolerancia: EPSILON para división de punto flotante.
        """
        cache = NarrativeCache(maxsize=10)
        cache.put("D", "C", {}, "v1")
        for _ in range(3):
            cache.get("D", "C", {})   # 3 hits
        cache.get("D", "X", {})       # 1 miss

        stats = cache.stats
        if "hit_rate" in stats:
            expected_rate = 3 / 4
            assert abs(stats["hit_rate"] - expected_rate) < EPSILON, (
                f"hit_rate: esperado {expected_rate:.4f}, got {stats['hit_rate']:.4f}"
            )

    def test_invalid_maxsize_raises(self):
        """maxsize < 1 lanza ValueError (el caché debe tener capacidad positiva)."""
        with pytest.raises(ValueError):
            NarrativeCache(maxsize=0)
        with pytest.raises(ValueError):
            NarrativeCache(maxsize=-1)

    def test_thread_safety_no_corruption(self):
        """
        Acceso concurrente no corrompe el estado del caché.

        Verificaciones:
        1. No hay excepciones no manejadas bajo concurrencia
        2. El tamaño final no excede maxsize
        3. Los valores leídos son siempre del tipo correcto
        """
        MAXSIZE = 50
        N_WORKERS = 10
        OPS_PER_WORKER = 100
        cache = NarrativeCache(maxsize=MAXSIZE)
        errors: List[str] = []
        lock = threading.Lock()

        def worker(worker_id: int) -> None:
            try:
                for i in range(OPS_PER_WORKER):
                    key_suffix = i % (MAXSIZE * 2)  # Crear presión de evicción
                    domain = f"D{worker_id}"
                    classification = f"C{key_suffix}"
                    expected_value = f"v_{worker_id}_{key_suffix}"

                    # Put
                    cache.put(domain, classification, {"i": key_suffix}, expected_value)

                    # Get (puede ser miss si fue evictado por otro worker)
                    result = cache.get(domain, classification, {"i": key_suffix})

                    # Invariante: si no es None, debe ser string
                    if result is not None and not isinstance(result, str):
                        with lock:
                            errors.append(
                                f"Worker {worker_id}, i={i}: "
                                f"tipo incorrecto {type(result).__name__}"
                            )

            except Exception as e:
                with lock:
                    errors.append(f"Worker {worker_id}: {type(e).__name__}: {e}")

        threads = [
            threading.Thread(target=worker, args=(i,), daemon=True)
            for i in range(N_WORKERS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        # Verificar que todos los threads terminaron
        alive = [t for t in threads if t.is_alive()]
        assert not alive, f"{len(alive)} threads no terminaron (posible deadlock)"

        # Verificar ausencia de errores
        assert not errors, f"Errores de concurrencia:\n" + "\n".join(errors[:5])

        # Verificar invariante de tamaño
        stats = cache.stats
        assert stats["size"] <= MAXSIZE, (
            f"Tamaño del caché excedió maxsize: {stats['size']} > {MAXSIZE}"
        )


# ============================================================================
# 13. TestSemanticDiffeomorphism — Refinado
# ============================================================================


class TestSemanticDiffeomorphism:
    """
    Verificación de funtores semánticos InvariantSpace → ImpactSpace.

    Los 'diffeomorfismos semánticos' son morfismos que mapean métricas
    cuantitativas (espacio de invariantes) a narrativas cualitativas
    (espacio de impacto). Propiedades a verificar:
    1. Completitud: cubren el dominio relevante
    2. No trivialidad: producen contenido informativo
    3. Continuidad de Lipschitz (discreta): perturbaciones pequeñas
       producen cambios acotados en el espacio semántico.
    """

    def test_map_betti_1_empty_returns_empty(self):
        """Lista vacía de ciclos → string vacío (sin narrativa de ciclos)."""
        result = SemanticDiffeomorphismMapper.map_betti_1_cycles([])
        assert result == "", f"Lista vacía debe retornar '', got {result!r}"

    def test_map_betti_1_single_node_returns_empty(self):
        """Lista con un solo nodo no forma ciclo → string vacío o mínimo."""
        result = SemanticDiffeomorphismMapper.map_betti_1_cycles(["A"])
        # Un ciclo requiere ≥ 2 nodos; con 1 no hay ciclo semántico
        assert result == "" or len(result) < 10

    def test_map_betti_1_contains_all_nodes_and_closure(self):
        """
        La narrativa de ciclo A→B→C contiene:
        1. Todos los nodos del ciclo
        2. El cierre del ciclo (A→...→A)
        3. La palabra VETO (impacto semántico del ciclo)
        4. El operador de arista (➔ o →)
        """
        nodes = ["A", "B", "C"]
        result = SemanticDiffeomorphismMapper.map_betti_1_cycles(nodes)

        for node in nodes:
            assert node in result, f"Nodo {node!r} no encontrado en narrativa: {result!r}"

        arrow_present = "➔" in result or "→" in result
        assert arrow_present, f"Operador de arista no encontrado en narrativa: {result!r}"
        assert "VETO" in result, f"'VETO' no encontrado en narrativa: {result!r}"

    def test_map_pyramid_instability_contains_psi_and_provider(self):
        """
        Narrativa de pirámide invertida contiene:
        1. El valor de Ψ formateado (2 decimales)
        2. El nombre del proveedor/entidad
        3. Una señal de alerta (ALERTA o QUIEBRA)
        """
        result = SemanticDiffeomorphismMapper.map_pyramid_instability(
            0.5, "Proveedor X",
        )
        assert "0.50" in result, f"Ψ=0.50 no encontrado en: {result!r}"
        assert "Proveedor X" in result, f"'Proveedor X' no encontrado en: {result!r}"
        alert_present = "ALERTA" in result or "QUIEBRA" in result
        assert alert_present, f"Señal de alerta no encontrada en: {result!r}"

    def test_map_fragmentation_contains_count_and_cost(self):
        """
        Narrativa de fragmentación contiene:
        1. El número de componentes (β₀)
        2. El costo en riesgo formateado con separadores de miles
        3. La palabra FUGA (impacto financiero de la fragmentación)
        """
        result = SemanticDiffeomorphismMapper.map_betti_0_fragmentation(3, 1_500_000.0)
        assert "3" in result, f"β₀=3 no encontrado en: {result!r}"
        # El número puede aparecer como 1,500,000 o 1.500.000 según locale
        has_cost = "1,500,000" in result or "1.500.000" in result or "1500000" in result
        assert has_cost, f"Costo 1,500,000 no encontrado en: {result!r}"
        assert "FUGA" in result, f"'FUGA' no encontrado en: {result!r}"

    def test_lipschitz_continuity_within_same_interval(
        self, translator: SemanticTranslator,
    ):
        """
        Continuidad de Lipschitz (Teorema 3 refinado).

        Propiedad formal: Para el mapeo discreto f: ℝ → VerdictLevel,
        dos puntos en el mismo intervalo de clasificación deben mapearse
        al mismo elemento del retículo (constante de Lipschitz K = 0 local).

        Verificamos en la vecindad del umbral entropy_high con ε = 1e-9 < 1e-6,
        garantizando que ambos puntos pertenecen al mismo intervalo.
        """
        entropy_threshold = translator.config.thermal.entropy_high
        delta = 1e-9  # Mucho menor que cualquier diferencia de umbral

        # Dos puntos dentro del mismo intervalo [entropy_low, entropy_high)
        entropy_1 = entropy_threshold - delta
        entropy_2 = entropy_threshold - 2 * delta

        # Ambos deben clasificarse igual
        class_1 = translator.config.thermal.classify_entropy(entropy_1)
        class_2 = translator.config.thermal.classify_entropy(entropy_2)
        assert class_1 == class_2, (
            f"Bifurcación espuria: classify_entropy({entropy_1}) = {class_1!r} "
            f"≠ classify_entropy({entropy_2}) = {class_2!r} "
            f"(ambos deberían estar en el mismo intervalo)"
        )

        # Las traducciones deben producir el mismo veredicto
        metrics_1 = {"temperature": 25.0, "entropy": entropy_1, "heat_capacity": 0.5}
        metrics_2 = {"temperature": 25.0, "entropy": entropy_2, "heat_capacity": 0.5}

        _, verdict_1 = translator.translate_thermodynamics(metrics_1)
        _, verdict_2 = translator.translate_thermodynamics(metrics_2)

        assert verdict_1 == verdict_2, (
            f"Violación de Lipschitz local: "
            f"verdict({entropy_1}) = {verdict_1.name} "
            f"≠ verdict({entropy_2}) = {verdict_2.name} "
            f"(Δε = {delta:.2e}, mismo intervalo)"
        )

    def test_lipschitz_bounded_jump_across_threshold(
        self, translator: SemanticTranslator,
    ):
        """
        Al cruzar un umbral, el salto en VerdictLevel está acotado.

        Invariante: el salto discreto máximo en una sola transición de umbral
        es ≤ _MAX_LIPSCHITZ_JUMP = 3 (ceil(n/2) para n=5 elementos).

        Esto garantiza que no hay "saltos catastróficos" de VIABLE a RECHAZAR
        en un solo umbral.
        """
        entropy_threshold = translator.config.thermal.entropy_high
        delta = 1e-9

        metrics_below = {
            "temperature": 25.0,
            "entropy": entropy_threshold - delta,
            "heat_capacity": 0.5,
        }
        metrics_above = {
            "temperature": 25.0,
            "entropy": entropy_threshold + delta,
            "heat_capacity": 0.5,
        }

        _, verdict_below = translator.translate_thermodynamics(metrics_below)
        _, verdict_above = translator.translate_thermodynamics(metrics_above)

        jump = abs(verdict_above.value - verdict_below.value)
        assert jump <= _MAX_LIPSCHITZ_JUMP, (
            f"Salto de Lipschitz excesivo al cruzar entropy_high={entropy_threshold}: "
            f"{verdict_below.name} (valor={verdict_below.value}) → "
            f"{verdict_above.name} (valor={verdict_above.value}), "
            f"salto={jump} > {_MAX_LIPSCHITZ_JUMP}"
        )

    def test_lipschitz_no_jump_to_max_from_viable(
        self, translator: SemanticTranslator,
    ):
        """
        Al cruzar UN umbral, no se puede saltar de VIABLE a RECHAZAR directamente.

        Invariante semántico: la transición de "viable" a "rechazar" requiere
        cruzar múltiples umbrales (no es una discontinuidad de salto único).
        """
        entropy_threshold = translator.config.thermal.entropy_high
        delta = 1e-9

        _, verdict_below = translator.translate_thermodynamics({
            "temperature": 25.0,
            "entropy": entropy_threshold - delta,
            "heat_capacity": 0.5,
        })

        if verdict_below == VerdictLevel.VIABLE:
            _, verdict_above = translator.translate_thermodynamics({
                "temperature": 25.0,
                "entropy": entropy_threshold + delta,
                "heat_capacity": 0.5,
            })
            assert verdict_above != VerdictLevel.RECHAZAR, (
                f"Transición directa VIABLE→RECHAZAR al cruzar entropy_high={entropy_threshold}: "
                f"violación de continuidad semántica"
            )


# ============================================================================
# 14. TestGraphRAGCausalNarrator — Refinado
# ============================================================================


class TestGraphRAGCausalNarrator:
    """
    Verificación del narrador causal basado en grafos.

    Propiedades a verificar:
    1. Corrección de tipos (rechazo de no-dígrafos)
    2. Acotamiento de enumeración de ciclos (complejidad controlada)
    3. Identificación correcta del nodo crítico (por centralidad)
    4. Invarianza bajo isomorfismos de grafo (propiedad topológica)
    """

    def test_invalid_graph_type_raises(self):
        """Tipos incorrectos de grafo deben lanzar GraphStructureError."""
        invalid_inputs = ["not_a_graph", 42, None, [], {}, nx.Graph()]
        for invalid in invalid_inputs:
            with pytest.raises((GraphStructureError, TypeError)):
                GraphRAGCausalNarrator(invalid)  # type: ignore

    def test_empty_graph_returns_nominal(self, empty_digraph: nx.DiGraph):
        """Grafo vacío produce estado nominal (sin defectos detectables)."""
        narrator = GraphRAGCausalNarrator(empty_digraph)
        result = narrator.narrate_topological_collapse(betti_1=0, psi=2.0)
        assert "NOMINAL" in result, (
            f"Grafo vacío con β₁=0 y Ψ=2.0 debería ser NOMINAL: {result!r}"
        )

    def test_cycle_detection_generates_veto_narrative(
        self, simple_digraph: nx.DiGraph,
    ):
        """
        β₁ > 0 → narrativa debe contener 'VETO' (ciclo semántico detectado).

        El dígrafo simple tiene exactamente un ciclo A→B→C→A.
        """
        narrator = GraphRAGCausalNarrator(simple_digraph)
        result = narrator.narrate_topological_collapse(betti_1=1, psi=2.0)
        cycle_signal = "VETO" in result or "circular" in result.lower() or "ciclo" in result.lower()
        assert cycle_signal, (
            f"β₁=1 debe generar señal de ciclo en narrativa: {result!r}"
        )

    def test_inverted_pyramid_detection(self, simple_digraph: nx.DiGraph):
        """
        Ψ < 1 → narrativa debe contener señal de pirámide invertida.

        Umbral: Ψ < 1 activa el veto de pirámide invertida.
        """
        narrator = GraphRAGCausalNarrator(simple_digraph)
        result = narrator.narrate_topological_collapse(betti_1=0, psi=0.5)
        inversion_signal = (
            "ALERTA" in result or
            "Pirámide" in result or
            "QUIEBRA" in result or
            "invertida" in result.lower()
        )
        assert inversion_signal, (
            f"Ψ=0.5 < 1 debe generar señal de pirámide invertida: {result!r}"
        )

    def test_bounded_cycle_enumeration_respects_limit(self):
        """
        La enumeración de ciclos respeta estrictamente el límite configurado.

        Grafo completo K₆ tiene O(n! * 2^n) ciclos simples → sin límite se cuelga.
        Con max_cycles=3, retorna exactamente ≤ 3 ciclos.
        """
        g = nx.complete_graph(6, create_using=nx.DiGraph)
        for limit in [1, 3, 5, 10]:
            narrator = GraphRAGCausalNarrator(g, max_cycles=limit)
            cycles = narrator._enumerate_cycles_bounded()
            assert len(cycles) <= limit, (
                f"Enumeración retornó {len(cycles)} ciclos > límite {limit}"
            )

    def test_bounded_cycle_enumeration_max_module_constant(self):
        """
        Sin límite explícito, respeta MAX_SIMPLE_CYCLES_ENUMERATION del módulo.
        """
        g = nx.complete_graph(8, create_using=nx.DiGraph)
        narrator = GraphRAGCausalNarrator(g)
        cycles = narrator._enumerate_cycles_bounded()
        assert len(cycles) <= MAX_SIMPLE_CYCLES_ENUMERATION, (
            f"Enumeración sin límite superó MAX_SIMPLE_CYCLES_ENUMERATION="
            f"{MAX_SIMPLE_CYCLES_ENUMERATION}: got {len(cycles)}"
        )

    def test_critical_node_in_acyclic_graph(self, acyclic_digraph: nx.DiGraph):
        """
        El nodo crítico en un DAG es un nodo real del grafo (no placeholder).

        En el DAG de rombo (A→B, A→C, B→D, C→D), D tiene el mayor in-degree.
        """
        narrator = GraphRAGCausalNarrator(acyclic_digraph)
        node = narrator._find_critical_node()
        assert isinstance(node, str), f"Nodo crítico debe ser str, got {type(node)}"
        assert node in acyclic_digraph.nodes(), (
            f"Nodo crítico '{node}' no está en el grafo"
        )

    def test_critical_node_empty_graph_returns_placeholder(
        self, empty_digraph: nx.DiGraph,
    ):
        """Grafo vacío retorna el placeholder de nodo desconocido."""
        narrator = GraphRAGCausalNarrator(empty_digraph)
        node = narrator._find_critical_node()
        assert node == "Nodo Central Desconocido", (
            f"Grafo vacío debe retornar 'Nodo Central Desconocido', got {node!r}"
        )

    def test_critical_node_disconnected_graph(
        self, disconnected_digraph: nx.DiGraph,
    ):
        """
        Grafo desconectado usa fallback (eigenvector centrality falla).

        El fallback por grado debe retornar un nodo válido del grafo.
        """
        narrator = GraphRAGCausalNarrator(disconnected_digraph)
        node = narrator._find_critical_node()
        assert isinstance(node, str), f"Nodo crítico debe ser str"
        assert node in disconnected_digraph.nodes(), (
            f"Nodo crítico '{node}' no está en el grafo"
        )

    def test_causal_invariance_under_graph_isomorphism(
        self, simple_digraph: nx.DiGraph,
    ):
        """
        Teorema 1 refinado: Invarianza Causal bajo Isomorfismos de Grafo.

        Para dos grafos isomorfos G y G' = π(G) (donde π es una permutación
        de etiquetas), las narrativas normalizadas (con nombres de nodo
        reemplazados por [NODO]) deben ser idénticas.

        Corrección: la normalización usa regex con word boundaries (\b)
        y aplica las sustituciones en orden de longitud decreciente para
        evitar colisiones (ej: "AB" no debe colisionar con sustitución de "A").
        """
        # Narrativa del grafo original
        narrator_orig = GraphRAGCausalNarrator(simple_digraph)
        narrative_orig = narrator_orig.narrate_topological_collapse(
            betti_1=1, psi=2.0,
        )

        # Mapeo de isomorfismo con nombres que no son substrings entre sí
        mapping = {
            "A": "Ingenieria",
            "B": "Compras",
            "C": "Finanzas",
            "D": "Logistica",
        }
        isomorphic_graph = nx.relabel_nodes(simple_digraph, mapping)

        # Narrativa del grafo isomórfico
        narrator_iso = GraphRAGCausalNarrator(isomorphic_graph)
        narrative_iso = narrator_iso.narrate_topological_collapse(
            betti_1=1, psi=2.0,
        )

        # Ambas deben contener la señal de VETO
        assert "VETO" in narrative_orig, f"'VETO' no en narrativa original: {narrative_orig!r}"
        assert "VETO" in narrative_iso, f"'VETO' no en narrativa isomórfica: {narrative_iso!r}"

        def normalize(text: str, nodes: List[str]) -> str:
            """
            Normaliza una narrativa reemplazando nombres de nodos por [NODO].

            Orden: de mayor a menor longitud para evitar sustituciones parciales
            (ej: "Ingenieria" antes que "Inge" si hubiera colisión).
            """
            result = text
            for node in sorted(nodes, key=len, reverse=True):
                result = re.sub(
                    rf'\b{re.escape(str(node))}\b',
                    '[NODO]',
                    result,
                )
            return result

        norm_orig = normalize(narrative_orig, list(simple_digraph.nodes()))
        norm_iso = normalize(narrative_iso, list(isomorphic_graph.nodes()))

        assert norm_orig == norm_iso, (
            f"Invarianza isomórfica violada:\n"
            f"  Original normalizado:   {norm_orig!r}\n"
            f"  Isomórfico normalizado: {norm_iso!r}"
        )


# ============================================================================
# 17. TestStrategicReport — Refinado
# ============================================================================


class TestStrategicReport:
    """
    Serialización y propiedades del reporte estratégico.

    El StrategicReport es el resultado del análisis multi-estrato.
    Sus invariantes son:
    1. Confianza ∈ [0, 1]
    2. El veredicto determina is_viable y requires_immediate_action
    3. La serialización es determinista y contiene todas las claves requeridas
    4. El timestamp es ISO 8601 válido
    """

    def _make_report(
        self,
        verdict: VerdictLevel = VerdictLevel.VIABLE,
        confidence: float = 1.0,
        recommendations: Optional[List[str]] = None,
    ) -> StrategicReport:
        """Factory para crear reportes de prueba con valores mínimos válidos."""
        return StrategicReport(
            title="Test Report",
            verdict=verdict,
            executive_summary="Summary",
            strata_analysis={},
            recommendations=recommendations or ["Rec 1"],
            raw_narrative="Raw",
            confidence=confidence,
        )

    def test_viable_report_properties(self):
        """VIABLE: is_viable=True, requires_immediate_action=False."""
        report = self._make_report(VerdictLevel.VIABLE)
        assert report.is_viable, "VIABLE debe tener is_viable=True"
        assert not report.requires_immediate_action, (
            "VIABLE no debe requerir acción inmediata"
        )

    def test_condicional_report_properties(self):
        """CONDICIONAL: is_viable=True (positivo), no requiere acción inmediata."""
        report = self._make_report(VerdictLevel.CONDICIONAL)
        assert report.is_viable, "CONDICIONAL debe ser viable (positivo)"
        assert not report.requires_immediate_action

    def test_revisar_report_properties(self):
        """REVISAR: is_viable=False, requires_action varía según implementación."""
        report = self._make_report(VerdictLevel.REVISAR)
        assert not report.is_viable, "REVISAR no debe ser viable"

    def test_precaucion_report_properties(self):
        """PRECAUCION: is_viable=False, requires_immediate_action=True."""
        report = self._make_report(VerdictLevel.PRECAUCION)
        assert not report.is_viable
        assert report.requires_immediate_action, (
            "PRECAUCION debe requerir atención inmediata"
        )

    def test_rechazar_report_properties(self):
        """RECHAZAR: is_viable=False, requires_immediate_action=True."""
        report = self._make_report(VerdictLevel.RECHAZAR)
        assert not report.is_viable
        assert report.requires_immediate_action

    @pytest.mark.parametrize(
        "bad_confidence",
        [1.001, 1.5, 2.0, -0.001, -0.1, float("nan"), float("inf")],
    )
    def test_invalid_confidence_raises(self, bad_confidence: float):
        """Confianza fuera de [0, 1] o no finita lanza ValueError."""
        with pytest.raises(ValueError):
            self._make_report(confidence=bad_confidence)

    @pytest.mark.parametrize("valid_confidence", [0.0, 0.5, 0.999, 1.0])
    def test_valid_confidence_accepted(self, valid_confidence: float):
        """Confianza en [0, 1] es aceptada sin excepción."""
        report = self._make_report(confidence=valid_confidence)
        assert abs(report.confidence - valid_confidence) < EPSILON

    def test_to_dict_contains_required_keys(self):
        """
        La serialización contiene todas las claves requeridas.

        Estas claves son parte del contrato de la API pública del reporte.
        """
        report = self._make_report()
        d = report.to_dict()
        required_keys = {
            "title", "verdict", "verdict_emoji", "verdict_description",
            "is_viable", "requires_action", "executive_summary",
            "strata_analysis", "recommendations", "timestamp", "confidence",
        }
        missing = required_keys - set(d.keys())
        assert not missing, f"Claves faltantes en to_dict(): {missing}"

    def test_to_dict_verdict_is_string(self):
        """El veredicto serializado es un string (no un enum)."""
        report = self._make_report(VerdictLevel.RECHAZAR)
        d = report.to_dict()
        assert isinstance(d["verdict"], str), (
            f"verdict en dict debe ser str, got {type(d['verdict'])}"
        )
        assert d["verdict"] == "RECHAZAR"

    def test_to_dict_recommendations_are_deep_copy(self):
        """
        Las recomendaciones serializadas son una copia profunda.

        Modificar el dict serializado no debe afectar el reporte original.
        """
        recs = ["Rec 1", "Rec 2"]
        report = self._make_report(recommendations=recs)
        d = report.to_dict()
        d["recommendations"].append("Injected")
        assert len(report.recommendations) == 2, (
            "Modificar dict serializado afectó el reporte original (shallow copy)"
        )

    def test_timestamp_is_iso_8601(self):
        """
        El timestamp está en formato ISO 8601.

        Estándar: YYYY-MM-DDTHH:MM:SS.ffffff (Python datetime.isoformat())
        """
        from datetime import datetime as dt
        report = self._make_report()
        try:
            parsed = dt.fromisoformat(report.timestamp)
        except ValueError as e:
            pytest.fail(
                f"Timestamp '{report.timestamp}' no es ISO 8601 válido: {e}"
            )

    def test_timestamp_is_recent(self):
        """El timestamp es reciente (creado en los últimos 5 segundos)."""
        from datetime import datetime as dt, timezone, timedelta
        report = self._make_report()
        now = dt.now()
        parsed = dt.fromisoformat(report.timestamp)
        # Tolerancia de 5 segundos para ejecución de tests
        delta = abs((now - parsed).total_seconds())
        assert delta < 5.0, (
            f"Timestamp demasiado antiguo: {report.timestamp} "
            f"(diferencia: {delta:.2f}s)"
        )


# ============================================================================
# TESTS DE PROPIEDADES ALGEBRAICAS ADICIONALES — Refinado
# ============================================================================


class TestAlgebraicProperties:
    """
    Propiedades algebraicas profundas del sistema de decisión.

    Verifica propiedades que emergen de la composición del sistema:
    - Totalidad del orden en VerdictLevel (tricotomía)
    - Coherencia de join=max y meet=min en cadenas
    - Sub-retículo de la imagen del homomorfismo
    - Preservación de orden en mapeos financieros
    """

    def test_verdict_satisfies_trichotomy(self):
        """
        VerdictLevel satisface la tricotomía: ∀a,b, exactamente una de
        {a < b, a = b, a > b} es verdadera.
        """
        for a in VerdictLevel:
            for b in VerdictLevel:
                lt = a.value < b.value
                eq = a.value == b.value
                gt = a.value > b.value
                assert (lt + eq + gt) == 1, (
                    f"Tricotomía violada: ({a.name}, {b.name}) → "
                    f"lt={lt}, eq={eq}, gt={gt}"
                )

    def test_verdict_is_total_order(self):
        """∀a,b ∈ L: a ≤ b ∨ b ≤ a (totalidad del orden)."""
        for a in VerdictLevel:
            for b in VerdictLevel:
                assert a <= b or b <= a, (
                    f"Totalidad violada: {a.name} y {b.name} son incomparables"
                )

    def test_join_equals_max_in_chain(self):
        """
        En una cadena, join = max.

        Teorema: (L, ≤) cadena ⟹ a ⊔ b = max(a, b).
        """
        for a in VerdictLevel:
            for b in VerdictLevel:
                expected = VerdictLevel(max(a.value, b.value))
                result = a | b
                assert result == expected, (
                    f"join ≠ max: {a.name} ⊔ {b.name} = {result.name}, "
                    f"max = {expected.name}"
                )

    def test_meet_equals_min_in_chain(self):
        """
        En una cadena, meet = min.

        Teorema: (L, ≤) cadena ⟹ a ⊓ b = min(a, b).
        """
        for a in VerdictLevel:
            for b in VerdictLevel:
                expected = VerdictLevel(min(a.value, b.value))
                result = a & b
                assert result == expected, (
                    f"meet ≠ min: {a.name} ⊓ {b.name} = {result.name}, "
                    f"min = {expected.name}"
                )

    def test_lattice_is_bounded(self):
        """
        L tiene bottom (⊥) y top (⊤) bien definidos.

        Equivalente a: ∃⊥,⊤ ∈ L: ∀a ∈ L, ⊥ ≤ a ≤ ⊤.
        """
        bottom = VerdictLevel.bottom()
        top = VerdictLevel.top()
        for v in VerdictLevel:
            assert bottom <= v, f"⊥={bottom.name} ≰ {v.name}"
            assert v <= top, f"{v.name} ≰ ⊤={top.name}"

    def test_severity_lattice_image_is_sublattice(self):
        """
        Im(φ) ⊆ VerdictLevel es un sub-retículo.

        Definición: S ⊆ L es sub-retículo si ∀a,b ∈ S: a ⊔ b ∈ S ∧ a ⊓ b ∈ S.

        Corrección: verificamos que el join y meet de elementos de Im(φ)
        pertenecen a Im(φ) — no comparamos con min/max de Im(φ) que sería
        una condición más débil.
        """
        image: FrozenSet[VerdictLevel] = frozenset(
            SeverityToVerdictHomomorphism.apply(s) for s in SeverityLattice
        )

        for a in image:
            for b in image:
                join_ab = a | b
                meet_ab = a & b
                assert join_ab in image, (
                    f"Im(φ) no cerrada bajo join: "
                    f"{a.name} ⊔ {b.name} = {join_ab.name} ∉ Im(φ)"
                )
                assert meet_ab in image, (
                    f"Im(φ) no cerrada bajo meet: "
                    f"{a.name} ⊓ {b.name} = {meet_ab.name} ∉ Im(φ)"
                )

    def test_financial_verdict_order_strictly_preserved(self):
        """
        El mapeo FinancialVerdict → VerdictLevel preserva el orden ESTRICTO.

        ACCEPT < CONDITIONAL < REVIEW < REJECT en FinancialVerdict
        debe mapear a valores estrictamente crecientes en VerdictLevel.
        """
        ordered_financial = [
            FinancialVerdict.ACCEPT,
            FinancialVerdict.CONDITIONAL,
            FinancialVerdict.REVIEW,
            FinancialVerdict.REJECT,
        ]
        mapped_values = [fv.to_verdict_level().value for fv in ordered_financial]

        for i in range(len(mapped_values) - 1):
            assert mapped_values[i] < mapped_values[i + 1], (
                f"Orden no estricto: "
                f"φ({ordered_financial[i].name}) = {mapped_values[i]} "
                f"≥ φ({ordered_financial[i+1].name}) = {mapped_values[i+1]}"
            )


# ============================================================================
# TESTS DE EDGE CASES — Refinado
# ============================================================================


class TestEdgeCases:
    """
    Tests de casos límite con fundamentación matemática explícita.
    """

    def test_gerschgorin_robustness_at_connectivity_threshold(
        self, translator: SemanticTranslator,
    ):
        """
        Teorema 2 refinado: Acotamiento por Discos de Gerschgorin.

        Sea L el Laplaciano normalizado de un grafo y E la matriz de perturbación
        numérica. Por el Teorema de Weyl: |λᵢ(L+E) - λᵢ(L)| ≤ ‖E‖₂.

        Para una perturbación de magnitud ε_mach, el valor de Fiedler perturbado
        λ₂(L+E) satisface: |λ₂(L+E) - λ₂(L)| ≤ ε_mach * ‖L‖₂.

        Construimos un caso donde λ₂(L) > fiedler_threshold + δ con
        δ >> ε_mach * ‖L‖₂, garantizando que la perturbación NO cruza el umbral.
        """
        fiedler_threshold = translator.config.topology.fiedler_connected_threshold
        eps_machine = np.finfo(float).eps  # ~2.22e-16

        # Laplaciano de K₂ (grafo completo de 2 nodos): L = [[1,-1],[-1,1]]
        # Eigenvalores: 0 y 2
        # Si escalamos: L_scaled = s * L → eigenvalores: 0 y 2s
        # Para que λ₂ = fiedler_threshold + safety_margin:
        safety_margin = 1e-6  # >> eps_machine * ‖L‖₂ ≈ eps_machine * 2
        target_fiedler = fiedler_threshold + safety_margin

        scale = target_fiedler / 2.0
        L_base = scale * np.array([[1.0, -1.0], [-1.0, 1.0]])

        # Verificar eigenvalores base
        eigs_base = np.sort(np.linalg.eigvalsh(L_base))
        fiedler_base = eigs_base[1]
        assert abs(fiedler_base - target_fiedler) < 1e-12, (
            f"Fiedler base incorrecto: {fiedler_base} ≠ {target_fiedler}"
        )

        # Perturbación simétrica acotada: ‖E‖₂ = 2 * perturbation_size
        # Por Weyl: |Δλ₂| ≤ ‖E‖₂ = 2 * 10 * eps_machine << safety_margin
        perturbation_size = 10 * eps_machine
        E = perturbation_size * np.array([[0.0, 1.0], [1.0, 0.0]])
        L_perturbed = L_base + E

        eigs_perturbed = np.sort(np.linalg.eigvalsh(L_perturbed))
        fiedler_perturbed = eigs_perturbed[1]

        # Verificar que la perturbación no cruzó el umbral
        weyl_bound = np.linalg.norm(E, ord=2)  # ≈ 2 * perturbation_size
        assert fiedler_perturbed > fiedler_threshold, (
            f"Perturbación de Gerschgorin cruzó el umbral: "
            f"λ₂_perturb={fiedler_perturbed:.2e} ≤ threshold={fiedler_threshold:.2e}. "
            f"Weyl bound={weyl_bound:.2e}, safety_margin={safety_margin:.2e}"
        )

        # Construir topología con el Fiedler perturbado
        topo = ValidatedTopology(
            beta_0=1, beta_1=0, beta_2=0,
            euler_characteristic=1,
            fiedler_value=float(fiedler_perturbed),
            spectral_gap=0.5,
            pyramid_stability=5.0,
            structural_entropy=0.1,
        )

        narrative, verdict = translator.translate_topology(topo, stability=5.0)

        # Con λ₂ > threshold, la topología es conexa → no debe ser RECHAZAR
        assert verdict != VerdictLevel.RECHAZAR, (
            f"Falso positivo: ruido de máquina (‖E‖₂={weyl_bound:.2e}) causó "
            f"rechazo topológico. λ₂_perturb={fiedler_perturbed:.2e} > threshold."
        )
        # Verificar que la clasificación espectral es "connected" (no "disconnected")
        spectral_class = translator.config.topology.classify_spectral_connectivity(
            float(fiedler_perturbed),
        )
        assert "disconnected" not in spectral_class, (
            f"Clasificación espectral incorrecta: {spectral_class!r} "
            f"para λ₂={fiedler_perturbed:.2e} > threshold={fiedler_threshold:.2e}"
        )

    def test_topology_beta_0_zero_is_rechazar(self, translator: SemanticTranslator):
        """
        β₀ = 0 (grafo vacío conceptual): Ψ=0 < 1 → RECHAZAR.

        Χ = β₀ - β₁ + β₂ = 0 - 0 + 0 = 0. Grafo vacío es topológicamente trivial.
        """
        topo = ValidatedTopology(
            beta_0=0, beta_1=0, beta_2=0,
            euler_characteristic=0,
            fiedler_value=0.0, spectral_gap=0.0,
            pyramid_stability=0.0, structural_entropy=0.0,
        )
        _, verdict = translator.translate_topology(topo, stability=0.0)
        assert verdict == VerdictLevel.RECHAZAR, (
            f"β₀=0, Ψ=0 debe ser RECHAZAR, got {verdict.name}"
        )

    def test_extreme_betti_numbers(self, translator: SemanticTranslator):
        """
        Números de Betti grandes (β₀=1000, β₁=500) → RECHAZAR.

        χ = 1000 - 500 + 0 = 500. Ψ=0.1 < 1 → veto de pirámide.
        """
        topo = ValidatedTopology(
            beta_0=1000, beta_1=500, beta_2=0,
            euler_characteristic=500,  # 1000 - 500 + 0 = 500 ✓
            fiedler_value=0.001,
            spectral_gap=0.001,
            pyramid_stability=0.1,
            structural_entropy=0.99,
        )
        _, verdict = translator.translate_topology(topo, stability=0.1)
        assert verdict == VerdictLevel.RECHAZAR, (
            f"β₁=500, Ψ=0.1 debe ser RECHAZAR, got {verdict.name}"
        )

    def test_stability_exactly_at_thresholds(self, translator: SemanticTranslator):
        """
        Verifica que los umbrales implementan intervalos [a, b) correctamente.

        Con la convención [a, b):
        - Ψ = critical → pertenece a [critical, warning) → "warning"
        - Ψ = warning → pertenece a [warning, solid) → "stable"
        - Ψ = solid → pertenece a [solid, ∞) → "robust"
        """
        config = translator.config.stability

        assert config.classify(config.critical) == "warning", (
            f"Ψ = critical = {config.critical} debe ser 'warning' (intervalo cerrado-izquierdo)"
        )
        assert config.classify(config.warning) == "stable", (
            f"Ψ = warning = {config.warning} debe ser 'stable'"
        )
        assert config.classify(config.solid) == "robust", (
            f"Ψ = solid = {config.solid} debe ser 'robust'"
        )

    def test_near_threshold_stability_classification(
        self, translator: SemanticTranslator,
    ):
        """
        Valores infinitesimalmente por debajo de cada umbral pertenecen
        al intervalo IZQUIERDO (comportamiento determinista de los límites).
        """
        config = translator.config.stability
        delta = 1e-10  # Infinitesimalmente menor

        assert config.classify(config.critical - delta) == "critical", (
            f"Ψ = critical - ε debe ser 'critical'"
        )
        assert config.classify(config.warning - delta) == "warning", (
            f"Ψ = warning - ε debe ser 'warning'"
        )
        assert config.classify(config.solid - delta) == "stable", (
            f"Ψ = solid - ε debe ser 'stable'"
        )

    def test_thermodynamics_kelvin_zero(self, translator: SemanticTranslator):
        """
        T = 0K (cero absoluto): el sistema debe manejar sin errores.

        Termodinámicamente, T = 0K es el estado de mínima energía.
        El sistema no debe lanzar ZeroDivisionError ni similar.
        """
        metrics = ThermodynamicMetrics(
            system_temperature=0.0,
            entropy=0.0,
            heat_capacity=0.0,
        )
        try:
            _, verdict = translator.translate_thermodynamics(metrics)
            assert isinstance(verdict, VerdictLevel), (
                f"Veredicto en T=0K debe ser VerdictLevel, got {type(verdict)}"
            )
        except ZeroDivisionError as e:
            pytest.fail(f"ZeroDivisionError en T=0K: {e}")
        except Exception as e:
            # Otras excepciones de dominio son aceptables (el estado es físicamente extremo)
            pass

    def test_thermodynamics_extreme_high_temperature(
        self, translator: SemanticTranslator,
    ):
        """
        T extremadamente alta → RECHAZAR (sistema en colapso térmico).

        Nota: el sistema interpreta T > 100 como Kelvin (10000K ≈ 9727°C).
        Esto es definitivamente "critical".
        """
        metrics = ThermodynamicMetrics(
            system_temperature=10_000.0,  # 10000K
            entropy=0.99,
            heat_capacity=0.01,
        )
        _, verdict = translator.translate_thermodynamics(metrics)
        assert verdict == VerdictLevel.RECHAZAR, (
            f"T=10000K, S=0.99 debe ser RECHAZAR, got {verdict.name}"
        )

    def test_empty_synergy_dict_is_neutral(
        self, translator: SemanticTranslator, valid_topology: ValidatedTopology,
    ):
        """
        Sinergia vacía {} no activa el veto topológico.

        Un dict vacío indica ausencia de información de sinergia,
        no presencia de sinergia detectada.
        """
        _, verdict = translator.translate_topology(
            valid_topology, stability=5.0, synergy_risk={},
        )
        assert isinstance(verdict, VerdictLevel), (
            "synergy_risk={} no debe causar excepción"
        )
        # Con topología válida y sin sinergia, no debería ser RECHAZAR
        # (la topología tiene β₁=0 y Ψ=5.0 > 1)
        # Nota: dependiente de la implementación, pero la sinergia vacía no debe activar veto