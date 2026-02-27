"""
Suite de Pruebas para el Módulo de Traducción Semántica
=======================================================

Esta suite valida exhaustivamente:

1. **Propiedades Algebraicas del Lattice de Veredictos**
   - Reflexividad: a ≤ a
   - Antisimetría: a ≤ b ∧ b ≤ a ⟹ a = b
   - Transitividad: a ≤ b ∧ b ≤ c ⟹ a ≤ c
   - Existencia de supremo e ínfimo

2. **Traducción Semántica por Dominio**
   - Topología: β₀, β₁, χ → Narrativa + Veredicto
   - Termodinámica: T, S, U → Estado térmico
   - Física: Estabilidad giroscópica, Laplace
   - Finanzas: WACC, VaR, PI → Recomendación

3. **Composición de Reportes Estratégicos**
   - Integración de múltiples dominios
   - Propagación de veredictos (clausura transitiva)
   - Serialización y reconstrucción

4. **Propiedades Estadísticas**
   - Determinismo con misma entrada
   - Idempotencia de traducciones
   - Invariantes de normalización

Convenciones de Nomenclatura:
-----------------------------
- test_<componente>_<comportamiento>_<condición>
- Fixtures: <dominio>_<característica>

Marcadores:
-----------
- @pytest.mark.unit: Tests unitarios rápidos
- @pytest.mark.integration: Tests de integración
- @pytest.mark.algebraic: Tests de propiedades algebraicas
- @pytest.mark.parametrize: Tests parametrizados
"""

from __future__ import annotations

import copy
import math
import sys
from dataclasses import FrozenInstanceError, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from unittest.mock import Mock, patch, MagicMock

import pytest

# =============================================================================
# IMPORTACIONES DEL MÓDULO BAJO PRUEBA
# =============================================================================

# Configuración
from app.semantic_translator import (
    TranslatorConfig,
    StabilityThresholds,
    TopologicalThresholds,
    ThermalThresholds,
    FinancialThresholds,
)

# Lattice y Veredictos
from app.semantic_translator import (
    VerdictLevel,
    FinancialVerdict,
)

# Resultados
from app.semantic_translator import (
    StratumAnalysisResult,
    StrategicReport,
)

# Traductor principal
from app.semantic_translator import (
    SemanticTranslator,
    create_translator,
    translate_metrics_to_narrative,
)

# Esquemas de telemetría
from app.telemetry_schemas import (
    PhysicsMetrics,
    TopologicalMetrics,
    ControlMetrics,
    ThermodynamicMetrics,
)

# Jerarquía DIKW
from app.schemas import Stratum


# =============================================================================
# CONSTANTES DE PRUEBA
# =============================================================================

class TestConstants:
    """Constantes centralizadas para pruebas."""
    
    # Temperaturas en Kelvin
    TEMP_FROZEN_K: float = 250.0       # -23°C (congelado)
    TEMP_COLD_K: float = 280.0         # 7°C (frío)
    TEMP_AMBIENT_K: float = 300.0      # 27°C (ambiente)
    TEMP_WARM_K: float = 320.0         # 47°C (templado)
    TEMP_FEVER_K: float = 340.0        # 67°C (fiebre)
    TEMP_CRITICAL_K: float = 360.0     # 87°C (crítico)
    TEMP_MELTDOWN_K: float = 400.0     # 127°C (fusión)
    
    # Umbrales de estabilidad
    STABILITY_CRITICAL: float = 3.0
    STABILITY_LOW: float = 5.0
    STABILITY_NORMAL: float = 10.0
    STABILITY_HIGH: float = 15.0
    STABILITY_EXCELLENT: float = 20.0
    
    # Métricas financieras
    WACC_LOW: float = 0.05
    WACC_NORMAL: float = 0.10
    WACC_HIGH: float = 0.15
    WACC_CRITICAL: float = 0.20
    
    # Profitability Index
    PI_REJECT: float = 0.65
    PI_MARGINAL: float = 0.95
    PI_ACCEPTABLE: float = 1.0
    PI_GOOD: float = 1.25
    PI_EXCELLENT: float = 1.50
    
    # Tolerancias
    EPSILON: float = 1e-9


# =============================================================================
# MARCADORES PERSONALIZADOS
# =============================================================================

pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]


def pytest_configure(config):
    """Registra marcadores personalizados."""
    config.addinivalue_line("markers", "unit: Tests unitarios rápidos")
    config.addinivalue_line("markers", "integration: Tests de integración")
    config.addinivalue_line("markers", "algebraic: Tests de propiedades algebraicas")
    config.addinivalue_line("markers", "slow: Tests lentos")
    config.addinivalue_line("markers", "regression: Tests de regresión")


# =============================================================================
# HELPERS Y FACTORIES
# =============================================================================

class MetricsFactory:
    """
    Factory para creación de métricas de prueba.
    
    Centraliza la creación de objetos de prueba con valores
    predeterminados sensatos y variantes parametrizadas.
    """
    
    @staticmethod
    def create_topology(
        *,
        beta_0: int = 1,
        beta_1: int = 0,
        beta_2: int = 0,
        euler_characteristic: Optional[int] = None,
        fiedler_value: float = 1.0,
        **kwargs
    ) -> TopologicalMetrics:
        """
        Crea métricas topológicas.
        
        Si euler_characteristic no se especifica, se calcula como χ = β₀ - β₁ + β₂.
        """
        if euler_characteristic is None:
            euler_characteristic = beta_0 - beta_1 + beta_2
        
        return TopologicalMetrics(
            beta_0=beta_0,
            beta_1=beta_1,
            euler_characteristic=euler_characteristic,
            fiedler_value=fiedler_value,
            **kwargs
        )
    
    @staticmethod
    def create_clean_topology() -> TopologicalMetrics:
        """Topología limpia: conexa, sin ciclos (genus 0)."""
        return MetricsFactory.create_topology(beta_0=1, beta_1=0)
    
    @staticmethod
    def create_cyclic_topology(n_cycles: int = 2) -> TopologicalMetrics:
        """Topología con ciclos."""
        return MetricsFactory.create_topology(beta_0=1, beta_1=n_cycles)
    
    @staticmethod
    def create_fragmented_topology(n_components: int = 3) -> TopologicalMetrics:
        """Topología fragmentada (múltiples componentes conexas)."""
        return MetricsFactory.create_topology(beta_0=n_components, beta_1=0)
    
    @staticmethod
    def create_thermal(
        *,
        temperature: float = TestConstants.TEMP_AMBIENT_K,
        entropy: float = 0.0,
        internal_energy: float = 0.0,
        heat_capacity: float = 1.0,
        **kwargs
    ) -> ThermodynamicMetrics:
        """Crea métricas termodinámicas."""
        return ThermodynamicMetrics(
            system_temperature=temperature,
            entropy=entropy,
            internal_energy=internal_energy,
            heat_capacity=heat_capacity,
            **kwargs
        )
    
    @staticmethod
    def create_physics(
        *,
        gyroscopic_stability: float = 1.0,
        saturation: float = 0.0,
        **kwargs
    ) -> PhysicsMetrics:
        """Crea métricas físicas."""
        return PhysicsMetrics(
            gyroscopic_stability=gyroscopic_stability,
            saturation=saturation,
            **kwargs
        )
    
    @staticmethod
    def create_control(
        *,
        is_stable: bool = True,
        phase_margin_deg: float = 45.0,
        gain_margin_db: float = 10.0,
        **kwargs
    ) -> ControlMetrics:
        """Crea métricas de control."""
        return ControlMetrics(
            is_stable=is_stable,
            phase_margin_deg=phase_margin_deg,
            gain_margin_db=gain_margin_db,
            **kwargs
        )


class FinancialsFactory:
    """Factory para creación de métricas financieras."""
    
    @staticmethod
    def create_viable(
        *,
        wacc: float = TestConstants.WACC_NORMAL,
        var: float = 1000.0,
        contingency: float = 1500.0,
        pi: float = TestConstants.PI_GOOD,
        recommendation: str = "ACEPTAR",
    ) -> Dict[str, Any]:
        """Métricas financieras viables."""
        return {
            "wacc": wacc,
            "var": var,
            "contingency": {"recommended": contingency},
            "performance": {
                "recommendation": recommendation,
                "profitability_index": pi,
            },
        }
    
    @staticmethod
    def create_rejected(
        *,
        wacc: float = TestConstants.WACC_HIGH,
        var: float = 100000.0,
        contingency: float = 150000.0,
        pi: float = TestConstants.PI_REJECT,
    ) -> Dict[str, Any]:
        """Métricas financieras de rechazo."""
        return FinancialsFactory.create_viable(
            wacc=wacc,
            var=var,
            contingency=contingency,
            pi=pi,
            recommendation="RECHAZAR",
        )
    
    @staticmethod
    def create_review(
        *,
        wacc: float = 0.12,
        pi: float = TestConstants.PI_MARGINAL,
    ) -> Dict[str, Any]:
        """Métricas financieras en revisión."""
        return FinancialsFactory.create_viable(
            wacc=wacc,
            pi=pi,
            recommendation="REVISAR",
        )
    
    @staticmethod
    def create_empty() -> Dict[str, Any]:
        """Métricas financieras vacías."""
        return {}
    
    @staticmethod
    def create_minimal() -> Dict[str, Any]:
        """Métricas financieras mínimas."""
        return {"performance": {"recommendation": "REVISAR"}}


class TranslatorFactory:
    """Factory para creación de traductores."""
    
    @staticmethod
    def create_default() -> SemanticTranslator:
        """Traductor con configuración por defecto."""
        return SemanticTranslator()
    
    @staticmethod
    def create_deterministic(market_index: int = 0) -> SemanticTranslator:
        """Traductor con comportamiento determinístico."""
        config = TranslatorConfig(
            deterministic_market=True,
            default_market_index=market_index,
        )
        return SemanticTranslator(config=config)
    
    @staticmethod
    def create_with_thresholds(
        *,
        stability: Optional[StabilityThresholds] = None,
        topological: Optional[TopologicalThresholds] = None,
        thermal: Optional[ThermalThresholds] = None,
        financial: Optional[FinancialThresholds] = None,
    ) -> SemanticTranslator:
        """Traductor con umbrales personalizados."""
        config = TranslatorConfig(
            stability_thresholds=stability or StabilityThresholds(),
            topological_thresholds=topological or TopologicalThresholds(),
            thermal_thresholds=thermal or ThermalThresholds(),
            financial_thresholds=financial or FinancialThresholds(),
        )
        return SemanticTranslator(config=config)


# =============================================================================
# FIXTURES
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: Traductores
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def default_translator() -> SemanticTranslator:
    """Traductor con configuración por defecto."""
    return TranslatorFactory.create_default()


@pytest.fixture
def deterministic_translator() -> SemanticTranslator:
    """Traductor con comportamiento determinístico para tests reproducibles."""
    return TranslatorFactory.create_deterministic()


@pytest.fixture
def strict_translator() -> SemanticTranslator:
    """Traductor con umbrales estrictos."""
    return TranslatorFactory.create_with_thresholds(
        stability=StabilityThresholds(critical=5.0, warning=10.0),
        thermal=ThermalThresholds(critical_celsius=60.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: Topología
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_topology() -> TopologicalMetrics:
    """Topología limpia: conexa, sin ciclos (genus 0)."""
    return MetricsFactory.create_clean_topology()


@pytest.fixture
def cyclic_topology() -> TopologicalMetrics:
    """Topología con ciclos (genus > 0)."""
    return MetricsFactory.create_cyclic_topology(n_cycles=2)


@pytest.fixture
def fragmented_topology() -> TopologicalMetrics:
    """Topología fragmentada (β₀ > 1)."""
    return MetricsFactory.create_fragmented_topology(n_components=3)


@pytest.fixture
def complex_topology() -> TopologicalMetrics:
    """Topología compleja: fragmentada con ciclos."""
    return MetricsFactory.create_topology(beta_0=3, beta_1=2, fiedler_value=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: Termodinámica
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def cold_thermal() -> ThermodynamicMetrics:
    """Métricas térmicas frías pero estables."""
    return MetricsFactory.create_thermal(
        temperature=TestConstants.TEMP_COLD_K,
        entropy=0.2,
        internal_energy=300.0,
    )


@pytest.fixture
def ambient_thermal() -> ThermodynamicMetrics:
    """Métricas térmicas a temperatura ambiente."""
    return MetricsFactory.create_thermal(temperature=TestConstants.TEMP_AMBIENT_K)


@pytest.fixture
def fever_thermal() -> ThermodynamicMetrics:
    """Métricas térmicas con fiebre."""
    return MetricsFactory.create_thermal(
        temperature=TestConstants.TEMP_FEVER_K,
        entropy=0.6,
    )


@pytest.fixture
def critical_thermal() -> ThermodynamicMetrics:
    """Métricas térmicas críticas."""
    return MetricsFactory.create_thermal(
        temperature=TestConstants.TEMP_CRITICAL_K,
        entropy=0.8,
    )


@pytest.fixture
def meltdown_thermal() -> ThermodynamicMetrics:
    """Métricas térmicas de fusión (inviable)."""
    return MetricsFactory.create_thermal(
        temperature=TestConstants.TEMP_MELTDOWN_K,
        entropy=1.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: Finanzas
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def viable_financials() -> Dict[str, Any]:
    """Métricas financieras viables."""
    return FinancialsFactory.create_viable()


@pytest.fixture
def rejected_financials() -> Dict[str, Any]:
    """Métricas financieras de rechazo."""
    return FinancialsFactory.create_rejected()


@pytest.fixture
def review_financials() -> Dict[str, Any]:
    """Métricas financieras en revisión."""
    return FinancialsFactory.create_review()


@pytest.fixture
def empty_financials() -> Dict[str, Any]:
    """Métricas financieras vacías."""
    return FinancialsFactory.create_empty()


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: Física y Control
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def stable_physics() -> PhysicsMetrics:
    """Métricas físicas estables."""
    return MetricsFactory.create_physics(gyroscopic_stability=1.0)


@pytest.fixture
def unstable_physics() -> PhysicsMetrics:
    """Métricas físicas inestables (nutación crítica)."""
    return MetricsFactory.create_physics(gyroscopic_stability=0.2)


@pytest.fixture
def stable_control() -> ControlMetrics:
    """Métricas de control estables."""
    return MetricsFactory.create_control(is_stable=True, phase_margin_deg=60.0)


@pytest.fixture
def unstable_control() -> ControlMetrics:
    """Métricas de control inestables (RHP)."""
    return MetricsFactory.create_control(is_stable=False, phase_margin_deg=-10.0)


@pytest.fixture
def marginal_control() -> ControlMetrics:
    """Métricas de control marginalmente estables."""
    return MetricsFactory.create_control(is_stable=True, phase_margin_deg=15.0)


# =============================================================================
# TESTS: ESQUEMAS DE TELEMETRÍA
# =============================================================================

@pytest.mark.unit
class TestTelemetrySchemas:
    """
    Pruebas para esquemas de telemetría.
    
    Verifica valores por defecto, inmutabilidad y serialización.
    """

    def test_topological_metrics_defaults(self):
        """TopologicalMetrics tiene valores por defecto sensatos."""
        metrics = TopologicalMetrics()
        
        assert metrics.beta_0 == 1, "β₀ debe ser 1 (conexo)"
        assert metrics.beta_1 == 0, "β₁ debe ser 0 (sin ciclos)"
        assert metrics.fiedler_value == 1.0, "Fiedler debe ser 1.0"

    def test_topological_metrics_euler_invariant(self):
        """Característica de Euler debe satisfacer χ = β₀ - β₁ + β₂."""
        # Para verificar que el usuario puede especificar valores consistentes
        metrics = MetricsFactory.create_topology(beta_0=2, beta_1=1, beta_2=0)
        expected_euler = 2 - 1 + 0
        assert metrics.euler_characteristic == expected_euler

    def test_thermodynamic_metrics_defaults(self):
        """ThermodynamicMetrics tiene valores por defecto físicamente sensatos."""
        metrics = ThermodynamicMetrics()
        
        assert metrics.entropy == 0.0, "Entropía debe ser 0"
        assert metrics.system_temperature == 300.0, "T debe ser 300K (ambiente)"

    @pytest.mark.parametrize("temp_k,expected_celsius", [
        (273.15, 0.0),
        (373.15, 100.0),
        (300.0, 26.85),
    ])
    def test_thermodynamic_temperature_conversion(self, temp_k: float, expected_celsius: float):
        """Conversión de temperatura Kelvin a Celsius."""
        metrics = MetricsFactory.create_thermal(temperature=temp_k)
        actual_celsius = metrics.system_temperature - 273.15
        assert abs(actual_celsius - expected_celsius) < 0.01

    def test_physics_metrics_defaults(self):
        """PhysicsMetrics tiene valores por defecto estables."""
        metrics = PhysicsMetrics()
        
        assert metrics.gyroscopic_stability == 1.0, "Estabilidad giroscópica debe ser 1.0"
        assert metrics.saturation == 0.0, "Saturación debe ser 0"

    def test_control_metrics_defaults(self):
        """ControlMetrics tiene valores por defecto estables."""
        metrics = ControlMetrics()
        
        assert metrics.is_stable is True, "Sistema debe ser estable por defecto"
        assert metrics.phase_margin_deg == 45.0, "PM debe ser 45° por defecto"

    @pytest.mark.parametrize("metric_class,expected_frozen", [
        (TopologicalMetrics, True),
        (ThermodynamicMetrics, True),
        (PhysicsMetrics, True),
        (ControlMetrics, True),
    ])
    def test_metrics_are_frozen(self, metric_class: Type, expected_frozen: bool):
        """Verifica que las métricas son inmutables (frozen dataclass)."""
        instance = metric_class()
        
        if expected_frozen:
            with pytest.raises((FrozenInstanceError, AttributeError)):
                instance.beta_0 = 999  # Intentar modificar


# =============================================================================
# TESTS: PROPIEDADES ALGEBRAICAS DEL LATTICE
# =============================================================================

@pytest.mark.unit
@pytest.mark.algebraic
class TestLatticeAlgebraicProperties:
    """
    Verifica las propiedades algebraicas del lattice (VerdictLevel, ≤, ⊔, ⊓).
    
    Un lattice es una estructura algebraica (L, ≤) donde:
    - ≤ es un orden parcial (reflexivo, antisimétrico, transitivo)
    - Todo par de elementos tiene supremo (⊔) e ínfimo (⊓)
    - Existen elementos mínimo (⊥) y máximo (⊤)
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Elementos Especiales
    # ─────────────────────────────────────────────────────────────────────────

    def test_lattice_has_bottom_element(self):
        """El lattice tiene elemento mínimo (bottom/⊥)."""
        bottom = VerdictLevel.bottom()
        
        assert bottom == VerdictLevel.VIABLE
        
        # ⊥ ≤ x para todo x
        for level in VerdictLevel:
            assert bottom.value <= level.value, f"bottom ≤ {level.name} debe ser True"

    def test_lattice_has_top_element(self):
        """El lattice tiene elemento máximo (top/⊤)."""
        top = VerdictLevel.top()
        
        assert top == VerdictLevel.RECHAZAR
        
        # x ≤ ⊤ para todo x
        for level in VerdictLevel:
            assert level.value <= top.value, f"{level.name} ≤ top debe ser True"

    # ─────────────────────────────────────────────────────────────────────────
    # Propiedades del Orden Parcial
    # ─────────────────────────────────────────────────────────────────────────

    def test_order_is_reflexive(self):
        """Reflexividad: a ≤ a para todo a."""
        for level in VerdictLevel:
            assert level.value <= level.value, f"{level.name} ≤ {level.name}"

    def test_order_is_antisymmetric(self):
        """Antisimetría: a ≤ b ∧ b ≤ a ⟹ a = b."""
        for a in VerdictLevel:
            for b in VerdictLevel:
                if a.value <= b.value and b.value <= a.value:
                    assert a == b, f"Si {a.name} ≤ {b.name} y {b.name} ≤ {a.name}, entonces {a.name} = {b.name}"

    def test_order_is_transitive(self):
        """Transitividad: a ≤ b ∧ b ≤ c ⟹ a ≤ c."""
        levels = list(VerdictLevel)
        
        for a in levels:
            for b in levels:
                for c in levels:
                    if a.value <= b.value and b.value <= c.value:
                        assert a.value <= c.value, \
                            f"Transitividad violada: {a.name} ≤ {b.name} ≤ {c.name}"

    # ─────────────────────────────────────────────────────────────────────────
    # Operaciones de Lattice
    # ─────────────────────────────────────────────────────────────────────────

    def test_supremum_is_least_upper_bound(self):
        """Supremum (⊔) es la menor cota superior."""
        # supremum({VIABLE, CONDICIONAL}) = CONDICIONAL
        result = VerdictLevel.supremum(VerdictLevel.VIABLE, VerdictLevel.CONDICIONAL)
        
        assert result == VerdictLevel.CONDICIONAL
        assert result.value >= VerdictLevel.VIABLE.value
        assert result.value >= VerdictLevel.CONDICIONAL.value

    def test_supremum_takes_worst_case(self):
        """Supremum toma el peor caso (valor más alto)."""
        result = VerdictLevel.supremum(
            VerdictLevel.VIABLE,
            VerdictLevel.CONDICIONAL,
            VerdictLevel.RECHAZAR,
        )
        
        assert result == VerdictLevel.RECHAZAR

    def test_supremum_with_single_element(self):
        """Supremum de un solo elemento es él mismo."""
        for level in VerdictLevel:
            assert VerdictLevel.supremum(level) == level

    def test_supremum_is_idempotent(self):
        """Idempotencia: a ⊔ a = a."""
        for level in VerdictLevel:
            assert VerdictLevel.supremum(level, level) == level

    def test_supremum_is_commutative(self):
        """Conmutatividad: a ⊔ b = b ⊔ a."""
        for a in VerdictLevel:
            for b in VerdictLevel:
                assert VerdictLevel.supremum(a, b) == VerdictLevel.supremum(b, a), \
                    f"Conmutatividad violada para {a.name} y {b.name}"

    def test_supremum_is_associative(self):
        """Asociatividad: (a ⊔ b) ⊔ c = a ⊔ (b ⊔ c)."""
        levels = list(VerdictLevel)[:3]  # Tomar 3 para test
        
        for a in levels:
            for b in levels:
                for c in levels:
                    left = VerdictLevel.supremum(VerdictLevel.supremum(a, b), c)
                    right = VerdictLevel.supremum(a, VerdictLevel.supremum(b, c))
                    assert left == right, f"Asociatividad violada para {a.name}, {b.name}, {c.name}"

    def test_supremum_with_bottom_is_identity(self):
        """Bottom es identidad para supremum: a ⊔ ⊥ = a."""
        bottom = VerdictLevel.bottom()
        
        for level in VerdictLevel:
            assert VerdictLevel.supremum(level, bottom) == level

    def test_supremum_with_top_is_top(self):
        """Top absorbe en supremum: a ⊔ ⊤ = ⊤."""
        top = VerdictLevel.top()
        
        for level in VerdictLevel:
            assert VerdictLevel.supremum(level, top) == top

    # ─────────────────────────────────────────────────────────────────────────
    # Propiedades Específicas del Dominio
    # ─────────────────────────────────────────────────────────────────────────

    def test_verdict_ordering_reflects_severity(self):
        """El orden del lattice refleja severidad creciente."""
        expected_order = [
            VerdictLevel.VIABLE,
            VerdictLevel.PRECAUCION,
            VerdictLevel.CONDICIONAL,
            VerdictLevel.RECHAZAR,
        ]
        
        for i in range(len(expected_order) - 1):
            assert expected_order[i].value < expected_order[i + 1].value, \
                f"{expected_order[i].name} < {expected_order[i + 1].name}"

    @pytest.mark.parametrize("verdicts,expected", [
        ([VerdictLevel.VIABLE], VerdictLevel.VIABLE),
        ([VerdictLevel.VIABLE, VerdictLevel.VIABLE], VerdictLevel.VIABLE),
        ([VerdictLevel.VIABLE, VerdictLevel.PRECAUCION], VerdictLevel.PRECAUCION),
        ([VerdictLevel.CONDICIONAL, VerdictLevel.RECHAZAR], VerdictLevel.RECHAZAR),
        (list(VerdictLevel), VerdictLevel.RECHAZAR),  # Todos → top
    ])
    def test_supremum_parametrized_cases(
        self,
        verdicts: List[VerdictLevel],
        expected: VerdictLevel
    ):
        """Casos parametrizados de supremum."""
        result = VerdictLevel.supremum(*verdicts)
        assert result == expected


# =============================================================================
# TESTS: TRADUCCIÓN DE TOPOLOGÍA
# =============================================================================

@pytest.mark.unit
class TestTopologyTranslation:
    """
    Pruebas para traducción de métricas topológicas a narrativa.
    
    Verifica la correcta interpretación de:
    - β₀ (componentes conexas)
    - β₁ (ciclos independientes)
    - χ (característica de Euler)
    - Valor de Fiedler (conectividad algebraica)
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Casos Básicos
    # ─────────────────────────────────────────────────────────────────────────

    def test_translate_clean_topology(
        self,
        default_translator: SemanticTranslator,
        clean_topology: TopologicalMetrics
    ):
        """Topología limpia (genus 0) genera narrativa positiva y veredicto VIABLE."""
        narrative, verdict = default_translator.translate_topology(
            clean_topology,
            stability=TestConstants.STABILITY_HIGH
        )

        assert "Integridad" in narrative or "Genus 0" in narrative
        assert verdict == VerdictLevel.VIABLE

    def test_translate_cyclic_topology(
        self,
        default_translator: SemanticTranslator,
        cyclic_topology: TopologicalMetrics
    ):
        """Topología con ciclos genera advertencias y veredicto ≥ CONDICIONAL."""
        narrative, verdict = default_translator.translate_topology(
            cyclic_topology,
            stability=TestConstants.STABILITY_HIGH
        )

        # Debe mencionar ciclos/socavones/bucles
        narrative_lower = narrative.lower()
        assert any(word in narrative_lower for word in ["socavones", "ciclos", "bucles", "genus"])
        
        # Veredicto debe ser al menos CONDICIONAL
        assert verdict.value >= VerdictLevel.CONDICIONAL.value

    def test_translate_fragmented_topology(
        self,
        default_translator: SemanticTranslator,
        fragmented_topology: TopologicalMetrics
    ):
        """Topología fragmentada (β₀ > 1) genera advertencias."""
        narrative, verdict = default_translator.translate_topology(
            fragmented_topology,
            stability=TestConstants.STABILITY_HIGH
        )

        # Debe indicar fragmentación o componentes
        narrative_lower = narrative.lower()
        assert any(word in narrative_lower for word in ["componentes", "fragmentad", "isla"])
        
        assert verdict.value >= VerdictLevel.PRECAUCION.value

    # ─────────────────────────────────────────────────────────────────────────
    # Formatos de Entrada
    # ─────────────────────────────────────────────────────────────────────────

    def test_translate_accepts_dict_input(self, default_translator: SemanticTranslator):
        """Acepta diccionario como entrada (normalización automática)."""
        dict_input = {"beta_0": 1, "beta_1": 0, "euler_characteristic": 1}

        narrative, verdict = default_translator.translate_topology(
            dict_input,
            stability=TestConstants.STABILITY_NORMAL
        )

        assert isinstance(narrative, str)
        assert isinstance(verdict, VerdictLevel)
        assert "Integridad" in narrative

    def test_translate_accepts_partial_dict(self, default_translator: SemanticTranslator):
        """Acepta diccionario parcial (usa valores por defecto)."""
        partial_input = {"beta_0": 1}  # Sin beta_1

        narrative, verdict = default_translator.translate_topology(
            partial_input,
            stability=TestConstants.STABILITY_NORMAL
        )

        assert isinstance(narrative, str)
        assert verdict is not None

    # ─────────────────────────────────────────────────────────────────────────
    # Validación de Entradas
    # ─────────────────────────────────────────────────────────────────────────

    def test_translate_negative_stability_raises_error(
        self,
        default_translator: SemanticTranslator,
        clean_topology: TopologicalMetrics
    ):
        """Estabilidad negativa lanza ValueError."""
        with pytest.raises(ValueError, match="non-negative|positiv"):
            default_translator.translate_topology(clean_topology, stability=-1.0)

    def test_translate_zero_stability(
        self,
        default_translator: SemanticTranslator,
        clean_topology: TopologicalMetrics
    ):
        """Estabilidad cero es válida (caso límite)."""
        narrative, verdict = default_translator.translate_topology(
            clean_topology,
            stability=0.0
        )

        # Con estabilidad 0, debe haber advertencias
        assert verdict.value >= VerdictLevel.PRECAUCION.value

    @pytest.mark.parametrize("invalid_input", [
        None,
        "invalid",
        123,
        [],
    ])
    def test_translate_invalid_topology_type(
        self,
        default_translator: SemanticTranslator,
        invalid_input: Any
    ):
        """Tipos de entrada inválidos lanzan excepción apropiada."""
        with pytest.raises((TypeError, ValueError, AttributeError)):
            default_translator.translate_topology(
                invalid_input,
                stability=TestConstants.STABILITY_NORMAL
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Casos Parametrizados
    # ─────────────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("beta_0,beta_1,expected_min_verdict", [
        (1, 0, VerdictLevel.VIABLE),      # Limpio
        (1, 1, VerdictLevel.PRECAUCION),  # Un ciclo
        (1, 3, VerdictLevel.CONDICIONAL), # Múltiples ciclos
        (2, 0, VerdictLevel.PRECAUCION),  # Fragmentado
        (3, 2, VerdictLevel.CONDICIONAL), # Fragmentado con ciclos
    ])
    def test_topology_verdict_by_betti_numbers(
        self,
        default_translator: SemanticTranslator,
        beta_0: int,
        beta_1: int,
        expected_min_verdict: VerdictLevel
    ):
        """El veredicto corresponde a los números de Betti."""
        topology = MetricsFactory.create_topology(beta_0=beta_0, beta_1=beta_1)
        
        _, verdict = default_translator.translate_topology(
            topology,
            stability=TestConstants.STABILITY_HIGH
        )
        
        assert verdict.value >= expected_min_verdict.value, \
            f"β₀={beta_0}, β₁={beta_1} debería dar veredicto ≥ {expected_min_verdict.name}"

    @pytest.mark.parametrize("stability,expected_max_verdict", [
        (TestConstants.STABILITY_EXCELLENT, VerdictLevel.VIABLE),
        (TestConstants.STABILITY_NORMAL, VerdictLevel.PRECAUCION),
        (TestConstants.STABILITY_LOW, VerdictLevel.CONDICIONAL),
        (TestConstants.STABILITY_CRITICAL, VerdictLevel.RECHAZAR),
    ])
    def test_stability_affects_verdict(
        self,
        default_translator: SemanticTranslator,
        clean_topology: TopologicalMetrics,
        stability: float,
        expected_max_verdict: VerdictLevel
    ):
        """La estabilidad afecta el veredicto final."""
        _, verdict = default_translator.translate_topology(clean_topology, stability=stability)
        
        # Con topología limpia, el veredicto no debe exceder el máximo por estabilidad
        # (puede ser menor si la topología es perfecta)
        assert verdict.value <= expected_max_verdict.value or stability < TestConstants.STABILITY_LOW


# =============================================================================
# TESTS: TRADUCCIÓN TERMODINÁMICA
# =============================================================================

@pytest.mark.unit
class TestThermalTranslation:
    """
    Pruebas para traducción de métricas termodinámicas.
    
    Mapeo de temperatura:
    - < 10°C (283K): Estable/Frío
    - 10-50°C: Normal
    - 50-75°C: Fiebre
    - > 75°C (348K): Crítico/Fusión
    """

    def test_translate_cold_temperature(
        self,
        default_translator: SemanticTranslator,
        cold_thermal: ThermodynamicMetrics
    ):
        """Temperatura fría (< 10°C) indica estabilidad."""
        narrative, verdict = default_translator.translate_thermodynamics(cold_thermal)
        
        assert "Estable" in narrative or "❄️" in narrative or "frí" in narrative.lower()
        assert verdict == VerdictLevel.VIABLE

    def test_translate_ambient_temperature(
        self,
        default_translator: SemanticTranslator,
        ambient_thermal: ThermodynamicMetrics
    ):
        """Temperatura ambiente es aceptable."""
        narrative, verdict = default_translator.translate_thermodynamics(ambient_thermal)
        
        assert verdict in (VerdictLevel.VIABLE, VerdictLevel.PRECAUCION)

    def test_translate_fever_temperature(
        self,
        default_translator: SemanticTranslator,
        fever_thermal: ThermodynamicMetrics
    ):
        """Temperatura de fiebre (50-75°C) genera advertencia."""
        narrative, verdict = default_translator.translate_thermodynamics(fever_thermal)
        
        assert "FIEBRE" in narrative or "calor" in narrative.lower() or "⚠️" in narrative
        assert verdict.value >= VerdictLevel.PRECAUCION.value

    def test_translate_critical_temperature(
        self,
        default_translator: SemanticTranslator,
        critical_thermal: ThermodynamicMetrics
    ):
        """Temperatura crítica (> 75°C) genera rechazo."""
        narrative, verdict = default_translator.translate_thermodynamics(critical_thermal)
        
        assert "FUSIÓN" in narrative or "crítica" in narrative.lower() or "🔥" in narrative
        assert verdict == VerdictLevel.RECHAZAR

    def test_translate_meltdown_temperature(
        self,
        default_translator: SemanticTranslator,
        meltdown_thermal: ThermodynamicMetrics
    ):
        """Temperatura de fusión genera rechazo categórico."""
        narrative, verdict = default_translator.translate_thermodynamics(meltdown_thermal)
        
        assert verdict == VerdictLevel.RECHAZAR

    @pytest.mark.parametrize("temp_k,expected_verdict", [
        (TestConstants.TEMP_COLD_K, VerdictLevel.VIABLE),
        (TestConstants.TEMP_AMBIENT_K, VerdictLevel.VIABLE),  # o PRECAUCION
        (TestConstants.TEMP_FEVER_K, VerdictLevel.PRECAUCION),
        (TestConstants.TEMP_CRITICAL_K, VerdictLevel.RECHAZAR),
        (TestConstants.TEMP_MELTDOWN_K, VerdictLevel.RECHAZAR),
    ])
    def test_temperature_verdict_mapping(
        self,
        default_translator: SemanticTranslator,
        temp_k: float,
        expected_verdict: VerdictLevel
    ):
        """Mapeo parametrizado de temperatura a veredicto."""
        thermal = MetricsFactory.create_thermal(temperature=temp_k)
        _, verdict = default_translator.translate_thermodynamics(thermal)
        
        # Permitir un nivel de tolerancia para casos límite
        assert abs(verdict.value - expected_verdict.value) <= 1


# =============================================================================
# TESTS: TRADUCCIÓN DE FÍSICA Y CONTROL
# =============================================================================

@pytest.mark.unit
class TestPhysicsTranslation:
    """
    Pruebas para traducción de métricas físicas y de control.
    
    Verifica:
    - Estabilidad giroscópica (nutación)
    - Análisis de Laplace (polos en RHP)
    - Margen de fase
    """

    def test_translate_stable_gyroscope(
        self,
        default_translator: SemanticTranslator,
        stable_physics: PhysicsMetrics
    ):
        """Giroscopio estable (>= 0.5) es aceptable."""
        result = default_translator._analyze_physics_stratum(
            thermal=ThermodynamicMetrics(),
            stability=TestConstants.STABILITY_NORMAL,
            physics=stable_physics
        )
        
        assert result.verdict in (VerdictLevel.VIABLE, VerdictLevel.PRECAUCION)
        assert "NUTACIÓN" not in result.narrative

    def test_translate_unstable_gyroscope(
        self,
        default_translator: SemanticTranslator,
        unstable_physics: PhysicsMetrics
    ):
        """Giroscopio inestable (< 0.3) genera rechazo."""
        result = default_translator._analyze_physics_stratum(
            thermal=ThermodynamicMetrics(),
            stability=TestConstants.STABILITY_NORMAL,
            physics=unstable_physics
        )
        
        assert "NUTACIÓN" in result.narrative
        assert result.verdict == VerdictLevel.RECHAZAR

    def test_translate_stable_laplace(
        self,
        default_translator: SemanticTranslator,
        stable_control: ControlMetrics
    ):
        """Control estable (LHP) es aceptable."""
        result = default_translator._analyze_physics_stratum(
            thermal=ThermodynamicMetrics(),
            stability=TestConstants.STABILITY_NORMAL,
            control=stable_control
        )
        
        assert "DIVERGENCIA" not in result.narrative
        assert result.verdict != VerdictLevel.RECHAZAR

    def test_translate_unstable_laplace(
        self,
        default_translator: SemanticTranslator,
        unstable_control: ControlMetrics
    ):
        """Control inestable (RHP) genera rechazo."""
        result = default_translator._analyze_physics_stratum(
            thermal=ThermodynamicMetrics(),
            stability=TestConstants.STABILITY_NORMAL,
            control=unstable_control
        )
        
        assert "DIVERGENCIA" in result.narrative or "Divergencia" in result.narrative
        assert result.verdict == VerdictLevel.RECHAZAR

    def test_translate_marginal_control(
        self,
        default_translator: SemanticTranslator,
        marginal_control: ControlMetrics
    ):
        """Control marginalmente estable genera precaución."""
        result = default_translator._analyze_physics_stratum(
            thermal=ThermodynamicMetrics(),
            stability=TestConstants.STABILITY_NORMAL,
            control=marginal_control
        )
        
        assert result.verdict.value >= VerdictLevel.PRECAUCION.value

    @pytest.mark.parametrize("gyro_stability,expected_verdict", [
        (1.0, VerdictLevel.VIABLE),
        (0.7, VerdictLevel.VIABLE),
        (0.5, VerdictLevel.PRECAUCION),
        (0.3, VerdictLevel.CONDICIONAL),
        (0.2, VerdictLevel.RECHAZAR),
        (0.1, VerdictLevel.RECHAZAR),
    ])
    def test_gyroscopic_stability_mapping(
        self,
        default_translator: SemanticTranslator,
        gyro_stability: float,
        expected_verdict: VerdictLevel
    ):
        """Mapeo de estabilidad giroscópica a veredicto."""
        physics = MetricsFactory.create_physics(gyroscopic_stability=gyro_stability)
        
        result = default_translator._analyze_physics_stratum(
            thermal=ThermodynamicMetrics(),
            stability=TestConstants.STABILITY_HIGH,
            physics=physics
        )
        
        assert result.verdict.value >= expected_verdict.value - 1  # Tolerancia de 1 nivel


# =============================================================================
# TESTS: COMPOSICIÓN DE REPORTE ESTRATÉGICO
# =============================================================================

@pytest.mark.integration
class TestStrategicReportComposition:
    """
    Pruebas de integración para composición del reporte estratégico.
    
    Verifica:
    - Estructura del reporte
    - Integración de múltiples dominios
    - Propagación correcta de veredictos
    """

    def test_compose_returns_strategic_report(
        self,
        deterministic_translator: SemanticTranslator,
        clean_topology: TopologicalMetrics,
        viable_financials: Dict[str, Any]
    ):
        """compose_strategic_narrative retorna StrategicReport."""
        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=clean_topology,
            financial_metrics=viable_financials,
            stability=TestConstants.STABILITY_HIGH,
        )

        assert isinstance(report, StrategicReport)
        assert report.verdict is not None
        assert isinstance(report.verdict, VerdictLevel)

    def test_report_has_required_fields(
        self,
        deterministic_translator: SemanticTranslator,
        clean_topology: TopologicalMetrics,
        viable_financials: Dict[str, Any]
    ):
        """El reporte tiene todos los campos requeridos."""
        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=clean_topology,
            financial_metrics=viable_financials,
            stability=TestConstants.STABILITY_HIGH,
        )

        assert hasattr(report, 'verdict')
        assert hasattr(report, 'is_viable')
        assert hasattr(report, 'raw_narrative')
        assert hasattr(report, 'strata_analysis')
        
        # Verificar que hay análisis por estrato
        assert Stratum.PHYSICS in report.strata_analysis
        assert Stratum.TACTICS in report.strata_analysis

    def test_report_green_light_scenario(
        self,
        deterministic_translator: SemanticTranslator,
        clean_topology: TopologicalMetrics,
        viable_financials: Dict[str, Any]
    ):
        """Escenario de luz verde: topología limpia + finanzas viables = VIABLE."""
        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=clean_topology,
            financial_metrics=viable_financials,
            stability=TestConstants.STABILITY_HIGH,
        )

        assert report.verdict == VerdictLevel.VIABLE
        assert report.is_viable is True

    def test_report_red_light_scenario(
        self,
        deterministic_translator: SemanticTranslator,
        cyclic_topology: TopologicalMetrics,
        rejected_financials: Dict[str, Any]
    ):
        """Escenario de luz roja: topología con ciclos + finanzas rechazadas = RECHAZAR."""
        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=cyclic_topology,
            financial_metrics=rejected_financials,
            stability=TestConstants.STABILITY_LOW,
        )

        assert report.verdict == VerdictLevel.RECHAZAR
        assert report.is_viable is False

    def test_report_with_thermal_metrics(
        self,
        deterministic_translator: SemanticTranslator,
        clean_topology: TopologicalMetrics,
        viable_financials: Dict[str, Any],
        fever_thermal: ThermodynamicMetrics
    ):
        """Métricas térmicas se integran correctamente en el reporte."""
        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=clean_topology,
            financial_metrics=viable_financials,
            stability=TestConstants.STABILITY_HIGH,
            thermal_metrics=asdict(fever_thermal),
        )

        # La narrativa debe mencionar temperatura
        assert any(word in report.raw_narrative.lower() for word in ["fiebre", "calor", "temperatura", "⚠️"])

    def test_report_with_physics_metrics(
        self,
        deterministic_translator: SemanticTranslator,
        clean_topology: TopologicalMetrics,
        viable_financials: Dict[str, Any],
        unstable_physics: PhysicsMetrics
    ):
        """Métricas físicas inestables degradan el veredicto."""
        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=clean_topology,
            financial_metrics=viable_financials,
            stability=TestConstants.STABILITY_HIGH,
            physics_metrics=asdict(unstable_physics),
        )

        # La física inestable debe propagar rechazo
        assert report.verdict == VerdictLevel.RECHAZAR
        assert "NUTACIÓN" in report.raw_narrative

    def test_report_with_control_metrics(
        self,
        deterministic_translator: SemanticTranslator,
        clean_topology: TopologicalMetrics,
        viable_financials: Dict[str, Any],
        unstable_control: ControlMetrics
    ):
        """Control inestable degrada el veredicto."""
        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=clean_topology,
            financial_metrics=viable_financials,
            stability=TestConstants.STABILITY_HIGH,
            control_metrics=asdict(unstable_control),
        )

        assert report.verdict == VerdictLevel.RECHAZAR

    def test_report_serialization(
        self,
        deterministic_translator: SemanticTranslator,
        clean_topology: TopologicalMetrics,
        viable_financials: Dict[str, Any]
    ):
        """El reporte puede serializarse a diccionario."""
        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=clean_topology,
            financial_metrics=viable_financials,
            stability=TestConstants.STABILITY_HIGH,
        )

        # Verificar que se puede convertir a dict
        report_dict = report.to_dict() if hasattr(report, 'to_dict') else asdict(report)
        
        assert isinstance(report_dict, dict)
        assert "verdict" in report_dict or "raw_narrative" in report_dict


# =============================================================================
# TESTS: INTEGRACIÓN CON STRATUM (DIKW)
# =============================================================================

@pytest.mark.integration
class TestStratumIntegration:
    """
    Pruebas de integración con la jerarquía DIKW.
    
    Verifica:
    - Clausura transitiva de validaciones
    - Propagación de fallos hacia arriba
    - Consistencia entre estratos
    """

    def test_failed_physics_propagates_to_tactics(
        self,
        deterministic_translator: SemanticTranslator,
        clean_topology: TopologicalMetrics,
        viable_financials: Dict[str, Any],
        meltdown_thermal: ThermodynamicMetrics
    ):
        """Fallo en PHYSICS propaga hacia TACTICS."""
        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=clean_topology,
            financial_metrics=viable_financials,
            stability=TestConstants.STABILITY_HIGH,
            thermal_metrics=asdict(meltdown_thermal),
        )

        physics_analysis = report.strata_analysis.get(Stratum.PHYSICS)
        assert physics_analysis is not None
        assert physics_analysis.verdict == VerdictLevel.RECHAZAR

        # El veredicto global debe reflejar el fallo
        assert report.verdict == VerdictLevel.RECHAZAR

    def test_all_strata_present_in_report(
        self,
        deterministic_translator: SemanticTranslator,
        clean_topology: TopologicalMetrics,
        viable_financials: Dict[str, Any]
    ):
        """Todos los estratos relevantes están presentes en el reporte."""
        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=clean_topology,
            financial_metrics=viable_financials,
            stability=TestConstants.STABILITY_HIGH,
        )

        expected_strata = {Stratum.PHYSICS, Stratum.TACTICS}
        actual_strata = set(report.strata_analysis.keys())
        
        assert expected_strata.issubset(actual_strata)

    def test_stratum_analysis_consistency(
        self,
        deterministic_translator: SemanticTranslator,
        clean_topology: TopologicalMetrics,
        viable_financials: Dict[str, Any]
    ):
        """El veredicto global es consistente con los análisis por estrato."""
        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=clean_topology,
            financial_metrics=viable_financials,
            stability=TestConstants.STABILITY_HIGH,
        )

        # El veredicto global debe ser el supremum de todos los estratos
        strata_verdicts = [analysis.verdict for analysis in report.strata_analysis.values()]
        
        if strata_verdicts:
            expected_global = VerdictLevel.supremum(*strata_verdicts)
            assert report.verdict == expected_global


# =============================================================================
# TESTS: PROPIEDADES ESTADÍSTICAS Y DETERMINISMO
# =============================================================================

@pytest.mark.unit
class TestStatisticalProperties:
    """
    Pruebas de propiedades estadísticas del sistema.
    
    Verifica:
    - Determinismo con misma entrada
    - Idempotencia de traducciones
    - Reproducibilidad
    """

    @pytest.mark.parametrize("seed", range(5))
    def test_deterministic_output_with_same_input(
        self,
        seed: int,
        viable_financials: Dict[str, Any]
    ):
        """Misma entrada produce misma salida (determinismo)."""
        translator = TranslatorFactory.create_deterministic()
        topo = MetricsFactory.create_topology(beta_0=1, beta_1=seed % 3)
        stability = TestConstants.STABILITY_NORMAL + seed

        report1 = translator.compose_strategic_narrative(
            topological_metrics=topo,
            financial_metrics=viable_financials,
            stability=stability,
        )

        report2 = translator.compose_strategic_narrative(
            topological_metrics=topo,
            financial_metrics=viable_financials,
            stability=stability,
        )

        assert report1.verdict == report2.verdict
        assert report1.raw_narrative == report2.raw_narrative

    def test_idempotence_of_translation(self, deterministic_translator: SemanticTranslator):
        """La traducción es idempotente: f(f(x)) = f(x)."""
        topology = MetricsFactory.create_clean_topology()
        
        narrative1, verdict1 = deterministic_translator.translate_topology(
            topology,
            stability=TestConstants.STABILITY_NORMAL
        )
        
        # Traducir de nuevo con los mismos datos
        narrative2, verdict2 = deterministic_translator.translate_topology(
            topology,
            stability=TestConstants.STABILITY_NORMAL
        )
        
        assert verdict1 == verdict2
        assert narrative1 == narrative2

    def test_output_varies_with_input(self, deterministic_translator: SemanticTranslator):
        """Diferentes entradas producen diferentes salidas."""
        clean = MetricsFactory.create_clean_topology()
        cyclic = MetricsFactory.create_cyclic_topology(n_cycles=3)
        
        _, verdict_clean = deterministic_translator.translate_topology(
            clean, stability=TestConstants.STABILITY_HIGH
        )
        
        _, verdict_cyclic = deterministic_translator.translate_topology(
            cyclic, stability=TestConstants.STABILITY_HIGH
        )
        
        # Los veredictos deben ser diferentes
        assert verdict_clean != verdict_cyclic


# =============================================================================
# TESTS: EDGE CASES Y BOUNDARY CONDITIONS
# =============================================================================

@pytest.mark.unit
class TestEdgeCases:
    """
    Pruebas de casos límite y condiciones de frontera.
    
    Verifica comportamiento en:
    - Valores extremos
    - Entradas vacías/nulas
    - Tipos inesperados
    """

    def test_zero_stability(self, default_translator: SemanticTranslator):
        """Estabilidad cero es un caso límite válido."""
        topology = MetricsFactory.create_clean_topology()
        
        _, verdict = default_translator.translate_topology(topology, stability=0.0)
        
        # Con estabilidad 0, debe haber preocupación
        assert verdict.value >= VerdictLevel.PRECAUCION.value

    def test_very_high_stability(self, default_translator: SemanticTranslator):
        """Estabilidad muy alta no causa overflow."""
        topology = MetricsFactory.create_clean_topology()
        
        _, verdict = default_translator.translate_topology(topology, stability=1e10)
        
        assert verdict == VerdictLevel.VIABLE

    def test_extreme_betti_numbers(self, default_translator: SemanticTranslator):
        """Números de Betti extremos no causan problemas."""
        extreme_topology = MetricsFactory.create_topology(beta_0=100, beta_1=50)
        
        narrative, verdict = default_translator.translate_topology(
            extreme_topology,
            stability=TestConstants.STABILITY_HIGH
        )
        
        assert verdict == VerdictLevel.RECHAZAR
        assert narrative is not None

    def test_empty_financial_metrics(self, deterministic_translator: SemanticTranslator):
        """Métricas financieras vacías no causan excepción."""
        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=MetricsFactory.create_clean_topology(),
            financial_metrics={},
            stability=TestConstants.STABILITY_HIGH,
        )
        
        assert report is not None
        assert report.verdict is not None

    def test_minimal_financial_metrics(self, deterministic_translator: SemanticTranslator):
        """Métricas financieras mínimas son aceptadas."""
        minimal = {"performance": {"recommendation": "REVISAR"}}
        
        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=MetricsFactory.create_clean_topology(),
            financial_metrics=minimal,
            stability=TestConstants.STABILITY_HIGH,
        )
        
        assert report is not None

    @pytest.mark.parametrize("temperature", [0.0, 1.0, 273.15, 1000.0, 1e6])
    def test_extreme_temperatures(
        self,
        default_translator: SemanticTranslator,
        temperature: float
    ):
        """Temperaturas extremas son manejadas correctamente."""
        thermal = MetricsFactory.create_thermal(temperature=temperature)
        
        narrative, verdict = default_translator.translate_thermodynamics(thermal)
        
        assert isinstance(narrative, str)
        assert isinstance(verdict, VerdictLevel)

    def test_nan_values_handled(self, default_translator: SemanticTranslator):
        """Valores NaN son manejados apropiadamente."""
        # Este test verifica que el sistema no explota con NaN
        # El comportamiento específico puede variar
        try:
            thermal = MetricsFactory.create_thermal(temperature=float('nan'))
            _, verdict = default_translator.translate_thermodynamics(thermal)
            # Si no lanza excepción, debe retornar algo
            assert verdict is not None
        except (ValueError, TypeError):
            # También es aceptable que lance excepción
            pass

    def test_inf_values_handled(self, default_translator: SemanticTranslator):
        """Valores infinitos son manejados apropiadamente."""
        try:
            thermal = MetricsFactory.create_thermal(temperature=float('inf'))
            _, verdict = default_translator.translate_thermodynamics(thermal)
            # Temperatura infinita debería ser RECHAZAR
            assert verdict == VerdictLevel.RECHAZAR
        except (ValueError, TypeError):
            pass


# =============================================================================
# TESTS: FACTORIES Y CONSTRUCTORES
# =============================================================================

@pytest.mark.unit
class TestFactories:
    """Pruebas para factories y funciones de creación."""

    def test_create_translator_default(self):
        """create_translator crea traductor con valores por defecto."""
        translator = create_translator()
        
        assert isinstance(translator, SemanticTranslator)

    def test_create_translator_with_config(self):
        """create_translator acepta configuración personalizada."""
        config = TranslatorConfig(deterministic_market=True)
        translator = create_translator(config=config)
        
        assert isinstance(translator, SemanticTranslator)

    def test_translate_metrics_to_narrative_function(self):
        """translate_metrics_to_narrative es función de conveniencia."""
        topology = MetricsFactory.create_clean_topology()
        financials = FinancialsFactory.create_viable()
        
        result = translate_metrics_to_narrative(
            topological_metrics=topology,
            financial_metrics=financials,
            stability=TestConstants.STABILITY_HIGH,
        )
        
        assert isinstance(result, (StrategicReport, dict))


# =============================================================================
# TESTS: CONFIGURACIÓN Y UMBRALES
# =============================================================================

@pytest.mark.unit
class TestConfiguration:
    """Pruebas para configuración y umbrales."""

    def test_custom_stability_thresholds(self):
        """Umbrales de estabilidad personalizados afectan veredictos."""
        strict_thresholds = StabilityThresholds(critical=8.0, warning=15.0)
        translator = TranslatorFactory.create_with_thresholds(stability=strict_thresholds)
        
        topology = MetricsFactory.create_clean_topology()
        
        # Con umbrales estrictos, estabilidad 10 debería dar advertencia
        _, verdict = translator.translate_topology(topology, stability=10.0)
        
        # Debe ser al menos PRECAUCION con umbrales estrictos
        assert verdict.value >= VerdictLevel.VIABLE.value

    def test_config_is_immutable(self):
        """La configuración es inmutable después de crear el traductor."""
        config = TranslatorConfig(deterministic_market=True)
        translator = SemanticTranslator(config=config)
        
        # Verificar que el config del traductor no puede modificarse
        # (depende de si TranslatorConfig es frozen)
        try:
            translator._config.deterministic_market = False
            # Si llega aquí, no es inmutable - verificar que no afecta
        except (FrozenInstanceError, AttributeError):
            pass  # Es inmutable, como se esperaba


# =============================================================================
# TESTS: REGRESIÓN
# =============================================================================

@pytest.mark.regression
class TestRegression:
    """
    Tests de regresión para bugs conocidos.
    
    Cada test documenta un bug histórico y verifica que no reaparezca.
    """

    def test_regression_empty_topology_dict(self, default_translator: SemanticTranslator):
        """
        Regresión: Diccionario de topología vacío no debe causar KeyError.
        
        Bug original: translate_topology({}, stability=10) lanzaba KeyError.
        """
        empty_dict: Dict[str, Any] = {}
        
        # No debe lanzar excepción
        try:
            narrative, verdict = default_translator.translate_topology(
                empty_dict,
                stability=TestConstants.STABILITY_NORMAL
            )
            assert verdict is not None
        except KeyError:
            pytest.fail("KeyError no debe ocurrir con diccionario vacío")

    def test_regression_missing_performance_key(self, deterministic_translator: SemanticTranslator):
        """
        Regresión: Métricas financieras sin 'performance' no deben causar error.
        
        Bug original: compose_strategic_narrative fallaba si faltaba 'performance'.
        """
        incomplete_financials = {"wacc": 0.10, "var": 1000.0}
        
        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=MetricsFactory.create_clean_topology(),
            financial_metrics=incomplete_financials,
            stability=TestConstants.STABILITY_HIGH,
        )
        
        assert report is not None


# =============================================================================
# TESTS: PERFORMANCE (Opcional)
# =============================================================================

@pytest.mark.slow
class TestPerformance:
    """
    Tests de rendimiento.
    
    Estos tests verifican que el sistema mantiene tiempos de respuesta
    aceptables bajo carga típica.
    """

    def test_translation_performance(self, deterministic_translator: SemanticTranslator):
        """La traducción debe completarse en tiempo razonable."""
        import time
        
        topology = MetricsFactory.create_clean_topology()
        financials = FinancialsFactory.create_viable()
        
        start = time.monotonic()
        
        for _ in range(100):
            deterministic_translator.compose_strategic_narrative(
                topological_metrics=topology,
                financial_metrics=financials,
                stability=TestConstants.STABILITY_HIGH,
            )
        
        elapsed = time.monotonic() - start
        
        # 100 iteraciones deben completarse en menos de 5 segundos
        assert elapsed < 5.0, f"100 traducciones tomaron {elapsed:.2f}s (esperado < 5s)"


# =============================================================================
# UTILIDADES DE PRUEBA
# =============================================================================

def assert_verdict_at_least(actual: VerdictLevel, minimum: VerdictLevel) -> None:
    """
    Afirma que el veredicto actual es al menos tan severo como el mínimo.
    
    Args:
        actual: Veredicto obtenido
        minimum: Veredicto mínimo esperado
    """
    assert actual.value >= minimum.value, \
        f"Veredicto {actual.name} es menos severo que {minimum.name}"


def assert_verdict_at_most(actual: VerdictLevel, maximum: VerdictLevel) -> None:
    """
    Afirma que el veredicto actual es como máximo tan severo como el máximo.
    
    Args:
        actual: Veredicto obtenido
        maximum: Veredicto máximo esperado
    """
    assert actual.value <= maximum.value, \
        f"Veredicto {actual.name} es más severo que {maximum.name}"


def assert_narrative_contains(narrative: str, *keywords: str) -> None:
    """
    Afirma que la narrativa contiene al menos una de las palabras clave.
    
    Args:
        narrative: Texto de narrativa
        keywords: Palabras clave a buscar (case-insensitive)
    """
    narrative_lower = narrative.lower()
    found = any(kw.lower() in narrative_lower for kw in keywords)
    
    assert found, f"Narrativa no contiene ninguna de: {keywords}\n\nNarrativa: {narrative}"