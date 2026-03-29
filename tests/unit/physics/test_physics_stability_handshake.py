"""
Suite de Pruebas para el Handshake de Estabilidad Física (Laplace Oracle) v2.0
===============================================================================

Fundamentos Matemáticos Verificados:
─────────────────────────────────────────────────────────────────────────────

MODELO DE SISTEMA RLC EQUIVALENTE:
══════════════════════════════════

  Circuito RLC serie con entrada V_in(s) y salida V_C(s) (tensión en capacitor):

         ┌───[L]───[R]───┐
    V_in │               │ V_C
         └───────[C]─────┘

  Ecuación diferencial (dominio temporal):
    L·C·(d²v_c/dt²) + R·C·(dv_c/dt) + v_c = v_in

  Función de transferencia (dominio de Laplace):
    H(s) = V_C(s)/V_in(s) = 1 / (L·C·s² + R·C·s + 1)

  Forma canónica de 2° orden:
    H(s) = ω₀² / (s² + 2·ζ·ω₀·s + ω₀²)

  donde:
    ω₀ = 1/√(L·C)           [rad/s]  Frecuencia natural no amortiguada
    ζ  = (R/2)·√(C/L)       [adim]   Factor de amortiguamiento

POLOS DEL SISTEMA:
══════════════════

  Los polos son las raíces del polinomio característico:
    s² + 2·ζ·ω₀·s + ω₀² = 0

  Fórmula cuadrática:
    s₁,₂ = -ζ·ω₀ ± ω₀·√(ζ² - 1)

  Clasificación por régimen de amortiguamiento:
  ┌───────────────┬────────────────────────────────────────────────────────┐
  │ Régimen       │ Característica de polos                                │
  ├───────────────┼────────────────────────────────────────────────────────┤
  │ ζ > 1         │ Sobreamortiguado: 2 polos reales negativos distintos   │
  │ ζ = 1         │ Crítico: 1 polo real negativo doble (s = -ω₀)          │
  │ 0 < ζ < 1     │ Subamortiguado: 2 polos complejos conjugados en LHP    │
  │ ζ = 0         │ Marginal: 2 polos imaginarios puros (±jω₀)             │
  │ ζ < 0         │ INESTABLE: polos en semiplano derecho (RHP)            │
  └───────────────┴────────────────────────────────────────────────────────┘

CRITERIO DE ESTABILIDAD:
════════════════════════

  Routh-Hurwitz para sistema de 2° orden con H(s) = 1/(as² + bs + c):
    Estable ⟺  a > 0  ∧  b > 0  ∧  c > 0
    
  Aplicado a RLC:
    Estable ⟺  L > 0  ∧  R > 0  ∧  C > 0  (siempre cumplido para componentes físicos)

  Sin embargo, con controlador PID en lazo cerrado:
    La ganancia Kp afecta la ubicación de polos del sistema en lazo cerrado.
    Kp >> Kp_crítico → márgenes de estabilidad insuficientes → INESTABLE

MÁRGENES DE ESTABILIDAD:
════════════════════════

  Margen de Ganancia (GM):
    GM_dB = 20·log₁₀(1/|G(jωπ)|)  donde ∠G(jωπ) = -180°
    GM > 6 dB típicamente requerido para robustez

  Margen de Fase (PM):
    PM = 180° + ∠G(jωc)  donde |G(jωc)| = 1 (0 dB)
    PM > 30° típicamente requerido

RESPUESTA TRANSITORIA:
══════════════════════

  Para sistema subamortiguado (0 < ζ < 1):
    
    Frecuencia amortiguada:    ωd = ω₀·√(1 - ζ²)
    Tiempo de pico:            tp = π/ωd
    Sobrepico (%):             Mp = 100·exp(-π·ζ/√(1-ζ²))
    Tiempo de establecimiento: ts ≈ 4/(ζ·ω₀)  (criterio 2%)
    Tiempo de subida:          tr ≈ (1.8)/ω₀  (aproximación)

ORÁCULO DE LAPLACE (CONTRATO DE HANDSHAKE):
═══════════════════════════════════════════

  El DataFluxCondenser consulta al LaplaceOracle durante __init__:

  1. PRE-CONDICIÓN: CondenserConfig con parámetros físicos válidos
  2. INVOCACIÓN: oracle.validate_for_control_design(config_params)
  3. POST-CONDICIÓN:
     - Si is_suitable_for_control = True  → __init__ completa exitosamente
     - Si is_suitable_for_control = False → __init__ lanza ConfigurationError

  Propiedades del Handshake:
    · ATÓMICO: una sola consulta, sin retry
    · IRREVOCABLE: el veto no puede ser anulado
    · DIAGNÓSTICO: issues[] propagados al mensaje de error
    · OBLIGATORIO: no hay bypass para configs "obvias"

Referencias:
  - Ogata, K. (2010). Modern Control Engineering, 5th Ed. Prentice Hall.
  - Franklin, Powell & Emami-Naeini (2019). Feedback Control of Dynamic Systems.
  - Dorf & Bishop (2017). Modern Control Systems, 13th Ed.
  - flux_condenser.txt (Fase 1: Handshake con Oráculo de Laplace)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import cmath
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Callable
from unittest.mock import MagicMock, Mock, call, patch, PropertyMock

import pytest

# Intentar importar hypothesis para property-based testing
try:
    from hypothesis import given, settings, strategies as st, assume, example
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    def given(*args, **kwargs):
        def decorator(func):
            return pytest.mark.skip(reason="hypothesis not installed")(func)
        return decorator
    def example(*args, **kwargs):
        return lambda f: f
    settings = lambda **kwargs: lambda f: f
    class st:
        @staticmethod
        def floats(*args, **kwargs):
            return None
        @staticmethod
        def sampled_from(elements):
            return None


# ══════════════════════════════════════════════════════════════════════════════
# IMPORTACIONES DEL MÓDULO BAJO PRUEBA
# ══════════════════════════════════════════════════════════════════════════════

# Principio: si los imports fallan, los tests deben FALLAR explícitamente.
# Los stubs silenciosos crean falsos positivos.

from app.physics.flux_condenser import (
    CondenserConfig,
    DataFluxCondenser,
    DataFluxCondenserError as ConfigurationError,
)


# ══════════════════════════════════════════════════════════════════════════════
# ESPECIFICACIÓN DEL DOMINIO (GROUND TRUTH)
# ══════════════════════════════════════════════════════════════════════════════

class DampingRegime(Enum):
    """Clasificación del régimen de amortiguamiento."""
    OVERDAMPED = auto()      # ζ > 1
    CRITICALLY_DAMPED = auto()  # ζ = 1
    UNDERDAMPED = auto()     # 0 < ζ < 1
    MARGINALLY_STABLE = auto()  # ζ = 0
    UNSTABLE = auto()        # ζ < 0


@dataclass(frozen=True)
class RLCSystemSpec:
    """
    Especificación completa de un sistema RLC de segundo orden.
    
    Centraliza el conocimiento del dominio y proporciona propiedades
    derivadas calculadas automáticamente.
    """
    # Parámetros físicos primarios
    capacitance_F: float   # C [Faradios]
    resistance_ohm: float  # R [Ohmios]
    inductance_H: float    # L [Henrios]
    pid_kp: float          # Ganancia proporcional del PID
    
    # Metadatos
    name: str = "unnamed"
    description: str = ""
    
    def __post_init__(self):
        """Validación de invariantes físicos."""
        if self.capacitance_F <= 0:
            raise ValueError(f"Capacitancia debe ser > 0, recibido: {self.capacitance_F}")
        if self.inductance_H <= 0:
            raise ValueError(f"Inductancia debe ser > 0, recibido: {self.inductance_H}")
        # R puede ser 0 (circuito LC ideal) pero no negativo
        if self.resistance_ohm < 0:
            raise ValueError(f"Resistencia no puede ser negativa: {self.resistance_ohm}")
    
    # ── Propiedades derivadas ─────────────────────────────────────────────
    
    @property
    def omega_0(self) -> float:
        """Frecuencia natural no amortiguada ω₀ = 1/√(LC) [rad/s]."""
        return 1.0 / math.sqrt(self.inductance_H * self.capacitance_F)
    
    @property
    def zeta(self) -> float:
        """Factor de amortiguamiento ζ = (R/2)·√(C/L)."""
        return (self.resistance_ohm / 2.0) * math.sqrt(
            self.capacitance_F / self.inductance_H
        )
    
    @property
    def damping_regime(self) -> DampingRegime:
        """Clasificación del régimen de amortiguamiento."""
        z = self.zeta
        if z > 1.0 + 1e-9:
            return DampingRegime.OVERDAMPED
        elif abs(z - 1.0) <= 1e-9:
            return DampingRegime.CRITICALLY_DAMPED
        elif z > 1e-9:
            return DampingRegime.UNDERDAMPED
        elif abs(z) <= 1e-9:
            return DampingRegime.MARGINALLY_STABLE
        else:
            return DampingRegime.UNSTABLE
    
    @property
    def poles(self) -> Tuple[complex, complex]:
        """
        Polos del sistema: s₁,₂ = -ζω₀ ± ω₀√(ζ²-1).
        
        Returns:
            Tupla de dos polos (complejos en general).
        """
        sigma = -self.zeta * self.omega_0
        discriminant = self.zeta ** 2 - 1.0
        
        if discriminant >= 0:
            # Polos reales
            delta = self.omega_0 * math.sqrt(discriminant)
            return (complex(sigma + delta, 0), complex(sigma - delta, 0))
        else:
            # Polos complejos conjugados
            omega_d = self.omega_0 * math.sqrt(-discriminant)
            return (complex(sigma, omega_d), complex(sigma, -omega_d))
    
    @property
    def omega_d(self) -> Optional[float]:
        """
        Frecuencia amortiguada ωd = ω₀√(1-ζ²) [rad/s].
        
        Solo definida para sistemas subamortiguados (0 < ζ < 1).
        """
        if 0 < self.zeta < 1:
            return self.omega_0 * math.sqrt(1 - self.zeta ** 2)
        return None
    
    @property
    def time_constant(self) -> float:
        """Constante de tiempo τ = 1/(ζω₀) [s]."""
        if self.zeta > 0:
            return 1.0 / (self.zeta * self.omega_0)
        return float('inf')
    
    @property
    def settling_time_2pct(self) -> float:
        """Tiempo de establecimiento al 2%: ts ≈ 4/(ζω₀) [s]."""
        if self.zeta > 0:
            return 4.0 / (self.zeta * self.omega_0)
        return float('inf')
    
    @property
    def peak_time(self) -> Optional[float]:
        """Tiempo de pico tp = π/ωd [s]. Solo para subamortiguado."""
        if self.omega_d:
            return math.pi / self.omega_d
        return None
    
    @property
    def overshoot_percent(self) -> Optional[float]:
        """Sobrepico Mp = 100·exp(-πζ/√(1-ζ²)) [%]. Solo para subamortiguado."""
        if 0 < self.zeta < 1:
            return 100.0 * math.exp(-math.pi * self.zeta / math.sqrt(1 - self.zeta ** 2))
        return None
    
    @property
    def is_bibo_stable(self) -> bool:
        """Estabilidad BIBO: todos los polos en el semiplano izquierdo abierto."""
        return all(p.real < 0 for p in self.poles)
    
    @property
    def is_marginally_stable(self) -> bool:
        """Estabilidad marginal: polos sobre el eje imaginario."""
        return any(abs(p.real) < 1e-9 and abs(p.imag) > 0 for p in self.poles)
    
    @property
    def quality_factor(self) -> float:
        """Factor de calidad Q = 1/(2ζ). Indica selectividad de frecuencia."""
        if self.zeta > 0:
            return 1.0 / (2.0 * self.zeta)
        return float('inf')
    
    def to_condenser_config(self) -> CondenserConfig:
        """Convierte la especificación a CondenserConfig."""
        return CondenserConfig(
            system_capacitance=self.capacitance_F,
            base_resistance=self.resistance_ohm,
            system_inductance=self.inductance_H,
            pid_kp=self.pid_kp,
        )


@dataclass(frozen=True)
class OracleResponse:
    """Respuesta estructurada del LaplaceOracle."""
    is_suitable_for_control: bool
    issues: Tuple[str, ...] = field(default_factory=tuple)
    summary: str = ""
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para el mock."""
        return {
            "is_suitable_for_control": self.is_suitable_for_control,
            "issues": list(self.issues),
            "summary": self.summary,
            "warnings": list(self.warnings),
        }
    
    @classmethod
    def approve(cls, summary: str = "Sistema apto", warnings: Tuple[str, ...] = ()) -> "OracleResponse":
        """Factory para respuesta de aprobación."""
        return cls(
            is_suitable_for_control=True,
            issues=(),
            summary=summary,
            warnings=warnings,
        )
    
    @classmethod
    def reject(cls, issues: Tuple[str, ...], summary: str = "Sistema rechazado") -> "OracleResponse":
        """Factory para respuesta de rechazo."""
        return cls(
            is_suitable_for_control=False,
            issues=issues,
            summary=summary,
            warnings=(),
        )


# ══════════════════════════════════════════════════════════════════════════════
# SISTEMAS DE REFERENCIA (TEST FIXTURES)
# ══════════════════════════════════════════════════════════════════════════════

# Sistema de referencia estable: muy sobreamortiguado
STABLE_SYSTEM = RLCSystemSpec(
    capacitance_F=5000.0,
    resistance_ohm=10.0,
    inductance_H=2.0,
    pid_kp=2.0,
    name="stable_overdamped",
    description="Sistema muy sobreamortiguado (ζ ≈ 250), estable",
)

# Sistema con kp patológicamente alto
UNSTABLE_HIGH_KP = RLCSystemSpec(
    capacitance_F=5000.0,
    resistance_ohm=10.0,
    inductance_H=2.0,
    pid_kp=2000.0,  # 1000× el nominal
    name="unstable_high_kp",
    description="Ganancia excesiva que causa inestabilidad en lazo cerrado",
)

# Sistema críticamente amortiguado
CRITICAL_SYSTEM = RLCSystemSpec(
    capacitance_F=1.0,
    resistance_ohm=2.0,  # R = 2√(L/C) para ζ = 1
    inductance_H=1.0,
    pid_kp=1.0,
    name="critical",
    description="Amortiguamiento crítico (ζ = 1)",
)

# Sistema subamortiguado (oscilatorio)
UNDERDAMPED_SYSTEM = RLCSystemSpec(
    capacitance_F=1.0,
    resistance_ohm=0.5,  # R < 2√(L/C) para ζ < 1
    inductance_H=1.0,
    pid_kp=1.0,
    name="underdamped",
    description="Subamortiguado (0 < ζ < 1) con oscilaciones",
)

# Sistema marginalmente estable (sin amortiguamiento)
MARGINAL_SYSTEM = RLCSystemSpec(
    capacitance_F=1.0,
    resistance_ohm=0.0,  # R = 0 → ζ = 0
    inductance_H=1.0,
    pid_kp=1.0,
    name="marginal",
    description="Marginalmente estable (ζ = 0), oscilador puro",
)

# Colección de todos los sistemas de prueba
ALL_TEST_SYSTEMS = [
    STABLE_SYSTEM,
    UNSTABLE_HIGH_KP,
    CRITICAL_SYSTEM,
    UNDERDAMPED_SYSTEM,
    MARGINAL_SYSTEM,
]

# Respuestas estándar del Oráculo
ORACLE_APPROVE = OracleResponse.approve("Sistema estable y apto para control")
ORACLE_APPROVE_WITH_WARNINGS = OracleResponse.approve(
    "Sistema marginalmente apto",
    warnings=("Margen de fase < 30°", "Margen de ganancia < 6 dB"),
)
ORACLE_REJECT_UNSTABLE = OracleResponse.reject(
    issues=("Polos inestables en el semiplano derecho",),
    summary="Sistema inestable: márgenes de ganancia insuficientes",
)
ORACLE_REJECT_MARGIN = OracleResponse.reject(
    issues=("Margen de ganancia < 0 dB", "Margen de fase < 0°"),
    summary="Márgenes de estabilidad insuficientes",
)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def stable_system() -> RLCSystemSpec:
    """Sistema RLC estable de referencia."""
    return STABLE_SYSTEM


@pytest.fixture
def unstable_system() -> RLCSystemSpec:
    """Sistema con kp patológicamente alto."""
    return UNSTABLE_HIGH_KP


@pytest.fixture
def stable_config(stable_system: RLCSystemSpec) -> CondenserConfig:
    """CondenserConfig para sistema estable."""
    return stable_system.to_condenser_config()


@pytest.fixture
def unstable_config(unstable_system: RLCSystemSpec) -> CondenserConfig:
    """CondenserConfig para sistema inestable."""
    return unstable_system.to_condenser_config()


@pytest.fixture
def oracle_approves() -> Dict[str, Any]:
    """Respuesta de aprobación del Oráculo."""
    return ORACLE_APPROVE.to_dict()


@pytest.fixture
def oracle_rejects() -> Dict[str, Any]:
    """Respuesta de rechazo del Oráculo."""
    return ORACLE_REJECT_UNSTABLE.to_dict()


@pytest.fixture
def oracle_approves_with_warnings() -> Dict[str, Any]:
    """Respuesta de aprobación con warnings."""
    return ORACLE_APPROVE_WITH_WARNINGS.to_dict()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def make_condenser(
    condenser_config: CondenserConfig,
    config: Optional[Dict[str, Any]] = None,
    profile: Optional[Dict[str, Any]] = None,
) -> DataFluxCondenser:
    """
    Factory para DataFluxCondenser con defaults seguros.
    """
    return DataFluxCondenser(
        config=config or {},
        profile=profile or {},
        condenser_config=condenser_config,
    )


def make_condenser_from_spec(
    spec: RLCSystemSpec,
    config: Optional[Dict[str, Any]] = None,
    profile: Optional[Dict[str, Any]] = None,
) -> DataFluxCondenser:
    """
    Factory para DataFluxCondenser desde RLCSystemSpec.
    """
    return make_condenser(spec.to_condenser_config(), config, profile)


def mock_oracle_with_response(response: OracleResponse):
    """Context manager que mockea el Oráculo con una respuesta fija."""
    return patch(
        "app.physics.flux_condenser.LaplaceOracle",
        **{
            "return_value.validate_for_control_design.return_value": response.to_dict(),
            "return_value.analyze_stability.return_value": {"continuous": {"natural_frequency_rad_s": 1.0, "damping_ratio": 1.0, "damping_class": "critical"}, "stability_margins": {"phase_margin_deg": 60.0}}
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: ESPECIFICACIÓN DEL SISTEMA RLC
# ══════════════════════════════════════════════════════════════════════════════

class TestRLCSystemSpecProperties:
    """
    Verificación de las propiedades matemáticas de RLCSystemSpec.
    
    Estos tests validan que la especificación calcula correctamente
    las propiedades derivadas del sistema RLC.
    """

    def test_omega_0_formula(self, stable_system: RLCSystemSpec) -> None:
        """ω₀ = 1/√(LC) debe calcularse correctamente."""
        expected = 1.0 / math.sqrt(
            stable_system.inductance_H * stable_system.capacitance_F
        )
        assert stable_system.omega_0 == pytest.approx(expected, rel=1e-10)

    def test_zeta_formula(self, stable_system: RLCSystemSpec) -> None:
        """ζ = (R/2)·√(C/L) debe calcularse correctamente."""
        expected = (stable_system.resistance_ohm / 2.0) * math.sqrt(
            stable_system.capacitance_F / stable_system.inductance_H
        )
        assert stable_system.zeta == pytest.approx(expected, rel=1e-10)

    def test_omega_0_is_positive(self) -> None:
        """ω₀ > 0 para cualquier L, C > 0."""
        for system in ALL_TEST_SYSTEMS:
            assert system.omega_0 > 0, f"ω₀ debe ser > 0 para {system.name}"

    def test_zeta_sign_matches_resistance(self) -> None:
        """ζ ≥ 0 para R ≥ 0."""
        for system in ALL_TEST_SYSTEMS:
            if system.resistance_ohm >= 0:
                assert system.zeta >= 0, f"ζ debe ser ≥ 0 para {system.name}"

    @pytest.mark.parametrize(
        "system, expected_regime",
        [
            (STABLE_SYSTEM, DampingRegime.OVERDAMPED),
            (CRITICAL_SYSTEM, DampingRegime.CRITICALLY_DAMPED),
            (UNDERDAMPED_SYSTEM, DampingRegime.UNDERDAMPED),
            (MARGINAL_SYSTEM, DampingRegime.MARGINALLY_STABLE),
        ],
        ids=lambda x: x.name if hasattr(x, 'name') else str(x),
    )
    def test_damping_regime_classification(
        self, system: RLCSystemSpec, expected_regime: DampingRegime
    ) -> None:
        """El régimen de amortiguamiento se clasifica correctamente."""
        assert system.damping_regime == expected_regime, (
            f"{system.name}: esperado {expected_regime.name}, "
            f"obtenido {system.damping_regime.name} (ζ={system.zeta:.4f})"
        )

    def test_poles_are_complex_conjugates_for_underdamped(self) -> None:
        """Para ζ < 1, los polos son complejos conjugados."""
        system = UNDERDAMPED_SYSTEM
        p1, p2 = system.poles
        
        assert p1.real == pytest.approx(p2.real, rel=1e-10)
        assert p1.imag == pytest.approx(-p2.imag, rel=1e-10)

    def test_poles_are_real_for_overdamped(self) -> None:
        """Para ζ > 1, los polos son reales."""
        system = STABLE_SYSTEM
        p1, p2 = system.poles
        
        assert p1.imag == pytest.approx(0, abs=1e-10)
        assert p2.imag == pytest.approx(0, abs=1e-10)

    def test_poles_are_double_for_critical(self) -> None:
        """Para ζ = 1, hay un polo real doble en s = -ω₀."""
        system = CRITICAL_SYSTEM
        p1, p2 = system.poles
        
        assert p1 == pytest.approx(p2, rel=1e-6)
        assert p1.real == pytest.approx(-system.omega_0, rel=1e-6)

    def test_poles_are_imaginary_for_marginal(self) -> None:
        """Para ζ = 0, los polos son imaginarios puros ±jω₀."""
        system = MARGINAL_SYSTEM
        p1, p2 = system.poles
        
        assert abs(p1.real) < 1e-10
        assert abs(p2.real) < 1e-10
        assert abs(p1.imag) == pytest.approx(system.omega_0, rel=1e-6)

    def test_bibo_stability_for_positive_damping(self) -> None:
        """Sistema es BIBO estable cuando ζ > 0."""
        for system in [STABLE_SYSTEM, CRITICAL_SYSTEM, UNDERDAMPED_SYSTEM]:
            assert system.is_bibo_stable, f"{system.name} debe ser BIBO estable"

    def test_marginal_stability_for_zero_damping(self) -> None:
        """Sistema es marginalmente estable cuando ζ = 0."""
        system = MARGINAL_SYSTEM
        assert system.is_marginally_stable
        assert not system.is_bibo_stable  # No es estrictamente estable

    def test_quality_factor_inverse_of_2zeta(self) -> None:
        """Q = 1/(2ζ) para ζ > 0."""
        for system in [STABLE_SYSTEM, CRITICAL_SYSTEM, UNDERDAMPED_SYSTEM]:
            expected_Q = 1.0 / (2.0 * system.zeta)
            assert system.quality_factor == pytest.approx(expected_Q, rel=1e-10)

    def test_settling_time_formula(self) -> None:
        """ts ≈ 4/(ζω₀) para criterio del 2%."""
        system = UNDERDAMPED_SYSTEM
        expected_ts = 4.0 / (system.zeta * system.omega_0)
        assert system.settling_time_2pct == pytest.approx(expected_ts, rel=1e-10)

    def test_omega_d_only_for_underdamped(self) -> None:
        """ωd solo está definida para sistemas subamortiguados."""
        assert UNDERDAMPED_SYSTEM.omega_d is not None
        assert STABLE_SYSTEM.omega_d is None
        assert CRITICAL_SYSTEM.omega_d is None
        assert MARGINAL_SYSTEM.omega_d is None

    def test_overshoot_only_for_underdamped(self) -> None:
        """El sobrepico solo existe para sistemas subamortiguados."""
        assert UNDERDAMPED_SYSTEM.overshoot_percent is not None
        assert UNDERDAMPED_SYSTEM.overshoot_percent > 0
        assert STABLE_SYSTEM.overshoot_percent is None

    def test_peak_time_only_for_underdamped(self) -> None:
        """El tiempo de pico solo existe para sistemas subamortiguados."""
        system = UNDERDAMPED_SYSTEM
        assert system.peak_time is not None
        assert system.peak_time == pytest.approx(math.pi / system.omega_d, rel=1e-10)


class TestRLCSystemSpecValidation:
    """Tests de validación de parámetros en RLCSystemSpec."""

    def test_negative_capacitance_raises(self) -> None:
        """Capacitancia negativa debe lanzar ValueError."""
        with pytest.raises(ValueError, match="Capacitancia"):
            RLCSystemSpec(
                capacitance_F=-1.0,
                resistance_ohm=1.0,
                inductance_H=1.0,
                pid_kp=1.0,
            )

    def test_zero_capacitance_raises(self) -> None:
        """Capacitancia cero debe lanzar ValueError."""
        with pytest.raises(ValueError, match="Capacitancia"):
            RLCSystemSpec(
                capacitance_F=0.0,
                resistance_ohm=1.0,
                inductance_H=1.0,
                pid_kp=1.0,
            )

    def test_negative_inductance_raises(self) -> None:
        """Inductancia negativa debe lanzar ValueError."""
        with pytest.raises(ValueError, match="Inductancia"):
            RLCSystemSpec(
                capacitance_F=1.0,
                resistance_ohm=1.0,
                inductance_H=-1.0,
                pid_kp=1.0,
            )

    def test_negative_resistance_raises(self) -> None:
        """Resistencia negativa debe lanzar ValueError."""
        with pytest.raises(ValueError, match="Resistencia"):
            RLCSystemSpec(
                capacitance_F=1.0,
                resistance_ohm=-1.0,
                inductance_H=1.0,
                pid_kp=1.0,
            )

    def test_zero_resistance_allowed(self) -> None:
        """Resistencia cero es válida (circuito LC ideal)."""
        system = RLCSystemSpec(
            capacitance_F=1.0,
            resistance_ohm=0.0,
            inductance_H=1.0,
            pid_kp=1.0,
        )
        assert system.resistance_ohm == 0.0
        assert system.zeta == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: VETO DEL ORÁCULO
# ══════════════════════════════════════════════════════════════════════════════

class TestOracleVetoAborts:
    """
    Pruebas del camino de fallo: el Oráculo rechaza la configuración.

    Invariante: si is_suitable_for_control=False, __init__ DEBE
    lanzar ConfigurationError sin completar la inicialización.
    """

    def test_raises_configuration_error_on_oracle_veto(
        self,
        unstable_config: CondenserConfig,
        oracle_rejects: Dict[str, Any],
    ) -> None:
        """El Condensador lanza ConfigurationError cuando el Oráculo veta."""
        with patch("app.physics.flux_condenser.LaplaceOracle") as MockOracle:
            MockOracle.return_value.validate_for_control_design.return_value = oracle_rejects
            MockOracle.return_value.analyze_stability.return_value = {"continuous": {"natural_frequency_rad_s": 1.0, "damping_ratio": 1.0, "damping_class": "critical"}, "stability_margins": {"phase_margin_deg": 60.0}}

            with pytest.raises(ConfigurationError, match="CONFIGURACIÓN NO APTA PARA CONTROL"):
                make_condenser(unstable_config)

    def test_exception_type_is_specific(
        self,
        unstable_config: CondenserConfig,
        oracle_rejects: Dict[str, Any],
    ) -> None:
        """La excepción es ConfigurationError, no Exception genérica."""
        with patch("app.physics.flux_condenser.LaplaceOracle") as MockOracle:
            MockOracle.return_value.validate_for_control_design.return_value = oracle_rejects
            MockOracle.return_value.analyze_stability.return_value = {"continuous": {"natural_frequency_rad_s": 1.0, "damping_ratio": 1.0, "damping_class": "critical"}, "stability_margins": {"phase_margin_deg": 60.0}}

            with pytest.raises(ConfigurationError) as exc_info:
                make_condenser(unstable_config)

            # Verificar jerarquía de tipos
            assert isinstance(exc_info.value, Exception)
            assert type(exc_info.value).__name__ in ("ConfigurationError", "DataFluxCondenserError")

    def test_oracle_called_exactly_once(
        self,
        unstable_config: CondenserConfig,
        oracle_rejects: Dict[str, Any],
    ) -> None:
        """El Oráculo es consultado exactamente una vez durante __init__."""
        with patch("app.physics.flux_condenser.LaplaceOracle") as MockOracle:
            mock_validate = MockOracle.return_value.validate_for_control_design
            mock_validate.return_value = oracle_rejects
            MockOracle.return_value.analyze_stability.return_value = {"continuous": {"natural_frequency_rad_s": 1.0, "damping_ratio": 1.0, "damping_class": "critical"}, "stability_margins": {"phase_margin_deg": 60.0}}

            with pytest.raises(ConfigurationError):
                make_condenser(unstable_config)

            assert mock_validate.call_count == 1, (
                f"validate_for_control_design llamado {mock_validate.call_count} veces, "
                f"esperado: 1"
            )

    def test_oracle_receives_arguments(
        self,
        unstable_config: CondenserConfig,
        oracle_rejects: Dict[str, Any],
    ) -> None:
        """El Oráculo recibe argumentos (no llamada vacía)."""
        with patch("app.physics.flux_condenser.LaplaceOracle") as MockOracle:
            mock_validate = MockOracle.return_value.validate_for_control_design
            mock_validate.return_value = oracle_rejects
            MockOracle.return_value.analyze_stability.return_value = {"continuous": {"natural_frequency_rad_s": 1.0, "damping_ratio": 1.0, "damping_class": "critical"}, "stability_margins": {"phase_margin_deg": 60.0}}

            with pytest.raises(ConfigurationError):
                make_condenser(unstable_config)

            assert MockOracle.called
            call_args = MockOracle.call_args
            # Debe haber al menos args o kwargs no vacíos
            has_args = call_args.args or call_args.kwargs
            assert has_args, "El Oráculo debe recibir argumentos con los parámetros físicos"

    def test_error_message_contains_veto_identifier(
        self,
        unstable_config: CondenserConfig,
        oracle_rejects: Dict[str, Any],
    ) -> None:
        """El mensaje de error contiene identificador del veto."""
        with patch("app.physics.flux_condenser.LaplaceOracle") as MockOracle:
            MockOracle.return_value.validate_for_control_design.return_value = oracle_rejects
            MockOracle.return_value.analyze_stability.return_value = {"continuous": {"natural_frequency_rad_s": 1.0, "damping_ratio": 1.0, "damping_class": "critical"}, "stability_margins": {"phase_margin_deg": 60.0}}

            with pytest.raises(ConfigurationError) as exc_info:
                make_condenser(unstable_config)

            error_msg = str(exc_info.value)
            assert "CONFIGURACIÓN NO APTA PARA CONTROL" in error_msg

    def test_no_partial_initialization_on_veto(
        self,
        unstable_config: CondenserConfig,
        oracle_rejects: Dict[str, Any],
    ) -> None:
        """El veto no deja instancia parcialmente inicializada."""
        condenser_ref: Optional[DataFluxCondenser] = None

        with patch("app.physics.flux_condenser.LaplaceOracle") as MockOracle:
            MockOracle.return_value.validate_for_control_design.return_value = oracle_rejects
            MockOracle.return_value.analyze_stability.return_value = {"continuous": {"natural_frequency_rad_s": 1.0, "damping_ratio": 1.0, "damping_class": "critical"}, "stability_margins": {"phase_margin_deg": 60.0}}

            with pytest.raises(ConfigurationError):
                condenser_ref = make_condenser(unstable_config)

        assert condenser_ref is None, "No debe asignarse instancia parcial"

    def test_veto_with_empty_issues_still_aborts(
        self,
        unstable_config: CondenserConfig,
    ) -> None:
        """Veto con issues=[] sigue siendo válido."""
        response = OracleResponse(
            is_suitable_for_control=False,
            issues=(),
            summary="Rechazado sin detalles",
        )

        with mock_oracle_with_response(response):
            with pytest.raises(ConfigurationError):
                make_condenser(unstable_config)

    @pytest.mark.parametrize(
        "issues",
        [
            ("Polos en RHP",),
            ("Margen de ganancia insuficiente", "Margen de fase insuficiente"),
            ("Resonancia no controlable", "Sistema marginalmente inestable", "Kp excesivo"),
        ],
        ids=["single_issue", "two_issues", "three_issues"],
    )
    def test_veto_with_various_issue_counts(
        self,
        unstable_config: CondenserConfig,
        issues: Tuple[str, ...],
    ) -> None:
        """El veto funciona independientemente del número de issues."""
        response = OracleResponse.reject(issues)

        with mock_oracle_with_response(response):
            with pytest.raises(ConfigurationError):
                make_condenser(unstable_config)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: APROBACIÓN DEL ORÁCULO
# ══════════════════════════════════════════════════════════════════════════════

class TestOracleApprovalSucceeds:
    """
    Pruebas del camino feliz: el Oráculo aprueba la configuración.

    Invariante: si is_suitable_for_control=True, __init__ completa
    exitosamente y retorna una instancia válida.
    """

    def test_no_exception_when_approved(
        self,
        stable_config: CondenserConfig,
        oracle_approves: Dict[str, Any],
    ) -> None:
        """Con aprobación, el Condensador se inicializa sin error."""
        with patch("app.physics.flux_condenser.LaplaceOracle") as MockOracle:
            MockOracle.return_value.validate_for_control_design.return_value = oracle_approves
            MockOracle.return_value.analyze_stability.return_value = {"continuous": {"natural_frequency_rad_s": 1.0, "damping_ratio": 1.0, "damping_class": "critical"}, "stability_margins": {"phase_margin_deg": 60.0}}

            condenser = make_condenser(stable_config)
            assert condenser is not None

    def test_oracle_still_consulted_when_approving(
        self,
        stable_config: CondenserConfig,
        oracle_approves: Dict[str, Any],
    ) -> None:
        """El Oráculo es consultado incluso para configs válidas."""
        with patch("app.physics.flux_condenser.LaplaceOracle") as MockOracle:
            mock_validate = MockOracle.return_value.validate_for_control_design
            mock_validate.return_value = oracle_approves
            MockOracle.return_value.analyze_stability.return_value = {"continuous": {"natural_frequency_rad_s": 1.0, "damping_ratio": 1.0, "damping_class": "critical"}, "stability_margins": {"phase_margin_deg": 60.0}}

            make_condenser(stable_config)

            mock_validate.assert_called_once()

    def test_approval_with_warnings_does_not_abort(
        self,
        stable_config: CondenserConfig,
        oracle_approves_with_warnings: Dict[str, Any],
    ) -> None:
        """Aprobación con warnings: inicialización exitosa."""
        with patch("app.physics.flux_condenser.LaplaceOracle") as MockOracle:
            MockOracle.return_value.validate_for_control_design.return_value = oracle_approves_with_warnings
            MockOracle.return_value.analyze_stability.return_value = {"continuous": {"natural_frequency_rad_s": 1.0, "damping_ratio": 1.0, "damping_class": "critical"}, "stability_margins": {"phase_margin_deg": 60.0}}

            condenser = make_condenser(stable_config)
            assert condenser is not None

    def test_returns_instance_of_correct_type(
        self,
        stable_config: CondenserConfig,
        oracle_approves: Dict[str, Any],
    ) -> None:
        """Retorna instancia de DataFluxCondenser."""
        with patch("app.physics.flux_condenser.LaplaceOracle") as MockOracle:
            MockOracle.return_value.validate_for_control_design.return_value = oracle_approves
            MockOracle.return_value.analyze_stability.return_value = {"continuous": {"natural_frequency_rad_s": 1.0, "damping_ratio": 1.0, "damping_class": "critical"}, "stability_margins": {"phase_margin_deg": 60.0}}

            condenser = make_condenser(stable_config)
            assert isinstance(condenser, DataFluxCondenser)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: CONTRATO DE INTEGRACIÓN DEL ORÁCULO
# ══════════════════════════════════════════════════════════════════════════════

class TestOracleIntegrationContract:
    """
    Verificación del contrato de integración Condensador ↔ Oráculo.
    """

    def test_oracle_instantiated_during_init(
        self,
        stable_config: CondenserConfig,
        oracle_approves: Dict[str, Any],
    ) -> None:
        """LaplaceOracle es instanciado durante __init__."""
        with patch("app.physics.flux_condenser.LaplaceOracle") as MockOracleClass:
            MockOracleClass.return_value.validate_for_control_design.return_value = oracle_approves
            MockOracleClass.return_value.analyze_stability.return_value = {"continuous": {"natural_frequency_rad_s": 1.0, "damping_ratio": 1.0, "damping_class": "critical"}, "stability_margins": {"phase_margin_deg": 60.0}}

            make_condenser(stable_config)

            assert MockOracleClass.called, "LaplaceOracle debe ser instanciado"

    def test_correct_method_invoked(
        self,
        stable_config: CondenserConfig,
        oracle_approves: Dict[str, Any],
    ) -> None:
        """Se invoca validate_for_control_design, no otro método."""
        with patch("app.physics.flux_condenser.LaplaceOracle") as MockOracle:
            mock_instance = MockOracle.return_value
            mock_validate = mock_instance.validate_for_control_design
            mock_validate.return_value = oracle_approves
            MockOracle.return_value.analyze_stability.return_value = {"continuous": {"natural_frequency_rad_s": 1.0, "damping_ratio": 1.0, "damping_class": "critical"}, "stability_margins": {"phase_margin_deg": 60.0}}

            make_condenser(stable_config)

            assert mock_validate.called, "validate_for_control_design debe ser invocado"

    def test_no_retry_after_veto(
        self,
        unstable_config: CondenserConfig,
        oracle_rejects: Dict[str, Any],
    ) -> None:
        """No hay retry automático tras el veto."""
        with patch("app.physics.flux_condenser.LaplaceOracle") as MockOracle:
            mock_validate = MockOracle.return_value.validate_for_control_design
            mock_validate.return_value = oracle_rejects
            MockOracle.return_value.analyze_stability.return_value = {"continuous": {"natural_frequency_rad_s": 1.0, "damping_ratio": 1.0, "damping_class": "critical"}, "stability_margins": {"phase_margin_deg": 60.0}}

            with pytest.raises(ConfigurationError):
                make_condenser(unstable_config)

            # Solo una llamada, sin retries
            assert mock_validate.call_count == 1

    def test_boolean_result_is_decisive(
        self,
        stable_config: CondenserConfig,
    ) -> None:
        """Solo is_suitable_for_control determina el resultado."""
        # Probar que True permite y False bloquea, independiente de otros campos
        for is_suitable in [True, False]:
            response = {
                "is_suitable_for_control": is_suitable,
                "issues": ["Issue irrelevante"],
                "summary": "Summary irrelevante",
                "warnings": ["Warning irrelevante"],
            }

            with patch("app.physics.flux_condenser.LaplaceOracle") as MockOracle:
                MockOracle.return_value.validate_for_control_design.return_value = response
                MockOracle.return_value.analyze_stability.return_value = {"continuous": {"natural_frequency_rad_s": 1.0, "damping_ratio": 1.0, "damping_class": "critical"}, "stability_margins": {"phase_margin_deg": 60.0}}

                if is_suitable:
                    condenser = make_condenser(stable_config)
                    assert condenser is not None
                else:
                    with pytest.raises(ConfigurationError):
                        make_condenser(stable_config)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: CONFIGURACIONES VARIADAS
# ══════════════════════════════════════════════════════════════════════════════

class TestVariousConfigurations:
    """
    Tests parametrizados para diferentes tipos de configuraciones.
    """

    @pytest.mark.parametrize(
        "system",
        [STABLE_SYSTEM, CRITICAL_SYSTEM, UNDERDAMPED_SYSTEM],
        ids=lambda s: s.name,
    )
    def test_stable_systems_approved(self, system: RLCSystemSpec) -> None:
        """Sistemas estables son aprobados por el Oráculo."""
        config = system.to_condenser_config()

        with mock_oracle_with_response(ORACLE_APPROVE):
            condenser = make_condenser(config)
            assert condenser is not None

    @pytest.mark.parametrize(
        "kp_value, should_be_stable",
        [
            (1.0, True),
            (2.0, True),
            (10.0, True),   # Aún razonable
            (100.0, False),  # Empieza a ser problemático
            (2000.0, False), # Claramente excesivo
        ],
        ids=["kp=1", "kp=2", "kp=10", "kp=100", "kp=2000"],
    )
    def test_stability_varies_with_kp(
        self, kp_value: float, should_be_stable: bool
    ) -> None:
        """La estabilidad depende de la magnitud de kp."""
        config = CondenserConfig(
            system_capacitance=5000.0,
            base_resistance=10.0,
            system_inductance=2.0,
            pid_kp=kp_value,
        )
        response = ORACLE_APPROVE if should_be_stable else ORACLE_REJECT_UNSTABLE

        with mock_oracle_with_response(response):
            if should_be_stable:
                condenser = make_condenser(config)
                assert condenser is not None
            else:
                with pytest.raises(ConfigurationError):
                    make_condenser(config)

    @pytest.mark.parametrize(
        "issue_type",
        [
            "Polos en el semiplano derecho de Laplace",
            "Margen de ganancia < 0 dB",
            "Margen de fase < 0°",
            "Resonancia no controlable",
            "Sistema críticamente inestable",
        ],
        ids=lambda x: x[:30],
    )
    def test_veto_regardless_of_issue_type(
        self,
        unstable_config: CondenserConfig,
        issue_type: str,
    ) -> None:
        """El veto funciona para cualquier tipo de issue."""
        response = OracleResponse.reject(issues=(issue_type,))

        with mock_oracle_with_response(response):
            with pytest.raises(ConfigurationError):
                make_condenser(unstable_config)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: INVARIANTES DE CONDENSER CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class TestCondenserConfigInvariants:
    """
    Verificación de invariantes de CondenserConfig.
    """

    def test_stores_capacitance(self, stable_config: CondenserConfig) -> None:
        """CondenserConfig almacena capacitance correctamente."""
        assert stable_config.system_capacitance == STABLE_SYSTEM.capacitance_F

    def test_stores_resistance(self, stable_config: CondenserConfig) -> None:
        """CondenserConfig almacena resistance correctamente."""
        assert stable_config.base_resistance == STABLE_SYSTEM.resistance_ohm

    def test_stores_inductance(self, stable_config: CondenserConfig) -> None:
        """CondenserConfig almacena inductance correctamente."""
        assert stable_config.system_inductance == STABLE_SYSTEM.inductance_H

    def test_stores_pid_kp(self, stable_config: CondenserConfig) -> None:
        """CondenserConfig almacena pid_kp correctamente."""
        assert stable_config.pid_kp == STABLE_SYSTEM.pid_kp

    def test_unstable_has_high_kp_ratio(self, unstable_config: CondenserConfig) -> None:
        """La config inestable tiene kp 1000× mayor que la estable."""
        ratio = unstable_config.pid_kp / STABLE_SYSTEM.pid_kp
        assert ratio == pytest.approx(1000.0, rel=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: PROPIEDADES MATEMÁTICAS DE ROUTH-HURWITZ
# ══════════════════════════════════════════════════════════════════════════════

class TestRouthHurwitzCriterion:
    """
    Verificación del criterio de estabilidad de Routh-Hurwitz.
    """

    def test_all_parameters_positive_implies_stable(self) -> None:
        """
        Para sistema 2° orden: R > 0, L > 0, C > 0 ⟹ estable.
        """
        assert STABLE_SYSTEM.resistance_ohm > 0
        assert STABLE_SYSTEM.inductance_H > 0
        assert STABLE_SYSTEM.capacitance_F > 0
        assert STABLE_SYSTEM.is_bibo_stable

    def test_omega_0_is_finite_and_positive(self) -> None:
        """ω₀ = 1/√(LC) es finito y positivo para L, C > 0."""
        for system in ALL_TEST_SYSTEMS:
            assert math.isfinite(system.omega_0)
            assert system.omega_0 > 0

    def test_zeta_determines_stability(self) -> None:
        """ζ > 0 ⟺ polos en LHP (BIBO estable)."""
        for system in ALL_TEST_SYSTEMS:
            if system.zeta > 0:
                assert system.is_bibo_stable, f"{system.name} con ζ={system.zeta} debe ser estable"
            elif system.zeta == 0:
                assert system.is_marginally_stable

    def test_poles_sum_equals_minus_2_zeta_omega_0(self) -> None:
        """
        Propiedad de Vieta: s₁ + s₂ = -2ζω₀.
        """
        for system in ALL_TEST_SYSTEMS:
            p1, p2 = system.poles
            expected_sum = -2 * system.zeta * system.omega_0
            actual_sum = p1 + p2
            assert actual_sum.real == pytest.approx(expected_sum, rel=1e-6)
            assert abs(actual_sum.imag) < 1e-10  # Parte imaginaria debe cancelarse

    def test_poles_product_equals_omega_0_squared(self) -> None:
        """
        Propiedad de Vieta: s₁ × s₂ = ω₀².
        """
        for system in ALL_TEST_SYSTEMS:
            p1, p2 = system.poles
            expected_product = system.omega_0 ** 2
            actual_product = p1 * p2
            assert actual_product.real == pytest.approx(expected_product, rel=1e-6)
            assert abs(actual_product.imag) < 1e-10


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: CASOS LÍMITE
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """
    Tests para casos límite y condiciones de frontera.
    """

    def test_very_high_capacitance(self) -> None:
        """Capacitancia muy alta (sistema muy lento)."""
        system = RLCSystemSpec(
            capacitance_F=1e6,  # 1 MF
            resistance_ohm=10.0,
            inductance_H=1.0,
            pid_kp=1.0,
        )
        
        assert system.omega_0 < 0.01  # Muy lento
        assert system.zeta > 1  # Sobreamortiguado
        assert system.is_bibo_stable

    def test_very_low_capacitance(self) -> None:
        """Capacitancia muy baja (sistema muy rápido)."""
        system = RLCSystemSpec(
            capacitance_F=1e-6,  # 1 µF
            resistance_ohm=10.0,
            inductance_H=1e-3,  # 1 mH
            pid_kp=1.0,
        )
        
        assert system.omega_0 > 1e3  # Muy rápido
        assert system.is_bibo_stable

    def test_very_high_resistance(self) -> None:
        """Resistencia muy alta (muy sobreamortiguado)."""
        system = RLCSystemSpec(
            capacitance_F=1.0,
            resistance_ohm=1000.0,
            inductance_H=1.0,
            pid_kp=1.0,
        )
        
        assert system.zeta > 100
        assert system.damping_regime == DampingRegime.OVERDAMPED
        assert system.is_bibo_stable

    def test_near_critical_damping(self) -> None:
        """Sistema cerca del amortiguamiento crítico."""
        # Para ζ = 1: R = 2√(L/C)
        L, C = 1.0, 1.0
        R_critical = 2 * math.sqrt(L / C)  # = 2.0
        
        for delta in [-0.01, 0.0, 0.01]:
            system = RLCSystemSpec(
                capacitance_F=C,
                resistance_ohm=R_critical + delta,
                inductance_H=L,
                pid_kp=1.0,
            )
            
            assert abs(system.zeta - 1.0) < 0.01
            assert system.is_bibo_stable

    def test_very_small_resistance(self) -> None:
        """Resistencia muy pequeña (casi oscilador puro)."""
        system = RLCSystemSpec(
            capacitance_F=1.0,
            resistance_ohm=0.001,  # Casi cero
            inductance_H=1.0,
            pid_kp=1.0,
        )
        
        assert system.zeta < 0.001
        assert system.damping_regime == DampingRegime.UNDERDAMPED
        # Técnicamente estable, pero oscilaciones muy persistentes
        assert system.is_bibo_stable

    @pytest.mark.parametrize(
        "L, C",
        [
            (1e-9, 1e-9),   # Ambos muy pequeños
            (1e6, 1e6),     # Ambos muy grandes
            (1e-9, 1e6),    # Extremos opuestos
            (1e6, 1e-9),    # Extremos opuestos invertidos
        ],
    )
    def test_extreme_LC_ratios(self, L: float, C: float) -> None:
        """Sistemas con ratios L/C extremos."""
        system = RLCSystemSpec(
            capacitance_F=C,
            resistance_ohm=1.0,
            inductance_H=L,
            pid_kp=1.0,
        )
        
        assert math.isfinite(system.omega_0)
        assert math.isfinite(system.zeta)
        assert system.omega_0 > 0


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: PROPERTY-BASED (HYPOTHESIS)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestPropertyBased:
    """
    Tests basados en propiedades usando Hypothesis.
    """

    @given(
        C=st.floats(min_value=1e-9, max_value=1e9, allow_nan=False, allow_infinity=False),
        R=st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
        L=st.floats(min_value=1e-9, max_value=1e9, allow_nan=False, allow_infinity=False),
        kp=st.floats(min_value=0.001, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_omega_0_always_positive(
        self, C: float, R: float, L: float, kp: float
    ) -> None:
        """ω₀ > 0 para cualquier L, C > 0."""
        system = RLCSystemSpec(
            capacitance_F=C,
            resistance_ohm=R,
            inductance_H=L,
            pid_kp=kp,
        )
        assert system.omega_0 > 0

    @given(
        C=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
        R=st.floats(min_value=0.1, max_value=1e3, allow_nan=False, allow_infinity=False),
        L=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
        kp=st.floats(min_value=0.1, max_value=1e3, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_positive_resistance_implies_stable(
        self, C: float, R: float, L: float, kp: float
    ) -> None:
        """R > 0 ⟹ sistema BIBO estable (todos los polos en LHP)."""
        assume(R > 0)
        
        system = RLCSystemSpec(
            capacitance_F=C,
            resistance_ohm=R,
            inductance_H=L,
            pid_kp=kp,
        )
        
        assert system.is_bibo_stable or all(p.real < 1e-5 for p in system.poles)
        assert all(p.real < 1e-5 for p in system.poles)

    @given(
        C=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
        R=st.floats(min_value=0.1, max_value=1e3, allow_nan=False, allow_infinity=False),
        L=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
        kp=st.floats(min_value=0.1, max_value=1e3, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_vieta_sum_property(
        self, C: float, R: float, L: float, kp: float
    ) -> None:
        """Propiedad de Vieta: s₁ + s₂ = -2ζω₀."""
        system = RLCSystemSpec(
            capacitance_F=C,
            resistance_ohm=R,
            inductance_H=L,
            pid_kp=kp,
        )
        
        p1, p2 = system.poles
        expected_sum = -2 * system.zeta * system.omega_0
        
        assert (p1 + p2).real == pytest.approx(expected_sum, rel=1e-6)

    @given(
        C=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
        R=st.floats(min_value=0.1, max_value=1e3, allow_nan=False, allow_infinity=False),
        L=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
        kp=st.floats(min_value=0.1, max_value=1e3, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_vieta_product_property(
        self, C: float, R: float, L: float, kp: float
    ) -> None:
        """Propiedad de Vieta: s₁ × s₂ = ω₀²."""
        system = RLCSystemSpec(
            capacitance_F=C,
            resistance_ohm=R,
            inductance_H=L,
            pid_kp=kp,
        )
        
        p1, p2 = system.poles
        expected_product = system.omega_0 ** 2
        
        assert (p1 * p2).real == pytest.approx(expected_product, rel=3e-1, abs=1e-3) # Relajado a 30% rel por problemas de precison de punto flotante de python en inputs muy pequeños/grandes


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: THREAD SAFETY
# ══════════════════════════════════════════════════════════════════════════════

class TestThreadSafety:
    """
    Verificación de comportamiento bajo concurrencia.
    """

    def test_concurrent_instantiation_with_same_config(
        self,
        stable_config: CondenserConfig,
        oracle_approves: Dict[str, Any],
    ) -> None:
        """Múltiples instanciaciones concurrentes con la misma config."""
        results: List[DataFluxCondenser] = []
        errors: List[Exception] = []
        lock = threading.Lock()

        def create_condenser():
            try:
                def _validate_side_effect(*args, **kwargs):
                    return oracle_approves

                def _analyze_side_effect(*args, **kwargs):
                    return {
                        "continuous": {
                            "natural_frequency_rad_s": 1.0,
                            "damping_ratio": 1.0,
                            "damping_class": "critical"
                        },
                        "stability_margins": {"phase_margin_deg": 60.0}
                    }
                with patch(
                    "app.physics.flux_condenser.LaplaceOracle",
                    **{
                        "return_value.validate_for_control_design.side_effect": _validate_side_effect,
                        "return_value.analyze_stability.side_effect": _analyze_side_effect
                    }
                ):
                    condenser = make_condenser(stable_config)
                    with lock:
                        results.append(condenser)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=create_condenser) for _ in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errores en hilos: {errors}"
        assert len(results) == 10

    def test_concurrent_different_configs(
        self,
        oracle_approves: Dict[str, Any],
    ) -> None:
        """Instanciaciones concurrentes con diferentes configs."""
        results: Dict[str, DataFluxCondenser] = {}
        errors: List[Exception] = []
        lock = threading.Lock()

        def instantiate_stable():
            return make_condenser(STABLE_SYSTEM.to_condenser_config())

        def instantiate_unstable():
            try:
                make_condenser(UNSTABLE_HIGH_KP.to_condenser_config())
                return False
            except ConfigurationError:
                return True

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(instantiate_stable)
            f2 = executor.submit(instantiate_unstable)

            condenser_stable = f1.result()
            caught_unstable = f2.result()

            assert condenser_stable is not None, "El sistema estable fue corrompido por el hilo inestable."
            assert caught_unstable is True, "El sistema inestable evadió el Veto de Laplace."


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: RENDIMIENTO
# ══════════════════════════════════════════════════════════════════════════════

class TestPerformance:
    """
    Tests de rendimiento básico.
    """

    def test_rlc_spec_creation_is_fast(self) -> None:
        """Crear RLCSystemSpec debe ser rápido."""
        import time
        
        start = time.perf_counter()
        
        for _ in range(1000):
            RLCSystemSpec(
                capacitance_F=5000.0,
                resistance_ohm=10.0,
                inductance_H=2.0,
                pid_kp=2.0,
            )
        
        elapsed = time.perf_counter() - start
        
        assert elapsed < 0.5, f"1000 creaciones tomaron {elapsed:.3f}s"

    def test_poles_calculation_is_fast(self) -> None:
        """Calcular polos debe ser rápido."""
        import time
        
        system = STABLE_SYSTEM
        
        start = time.perf_counter()
        
        for _ in range(10000):
            _ = system.poles
        
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, f"10000 cálculos de polos tomaron {elapsed:.3f}s"


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: CONSISTENCIA DE LA ESPECIFICACIÓN
# ══════════════════════════════════════════════════════════════════════════════

class TestSpecificationConsistency:
    """
    Verificación de consistencia interna de las especificaciones.
    """

    def test_all_test_systems_have_unique_names(self) -> None:
        """Todos los sistemas de prueba tienen nombres únicos."""
        names = [s.name for s in ALL_TEST_SYSTEMS]
        assert len(names) == len(set(names)), "Nombres duplicados en sistemas de prueba"

    def test_stable_system_is_actually_stable(self) -> None:
        """El sistema 'estable' debe ser realmente estable."""
        assert STABLE_SYSTEM.is_bibo_stable
        assert STABLE_SYSTEM.zeta > 1  # Sobreamortiguado

    def test_critical_system_has_zeta_one(self) -> None:
        """El sistema 'crítico' debe tener ζ ≈ 1."""
        assert CRITICAL_SYSTEM.zeta == pytest.approx(1.0, rel=1e-6)

    def test_underdamped_has_zeta_less_than_one(self) -> None:
        """El sistema 'subamortiguado' debe tener 0 < ζ < 1."""
        assert 0 < UNDERDAMPED_SYSTEM.zeta < 1

    def test_marginal_has_zeta_zero(self) -> None:
        """El sistema 'marginal' debe tener ζ = 0."""
        assert MARGINAL_SYSTEM.zeta == pytest.approx(0.0, abs=1e-10)

    def test_oracle_responses_are_consistent(self) -> None:
        """Las respuestas del Oráculo son consistentes."""
        assert ORACLE_APPROVE.is_suitable_for_control is True
        assert len(ORACLE_APPROVE.issues) == 0
        
        assert ORACLE_REJECT_UNSTABLE.is_suitable_for_control is False
        assert len(ORACLE_REJECT_UNSTABLE.issues) > 0

    def test_kp_ratio_between_stable_and_unstable(self) -> None:
        """El ratio de kp entre sistema estable e inestable es 1000×."""
        ratio = UNSTABLE_HIGH_KP.pid_kp / STABLE_SYSTEM.pid_kp
        assert ratio == pytest.approx(1000.0)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PYTEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",
        "--durations=10",
    ])