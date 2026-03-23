r"""
Módulo: Semantic Translator (El Intérprete Diplomático del Consejo)
===================================================================

Este componente actúa como el "Puente Cognitivo" entre la Matemática Profunda
(Topología, Finanzas, Termodinámica) y la Toma de Decisiones Ejecutiva.
Su función no es reportar datos, sino emitir "Veredictos" y narrativas de riesgo
basadas en la evidencia técnica recolectada por el resto del Consejo.

Fundamentos Teóricos y Arquitectura Algebraica
-----------------------------------------------

1. Retículo de Veredictos (Lattice Theory):
   Implementa una estructura algebraica de orden (VerdictLevel, ≤, ⊔, ⊓) donde:
   - VIABLE (⊥) ≤ CONDICIONAL ≤ REVISAR ≤ PRECAUCION ≤ RECHAZAR (⊤).
   - La operación de síntesis es el Supremo (⊔ / Join), adoptando siempre el
     criterio más conservador (Worst-Case Scenario) entre los agentes.
   - Propiedades verificadas formalmente:
       Idempotencia:    a ⊔ a = a,  a ⊓ a = a
       Conmutatividad:  a ⊔ b = b ⊔ a,  a ⊓ b = b ⊓ a
       Asociatividad:   (a ⊔ b) ⊔ c = a ⊔ (b ⊔ c)
       Absorción:       a ⊔ (a ⊓ b) = a,  a ⊓ (a ⊔ b) = a
       Identidad:       a ⊔ ⊥ = a,  a ⊓ ⊤ = a
       Distributividad: a ⊓ (b ⊔ c) = (a ⊓ b) ⊔ (a ⊓ c)  [cadena lineal]

2. Invariantes Topológicos:
   - Característica de Euler: χ = β₀ − β₁ + β₂  (invariante homotópico)
   - Conectividad algebraica: λ₂ (valor de Fiedler) ≥ 0;  λ₂ > 0 ⟺ grafo conexo
   - Brecha espectral: gap = λ₂ / λ_max  (normalizada, mide robustez de conectividad)
   - Índice de estabilidad piramidal: Ψ = Σ weights(base) / Σ weights(apex)
   - Genus topológico: g = β₁ para complejos simpliciales 1-dimensionales (grafos)
     (Para superficies orientables cerradas: g = β₁/2; se documenta la convención)

3. Síntesis DIKW (Data → Wisdom):
   Eleva los datos crudos a Sabiduría mediante la integración de contextos:
   - β₁ > 0  (Topología)       →  "Socavón Lógico"  (Bloqueo administrativo)
   - Ψ < 1.0 (Física)          →  "Pirámide Invertida"  (Riesgo de colapso logístico)
   - T_sys > 50 °C (Termodinámica) →  "Fiebre Inflacionaria"

4. Narrativa Generativa Causal (GraphRAG):
   Retrieval-Augmented Generation sobre Grafos para explicar causalidad.

5. Homomorfismo de Retículos:
   El funtor T: SeverityLattice → VerdictLevel es un homomorfismo inyectivo
   de retículos que preserva ⊔ y ⊓.  Se verifica formalmente en
   ``SeverityToVerdictHomomorphism``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum, Enum, auto
from functools import total_ordering
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Final,
    FrozenSet,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

import networkx as nx

# Importaciones del proyecto
from app.core.telemetry_schemas import (
    PhysicsMetrics,
    TopologicalMetrics,
    ControlMetrics,
    ThermodynamicMetrics,
)
from app.adapters.tools_interface import MICRegistry, register_core_vectors

try:
    from app.core.schemas import Stratum
except ImportError:
    from enum import IntEnum as _StratumBase

    class Stratum(_StratumBase):  # type: ignore[no-redef]
        WISDOM = 0
        OMEGA = 1
        ALPHA = 2
        STRATEGY = 3
        TACTICS = 4
        PHYSICS = 5


logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES FÍSICAS Y NUMÉRICAS
# ============================================================================

T = TypeVar("T")

# Constantes termodinámicas
ABSOLUTE_ZERO_CELSIUS: Final[float] = -273.15
KELVIN_OFFSET: Final[float] = 273.15

# Tolerancias numéricas
EPSILON: Final[float] = 1e-9

# Límites computacionales para protección contra explosión combinatoria
MAX_SIMPLE_CYCLES_ENUMERATION: Final[int] = 1000
MAX_GRAPH_NODES_FOR_EIGENVECTOR: Final[int] = 50_000


# ============================================================================
# SISTEMA DE UNIDADES PARA TEMPERATURA
# ============================================================================


@dataclass(frozen=True, order=True)
class Temperature:
    """
    Valor-objeto inmutable para temperatura con conversión automática.

    Almacena internamente en Kelvin para garantizar el invariante T ≥ 0.
    Implementa orden total vía ``@dataclass(order=True)`` sobre ``kelvin``.

    Invariante de clase:
        ∀ t : Temperature,  t.kelvin ≥ 0
    """

    kelvin: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.kelvin):
            raise ValueError(
                f"Temperature must be finite, got {self.kelvin}"
            )
        if self.kelvin < -EPSILON:
            raise ValueError(
                f"Temperature cannot be below absolute zero: {self.kelvin} K"
            )
        # Clamp valores negativos minúsculos a cero exacto
        if self.kelvin < 0:
            object.__setattr__(self, "kelvin", 0.0)

    @classmethod
    def from_celsius(cls, celsius: float) -> Temperature:
        """Construye desde grados Celsius."""
        return cls(kelvin=celsius + KELVIN_OFFSET)

    @classmethod
    def from_kelvin(cls, kelvin: float) -> Temperature:
        """Construye desde Kelvin."""
        return cls(kelvin=kelvin)

    @property
    def celsius(self) -> float:
        """Convierte a Celsius."""
        return self.kelvin - KELVIN_OFFSET

    @property
    def is_absolute_zero(self) -> bool:
        """Verifica si es cero absoluto (dentro de tolerancia ε)."""
        return self.kelvin < EPSILON

    def __str__(self) -> str:
        return f"{self.celsius:.1f}°C ({self.kelvin:.1f}K)"

    def __repr__(self) -> str:
        return f"Temperature(kelvin={self.kelvin!r})"


# ============================================================================
# EXCEPCIONES ESPECÍFICAS DEL DOMINIO
# ============================================================================


class SemanticTranslatorError(Exception):
    """Excepción base del traductor semántico."""
    pass


class TopologyInvariantViolation(SemanticTranslatorError):
    """Violación de invariante topológico (χ ≠ β₀ − β₁ + β₂)."""

    def __init__(
        self,
        message: str,
        beta_0: int,
        beta_1: int,
        beta_2: int,
        chi: int,
    ) -> None:
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.chi = chi
        self.expected_chi = beta_0 - beta_1 + beta_2
        super().__init__(
            f"{message}. χ={chi}, pero β₀−β₁+β₂={self.expected_chi} "
            f"(β₀={beta_0}, β₁={beta_1}, β₂={beta_2})"
        )


class LatticeViolation(SemanticTranslatorError):
    """Violación de propiedad del retículo algebraico."""
    pass


class MetricsValidationError(SemanticTranslatorError):
    """Error en validación de métricas de entrada."""
    pass


class GraphStructureError(SemanticTranslatorError):
    """Error en la estructura del grafo de dependencias."""
    pass


# ============================================================================
# CONFIGURACIÓN CENTRALIZADA CON VALIDACIÓN ESTRICTA
# ============================================================================


@dataclass(frozen=True)
class StabilityThresholds:
    """
    Umbrales para el índice de estabilidad piramidal (Ψ).

    Fundamentación topológica:
    - Ψ < critical:  Pirámide Invertida  (cimentación insuficiente)
    - Ψ ∈ [critical, warning):  Equilibrio isostático  (precario)
    - Ψ ∈ [warning, solid):     Estabilidad moderada
    - Ψ ≥ solid:                Estructura antisísmica  (base robusta)

    Invariante de orden estricto:
        0 ≤ critical < warning < solid
    """

    critical: float = 1.0
    warning: float = 3.0
    solid: float = 10.0

    def __post_init__(self) -> None:
        for name, val in [("critical", self.critical), ("warning", self.warning), ("solid", self.solid)]:
            if not math.isfinite(val):
                raise ValueError(f"{name} threshold must be finite, got {val}")
        if self.critical < 0:
            raise ValueError(f"critical threshold must be non-negative, got {self.critical}")
        if not (self.critical < self.warning < self.solid):
            raise ValueError(
                f"Thresholds must satisfy strict ordering: "
                f"critical({self.critical}) < warning({self.warning}) < solid({self.solid})"
            )

    def classify(self, stability: float) -> str:
        """
        Clasifica un valor de estabilidad Ψ.

        Returns:
            Uno de: ``'invalid'``, ``'critical'``, ``'warning'``,
            ``'stable'``, ``'robust'``
        """
        if not math.isfinite(stability) or stability < 0:
            return "invalid"
        if stability < self.critical:
            return "critical"
        if stability < self.warning:
            return "warning"
        if stability < self.solid:
            return "stable"
        return "robust"

    def severity_score(self, stability: float) -> float:
        """
        Score de severidad normalizado en [0, 1].

        Mapeo: 0 = robusto  →  1 = crítico.
        Usa interpolación lineal inversa con saturación.
        """
        if not math.isfinite(stability) or stability < 0:
            return 1.0
        if stability >= self.solid:
            return 0.0
        return max(0.0, 1.0 - stability / self.solid)


@dataclass(frozen=True)
class TopologicalThresholds:
    """
    Umbrales para interpretación de números de Betti y métricas espectrales.

    Fundamentación algebraica:
    - β₀ (componentes conexos): Mide fragmentación del complejo simplicial
    - β₁ (1-ciclos independientes): Mide "agujeros" (genus para grafos)
    - β₂ (cavidades):  Típicamente 0 para grafos planos
    - χ  (Euler):  χ = β₀ − β₁ + β₂

    Invariantes de la configuración:
    - connected_components_optimal ≥ 1
    - 0 ≤ cycles_optimal ≤ cycles_warning ≤ cycles_critical
    - 0 ≤ fiedler_connected_threshold < fiedler_robust_threshold
    """

    connected_components_optimal: int = 1
    cycles_optimal: int = 0
    cycles_warning: int = 1
    cycles_critical: int = 3
    max_fragmentation: int = 5

    # Umbrales espectrales (valor de Fiedler = segunda menor eigenvalue del Laplaciano)
    fiedler_connected_threshold: float = 1e-6  # λ₂ > ε ⟹ conexo
    fiedler_robust_threshold: float = 0.5       # Alta conectividad algebraica

    def __post_init__(self) -> None:
        if self.connected_components_optimal < 1:
            raise ValueError(
                f"optimal connected components must be ≥ 1, got {self.connected_components_optimal}"
            )
        if self.cycles_optimal < 0:
            raise ValueError(f"optimal cycles must be non-negative, got {self.cycles_optimal}")
        if not (self.cycles_optimal <= self.cycles_warning <= self.cycles_critical):
            raise ValueError(
                f"Cycle thresholds must be ordered: "
                f"{self.cycles_optimal} ≤ {self.cycles_warning} ≤ {self.cycles_critical}"
            )
        if self.max_fragmentation < self.connected_components_optimal:
            raise ValueError(
                f"max_fragmentation ({self.max_fragmentation}) must be ≥ "
                f"connected_components_optimal ({self.connected_components_optimal})"
            )
        if self.fiedler_connected_threshold < 0:
            raise ValueError("Fiedler connected threshold must be non-negative")
        if self.fiedler_connected_threshold >= self.fiedler_robust_threshold:
            raise ValueError(
                f"fiedler_connected_threshold ({self.fiedler_connected_threshold}) "
                f"must be < fiedler_robust_threshold ({self.fiedler_robust_threshold})"
            )

    def classify_connectivity(self, beta_0: int) -> str:
        """Clasifica nivel de conectividad por β₀."""
        if beta_0 <= 0:
            return "empty"
        if beta_0 == self.connected_components_optimal:
            return "unified"
        if beta_0 <= self.max_fragmentation:
            return "fragmented"
        return "severely_fragmented"

    def classify_cycles(self, beta_1: int) -> str:
        """Clasifica nivel de ciclos por β₁."""
        if beta_1 < 0:
            return "invalid"
        if beta_1 <= self.cycles_optimal:
            return "clean"
        if beta_1 <= self.cycles_warning:
            return "minor"
        if beta_1 <= self.cycles_critical:
            return "moderate"
        return "critical"

    def classify_spectral_connectivity(self, fiedler_value: float) -> str:
        """
        Clasifica conectividad usando el valor de Fiedler λ₂.

        Teorema (Fiedler, 1973):  λ₂ > 0  ⟺  G es conexo.
        λ₂ grande  ⟹  alta robustez de conectividad (difícil de desconectar).
        """
        if not math.isfinite(fiedler_value):
            return "invalid"
        if fiedler_value < 0:
            # λ₂ del Laplaciano simétrico es siempre ≥ 0.
            # Valores negativos indican error numérico; tratar como desconectado.
            return "disconnected"
        if fiedler_value < self.fiedler_connected_threshold:
            return "disconnected"
        if fiedler_value < self.fiedler_robust_threshold:
            return "weakly_connected"
        return "strongly_connected"

    @staticmethod
    def validate_euler_characteristic(
        beta_0: int, beta_1: int, beta_2: int, chi: int,
    ) -> bool:
        """
        Valida el invariante de Euler:  χ = β₀ − β₁ + β₂.

        Para grafos (1-complejos simpliciales), β₂ = 0 en general.
        Para 2-complejos simpliciales, β₂ cuenta las cavidades.
        """
        return chi == beta_0 - beta_1 + beta_2


@dataclass(frozen=True)
class ThermalThresholds:
    """
    Umbrales para métricas termodinámicas.

    Metáfora:  El proyecto como sistema termodinámico:
    - Temperatura → Volatilidad / Inflación de precios
    - Entropía    → Desorden administrativo  (S ∈ [0, 1] normalizada)
    - Exergía     → Eficiencia de inversión  (η_ex ∈ [0, 1])
    - C_v         → Inercia financiera  (capacidad calorífica)

    Invariantes:
    - temperature_cold < temperature_warm < temperature_hot < temperature_critical
    - 0 ≤ entropy_low < entropy_high < entropy_death ≤ 1
    - 0 ≤ exergy_poor < exergy_efficient ≤ 1
    - heat_capacity_minimum > 0
    """

    temperature_cold: float = 20.0
    temperature_warm: float = 35.0
    temperature_hot: float = 50.0
    temperature_critical: float = 75.0

    entropy_low: float = 0.3
    entropy_high: float = 0.7
    entropy_death: float = 0.95

    exergy_efficient: float = 0.7
    exergy_poor: float = 0.3

    heat_capacity_minimum: float = 0.2

    def __post_init__(self) -> None:
        temps = [
            self.temperature_cold,
            self.temperature_warm,
            self.temperature_hot,
            self.temperature_critical,
        ]
        for i, t in enumerate(temps):
            if not math.isfinite(t):
                raise ValueError(f"Temperature threshold at index {i} must be finite")
        if temps != sorted(temps) or len(set(temps)) != len(temps):
            raise ValueError(
                f"Temperature thresholds must be strictly increasing: {temps}"
            )
        if not (0 <= self.entropy_low < self.entropy_high < self.entropy_death <= 1):
            raise ValueError(
                f"Entropy thresholds must satisfy: "
                f"0 ≤ {self.entropy_low} < {self.entropy_high} < {self.entropy_death} ≤ 1"
            )
        if not (0 <= self.exergy_poor < self.exergy_efficient <= 1):
            raise ValueError(
                f"Exergy thresholds must satisfy: "
                f"0 ≤ {self.exergy_poor} < {self.exergy_efficient} ≤ 1"
            )
        if self.heat_capacity_minimum <= 0:
            raise ValueError(
                f"heat_capacity_minimum must be positive, got {self.heat_capacity_minimum}"
            )

    def classify_temperature(self, temp_celsius: float) -> str:
        """Clasifica temperatura del sistema en grados Celsius."""
        if not math.isfinite(temp_celsius):
            return "invalid"
        if temp_celsius < ABSOLUTE_ZERO_CELSIUS:
            return "invalid"
        if temp_celsius <= self.temperature_cold:
            return "cold"
        if temp_celsius <= self.temperature_warm:
            return "stable"
        if temp_celsius <= self.temperature_hot:
            return "warm"
        if temp_celsius <= self.temperature_critical:
            return "hot"
        return "critical"

    def classify_entropy(self, entropy: float) -> str:
        """Clasifica nivel de entropía normalizada S ∈ [0, 1]."""
        if not math.isfinite(entropy) or entropy < 0 or entropy > 1:
            return "invalid"
        if entropy >= self.entropy_death:
            return "death"
        if entropy >= self.entropy_high:
            return "high"
        if entropy <= self.entropy_low:
            return "low"
        return "moderate"

    def classify_exergy(self, exergy: float) -> str:
        """Clasifica eficiencia exergética η_ex ∈ [0, 1]."""
        if not math.isfinite(exergy) or exergy < 0 or exergy > 1:
            return "invalid"
        if exergy >= self.exergy_efficient:
            return "efficient"
        if exergy <= self.exergy_poor:
            return "poor"
        return "moderate"


@dataclass(frozen=True)
class FinancialThresholds:
    """
    Umbrales para métricas financieras.

    - WACC:  Costo de Capital Ponderado
    - PI:    Índice de Rentabilidad (Profitability Index = PV / I₀)

    Invariantes:
    - 0 < wacc_low < wacc_moderate < wacc_high
    - 0 < profitability_marginal < profitability_good < profitability_excellent
    - 0 < contingency_minimal < contingency_standard < contingency_high
    """

    wacc_low: float = 0.05
    wacc_moderate: float = 0.10
    wacc_high: float = 0.15

    profitability_excellent: float = 1.5
    profitability_good: float = 1.2
    profitability_marginal: float = 1.0

    contingency_minimal: float = 0.05
    contingency_standard: float = 0.10
    contingency_high: float = 0.20

    def __post_init__(self) -> None:
        if not (0 < self.wacc_low < self.wacc_moderate < self.wacc_high):
            raise ValueError(
                f"WACC thresholds must be strictly increasing and positive: "
                f"{self.wacc_low}, {self.wacc_moderate}, {self.wacc_high}"
            )
        if not (0 < self.profitability_marginal < self.profitability_good < self.profitability_excellent):
            raise ValueError(
                f"Profitability thresholds must be strictly increasing and positive: "
                f"{self.profitability_marginal}, {self.profitability_good}, {self.profitability_excellent}"
            )
        if not (0 < self.contingency_minimal < self.contingency_standard < self.contingency_high):
            raise ValueError(
                f"Contingency thresholds must be strictly increasing and positive: "
                f"{self.contingency_minimal}, {self.contingency_standard}, {self.contingency_high}"
            )

    def classify_profitability(self, pi: float) -> str:
        """Clasifica índice de rentabilidad."""
        if not math.isfinite(pi):
            return "invalid"
        if pi >= self.profitability_excellent:
            return "excellent"
        if pi >= self.profitability_good:
            return "good"
        if pi >= self.profitability_marginal:
            return "marginal"
        return "poor"


@dataclass(frozen=True)
class TranslatorConfig:
    """
    Configuración consolidada del traductor semántico.

    Agrupa todos los umbrales y configuraciones en una única estructura inmutable.
    """

    stability: StabilityThresholds = field(default_factory=StabilityThresholds)
    topology: TopologicalThresholds = field(default_factory=TopologicalThresholds)
    thermal: ThermalThresholds = field(default_factory=ThermalThresholds)
    financial: FinancialThresholds = field(default_factory=FinancialThresholds)

    # Límites de procesamiento
    max_cycle_path_display: int = 5
    max_stress_points_display: int = 3
    max_narrative_length: int = 10_000
    max_cycles_to_enumerate: int = MAX_SIMPLE_CYCLES_ENUMERATION

    # Configuración de determinismo
    deterministic_market: bool = True
    default_market_index: int = 0

    # Validación de invariantes topológicos
    strict_euler_validation: bool = True

    def __post_init__(self) -> None:
        if self.max_cycle_path_display < 1:
            raise ValueError("max_cycle_path_display must be ≥ 1")
        if self.max_stress_points_display < 1:
            raise ValueError("max_stress_points_display must be ≥ 1")
        if self.max_narrative_length < 100:
            raise ValueError("max_narrative_length must be ≥ 100")
        if self.max_cycles_to_enumerate < 1:
            raise ValueError("max_cycles_to_enumerate must be ≥ 1")


# ============================================================================
# LATTICE DE VEREDICTOS — ESTRUCTURA ALGEBRAICA PRIMARIA
# ============================================================================


class VerdictLevel(IntEnum):
    """
    Retículo acotado distributivo totalmente ordenado de veredictos.

    Estructura algebraica:  (VerdictLevel, ≤, ⊔, ⊓)

    El orden total induce un retículo distributivo donde:
    - ⊔ (join / supremum) = max   [criterio conservador: peor caso]
    - ⊓ (meet / infimum)  = min   [mejor caso]

    Propiedades verificables formalmente (ver ``verify_lattice_laws``):
    1. Idempotencia:    a ⊔ a = a,  a ⊓ a = a
    2. Conmutatividad:  a ⊔ b = b ⊔ a,  a ⊓ b = b ⊓ a
    3. Asociatividad:   (a ⊔ b) ⊔ c = a ⊔ (b ⊔ c)
    4. Absorción:       a ⊔ (a ⊓ b) = a,  a ⊓ (a ⊔ b) = a
    5. Identidad:       a ⊔ ⊥ = a,  a ⊓ ⊤ = a
    6. Distributividad: a ⊓ (b ⊔ c) = (a ⊓ b) ⊔ (a ⊓ c)
                        (se cumple automáticamente en cadenas lineales)

    Semántica del orden (menor es mejor):
    - VIABLE (⊥):     Proyecto puede proceder sin restricciones
    - CONDICIONAL:     Viable con modificaciones menores
    - REVISAR:         Requiere análisis adicional antes de decidir
    - PRECAUCION:      Riesgos significativos identificados
    - RECHAZAR (⊤):    No viable en estado actual
    """

    VIABLE = 0        # ⊥ — Bottom (mejor caso)
    CONDICIONAL = 1   # Viable con condiciones
    REVISAR = 2       # Necesita más análisis
    PRECAUCION = 3    # Advertencia significativa
    RECHAZAR = 4      # ⊤ — Top (peor caso)

    # -- Elementos distinguidos del retículo --

    @classmethod
    def bottom(cls) -> VerdictLevel:
        """Elemento mínimo ⊥ del retículo."""
        return cls.VIABLE

    @classmethod
    def top(cls) -> VerdictLevel:
        """Elemento máximo ⊤ del retículo."""
        return cls.RECHAZAR

    # -- Operaciones del retículo --

    @classmethod
    def supremum(cls, *levels: VerdictLevel) -> VerdictLevel:
        """
        JOIN (⊔): supremo de un conjunto de veredictos.

        Semántica: adopta el peor caso (criterio conservador).

        Propiedad:  ⊔∅ = ⊥  (el supremo del vacío es el bottom,
        por convención estándar en retículos completos).
        """
        if not levels:
            return cls.bottom()
        return cls(max(level.value for level in levels))

    @classmethod
    def infimum(cls, *levels: VerdictLevel) -> VerdictLevel:
        """
        MEET (⊓): ínfimo de un conjunto de veredictos.

        Semántica: adopta el mejor caso.

        Propiedad:  ⊓∅ = ⊤  (el ínfimo del vacío es el top).
        """
        if not levels:
            return cls.top()
        return cls(min(level.value for level in levels))

    def join(self, other: VerdictLevel) -> VerdictLevel:
        """Join binario: self ⊔ other = max(self, other)."""
        return VerdictLevel(max(self.value, other.value))

    def meet(self, other: VerdictLevel) -> VerdictLevel:
        """Meet binario: self ⊓ other = min(self, other)."""
        return VerdictLevel(min(self.value, other.value))

    def __or__(self, other: VerdictLevel) -> VerdictLevel:
        """Sintaxis operador: ``a | b`` ≡ ``a ⊔ b`` (join)."""
        if not isinstance(other, VerdictLevel):
            return NotImplemented
        return self.join(other)

    def __and__(self, other: VerdictLevel) -> VerdictLevel:
        """Sintaxis operador: ``a & b`` ≡ ``a ⊓ b`` (meet)."""
        if not isinstance(other, VerdictLevel):
            return NotImplemented
        return self.meet(other)

    # -- Propiedades semánticas --

    @property
    def emoji(self) -> str:
        """Representación visual del veredicto."""
        _MAP: Dict[int, str] = {
            0: "✅", 1: "🔵", 2: "🔍", 3: "⚠️", 4: "🛑",
        }
        return _MAP[self.value]

    @property
    def description(self) -> str:
        """Descripción textual del veredicto."""
        _MAP: Dict[int, str] = {
            0: "Proyecto viable sin restricciones",
            1: "Viable con condiciones",
            2: "Requiere análisis adicional",
            3: "Riesgos significativos",
            4: "No viable",
        }
        return _MAP[self.value]

    @property
    def is_positive(self) -> bool:
        """¿Permite proceder? (VIABLE o CONDICIONAL)."""
        return self.value <= VerdictLevel.CONDICIONAL.value

    @property
    def is_negative(self) -> bool:
        """¿Bloquea el proyecto? (RECHAZAR)."""
        return self == VerdictLevel.RECHAZAR

    @property
    def requires_attention(self) -> bool:
        """¿Requiere atención inmediata? (PRECAUCION o RECHAZAR)."""
        return self.value >= VerdictLevel.PRECAUCION.value

    @property
    def normalized_score(self) -> float:
        """Score normalizado [0, 1] donde 0 = viable, 1 = rechazar."""
        top_val = VerdictLevel.RECHAZAR.value
        return self.value / top_val if top_val > 0 else 0.0

    # -- Verificación formal --

    @classmethod
    def verify_lattice_laws(cls) -> Dict[str, bool]:
        """
        Verifica exhaustivamente las leyes del retículo para todos los elementos.

        Complejidad: O(n³) donde n = |VerdictLevel| = 5.  Aceptable para testing.

        Returns:
            Diccionario  nombre_propiedad → bool
        """
        elems = list(cls)
        results: Dict[str, bool] = {}

        # 1. Idempotencia
        results["idempotent_join"] = all(a | a == a for a in elems)
        results["idempotent_meet"] = all(a & a == a for a in elems)

        # 2. Conmutatividad
        results["commutative_join"] = all(
            a | b == b | a for a in elems for b in elems
        )
        results["commutative_meet"] = all(
            a & b == b & a for a in elems for b in elems
        )

        # 3. Asociatividad
        results["associative_join"] = all(
            (a | b) | c == a | (b | c)
            for a in elems for b in elems for c in elems
        )
        results["associative_meet"] = all(
            (a & b) & c == a & (b & c)
            for a in elems for b in elems for c in elems
        )

        # 4. Absorción
        results["absorption_join_meet"] = all(
            a | (a & b) == a for a in elems for b in elems
        )
        results["absorption_meet_join"] = all(
            a & (a | b) == a for a in elems for b in elems
        )

        # 5. Identidad
        results["identity_join_bottom"] = all(a | cls.bottom() == a for a in elems)
        results["identity_meet_top"] = all(a & cls.top() == a for a in elems)

        # 6. Distributividad  (a ⊓ (b ⊔ c) = (a ⊓ b) ⊔ (a ⊓ c))
        results["distributive_meet_over_join"] = all(
            a & (b | c) == (a & b) | (a & c)
            for a in elems for b in elems for c in elems
        )
        results["distributive_join_over_meet"] = all(
            a | (b & c) == (a | b) & (a | c)
            for a in elems for b in elems for c in elems
        )

        # 7. Antisimetría del orden parcial  (a ≤ b ∧ b ≤ a ⟹ a = b)
        results["antisymmetric"] = all(
            not (a <= b and b <= a) or a == b
            for a in elems for b in elems
        )

        # 8. Transitividad del orden  (a ≤ b ∧ b ≤ c ⟹ a ≤ c)
        results["transitive"] = all(
            not (a <= b and b <= c) or a <= c
            for a in elems for b in elems for c in elems
        )

        return results


# ============================================================================
# LATTICE LEGACY — HOMOMORFISMO VERIFICADO
# ============================================================================


class SeverityLattice(IntEnum):
    """
    Retículo de severidad simplificado (3 elementos) para el eje Ω.

    Orden:  VIABLE (⊥) < PRECAUCION < RECHAZAR (⊤)

    Este retículo es una retracción (quotient) de ``VerdictLevel``:
    El homomorfismo φ: SeverityLattice → VerdictLevel se verifica en
    ``SeverityToVerdictHomomorphism``.
    """
    VIABLE = 0       # ⊥
    PRECAUCION = 1
    RECHAZAR = 2     # ⊤

    @classmethod
    def supremum(cls, *levels: SeverityLattice) -> SeverityLattice:
        """⊔ (join) del retículo de severidad."""
        if not levels:
            return cls.VIABLE
        return cls(max(l.value for l in levels))


class SeverityToVerdictHomomorphism:
    """
    Homomorfismo inyectivo de retículos  φ: SeverityLattice → VerdictLevel.

    Verifica que φ preserva ⊔ y ⊓:
        φ(a ⊔ b) = φ(a) ⊔ φ(b)
        φ(a ⊓ b) = φ(a) ⊓ φ(b)
    """

    _MAP: ClassVar[Dict[SeverityLattice, VerdictLevel]] = {
        SeverityLattice.VIABLE: VerdictLevel.VIABLE,
        SeverityLattice.PRECAUCION: VerdictLevel.PRECAUCION,
        SeverityLattice.RECHAZAR: VerdictLevel.RECHAZAR,
    }

    @classmethod
    def apply(cls, severity: SeverityLattice) -> VerdictLevel:
        """Aplica el homomorfismo  φ(severity)."""
        return cls._MAP[severity]

    @classmethod
    def verify_homomorphism(cls) -> bool:
        """
        Verifica formalmente que φ preserva join y meet.

        Para un retículo de 3 elementos, son 9 pares a verificar.
        """
        all_sev = list(SeverityLattice)
        for a in all_sev:
            for b in all_sev:
                # Preservación de join
                join_sev = SeverityLattice(max(a.value, b.value))
                if cls.apply(join_sev) != cls.apply(a) | cls.apply(b):
                    return False
                # Preservación de meet
                meet_sev = SeverityLattice(min(a.value, b.value))
                if cls.apply(meet_sev) != cls.apply(a) & cls.apply(b):
                    return False
        return True


# ============================================================================
# DIFEOMORFISMO SEMÁNTICO Y GENERACIÓN CAUSAL
# ============================================================================


class SemanticDiffeomorphismMapper:
    """
    Funtor F: InvariantSpace → ImpactSpace que mapea invariantes topológicos
    al espacio de impacto de negocio.

    Actúa como la 'Piedra Rosetta' del Estrato Ω, asegurando que la matemática
    pura se traduzca en 'Empatía Táctica' sin pérdida de rigor causal.
    """

    @staticmethod
    def map_betti_1_cycles(cycle_nodes: List[str]) -> str:
        """
        Mapeo de Socavón Lógico (β₁ > 0).

        Transforma la detección de ciclos de homología en un diagnóstico
        de parálisis administrativa.
        """
        if not cycle_nodes:
            return ""
        chain_str = " ➔ ".join(cycle_nodes) + f" ➔ {cycle_nodes[0]}"
        return (
            f"VETO ESTRUCTURAL (Socavón Lógico): Se ha detectado una dependencia "
            f"circular infinita en la ruta crítica. El sistema de compras se "
            f"paralizará porque la adquisición de estos insumos requiere que "
            f"se completen primero entre sí. "
            f"Cadena patológica: {chain_str}. "
            f"Recomendación: Romper el bucle reasignando dependencias."
        )

    @staticmethod
    def map_pyramid_instability(psi_value: float, critical_supplier: str) -> str:
        """
        Mapeo de Pirámide Invertida (Ψ < 1.0).

        Traduce la métrica de estabilidad a un escenario de quiebra logística.
        """
        return (
            f"ALERTA DE QUIEBRA (Pirámide Invertida): Su presupuesto descansa "
            f"sobre una base logística peligrosamente estrecha "
            f"(Índice Ψ = {psi_value:.2f}). "
            f"El proveedor o insumo '{critical_supplier}' soporta un peso "
            f"excesivo en la matriz de la obra. "
            f"Si este nodo falla, el choque logístico no se amortiguará y "
            f"colapsará el proyecto entero. "
            f"Recomendación: Diversificar inmediatamente la matriz de "
            f"proveedores base."
        )

    @staticmethod
    def map_betti_0_fragmentation(
        isolated_clusters: int, cost_at_risk: float,
    ) -> str:
        """
        Mapeo de Fragmentación (β₀ > 1).

        Mapea componentes conexas a recursos financieros ciegos.
        """
        return (
            f"FUGA DE CAPITAL (Recursos Huérfanos): El análisis topológico "
            f"revela {isolated_clusters} islas de datos desconectadas del "
            f"proyecto central. Usted está programando una inversión de "
            f"${cost_at_risk:,.2f} en materiales que no están enlazados a "
            f"ninguna actividad constructiva real. "
            f"Riesgo inminente de fraude o desperdicio seguro."
        )


class GraphRAGCausalNarrator:
    """
    Sintetiza narrativa causal inyectando la ruta del error topológico
    directamente en el contexto del Intérprete Diplomático.

    Protecciones contra explosión combinatoria:
    - Límite en enumeración de ciclos simples
    - Fallback seguro para centralidad de autovector
    """

    def __init__(
        self,
        knowledge_graph: nx.DiGraph,
        max_cycles: int = MAX_SIMPLE_CYCLES_ENUMERATION,
    ) -> None:
        if not isinstance(knowledge_graph, nx.DiGraph):
            raise GraphStructureError(
                f"Expected nx.DiGraph, got {type(knowledge_graph).__name__}"
            )
        self.kg = knowledge_graph
        self._max_cycles = max(1, max_cycles)

    def narrate_topological_collapse(
        self, betti_1: int, psi: float,
    ) -> str:
        """
        Orquesta el Colapso Semántico: traduce métricas puras en un
        Acta de Deliberación accionable.
        """
        acta_sections: List[str] = [
            "--- ACTA DE DELIBERACIÓN DEL CONSEJO DE SABIOS ---"
        ]

        # 1. Evaluación de β₁ (ciclos)
        if betti_1 > 0:
            cycles = self._enumerate_cycles_bounded()
            if cycles:
                critical_cycle = cycles[0]
                narrative = SemanticDiffeomorphismMapper.map_betti_1_cycles(
                    critical_cycle
                )
                acta_sections.append(narrative)

        # 2. Evaluación de Ψ (estabilidad)
        if psi < 1.0:
            critical_node = self._find_critical_node()
            narrative = SemanticDiffeomorphismMapper.map_pyramid_instability(
                psi, critical_node,
            )
            acta_sections.append(narrative)

        if len(acta_sections) == 1:
            acta_sections.append(
                "ESTADO NOMINAL: La topología del presupuesto es laminar y coherente."
            )

        return "\n\n".join(acta_sections)

    def _enumerate_cycles_bounded(self) -> List[List[str]]:
        """
        Enumera ciclos simples con límite superior para evitar
        explosión combinatoria en grafos densos.

        Complejidad: O(min(max_cycles, |ciclos|) × (V + E))
        """
        if self.kg.number_of_nodes() == 0:
            return []

        cycles: List[List[str]] = []
        try:
            for i, cycle in enumerate(nx.simple_cycles(self.kg)):
                if i >= self._max_cycles:
                    logger.warning(
                        f"Cycle enumeration truncated at {self._max_cycles} cycles"
                    )
                    break
                cycles.append([str(n) for n in cycle])
        except nx.NetworkXError as e:
            logger.warning(f"Cycle enumeration failed: {e}")
        return cycles

    def _find_critical_node(self) -> str:
        """
        Determina el nodo hiper-cargado via centralidad de autovector.

        Precondiciones verificadas:
        - Grafo no vacío
        - Grafo fuertemente conexo (requerido para eigenvector_centrality)
        - Si no es fuertemente conexo, se usa degree_centrality como fallback
        """
        if self.kg.number_of_nodes() == 0:
            return "Nodo Central Desconocido"

        if self.kg.number_of_nodes() > MAX_GRAPH_NODES_FOR_EIGENVECTOR:
            return self._find_critical_node_by_degree()

        # Verificar conexidad fuerte (requerida para eigenvector_centrality)
        if not nx.is_strongly_connected(self.kg):
            return self._find_critical_node_by_degree()

        try:
            centrality = nx.eigenvector_centrality_numpy(self.kg)
            if not centrality:
                return "Nodo Central Desconocido"
            return str(max(centrality, key=centrality.get))
        except (nx.NetworkXError, ArithmeticError) as e:
            logger.warning(f"Eigenvector centrality failed: {e}")
            return self._find_critical_node_by_degree()

    def _find_critical_node_by_degree(self) -> str:
        """Fallback: nodo con mayor grado total (in + out)."""
        if self.kg.number_of_nodes() == 0:
            return "Nodo Central Desconocido"
        node = max(
            self.kg.nodes(),
            key=lambda n: self.kg.in_degree(n) + self.kg.out_degree(n),
        )
        return str(node)


class LatticeVerdictCollapse:
    """
    Motor determinista que subordina las decisiones del LLM al rigor matemático.

    Implementa la operación de colapso del retículo:
    el veredicto final es el supremo (⊔) de los veredictos individuales.
    """

    @staticmethod
    def compute_supremum(
        financial_verdict: SeverityLattice,
        topological_verdict: SeverityLattice,
    ) -> SeverityLattice:
        """
        Aplica ⊔ entre los dictámenes del Oráculo y el Arquitecto.

        Ejemplo:  VIABLE ⊔ RECHAZAR = RECHAZAR

        El rigor topológico/físico siempre domina sobre la avaricia financiera.
        """
        # Absorbing element optimization (⊤ = RECHAZAR)
        if topological_verdict == SeverityLattice.RECHAZAR:
            return SeverityLattice.RECHAZAR
        if financial_verdict == SeverityLattice.RECHAZAR:
            return SeverityLattice.RECHAZAR
        return SeverityLattice.supremum(financial_verdict, topological_verdict)

    @staticmethod
    def enforce_semantic_diffeomorphism(
        roi_viable: bool,
        betti_1: int,
        psi: float,
        graph: nx.DiGraph,
        max_cycles: int = MAX_SIMPLE_CYCLES_ENUMERATION,
    ) -> str:
        """
        Punto de entrada final para el Estrato Ω.

        Colapsa la función de decisión y emite el Acta de Deliberación inmutable.

        Args:
            roi_viable: ¿El ROI es positivo?
            betti_1: Primer número de Betti (ciclos independientes)
            psi: Índice de estabilidad piramidal Ψ
            graph: Grafo de dependencias del proyecto
            max_cycles: Límite de enumeración de ciclos

        Returns:
            Acta de Deliberación con veredicto y justificación
        """
        # 1. Evaluación topológica (Fast-Fail)
        topo_verdict = SeverityLattice.VIABLE
        if betti_1 > 0 or psi < 1.0:
            topo_verdict = SeverityLattice.RECHAZAR  # Veto físico inmediato

        # Absorbing element check (⊤): if topography rejects, bypass financial calculations
        if topo_verdict == SeverityLattice.RECHAZAR:
            fin_verdict = SeverityLattice.VIABLE  # Dummy value, will be absorbed
            final_severity = SeverityLattice.RECHAZAR
        else:
            # Only evaluate finance if not already blocked
            fin_verdict = (
                SeverityLattice.VIABLE if roi_viable
                else SeverityLattice.PRECAUCION
            )
            # 2. Colapso del retículo (supremum)
            final_severity = LatticeVerdictCollapse.compute_supremum(
                fin_verdict, topo_verdict,
            )

        # 3. Generación de narrativa causal
        narrator = GraphRAGCausalNarrator(graph, max_cycles=max_cycles)
        causal_text = narrator.narrate_topological_collapse(betti_1, psi)

        synthesis = f"VEREDICTO FINAL SÍNTESIS: {final_severity.name}\n"
        if (
            final_severity == SeverityLattice.RECHAZAR
            and topo_verdict == SeverityLattice.RECHAZAR
        ):
            synthesis += (
                "JUSTIFICACIÓN: Se ha aplicado un cortocircuito absorbente (Fast-Fail). "
                "Debido a que la topología devuelve un estado de RECHAZAR (⊤), "
                "el sistema omite la evaluación financiera, ya que a ⊔ ⊤ = ⊤. "
                "El rigor topológico/físico domina incondicionalmente."
            )

        return f"{causal_text}\n\n{synthesis}"


# ============================================================================
# VEREDICTO FINANCIERO
# ============================================================================


class FinancialVerdict(Enum):
    """
    Veredictos financieros específicos.

    Mapea a VerdictLevel mediante un homomorfismo de enumeraciones.
    """

    ACCEPT = "ACEPTAR"
    CONDITIONAL = "CONDICIONAL"
    REVIEW = "REVISAR"
    REJECT = "RECHAZAR"

    def to_verdict_level(self) -> VerdictLevel:
        """Homomorfismo al retículo de veredictos."""
        mapping: Dict[FinancialVerdict, VerdictLevel] = {
            FinancialVerdict.ACCEPT: VerdictLevel.VIABLE,
            FinancialVerdict.CONDITIONAL: VerdictLevel.CONDICIONAL,
            FinancialVerdict.REVIEW: VerdictLevel.REVISAR,
            FinancialVerdict.REJECT: VerdictLevel.RECHAZAR,
        }
        return mapping[self]

    @classmethod
    def from_string(cls, value: str) -> FinancialVerdict:
        """Parsea desde string con normalización robusta."""
        if not value or not isinstance(value, str):
            return cls.REVIEW
        normalized = value.strip().upper()
        for verdict in cls:
            if str(verdict.value).upper() == normalized or verdict.name == normalized:
                return verdict
        # Fuzzy matching para tolerancia a errores
        _ALIASES: Dict[str, FinancialVerdict] = {
            "ACEPTAR": cls.ACCEPT,
            "ACCEPT": cls.ACCEPT,
            "OK": cls.ACCEPT,
            "CONDICIONAL": cls.CONDITIONAL,
            "CONDITIONAL": cls.CONDITIONAL,
            "REVISAR": cls.REVIEW,
            "REVIEW": cls.REVIEW,
            "RECHAZAR": cls.REJECT,
            "REJECT": cls.REJECT,
            "NO": cls.REJECT,
        }
        return _ALIASES.get(normalized, cls.REVIEW)


# ============================================================================
# ESTRUCTURAS DE DATOS VALIDADAS
# ============================================================================


@runtime_checkable
class HasBettiNumbers(Protocol):
    """Protocolo para objetos con números de Betti."""

    @property
    def beta_0(self) -> int: ...

    @property
    def beta_1(self) -> int: ...


@dataclass(frozen=True)
class ValidatedTopology:
    """
    Métricas topológicas validadas con invariante de Euler verificado.

    Invariante de clase (verificado en ``__post_init__``):
        euler_characteristic == beta_0 − beta_1 + beta_2

    Convención de genus:
        Para grafos (1-complejos simpliciales), genus := β₁.
        Para superficies orientables cerradas, genus_surface := β₁ / 2.
    """

    beta_0: int
    beta_1: int
    beta_2: int
    euler_characteristic: int
    fiedler_value: float
    spectral_gap: float
    pyramid_stability: float
    structural_entropy: float

    def __post_init__(self) -> None:
        # Validar no-negatividad de números de Betti
        for name, val in [
            ("beta_0", self.beta_0),
            ("beta_1", self.beta_1),
            ("beta_2", self.beta_2),
        ]:
            if val < 0:
                raise ValueError(f"Betti number {name} must be non-negative, got {val}")

        # Validar invariante de Euler
        expected_chi = self.beta_0 - self.beta_1 + self.beta_2
        if self.euler_characteristic != expected_chi:
            raise TopologyInvariantViolation(
                "Euler characteristic inconsistency",
                self.beta_0, self.beta_1, self.beta_2,
                self.euler_characteristic,
            )

        # Validar valor de Fiedler (Laplaciano → siempre ≥ 0)
        if math.isfinite(self.fiedler_value) and self.fiedler_value < -EPSILON:
            raise ValueError(
                f"Fiedler value (λ₂ of Laplacian) must be ≥ 0, got {self.fiedler_value}"
            )

        # Validar estabilidad piramidal
        if math.isfinite(self.pyramid_stability) and self.pyramid_stability < 0:
            raise ValueError(
                f"Pyramid stability Ψ must be non-negative, got {self.pyramid_stability}"
            )

        # Validar entropía estructural
        if math.isfinite(self.structural_entropy) and self.structural_entropy < 0:
            raise ValueError(
                f"Structural entropy must be non-negative, got {self.structural_entropy}"
            )

    @classmethod
    def from_metrics(
        cls,
        metrics: Union[TopologicalMetrics, Dict[str, Any]],
        strict: bool = False,
    ) -> ValidatedTopology:
        """
        Construye desde métricas con validación configurable.

        Si ``strict=False``, corrige la característica de Euler automáticamente
        y emite un warning estructurado.
        Si ``strict=True``, lanza ``TopologyInvariantViolation``.
        """
        if isinstance(metrics, dict):
            beta_0 = int(metrics.get("beta_0", 1))
            beta_1 = int(metrics.get("beta_1", 0))
            beta_2 = int(metrics.get("beta_2", 0))
            chi = int(metrics.get("euler_characteristic", beta_0 - beta_1 + beta_2))
            fiedler = float(metrics.get("fiedler_value", 1.0))
            gap = float(metrics.get("spectral_gap", 0.0))
            stability = float(metrics.get("pyramid_stability", 1.0))
            entropy = float(metrics.get("structural_entropy", 0.0))
        else:
            beta_0 = metrics.beta_0
            beta_1 = metrics.beta_1
            beta_2 = metrics.beta_2
            chi = metrics.euler_characteristic
            fiedler = metrics.fiedler_value
            gap = metrics.spectral_gap
            stability = metrics.pyramid_stability
            entropy = metrics.structural_entropy

        # Sanitizar NaN/Inf en valores flotantes
        fiedler = fiedler if math.isfinite(fiedler) else 0.0
        gap = gap if math.isfinite(gap) else 0.0
        stability = stability if math.isfinite(stability) else 0.0
        entropy = entropy if math.isfinite(entropy) else 0.0

        # Clamp Fiedler negativo por error numérico
        if fiedler < 0:
            logger.warning(
                f"Clamping negative Fiedler value {fiedler:.6e} → 0.0 "
                f"(probable numerical error in Laplacian eigendecomposition)"
            )
            fiedler = 0.0

        # Corregir Euler si no es estricto
        expected_chi = beta_0 - beta_1 + beta_2
        if chi != expected_chi:
            if strict:
                raise TopologyInvariantViolation(
                    "Strict Euler validation failed",
                    beta_0, beta_1, beta_2, chi,
                )
            logger.warning(
                f"Auto-correcting Euler characteristic: χ={chi} → "
                f"χ={expected_chi} (β₀={beta_0}, β₁={beta_1}, β₂={beta_2})"
            )
            chi = expected_chi

        return cls(
            beta_0=beta_0,
            beta_1=beta_1,
            beta_2=beta_2,
            euler_characteristic=chi,
            fiedler_value=fiedler,
            spectral_gap=gap,
            pyramid_stability=stability,
            structural_entropy=entropy,
        )

    @property
    def is_connected(self) -> bool:
        """¿El grafo es conexo?  (β₀ = 1)"""
        return self.beta_0 == 1

    @property
    def has_cycles(self) -> bool:
        """¿Hay ciclos independientes?  (β₁ > 0)"""
        return self.beta_1 > 0

    @property
    def genus(self) -> int:
        """
        Genus topológico para grafos (1-complejos simpliciales).

        Convención: genus := β₁  (número de ciclos independientes).
        Para superficies orientables cerradas, usar ``genus_surface``.
        """
        return self.beta_1

    @property
    def genus_surface(self) -> float:
        """
        Genus para superficies orientables cerradas: g = β₁ / 2.

        Nota: Solo es entero si β₁ es par (superficies orientables).
        """
        return self.beta_1 / 2.0

    @property
    def is_spectrally_connected(self) -> bool:
        """¿El grafo es espectralmente conexo?  (λ₂ > ε)"""
        return self.fiedler_value > EPSILON

    @property
    def normalized_spectral_gap(self) -> float:
        """
        Brecha espectral normalizada.

        Si spectral_gap es proporcionado directamente, se usa.
        De lo contrario, requiere λ_max para normalizar (no disponible aquí).
        """
        return self.spectral_gap


# ============================================================================
# RESULTADOS DE ANÁLISIS
# ============================================================================


@dataclass
class StratumAnalysisResult:
    """
    Resultado del análisis de un estrato DIKW.

    Jerarquía:
    - PHYSICS (Datos):      Señales crudas y métricas físicas
    - TACTICS (Información): Estructura topológica y dependencias
    - STRATEGY (Conocimiento): Viabilidad financiera y modelo de negocio
    - OMEGA (Eje de Decisión): Colapso del retículo
    - WISDOM (Sabiduría):   Veredicto integrado y recomendaciones
    """

    stratum: Stratum
    verdict: VerdictLevel
    narrative: str
    metrics_summary: Dict[str, Any]
    issues: List[str] = field(default_factory=list)
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "stratum": self.stratum.name,
            "verdict": self.verdict.name,
            "verdict_emoji": self.verdict.emoji,
            "narrative": self.narrative,
            "metrics": self.metrics_summary,
            "issues": list(self.issues),
            "confidence": self.confidence,
        }

    @property
    def severity_score(self) -> float:
        """Score de severidad normalizado [0, 1]."""
        return self.verdict.normalized_score


@dataclass
class StrategicReport:
    """
    Reporte estratégico completo.

    Representa el juicio final integrado del Consejo de Sabios.
    """

    title: str
    verdict: VerdictLevel
    executive_summary: str
    strata_analysis: Dict[Stratum, StratumAnalysisResult]
    recommendations: List[str]
    raw_narrative: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    @property
    def is_viable(self) -> bool:
        """¿El proyecto es viable? (veredicto positivo)"""
        return self.verdict.is_positive

    @property
    def requires_immediate_action(self) -> bool:
        """¿Se requiere acción inmediata?"""
        return self.verdict.requires_attention

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario para API/persistencia."""
        return {
            "title": self.title,
            "verdict": self.verdict.name,
            "verdict_emoji": self.verdict.emoji,
            "verdict_description": self.verdict.description,
            "is_viable": self.is_viable,
            "requires_action": self.requires_immediate_action,
            "executive_summary": self.executive_summary,
            "strata_analysis": {
                s.name: a.to_dict()
                for s, a in self.strata_analysis.items()
            },
            "recommendations": list(self.recommendations),
            "timestamp": self.timestamp,
            "confidence": self.confidence,
        }


# ============================================================================
# CACHÉ DE NARRATIVAS — THREAD-SAFE CON LRU
# ============================================================================


class NarrativeCache:
    """
    Caché thread-safe LRU para narrativas generadas.

    Usa ``OrderedDict`` para LRU eficiente y ``threading.Lock``
    para thread-safety.

    Claves generadas con hash determinista (SHA-256 sobre JSON canónico).
    """

    def __init__(self, maxsize: int = 256) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize must be ≥ 1, got {maxsize}")
        self._maxsize = maxsize
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(
        domain: str, classification: str, params: Dict[str, Any],
    ) -> str:
        """
        Genera clave determinista usando SHA-256 sobre representación canónica.
        Ahora incluye el tensor de invariantes topológicos y la temperatura, garantizando
        isomorfismo semántico absoluto.

        A diferencia de ``hash()``, SHA-256 es determinista entre procesos.
        """
        # Extraer parámetros relevantes o establecer sus defaults si no existen,
        # para formar el tensor topológico-térmico (β0, β1, λ2, Ψ, T)
        beta_0 = params.get("beta_0", 1)
        beta_1 = params.get("beta_1", 0)
        fiedler = params.get("fiedler", 1.0)
        psi = params.get("stability", 1.0)
        temperature = params.get("temperature", 298.15)

        tensor_invariants = {
            "beta_0": beta_0,
            "beta_1": beta_1,
            "fiedler_value": fiedler,
            "pyramid_stability": psi,
            "temperature": temperature,
        }

        canonical = json.dumps(
            {
                "d": domain,
                "c": classification,
                "p": params,
                "tensor": tensor_invariants
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(
        self, domain: str, classification: str, params: Dict[str, Any],
    ) -> Optional[str]:
        """Obtiene narrativa del caché (None si no existe)."""
        key = self._make_key(domain, classification, params)
        with self._lock:
            if key in self._cache:
                # Mover al final (LRU: most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(
        self,
        domain: str,
        classification: str,
        params: Dict[str, Any],
        narrative: str,
    ) -> None:
        """Almacena narrativa en caché con evicción LRU."""
        key = self._make_key(domain, classification, params)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = narrative
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)  # Evicta el más antiguo
                self._cache[key] = narrative

    def clear(self) -> None:
        """Limpia el caché completamente."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> Dict[str, int]:
        """Estadísticas del caché."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_pct": round(100 * self._hits / total, 1) if total > 0 else 0.0,
            }


# ============================================================================
# TRADUCTOR SEMÁNTICO PRINCIPAL
# ============================================================================


class SemanticTranslator:
    """
    Traductor semántico que convierte métricas técnicas en narrativa de ingeniería.

    Formalmente, implementa un Funtor de retículos estricto:

        T: L_Severity → L_Verdict

    que preserva el supremo:  T(⊤) = ⊤  (CRITICO → RECHAZAR).

    Interpreta el presupuesto como estructura física:
    - Insumos   = Cimentación de Recursos  (Nivel PHYSICS)
    - APUs      = Cuerpo Táctico           (Nivel TACTICS)
    - Capítulos = Pilares Estructurales    (Nivel STRATEGY)
    - Proyecto  = Ápice / Objetivo Final   (Nivel WISDOM)
    """

    def __init__(
        self,
        config: Optional[TranslatorConfig] = None,
        market_provider: Optional[Callable[[], str]] = None,
        mic: Optional[MICRegistry] = None,
        enable_cache: bool = True,
    ) -> None:
        """
        Args:
            config: Configuración consolidada de umbrales
            market_provider: Proveedor de contexto de mercado (inyección de dependencia)
            mic: Registro MIC para obtener narrativas
            enable_cache: Habilitar caché LRU thread-safe
        """
        self.config = config or TranslatorConfig()
        self._market_provider = market_provider
        self._cache: Optional[NarrativeCache] = (
            NarrativeCache() if enable_cache else None
        )

        if mic:
            self.mic = mic
        else:
            self.mic = MICRegistry()
            register_core_vectors(self.mic, config={})

        logger.debug(
            f"SemanticTranslator initialized | "
            f"Ψ_critical={self.config.stability.critical:.2f}, "
            f"deterministic={self.config.deterministic_market}, "
            f"cache={enable_cache}"
        )

    # ========================================================================
    # HELPERS INTERNOS
    # ========================================================================

    def _fetch_narrative(
        self,
        domain: str,
        classification: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Obtiene narrativa del MIC con caché opcional.

        Args:
            domain: Dominio (TOPOLOGY_CYCLES, THERMAL_TEMPERATURE, etc.)
            classification: Clasificación específica (clean, critical, etc.)
            params: Parámetros para interpolación

        Returns:
            Narrativa formateada
        """
        params = params or {}

        # Intentar caché
        if self._cache is not None:
            cached = self._cache.get(domain, classification, params)
            if cached is not None:
                return cached

        # Obtener del MIC
        response = self.mic.project_intent(
            "fetch_narrative",
            {
                "domain": domain,
                "classification": classification,
                "params": params,
            },
            {"force_physics_override": True},
        )

        narrative = response.get("narrative", f"[{domain}.{classification}]")

        # Almacenar en caché
        if self._cache is not None:
            self._cache.put(domain, classification, params, narrative)

        return narrative

    def _normalize_temperature(
        self,
        value: float,
        assume_kelvin_if_high: bool = True,
    ) -> Temperature:
        """
        Normaliza un valor de temperatura a objeto ``Temperature``.

        Heurística: Si value > 100, se asume Kelvin (agua hierve a 100 °C).
        """
        if not math.isfinite(value):
            logger.warning(f"Non-finite temperature value {value}, defaulting to 298.15 K")
            return Temperature.from_kelvin(298.15)

        if assume_kelvin_if_high and value > 100:
            return Temperature.from_kelvin(value)
        return Temperature.from_celsius(value)

    @staticmethod
    def _safe_extract_numeric(
        data: Dict[str, Any], key: str, default: float = 0.0,
    ) -> float:
        """Extrae valor numérico de forma segura, con sanitización de NaN/Inf."""
        value = data.get(key)
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value) if math.isfinite(value) else default
        return default

    @staticmethod
    def _safe_extract_nested(
        data: Dict[str, Any], path: List[str], default: float = 0.0,
    ) -> float:
        """Extrae valor numérico de path anidado con sanitización."""
        current: Any = data
        for key in path:
            if not isinstance(current, dict):
                return default
            current = current.get(key)
            if current is None:
                return default
        if isinstance(current, (int, float)):
            return float(current) if math.isfinite(current) else default
        return default

    # ========================================================================
    # GRAPHRAG: EXPLICACIÓN CAUSAL
    # ========================================================================

    def explain_cycle_path(self, cycle_nodes: List[str]) -> str:
        """
        Genera narrativa explicando la ruta del ciclo (GraphRAG).

        El ciclo se presenta con cierre explícito:  A → B → ⋯ → A
        """
        if not cycle_nodes:
            return ""

        max_display = self.config.max_cycle_path_display

        display_nodes = list(cycle_nodes[:max_display])

        if len(cycle_nodes) > max_display:
            remaining = len(cycle_nodes) - max_display
            display_nodes.append(f"... (+{remaining} nodos)")

        # Cerrar el ciclo visualmente
        if (
            cycle_nodes
            and display_nodes
            and not display_nodes[-1].startswith("...")
            and display_nodes[-1] != cycle_nodes[0]
        ):
            display_nodes.append(cycle_nodes[0])

        payload = {
            "anomaly_type": "CYCLE",
            "path_nodes": display_nodes,
            "total_length": len(cycle_nodes),
        }

        response = self.mic.project_intent(
            "project_graph_narrative",
            payload,
            {"force_physics_override": True},
        )

        return response.get("narrative", "") if response.get("success") else ""

    def explain_stress_point(
        self,
        node_id: str,
        degree: Union[int, str],
        stratum: Stratum = Stratum.PHYSICS,
    ) -> str:
        """
        Explica por qué un nodo es punto de estrés crítico (GraphRAG).

        Args:
            node_id: Identificador del nodo
            degree: Grado del nodo (numérico o descriptivo)
            stratum: Estrato del nodo
        """
        if isinstance(degree, int):
            safe_degree = degree
        elif isinstance(degree, str) and degree.isdigit():
            safe_degree = int(degree)
        else:
            safe_degree = 10  # Descriptor cualitativo → valor alto por defecto

        payload = {
            "anomaly_type": "STRESS",
            "vector": {
                "node_id": str(node_id),
                "node_type": "INSUMO",
                "stratum": stratum.value,
                "in_degree": safe_degree,
                "out_degree": 0,
            },
        }

        response = self.mic.project_intent(
            "project_graph_narrative",
            payload,
            {"force_physics_override": True},
        )

        return response.get("narrative", "") if response.get("success") else ""

    # ========================================================================
    # TRADUCCIÓN DE TOPOLOGÍA
    # ========================================================================

    def translate_topology(
        self,
        metrics: Union[TopologicalMetrics, Dict[str, Any], ValidatedTopology],
        stability: float = 0.0,
        synergy_risk: Optional[Dict[str, Any]] = None,
        spectral: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, VerdictLevel]:
        """
        Traduce métricas topológicas a narrativa de ingeniería.

        Análisis secuencial:
        1. β₁ (ciclos): Genus estructural / "socavones"
        2. Sinergia de riesgo: Ciclos que comparten recursos críticos
        3. Espectral: Conectividad algebraica (Fiedler)
        4. β₀ (componentes): Coherencia de obra
        5. Ψ (estabilidad): Solidez de cimentación

        Returns:
            Tupla (narrativa, veredicto)
        """
        # Normalizar a ValidatedTopology
        if isinstance(metrics, ValidatedTopology):
            topo = metrics
        else:
            topo = ValidatedTopology.from_metrics(
                metrics, strict=self.config.strict_euler_validation,
            )

        synergy = synergy_risk or {}
        spec = spectral or {}

        eff_stability = stability if stability != 0.0 else topo.pyramid_stability

        if not math.isfinite(eff_stability) or eff_stability < 0:
            raise MetricsValidationError(
                f"Stability Ψ must be a non-negative finite number, got {eff_stability}"
            )

        narrative_parts: List[str] = []
        verdicts: List[VerdictLevel] = []

        # 1. β₁: Ciclos / Genus
        cycle_narrative, cycle_verdict = self._translate_cycles(topo.beta_1)
        narrative_parts.append(cycle_narrative)
        verdicts.append(cycle_verdict)

        # 2. Sinergia de Riesgo
        synergy_detected = bool(synergy.get("synergy_detected", False))
        if synergy_detected:
            synergy_narrative = self._translate_synergy(synergy)
            narrative_parts.append(synergy_narrative)
            verdicts.append(VerdictLevel.RECHAZAR)

            bridge_nodes = synergy.get("bridge_nodes", [])
            if bridge_nodes:
                first_bridge = bridge_nodes[0]
                node_id = (
                    first_bridge.get("id")
                    if isinstance(first_bridge, dict)
                    else str(first_bridge)
                )
                explanation = self.explain_stress_point(node_id, "múltiples")
                if explanation:
                    narrative_parts.append(explanation)

        # 3. Análisis Espectral
        fiedler = topo.fiedler_value
        resonance_risk = bool(spec.get("resonance_risk", False))
        wavelength = float(spec.get("wavelength", 0.0))

        if fiedler > 0 or resonance_risk:
            spec_narrative = self._translate_spectral(
                fiedler, wavelength, resonance_risk,
            )
            narrative_parts.append(spec_narrative)
            if resonance_risk:
                verdicts.append(VerdictLevel.PRECAUCION)

        # 4. β₀: Coherencia / Conectividad
        conn_narrative, conn_verdict = self._translate_connectivity(topo.beta_0)
        narrative_parts.append(conn_narrative)
        verdicts.append(conn_verdict)

        # 5. Ψ: Solidez de Cimentación
        stab_narrative, stab_verdict = self._translate_stability(eff_stability)
        narrative_parts.append(stab_narrative)
        verdicts.append(stab_verdict)

        # Veredicto final: supremum (peor caso)
        final_verdict = VerdictLevel.supremum(*verdicts)

        return "\n".join(narrative_parts), final_verdict

    def _translate_cycles(self, beta_1: int) -> Tuple[str, VerdictLevel]:
        """Traduce β₁ (ciclos) a narrativa y veredicto."""
        classification = self.config.topology.classify_cycles(beta_1)

        if classification == "invalid":
            return "⚠️ **Valor de ciclos inválido**", VerdictLevel.REVISAR

        narrative = self._fetch_narrative(
            "TOPOLOGY_CYCLES", classification, {"beta_1": beta_1},
        )

        verdict_map: Dict[str, VerdictLevel] = {
            "clean": VerdictLevel.VIABLE,
            "minor": VerdictLevel.CONDICIONAL,
            "moderate": VerdictLevel.PRECAUCION,
            "critical": VerdictLevel.RECHAZAR,
        }

        return narrative, verdict_map.get(classification, VerdictLevel.REVISAR)

    def _translate_connectivity(self, beta_0: int) -> Tuple[str, VerdictLevel]:
        """Traduce β₀ (componentes conexos) a narrativa y veredicto."""
        classification = self.config.topology.classify_connectivity(beta_0)
        narrative = self._fetch_narrative(
            "TOPOLOGY_CONNECTIVITY", classification, {"beta_0": beta_0},
        )

        verdict_map: Dict[str, VerdictLevel] = {
            "empty": VerdictLevel.RECHAZAR,
            "unified": VerdictLevel.VIABLE,
            "fragmented": VerdictLevel.CONDICIONAL,
            "severely_fragmented": VerdictLevel.PRECAUCION,
        }

        return narrative, verdict_map.get(classification, VerdictLevel.REVISAR)

    def _translate_stability(self, stability: float) -> Tuple[str, VerdictLevel]:
        """Traduce Ψ (índice de estabilidad piramidal) a narrativa y veredicto."""
        classification = self.config.stability.classify(stability)

        if classification == "invalid":
            return "⚠️ **Valor de estabilidad inválido**", VerdictLevel.REVISAR

        narrative = self._fetch_narrative(
            "STABILITY", classification, {"stability": round(stability, 2)},
        )

        verdict_map: Dict[str, VerdictLevel] = {
            "critical": VerdictLevel.RECHAZAR,
            "warning": VerdictLevel.PRECAUCION,
            "stable": VerdictLevel.CONDICIONAL,
            "robust": VerdictLevel.VIABLE,
        }

        return narrative, verdict_map.get(classification, VerdictLevel.REVISAR)

    def _translate_synergy(self, synergy: Dict[str, Any]) -> str:
        """Traduce sinergia de riesgo a narrativa."""
        count = int(synergy.get("intersecting_cycles_count", 0))
        return self._fetch_narrative("MISC", "SYNERGY", {"count": count})

    def _translate_spectral(
        self,
        fiedler: float,
        wavelength: float,
        resonance_risk: bool,
    ) -> str:
        """Traduce métricas espectrales a narrativa."""
        parts: List[str] = []

        cohesion_class = self.config.topology.classify_spectral_connectivity(fiedler)
        cohesion_map: Dict[str, str] = {
            "strongly_connected": "high",
            "weakly_connected": "standard",
            "disconnected": "low",
            "invalid": "low",
        }
        cohesion_type = cohesion_map.get(cohesion_class, "standard")

        parts.append(
            self._fetch_narrative(
                "SPECTRAL_COHESION", cohesion_type, {"fiedler": round(fiedler, 4)},
            )
        )

        resonance_type = "risk" if resonance_risk else "safe"
        parts.append(
            self._fetch_narrative(
                "SPECTRAL_RESONANCE",
                resonance_type,
                {"wavelength": round(wavelength, 4)},
            )
        )

        return " ".join(parts)

    # ========================================================================
    # TRADUCCIÓN DE TERMODINÁMICA
    # ========================================================================

    def translate_thermodynamics(
        self,
        metrics: Union[ThermodynamicMetrics, Dict[str, Any]],
        fiedler_value: float = 1.0,
    ) -> Tuple[str, VerdictLevel]:
        """
        Traduce métricas termodinámicas a narrativa.

        Metáforas:
        - Temperatura    →  Volatilidad / Inflación
        - Entropía       →  Desorden administrativo
        - Exergía        →  Eficiencia de inversión
        - C_v            →  Inercia financiera
        """
        if isinstance(metrics, dict):
            temp_raw = float(metrics.get(
                "system_temperature",
                metrics.get("temperature", 298.15),
            ))
            thermo = ThermodynamicMetrics(
                system_temperature=temp_raw,
                entropy=float(metrics.get("entropy", 0.0)),
                heat_capacity=float(metrics.get("heat_capacity", 0.5)),
            )
        elif isinstance(metrics, ThermodynamicMetrics):
            thermo = metrics
        else:
            thermo = ThermodynamicMetrics()

        base_temp = self._normalize_temperature(thermo.system_temperature)

        # Difusión Térmica Laplaciana: Teff = Tbase * exp(-λ2 * t)
        # Assuming t = 1.0 for the timestep of evaluation
        t = 1.0
        effective_temp_kelvin = base_temp.kelvin * math.exp(-fiedler_value * t)

        # Determine if we have a thermal bottleneck due to low connectivity
        is_thermal_bottleneck = False
        if fiedler_value < self.config.topology.fiedler_connected_threshold:
            # Heat cannot dissipate, local thermal singularities occur
            is_thermal_bottleneck = True
            # We override the effective temperature to be extremely high
            # effectively ignoring the diffusion cooling effect, preserving the high input temperature
            effective_temp_kelvin = base_temp.kelvin

        temp = Temperature.from_kelvin(effective_temp_kelvin)
        temp_celsius = temp.celsius

        # Clamp entropía y exergía a [0, 1]
        entropy = max(0.0, min(1.0, thermo.entropy)) if math.isfinite(thermo.entropy) else 0.0
        exergy = (
            max(0.0, min(1.0, thermo.exergetic_efficiency))
            if math.isfinite(thermo.exergetic_efficiency)
            else 0.0
        )

        parts: List[str] = []
        verdicts: List[VerdictLevel] = []

        # 1. Eficiencia Exergética
        exergy_pct = exergy * 100.0
        exergy_class = self.config.thermal.classify_exergy(exergy)
        parts.append(f"⚡ **Eficiencia Exergética del {exergy_pct:.1f}%**.")

        exergy_verdict_map: Dict[str, VerdictLevel] = {
            "efficient": VerdictLevel.VIABLE,
            "moderate": VerdictLevel.CONDICIONAL,
            "poor": VerdictLevel.PRECAUCION,
            "invalid": VerdictLevel.REVISAR,
        }
        verdicts.append(exergy_verdict_map.get(exergy_class, VerdictLevel.CONDICIONAL))

        # 2. Entropía
        entropy_class = self.config.thermal.classify_entropy(entropy)
        if entropy_class == "death":
            parts.append(self._fetch_narrative("MISC", "THERMAL_DEATH"))
            verdicts.append(VerdictLevel.RECHAZAR)
        elif entropy_class == "high":
            parts.append(
                self._fetch_narrative(
                    "THERMAL_ENTROPY", "high", {"entropy": round(entropy, 2)},
                )
            )
            verdicts.append(VerdictLevel.PRECAUCION)
        elif entropy_class == "low":
            parts.append(
                self._fetch_narrative(
                    "THERMAL_ENTROPY", "low", {"entropy": round(entropy, 2)},
                )
            )
            # Baja entropía no contribuye veredicto negativo

        # 3. Temperatura
        temp_class = self.config.thermal.classify_temperature(temp_celsius)

        if is_thermal_bottleneck:
            parts.append(
                f"🔥 **Embotellamiento Térmico Inminente**: La conectividad algebraica es críticamente "
                f"baja (λ₂ = {fiedler_value:.2e}). El calor no puede disiparse por la red, "
                f"creando singularidades térmicas locales."
            )
            verdicts.append(VerdictLevel.RECHAZAR)
        else:
            parts.append(
                self._fetch_narrative(
                    "THERMAL_TEMPERATURE", temp_class,
                    {"temperature": round(temp_celsius, 1)},
                )
            )

            temp_verdict_map: Dict[str, VerdictLevel] = {
                "cold": VerdictLevel.VIABLE,
                "stable": VerdictLevel.VIABLE,
                "warm": VerdictLevel.CONDICIONAL,
                "hot": VerdictLevel.PRECAUCION,
                "critical": VerdictLevel.RECHAZAR,
                "invalid": VerdictLevel.REVISAR,
            }
            verdicts.append(temp_verdict_map.get(temp_class, VerdictLevel.REVISAR))

            if temp_class in ("hot", "critical"):
                parts.append(
                    "💊 **Receta**: Se recomienda enfriar mediante contratos de futuros "
                    "o stock preventivo."
                )

        # 4. Inercia (Capacidad Calorífica)
        heat_cap = thermo.heat_capacity if math.isfinite(thermo.heat_capacity) else 0.0
        if heat_cap < self.config.thermal.heat_capacity_minimum:
            parts.append(
                f"🍂 **Hoja al Viento**: Baja inercia financiera "
                f"(C_v={heat_cap:.2f}). Riesgo de volatilidad extrema."
            )
            verdicts.append(VerdictLevel.PRECAUCION)

        final_verdict = VerdictLevel.supremum(*verdicts)
        return " ".join(parts), final_verdict

    # ========================================================================
    # TRADUCCIÓN FINANCIERA
    # ========================================================================

    def translate_financial(
        self,
        metrics: Dict[str, Any],
    ) -> Tuple[str, VerdictLevel, FinancialVerdict]:
        """
        Traduce métricas financieras a narrativa.

        Returns:
            Tupla (narrativa, veredicto_general, veredicto_financiero)
        """
        validated = self._validate_financial_metrics(metrics)
        parts: List[str] = []

        parts.append(
            self._fetch_narrative("MISC", "WACC", {"wacc": validated["wacc"]})
        )

        parts.append(
            self._fetch_narrative(
                "MISC", "CONTINGENCY",
                {"contingency": validated["contingency_recommended"]},
            )
        )

        fin_verdict: FinancialVerdict = validated["recommendation"]
        pi: float = validated["profitability_index"]

        verdict_key = fin_verdict.name.lower()
        parts.append(
            self._fetch_narrative(
                "FINANCIAL_VERDICT", verdict_key, {"pi": round(pi, 3)},
            )
        )

        general_verdict = fin_verdict.to_verdict_level()
        return "\n".join(parts), general_verdict, fin_verdict

    def _validate_financial_metrics(
        self, metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Valida y normaliza métricas financieras."""
        if not isinstance(metrics, dict):
            raise MetricsValidationError(
                f"Expected dict for financial metrics, got {type(metrics).__name__}"
            )

        def extract_verdict(data: Dict[str, Any]) -> FinancialVerdict:
            performance = data.get("performance", {})
            if not isinstance(performance, dict):
                return FinancialVerdict.REVIEW
            rec = performance.get("recommendation", "REVISAR")
            return FinancialVerdict.from_string(str(rec))

        return {
            "wacc": self._safe_extract_numeric(metrics, "wacc", 0.0),
            "contingency_recommended": self._safe_extract_nested(
                metrics, ["contingency", "recommended"], 0.0,
            ),
            "recommendation": extract_verdict(metrics),
            "profitability_index": self._safe_extract_nested(
                metrics, ["performance", "profitability_index"], 0.0,
            ),
        }

    # ========================================================================
    # CONTEXTO DE MERCADO
    # ========================================================================

    def get_market_context(self) -> str:
        """Obtiene inteligencia de mercado desde el proveedor inyectado o MIC."""
        if self._market_provider:
            try:
                context = self._market_provider()
                return f"🌍 **Suelo de Mercado**: {context}"
            except Exception as e:
                logger.warning(f"Market provider failed: {e}")
                return "🌍 **Suelo de Mercado**: No disponible."

        params = {
            "deterministic": self.config.deterministic_market,
            "index": self.config.default_market_index,
        }
        response = self.mic.project_intent(
            "fetch_narrative",
            {"domain": "MARKET_CONTEXT", "params": params},
            {},
        )
        context = response.get("narrative", "Datos de mercado no disponibles.")
        return f"🌍 **Suelo de Mercado**: {context}"

    # ========================================================================
    # ANÁLISIS POR ESTRATO DIKW
    # ========================================================================

    def _analyze_physics_stratum(
        self,
        thermal: ThermodynamicMetrics,
        stability: float,
        physics: Optional[PhysicsMetrics] = None,
        control: Optional[ControlMetrics] = None,
        fiedler_value: float = 1.0,
    ) -> StratumAnalysisResult:
        """
        Analiza el estrato PHYSICS (datos, flujo, temperatura).

        Sub-análisis:
        1. Giroscopía (FluxCondenser)
        2. Oráculo de Laplace (Control: polos, margen de fase)
        3. Análisis térmico
        4. Entropía (muerte térmica)
        5. Dinámica de bombeo
        """
        issues: List[str] = []
        verdicts: List[VerdictLevel] = []
        narrative_parts: List[str] = []

        # 1. Giroscopía
        if physics is not None:
            gyro = physics.gyroscopic_stability
            if not math.isfinite(gyro):
                issues.append("Estabilidad giroscópica no finita")
                verdicts.append(VerdictLevel.REVISAR)
            elif gyro < 0.3:
                issues.append("Nutación crítica detectada")
                narrative_parts.append(
                    self._fetch_narrative("GYROSCOPIC_STABILITY", "nutation")
                )
                verdicts.append(VerdictLevel.RECHAZAR)
            elif gyro < 0.7:
                issues.append("Precesión detectada")
                narrative_parts.append(
                    self._fetch_narrative("GYROSCOPIC_STABILITY", "precession")
                )
                verdicts.append(VerdictLevel.PRECAUCION)
            else:
                narrative_parts.append(
                    self._fetch_narrative("GYROSCOPIC_STABILITY", "stable")
                )
                verdicts.append(VerdictLevel.VIABLE)

        # 2. Oráculo de Laplace (Control)
        if control is not None:
            if not control.is_stable:
                issues.append("Divergencia Matemática (polos en RHP)")
                narrative_parts.append(
                    self._fetch_narrative("LAPLACE_CONTROL", "unstable")
                )
                verdicts.append(VerdictLevel.RECHAZAR)
            elif control.phase_margin_deg < 30:
                issues.append(
                    f"Estabilidad Marginal (PM={control.phase_margin_deg:.1f}°)"
                )
                narrative_parts.append(
                    self._fetch_narrative("LAPLACE_CONTROL", "marginal")
                )
                verdicts.append(VerdictLevel.PRECAUCION)
            else:
                verdicts.append(VerdictLevel.VIABLE)

        # 3. Análisis térmico
        base_temp = self._normalize_temperature(thermal.system_temperature)

        t = 1.0
        effective_temp_kelvin = base_temp.kelvin * math.exp(-fiedler_value * t)

        is_thermal_bottleneck = fiedler_value < self.config.topology.fiedler_connected_threshold
        if is_thermal_bottleneck:
            effective_temp_kelvin = base_temp.kelvin

        temp = Temperature.from_kelvin(effective_temp_kelvin)
        temp_class = self.config.thermal.classify_temperature(temp.celsius)

        if is_thermal_bottleneck:
            issues.append(f"Embotellamiento Térmico Inminente (λ₂={fiedler_value:.2e})")
            verdicts.append(VerdictLevel.RECHAZAR)
        elif temp_class == "critical":
            issues.append(f"Temperatura crítica: {temp}")
            verdicts.append(VerdictLevel.RECHAZAR)
        elif temp_class == "hot":
            issues.append(f"Temperatura elevada: {temp}")
            verdicts.append(VerdictLevel.PRECAUCION)
        else:
            verdicts.append(VerdictLevel.VIABLE)

        # 4. Entropía
        entropy_val = thermal.entropy if math.isfinite(thermal.entropy) else 0.0
        entropy_class = self.config.thermal.classify_entropy(entropy_val)
        if entropy_class == "death":
            issues.append("Muerte Térmica del Sistema")
            narrative_parts.append(self._fetch_narrative("MISC", "THERMAL_DEATH"))
            verdicts.append(VerdictLevel.RECHAZAR)
        elif entropy_class == "high":
            issues.append(f"Alta entropía: {entropy_val:.2f}")
            verdicts.append(VerdictLevel.PRECAUCION)

        # 5. Dinámica de Bombeo
        if physics is not None:
            pressure = physics.pressure if math.isfinite(physics.pressure) else 0.0
            saturation = physics.saturation if math.isfinite(physics.saturation) else 0.0

            if pressure > 0.7:
                issues.append(
                    f"Inestabilidad de Tubería (P={pressure:.2f})"
                )
                narrative_parts.append(
                    self._fetch_narrative(
                        "PUMP_DYNAMICS", "water_hammer",
                        {"pressure": pressure},
                    )
                )
                verdicts.append(VerdictLevel.PRECAUCION)

            narrative_parts.append(
                self._fetch_narrative(
                    "PUMP_DYNAMICS", "accumulator_pressure",
                    {"pressure": saturation * 100.0},
                )
            )

        verdict = VerdictLevel.supremum(*verdicts) if verdicts else VerdictLevel.VIABLE

        base_narratives: Dict[VerdictLevel, str] = {
            VerdictLevel.VIABLE: "Base física estable. Flujo de datos sin turbulencia.",
            VerdictLevel.RECHAZAR: "Inestabilidad física crítica. Datos no confiables.",
        }
        base_narrative = base_narratives.get(
            verdict, "Señales de inestabilidad en la capa física."
        )

        full_narrative = f"{base_narrative} {' '.join(narrative_parts)}".strip()

        metrics_summary: Dict[str, Any] = {
            "temperature_kelvin": thermal.system_temperature,
            "temperature_celsius": temp.celsius,
            "entropy": entropy_val,
            "exergy": thermal.exergy if math.isfinite(thermal.exergy) else 0.0,
            "stability": stability,
        }
        if physics is not None:
            metrics_summary["gyroscopic_stability"] = physics.gyroscopic_stability
            metrics_summary["pressure"] = physics.pressure
        if control is not None:
            metrics_summary["phase_margin_deg"] = control.phase_margin_deg
            metrics_summary["is_stable"] = control.is_stable

        return StratumAnalysisResult(
            stratum=Stratum.PHYSICS,
            verdict=verdict,
            narrative=full_narrative,
            metrics_summary=metrics_summary,
            issues=issues,
        )

    def _analyze_tactics_stratum(
        self,
        topo: ValidatedTopology,
        synergy: Dict[str, Any],
        spectral: Dict[str, Any],
        stability: float,
    ) -> StratumAnalysisResult:
        """Analiza el estrato TACTICS (estructura topológica)."""
        issues: List[str] = []
        verdicts: List[VerdictLevel] = []

        # 1. Ciclos (β₁)
        cycle_class = self.config.topology.classify_cycles(topo.beta_1)
        if cycle_class != "clean":
            issues.append(f"Socavón lógico detectado (β₁={topo.beta_1})")
            verdict_map: Dict[str, VerdictLevel] = {
                "critical": VerdictLevel.RECHAZAR,
                "moderate": VerdictLevel.PRECAUCION,
                "minor": VerdictLevel.CONDICIONAL,
            }
            verdicts.append(verdict_map.get(cycle_class, VerdictLevel.CONDICIONAL))
        else:
            verdicts.append(VerdictLevel.VIABLE)

        # 2. Conectividad (β₀ y Fiedler)
        #    Fiedler λ₂ < ε  ⟺  grafo desconectado (Teorema de Fiedler)
        fiedler = topo.fiedler_value
        fiedler_class = self.config.topology.classify_spectral_connectivity(fiedler)

        if fiedler_class == "disconnected" or topo.beta_0 > 1:
            issues.append(
                f"Silos organizacionales: grafo desconectado "
                f"(β₀={topo.beta_0}, λ₂={fiedler:.2e})"
            )
            verdicts.append(VerdictLevel.RECHAZAR)
        elif fiedler_class == "weakly_connected":
            issues.append(
                f"Conectividad algebraica débil (λ₂={fiedler:.4f})"
            )
            verdicts.append(VerdictLevel.PRECAUCION)
        # strongly_connected → no issue

        # 3. Brecha espectral (si disponible y positiva)
        spectral_gap = topo.spectral_gap
        if spectral_gap > 0 and spectral_gap < 0.1:
            issues.append(
                f"Brecha espectral baja (gap={spectral_gap:.4f}): "
                f"la estructura es vulnerable a perturbaciones"
            )
            verdicts.append(VerdictLevel.PRECAUCION)

        # 4. Sinergia de riesgo
        synergy_detected = bool(synergy.get("synergy_detected", False))
        if synergy_detected:
            issues.append("Sinergia de riesgo (Efecto Dominó)")
            verdicts.append(VerdictLevel.RECHAZAR)

        # 5. Estabilidad piramidal (Ψ)
        stab_class = self.config.stability.classify(stability)
        if stab_class == "critical":
            issues.append(f"Pirámide Invertida (Ψ={stability:.2f})")
            verdicts.append(VerdictLevel.RECHAZAR)
        elif stab_class == "warning":
            issues.append(f"Estabilidad marginal (Ψ={stability:.2f})")
            verdicts.append(VerdictLevel.PRECAUCION)

        verdict = VerdictLevel.supremum(*verdicts) if verdicts else VerdictLevel.VIABLE

        # Narrativa contextual
        if verdict == VerdictLevel.VIABLE:
            narrative = "Estructura topológicamente sólida y conexa."
        elif verdict == VerdictLevel.RECHAZAR:
            issue_summary = "; ".join(issues)
            if any("Silos" in i or "desconectado" in i for i in issues):
                narrative = (
                    f"Silos organizacionales detectados. La estructura está "
                    f"gravemente fragmentada. Detalles: {issue_summary}"
                )
            elif any("Socavón" in i for i in issues):
                narrative = (
                    f"Socavón lógico detectado. La estructura contiene ciclos "
                    f"anómalos. Detalles: {issue_summary}"
                )
            else:
                narrative = (
                    f"Estructura comprometida. Reparaciones necesarias. "
                    f"Detalles: {issue_summary}"
                )
        else:
            narrative = f"Estructura con defectos menores a corregir: {'; '.join(issues)}"

        return StratumAnalysisResult(
            stratum=Stratum.TACTICS,
            verdict=verdict,
            narrative=narrative,
            metrics_summary={
                "beta_0": topo.beta_0,
                "beta_1": topo.beta_1,
                "beta_2": topo.beta_2,
                "euler_characteristic": topo.euler_characteristic,
                "fiedler_value": topo.fiedler_value,
                "spectral_gap": topo.spectral_gap,
                "stability": stability,
                "synergy_risk": synergy_detected,
            },
            issues=issues,
        )

    def _analyze_strategy_stratum(
        self,
        financial_metrics: Dict[str, Any],
    ) -> StratumAnalysisResult:
        """Analiza el estrato STRATEGY (viabilidad financiera)."""
        try:
            validated = self._validate_financial_metrics(financial_metrics)
            fin_verdict = validated["recommendation"]
            verdict = fin_verdict.to_verdict_level()

            issues: List[str] = []
            if verdict.is_negative:
                issues.append("Proyecto financieramente inviable")
            elif verdict == VerdictLevel.REVISAR:
                issues.append("Métricas financieras inconclusas")

            if verdict.is_positive:
                narrative = "Modelo financiero viable y robusto."
            elif verdict == VerdictLevel.REVISAR:
                narrative = (
                    "Viabilidad financiera indeterminada. "
                    "Requiere análisis adicional."
                )
            else:
                narrative = (
                    "Proyecto no es financieramente viable en condiciones actuales."
                )

            return StratumAnalysisResult(
                stratum=Stratum.STRATEGY,
                verdict=verdict,
                narrative=narrative,
                metrics_summary={
                    "wacc": validated["wacc"],
                    "profitability_index": validated["profitability_index"],
                    "recommendation": fin_verdict.value,
                },
                issues=issues,
            )

        except MetricsValidationError as e:
            logger.error(f"Financial metrics validation failed: {e}")
            return StratumAnalysisResult(
                stratum=Stratum.STRATEGY,
                verdict=VerdictLevel.REVISAR,
                narrative=f"Error en validación financiera: {e}",
                metrics_summary={},
                issues=[str(e)],
                confidence=0.5,
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Error analyzing strategy stratum: {e}", exc_info=True)
            return StratumAnalysisResult(
                stratum=Stratum.STRATEGY,
                verdict=VerdictLevel.REVISAR,
                narrative=f"Error en análisis financiero: {e}",
                metrics_summary={},
                issues=[str(e)],
                confidence=0.5,
            )

    def _analyze_wisdom_stratum(
        self,
        topo: ValidatedTopology,
        financial_metrics: Dict[str, Any],
        stability: float,
        synergy: Dict[str, Any],
        final_verdict: VerdictLevel,
        has_errors: bool,
    ) -> StratumAnalysisResult:
        """Genera el análisis del estrato WISDOM (veredicto integrado)."""
        narrative = self._generate_final_advice(
            topo, financial_metrics, stability, synergy, has_errors,
        )

        confidence = 0.7 if has_errors else 1.0

        return StratumAnalysisResult(
            stratum=Stratum.WISDOM,
            verdict=final_verdict,
            narrative=narrative,
            metrics_summary={
                "final_verdict": final_verdict.name,
                "is_viable": final_verdict.is_positive,
                "requires_action": final_verdict.requires_attention,
            },
            issues=["Errores en análisis previo"] if has_errors else [],
            confidence=confidence,
        )

    # ========================================================================
    # GENERACIÓN DE NARRATIVAS COMPUESTAS
    # ========================================================================

    def _generate_report_header(self) -> str:
        """Genera el encabezado del reporte."""
        return (
            "# 🏗️ INFORME DE INGENIERÍA ESTRATÉGICA\n\n"
            f"*Análisis de Coherencia Fractal | "
            f"Estabilidad Crítica: Ψ < {self.config.stability.critical}*\n"
        )

    def _generate_integrated_diagnosis(
        self,
        stability: float,
        temperature_celsius: float,
        physics_verdict: VerdictLevel,
    ) -> str:
        """Genera diagnóstico integrado (estructura + entorno térmico)."""
        parts: List[str] = []

        stab_class = self.config.stability.classify(stability)

        if stab_class in ("critical", "warning"):
            parts.append(
                f"El edificio se apoya sobre una **base inestable** "
                f"(Ψ={stability:.2f}). La cimentación de recursos es "
                f"insuficiente para la carga táctica."
            )
        else:
            parts.append(
                f"El edificio tiene una **cimentación sólida** "
                f"(Ψ={stability:.2f}). La base de recursos es amplia "
                f"y redundante."
            )

        temp_class = self.config.thermal.classify_temperature(temperature_celsius)

        if temp_class in ("hot", "critical"):
            parts.append(
                f"Sin embargo, el entorno es hostil. Detectamos una "
                f"**Fiebre Inflacionaria** de {temperature_celsius:.1f}°C "
                f"entrando por los insumos (volatilidad de precios)."
            )

            # Interacción base + calor (disipación)
            dissipation_narratives: Dict[str, str] = {
                "robust": (
                    "✅ Gracias a la base ancha, la estructura actúa como un "
                    "**disipador de calor** eficiente. El riesgo de sobrecostos "
                    "se diluye en la red de proveedores resiliente."
                ),
                "critical": (
                    "🚨 Debido a la base estrecha (Pirámide Invertida), "
                    "**el calor no se disipa**. Se concentra en los pocos "
                    "puntos de apoyo, creando riesgo crítico de fractura."
                ),
                "warning": (
                    "🚨 Debido a la base estrecha (Pirámide Invertida), "
                    "**el calor no se disipa**. Se concentra en los pocos "
                    "puntos de apoyo, creando riesgo crítico de fractura."
                ),
            }
            default_dissipation = (
                "⚠️ La estructura tiene capacidad moderada de disipación, "
                "pero una ola de calor sostenida podría comprometer "
                "la integridad financiera."
            )
            parts.append(
                dissipation_narratives.get(stab_class, default_dissipation)
            )
        else:
            parts.append(
                f"El entorno es térmicamente estable ({temperature_celsius:.1f}°C). "
                f"El riesgo inflacionario externo es bajo."
            )

        return " ".join(parts)

    def _generate_final_advice(
        self,
        topo: ValidatedTopology,
        financial_metrics: Dict[str, Any],
        stability: float,
        synergy: Dict[str, Any],
        has_errors: bool,
    ) -> str:
        """Genera el dictamen final del Ingeniero Jefe."""
        if has_errors:
            return self._fetch_narrative("FINAL_VERDICTS", "analysis_failed")

        synergy_detected = bool(synergy.get("synergy_detected", False))
        if synergy_detected:
            result = self._fetch_narrative("FINAL_VERDICTS", "synergy_risk")
            intersecting_cycles = synergy.get("intersecting_cycles", [])
            if intersecting_cycles and isinstance(intersecting_cycles[0], list):
                cycle_explanation = self.explain_cycle_path(intersecting_cycles[0])
                if cycle_explanation:
                    result += f"\n\n{cycle_explanation}"
            return result

        stab_class = self.config.stability.classify(stability)
        is_inverted = stab_class == "critical"

        try:
            fin_verdict = self._validate_financial_metrics(financial_metrics)[
                "recommendation"
            ]
        except (MetricsValidationError, KeyError, TypeError):
            fin_verdict = FinancialVerdict.REVIEW

        if is_inverted:
            if fin_verdict == FinancialVerdict.ACCEPT:
                return self._fetch_narrative(
                    "FINAL_VERDICTS", "inverted_pyramid_viable",
                    {"stability": round(stability, 2)},
                )
            return self._fetch_narrative("FINAL_VERDICTS", "inverted_pyramid_reject")

        if topo.has_cycles:
            return self._fetch_narrative(
                "FINAL_VERDICTS", "has_holes", {"beta_1": topo.beta_1},
            )

        if fin_verdict == FinancialVerdict.ACCEPT:
            return self._fetch_narrative("FINAL_VERDICTS", "certified")

        return self._fetch_narrative("FINAL_VERDICTS", "review_required")

    def _generate_executive_summary(
        self,
        final_verdict: VerdictLevel,
        strata: Dict[Stratum, StratumAnalysisResult],
        errors: List[str],
    ) -> str:
        """Genera resumen ejecutivo del reporte."""
        parts: List[str] = [
            f"**Veredicto: {final_verdict.emoji} {final_verdict.name}**\n"
            f"*{final_verdict.description}*\n"
        ]

        for stratum in [Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY]:
            if stratum in strata:
                analysis = strata[stratum]
                parts.append(
                    f"- **{stratum.name}**: "
                    f"{analysis.verdict.emoji} {analysis.verdict.name}"
                )

        if errors:
            parts.append(
                f"\n⚠️ Se detectaron {len(errors)} errores durante el análisis."
            )

        return "\n".join(parts)

    def _generate_recommendations(
        self,
        final_verdict: VerdictLevel,
        strata: Dict[Stratum, StratumAnalysisResult],
        synergy: Dict[str, Any],
    ) -> List[str]:
        """Genera recomendaciones accionables basadas en el análisis."""
        recommendations: List[str] = []

        if final_verdict == VerdictLevel.VIABLE:
            recommendations.append("✅ Proceder con la ejecución del proyecto.")
            return recommendations

        for stratum, analysis in strata.items():
            if analysis.verdict.requires_attention or analysis.verdict.is_negative:
                if stratum == Stratum.PHYSICS:
                    recommendations.append(
                        "🔧 Revisar fuentes de datos y estabilizar flujo de información."
                    )
                elif stratum == Stratum.TACTICS:
                    if synergy.get("synergy_detected", False):
                        recommendations.append(
                            "⚡ Desacoplar ciclos de dependencia que "
                            "comparten recursos críticos."
                        )
                    beta_1 = analysis.metrics_summary.get("beta_1", 0)
                    if isinstance(beta_1, int) and beta_1 > 0:
                        recommendations.append(
                            "🔄 Sanear topología eliminando ciclos de dependencia."
                        )
                    beta_0 = analysis.metrics_summary.get("beta_0", 1)
                    if isinstance(beta_0, int) and beta_0 > 1:
                        recommendations.append(
                            "🔗 Conectar componentes aislados del proyecto."
                        )
                elif stratum == Stratum.STRATEGY:
                    recommendations.append(
                        "💰 Revisar modelo financiero y ajustar "
                        "parámetros de viabilidad."
                    )

        if not recommendations:
            recommendations.append(
                "🔍 Realizar análisis adicional para determinar siguiente paso."
            )

        # Deduplicar preservando orden
        seen: Set[str] = set()
        unique: List[str] = []
        for r in recommendations:
            if r not in seen:
                seen.add(r)
                unique.append(r)

        return unique[:5]

    # ========================================================================
    # COMPOSICIÓN DE REPORTE ESTRATÉGICO
    # ========================================================================

    def compose_strategic_narrative(
        self,
        topological_metrics: Union[TopologicalMetrics, Dict[str, Any], ValidatedTopology],
        financial_metrics: Dict[str, Any],
        stability: float = 0.0,
        synergy_risk: Optional[Dict[str, Any]] = None,
        spectral: Optional[Dict[str, Any]] = None,
        thermal_metrics: Optional[Union[ThermodynamicMetrics, Dict[str, Any]]] = None,
        physics_metrics: Optional[Union[PhysicsMetrics, Dict[str, Any]]] = None,
        control_metrics: Optional[Union[ControlMetrics, Dict[str, Any]]] = None,
        critical_resources: Optional[List[Dict[str, Any]]] = None,
        raw_cycles: Optional[List[List[str]]] = None,
        **kwargs: Any,
    ) -> StrategicReport:
        """
        Compone el reporte ejecutivo completo integrando todos los estratos DIKW.

        El veredicto final es el supremo (⊔) de todos los veredictos parciales,
        con veto incondicional si β₁ > 0 o Ψ < 1.0.

        Args:
            topological_metrics: Métricas topológicas (Betti, Fiedler, etc.)
            financial_metrics: Métricas financieras (WACC, PI, etc.)
            stability: Índice de estabilidad piramidal Ψ
            synergy_risk: Información de sinergia de riesgo
            spectral: Métricas espectrales adicionales
            thermal_metrics: Métricas termodinámicas
            physics_metrics: Métricas físicas (giroscopía, presión)
            control_metrics: Métricas de control (Laplace)
            critical_resources: Recursos críticos para GraphRAG
            raw_cycles: Ciclos crudos para GraphRAG

        Returns:
            StrategicReport completo con veredicto integrado
        """
        # --- Normalización de entradas ---

        if isinstance(topological_metrics, ValidatedTopology):
            topo = topological_metrics
        else:
            topo = ValidatedTopology.from_metrics(
                topological_metrics,
                strict=self.config.strict_euler_validation,
            )

        # Termodinámica
        tm = thermal_metrics or {}
        if isinstance(tm, ThermodynamicMetrics):
            thermal = tm
        elif isinstance(tm, dict):
            safe_fields = {
                k: v for k, v in tm.items()
                if k in ThermodynamicMetrics.__annotations__
            }
            thermal = ThermodynamicMetrics(**safe_fields) if safe_fields else ThermodynamicMetrics()
        else:
            thermal = ThermodynamicMetrics()

        # Física
        pm = physics_metrics or {}
        physics: Optional[PhysicsMetrics] = None
        if isinstance(pm, PhysicsMetrics):
            physics = pm
        elif isinstance(pm, dict) and pm:
            safe_fields = {
                k: v for k, v in pm.items()
                if k in PhysicsMetrics.__annotations__
            }
            if safe_fields:
                physics = PhysicsMetrics(**safe_fields)

        # Control
        cm = control_metrics or kwargs.get("control") or {}
        control: Optional[ControlMetrics] = None
        if isinstance(cm, ControlMetrics):
            control = cm
        elif isinstance(cm, dict):
            # Fallback legacy
            if not cm and isinstance(pm, dict):
                cm = {
                    "is_stable": pm.get("is_stable_lhp", True),
                    "phase_margin_deg": pm.get("phase_margin_deg", 45.0),
                }
            safe_fields = {
                k: v for k, v in cm.items()
                if k in ControlMetrics.__annotations__
            }
            if safe_fields:
                control = ControlMetrics(**safe_fields)

        synergy = synergy_risk or {}
        spec = spectral or {}

        eff_stability = stability if stability != 0.0 else topo.pyramid_stability

        # --- Acumuladores ---
        strata_analysis: Dict[Stratum, StratumAnalysisResult] = {}
        section_narratives: List[str] = []
        all_verdicts: List[VerdictLevel] = []
        errors: List[str] = []

        # ====== HEADER ======
        section_narratives.append(self._generate_report_header())

        # ====== PHYSICS ======
        physics_result = self._analyze_physics_stratum(
            thermal, eff_stability, physics, control, topo.fiedler_value,
        )
        strata_analysis[Stratum.PHYSICS] = physics_result
        all_verdicts.append(physics_result.verdict)

        # ====== TACTICS ======
        tactics_result = self._analyze_tactics_stratum(
            topo, synergy, spec, eff_stability,
        )
        strata_analysis[Stratum.TACTICS] = tactics_result
        all_verdicts.append(tactics_result.verdict)

        # ====== STRATEGY ======
        strategy_result = self._analyze_strategy_stratum(financial_metrics)
        strata_analysis[Stratum.STRATEGY] = strategy_result
        all_verdicts.append(strategy_result.verdict)

        # ====== Diagnóstico integrado ======
        section_narratives.append("## 🏗️ Diagnóstico del Edificio Vivo\n")
        temp = self._normalize_temperature(thermal.system_temperature)
        integrated_diagnosis = self._generate_integrated_diagnosis(
            eff_stability, temp.celsius, physics_result.verdict,
        )
        section_narratives.append(integrated_diagnosis)
        section_narratives.append("")

        # ====== Dinámica de Bombeo ======
        section_narratives.append("### 0. Dinámica de Bombeo (Física)\n")
        section_narratives.append(physics_result.narrative)
        section_narratives.append("")

        # ====== Topología ======
        section_narratives.append("### 1. Auditoría de Integridad Estructural\n")
        try:
            topo_narrative, topo_verdict = self.translate_topology(
                topo, eff_stability, synergy, spec,
            )
            section_narratives.append(topo_narrative)
            all_verdicts.append(topo_verdict)

            if critical_resources:
                limit = self.config.max_stress_points_display
                for resource in critical_resources[:limit]:
                    node_id = resource.get("id")
                    degree = resource.get("in_degree", 0)
                    if node_id:
                        explanation = self.explain_stress_point(node_id, degree)
                        if explanation:
                            section_narratives.append(f"- {explanation}")

            if raw_cycles:
                limit_cycles = self.config.max_cycle_path_display
                for cycle_path in raw_cycles[:limit_cycles]:
                    explanation = self.explain_cycle_path(cycle_path)
                    if explanation:
                        section_narratives.append(f"- {explanation}")

        except (MetricsValidationError, TopologyInvariantViolation) as e:
            error_msg = f"Error analizando estructura: {e}"
            section_narratives.append(f"❌ {error_msg}")
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
        section_narratives.append("")

        # ====== Finanzas ======
        section_narratives.append("### 2. Análisis de Cargas Financieras\n")
        try:
            fin_narrative, fin_verdict, _ = self.translate_financial(financial_metrics)
            section_narratives.append(fin_narrative)
            all_verdicts.append(fin_verdict)
        except (MetricsValidationError, KeyError, TypeError) as e:
            error_msg = f"Error analizando finanzas: {e}"
            section_narratives.append(f"❌ {error_msg}")
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
        section_narratives.append("")

        # ====== Mercado ======
        section_narratives.append("### 3. Geotecnia de Mercado\n")
        section_narratives.append(self.get_market_context())
        section_narratives.append("")

        # ====== OMEGA: Colapso del retículo ======
        is_financially_viable = strategy_result.verdict.is_positive

        graph_for_narrator: nx.DiGraph = kwargs.get("graph", nx.DiGraph())
        if not isinstance(graph_for_narrator, nx.DiGraph):
            graph_for_narrator = nx.DiGraph()

        omega_narrative = LatticeVerdictCollapse.enforce_semantic_diffeomorphism(
            roi_viable=is_financially_viable,
            betti_1=topo.beta_1,
            psi=eff_stability,
            graph=graph_for_narrator,
            max_cycles=self.config.max_cycles_to_enumerate,
        )

        # Colapso final: supremum de todos + veto incondicional
        final_verdict = VerdictLevel.supremum(*all_verdicts)

        # Veto topológico/físico incondicional
        if topo.beta_1 > 0 or eff_stability < 1.0:
            final_verdict = final_verdict | VerdictLevel.RECHAZAR

        if errors:
            final_verdict = final_verdict | VerdictLevel.REVISAR
            omega_narrative += (
                f"\nSe detectaron {len(errors)} errores en la evaluación."
            )

        # Verificar coherencia del homomorfismo SeverityLattice → VerdictLevel
        severity_verdict = SeverityLattice.RECHAZAR if (topo.beta_1 > 0 or eff_stability < 1.0) else SeverityLattice.VIABLE
        mapped_verdict = SeverityToVerdictHomomorphism.apply(severity_verdict)
        if not (mapped_verdict.value <= final_verdict.value):
            logger.error(
                f"Homomorphism coherence violation: "
                f"φ({severity_verdict.name})={mapped_verdict.name} "
                f"but final={final_verdict.name}"
            )
            final_verdict = final_verdict | mapped_verdict

        # Registrar estrato OMEGA
        omega_confidence = 1.0 if not errors else 0.5
        omega_result = StratumAnalysisResult(
            stratum=Stratum.OMEGA,
            verdict=final_verdict,
            narrative=omega_narrative,
            metrics_summary={
                "errors_count": len(errors),
                "verdict": final_verdict.name,
                "homomorphism_verified": True,
            },
            issues=errors,
            confidence=omega_confidence,
        )
        strata_analysis[Stratum.OMEGA] = omega_result

        # Registrar estrato ALPHA (si existe en Stratum)
        if hasattr(Stratum, "ALPHA"):
            alpha_result = StratumAnalysisResult(
                stratum=Stratum.ALPHA,
                verdict=final_verdict,
                narrative="Viabilidad de negocio evaluada.",
                metrics_summary={"business_valid": final_verdict.is_positive},
                issues=[],
                confidence=omega_confidence,
            )
            strata_analysis[Stratum.ALPHA] = alpha_result

        # ====== WISDOM ======
        wisdom_result = self._analyze_wisdom_stratum(
            topo, financial_metrics, eff_stability, synergy,
            final_verdict, bool(errors),
        )
        strata_analysis[Stratum.WISDOM] = wisdom_result

        section_narratives.append("### 💡 Dictamen del Ingeniero Jefe\n")
        section_narratives.append(wisdom_result.narrative)

        # --- Ensamblar reporte ---
        executive_summary = self._generate_executive_summary(
            final_verdict, strata_analysis, errors,
        )

        recommendations = self._generate_recommendations(
            final_verdict, strata_analysis, synergy,
        )

        confidences = [a.confidence for a in strata_analysis.values()]
        global_confidence = sum(confidences) / len(confidences) if confidences else 1.0
        global_confidence = max(0.0, min(1.0, global_confidence))

        raw_narrative = "\n".join(section_narratives)
        if len(raw_narrative) > self.config.max_narrative_length:
            raw_narrative = raw_narrative[:self.config.max_narrative_length - 3] + "..."

        return StrategicReport(
            title="INFORME DE INGENIERÍA ESTRATÉGICA",
            verdict=final_verdict,
            executive_summary=executive_summary,
            strata_analysis=strata_analysis,
            recommendations=recommendations,
            raw_narrative=raw_narrative,
            confidence=global_confidence,
        )

    def assemble_data_product(
        self,
        graph: nx.DiGraph,
        report: Any,
    ) -> Dict[str, Any]:
        """
        Ensambla el Producto de Datos Final (Sabiduría).

        Integra la estructura (Grafo) y el análisis (Reporte) en un artefacto
        consumible por el negocio.
        """
        narrative = getattr(report, "strategic_narrative", "")
        verdict = getattr(report, "integrity_score", 0.0)

        return {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "verdict_score": verdict,
                "complexity_level": getattr(report, "complexity_level", "Unknown"),
            },
            "narrative": {
                "executive_summary": narrative,
                "alerts": (
                    getattr(report, "waste_alerts", [])
                    + getattr(report, "circular_risks", [])
                ),
            },
            "topology": {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "details": getattr(report, "details", {}),
            },
        }

    # ========================================================================
    # COMPATIBILIDAD LEGACY
    # ========================================================================

    def compose_strategic_narrative_legacy(
        self,
        topological_metrics: Any,
        financial_metrics: Dict[str, Any],
        stability: float = 0.0,
        synergy_risk: Optional[Dict[str, Any]] = None,
        spectral: Optional[Dict[str, Any]] = None,
        thermal_metrics: Optional[Dict[str, Any]] = None,
        physics_metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Versión legacy que retorna solo el string de narrativa.

        Mantiene compatibilidad con código existente.
        """
        report = self.compose_strategic_narrative(
            topological_metrics=topological_metrics,
            financial_metrics=financial_metrics,
            stability=stability,
            synergy_risk=synergy_risk,
            spectral=spectral,
            thermal_metrics=thermal_metrics,
            physics_metrics=physics_metrics,
        )
        return report.raw_narrative


# ============================================================================
# FUNCIONES DE UTILIDAD Y FACTORY
# ============================================================================


def create_translator(
    config: Optional[TranslatorConfig] = None,
    market_provider: Optional[Callable[[], str]] = None,
    mic: Optional[MICRegistry] = None,
    enable_cache: bool = True,
) -> SemanticTranslator:
    """Factory function para crear un traductor configurado."""
    return SemanticTranslator(
        config=config,
        market_provider=market_provider,
        mic=mic,
        enable_cache=enable_cache,
    )


def translate_metrics_to_narrative(
    topological_metrics: Any,
    financial_metrics: Dict[str, Any],
    stability: float = 0.0,
    **kwargs: Any,
) -> str:
    """Función de conveniencia para traducción rápida."""
    translator = SemanticTranslator()
    report = translator.compose_strategic_narrative(
        topological_metrics=topological_metrics,
        financial_metrics=financial_metrics,
        stability=stability,
        **kwargs,
    )
    return report.raw_narrative


def verify_verdict_lattice() -> bool:
    """
    Verifica que VerdictLevel cumple todas las leyes del retículo.

    Returns:
        True si todas las propiedades algebraicas se cumplen
    """
    results = VerdictLevel.verify_lattice_laws()
    all_pass = all(results.values())

    if not all_pass:
        failed = [k for k, v in results.items() if not v]
        logger.error(f"Lattice laws violated: {failed}")

    return all_pass


def verify_severity_homomorphism() -> bool:
    """
    Verifica que el homomorfismo SeverityLattice → VerdictLevel
    preserva las operaciones del retículo.

    Returns:
        True si el homomorfismo es válido
    """
    result = SeverityToVerdictHomomorphism.verify_homomorphism()
    if not result:
        logger.error("SeverityLattice → VerdictLevel homomorphism is invalid")
    return result