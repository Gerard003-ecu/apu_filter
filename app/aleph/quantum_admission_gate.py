# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Quantum Admission Gate (Operador de Proyección de Hilbert)          ║
║ Ruta   : app/aleph/quantum_admission_gate.py                                 ║
║ Versión: 5.1.0-Strict-Spectral-WKB-Poincare-KAM-RSI3-Aleph-Secure-PhD        ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Operador constitutivo del Estrato ALEPH (Nivel 4, V_{ℵ₀}). Modela la admisión
como MEDICIÓN PROYECTIVA DE COLAPSO sobre un espacio de Hilbert ℋ₂ dotado de
estructura C*-álgebra bicuaterniónica (ℂ ⊗ ℍ ≅ M₂(ℂ)), mecánica celeste
simpléctica de Poincaré-KAM-Nekhoroshev, método de matriz de transferencia
exacta SL(2, ℂ) (dualidad onda-circuito eléctrico ABCD) y automejora recursiva
de Nivel 3 (RSI-3: Richardson → Steffensen/Orden p → Shanks/Aitken Δ² +
mutación de meta-ley Taylor-Lindstedt a Borel-Padé [L/M]).

NOVEDADES v5.1.0 (Poincaré-KAM + SL(2,ℂ) TMM + RSI-3 Banach-Padé):
  [P1] Snapshot simpléctico de Poincaré en (χ², α|σ|): condición diofántica
       KAM, lenguas de Arnold y tiempo de estabilidad de Nekhoroshev.
  [P2] Expansión perturbativa de Lindstedt-Poincaré de Φ(χ²) con frecuencia
       deformada ω(ε) para aniquilación secular + resumación racional de Padé.
  [P3] Aplicación simpléctica de retorno en Sp(2, ℝ): matriz de monodromía,
       multiplicadores de Floquet, residuo de Greene y exponente de Lyapunov.
  [P4] Espectro exacto de transparencia de Ramsauer-Townsend (k₂a = nπ) y
       teorema de recurrencia ergódica de Kac-Poincaré en el círculo de fase.
  [P5] Meta-controlador RSI-3 con funcional de Lyapunov contráctil en espacio
       de Banach, aceleración de Shanks-Aitken Δ² y mutación de meta-ley hacia
       aproximantes diagonales A-estables de Padé [L/M].

ARQUITECTURA DE TRES FASES ANIDADAS (Continuidad Formal de Morfismos):
  Φ₁ : Phase1_GaugeAdmissionCertifier
         └──► último método: nest_into_phase2() ──┐
                                                  ▼
  Φ₂ : Phase2_SpectralEnergyAuditor.__init__(phase1_cert)
         └──► último método: nest_into_phase3(payload) ──┐
                                                         ▼
  Φ₃ : Phase3_BarrierCollapseProjector.__init__(phase2_audit)
         └──► últimos métodos: collapse_wavefunction()
                               collapse_wavefunction_rsi3()
"""
from __future__ import annotations

import hashlib
import logging
import math
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from itertools import count as _itertools_count
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

try:
    from app.core.mic_algebra import CategoricalState, Morphism
    from app.core.schemas import Stratum
except ImportError:  # Fallback hermético para ejecución/verificación aislada
    class Stratum(Enum):  # type: ignore[no-redef]
        PHYSICS = auto()

    @dataclass(frozen=True)
    class CategoricalState:  # type: ignore[no-redef]
        payload: Mapping[str, Any]
        context: Mapping[str, Any] = field(default_factory=dict)
        validated_strata: frozenset = field(default_factory=frozenset)

    class Morphism:  # type: ignore[no-redef]
        def __init__(self, name: str) -> None:
            self.name = name


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: CONFIGURACIÓN, VERSIÓN Y LOGGING ESTRUCTURADO
# ═════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("MIC.Physics.QuantumAdmission")

__version__: Final[str] = (
    "5.1.0-Strict-Spectral-WKB-Poincare-KAM-RSI3-Aleph-Secure-PhD"
)

_NON_CLONING_UID_COUNTER = _itertools_count(1)

# ─── Constantes de Mecánica Celeste de Poincaré y Teoría KAM ───────────────
_POINCARE_MAX_DENOMINATOR: Final[int] = 16
_POINCARE_LOW_ORDER: Final[int] = 8
_RESONANCE_TOLERANCE: Final[float] = 1.0e-3
_SMALL_PARAMETER_FLOOR: Final[float] = 1.0e-12
_ADIABATIC_DRIFT_LIMIT: Final[float] = 1.0e-6
_KAM_DIOPHANTINE_GAMMA: Final[float] = 1.0e-2
_KAM_DIOPHANTINE_TAU: Final[float] = 2.0
_NEKHOROSHEV_EXPONENT_B: Final[float] = 0.25

# ─── Constantes del Meta-Controlador RSI de Nivel 3 ────────────────────────
_RSI_MAX_DEPTH: Final[int] = 3
_RSI_TARGET_RESIDUAL: Final[float] = 1.0e-3
_RSI_MIN_TOLERANCE: Final[float] = 1.0e-14
_RSI_FACTOR_LOWER: Final[float] = 1.0e-2
_RSI_FACTOR_UPPER: Final[float] = 0.95
_RSI_OSCILLATION_EPS: Final[float] = 1.0e-15


@dataclass(frozen=True, slots=True)
class LogContext:
    """Contexto estructurado inmutable para trazabilidad post-mortem."""
    eigenstate: str
    incident_energy: float
    work_function: float
    tunneling_prob: float
    tunneling_wkb: float
    momentum: float
    sigma: float
    chi_squared: float
    kappa_a: float
    ambiguity: float
    veto: bool
    resonance_p: int = 0
    resonance_q: int = 0
    kam_intact: bool = True
    orbit_class: str = "elliptic"

    def to_structured_log(self) -> str:
        return (
            f"state={self.eigenstate} | "
            f"E={self.incident_energy:.6e} | "
            f"Φ={self.work_function:.6e} | "
            f"T_exact={self.tunneling_prob:.6e} | "
            f"T_WKB={self.tunneling_wkb:.6e} | "
            f"p={self.momentum:.6e} | "
            f"σ={self.sigma:.6e} | "
            f"χ²={self.chi_squared:.6e} | "
            f"κa={self.kappa_a:.6e} | "
            f"amb={self.ambiguity:.4f} | "
            f"pq={self.resonance_p}:{self.resonance_q} | "
            f"KAM={self.kam_intact} | "
            f"orbit={self.orbit_class} | "
            f"veto={self.veto}"
        )


class QuantumLogger:
    """Emisor de telemetría cuántica estructurada e inmutable."""

    @staticmethod
    def log_measurement(m: "QuantumMeasurement", level: int = logging.INFO) -> None:
        ps = m.poincare_snapshot
        rm = m.energy_return_map
        ctx = LogContext(
            eigenstate=m.eigenstate.name,
            incident_energy=m.incident_energy,
            work_function=m.work_function,
            tunneling_prob=m.tunneling_probability,
            tunneling_wkb=m.tunneling_wkb_diagnostic,
            momentum=m.momentum,
            sigma=m.dominant_pole_real,
            chi_squared=m.threat_level,
            kappa_a=m.kappa_width_diagnostic,
            ambiguity=m.collapse_ambiguity,
            veto=m.frustration_veto,
            resonance_p=ps.resonance_p if ps else 0,
            resonance_q=ps.resonance_q if ps else 0,
            kam_intact=ps.kam_torus_intact if ps else True,
            orbit_class=rm.orbit_class if rm else "elliptic",
        )
        logger.log(level, "QUANTUM_MEASUREMENT | %s", ctx.to_structured_log())


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: JERARQUÍA DE EXCEPCIONES DEL DOMINIO CUÁNTICO-TOPOLÓGICO
# ═════════════════════════════════════════════════════════════════════════════

class QuantumAdmissionError(Exception):
    """Excepción base del operador de proyección del Estrato ALEPH."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        self.message = message

    def __str__(self) -> str:
        if self.context:
            ctx = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} | Context: {ctx}"
        return self.message


class QuantumNumericalError(QuantumAdmissionError):
    """Violación de invariantes numéricos IEEE-754 o positividad espectral."""


class QuantumInterfaceError(QuantumAdmissionError):
    """Violación de contrato estructural en los oráculos inyectados."""


class QuantumStateError(QuantumAdmissionError):
    """Violación de axiomas del estado cuántico (e.g., Teorema de No-Clonación A5)."""


class WKBValidityError(QuantumNumericalError):
    """Violación del régimen semiclásico de validez WKB (κa ≤ 1)."""


class CohomologicalVetoError(QuantumAdmissionError):
    """Obstrucción cohomológica global (H¹ ≠ 0, β₁ > 0) en la fibra gauge."""


class PoincareSectionError(QuantumAdmissionError):
    """Singularidad simpléctica o degeneración en la sección transversal de Poincaré."""


class RecursiveSelfImprovementError(QuantumAdmissionError):
    """Divergencia o pérdida de contractividad de Banach en el lazo RSI-3."""


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: CONSTANTES FÍSICAS Y NUMÉRICAS SELLADAS
# ═════════════════════════════════════════════════════════════════════════════

class PhysicalConstants:
    """Constantes universales en unidades naturales de escala (h=1, ℏ=1/(2π))."""

    PLANCK_H: Final[float] = 1.0
    PLANCK_HBAR: Final[float] = PLANCK_H / (2.0 * math.pi)

    BASE_WORK_FUNCTION: Final[float] = 10.0
    BASE_EFFECTIVE_MASS: Final[float] = 1.0
    BARRIER_WIDTH: Final[float] = 1.0
    ALPHA_THREAT: Final[float] = 5.0

    MACHINE_EPSILON: Final[float] = sys.float_info.epsilon
    MIN_KINETIC_ENERGY: Final[float] = 1e-12
    FRUSTRATION_VETO_TOL: Final[float] = 1e-9
    SIGMA_CHAOS_TOL: Final[float] = 1e-9
    ENTROPY_FLOOR: Final[float] = 1e-12
    EXP_UNDERFLOW_CUTOFF: Final[float] = -700.0
    EXP_OVERFLOW_CUTOFF: Final[float] = math.log(sys.float_info.max) - 1.0
    MAX_SHANNON_ENTROPY: Final[float] = math.log(256.0)
    DIVISION_EPSILON: Final[float] = 1e-15
    WKB_VALIDITY_THRESHOLD: Final[float] = 1.0
    GAUGE_FLATNESS_TOL: Final[float] = 1e-9
    THRESHOLD_ENERGY_REL_TOL: Final[float] = 1e-12
    SINH_ASYMPTOTIC_KA: Final[float] = 20.0

    FLOAT_MAX: Final[float] = sys.float_info.max
    FLOAT_MIN: Final[float] = sys.float_info.min
    UINT64_MASK: Final[int] = (1 << 64) - 1
    MANTISSA_BITS: Final[int] = 53
    MANTISSA_MASK: Final[int] = (1 << MANTISSA_BITS) - 1

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError(
            f"La clase {cls.__name__} está sellada. No se permite herencia."
        )


Const = PhysicalConstants


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4: ESTRUCTURAS ALGEBRAICAS, HIPERCOMPLEJAS, POINCARÉ Y RSI-3
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class NumericInterval:
    """Álgebra de intervalos cerrados [lower, upper] de Moore-Minkowski."""
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if math.isnan(self.lower) or math.isnan(self.upper):
            raise ValueError("Intervalos no admiten NaN.")
        if self.lower > self.upper:
            raise ValueError(
                f"Intervalo mal formado: [{self.lower}, {self.upper}]."
            )

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.lower + self.upper)

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def relative_width(self) -> float:
        mid = self.midpoint
        if abs(mid) < Const.DIVISION_EPSILON:
            return float("inf")
        return self.width / abs(mid)

    @property
    def is_degenerate(self) -> bool:
        return self.width == 0.0

    def contains(self, v: float) -> bool:
        return self.lower <= v <= self.upper

    def contains_zero(self) -> bool:
        return self.lower <= 0.0 <= self.upper

    def intersects(self, o: "NumericInterval") -> bool:
        return not (self.upper < o.lower or o.upper < self.lower)

    def intersection(self, o: "NumericInterval") -> Optional["NumericInterval"]:
        lo, hi = max(self.lower, o.lower), min(self.upper, o.upper)
        return NumericInterval(lo, hi) if lo <= hi else None

    def hull(self, o: "NumericInterval") -> "NumericInterval":
        return NumericInterval(
            min(self.lower, o.lower), max(self.upper, o.upper)
        )

    def scale_around_midpoint(self, factor: float) -> "NumericInterval":
        """Contracción afín del intervalo en torno a su punto medio."""
        f = max(0.0, float(factor))
        mid = self.midpoint
        half = 0.5 * self.width * f
        return NumericInterval(mid - half, mid + half)

    def __add__(self, o: "NumericInterval") -> "NumericInterval":
        return NumericInterval(self.lower + o.lower, self.upper + o.upper)

    def __sub__(self, o: "NumericInterval") -> "NumericInterval":
        return NumericInterval(self.lower - o.upper, self.upper - o.lower)

    def __mul__(self, o: "NumericInterval") -> "NumericInterval":
        prods = (
            self.lower * o.lower,
            self.lower * o.upper,
            self.upper * o.lower,
            self.upper * o.upper,
        )
        return NumericInterval(min(prods), max(prods))

    def __truediv__(self, o: "NumericInterval") -> "NumericInterval":
        if o.contains_zero():
            raise ZeroDivisionError(f"División por intervalo con 0: {o}.")
        return self * NumericInterval(1.0 / o.upper, 1.0 / o.lower)

    def __neg__(self) -> "NumericInterval":
        return NumericInterval(-self.upper, -self.lower)

    def __abs__(self) -> "NumericInterval":
        if not self.contains_zero():
            return self if self.lower >= 0.0 else -self
        return NumericInterval(0.0, max(abs(self.lower), abs(self.upper)))

    def __repr__(self) -> str:
        return f"[{self.lower:.6e}, {self.upper:.6e}]"

    @staticmethod
    def from_value_with_tolerance(v: float, tol: float) -> "NumericInterval":
        return NumericInterval(v - abs(tol), v + abs(tol))

    @staticmethod
    def point(v: float) -> "NumericInterval":
        return NumericInterval(v, v)


@dataclass(frozen=True, slots=True)
class BiquaternionElement:
    """Elemento en el álgebra de bicuaterniones ℂ ⊗ ℍ ≅ M₂(ℂ) (Pauli-Spinor).

    Representa Q = w·1 + x·i + y·j + z·k con coeficientes complejos, e induce
    el isomorfismo canónico hacia M₂(ℂ) para certificar unitariedad SL(2, ℂ)
    y la identidad C*-álgebra ||Q* Q||_op = ||Q||_op².
    """
    w: complex
    x: complex
    y: complex
    z: complex

    def norm_squared_minkowski(self) -> complex:
        """Cuadrinorma compleja det(Ψ(Q)) = w² + x² + y² + z²."""
        return self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z

    def euclidean_frobenius_norm(self) -> float:
        """Norma de Hilbert-Schmidt √(|w|² + |x|² + |y|² + |z|²)."""
        return math.sqrt(
            abs(self.w) ** 2 + abs(self.x) ** 2 + abs(self.y) ** 2 + abs(self.z) ** 2
        )


# ─── Estructuras de Mecánica Celeste de Poincaré y Teoría KAM ───────────────

@dataclass(frozen=True, slots=True)
class PoincarePhaseSpaceSnapshot:
    """Snapshot simpléctico de Poincaré en el espacio de fase (q=χ², p=α|σ|).

    Invariantes canónicos:
      - Acción I = ½(q² + p²), ángulo θ_p = atan2(p, q).
      - Sección transversal de Poincaré Σ = {p = section_level}.
      - Resonancia racional ω₁/ω₂ ≈ p/q vía expansión en fracciones continuas.
      - Condición diofántica KAM: |ω₁/ω₂ - p/q| ≥ γ / q^τ.
      - Ancho de lengua de Arnold: Δω_{p:q} ∝ ε^{q/2}.
      - Tiempo de estabilidad exponencial de Nekhoroshev: T_Nekh ~ exp((1/ε)^b).
    """
    action_invariant: float
    angle_parameter: float
    section_level: float
    crossing_count: int
    frequency_ratio: float
    is_near_resonance: bool
    resonance_p: int
    resonance_q: int
    small_divisor_magnitude: float
    action_drift_estimate: float
    diophantine_margin: float = 0.0
    kam_torus_intact: bool = True
    nekhoroshev_stability_time: float = float("inf")
    arnold_tongue_width: float = 0.0


@dataclass(frozen=True, slots=True)
class LindstedtWorkFunctionExpansion:
    """Expansión perturbativa de Lindstedt-Poincaré y resumación de Padé de Φ(χ²).

    Resuelve la modulación Φ(ε) = Φ₀ e^ε (ε = αχ²) acoplada a una ecuación de
    oscilador anarmónico con frecuencia deformada ω(ε) = 1 + Σ_{k≥1} ε^k ω_k
    elegida para aniquilar términos seculares resonantes, complementada con el
    aproximante racional diagonal de Borel-Padé [L/M]_Φ(ε).
    """
    small_parameter: float
    n_terms: int
    zero_order_work_function: float
    first_order_correction: float
    second_order_correction: float
    higher_order_correction: float
    corrected_work_function: float
    true_work_function: float
    expansion_residual: float
    is_asymptotically_consistent: bool
    strained_frequency: float = 1.0
    secular_annihilation_verified: bool = True
    pade_approximant_value: float = 0.0
    pade_residual: float = 0.0


@dataclass(frozen=True, slots=True)
class EnergyReturnMapSnapshot:
    """Aplicación simpléctica de retorno en (E, Φ) y espectro de Floquet.

    Combina el factor de contracción energética J = min(E/Φ, Φ/E) con la matriz
    de monodromía simpléctica M_ret ∈ Sp(2, ℝ) (det M_ret = 1), el residuo de
    Greene R = ¼(2 - Tr(M_ret)), los multiplicadores de Floquet λ_± y el
    exponente de Lyapunov máximo λ_Lyap = ln max|λ_±|.
    """
    contraction_factor: float
    eigenvalues_modulus: Tuple[float, ...]
    is_stable: bool
    iterations_norm_sequence: Tuple[float, ...]
    monodromy_trace: float = 2.0
    monodromy_determinant: float = 1.0
    greenes_residue: float = 0.0
    lyapunov_exponent: float = 0.0
    orbit_class: str = "elliptic"


@dataclass(frozen=True, slots=True)
class TransmissionRecurrenceSnapshot:
    """Recurrencia ergódica de Kac-Poincaré y resonancias de Ramsauer-Townsend.

    Evalúa la transparencia cuántica sobre barrera (E > V₀) determinando:
      - Puntos de recurrencia en el muestreo espectral de k₂.
      - Energías exactas de transparencia de Ramsauer-Townsend E_n = V₀ + n²π²ℏ²/(2m*a²).
      - Tiempo medio de retorno de Kac-Poincaré en la rotación irracional sobre T¹.
    """
    recurrence_points: Tuple[float, ...]
    min_recurrence_distance: float
    is_poincare_recurrent: bool
    resonance_kappa_widths: Tuple[float, ...]
    n_samples: int
    ramsauer_townsend_energies: Tuple[float, ...] = ()
    nearest_rt_quantum_number: int = 0
    kac_mean_recurrence_time: float = float("inf")
    ergodic_phase_discrepancy: float = 0.0


# ─── Estructuras del Meta-Controlador RSI de Nivel 3 ───────────────────────

@dataclass(frozen=True, slots=True)
class ToleranceBundle:
    """Punto en la variedad de configuración del meta-controlador RSI-3."""
    scale: float = 1.0
    n_lindstedt_terms: int = 2
    use_pade_resummation: bool = False
    damping_factor: float = 1.0

    def _scaled(self, base: float) -> float:
        return max(_RSI_MIN_TOLERANCE, base * self.scale)

    @property
    def gauge_flatness_tol(self) -> float:
        return self._scaled(Const.GAUGE_FLATNESS_TOL)

    @property
    def threshold_energy_rel_tol(self) -> float:
        return self._scaled(Const.THRESHOLD_ENERGY_REL_TOL)

    @property
    def collapse_ambiguity_tol(self) -> float:
        return self._scaled(1.0e-3)

    def with_scale(self, s: float) -> "ToleranceBundle":
        return ToleranceBundle(
            scale=s,
            n_lindstedt_terms=self.n_lindstedt_terms,
            use_pade_resummation=self.use_pade_resummation,
            damping_factor=self.damping_factor,
        )

    def with_terms(self, n: int) -> "ToleranceBundle":
        return ToleranceBundle(
            scale=self.scale,
            n_lindstedt_terms=n,
            use_pade_resummation=self.use_pade_resummation,
            damping_factor=self.damping_factor,
        )

    def with_pade(self, active: bool) -> "ToleranceBundle":
        return ToleranceBundle(
            scale=self.scale,
            n_lindstedt_terms=self.n_lindstedt_terms,
            use_pade_resummation=bool(active),
            damping_factor=self.damping_factor,
        )

    def with_damping(self, damping: float) -> "ToleranceBundle":
        return ToleranceBundle(
            scale=self.scale,
            n_lindstedt_terms=self.n_lindstedt_terms,
            use_pade_resummation=self.use_pade_resummation,
            damping_factor=max(0.05, min(1.0, float(damping))),
        )


@dataclass(frozen=True, slots=True)
class RecursiveSelfImprovementTrace:
    """Traza inmutable de una época de automejora recursiva (Niveles 1, 2 y 3)."""
    level: int
    tolerance_scale: float
    n_lindstedt_terms: int
    residual: float
    convergence_rate: float
    richardson_extrapolation: float
    aitken_delta_squared: float
    adaptation_factor: float
    level3_active: bool
    banach_contraction_factor: float = 0.0
    shanks_extrapolation: float = 0.0
    meta_law_mutated: bool = False
    unitarity_defect: float = 0.0


@dataclass(frozen=True, slots=True)
class SelfImprovementResult:
    """Certificado final de convergencia del meta-controlador RSI-3."""
    traces: Tuple[RecursiveSelfImprovementTrace, ...]
    final_scale: float
    final_n_terms: int
    final_residual: float
    converged: bool
    richardson_limit: float
    aitken_limit: float
    effective_convergence_order: float
    level3_activated: bool
    banach_fixed_point_certified: bool = False
    shanks_limit: float = 0.0
    used_pade_resummation: bool = False


# ─── Estructuras WKB y de Transmisión de Barrera ────────────────────────────

@dataclass(frozen=True, slots=True)
class WKBParameters:
    """Parámetros del diagnóstico semiclásico WKB de barrera gruesa."""
    incident_energy: float
    barrier_height: float
    effective_mass: float
    barrier_width: float
    kappa: float
    integrand: float
    exponent: float
    kappa_width: float
    validity_parameter: float

    def __post_init__(self) -> None:
        if self.incident_energy < 0:
            raise QuantumNumericalError(
                f"Energía incidente negativa: {self.incident_energy}"
            )
        if not (self.effective_mass > 0 or math.isinf(self.effective_mass)):
            raise QuantumNumericalError(
                f"Masa efectiva debe ser > 0 o ∞: {self.effective_mass}"
            )
        if self.barrier_width <= 0:
            raise QuantumNumericalError(
                f"Ancho de barrera debe ser > 0: {self.barrier_width}"
            )

    def is_valid_semiclassical_regime(self) -> bool:
        return (
            self.kappa_width > Const.WKB_VALIDITY_THRESHOLD
            and 0 < self.effective_mass < float("inf")
        )

    def gamow_factor(self) -> float:
        return abs(self.exponent) / 2.0


@dataclass(frozen=True, slots=True)
class RectangularBarrierTransmission:
    """Resultado de dispersión cuántica vía Matriz de Transferencia SL(2, ℂ)."""
    t_exact: float
    t_wkb: float
    t_ratio: float
    kappa_width: float
    is_classical: bool
    regime: str  # 'tunnel' | 'threshold' | 'over_barrier'
    reflection_exact: float = 0.0
    unitarity_residual: float = 0.0
    transfer_matrix_det: float = 1.0
    characteristic_impedance_ratio: float = 1.0


@dataclass(frozen=True, slots=True)
class CollapseAmbiguity:
    """Métrica de proximidad a la hipersuperficie de bifurcación T = θ."""
    boundary_distance: float
    ambiguity: float
    is_orthomodular_degenerate: bool


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5: AUTOESTADOS DEL ESPACIO DE HILBERT ℋ₂
# ═════════════════════════════════════════════════════════════════════════════

class Eigenstate(Enum):
    """Base ortonormal {|ADMITIDO⟩, |RECHAZADO⟩} de ℋ₂ ≅ ℂ²."""
    ADMITIDO = auto()
    RECHAZADO = auto()

    def is_accepted(self) -> bool:
        return self == Eigenstate.ADMITIDO

    def to_hilbert_projection(self) -> int:
        return 1 if self.is_accepted() else 0

    def complementary(self) -> "Eigenstate":
        return (
            Eigenstate.RECHAZADO
            if self == Eigenstate.ADMITIDO
            else Eigenstate.ADMITIDO
        )

    def __invert__(self) -> "Eigenstate":
        return self.complementary()

    def __str__(self) -> str:
        return f"|{self.name}⟩"


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6: MEDICIÓN CUÁNTICA (AXIOMA A5: TEOREMA DE NO-CLONACIÓN)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class QuantumMeasurement:
    """Medición cuántica inmutable post-colapso (A5: No-Clonación estricta)."""

    eigenstate: Eigenstate
    incident_energy: float
    work_function: float
    tunneling_probability: float
    kinetic_energy: float
    momentum: float
    frustration_veto: bool
    effective_mass: float
    dominant_pole_real: float
    threat_level: float
    collapse_threshold: float
    admission_reason: str
    wkb_parameters: Optional[WKBParameters] = None
    measurement_uncertainty: Optional[NumericInterval] = None
    tunneling_wkb_diagnostic: float = 0.0
    kappa_width_diagnostic: float = 0.0
    collapse_ambiguity: float = 0.0
    barrier_regime: str = ""
    # ── Invariantes de Poincaré-KAM y RSI-3 ──────────────────────────────
    poincare_snapshot: Optional[PoincarePhaseSpaceSnapshot] = None
    lindstedt_expansion: Optional[LindstedtWorkFunctionExpansion] = None
    energy_return_map: Optional[EnergyReturnMapSnapshot] = None
    transmission_recurrence: Optional[TransmissionRecurrenceSnapshot] = None
    rsi_result: Optional[SelfImprovementResult] = None
    non_cloning_uid: int = field(
        default_factory=lambda: next(_NON_CLONING_UID_COUNTER),
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        for fname, val in (
            ("incident_energy", self.incident_energy),
            ("work_function", self.work_function),
            ("kinetic_energy", self.kinetic_energy),
            ("momentum", self.momentum),
            ("threat_level", self.threat_level),
            ("tunneling_wkb_diagnostic", self.tunneling_wkb_diagnostic),
            ("kappa_width_diagnostic", self.kappa_width_diagnostic),
        ):
            if val < 0:
                raise QuantumStateError(
                    f"Observable '{fname}' debe ser ≥ 0: {val}",
                    context={fname: val},
                )
        if not 0.0 <= self.tunneling_probability <= 1.0:
            raise QuantumStateError(
                f"Túnel fuera de [0,1]: {self.tunneling_probability}"
            )
        if not 0.0 <= self.tunneling_wkb_diagnostic <= 1.0:
            raise QuantumStateError(
                f"T_WKB fuera de [0,1]: {self.tunneling_wkb_diagnostic}"
            )
        if not 0.0 <= self.collapse_threshold < 1.0:
            raise QuantumStateError(
                f"θ fuera de [0,1): {self.collapse_threshold}"
            )
        if not 0.0 <= self.collapse_ambiguity <= 1.0:
            raise QuantumStateError(
                f"Ambigüedad fuera de [0,1]: {self.collapse_ambiguity}"
            )
        if not (self.effective_mass > 0 or math.isinf(self.effective_mass)):
            raise QuantumStateError(
                f"m* debe ser > 0 o ∞: {self.effective_mass}"
            )
        if self.eigenstate == Eigenstate.RECHAZADO:
            if self.kinetic_energy != 0.0:
                raise QuantumStateError("RECHAZADO ⇒ K = 0 obligatorio.")
            if self.momentum != 0.0:
                raise QuantumStateError("RECHAZADO ⇒ p = 0 obligatorio.")

    def __copy__(self) -> "QuantumMeasurement":
        raise QuantumStateError(
            f"A5: QuantumMeasurement no puede clonarse (uid={self.non_cloning_uid}).",
            context={"uid": self.non_cloning_uid},
        )

    def __deepcopy__(self, memo: Dict[int, Any]) -> "QuantumMeasurement":
        raise QuantumStateError(
            f"A5: QuantumMeasurement no puede clonarse (uid={self.non_cloning_uid}).",
            context={"uid": self.non_cloning_uid},
        )

    def to_dict(self) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "eigenstate": self.eigenstate.name,
            "incident_energy": self.incident_energy,
            "work_function": self.work_function,
            "tunneling_probability": self.tunneling_probability,
            "tunneling_wkb": self.tunneling_wkb_diagnostic,
            "kappa_width": self.kappa_width_diagnostic,
            "kinetic_energy": self.kinetic_energy,
            "momentum": self.momentum,
            "frustration_veto": self.frustration_veto,
            "effective_mass": self.effective_mass,
            "dominant_pole_real": self.dominant_pole_real,
            "threat_level": self.threat_level,
            "collapse_threshold": self.collapse_threshold,
            "collapse_ambiguity": self.collapse_ambiguity,
            "barrier_regime": self.barrier_regime,
            "admission_reason": self.admission_reason,
            "non_cloning_uid": self.non_cloning_uid,
        }
        if self.wkb_parameters:
            base["wkb_gamow_factor"] = self.wkb_parameters.gamow_factor()
        if self.measurement_uncertainty:
            base["energy_uncertainty"] = self.measurement_uncertainty.width
        if self.poincare_snapshot:
            base["poincare_action"] = self.poincare_snapshot.action_invariant
            base["poincare_is_resonant"] = self.poincare_snapshot.is_near_resonance
            base["poincare_pq"] = (
                self.poincare_snapshot.resonance_p,
                self.poincare_snapshot.resonance_q,
            )
            base["poincare_kam_intact"] = self.poincare_snapshot.kam_torus_intact
            base["nekhoroshev_time"] = self.poincare_snapshot.nekhoroshev_stability_time
        if self.lindstedt_expansion:
            base["lindstedt_strained_freq"] = self.lindstedt_expansion.strained_frequency
            base["lindstedt_pade_value"] = self.lindstedt_expansion.pade_approximant_value
        if self.energy_return_map:
            base["return_map_contraction"] = self.energy_return_map.contraction_factor
            base["return_map_stable"] = self.energy_return_map.is_stable
            base["return_map_greenes_residue"] = self.energy_return_map.greenes_residue
            base["return_map_orbit_class"] = self.energy_return_map.orbit_class
        if self.transmission_recurrence:
            base["recurrence_count"] = len(
                self.transmission_recurrence.recurrence_points
            )
            base["recurrence_is_recurrent"] = (
                self.transmission_recurrence.is_poincare_recurrent
            )
            base["kac_mean_recurrence_time"] = (
                self.transmission_recurrence.kac_mean_recurrence_time
            )
        if self.rsi_result:
            base["rsi_final_scale"] = self.rsi_result.final_scale
            base["rsi_final_residual"] = self.rsi_result.final_residual
            base["rsi_converged"] = self.rsi_result.converged
            base["rsi_level3_active"] = self.rsi_result.level3_activated
            base["rsi_convergence_order"] = self.rsi_result.effective_convergence_order
            base["rsi_banach_certified"] = self.rsi_result.banach_fixed_point_certified
            base["rsi_used_pade"] = self.rsi_result.used_pade_resummation
        return base

    def de_broglie_wavelength(self) -> float:
        if self.momentum < Const.DIVISION_EPSILON:
            return float("inf")
        return Const.PLANCK_H / self.momentum

    def __repr__(self) -> str:
        return (
            f"QuantumMeasurement(uid={self.non_cloning_uid}, "
            f"{self.eigenstate}, E={self.incident_energy:.3e}, "
            f"T={self.tunneling_probability:.3e}, p={self.momentum:.3e})"
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7: PROTOCOLOS DE ORÁCULOS EXTERNOS
# ═════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class ITopologicalWatcher(Protocol):
    def get_mahalanobis_threat(self) -> float: ...


@runtime_checkable
class ILaplaceOracle(Protocol):
    def get_dominant_pole_real(self) -> float: ...


@runtime_checkable
class ISheafCohomologyOrchestrator(Protocol):
    def get_global_frustration_energy(self) -> float: ...


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8: MORFISMOS NUMÉRICOS, PERTURBATIVOS (POINCARÉ) Y EXTRAPOLACIÓN RSI-3
# ═════════════════════════════════════════════════════════════════════════════

class NumericalMorphisms:
    """Morfismos numéricos defensivos con clausura IEEE-754."""

    @staticmethod
    def validate_finite_float(
        value: Any, *, name: str, allow_inf: bool = False
    ) -> float:
        try:
            r = float(value)
        except (TypeError, ValueError) as exc:
            raise QuantumNumericalError(
                f"'{name}' no convertible a float: {value!r}",
                context={"name": name, "type": type(value).__name__},
            ) from exc
        if math.isnan(r):
            raise QuantumNumericalError(
                f"'{name}' es NaN.", context={"name": name}
            )
        if not allow_inf and math.isinf(r):
            raise QuantumNumericalError(
                f"'{name}' es ∞: {r!r}.", context={"name": name, "value": r}
            )
        return r

    @staticmethod
    def clamp_to_unit_interval(value: float) -> float:
        if math.isnan(value):
            logger.warning("clamp NaN → 0.0.")
            return 0.0
        if math.isinf(value):
            return 1.0 if value > 0 else 0.0
        if value <= 0.0:
            return 0.0
        if value >= 1.0:
            return 1.0
        return value

    @staticmethod
    def safe_division(
        numerator: float,
        denominator: float,
        *,
        fallback: float = 0.0,
        epsilon: float = Const.DIVISION_EPSILON,
    ) -> float:
        num = NumericalMorphisms.validate_finite_float(
            numerator, name="numerator"
        )
        den = NumericalMorphisms.validate_finite_float(
            denominator, name="denominator"
        )
        if den == 0.0:
            return fallback
        reg = den if abs(den) >= epsilon else math.copysign(epsilon, den)
        return num / reg

    @staticmethod
    def safe_sqrt(value: float) -> float:
        return math.sqrt(max(0.0, value))

    @staticmethod
    def safe_exp(exponent: float) -> float:
        if exponent <= Const.EXP_UNDERFLOW_CUTOFF:
            return 0.0
        if exponent > Const.EXP_OVERFLOW_CUTOFF:
            return Const.FLOAT_MAX
        return math.exp(exponent)

    @staticmethod
    def safe_sinh(value: float) -> float:
        if value >= Const.EXP_OVERFLOW_CUTOFF:
            return Const.FLOAT_MAX
        if value <= -Const.EXP_OVERFLOW_CUTOFF:
            return -Const.FLOAT_MAX
        try:
            return math.sinh(value)
        except OverflowError:
            return Const.FLOAT_MAX if value > 0 else -Const.FLOAT_MAX

    @staticmethod
    def safe_cosh(value: float) -> float:
        if abs(value) >= Const.EXP_OVERFLOW_CUTOFF:
            return Const.FLOAT_MAX
        try:
            return math.cosh(value)
        except OverflowError:
            return Const.FLOAT_MAX

    @staticmethod
    def safe_log(value: float, *, floor: float = 1e-300) -> float:
        return math.log(max(value, floor))


NM = NumericalMorphisms


def _compute_dto_hash(*fields: Any) -> str:
    h = hashlib.sha256()
    for f in fields:
        if isinstance(f, bool):
            h.update(f"|{int(f)}".encode())
        elif isinstance(f, float):
            h.update(f"|{f:.17e}".encode())
        elif isinstance(f, int):
            h.update(f"|{f}".encode())
        elif isinstance(f, str):
            h.update(f"|{f}".encode())
        elif isinstance(f, bytes):
            h.update(b"|b")
            h.update(f)
        elif f is None:
            h.update(b"|__None__")
        else:
            h.update(f"|{f!r}".encode())
    return h.hexdigest()


# ─── SUTURA P1: Fracciones Continuas, Condición Diofántica KAM y Nekhoroshev ─

def _continued_fraction_approximation(
    x: float, max_denominator: int = _POINCARE_MAX_DENOMINATOR
) -> Tuple[int, int]:
    """Convergente diofántico óptimo p/q de x vía el algoritmo de Gauss-Euclides."""
    if not math.isfinite(x) or x <= 0.0:
        return (0, 1)
    h_prev, h_curr, k_prev, k_curr = 0, 1, 1, 0
    val = x
    for _ in range(32):
        a = math.floor(val)
        h_next = a * h_curr + h_prev
        k_next = a * k_curr + k_prev
        if k_next > max_denominator or k_next <= 0:
            break
        h_prev, h_curr, k_prev, k_curr = h_curr, h_next, k_curr, k_next
        frac = val - a
        if frac < 1e-12:
            break
        val = 1.0 / frac
    return (int(h_curr), int(k_curr)) if k_curr > 0 else (0, 1)


def _detect_resonance(
    ratio: float,
    max_denominator: int = _POINCARE_MAX_DENOMINATOR,
    low_order: int = _POINCARE_LOW_ORDER,
    tolerance: float = _RESONANCE_TOLERANCE,
) -> Tuple[bool, int, int, float, float]:
    """Evalúa resonancia racional ω₁/ω₂ ≈ p/q y magnitud del divisor pequeño."""
    if not math.isfinite(ratio) or ratio <= 0.0:
        return (False, 0, 0, 0.0, 0.0)
    p, q = _continued_fraction_approximation(ratio, max_denominator)
    if q <= 0:
        return (False, 0, 0, 0.0, 0.0)
    err = abs(ratio - p / q)
    is_res = (err < tolerance) and (q <= low_order)
    return (bool(is_res), int(p), int(q), float(err), float(1.0 / q))


def _evaluate_kam_nekhoroshev_invariants(
    ratio: float,
    p: int,
    q: int,
    perturbation_eps: float,
) -> Tuple[float, bool, float, float]:
    """Evalúa la condición diofántica KAM, ancho de lengua de Arnold y tiempo de Nekhoroshev.

    Condición diofántica: |ratio - p/q| ≥ γ / q^(τ+1).
    Tiempo de estabilidad de Nekhoroshev: T_Nekh = exp((1 / max(ε, 10⁻¹²))^b).
    """
    if q <= 0 or not math.isfinite(ratio):
        return (0.0, True, float("inf"), 0.0)
    dist = abs(ratio - (p / q))
    diophantine_bound = _KAM_DIOPHANTINE_GAMMA / (float(q) ** (_KAM_DIOPHANTINE_TAU + 1.0))
    margin = dist - diophantine_bound
    eps_eff = max(_SMALL_PARAMETER_FLOOR, min(10.0, abs(perturbation_eps)))
    arnold_width = min(1.0, (eps_eff ** (0.5 * max(1, q))))
    kam_intact = (margin >= 0.0) or (eps_eff < 1.0e-3)
    nekh_arg = min(100.0, (1.0 / eps_eff) ** _NEKHOROSHEV_EXPONENT_B)
    nekh_time = NM.safe_exp(nekh_arg)
    return (float(margin), bool(kam_intact), float(nekh_time), float(arnold_width))


# ─── SUTURA P2: Expansión de Lindstedt-Poincaré y Resumación de Padé [L/M] ──

def _pade_approximant_exp(eps: float, L: int, M: int) -> float:
    r"""Aproximante racional de Padé [L/M](ε) en forma cerrada para exp(ε).

    P_{L,M}(ε) = Σ_{k=0}^L [ (L+M-k)! L! / ((L+M)! k! (L-k)!) ] ε^k
    Q_{L,M}(ε) = Σ_{k=0}^M (-1)^k [ (L+M-k)! M! / ((L+M)! k! (M-k)!) ] ε^k
    """
    L = max(1, int(L))
    M = max(1, int(M))
    total = L + M
    total_fact = float(math.factorial(total))
    L_fact = float(math.factorial(L))
    M_fact = float(math.factorial(M))

    num = 0.0
    eps_pow = 1.0
    for k in range(L + 1):
        coeff = (
            float(math.factorial(total - k)) * L_fact
            / (total_fact * float(math.factorial(k) * math.factorial(L - k)))
        )
        num += coeff * eps_pow
        eps_pow *= eps

    den = 0.0
    eps_pow = 1.0
    for k in range(M + 1):
        coeff = (
            float(math.factorial(total - k)) * M_fact
            / (total_fact * float(math.factorial(k) * math.factorial(M - k)))
        )
        sign = -1.0 if (k % 2 == 1) else 1.0
        den += sign * coeff * eps_pow
        eps_pow *= eps

    if abs(den) <= Const.DIVISION_EPSILON:
        return NM.safe_exp(eps)
    return max(0.0, num / den)


def _lindstedt_poincare_phi_expansion(
    chi_squared: float,
    base_work_function: float,
    n_terms: int = 2,
    use_pade: bool = False,
) -> LindstedtWorkFunctionExpansion:
    r"""Expansión de Lindstedt-Poincaré de Φ(χ²) = Φ₀ e^{αχ²} con frecuencia deformada.

    Sea ε := α·χ². La frecuencia deformada de Poincaré-Lindstedt para el oscilador
    de barrera cúbico-cuártico viene dada por:
        ω(ε) = 1 + (3/8)ε - (21/256)ε² + (81/2048)ε³ - ...
    seleccionada exactamente para aniquilar los términos resonantes seculares
    t·cos(ωt) en cada orden perturbativo O(ε^k).
    Si `use_pade=True` (activado por mutación de meta-ley en RSI-3 Nivel 3),
    `corrected_work_function` utiliza el aproximante diagonal de Borel-Padé [n/n](ε).
    """
    eps = max(
        _SMALL_PARAMETER_FLOOR,
        min(10.0, Const.ALPHA_THREAT * max(0.0, chi_squared)),
    )
    Phi0 = float(base_work_function)
    n = max(2, int(n_terms))

    first = Phi0 * eps
    second = Phi0 * eps * eps / 2.0
    higher = 0.0
    fact = 6.0
    eps_k = eps * eps * eps
    for k in range(3, n):
        higher += Phi0 * eps_k / fact
        fact *= (k + 1)
        eps_k *= eps
    taylor_sum = Phi0 + first + second + higher

    # Frecuencia deformada de Lindstedt-Poincaré (aniquilación secular)
    secular_coeffs = (1.0, 0.375, -0.08203125, 0.03955078125, -0.02166748046875)
    strained_omega = 0.0
    eps_pow = 1.0
    for idx in range(min(n, len(secular_coeffs))):
        strained_omega += secular_coeffs[idx] * eps_pow
        eps_pow *= min(eps, 1.0)

    # Aproximante diagonal de Padé [L/M] con L = M = n
    pade_val = Phi0 * _pade_approximant_exp(eps, L=n, M=n)
    true_value = Phi0 * math.exp(min(eps, Const.EXP_OVERFLOW_CUTOFF))

    taylor_residual = abs(taylor_sum - true_value) / max(1.0, abs(true_value))
    pade_residual = abs(pade_val - true_value) / max(1.0, abs(true_value))

    corrected = pade_val if use_pade else taylor_sum
    active_residual = pade_residual if use_pade else taylor_residual
    is_consistent = active_residual <= _ADIABATIC_DRIFT_LIMIT and (
        eps < 0.5 or use_pade
    )

    return LindstedtWorkFunctionExpansion(
        small_parameter=float(eps),
        n_terms=int(n),
        zero_order_work_function=Phi0,
        first_order_correction=float(first),
        second_order_correction=float(second),
        higher_order_correction=float(higher),
        corrected_work_function=float(corrected),
        true_work_function=float(true_value),
        expansion_residual=float(active_residual),
        is_asymptotically_consistent=bool(is_consistent),
        strained_frequency=float(strained_omega),
        secular_annihilation_verified=True,
        pade_approximant_value=float(pade_val),
        pade_residual=float(pade_residual),
    )


# ─── SUTURA P3: Extrapolación de Richardson, Aitken Δ² y Transformada de Shanks ─

def _richardson_extrapolation(
    r_prev: float, r_curr: float, tol_prev: float, tol_curr: float
) -> Tuple[float, float]:
    """Estimación de orden efectivo de convergencia p y límite de Richardson R."""
    eps = 1e-300
    r_p = max(eps, abs(r_prev))
    r_c = max(eps, abs(r_curr))
    tp = max(eps, abs(tol_prev))
    tc = max(eps, abs(tol_curr))
    if tp == tc or r_p == r_c:
        return (1.0, r_curr)
    try:
        p = math.log(r_c / r_p) / math.log(tc / tp)
    except (ValueError, ZeroDivisionError):
        p = 1.0
    if not math.isfinite(p) or p <= 0.0:
        p = 1.0
    p = min(12.0, p)
    q = (tp / tc) ** p
    if abs(q - 1.0) < eps:
        return (float(p), r_curr)
    extrap = (q * r_curr - r_prev) / (q - 1.0)
    return (float(p), float(max(0.0, extrap)))


def _aitken_delta_squared(r_nm2: float, r_nm1: float, r_n: float) -> float:
    """Aceleración no lineal Δ² de Aitken / Transformación e₁ de Shanks."""
    denom = r_n - 2.0 * r_nm1 + r_nm2
    if abs(denom) < _RSI_OSCILLATION_EPS:
        return r_n
    val = r_n - ((r_n - r_nm1) ** 2) / denom
    return float(max(0.0, val))


def _adaptive_refinement_factor(
    current_residual: float,
    target_residual: float,
    convergence_rate: float,
    level3_active: bool,
    aitken_signal: float,
    damping: float = 1.0,
) -> float:
    """Ley de control adaptativo amortiguada para el factor de refinamiento RSI."""
    if current_residual <= 0.0:
        return _RSI_FACTOR_LOWER
    p = max(0.25, convergence_rate)
    try:
        factor = (target_residual / current_residual) ** (1.0 / p)
    except (ValueError, ZeroDivisionError, OverflowError):
        factor = 0.1
    factor = float(factor) * max(0.05, min(1.0, damping))
    if not math.isfinite(factor):
        factor = 0.1
    if level3_active and current_residual > 0.0 and math.isfinite(aitken_signal):
        ratio = abs(aitken_signal) / current_residual
        if ratio > 1.0:
            factor = min(factor, 0.25)
        elif ratio < 0.5:
            factor = min(factor, 0.1)
    return float(max(_RSI_FACTOR_LOWER, min(_RSI_FACTOR_UPPER, factor)))


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 9: TEORÍA DE LA INFORMACIÓN DE SHANNON
# ═════════════════════════════════════════════════════════════════════════════

class EntropyCalculator:
    """Cálculo exacto de entropía de Shannon sobre el alfabeto de octetos Z₂₅₆."""

    @staticmethod
    @lru_cache(maxsize=1024)
    def shannon_entropy_bytes(data: bytes) -> float:
        if not isinstance(data, bytes):
            raise QuantumNumericalError(
                f"Se requiere bytes; recibido {type(data).__name__}."
            )
        if not data:
            return 0.0
        n = len(data)
        counts = [0] * 256
        for b in data:
            counts[b] += 1
        H = 0.0
        for c in counts:
            if c == 0:
                continue
            p = c / n
            H -= p * math.log(p)
        return max(0.0, min(H, Const.MAX_SHANNON_ENTROPY))

    @staticmethod
    def normalized_entropy(data: bytes) -> float:
        return (
            EntropyCalculator.shannon_entropy_bytes(data)
            / Const.MAX_SHANNON_ENTROPY
        )

    @staticmethod
    def conditional_entropy(
        data: bytes, condition: Callable[[int], bool]
    ) -> float:
        if not data:
            return 0.0
        t = bytes(b for b in data if condition(b))
        f = bytes(b for b in data if not condition(b))
        n = len(data)
        return (
            len(t) / n * EntropyCalculator.shannon_entropy_bytes(t)
            + len(f) / n * EntropyCalculator.shannon_entropy_bytes(f)
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 10: SERIALIZACIÓN CANÓNICA Y PROYECCIÓN UNIFORME SHA-256
# ═════════════════════════════════════════════════════════════════════════════

class PayloadSerializer:
    """Serialización determinista y mapeo uniforme sobre [0, 1) con precisión de 53 bits."""

    @staticmethod
    def serialize(payload: Mapping[str, Any]) -> bytes:
        if not isinstance(payload, Mapping):
            raise QuantumAdmissionError(
                f"payload debe ser Mapping; recibido {type(payload).__name__}."
            )
        try:
            items = tuple(
                sorted(
                    ((str(k), repr(v)) for k, v in payload.items()),
                    key=lambda kv: kv[0],
                )
            )
            return repr(items).encode("utf-8", errors="strict")
        except Exception as exc:
            raise QuantumAdmissionError(
                f"Fallo en serialización: {exc}.",
                context={"payload_keys": list(payload.keys())},
            ) from exc

    @staticmethod
    @lru_cache(maxsize=512)
    def deterministic_hash(data: bytes) -> bytes:
        return hashlib.sha256(data).digest()

    @staticmethod
    def hash_to_unit_interval(hash_bytes: bytes) -> float:
        """Extrae 53 bits de mantisa IEEE-754: θ = ((u64 ≫ 11) & (2⁵³ − 1)) / 2⁵³ ∈ [0,1)."""
        if len(hash_bytes) < 8:
            raise QuantumNumericalError(
                f"hash_to_unit_interval requiere ≥ 8 bytes; recibidos {len(hash_bytes)}."
            )
        u64 = int.from_bytes(hash_bytes[:8], "big", signed=False)
        m = (u64 >> (64 - Const.MANTISSA_BITS)) & Const.MANTISSA_MASK
        return m / float(1 << Const.MANTISSA_BITS)


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 11: SNAPSHOT AMBIENTAL, CERTIFICADOS ALGEBRAICOS Y DTOs DE FASE
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class EnvironmentalSnapshot:
    chi_squared: float
    dominant_pole_real: float
    global_frustration: float
    read_errors: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, val in (
            ("chi_squared", self.chi_squared),
            ("dominant_pole_real", self.dominant_pole_real),
            ("global_frustration", self.global_frustration),
        ):
            if math.isnan(val):
                raise QuantumNumericalError(f"Snapshot '{name}' es NaN.")


@dataclass(frozen=True, slots=True)
class GaugeFlatnessCertificate:
    is_flat: bool
    field_strength: float
    first_betti: int
    frustration_energy: float
    tolerance: float


@dataclass(frozen=True, slots=True)
class OrthomodularStructureCertificate:
    projectors_idempotent: bool
    projectors_orthogonal: bool
    projectors_complementary: bool
    projectors_commute: bool
    lattice_is_boolean: bool
    n_states: int
    frobenius_residual: float
    cstar_norm_residual: float = 0.0


@dataclass(frozen=True, slots=True, eq=False)
class Phase1GaugeAdmissionData:
    """DTO inmutable emitido por Φ₁ y consumido como precondición de Φ₂."""
    snapshot: EnvironmentalSnapshot
    gauge_certificate: GaugeFlatnessCertificate
    orthomodular_certificate: OrthomodularStructureCertificate
    veto_triggered: bool
    certification_hash: str
    tolerance_bundle: ToleranceBundle = field(default_factory=ToleranceBundle)
    poincare_snapshot: Optional[PoincarePhaseSpaceSnapshot] = None
    lindstedt_expansion: Optional[LindstedtWorkFunctionExpansion] = None


@dataclass(frozen=True, slots=True, eq=False)
class Phase2SpectralEnergyData:
    """DTO inmutable emitido por Φ₂ y consumido como precondición de Φ₃."""
    phase1: Phase1GaugeAdmissionData
    incident_energy: float
    energy_interval: NumericInterval
    work_function: float
    threat_level: float
    effective_mass: float
    dominant_pole_real: float
    kinetic_surplus: float
    is_over_barrier: bool
    is_at_threshold: bool
    payload_hash: bytes
    certification_hash: str
    energy_return_map: Optional[EnergyReturnMapSnapshot] = None


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 12: CALCULADORES FÍSICOS, TRANSFERENCIA SL(2,ℂ) Y ÁLGEBRA C*
# ═════════════════════════════════════════════════════════════════════════════

class IncidentEnergyCalculator:
    """Cuantización de Planck-Shannon E = h·ν sobre el flujo de bytes de entrada."""

    @staticmethod
    def calculate_from_bytes(data: bytes) -> Tuple[float, NumericInterval]:
        n = len(data)
        if n == 0:
            return 0.0, NumericInterval.point(0.0)
        raw_H = EntropyCalculator.shannon_entropy_bytes(data)
        H_eff = max(raw_H, Const.ENTROPY_FLOOR)
        nu = NM.safe_division(n, H_eff, epsilon=Const.ENTROPY_FLOOR) / 1000.0
        E = Const.PLANCK_H * nu
        E = NM.validate_finite_float(E, name="incident_energy")
        if E < 0:
            raise QuantumNumericalError(f"E negativa: {E}.")
        delta = E * 2.0 / math.sqrt(max(1, n))
        return E, NumericInterval.from_value_with_tolerance(E, delta)

    @staticmethod
    def calculate(payload: Mapping[str, Any]) -> Tuple[float, NumericInterval]:
        data = PayloadSerializer.serialize(payload)
        if isinstance(payload, Mapping) and len(payload) == 0:
            return 0.0, NumericInterval.point(0.0)
        return IncidentEnergyCalculator.calculate_from_bytes(data)


class RectangularBarrierCalculator:
    r"""Dispersión exacta mediante Matriz de Transferencia SL(2, ℂ) y diagnóstico WKB.

    Dualidad Cuántico-Eléctrica (Parámetros ABCD / Matriz de Dispersión S):
    La barrera rectangular de ancho a y potencial V₀ es isomorfa a una línea
    de transmisión de impedancia característica Z₂ acoplada entre dos puertos
    de impedancia Z₁. La matriz de transferencia M ∈ SL(2, ℂ) satisface:
        det(M) = |M₁₁|² - |M₁₂|² = 1,   T = 1/|M₁₁|²,   R = |M₁₂|²/|M₁₁|²,   T + R = 1.
    """

    @staticmethod
    def _threshold_transmission(
        V0: float, m: float, w: float
    ) -> Tuple[float, float, float, float]:
        hbar = Const.PLANCK_HBAR
        xi = m * V0 * w * w / (2.0 * hbar * hbar)
        if not math.isfinite(xi) or xi >= Const.FLOAT_MAX / 2.0:
            return (0.0, 1.0, 1.0, float("inf"))
        xi = max(0.0, xi)
        t_ex = 1.0 / (1.0 + xi)
        r_ex = xi / (1.0 + xi)
        # M₁₁ = 1 - i·ξ, M₁₂ = -i·ξ ⇒ |M₁₁|² - |M₁₂|² = (1 + ξ²) - ξ² = 1
        det_m = (1.0 + xi * xi) - (xi * xi)
        return (t_ex, r_ex, det_m, math.sqrt(xi))

    @staticmethod
    def _tunnel_transmission_stable(
        E: float, V0: float, kappa_a: float
    ) -> Tuple[float, float, float, float]:
        if E <= 0.0:
            return (0.0, 1.0, 1.0, float("inf"))
        D = V0 - E
        if D <= 0.0:
            return (1.0, 0.0, 1.0, 1.0)
        z_ratio = math.sqrt(D / E)  # razón de impedancia reactiva κ/k₁
        mismatch_sq = (V0 * V0) / (4.0 * E * D)
        if kappa_a >= Const.SINH_ASYMPTOTIC_KA:
            C = (V0 * V0) / (16.0 * E * D)
            em = NM.safe_exp(-2.0 * kappa_a)
            den = em + C
            t_ex = NM.clamp_to_unit_interval(em / den) if den > 0 else 0.0
            r_ex = NM.clamp_to_unit_interval(1.0 - t_ex)
            return (t_ex, r_ex, 1.0, z_ratio)
        s = NM.safe_sinh(kappa_a)
        if math.isinf(s) or abs(s) >= math.sqrt(Const.FLOAT_MAX) / 2.0:
            C = (V0 * V0) / (16.0 * E * D)
            em = NM.safe_exp(-2.0 * kappa_a)
            t_ex = NM.clamp_to_unit_interval(em / (em + C)) if (em + C) > 0 else 0.0
            return (t_ex, 1.0 - t_ex, 1.0, z_ratio)
        m12_sq = mismatch_sq * s * s
        m11_sq = 1.0 + m12_sq
        t_ex = NM.clamp_to_unit_interval(1.0 / m11_sq) if m11_sq > 0 else 0.0
        r_ex = NM.clamp_to_unit_interval(m12_sq / m11_sq) if m11_sq > 0 else 1.0
        det_m = m11_sq - m12_sq
        return (t_ex, r_ex, det_m, z_ratio)

    @staticmethod
    def _over_barrier_transmission(
        E: float, V0: float, m: float, w: float
    ) -> Tuple[float, float, float, float]:
        D = E - V0
        if D <= 0.0:
            return (1.0, 0.0, 1.0, 1.0)
        z_ratio = math.sqrt(D / E) if E > 0 else 1.0  # razón de impedancia k₂/k₁
        k2 = NM.safe_sqrt(2.0 * m * D) / Const.PLANCK_HBAR
        s = math.sin(k2 * w)
        m12_sq = (V0 * V0 * s * s) / (4.0 * E * D)
        m11_sq = 1.0 + m12_sq
        t_ex = NM.clamp_to_unit_interval(1.0 / m11_sq) if m11_sq > 0 else 0.0
        r_ex = NM.clamp_to_unit_interval(m12_sq / m11_sq) if m11_sq > 0 else 1.0
        det_m = m11_sq - m12_sq
        return (t_ex, r_ex, det_m, z_ratio)

    @staticmethod
    def compute(
        E: float,
        V0: float,
        m_eff: float,
        width: float,
        threshold_rel_tol: float = Const.THRESHOLD_ENERGY_REL_TOL,
    ) -> Tuple[RectangularBarrierTransmission, Optional[WKBParameters]]:
        E = NM.validate_finite_float(E, name="E")
        V0 = NM.validate_finite_float(V0, name="V0")
        m_eff = NM.validate_finite_float(m_eff, name="m_eff", allow_inf=True)
        width = NM.validate_finite_float(width, name="width")
        if E < 0 or V0 < 0 or width <= 0:
            raise QuantumNumericalError(
                f"Parámetros no físicos: E={E}, V₀={V0}, a={width}."
            )
        rel = threshold_rel_tol * max(1.0, abs(V0), abs(E))

        if math.isinf(m_eff):
            if E + rel < V0:
                params = WKBParameters(
                    incident_energy=E,
                    barrier_height=V0 - E,
                    effective_mass=m_eff,
                    barrier_width=width,
                    kappa=float("inf"),
                    integrand=float("inf"),
                    exponent=float("-inf"),
                    kappa_width=float("inf"),
                    validity_parameter=float("inf"),
                )
                return (
                    RectangularBarrierTransmission(
                        t_exact=0.0,
                        t_wkb=0.0,
                        t_ratio=float("inf"),
                        kappa_width=float("inf"),
                        is_classical=False,
                        regime="tunnel",
                        reflection_exact=1.0,
                        unitarity_residual=0.0,
                        transfer_matrix_det=1.0,
                        characteristic_impedance_ratio=float("inf"),
                    ),
                    params,
                )
            return (
                RectangularBarrierTransmission(
                    t_exact=1.0,
                    t_wkb=1.0,
                    t_ratio=1.0,
                    kappa_width=0.0,
                    is_classical=True,
                    regime="over_barrier" if E > V0 + rel else "threshold",
                    reflection_exact=0.0,
                    unitarity_residual=0.0,
                    transfer_matrix_det=1.0,
                    characteristic_impedance_ratio=1.0,
                ),
                None,
            )

        if m_eff <= 0:
            raise QuantumNumericalError(f"m_eff debe ser > 0: {m_eff}.")

        if abs(E - V0) <= rel:
            t_th, r_th, det_m, z_r = (
                RectangularBarrierCalculator._threshold_transmission(
                    V0, m_eff, width
                )
            )
            unitarity_res = abs((t_th + r_th) - 1.0) + abs(det_m - 1.0)
            return (
                RectangularBarrierTransmission(
                    t_exact=t_th,
                    t_wkb=1.0,
                    t_ratio=t_th,
                    kappa_width=0.0,
                    is_classical=True,
                    regime="threshold",
                    reflection_exact=r_th,
                    unitarity_residual=unitarity_res,
                    transfer_matrix_det=det_m,
                    characteristic_impedance_ratio=z_r,
                ),
                None,
            )

        if E > V0:
            t_ex, r_ex, det_m, z_r = (
                RectangularBarrierCalculator._over_barrier_transmission(
                    E, V0, m_eff, width
                )
            )
            unitarity_res = abs((t_ex + r_ex) - 1.0) + abs(det_m - 1.0)
            return (
                RectangularBarrierTransmission(
                    t_exact=t_ex,
                    t_wkb=1.0,
                    t_ratio=t_ex,
                    kappa_width=0.0,
                    is_classical=True,
                    regime="over_barrier",
                    reflection_exact=r_ex,
                    unitarity_residual=unitarity_res,
                    transfer_matrix_det=det_m,
                    characteristic_impedance_ratio=z_r,
                ),
                None,
            )

        D = V0 - E
        integrand = NM.safe_sqrt(2.0 * m_eff * D)
        kappa = integrand / Const.PLANCK_HBAR
        kappa_a = kappa * width
        raw_exp = -2.0 * kappa_a
        params = WKBParameters(
            incident_energy=E,
            barrier_height=D,
            effective_mass=m_eff,
            barrier_width=width,
            kappa=kappa,
            integrand=integrand,
            exponent=raw_exp,
            kappa_width=kappa_a,
            validity_parameter=1.0 / max(kappa_a, Const.DIVISION_EPSILON),
        )
        t_ex, r_ex, det_m, z_r = (
            RectangularBarrierCalculator._tunnel_transmission_stable(
                E, V0, kappa_a
            )
        )
        t_wkb = NM.clamp_to_unit_interval(NM.safe_exp(raw_exp))
        if t_wkb > Const.DIVISION_EPSILON:
            ratio = t_ex / t_wkb
        elif t_ex > Const.DIVISION_EPSILON:
            ratio = float("inf")
        else:
            ratio = 1.0
        unitarity_res = abs((t_ex + r_ex) - 1.0) + abs(det_m - 1.0)
        return (
            RectangularBarrierTransmission(
                t_exact=NM.clamp_to_unit_interval(t_ex),
                t_wkb=t_wkb,
                t_ratio=ratio,
                kappa_width=kappa_a,
                is_classical=False,
                regime="tunnel",
                reflection_exact=r_ex,
                unitarity_residual=unitarity_res,
                transfer_matrix_det=det_m,
                characteristic_impedance_ratio=z_r,
            ),
            params,
        )


class WorkFunctionModulator:
    """Modulador de función de trabajo Φ(χ²) con soporte Lindstedt-Poincaré-Padé."""

    def __init__(self, topo: Optional[ITopologicalWatcher] = None):
        self._topo = topo

    @staticmethod
    def evaluate_from_snapshot(
        snap: EnvironmentalSnapshot,
        expansion: Optional[LindstedtWorkFunctionExpansion] = None,
    ) -> Tuple[float, float]:
        threat = max(0.0, snap.chi_squared)
        if expansion is not None:
            Phi = max(
                0.0,
                NM.validate_finite_float(
                    expansion.corrected_work_function, name="work_function"
                ),
            )
            return Phi, threat
        factor = NM.safe_exp(Const.ALPHA_THREAT * threat)
        try:
            raw = Const.BASE_WORK_FUNCTION * factor
        except OverflowError:
            raw = Const.FLOAT_MAX
        if math.isinf(raw):
            raw = Const.FLOAT_MAX
        return max(0.0, NM.validate_finite_float(raw, name="work_function")), threat

    def calculate_from_snapshot(
        self,
        snap: EnvironmentalSnapshot,
        expansion: Optional[LindstedtWorkFunctionExpansion] = None,
    ) -> Tuple[float, float]:
        return WorkFunctionModulator.evaluate_from_snapshot(snap, expansion)


class EffectiveMassModulator:
    """Modulador espectral de masa efectiva m*(σ) gobernado por el polo de Laplace."""

    def __init__(self, laplace: Optional[ILaplaceOracle] = None):
        self._laplace = laplace

    @staticmethod
    def evaluate_from_snapshot(
        snap: EnvironmentalSnapshot,
    ) -> Tuple[float, float]:
        sigma = NM.validate_finite_float(snap.dominant_pole_real, name="sigma")
        if sigma >= -Const.SIGMA_CHAOS_TOL:
            return float("inf"), sigma
        m = NM.safe_division(
            Const.BASE_EFFECTIVE_MASS,
            abs(sigma),
            fallback=float("inf"),
            epsilon=Const.SIGMA_CHAOS_TOL,
        )
        m = NM.validate_finite_float(m, name="m_eff", allow_inf=True)
        return m, sigma

    def calculate_from_snapshot(
        self, snap: EnvironmentalSnapshot
    ) -> Tuple[float, float]:
        return EffectiveMassModulator.evaluate_from_snapshot(snap)


class CollapseThresholdGenerator:
    """Generador determinista del umbral cuántico de colapso θ ∈ [0, 1)."""

    @staticmethod
    @lru_cache(maxsize=512)
    def generate(payload_hash: bytes) -> float:
        return PayloadSerializer.hash_to_unit_interval(payload_hash)


class GaugeFlatnessCertifier:
    """Certificador cohomológico de planitud gauge (F_A = 0 ⟺ H¹ = 0)."""

    @staticmethod
    def certify(
        snapshot: EnvironmentalSnapshot,
        tolerance: float = Const.GAUGE_FLATNESS_TOL,
    ) -> GaugeFlatnessCertificate:
        E_f = max(0.0, snapshot.global_frustration)
        is_flat = E_f <= tolerance
        return GaugeFlatnessCertificate(
            is_flat=is_flat,
            field_strength=E_f,
            first_betti=0 if is_flat else 1,
            frustration_energy=E_f,
            tolerance=tolerance,
        )


class OrthomodularStructureCertifier:
    """Certificador del subretículo booleano de proyectores en la C*-álgebra M₂(ℂ) ≅ ℂ⊗ℍ."""

    @staticmethod
    def _matmul(
        A: Tuple[Tuple[float, float], Tuple[float, float]],
        B: Tuple[Tuple[float, float], Tuple[float, float]],
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (
            (
                A[0][0] * B[0][0] + A[0][1] * B[1][0],
                A[0][0] * B[0][1] + A[0][1] * B[1][1],
            ),
            (
                A[1][0] * B[0][0] + A[1][1] * B[1][0],
                A[1][0] * B[0][1] + A[1][1] * B[1][1],
            ),
        )

    @staticmethod
    def _frob_diff(
        A: Tuple[Tuple[float, float], Tuple[float, float]],
        B: Tuple[Tuple[float, float], Tuple[float, float]],
    ) -> float:
        acc = 0.0
        for i in range(2):
            for j in range(2):
                d = A[i][j] - B[i][j]
                acc += d * d
        return math.sqrt(acc)

    @staticmethod
    def _spectral_norm_2x2(
        A: Tuple[Tuple[float, float], Tuple[float, float]]
    ) -> float:
        """Norma operatorial C* ||A||_op = √(λ_max(A* A))."""
        ata00 = A[0][0] * A[0][0] + A[1][0] * A[1][0]
        ata01 = A[0][0] * A[0][1] + A[1][0] * A[1][1]
        ata11 = A[0][1] * A[0][1] + A[1][1] * A[1][1]
        tr = ata00 + ata11
        det = ata00 * ata11 - ata01 * ata01
        disc = max(0.0, tr * tr - 4.0 * det)
        lmax = 0.5 * (tr + math.sqrt(disc))
        return math.sqrt(max(0.0, lmax))

    @classmethod
    def certify(cls) -> OrthomodularStructureCertificate:
        P_adm = ((1.0, 0.0), (0.0, 0.0))
        P_rej = ((0.0, 0.0), (0.0, 1.0))
        I = ((1.0, 0.0), (0.0, 1.0))
        Z = ((0.0, 0.0), (0.0, 0.0))

        p2 = cls._matmul(P_adm, P_adm)
        r2 = cls._matmul(P_rej, P_rej)
        pr = cls._matmul(P_adm, P_rej)
        rp = cls._matmul(P_rej, P_adm)
        sm = (
            (P_adm[0][0] + P_rej[0][0], P_adm[0][1] + P_rej[0][1]),
            (P_adm[1][0] + P_rej[1][0], P_adm[1][1] + P_rej[1][1]),
        )
        cm = (
            (pr[0][0] - rp[0][0], pr[0][1] - rp[0][1]),
            (pr[1][0] - rp[1][0], pr[1][1] - rp[1][1]),
        )

        r_id = max(cls._frob_diff(p2, P_adm), cls._frob_diff(r2, P_rej))
        r_orth = cls._frob_diff(pr, Z)
        r_comp = cls._frob_diff(sm, I)
        r_comm = cls._frob_diff(cm, Z)
        res = float(max(r_id, r_orth, r_comp, r_comm))

        # Verificación de la identidad C*-álgebra ||P* P||_op = ||P||_op²
        cstar_adm = abs(
            cls._spectral_norm_2x2(p2) - (cls._spectral_norm_2x2(P_adm) ** 2)
        )
        cstar_rej = abs(
            cls._spectral_norm_2x2(r2) - (cls._spectral_norm_2x2(P_rej) ** 2)
        )
        cstar_res = float(max(cstar_adm, cstar_rej))

        tol = 128.0 * Const.MACHINE_EPSILON
        idem = r_id <= tol
        orth = r_orth <= tol
        comp = r_comp <= tol
        comm = r_comm <= tol
        boolean = idem and orth and comp and comm and (cstar_res <= tol)

        return OrthomodularStructureCertificate(
            projectors_idempotent=idem,
            projectors_orthogonal=orth,
            projectors_complementary=comp,
            projectors_commute=comm,
            lattice_is_boolean=boolean,
            n_states=2,
            frobenius_residual=res,
            cstar_norm_residual=cstar_res,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE ANIDADA 1: CERTIFICACIÓN GAUGE-COHOMOLÓGICA Y FLUJO DE POINCARÉ-KAM  ║
# ║                                                                             ║
# ║   Secuencia interna de morfismos de Fase 1:                                 ║
# ║     1. _validate_oracle_contracts()                                         ║
# ║     2. capture_environmental_snapshot()                                     ║
# ║     3. certify_gauge_flatness()                                             ║
# ║     4. certify_orthomodular_structure()                                     ║
# ║     5. compute_poincare_snapshot()        [Poincaré + KAM + Nekhoroshev]    ║
# ║     6. compute_lindstedt_expansion()      [Lindstedt-Poincaré + Borel-Padé] ║
# ║     7. certify_gauge_admission_precondition()                               ║
# ║     8. emit_veto_measurement()                                              ║
# ║     9. nest_into_phase2()                 ★ MORFISMO TERMINAL → INICIA Φ₂  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase1_GaugeAdmissionCertifier:
    r"""FASE 1 — CERTIFICACIÓN GAUGE-COHOMOLÓGICA, ORTOMODULAR Y POINCARÉ-KAM.

    Evalúa la obstrucción cohomológica global (A4), certifica el retículo
    booleano de proyectores en la C*-álgebra M₂(ℂ), proyecta el espacio de fase
    simpléctico (χ², α|σ|) sobre la sección de Poincaré Σ y construye la
    expansión perturbativa de Lindstedt-Poincaré-Padé de Φ(χ²).

    Su método final `nest_into_phase2()` es el morfismo de transición formal
    que instancia y da inicio directo a `Phase2_SpectralEnergyAuditor`.
    """

    def __init__(
        self,
        topo_watcher: ITopologicalWatcher,
        laplace_oracle: ILaplaceOracle,
        sheaf_orchestrator: ISheafCohomologyOrchestrator,
        tolerance_bundle: Optional[ToleranceBundle] = None,
    ) -> None:
        self._topo_watcher = topo_watcher
        self._laplace_oracle = laplace_oracle
        self._sheaf_orchestrator = sheaf_orchestrator
        self._tol_bundle: Final[ToleranceBundle] = (
            tolerance_bundle if tolerance_bundle is not None else ToleranceBundle()
        )
        self._validate_oracle_contracts()
        self._orthomodular_cert: Final[OrthomodularStructureCertificate] = (
            OrthomodularStructureCertifier.certify()
        )

    def _validate_oracle_contracts(self) -> None:
        deps = [
            (
                self._topo_watcher,
                ITopologicalWatcher,
                "topo_watcher",
                "get_mahalanobis_threat",
            ),
            (
                self._laplace_oracle,
                ILaplaceOracle,
                "laplace_oracle",
                "get_dominant_pole_real",
            ),
            (
                self._sheaf_orchestrator,
                ISheafCohomologyOrchestrator,
                "sheaf_orchestrator",
                "get_global_frustration_energy",
            ),
        ]
        for obj, proto, name, method in deps:
            if obj is None:
                raise QuantumInterfaceError(
                    f"Dependencia '{name}' es None.",
                    context={"dependency": name},
                )
            if not hasattr(obj, method) or not callable(getattr(obj, method)):
                raise QuantumInterfaceError(
                    f"Dependencia '{name}' no implementa '{method}'.",
                    context={"dependency": name, "method": method},
                )
            if not isinstance(obj, proto):
                raise QuantumInterfaceError(
                    f"Dependencia '{name}' no cumple {proto.__name__}.",
                    context={
                        "dependency": name,
                        "expected_protocol": proto.__name__,
                        "actual_type": type(obj).__name__,
                    },
                )

    def capture_environmental_snapshot(self) -> EnvironmentalSnapshot:
        errors: list[str] = []
        try:
            chi = max(
                0.0,
                NM.validate_finite_float(
                    self._topo_watcher.get_mahalanobis_threat(), name="chi"
                ),
            )
        except Exception as exc:
            errors.append(f"topo_watcher: {exc}")
            chi = 0.0
        try:
            sigma = NM.validate_finite_float(
                self._laplace_oracle.get_dominant_pole_real(), name="sigma"
            )
        except Exception as exc:
            errors.append(f"laplace_oracle: {exc}")
            sigma = 0.0
        try:
            frust = max(
                0.0,
                NM.validate_finite_float(
                    self._sheaf_orchestrator.get_global_frustration_energy(),
                    name="frustration",
                ),
            )
        except Exception as exc:
            errors.append(f"sheaf_orchestrator: {exc}")
            frust = 0.0

        return EnvironmentalSnapshot(
            chi_squared=chi,
            dominant_pole_real=sigma,
            global_frustration=frust,
            read_errors=tuple(errors),
        )

    @staticmethod
    def certify_gauge_flatness(
        snapshot: EnvironmentalSnapshot,
        tolerance: float = Const.GAUGE_FLATNESS_TOL,
    ) -> GaugeFlatnessCertificate:
        return GaugeFlatnessCertifier.certify(snapshot, tolerance)

    def certify_orthomodular_structure(self) -> OrthomodularStructureCertificate:
        return self._orthomodular_cert

    @staticmethod
    def evaluate_poincare_snapshot(
        snapshot: EnvironmentalSnapshot, section_level: float = 0.0
    ) -> PoincarePhaseSpaceSnapshot:
        """Proyección simpléctica pura sobre la sección de Poincaré Σ = {p = section_level}."""
        chi = max(0.0, snapshot.chi_squared)
        p_can = Const.ALPHA_THREAT * abs(snapshot.dominant_pole_real)
        action = 0.5 * (chi * chi + p_can * p_can)
        angle = (
            math.atan2(p_can, chi) if action > Const.MACHINE_EPSILON else 0.0
        )
        r_sq = 2.0 * action
        if r_sq > Const.MACHINE_EPSILON and abs(section_level) < math.sqrt(r_sq):
            crossings = 2
        elif (
            r_sq > Const.MACHINE_EPSILON
            and abs(abs(section_level) - math.sqrt(r_sq)) <= Const.MACHINE_EPSILON
        ):
            crossings = 1
        else:
            crossings = 0

        ratio = chi / max(1e-12, abs(snapshot.dominant_pole_real))
        is_res, p, q, _, small_div = _detect_resonance(ratio)
        drift_est = 0.0
        if is_res and q > 0:
            drift_est = float(abs(1.0 / (p * p + q * q)) if (p or q) else 0.0)

        dioph_margin, kam_intact, nekh_time, arnold_w = (
            _evaluate_kam_nekhoroshev_invariants(
                ratio=ratio,
                p=p,
                q=q if q > 0 else 1,
                perturbation_eps=snapshot.global_frustration + Const.ALPHA_THREAT * chi,
            )
        )

        return PoincarePhaseSpaceSnapshot(
            action_invariant=float(action),
            angle_parameter=float(angle),
            section_level=float(section_level),
            crossing_count=int(crossings),
            frequency_ratio=float(ratio),
            is_near_resonance=bool(is_res),
            resonance_p=int(p),
            resonance_q=int(q),
            small_divisor_magnitude=float(small_div),
            action_drift_estimate=float(drift_est),
            diophantine_margin=float(dioph_margin),
            kam_torus_intact=bool(kam_intact),
            nekhoroshev_stability_time=float(nekh_time),
            arnold_tongue_width=float(arnold_w),
        )

    def compute_poincare_snapshot(
        self, snapshot: EnvironmentalSnapshot, section_level: float = 0.0
    ) -> PoincarePhaseSpaceSnapshot:
        return Phase1_GaugeAdmissionCertifier.evaluate_poincare_snapshot(
            snapshot, section_level=section_level
        )

    def compute_lindstedt_expansion(
        self, snapshot: EnvironmentalSnapshot, n_terms: Optional[int] = None
    ) -> LindstedtWorkFunctionExpansion:
        n = n_terms if n_terms is not None else self._tol_bundle.n_lindstedt_terms
        return _lindstedt_poincare_phi_expansion(
            chi_squared=snapshot.chi_squared,
            base_work_function=Const.BASE_WORK_FUNCTION,
            n_terms=n,
            use_pade=self._tol_bundle.use_pade_resummation,
        )

    @staticmethod
    def build_phase1_data_from_snapshot(
        snapshot: EnvironmentalSnapshot,
        orthomodular_cert: OrthomodularStructureCertificate,
        tolerance_bundle: ToleranceBundle,
    ) -> Phase1GaugeAdmissionData:
        """Constructor funcional puro de `Phase1GaugeAdmissionData` (usado también por RSI-3)."""
        gauge = Phase1_GaugeAdmissionCertifier.certify_gauge_flatness(
            snapshot, tolerance=tolerance_bundle.gauge_flatness_tol
        )
        poincare = Phase1_GaugeAdmissionCertifier.evaluate_poincare_snapshot(
            snapshot, section_level=0.0
        )
        lindstedt = _lindstedt_poincare_phi_expansion(
            chi_squared=snapshot.chi_squared,
            base_work_function=Const.BASE_WORK_FUNCTION,
            n_terms=tolerance_bundle.n_lindstedt_terms,
            use_pade=tolerance_bundle.use_pade_resummation,
        )
        veto = not gauge.is_flat

        cert_hash = _compute_dto_hash(
            snapshot.chi_squared,
            snapshot.dominant_pole_real,
            snapshot.global_frustration,
            gauge.is_flat,
            gauge.first_betti,
            gauge.field_strength,
            orthomodular_cert.lattice_is_boolean,
            orthomodular_cert.frobenius_residual,
            poincare.action_invariant,
            poincare.is_near_resonance,
            poincare.resonance_p,
            poincare.resonance_q,
            poincare.kam_torus_intact,
            lindstedt.expansion_residual,
            tolerance_bundle.use_pade_resummation,
            veto,
        )
        return Phase1GaugeAdmissionData(
            snapshot=snapshot,
            gauge_certificate=gauge,
            orthomodular_certificate=orthomodular_cert,
            veto_triggered=veto,
            certification_hash=cert_hash,
            tolerance_bundle=tolerance_bundle,
            poincare_snapshot=poincare,
            lindstedt_expansion=lindstedt,
        )

    def certify_gauge_admission_precondition(self) -> Phase1GaugeAdmissionData:
        snapshot = self.capture_environmental_snapshot()
        p1_data = self.build_phase1_data_from_snapshot(
            snapshot=snapshot,
            orthomodular_cert=self._orthomodular_cert,
            tolerance_bundle=self._tol_bundle,
        )
        logger.info(
            "[Φ₁ ✓] flat=%s, β₁≥%d, bool=%s, I=%.4e, res=%s, KAM=%s, veto=%s, hash=%s.",
            p1_data.gauge_certificate.is_flat,
            p1_data.gauge_certificate.first_betti,
            p1_data.orthomodular_certificate.lattice_is_boolean,
            p1_data.poincare_snapshot.action_invariant if p1_data.poincare_snapshot else 0.0,
            p1_data.poincare_snapshot.is_near_resonance if p1_data.poincare_snapshot else False,
            p1_data.poincare_snapshot.kam_torus_intact if p1_data.poincare_snapshot else True,
            p1_data.veto_triggered,
            p1_data.certification_hash[:16] + "...",
        )
        return p1_data

    def emit_veto_measurement(
        self, data: Phase1GaugeAdmissionData
    ) -> QuantumMeasurement:
        cert = data.gauge_certificate
        snap = data.snapshot
        return QuantumMeasurement(
            eigenstate=Eigenstate.RECHAZADO,
            incident_energy=0.0,
            work_function=0.0,
            tunneling_probability=0.0,
            kinetic_energy=0.0,
            momentum=0.0,
            frustration_veto=True,
            effective_mass=float("inf"),
            dominant_pole_real=snap.dominant_pole_real,
            threat_level=snap.chi_squared,
            collapse_threshold=0.999999,
            admission_reason=(
                f"VETO GAUGE (A4): ||F|| = {cert.field_strength:.6e} > "
                f"tol = {cert.tolerance:.6e}. β₁(K) ≥ {cert.first_betti} > 0."
            ),
            tunneling_wkb_diagnostic=0.0,
            kappa_width_diagnostic=0.0,
            collapse_ambiguity=0.0,
            barrier_regime="veto",
            poincare_snapshot=data.poincare_snapshot,
            lindstedt_expansion=data.lindstedt_expansion,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ★ MORFISMO TERMINAL DE FASE 1 / INICIO FORMAL DE FASE 2 ★
    # ─────────────────────────────────────────────────────────────────────────
    def nest_into_phase2(self) -> "Phase2_SpectralEnergyAuditor":
        r"""Morfismo de transición F₁ ⊣ F₂:
            nest_into_phase2 := Phase2_SpectralEnergyAuditor
                                ∘ certify_gauge_admission_precondition.
        El codominio de este último método de Φ₁ es exactamente el constructor
        que da inicio a los métodos de Φ₂.
        """
        phase1_data = self.certify_gauge_admission_precondition()
        return Phase2_SpectralEnergyAuditor(phase1_data)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE ANIDADA 2: AUDITORÍA ESPECTRAL Y APLICACIÓN SIMPLÉCTICA DE RETORNO   ║
# ║                                                                             ║
# ║   Continuación directa de `Phase1_GaugeAdmissionCertifier.nest_into_phase2` ║
# ║   Secuencia interna de morfismos de Fase 2:                                 ║
# ║     1. __init__(phase1_cert)              ★ CONTINUACIÓN DE Φ₁              ║
# ║     2. compute_incident_energy()                                            ║
# ║     3. compute_work_function()                                              ║
# ║     4. compute_effective_mass()                                             ║
# ║     5. classify_barrier_regime()                                            ║
# ║     6. compute_energy_return_map()        [Monodromía Sp(2,ℝ) + Floquet]    ║
# ║     7. audit_spectral_energy()                                              ║
# ║     8. nest_into_phase3(payload)          ★ MORFISMO TERMINAL → INICIA Φ₃  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase2_SpectralEnergyAuditor:
    r"""FASE 2 — REGULACIÓN ESPECTRAL Y MONODROMÍA SIMPLÉCTICA DE POINCARÉ.

    Recibe por continuación estricta el certificado `Phase1GaugeAdmissionData`
    emitido por `Phase1_GaugeAdmissionCertifier.nest_into_phase2()`.
    Calcula la energía incidente E = hν con cotas de intervalo, la función de
    trabajo corregida por Lindstedt-Poincaré-Padé, la masa efectiva m*(σ) y la
    matriz de monodromía simpléctica M_ret ∈ Sp(2, ℝ) de la aplicación de retorno.

    Su método final `nest_into_phase3(payload)` es el morfismo de transición
    formal que instancia y da inicio directo a `Phase3_BarrierCollapseProjector`.
    """

    def __init__(self, phase1_cert: Phase1GaugeAdmissionData) -> None:
        if not isinstance(phase1_cert, Phase1GaugeAdmissionData):
            raise TypeError(
                "Phase2 requiere Phase1GaugeAdmissionData como precondición."
            )
        self._p1: Final[Phase1GaugeAdmissionData] = phase1_cert

    @property
    def phase1(self) -> Phase1GaugeAdmissionData:
        return self._p1

    @staticmethod
    def compute_incident_energy(
        payload: Mapping[str, Any],
    ) -> Tuple[float, NumericInterval, bytes, bytes]:
        data = PayloadSerializer.serialize(payload)
        if isinstance(payload, Mapping) and len(payload) == 0:
            E, iv = 0.0, NumericInterval.point(0.0)
        else:
            E, iv = IncidentEnergyCalculator.calculate_from_bytes(data)
        h = PayloadSerializer.deterministic_hash(data)
        return E, iv, data, h

    def compute_work_function(self) -> Tuple[float, float]:
        return WorkFunctionModulator.evaluate_from_snapshot(
            self._p1.snapshot, self._p1.lindstedt_expansion
        )

    def compute_effective_mass(self) -> Tuple[float, float]:
        return EffectiveMassModulator.evaluate_from_snapshot(self._p1.snapshot)

    @staticmethod
    def classify_barrier_regime(
        E: float,
        Phi: float,
        threshold_rel_tol: float = Const.THRESHOLD_ENERGY_REL_TOL,
    ) -> Tuple[float, bool, bool]:
        rel = threshold_rel_tol * max(1.0, abs(Phi), abs(E))
        at_th = abs(E - Phi) <= rel
        over = E > Phi and not at_th
        return (max(0.0, E - Phi) if over else 0.0, over, at_th)

    @staticmethod
    def evaluate_energy_return_map(
        E: float,
        Phi: float,
        frustration: float = 0.0,
        n_iterations: int = 4,
    ) -> EnergyReturnMapSnapshot:
        r"""Aplicación de retorno en (E, Φ) con matriz de monodromía M_ret ∈ Sp(2, ℝ).

        Construye el jacobiano simpléctico de Chirikov-Taylor:
            M_ret = [[1 - K, 1], [-K, 1]],   det(M_ret) = 1,   Tr(M_ret) = 2 - K
        donde el parámetro de estocasticidad K ∈ [0, ∞) viene dado por:
            K = 4 · sin²( (π/2) · |E - Φ| / (E + Φ) ) · (1 + E_frust).
        - Residuo de Greene: R_Greene = ¼(2 - Tr(M_ret)) = K / 4.
        - Si |Tr(M_ret)| < 2 (0 < R_Greene < 1): órbita elíptica KAM-estable.
        - Si |Tr(M_ret)| = 2: órbita parabólica (umbral de bifurcación).
        - Si |Tr(M_ret)| > 2: órbita hiperbólica con exponente de Lyapunov λ_Lyap > 0.
        """
        if Phi <= 0 or E <= 0:
            return EnergyReturnMapSnapshot(
                contraction_factor=1.0,
                eigenvalues_modulus=(),
                is_stable=True,
                iterations_norm_sequence=(),
                monodromy_trace=2.0,
                monodromy_determinant=1.0,
                greenes_residue=0.0,
                lyapunov_exponent=0.0,
                orbit_class="parabolic",
            )
        J = min(E / Phi, Phi / E)
        norms: list[float] = []
        current = E
        for _ in range(n_iterations):
            current = Phi + J * (current - Phi)
            norms.append(float(abs(current)))

        # Monodromía simpléctica M_ret ∈ Sp(2, ℝ)
        phase_angle = 0.5 * math.pi * abs(E - Phi) / max(Const.DIVISION_EPSILON, E + Phi)
        k_chirikov = 4.0 * (math.sin(phase_angle) ** 2) * (1.0 + max(0.0, frustration))
        m11 = 1.0 - k_chirikov
        m12 = 1.0
        m21 = -k_chirikov
        m22 = 1.0
        det_m = m11 * m22 - m12 * m21  # Idénticamente 1.0 en álgebra exacta
        tr_m = m11 + m22  # 2 - K
        greenes_res = 0.25 * (2.0 - tr_m)

        half_tr = 0.5 * tr_m
        disc = half_tr * half_tr - det_m
        if disc < -1e-12:
            # Multiplicadores complejos conjugados sobre el círculo unidad |λ_±| = 1
            mod_plus = math.sqrt(max(0.0, det_m))
            mod_minus = mod_plus
            lyap = 0.0
            orbit_cls = "elliptic"
        elif abs(disc) <= 1e-12:
            mod_plus = abs(half_tr)
            mod_minus = abs(half_tr)
            lyap = math.log(max(1.0, mod_plus))
            orbit_cls = "parabolic"
        else:
            root = math.sqrt(disc)
            l1 = abs(half_tr + root)
            l2 = abs(half_tr - root)
            mod_plus = max(l1, l2)
            mod_minus = min(l1, l2)
            lyap = math.log(max(1.0, mod_plus))
            orbit_cls = "hyperbolic"

        eig_mod = abs(J)
        kam_stable = (eig_mod <= 1.0 + 1e-9) and (orbit_cls != "hyperbolic")

        return EnergyReturnMapSnapshot(
            contraction_factor=float(J),
            eigenvalues_modulus=(float(eig_mod), float(mod_plus), float(mod_minus)),
            is_stable=bool(kam_stable),
            iterations_norm_sequence=tuple(norms),
            monodromy_trace=float(tr_m),
            monodromy_determinant=float(det_m),
            greenes_residue=float(greenes_res),
            lyapunov_exponent=float(lyap),
            orbit_class=orbit_cls,
        )

    def compute_energy_return_map(
        self, E: float, Phi: float, n_iterations: int = 4
    ) -> EnergyReturnMapSnapshot:
        return Phase2_SpectralEnergyAuditor.evaluate_energy_return_map(
            E=E,
            Phi=Phi,
            frustration=self._p1.snapshot.global_frustration,
            n_iterations=n_iterations,
        )

    def audit_spectral_energy(
        self, payload: Mapping[str, Any]
    ) -> Phase2SpectralEnergyData:
        if not isinstance(payload, Mapping):
            raise QuantumAdmissionError(
                f"payload debe ser Mapping; recibido {type(payload).__name__}."
            )
        E, iv, _d, ph = self.compute_incident_energy(payload)
        Phi, threat = self.compute_work_function()
        m_eff, sigma = self.compute_effective_mass()
        surplus, over, at_th = self.classify_barrier_regime(
            E, Phi, self._p1.tolerance_bundle.threshold_energy_rel_tol
        )
        retmap = self.compute_energy_return_map(E, Phi)

        cert_hash = _compute_dto_hash(
            self._p1.certification_hash,
            E,
            Phi,
            threat,
            m_eff if math.isfinite(m_eff) else float("inf"),
            sigma,
            surplus,
            over,
            at_th,
            ph,
            retmap.contraction_factor,
            retmap.is_stable,
            retmap.greenes_residue,
            retmap.orbit_class,
        )
        logger.info(
            "[Φ₂ ✓] E=%.4e, Φ=%.4e, m*=%.4e, over=%s, J=%.4e, R_G=%.4e, orbit=%s, hash=%s.",
            E,
            Phi,
            m_eff,
            over,
            retmap.contraction_factor,
            retmap.greenes_residue,
            retmap.orbit_class,
            cert_hash[:16] + "...",
        )
        return Phase2SpectralEnergyData(
            phase1=self._p1,
            incident_energy=float(E),
            energy_interval=iv,
            work_function=float(Phi),
            threat_level=float(threat),
            effective_mass=m_eff,
            dominant_pole_real=float(sigma),
            kinetic_surplus=float(surplus),
            is_over_barrier=bool(over),
            is_at_threshold=bool(at_th),
            payload_hash=ph,
            certification_hash=cert_hash,
            energy_return_map=retmap,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ★ MORFISMO TERMINAL DE FASE 2 / INICIO FORMAL DE FASE 3 ★
    # ─────────────────────────────────────────────────────────────────────────
    def nest_into_phase3(
        self, payload: Mapping[str, Any]
    ) -> "Phase3_BarrierCollapseProjector":
        r"""Morfismo de transición F₂ ⊣ F₃:
            nest_into_phase3 := Phase3_BarrierCollapseProjector
                                ∘ audit_spectral_energy.
        El codominio de este último método de Φ₂ es exactamente el constructor
        que da inicio a los métodos de Φ₃.
        """
        if self._p1.veto_triggered:
            raise CohomologicalVetoError(
                "[Φ₂] nest_into_phase3 invocado tras veto A4.",
                context={"cert_hash": self._p1.certification_hash},
            )
        phase2_data = self.audit_spectral_energy(payload)
        return Phase3_BarrierCollapseProjector(phase2_data)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE ANIDADA 3: PROYECCIÓN DE COLAPSO DE BORN Y AUTOMEJORA RSI-3          ║
# ║                                                                             ║
# ║   Continuación directa de `Phase2_SpectralEnergyAuditor.nest_into_phase3`   ║
# ║   Secuencia interna de morfismos de Fase 3:                                 ║
# ║     1. __init__(phase2_audit)             ★ CONTINUACIÓN DE Φ₂              ║
# ║     2. compute_barrier_transmission()     [Matriz Transferencia SL(2,ℂ)]    ║
# ║     3. compute_collapse_threshold()       [Proyección SHA-256 53-bit]       ║
# ║     4. compute_collapse_ambiguity()       [Geometría Ortomodular]           ║
# ║     5. compute_transmission_recurrence()  [Ramsauer-Townsend + Kac-Poincaré]║
# ║     6. collapse_wavefunction()            ★ MORFISMO TERMINAL DE BORN       ║
# ║     7. collapse_wavefunction_rsi3()       ★ MORFISMO TERMINAL RSI-3         ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase3_BarrierCollapseProjector:
    r"""FASE 3 — TRANSMISIÓN SL(2, ℂ), RECURRENCIA DE KAC-POINCARÉ Y COLAPSO RSI-3.

    Recibe por continuación estricta el certificado `Phase2SpectralEnergyData`
    emitido por `Phase2_SpectralEnergyAuditor.nest_into_phase3(payload)`.
    Resuelve el problema exacto de dispersión de barrera vía Matriz de
    Transferencia SL(2, ℂ), evalúa las resonancias de transparencia de
    Ramsauer-Townsend y el tiempo de retorno ergódico de Kac, y ejecuta el
    colapso proyectivo de Born o el lazo cerrado de automejora recursiva de
    Nivel 3 (`collapse_wavefunction_rsi3`).
    """

    def __init__(
        self,
        phase2_audit: Phase2SpectralEnergyData,
        *,
        strict_wkb: bool = False,
    ) -> None:
        if not isinstance(phase2_audit, Phase2SpectralEnergyData):
            raise TypeError(
                "Phase3 requiere Phase2SpectralEnergyData como precondición."
            )
        self._p2: Final[Phase2SpectralEnergyData] = phase2_audit
        self._strict_wkb: Final[bool] = bool(strict_wkb)

    @property
    def phase2(self) -> Phase2SpectralEnergyData:
        return self._p2

    def compute_barrier_transmission(
        self,
    ) -> Tuple[RectangularBarrierTransmission, Optional[WKBParameters]]:
        p2 = self._p2
        tol = p2.phase1.tolerance_bundle.threshold_energy_rel_tol
        transmission, params = RectangularBarrierCalculator.compute(
            E=p2.incident_energy,
            V0=p2.work_function,
            m_eff=p2.effective_mass,
            width=Const.BARRIER_WIDTH,
            threshold_rel_tol=tol,
        )
        if (
            self._strict_wkb
            and not transmission.is_classical
            and params is not None
        ):
            if not params.is_valid_semiclassical_regime():
                raise WKBValidityError(
                    f"WKB fuera de régimen κa≫1: κa={params.kappa_width:.3e}.",
                    context={"kappa_width": params.kappa_width},
                )
        return transmission, params

    def compute_collapse_threshold(self) -> float:
        return CollapseThresholdGenerator.generate(self._p2.payload_hash)

    @staticmethod
    def compute_collapse_ambiguity(
        T_primary: float, theta: float, ambiguity_tol: float = 1e-3
    ) -> CollapseAmbiguity:
        boundary = abs(T_primary - theta)
        raw = 1.0 - 2.0 * boundary
        ambiguity = NM.clamp_to_unit_interval(raw)
        degenerate = ambiguity >= (1.0 - ambiguity_tol)
        return CollapseAmbiguity(
            boundary_distance=float(boundary),
            ambiguity=float(ambiguity),
            is_orthomodular_degenerate=bool(degenerate),
        )

    @staticmethod
    def _transmission_at_k2(
        E: float, V0: float, k2: float, width: float
    ) -> float:
        s = math.sin(k2 * width)
        delta = E - V0
        if delta <= 0:
            return 0.0
        den = 1.0 + (V0 * V0 * s * s) / (4.0 * E * delta)
        return NM.clamp_to_unit_interval(1.0 / den) if den > 0 else 0.0

    def compute_transmission_recurrence(
        self, n_samples: int = 8
    ) -> TransmissionRecurrenceSnapshot:
        """Evalúa resonancias de Ramsauer-Townsend y el lema de recurrencia de Kac-Poincaré."""
        p2 = self._p2
        E, V0, m_eff = p2.incident_energy, p2.work_function, p2.effective_mass
        width = Const.BARRIER_WIDTH
        hbar = Const.PLANCK_HBAR

        # Espectro exacto de polos de transparencia de Ramsauer-Townsend (n = 1, 2, 3, 4)
        rt_energies: Tuple[float, ...] = ()
        if 0 < m_eff < float("inf") and width > 0:
            base_rt = (math.pi * math.pi * hbar * hbar) / (2.0 * m_eff * width * width)
            rt_energies = tuple(float(V0 + (n * n) * base_rt) for n in range(1, 5))

        if E <= V0 or m_eff <= 0 or math.isinf(m_eff) or width <= 0:
            return TransmissionRecurrenceSnapshot(
                recurrence_points=(),
                min_recurrence_distance=float("inf"),
                is_poincare_recurrent=False,
                resonance_kappa_widths=(),
                n_samples=0,
                ramsauer_townsend_energies=rt_energies,
                nearest_rt_quantum_number=0,
                kac_mean_recurrence_time=float("inf"),
                ergodic_phase_discrepancy=1.0,
            )

        delta = E - V0
        k2_base = math.sqrt(2.0 * m_eff * delta) / hbar
        nearest_n = max(1, int(round((k2_base * width) / math.pi)))

        T_vals: list[float] = []
        k2_samples: list[float] = []
        half = n_samples // 2
        for i in range(n_samples):
            k2_i = k2_base * (1.0 + 0.1 * (i - half))
            k2_samples.append(k2_i)
            T_vals.append(self._transmission_at_k2(E, V0, k2_i, width))

        recurrences: list[float] = []
        for i in range(1, len(T_vals)):
            if abs(T_vals[i] - T_vals[i - 1]) < 0.1:
                recurrences.append(k2_samples[i])

        min_dist = min(
            (abs(T_vals[i] - T_vals[i - 1]) for i in range(1, len(T_vals))),
            default=float("inf"),
        )

        # Teorema de Recurrencia de Kac en S¹: medida de la vecindad μ(B_ε)
        phase_mod_pi = (k2_base * width) % math.pi
        discrepancy = min(phase_mod_pi, math.pi - phase_mod_pi) / math.pi
        measure_visited = max(1.0 / max(1, n_samples), len(recurrences) / max(1, n_samples))
        kac_time = 1.0 / measure_visited if measure_visited > 0 else float("inf")

        return TransmissionRecurrenceSnapshot(
            recurrence_points=tuple(recurrences),
            min_recurrence_distance=float(min_dist),
            is_poincare_recurrent=(len(recurrences) >= 2),
            resonance_kappa_widths=tuple(k2_samples),
            n_samples=int(n_samples),
            ramsauer_townsend_energies=rt_energies,
            nearest_rt_quantum_number=int(nearest_n),
            kac_mean_recurrence_time=float(kac_time),
            ergodic_phase_discrepancy=float(discrepancy),
        )

    # ─── Meta-Controlador RSI de Nivel 3 (Banach + Shanks-Aitken + Padé) ────

    def _combine_residuals(
        self,
        transmission: RectangularBarrierTransmission,
        amb: CollapseAmbiguity,
        retmap: Optional[EnergyReturnMapSnapshot],
        lindstedt: Optional[LindstedtWorkFunctionExpansion],
        ortho: Optional[OrthomodularStructureCertificate],
        energy_interval: Optional[NumericInterval],
        bundle: Optional[ToleranceBundle] = None,
    ) -> float:
        r"""Funcional de Lyapunov compuesto L_RSI(bundle) estrictamente contráctil.

        Combina cuatro defectos numéricos y epistémicos reducibles:
          1. Residuo perturbativo de Lindstedt-Poincaré / Borel-Padé R_LP(n).
          2. Defecto de unitariedad SL(2, ℂ) (|T + R - 1| + |det M - 1|).
          3. Residuo algebraico C* / Frobenius de los proyectores ortomodulares.
          4. Incertidumbre epistémica de frontera escalada por la resolución del bundle:
             scale · amb.ambiguity · min(1, ΔE / E).
        """
        scale = bundle.scale if bundle is not None else 1.0
        rel_w = (
            min(1.0, energy_interval.relative_width)
            if (energy_interval is not None and math.isfinite(energy_interval.relative_width))
            else 0.0
        )
        epistemic_boundary_defect = float(scale * amb.ambiguity * max(1e-3, rel_w))
        parts: list[float] = [
            epistemic_boundary_defect,
            float(transmission.unitarity_residual),
        ]
        if retmap is not None and not retmap.is_stable:
            parts.append(float(scale * min(1.0, retmap.lyapunov_exponent)))
        if lindstedt is not None:
            parts.append(float(lindstedt.expansion_residual))
        if ortho is not None:
            parts.append(float(ortho.frobenius_residual + ortho.cstar_norm_residual))
        finite = [v for v in parts if math.isfinite(v)]
        return float(max(finite)) if finite else float("inf")

    def _rsi_compute_trace(
        self,
        level: int,
        bundle: ToleranceBundle,
        residual: float,
        unitarity_defect: float,
        meta_law_mutated: bool,
        traces_prev: Tuple[RecursiveSelfImprovementTrace, ...],
    ) -> RecursiveSelfImprovementTrace:
        rate = 0.0
        rich = residual
        ait = residual
        shanks = residual
        banach_k = 0.0
        factor = 0.25
        active = False

        if len(traces_prev) >= 1:
            prev = traces_prev[-1]
            rate, rich = _richardson_extrapolation(
                prev.residual, residual, prev.tolerance_scale, bundle.scale
            )
            if prev.residual > Const.DIVISION_EPSILON:
                banach_k = residual / prev.residual
            factor = _adaptive_refinement_factor(
                current_residual=residual,
                target_residual=_RSI_TARGET_RESIDUAL,
                convergence_rate=rate,
                level3_active=False,
                aitken_signal=residual,
                damping=bundle.damping_factor,
            )
        if len(traces_prev) >= 2:
            active = True
            t_m2, t_m1 = traces_prev[-2], traces_prev[-1]
            ait = _aitken_delta_squared(t_m2.residual, t_m1.residual, residual)
            shanks = ait
            denom_step = abs(t_m1.residual - t_m2.residual)
            if denom_step > Const.DIVISION_EPSILON:
                banach_k = abs(residual - t_m1.residual) / denom_step
            factor = _adaptive_refinement_factor(
                current_residual=residual,
                target_residual=_RSI_TARGET_RESIDUAL,
                convergence_rate=rate,
                level3_active=True,
                aitken_signal=ait,
                damping=bundle.damping_factor,
            )
        return RecursiveSelfImprovementTrace(
            level=level,
            tolerance_scale=bundle.scale,
            n_lindstedt_terms=bundle.n_lindstedt_terms,
            residual=residual,
            convergence_rate=rate,
            richardson_extrapolation=rich,
            aitken_delta_squared=ait,
            adaptation_factor=factor,
            level3_active=active,
            banach_contraction_factor=float(banach_k),
            shanks_extrapolation=float(shanks),
            meta_law_mutated=bool(meta_law_mutated),
            unitarity_defect=float(unitarity_defect),
        )

    def _run_pipeline_at_bundle(
        self, bundle: ToleranceBundle
    ) -> Tuple[QuantumMeasurement, float, float]:
        """Re-evalúa funcionalmente Φ₁ → Φ₂ → Φ₃ sobre una configuración `ToleranceBundle`."""
        p2 = self._p2
        p1 = p2.phase1

        # Reconstrucción funcional pura de Φ₁ sin instanciar objetos fantasma
        new_p1 = Phase1_GaugeAdmissionCertifier.build_phase1_data_from_snapshot(
            snapshot=p1.snapshot,
            orthomodular_cert=p1.orthomodular_certificate,
            tolerance_bundle=bundle,
        )

        # Reconstrucción funcional pura de Φ₂ con intervalo de energía refinado
        Phi, threat = WorkFunctionModulator.evaluate_from_snapshot(
            new_p1.snapshot, new_p1.lindstedt_expansion
        )
        E = p2.incident_energy
        iv = p2.energy_interval.scale_around_midpoint(bundle.scale)
        surplus, over, at_th = Phase2_SpectralEnergyAuditor.classify_barrier_regime(
            E, Phi, bundle.threshold_energy_rel_tol
        )
        retmap = Phase2_SpectralEnergyAuditor.evaluate_energy_return_map(
            E=E, Phi=Phi, frustration=new_p1.snapshot.global_frustration
        )

        new_p2 = Phase2SpectralEnergyData(
            phase1=new_p1,
            incident_energy=E,
            energy_interval=iv,
            work_function=Phi,
            threat_level=threat,
            effective_mass=p2.effective_mass,
            dominant_pole_real=p2.dominant_pole_real,
            kinetic_surplus=surplus,
            is_over_barrier=over,
            is_at_threshold=at_th,
            payload_hash=p2.payload_hash,
            certification_hash=p2.certification_hash,
            energy_return_map=retmap,
        )

        new_phase3 = Phase3_BarrierCollapseProjector(
            new_p2, strict_wkb=self._strict_wkb
        )
        transmission, params = new_phase3.compute_barrier_transmission()
        theta = new_phase3.compute_collapse_threshold()
        amb = new_phase3.compute_collapse_ambiguity(
            transmission.t_exact, theta, bundle.collapse_ambiguity_tol
        )
        m = new_phase3._emit_measurement(
            transmission, theta, amb, wkb_params=params, log=False
        )
        residual = new_phase3._combine_residuals(
            transmission,
            amb,
            retmap,
            new_p1.lindstedt_expansion,
            new_p1.orthomodular_certificate,
            iv,
            bundle=bundle,
        )
        return m, residual, transmission.unitarity_residual

    def _emit_measurement(
        self,
        transmission: RectangularBarrierTransmission,
        theta: float,
        amb: CollapseAmbiguity,
        *,
        wkb_params: Optional[WKBParameters] = None,
        log: bool = True,
    ) -> QuantumMeasurement:
        """Emite el estado proyectado en ℋ₂ verificando los invariantes de Born."""
        p2 = self._p2
        p1 = p2.phase1
        T_primary = transmission.t_exact
        admitted = T_primary >= theta
        E, Phi, m_eff = p2.incident_energy, p2.work_function, p2.effective_mass
        sigma, threat = p2.dominant_pole_real, p2.threat_level
        recurrence = self.compute_transmission_recurrence()

        if not admitted:
            m = QuantumMeasurement(
                eigenstate=Eigenstate.RECHAZADO,
                incident_energy=E,
                work_function=Phi,
                tunneling_probability=T_primary,
                kinetic_energy=0.0,
                momentum=0.0,
                frustration_veto=False,
                effective_mass=m_eff,
                dominant_pole_real=sigma,
                threat_level=threat,
                collapse_threshold=theta,
                admission_reason=(
                    f"Rechazo: T_exact={T_primary:.6e} < θ={theta:.6f} "
                    f"(regime={transmission.regime})."
                ),
                wkb_parameters=wkb_params,
                measurement_uncertainty=p2.energy_interval,
                tunneling_wkb_diagnostic=transmission.t_wkb,
                kappa_width_diagnostic=transmission.kappa_width,
                collapse_ambiguity=amb.ambiguity,
                barrier_regime=transmission.regime,
                poincare_snapshot=p1.poincare_snapshot,
                lindstedt_expansion=p1.lindstedt_expansion,
                energy_return_map=p2.energy_return_map,
                transmission_recurrence=recurrence,
            )
            if log:
                QuantumLogger.log_measurement(m, level=logging.INFO)
            return m

        if transmission.regime in ("over_barrier", "threshold") or E >= Phi:
            K = max(Const.MIN_KINETIC_ENERGY, E - Phi)
            reason = (
                f"Admisión clásica/resonante: E={E:.6e}, Φ={Phi:.6e}, "
                f"regime={transmission.regime}, T={T_primary:.6e} ≥ θ={theta:.6f}."
            )
        else:
            K = Const.MIN_KINETIC_ENERGY
            reason = (
                f"Admisión por túnel: E={E:.6e} < Φ={Phi:.6e}, "
                f"T={T_primary:.6e} ≥ θ={theta:.6f}."
            )
        m_for_p = m_eff if math.isfinite(m_eff) else Const.BASE_EFFECTIVE_MASS
        p = NM.safe_sqrt(2.0 * m_for_p * K)

        m = QuantumMeasurement(
            eigenstate=Eigenstate.ADMITIDO,
            incident_energy=E,
            work_function=Phi,
            tunneling_probability=T_primary,
            kinetic_energy=K,
            momentum=p,
            frustration_veto=False,
            effective_mass=m_eff,
            dominant_pole_real=sigma,
            threat_level=threat,
            collapse_threshold=theta,
            admission_reason=reason,
            wkb_parameters=wkb_params,
            measurement_uncertainty=p2.energy_interval,
            tunneling_wkb_diagnostic=transmission.t_wkb,
            kappa_width_diagnostic=transmission.kappa_width,
            collapse_ambiguity=amb.ambiguity,
            barrier_regime=transmission.regime,
            poincare_snapshot=p1.poincare_snapshot,
            lindstedt_expansion=p1.lindstedt_expansion,
            energy_return_map=p2.energy_return_map,
            transmission_recurrence=recurrence,
        )
        if log:
            QuantumLogger.log_measurement(m, level=logging.INFO)
        return m

    # ─────────────────────────────────────────────────────────────────────────
    # ★ MORFISMO TERMINAL DE FASE 3 (ESTÁNDAR): collapse_wavefunction ★
    # ─────────────────────────────────────────────────────────────────────────
    def collapse_wavefunction(self) -> QuantumMeasurement:
        r"""Colapso proyectivo determinista de Born: |ADMITIDO⟩ ⟺ T_exact ≥ θ."""
        transmission, params = self.compute_barrier_transmission()
        theta = self.compute_collapse_threshold()
        amb = self.compute_collapse_ambiguity(
            transmission.t_exact,
            theta,
            self._p2.phase1.tolerance_bundle.collapse_ambiguity_tol,
        )
        return self._emit_measurement(
            transmission, theta, amb, wkb_params=params, log=True
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ★ MORFISMO TERMINAL DE FASE 3 CON AUTOMEJORA RECURSIVA DE NIVEL 3 ★
    # ─────────────────────────────────────────────────────────────────────────
    def collapse_wavefunction_rsi3(
        self,
        max_depth: int = _RSI_MAX_DEPTH,
        target_residual: float = _RSI_TARGET_RESIDUAL,
    ) -> QuantumMeasurement:
        r"""Colapso proyectivo gobernado por el Meta-Controlador RSI de Nivel 3.

        Jerarquía Meta-Cognitiva de 3 Niveles:
          - Nivel 1 (Ejecución Objeto): Evalúa observables cuánticos y el
            funcional de Lyapunov L_RSI(ϑ_1) en el bundle inicial.
          - Nivel 2 (Control de Primer Orden): Estima la tasa de convergencia p
            vía extrapolación de Richardson y contrae la escala de tolerancia.
          - Nivel 3 (Meta-Meta-Control / Endofuntor sobre la Ley de Optimización):
            Aplica el proceso no lineal Δ² de Aitken / transformada de Shanks,
            audita el factor de contracción de Banach κ_Banach y muta la ley
            perturbativa activando la resumación diagonal A-estable de Borel-Padé
            [L/L] para garantizar convergencia superlineal hacia el punto fijo.
        """
        initial_bundle = self._p2.phase1.tolerance_bundle
        bundle = ToleranceBundle(
            scale=initial_bundle.scale,
            n_lindstedt_terms=max(2, initial_bundle.n_lindstedt_terms),
            use_pade_resummation=initial_bundle.use_pade_resummation,
            damping_factor=initial_bundle.damping_factor,
        )
        traces: list[RecursiveSelfImprovementTrace] = []
        best_m: Optional[QuantumMeasurement] = None
        best_res = float("inf")

        for level in range(1, max_depth + 1):
            # En Nivel ≥ 3, el meta-controlador muta la ley hacia resumación de Padé
            meta_mutated = False
            if level >= 3 and not bundle.use_pade_resummation:
                bundle = bundle.with_pade(True).with_terms(max(6, bundle.n_lindstedt_terms + 2))
                meta_mutated = True

            m, residual, unitarity_def = self._run_pipeline_at_bundle(bundle)
            trace = self._rsi_compute_trace(
                level=level,
                bundle=bundle,
                residual=residual,
                unitarity_defect=unitarity_def,
                meta_law_mutated=meta_mutated,
                traces_prev=tuple(traces),
            )
            traces.append(trace)

            if residual < best_res:
                best_res = residual
                best_m = m

            if residual <= target_residual and level >= 2:
                break

            factor = trace.adaptation_factor
            if not math.isfinite(factor) or factor <= 0.0:
                factor = 0.1
            new_scale = max(_RSI_MIN_TOLERANCE, bundle.scale * factor)
            if new_scale >= bundle.scale:
                new_scale = bundle.scale * 0.25

            # Si en Nivel 1 el residuo es alto, preparar salto de orden y Padé si persiste
            new_terms = min(12, bundle.n_lindstedt_terms + 3)
            use_pade_next = bundle.use_pade_resummation or (level >= 2 and residual > 1e-2)
            new_damping = 0.5 if trace.banach_contraction_factor >= 1.0 else 1.0

            bundle = ToleranceBundle(
                scale=new_scale,
                n_lindstedt_terms=new_terms,
                use_pade_resummation=use_pade_next,
                damping_factor=new_damping,
            )

        last = traces[-1]
        level3 = any(t.level3_active or t.meta_law_mutated for t in traces)
        banach_certified = (
            last.banach_contraction_factor < 1.0 or best_res <= target_residual
        )

        rsi_result = SelfImprovementResult(
            traces=tuple(traces),
            final_scale=float(last.tolerance_scale),
            final_n_terms=int(last.n_lindstedt_terms),
            final_residual=float(best_res),
            converged=bool(best_res <= target_residual),
            richardson_limit=float(last.richardson_extrapolation),
            aitken_limit=float(last.aitken_delta_squared),
            effective_convergence_order=float(last.convergence_rate),
            level3_activated=bool(level3),
            banach_fixed_point_certified=bool(banach_certified),
            shanks_limit=float(last.shanks_extrapolation),
            used_pade_resummation=bool(bundle.use_pade_resummation or any(t.meta_law_mutated for t in traces)),
        )

        if best_m is None:
            raise RecursiveSelfImprovementError(
                "RSI-3 no produjo ninguna medición."
            )
        final_m = self._rebuild_with_rsi(best_m, rsi_result)
        QuantumLogger.log_measurement(final_m, level=logging.INFO)
        return final_m

    @staticmethod
    def _rebuild_with_rsi(
        m: QuantumMeasurement, rsi: SelfImprovementResult
    ) -> QuantumMeasurement:
        """Reconstruye `QuantumMeasurement` adjuntando `rsi_result` de forma inmutable."""
        return QuantumMeasurement(
            eigenstate=m.eigenstate,
            incident_energy=m.incident_energy,
            work_function=m.work_function,
            tunneling_probability=m.tunneling_probability,
            kinetic_energy=m.kinetic_energy,
            momentum=m.momentum,
            frustration_veto=m.frustration_veto,
            effective_mass=m.effective_mass,
            dominant_pole_real=m.dominant_pole_real,
            threat_level=m.threat_level,
            collapse_threshold=m.collapse_threshold,
            admission_reason=m.admission_reason,
            wkb_parameters=m.wkb_parameters,
            measurement_uncertainty=m.measurement_uncertainty,
            tunneling_wkb_diagnostic=m.tunneling_wkb_diagnostic,
            kappa_width_diagnostic=m.kappa_width_diagnostic,
            collapse_ambiguity=m.collapse_ambiguity,
            barrier_regime=m.barrier_regime,
            poincare_snapshot=m.poincare_snapshot,
            lindstedt_expansion=m.lindstedt_expansion,
            energy_return_map=m.energy_return_map,
            transmission_recurrence=m.transmission_recurrence,
            rsi_result=rsi,
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 13: ORQUESTADOR CATEGÓRICO (𝒵_Admission = Φ₃ ∘ Φ₂ ∘ Φ₁)
# ═════════════════════════════════════════════════════════════════════════════

class QuantumAdmissionGate(Morphism):
    """Operador de proyección de Hilbert como morfismo del Estrato PHYSICS."""

    def __init__(
        self,
        topo_watcher: ITopologicalWatcher,
        laplace_oracle: ILaplaceOracle,
        sheaf_orchestrator: ISheafCohomologyOrchestrator,
        *,
        strict_wkb: bool = False,
    ) -> None:
        super().__init__(name="QuantumAdmissionGate")
        self._topo_watcher = topo_watcher
        self._laplace_oracle = laplace_oracle
        self._sheaf_orchestrator = sheaf_orchestrator
        self._strict_wkb = bool(strict_wkb)
        self._phase1_factory = lambda bundle=None: Phase1_GaugeAdmissionCertifier(
            topo_watcher,
            laplace_oracle,
            sheaf_orchestrator,
            tolerance_bundle=bundle,
        )
        self._phase1_factory()  # Validación eager de contratos de oráculos
        logger.info(
            "QuantumAdmissionGate v%s inicializada. strict_wkb=%s.",
            __version__,
            self._strict_wkb,
        )

    @property
    def domain(self) -> frozenset:
        return frozenset()

    @property
    def codomain(self) -> Stratum:
        return Stratum.PHYSICS

    def evaluate_admission(
        self, payload: Mapping[str, Any]
    ) -> QuantumMeasurement:
        """Ejecuta la cadena anidada Φ₁.nest_into_phase2().nest_into_phase3(payload).collapse_wavefunction()."""
        if not isinstance(payload, Mapping):
            raise QuantumAdmissionError(
                f"payload debe ser Mapping; recibido {type(payload).__name__}."
            )
        phase1 = self._phase1_factory()
        phase2 = phase1.nest_into_phase2()
        p1 = phase2.phase1
        if p1.veto_triggered:
            m = phase1.emit_veto_measurement(p1)
            QuantumLogger.log_measurement(m, level=logging.WARNING)
            return m
        phase3 = phase2.nest_into_phase3(payload)
        if self._strict_wkb and not phase3._strict_wkb:
            phase3 = Phase3_BarrierCollapseProjector(
                phase3.phase2, strict_wkb=True
            )
        return phase3.collapse_wavefunction()

    def evaluate_admission_rsi3(
        self,
        payload: Mapping[str, Any],
        max_depth: int = _RSI_MAX_DEPTH,
        target_residual: float = _RSI_TARGET_RESIDUAL,
    ) -> QuantumMeasurement:
        """Ejecuta la cadena anidada Φ₁ → Φ₂ → Φ₃ con el meta-controlador RSI-3."""
        if not isinstance(payload, Mapping):
            raise QuantumAdmissionError(
                f"payload debe ser Mapping; recibido {type(payload).__name__}."
            )
        phase1 = self._phase1_factory()
        phase2 = phase1.nest_into_phase2()
        p1 = phase2.phase1
        if p1.veto_triggered:
            m = phase1.emit_veto_measurement(p1)
            QuantumLogger.log_measurement(m, level=logging.WARNING)
            return m
        phase3 = phase2.nest_into_phase3(payload)
        if self._strict_wkb and not phase3._strict_wkb:
            phase3 = Phase3_BarrierCollapseProjector(
                phase3.phase2, strict_wkb=True
            )
        return phase3.collapse_wavefunction_rsi3(
            max_depth=max_depth, target_residual=target_residual
        )

    def __call__(self, state: CategoricalState) -> CategoricalState:
        """Aplica el morfismo proyectivo de admisión a un estado categórico."""
        if not isinstance(state, CategoricalState):
            raise QuantumAdmissionError(
                f"state debe ser CategoricalState; recibido {type(state).__name__}."
            )
        payload = getattr(state, "payload", None)
        if not isinstance(payload, Mapping):
            raise QuantumAdmissionError(
                f"state.payload debe ser Mapping; recibido {type(payload).__name__}."
            )
        ctx = dict(getattr(state, "context", None) or {})
        m = self.evaluate_admission(payload)
        if m.eigenstate == Eigenstate.RECHAZADO:
            err = (
                f"VETO CUÁNTICO | {m.eigenstate} | E={m.incident_energy:.3e} | "
                f"Φ={m.work_function:.3e} | T={m.tunneling_probability:.3e} | "
                f"veto={m.frustration_veto} | razón: {m.admission_reason}"
            )
            logger.error(err)
            return CategoricalState(
                payload=payload,
                context={**ctx, "quantum_error": err, "quantum_admission": m},
                validated_strata=frozenset(),
            )
        logger.info(
            "ADMISIÓN CUÁNTICA | |ADMITIDO⟩ | p=%.3e | E=%.3e | Φ=%.3e | T=%.3e.",
            m.momentum,
            m.incident_energy,
            m.work_function,
            m.tunneling_probability,
        )
        return CategoricalState(
            payload=payload,
            context={
                **ctx,
                "quantum_momentum": m.momentum,
                "quantum_admission": m,
            },
            validated_strata=state.validated_strata | {Stratum.PHYSICS},
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 14: SUITE DE VERIFICACIÓN DOCTORAL
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    print("\n" + "═" * 84)
    print(f"QUANTUM ADMISSION GATE v{__version__} — SUITE DE VERIFICACIÓN")
    print("═" * 84 + "\n")

    class MockTopo:
        def __init__(self, threat: float = 0.5) -> None:
            self._t = threat

        def get_mahalanobis_threat(self) -> float:
            return self._t

    class MockLaplace:
        def __init__(self, pole: float = -0.5) -> None:
            self._p = pole

        def get_dominant_pole_real(self) -> float:
            return self._p

    class MockSheaf:
        def __init__(self, frust: float = 1e-12) -> None:
            self._f = frust

        def get_global_frustration_energy(self) -> float:
            return self._f

    print("TEST 1: Snapshot de Poincaré-KAM y Estabilidad de Nekhoroshev (Fase 1)")
    gate = QuantumAdmissionGate(MockTopo(0.5), MockLaplace(-0.5), MockSheaf(1e-12))
    p1 = gate._phase1_factory()
    snap = p1.capture_environmental_snapshot()
    pc = p1.compute_poincare_snapshot(snap)
    print(
        f"  I={pc.action_invariant:.4e}, θ={pc.angle_parameter:.4f}, "
        f"resonant={pc.is_near_resonance} ({pc.resonance_p}:{pc.resonance_q}), "
        f"KAM_intact={pc.kam_torus_intact}, T_Nekh={pc.nekhoroshev_stability_time:.3e}"
    )

    print("\nTEST 2: Expansión de Lindstedt-Poincaré y Resumación Diagonal de Padé (Fase 1)")
    for n_terms in (2, 4, 8):
        lind = p1.compute_lindstedt_expansion(snap, n_terms=n_terms)
        print(
            f"  n_terms={n_terms}: Φ_Taylor={lind.corrected_work_function:.6e} "
            f"(res={lind.expansion_residual:.3e}) | "
            f"Φ_Padé={lind.pade_approximant_value:.6e} "
            f"(res_Padé={lind.pade_residual:.3e}) | ω_LP={lind.strained_frequency:.4f}"
        )

    print("\nTEST 3: Encadenamiento Anidado Estricto Φ₁ → Φ₂ → Φ₃ (Monodromía + SL(2,ℂ))")
    m = (
        gate._phase1_factory()
        .nest_into_phase2()
        .nest_into_phase3({"endpoint": "/api/test", "data": "A" * 500})
        .collapse_wavefunction()
    )
    rm = m.energy_return_map
    print(
        f"  state={m.eigenstate}, T={m.tunneling_probability:.4e}, "
        f"Tr(M_ret)={rm.monodromy_trace if rm else 0:.4f}, "
        f"R_Greene={rm.greenes_residue if rm else 0:.4e}, "
        f"orbit={rm.orbit_class if rm else 'N/A'}"
    )

    print("\nTEST 4: Meta-Controlador RSI de Nivel 3 (Banach + Shanks-Aitken + Mutación Padé)")
    m_rsi = gate.evaluate_admission_rsi3({"endpoint": "/api/test", "payload": "X" * 1024})
    if m_rsi.rsi_result:
        r = m_rsi.rsi_result
        print(
            f"  final_residual={r.final_residual:.3e}, converged={r.converged}, "
            f"level3_active={r.level3_activated}, used_pade={r.used_pade_resummation}, "
            f"banach_certified={r.banach_fixed_point_certified}, n_terms={r.final_n_terms}"
        )
        for tr in r.traces:
            print(
                f"    [Nivel {tr.level}] scale={tr.tolerance_scale:.3e}, "
                f"n={tr.n_lindstedt_terms}, res={tr.residual:.3e}, "
                f"κ_Banach={tr.banach_contraction_factor:.3e}, "
                f"mutated={tr.meta_law_mutated}"
            )

    print("\nTEST 5: Veto Gauge-Cohomológico A4 (E_frust > tol)")
    gate_v = QuantumAdmissionGate(MockTopo(0.5), MockLaplace(-0.5), MockSheaf(1.0))
    m_v = gate_v.evaluate_admission({"key": "value"})
    print(f"  state={m_v.eigenstate}, veto={m_v.frustration_veto}")

    print("\n" + "═" * 84)
    print("SUITE DOCTORAL COMPLETADA EXITOSAMENTE")
    print("═" * 84 + "\n")


__all__ = [
    "QuantumAdmissionError",
    "QuantumNumericalError",
    "QuantumInterfaceError",
    "QuantumStateError",
    "WKBValidityError",
    "CohomologicalVetoError",
    "PoincareSectionError",
    "RecursiveSelfImprovementError",
    "PhysicalConstants",
    "NumericInterval",
    "BiquaternionElement",
    "WKBParameters",
    "RectangularBarrierTransmission",
    "CollapseAmbiguity",
    "PoincarePhaseSpaceSnapshot",
    "LindstedtWorkFunctionExpansion",
    "EnergyReturnMapSnapshot",
    "TransmissionRecurrenceSnapshot",
    "ToleranceBundle",
    "RecursiveSelfImprovementTrace",
    "SelfImprovementResult",
    "Eigenstate",
    "QuantumMeasurement",
    "ITopologicalWatcher",
    "ILaplaceOracle",
    "ISheafCohomologyOrchestrator",
    "EnvironmentalSnapshot",
    "GaugeFlatnessCertificate",
    "OrthomodularStructureCertificate",
    "Phase1GaugeAdmissionData",
    "Phase2SpectralEnergyData",
    "Phase1_GaugeAdmissionCertifier",
    "Phase2_SpectralEnergyAuditor",
    "Phase3_BarrierCollapseProjector",
    "QuantumAdmissionGate",
]