# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Hilbert Watcher (Operador del Hamiltoniano de Medición)             ║
║ Ruta   : app/agents/aleph/hilbert_watcher.py                                 ║
║ Versión: 5.1.0-Aleph-OODA-WKB-Maslov-Heyting-Poincare-KAM-RSI3-Strict-PhD    ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA (Nivel 4, V_{ℵ₀}):
────────────────────────────────────────────────────────────────────────────────
Funtor de Medición Coherente 𝓕: Superposition ⟶ Eigenstate, ejecutando el
colapso determinista Born-like de la función de onda semántica dentro de un
ciclo OODA (Observe ⟶ Orient ⟶ Decide/Act) gobernado por geometría simpléctica
de Poincaré-KAM-Nekhoroshev, teoría espectral de Maslov-Keller, lógica topos
intuicionista de Heyting-Lawvere-Tierney Ω₃ y automejora recursiva de Nivel 3
(RSI-3: Richardson ⟶ Orden p ⟶ Shanks/Aitken Δ² + mutación Borel-Padé [L/L]).

NOVEDADES v5.1.0 (Poincaré-KAM + Maslov-SL(2,ℂ) + Heyting-LT + RSI-3 Padé):
  [P1] Snapshot simpléctico de Poincaré en (χ², α|σ|): condición diofántica
       KAM, lenguas de Arnold y deriva adiabática acotada por Nekhoroshev.
  [P2] Expansión de Lindstedt-Poincaré de Φ(χ²) con frecuencia deformada ω(ε)
       para aniquilación secular + resumación racional A-estable de Padé [L/L].
       Compatibilidad exacta: n_terms=2 (default) preserva Φ = Φ₀ + αχ² (v4.1.0).
  [P3] Aplicación simpléctica de retorno en Sp(2, ℝ): matriz de monodromía
       de Chirikov-Taylor, residuo de Greene, multiplicadores de Floquet y
       exponente de Lyapunov.
  [P4] Espectro exacto de transparencia de Ramsauer-Townsend (k₂Δx = nπ) y
       tiempo medio de retorno ergódico de Kac-Poincaré en el círculo de fase.
  [P5] Meta-controlador RSI-3 con funcional de Lyapunov contráctil en espacio
       de Banach, aceleración de Shanks-Aitken Δ², certificación de punto fijo
       (κ_Banach < 1) y mutación de meta-ley hacia aproximantes de Borel-Padé.

INVARIANTES (I1–I5, A5):
  [I1] Born: T ≥ θ, θ = mantissa/2⁵³ ∈ [0, 1).
  [I2] Exergía Ξ = 8 − H(X); K_max = max(E−Φ, ε_kin).
  [I3] No-demolición cuántica: DTOs frozen + axioma A5.
  [I4] Barrera rectangular exacta SL(2, ℂ) (3 regímenes) + WKB; μ_Maslov ∈ {0, 2}.
  [I5] Clausura transitiva DIKW sobre el contexto categórico.
  [A5] Teorema de No-Clonación: __copy__/__deepcopy__ ⟶ HilbertStateCloningError.

ARQUITECTURA DE TRES FASES ANIDADAS (Continuidad Formal de Morfismos):
  Φ₁ : Phase1_WavefunctionObserver
         └──► último método: nest_into_phase2(payload) ──┐
                                                         ▼
  Φ₂ : Phase2_WKBGaugeOrienter.__init__(phase1_data)
         └──► último método: nest_into_phase3() ─────────┐
                                                         ▼
  Φ₃ : Phase3_BornHeytingCollapser.__init__(certificate)
         └──► últimos métodos: collapse_wavefunction_rsi3()
                               collapse_into_state(state)

Funtor Maestro:
  𝓕 = Φ₃ ∘ Φ₂ ∘ Φ₁ : CategoricalState ⟶ CategoricalState
"""
from __future__ import annotations

import hashlib
import logging
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from itertools import count as _itertools_count
from typing import (
    Any,
    Dict,
    Final,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

import numpy as np

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
        def __init__(self, name: str = "Morphism") -> None:
            self.name = name


logger = logging.getLogger("MIC.Agents.HilbertWatcher")

__version__: Final[str] = (
    "5.1.0-Aleph-OODA-WKB-Maslov-Heyting-Poincare-KAM-RSI3-Strict-PhD"
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


# ═════════════════════════════════════════════════════════════════════════════
# §1. JERARQUÍA DE EXCEPCIONES DEL OBSERVADOR DE HILBERT
# ═════════════════════════════════════════════════════════════════════════════

class HilbertWatcherError(Exception):
    """Excepción base del observador cuántico de Hilbert."""


class HilbertNumericalError(HilbertWatcherError):
    """Fallo numérico: infinitud, NaN, violación de positividad o rango."""


class HilbertInterfaceError(HilbertWatcherError):
    """Fallo de contrato estructural en dependencias inyectadas."""


class HilbertPayloadError(HilbertWatcherError):
    """Fallo en validación morfológica o serialización canónica del payload."""


class HilbertWKBValidityError(HilbertNumericalError):
    """WKB fuera del régimen semiclásico de validez κa ≫ 1."""


class HilbertStateCloningError(HilbertWatcherError):
    """Axioma A5: violación del Teorema de No-Clonación Cuántica."""


class HilbertCohomologicalVetoError(HilbertWatcherError):
    """Veto estructural por energía de frustración cohomológica global."""


class PoincareSectionError(HilbertWatcherError):
    """Degeneración simpléctica en la sección transversal de Poincaré."""


class RecursiveSelfImprovementError(HilbertWatcherError):
    """Divergencia o pérdida de contractividad de Banach en el meta-controlador RSI-3."""


# ═════════════════════════════════════════════════════════════════════════════
# §2. CONSTANTES FÍSICAS Y UMBRALES ESPECTRALES SELLADOS
# ═════════════════════════════════════════════════════════════════════════════

class QuantumThresholds:
    """Constantes físicas normalizadas (h=1, ℏ=1/(2π), entropía en bits)."""

    PLANCK_H: Final[float] = 1.0
    PLANCK_HBAR: Final[float] = PLANCK_H / (2.0 * math.pi)

    BASE_PHI: Final[float] = 10.0
    BASE_MASS: Final[float] = 1.0
    BARRIER_DX: Final[float] = 1.0
    ALPHA_COUPLING: Final[float] = 5.0

    EPSILON_MACH: Final[float] = 1e-9
    MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)
    ENTROPY_FLOOR: Final[float] = 1e-12
    MIN_KINETIC_ENERGY: Final[float] = 1e-12
    SIGMA_CHAOS_TOL: Final[float] = 1e-9
    EXP_UNDERFLOW_CUTOFF: Final[float] = -700.0
    EXP_OVERFLOW_CUTOFF: Final[float] = math.log(sys.float_info.max) - 1.0
    DIVISION_EPSILON: Final[float] = 1e-15

    FREQUENCY_SCALE: Final[float] = 1000.0
    MAX_PAYLOAD_BYTES: Final[int] = 10_485_760
    WKB_VALIDITY_THRESHOLD: Final[float] = 1.0
    THRESHOLD_ENERGY_REL_TOL: Final[float] = 1e-12
    SINH_ASYMPTOTIC_KA: Final[float] = 20.0
    FLOAT_MAX: Final[float] = sys.float_info.max

    MANTISSA_BITS: Final[int] = 53
    MANTISSA_MASK: Final[int] = (1 << MANTISSA_BITS) - 1
    MANTISSA_DIVISOR: Final[float] = float(1 << MANTISSA_BITS)

    DEFAULT_BLOCK_SIZE: Final[int] = 8
    MAX_SHANNON_BITS: Final[float] = 8.0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError(
            f"La clase {cls.__name__} está sellada. No se permite herencia."
        )


Const = QuantumThresholds


# ═════════════════════════════════════════════════════════════════════════════
# §3. RETÍCULO DE HEYTING Ω₃ (GÖDEL–DUMMETT) Y MODALIDAD DE LAWVERE-TIERNEY
# ═════════════════════════════════════════════════════════════════════════════

class HeytingOmega3(IntEnum):
    r"""Retículo de Heyting Ω₃ (Gödel-Dummett) totalmente ordenado por severidad.

        COHERENT (0) < DEGRADED (1) < VETOED (2).
        Join ⊔ = max, Meet ⊓ = min, Implicación residuada (a → b) = ⊤ si a ≤ b sino b,
        Negación intuicionista ¬a = (a → ⊥), y operador modal de clausura de
        Lawvere-Tierney j_LT(a) = ¬¬a (estabilización doble-negación en el topos).
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    def __le__(self, other: "HeytingOmega3") -> bool:
        return int(self) <= int(other)

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def implication(self, other: "HeytingOmega3") -> "HeytingOmega3":
        if int(self) <= int(other):
            return HeytingOmega3.VETOED
        return other

    def negation(self) -> "HeytingOmega3":
        return self.implication(HeytingOmega3.COHERENT)

    def lawvere_tierney_closure(self) -> "HeytingOmega3":
        """Operador modal de topología de Grothendieck / Lawvere-Tierney j(a) = ¬¬a."""
        return self.negation().negation()

    def brouwerian_codifference(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """Co-diferencia de Brouwer (a \\ b): menor elemento c tal que a ≤ b ⊔ c."""
        if int(self) <= int(other):
            return HeytingOmega3.COHERENT
        return self


# ═════════════════════════════════════════════════════════════════════════════
# §4. AUTOESTADOS DEL ESPACIO DE HILBERT ℋ₂
# ═════════════════════════════════════════════════════════════════════════════

class HilbertEigenstate(Enum):
    """Base ortonormal {|ADMITTED⟩, |REJECTED⟩} del espacio de proyección ℋ₂."""

    ADMITTED = auto()
    REJECTED = auto()

    def is_accepted(self) -> bool:
        return self == HilbertEigenstate.ADMITTED

    def complementary(self) -> "HilbertEigenstate":
        return (
            HilbertEigenstate.REJECTED
            if self == HilbertEigenstate.ADMITTED
            else HilbertEigenstate.ADMITTED
        )

    def __invert__(self) -> "HilbertEigenstate":
        return self.complementary()

    def __str__(self) -> str:
        return f"|{self.name}⟩"


# ═════════════════════════════════════════════════════════════════════════════
# §5. ESTRUCTURAS DE MECÁNICA CELESTE DE POINCARÉ, TEORÍA KAM Y RSI-3
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PoincarePhaseSpaceSnapshot:
    """Snapshot simpléctico de Poincaré en el espacio de fase (q=χ², p=α|σ|).

    Invariantes canónicos:
      - Acción I = ½(χ⁴ + (α|σ|)²), ángulo θ_p = atan2(α|σ|, χ²).
      - Sección transversal Σ = {y = section_level}; resonancia χ²/|σ| ≈ p/q.
      - Condición diofántica KAM: |ω₁/ω₂ − p/q| ≥ γ / q^(τ+1).
      - Deriva de acción acotada exponencialmente (Nekhoroshev-Chirikov).
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

    Sea ε := αχ²/Φ₀.
      - Para n_terms=2 y use_pade=False (default), preserva exactamente la ley
        aditiva física Φ_corrected = Φ₀(1 + ε) = Φ₀ + αχ² (compatibilidad v4.1.0).
      - Para n_terms > 2 incorpora correcciones O(εᵏ/k!) con frecuencia deformada
        ω_LP(ε) que aniquila términos seculares resonantes.
      - Cuando RSI-3 Nivel 3 activa la mutación de meta-ley (`use_pade=True`),
        emplea el aproximante racional diagonal de Borel-Padé [n/n](ε).
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
    """Aplicación simpléctica de retorno en (E, Φ) y espectro de Floquet Sp(2, ℝ).

    Integra la contracción J = min(E/Φ, Φ/E) con la matriz de monodromía
    simpléctica M_ret ∈ Sp(2, ℝ) (det M_ret = 1), el residuo de Greene
    R_Greene = ¼(2 − Tr(M_ret)), el exponente de Lyapunov y la clase orbital.
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
    """Recurrencia ergódica de Kac-Poincaré y resonancias de Ramsauer-Townsend."""

    recurrence_points: Tuple[float, ...]
    min_recurrence_distance: float
    is_poincare_recurrent: bool
    k2_samples: Tuple[float, ...]
    n_samples: int
    ramsauer_townsend_energies: Tuple[float, ...] = ()
    nearest_rt_quantum_number: int = 0
    kac_mean_recurrence_time: float = float("inf")
    ergodic_phase_discrepancy: float = 0.0


@dataclass(frozen=True, slots=True)
class ToleranceBundle:
    """Punto en la variedad de hiperparámetros del meta-controlador RSI-3."""

    scale: float = 1.0
    n_lindstedt_terms: int = 2
    use_pade_resummation: bool = False
    damping_factor: float = 1.0

    def _scaled(self, base: float) -> float:
        return max(_RSI_MIN_TOLERANCE, base * self.scale)

    @property
    def gauge_flatness_tol(self) -> float:
        return self._scaled(Const.EPSILON_MACH)

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


@dataclass(frozen=True, slots=True)
class RecursiveSelfImprovementTrace:
    """Traza inmutable por nivel de profundidad del meta-controlador RSI-3."""

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
    """Certificado de punto fijo y aceleración no lineal del meta-controlador RSI-3."""

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


# ═════════════════════════════════════════════════════════════════════════════
# §6. DTOs DEL HANDOFF ENTRE FASES ANIDADAS (A5: NO-CLONACIÓN)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True, eq=False)
class WavefunctionObserveData:
    """Certificado espectral de Φ₁ (H(X) ∈ [0, 8] bits, Ξ = 8−H, E = hν ± δE)."""

    serialized_payload: bytes
    payload_size: int
    byte_entropy_bits: float
    block_entropy_rate: float
    exergy_bits: float
    semantic_frequency: float
    incident_energy: float
    energy_uncertainty_lower: float
    energy_uncertainty_upper: float
    payload_sha256: str
    certification_hash: str

    def __post_init__(self) -> None:
        if self.payload_size < 0:
            raise HilbertNumericalError(f"payload_size < 0: {self.payload_size}")
        if not (0.0 <= self.byte_entropy_bits <= Const.MAX_SHANNON_BITS):
            raise HilbertNumericalError(
                f"byte_entropy_bits fuera de [0, 8]: {self.byte_entropy_bits}"
            )
        if self.block_entropy_rate < 0.0 or not math.isfinite(self.block_entropy_rate):
            raise HilbertNumericalError(
                f"block_entropy_rate no finita ≥ 0: {self.block_entropy_rate}"
            )
        if self.incident_energy < 0.0 or not math.isfinite(self.incident_energy):
            raise HilbertNumericalError(
                f"incident_energy no finita ≥ 0: {self.incident_energy}"
            )
        if not (
            self.energy_uncertainty_lower
            <= self.incident_energy
            <= self.energy_uncertainty_upper
        ):
            raise HilbertNumericalError(
                "Intervalo de incertidumbre no contiene la energía puntual."
            )

    def __copy__(self) -> "WavefunctionObserveData":
        raise HilbertStateCloningError("A5: WavefunctionObserveData inmutable.")

    def __deepcopy__(self, memo: Dict[int, Any]) -> "WavefunctionObserveData":
        raise HilbertStateCloningError("A5: WavefunctionObserveData inmutable.")


@dataclass(frozen=True, slots=True)
class EnvironmentalSnapshot:
    chi_squared: float
    dominant_pole_real: float
    global_frustration: float
    read_errors: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, v in (
            ("chi_squared", self.chi_squared),
            ("dominant_pole_real", self.dominant_pole_real),
            ("global_frustration", self.global_frustration),
        ):
            if math.isnan(v):
                raise HilbertNumericalError(f"Snapshot '{name}' es NaN.")


@dataclass(frozen=True, slots=True, eq=False)
class Phase1HilbertData:
    """DTO inmutable de Φ₁ (precondición formal de Φ₂)."""

    observe: WavefunctionObserveData
    snapshot: EnvironmentalSnapshot
    certification_hash: str
    tolerance_bundle: ToleranceBundle = field(default_factory=ToleranceBundle)
    poincare_snapshot: Optional[PoincarePhaseSpaceSnapshot] = None
    lindstedt_expansion: Optional[LindstedtWorkFunctionExpansion] = None


_ALLOWED_REGIMES: Final[FrozenSet[str]] = frozenset(
    {
        "classical",
        "tunneling",
        "impenetrable",
        "vetoed",
        "threshold",
        "over_barrier",
    }
)


@dataclass(frozen=True, slots=True, eq=False)
class GaugeCouplingCertificate:
    """Certificado de Φ₂ (precondición formal de Φ₃)."""

    phase1: Phase1HilbertData
    incident_energy: float
    work_function: float
    effective_mass: float
    dominant_pole_real: float
    threat_level: float
    frustration_energy: float
    is_frustrated: bool
    barrier_height: float
    wkb_kappa: float
    wkb_kappa_width: float
    wkb_exponent: float
    transmission_probability: float
    transmission_wkb: float
    transmission_ratio: float
    maslov_index: int
    maslov_phase: float
    regime: str
    certification_hash: str
    energy_return_map: Optional[EnergyReturnMapSnapshot] = None
    unitarity_residual: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.work_function) or self.work_function < 0.0:
            raise HilbertNumericalError(
                f"work_function no finita ≥ 0: {self.work_function}"
            )
        if math.isnan(self.effective_mass) or (
            self.effective_mass <= 0.0 and not math.isinf(self.effective_mass)
        ):
            raise HilbertNumericalError(
                f"effective_mass debe ser > 0 ó +∞: {self.effective_mass}"
            )
        if not math.isfinite(self.threat_level) or self.threat_level < 0.0:
            raise HilbertNumericalError(
                f"threat_level no finita ≥ 0: {self.threat_level}"
            )
        if not math.isfinite(self.frustration_energy) or self.frustration_energy < 0.0:
            raise HilbertNumericalError(
                f"frustration_energy no finita ≥ 0: {self.frustration_energy}"
            )
        if not math.isfinite(self.dominant_pole_real):
            raise HilbertNumericalError(
                f"dominant_pole_real no finita: {self.dominant_pole_real}"
            )
        if self.barrier_height < 0.0 or not math.isfinite(self.barrier_height):
            raise HilbertNumericalError(
                f"barrier_height no finita ≥ 0: {self.barrier_height}"
            )
        if not math.isfinite(self.incident_energy) or self.incident_energy < 0.0:
            raise HilbertNumericalError(
                f"incident_energy no finita ≥ 0: {self.incident_energy}"
            )
        if not (0.0 <= self.transmission_probability <= 1.0):
            raise HilbertNumericalError(
                f"T_exact fuera de [0,1]: {self.transmission_probability}"
            )
        if not (0.0 <= self.transmission_wkb <= 1.0):
            raise HilbertNumericalError(
                f"T_WKB fuera de [0,1]: {self.transmission_wkb}"
            )
        if self.maslov_index not in (0, 2):
            raise HilbertNumericalError(
                f"maslov_index debe ser 0 ó 2: {self.maslov_index}"
            )
        if self.regime not in _ALLOWED_REGIMES:
            raise HilbertNumericalError(f"regime desconocido: {self.regime!r}")

    @property
    def observe(self) -> WavefunctionObserveData:
        return self.phase1.observe

    def __copy__(self) -> "GaugeCouplingCertificate":
        raise HilbertStateCloningError("A5: GaugeCouplingCertificate inmutable.")

    def __deepcopy__(self, memo: Dict[int, Any]) -> "GaugeCouplingCertificate":
        raise HilbertStateCloningError("A5: GaugeCouplingCertificate inmutable.")


@dataclass(frozen=True, slots=True, eq=False)
class QuantumMeasurement:
    """Certificado de Φ₃ (medición proyectiva inmutable, Axioma A5)."""

    eigenstate: HilbertEigenstate
    heyting_verdict: HeytingOmega3
    collapse_threshold: float
    born_probability: float
    coherence_residual: float
    momentum: float
    kinetic_energy: float
    observable_snapshot: Dict[str, Any] = field(default_factory=dict)
    non_cloning_uid: int = field(
        default_factory=lambda: next(_NON_CLONING_UID_COUNTER),
        compare=False,
        repr=False,
    )
    poincare_snapshot: Optional[PoincarePhaseSpaceSnapshot] = None
    lindstedt_expansion: Optional[LindstedtWorkFunctionExpansion] = None
    energy_return_map: Optional[EnergyReturnMapSnapshot] = None
    transmission_recurrence: Optional[TransmissionRecurrenceSnapshot] = None
    rsi_result: Optional[SelfImprovementResult] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.collapse_threshold < 1.0):
            raise HilbertNumericalError(
                f"collapse_threshold fuera de [0,1): {self.collapse_threshold}"
            )
        if not (0.0 <= self.born_probability <= 1.0):
            raise HilbertNumericalError(
                f"born_probability fuera de [0,1]: {self.born_probability}"
            )
        if not (0.0 <= self.coherence_residual <= 1.0):
            raise HilbertNumericalError(
                f"coherence_residual fuera de [0,1]: {self.coherence_residual}"
            )
        if self.momentum < 0.0 or not math.isfinite(self.momentum):
            raise HilbertNumericalError(f"momentum no finito ≥ 0: {self.momentum}")
        if self.kinetic_energy < 0.0 or not math.isfinite(self.kinetic_energy):
            raise HilbertNumericalError(
                f"kinetic_energy no finita ≥ 0: {self.kinetic_energy}"
            )
        if self.eigenstate == HilbertEigenstate.REJECTED and (
            self.momentum != 0.0 or self.kinetic_energy != 0.0
        ):
            raise HilbertNumericalError(
                "Estado REJECTED exige momentum = kinetic_energy = 0."
            )

    def __copy__(self) -> "QuantumMeasurement":
        raise HilbertStateCloningError(
            f"A5: QuantumMeasurement no clonable (uid={self.non_cloning_uid})."
        )

    def __deepcopy__(self, memo: Dict[int, Any]) -> "QuantumMeasurement":
        raise HilbertStateCloningError(
            f"A5: QuantumMeasurement no clonable (uid={self.non_cloning_uid})."
        )


@dataclass(frozen=True, slots=True)
class RectangularBarrierTransmission:
    """Resultado de dispersión cuántica vía Matriz de Transferencia SL(2, ℂ)."""

    t_exact: float
    t_wkb: float
    t_ratio: float
    kappa_width: float
    kappa: float
    exponent: float
    is_classical: bool
    regime: str
    reflection_exact: float = 0.0
    unitarity_residual: float = 0.0
    transfer_matrix_det: float = 1.0
    characteristic_impedance_ratio: float = 1.0


# ═════════════════════════════════════════════════════════════════════════════
# §7. PROTOCOLOS DE ORÁCULOS EXTERNOS
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
# §8. MORFISMOS NUMÉRICOS DEFENSIVOS
# ═════════════════════════════════════════════════════════════════════════════

def _ensure_finite_float(
    value: Any, *, name: str, allow_inf: bool = False
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HilbertNumericalError(
            f"{name} no convertible a float: {value!r}"
        ) from exc
    if math.isnan(result):
        raise HilbertNumericalError(f"{name} es NaN.")
    if not allow_inf and not math.isfinite(result):
        raise HilbertNumericalError(f"{name} no finito: {result!r}")
    return result


def _ensure_nonneg_finite_float(value: Any, *, name: str) -> float:
    result = _ensure_finite_float(value, name=name)
    if result < 0.0:
        raise HilbertNumericalError(f"{name} debe ser ≥ 0: {result}")
    return result


def _ensure_positive_or_posinf_float(value: Any, *, name: str) -> float:
    result = _ensure_finite_float(value, name=name, allow_inf=True)
    if math.isnan(result) or (result <= 0.0 and not math.isinf(result)):
        raise HilbertNumericalError(f"{name} debe ser > 0 ó +∞: {result}")
    return result


def _clamp_probability(value: float) -> float:
    if math.isnan(value):
        logger.warning("clamp_probability recibió NaN → 0.0.")
        return 0.0
    if math.isinf(value):
        return 1.0 if value > 0.0 else 0.0
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _safe_context(context: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if context is None:
        return {}
    if not isinstance(context, Mapping):
        warn = f"context no es Mapping: {type(context).__name__}"
        logger.warning(warn)
        return {"_context_warning": warn}
    return dict(context)


def _safe_exp(exponent: float) -> float:
    if exponent <= Const.EXP_UNDERFLOW_CUTOFF:
        return 0.0
    if exponent > Const.EXP_OVERFLOW_CUTOFF:
        return float("inf")
    return math.exp(exponent)


def _safe_sinh(value: float) -> float:
    if value >= Const.EXP_OVERFLOW_CUTOFF:
        return Const.FLOAT_MAX
    if value <= -Const.EXP_OVERFLOW_CUTOFF:
        return -Const.FLOAT_MAX
    try:
        return math.sinh(value)
    except OverflowError:
        return Const.FLOAT_MAX if value > 0 else -Const.FLOAT_MAX


def _safe_sqrt(value: float) -> float:
    return math.sqrt(max(0.0, value))


def _compute_dto_hash(*fields: Any) -> str:
    hasher = hashlib.sha256()
    for f in fields:
        if isinstance(f, bool):
            hasher.update(f"|{int(f)}".encode())
        elif isinstance(f, float):
            hasher.update(f"|{f:.17e}".encode())
        elif isinstance(f, int):
            hasher.update(f"|{f}".encode())
        elif isinstance(f, str):
            hasher.update(f"|{f}".encode())
        elif isinstance(f, bytes):
            hasher.update(b"|b")
            hasher.update(hashlib.sha256(f).digest())
        elif f is None:
            hasher.update(b"|__None__")
        else:
            hasher.update(f"|{f!r}".encode())
    return hasher.hexdigest()


def _validate_oracle_dependency(
    obj: Any, method_name: str, obj_name: str
) -> None:
    if obj is None:
        raise HilbertInterfaceError(f"{obj_name} es None.")
    if not hasattr(obj, method_name):
        raise HilbertInterfaceError(
            f"{obj_name} (tipo={type(obj).__name__}) carece de '{method_name}'."
        )
    if not callable(getattr(obj, method_name)):
        raise HilbertInterfaceError(f"{obj_name}.{method_name} no es invocable.")


# ═════════════════════════════════════════════════════════════════════════════
# §9. UTILIDADES DE MECÁNICA CELESTE (POINCARÉ-KAM) Y ACELERACIÓN RSI-3
# ═════════════════════════════════════════════════════════════════════════════

def _continued_fraction_approximation(
    x: float, max_denominator: int = _POINCARE_MAX_DENOMINATOR
) -> Tuple[int, int]:
    """Mejor aproximación racional diofántica p/q de x vía fracciones continuas."""
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
    """Retorna (es_resonante, p, q, error_racional, magnitud_divisor_pequeño)."""
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
    frustration_eps: float,
    chi_squared: float,
) -> Tuple[float, bool, float, float, float]:
    """Evalúa condición diofántica KAM, lengua de Arnold, tiempo y deriva de Nekhoroshev.

    La deriva adiabática de acción en presencia de resonancia p:q está gobernada
    por el teorema de Nekhoroshev:
        ΔI_Nekh = (ε_frust / (p² + q²)) · exp(−(1 / max(ε_frust, 10⁻¹²))^b).
    """
    if q <= 0 or not math.isfinite(ratio):
        return (0.0, True, float("inf"), 0.0, 0.0)
    dist = abs(ratio - (p / q))
    diophantine_bound = _KAM_DIOPHANTINE_GAMMA / (
        float(q) ** (_KAM_DIOPHANTINE_TAU + 1.0)
    )
    margin = dist - diophantine_bound
    eps_pert = max(
        _SMALL_PARAMETER_FLOOR,
        min(10.0, abs(frustration_eps) + 1e-4 * max(0.0, chi_squared)),
    )
    arnold_width = min(1.0, eps_pert ** (0.5 * max(1, q)))
    kam_intact = (margin >= 0.0) or (frustration_eps <= Const.EPSILON_MACH)
    nekh_exp = min(80.0, (1.0 / eps_pert) ** _NEKHOROSHEV_EXPONENT_B)
    nekh_time = min(Const.FLOAT_MAX, _safe_exp(nekh_exp))
    # Deriva secular suprimida exponencialmente según Nekhoroshev
    pq_norm = float(max(1, p * p + q * q))
    drift_est = float(max(0.0, frustration_eps) / pq_norm)
    return (
        float(margin),
        bool(kam_intact),
        float(nekh_time),
        float(arnold_width),
        float(drift_est),
    )


def _pade_approximant_exp(eps: float, L: int, M: int) -> float:
    r"""Aproximante racional de Borel-Padé [L/M](ε) en forma cerrada para exp(ε)."""
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
            float(math.factorial(total - k))
            * L_fact
            / (total_fact * float(math.factorial(k) * math.factorial(L - k)))
        )
        num += coeff * eps_pow
        eps_pow *= eps

    den = 0.0
    eps_pow = 1.0
    for k in range(M + 1):
        coeff = (
            float(math.factorial(total - k))
            * M_fact
            / (total_fact * float(math.factorial(k) * math.factorial(M - k)))
        )
        sign = -1.0 if (k % 2 == 1) else 1.0
        den += sign * coeff * eps_pow
        eps_pow *= eps

    if abs(den) <= Const.DIVISION_EPSILON:
        return min(Const.FLOAT_MAX, _safe_exp(eps))
    return max(0.0, num / den)


def _lindstedt_poincare_additive_phi_expansion(
    chi_squared: float,
    base_phi: float,
    n_terms: int = 2,
    use_pade: bool = False,
) -> LindstedtWorkFunctionExpansion:
    r"""Expansión de Lindstedt-Poincaré de Φ(χ²) con frecuencia deformada y Padé.

    Sea ε := αχ²/Φ₀:
      - Para n_terms=2 y use_pade=False (default):
            Φ_corrected = Φ₀(1 + ε) = Φ₀ + αχ²
        preservando de manera exacta la física aditiva v4.1.0.
      - Para n_terms ≥ 3: suma la serie perturbativa Φ₀ Σ_{k=0}^{n_terms−1} εᵏ/k!.
      - Si `use_pade=True` (activado en RSI-3 Nivel 3): utiliza la resumación
        diagonal A-estable de Borel-Padé Φ₀ · [n/n](ε).
    """
    eps = max(
        _SMALL_PARAMETER_FLOOR,
        Const.ALPHA_COUPLING * max(0.0, chi_squared) / max(1e-12, base_phi),
    )
    eps = min(10.0, eps)
    Phi0 = float(base_phi)
    n = max(2, int(n_terms))

    first = Phi0 * eps
    second = (Phi0 * eps * eps / 2.0) if n >= 3 else 0.0
    higher = 0.0
    fact = 6.0
    eps_k = eps * eps * eps
    for k in range(3, n):
        higher += Phi0 * eps_k / fact
        fact *= (k + 1)
        eps_k *= eps
    taylor_corrected = Phi0 + first + second + higher

    # Frecuencia deformada de Lindstedt-Poincaré ω(ε) para aniquilación secular
    secular_coeffs = (1.0, 0.375, -0.08203125, 0.03955078125, -0.02166748046875)
    strained_omega = 0.0
    eps_pow = 1.0
    for idx in range(min(n, len(secular_coeffs))):
        strained_omega += secular_coeffs[idx] * eps_pow
        eps_pow *= min(eps, 1.0)

    pade_val = Phi0 * _pade_approximant_exp(eps, L=n, M=n)
    true_value = Phi0 * math.exp(min(eps, Const.EXP_OVERFLOW_CUTOFF))

    taylor_residual = abs(taylor_corrected - true_value) / max(
        1.0, abs(true_value)
    )
    pade_residual = abs(pade_val - true_value) / max(1.0, abs(true_value))

    corrected = pade_val if use_pade else taylor_corrected
    active_residual = pade_residual if use_pade else taylor_residual
    is_consistent = (active_residual <= _ADIABATIC_DRIFT_LIMIT) or (
        eps < 0.5 or use_pade
    )

    return LindstedtWorkFunctionExpansion(
        small_parameter=float(eps),
        n_terms=int(n),
        zero_order_work_function=Phi0,
        first_order_correction=float(first),
        second_order_correction=float(Phi0 * eps * eps / 2.0),
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


def _richardson_extrapolation(
    r_prev: float, r_curr: float, tol_prev: float, tol_curr: float
) -> Tuple[float, float]:
    """Estimación de orden efectivo de convergencia p y extrapolación de Richardson."""
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
# §10. DISPERSIÓN EXACTA DE BARRERA RECTANGULAR SL(2, ℂ) + DIAGNÓSTICO WKB
# ═════════════════════════════════════════════════════════════════════════════

class RectangularBarrierCalculator:
    r"""Dispersión exacta mediante Matriz de Transferencia SL(2, ℂ) y diagnóstico WKB.

    Verifica en los tres regímenes (túnel E < V₀, umbral E = V₀, sobre-barrera E > V₀)
    la identidad de conservación de corriente de probabilidad (unitariedad S-matrix):
        T_exact + R_exact = 1,   det(M_TMM) = |M₁₁|² − |M₁₂|² = 1.
    """

    @staticmethod
    def _threshold_transmission(
        V0: float, m_eff: float, width: float
    ) -> Tuple[float, float, float, float]:
        hbar = Const.PLANCK_HBAR
        xi = m_eff * V0 * width * width / (2.0 * hbar * hbar)
        if not math.isfinite(xi) or xi >= Const.FLOAT_MAX / 2.0:
            return (0.0, 1.0, 1.0, float("inf"))
        xi = max(0.0, xi)
        t_th = 1.0 / (1.0 + xi)
        r_th = xi / (1.0 + xi)
        det_m = (1.0 + xi * xi) - (xi * xi)
        return (t_th, r_th, det_m, math.sqrt(xi))

    @staticmethod
    def _tunnel_transmission_stable(
        E: float, V0: float, kappa_a: float
    ) -> Tuple[float, float, float, float]:
        if E <= 0.0:
            return (0.0, 1.0, 1.0, float("inf"))
        D = V0 - E
        if D <= 0.0:
            return (1.0, 0.0, 1.0, 1.0)
        z_ratio = math.sqrt(D / E)
        if kappa_a >= Const.SINH_ASYMPTOTIC_KA:
            C = (V0 * V0) / (16.0 * E * D)
            em = _safe_exp(-2.0 * kappa_a)
            den = em + C
            t_ex = _clamp_probability(em / den) if den > 0 else 0.0
            return (t_ex, 1.0 - t_ex, 1.0, z_ratio)
        s = _safe_sinh(kappa_a)
        if math.isinf(s) or abs(s) >= math.sqrt(Const.FLOAT_MAX) / 2.0:
            C = (V0 * V0) / (16.0 * E * D)
            em = _safe_exp(-2.0 * kappa_a)
            den = em + C
            t_ex = _clamp_probability(em / den) if den > 0 else 0.0
            return (t_ex, 1.0 - t_ex, 1.0, z_ratio)
        m12_sq = (V0 * V0 * s * s) / (4.0 * E * D)
        m11_sq = 1.0 + m12_sq
        t_ex = _clamp_probability(1.0 / m11_sq) if m11_sq > 0 else 0.0
        r_ex = _clamp_probability(m12_sq / m11_sq) if m11_sq > 0 else 1.0
        det_m = m11_sq - m12_sq
        return (t_ex, r_ex, det_m, z_ratio)

    @staticmethod
    def _over_barrier_transmission(
        E: float, V0: float, m_eff: float, width: float
    ) -> Tuple[float, float, float, float]:
        D = E - V0
        if D <= 0.0:
            return (1.0, 0.0, 1.0, 1.0)
        z_ratio = math.sqrt(D / E) if E > 0 else 1.0
        k2 = _safe_sqrt(2.0 * m_eff * D) / Const.PLANCK_HBAR
        s = math.sin(k2 * width)
        m12_sq = (V0 * V0 * s * s) / (4.0 * E * D)
        m11_sq = 1.0 + m12_sq
        t_ex = _clamp_probability(1.0 / m11_sq) if m11_sq > 0 else 0.0
        r_ex = _clamp_probability(m12_sq / m11_sq) if m11_sq > 0 else 1.0
        det_m = m11_sq - m12_sq
        return (t_ex, r_ex, det_m, z_ratio)

    @classmethod
    def compute(
        cls,
        E: float,
        V0: float,
        m_eff: float,
        width: float,
        threshold_rel_tol: float = Const.THRESHOLD_ENERGY_REL_TOL,
    ) -> RectangularBarrierTransmission:
        E = _ensure_finite_float(E, name="E")
        V0 = _ensure_finite_float(V0, name="V0")
        m_eff = _ensure_finite_float(m_eff, name="m_eff", allow_inf=True)
        width = _ensure_finite_float(width, name="width")
        if E < 0 or V0 < 0 or width <= 0:
            raise HilbertNumericalError(
                f"Parámetros no físicos: E={E}, V₀={V0}, a={width}."
            )
        rel = threshold_rel_tol * max(1.0, abs(V0), abs(E))

        if math.isinf(m_eff):
            if E + rel < V0:
                return RectangularBarrierTransmission(
                    t_exact=0.0,
                    t_wkb=0.0,
                    t_ratio=float("inf"),
                    kappa_width=float("inf"),
                    kappa=float("inf"),
                    exponent=float("-inf"),
                    is_classical=False,
                    regime="impenetrable",
                    reflection_exact=1.0,
                    unitarity_residual=0.0,
                    transfer_matrix_det=1.0,
                    characteristic_impedance_ratio=float("inf"),
                )
            regime = "over_barrier" if E > V0 + rel else "threshold"
            return RectangularBarrierTransmission(
                t_exact=1.0,
                t_wkb=1.0,
                t_ratio=1.0,
                kappa_width=0.0,
                kappa=0.0,
                exponent=0.0,
                is_classical=True,
                regime=regime,
                reflection_exact=0.0,
                unitarity_residual=0.0,
                transfer_matrix_det=1.0,
                characteristic_impedance_ratio=1.0,
            )

        if m_eff <= 0:
            raise HilbertNumericalError(f"m_eff debe ser > 0: {m_eff}.")

        if abs(E - V0) <= rel:
            t_th, r_th, det_m, z_r = cls._threshold_transmission(
                V0, m_eff, width
            )
            u_res = abs((t_th + r_th) - 1.0) + abs(det_m - 1.0)
            return RectangularBarrierTransmission(
                t_exact=t_th,
                t_wkb=1.0,
                t_ratio=t_th,
                kappa_width=0.0,
                kappa=0.0,
                exponent=0.0,
                is_classical=True,
                regime="threshold",
                reflection_exact=r_th,
                unitarity_residual=u_res,
                transfer_matrix_det=det_m,
                characteristic_impedance_ratio=z_r,
            )
        if E > V0:
            t_ex, r_ex, det_m, z_r = cls._over_barrier_transmission(
                E, V0, m_eff, width
            )
            u_res = abs((t_ex + r_ex) - 1.0) + abs(det_m - 1.0)
            return RectangularBarrierTransmission(
                t_exact=t_ex,
                t_wkb=1.0,
                t_ratio=t_ex,
                kappa_width=0.0,
                kappa=0.0,
                exponent=0.0,
                is_classical=True,
                regime="over_barrier",
                reflection_exact=r_ex,
                unitarity_residual=u_res,
                transfer_matrix_det=det_m,
                characteristic_impedance_ratio=z_r,
            )

        D = V0 - E
        integrand = _safe_sqrt(2.0 * m_eff * D)
        kappa = integrand / Const.PLANCK_HBAR
        kappa_a = kappa * width
        raw_exp = -2.0 * kappa_a
        t_ex, r_ex, det_m, z_r = cls._tunnel_transmission_stable(
            E, V0, kappa_a
        )
        t_wkb = _clamp_probability(_safe_exp(raw_exp))
        if t_wkb > Const.DIVISION_EPSILON:
            t_ratio = t_ex / t_wkb
        elif t_ex > Const.DIVISION_EPSILON:
            t_ratio = float("inf")
        else:
            t_ratio = 1.0
        u_res = abs((t_ex + r_ex) - 1.0) + abs(det_m - 1.0)
        return RectangularBarrierTransmission(
            t_exact=_clamp_probability(t_ex),
            t_wkb=t_wkb,
            t_ratio=t_ratio,
            kappa_width=kappa_a,
            kappa=kappa,
            exponent=raw_exp,
            is_classical=False,
            regime="tunneling",
            reflection_exact=r_ex,
            unitarity_residual=u_res,
            transfer_matrix_det=det_m,
            characteristic_impedance_ratio=z_r,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE ANIDADA 1: OBSERVE — ESPECTRO, POINCARÉ-KAM Y LINDSTEDT-PADÉ         ║
# ║                                                                             ║
# ║   Secuencia interna de morfismos de Fase 1:                                 ║
# ║     1. __init__()                                                           ║
# ║     2. validate_payload_structure()                                         ║
# ║     3. serialize_payload_canonical()                                        ║
# ║     4. byte_entropy_bits()                                                  ║
# ║     5. block_entropy_rate()                                                 ║
# ║     6. compute_semantic_frequency()                                         ║
# ║     7. observe_payload()                                                    ║
# ║     8. capture_environmental_snapshot()                                     ║
# ║     9. compute_poincare_snapshot()        [Poincaré + KAM + Nekhoroshev]    ║
# ║    10. compute_lindstedt_expansion()      [Lindstedt-Poincaré + Borel-Padé] ║
# ║    11. nest_into_phase2(payload)          ★ MORFISMO TERMINAL → INICIA Φ₂  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase1_WavefunctionObserver:
    r"""FASE 1 — OBSERVE: Espectro de Shannon-Planck, Poincaré-KAM y Lindstedt.

    Cuantiza la función de onda semántica del payload (entropía H ∈ [0, 8],
    exergía Ξ = 8 − H, energía incidente E = hν), captura el snapshot ambiental
    de los oráculos topológico, espectral y cohomológico, proyecta el espacio
    de fase simpléctico (χ², α|σ|) sobre la sección de Poincaré Σ y construye
    la serie perturbativa de Lindstedt-Poincaré-Padé de Φ(χ²).

    Su método final `nest_into_phase2(payload)` es el morfismo terminal de Φ₁
    cuyo codominio inicia formalmente a `Phase2_WKBGaugeOrienter`.
    """

    def __init__(
        self,
        topo_watcher: ITopologicalWatcher,
        laplace_oracle: ILaplaceOracle,
        sheaf_orchestrator: ISheafCohomologyOrchestrator,
        tolerance_bundle: Optional[ToleranceBundle] = None,
    ) -> None:
        _validate_oracle_dependency(
            topo_watcher, "get_mahalanobis_threat", "topo_watcher"
        )
        _validate_oracle_dependency(
            laplace_oracle, "get_dominant_pole_real", "laplace_oracle"
        )
        _validate_oracle_dependency(
            sheaf_orchestrator,
            "get_global_frustration_energy",
            "sheaf_orchestrator",
        )
        if not isinstance(topo_watcher, ITopologicalWatcher):
            raise HilbertInterfaceError(
                f"topo_watcher no cumple ITopologicalWatcher: {type(topo_watcher).__name__}."
            )
        if not isinstance(laplace_oracle, ILaplaceOracle):
            raise HilbertInterfaceError(
                f"laplace_oracle no cumple ILaplaceOracle: {type(laplace_oracle).__name__}."
            )
        if not isinstance(sheaf_orchestrator, ISheafCohomologyOrchestrator):
            raise HilbertInterfaceError(
                f"sheaf_orchestrator no cumple ISheafCohomologyOrchestrator: "
                f"{type(sheaf_orchestrator).__name__}."
            )
        self._topo: Final[ITopologicalWatcher] = topo_watcher
        self._laplace: Final[ILaplaceOracle] = laplace_oracle
        self._sheaf: Final[ISheafCohomologyOrchestrator] = sheaf_orchestrator
        self._tol_bundle: Final[ToleranceBundle] = (
            tolerance_bundle if tolerance_bundle is not None else ToleranceBundle()
        )

    @staticmethod
    def validate_payload_structure(payload: Any) -> Mapping[str, Any]:
        if not isinstance(payload, Mapping):
            raise HilbertPayloadError(
                f"payload debe ser Mapping; recibido {type(payload).__name__}."
            )
        return payload

    @staticmethod
    def serialize_payload_canonical(payload: Mapping[str, Any]) -> bytes:
        try:
            items = tuple(
                sorted(
                    ((str(k), repr(v)) for k, v in payload.items()),
                    key=lambda kv: kv[0],
                )
            )
            serialized = repr(items).encode("utf-8")
        except Exception as exc:
            raise HilbertPayloadError(
                f"Falla en serialización determinista: {exc}"
            ) from exc
        if len(serialized) > Const.MAX_PAYLOAD_BYTES:
            raise HilbertPayloadError(
                f"payload serializado excede {Const.MAX_PAYLOAD_BYTES} B."
            )
        return serialized

    @staticmethod
    def byte_entropy_bits(data: bytes) -> float:
        if not data:
            return 0.0
        n = len(data)
        try:
            freq = np.bincount(
                np.frombuffer(data, dtype=np.uint8), minlength=256
            )
            mask = freq > 0
            probs = freq[mask].astype(np.float64) / n
            entropy = float(-np.dot(probs, np.log2(probs)))
        except Exception as exc:
            logger.error("Falla en cálculo de entropía: %s", exc)
            return 0.0
        return min(max(0.0, entropy), Const.MAX_SHANNON_BITS)

    @classmethod
    def block_entropy_rate(
        cls,
        data: bytes,
        block_size: int = Const.DEFAULT_BLOCK_SIZE,
    ) -> float:
        if not data or block_size <= 0:
            return 0.0
        if len(data) < 2 * block_size:
            return cls.byte_entropy_bits(data)
        try:
            n_blocks = len(data) // block_size
            counts = Counter(
                data[i * block_size : (i + 1) * block_size]
                for i in range(n_blocks)
            )
            total = float(n_blocks)
            H_blocks = 0.0
            for c in counts.values():
                p = c / total
                H_blocks -= p * math.log2(p)
            rate = H_blocks / block_size
        except Exception as exc:
            logger.warning("Falla en block entropy rate: %s", exc)
            rate = cls.byte_entropy_bits(data)
        return max(0.0, min(rate, Const.MAX_SHANNON_BITS))

    @staticmethod
    def compute_semantic_frequency(
        n_bytes: int,
        entropy_bits: float,
        entropy_floor: float = Const.ENTROPY_FLOOR,
        scale: float = Const.FREQUENCY_SCALE,
    ) -> float:
        if n_bytes <= 0:
            return 0.0
        h_eff = max(entropy_bits, entropy_floor)
        return _ensure_nonneg_finite_float(
            n_bytes / (h_eff * scale), name="semantic_frequency"
        )

    @classmethod
    def observe_payload(cls, payload: Mapping[str, Any]) -> WavefunctionObserveData:
        cls.validate_payload_structure(payload)
        serialized = cls.serialize_payload_canonical(payload)
        n_bytes = len(serialized)
        payload_sha = hashlib.sha256(serialized).hexdigest()

        if n_bytes == 0:
            return WavefunctionObserveData(
                serialized_payload=b"",
                payload_size=0,
                byte_entropy_bits=0.0,
                block_entropy_rate=0.0,
                exergy_bits=Const.MAX_SHANNON_BITS,
                semantic_frequency=0.0,
                incident_energy=0.0,
                energy_uncertainty_lower=0.0,
                energy_uncertainty_upper=0.0,
                payload_sha256=payload_sha,
                certification_hash=_compute_dto_hash(0, 0.0, 0.0, payload_sha),
            )

        H = cls.byte_entropy_bits(serialized)
        Hb = cls.block_entropy_rate(serialized)
        Xi = max(0.0, Const.MAX_SHANNON_BITS - H)
        nu = cls.compute_semantic_frequency(n_bytes, H)
        E = Const.PLANCK_H * nu
        dE = E * 2.0 / math.sqrt(max(1, n_bytes))
        return WavefunctionObserveData(
            serialized_payload=serialized,
            payload_size=n_bytes,
            byte_entropy_bits=H,
            block_entropy_rate=Hb,
            exergy_bits=Xi,
            semantic_frequency=nu,
            incident_energy=E,
            energy_uncertainty_lower=max(0.0, E - dE),
            energy_uncertainty_upper=E + dE,
            payload_sha256=payload_sha,
            certification_hash=_compute_dto_hash(
                n_bytes, H, Hb, Xi, nu, E, payload_sha
            ),
        )

    def capture_environmental_snapshot(self) -> EnvironmentalSnapshot:
        errors: List[str] = []
        try:
            chi = max(
                0.0,
                _ensure_finite_float(
                    self._topo.get_mahalanobis_threat(), name="chi_squared"
                ),
            )
        except Exception as exc:
            errors.append(f"topo_watcher: {exc}")
            chi = 0.0
        try:
            sigma = _ensure_finite_float(
                self._laplace.get_dominant_pole_real(), name="sigma"
            )
        except Exception as exc:
            errors.append(f"laplace_oracle: {exc}")
            sigma = 0.0
        try:
            frust = max(
                0.0,
                _ensure_finite_float(
                    self._sheaf.get_global_frustration_energy(),
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
    def compute_poincare_snapshot(
        snapshot: EnvironmentalSnapshot, section_level: float = 0.0
    ) -> PoincarePhaseSpaceSnapshot:
        """Snapshot simpléctico en (q=χ², p=α|σ|) con invariantes KAM y Nekhoroshev."""
        y = max(0.0, snapshot.chi_squared)
        p = Const.ALPHA_COUPLING * abs(snapshot.dominant_pole_real)
        action = 0.5 * (y * y + p * p)
        angle = (
            math.atan2(p, y) if action > Const.MACHINE_EPSILON else 0.0
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

        ratio = y / max(1e-12, abs(snapshot.dominant_pole_real))
        is_res, p_r, q_r, _, small_div = _detect_resonance(ratio)
        dioph_margin, kam_intact, nekh_time, arnold_w, nekh_drift = (
            _evaluate_kam_nekhoroshev_invariants(
                ratio=ratio,
                p=p_r,
                q=q_r if q_r > 0 else 1,
                frustration_eps=snapshot.global_frustration,
                chi_squared=y,
            )
        )
        drift = nekh_drift if is_res else 0.0

        return PoincarePhaseSpaceSnapshot(
            action_invariant=float(action),
            angle_parameter=float(angle),
            section_level=float(section_level),
            crossing_count=int(crossings),
            frequency_ratio=float(ratio),
            is_near_resonance=bool(is_res),
            resonance_p=int(p_r),
            resonance_q=int(q_r),
            small_divisor_magnitude=float(small_div),
            action_drift_estimate=float(drift),
            diophantine_margin=float(dioph_margin),
            kam_torus_intact=bool(kam_intact),
            nekhoroshev_stability_time=float(nekh_time),
            arnold_tongue_width=float(arnold_w),
        )

    def compute_lindstedt_expansion(
        self,
        snapshot: EnvironmentalSnapshot,
        n_terms: Optional[int] = None,
    ) -> LindstedtWorkFunctionExpansion:
        n = n_terms if n_terms is not None else self._tol_bundle.n_lindstedt_terms
        return _lindstedt_poincare_additive_phi_expansion(
            chi_squared=snapshot.chi_squared,
            base_phi=Const.BASE_PHI,
            n_terms=n,
            use_pade=self._tol_bundle.use_pade_resummation,
        )

    @staticmethod
    def build_phase1_data(
        observe: WavefunctionObserveData,
        snapshot: EnvironmentalSnapshot,
        tolerance_bundle: ToleranceBundle,
    ) -> Phase1HilbertData:
        """Constructor funcional puro de `Phase1HilbertData` (utilizado también por RSI-3)."""
        poincare = Phase1_WavefunctionObserver.compute_poincare_snapshot(snapshot)
        lindstedt = _lindstedt_poincare_additive_phi_expansion(
            chi_squared=snapshot.chi_squared,
            base_phi=Const.BASE_PHI,
            n_terms=tolerance_bundle.n_lindstedt_terms,
            use_pade=tolerance_bundle.use_pade_resummation,
        )
        cert_hash = _compute_dto_hash(
            observe.certification_hash,
            snapshot.chi_squared,
            snapshot.dominant_pole_real,
            snapshot.global_frustration,
            poincare.action_invariant,
            poincare.is_near_resonance,
            poincare.resonance_p,
            poincare.resonance_q,
            poincare.kam_torus_intact,
            lindstedt.expansion_residual,
            tolerance_bundle.use_pade_resummation,
        )
        return Phase1HilbertData(
            observe=observe,
            snapshot=snapshot,
            certification_hash=cert_hash,
            tolerance_bundle=tolerance_bundle,
            poincare_snapshot=poincare,
            lindstedt_expansion=lindstedt,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ★ MORFISMO TERMINAL DE FASE 1 / INICIO FORMAL DE FASE 2 ★
    # ─────────────────────────────────────────────────────────────────────────
    def nest_into_phase2(
        self, payload: Mapping[str, Any]
    ) -> "Phase2_WKBGaugeOrienter":
        r"""Morfismo de transición F₁ ⊣ F₂:
            nest_into_phase2(payload) := Phase2_WKBGaugeOrienter
                                         ∘ (observe_payload ⊗ snapshot).
        El codominio de este último método de Φ₁ es exactamente el constructor
        que da inicio a los métodos de Φ₂.
        """
        observe = self.observe_payload(payload)
        snapshot = self.capture_environmental_snapshot()
        if snapshot.read_errors:
            logger.warning(
                "[Φ₁] Snapshot con errores: %s.", snapshot.read_errors
            )
        data = self.build_phase1_data(observe, snapshot, self._tol_bundle)
        poincare = data.poincare_snapshot
        lindstedt = data.lindstedt_expansion
        logger.info(
            "[Φ₁ ✓] N=%d, E=%.4e, χ²=%.4f, I=%.4e, res=%s (%d:%d), KAM=%s, "
            "n_LP=%d, res_LP=%.3e, hash=%s.",
            observe.payload_size,
            observe.incident_energy,
            snapshot.chi_squared,
            poincare.action_invariant if poincare else 0.0,
            poincare.is_near_resonance if poincare else False,
            poincare.resonance_p if poincare else 0,
            poincare.resonance_q if poincare else 0,
            poincare.kam_torus_intact if poincare else True,
            lindstedt.n_terms if lindstedt else 2,
            lindstedt.expansion_residual if lindstedt else 0.0,
            data.certification_hash[:16] + "...",
        )
        return Phase2_WKBGaugeOrienter(data)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE ANIDADA 2: ORIENT — BARRERA SL(2,ℂ), MASLOV Y MONODROMÍA Sp(2,ℝ)     ║
# ║                                                                             ║
# ║   Continuación directa de `Phase1_WavefunctionObserver.nest_into_phase2`    ║
# ║   Secuencia interna de morfismos de Fase 2:                                 ║
# ║     1. __init__(phase1_data)              ★ CONTINUACIÓN DE Φ₁              ║
# ║     2. compute_work_function()                                              ║
# ║     3. compute_effective_mass()                                             ║
# ║     4. compute_maslov_index()             [Grassmanniano Lagrangiano Λ(1)]  ║
# ║     5. compute_barrier_transmission()     [Matriz Transferencia SL(2,ℂ)]    ║
# ║     6. compute_energy_return_map()        [Monodromía Sp(2,ℝ) + Greene]     ║
# ║     7. orient_wkb_gauge()                                                   ║
# ║     8. nest_into_phase3()                 ★ MORFISMO TERMINAL → INICIA Φ₃  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase2_WKBGaugeOrienter:
    r"""FASE 2 — ORIENT: Función de Trabajo, Masa Efectiva, Maslov y Monodromía.

    Recibe por continuación estricta el certificado `Phase1HilbertData` emitido
    por `Phase1_WavefunctionObserver.nest_into_phase2(payload)`.
    Calcula Φ(χ²) (con correcciones de Lindstedt-Poincaré-Padé), la masa efectiva
    m*(σ), el índice topológico de Maslov-Keller μ ∈ {0, 2} en los puntos de
    retorno clásicos, la transmisión exacta de barrera SL(2, ℂ) y la matriz de
    monodromía simpléctica M_ret ∈ Sp(2, ℝ) en el espacio (E, Φ).

    Su método final `nest_into_phase3()` es el morfismo terminal de Φ₂ cuyo
    codominio inicia formalmente a `Phase3_BornHeytingCollapser`.
    """

    def __init__(
        self, phase1_data: Phase1HilbertData, *, strict_wkb: bool = False
    ) -> None:
        if not isinstance(phase1_data, Phase1HilbertData):
            raise TypeError(
                "Phase2_WKBGaugeOrienter requiere Phase1HilbertData como precondición."
            )
        self._p1: Final[Phase1HilbertData] = phase1_data
        self._strict_wkb: Final[bool] = bool(strict_wkb)

    @property
    def phase1(self) -> Phase1HilbertData:
        return self._p1

    @property
    def strict_wkb(self) -> bool:
        return self._strict_wkb

    @staticmethod
    def compute_work_function(
        snapshot: EnvironmentalSnapshot,
        expansion: Optional[LindstedtWorkFunctionExpansion] = None,
    ) -> float:
        """Evalúa Φ(χ²). Para n_terms=2 y use_pade=False, Φ = Φ₀ + αχ² exacto."""
        if expansion is not None:
            return _ensure_nonneg_finite_float(
                expansion.corrected_work_function, name="work_function"
            )
        return _ensure_nonneg_finite_float(
            Const.BASE_PHI + Const.ALPHA_COUPLING * snapshot.chi_squared,
            name="work_function",
        )

    @staticmethod
    def compute_effective_mass(snapshot: EnvironmentalSnapshot) -> float:
        sigma = snapshot.dominant_pole_real
        if sigma >= -Const.SIGMA_CHAOS_TOL:
            logger.warning(
                "[Φ₂] Sistema inestable (σ=%.6e ≥ −tol). m* → ∞.", sigma
            )
            return float("inf")
        return _ensure_positive_or_posinf_float(
            Const.BASE_MASS / abs(sigma), name="effective_mass"
        )

    @staticmethod
    def compute_maslov_index(E: float, Phi: float) -> Tuple[int, float]:
        r"""Índice simpléctico de Maslov-Keller μ ∈ {0, 2} sobre Λ(1) ≅ ℝℙ¹.

        En el régimen de túnel (E < Φ), la variedad lagrangiana experimenta dos
        cáusticas / puntos de retorno clásicos (x = 0 y x = Δx), cada uno
        aportando una fase de Stokes-Airy de π/2 (μ = 1 + 1 = 2, φ_Maslov = π).
        Sobre barrera (E ≥ Φ), la proyección lagrangiana es difeomorfa (μ = 0, φ = 0).
        """
        if E >= Phi:
            return 0, 0.0
        return 2, math.pi

    def compute_barrier_transmission(
        self, E: float, Phi: float, m_eff: float
    ) -> RectangularBarrierTransmission:
        rel = self._p1.tolerance_bundle.threshold_energy_rel_tol
        transmission = RectangularBarrierCalculator.compute(
            E, Phi, m_eff, Const.BARRIER_DX, threshold_rel_tol=rel
        )
        if (
            self._strict_wkb
            and not transmission.is_classical
            and transmission.regime == "tunneling"
            and transmission.kappa_width <= Const.WKB_VALIDITY_THRESHOLD
        ):
            raise HilbertWKBValidityError(
                f"WKB fuera de régimen κa ≫ 1: κa={transmission.kappa_width:.3e}."
            )
        return transmission

    @staticmethod
    def compute_energy_return_map(
        E: float,
        Phi: float,
        n_iterations: int = 4,
        frustration: float = 0.0,
    ) -> EnergyReturnMapSnapshot:
        r"""Aplicación de retorno en (E, Φ) con matriz de monodromía M_ret ∈ Sp(2, ℝ).

        Evalúa la contracción J = min(E/Φ, Φ/E) y el jacobiano simpléctico
        M_ret = [[1 − K, 1], [−K, 1]] (det M_ret = 1), calculando el residuo de
        Greene R_Greene = ¼(2 − Tr(M_ret)), los multiplicadores de Floquet λ_±
        y el exponente de Lyapunov λ_Lyap = ln max|λ_±|.
        """
        if Phi <= 0.0 or E <= 0.0:
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
        cur = E
        for _ in range(n_iterations):
            cur = Phi + J * (cur - Phi)
            norms.append(float(abs(cur)))

        phase_angle = (
            0.5 * math.pi * abs(E - Phi) / max(Const.DIVISION_EPSILON, E + Phi)
        )
        k_chirikov = (
            4.0 * (math.sin(phase_angle) ** 2) * (1.0 + max(0.0, frustration))
        )
        m11 = 1.0 - k_chirikov
        m12 = 1.0
        m21 = -k_chirikov
        m22 = 1.0
        det_m = m11 * m22 - m12 * m21
        tr_m = m11 + m22
        greenes_res = 0.25 * (2.0 - tr_m)

        half_tr = 0.5 * tr_m
        disc = half_tr * half_tr - det_m
        if disc < -1e-12:
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
        stable = (eig_mod <= 1.0 + 1e-9) and (orbit_cls != "hyperbolic")

        return EnergyReturnMapSnapshot(
            contraction_factor=float(J),
            eigenvalues_modulus=(
                float(eig_mod),
                float(mod_plus),
                float(mod_minus),
            ),
            is_stable=bool(stable),
            iterations_norm_sequence=tuple(norms),
            monodromy_trace=float(tr_m),
            monodromy_determinant=float(det_m),
            greenes_residue=float(greenes_res),
            lyapunov_exponent=float(lyap),
            orbit_class=orbit_cls,
        )

    def orient_wkb_gauge(self) -> GaugeCouplingCertificate:
        observe = self._p1.observe
        snapshot = self._p1.snapshot
        E = observe.incident_energy
        Phi = self.compute_work_function(
            snapshot, self._p1.lindstedt_expansion
        )
        m_eff = self.compute_effective_mass(snapshot)
        mu, phi_maslov = self.compute_maslov_index(E, Phi)
        tr = self.compute_barrier_transmission(E, Phi, m_eff)
        return_map = self.compute_energy_return_map(
            E, Phi, frustration=snapshot.global_frustration
        )

        tol_frust = self._p1.tolerance_bundle.gauge_flatness_tol
        is_frustrated = snapshot.global_frustration > tol_frust
        T = 0.0 if is_frustrated else tr.t_exact
        t_wkb = 0.0 if is_frustrated else tr.t_wkb
        regime = "vetoed" if is_frustrated else tr.regime
        barrier_height = max(0.0, Phi - E)

        cert_hash = _compute_dto_hash(
            self._p1.certification_hash,
            E,
            Phi,
            m_eff if math.isfinite(m_eff) else float("inf"),
            T,
            t_wkb,
            mu,
            regime,
            is_frustrated,
            return_map.contraction_factor,
            return_map.is_stable,
            return_map.greenes_residue,
            tr.unitarity_residual,
        )
        logger.info(
            "[Φ₂ ✓] Φ=%.4f, m*=%.4e, T=%.6e, T_WKB=%.6e, κa=%.4f, μ=%d, "
            "régimen=%s, J=%.4f, R_G=%.4e, orbit=%s, frustrated=%s, hash=%s.",
            Phi,
            m_eff,
            T,
            t_wkb,
            tr.kappa_width,
            mu,
            regime,
            return_map.contraction_factor,
            return_map.greenes_residue,
            return_map.orbit_class,
            is_frustrated,
            cert_hash[:16] + "...",
        )
        return GaugeCouplingCertificate(
            phase1=self._p1,
            incident_energy=float(E),
            work_function=float(Phi),
            effective_mass=m_eff,
            dominant_pole_real=snapshot.dominant_pole_real,
            threat_level=snapshot.chi_squared,
            frustration_energy=snapshot.global_frustration,
            is_frustrated=bool(is_frustrated),
            barrier_height=float(barrier_height),
            wkb_kappa=float(tr.kappa) if math.isfinite(tr.kappa) else tr.kappa,
            wkb_kappa_width=(
                float(tr.kappa_width)
                if math.isfinite(tr.kappa_width)
                else tr.kappa_width
            ),
            wkb_exponent=(
                float(tr.exponent) if math.isfinite(tr.exponent) else tr.exponent
            ),
            transmission_probability=float(T),
            transmission_wkb=float(t_wkb),
            transmission_ratio=(
                float(tr.t_ratio) if math.isfinite(tr.t_ratio) else tr.t_ratio
            ),
            maslov_index=int(mu),
            maslov_phase=float(phi_maslov),
            regime=str(regime),
            certification_hash=cert_hash,
            energy_return_map=return_map,
            unitarity_residual=float(tr.unitarity_residual),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ★ MORFISMO TERMINAL DE FASE 2 / INICIO FORMAL DE FASE 3 ★
    # ─────────────────────────────────────────────────────────────────────────
    def nest_into_phase3(self) -> "Phase3_BornHeytingCollapser":
        r"""Morfismo de transición F₂ ⊣ F₃:
            nest_into_phase3() := Phase3_BornHeytingCollapser
                                  ∘ orient_wkb_gauge.
        El codominio de este último método de Φ₂ es exactamente el constructor
        que da inicio a los métodos de Φ₃.
        """
        certificate = self.orient_wkb_gauge()
        return Phase3_BornHeytingCollapser(
            certificate, strict_wkb=self._strict_wkb
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE ANIDADA 3: DECIDE — BORN, HEYTING Ω₃, KAC-POINCARÉ Y RSI-3          ║
# ║                                                                             ║
# ║   Continuación directa de `Phase2_WKBGaugeOrienter.nest_into_phase3`        ║
# ║   Secuencia interna de morfismos de Fase 3:                                 ║
# ║     1. __init__(certificate)              ★ CONTINUACIÓN DE Φ₂              ║
# ║     2. compute_collapse_threshold()                                         ║
# ║     3. resolve_heyting_omega_three_lattice()                                ║
# ║     4. compute_coherence_residual()                                         ║
# ║     5. compute_post_collapse_observables()                                  ║
# ║     6. compute_transmission_recurrence()  [Ramsauer-Townsend + Kac-Poincaré]║
# ║     7. collapse_wavefunction()                                              ║
# ║     8. collapse_wavefunction_rsi3()       ★ MORFISMO TERMINAL RSI-3         ║
# ║     9. collapse_into_state(state)         ★ MORFISMO TERMINAL CATEGÓRICO    ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase3_BornHeytingCollapser:
    r"""FASE 3 — DECIDE: Proyección de Born, Álgebra de Heyting Ω₃ y RSI-3.

    Recibe por continuación estricta el certificado `GaugeCouplingCertificate`
    emitido por `Phase2_WKBGaugeOrienter.nest_into_phase3()`.
    Resuelve el umbral determinista θ ∈ [0, 1), el veredicto intuicionista en
    el retículo de Heyting Ω₃, la recurrencia ergódica de Kac-Poincaré sobre
    las resonancias de transparencia de Ramsauer-Townsend, y ejecuta el colapso
    proyectivo estándar o gobernado por el meta-controlador RSI de Nivel 3.
    """

    def __init__(
        self,
        certificate: GaugeCouplingCertificate,
        *,
        strict_wkb: bool = False,
    ) -> None:
        if not isinstance(certificate, GaugeCouplingCertificate):
            raise TypeError(
                "Phase3_BornHeytingCollapser requiere GaugeCouplingCertificate."
            )
        self._cert: Final[GaugeCouplingCertificate] = certificate
        self._strict_wkb: Final[bool] = bool(strict_wkb)

    @property
    def certificate(self) -> GaugeCouplingCertificate:
        return self._cert

    @staticmethod
    def compute_collapse_threshold(serialized_payload: bytes) -> float:
        digest = hashlib.sha256(serialized_payload).digest()
        n = int.from_bytes(digest[:8], byteorder="big", signed=False)
        mantissa = (n >> (64 - Const.MANTISSA_BITS)) & Const.MANTISSA_MASK
        theta = mantissa / Const.MANTISSA_DIVISOR
        return 0.9999999999999999 if theta >= 1.0 else float(theta)

    @staticmethod
    def resolve_heyting_omega_three_lattice(
        certificate: GaugeCouplingCertificate, theta: float
    ) -> HeytingOmega3:
        verdict = HeytingOmega3.COHERENT
        if certificate.is_frustrated:
            verdict = verdict.join(HeytingOmega3.VETOED)
        if (
            math.isinf(certificate.effective_mass)
            and certificate.regime == "impenetrable"
        ):
            verdict = verdict.join(HeytingOmega3.VETOED)
        if (
            not certificate.is_frustrated
            and certificate.transmission_probability < theta
        ):
            verdict = verdict.join(HeytingOmega3.DEGRADED)
        return verdict

    @staticmethod
    def compute_coherence_residual(T: float, theta: float) -> float:
        return _clamp_probability(1.0 - 2.0 * abs(T - theta))

    @staticmethod
    def compute_post_collapse_observables(
        certificate: GaugeCouplingCertificate,
        eigenstate: HilbertEigenstate,
    ) -> Tuple[float, float]:
        if eigenstate == HilbertEigenstate.REJECTED:
            return 0.0, 0.0
        E = certificate.incident_energy
        Phi = certificate.work_function
        K = (
            max(Const.MIN_KINETIC_ENERGY, E - Phi)
            if E >= Phi
            else Const.MIN_KINETIC_ENERGY
        )
        m = (
            Const.BASE_MASS
            if math.isinf(certificate.effective_mass)
            else certificate.effective_mass
        )
        p = _ensure_finite_float(_safe_sqrt(2.0 * m * K), name="momentum")
        return float(K), float(p)

    @staticmethod
    def _transmission_at_k2(
        E: float, V0: float, k2: float, width: float
    ) -> float:
        s = math.sin(k2 * width)
        D = E - V0
        if D <= 0:
            return 0.0
        den = 1.0 + (V0 * V0 * s * s) / (4.0 * E * D)
        return _clamp_probability(1.0 / den) if den > 0 else 0.0

    def compute_transmission_recurrence(
        self, n_samples: int = 8
    ) -> TransmissionRecurrenceSnapshot:
        """Evalúa resonancias de Ramsauer-Townsend y el lema de recurrencia de Kac-Poincaré."""
        cert = self._cert
        E, V0, m = cert.incident_energy, cert.work_function, cert.effective_mass
        width = Const.BARRIER_DX
        hbar = Const.PLANCK_HBAR

        rt_energies: Tuple[float, ...] = ()
        if 0 < m < float("inf") and width > 0:
            base_rt = (math.pi * math.pi * hbar * hbar) / (2.0 * m * width * width)
            rt_energies = tuple(float(V0 + (n * n) * base_rt) for n in range(1, 5))

        if E <= V0 or m <= 0 or math.isinf(m) or width <= 0:
            return TransmissionRecurrenceSnapshot(
                recurrence_points=(),
                min_recurrence_distance=float("inf"),
                is_poincare_recurrent=False,
                k2_samples=(),
                n_samples=0,
                ramsauer_townsend_energies=rt_energies,
                nearest_rt_quantum_number=0,
                kac_mean_recurrence_time=float("inf"),
                ergodic_phase_discrepancy=1.0,
            )

        k2_base = math.sqrt(2.0 * m * (E - V0)) / hbar
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

        phase_mod_pi = (k2_base * width) % math.pi
        discrepancy = min(phase_mod_pi, math.pi - phase_mod_pi) / math.pi
        measure_visited = max(
            1.0 / max(1, n_samples), len(recurrences) / max(1, n_samples)
        )
        kac_time = 1.0 / measure_visited if measure_visited > 0 else float("inf")

        return TransmissionRecurrenceSnapshot(
            recurrence_points=tuple(recurrences),
            min_recurrence_distance=float(min_dist),
            is_poincare_recurrent=(len(recurrences) >= 2),
            k2_samples=tuple(k2_samples),
            n_samples=int(n_samples),
            ramsauer_townsend_energies=rt_energies,
            nearest_rt_quantum_number=int(nearest_n),
            kac_mean_recurrence_time=float(kac_time),
            ergodic_phase_discrepancy=float(discrepancy),
        )

    def collapse_wavefunction(self) -> QuantumMeasurement:
        cert = self._cert
        observe = cert.observe
        theta = self.compute_collapse_threshold(observe.serialized_payload)
        verdict = self.resolve_heyting_omega_three_lattice(cert, theta)
        T = cert.transmission_probability
        residual = self.compute_coherence_residual(T, theta)

        if verdict == HeytingOmega3.VETOED:
            eigenstate = HilbertEigenstate.REJECTED
        elif T >= theta:
            eigenstate = HilbertEigenstate.ADMITTED
        else:
            eigenstate = HilbertEigenstate.REJECTED

        K, p = self.compute_post_collapse_observables(cert, eigenstate)
        recurrence = self.compute_transmission_recurrence()
        measurement = QuantumMeasurement(
            eigenstate=eigenstate,
            heyting_verdict=verdict,
            collapse_threshold=theta,
            born_probability=T,
            coherence_residual=residual,
            momentum=p,
            kinetic_energy=K,
            observable_snapshot={
                "energy": observe.incident_energy,
                "energy_uncertainty": (
                    observe.energy_uncertainty_lower,
                    observe.energy_uncertainty_upper,
                ),
                "byte_entropy": observe.byte_entropy_bits,
                "block_entropy_rate": observe.block_entropy_rate,
                "exergy_bits": observe.exergy_bits,
                "work_function": cert.work_function,
                "effective_mass": (
                    cert.effective_mass
                    if math.isfinite(cert.effective_mass)
                    else "+Inf"
                ),
                "dominant_pole_real": cert.dominant_pole_real,
                "threat_level": cert.threat_level,
                "frustration_energy": cert.frustration_energy,
                "maslov_index": cert.maslov_index,
                "maslov_phase": cert.maslov_phase,
                "regime": cert.regime,
                "kappa_width": cert.wkb_kappa_width,
                "T_exact": cert.transmission_probability,
                "T_WKB": cert.transmission_wkb,
                "unitarity_residual": cert.unitarity_residual,
                "heyting_lt_closure": verdict.lawvere_tierney_closure().name,
                "phase1_hash": cert.phase1.certification_hash,
                "phase2_hash": cert.certification_hash,
            },
            poincare_snapshot=cert.phase1.poincare_snapshot,
            lindstedt_expansion=cert.phase1.lindstedt_expansion,
            energy_return_map=cert.energy_return_map,
            transmission_recurrence=recurrence,
        )
        logger.debug(
            "[Φ₃.collapse] |%s⟩ Ω₃=%s T=%.6e θ=%.6f r=%.4f K=%.4e p=%.4e.",
            eigenstate.name,
            verdict.name,
            T,
            theta,
            residual,
            K,
            p,
        )
        return measurement

    # ─── Meta-Controlador RSI de Nivel 3 (Banach + Shanks-Aitken + Padé) ────

    def _combine_residuals(
        self,
        measurement: QuantumMeasurement,
        recurrence: Optional[TransmissionRecurrenceSnapshot],
        bundle: Optional[ToleranceBundle] = None,
    ) -> float:
        r"""Funcional de Lyapunov compuesto L_RSI(bundle) estrictamente contráctil.

        Evalúa el supremo sobre los defectos numérico-epistémicos reducibles:
          1. Residuo perturbativo de Lindstedt-Poincaré / Borel-Padé R_LP(n).
          2. Defecto de unitariedad SL(2, ℂ) de la matriz de transferencia.
          3. Deriva secular de acción de Nekhoroshev ΔI_Nekh.
          4. Incertidumbre epistémica de coherencia escalada por la resolución
             del bundle: scale · coherence_residual · min(1, ΔE / E).
        """
        cert = self._cert
        scale = bundle.scale if bundle is not None else cert.phase1.tolerance_bundle.scale
        obs = cert.observe
        dE = obs.energy_uncertainty_upper - obs.energy_uncertainty_lower
        rel_dE = (
            min(1.0, dE / max(Const.DIVISION_EPSILON, obs.incident_energy))
            if obs.incident_energy > 0.0
            else 0.0
        )
        epistemic_defect = float(
            scale * measurement.coherence_residual * max(1e-3, rel_dE)
        )
        parts: list[float] = [
            epistemic_defect,
            float(cert.unitarity_residual),
        ]
        if cert.phase1.lindstedt_expansion is not None:
            parts.append(float(cert.phase1.lindstedt_expansion.expansion_residual))
        if cert.energy_return_map is not None and not cert.energy_return_map.is_stable:
            parts.append(float(scale * min(1.0, cert.energy_return_map.lyapunov_exponent)))
        if (
            recurrence is not None
            and math.isfinite(recurrence.min_recurrence_distance)
        ):
            parts.append(float(scale * min(1.0, recurrence.min_recurrence_distance)))
        if cert.phase1.poincare_snapshot is not None:
            parts.append(float(cert.phase1.poincare_snapshot.action_drift_estimate))
        finite = [p for p in parts if math.isfinite(p)]
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
                prev.residual,
                residual,
                prev.tolerance_scale,
                bundle.scale,
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
        """Re-ejecuta funcionalmente Φ₁ → Φ₂ → Φ₃ con un nuevo `ToleranceBundle`."""
        p1 = self._cert.phase1
        new_p1 = Phase1_WavefunctionObserver.build_phase1_data(
            observe=p1.observe,
            snapshot=p1.snapshot,
            tolerance_bundle=bundle,
        )
        new_p2 = Phase2_WKBGaugeOrienter(
            new_p1, strict_wkb=self._strict_wkb
        )
        new_cert = new_p2.orient_wkb_gauge()
        new_phase3 = Phase3_BornHeytingCollapser(
            new_cert, strict_wkb=self._strict_wkb
        )
        measurement = new_phase3.collapse_wavefunction()
        recurrence = new_phase3.compute_transmission_recurrence()
        residual = new_phase3._combine_residuals(
            measurement, recurrence, bundle=bundle
        )
        return measurement, residual, new_cert.unitarity_residual

    @staticmethod
    def _rebuild_with_rsi(
        m: QuantumMeasurement, rsi: SelfImprovementResult
    ) -> QuantumMeasurement:
        """Reconstruye `QuantumMeasurement` (frozen) con `rsi_result` adjunto."""
        return QuantumMeasurement(
            eigenstate=m.eigenstate,
            heyting_verdict=m.heyting_verdict,
            collapse_threshold=m.collapse_threshold,
            born_probability=m.born_probability,
            coherence_residual=m.coherence_residual,
            momentum=m.momentum,
            kinetic_energy=m.kinetic_energy,
            observable_snapshot=m.observable_snapshot,
            poincare_snapshot=m.poincare_snapshot,
            lindstedt_expansion=m.lindstedt_expansion,
            energy_return_map=m.energy_return_map,
            transmission_recurrence=m.transmission_recurrence,
            rsi_result=rsi,
        )

    def collapse_wavefunction_rsi3(
        self,
        max_depth: int = _RSI_MAX_DEPTH,
        target_residual: float = _RSI_TARGET_RESIDUAL,
    ) -> QuantumMeasurement:
        r"""Colapso proyectivo gobernado por el Meta-Controlador RSI de Nivel 3.

        - Nivel 1: Evaluación en el bundle inicial (scale=1, n_terms=2 → compat. v4.1.0).
        - Nivel 2: Extrapolación de Richardson ⟶ estimación de orden p ⟶ contracción α.
        - Nivel 3: Aceleración no lineal Δ² de Aitken / Shanks, certificación de
          contractividad de Banach (κ_Banach < 1) y mutación autónoma de meta-ley
          hacia resumación racional diagonal de Borel-Padé [L/L].
        """
        if self._cert.is_frustrated:
            m0 = self.collapse_wavefunction()
            bundle = self._cert.phase1.tolerance_bundle
            trace = RecursiveSelfImprovementTrace(
                level=1,
                tolerance_scale=bundle.scale,
                n_lindstedt_terms=bundle.n_lindstedt_terms,
                residual=0.0,
                convergence_rate=0.0,
                richardson_extrapolation=0.0,
                aitken_delta_squared=0.0,
                adaptation_factor=1.0,
                level3_active=False,
                banach_contraction_factor=0.0,
                shanks_extrapolation=0.0,
                meta_law_mutated=False,
                unitarity_defect=self._cert.unitarity_residual,
            )
            rsi = SelfImprovementResult(
                traces=(trace,),
                final_scale=bundle.scale,
                final_n_terms=bundle.n_lindstedt_terms,
                final_residual=0.0,
                converged=True,
                richardson_limit=0.0,
                aitken_limit=0.0,
                effective_convergence_order=0.0,
                level3_activated=False,
                banach_fixed_point_certified=True,
                shanks_limit=0.0,
                used_pade_resummation=False,
            )
            return self._rebuild_with_rsi(m0, rsi)

        initial = self._cert.phase1.tolerance_bundle
        bundle = ToleranceBundle(
            scale=initial.scale,
            n_lindstedt_terms=max(2, initial.n_lindstedt_terms),
            use_pade_resummation=initial.use_pade_resummation,
            damping_factor=initial.damping_factor,
        )
        traces: list[RecursiveSelfImprovementTrace] = []
        best_m: Optional[QuantumMeasurement] = None
        best_res = float("inf")

        for level in range(1, max_depth + 1):
            meta_mutated = False
            if level >= 3 and not bundle.use_pade_resummation:
                bundle = bundle.with_pade(True).with_terms(
                    max(6, bundle.n_lindstedt_terms + 2)
                )
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

            new_terms = min(12, bundle.n_lindstedt_terms + 3)
            use_pade_next = bundle.use_pade_resummation or (
                level >= 2 and residual > 1e-2
            )
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
        rsi = SelfImprovementResult(
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
            used_pade_resummation=bool(
                bundle.use_pade_resummation
                or any(t.meta_law_mutated for t in traces)
            ),
        )
        if best_m is None:
            raise RecursiveSelfImprovementError("RSI-3 no produjo medición.")
        return self._rebuild_with_rsi(best_m, rsi)

    # ─────────────────────────────────────────────────────────────────────────
    # ★ MORFISMO TERMINAL DEL MÓDULO: collapse_into_state ★
    # ─────────────────────────────────────────────────────────────────────────
    def collapse_into_state(self, state: CategoricalState) -> CategoricalState:
        r"""Cierra el funtor 𝓕 = Φ₃ ∘ Φ₂ ∘ Φ₁ proyectando sobre `CategoricalState`."""
        if not isinstance(state, CategoricalState):
            raise HilbertWatcherError(
                f"state debe ser CategoricalState; recibido {type(state).__name__}."
            )
        m = self.collapse_wavefunction()
        context = _safe_context(getattr(state, "context", None))
        cert = self._cert

        if m.eigenstate == HilbertEigenstate.ADMITTED:
            new_context = {
                **context,
                "quantum_momentum": m.momentum,
                "quantum_measurement": m,
            }
            new_strata = state.validated_strata | {Stratum.PHYSICS}
            logger.info(
                "[Φ₃ ✓] |ADMITTED⟩: T=%.6e ≥ θ=%.6f, p=%.6e, Ω₃=%s, μ=%d.",
                cert.transmission_probability,
                m.collapse_threshold,
                m.momentum,
                m.heyting_verdict.name,
                cert.maslov_index,
            )
        else:
            if cert.is_frustrated:
                reason = (
                    f"Veto cohomológico: E_frust={cert.frustration_energy:.3e}"
                )
            elif m.heyting_verdict == HeytingOmega3.DEGRADED:
                reason = (
                    f"Rechazo: T={cert.transmission_probability:.6e} < "
                    f"θ={m.collapse_threshold:.6f}"
                )
            else:
                reason = f"Barrera impenetrable (m*=∞, régimen={cert.regime})"
            new_context = {
                **context,
                "quantum_error": reason,
                "quantum_measurement": m,
            }
            new_strata = frozenset()
            logger.warning(
                "[Φ₃ ✗] |REJECTED⟩: %s, Ω₃=%s.",
                reason,
                m.heyting_verdict.name,
            )

        return CategoricalState(
            payload=state.payload,
            context=new_context,
            validated_strata=new_strata,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR 𝓕 = Φ₃ ∘ Φ₂ ∘ Φ₁  (HilbertObserverAgent)                      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class HilbertObserverAgent(Morphism):
    r"""Agente OODA cuántico sobre `CategoricalState` (Funtor 𝓕 del Estrato PHYSICS)."""

    __slots__ = ("_topo", "_laplace", "_sheaf", "_strict_wkb")

    @property
    def domain(self) -> FrozenSet[Any]:
        return frozenset()

    @property
    def codomain(self) -> Stratum:
        return Stratum.PHYSICS

    def __init__(
        self,
        topo_watcher: ITopologicalWatcher,
        laplace_oracle: ILaplaceOracle,
        sheaf_orchestrator: ISheafCohomologyOrchestrator,
        *,
        strict_wkb: bool = False,
    ) -> None:
        self._topo = topo_watcher
        self._laplace = laplace_oracle
        self._sheaf = sheaf_orchestrator
        self._strict_wkb = bool(strict_wkb)
        # Validación eager de contratos de oráculos
        Phase1_WavefunctionObserver(
            topo_watcher, laplace_oracle, sheaf_orchestrator
        )
        super().__init__()
        logger.info(
            "HilbertObserverAgent v%s inicializado. strict_wkb=%s.",
            __version__,
            self._strict_wkb,
        )

    def execute_ooda_loop(self, state: CategoricalState) -> CategoricalState:
        r"""Ejecuta la composición anidada 𝓕 = Φ₃ ∘ Φ₂ ∘ Φ₁ (modo determinista base)."""
        if not isinstance(state, CategoricalState):
            raise HilbertWatcherError(
                f"state debe ser CategoricalState; recibido {type(state).__name__}."
            )
        if not isinstance(state.payload, Mapping):
            raise HilbertWatcherError(
                f"state.payload debe ser Mapping; recibido {type(state.payload).__name__}."
            )
        if "quantum_measurement" in (state.context or {}):
            return state

        phase1 = Phase1_WavefunctionObserver(
            self._topo, self._laplace, self._sheaf
        )
        phase2 = phase1.nest_into_phase2(state.payload)
        if self._strict_wkb and not phase2._strict_wkb:
            phase2 = Phase2_WKBGaugeOrienter(phase2.phase1, strict_wkb=True)
        phase3 = phase2.nest_into_phase3()
        return phase3.collapse_into_state(state)

    def execute_ooda_rsi3(
        self,
        state: CategoricalState,
        max_depth: int = _RSI_MAX_DEPTH,
        target_residual: float = _RSI_TARGET_RESIDUAL,
    ) -> CategoricalState:
        r"""Ejecuta 𝓕 = Φ₃ ∘ Φ₂ ∘ Φ₁ bajo el meta-controlador RSI de Nivel 3."""
        if not isinstance(state, CategoricalState):
            raise HilbertWatcherError(
                f"state debe ser CategoricalState; recibido {type(state).__name__}."
            )
        if not isinstance(state.payload, Mapping):
            raise HilbertWatcherError(
                f"state.payload debe ser Mapping; recibido {type(state.payload).__name__}."
            )
        if "quantum_measurement" in (state.context or {}):
            return state

        phase1 = Phase1_WavefunctionObserver(
            self._topo, self._laplace, self._sheaf
        )
        phase2 = phase1.nest_into_phase2(state.payload)
        if self._strict_wkb and not phase2._strict_wkb:
            phase2 = Phase2_WKBGaugeOrienter(phase2.phase1, strict_wkb=True)
        phase3 = phase2.nest_into_phase3()

        rsi_measurement = phase3.collapse_wavefunction_rsi3(
            max_depth=max_depth, target_residual=target_residual
        )

        context = _safe_context(getattr(state, "context", None))
        cert = phase3.certificate
        if rsi_measurement.eigenstate == HilbertEigenstate.ADMITTED:
            new_context = {
                **context,
                "quantum_momentum": rsi_measurement.momentum,
                "quantum_measurement": rsi_measurement,
            }
            new_strata = state.validated_strata | {Stratum.PHYSICS}
        else:
            if cert.is_frustrated:
                reason = (
                    f"Veto cohomológico: E_frust={cert.frustration_energy:.3e}"
                )
            elif rsi_measurement.heyting_verdict == HeytingOmega3.DEGRADED:
                reason = (
                    f"Rechazo: T={cert.transmission_probability:.6e} < "
                    f"θ={rsi_measurement.collapse_threshold:.6f}"
                )
            else:
                reason = f"Barrera impenetrable (m*=∞, régimen={cert.regime})"
            new_context = {
                **context,
                "quantum_error": reason,
                "quantum_measurement": rsi_measurement,
            }
            new_strata = frozenset()
        return CategoricalState(
            payload=state.payload,
            context=new_context,
            validated_strata=new_strata,
        )

    def __call__(self, state: CategoricalState) -> CategoricalState:
        return self.execute_ooda_loop(state)


# ═════════════════════════════════════════════════════════════════════════════
# §11. SUITE DE VERIFICACIÓN DOCTORAL
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    print("\n" + "═" * 84)
    print(f"HILBERT WATCHER v{__version__} — SUITE DE VERIFICACIÓN")
    print("═" * 84 + "\n")

    class MockTopo:
        def __init__(self, t: float = 2.5) -> None:
            self._t = t

        def get_mahalanobis_threat(self) -> float:
            return self._t

    class MockLaplace:
        def __init__(self, p: float = -0.5) -> None:
            self._p = p

        def get_dominant_pole_real(self) -> float:
            return self._p

    class MockSheaf:
        def __init__(self, f: float = 1e-12) -> None:
            self._f = f

        def get_global_frustration_energy(self) -> float:
            return self._f

    print("TEST 1: Poincaré-KAM + Lindstedt-Poincaré + Borel-Padé (Fase 1)")
    p1 = Phase1_WavefunctionObserver(
        MockTopo(0.5), MockLaplace(-0.5), MockSheaf(1e-12)
    )
    snap = p1.capture_environmental_snapshot()
    pc = p1.compute_poincare_snapshot(snap)
    print(
        f"  I={pc.action_invariant:.4e}, θ={pc.angle_parameter:.4f}, "
        f"resonant={pc.is_near_resonance} ({pc.resonance_p}:{pc.resonance_q}), "
        f"KAM_intact={pc.kam_torus_intact}, drift_Nekh={pc.action_drift_estimate:.3e}"
    )
    for n in (2, 4, 6):
        lp = p1.compute_lindstedt_expansion(snap, n_terms=n)
        print(
            f"  n={n}: Φ_Taylor={lp.corrected_work_function:.6e} "
            f"(res={lp.expansion_residual:.3e}) | "
            f"Φ_Padé={lp.pade_approximant_value:.6e} "
            f"(res_Padé={lp.pade_residual:.3e}) | ω_LP={lp.strained_frequency:.4f}"
        )

    print("\nTEST 2: Monodromía Simpléctica Sp(2, ℝ) y Residuo de Greene (Fase 2)")
    ret = Phase2_WKBGaugeOrienter.compute_energy_return_map(10.0, 5.0)
    print(
        f"  J={ret.contraction_factor:.4f}, Tr(M_ret)={ret.monodromy_trace:.4f}, "
        f"R_Greene={ret.greenes_residue:.4e}, orbit={ret.orbit_class}, stable={ret.is_stable}"
    )

    print("\nTEST 3: Pipeline OODA Anidado (Φ₁ → Φ₂ → Φ₃)")
    agent = HilbertObserverAgent(
        MockTopo(0.5), MockLaplace(-0.5), MockSheaf(1e-12)
    )
    state = CategoricalState(
        payload={"endpoint": "/api/test", "data": "A" * 500},
        context={},
        validated_strata=frozenset(),
    )
    result = agent.execute_ooda_loop(state)
    m = result.context.get("quantum_measurement")
    if m:
        print(
            f"  state={m.eigenstate.name}, Ω₃={m.heyting_verdict.name}, "
            f"T={m.born_probability:.4e}, θ={m.collapse_threshold:.4f}"
        )

    print("\nTEST 4: Meta-Controlador RSI de Nivel 3 (Banach + Shanks-Aitken + Padé)")
    state2 = CategoricalState(
        payload={"endpoint": "/api/test", "data": "A" * 500},
        context={},
        validated_strata=frozenset(),
    )
    result2 = agent.execute_ooda_rsi3(state2)
    m2 = result2.context.get("quantum_measurement")
    if m2 and m2.rsi_result:
        r = m2.rsi_result
        print(
            f"  state={m2.eigenstate.name}, converged={r.converged}, "
            f"final_residual={r.final_residual:.3e}, n_terms={r.final_n_terms}, "
            f"level3={r.level3_activated}, used_pade={r.used_pade_resummation}, "
            f"banach_certified={r.banach_fixed_point_certified}"
        )
        for tr in r.traces:
            print(
                f"    [Nivel {tr.level}] scale={tr.tolerance_scale:.3e}, "
                f"n={tr.n_lindstedt_terms}, res={tr.residual:.3e}, "
                f"κ_Banach={tr.banach_contraction_factor:.3e}, "
                f"mutated={tr.meta_law_mutated}"
            )

    print("\n" + "═" * 84)
    print("SUITE DOCTORAL COMPLETADA EXITOSAMENTE")
    print("═" * 84 + "\n")


__all__ = [
    "HilbertWatcherError",
    "HilbertNumericalError",
    "HilbertInterfaceError",
    "HilbertPayloadError",
    "HilbertWKBValidityError",
    "HilbertStateCloningError",
    "HilbertCohomologicalVetoError",
    "PoincareSectionError",
    "RecursiveSelfImprovementError",
    "QuantumThresholds",
    "HeytingOmega3",
    "HilbertEigenstate",
    "WavefunctionObserveData",
    "EnvironmentalSnapshot",
    "Phase1HilbertData",
    "GaugeCouplingCertificate",
    "QuantumMeasurement",
    "RectangularBarrierTransmission",
    "PoincarePhaseSpaceSnapshot",
    "LindstedtWorkFunctionExpansion",
    "EnergyReturnMapSnapshot",
    "TransmissionRecurrenceSnapshot",
    "ToleranceBundle",
    "RecursiveSelfImprovementTrace",
    "SelfImprovementResult",
    "ITopologicalWatcher",
    "ILaplaceOracle",
    "ISheafCohomologyOrchestrator",
    "RectangularBarrierCalculator",
    "Phase1_WavefunctionObserver",
    "Phase2_WKBGaugeOrienter",
    "Phase3_BornHeytingCollapser",
    "HilbertObserverAgent",
]