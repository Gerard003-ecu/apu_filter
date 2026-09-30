# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Hilbert Watcher (Operador del Hamiltoniano de Medición)             ║
║ Ruta   : app/agents/aleph/hilbert_watcher.py                                 ║
║ Versión: 4.1.0-Aleph-OODA-WKB-Maslov-Heyting-Handoff-Nested-Strict-PhD       ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y COLAPSO DE ONDA EN EL ESTRATO ALEPH (ℵ₀)
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra la aduana cuántico-informacional primaria del sistema,
operando formalmente como el Funtor de Medición Coherente:

                     𝓕 : Superposition ⟶ Eigenstate

Habita estrictamente en el Estrato Aleph (ℵ₀, Nivel 4), constituyendo el vacío
topológico y la capa límite termodinámica que precede a la variedad
diferenciable de la base física (V_ℙ). Su mandato axiomático es ejecutar el
colapso determinista de la función de onda semántica asociada a los payloads
y solicitudes entrantes, aniquilando el caos estocástico y la redundancia
sintáctica («fango informativo») antes de que exciten los motores analíticos.

A través de la cuantización de la energía semántica, el modelado de
penetración de barrera (T exacta rectangular ∧ WKB diagnóstico) y la lógica
intuicionista sobre retículos, el agente actúa como un filtro pasabajo
infranqueable de-confinado en el software.

Política de unidades: h = 1,  ℏ = h/(2π).  Entropía en bits (log₂).
Φ(χ²) = Φ₀ + α·χ²  (acoplamiento aditivo; distinto del portal ALEPH exponencial).

INVARIANTES MATEMÁTICOS, GEOMÉTRICOS Y LEYES DE CONSERVACIÓN
────────────────────────────────────────────────────────────────────────────────
  [I1] Espectro binario de proyección hermítica (Born):
       P = P† ∧ P² = P ⟹ σ(P) ⊆ {0, 1};  p_i = ‖P_i ψ‖² / ‖ψ‖².
       Colapso Born-like DETERMINISTA: T_exact ≥ θ, θ = Φ_SHA256 ∈ [0, 1).

  [I2] Conservación de información y confinamiento de exergía:
       Ξ = H_max − H(X) ≥ 0  (H_max = 8 bits/byte);
       K_max = max(E − Φ, ε) ≥ 0.

  [I3] No-demolición cuántica:
       [H, O_api] = 0 — el operador de lectura no perturba el Hamiltoniano basal.
       Realizado por DTOs frozen + A5 (no-clonación).

  [I4] Transmisión exacta de barrera rectangular + diagnóstico WKB:
       E < Φ :  T = [1 + Φ² sinh²(κa) / (4 E (Φ−E))]⁻¹,
                κ = √(2m*(Φ−E))/ℏ.
       E = Φ :  T = [1 + m* Φ a² / (2 ℏ²)]⁻¹.
       E > Φ :  T = [1 + Φ² sin²(k₂ a) / (4 E (E−Φ))]⁻¹,
                k₂ = √(2m*(E−Φ))/ℏ.
       WKB (testigo de barrera gruesa, NUNCA sustituto de T_exact):
                T_WKB ≈ exp(−2 κ a)  (E < Φ),  T_WKB = 1 (E ≥ Φ).
       Índice de Maslov μ ∈ {0, 2} (0 clásico / 2 túnel rectangular).
       El potencial rectangular es DISCONTINUO: WKB no es uniformemente válido.

  [I5] Monotonicidad causal de la filtración (clausura transitiva DIKW):
       V_ℵ₀ ⊊ V_ℙ ⊊ V_𝕋 ⊊ V_𝕊 ⊊ V_𝕎.

  [A5] No-clonación categórica:
       ∄ U ∈ U(ℋ ⊗ ℋ): U(|ψ⟩ ⊗ |0⟩) = |ψ⟩ ⊗ |ψ⟩.
       __copy__/__deepcopy__ → HilbertStateCloningError.

ARQUITECTURA DE TRES FASES ANIDADAS (Handoff por Constructor Estricto)
────────────────────────────────────────────────────────────────────────────────
El tipo de retorno del último método de Φᵢ ES el objeto inicial de Φᵢ₊₁:

  Fase 1 ──► OBSERVE — ANÁLISIS ESPECTRAL Y EXERGÍA DE SHANNON
             (Phase1_WavefunctionObserver)
             Sanea el payload, calcula H(X) bitewise y Ĥ_b (entropy rate de
             bloques), deriva E = hν con incertidumbre Poisson 2/√N.
             Morfismo terminal: nest_into_phase2
                 payload  ⟶  Phase2_WKBGaugeOrienter

  Fase 2 ──► ORIENT — PENETRACIÓN EXACTA/WKB Y ACOPLAMIENTO DE GAUGE
             (Phase2_WKBGaugeOrienter)
             ★ INICIO FORMAL = continuación de nest_into_phase2 ★
             Snapshot ambiental único, Φ(χ²), m*(σ), Maslov μ, T_exact ∧ T_WKB.
             R1: m*=∞ ∧ E ≥ Φ ⟹ T=1 (clásico).  R2: m*=∞ ∧ E < Φ ⟹ T=0.
             Morfismo terminal: nest_into_phase3
                 Phase2  ⟶  Phase3_BornHeytingCollapser

  Fase 3 ──► DECIDE & ACT — COLAPSO BORN Y VETO HEYTING
             (Phase3_BornHeytingCollapser)
             ★ INICIO FORMAL = continuación de nest_into_phase3 ★
             θ ∈ [0, 1) (53 bits de mantisa), Ω₃, K = E−Φ (E ya no se pierde),
             medición proyectiva y colapso del CategoricalState.
             Morfismo terminal: collapse_into_state
                 (Phase3, state)  ⟶  CategoricalState

RETÍCULO HEYTING Ω₃ (álgebra de Gödel–Dummett)
────────────────────────────────────────────────────────────────────────────────
  Ω₃ = { COHERENT := ⊥ = 0, DEGRADED := 1, VETOED := ⊤ = 2 }
  Orden: COHERENT < DEGRADED < VETOED
  Join ⊔ = max, Meet ⊓ = min
  Implicación: a → b = ⊤ si a ≤ b,  a → b = b si a > b
  Negación: ¬a = a → ⊥

  Colapso (precedencia estricta, join monótono):
      1. is_frustrated              ⟹ VETOED (⊤)
      2. m* = ∞ ∧ E < Φ             ⟹ VETOED
      3. T < θ ∧ ¬frustrated        ⟹ DEGRADED
      4. en otro caso               ⟹ COHERENT (⊥)

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
from typing import Any, Dict, Final, FrozenSet, List, Mapping, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from app.core.mic_algebra import CategoricalState, Morphism
from app.core.schemas import Stratum

logger = logging.getLogger("MIC.Agents.HilbertWatcher")

__version__: Final[str] = (
    "4.1.0-Aleph-OODA-WKB-Maslov-Heyting-Handoff-Nested-Strict-PhD"
)

_NON_CLONING_UID_COUNTER = _itertools_count(1)


# ═════════════════════════════════════════════════════════════════════════════
# §1. EXCEPCIONES
# ═════════════════════════════════════════════════════════════════════════════


class HilbertWatcherError(Exception):
    """Excepción base del observador Hilbert."""


class HilbertNumericalError(HilbertWatcherError):
    """Fallo numérico: infinitud, NaN, rango fuera de contrato."""


class HilbertInterfaceError(HilbertWatcherError):
    """Fallo de contrato en dependencias inyectadas."""


class HilbertPayloadError(HilbertWatcherError):
    """Fallo en validación o serialización del payload."""


class HilbertWKBValidityError(HilbertNumericalError):
    """Aproximación WKB fuera del régimen de barrera gruesa (κa ≲ 1)."""


class HilbertStateCloningError(HilbertWatcherError):
    """A5: intento de clonado de una medición cuántica inmutable."""


class HilbertCohomologicalVetoError(HilbertWatcherError):
    """Veto estructural por frustración cohomológica activa."""


# ═════════════════════════════════════════════════════════════════════════════
# §2. CONSTANTES FÍSICAS DISCRETIZADAS
# ═════════════════════════════════════════════════════════════════════════════


class QuantumThresholds:
    """Constantes normalizadas del hiperespacio de información.

    Unidades: h = 1, ℏ = h/(2π). Entropía en bits. Clase sellada.
    """

    PLANCK_H: Final[float] = 1.0
    PLANCK_HBAR: Final[float] = PLANCK_H / (2.0 * math.pi)

    BASE_PHI: Final[float] = 10.0
    BASE_MASS: Final[float] = 1.0
    BARRIER_DX: Final[float] = 1.0
    ALPHA_COUPLING: Final[float] = 5.0  # Φ = Φ₀ + α χ² (aditivo)

    EPSILON_MACH: Final[float] = 1e-9
    MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)
    ENTROPY_FLOOR: Final[float] = 1e-12
    MIN_KINETIC_ENERGY: Final[float] = 1e-12
    SIGMA_CHAOS_TOL: Final[float] = 1e-9
    EXP_UNDERFLOW_CUTOFF: Final[float] = -700.0
    EXP_OVERFLOW_CUTOFF: Final[float] = math.log(sys.float_info.max) - 1.0
    DIVISION_EPSILON: Final[float] = 1e-15

    FREQUENCY_SCALE: Final[float] = 1000.0
    MAX_PAYLOAD_BYTES: Final[int] = 10_485_760  # 10 MiB
    WKB_VALIDITY_THRESHOLD: Final[float] = 1.0
    THRESHOLD_ENERGY_REL_TOL: Final[float] = 1e-12
    SINH_ASYMPTOTIC_KA: Final[float] = 20.0
    FLOAT_MAX: Final[float] = sys.float_info.max

    MANTISSA_BITS: Final[int] = 53
    MANTISSA_MASK: Final[int] = (1 << MANTISSA_BITS) - 1
    MANTISSA_DIVISOR: Final[float] = float(1 << MANTISSA_BITS)

    DEFAULT_BLOCK_SIZE: Final[int] = 8  # 64 bits
    MAX_SHANNON_BITS: Final[float] = 8.0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError(
            f"La clase {cls.__name__} está sellada. No se permite herencia."
        )


Const = QuantumThresholds


# ═════════════════════════════════════════════════════════════════════════════
# §3. RETÍCULO DISTRIBUTIVO DE HEYTING Ω₃ (Gödel–Dummett)
# ═════════════════════════════════════════════════════════════════════════════


class HeytingOmega3(IntEnum):
    r"""Retículo distributivo de Heyting Ω₃ totalmente ordenado.

        Ω₃ = { COHERENT := ⊥ = 0, DEGRADED := 1, VETOED := ⊤ = 2 }
        Orden: COHERENT < DEGRADED < VETOED.
        Join ⊔ = max, Meet ⊓ = min.
        Implicación de Gödel: a → b = ⊤ si a ≤ b, a → b = b si a > b.
        ¬a = a → ⊥.

    Tabla de a → b sobre {0, 1, 2}:

            b\\a   0   1   2
             0     2   0   0
             1     2   2   1
             2     2   2   2
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    def __le__(self, other: "HeytingOmega3") -> bool:
        return int(self) <= int(other)

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """Supremo en Ω₃: máximo."""
        return HeytingOmega3(max(int(self), int(other)))

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """Ínfimo en Ω₃: mínimo."""
        return HeytingOmega3(min(int(self), int(other)))

    def implication(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """Implicación de Gödel: a → b = ⊤ si a ≤ b, else b."""
        if int(self) <= int(other):
            return HeytingOmega3.VETOED
        return other

    def negation(self) -> "HeytingOmega3":
        """Negación de Heyting: ¬a = a → ⊥."""
        return self.implication(HeytingOmega3.COHERENT)


# ═════════════════════════════════════════════════════════════════════════════
# §4. AUTOESTADOS DEL OPERADOR DE MEDICIÓN
# ═════════════════════════════════════════════════════════════════════════════


class HilbertEigenstate(Enum):
    """Autoestados de Ĥ sobre ℋ₂ = span{|ADMITTED⟩, |REJECTED⟩}.

    ⟨ADMITTED|REJECTED⟩ = 0, ⟨n|n⟩ = 1, P_adm + P_rej = 𝟙.
    """

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
# §5. DTOs DEL HANDOFF ENTRE FASES
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True, eq=False)
class WavefunctionObserveData:
    """Certificado espectral de Φ₁. Ladrillo de Phase1HilbertData.

    H(X) bitewise ∈ [0, 8] bits.  Ξ = 8 − H(X).  E = h ν ± δE, δE/E = 2/√N.
    """

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
        raise HilbertStateCloningError(
            "A5: WavefunctionObserveData es inmutable y no clonable."
        )

    def __deepcopy__(self, memo: Dict[int, Any]) -> "WavefunctionObserveData":
        raise HilbertStateCloningError(
            "A5: WavefunctionObserveData es inmutable y no clonable."
        )


@dataclass(frozen=True, slots=True)
class EnvironmentalSnapshot:
    """Snapshot consistente de los oráculos al inicio de Φ₂ (una sola lectura)."""

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
    """DTO inmutable de Φ₁. Precondición constructora de Φ₂.

    Empaqueta el certificado espectral y el snapshot ambiental congelado.
    """

    observe: WavefunctionObserveData
    snapshot: EnvironmentalSnapshot
    certification_hash: str


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
    """Certificado de Φ₂. Precondición constructora de Φ₃.

    T_exact es el observable de Born; T_WKB es diagnóstico de barrera gruesa.
    incident_energy se CONSERVA explícitamente (bug v4.0: se perdía si E ≥ Φ).
    """

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
    transmission_probability: float  # T_exact (Born)
    transmission_wkb: float
    transmission_ratio: float
    maslov_index: int
    maslov_phase: float
    regime: str
    certification_hash: str

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
        raise HilbertStateCloningError(
            "A5: GaugeCouplingCertificate es inmutable y no clonable."
        )

    def __deepcopy__(self, memo: Dict[int, Any]) -> "GaugeCouplingCertificate":
        raise HilbertStateCloningError(
            "A5: GaugeCouplingCertificate es inmutable y no clonable."
        )


@dataclass(frozen=True, slots=True, eq=False)
class QuantumMeasurement:
    """Certificado de Φ₃ (DECIDE). Medición proyectiva inmutable (A5)."""

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
    """T_exact (tres regímenes) + diagnóstico WKB."""

    t_exact: float
    t_wkb: float
    t_ratio: float
    kappa_width: float
    kappa: float
    exponent: float
    is_classical: bool
    regime: str


# ═════════════════════════════════════════════════════════════════════════════
# §6. PROTOCOLOS DE INTERFAZ
# ═════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class ITopologicalWatcher(Protocol):
    """Observador topológico: χ² ∈ [0, +∞)."""

    def get_mahalanobis_threat(self) -> float: ...


@runtime_checkable
class ILaplaceOracle(Protocol):
    """Oráculo espectral: σ = Re(polo dominante) ∈ ℝ."""

    def get_dominant_pole_real(self) -> float: ...


@runtime_checkable
class ISheafCohomologyOrchestrator(Protocol):
    """Orquestador cohomológico: E_frust ∈ [0, +∞)."""

    def get_global_frustration_energy(self) -> float: ...


# ═════════════════════════════════════════════════════════════════════════════
# §7. MORFISMOS NUMÉRICOS AUXILIARES
# ═════════════════════════════════════════════════════════════════════════════


def _ensure_finite_float(value: Any, *, name: str, allow_inf: bool = False) -> float:
    """Morfismo Any → ℝ_finito; rechaza NaN (e ±∞ salvo allow_inf)."""
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
    """Proyección ℝ → [0, 1] con saneo IEEE-754."""
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
    """Firma determinista SHA-256 de campos heterogéneos."""
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


def _validate_oracle_dependency(obj: Any, method_name: str, obj_name: str) -> None:
    if obj is None:
        raise HilbertInterfaceError(f"{obj_name} es None.")
    if not hasattr(obj, method_name):
        raise HilbertInterfaceError(
            f"{obj_name} (tipo={type(obj).__name__}) carece de '{method_name}'."
        )
    if not callable(getattr(obj, method_name)):
        raise HilbertInterfaceError(f"{obj_name}.{method_name} no es invocable.")


# ═════════════════════════════════════════════════════════════════════════════
# §8. TRANSMISIÓN EXACTA DE BARRERA RECTANGULAR + WKB
# ═════════════════════════════════════════════════════════════════════════════


class RectangularBarrierCalculator:
    """T_exact (Griffiths QM §2.5) en tres regímenes + diagnóstico WKB.

    Estabilidad: |E−V₀| ≤ ε_rel → umbral; κa ≥ 20 → forma exp(−2κa)/(e^{−2κa}+C).
    """

    @staticmethod
    def _threshold_transmission(V0: float, m_eff: float, width: float) -> float:
        r"""T(E=V₀) = [1 + m* V₀ a² / (2 ℏ²)]⁻¹."""
        hbar = Const.PLANCK_HBAR
        xi = m_eff * V0 * width * width / (2.0 * hbar * hbar)
        if not math.isfinite(xi) or xi >= Const.FLOAT_MAX / 2.0:
            return 0.0
        return 1.0 / (1.0 + max(xi, 0.0))

    @staticmethod
    def _tunnel_transmission_stable(E: float, V0: float, kappa_a: float) -> float:
        r"""T_túnel estable, incluida asintótica κa ≫ 1.

        C = V₀²/(16 E Δ),  T = e^{−2κa} / (e^{−2κa} + C)  (κa grande).
        """
        if E <= 0.0:
            return 0.0
        Delta = V0 - E
        if Delta <= 0.0:
            return 1.0
        if kappa_a >= Const.SINH_ASYMPTOTIC_KA:
            C = (V0 * V0) / (16.0 * E * Delta)
            exp_m = _safe_exp(-2.0 * kappa_a)
            denom = exp_m + C
            if denom <= 0.0 or not math.isfinite(denom):
                return 0.0
            return _clamp_probability(exp_m / denom)
        sinh_val = _safe_sinh(kappa_a)
        if math.isinf(sinh_val) or abs(sinh_val) >= math.sqrt(Const.FLOAT_MAX) / 2.0:
            C = (V0 * V0) / (16.0 * E * Delta)
            exp_m = _safe_exp(-2.0 * kappa_a)
            denom = exp_m + C
            return _clamp_probability(exp_m / denom) if denom > 0 else 0.0
        denom = 1.0 + (V0 * V0 * sinh_val * sinh_val) / (4.0 * E * Delta)
        if denom <= 0.0 or not math.isfinite(denom):
            return 0.0
        return _clamp_probability(1.0 / denom)

    @staticmethod
    def _over_barrier_transmission(
        E: float, V0: float, m_eff: float, width: float,
    ) -> float:
        r"""T(E>V₀) = [1 + V₀² sin²(k₂ a) / (4 E (E−V₀))]⁻¹."""
        Delta = E - V0
        if Delta <= 0.0:
            return 1.0
        k2 = _safe_sqrt(2.0 * m_eff * Delta) / Const.PLANCK_HBAR
        s = math.sin(k2 * width)
        denom = 1.0 + (V0 * V0 * s * s) / (4.0 * E * Delta)
        if denom <= 0.0 or not math.isfinite(denom):
            return 0.0
        return _clamp_probability(1.0 / denom)

    @classmethod
    def compute(
        cls,
        E: float,
        V0: float,
        m_eff: float,
        width: float,
    ) -> RectangularBarrierTransmission:
        E = _ensure_finite_float(E, name="E_barrier")
        V0 = _ensure_finite_float(V0, name="V0_barrier")
        m_eff = _ensure_finite_float(m_eff, name="m_eff_barrier", allow_inf=True)
        width = _ensure_finite_float(width, name="width_barrier")
        if E < 0 or V0 < 0 or width <= 0:
            raise HilbertNumericalError(
                f"Parámetros no físicos: E={E}, V₀={V0}, a={width}."
            )
        rel_tol = Const.THRESHOLD_ENERGY_REL_TOL * max(1.0, abs(V0), abs(E))

        if math.isinf(m_eff):
            if E + rel_tol < V0:
                return RectangularBarrierTransmission(
                    t_exact=0.0, t_wkb=0.0, t_ratio=float("inf"),
                    kappa_width=float("inf"), kappa=float("inf"),
                    exponent=float("-inf"), is_classical=False, regime="impenetrable",
                )
            regime = "over_barrier" if E > V0 + rel_tol else "threshold"
            return RectangularBarrierTransmission(
                t_exact=1.0, t_wkb=1.0, t_ratio=1.0,
                kappa_width=0.0, kappa=0.0, exponent=0.0,
                is_classical=True, regime=regime,
            )

        if m_eff <= 0:
            raise HilbertNumericalError(f"Masa efectiva debe ser > 0: {m_eff}.")

        if abs(E - V0) <= rel_tol:
            t_th = cls._threshold_transmission(V0, m_eff, width)
            return RectangularBarrierTransmission(
                t_exact=t_th, t_wkb=1.0, t_ratio=t_th,
                kappa_width=0.0, kappa=0.0, exponent=0.0,
                is_classical=True, regime="threshold",
            )

        if E > V0:
            t_ex = cls._over_barrier_transmission(E, V0, m_eff, width)
            return RectangularBarrierTransmission(
                t_exact=t_ex, t_wkb=1.0, t_ratio=t_ex,
                kappa_width=0.0, kappa=0.0, exponent=0.0,
                is_classical=True, regime="over_barrier",
            )

        Delta = V0 - E
        integrand = _safe_sqrt(2.0 * m_eff * Delta)
        kappa = integrand / Const.PLANCK_HBAR
        kappa_a = kappa * width
        raw_exponent = -2.0 * kappa_a
        t_exact = cls._tunnel_transmission_stable(E, V0, kappa_a)
        t_wkb = _clamp_probability(_safe_exp(raw_exponent))
        if t_wkb > Const.DIVISION_EPSILON:
            t_ratio = t_exact / t_wkb
        elif t_exact > Const.DIVISION_EPSILON:
            t_ratio = float("inf")
        else:
            t_ratio = 1.0
        return RectangularBarrierTransmission(
            t_exact=_clamp_probability(t_exact),
            t_wkb=t_wkb,
            t_ratio=t_ratio,
            kappa_width=kappa_a,
            kappa=kappa,
            exponent=raw_exponent,
            is_classical=False,
            regime="tunneling",
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 1: OBSERVE — ANÁLISIS ESPECTRAL Y EXERGÍA DE SHANNON                 ║
# ║                                                                             ║
# ║   Marco formal:                                                             ║
# ║   ─────────                                                                 ║
# ║   Payload ↦ bytes canónicos ↦ H(X) bitewise ∧ Ĥ_b (entropy rate).           ║
# ║   ν = N / max(H, ε) / scale,  E = h ν,  δE/E = 2/√N (Poisson+SE entropía).  ║
# ║   Snapshot ambiental único (χ², σ, E_frust) congelado para Φ₂.              ║
# ║                                                                             ║
# ║   ULTIMO MÉTODO DE FASE 1 (morfismo terminal = unidad de Fase 2):            ║
# ║       nest_into_phase2(payload) → Phase2_WKBGaugeOrienter                    ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase1_WavefunctionObserver:
    r"""FASE 1 — OBSERVE: espectro, exergía y snapshot ambiental.

    Cadena interna:
        _validate_oracle_contracts
            → validate_payload_structure
            → serialize_payload_canonical
            → byte_entropy_bits
            → block_entropy_rate
            → compute_semantic_frequency
            → observe_payload                  (pre-terminal)
            → capture_environmental_snapshot
            → nest_into_phase2                 ★ morfismo terminal = unidad de Φ₂
    """

    def __init__(
        self,
        topo_watcher: ITopologicalWatcher,
        laplace_oracle: ILaplaceOracle,
        sheaf_orchestrator: ISheafCohomologyOrchestrator,
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

    # ─────────────────────────────────────────────────────────────────────
    # 1.1 Validación y serialización canónica
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def validate_payload_structure(payload: Any) -> Mapping[str, Any]:
        if not isinstance(payload, Mapping):
            raise HilbertPayloadError(
                f"payload debe ser Mapping; recibido {type(payload).__name__}."
            )
        return payload

    @staticmethod
    def serialize_payload_canonical(payload: Mapping[str, Any]) -> bytes:
        """Serializa con ordenamiento lexicográfico determinista (UTF-8)."""
        try:
            ordered_items = tuple(
                sorted(
                    ((str(k), repr(v)) for k, v in payload.items()),
                    key=lambda kv: kv[0],
                )
            )
            serialized = repr(ordered_items).encode("utf-8")
        except Exception as exc:
            raise HilbertPayloadError(
                f"Falla en serialización determinista: {exc}"
            ) from exc
        if len(serialized) > Const.MAX_PAYLOAD_BYTES:
            raise HilbertPayloadError(
                f"payload serializado excede {Const.MAX_PAYLOAD_BYTES} B: "
                f"{len(serialized)} B."
            )
        return serialized

    # ─────────────────────────────────────────────────────────────────────
    # 1.2 Entropía bitewise H(X) ∈ [0, 8] bits  (SIN lru_cache: payloads ≤ 10 MiB)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def byte_entropy_bits(data: bytes) -> float:
        r"""H(X) = −Σ pᵢ log₂(pᵢ). Convención 0·log₂(0) = 0."""
        if not data:
            return 0.0
        n = len(data)
        try:
            frequencies = np.bincount(
                np.frombuffer(data, dtype=np.uint8), minlength=256
            )
            mask = frequencies > 0
            probabilities = frequencies[mask].astype(np.float64) / n
            entropy = float(-np.dot(probabilities, np.log2(probabilities)))
        except Exception as exc:
            logger.error("Falla en cálculo de entropía: %s", exc)
            return 0.0
        if entropy < 0.0:
            entropy = 0.0
        return min(entropy, Const.MAX_SHANNON_BITS)

    # ─────────────────────────────────────────────────────────────────────
    # 1.3 Entropy rate de bloques Ĥ_b ≈ H(B)/b  (cota superior de h_μ)
    # ─────────────────────────────────────────────────────────────────────
    @classmethod
    def block_entropy_rate(
        cls,
        data: bytes,
        block_size: int = Const.DEFAULT_BLOCK_SIZE,
    ) -> float:
        r"""Ĥ_b(X) = H(B)/b bits/byte. Cota superior de la entropy rate h_μ.

        No es H(B_n | B_{<n}) (eso exigiría un modelo de Markov); es la
        entropía i.i.d. de bloques de longitud b, que cumple h_μ ≤ H(B)/b.
        """
        if not data or block_size <= 0:
            return 0.0
        if len(data) < 2 * block_size:
            return cls.byte_entropy_bits(data)
        try:
            n_blocks = len(data) // block_size
            counts = Counter(
                data[i * block_size : (i + 1) * block_size] for i in range(n_blocks)
            )
            total = float(n_blocks)
            entropy_blocks = 0.0
            for count in counts.values():
                p = count / total
                entropy_blocks -= p * math.log2(p)
            rate = entropy_blocks / block_size
        except Exception as exc:
            logger.warning("Falla en block entropy rate: %s", exc)
            rate = cls.byte_entropy_bits(data)
        return max(0.0, min(rate, Const.MAX_SHANNON_BITS))

    # ─────────────────────────────────────────────────────────────────────
    # 1.4 Frecuencia semántica ν = N / max(H, ε) / scale
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def compute_semantic_frequency(
        n_bytes: int,
        entropy_bits: float,
        entropy_floor: float = Const.ENTROPY_FLOOR,
        scale: float = Const.FREQUENCY_SCALE,
    ) -> float:
        """Payloads grandes de baja entropía ⟹ ν alta (oscilación repetitiva)."""
        if n_bytes <= 0:
            return 0.0
        h_eff = max(entropy_bits, entropy_floor)
        nu = n_bytes / (h_eff * scale)
        return _ensure_nonneg_finite_float(nu, name="semantic_frequency")

    # ─────────────────────────────────────────────────────────────────────
    # 1.5 Observe (pre-terminal de Φ₁)
    # ─────────────────────────────────────────────────────────────────────
    @classmethod
    def observe_payload(cls, payload: Mapping[str, Any]) -> WavefunctionObserveData:
        r"""Certifica el espectro de Shannon y la energía incidente.

        Cadena:
            payload ──(validate)──▶ Mapping
                    ──(serialize)──▶ bytes
                    ──(H bitewise ∧ Ĥ_b)──▶ bits
                    ──(ν, E, δE=2E/√N)──▶ WavefunctionObserveData
        """
        cls.validate_payload_structure(payload)
        serialized = cls.serialize_payload_canonical(payload)
        n_bytes = len(serialized)
        payload_sha = hashlib.sha256(serialized).hexdigest()

        if n_bytes == 0:
            cert_hash = _compute_dto_hash(0, 0.0, 0.0, payload_sha)
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
                certification_hash=cert_hash,
            )

        entropy_bits = cls.byte_entropy_bits(serialized)
        block_rate = cls.block_entropy_rate(serialized)
        exergy_bits = max(0.0, Const.MAX_SHANNON_BITS - entropy_bits)
        nu = cls.compute_semantic_frequency(n_bytes, entropy_bits)
        energy = Const.PLANCK_H * nu
        delta_energy = energy * 2.0 / math.sqrt(max(1, n_bytes))
        lower = max(0.0, energy - delta_energy)
        upper = energy + delta_energy
        cert_hash = _compute_dto_hash(
            n_bytes, entropy_bits, block_rate, exergy_bits, nu, energy, payload_sha
        )
        logger.debug(
            "[Φ₁.observe] N=%d, H=%.4f bits, Ĥ_b=%.4f, Ξ=%.4f, ν=%.6e, "
            "E=%.6e ± %.3e.",
            n_bytes, entropy_bits, block_rate, exergy_bits, nu, energy, delta_energy,
        )
        return WavefunctionObserveData(
            serialized_payload=serialized,
            payload_size=n_bytes,
            byte_entropy_bits=entropy_bits,
            block_entropy_rate=block_rate,
            exergy_bits=exergy_bits,
            semantic_frequency=nu,
            incident_energy=energy,
            energy_uncertainty_lower=lower,
            energy_uncertainty_upper=upper,
            payload_sha256=payload_sha,
            certification_hash=cert_hash,
        )

    # ─────────────────────────────────────────────────────────────────────
    # 1.6 Snapshot ambiental (lectura única)
    # ─────────────────────────────────────────────────────────────────────
    def capture_environmental_snapshot(self) -> EnvironmentalSnapshot:
        """Lee (χ², σ, E_frust) UNA sola vez para consistencia temporal."""
        read_errors: List[str] = []
        try:
            chi_raw = self._topo.get_mahalanobis_threat()
            chi = max(0.0, _ensure_finite_float(chi_raw, name="chi_squared"))
        except Exception as exc:
            read_errors.append(f"topo_watcher: {exc}")
            chi = 0.0
        try:
            sigma_raw = self._laplace.get_dominant_pole_real()
            sigma = _ensure_finite_float(sigma_raw, name="sigma")
        except Exception as exc:
            read_errors.append(f"laplace_oracle: {exc}")
            sigma = 0.0
        try:
            frust_raw = self._sheaf.get_global_frustration_energy()
            frust = max(0.0, _ensure_finite_float(frust_raw, name="frustration"))
        except Exception as exc:
            read_errors.append(f"sheaf_orchestrator: {exc}")
            frust = 0.0
        snap = EnvironmentalSnapshot(
            chi_squared=chi,
            dominant_pole_real=sigma,
            global_frustration=frust,
            read_errors=tuple(read_errors),
        )
        logger.debug(
            "[Φ₁.snapshot] χ²=%.4f, σ=%.6f, E_frust=%.4e, errors=%d.",
            chi, sigma, frust, len(read_errors),
        )
        return snap

    # ─────────────────────────────────────────────────────────────────────
    # 1.7 ★ MORFISMO TERMINAL DE FASE 1 ★
    #     Tipo de retorno = objeto inicial de la FASE 2.
    #     last(Φ₁) = unit(Φ₂) = Phase2_WKBGaugeOrienter.
    # ─────────────────────────────────────────────────────────────────────
    def nest_into_phase2(
        self, payload: Mapping[str, Any],
    ) -> "Phase2_WKBGaugeOrienter":
        r"""★ MORFISMO TERMINAL DE FASE 1 / UNIDAD DE LA FASE 2 ★

        Composición estricta F₁ ⊣ F₂:
            nest_into_phase2(payload)  :=  Phase2_WKBGaugeOrienter
                                           ∘ (observe_payload ⊗ snapshot).

        El orienter WKB nace ya alimentado con Phase1HilbertData;
        su constructor ES la continuación formal de este método.
        """
        observe = self.observe_payload(payload)
        snapshot = self.capture_environmental_snapshot()
        if snapshot.read_errors:
            logger.warning(
                "[Φ₁] Snapshot con errores de lectura: %s.", snapshot.read_errors
            )
        cert_hash = _compute_dto_hash(
            observe.certification_hash,
            snapshot.chi_squared,
            snapshot.dominant_pole_real,
            snapshot.global_frustration,
        )
        logger.info(
            "[Φ₁ ✓] Phase1HilbertData: N=%d, E=%.4e, χ²=%.4f, hash=%s.",
            observe.payload_size, observe.incident_energy,
            snapshot.chi_squared, cert_hash[:16] + "...",
        )
        data = Phase1HilbertData(
            observe=observe, snapshot=snapshot, certification_hash=cert_hash
        )
        return Phase2_WKBGaugeOrienter(data)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 2: ORIENT — PENETRACIÓN EXACTA/WKB Y ACOPLAMIENTO DE GAUGE           ║
# ║                                                                             ║
# ║   ★ INICIO FORMAL = continuación de Phase1.nest_into_phase2 ★               ║
# ║   Precondición constructora: Phase1HilbertData.                              ║
# ║                                                                             ║
# ║   ULTIMO MÉTODO DE FASE 2 (morfismo terminal = unidad de Fase 3):            ║
# ║       nest_into_phase3() → Phase3_BornHeytingCollapser                       ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase2_WKBGaugeOrienter:
    r"""FASE 2 — ORIENT: Φ, m*, Maslov, T_exact ∧ T_WKB.

    ★ INICIO FORMAL = continuación del morfismo terminal de la FASE 1 ★

    Cadena interna:
        __init__(Phase1HilbertData)            ← unidad heredada de Φ₁
            → compute_work_function
            → compute_effective_mass
            → compute_maslov_index
            → compute_barrier_transmission     (exacta + WKB)
            → orient_wkb_gauge                 (pre-terminal)
            → nest_into_phase3                 ★ morfismo terminal = unidad de Φ₃
    """

    def __init__(
        self,
        phase1_data: Phase1HilbertData,
        *,
        strict_wkb: bool = False,
    ) -> None:
        r"""★ CONTINUACIÓN DE FASE 1 / INICIO DE FASE 2 ★

        Args
        ────
        phase1_data : Phase1HilbertData
            Salida de nest_into_phase2 (observe ⊗ snapshot).
        strict_wkb : si True, κa ≤ umbral en régimen túnel → HilbertWKBValidityError.
        """
        if not isinstance(phase1_data, Phase1HilbertData):
            raise TypeError(
                "Phase2_WKBGaugeOrienter requiere Phase1HilbertData "
                "como precondición (Fase 1)."
            )
        self._p1: Final[Phase1HilbertData] = phase1_data
        self._strict_wkb: Final[bool] = bool(strict_wkb)

    @property
    def phase1(self) -> Phase1HilbertData:
        return self._p1

    # ─────────────────────────────────────────────────────────────────────
    # 2.1 Función de trabajo Φ(χ²) = Φ₀ + α·χ²  (acoplamiento aditivo)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def compute_work_function(snapshot: EnvironmentalSnapshot) -> float:
        """Mayor amenaza topológica ⟹ mayor barrera (modulación lineal)."""
        phi = Const.BASE_PHI + Const.ALPHA_COUPLING * snapshot.chi_squared
        return _ensure_nonneg_finite_float(phi, name="work_function")

    # ─────────────────────────────────────────────────────────────────────
    # 2.2 Masa efectiva m*(σ) = m₀/|σ|  (σ < −ε);  +∞ si σ ≥ −ε
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def compute_effective_mass(snapshot: EnvironmentalSnapshot) -> float:
        sigma = snapshot.dominant_pole_real
        if sigma >= -Const.SIGMA_CHAOS_TOL:
            logger.warning(
                "[Φ₂] Sistema inestable (σ=%.6e ≥ −tol). m* → ∞.", sigma
            )
            return float("inf")
        m_eff = Const.BASE_MASS / abs(sigma)
        return _ensure_positive_or_posinf_float(m_eff, name="effective_mass")

    # ─────────────────────────────────────────────────────────────────────
    # 2.3 Índice de Maslov para barrera rectangular
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def compute_maslov_index(E: float, Phi: float) -> Tuple[int, float]:
        r"""μ = 0 (E ≥ Φ, sin puntos de retorno), μ = 2 (E < Φ, dos muros).

        Fase φ = π μ / 2 ∈ {0, π}. Contribuye fase global; no altera |T|².
        """
        if E >= Phi:
            return 0, 0.0
        return 2, math.pi

    # ─────────────────────────────────────────────────────────────────────
    # 2.4 Transmisión exacta + diagnóstico WKB (R1–R4 preservadas)
    # ─────────────────────────────────────────────────────────────────────
    def compute_barrier_transmission(
        self, E: float, Phi: float, m_eff: float,
    ) -> RectangularBarrierTransmission:
        r"""R1: m*=∞ ∧ E ≥ Φ ⟹ T=1.  R2: m*=∞ ∧ E < Φ ⟹ T=0.
        R3–R4: masa finita → T_exact de los tres regímenes.
        """
        transmission = RectangularBarrierCalculator.compute(
            E, Phi, m_eff, Const.BARRIER_DX
        )
        if (
            self._strict_wkb
            and not transmission.is_classical
            and transmission.regime == "tunneling"
            and transmission.kappa_width <= Const.WKB_VALIDITY_THRESHOLD
        ):
            raise HilbertWKBValidityError(
                f"WKB fuera del régimen κa ≫ 1: κa={transmission.kappa_width:.3e} "
                f"≤ {Const.WKB_VALIDITY_THRESHOLD:.3e}.",
            )
        return transmission

    # ─────────────────────────────────────────────────────────────────────
    # 2.5 Orient (pre-terminal de Φ₂)
    # ─────────────────────────────────────────────────────────────────────
    def orient_wkb_gauge(self) -> GaugeCouplingCertificate:
        """Produce GaugeCouplingCertificate, precondición estricta de Φ₃."""
        observe = self._p1.observe
        snapshot = self._p1.snapshot
        E = observe.incident_energy
        Phi = self.compute_work_function(snapshot)
        m_eff = self.compute_effective_mass(snapshot)
        mu, phi_maslov = self.compute_maslov_index(E, Phi)
        tr = self.compute_barrier_transmission(E, Phi, m_eff)

        is_frustrated = snapshot.global_frustration > Const.EPSILON_MACH
        T = 0.0 if is_frustrated else tr.t_exact
        t_wkb = 0.0 if is_frustrated else tr.t_wkb
        regime = "vetoed" if is_frustrated else tr.regime
        barrier_height = max(0.0, Phi - E)

        cert_hash = _compute_dto_hash(
            self._p1.certification_hash,
            E, Phi, m_eff if math.isfinite(m_eff) else float("inf"),
            T, t_wkb, mu, regime, is_frustrated,
        )
        logger.info(
            "[Φ₂ ✓] GaugeCouplingCertificate: Φ=%.4f, m*=%.4e, T_exact=%.6e, "
            "T_WKB=%.6e, κa=%.4f, μ=%d, régimen=%s, frustrated=%s, hash=%s.",
            Phi, m_eff, T, t_wkb, tr.kappa_width, mu, regime, is_frustrated,
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
            wkb_kappa_width=float(tr.kappa_width)
            if math.isfinite(tr.kappa_width)
            else tr.kappa_width,
            wkb_exponent=float(tr.exponent)
            if math.isfinite(tr.exponent)
            else tr.exponent,
            transmission_probability=float(T),
            transmission_wkb=float(t_wkb),
            transmission_ratio=float(tr.t_ratio)
            if math.isfinite(tr.t_ratio)
            else tr.t_ratio,
            maslov_index=int(mu),
            maslov_phase=float(phi_maslov),
            regime=str(regime),
            certification_hash=cert_hash,
        )

    # ─────────────────────────────────────────────────────────────────────
    # 2.6 ★ MORFISMO TERMINAL DE FASE 2 ★
    #     Tipo de retorno = objeto inicial de la FASE 3.
    #     last(Φ₂) = unit(Φ₃) = Phase3_BornHeytingCollapser.
    # ─────────────────────────────────────────────────────────────────────
    def nest_into_phase3(self) -> "Phase3_BornHeytingCollapser":
        r"""★ MORFISMO TERMINAL DE FASE 2 / UNIDAD DE LA FASE 3 ★

        Composición estricta F₂ ⊣ F₃:
            nest_into_phase3()  :=  Phase3_BornHeytingCollapser
                                    ∘ orient_wkb_gauge.

        El collapser de Born nace ya alimentado con GaugeCouplingCertificate;
        su constructor ES la continuación formal de este método.
        """
        certificate = self.orient_wkb_gauge()
        return Phase3_BornHeytingCollapser(certificate)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 3: DECIDE & ACT — COLAPSO BORN Y VETO HEYTING                        ║
# ║                                                                             ║
# ║   ★ INICIO FORMAL = continuación de Phase2.nest_into_phase3 ★               ║
# ║   Precondición constructora: GaugeCouplingCertificate.                       ║
# ║                                                                             ║
# ║   ULTIMO MÉTODO DE FASE 3 (morfismo terminal del módulo):                    ║
# ║       collapse_into_state(state) → CategoricalState                          ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase3_BornHeytingCollapser:
    r"""FASE 3 — DECIDE & ACT: θ, Ω₃, K = E−Φ, medición, CategoricalState.

    ★ INICIO FORMAL = continuación del morfismo terminal de la FASE 2 ★

    Cadena interna:
        __init__(GaugeCouplingCertificate)     ← unidad heredada de Φ₂
            → compute_collapse_threshold
            → resolve_heyting_omega_three_lattice
            → compute_coherence_residual
            → compute_post_collapse_observables  (E conservada)
            → collapse_wavefunction              (pre-terminal)
            → collapse_into_state                ★ morfismo terminal del módulo
    """

    def __init__(self, certificate: GaugeCouplingCertificate) -> None:
        r"""★ CONTINUACIÓN DE FASE 2 / INICIO DE FASE 3 ★

        Args
        ────
        certificate : GaugeCouplingCertificate
            Salida de orient_wkb_gauge, inyectada por nest_into_phase3.
        """
        if not isinstance(certificate, GaugeCouplingCertificate):
            raise TypeError(
                "Phase3_BornHeytingCollapser requiere GaugeCouplingCertificate "
                "como precondición (Fase 2)."
            )
        self._cert: Final[GaugeCouplingCertificate] = certificate

    @property
    def certificate(self) -> GaugeCouplingCertificate:
        return self._cert

    # ─────────────────────────────────────────────────────────────────────
    # 3.1 Umbral determinista θ ∈ [0, 1) vía SHA-256 (53 bits de mantisa)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def compute_collapse_threshold(serialized_payload: bytes) -> float:
        """θ = mantissa / 2⁵³ ∈ [0, 1). Evita n/2⁶⁴ (redondeo a 1.0)."""
        digest = hashlib.sha256(serialized_payload).digest()
        n = int.from_bytes(digest[:8], byteorder="big", signed=False)
        mantissa = (n >> (64 - Const.MANTISSA_BITS)) & Const.MANTISSA_MASK
        theta = mantissa / Const.MANTISSA_DIVISOR
        if theta >= 1.0:  # imposible por construcción; defensa en profundidad
            theta = 0.9999999999999999
        return float(theta)

    # ─────────────────────────────────────────────────────────────────────
    # 3.2 Resolución del retículo Ω₃
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def resolve_heyting_omega_three_lattice(
        certificate: GaugeCouplingCertificate,
        theta: float,
    ) -> HeytingOmega3:
        r"""Colapsa Ω₃ con precedencia estricta y join monótono.

            1. is_frustrated                         ⟹ VETOED
            2. m* = ∞ ∧ régimen impenetrable         ⟹ VETOED
            3. T < θ ∧ ¬frustrated                   ⟹ DEGRADED
            4. en otro caso                          ⟹ COHERENT
        """
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

    # ─────────────────────────────────────────────────────────────────────
    # 3.3 Residual de coherencia r = clamp(1 − 2|T − θ|, 0, 1)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def compute_coherence_residual(T: float, theta: float) -> float:
        """r ≈ 1 ⟺ colapso fronterizo; r ≈ 0 ⟺ veredicto inequívoco."""
        boundary = abs(T - theta)
        return _clamp_probability(1.0 - 2.0 * boundary)

    # ─────────────────────────────────────────────────────────────────────
    # 3.4 Observables cinéticos post-colapso (E ya no se reconstruye)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def compute_post_collapse_observables(
        certificate: GaugeCouplingCertificate,
        eigenstate: HilbertEigenstate,
    ) -> Tuple[float, float]:
        r"""Si ADMITTED: K = max(E−Φ, ε) si E ≥ Φ, else ε (túnel);
        p = √(2 m_kin K). Si REJECTED: K = p = 0.

        E se lee de certificate.incident_energy (conservada desde Φ₁).
        """
        if eigenstate == HilbertEigenstate.REJECTED:
            return 0.0, 0.0
        E = certificate.incident_energy
        Phi = certificate.work_function
        if E >= Phi:
            kinetic_energy = max(Const.MIN_KINETIC_ENERGY, E - Phi)
        else:
            kinetic_energy = Const.MIN_KINETIC_ENERGY
        m_for_momentum = (
            Const.BASE_MASS
            if math.isinf(certificate.effective_mass)
            else certificate.effective_mass
        )
        momentum = _safe_sqrt(2.0 * m_for_momentum * kinetic_energy)
        momentum = _ensure_finite_float(momentum, name="quantum_momentum")
        return float(kinetic_energy), float(momentum)

    # ─────────────────────────────────────────────────────────────────────
    # 3.5 Colapso de la función de onda (pre-terminal)
    # ─────────────────────────────────────────────────────────────────────
    def collapse_wavefunction(self) -> QuantumMeasurement:
        r"""Emite QuantumMeasurement (Born T ≥ θ, veto Ω₃ fuerza REJECTED)."""
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

        kinetic_energy, momentum = self.compute_post_collapse_observables(
            cert, eigenstate
        )
        measurement = QuantumMeasurement(
            eigenstate=eigenstate,
            heyting_verdict=verdict,
            collapse_threshold=theta,
            born_probability=T,
            coherence_residual=residual,
            momentum=momentum,
            kinetic_energy=kinetic_energy,
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
                "phase1_hash": cert.phase1.certification_hash,
                "phase2_hash": cert.certification_hash,
            },
        )
        logger.debug(
            "[Φ₃.collapse] |%s⟩ Ω₃=%s T=%.6e θ=%.6f r=%.4f K=%.4e p=%.4e.",
            eigenstate.name, verdict.name, T, theta, residual,
            kinetic_energy, momentum,
        )
        return measurement

    # ─────────────────────────────────────────────────────────────────────
    # 3.6 ★ MORFISMO TERMINAL DE FASE 3 / DEL MÓDULO ★
    #     Cierra el funtor maestro 𝓕 = Φ₃ ∘ Φ₂ ∘ Φ₁.
    # ─────────────────────────────────────────────────────────────────────
    def collapse_into_state(self, state: CategoricalState) -> CategoricalState:
        r"""★ MORFISMO TERMINAL DEL MÓDULO ★

        Cadena funtorial de Φ₃:
            GaugeCouplingCertificate
              ──(collapse_wavefunction)──▶ QuantumMeasurement
              ──(ADMITTED ⇒ PHYSICS ∪ strata / REJECTED ⇒ ∅)──▶ CategoricalState
        """
        if not isinstance(state, CategoricalState):
            raise HilbertWatcherError(
                f"state debe ser CategoricalState; recibido {type(state).__name__}."
            )
        measurement = self.collapse_wavefunction()
        context = _safe_context(getattr(state, "context", None))
        cert = self._cert
        T = cert.transmission_probability
        theta = measurement.collapse_threshold
        verdict = measurement.heyting_verdict

        if measurement.eigenstate == HilbertEigenstate.ADMITTED:
            new_context = {
                **context,
                "quantum_momentum": measurement.momentum,
                "quantum_measurement": measurement,
            }
            new_strata = state.validated_strata | {Stratum.PHYSICS}
            logger.info(
                "[Φ₃ ✓] |ADMITTED⟩: T=%.6e ≥ θ=%.6f, p=%.6e, Ω₃=%s, r=%.4f, μ=%d.",
                T, theta, measurement.momentum, verdict.name,
                measurement.coherence_residual, cert.maslov_index,
            )
        else:
            if cert.is_frustrated:
                reason = (
                    f"Veto cohomológico: E_frust={cert.frustration_energy:.3e}"
                )
            elif verdict == HeytingOmega3.DEGRADED:
                reason = f"Rechazo: T={T:.6e} < θ={theta:.6f}"
            else:
                reason = (
                    f"Barrera impenetrable (m*=∞, régimen={cert.regime})"
                )
            new_context = {
                **context,
                "quantum_error": reason,
                "quantum_measurement": measurement,
            }
            new_strata = frozenset()
            logger.warning("[Φ₃ ✗] |REJECTED⟩: %s, Ω₃=%s.", reason, verdict.name)

        return CategoricalState(
            payload=state.payload,
            context=new_context,
            validated_strata=new_strata,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   ORQUESTADOR: 𝓕 = Φ₃ ∘ Φ₂ ∘ Φ₁  (HilbertObserverAgent)                     ║
# ║                                                                             ║
# ║   Anidamiento EXCLUSIVO vía morfismos terminales:                           ║
# ║       Φ₁.nest_into_phase2(payload)  ⟶  Phase2                               ║
# ║       Φ₂.nest_into_phase3()         ⟶  Phase3                               ║
# ║       Φ₃.collapse_into_state(state) ⟶  CategoricalState                     ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class HilbertObserverAgent(Morphism):
    r"""Agente cuántico OODA sobre CategoricalState.

    Diagrama conmutativo:
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │ OBSERVE  │───▶│  ORIENT  │───▶│DECIDE+ACT│
        │ E←H(ρ)   │    │ Φ,m*,T   │    │ Ω₃,|λ⟩   │
        └──────────┘    └──────────┘    └──────────┘

    Propiedades funtoriales: preserva composición MIC, morfismo puro entre
    invocaciones, telemetría embebida, idempotencia estructural por marca
    de contexto ('quantum_measurement').
    """

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
        # Validación eager de contratos (falla en construcción, no en execute).
        Phase1_WavefunctionObserver(topo_watcher, laplace_oracle, sheaf_orchestrator)
        super().__init__()
        logger.info(
            "HilbertObserverAgent v%s inicializado. strict_wkb=%s.",
            __version__, self._strict_wkb,
        )

    def execute_ooda_loop(self, state: CategoricalState) -> CategoricalState:
        r"""Punto de entrada: 𝓕 = Φ₃ ∘ Φ₂ ∘ Φ₁, anidado por morfismos terminales.

        Idempotencia: si state.context ya contiene 'quantum_measurement',
        retorna el mismo objeto (morfismo idempotente sobre estados colapsados).
        """
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

    def __call__(self, state: CategoricalState) -> CategoricalState:
        """Implementación del funtor Morphism para integración MIC."""
        return self.execute_ooda_loop(state)


# ═════════════════════════════════════════════════════════════════════════════
# §9. TESTING Y EJEMPLOS
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    print("\n" + "═" * 80)
    print(f"HILBERT WATCHER v{__version__} — SUITE DE TESTING")
    print("═" * 80 + "\n")

    class MockTopologicalWatcher:
        def __init__(self, threat: float = 2.5):
            self._threat = threat

        def get_mahalanobis_threat(self) -> float:
            return self._threat

    class MockLaplaceOracle:
        def __init__(self, pole: float = -0.5):
            self._pole = pole

        def get_dominant_pole_real(self) -> float:
            return self._pole

    class MockSheafOrchestrator:
        def __init__(self, frustration: float = 1e-12):
            self._frustration = frustration

        def get_global_frustration_energy(self) -> float:
            return self._frustration

    print("TEST 1: Heyting Ω₃ (Gödel–Dummett)")
    a, b = HeytingOmega3.DEGRADED, HeytingOmega3.VETOED
    print(f"  join({a.name}, {b.name}) = {a.join(b).name}")
    print(f"  meet({a.name}, {b.name}) = {a.meet(b).name}")
    print(
        f"  {a.name} → COHERENT = {a.implication(HeytingOmega3.COHERENT).name}  "
        f"(esperado ⊥)"
    )
    print(f"  ¬{a.name} = {a.negation().name}  (esperado ⊥)")
    print(
        f"  COHERENT → DEGRADED = "
        f"{HeytingOmega3.COHERENT.implication(HeytingOmega3.DEGRADED).name}  "
        f"(esperado ⊤)"
    )
    print()

    print("TEST 2: Construcción del agente")
    agent = HilbertObserverAgent(
        topo_watcher=MockTopologicalWatcher(threat=0.5),
        laplace_oracle=MockLaplaceOracle(pole=-0.5),
        sheaf_orchestrator=MockSheafOrchestrator(frustration=1e-12),
    )
    print(f"  ✓ Agente v{__version__}. domain={agent.domain}, codomain={agent.codomain}")
    print()

    print("TEST 3: Φ₁.observe_payload")
    payload = {"endpoint": "/api/test", "data": "A" * 500}
    obs = Phase1_WavefunctionObserver.observe_payload(payload)
    print(
        f"  N={obs.payload_size}, H={obs.byte_entropy_bits:.4f} bits, "
        f"Ĥ_b={obs.block_entropy_rate:.4f}"
    )
    print(
        f"  E={obs.incident_energy:.6e}, "
        f"δE=[{obs.energy_uncertainty_lower:.3e}, {obs.energy_uncertainty_upper:.3e}]"
    )
    print()

    print("TEST 4: Anidamiento Φ₁ ⊣ Φ₂")
    phase1 = Phase1_WavefunctionObserver(
        MockTopologicalWatcher(threat=0.5),
        MockLaplaceOracle(pole=-0.5),
        MockSheafOrchestrator(frustration=1e-12),
    )
    phase2 = phase1.nest_into_phase2(payload)
    cert = phase2.orient_wkb_gauge()
    print(f"  Φ={cert.work_function:.4f}, m*={cert.effective_mass:.4e}")
    print(
        f"  κa={cert.wkb_kappa_width:.4f}, régimen={cert.regime}, "
        f"T_exact={cert.transmission_probability:.6e}, T_WKB={cert.transmission_wkb:.6e}"
    )
    print(
        f"  μ_Maslov={cert.maslov_index}, φ={cert.maslov_phase:.4f}, "
        f"E conservada={cert.incident_energy:.6e}"
    )
    print()

    print("TEST 5: Pipeline completo 𝓕 = Φ₃ ∘ Φ₂ ∘ Φ₁")
    initial = CategoricalState(
        payload=payload, context={}, validated_strata=frozenset()
    )
    result = agent.execute_ooda_loop(initial)
    m = result.context.get("quantum_measurement")
    if m is not None:
        print(f"  eigenstate={m.eigenstate.name}, Ω₃={m.heyting_verdict.name}")
        print(
            f"  θ={m.collapse_threshold:.6f}, T={m.born_probability:.6e}, "
            f"r={m.coherence_residual:.4f}"
        )
        print(f"  p={m.momentum:.6e}, K={m.kinetic_energy:.6e}")
        print(f"  strata={result.validated_strata}")
    print()

    print("TEST 6: No-clonación A5")
    import copy as _cp

    try:
        _ = _cp.copy(m)
        print("  ✗ copy() NO bloqueado")
    except HilbertStateCloningError as exc:
        print(f"  ✓ copy() bloqueado: {str(exc)[:80]}")
    try:
        _ = _cp.deepcopy(m)
        print("  ✗ deepcopy() NO bloqueado")
    except HilbertStateCloningError as exc:
        print(f"  ✓ deepcopy() bloqueado: {str(exc)[:80]}")
    print()

    print("TEST 7: Veto cohomológico")
    agent_veto = HilbertObserverAgent(
        topo_watcher=MockTopologicalWatcher(threat=0.5),
        laplace_oracle=MockLaplaceOracle(pole=-0.5),
        sheaf_orchestrator=MockSheafOrchestrator(frustration=1.0),
    )
    initial2 = CategoricalState(
        payload={"k": "v"}, context={}, validated_strata=frozenset()
    )
    result2 = agent_veto.execute_ooda_loop(initial2)
    m2 = result2.context.get("quantum_measurement")
    print(f"  eigenstate={m2.eigenstate.name}, Ω₃={m2.heyting_verdict.name}")
    print(f"  reason={str(result2.context.get('quantum_error', 'N/A'))[:80]}")
    print()

    print("TEST 8: R1 — m*=∞ con E≥Φ debe dar T=1 (clásico)")
    payload_big = {"data": "A" * 100_000}
    obs_big = Phase1_WavefunctionObserver.observe_payload(payload_big)
    snap_unstable = EnvironmentalSnapshot(
        chi_squared=0.0, dominant_pole_real=0.1, global_frustration=1e-12
    )
    p1_unstable = Phase1HilbertData(
        observe=obs_big, snapshot=snap_unstable, certification_hash="test"
    )
    cert_unstable = Phase2_WKBGaugeOrienter(p1_unstable).orient_wkb_gauge()
    print(f"  E={obs_big.incident_energy:.4e}, Φ={cert_unstable.work_function:.4f}")
    print(
        f"  m*={cert_unstable.effective_mass}, régimen={cert_unstable.regime}, "
        f"T={cert_unstable.transmission_probability}"
    )
    print()

    print("TEST 9: Idempotencia del funtor")
    result_again = agent.execute_ooda_loop(result)
    print(f"  Idempotente: {result is result_again}")
    print()

    print("TEST 10: T_exact vs T_WKB (túnel, umbral, sobrebarrera)")
    for E in [0.1, 1.0, 5.0, 9.9, 10.0, 12.0, 20.0]:
        t = RectangularBarrierCalculator.compute(
            E, V0=10.0, m_eff=1.0, width=1.0
        )
        print(
            f"  E={E:5.1f}: T_exact={t.t_exact:.6e}, T_WKB={t.t_wkb:.6e}, "
            f"κa={t.kappa_width:.3f}, regime={t.regime}"
        )
    print()

    print("TEST 11: K se conserva cuando E ≥ Φ (bug v4.0)")
    # Forzar certificado clásico con E conocida.
    if cert.incident_energy >= cert.work_function:
        K, p = Phase3_BornHeytingCollapser.compute_post_collapse_observables(
            cert, HilbertEigenstate.ADMITTED
        )
        expected = max(Const.MIN_KINETIC_ENERGY, cert.incident_energy - cert.work_function)
        print(f"  K={K:.6e}, esperado≈{expected:.6e}, match={abs(K - expected) < 1e-12}")
    else:
        print("  (E < Φ en este payload; K de túnel = ε)")
        K, p = Phase3_BornHeytingCollapser.compute_post_collapse_observables(
            cert, HilbertEigenstate.ADMITTED
        )
        print(f"  K_túnel={K:.6e} (ε={Const.MIN_KINETIC_ENERGY:.6e})")
    print()

    print("═" * 80)
    print("SUITE DE TESTING COMPLETADA")
    print("═" * 80 + "\n")


__all__ = [
    "HilbertWatcherError",
    "HilbertNumericalError",
    "HilbertInterfaceError",
    "HilbertPayloadError",
    "HilbertWKBValidityError",
    "HilbertStateCloningError",
    "HilbertCohomologicalVetoError",
    "QuantumThresholds",
    "HeytingOmega3",
    "HilbertEigenstate",
    "WavefunctionObserveData",
    "EnvironmentalSnapshot",
    "Phase1HilbertData",
    "GaugeCouplingCertificate",
    "QuantumMeasurement",
    "RectangularBarrierTransmission",
    "ITopologicalWatcher",
    "ILaplaceOracle",
    "ISheafCohomologyOrchestrator",
    "RectangularBarrierCalculator",
    "Phase1_WavefunctionObserver",
    "Phase2_WKBGaugeOrienter",
    "Phase3_BornHeytingCollapser",
    "HilbertObserverAgent",
]