# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Quantum Admission Gate (Operador de Proyección de Hilbert)          ║
║ Ruta   : app/aleph/quantum_admission_gate.py                                 ║
║ Versión: 4.1.0-Strict-Spectral-WKB-Exact-Gauge-Orthomodular-Aleph-Secure-PhD ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS CUÁNTICA Y GOBERNANZA DE FRONTERA (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este operador constitutivo y resolvedor ciego de-confinado del Estrato ALEPH
(Nivel 4, V_{ℵ₀}) actúa como el Miembro Cero del Consejo de Sabios. Su propósito
es salvaguardar la «Fortaleza Matemática» frente a perturbaciones y payloads
crudos externos, modelando la admisión de datos no como un filtrado sintáctico
pasivo, sino como una MEDICIÓN PROYECTIVA DE COLAPSO DE FASE.

El sistema trata el payload entrante como un paquete de ondas en superposición.
Calcula su entropía de Shannon (nats) para derivar su energía semántica E = hν.
Si el cuanto incidente supera la Función de Trabajo Φ de la barrera —modulada
por la topología local y el Oráculo de Laplace—, la señal se admite y se
inyecta síncronamente como condiciones iniciales (t₀) en forma de momentum
ciber-físico (p = √(2m* K)) sobre el condensador de flujo.
Si la energía es sub-umbral, se evalúa la probabilidad de penetración por
efecto túnel mediante (i) la fórmula EXACTA de barrera rectangular (los tres
regímenes E ≶ V₀) y (ii) la aproximación semiclásica WKB como diagnóstico
cruzado de barrera gruesa (κa ≫ 1). Un fallo en el túnel aniquila
idempotentemente el estado.

Política de unidades: h = 1,  ℏ = h/(2π)  (unidades de Planck con h-barra
derivada). Toda energía se expresa en cuantos de h·Hz.

AXIOMÁTICA ALGEBRAICA, CATEGORIAL Y FÍSICO-MATEMÁTICA:
────────────────────────────────────────────────────────────────────────────────

  [A1] Espacio de Hilbert complejo separable:
       ℋ := L²(ℝ, ℂ),  ⟨ψ|φ⟩ = ∫_ℝ ψ*(x) φ(x) dx.

  [A2] Medición de Born y hermiticidad espectral:
       Ĥ = Ĥ† ⟹ σ(Ĥ) ⊂ ℝ;  P(a_i) = |⟨a_i|ψ⟩|².
       El colapso de este portal es Born-like DETERMINISTA: T ≥ θ, con
       θ = Φ_SHA256(payload) ∈ [0, 1) (53 bits de mantisa IEEE-754).

  [A3] Transmisión exacta de barrera rectangular + diagnóstico WKB:
       E < V₀ :  T = [1 + V₀² sinh²(κa) / (4 E (V₀−E))]⁻¹,
                 κ = √(2m*(V₀−E))/ℏ.
       E = V₀ :  T = [1 + m* V₀ a² / (2 ℏ²)]⁻¹.
       E > V₀ :  T = [1 + V₀² sin²(k₂ a) / (4 E (E−V₀))]⁻¹,
                 k₂ = √(2m*(E−V₀))/ℏ.
       WKB (diagnóstico de barrera gruesa, NUNCA sustituto de T_exact):
                 T_WKB ≈ exp(−2 κ a)  (E < V₀),  T_WKB = 1 (E ≥ V₀).
       Para el potencial rectangular, V' es nulo en el interior y
       singular en los muros: WKB NO es uniformemente válido. Se usa
       exclusivamente como testigo asintótico cuando κa ≫ 1.

  [A4] Fibrado principal de gauge U(1) del portal:
       π: P → M, G = U(1).  F = dω ≡ 0  ⟹  β₁(K) = 0 en el 1-esqueleto.
       PROXY OPERACIONAL: E_frust ≤ ε_gauge  ⟺  is_flat.
       β₁ se reporta como COTA INFERIOR {0,1}, jamás como ceil(E_frust).
       El orquestador de haces es la fuente de verdad de los números de Betti.

  [A5] No-clonación categórica (zero side-effects):
       ∄ U ∈ U(ℋ ⊗ ℋ): U(|ψ⟩ ⊗ |0⟩) = |ψ⟩ ⊗ |ψ⟩  ∀|ψ⟩.
       Realizado por __copy__/__deepcopy__ → QuantumStateError y por el
       token _non_cloning_uid.

  [A6] Retículo ortomodular de proyecciones (Birkhoff–von Neumann):
       𝒫(ℋ) es ortomodular y, en general, no distributivo.
       En ℋ₂ = span{|ADMITIDO⟩, |RECHAZADO⟩} los proyectores espectrales
       de Ĥ conmutan, son ortogonales y se complementan a 𝟙: el subretículo
       generado es BOOLEANO (álgebra de Boole de 4 elementos). Se certifica
       por álgebra matricial 2×2, no por fiat.

ARQUITECTURA DE TRES FASES ANIDADAS (Handoff por Constructor Estricto):
────────────────────────────────────────────────────────────────────────────────
La admisión se rige por un contrato covariante F₁ ⊣ F₂ ⊣ F₃. El tipo de
retorno del último método de Φᵢ ES el objeto inicial de Φᵢ₊₁:

  Fase 1 ──► CERTIFICACIÓN GAUGE-COHOMOLÓGICA DEL PORTAL
             (Phase1_GaugeAdmissionCertifier)
             Snapshot ambiental único (χ², σ, E_frust), certificación de
             planitud U(1) y del retículo ortomodular ℋ₂.
             Morfismo terminal: nest_into_phase2
                 (oráculos)  ⟶  Phase2_SpectralEnergyAuditor

  Fase 2 ──► REGULACIÓN ESPECTRAL DE ENERGÍA, Φ Y MASA EFECTIVA
             (Phase2_SpectralEnergyAuditor)
             ★ INICIO FORMAL = continuación de nest_into_phase2 ★
             E = hν con aritmética de intervalos, Φ(χ²) = Φ₀ e^{α χ²},
             m*(σ) = m₀/|σ|. Clasifica el régimen E ≶ Φ.
             Morfismo terminal: nest_into_phase3
                 (Phase2, payload)  ⟶  Phase3_BarrierCollapseProjector

  Fase 3 ──► TRANSMISIÓN EXACTA DE BARRERA Y COLAPSO DE BORN
             (Phase3_BarrierCollapseProjector)
             ★ INICIO FORMAL = continuación de nest_into_phase3 ★
             T_exact (tres regímenes) ∧ T_WKB (diagnóstico), umbral θ,
             ambigüedad ortomodular y emisión de QuantumMeasurement.
             Morfismo terminal: collapse_wavefunction
                 Phase3  ⟶  QuantumMeasurement

Funtor Maestro:
  𝒵_Admission = Φ₃ ∘ Φ₂ ∘ Φ₁ : Payload × Oracles ⟶ QuantumMeasurement
"""

from __future__ import annotations

import copy as _copy_module
import hashlib
import logging
import math
import struct
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

from app.core.mic_algebra import CategoricalState, Morphism
from app.core.schemas import Stratum


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: CONFIGURACIÓN, VERSIÓN Y LOGGING
# ═════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("MIC.Physics.QuantumAdmission")

__version__: Final[str] = (
    "4.1.0-Strict-Spectral-WKB-Exact-Gauge-Orthomodular-Aleph-Secure-PhD"
)

_NON_CLONING_UID_COUNTER = _itertools_count(1)


@dataclass(frozen=True, slots=True)
class LogContext:
    """Contexto estructurado inmutable para trazabilidad cuántica post-mortem."""

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

    def to_structured_log(self) -> str:
        """Serialización canónica key=value."""
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
            f"veto={self.veto}"
        )


class QuantumLogger:
    """Logger especializado con contexto cuántico enriquecido."""

    @staticmethod
    def log_measurement(
        measurement: "QuantumMeasurement",
        level: int = logging.INFO,
    ) -> None:
        context = LogContext(
            eigenstate=measurement.eigenstate.name,
            incident_energy=measurement.incident_energy,
            work_function=measurement.work_function,
            tunneling_prob=measurement.tunneling_probability,
            tunneling_wkb=measurement.tunneling_wkb_diagnostic,
            momentum=measurement.momentum,
            sigma=measurement.dominant_pole_real,
            chi_squared=measurement.threat_level,
            kappa_a=measurement.kappa_width_diagnostic,
            ambiguity=measurement.collapse_ambiguity,
            veto=measurement.frustration_veto,
        )
        logger.log(level, "QUANTUM_MEASUREMENT | %s", context.to_structured_log())


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: JERARQUÍA DE EXCEPCIONES
# ═════════════════════════════════════════════════════════════════════════════


class QuantumAdmissionError(Exception):
    """Objeto inicial de la categoría de errores cuánticos del portal."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        self.message = message

    def __str__(self) -> str:
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} | Context: {ctx_str}"
        return self.message


class QuantumNumericalError(QuantumAdmissionError):
    """Error numérico con propagación de incertidumbre (NaN, overflow, dominio)."""


class QuantumInterfaceError(QuantumAdmissionError):
    """Violación de contrato funtorial (protocolo de oráculo)."""


class QuantumStateError(QuantumAdmissionError):
    """Estado fuera del espacio de Hilbert admisible (A1) o violación de A5."""


class WKBValidityError(QuantumNumericalError):
    """Aproximación WKB fuera del régimen de barrera gruesa (κa ≲ 1)."""


class CohomologicalVetoError(QuantumAdmissionError):
    """Veto estructural por frustración cohomológica / curvatura F ≢ 0 (A4)."""


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: CONSTANTES FÍSICAS (selladas)
# ═════════════════════════════════════════════════════════════════════════════


class PhysicalConstants:
    """Constantes físicas fundamentales con validación estática.

    Sistema: h = 1,  ℏ = h/(2π).  Invariante: ∀c,  c ∈ ℝ₊ ∪ {+∞}.
    La clase está sellada (__init_subclass__).
    """

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
    MAX_SHANNON_ENTROPY: Final[float] = math.log(256.0)  # ln(256) nats
    DIVISION_EPSILON: Final[float] = 1e-15
    WKB_VALIDITY_THRESHOLD: Final[float] = 1.0  # κa ≥ 1 ⇒ diagnóstico WKB útil
    GAUGE_FLATNESS_TOL: Final[float] = 1e-9
    THRESHOLD_ENERGY_REL_TOL: Final[float] = 1e-12
    SINH_ASYMPTOTIC_KA: Final[float] = 20.0  # κa ≥ 20 ⇒ sinh ≈ ½ e^{κa}

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
# SECCIÓN 4: ESTRUCTURAS ALGEBRAICAS (Intervalos, WKB, Colapso)
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class NumericInterval:
    """Intervalo cerrado [lower, upper] ⊂ ℝ con aritmética de Minkowski.

    Estructura:
        · Espacio métrico d(I₁, I₂) = |mid(I₁) − mid(I₂)|.
        · Retícula: I₁ ≤ I₂ ⟺ upper₁ ≤ lower₂.
        · Álgebra:
            I₁ + I₂ = [a+c, b+d]
            I₁ − I₂ = [a−d, b−c]
            I₁ · I₂ = [min(ac,ad,bc,bd), max(ac,ad,bc,bd)]
            I₁ / I₂ = I₁ · I₂⁻¹   (0 ∉ I₂)

    Invariantes: lower ≤ upper; extremos finitos o ±∞; width ≥ 0.
    """

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

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def contains_zero(self) -> bool:
        return self.lower <= 0.0 <= self.upper

    def intersects(self, other: "NumericInterval") -> bool:
        return not (self.upper < other.lower or other.upper < self.lower)

    def intersection(self, other: "NumericInterval") -> Optional["NumericInterval"]:
        lo = max(self.lower, other.lower)
        hi = min(self.upper, other.upper)
        return NumericInterval(lo, hi) if lo <= hi else None

    def hull(self, other: "NumericInterval") -> "NumericInterval":
        return NumericInterval(
            min(self.lower, other.lower),
            max(self.upper, other.upper),
        )

    def __add__(self, other: "NumericInterval") -> "NumericInterval":
        return NumericInterval(self.lower + other.lower, self.upper + other.upper)

    def __sub__(self, other: "NumericInterval") -> "NumericInterval":
        return NumericInterval(self.lower - other.upper, self.upper - other.lower)

    def __mul__(self, other: "NumericInterval") -> "NumericInterval":
        products = (
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        )
        return NumericInterval(min(products), max(products))

    def __truediv__(self, other: "NumericInterval") -> "NumericInterval":
        if other.contains_zero():
            raise ZeroDivisionError(
                f"División por intervalo que contiene 0: {other}."
            )
        # 0 ∉ I₂ ⇒ ambos extremos del mismo signo: [1/upper, 1/lower] está ordenado.
        inv = NumericInterval(1.0 / other.upper, 1.0 / other.lower)
        return self * inv

    def __neg__(self) -> "NumericInterval":
        return NumericInterval(-self.upper, -self.lower)

    def __abs__(self) -> "NumericInterval":
        if not self.contains_zero():
            return self if self.lower >= 0.0 else -self
        mag = max(abs(self.lower), abs(self.upper))
        return NumericInterval(0.0, mag)

    def __contains__(self, value: float) -> bool:
        return self.contains(value)

    def __repr__(self) -> str:
        return f"[{self.lower:.6e}, {self.upper:.6e}]"

    @staticmethod
    def from_value_with_tolerance(value: float, tolerance: float) -> "NumericInterval":
        abs_tol = abs(tolerance)
        return NumericInterval(value - abs_tol, value + abs_tol)

    @staticmethod
    def point(value: float) -> "NumericInterval":
        return NumericInterval(value, value)


@dataclass(frozen=True, slots=True)
class WKBParameters:
    """Parámetros del diagnóstico WKB de barrera gruesa.

    γ = κ a  (factor de Gamow rectangular).  T_WKB ≈ e^{−2γ}.
    validity_parameter = 1/κa : pequeño ⟺ régimen asintótico útil.
    El potencial rectangular es DISCONTINUO: WKB no es uniformemente válido.
    """

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
                f"Energía incidente negativa: {self.incident_energy}",
                context={"incident_energy": self.incident_energy},
            )
        if not (self.effective_mass > 0 or math.isinf(self.effective_mass)):
            raise QuantumNumericalError(
                f"Masa efectiva debe ser > 0 o ∞: {self.effective_mass}",
                context={"effective_mass": self.effective_mass},
            )
        if self.barrier_width <= 0:
            raise QuantumNumericalError(
                f"Ancho de barrera debe ser > 0: {self.barrier_width}",
                context={"barrier_width": self.barrier_width},
            )

    def is_valid_semiclassical_regime(self) -> bool:
        """True ⟺ κa > umbral y masa finita (testigo de barrera gruesa)."""
        return (
            self.kappa_width > Const.WKB_VALIDITY_THRESHOLD
            and 0 < self.effective_mass < float("inf")
        )

    def gamow_factor(self) -> float:
        """γ = κa ≥ 0."""
        return abs(self.exponent) / 2.0


@dataclass(frozen=True, slots=True)
class RectangularBarrierTransmission:
    """Transmisión EXACTA de barrera rectangular + diagnóstico WKB.

    Tres regímenes (Griffiths QM §2.5, ecs. 2.168 y 2.169 + umbral):
        E < V₀, E = V₀, E > V₀.
    """

    t_exact: float
    t_wkb: float
    t_ratio: float
    kappa_width: float
    is_classical: bool
    regime: str  # 'tunnel' | 'threshold' | 'over_barrier'


@dataclass(frozen=True, slots=True)
class CollapseAmbiguity:
    """Ambigüedad del colapso Born cerca de la frontera de decisión (A6).

    boundary_distance = |T − θ| ∈ [0, 1].
    ambiguity = clamp(1 − 2·boundary_distance, 0, 1).
    ambiguity ≈ 1  ⟺  colapso en la frontera de máxima superposición.
    is_orthomodular_degenerate ≈ (ambiguity → 1): el veredicto es
    numéricamente indistinguible de una superposición proyectiva.
    """

    boundary_distance: float
    ambiguity: float
    is_orthomodular_degenerate: bool


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5: AUTOESTADOS DEL OPERADOR HERMÍTICO Ĥ
# ═════════════════════════════════════════════════════════════════════════════


class Eigenstate(Enum):
    """Autoestados de Ĥ sobre ℋ₂ = span{|ADMITIDO⟩, |RECHAZADO⟩}.

    ⟨ADMITIDO|RECHAZADO⟩ = 0,  ⟨n|n⟩ = 1,
    |ADMITIDO⟩⟨ADMITIDO| + |RECHAZADO⟩⟨RECHAZADO| = 𝟙.
    """

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
# SECCIÓN 6: MEDICIÓN CUÁNTICA POST-COLAPSO (no-clonación A5)
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class QuantumMeasurement:
    """Medición cuántica inmutable post-colapso.

    TEOREMA (No-Clonación, A5):
        ∄ U ∈ U(ℋ ⊗ ℋ) tal que U(|ψ⟩⊗|0⟩) = |ψ⟩⊗|ψ⟩.
        Cada instancia porta un token _non_cloning_uid único;
        __copy__ y __deepcopy__ lanzan QuantumStateError.
    """

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
    non_cloning_uid: int = field(
        default_factory=lambda: next(_NON_CLONING_UID_COUNTER),
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        for fname, value in (
            ("incident_energy", self.incident_energy),
            ("work_function", self.work_function),
            ("kinetic_energy", self.kinetic_energy),
            ("momentum", self.momentum),
            ("threat_level", self.threat_level),
            ("tunneling_wkb_diagnostic", self.tunneling_wkb_diagnostic),
            ("kappa_width_diagnostic", self.kappa_width_diagnostic),
        ):
            if value < 0:
                raise QuantumStateError(
                    f"Observable '{fname}' debe ser no negativo: {value}",
                    context={fname: value},
                )

        if not 0.0 <= self.tunneling_probability <= 1.0:
            raise QuantumStateError(
                f"Probabilidad de túnel fuera de [0,1]: {self.tunneling_probability}",
                context={"tunneling_probability": self.tunneling_probability},
            )
        if not 0.0 <= self.tunneling_wkb_diagnostic <= 1.0:
            raise QuantumStateError(
                f"T_WKB fuera de [0,1]: {self.tunneling_wkb_diagnostic}",
            )
        if not 0.0 <= self.collapse_threshold < 1.0:
            raise QuantumStateError(
                f"Umbral de colapso fuera de [0,1): {self.collapse_threshold}",
                context={"collapse_threshold": self.collapse_threshold},
            )
        if not 0.0 <= self.collapse_ambiguity <= 1.0:
            raise QuantumStateError(
                f"Ambigüedad de colapso fuera de [0,1]: {self.collapse_ambiguity}",
            )
        if not (self.effective_mass > 0 or math.isinf(self.effective_mass)):
            raise QuantumStateError(
                f"Masa efectiva debe ser > 0 o ∞: {self.effective_mass}",
                context={"effective_mass": self.effective_mass},
            )
        if self.eigenstate == Eigenstate.RECHAZADO:
            if self.kinetic_energy != 0.0:
                raise QuantumStateError(
                    "Estado RECHAZADO no puede tener K ≠ 0.",
                    context={"kinetic_energy": self.kinetic_energy},
                )
            if self.momentum != 0.0:
                raise QuantumStateError(
                    "Estado RECHAZADO no puede tener p ≠ 0.",
                    context={"momentum": self.momentum},
                )

    def __copy__(self) -> "QuantumMeasurement":
        raise QuantumStateError(
            "A5 (No-Clonación): QuantumMeasurement no puede ser clonado "
            "vía copy.copy(). El estado cuántico es único por instancia "
            f"(uid={self.non_cloning_uid}).",
            context={"uid": self.non_cloning_uid},
        )

    def __deepcopy__(self, memo: Dict[int, Any]) -> "QuantumMeasurement":
        raise QuantumStateError(
            "A5 (No-Clonación): QuantumMeasurement no puede ser clonado "
            f"vía copy.deepcopy() (uid={self.non_cloning_uid}).",
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
        return base

    def de_broglie_wavelength(self) -> float:
        """λ = h/p si p > 0, else +∞."""
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
# SECCIÓN 7: PROTOCOLOS DE INTERFACES
# ═════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class ITopologicalWatcher(Protocol):
    """Funtor Top → ℝ₊ vía métrica de Mahalanobis χ²."""

    def get_mahalanobis_threat(self) -> float: ...


@runtime_checkable
class ILaplaceOracle(Protocol):
    """Funtor LTI → ℂ vía polo dominante σ = Re(p*)."""

    def get_dominant_pole_real(self) -> float: ...


@runtime_checkable
class ISheafCohomologyOrchestrator(Protocol):
    """Funtor Sh(X) → ℝ₊ vía energía de frustración E_frust."""

    def get_global_frustration_energy(self) -> float: ...


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8: ÁLGEBRA NUMÉRICA (Morfismos Seguros)
# ═════════════════════════════════════════════════════════════════════════════


class NumericalMorphisms:
    """Morfismos numéricos con garantías algebraicas sobre ℝ ∪ {±∞}."""

    @staticmethod
    def validate_finite_float(
        value: Any, *, name: str, allow_inf: bool = False,
    ) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise QuantumNumericalError(
                f"Parámetro '{name}' no convertible a float: {value!r}",
                context={"name": name, "type": type(value).__name__},
            ) from exc
        if math.isnan(result):
            raise QuantumNumericalError(
                f"Parámetro '{name}' es NaN.", context={"name": name},
            )
        if not allow_inf and math.isinf(result):
            raise QuantumNumericalError(
                f"Parámetro '{name}' es infinito: {result!r}.",
                context={"name": name, "value": result},
            )
        return result

    @staticmethod
    def clamp_to_unit_interval(value: float) -> float:
        if math.isnan(value):
            logger.warning("clamp_to_unit_interval recibió NaN → 0.0.")
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
        num = NumericalMorphisms.validate_finite_float(numerator, name="numerator")
        denom = NumericalMorphisms.validate_finite_float(
            denominator, name="denominator"
        )
        if denom == 0.0:
            logger.warning("División por cero: %s/0 → fallback %s.", num, fallback)
            return fallback
        reg_denom = (
            denom if abs(denom) >= epsilon else math.copysign(epsilon, denom)
        )
        return num / reg_denom

    @staticmethod
    def safe_sqrt(value: float) -> float:
        return math.sqrt(max(0.0, value))

    @staticmethod
    def safe_exp(exponent: float) -> float:
        if exponent <= Const.EXP_UNDERFLOW_CUTOFF:
            return 0.0
        if exponent > Const.EXP_OVERFLOW_CUTOFF:
            logger.warning("Exponente %.2f > cutoff → FLOAT_MAX.", exponent)
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
    def safe_log(value: float, *, floor: float = 1e-300) -> float:
        return math.log(max(value, floor))


NM = NumericalMorphisms


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
            hasher.update(f)
        elif f is None:
            hasher.update(b"|__None__")
        else:
            hasher.update(f"|{f!r}".encode())
    return hasher.hexdigest()


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 9: TEORÍA DE LA INFORMACIÓN
# ═════════════════════════════════════════════════════════════════════════════


class EntropyCalculator:
    """Entropía de Shannon en nats sobre el alfabeto de bytes {0,…,255}.

    H(X) = −Σ p_i ln(p_i).  0 ≤ H ≤ ln(256).  H = 0 ⟺ determinista.
    """

    @staticmethod
    @lru_cache(maxsize=1024)
    def shannon_entropy_bytes(data: bytes) -> float:
        if not isinstance(data, bytes):
            raise QuantumNumericalError(
                f"shannon_entropy_bytes requiere bytes; recibido {type(data).__name__}.",
            )
        if not data:
            return 0.0

        n = len(data)
        byte_counts = [0] * 256
        for b in data:
            byte_counts[b] += 1

        entropy = 0.0
        for c in byte_counts:
            if c == 0:
                continue
            p = c / n
            entropy -= p * math.log(p)

        return max(0.0, min(entropy, Const.MAX_SHANNON_ENTROPY))

    @staticmethod
    def normalized_entropy(data: bytes) -> float:
        """H / ln(256) ∈ [0, 1]."""
        return EntropyCalculator.shannon_entropy_bytes(data) / Const.MAX_SHANNON_ENTROPY

    @staticmethod
    def conditional_entropy(
        data: bytes, condition: Callable[[int], bool],
    ) -> float:
        """H(X|Y) con Y = condition(byte)."""
        if not data:
            return 0.0
        true_bytes = bytes(b for b in data if condition(b))
        false_bytes = bytes(b for b in data if not condition(b))
        n = len(data)
        p_true = len(true_bytes) / n
        p_false = len(false_bytes) / n
        return (
            p_true * EntropyCalculator.shannon_entropy_bytes(true_bytes)
            + p_false * EntropyCalculator.shannon_entropy_bytes(false_bytes)
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 10: SERIALIZACIÓN CANÓNICA
# ═════════════════════════════════════════════════════════════════════════════


class PayloadSerializer:
    """Serialización canónica + SHA-256 + proyección uniforme a [0, 1).

    La proyección usa los 53 bits superiores del digest como mantisa
    canónica de IEEE-754 binary64:  mantissa / 2⁵³ ∈ [0, 1).
    """

    @staticmethod
    def serialize(payload: Mapping[str, Any]) -> bytes:
        if not isinstance(payload, Mapping):
            raise QuantumAdmissionError(
                f"payload debe ser Mapping; recibido {type(payload).__name__}.",
                context={"type": type(payload).__name__},
            )
        try:
            ordered_items = tuple(
                sorted(
                    ((str(k), repr(v)) for k, v in payload.items()),
                    key=lambda kv: kv[0],
                )
            )
            return repr(ordered_items).encode("utf-8", errors="strict")
        except Exception as exc:
            raise QuantumAdmissionError(
                f"Fallo en serialización canónica: {exc}",
                context={"payload_keys": list(payload.keys())},
            ) from exc

    @staticmethod
    @lru_cache(maxsize=512)
    def deterministic_hash(data: bytes) -> bytes:
        return hashlib.sha256(data).digest()

    @staticmethod
    def hash_to_unit_interval(hash_bytes: bytes) -> float:
        """Proyección uniforme bytes → [0, 1) vía 53 bits de mantisa.

        mantissa = (u64 ≫ 11) & (2⁵³−1);  θ = mantissa / 2⁵³ ∈ [0, 1).
        El caso θ = 1 es imposible por construcción.
        """
        if len(hash_bytes) < 8:
            raise QuantumNumericalError(
                f"hash_to_unit_interval requiere ≥ 8 bytes; recibido {len(hash_bytes)}.",
            )
        u64 = int.from_bytes(hash_bytes[:8], byteorder="big", signed=False)
        mantissa = (u64 >> (64 - Const.MANTISSA_BITS)) & Const.MANTISSA_MASK
        return mantissa / float(1 << Const.MANTISSA_BITS)


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 11: SNAPSHOT AMBIENTAL Y CERTIFICADOS
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class EnvironmentalSnapshot:
    """Snapshot consistente de los oráculos al inicio de Φ₁.

    Una sola lectura de cada fuente; se propaga inmutable por el pipeline.
    """

    chi_squared: float
    dominant_pole_real: float
    global_frustration: float
    read_errors: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for fname, val in (
            ("chi_squared", self.chi_squared),
            ("dominant_pole_real", self.dominant_pole_real),
            ("global_frustration", self.global_frustration),
        ):
            if math.isnan(val):
                raise QuantumNumericalError(
                    f"Snapshot '{fname}' es NaN.", context={fname: val},
                )


@dataclass(frozen=True, slots=True)
class GaugeFlatnessCertificate:
    """A4: certificado de planitud gauge U(1) del portal.

    PROXY: E_frust ≤ ε_gauge  ⟺  is_flat.  first_betti ∈ {0,1} es cota
    inferior (F ≢ 0 ⇒ β₁ ≥ 1 en el 1-esqueleto), NUNCA ceil(E_frust).
    """

    is_flat: bool
    field_strength: float
    first_betti: int
    frustration_energy: float
    tolerance: float


@dataclass(frozen=True, slots=True)
class OrthomodularStructureCertificate:
    """A6: certificado del subretículo de ℋ₂ generado por {P_adm, P_rej}.

    Se verifica álgebra 2×2: P²=P, P_adm ⊥ P_rej, P_adm+P_rej=𝟙, [P,Q]=0.
    El subretículo es booleano (4 elementos); 𝒫(ℋ) ambiente sigue siendo
    ortomodular no distributivo.
    """

    projectors_idempotent: bool
    projectors_orthogonal: bool
    projectors_complementary: bool
    projectors_commute: bool
    lattice_is_boolean: bool
    n_states: int
    frobenius_residual: float


@dataclass(frozen=True, slots=True, eq=False)
class Phase1GaugeAdmissionData:
    """DTO inmutable de Φ₁. Precondición constructora de Φ₂."""

    snapshot: EnvironmentalSnapshot
    gauge_certificate: GaugeFlatnessCertificate
    orthomodular_certificate: OrthomodularStructureCertificate
    veto_triggered: bool
    certification_hash: str


@dataclass(frozen=True, slots=True, eq=False)
class Phase2SpectralEnergyData:
    """DTO inmutable de Φ₂. Precondición constructora de Φ₃."""

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


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 12: CALCULADORES FÍSICOS (morfismos reutilizables)
# ═════════════════════════════════════════════════════════════════════════════


class IncidentEnergyCalculator:
    """Energía incidente E = h ν_eff, ν_eff = n / H_eff / 1000.

    El factor 1000 es una escala de calibración del portal (cuantos por
    kilobyte-nat). La incertidumbre se modela por aritmética de intervalos
    con heurística Poisson+entropía: δE/E = 2/√n.
    """

    @staticmethod
    def calculate_from_bytes(data: bytes) -> Tuple[float, NumericInterval]:
        n_bytes = len(data)
        if n_bytes == 0:
            return 0.0, NumericInterval.point(0.0)

        raw_entropy = EntropyCalculator.shannon_entropy_bytes(data)
        effective_entropy = max(raw_entropy, Const.ENTROPY_FLOOR)
        nu = (
            NM.safe_division(
                n_bytes, effective_entropy, epsilon=Const.ENTROPY_FLOOR
            )
            / 1000.0
        )
        E = Const.PLANCK_H * nu
        E = NM.validate_finite_float(E, name="incident_energy")
        if E < 0:
            raise QuantumNumericalError(
                f"Energía incidente negativa: {E}.",
                context={"E": E, "nu": nu, "n_bytes": n_bytes},
            )
        delta_E = E * 2.0 / math.sqrt(max(1, n_bytes))
        interval = NumericInterval.from_value_with_tolerance(E, delta_E)
        logger.debug(
            "Energía incidente: E=%.6e ± %.6e, n=%d B, H=%.4f nats.",
            E, delta_E, n_bytes, raw_entropy,
        )
        return E, interval

    @staticmethod
    def calculate(payload: Mapping[str, Any]) -> Tuple[float, NumericInterval]:
        data = PayloadSerializer.serialize(payload)
        if isinstance(payload, Mapping) and len(payload) == 0:
            return 0.0, NumericInterval.point(0.0)
        return IncidentEnergyCalculator.calculate_from_bytes(data)


class RectangularBarrierCalculator:
    """Transmisión EXACTA de barrera rectangular (tres regímenes) + WKB.

    Estabilidad numérica:
        · |E − V₀| ≤ ε_rel  → fórmula de umbral (evita 0/0).
        · κa ≥ SINH_ASYMPTOTIC_KA  → forma exp(−2κa) / (e^{−2κa} + C)
          en lugar de sinh que overflow.
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
    def _tunnel_transmission_stable(
        E: float, V0: float, kappa_a: float,
    ) -> float:
        r"""T_tunel numéricamente estable, incluida asintótica κa ≫ 1.

        C = V₀² / (16 E Δ),  T = e^{−2κa} / (e^{−2κa} + C)   (κa grande),
        T = [1 + V₀² sinh²(κa) / (4 E Δ)]⁻¹                 (κa moderado).
        """
        if E <= 0.0:
            return 0.0
        Delta = V0 - E
        if Delta <= 0.0:
            return 1.0
        if kappa_a >= Const.SINH_ASYMPTOTIC_KA:
            # T = 1 / (1 + (V₀² / (16 E Δ)) e^{2κa})
            #   = e^{−2κa} / (e^{−2κa} + V₀²/(16 E Δ))
            C = (V0 * V0) / (16.0 * E * Delta)
            exp_m = NM.safe_exp(-2.0 * kappa_a)
            denom = exp_m + C
            if denom <= 0.0 or not math.isfinite(denom):
                return 0.0
            return NM.clamp_to_unit_interval(exp_m / denom)

        sinh_val = NM.safe_sinh(kappa_a)
        if math.isinf(sinh_val) or abs(sinh_val) >= math.sqrt(Const.FLOAT_MAX) / 2.0:
            C = (V0 * V0) / (16.0 * E * Delta)
            exp_m = NM.safe_exp(-2.0 * kappa_a)
            denom = exp_m + C
            return NM.clamp_to_unit_interval(exp_m / denom) if denom > 0 else 0.0
        denom = 1.0 + (V0 * V0 * sinh_val * sinh_val) / (4.0 * E * Delta)
        if denom <= 0.0 or not math.isfinite(denom):
            return 0.0
        return NM.clamp_to_unit_interval(1.0 / denom)

    @staticmethod
    def _over_barrier_transmission(
        E: float, V0: float, m_eff: float, width: float,
    ) -> float:
        r"""T(E>V₀) = [1 + V₀² sin²(k₂ a) / (4 E (E−V₀))]⁻¹."""
        Delta = E - V0
        if Delta <= 0.0:
            return 1.0
        k2 = NM.safe_sqrt(2.0 * m_eff * Delta) / Const.PLANCK_HBAR
        s = math.sin(k2 * width)
        denom = 1.0 + (V0 * V0 * s * s) / (4.0 * E * Delta)
        if denom <= 0.0 or not math.isfinite(denom):
            return 0.0
        return NM.clamp_to_unit_interval(1.0 / denom)

    @staticmethod
    def compute(
        E: float, V0: float, m_eff: float, width: float,
    ) -> Tuple[RectangularBarrierTransmission, Optional[WKBParameters]]:
        E = NM.validate_finite_float(E, name="E_barrier")
        V0 = NM.validate_finite_float(V0, name="V0_barrier")
        m_eff = NM.validate_finite_float(m_eff, name="m_eff_barrier", allow_inf=True)
        width = NM.validate_finite_float(width, name="width_barrier")

        if E < 0 or V0 < 0 or width <= 0:
            raise QuantumNumericalError(
                f"Parámetros no físicos: E={E}, V₀={V0}, a={width}.",
            )

        rel_tol = Const.THRESHOLD_ENERGY_REL_TOL * max(1.0, abs(V0), abs(E))

        # ── Masa infinita: impenetrable si E < V₀; clásica si E > V₀.
        if math.isinf(m_eff):
            if E + rel_tol < V0:
                params = WKBParameters(
                    incident_energy=E, barrier_height=V0 - E,
                    effective_mass=m_eff, barrier_width=width,
                    kappa=float("inf"), integrand=float("inf"),
                    exponent=float("-inf"), kappa_width=float("inf"),
                    validity_parameter=float("inf"),
                )
                return RectangularBarrierTransmission(
                    t_exact=0.0, t_wkb=0.0, t_ratio=float("inf"),
                    kappa_width=float("inf"), is_classical=False, regime="tunnel",
                ), params
            return RectangularBarrierTransmission(
                t_exact=1.0, t_wkb=1.0, t_ratio=1.0,
                kappa_width=0.0, is_classical=True,
                regime="over_barrier" if E > V0 + rel_tol else "threshold",
            ), None

        if m_eff <= 0:
            raise QuantumNumericalError(f"Masa efectiva debe ser > 0: {m_eff}.")

        # ── Umbral E ≈ V₀.
        if abs(E - V0) <= rel_tol:
            t_th = RectangularBarrierCalculator._threshold_transmission(
                V0, m_eff, width
            )
            return RectangularBarrierTransmission(
                t_exact=t_th, t_wkb=1.0, t_ratio=t_th,
                kappa_width=0.0, is_classical=True, regime="threshold",
            ), None

        # ── Sobrebarrera E > V₀ (resonancias de Ramsauer–Townsend).
        if E > V0:
            t_ex = RectangularBarrierCalculator._over_barrier_transmission(
                E, V0, m_eff, width
            )
            return RectangularBarrierTransmission(
                t_exact=t_ex, t_wkb=1.0, t_ratio=t_ex,
                kappa_width=0.0, is_classical=True, regime="over_barrier",
            ), None

        # ── Túnel E < V₀.
        Delta = V0 - E
        integrand = NM.safe_sqrt(2.0 * m_eff * Delta)
        kappa = integrand / Const.PLANCK_HBAR
        kappa_a = kappa * width
        raw_exponent = -2.0 * kappa_a
        validity_param = 1.0 / max(kappa_a, Const.DIVISION_EPSILON)

        params = WKBParameters(
            incident_energy=E, barrier_height=Delta,
            effective_mass=m_eff, barrier_width=width,
            kappa=kappa, integrand=integrand,
            exponent=raw_exponent, kappa_width=kappa_a,
            validity_parameter=validity_param,
        )
        t_exact = RectangularBarrierCalculator._tunnel_transmission_stable(
            E, V0, kappa_a
        )
        t_wkb = NM.clamp_to_unit_interval(NM.safe_exp(raw_exponent))
        if t_wkb > Const.DIVISION_EPSILON:
            t_ratio = t_exact / t_wkb
        elif t_exact > Const.DIVISION_EPSILON:
            t_ratio = float("inf")
        else:
            t_ratio = 1.0

        transmission = RectangularBarrierTransmission(
            t_exact=NM.clamp_to_unit_interval(t_exact),
            t_wkb=t_wkb,
            t_ratio=t_ratio,
            kappa_width=kappa_a,
            is_classical=False,
            regime="tunnel",
        )
        logger.debug(
            "Barrera: E=%.3e, V₀=%.3e, κa=%.3e, T_exact=%.6e, T_WKB=%.6e, "
            "ratio=%.4f, regime=%s.",
            E, V0, kappa_a, t_exact, t_wkb, t_ratio, transmission.regime,
        )
        return transmission, params


class WorkFunctionModulator:
    """Φ(χ²) = Φ₀ exp(α·χ²) vía acoplamiento gauge-topológico."""

    def __init__(self, topo_watcher: ITopologicalWatcher):
        self._topo_watcher = topo_watcher

    def calculate_from_snapshot(
        self, snapshot: EnvironmentalSnapshot,
    ) -> Tuple[float, float]:
        threat = max(0.0, snapshot.chi_squared)
        exponential_factor = NM.safe_exp(Const.ALPHA_THREAT * threat)
        try:
            phi_raw = Const.BASE_WORK_FUNCTION * exponential_factor
        except OverflowError:
            phi_raw = Const.FLOAT_MAX
        if math.isinf(phi_raw):
            phi_raw = Const.FLOAT_MAX
        Phi = max(0.0, NM.validate_finite_float(phi_raw, name="work_function"))
        return Phi, threat


class EffectiveMassModulator:
    """m*(σ) = m₀ / |σ| para σ < −ε;  +∞ para σ ≥ −ε (polo inestable)."""

    def __init__(self, laplace_oracle: ILaplaceOracle):
        self._laplace_oracle = laplace_oracle

    def calculate_from_snapshot(
        self, snapshot: EnvironmentalSnapshot,
    ) -> Tuple[float, float]:
        sigma = NM.validate_finite_float(
            snapshot.dominant_pole_real, name="sigma",
        )
        if sigma >= -Const.SIGMA_CHAOS_TOL:
            logger.warning(
                "Sistema inestable (σ=%.6e ≥ −tol). m* → ∞.", sigma
            )
            return float("inf"), sigma

        m_eff = NM.safe_division(
            Const.BASE_EFFECTIVE_MASS, abs(sigma),
            fallback=float("inf"), epsilon=Const.SIGMA_CHAOS_TOL,
        )
        m_eff = NM.validate_finite_float(m_eff, name="m_eff", allow_inf=True)
        if not math.isinf(m_eff) and m_eff <= 0:
            raise QuantumNumericalError(
                f"Masa efectiva no positiva: m_eff={m_eff}.",
            )
        return m_eff, sigma


class CollapseThresholdGenerator:
    """Umbral determinista θ ∈ [0, 1) vía SHA-256 del payload."""

    @staticmethod
    @lru_cache(maxsize=512)
    def generate(payload_hash: bytes) -> float:
        return PayloadSerializer.hash_to_unit_interval(payload_hash)


class GaugeFlatnessCertifier:
    """A4: certificador de planitud gauge U(1).

    E_frust ≤ ε  ⟺  is_flat.  first_betti = 0 si plano, 1 si no
    (cota inferior; el orquestador de haces posee los Betti reales).
    """

    @staticmethod
    def certify(snapshot: EnvironmentalSnapshot) -> GaugeFlatnessCertificate:
        E_frust = max(0.0, snapshot.global_frustration)
        is_flat = E_frust <= Const.GAUGE_FLATNESS_TOL
        first_betti = 0 if is_flat else 1
        return GaugeFlatnessCertificate(
            is_flat=is_flat,
            field_strength=E_frust,
            first_betti=first_betti,
            frustration_energy=E_frust,
            tolerance=Const.GAUGE_FLATNESS_TOL,
        )


class OrthomodularStructureCertifier:
    """A6: certificador del subretículo booleano de ℋ₂.

    Verifica por álgebra 2×2 (sin NumPy):
        P_adm = |ADMITIDO⟩⟨ADMITIDO| = diag(1, 0)
        P_rej = |RECHAZADO⟩⟨RECHAZADO| = diag(0, 1)
        P² = P,  P_adm P_rej = 0,  P_adm + P_rej = 𝟙,  [P_adm, P_rej] = 0.
    """

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

    @classmethod
    def certify(cls) -> OrthomodularStructureCertificate:
        p_adm: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (1.0, 0.0), (0.0, 0.0)
        )
        p_rej: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (0.0, 0.0), (0.0, 1.0)
        )
        ident: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (1.0, 0.0), (0.0, 1.0)
        )
        zero: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (0.0, 0.0), (0.0, 0.0)
        )

        p_adm2 = cls._matmul(p_adm, p_adm)
        p_rej2 = cls._matmul(p_rej, p_rej)
        prod_ar = cls._matmul(p_adm, p_rej)
        prod_ra = cls._matmul(p_rej, p_adm)
        summ = (
            (p_adm[0][0] + p_rej[0][0], p_adm[0][1] + p_rej[0][1]),
            (p_adm[1][0] + p_rej[1][0], p_adm[1][1] + p_rej[1][1]),
        )
        comm = (
            (prod_ar[0][0] - prod_ra[0][0], prod_ar[0][1] - prod_ra[0][1]),
            (prod_ar[1][0] - prod_ra[1][0], prod_ar[1][1] - prod_ra[1][1]),
        )

        r_idemp = max(cls._frob_diff(p_adm2, p_adm), cls._frob_diff(p_rej2, p_rej))
        r_orth = cls._frob_diff(prod_ar, zero)
        r_comp = cls._frob_diff(summ, ident)
        r_comm = cls._frob_diff(comm, zero)
        residual = float(max(r_idemp, r_orth, r_comp, r_comm))
        tol = 128.0 * Const.MACHINE_EPSILON

        idempotent = r_idemp <= tol
        orthogonal = r_orth <= tol
        complementary = r_comp <= tol
        commute = r_comm <= tol
        boolean = idempotent and orthogonal and complementary and commute

        if not boolean:
            logger.warning(
                "A6: subretículo ℋ₂ degradado. residual=%.3e.", residual
            )

        return OrthomodularStructureCertificate(
            projectors_idempotent=idempotent,
            projectors_orthogonal=orthogonal,
            projectors_complementary=complementary,
            projectors_commute=commute,
            lattice_is_boolean=boolean,
            n_states=2,
            frobenius_residual=residual,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 1: CERTIFICACIÓN GAUGE-COHOMOLÓGICA DEL PORTAL                       ║
# ║                                                                             ║
# ║   Marco formal:                                                             ║
# ║   ─────────                                                                 ║
# ║   Se toma un snapshot ambiental único (χ², σ, E_frust) para congelar        ║
# ║   oráculos potencialmente stateful. Se certifica F ≡ 0 (A4) por el          ║
# ║   proxy E_frust ≤ ε_gauge y el subretículo booleano de ℋ₂ (A6).             ║
# ║                                                                             ║
# ║   ULTIMO MÉTODO DE FASE 1 (morfismo terminal = unidad de Fase 2):            ║
# ║       nest_into_phase2() → Phase2_SpectralEnergyAuditor                      ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase1_GaugeAdmissionCertifier:
    r"""FASE 1 — CERTIFICACIÓN GAUGE-COHOMOLÓGICA DEL PORTAL.

    Cadena interna:
        _validate_oracle_contracts
            → capture_environmental_snapshot
            → certify_gauge_flatness
            → certify_orthomodular_structure
            → certify_gauge_admission_precondition
            → nest_into_phase2          ★ morfismo terminal = unidad de Φ₂
    """

    def __init__(
        self,
        topo_watcher: ITopologicalWatcher,
        laplace_oracle: ILaplaceOracle,
        sheaf_orchestrator: ISheafCohomologyOrchestrator,
    ) -> None:
        self._topo_watcher = topo_watcher
        self._laplace_oracle = laplace_oracle
        self._sheaf_orchestrator = sheaf_orchestrator
        self._validate_oracle_contracts()
        self._orthomodular_cert: Final[OrthomodularStructureCertificate] = (
            OrthomodularStructureCertifier.certify()
        )

    # ─────────────────────────────────────────────────────────────────────
    # 1.1 Contratos de oráculo
    # ─────────────────────────────────────────────────────────────────────
    def _validate_oracle_contracts(self) -> None:
        deps = [
            (self._topo_watcher, ITopologicalWatcher, "topo_watcher",
             ["get_mahalanobis_threat"]),
            (self._laplace_oracle, ILaplaceOracle, "laplace_oracle",
             ["get_dominant_pole_real"]),
            (self._sheaf_orchestrator, ISheafCohomologyOrchestrator,
             "sheaf_orchestrator", ["get_global_frustration_energy"]),
        ]
        for obj, protocol, name, methods in deps:
            if obj is None:
                raise QuantumInterfaceError(
                    f"Dependencia '{name}' es None.",
                    context={"dependency": name},
                )
            for method in methods:
                if not hasattr(obj, method) or not callable(getattr(obj, method)):
                    raise QuantumInterfaceError(
                        f"Dependencia '{name}' no implementa '{method}'.",
                        context={"dependency": name, "method": method},
                    )
            if not isinstance(obj, protocol):
                raise QuantumInterfaceError(
                    f"Dependencia '{name}' no cumple {protocol.__name__}.",
                    context={
                        "dependency": name,
                        "expected_protocol": protocol.__name__,
                        "actual_type": type(obj).__name__,
                    },
                )

    # ─────────────────────────────────────────────────────────────────────
    # 1.2 Snapshot ambiental (lectura única)
    # ─────────────────────────────────────────────────────────────────────
    def capture_environmental_snapshot(self) -> EnvironmentalSnapshot:
        """Lectura defensiva y única de (χ², σ, E_frust)."""
        read_errors: list[str] = []

        try:
            chi_raw = self._topo_watcher.get_mahalanobis_threat()
            chi = max(0.0, NM.validate_finite_float(chi_raw, name="chi_squared"))
        except Exception as exc:
            read_errors.append(f"topo_watcher: {exc}")
            chi = 0.0

        try:
            sigma_raw = self._laplace_oracle.get_dominant_pole_real()
            sigma = NM.validate_finite_float(sigma_raw, name="sigma")
        except Exception as exc:
            read_errors.append(f"laplace_oracle: {exc}")
            sigma = 0.0

        try:
            frust_raw = self._sheaf_orchestrator.get_global_frustration_energy()
            frustration = max(
                0.0, NM.validate_finite_float(frust_raw, name="frustration")
            )
        except Exception as exc:
            read_errors.append(f"sheaf_orchestrator: {exc}")
            frustration = 0.0

        snapshot = EnvironmentalSnapshot(
            chi_squared=chi,
            dominant_pole_real=sigma,
            global_frustration=frustration,
            read_errors=tuple(read_errors),
        )
        logger.debug(
            "[Φ₁.1] Snapshot: χ²=%.3e, σ=%.3e, E_frust=%.3e, errors=%d.",
            chi, sigma, frustration, len(read_errors),
        )
        return snapshot

    # ─────────────────────────────────────────────────────────────────────
    # 1.3 Planitud gauge U(1)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def certify_gauge_flatness(
        snapshot: EnvironmentalSnapshot,
    ) -> GaugeFlatnessCertificate:
        """A4: F ≡ 0  ⟺  E_frust ≤ ε_gauge;  β₁ ∈ {0,1} cota inferior."""
        cert = GaugeFlatnessCertifier.certify(snapshot)
        if cert.is_flat:
            logger.debug(
                "[Φ₁.2] A4 ✓ F ≡ 0: E_frust=%.3e, β₁≥%d.",
                cert.frustration_energy, cert.first_betti,
            )
        else:
            logger.warning(
                "[Φ₁.2] A4 ✗ F ≢ 0: E_frust=%.3e, β₁≥%d.",
                cert.frustration_energy, cert.first_betti,
            )
        return cert

    # ─────────────────────────────────────────────────────────────────────
    # 1.4 Retículo ortomodular ℋ₂
    # ─────────────────────────────────────────────────────────────────────
    def certify_orthomodular_structure(self) -> OrthomodularStructureCertificate:
        """A6: álgebra 2×2 de {P_adm, P_rej} (certificado cacheado)."""
        return self._orthomodular_cert

    # ─────────────────────────────────────────────────────────────────────
    # 1.5 Emisión del DTO de Φ₁ (pre-terminal)
    # ─────────────────────────────────────────────────────────────────────
    def certify_gauge_admission_precondition(self) -> Phase1GaugeAdmissionData:
        r"""Certifica A4 ∧ A6 y emite Phase1GaugeAdmissionData.

        Cadena:
            oráculos ──(snapshot)──────────▶ (χ², σ, E_frust)
                     ──(gauge_flatness)────▶ GaugeFlatnessCertificate
                     ──(orthomodular)──────▶ OrthomodularStructureCertificate
                     ──(emit DTO)──────────▶ precondición de Φ₂
        """
        snapshot = self.capture_environmental_snapshot()
        if snapshot.read_errors:
            logger.warning(
                "[Φ₁] Snapshot con errores de lectura: %s.", snapshot.read_errors
            )
        gauge = self.certify_gauge_flatness(snapshot)
        ortho = self.certify_orthomodular_structure()
        veto = not gauge.is_flat
        cert_hash = _compute_dto_hash(
            snapshot.chi_squared,
            snapshot.dominant_pole_real,
            snapshot.global_frustration,
            gauge.is_flat,
            gauge.first_betti,
            gauge.field_strength,
            ortho.lattice_is_boolean,
            ortho.frobenius_residual,
            veto,
        )
        logger.info(
            "[Φ₁ ✓] Phase1GaugeAdmissionData: flat=%s, β₁≥%d, "
            "boolean=%s, veto=%s, hash=%s.",
            gauge.is_flat, gauge.first_betti, ortho.lattice_is_boolean,
            veto, cert_hash[:16] + "...",
        )
        return Phase1GaugeAdmissionData(
            snapshot=snapshot,
            gauge_certificate=gauge,
            orthomodular_certificate=ortho,
            veto_triggered=veto,
            certification_hash=cert_hash,
        )

    def emit_veto_measurement(
        self, data: Phase1GaugeAdmissionData,
    ) -> QuantumMeasurement:
        """Colapsa a |RECHAZADO⟩ por veto gauge A4 (fail-secure, no excepción)."""
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
                f"tol = {cert.tolerance:.6e}. β₁(K) ≥ {cert.first_betti} > 0. "
                "Obstrucción topológica impide admisión."
            ),
            tunneling_wkb_diagnostic=0.0,
            kappa_width_diagnostic=0.0,
            collapse_ambiguity=0.0,
            barrier_regime="veto",
        )

    # ─────────────────────────────────────────────────────────────────────
    # 1.6 ★ MORFISMO TERMINAL DE FASE 1 ★
    #     Tipo de retorno = objeto inicial de la FASE 2.
    #     last(Φ₁) = unit(Φ₂) = Phase2_SpectralEnergyAuditor.
    # ─────────────────────────────────────────────────────────────────────
    def nest_into_phase2(self) -> "Phase2_SpectralEnergyAuditor":
        r"""★ MORFISMO TERMINAL DE FASE 1 / UNIDAD DE LA FASE 2 ★

        Composición estricta F₁ ⊣ F₂:
            nest_into_phase2  :=  Phase2_SpectralEnergyAuditor
                                  ∘ certify_gauge_admission_precondition.

        El auditor espectral nace ya alimentado con Phase1GaugeAdmissionData;
        su constructor ES la continuación formal de este método.
        """
        phase1_data = self.certify_gauge_admission_precondition()
        return Phase2_SpectralEnergyAuditor(phase1_data)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 2: REGULACIÓN ESPECTRAL DE ENERGÍA, Φ Y MASA EFECTIVA                ║
# ║                                                                             ║
# ║   ★ INICIO FORMAL = continuación de Phase1.nest_into_phase2 ★               ║
# ║   Precondición constructora: Phase1GaugeAdmissionData.                       ║
# ║                                                                             ║
# ║   ULTIMO MÉTODO DE FASE 2 (morfismo terminal = unidad de Fase 3):            ║
# ║       nest_into_phase3(payload) → Phase3_BarrierCollapseProjector            ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase2_SpectralEnergyAuditor:
    r"""FASE 2 — REGULACIÓN ESPECTRAL DE ENERGÍA, Φ Y MASA EFECTIVA.

    ★ INICIO FORMAL = continuación del morfismo terminal de la FASE 1 ★

    Cadena interna:
        __init__(Phase1GaugeAdmissionData)     ← unidad heredada de Φ₁
            → compute_incident_energy
            → compute_work_function
            → compute_effective_mass
            → classify_barrier_regime
            → audit_spectral_energy
            → nest_into_phase3                 ★ morfismo terminal = unidad de Φ₃
    """

    def __init__(self, phase1_certification: Phase1GaugeAdmissionData) -> None:
        r"""★ CONTINUACIÓN DE FASE 1 / INICIO DE FASE 2 ★

        Args
        ────
        phase1_certification : Phase1GaugeAdmissionData
            Salida de `certify_gauge_admission_precondition`, inyectada por
            `nest_into_phase2`.
        """
        if not isinstance(phase1_certification, Phase1GaugeAdmissionData):
            raise TypeError(
                "Phase2_SpectralEnergyAuditor requiere Phase1GaugeAdmissionData "
                "como precondición (Fase 1)."
            )
        self._p1: Final[Phase1GaugeAdmissionData] = phase1_certification
        self._work_mod = WorkFunctionModulator  # acceso estático vía snapshot
        self._mass_mod = EffectiveMassModulator

    @property
    def phase1(self) -> Phase1GaugeAdmissionData:
        return self._p1

    # ─────────────────────────────────────────────────────────────────────
    # 2.1 Energía incidente con intervalos
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def compute_incident_energy(
        payload: Mapping[str, Any],
    ) -> Tuple[float, NumericInterval, bytes, bytes]:
        """E = hν ± δE, más (payload_bytes, payload_hash) para Φ₃."""
        data = PayloadSerializer.serialize(payload)
        if isinstance(payload, Mapping) and len(payload) == 0:
            E, interval = 0.0, NumericInterval.point(0.0)
        else:
            E, interval = IncidentEnergyCalculator.calculate_from_bytes(data)
        payload_hash = PayloadSerializer.deterministic_hash(data)
        logger.debug("[Φ₂.1] E=%.6e, incertidumbre=%s.", E, interval)
        return E, interval, data, payload_hash

    # ─────────────────────────────────────────────────────────────────────
    # 2.2 Función de trabajo Φ(χ²)
    # ─────────────────────────────────────────────────────────────────────
    def compute_work_function(self) -> Tuple[float, float]:
        dummy = object.__new__(WorkFunctionModulator)
        Phi, threat = WorkFunctionModulator.calculate_from_snapshot(
            dummy, self._p1.snapshot
        )
        logger.debug("[Φ₂.2] Φ=%.6e (χ²=%.6e).", Phi, threat)
        return Phi, threat

    # ─────────────────────────────────────────────────────────────────────
    # 2.3 Masa efectiva m*(σ)
    # ─────────────────────────────────────────────────────────────────────
    def compute_effective_mass(self) -> Tuple[float, float]:
        dummy = object.__new__(EffectiveMassModulator)
        m_eff, sigma = EffectiveMassModulator.calculate_from_snapshot(
            dummy, self._p1.snapshot
        )
        logger.debug("[Φ₂.3] m*=%.6e (σ=%.6e).", m_eff, sigma)
        return m_eff, sigma

    # ─────────────────────────────────────────────────────────────────────
    # 2.4 Clasificación del régimen E ≶ Φ
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def classify_barrier_regime(
        E: float, Phi: float,
    ) -> Tuple[float, bool, bool]:
        """(kinetic_surplus, is_over_barrier, is_at_threshold)."""
        rel = Const.THRESHOLD_ENERGY_REL_TOL * max(1.0, abs(Phi), abs(E))
        at_th = abs(E - Phi) <= rel
        over = E > Phi and not at_th
        surplus = max(0.0, E - Phi) if over else 0.0
        return surplus, over, at_th

    # ─────────────────────────────────────────────────────────────────────
    # 2.5 Auditoría espectral (pre-terminal de Φ₂)
    # ─────────────────────────────────────────────────────────────────────
    def audit_spectral_energy(
        self, payload: Mapping[str, Any],
    ) -> Phase2SpectralEnergyData:
        """Produce Phase2SpectralEnergyData, precondición estricta de Φ₃."""
        if not isinstance(payload, Mapping):
            raise QuantumAdmissionError(
                f"payload debe ser Mapping; recibido {type(payload).__name__}.",
                context={"type": type(payload).__name__},
            )
        E, interval, _data, payload_hash = self.compute_incident_energy(payload)
        Phi, threat = self.compute_work_function()
        m_eff, sigma = self.compute_effective_mass()
        surplus, over, at_th = self.classify_barrier_regime(E, Phi)

        cert_hash = _compute_dto_hash(
            self._p1.certification_hash,
            E, Phi, threat, m_eff if math.isfinite(m_eff) else float("inf"),
            sigma, surplus, over, at_th, payload_hash,
        )
        logger.info(
            "[Φ₂ ✓] Phase2SpectralEnergyData: E=%.4e, Φ=%.4e, m*=%.4e, "
            "over=%s, thresh=%s, hash=%s.",
            E, Phi, m_eff, over, at_th, cert_hash[:16] + "...",
        )
        return Phase2SpectralEnergyData(
            phase1=self._p1,
            incident_energy=float(E),
            energy_interval=interval,
            work_function=float(Phi),
            threat_level=float(threat),
            effective_mass=m_eff,
            dominant_pole_real=float(sigma),
            kinetic_surplus=float(surplus),
            is_over_barrier=bool(over),
            is_at_threshold=bool(at_th),
            payload_hash=payload_hash,
            certification_hash=cert_hash,
        )

    # ─────────────────────────────────────────────────────────────────────
    # 2.6 ★ MORFISMO TERMINAL DE FASE 2 ★
    #     Tipo de retorno = objeto inicial de la FASE 3.
    #     last(Φ₂) = unit(Φ₃) = Phase3_BarrierCollapseProjector.
    # ─────────────────────────────────────────────────────────────────────
    def nest_into_phase3(
        self, payload: Mapping[str, Any],
    ) -> "Phase3_BarrierCollapseProjector":
        r"""★ MORFISMO TERMINAL DE FASE 2 / UNIDAD DE LA FASE 3 ★

        Composición estricta F₂ ⊣ F₃:
            nest_into_phase3(payload)  :=  Phase3_BarrierCollapseProjector
                                           ∘ audit_spectral_energy(payload).

        El projector de barrera nace ya alimentado con Phase2SpectralEnergyData;
        su constructor ES la continuación formal de este método.

        Raises
        ──────
        CohomologicalVetoError si Φ₁ marcó veto (la cadena no debió llegar aquí).
        """
        if self._p1.veto_triggered:
            raise CohomologicalVetoError(
                "[Φ₂] nest_into_phase3 invocado tras veto gauge A4. "
                "El orquestador debe cortocircuitar a emit_veto_measurement.",
                context={"cert_hash": self._p1.certification_hash},
            )
        phase2_data = self.audit_spectral_energy(payload)
        return Phase3_BarrierCollapseProjector(phase2_data)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 3: TRANSMISIÓN EXACTA DE BARRERA Y COLAPSO DE BORN                   ║
# ║                                                                             ║
# ║   ★ INICIO FORMAL = continuación de Phase2.nest_into_phase3 ★               ║
# ║   Precondición constructora: Phase2SpectralEnergyData.                       ║
# ║                                                                             ║
# ║   ULTIMO MÉTODO DE FASE 3 (morfismo terminal del módulo):                    ║
# ║       collapse_wavefunction() → QuantumMeasurement                           ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase3_BarrierCollapseProjector:
    r"""FASE 3 — TRANSMISIÓN EXACTA DE BARRERA Y COLAPSO DE BORN.

    ★ INICIO FORMAL = continuación del morfismo terminal de la FASE 2 ★

    Cadena interna:
        __init__(Phase2SpectralEnergyData)     ← unidad heredada de Φ₂
            → compute_barrier_transmission
            → compute_collapse_threshold
            → compute_collapse_ambiguity
            → collapse_wavefunction            ★ morfismo terminal del módulo
    """

    def __init__(
        self,
        phase2_audit: Phase2SpectralEnergyData,
        *,
        strict_wkb: bool = False,
    ) -> None:
        r"""★ CONTINUACIÓN DE FASE 2 / INICIO DE FASE 3 ★

        Args
        ────
        phase2_audit : Phase2SpectralEnergyData
            Salida de `audit_spectral_energy`, inyectada por `nest_into_phase3`.
        strict_wkb : si True, κa ≤ umbral en régimen túnel → WKBValidityError.
        """
        if not isinstance(phase2_audit, Phase2SpectralEnergyData):
            raise TypeError(
                "Phase3_BarrierCollapseProjector requiere Phase2SpectralEnergyData "
                "como precondición (Fase 2)."
            )
        self._p2: Final[Phase2SpectralEnergyData] = phase2_audit
        self._strict_wkb: Final[bool] = bool(strict_wkb)

    @property
    def phase2(self) -> Phase2SpectralEnergyData:
        return self._p2

    # ─────────────────────────────────────────────────────────────────────
    # 3.1 Transmisión exacta + diagnóstico WKB
    # ─────────────────────────────────────────────────────────────────────
    def compute_barrier_transmission(
        self,
    ) -> Tuple[RectangularBarrierTransmission, Optional[WKBParameters]]:
        p2 = self._p2
        transmission, params = RectangularBarrierCalculator.compute(
            E=p2.incident_energy,
            V0=p2.work_function,
            m_eff=p2.effective_mass,
            width=Const.BARRIER_WIDTH,
        )
        if self._strict_wkb and not transmission.is_classical and params is not None:
            if not params.is_valid_semiclassical_regime():
                raise WKBValidityError(
                    f"WKB fuera del régimen κa ≫ 1: κa={params.kappa_width:.3e} "
                    f"≤ {Const.WKB_VALIDITY_THRESHOLD:.3e}.",
                    context={
                        "kappa_width": params.kappa_width,
                        "E": p2.incident_energy,
                        "Phi": p2.work_function,
                    },
                )
        logger.debug(
            "[Φ₃.1] T_exact=%.6e, T_WKB=%.6e, κa=%.3e, regime=%s.",
            transmission.t_exact, transmission.t_wkb,
            transmission.kappa_width, transmission.regime,
        )
        return transmission, params

    # ─────────────────────────────────────────────────────────────────────
    # 3.2 Umbral de colapso θ ∈ [0, 1)
    # ─────────────────────────────────────────────────────────────────────
    def compute_collapse_threshold(self) -> float:
        theta = CollapseThresholdGenerator.generate(self._p2.payload_hash)
        logger.debug("[Φ₃.2] θ=%.6f.", theta)
        return theta

    # ─────────────────────────────────────────────────────────────────────
    # 3.3 Ambigüedad ortomodular del colapso
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def compute_collapse_ambiguity(
        T_primary: float, theta: float,
    ) -> CollapseAmbiguity:
        boundary = abs(T_primary - theta)
        raw = 1.0 - 2.0 * boundary
        ambiguity = NM.clamp_to_unit_interval(raw)
        degenerate = ambiguity >= (1.0 - 1e-3)
        return CollapseAmbiguity(
            boundary_distance=float(boundary),
            ambiguity=float(ambiguity),
            is_orthomodular_degenerate=bool(degenerate),
        )

    # ─────────────────────────────────────────────────────────────────────
    # 3.4 ★ MORFISMO TERMINAL DE FASE 3 / DEL MÓDULO ★
    #     Cierra el funtor maestro 𝒵_Admission = Φ₃ ∘ Φ₂ ∘ Φ₁.
    # ─────────────────────────────────────────────────────────────────────
    def collapse_wavefunction(self) -> QuantumMeasurement:
        r"""★ MORFISMO TERMINAL DE FASE 3 / DEL MÓDULO ★

        Cadena funtorial de Φ₃:
            Phase2SpectralEnergyData
              ──(compute_barrier_transmission)──▶ (T_exact, T_WKB, κa)
              ──(compute_collapse_threshold)────▶ θ ∈ [0, 1)
              ──(compute_collapse_ambiguity)────▶ CollapseAmbiguity
              ──(Born T ≥ θ)────────────────────▶ |ADMITIDO⟩ / |RECHAZADO⟩
              ──(emit QuantumMeasurement)───────▶ salida de 𝒵_Admission

        Criterio de colapso (A2, Born-like determinista):
            |ADMITIDO⟩  ⟺  T_exact ≥ θ.
        """
        p2 = self._p2
        transmission, wkb_params = self.compute_barrier_transmission()
        theta = self.compute_collapse_threshold()
        T_primary = transmission.t_exact
        amb = self.compute_collapse_ambiguity(T_primary, theta)
        admitted = T_primary >= theta

        E, Phi, m_eff = p2.incident_energy, p2.work_function, p2.effective_mass
        sigma, threat = p2.dominant_pole_real, p2.threat_level

        if not admitted:
            measurement = QuantumMeasurement(
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
                    f"Rechazo probabilístico: T_exact={T_primary:.6e} < "
                    f"θ={theta:.6f}. Colapso a |RECHAZADO⟩ según A2 "
                    f"(régimen={transmission.regime})."
                ),
                wkb_parameters=wkb_params,
                measurement_uncertainty=p2.energy_interval,
                tunneling_wkb_diagnostic=transmission.t_wkb,
                kappa_width_diagnostic=transmission.kappa_width,
                collapse_ambiguity=amb.ambiguity,
                barrier_regime=transmission.regime,
            )
            QuantumLogger.log_measurement(measurement, level=logging.INFO)
            return measurement

        if transmission.regime in ("over_barrier", "threshold") or E >= Phi:
            kinetic_energy = max(Const.MIN_KINETIC_ENERGY, E - Phi)
            reason = (
                f"Admisión clásica/resonante: E={E:.6e}, Φ={Phi:.6e}, "
                f"régimen={transmission.regime}, T_exact={T_primary:.6e} ≥ θ={theta:.6f}."
            )
        else:
            kinetic_energy = Const.MIN_KINETIC_ENERGY
            reason = (
                f"Admisión por túnel: E={E:.6e} < Φ={Phi:.6e}, "
                f"T_exact={T_primary:.6e} ≥ θ={theta:.6f}."
            )

        m_for_p = m_eff if math.isfinite(m_eff) else Const.BASE_EFFECTIVE_MASS
        momentum = NM.safe_sqrt(2.0 * m_for_p * kinetic_energy)
        momentum = NM.validate_finite_float(momentum, name="momentum")

        measurement = QuantumMeasurement(
            eigenstate=Eigenstate.ADMITIDO,
            incident_energy=E,
            work_function=Phi,
            tunneling_probability=T_primary,
            kinetic_energy=kinetic_energy,
            momentum=momentum,
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
        )
        QuantumLogger.log_measurement(measurement, level=logging.INFO)
        return measurement


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   ORQUESTADOR: 𝒵_Admission = Φ₃ ∘ Φ₂ ∘ Φ₁                                   ║
# ║                                                                             ║
# ║   Anidamiento EXCLUSIVO vía morfismos terminales:                           ║
# ║       Φ₁.nest_into_phase2()           ⟶  Phase2                             ║
# ║       Φ₂.nest_into_phase3(payload)    ⟶  Phase3                             ║
# ║       Φ₃.collapse_wavefunction()      ⟶  QuantumMeasurement                 ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class QuantumAdmissionGate(Morphism):
    r"""Operador de proyección de Hilbert como morfismo categórico.

    Funtor F: 𝒞_Ext → 𝒞_Phys.  Pipeline anidado Φ₁ ⊣ Φ₂ ⊣ Φ₃.
    Fast-fail A4: si el portal no es plano, se emite |RECHAZADO⟩ sin Φ₂ ni Φ₃.
    """

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
        self._phase1_factory = lambda: Phase1_GaugeAdmissionCertifier(
            topo_watcher, laplace_oracle, sheaf_orchestrator
        )
        # Validación eager de contratos (falla en construcción, no en evaluate).
        self._phase1_factory()
        logger.info(
            "QuantumAdmissionGate v%s inicializada. strict_wkb=%s.",
            __version__, self._strict_wkb,
        )

    @property
    def domain(self) -> frozenset:
        return frozenset()

    @property
    def codomain(self) -> Stratum:
        return Stratum.PHYSICS

    def evaluate_admission(
        self, payload: Mapping[str, Any],
    ) -> QuantumMeasurement:
        r"""Ejecuta 𝒵_Admission(payload) anidando Φ₁ ⊣ Φ₂ ⊣ Φ₃.

        Fast-fail [A4]: si F ≢ 0, se aborta Φ₂/Φ₃ y se colapsa a |RECHAZADO⟩.
        """
        if not isinstance(payload, Mapping):
            raise QuantumAdmissionError(
                f"payload debe ser Mapping; recibido {type(payload).__name__}.",
                context={"type": type(payload).__name__},
            )
        logger.debug("Iniciando 𝒵_Admission. keys=%s.", list(payload.keys()))

        # Φ₁ → unidad de Φ₂
        phase1 = self._phase1_factory()
        phase2 = phase1.nest_into_phase2()
        p1 = phase2.phase1

        if p1.veto_triggered:
            measurement = phase1.emit_veto_measurement(p1)
            QuantumLogger.log_measurement(measurement, level=logging.WARNING)
            return measurement

        # Φ₂ → unidad de Φ₃
        phase3 = phase2.nest_into_phase3(payload)
        # El flag strict_wkb vive en Φ₃; se reinyecta si el factory no lo porta.
        if self._strict_wkb and not phase3._strict_wkb:
            phase3 = Phase3_BarrierCollapseProjector(
                phase3.phase2, strict_wkb=True
            )

        # Φ₃ → DTO terminal
        return phase3.collapse_wavefunction()

    def __call__(self, state: CategoricalState) -> CategoricalState:
        """Aplica el morfismo de admisión a un estado categórico.

        Preservación funtorial: F(g ∘ f) = F(g) ∘ F(f), F(id) = id.
        Admitido  → strata' = strata ∪ {PHYSICS}.
        Rechazado → strata' = ∅, context con quantum_error.
        """
        if not isinstance(state, CategoricalState):
            raise QuantumAdmissionError(
                f"state debe ser CategoricalState; recibido {type(state).__name__}.",
                context={"type": type(state).__name__},
            )

        payload = getattr(state, "payload", None)
        if not isinstance(payload, Mapping):
            raise QuantumAdmissionError(
                f"state.payload debe ser Mapping; recibido {type(payload).__name__}.",
                context={"payload_type": type(payload).__name__},
            )

        original_context = getattr(state, "context", None)
        context = dict(original_context) if original_context else {}
        measurement = self.evaluate_admission(payload)

        if measurement.eigenstate == Eigenstate.RECHAZADO:
            error_msg = (
                f"VETO CUÁNTICO | {measurement.eigenstate} | "
                f"E={measurement.incident_energy:.3e} | "
                f"Φ={measurement.work_function:.3e} | "
                f"T={measurement.tunneling_probability:.3e} | "
                f"veto={measurement.frustration_veto} | "
                f"razón: {measurement.admission_reason}"
            )
            logger.error(error_msg)
            return CategoricalState(
                payload=payload,
                context={
                    **context,
                    "quantum_error": error_msg,
                    "quantum_admission": measurement,
                },
                validated_strata=frozenset(),
            )

        logger.info(
            "ADMISIÓN CUÁNTICA | |ADMITIDO⟩ | p=%.3e | E=%.3e | Φ=%.3e | T=%.3e.",
            measurement.momentum, measurement.incident_energy,
            measurement.work_function, measurement.tunneling_probability,
        )
        new_strata = state.validated_strata | {Stratum.PHYSICS}
        return CategoricalState(
            payload=payload,
            context={
                **context,
                "quantum_momentum": measurement.momentum,
                "quantum_admission": measurement,
            },
            validated_strata=new_strata,
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 14: TESTING Y EJEMPLOS
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    print("\n" + "═" * 80)
    print(f"QUANTUM ADMISSION GATE v{__version__} — SUITE DE TESTING")
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

    print("TEST 1: NumericInterval arithmetic")
    a = NumericInterval(1.0, 2.0)
    b = NumericInterval(3.0, 4.0)
    print(f"  {a} + {b} = {a + b}")
    print(f"  {a} * {b} = {a * b}")
    print(f"  {a} − {b} = {a - b}")
    print(f"  contains(1.5) = {a.contains(1.5)}")
    print()

    print("TEST 2: No-cloning guard (A5)")
    gate = QuantumAdmissionGate(
        topo_watcher=MockTopologicalWatcher(threat=0.5),
        laplace_oracle=MockLaplaceOracle(pole=-0.5),
        sheaf_orchestrator=MockSheafOrchestrator(frustration=1e-12),
    )
    m = gate.evaluate_admission({"key": "value"})
    print(f"  Medición uid={m.non_cloning_uid}, state={m.eigenstate}")
    try:
        _ = _copy_module.copy(m)
        print("  ✗ copy.copy() NO fue bloqueado")
    except QuantumStateError as exc:
        print(f"  ✓ copy.copy() bloqueado: {str(exc)[:80]}...")
    try:
        _ = _copy_module.deepcopy(m)
        print("  ✗ copy.deepcopy() NO fue bloqueado")
    except QuantumStateError as exc:
        print(f"  ✓ copy.deepcopy() bloqueado: {str(exc)[:80]}...")
    print()

    print("TEST 3: Admisión anidada Φ₁ ⊣ Φ₂ ⊣ Φ₃")
    m = gate.evaluate_admission({"endpoint": "/api/test", "data": "A" * 500})
    print(f"  State={m.eigenstate}, T_exact={m.tunneling_probability:.6e}")
    print(
        f"  T_WKB={m.tunneling_wkb_diagnostic:.6e}, "
        f"κa={m.kappa_width_diagnostic:.3e}, regime={m.barrier_regime}"
    )
    print(f"  ambiguity={m.collapse_ambiguity:.4f}")
    print()

    print("TEST 4: Veto gauge A4 (E_frust alta)")
    gate_veto = QuantumAdmissionGate(
        topo_watcher=MockTopologicalWatcher(threat=0.5),
        laplace_oracle=MockLaplaceOracle(pole=-0.5),
        sheaf_orchestrator=MockSheafOrchestrator(frustration=1.0),
    )
    m_veto = gate_veto.evaluate_admission({"key": "value"})
    print(f"  State={m_veto.eigenstate}, veto={m_veto.frustration_veto}")
    print(f"  Reason: {m_veto.admission_reason}")
    print()

    print("TEST 5: Reproducibilidad de θ")
    m1 = gate.evaluate_admission({"key": "value", "number": 42})
    m2 = gate.evaluate_admission({"key": "value", "number": 42})
    print(f"  θ₁={m1.collapse_threshold:.10f}")
    print(f"  θ₂={m2.collapse_threshold:.10f}")
    print(f"  Idénticos: {m1.collapse_threshold == m2.collapse_threshold}")
    print(f"  UIDs distintos: {m1.non_cloning_uid != m2.non_cloning_uid}")
    print()

    print("TEST 6: T_exact vs T_WKB (túnel, umbral, sobrebarrera)")
    for E in [0.1, 1.0, 5.0, 9.9, 10.0, 12.0, 20.0]:
        t, p = RectangularBarrierCalculator.compute(
            E, V0=10.0, m_eff=1.0, width=1.0
        )
        print(
            f"  E={E:5.1f}: T_exact={t.t_exact:.6e}, T_WKB={t.t_wkb:.6e}, "
            f"κa={t.kappa_width:.3f}, regime={t.regime}"
        )
    print()

    print("TEST 7: Subretículo ortomodular ℋ₂ (A6)")
    ortho = OrthomodularStructureCertifier.certify()
    print(
        f"  idempotent={ortho.projectors_idempotent}, "
        f"orthogonal={ortho.projectors_orthogonal}, "
        f"complementary={ortho.projectors_complementary}, "
        f"commute={ortho.projectors_commute}, "
        f"boolean={ortho.lattice_is_boolean}, "
        f"‖·‖_F residual={ortho.frobenius_residual:.3e}"
    )
    print()

    print("═" * 80)
    print("SUITE DE TESTING COMPLETADA")
    print("═" * 80 + "\n")


__all__ = [
    "QuantumAdmissionError",
    "QuantumNumericalError",
    "QuantumInterfaceError",
    "QuantumStateError",
    "WKBValidityError",
    "CohomologicalVetoError",
    "PhysicalConstants",
    "NumericInterval",
    "WKBParameters",
    "RectangularBarrierTransmission",
    "CollapseAmbiguity",
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