# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Routon Dependency Agent (Soberano de Calibre Routónico 128D).       ║
║ Ruta   : app/agents/core/routon_dependency_agent.py                          ║
║ Versión: 3.0.0-Doctoral-OODA-Heyting-CayleyDickson-128D-IRAM-ESP32-Nested3   ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y DE GOBERNANZA DE LAZO CERRADO:                         ║
║ Este agente supervisor ciber-físico opera en el Estrato de la Sabiduría      ║
║ (V_W, Nivel 0) u Omega (V_Ω, Nivel 0.5) para gobernar síncronamente al       ║
║ Motor de Calibre Routónico 128D [routon_dependency_engine.py] en la FPU.     ║
║                                                                              ║
║ Audita, como predicados locales en el retículo de Heyting Ω_3:               ║
║   1. Regularidad de Banach ℓ¹/ℓ²/ℓ^∞ sobre ℝ^{128} (equivalencia de normas).  ║
║   2. Asociador eneagonal de 9 vías A_9 y diámetro de Stasheff A_4.           ║
║   3. Distorsión de Moufang A_M(R,S,T)=(R(ST))R-(RS)(TR) (grado 4).           ║
║   4. Composición de Hurwitz / submultiplicatividad de Banach.                ║
║   5. Penetración no trivial del cono nulo 𝒩(ℝou), σ_min(L_R).                ║
║                                                                              ║
║ AXIOMAS DE GOBERNANZA:                                                       ║
║   (H3)  Ω_3 = {VETOED ≺ DEGRADED ≺ COHERENT} es la cadena de Heyting de      ║
║         tres elementos (álgebra de Gödel): ∧=mín, ∨=máx,                     ║
║         a → b = ⊤ si a ≼ b, si no b;  ¬a = a → ⊥.                            ║
║   (M)   El veredicto global es el meet de los predicados locales.            ║
║   (Γ)   Modalidad de gracia: Γ(DEGRADED)=DEGRADED si t<T, si no VETOED.      ║
║   (σ)   Override HMAC-SHA256 no promociona a ⊤: σ(DEGRADED)=DEGRADED         ║
║         con Γ desactivada. El veto duro no es anulable.                      ║
║   (B)   1 ≤ ‖x‖₁/‖x‖₂ ≤ √128,  1 ≤ ‖x‖₂/‖x‖_∞ ≤ √128,                        ║
║         1 ≤ ‖x‖₁/‖x‖_∞ ≤ 128  (equivalencia de normas en ℝ^{128}).           ║
║   (Mf)  A_M(R,S,T)=(R(ST))R-(RS)(TR);  δ_M=‖A_M‖/(‖R‖²‖S‖‖T‖).               ║
║   (A9)  A_9 es el asociador de asociación extrema de 9 factores.             ║
║                                                                              ║
║ ARQUITECTURA FUNCTORIAL EN TRES FASES ANIDADAS (OODA Ω_3):                   ║
║   Fase 1  Observe : saneamiento, Banach, estados FPU 128D.                   ║
║           Morfismo terminal : observe_nine_way → RoutonObservationKernel.    ║
║   Fase 2  Orient + Decide : motor 128D, A_M, A_9, meet de Heyting, Γ.        ║
║           Objeto inicial    : RoutonObservationKernel.                       ║
║           Morfismo terminal : govern_from_observation_kernel                 ║
║                               → RoutonGovernedDecision.                      ║
║   Fase 3  Act : Crowbar BT151 simulado (GPIO14, < 400 ns spec.) y sello.     ║
║           Objeto inicial    : RoutonGovernedDecision.                        ║
║           Morfismo terminal : audit_eneagonal_cycle                          ║
║                               → RoutonAgentCertificate.                      ║
║                                                                              ║
║ Anidación ontológica: Phase3 ⊏ Phase2 ⊏ Phase1  ⇒  Act∘Decide∘Observe.       ║
║ El último morfismo de la Fase k es el objeto inicial de la Fase k+1.         ║
║ En particular, Phase1.observe_nine_way devuelve un RoutonObservationKernel   ║
║ que es consumido por Phase2.continue_from_observation_kernel.                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
import struct
import sys
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Final, Iterable, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

__version__: Final[str] = (
    "3.0.0-Doctoral-OODA-Heyting-CayleyDickson-128D-IRAM-ESP32-Nested3"
)

# ------------------------------------------------------------------------------
# Rutas de compatibilidad para entornos de laboratorio / scratch / artifacts.
# ------------------------------------------------------------------------------
for extra_path in ("/workspace/scratch", "/workspace/artifacts"):
    if os.path.exists(extra_path) and extra_path not in sys.path:
        sys.path.insert(0, extra_path)

# ------------------------------------------------------------------------------
# Interfaces de resiliencia del ecosistema APU.
# ------------------------------------------------------------------------------
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:
    class Morphism:  # type: ignore[no-redef]
        """Morfismo stub: fuente (ℝ^{128})⁹ → certificado en Ω_3."""

    class TopologicalInvariantError(Exception):  # type: ignore[no-redef]
        pass

# ------------------------------------------------------------------------------
# Importación del motor routónico 128D y del álgebra de Cayley-Dickson.
# ------------------------------------------------------------------------------
try:
    from app.core.routon_dependency_engine import (
        RoutonDependencyEngine,
        RoutonState,
        RoutonEngineState,
        RoutonMetricsReport,
        RoutonEneagonalReport,
        CayleyDicksonAlgebra128,
        RoutonEngineError,
        RoutonDimensionError,
    )
except ImportError:
    from routon_dependency_engine import (
        RoutonDependencyEngine,
        RoutonState,
        RoutonEngineState,
        RoutonMetricsReport,
        RoutonEneagonalReport,
        CayleyDicksonAlgebra128,
        RoutonEngineError,
        RoutonDimensionError,
    )

try:
    from app.core.routon_dependency_engine import (
        KBNSummationKernel,
        RoutonThresholds,
        NullConeReport,
        RoutonNumericalSingularityError,
    )
except ImportError:
    try:
        from routon_dependency_engine import (
            KBNSummationKernel,
            RoutonThresholds,
            NullConeReport,
            RoutonNumericalSingularityError,
        )
    except ImportError:
        KBNSummationKernel = None  # type: ignore[assignment]
        RoutonThresholds = None  # type: ignore[assignment]
        NullConeReport = None  # type: ignore[assignment]
        RoutonNumericalSingularityError = RoutonEngineError  # type: ignore[assignment]


logger = logging.getLogger("APU.Agents.Wisdom.RoutonDependencyAgent")

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_FLOOR: Final[float] = 1e-15
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_ROUTON_DIM: Final[int] = 128
_BANACH_R128_UPPER_BOUND: Final[float] = float(math.sqrt(float(_ROUTON_DIM)))
_BANACH_L1_LINF_BOUND: Final[float] = float(_ROUTON_DIM)
_LOG_DBL_MAX: Final[float] = float(math.log(np.finfo(np.float64).max))
_LOG_DBL_TINY: Final[float] = float(math.log(np.finfo(np.float64).tiny))
_CROWBAR_GPIO: Final[str] = "GPIO14"
_CROWBAR_DEVICE: Final[str] = "BT151"

_DEFAULT_OVERRIDE_TOKENS: Final[Tuple[str, ...]] = (
    "AUT_POS_SABIDURIA_777",
    "OVERRIDE_NON_MOUFANG_ROUTON_2026",
    "HMAC_SUTURA_FOCK_SECURE_128D",
)


# ═══════════════════════════════════════════════════════════════════════════════
# §A. ÁLGEBRA DE HEYTING Ω_3 Y METROLOGÍA NUMÉRICA
# ═══════════════════════════════════════════════════════════════════════════════
class HeytingOmega3(IntEnum):
    r"""
    Cadena de Heyting de tres elementos (álgebra de Gödel).

    Orden: VETOED = ⊥ ≺ DEGRADED = ½ ≺ COHERENT = ⊤.
    Operaciones: a ∧ b = mín, a ∨ b = máx,
                 a → b = ⊤ si a ≼ b, en caso contrario b,
                 ¬a = a → ⊥.
    """

    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    @property
    def verdict(self) -> str:
        """Etiqueta estable de API pública."""
        return self.name

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """Producto (ínfimo) del retículo."""
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """Coproducto (supremo) del retículo."""
        return HeytingOmega3(max(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """Implicación de Heyting: c ≤ (a → b) ⇔ (c ∧ a) ≤ b."""
        if int(self) <= int(other):
            return HeytingOmega3.COHERENT
        return other

    def neg(self) -> "HeytingOmega3":
        """Negación intuicionista ¬a := a → ⊥."""
        return self.implies(HeytingOmega3.VETOED)


def _heyting_meet_all(values: Iterable[HeytingOmega3]) -> HeytingOmega3:
    """Meet de una familia; el neutro es ⊤."""
    acc = HeytingOmega3.COHERENT
    for value in values:
        acc = acc.meet(value)
    return acc


def _agent_log_norm(norm: float) -> float:
    """Logaritmo neperiano seguro de una norma no negativa (0 ↦ −∞)."""
    if norm <= 0.0:
        return -math.inf
    return math.log(norm)


def _agent_safe_exp(log_value: float) -> float:
    """Exponencial saturada en el rango de float64."""
    if log_value == -math.inf:
        return 0.0
    if log_value == math.inf:
        return math.inf
    if not math.isfinite(log_value):
        return math.inf if log_value > 0.0 else 0.0
    if log_value >= _LOG_DBL_MAX:
        return math.inf
    if log_value <= _LOG_DBL_TINY:
        return 0.0
    return float(math.exp(log_value))


def _canonical_float_bytes(value: float) -> bytes:
    """Serialización canónica little-endian IEEE-754; unifica ±0.0."""
    x = float(value)
    if math.isnan(x):
        return b"\x7fNAN\x00\x00\x00"
    if math.isinf(x):
        return b"\x7fPINF\x00\x00" if x > 0.0 else b"\x7fNINF\x00\x00"
    if x == 0.0:
        return struct.pack("<d", 0.0)
    return struct.pack("<d", x)


def _vector_norm_1(arr: np.ndarray) -> float:
    """Norma ℓ¹ con sumación KBN si el motor la expone."""
    abs_arr = np.abs(arr)
    if KBNSummationKernel is not None and hasattr(KBNSummationKernel, "sum"):
        return float(KBNSummationKernel.sum(abs_arr))
    return float(np.sum(abs_arr))


def _vector_norm_2(arr: np.ndarray) -> float:
    """Norma ℓ² con núcleo KBN escalado si está disponible."""
    if KBNSummationKernel is not None and hasattr(KBNSummationKernel, "norm"):
        return float(KBNSummationKernel.norm(arr))
    return float(la.norm(arr))


def _vector_norm_inf(arr: np.ndarray) -> float:
    """Norma ℓ^∞ = máx_i |x_i|."""
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))


def _attr_float(obj: Any, name: str, default: float) -> float:
    """Extracción defensiva de un escalar float desde un DTO del motor."""
    if obj is None:
        return default
    raw = getattr(obj, name, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value


def _attr_bool(obj: Any, name: str, default: bool) -> bool:
    """Extracción defensiva de un booleano desde un DTO del motor."""
    if obj is None:
        return default
    return bool(getattr(obj, name, default))


def _attr_str(obj: Any, name: str, default: str) -> str:
    """Extracción defensiva de una cadena desde un DTO del motor."""
    if obj is None:
        return default
    raw = getattr(obj, name, default)
    return default if raw is None else str(raw)


def _multiply_128(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Producto CD 128D con fallback al núcleo genérico."""
    if hasattr(CayleyDicksonAlgebra128, "multiply_128"):
        return CayleyDicksonAlgebra128.multiply_128(a, b)
    return CayleyDicksonAlgebra128.multiply(a, b)


# ═══════════════════════════════════════════════════════════════════════════════
# §B. DTOs INMUTABLES DEL AGENTE SOBERANO
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class RoutonAgentThresholds:
    r"""
    Umbrales de gobernanza (distintos de los umbrales FPU del motor).

    Las fracciones eneagonales parten la cota L_max en:
        banda elástica  (DEGRADED)  si  0.3 L_max < ‖A_9‖ ≤ 0.5 L_max,
        colapso duro    (VETOED)    si  ‖A_9‖ > 0.5 L_max  o no finito.

    A_Moufang no finito es veto duro; A_Moufang sobre cota es degradación.
    """

    eneagonal_soft_fraction: float = 0.3
    eneagonal_hard_fraction: float = 0.5
    condition_number_limit: float = 1e12
    banach_eps_factor: float = 10.0
    moufang_relative_limit: float = 1e-2
    quadratic_leakage_limit: float = 1e-8

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"El umbral {name} debe ser finito y estrictamente positivo.")
        if self.eneagonal_soft_fraction >= self.eneagonal_hard_fraction:
            raise ValueError("Se exige 0 < eneagonal_soft_fraction < eneagonal_hard_fraction.")
        if self.eneagonal_hard_fraction > 1.0:
            raise ValueError("eneagonal_hard_fraction no puede exceder 1.")


@dataclass(frozen=True, slots=True)
class BanachRegularityReport:
    r"""
    Certificado local de regularidad de Banach sobre ℝ^{128}:

        ‖x‖₂ ≤ ‖x‖₁ ≤ √128 ‖x‖₂,
        ‖x‖_∞ ≤ ‖x‖₂ ≤ √128 ‖x‖_∞,
        ‖x‖_∞ ≤ ‖x‖₁ ≤ 128 ‖x‖_∞.

    Para el vector nulo se define convencionalmente todo ratio = 1.0
    (el rayo más disperso, igualdad inferior de ℓ¹/ℓ²).
    `ratio` es el cociente ℓ¹/ℓ², preservado por compatibilidad 1.x.
    """

    norm_1: float
    norm_2: float
    ratio: float
    lower_bound: float
    upper_bound: float
    is_within_theoretical_bounds: bool
    norm_inf: float = 0.0
    ratio_2inf: float = 1.0
    ratio_1inf: float = 1.0


@dataclass(frozen=True, slots=True)
class RoutonObservationKernel:
    r"""
    Objeto terminal de la Fase 1 / objeto inicial de la Fase 2.

    Expediente inmutable de Observe: nueve actores saneados, estados FPU,
    certificados de Banach y sello de sesión canónico.
    """

    states_9way: Tuple[RoutonState, ...]
    sanitized_vectors: Tuple[np.ndarray, ...]
    banach_reports: Tuple[BanachRegularityReport, ...]
    banach_ratios_9way: Tuple[float, ...]
    session_sha256: str
    all_banach_regular: bool = True

    @property
    def states(self) -> Tuple[RoutonState, ...]:
        """Compatibilidad con la versión 1.x."""
        return self.states_9way

    @property
    def banach_ratios(self) -> Tuple[float, ...]:
        """Compatibilidad con la versión 1.x."""
        return self.banach_ratios_9way

    @property
    def cryptographic_seal(self) -> str:
        """Compatibilidad con la versión 1.x."""
        return self.session_sha256


@dataclass(frozen=True, slots=True)
class RoutonOrientationState:
    r"""
    Expediente inmutable de Orient: métricas algebraicas, espectrales
    y de cono nulo necesarias para el meet de Heyting.
    """

    kernel: RoutonObservationKernel

    trilateral_associator_norm: float
    moufang_distortion_norm: float
    eneagonal_associator_norm: float

    moufang_relative_norm: float
    eneagonal_relative_norm: float

    moufang_frustration_index: float
    eneagonal_frustration_index: float

    hurwitz_composition_error: float
    hurwitz_relative_error: float

    null_cone_friction: float
    null_cone_relative_defect: float
    null_cone_depth: float

    simplex_condition_number: float
    laplacian_connectivity: float

    is_moufang_stable: bool
    is_eneagonal_norm_stable: bool
    is_eneagonal_stable: bool
    is_hurwitz_stable: bool
    is_null_cone_stable: bool

    diagnosis: str

    moufang_threshold_used: float = 0.0
    eneagonal_threshold_used: float = 0.0
    left_moufang_defect: float = 0.0
    right_moufang_defect: float = 0.0
    middle_moufang_defect: float = 0.0
    alternative_strain_norm: float = 0.0
    right_alternative_strain_norm: float = 0.0
    flexibility_defect: float = 0.0
    power_associator_norm: float = 0.0
    stasheff_pentagon_diameter: float = 0.0
    simplex_volume: float = 0.0
    laplacian_spectral_gap: float = 0.0
    estimated_connected_components: int = 1
    sigma_min_left_r1: float = math.inf
    sigma_min_left_r2: float = math.inf
    commutator_norm: float = 0.0
    is_banach_submultiplicative: bool = True
    is_null_cone_penetrated: bool = False
    engine_cryptographic_seal: str = ""
    quadratic_leakage_max: float = 0.0


@dataclass(frozen=True, slots=True)
class HeytingDecision:
    r"""Decisión formal en Ω_3, con conjuncts locales y modalidad Γ."""

    verdict: str
    is_soft_veto: bool
    is_hard_veto: bool
    grace_expired: bool
    time_grace_remaining: float
    reasons: Tuple[str, ...]
    lattice_value: int = int(HeytingOmega3.COHERENT)
    conjuncts: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class RoutonGovernedDecision:
    r"""
    Objeto terminal de la Fase 2 / objeto inicial de la Fase 3.

    Empaqueta orientación, decisión de Heyting y sello del motor,
    de modo que Act no reinstancia Observe ni Orient.
    """

    orientation: RoutonOrientationState
    heyting: HeytingDecision


@dataclass(frozen=True, slots=True)
class CrowbarActuationReport:
    r"""Reporte de Act: actuación Crowbar BT151 (simulada) sobre GPIO14."""

    interlock_fired: bool
    actuation_latency_ns: float
    gpio: str
    device: str
    seed_sha256: str


@dataclass(frozen=True, slots=True)
class RoutonAgentCertificate:
    r"""Certificado formal de calibración, regularidad de calibre y veto ciber-físico."""

    phase: str
    heyting_verdict: str

    eneagonal_associator_norm: float
    moufang_distortion_norm: float
    is_eneagonal_stable: bool

    composition_error: float
    null_cone_friction: float
    is_null_cone_penetrated: bool

    is_soft_veto_active: bool
    override_grace_period_expired: bool
    hardware_interlock_fired: bool

    actuation_latency_ns: float
    time_grace_remaining: float
    digital_signature_sha256: str

    trilateral_associator_norm: float = 0.0
    moufang_relative_norm: float = 0.0
    eneagonal_relative_norm: float = 0.0
    moufang_frustration_index: float = 0.0
    eneagonal_frustration_index: float = 0.0

    moufang_threshold_used: float = 0.0
    eneagonal_threshold_used: float = 0.0

    is_moufang_stable: bool = False
    is_eneagonal_norm_stable: bool = False
    is_hard_veto_active: bool = False

    null_cone_relative_defect: float = 0.0
    null_cone_depth: float = 0.0

    simplex_condition_number: float = math.inf
    laplacian_connectivity: float = 0.0

    banach_ratios_9way: Tuple[float, ...] = ()
    reasons: Tuple[str, ...] = ()

    is_hurwitz_stable: bool = False
    is_null_cone_stable: bool = False
    left_moufang_defect: float = 0.0
    right_moufang_defect: float = 0.0
    middle_moufang_defect: float = 0.0
    stasheff_pentagon_diameter: float = 0.0
    sigma_min_left_r1: float = math.inf
    sigma_min_left_r2: float = math.inf
    engine_cryptographic_seal: str = ""
    heyting_conjuncts: Tuple[Tuple[str, str], ...] = ()
    agent_version: str = __version__


# ═══════════════════════════════════════════════════════════════════════════════
# §C. FASE 1 — OBSERVE
#     Morfismo terminal: observe_nine_way → RoutonObservationKernel
# ═══════════════════════════════════════════════════════════════════════════════
class Phase1_RoutonSovereignObserver(Morphism):
    r"""
    FASE 1 — Observe.

    Saneamiento de de Rham, regularidad de Banach ℓ¹/ℓ²/ℓ^∞ y construcción
    de estados routónicos FPU para los nueve actores del 8-símplex.

    Categoría fuente: \(\mathbf{Vec}_{128}^{9}\).
    Categoría meta  : \(\mathbf{ObservationKernel}\).

    El método terminal `observe_nine_way` es el objeto inicial de la Fase 2.
    """

    __slots__ = (
        "_tol",
        "_safety_margin",
        "_grace_limit",
        "_override_tokens",
        "_hmac_secret",
        "_agent_thresholds",
        "_is_soft_veto_active",
        "_soft_veto_timestamp",
        "_engine",
    )

    def __init__(
        self,
        tolerance: float = 1e-12,
        safety_margin: float = 1.0,
        grace_period_seconds: float = 3600.0,
        override_tokens: Optional[Sequence[str]] = None,
        engine: Optional[RoutonDependencyEngine] = None,
        hmac_secret: Optional[bytes] = None,
        agent_thresholds: Optional[RoutonAgentThresholds] = None,
    ) -> None:
        self._tol: Final[float] = self._validate_positive_float("tolerance", tolerance)
        self._safety_margin: Final[float] = self._validate_positive_float(
            "safety_margin", safety_margin
        )
        self._grace_limit: Final[float] = self._validate_positive_float(
            "grace_period_seconds", grace_period_seconds
        )
        self._override_tokens: Final[frozenset[str]] = frozenset(
            override_tokens if override_tokens is not None else _DEFAULT_OVERRIDE_TOKENS
        )
        self._agent_thresholds: Final[RoutonAgentThresholds] = (
            agent_thresholds or RoutonAgentThresholds()
        )

        secret = hmac_secret
        if secret is None:
            env = os.environ.get("ROUTON_HMAC_SECRET")
            secret = env.encode("utf-8") if env else None
        self._hmac_secret: Final[Optional[bytes]] = secret

        self._is_soft_veto_active: bool = False
        self._soft_veto_timestamp: Optional[float] = None

        self._engine: Final[RoutonDependencyEngine] = (
            engine if engine is not None else self._build_engine()
        )

    @staticmethod
    def _validate_positive_float(name: str, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} debe ser finito y estrictamente positivo.")
        return value

    @staticmethod
    def _validate_non_negative_float(name: str, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} debe ser finito y no negativo.")
        return value

    def _build_engine(self) -> RoutonDependencyEngine:
        """Construye el motor 128D, preferiendo umbrales evolucionados."""
        if RoutonThresholds is not None:
            base_kwargs = dict(
                absolute_tolerance=self._tol,
                relative_tolerance=max(self._tol * 10.0, 1e-10),
                zero_norm_threshold=max(self._tol, 1e-14),
                hurwitz_absolute=1e-9,
                hurwitz_relative=1e-8,
                moufang_absolute=500.0,
                moufang_relative=1e-2,
                eneagonal_absolute=15.0,
                eneagonal_relative=1e-2,
                null_absolute=1e-12,
                null_relative=1e-8,
                null_depth_threshold=0.99,
                condition_number_limit=self._agent_thresholds.condition_number_limit,
            )
            try:
                thresholds = RoutonThresholds(
                    **base_kwargs,
                    quadratic_leakage_limit=self._agent_thresholds.quadratic_leakage_limit,
                )
                return RoutonDependencyEngine(thresholds=thresholds)
            except TypeError:
                try:
                    return RoutonDependencyEngine(
                        thresholds=RoutonThresholds(**base_kwargs)
                    )
                except TypeError:
                    pass
            except Exception as exc:
                logger.debug("No se pudo construir motor con RoutonThresholds: %s", exc)

        try:
            return RoutonDependencyEngine(tolerance=self._tol)
        except TypeError:
            return RoutonDependencyEngine()

    def reset_grace_window(self) -> None:
        """Reinicia la modalidad de gracia Γ (único estado mutable del soberano)."""
        self._is_soft_veto_active = False
        self._soft_veto_timestamp = None

    @staticmethod
    def _sanitize_vector(value: np.ndarray, name: str) -> np.ndarray:
        """
        Sanea un vector 128D:

        - fuerza dtype float64 y forma estricta (128,);
        - exige componentes finitas;
        - colapsa ceros signados (−0.0 → +0.0);
        - congela la copia en RAM (write=False).
        """
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 1 or arr.shape != (_ROUTON_DIM,):
            raise RoutonDimensionError(
                f"{name} debe ser estrictamente de dimensión {_ROUTON_DIM}. "
                f"Obtenido: shape={arr.shape}."
            )

        arr = np.ascontiguousarray(arr, dtype=np.float64).copy()
        if not np.all(np.isfinite(arr)):
            raise RoutonNumericalSingularityError(
                f"{name} contiene NaN o Inf. El soberano no puede auditar "
                "singularidades no finitas."
            )

        arr = np.where(arr == 0.0, 0.0, arr)
        arr.setflags(write=False)
        return arr

    def _banach_report_from_sanitized(self, arr: np.ndarray) -> BanachRegularityReport:
        r"""Certifica las tres equivalencias de normas sobre un vector ya saneado."""
        norm1 = _vector_norm_1(arr)
        norm2 = _vector_norm_2(arr)
        norm_inf = _vector_norm_inf(arr)

        if not (math.isfinite(norm1) and math.isfinite(norm2) and math.isfinite(norm_inf)):
            raise RoutonNumericalSingularityError(
                "La regularidad de Banach produjo normas no finitas."
            )

        zero_threshold = max(self._tol, _WILKINSON_FLOOR)
        slack = self._agent_thresholds.banach_eps_factor * _MACHINE_EPS

        if norm2 <= zero_threshold:
            ratio12, ratio2inf, ratio1inf = 1.0, 1.0, 1.0
            within = True
        else:
            denom_inf = max(norm_inf, zero_threshold)
            ratio12 = norm1 / norm2
            ratio2inf = norm2 / denom_inf
            ratio1inf = norm1 / denom_inf
            within = (
                ratio12 + slack >= 1.0
                and ratio12 <= _BANACH_R128_UPPER_BOUND * (1.0 + slack)
                and ratio2inf + slack >= 1.0
                and ratio2inf <= _BANACH_R128_UPPER_BOUND * (1.0 + slack)
                and ratio1inf + slack >= 1.0
                and ratio1inf <= _BANACH_L1_LINF_BOUND * (1.0 + slack)
            )

        return BanachRegularityReport(
            norm_1=norm1,
            norm_2=norm2,
            ratio=ratio12,
            lower_bound=1.0,
            upper_bound=_BANACH_R128_UPPER_BOUND,
            is_within_theoretical_bounds=within,
            norm_inf=norm_inf,
            ratio_2inf=ratio2inf,
            ratio_1inf=ratio1inf,
        )

    def evaluate_banach_regularity_report(self, S: np.ndarray) -> BanachRegularityReport:
        """API pública granular: certificado de Banach de un vector 128D."""
        return self._banach_report_from_sanitized(self._sanitize_vector(S, "S"))

    def evaluate_banach_regularity(self, S: np.ndarray) -> float:
        """API pública 1.x: retorna únicamente el ratio ℓ¹/ℓ²."""
        return self.evaluate_banach_regularity_report(S).ratio

    def observe_nine_way(
        self,
        actors_raw: Sequence[np.ndarray],
    ) -> RoutonObservationKernel:
        r"""
        Morfismo terminal de la Fase 1.

        Sanea, certifica Banach y construye estados FPU para exactamente
        nueve actores. Firma functorial:

            \mathrm{observe\_nine\_way}:
                (\mathbb{R}^{128})^{9}\longrightarrow\mathbf{RoutonObservationKernel}.

        El valor de retorno es el objeto inicial de
        `Phase2_RoutonHeytingGovernor.continue_from_observation_kernel`.
        """
        if len(actors_raw) != 9:
            raise RoutonEngineError(
                "La auditoría routónica requiere exactamente nueve actores."
            )

        sanitized_vectors = []
        banach_reports = []
        banach_ratios = []
        states_128d = []

        hasher = hashlib.sha256()
        hasher.update(b"ROUTON_OBSERVATION_PHASE_1_V3")
        hasher.update(__version__.encode("utf-8"))

        all_regular = True
        for idx, vec in enumerate(actors_raw, start=1):
            name = f"actor_{idx}"
            clean_vec = self._sanitize_vector(np.asarray(vec), name)
            banach_report = self._banach_report_from_sanitized(clean_vec)
            state = self._engine.build_state(clean_vec)

            sanitized_vectors.append(clean_vec)
            banach_reports.append(banach_report)
            banach_ratios.append(banach_report.ratio)
            states_128d.append(state)
            all_regular = all_regular and banach_report.is_within_theoretical_bounds

            hasher.update(np.asarray(clean_vec, dtype="<f8").tobytes(order="C"))
            hasher.update(_canonical_float_bytes(banach_report.ratio))
            hasher.update(_canonical_float_bytes(banach_report.ratio_2inf))
            hasher.update(_canonical_float_bytes(banach_report.ratio_1inf))

            state_hash = getattr(state, "sha256_hash", "")
            if state_hash:
                hasher.update(str(state_hash).encode("ascii"))

        kernel = RoutonObservationKernel(
            states_9way=tuple(states_128d),
            sanitized_vectors=tuple(sanitized_vectors),
            banach_reports=tuple(banach_reports),
            banach_ratios_9way=tuple(banach_ratios),
            session_sha256=hasher.hexdigest(),
            all_banach_regular=all_regular,
        )

        logger.info(
            "Fase Observe [ROUTON_AGENT_128D]: 9 actores empaquetados en Banach 128D. Sello: %s",
            kernel.session_sha256[:16],
        )
        return kernel

    def _observe(self, actors_raw: Sequence[np.ndarray]) -> RoutonObservationKernel:
        """Alias interno de compatibilidad 2.x → morfismo terminal de la Fase 1."""
        return self.observe_nine_way(actors_raw)


# ═══════════════════════════════════════════════════════════════════════════════
# §D. FASE 2 — ORIENT + DECIDE
#     Objeto inicial = morfismo terminal de la Fase 1 (RoutonObservationKernel)
#     Morfismo terminal: govern_from_observation_kernel → RoutonGovernedDecision
# ═══════════════════════════════════════════════════════════════════════════════
class Phase2_RoutonHeytingGovernor(Phase1_RoutonSovereignObserver):
    r"""
    FASE 2 — Orient + Decide.

    Consume `RoutonObservationKernel` (terminal de Fase 1), orquesta el
    motor 128D (Hurwitz, Moufang L/M/R, A_9, Stasheff, cono nulo) y
    clasifica el 8-símplex en Ω_3 mediante el meet de predicados locales
    con modalidad de gracia Γ.

    El método inicial `continue_from_observation_kernel` es la continuación
    formal de `Phase1.observe_nine_way`.
    El método terminal `govern_from_observation_kernel` produce el objeto
    inicial de la Fase 3 (`RoutonGovernedDecision`).
    """

    __slots__ = ()

    def continue_from_observation_kernel(
        self,
        kernel: RoutonObservationKernel,
        eneagonal_threshold_Lmax: float,
        moufang_threshold_Lmax: float,
        null_critical_threshold: float,
        cota_enea_limite: float,
        cota_moufang_limite: float,
    ) -> RoutonOrientationState:
        r"""
        Morfismo de entrada de la Fase 2 (continuación formal de
        `Phase1_RoutonSovereignObserver.observe_nine_way`).

            \mathrm{continue\_from\_observation\_kernel}:
                \mathbf{RoutonObservationKernel}\times\mathbb{R}_{+}^{5}
                \longrightarrow\mathbf{RoutonOrientationState}.
        """
        return self._orient(
            kernel=kernel,
            eneagonal_threshold_Lmax=eneagonal_threshold_Lmax,
            moufang_threshold_Lmax=moufang_threshold_Lmax,
            null_critical_threshold=null_critical_threshold,
            cota_enea_limite=cota_enea_limite,
            cota_moufang_limite=cota_moufang_limite,
        )

    def _evaluate_null_metrics(
        self,
        r1_state: RoutonState,
        r2_state: RoutonState,
        null_critical_threshold: float,
        engine_null: Any = None,
    ) -> Tuple[float, float, float, bool, bool, float, float, float]:
        """
        Fricción, profundidad y espectro de 𝒩(ℝou) para el par (R₁, R₂).

        Devuelve:
            friction, relative_defect, null_depth,
            is_null_penetrated, is_null_stable,
            σ_min(L_{R₁}), σ_min(L_{R₂}), ‖[R₁,R₂]‖.
        """
        if engine_null is not None:
            friction = _attr_float(engine_null, "absolute_friction", math.inf)
            relative_defect = _attr_float(engine_null, "relative_defect", math.inf)
            null_depth = _attr_float(engine_null, "null_depth", 0.0)
            penetrated = _attr_bool(engine_null, "is_null_cone_penetrated", False)
            sigma1 = _attr_float(
                engine_null,
                "sigma_min_left_r1",
                _attr_float(engine_null, "sigma_min_left_x1", math.inf),
            )
            sigma2 = _attr_float(
                engine_null,
                "sigma_min_left_r2",
                _attr_float(engine_null, "sigma_min_left_x2", math.inf),
            )
            commutator = _attr_float(engine_null, "commutator_norm", 0.0)
            is_stable = (
                not penetrated
                and math.isfinite(friction)
                and friction <= (null_critical_threshold + self._tol)
            )
            return (
                friction,
                relative_defect,
                null_depth,
                penetrated,
                is_stable,
                sigma1,
                sigma2,
                commutator,
            )

        if hasattr(self._engine, "evaluate_null_cone"):
            report = self._engine.evaluate_null_cone(r1_state, r2_state)
            return self._evaluate_null_metrics(
                r1_state, r2_state, null_critical_threshold, engine_null=report
            )

        prod_vec = _multiply_128(r1_state.vector_rep, r2_state.vector_rep)
        prod_norm = _vector_norm_2(prod_vec)
        log_expected = _agent_log_norm(r1_state.norm) + _agent_log_norm(r2_state.norm)
        expected_norm = _agent_safe_exp(log_expected)

        if math.isfinite(prod_norm) and math.isfinite(expected_norm):
            friction = abs(prod_norm - expected_norm)
        else:
            friction = math.inf

        scale = (
            max(_WILKINSON_FLOOR, expected_norm)
            if math.isfinite(expected_norm)
            else _WILKINSON_FLOOR
        )
        relative_defect = friction / scale if math.isfinite(friction) else math.inf
        if (
            math.isfinite(expected_norm)
            and math.isfinite(prod_norm)
            and expected_norm > _WILKINSON_FLOOR
        ):
            null_depth = max(0.0, 1.0 - (prod_norm / expected_norm))
        else:
            null_depth = 0.0

        penetrated = math.isfinite(friction) and friction > null_critical_threshold
        is_stable = (
            not penetrated
            and math.isfinite(friction)
            and friction <= (null_critical_threshold + self._tol)
        )
        return friction, relative_defect, null_depth, penetrated, is_stable, math.inf, math.inf, 0.0

    def _orient(
        self,
        kernel: RoutonObservationKernel,
        eneagonal_threshold_Lmax: float,
        moufang_threshold_Lmax: float,
        null_critical_threshold: float,
        cota_enea_limite: float,
        cota_moufang_limite: float,
    ) -> RoutonOrientationState:
        """Orient: una pasada del motor 128D + diagnóstico compuesto."""
        states = kernel.states_9way
        sanitized = kernel.sanitized_vectors

        engine_state: Any = None
        metrics_report: Any = None
        ene_report: Any = None
        null_report: Any = None

        if hasattr(self._engine, "execute_eneagonal_audit"):
            try:
                engine_state = self._engine.execute_eneagonal_audit(
                    sanitized[0],
                    sanitized[1],
                    sanitized[2],
                    sanitized[3],
                    sanitized[4],
                    sanitized[5],
                    sanitized[6],
                    sanitized[7],
                    sanitized[8],
                    eneagonal_threshold=cota_enea_limite,
                    moufang_threshold=cota_moufang_limite,
                )
            except TypeError:
                engine_state = self._engine.execute_eneagonal_audit(
                    *sanitized,
                    eneagonal_threshold=cota_enea_limite,
                )
            metrics_report = getattr(engine_state, "metrics_report", None)
            ene_report = getattr(engine_state, "eneagonal_report", None)
            null_report = getattr(engine_state, "null_report", None)
        else:
            if hasattr(self._engine, "observe_metrics"):
                metrics_report = self._engine.observe_metrics(sanitized[0], sanitized[1])
            if hasattr(self._engine, "calculate_eneagonal_frustration"):
                try:
                    ene_report = self._engine.calculate_eneagonal_frustration(
                        *sanitized,
                        eneagonal_threshold=cota_enea_limite,
                        moufang_threshold=cota_moufang_limite,
                    )
                except TypeError:
                    try:
                        ene_report = self._engine.calculate_eneagonal_frustration(
                            *sanitized,
                            eneagonal_threshold=cota_enea_limite,
                        )
                    except TypeError:
                        ene_report = self._engine.calculate_eneagonal_frustration(*sanitized)
            else:
                raise RoutonEngineError(
                    "El motor routónico no expone calculate_eneagonal_frustration "
                    "ni execute_eneagonal_audit."
                )

        if metrics_report is not None:
            hurwitz_abs = _attr_float(
                metrics_report,
                "hurwitz_absolute_error",
                _attr_float(metrics_report, "hurwitz_composition_error", math.inf),
            )
            hurwitz_rel = _attr_float(metrics_report, "hurwitz_relative_error", math.inf)
            is_hurwitz_stable = _attr_bool(
                metrics_report,
                "is_hurwitz_stable",
                math.isfinite(hurwitz_abs) and hurwitz_abs < 1e-9,
            )
            is_banach_sub = _attr_bool(
                metrics_report, "is_banach_submultiplicative", is_hurwitz_stable
            )
        else:
            prod_vec = _multiply_128(states[0].vector_rep, states[1].vector_rep)
            prod_norm = _vector_norm_2(prod_vec)
            expected_norm = _agent_safe_exp(
                _agent_log_norm(states[0].norm) + _agent_log_norm(states[1].norm)
            )
            hurwitz_abs = (
                abs(prod_norm - expected_norm)
                if math.isfinite(prod_norm) and math.isfinite(expected_norm)
                else math.inf
            )
            scale = (
                max(_WILKINSON_FLOOR, expected_norm)
                if math.isfinite(expected_norm)
                else _WILKINSON_FLOOR
            )
            hurwitz_rel = hurwitz_abs / scale if math.isfinite(hurwitz_abs) else math.inf
            is_hurwitz_stable = math.isfinite(hurwitz_abs) and hurwitz_abs < 1e-9
            is_banach_sub = is_hurwitz_stable

        trilateral_norm = _attr_float(ene_report, "trilateral_associator_norm", 0.0)
        moufang_norm = _attr_float(ene_report, "moufang_distortion_norm", math.inf)
        enea_norm = _attr_float(ene_report, "eneagonal_associator_norm", math.inf)
        if math.isnan(moufang_norm):
            moufang_norm = math.inf
        if math.isnan(enea_norm):
            enea_norm = math.inf

        log_den_moufang = (
            2.0 * _agent_log_norm(max(float(states[0].norm), _WILKINSON_FLOOR))
            + _agent_log_norm(max(float(states[1].norm), _WILKINSON_FLOOR))
            + _agent_log_norm(max(float(states[2].norm), _WILKINSON_FLOOR))
        )
        log_den_enea = sum(
            _agent_log_norm(max(float(state.norm), _WILKINSON_FLOOR)) for state in states
        )
        default_moufang_rel = _agent_safe_exp(_agent_log_norm(moufang_norm) - log_den_moufang)
        default_enea_rel = _agent_safe_exp(_agent_log_norm(enea_norm) - log_den_enea)
        default_moufang_frustration = _agent_safe_exp(
            _agent_log_norm(moufang_norm) - max(0.0, log_den_moufang)
        )
        default_enea_frustration = _agent_safe_exp(
            _agent_log_norm(enea_norm) - max(0.0, log_den_enea)
        )

        moufang_rel = _attr_float(ene_report, "moufang_relative_norm", default_moufang_rel)
        enea_rel = _attr_float(ene_report, "eneagonal_relative_norm", default_enea_rel)
        moufang_frustration = _attr_float(
            ene_report, "moufang_frustration_index", default_moufang_frustration
        )
        enea_frustration = _attr_float(
            ene_report, "frustration_index", default_enea_frustration
        )
        moufang_threshold_used = _attr_float(
            ene_report, "moufang_threshold_used", cota_moufang_limite
        )
        eneagonal_threshold_used = _attr_float(
            ene_report, "eneagonal_threshold_used", cota_enea_limite
        )

        is_moufang_stable = _attr_bool(
            ene_report,
            "is_moufang_stable",
            math.isfinite(moufang_norm) and moufang_norm <= (cota_moufang_limite + self._tol),
        )
        is_enea_norm_stable = _attr_bool(
            ene_report,
            "is_eneagonal_norm_stable",
            math.isfinite(enea_norm) and enea_norm <= (cota_enea_limite + self._tol),
        )
        is_enea_stable = _attr_bool(
            ene_report,
            "is_eneagonal_stable",
            is_moufang_stable and is_enea_norm_stable,
        )

        condition_number = _attr_float(ene_report, "simplex_condition_number", math.inf)
        connectivity = _attr_float(ene_report, "laplacian_connectivity", 0.0)
        base_diagnosis = _attr_str(ene_report, "diagnosis", "")
        left_mf = _attr_float(ene_report, "left_moufang_defect", 0.0)
        right_mf = _attr_float(ene_report, "right_moufang_defect", 0.0)
        middle_mf = _attr_float(ene_report, "middle_moufang_defect", 0.0)
        alt_norm = _attr_float(ene_report, "alternative_strain_norm", 0.0)
        right_alt = _attr_float(ene_report, "right_alternative_strain_norm", 0.0)
        flexibility = _attr_float(ene_report, "flexibility_defect", 0.0)
        power_assoc = _attr_float(ene_report, "power_associator_norm", 0.0)
        stasheff = _attr_float(ene_report, "stasheff_pentagon_diameter", 0.0)
        simplex_volume = _attr_float(ene_report, "simplex_volume", 0.0)
        spectral_gap = _attr_float(ene_report, "laplacian_spectral_gap", 0.0)
        components = int(
            round(_attr_float(ene_report, "estimated_connected_components", 1.0))
        )

        (
            null_friction,
            null_relative,
            null_depth,
            is_null_penetrated,
            is_null_stable,
            sigma1,
            sigma2,
            commutator,
        ) = self._evaluate_null_metrics(
            states[0],
            states[1],
            null_critical_threshold,
            engine_null=null_report,
        )

        leakages = [
            float(getattr(state, "quadratic_leakage", 0.0))
            for state in states
            if hasattr(state, "quadratic_leakage")
        ]
        quadratic_leakage_max = max(leakages) if leakages else 0.0
        engine_seal = _attr_str(engine_state, "cryptographic_seal", "")

        extra_diagnosis = []
        if not math.isfinite(trilateral_norm):
            extra_diagnosis.append("Asociador trilateral no finito.")
        if not math.isfinite(moufang_norm):
            extra_diagnosis.append("A_Moufang no finito.")
        if not math.isfinite(enea_norm):
            extra_diagnosis.append("A_9 no finito.")
        if not is_moufang_stable:
            extra_diagnosis.append("Distorsión de Moufang fuera de cota.")
        if not is_enea_norm_stable:
            extra_diagnosis.append("A_9 fuera de cota eneagonal.")
        if not is_hurwitz_stable:
            extra_diagnosis.append("Deriva de Hurwitz.")
        if not is_null_stable:
            extra_diagnosis.append("Inestabilidad en el cono nulo.")
        if condition_number > self._agent_thresholds.condition_number_limit:
            extra_diagnosis.append("8-símplex con condicionamiento afín degenerado.")
        if connectivity <= _MACHINE_EPS:
            extra_diagnosis.append("Conectividad algebraica casi nula.")
        if not kernel.all_banach_regular:
            extra_diagnosis.append("Violación de equivalencia de normas de Banach.")

        diagnosis_parts = [part for part in (base_diagnosis, *extra_diagnosis) if part]
        diagnosis = " | ".join(diagnosis_parts) if diagnosis_parts else "Orientación estable."

        return RoutonOrientationState(
            kernel=kernel,
            trilateral_associator_norm=trilateral_norm,
            moufang_distortion_norm=moufang_norm,
            eneagonal_associator_norm=enea_norm,
            moufang_relative_norm=moufang_rel,
            eneagonal_relative_norm=enea_rel,
            moufang_frustration_index=moufang_frustration,
            eneagonal_frustration_index=enea_frustration,
            hurwitz_composition_error=hurwitz_abs,
            hurwitz_relative_error=hurwitz_rel,
            null_cone_friction=null_friction,
            null_cone_relative_defect=null_relative,
            null_cone_depth=null_depth,
            simplex_condition_number=condition_number,
            laplacian_connectivity=connectivity,
            is_moufang_stable=is_moufang_stable,
            is_eneagonal_norm_stable=is_enea_norm_stable,
            is_eneagonal_stable=is_enea_stable,
            is_hurwitz_stable=is_hurwitz_stable,
            is_null_cone_stable=is_null_stable,
            diagnosis=diagnosis,
            moufang_threshold_used=moufang_threshold_used,
            eneagonal_threshold_used=eneagonal_threshold_used,
            left_moufang_defect=left_mf,
            right_moufang_defect=right_mf,
            middle_moufang_defect=middle_mf,
            alternative_strain_norm=alt_norm,
            right_alternative_strain_norm=right_alt,
            flexibility_defect=flexibility,
            power_associator_norm=power_assoc,
            stasheff_pentagon_diameter=stasheff,
            simplex_volume=simplex_volume,
            laplacian_spectral_gap=spectral_gap,
            estimated_connected_components=components,
            sigma_min_left_r1=sigma1,
            sigma_min_left_r2=sigma2,
            commutator_norm=commutator,
            is_banach_submultiplicative=is_banach_sub,
            is_null_cone_penetrated=is_null_penetrated,
            engine_cryptographic_seal=engine_seal,
            quadratic_leakage_max=quadratic_leakage_max,
        )

    def _local_heyting_predicates(
        self,
        orientation: RoutonOrientationState,
        cota_moufang_limite: float,
        cota_enea_limite: float,
    ) -> Tuple[Tuple[str, HeytingOmega3, str], ...]:
        """Predicados locales en Ω_3 cuyo meet es el veredicto global."""
        thr = self._agent_thresholds
        enea = orientation.eneagonal_associator_norm
        moufang = orientation.moufang_distortion_norm

        if not math.isfinite(enea):
            enea_h = HeytingOmega3.VETOED
            enea_reason = "A_9 no finito: singularidad eneagonal."
        elif enea > (thr.eneagonal_hard_fraction * cota_enea_limite + self._tol):
            enea_h = HeytingOmega3.VETOED
            enea_reason = "Colapso eneagonal duro."
        elif enea > (thr.eneagonal_soft_fraction * cota_enea_limite + self._tol):
            enea_h = HeytingOmega3.DEGRADED
            enea_reason = "Frustración eneagonal en banda elástica."
        else:
            enea_h = HeytingOmega3.COHERENT
            enea_reason = ""

        if not math.isfinite(moufang):
            moufang_h = HeytingOmega3.VETOED
            moufang_reason = "A_Moufang no finito: singularidad de Moufang."
        elif moufang > (cota_moufang_limite + self._tol):
            moufang_h = HeytingOmega3.DEGRADED
            moufang_reason = "Distorsión de Moufang fuera de cota."
        else:
            moufang_h = HeytingOmega3.COHERENT
            moufang_reason = ""

        hurwitz_h = (
            HeytingOmega3.COHERENT
            if orientation.is_hurwitz_stable
            else HeytingOmega3.VETOED
        )
        null_h = (
            HeytingOmega3.COHERENT
            if orientation.is_null_cone_stable
            else HeytingOmega3.VETOED
        )
        banach_h = (
            HeytingOmega3.COHERENT
            if orientation.kernel.all_banach_regular
            else HeytingOmega3.DEGRADED
        )
        cond_h = (
            HeytingOmega3.DEGRADED
            if orientation.simplex_condition_number > thr.condition_number_limit
            else HeytingOmega3.COHERENT
        )
        conn_h = (
            HeytingOmega3.DEGRADED
            if orientation.laplacian_connectivity <= _MACHINE_EPS
            else HeytingOmega3.COHERENT
        )

        return (
            ("eneagonal", enea_h, enea_reason),
            ("moufang", moufang_h, moufang_reason),
            (
                "hurwitz",
                hurwitz_h,
                ""
                if hurwitz_h is HeytingOmega3.COHERENT
                else "Deriva de Hurwitz fuera del límite de Wilkinson.",
            ),
            (
                "null_cone",
                null_h,
                ""
                if null_h is HeytingOmega3.COHERENT
                else "Inminencia de divisor de cero en N(Rou).",
            ),
            (
                "banach",
                banach_h,
                ""
                if banach_h is HeytingOmega3.COHERENT
                else "Equivalencia de normas de Banach violada.",
            ),
            (
                "simplex_condition",
                cond_h,
                ""
                if cond_h is HeytingOmega3.COHERENT
                else "8-símplex con condicionamiento afín degenerado.",
            ),
            (
                "connectivity",
                conn_h,
                ""
                if conn_h is HeytingOmega3.COHERENT
                else "Conectividad algebraica casi nula.",
            ),
        )

    def _clear_soft_veto(self) -> None:
        self._is_soft_veto_active = False
        self._soft_veto_timestamp = None

    def _verify_hmac_override(self, token: str, session_sha256: str = "") -> bool:
        """
        Valida override:

        1. Comparación en tiempo constante contra la allowlist estática (API 2.x).
        2. Si hay `hmac_secret`, HMAC-SHA256(secret, session_sha256) ligado a la sesión.
        """
        if not isinstance(token, str) or not token:
            return False

        token_bytes = token.encode("utf-8")
        for allowed_token in self._override_tokens:
            if hmac.compare_digest(token_bytes, allowed_token.encode("utf-8")):
                return True

        if self._hmac_secret and session_sha256:
            expected = hmac.new(
                self._hmac_secret,
                session_sha256.encode("ascii"),
                hashlib.sha256,
            ).hexdigest()
            if hmac.compare_digest(token, expected):
                return True
        return False

    def decide_from_orientation(
        self,
        orientation: RoutonOrientationState,
        cota_moufang_limite: float,
        cota_enea_limite: float,
        override_token: Optional[str],
        simulate_grace_expired: bool,
        curr_time: float,
    ) -> HeytingDecision:
        r"""
        Clasifica la orientación en Ω_3.

        Meet de predicados locales, luego modalidad Γ sobre DEGRADED y
        aniquilación σ de la gracia (sin promoción a ⊤). El veto duro
        no admite override.
        """
        predicates = self._local_heyting_predicates(
            orientation, cota_moufang_limite, cota_enea_limite
        )
        lattice = _heyting_meet_all(item[1] for item in predicates)
        conjuncts = tuple((name, value.verdict) for name, value, _ in predicates)
        reasons = tuple(reason for _, value, reason in predicates if reason)

        if lattice is HeytingOmega3.VETOED:
            self._clear_soft_veto()
            return HeytingDecision(
                verdict=lattice.verdict,
                is_soft_veto=False,
                is_hard_veto=True,
                grace_expired=False,
                time_grace_remaining=0.0,
                reasons=reasons,
                lattice_value=int(lattice),
                conjuncts=conjuncts,
            )

        if lattice is HeytingOmega3.DEGRADED:
            reason_list = list(reasons)
            session = orientation.kernel.session_sha256

            if override_token is not None:
                if self._verify_hmac_override(override_token, session_sha256=session):
                    self._clear_soft_veto()
                    reason_list.append(
                        "Override HMAC validado: σ(DEGRADED)=DEGRADED, Γ desactivada."
                    )
                    return HeytingDecision(
                        verdict=lattice.verdict,
                        is_soft_veto=False,
                        is_hard_veto=False,
                        grace_expired=False,
                        time_grace_remaining=0.0,
                        reasons=tuple(reason_list),
                        lattice_value=int(lattice),
                        conjuncts=conjuncts,
                    )
                reason_list.append("Override inválido.")

            if not self._is_soft_veto_active and not simulate_grace_expired:
                self._is_soft_veto_active = True
                self._soft_veto_timestamp = curr_time
                reason_list.append("Veto suave activado: luz ámbar y ventana de gracia Γ.")
                return HeytingDecision(
                    verdict=lattice.verdict,
                    is_soft_veto=True,
                    is_hard_veto=False,
                    grace_expired=False,
                    time_grace_remaining=self._grace_limit,
                    reasons=tuple(reason_list),
                    lattice_value=int(lattice),
                    conjuncts=conjuncts,
                )

            if self._soft_veto_timestamp is None:
                elapsed = self._grace_limit + 1.0
            else:
                elapsed = curr_time - self._soft_veto_timestamp
            time_remaining = max(0.0, self._grace_limit - elapsed)

            if time_remaining <= self._tol or simulate_grace_expired:
                self._clear_soft_veto()
                reason_list.append("Ventana de gracia Γ expirada sin override válido.")
                return HeytingDecision(
                    verdict=HeytingOmega3.VETOED.verdict,
                    is_soft_veto=False,
                    is_hard_veto=True,
                    grace_expired=True,
                    time_grace_remaining=0.0,
                    reasons=tuple(reason_list),
                    lattice_value=int(HeytingOmega3.VETOED),
                    conjuncts=conjuncts,
                )

            reason_list.append("Ventana de gracia Γ activa.")
            return HeytingDecision(
                verdict=lattice.verdict,
                is_soft_veto=True,
                is_hard_veto=False,
                grace_expired=False,
                time_grace_remaining=time_remaining,
                reasons=tuple(reason_list),
                lattice_value=int(lattice),
                conjuncts=conjuncts,
            )

        self._clear_soft_veto()
        return HeytingDecision(
            verdict=lattice.verdict,
            is_soft_veto=False,
            is_hard_veto=False,
            grace_expired=False,
            time_grace_remaining=0.0,
            reasons=reasons,
            lattice_value=int(lattice),
            conjuncts=conjuncts,
        )

    def govern_from_observation_kernel(
        self,
        kernel: RoutonObservationKernel,
        eneagonal_threshold_Lmax: float,
        moufang_threshold_Lmax: float,
        null_critical_threshold: float,
        cota_enea_limite: float,
        cota_moufang_limite: float,
        override_token: Optional[str],
        simulate_grace_expired: bool,
        curr_time: float,
    ) -> RoutonGovernedDecision:
        r"""
        Morfismo terminal de la Fase 2.

        Compone Orient ∘ Decide sobre el kernel de la Fase 1 y entrega el
        `RoutonGovernedDecision`, objeto inicial de
        `Phase3_RoutonCrowbarActuator.continue_from_governed_decision`.
        """
        orientation = self.continue_from_observation_kernel(
            kernel=kernel,
            eneagonal_threshold_Lmax=eneagonal_threshold_Lmax,
            moufang_threshold_Lmax=moufang_threshold_Lmax,
            null_critical_threshold=null_critical_threshold,
            cota_enea_limite=cota_enea_limite,
            cota_moufang_limite=cota_moufang_limite,
        )
        heyting = self.decide_from_orientation(
            orientation=orientation,
            cota_moufang_limite=cota_moufang_limite,
            cota_enea_limite=cota_enea_limite,
            override_token=override_token,
            simulate_grace_expired=simulate_grace_expired,
            curr_time=curr_time,
        )
        return RoutonGovernedDecision(orientation=orientation, heyting=heyting)


# ═══════════════════════════════════════════════════════════════════════════════
# §E. FASE 3 — ACT
#     Objeto inicial = morfismo terminal de la Fase 2 (RoutonGovernedDecision)
#     Morfismo terminal: audit_eneagonal_cycle → RoutonAgentCertificate
# ═══════════════════════════════════════════════════════════════════════════════
class Phase3_RoutonCrowbarActuator(Phase2_RoutonHeytingGovernor):
    r"""
    FASE 3 — Act.

    Actuación Crowbar BT151 *simulada* (especificación IRAM: latencia
    estrictamente menor a 400 ns sobre GPIO14) y emisión del certificado
    terminal, encadenando el sello criptográfico del motor 128D.

    El método inicial `continue_from_governed_decision` es la continuación
    formal de `Phase2.govern_from_observation_kernel`.
    El método terminal `audit_eneagonal_cycle` cierra el ciclo OODA.
    """

    __slots__ = ()

    def continue_from_governed_decision(
        self,
        governed: RoutonGovernedDecision,
    ) -> RoutonAgentCertificate:
        r"""
        Morfismo de entrada de la Fase 3 (continuación formal de
        `Phase2_RoutonHeytingGovernor.govern_from_observation_kernel`).

            \mathrm{continue\_from\_governed\_decision}:
                \mathbf{RoutonGovernedDecision}
                \longrightarrow\mathbf{RoutonAgentCertificate}.
        """
        orientation = governed.orientation
        decision = governed.heyting
        kernel = orientation.kernel

        actuation = self._act_crowbar(
            decision=decision,
            kernel=kernel,
            orientation=orientation,
        )
        return self._issue_certificate(
            kernel=kernel,
            orientation=orientation,
            decision=decision,
            actuation=actuation,
        )

    def _act_crowbar(
        self,
        decision: HeytingDecision,
        kernel: RoutonObservationKernel,
        orientation: RoutonOrientationState,
    ) -> CrowbarActuationReport:
        """
        Simula la ISR IRAM del Crowbar BT151.

        La latencia es una función *determinista* del sello de sesión en el
        intervalo [395, 399.5] ns, auditable y reproducible. No conmuta
        hardware real. El digest se serializa con `.hex()` (los `bytes`
        de SHA-256 no exponen `.hexdigest()`).
        """
        if not decision.is_hard_veto:
            return CrowbarActuationReport(
                interlock_fired=False,
                actuation_latency_ns=0.0,
                gpio=_CROWBAR_GPIO,
                device=_CROWBAR_DEVICE,
                seed_sha256="",
            )

        hasher = hashlib.sha256()
        hasher.update(b"CROWBAR_BT151_GPIO14_IRAM_V3_128D")
        hasher.update(kernel.session_sha256.encode("ascii"))
        for scalar in (
            orientation.eneagonal_associator_norm,
            orientation.moufang_distortion_norm,
            orientation.null_cone_friction,
            orientation.hurwitz_composition_error,
        ):
            hasher.update(_canonical_float_bytes(scalar))
        digest = hasher.digest()
        seed_hex = digest.hex()

        unit = int.from_bytes(digest[:2], byteorder="big", signed=False) / 65535.0
        latency_ns = 395.0 + 4.5 * float(unit)
        if latency_ns >= _CROWBAR_IRAM_LATENCY_NS:
            latency_ns = _CROWBAR_IRAM_LATENCY_NS - _MACHINE_EPS

        logger.critical("¡COLA DE HEYTING COLAPSADA EN SOBERANO ROUTÓNICO 128D!")
        logger.critical("  - Simulando subrutina local isVerdictCoherent() en C++...")
        logger.critical("  - Simulando ISR en IRAM (spec. < 400 ns)...")
        logger.critical(
            "  - Conmutando %s a HIGH en %.2f ns (simulado)...",
            _CROWBAR_GPIO,
            latency_ns,
        )
        logger.critical("  - Tiristor %s (Crowbar) gatillado en simulación.", _CROWBAR_DEVICE)
        logger.critical("  - Interlock hidráulico marcado en el milisegundo cero.")

        return CrowbarActuationReport(
            interlock_fired=True,
            actuation_latency_ns=latency_ns,
            gpio=_CROWBAR_GPIO,
            device=_CROWBAR_DEVICE,
            seed_sha256=seed_hex,
        )

    def _issue_certificate(
        self,
        kernel: RoutonObservationKernel,
        orientation: RoutonOrientationState,
        decision: HeytingDecision,
        actuation: CrowbarActuationReport,
    ) -> RoutonAgentCertificate:
        """Sello SHA-256 canónico: versión, Ω_3, escalares IEEE-754 y sello del motor."""
        hasher = hashlib.sha256()
        hasher.update(b"G_WISDOM_ROUTON_SUTURATED_V3")
        hasher.update(__version__.encode("utf-8"))
        hasher.update(decision.verdict.encode("ascii"))
        hasher.update(kernel.session_sha256.encode("ascii"))
        hasher.update(orientation.engine_cryptographic_seal.encode("ascii"))
        hasher.update(actuation.seed_sha256.encode("ascii"))

        for scalar in (
            orientation.eneagonal_associator_norm,
            orientation.moufang_distortion_norm,
            orientation.hurwitz_composition_error,
            orientation.null_cone_friction,
            orientation.null_cone_depth,
            orientation.left_moufang_defect,
            orientation.right_moufang_defect,
            orientation.middle_moufang_defect,
            orientation.stasheff_pentagon_diameter,
            orientation.sigma_min_left_r1,
            orientation.sigma_min_left_r2,
            actuation.actuation_latency_ns,
            decision.time_grace_remaining,
            float(decision.lattice_value),
        ):
            hasher.update(_canonical_float_bytes(scalar))

        for name, value in decision.conjuncts:
            hasher.update(name.encode("ascii"))
            hasher.update(value.encode("ascii"))

        digital_signature = hasher.hexdigest()

        return RoutonAgentCertificate(
            phase="G_WISDOM_ROUTON_SUTURATED_V3",
            heyting_verdict=decision.verdict,
            eneagonal_associator_norm=orientation.eneagonal_associator_norm,
            moufang_distortion_norm=orientation.moufang_distortion_norm,
            is_eneagonal_stable=orientation.is_eneagonal_stable,
            composition_error=orientation.hurwitz_composition_error,
            null_cone_friction=orientation.null_cone_friction,
            is_null_cone_penetrated=orientation.is_null_cone_penetrated,
            is_soft_veto_active=decision.is_soft_veto,
            override_grace_period_expired=decision.grace_expired,
            hardware_interlock_fired=actuation.interlock_fired,
            actuation_latency_ns=actuation.actuation_latency_ns,
            time_grace_remaining=decision.time_grace_remaining,
            digital_signature_sha256=digital_signature,
            trilateral_associator_norm=orientation.trilateral_associator_norm,
            moufang_relative_norm=orientation.moufang_relative_norm,
            eneagonal_relative_norm=orientation.eneagonal_relative_norm,
            moufang_frustration_index=orientation.moufang_frustration_index,
            eneagonal_frustration_index=orientation.eneagonal_frustration_index,
            moufang_threshold_used=orientation.moufang_threshold_used,
            eneagonal_threshold_used=orientation.eneagonal_threshold_used,
            is_moufang_stable=orientation.is_moufang_stable,
            is_eneagonal_norm_stable=orientation.is_eneagonal_norm_stable,
            is_hard_veto_active=decision.is_hard_veto,
            null_cone_relative_defect=orientation.null_cone_relative_defect,
            null_cone_depth=orientation.null_cone_depth,
            simplex_condition_number=orientation.simplex_condition_number,
            laplacian_connectivity=orientation.laplacian_connectivity,
            banach_ratios_9way=kernel.banach_ratios_9way,
            reasons=decision.reasons,
            is_hurwitz_stable=orientation.is_hurwitz_stable,
            is_null_cone_stable=orientation.is_null_cone_stable,
            left_moufang_defect=orientation.left_moufang_defect,
            right_moufang_defect=orientation.right_moufang_defect,
            middle_moufang_defect=orientation.middle_moufang_defect,
            stasheff_pentagon_diameter=orientation.stasheff_pentagon_diameter,
            sigma_min_left_r1=orientation.sigma_min_left_r1,
            sigma_min_left_r2=orientation.sigma_min_left_r2,
            engine_cryptographic_seal=orientation.engine_cryptographic_seal,
            heyting_conjuncts=decision.conjuncts,
            agent_version=__version__,
        )

    def audit_eneagonal_cycle(
        self,
        r1_vec: np.ndarray,
        r2_vec: np.ndarray,
        r3_vec: np.ndarray,
        r4_vec: np.ndarray,
        r5_vec: np.ndarray,
        r6_vec: np.ndarray,
        r7_vec: np.ndarray,
        r8_vec: np.ndarray,
        r9_vec: np.ndarray,
        eneagonal_threshold_Lmax: float = 15.0,
        override_token: Optional[str] = None,
        simulate_grace_expired: bool = False,
        moufang_threshold_Lmax: float = 500.0,
        null_critical_threshold: float = 1e-3,
    ) -> RoutonAgentCertificate:
        r"""
        Morfismo terminal de la Fase 3 y del soberano.

        Orquesta el ciclo covariante OODA como composición functorial estricta:

            \mathrm{Observe}(R_1,\ldots,R_9)
                \xrightarrow{\text{Fase 1}} \mathbf{Kernel}
            \xrightarrow{\text{Fase 2}} \mathbf{GovernedDecision}
            \xrightarrow{\text{Fase 3}} \mathbf{Certificate}.

        Flujo:
            1. observe_nine_way(R1..R9) → RoutonObservationKernel.
            2. govern_from_observation_kernel(kernel, ·) → RoutonGovernedDecision.
            3. continue_from_governed_decision(governed) → RoutonAgentCertificate.
        """
        eneagonal_threshold_Lmax = self._validate_non_negative_float(
            "eneagonal_threshold_Lmax", eneagonal_threshold_Lmax
        )
        moufang_threshold_Lmax = self._validate_non_negative_float(
            "moufang_threshold_Lmax", moufang_threshold_Lmax
        )
        null_critical_threshold = self._validate_non_negative_float(
            "null_critical_threshold", null_critical_threshold
        )
        cota_moufang_limite = moufang_threshold_Lmax * self._safety_margin
        cota_enea_limite = eneagonal_threshold_Lmax * self._safety_margin
        curr_time = time.monotonic()

        kernel = self.observe_nine_way(
            (r1_vec, r2_vec, r3_vec, r4_vec, r5_vec, r6_vec, r7_vec, r8_vec, r9_vec)
        )

        governed = self.govern_from_observation_kernel(
            kernel=kernel,
            eneagonal_threshold_Lmax=eneagonal_threshold_Lmax,
            moufang_threshold_Lmax=moufang_threshold_Lmax,
            null_critical_threshold=null_critical_threshold,
            cota_enea_limite=cota_enea_limite,
            cota_moufang_limite=cota_moufang_limite,
            override_token=override_token,
            simulate_grace_expired=simulate_grace_expired,
            curr_time=curr_time,
        )

        certificate = self.continue_from_governed_decision(governed)
        decision = governed.heyting

        if decision.is_hard_veto:
            logger.error(
                "VETO DURO ROUTÓNICO. Veredicto=%s. Sello=%s",
                decision.verdict,
                certificate.digital_signature_sha256[:16],
            )
        elif decision.is_soft_veto:
            logger.warning(
                "VETO SUAVE ROUTÓNICO. Veredicto=%s. Gracia restante=%.2f s. Sello=%s",
                decision.verdict,
                decision.time_grace_remaining,
                certificate.digital_signature_sha256[:16],
            )
        else:
            logger.info(
                "Soberano Routónico 128D regulado síncronamente. Veredicto=%s. Sello=%s",
                decision.verdict,
                certificate.digital_signature_sha256[:16],
            )
        return certificate


# ═══════════════════════════════════════════════════════════════════════════════
# §F. FACHADA SOBERANA
# ═══════════════════════════════════════════════════════════════════════════════
class RoutonDependencyAgent(Phase3_RoutonCrowbarActuator):
    r"""
    Soberano de Calibre Routónico 128D (OODA lazo cerrado, 3 fases anidadas).

    Gobierna de forma covariante los estados hipercomplejos de 128 dimensiones
    de la Malla Agéntica, administrando la Rampa de Confianza de de Rham para
    censurar cartelizaciones de 9 vías, distorsiones de Moufang y colapsos
    por divisores de cero.

    Phase3 ⊏ Phase2 ⊏ Phase1  ⇒  Act contiene Decide/Orient contiene Observe.
    """

    __slots__ = ()


__all__ = [
    "RoutonDependencyAgent",
    "Phase1_RoutonSovereignObserver",
    "Phase2_RoutonHeytingGovernor",
    "Phase3_RoutonCrowbarActuator",
    "RoutonObservationKernel",
    "RoutonOrientationState",
    "RoutonGovernedDecision",
    "RoutonAgentCertificate",
    "RoutonAgentThresholds",
    "BanachRegularityReport",
    "HeytingDecision",
    "HeytingOmega3",
    "CrowbarActuationReport",
]