# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Chingon Dependency Agent (Soberano de Calibre Chingónico 64D).      ║
║ Ruta   : app/agents/core/chingon_dependency_agent.py                         ║
║ Versión: 3.0.0-Doctoral-OODA-Heyting-CayleyDickson-64D-IRAM-ESP32-Nested3    ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y DE GOBERNANZA DE LAZO CERRADO:                         ║
║ Este agente supervisor ciber-físico opera en el Estrato de la Sabiduría      ║
║ (V_W, Nivel 0) u Omega (V_Ω, Nivel 0.5) para gobernar síncronamente al       ║
║ Motor de Calibre Chingónico 64D [chingon_dependency_engine.py] en la FPU.    ║
║                                                                              ║
║ Audita, como predicados locales en el retículo de Heyting Ω_3:               ║
║   1. Regularidad de Banach ℓ¹/ℓ²/ℓ^∞ sobre ℝ^{64} (equivalencia de normas).   ║
║   2. Asociador heptagonal de 7 vías A_7 y diámetro de Stasheff A_4.          ║
║   3. Tensores A_alt^L=[X,X,Y] y A_alt^R=[X,Y,Y] (homogéneos de grado 3).     ║
║   4. Composición de Hurwitz / submultiplicatividad de Banach.                ║
║   5. Penetración no trivial del cono nulo 𝒩(𝕏), σ_min(L_X).                  ║
║                                                                              ║
║ AXIOMAS DE GOBERNANZA:                                                       ║
║   (H3)  Ω_3 = {VETOED ≺ DEGRADED ≺ COHERENT} es la cadena de Heyting de      ║
║         tres elementos (álgebra de Gödel): ∧=mín, ∨=máx,                     ║
║         a → b = ⊤ si a ≼ b, si no b;  ¬a = a → ⊥.                            ║
║   (M)   El veredicto global es el meet de los predicados locales.            ║
║   (Γ)   Modalidad de gracia: Γ(DEGRADED)=DEGRADED si t<T, si no VETOED.      ║
║   (σ)   Override HMAC-SHA256 no promociona a ⊤: σ(DEGRADED)=DEGRADED         ║
║         con Γ desactivada. El veto duro no es anulable.                      ║
║   (B)   1 ≤ ‖x‖₁/‖x‖₂ ≤ √64=8,  1 ≤ ‖x‖₂/‖x‖_∞ ≤ 8,                          ║
║         1 ≤ ‖x‖₁/‖x‖_∞ ≤ 64  (equivalencia de normas en ℝ^{64}).             ║
║   (Alt) A_alt^L(X,Y)=[X,X,Y];  δ_alt=‖A_alt‖/‖X‖²‖Y‖.                        ║
║   (A7)  A_7 es el asociador de asociación extrema de 7 factores.             ║
║                                                                              ║
║ ARQUITECTURA FUNCTORIAL EN TRES FASES ANIDADAS (OODA Ω_3):                   ║
║   Fase 1  Observe : saneamiento, Banach, estados FPU 64D.                    ║
║           Morfismo terminal : observe_seven_way → ChingonObservationKernel.  ║
║   Fase 2  Orient + Decide : motor 64D, A_alt, A_7, meet de Heyting, Γ.       ║
║           Objeto inicial    : ChingonObservationKernel.                      ║
║           Morfismo terminal : govern_from_observation_kernel                 ║
║                               → ChingonGovernedDecision.                     ║
║   Fase 3  Act : Crowbar BT151 simulado (GPIO14, < 400 ns spec.) y sello.     ║
║           Objeto inicial    : ChingonGovernedDecision.                       ║
║           Morfismo terminal : audit_heptagonal_cycle                         ║
║                               → ChingonAgentCertificate.                     ║
║                                                                              ║
║ Anidación ontológica: Phase3 ⊏ Phase2 ⊏ Phase1  ⇒  Act∘Decide∘Observe.       ║
║ El último morfismo de la Fase k es el objeto inicial de la Fase k+1.         ║
║ En particular, Phase1.observe_seven_way devuelve un ChingonObservationKernel ║
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
    "3.0.0-Doctoral-OODA-Heyting-CayleyDickson-64D-IRAM-ESP32-Nested3"
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
        """Morfismo stub: fuente (ℝ^{64})⁷ → certificado en Ω_3."""

    class TopologicalInvariantError(Exception):  # type: ignore[no-redef]
        pass

# ------------------------------------------------------------------------------
# Importación del motor chingónico 64D y del álgebra de Cayley-Dickson.
# ------------------------------------------------------------------------------
try:
    from app.core.chingon_dependency_engine import (
        ChingonDependencyEngine,
        ChingonState,
        ChingonEngineState,
        ChingonMetricsReport,
        ChingonHeptagonalReport,
        CayleyDicksonAlgebra64,
        ChingonEngineError,
        ChingonDimensionError,
    )
except ImportError:
    from chingon_dependency_engine import (
        ChingonDependencyEngine,
        ChingonState,
        ChingonEngineState,
        ChingonMetricsReport,
        ChingonHeptagonalReport,
        CayleyDicksonAlgebra64,
        ChingonEngineError,
        ChingonDimensionError,
    )

try:
    from app.core.chingon_dependency_engine import (
        KBNSummationKernel,
        ChingonThresholds,
        NullConeReport,
        ChingonNumericalSingularityError,
    )
except ImportError:
    try:
        from chingon_dependency_engine import (
            KBNSummationKernel,
            ChingonThresholds,
            NullConeReport,
            ChingonNumericalSingularityError,
        )
    except ImportError:
        KBNSummationKernel = None  # type: ignore[assignment]
        ChingonThresholds = None  # type: ignore[assignment]
        NullConeReport = None  # type: ignore[assignment]
        ChingonNumericalSingularityError = ChingonEngineError  # type: ignore[assignment]


logger = logging.getLogger("APU.Agents.Wisdom.ChingonDependencyAgent")

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_FLOOR: Final[float] = 1e-15
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_CHINGON_DIM: Final[int] = 64
_BANACH_X64_UPPER_BOUND: Final[float] = float(math.sqrt(float(_CHINGON_DIM)))
_BANACH_L1_LINF_BOUND: Final[float] = float(_CHINGON_DIM)
_LOG_DBL_MAX: Final[float] = float(math.log(np.finfo(np.float64).max))
_LOG_DBL_TINY: Final[float] = float(math.log(np.finfo(np.float64).tiny))
_CROWBAR_GPIO: Final[str] = "GPIO14"
_CROWBAR_DEVICE: Final[str] = "BT151"

_DEFAULT_OVERRIDE_TOKENS: Final[Tuple[str, ...]] = (
    "AUT_POS_SABIDURIA_777",
    "OVERRIDE_NON_ALTERNATIVE_CHINGON_2026",
    "HMAC_SUTURA_FOCK_SECURE_64D",
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


# ═══════════════════════════════════════════════════════════════════════════════
# §B. DTOs INMUTABLES DEL AGENTE SOBERANO
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class ChingonAgentThresholds:
    r"""
    Umbrales de gobernanza (distintos de los umbrales FPU del motor).

    Las fracciones heptagonales parten la cota L_max en:
        banda elástica  (DEGRADED)  si  0.3 L_max < ‖A_7‖ ≤ 0.5 L_max,
        colapso duro    (VETOED)    si  ‖A_7‖ > 0.5 L_max  o no finito.

    A_alt no finito es veto duro; A_alt sobre cota es degradación (no colapso).
    """

    heptagonal_soft_fraction: float = 0.3
    heptagonal_hard_fraction: float = 0.5
    condition_number_limit: float = 1e12
    banach_eps_factor: float = 10.0
    alternative_relative_limit: float = 1e-2
    quadratic_leakage_limit: float = 1e-8

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"El umbral {name} debe ser finito y estrictamente positivo.")
        if self.heptagonal_soft_fraction >= self.heptagonal_hard_fraction:
            raise ValueError("Se exige 0 < heptagonal_soft_fraction < heptagonal_hard_fraction.")
        if self.heptagonal_hard_fraction > 1.0:
            raise ValueError("heptagonal_hard_fraction no puede exceder 1.")


@dataclass(frozen=True, slots=True)
class BanachRegularityReport:
    r"""
    Certificado local de regularidad de Banach sobre ℝ^{64}:

        ‖x‖₂ ≤ ‖x‖₁ ≤ 8 ‖x‖₂,
        ‖x‖_∞ ≤ ‖x‖₂ ≤ 8 ‖x‖_∞,
        ‖x‖_∞ ≤ ‖x‖₁ ≤ 64 ‖x‖_∞.

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
class ChingonObservationKernel:
    r"""
    Objeto terminal de la Fase 1 / objeto inicial de la Fase 2.

    Expediente inmutable de Observe: siete actores saneados, estados FPU,
    certificados de Banach y sello de sesión canónico.
    """

    states_7way: Tuple[ChingonState, ...]
    sanitized_vectors: Tuple[np.ndarray, ...]
    banach_reports: Tuple[BanachRegularityReport, ...]
    banach_ratios_7way: Tuple[float, ...]
    session_sha256: str
    all_banach_regular: bool = True

    @property
    def states(self) -> Tuple[ChingonState, ...]:
        """Compatibilidad con la versión 1.x."""
        return self.states_7way

    @property
    def banach_ratios(self) -> Tuple[float, ...]:
        """Compatibilidad con la versión 1.x."""
        return self.banach_ratios_7way

    @property
    def cryptographic_seal(self) -> str:
        """Compatibilidad con la versión 1.x."""
        return self.session_sha256


@dataclass(frozen=True, slots=True)
class ChingonOrientationState:
    r"""
    Expediente inmutable de Orient: métricas algebraicas, espectrales
    y de cono nulo necesarias para el meet de Heyting.
    """

    kernel: ChingonObservationKernel

    trilateral_associator_norm: float
    alternative_strain_norm: float
    heptagonal_associator_norm: float

    alternative_relative_norm: float
    heptagonal_relative_norm: float

    alternative_frustration_index: float
    heptagonal_frustration_index: float

    hurwitz_composition_error: float
    hurwitz_relative_error: float

    null_cone_friction: float
    null_cone_relative_defect: float
    null_cone_depth: float

    simplex_condition_number: float
    laplacian_connectivity: float

    is_alternative_stable: bool
    is_heptagonal_norm_stable: bool
    is_heptagonal_stable: bool
    is_hurwitz_stable: bool
    is_null_cone_stable: bool

    diagnosis: str

    alternative_threshold_used: float = 0.0
    heptagonal_threshold_used: float = 0.0
    right_alternative_strain_norm: float = 0.0
    flexibility_defect: float = 0.0
    power_associator_norm: float = 0.0
    moufang_defect: float = 0.0
    stasheff_pentagon_diameter: float = 0.0
    simplex_volume: float = 0.0
    laplacian_spectral_gap: float = 0.0
    estimated_connected_components: int = 1
    sigma_min_left_x1: float = math.inf
    sigma_min_left_x2: float = math.inf
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
class ChingonGovernedDecision:
    r"""
    Objeto terminal de la Fase 2 / objeto inicial de la Fase 3.

    Empaqueta orientación, decisión de Heyting y sello del motor,
    de modo que Act no reinstancia Observe ni Orient.
    """

    orientation: ChingonOrientationState
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
class ChingonAgentCertificate:
    r"""Certificado formal de calibración, regularidad de calibre y veto ciber-físico."""

    phase: str
    heyting_verdict: str

    heptagonal_associator_norm: float
    alternative_strain_norm: float
    is_heptagonal_stable: bool

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
    alternative_relative_norm: float = 0.0
    heptagonal_relative_norm: float = 0.0
    alternative_frustration_index: float = 0.0
    heptagonal_frustration_index: float = 0.0

    alternative_threshold_used: float = 0.0
    heptagonal_threshold_used: float = 0.0

    is_alternative_stable: bool = False
    is_heptagonal_norm_stable: bool = False
    is_hard_veto_active: bool = False

    null_cone_relative_defect: float = 0.0
    null_cone_depth: float = 0.0

    simplex_condition_number: float = math.inf
    laplacian_connectivity: float = 0.0

    banach_ratios_7way: Tuple[float, ...] = ()
    reasons: Tuple[str, ...] = ()

    is_hurwitz_stable: bool = False
    is_null_cone_stable: bool = False
    right_alternative_strain_norm: float = 0.0
    moufang_defect: float = 0.0
    stasheff_pentagon_diameter: float = 0.0
    sigma_min_left_x1: float = math.inf
    sigma_min_left_x2: float = math.inf
    engine_cryptographic_seal: str = ""
    heyting_conjuncts: Tuple[Tuple[str, str], ...] = ()
    agent_version: str = __version__


# ═══════════════════════════════════════════════════════════════════════════════
# §C. FASE 1 — OBSERVE
#     Morfismo terminal: observe_seven_way → ChingonObservationKernel
# ═══════════════════════════════════════════════════════════════════════════════
class Phase1_ChingonSovereignObserver(Morphism):
    r"""
    FASE 1 — Observe.

    Saneamiento de de Rham, regularidad de Banach ℓ¹/ℓ²/ℓ^∞ y construcción
    de estados chingónicos FPU para los siete actores del 6-símplex.

    Categoría fuente: \(\mathbf{Vec}_{64}^{7}\).
    Categoría meta  : \(\mathbf{ObservationKernel}\).

    El método terminal `observe_seven_way` es el objeto inicial de la Fase 2.
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
        engine: Optional[ChingonDependencyEngine] = None,
        hmac_secret: Optional[bytes] = None,
        agent_thresholds: Optional[ChingonAgentThresholds] = None,
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
        self._agent_thresholds: Final[ChingonAgentThresholds] = (
            agent_thresholds or ChingonAgentThresholds()
        )

        secret = hmac_secret
        if secret is None:
            env = os.environ.get("CHINGON_HMAC_SECRET")
            secret = env.encode("utf-8") if env else None
        self._hmac_secret: Final[Optional[bytes]] = secret

        self._is_soft_veto_active: bool = False
        self._soft_veto_timestamp: Optional[float] = None

        self._engine: Final[ChingonDependencyEngine] = (
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

    def _build_engine(self) -> ChingonDependencyEngine:
        """Construye el motor 64D, preferiendo umbrales evolucionados."""
        if ChingonThresholds is not None:
            base_kwargs = dict(
                absolute_tolerance=self._tol,
                relative_tolerance=max(self._tol * 10.0, 1e-10),
                zero_norm_threshold=max(self._tol, 1e-14),
                hurwitz_absolute=1e-9,
                hurwitz_relative=1e-8,
                alternative_absolute=500.0,
                alternative_relative=1e-2,
                heptagonal_absolute=5_000_000.0,
                heptagonal_relative=1e-2,
                null_absolute=1e-12,
                null_relative=1e-8,
                null_depth_threshold=0.99,
                condition_number_limit=self._agent_thresholds.condition_number_limit,
            )
            try:
                thresholds = ChingonThresholds(
                    **base_kwargs,
                    quadratic_leakage_limit=self._agent_thresholds.quadratic_leakage_limit,
                )
                return ChingonDependencyEngine(thresholds=thresholds)
            except TypeError:
                try:
                    return ChingonDependencyEngine(
                        thresholds=ChingonThresholds(**base_kwargs)
                    )
                except TypeError:
                    pass
            except Exception as exc:
                logger.debug("No se pudo construir motor con ChingonThresholds: %s", exc)

        try:
            return ChingonDependencyEngine(tolerance=self._tol)
        except TypeError:
            return ChingonDependencyEngine()

    def reset_grace_window(self) -> None:
        """Reinicia la modalidad de gracia Γ (único estado mutable del soberano)."""
        self._is_soft_veto_active = False
        self._soft_veto_timestamp = None

    @staticmethod
    def _sanitize_vector(value: np.ndarray, name: str) -> np.ndarray:
        """
        Sanea un vector 64D:

        - fuerza dtype float64 y forma estricta (64,);
        - exige componentes finitas;
        - colapsa ceros signados (−0.0 → +0.0);
        - congela la copia en RAM (write=False).
        """
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 1 or arr.shape != (_CHINGON_DIM,):
            raise ChingonDimensionError(
                f"{name} debe ser estrictamente de dimensión {_CHINGON_DIM}. "
                f"Obtenido: shape={arr.shape}."
            )

        arr = np.ascontiguousarray(arr, dtype=np.float64).copy()
        if not np.all(np.isfinite(arr)):
            raise ChingonNumericalSingularityError(
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
            raise ChingonNumericalSingularityError(
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
                and ratio12 <= _BANACH_X64_UPPER_BOUND * (1.0 + slack)
                and ratio2inf + slack >= 1.0
                and ratio2inf <= _BANACH_X64_UPPER_BOUND * (1.0 + slack)
                and ratio1inf + slack >= 1.0
                and ratio1inf <= _BANACH_L1_LINF_BOUND * (1.0 + slack)
            )

        return BanachRegularityReport(
            norm_1=norm1,
            norm_2=norm2,
            ratio=ratio12,
            lower_bound=1.0,
            upper_bound=_BANACH_X64_UPPER_BOUND,
            is_within_theoretical_bounds=within,
            norm_inf=norm_inf,
            ratio_2inf=ratio2inf,
            ratio_1inf=ratio1inf,
        )

    def evaluate_banach_regularity_report(self, S: np.ndarray) -> BanachRegularityReport:
        """API pública granular: certificado de Banach de un vector 64D."""
        return self._banach_report_from_sanitized(self._sanitize_vector(S, "S"))

    def evaluate_banach_regularity(self, S: np.ndarray) -> float:
        """API pública 1.x: retorna únicamente el ratio ℓ¹/ℓ²."""
        return self.evaluate_banach_regularity_report(S).ratio

    def observe_seven_way(
        self,
        actors_raw: Sequence[np.ndarray],
    ) -> ChingonObservationKernel:
        r"""
        Morfismo terminal de la Fase 1.

        Sanea, certifica Banach y construye estados FPU para exactamente
        siete actores. Firma functorial:

            \mathrm{observe\_seven\_way}:
                (\mathbb{R}^{64})^{7}\longrightarrow\mathbf{ChingonObservationKernel}.

        El valor de retorno es el objeto inicial de
        `Phase2_ChingonHeytingGovernor.continue_from_observation_kernel`.
        """
        if len(actors_raw) != 7:
            raise ChingonEngineError(
                "La auditoría chingónica requiere exactamente siete actores."
            )

        sanitized_vectors = []
        banach_reports = []
        banach_ratios = []
        states_64d = []

        hasher = hashlib.sha256()
        hasher.update(b"CHINGON_OBSERVATION_PHASE_1_V3")
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
            states_64d.append(state)
            all_regular = all_regular and banach_report.is_within_theoretical_bounds

            hasher.update(np.asarray(clean_vec, dtype="<f8").tobytes(order="C"))
            hasher.update(_canonical_float_bytes(banach_report.ratio))
            hasher.update(_canonical_float_bytes(banach_report.ratio_2inf))
            hasher.update(_canonical_float_bytes(banach_report.ratio_1inf))

            state_hash = getattr(state, "sha256_hash", "")
            if state_hash:
                hasher.update(str(state_hash).encode("ascii"))

        kernel = ChingonObservationKernel(
            states_7way=tuple(states_64d),
            sanitized_vectors=tuple(sanitized_vectors),
            banach_reports=tuple(banach_reports),
            banach_ratios_7way=tuple(banach_ratios),
            session_sha256=hasher.hexdigest(),
            all_banach_regular=all_regular,
        )

        logger.info(
            "Fase Observe [CHINGON_AGENT_64D]: 7 actores empaquetados en Banach 64D. Sello: %s",
            kernel.session_sha256[:16],
        )
        return kernel

    def _observe(self, actors_raw: Sequence[np.ndarray]) -> ChingonObservationKernel:
        """Alias interno de compatibilidad 2.x → morfismo terminal de la Fase 1."""
        return self.observe_seven_way(actors_raw)


# ═══════════════════════════════════════════════════════════════════════════════
# §D. FASE 2 — ORIENT + DECIDE
#     Objeto inicial = morfismo terminal de la Fase 1 (ChingonObservationKernel)
#     Morfismo terminal: govern_from_observation_kernel → ChingonGovernedDecision
# ═══════════════════════════════════════════════════════════════════════════════
class Phase2_ChingonHeytingGovernor(Phase1_ChingonSovereignObserver):
    r"""
    FASE 2 — Orient + Decide.

    Consume `ChingonObservationKernel` (terminal de Fase 1), orquesta el
    motor 64D (Hurwitz, A_alt^{L/R}, A_7, Stasheff, Moufang, cono nulo)
    y clasifica el 6-símplex en Ω_3 mediante el meet de predicados locales
    con modalidad de gracia Γ.

    El método inicial `continue_from_observation_kernel` es la continuación
    formal de `Phase1.observe_seven_way`.
    El método terminal `govern_from_observation_kernel` produce el objeto
    inicial de la Fase 3 (`ChingonGovernedDecision`).
    """

    __slots__ = ()

    def continue_from_observation_kernel(
        self,
        kernel: ChingonObservationKernel,
        alternative_threshold_Lmax: float,
        heptagonal_threshold_Lmax: float,
        null_critical_threshold: float,
        cota_alt_limite: float,
        cota_hept_limite: float,
    ) -> ChingonOrientationState:
        r"""
        Morfismo de entrada de la Fase 2 (continuación formal de
        `Phase1_ChingonSovereignObserver.observe_seven_way`).

            \mathrm{continue\_from\_observation\_kernel}:
                \mathbf{ChingonObservationKernel}\times\mathbb{R}_{+}^{5}
                \longrightarrow\mathbf{ChingonOrientationState}.
        """
        return self._orient(
            kernel=kernel,
            alternative_threshold_Lmax=alternative_threshold_Lmax,
            heptagonal_threshold_Lmax=heptagonal_threshold_Lmax,
            null_critical_threshold=null_critical_threshold,
            cota_alt_limite=cota_alt_limite,
            cota_hept_limite=cota_hept_limite,
        )

    def _evaluate_null_metrics(
        self,
        x1_state: ChingonState,
        x2_state: ChingonState,
        null_critical_threshold: float,
        engine_null: Any = None,
    ) -> Tuple[float, float, float, bool, bool, float, float, float]:
        """
        Fricción, profundidad y espectro de 𝒩(𝕏) para el par (X₁, X₂).

        Devuelve:
            friction, relative_defect, null_depth,
            is_null_penetrated, is_null_stable,
            σ_min(L_{X₁}), σ_min(L_{X₂}), ‖[X₁,X₂]‖.
        """
        if engine_null is not None:
            friction = _attr_float(engine_null, "absolute_friction", math.inf)
            relative_defect = _attr_float(engine_null, "relative_defect", math.inf)
            null_depth = _attr_float(engine_null, "null_depth", 0.0)
            penetrated = _attr_bool(engine_null, "is_null_cone_penetrated", False)
            sigma1 = _attr_float(
                engine_null,
                "sigma_min_left_x1",
                _attr_float(engine_null, "sigma_min_left_p1", math.inf),
            )
            sigma2 = _attr_float(
                engine_null,
                "sigma_min_left_x2",
                _attr_float(engine_null, "sigma_min_left_p2", math.inf),
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
            report = self._engine.evaluate_null_cone(x1_state, x2_state)
            return self._evaluate_null_metrics(
                x1_state, x2_state, null_critical_threshold, engine_null=report
            )

        prod_vec = CayleyDicksonAlgebra64.multiply(x1_state.vector_rep, x2_state.vector_rep)
        prod_norm = _vector_norm_2(prod_vec)
        log_expected = _agent_log_norm(x1_state.norm) + _agent_log_norm(x2_state.norm)
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
        kernel: ChingonObservationKernel,
        alternative_threshold_Lmax: float,
        heptagonal_threshold_Lmax: float,
        null_critical_threshold: float,
        cota_alt_limite: float,
        cota_hept_limite: float,
    ) -> ChingonOrientationState:
        """Orient: una pasada del motor 64D + diagnóstico compuesto."""
        states = kernel.states_7way
        sanitized = kernel.sanitized_vectors

        engine_state: Any = None
        metrics_report: Any = None
        hept_report: Any = None
        null_report: Any = None

        if hasattr(self._engine, "execute_heptagonal_audit"):
            engine_state = self._engine.execute_heptagonal_audit(
                sanitized[0],
                sanitized[1],
                sanitized[2],
                sanitized[3],
                sanitized[4],
                sanitized[5],
                sanitized[6],
                alternative_threshold=cota_alt_limite,
                heptagonal_threshold=cota_hept_limite,
            )
            metrics_report = getattr(engine_state, "metrics_report", None)
            hept_report = getattr(engine_state, "heptagonal_report", None)
            null_report = getattr(engine_state, "null_report", None)
        else:
            if hasattr(self._engine, "observe_metrics"):
                metrics_report = self._engine.observe_metrics(sanitized[0], sanitized[1])
            if hasattr(self._engine, "calculate_heptagonal_frustration"):
                hept_report = self._engine.calculate_heptagonal_frustration(
                    *sanitized,
                    alternative_threshold=cota_alt_limite,
                    heptagonal_threshold=cota_hept_limite,
                )
            else:
                raise ChingonEngineError(
                    "El motor chingónico no expone calculate_heptagonal_frustration "
                    "ni execute_heptagonal_audit."
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
            prod_vec = CayleyDicksonAlgebra64.multiply(
                states[0].vector_rep, states[1].vector_rep
            )
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

        trilateral_norm = _attr_float(hept_report, "trilateral_associator_norm", 0.0)
        alt_norm = _attr_float(hept_report, "alternative_strain_norm", math.inf)
        hept_norm = _attr_float(hept_report, "heptagonal_associator_norm", math.inf)
        if math.isnan(alt_norm):
            alt_norm = math.inf
        if math.isnan(hept_norm):
            hept_norm = math.inf

        log_den_alt = (
            2.0 * _agent_log_norm(max(float(states[0].norm), _WILKINSON_FLOOR))
            + _agent_log_norm(max(float(states[1].norm), _WILKINSON_FLOOR))
        )
        log_den_hept = sum(
            _agent_log_norm(max(float(state.norm), _WILKINSON_FLOOR)) for state in states
        )
        default_alt_rel = _agent_safe_exp(_agent_log_norm(alt_norm) - log_den_alt)
        default_hept_rel = _agent_safe_exp(_agent_log_norm(hept_norm) - log_den_hept)
        default_alt_frustration = _agent_safe_exp(
            _agent_log_norm(alt_norm) - max(0.0, log_den_alt)
        )
        default_hept_frustration = _agent_safe_exp(
            _agent_log_norm(hept_norm) - max(0.0, log_den_hept)
        )

        alt_rel = _attr_float(hept_report, "alternative_relative_norm", default_alt_rel)
        hept_rel = _attr_float(hept_report, "heptagonal_relative_norm", default_hept_rel)
        alt_frustration = _attr_float(
            hept_report, "alternative_frustration_index", default_alt_frustration
        )
        hept_frustration = _attr_float(
            hept_report, "heptagonal_frustration_index", default_hept_frustration
        )
        alt_threshold_used = _attr_float(
            hept_report, "alternative_threshold_used", cota_alt_limite
        )
        hept_threshold_used = _attr_float(
            hept_report, "heptagonal_threshold_used", cota_hept_limite
        )

        is_alt_stable = _attr_bool(
            hept_report,
            "is_alternative_stable",
            math.isfinite(alt_norm) and alt_norm <= (cota_alt_limite + self._tol),
        )
        is_hept_norm_stable = _attr_bool(
            hept_report,
            "is_heptagonal_norm_stable",
            math.isfinite(hept_norm) and hept_norm <= (cota_hept_limite + self._tol),
        )
        is_hept_stable = _attr_bool(
            hept_report,
            "is_heptagonal_stable",
            is_alt_stable and is_hept_norm_stable,
        )

        condition_number = _attr_float(hept_report, "simplex_condition_number", math.inf)
        connectivity = _attr_float(hept_report, "laplacian_connectivity", 0.0)
        base_diagnosis = _attr_str(hept_report, "diagnosis", "")
        right_alt = _attr_float(hept_report, "right_alternative_strain_norm", 0.0)
        flexibility = _attr_float(hept_report, "flexibility_defect", 0.0)
        power_assoc = _attr_float(hept_report, "power_associator_norm", 0.0)
        moufang = _attr_float(hept_report, "moufang_defect", 0.0)
        stasheff = _attr_float(hept_report, "stasheff_pentagon_diameter", 0.0)
        simplex_volume = _attr_float(hept_report, "simplex_volume", 0.0)
        spectral_gap = _attr_float(hept_report, "laplacian_spectral_gap", 0.0)
        components = int(
            round(_attr_float(hept_report, "estimated_connected_components", 1.0))
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
        if not math.isfinite(alt_norm):
            extra_diagnosis.append("A_alt no finito.")
        if not math.isfinite(hept_norm):
            extra_diagnosis.append("A_7 no finito.")
        if not is_alt_stable:
            extra_diagnosis.append("Deformación no alternativa fuera de cota.")
        if not is_hept_norm_stable:
            extra_diagnosis.append("A_7 fuera de cota heptagonal.")
        if not is_hurwitz_stable:
            extra_diagnosis.append("Deriva de Hurwitz.")
        if not is_null_stable:
            extra_diagnosis.append("Inestabilidad en el cono nulo.")
        if condition_number > self._agent_thresholds.condition_number_limit:
            extra_diagnosis.append("6-símplex con condicionamiento afín degenerado.")
        if connectivity <= _MACHINE_EPS:
            extra_diagnosis.append("Conectividad algebraica casi nula.")
        if not kernel.all_banach_regular:
            extra_diagnosis.append("Violación de equivalencia de normas de Banach.")

        diagnosis_parts = [part for part in (base_diagnosis, *extra_diagnosis) if part]
        diagnosis = " | ".join(diagnosis_parts) if diagnosis_parts else "Orientación estable."

        return ChingonOrientationState(
            kernel=kernel,
            trilateral_associator_norm=trilateral_norm,
            alternative_strain_norm=alt_norm,
            heptagonal_associator_norm=hept_norm,
            alternative_relative_norm=alt_rel,
            heptagonal_relative_norm=hept_rel,
            alternative_frustration_index=alt_frustration,
            heptagonal_frustration_index=hept_frustration,
            hurwitz_composition_error=hurwitz_abs,
            hurwitz_relative_error=hurwitz_rel,
            null_cone_friction=null_friction,
            null_cone_relative_defect=null_relative,
            null_cone_depth=null_depth,
            simplex_condition_number=condition_number,
            laplacian_connectivity=connectivity,
            is_alternative_stable=is_alt_stable,
            is_heptagonal_norm_stable=is_hept_norm_stable,
            is_heptagonal_stable=is_hept_stable,
            is_hurwitz_stable=is_hurwitz_stable,
            is_null_cone_stable=is_null_stable,
            diagnosis=diagnosis,
            alternative_threshold_used=alt_threshold_used,
            heptagonal_threshold_used=hept_threshold_used,
            right_alternative_strain_norm=right_alt,
            flexibility_defect=flexibility,
            power_associator_norm=power_assoc,
            moufang_defect=moufang,
            stasheff_pentagon_diameter=stasheff,
            simplex_volume=simplex_volume,
            laplacian_spectral_gap=spectral_gap,
            estimated_connected_components=components,
            sigma_min_left_x1=sigma1,
            sigma_min_left_x2=sigma2,
            commutator_norm=commutator,
            is_banach_submultiplicative=is_banach_sub,
            is_null_cone_penetrated=is_null_penetrated,
            engine_cryptographic_seal=engine_seal,
            quadratic_leakage_max=quadratic_leakage_max,
        )

    def _local_heyting_predicates(
        self,
        orientation: ChingonOrientationState,
        cota_alt_limite: float,
        cota_hept_limite: float,
    ) -> Tuple[Tuple[str, HeytingOmega3, str], ...]:
        """Predicados locales en Ω_3 cuyo meet es el veredicto global."""
        thr = self._agent_thresholds
        hept = orientation.heptagonal_associator_norm
        alt = orientation.alternative_strain_norm

        if not math.isfinite(hept):
            hept_h = HeytingOmega3.VETOED
            hept_reason = "A_7 no finito: singularidad heptagonal."
        elif hept > (thr.heptagonal_hard_fraction * cota_hept_limite + self._tol):
            hept_h = HeytingOmega3.VETOED
            hept_reason = "Colapso heptagonal duro."
        elif hept > (thr.heptagonal_soft_fraction * cota_hept_limite + self._tol):
            hept_h = HeytingOmega3.DEGRADED
            hept_reason = "Frustración heptagonal en banda elástica."
        else:
            hept_h = HeytingOmega3.COHERENT
            hept_reason = ""

        if not math.isfinite(alt):
            alt_h = HeytingOmega3.VETOED
            alt_reason = "A_alt no finito: singularidad no alternativa."
        elif alt > (cota_alt_limite + self._tol):
            alt_h = HeytingOmega3.DEGRADED
            alt_reason = "Deformación no alternativa fuera de cota."
        else:
            alt_h = HeytingOmega3.COHERENT
            alt_reason = ""

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
            ("heptagonal", hept_h, hept_reason),
            ("alternative", alt_h, alt_reason),
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
                else "Inminencia de divisor de cero en N(X).",
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
                else "6-símplex con condicionamiento afín degenerado.",
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
        orientation: ChingonOrientationState,
        cota_alt_limite: float,
        cota_hept_limite: float,
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
            orientation, cota_alt_limite, cota_hept_limite
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
        kernel: ChingonObservationKernel,
        alternative_threshold_Lmax: float,
        heptagonal_threshold_Lmax: float,
        null_critical_threshold: float,
        cota_alt_limite: float,
        cota_hept_limite: float,
        override_token: Optional[str],
        simulate_grace_expired: bool,
        curr_time: float,
    ) -> ChingonGovernedDecision:
        r"""
        Morfismo terminal de la Fase 2.

        Compone Orient ∘ Decide sobre el kernel de la Fase 1 y entrega el
        `ChingonGovernedDecision`, objeto inicial de
        `Phase3_ChingonCrowbarActuator.continue_from_governed_decision`.
        """
        orientation = self.continue_from_observation_kernel(
            kernel=kernel,
            alternative_threshold_Lmax=alternative_threshold_Lmax,
            heptagonal_threshold_Lmax=heptagonal_threshold_Lmax,
            null_critical_threshold=null_critical_threshold,
            cota_alt_limite=cota_alt_limite,
            cota_hept_limite=cota_hept_limite,
        )
        heyting = self.decide_from_orientation(
            orientation=orientation,
            cota_alt_limite=cota_alt_limite,
            cota_hept_limite=cota_hept_limite,
            override_token=override_token,
            simulate_grace_expired=simulate_grace_expired,
            curr_time=curr_time,
        )
        return ChingonGovernedDecision(orientation=orientation, heyting=heyting)


# ═══════════════════════════════════════════════════════════════════════════════
# §E. FASE 3 — ACT
#     Objeto inicial = morfismo terminal de la Fase 2 (ChingonGovernedDecision)
#     Morfismo terminal: audit_heptagonal_cycle → ChingonAgentCertificate
# ═══════════════════════════════════════════════════════════════════════════════
class Phase3_ChingonCrowbarActuator(Phase2_ChingonHeytingGovernor):
    r"""
    FASE 3 — Act.

    Actuación Crowbar BT151 *simulada* (especificación IRAM: latencia
    estrictamente menor a 400 ns sobre GPIO14) y emisión del certificado
    terminal, encadenando el sello criptográfico del motor 64D.

    El método inicial `continue_from_governed_decision` es la continuación
    formal de `Phase2.govern_from_observation_kernel`.
    El método terminal `audit_heptagonal_cycle` cierra el ciclo OODA.
    """

    __slots__ = ()

    def continue_from_governed_decision(
        self,
        governed: ChingonGovernedDecision,
    ) -> ChingonAgentCertificate:
        r"""
        Morfismo de entrada de la Fase 3 (continuación formal de
        `Phase2_ChingonHeytingGovernor.govern_from_observation_kernel`).

            \mathrm{continue\_from\_governed\_decision}:
                \mathbf{ChingonGovernedDecision}
                \longrightarrow\mathbf{ChingonAgentCertificate}.
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
        kernel: ChingonObservationKernel,
        orientation: ChingonOrientationState,
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
        hasher.update(b"CROWBAR_BT151_GPIO14_IRAM_V3_64D")
        hasher.update(kernel.session_sha256.encode("ascii"))
        for scalar in (
            orientation.heptagonal_associator_norm,
            orientation.alternative_strain_norm,
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

        logger.critical("¡COLA DE HEYTING COLAPSADA EN SOBERANO CHINGÓNICO 64D!")
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
        kernel: ChingonObservationKernel,
        orientation: ChingonOrientationState,
        decision: HeytingDecision,
        actuation: CrowbarActuationReport,
    ) -> ChingonAgentCertificate:
        """Sello SHA-256 canónico: versión, Ω_3, escalares IEEE-754 y sello del motor."""
        hasher = hashlib.sha256()
        hasher.update(b"G_WISDOM_CHINGON_SUTURATED_V3")
        hasher.update(__version__.encode("utf-8"))
        hasher.update(decision.verdict.encode("ascii"))
        hasher.update(kernel.session_sha256.encode("ascii"))
        hasher.update(orientation.engine_cryptographic_seal.encode("ascii"))
        hasher.update(actuation.seed_sha256.encode("ascii"))

        for scalar in (
            orientation.heptagonal_associator_norm,
            orientation.alternative_strain_norm,
            orientation.hurwitz_composition_error,
            orientation.null_cone_friction,
            orientation.null_cone_depth,
            orientation.right_alternative_strain_norm,
            orientation.moufang_defect,
            orientation.stasheff_pentagon_diameter,
            orientation.sigma_min_left_x1,
            orientation.sigma_min_left_x2,
            actuation.actuation_latency_ns,
            decision.time_grace_remaining,
            float(decision.lattice_value),
        ):
            hasher.update(_canonical_float_bytes(scalar))

        for name, value in decision.conjuncts:
            hasher.update(name.encode("ascii"))
            hasher.update(value.encode("ascii"))

        digital_signature = hasher.hexdigest()

        return ChingonAgentCertificate(
            phase="G_WISDOM_CHINGON_SUTURATED_V3",
            heyting_verdict=decision.verdict,
            heptagonal_associator_norm=orientation.heptagonal_associator_norm,
            alternative_strain_norm=orientation.alternative_strain_norm,
            is_heptagonal_stable=orientation.is_heptagonal_stable,
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
            alternative_relative_norm=orientation.alternative_relative_norm,
            heptagonal_relative_norm=orientation.heptagonal_relative_norm,
            alternative_frustration_index=orientation.alternative_frustration_index,
            heptagonal_frustration_index=orientation.heptagonal_frustration_index,
            alternative_threshold_used=orientation.alternative_threshold_used,
            heptagonal_threshold_used=orientation.heptagonal_threshold_used,
            is_alternative_stable=orientation.is_alternative_stable,
            is_heptagonal_norm_stable=orientation.is_heptagonal_norm_stable,
            is_hard_veto_active=decision.is_hard_veto,
            null_cone_relative_defect=orientation.null_cone_relative_defect,
            null_cone_depth=orientation.null_cone_depth,
            simplex_condition_number=orientation.simplex_condition_number,
            laplacian_connectivity=orientation.laplacian_connectivity,
            banach_ratios_7way=kernel.banach_ratios_7way,
            reasons=decision.reasons,
            is_hurwitz_stable=orientation.is_hurwitz_stable,
            is_null_cone_stable=orientation.is_null_cone_stable,
            right_alternative_strain_norm=orientation.right_alternative_strain_norm,
            moufang_defect=orientation.moufang_defect,
            stasheff_pentagon_diameter=orientation.stasheff_pentagon_diameter,
            sigma_min_left_x1=orientation.sigma_min_left_x1,
            sigma_min_left_x2=orientation.sigma_min_left_x2,
            engine_cryptographic_seal=orientation.engine_cryptographic_seal,
            heyting_conjuncts=decision.conjuncts,
            agent_version=__version__,
        )

    def audit_heptagonal_cycle(
        self,
        x1_vec: np.ndarray,
        x2_vec: np.ndarray,
        x3_vec: np.ndarray,
        x4_vec: np.ndarray,
        x5_vec: np.ndarray,
        x6_vec: np.ndarray,
        x7_vec: np.ndarray,
        heptagonal_threshold_Lmax: float = 10.0,
        override_token: Optional[str] = None,
        simulate_grace_expired: bool = False,
        alternative_threshold_Lmax: float = 100.0,
        null_critical_threshold: float = 1e-3,
    ) -> ChingonAgentCertificate:
        r"""
        Morfismo terminal de la Fase 3 y del soberano.

        Orquesta el ciclo covariante OODA como composición functorial estricta:

            \mathrm{Observe}(X_1,\ldots,X_7)
                \xrightarrow{\text{Fase 1}} \mathbf{Kernel}
            \xrightarrow{\text{Fase 2}} \mathbf{GovernedDecision}
            \xrightarrow{\text{Fase 3}} \mathbf{Certificate}.

        Flujo:
            1. observe_seven_way(X1..X7) → ChingonObservationKernel.
            2. govern_from_observation_kernel(kernel, ·) → ChingonGovernedDecision.
            3. continue_from_governed_decision(governed) → ChingonAgentCertificate.
        """
        heptagonal_threshold_Lmax = self._validate_non_negative_float(
            "heptagonal_threshold_Lmax", heptagonal_threshold_Lmax
        )
        alternative_threshold_Lmax = self._validate_non_negative_float(
            "alternative_threshold_Lmax", alternative_threshold_Lmax
        )
        null_critical_threshold = self._validate_non_negative_float(
            "null_critical_threshold", null_critical_threshold
        )
        cota_alt_limite = alternative_threshold_Lmax * self._safety_margin
        cota_hept_limite = heptagonal_threshold_Lmax * self._safety_margin
        curr_time = time.monotonic()

        kernel = self.observe_seven_way(
            (x1_vec, x2_vec, x3_vec, x4_vec, x5_vec, x6_vec, x7_vec)
        )

        governed = self.govern_from_observation_kernel(
            kernel=kernel,
            alternative_threshold_Lmax=alternative_threshold_Lmax,
            heptagonal_threshold_Lmax=heptagonal_threshold_Lmax,
            null_critical_threshold=null_critical_threshold,
            cota_alt_limite=cota_alt_limite,
            cota_hept_limite=cota_hept_limite,
            override_token=override_token,
            simulate_grace_expired=simulate_grace_expired,
            curr_time=curr_time,
        )

        certificate = self.continue_from_governed_decision(governed)
        decision = governed.heyting

        if decision.is_hard_veto:
            logger.error(
                "VETO DURO CHINGÓNICO. Veredicto=%s. Sello=%s",
                decision.verdict,
                certificate.digital_signature_sha256[:16],
            )
        elif decision.is_soft_veto:
            logger.warning(
                "VETO SUAVE CHINGÓNICO. Veredicto=%s. Gracia restante=%.2f s. Sello=%s",
                decision.verdict,
                decision.time_grace_remaining,
                certificate.digital_signature_sha256[:16],
            )
        else:
            logger.info(
                "Soberano Chingónico 64D regulado síncronamente. Veredicto=%s. Sello=%s",
                decision.verdict,
                certificate.digital_signature_sha256[:16],
            )
        return certificate


# ═══════════════════════════════════════════════════════════════════════════════
# §F. FACHADA SOBERANA
# ═══════════════════════════════════════════════════════════════════════════════
class ChingonDependencyAgent(Phase3_ChingonCrowbarActuator):
    r"""
    Soberano de Calibre Chingónico 64D (OODA lazo cerrado, 3 fases anidadas).

    Gobierna de forma covariante los estados hipercomplejos de 64 dimensiones
    de la Malla Agéntica, administrando la Rampa de Confianza de de Rham para
    censurar cartelizaciones de 7 vías, deformaciones no alternativas y
    colapsos por divisores de cero.

    Phase3 ⊏ Phase2 ⊏ Phase1  ⇒  Act contiene Decide/Orient contiene Observe.
    """

    __slots__ = ()


__all__ = [
    "ChingonDependencyAgent",
    "Phase1_ChingonSovereignObserver",
    "Phase2_ChingonHeytingGovernor",
    "Phase3_ChingonCrowbarActuator",
    "ChingonObservationKernel",
    "ChingonOrientationState",
    "ChingonGovernedDecision",
    "ChingonAgentCertificate",
    "ChingonAgentThresholds",
    "BanachRegularityReport",
    "HeytingDecision",
    "HeytingOmega3",
    "CrowbarActuationReport",
]