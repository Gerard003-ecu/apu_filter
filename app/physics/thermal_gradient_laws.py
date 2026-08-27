# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Thermal Gradient Agent (Soberano de Calibre del Campo Térmico)      ║
║ Ruta   : app/agents/physics/thermal_gradient_agent.py                        ║
║ Versión: 3.1.0-Doctoral-Caputo-CechSheaf-KMS-Heyting-OODA-CAS-Secure         ║
║                                                                              ║
║ SINOPSIS (rigor doctoral; lo que SÍ se computa):                             ║
║ El agente es el endofunctor de supervisión  S = Act ∘ Orient ∘ Observe       ║
║ sobre sellos de ThermalGradientLaws. No re-simula Fourier. Audita el         ║
║ certificado y tres memorias / geometrías opcionales:                         ║
║                                                                              ║
║   (1) Caputo / Grünwald–Letnikov  α ∈ (0,1) sobre la serie {Φ_n, T_n}        ║
║       D^α f_n = Δt^{-α} ∑_{j=0}^{n} w_j^{(α)} f_{n-j}                        ║
║       w_0=1,  w_j = (1 − (α+1)/j) w_{j−1}                                    ║
║       I^α Φ  (Riemann–Liouville discreta) acumula fugas seculares.           ║
║       |D^α T| grande + I^α Φ ≥ 0  → transitorio (no veto).                   ║
║       I^α Φ < τ_frac persistente → veto secular.  NO es un D^α de Rham       ║
║       sobre un complejo simplicial: no hay 1-esqueleto aquí.                 ║
║                                                                              ║
║   (2) Haz de Heyting sobre un cubrimiento finito {U_i} de coordenadas        ║
║       Sección Γ(U_i): veredicto local de ⟨q,dT⟩|_{U_i}.                      ║
║       Restricción: meet en U_i ∩ U_j.                                        ║
║       H¹_Čech ≠ 0  ⇔  dos cartas solapadas con |Δ rank| ≥ 2.                 ║
║       Veto quirúrgico: se aislan las cartas VETOED; el resto opera.          ║
║       Sin cubrimiento: una carta, H¹=0.  NO es un topos de Grothendieck.     ║
║                                                                              ║
║   (3) KMS finito-dimensional (si se inyectan ρ, H)                           ║
║       ρ_β = e^{−βH}/Z ,  β = 1/T                                             ║
║       D(ρ‖ρ_β) y  ‖log ρ + βH − c I‖_HS  (defecto modular).                  ║
║       F_Uhlmann(ρ,ρ_β) = ‖√ρ √ρ_β‖_1² .                                      ║
║       Sin ρ: kms_available=False.  NO hay álgebra de von Neumann MAC.        ║
║                                                                              ║
║ ARQUITECTURA: In --F1→ Phase1 --F2→ Phase2 --F3→ Phase3 → Certificate        ║
║ Crowbar/GPIO SIMULADO. No hay acceso a silicio.                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, Final, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

try:
    from thermal_gradient_laws import ThermalGradientLaws
except ImportError:  # pragma: no cover
    try:
        from app.physics.thermal_gradient_laws import ThermalGradientLaws
    except ImportError:
        from physics.thermal_gradient_laws import ThermalGradientLaws

logger = logging.getLogger("APU.Agents.ThermalGradientAgent")

# ---------------------------------------------------------------------------
# Constantes metrológicas
# ---------------------------------------------------------------------------
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_LIMIT: Final[float] = 1e-15
_WILKINSON_FLOOR: Final[float] = _WILKINSON_LIMIT

_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_CROWBAR_LATENCY_FLOOR_NS: Final[float] = 380.0
_CROWBAR_LATENCY_CEIL_NS: Final[float] = 420.0

_DEFAULT_CD_THRESHOLD: Final[float] = -1e-4
_DEFAULT_EXERGY_THRESHOLD: Final[float] = 0.4
_DEGRADED_EXERGY_THRESHOLD: Final[float] = 0.6
_FOURIER_VETO: Final[float] = 1e-3
_FOURIER_DEGRADED: Final[float] = 1e-6
_CSMD_VETO: Final[float] = 1e-8
_LANDAUER_ABS_TOL: Final[float] = 100.0 * _MACHINE_EPS
_CONDITION_VETO: Final[float] = 1e12
_CONDITION_DEGRADED: Final[float] = 1e8

_DEFAULT_ALPHA: Final[float] = 0.5
_DEFAULT_HISTORY: Final[int] = 32
_DEFAULT_DT: Final[float] = 1.0
_FRAC_SECULAR_TAU: Final[float] = 1e-3
_KMS_VETO: Final[float] = 1e-2
_KMS_DEGRADED: Final[float] = 1e-4
_CECH_RANK_GAP: Final[int] = 2
_PSD_TOLERANCE: Final[float] = 1e-10
_DENSITY_TRACE_TOLERANCE: Final[float] = 1e-10


# =============================================================================
# Retículo de Heyting
# =============================================================================
class HeytingVerdict(str, Enum):
    r"""
    Cadena  ⊥ = VETOED ≺ DEGRADED ≺ COHERENT ≺ CERTIFIED = ⊤.

    El «join de seguridad» (peor gana) es el meet de verdad.
    CERTIFIED es ⊤, no un token desconocido.
    """

    VETOED = "VETOED"
    DEGRADED = "DEGRADED"
    COHERENT = "COHERENT"
    CERTIFIED = "CERTIFIED"

    @property
    def rank(self) -> int:
        return {
            HeytingVerdict.VETOED: 0,
            HeytingVerdict.DEGRADED: 1,
            HeytingVerdict.COHERENT: 2,
            HeytingVerdict.CERTIFIED: 3,
        }[self]

    def meet(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return self if self.rank <= other.rank else other

    def join(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return self if self.rank >= other.rank else other

    def implies(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict.CERTIFIED if self.rank <= other.rank else other

    def negate(self) -> "HeytingVerdict":
        return self.implies(HeytingVerdict.VETOED)

    @classmethod
    def parse(cls, token: Any, *, unknown_as_veto: bool = True) -> "HeytingVerdict":
        normalized = str(token).strip().upper()
        for member in cls:
            if member.value == normalized:
                return member
        if unknown_as_veto and normalized:
            return cls.VETOED
        return cls.COHERENT if not normalized else cls.VETOED


# =============================================================================
# Contratos inmutables
# =============================================================================
def _freeze_array(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, copy=True)
    out.setflags(write=False)
    return out


def _logsumexp(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64).ravel()
    if data.size == 0:
        return -np.inf
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return -np.inf
    shift = float(np.max(finite))
    return shift + float(np.log(np.sum(np.exp(finite - shift))))


@dataclass(frozen=True, slots=True)
class EngineCertificateAudit:
    available: bool
    engine_verdict: str
    engine_interlock_fired: bool
    fourier_residual: float
    csmd_error: float
    landauer_gap: float
    gouy_stodola: float
    conservation_residual: float
    thermal_entropy_production: float
    dirichlet_energy: float
    carnot_density: float
    T_hot: float
    T_cold: float
    pairing_q_dT: float
    metric_condition: float
    third_law_flag: bool
    temperature_clamped: bool
    engine_veto_reasons: Tuple[str, ...]
    engine_degraded_reasons: Tuple[str, ...]
    missing_keys: Tuple[str, ...]
    adaptive_tau_cd: float = float("nan")
    tellegen_residual: float = float("nan")


@dataclass(frozen=True, slots=True)
class FractionalMemoryChart:
    r"""Carta de memoria no-markoviana (Caputo / Grünwald–Letnikov / RL)."""

    alpha: float
    dt: float
    window: int
    samples: int
    gl_phi: float
    gl_temperature: float
    rl_integral_phi: float
    secular_leak: bool
    transient_flag: bool
    kernel_l1: float


@dataclass(frozen=True, slots=True)
class ChartSection:
    r"""Sección local Γ(U, ℋ) sobre una carta de coordenadas."""

    name: str
    indices: Tuple[int, ...]
    pairing: float
    phi_local: float
    dirichlet_mass: float
    verdict: str
    rank: int


@dataclass(frozen=True, slots=True)
class SheafCohomologyChart:
    r"""
    Čech H¹ de un cubrimiento finito con valores en la cadena de Heyting.

    `obstructed` ⇔ existe solape con |Δ rank| ≥ 2 (no pegable).
    `isolated_charts` = cartas cuyo veredicto local es VETOED.
    """

    cover_names: Tuple[str, ...]
    sections: Tuple[ChartSection, ...]
    overlap_disagreements: Tuple[Tuple[str, str, str, str], ...]
    cech_h1_obstructed: bool
    isolated_charts: Tuple[str, ...]
    surgical_possible: bool
    global_meet: str


@dataclass(frozen=True, slots=True)
class KMSChart:
    r"""Defecto KMS / modular a β = 1/T sobre M_n(ℂ)."""

    available: bool
    is_density: bool
    inverse_temperature: float
    relative_entropy: float
    modular_defect: float
    uhlmann_fidelity: float
    von_neumann_entropy: float
    kms_coherent: bool


@dataclass(frozen=True, slots=True)
class Phase1ThermalObservation:
    r"""Objeto terminal de la Fase 1 ≡ objeto inicial de la Fase 2."""

    clausius_duhem_residual: float
    carnot_efficiency: float
    exergy_potential: float
    heat_flux_norm: float
    heat_flux_energy: float
    temperature_system: float
    heat_flux_vector: np.ndarray
    grad_T: np.ndarray
    entropy_production_rate: float
    engine_stamp: Dict[str, Any]
    observation_valid: bool
    engine_alive: bool
    certificate_audit: EngineCertificateAudit
    history_length: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "heat_flux_vector", _freeze_array(self.heat_flux_vector))
        object.__setattr__(self, "grad_T", _freeze_array(self.grad_T))


@dataclass(frozen=True, slots=True)
class Phase2ThermalOrientation:
    r"""Objeto terminal de la Fase 2 ≡ segundo factor del dominio de la Fase 3."""

    is_cd_coherent: bool
    is_exergy_sufficient: bool
    is_fourier_coherent: bool
    is_csmd_coherent: bool
    is_landauer_coherent: bool
    is_temperature_physical: bool
    is_carnot_in_range: bool
    is_engine_alive: bool
    is_certificate_internally_consistent: bool
    is_fractional_secular: bool
    is_fractional_transient: bool
    is_sheaf_obstructed: bool
    is_kms_coherent: bool
    cd_margin: float
    exergy_margin: float
    fourier_margin: float
    landauer_margin: float
    physics_score: float
    policy_score: float
    fractional: Optional[FractionalMemoryChart]
    sheaf: Optional[SheafCohomologyChart]
    kms: Optional[KMSChart]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Phase3ThermalDecision:
    heyting_verdict: str
    heyting_rank: int
    heyting_score: float
    veto_reasons: Tuple[str, ...]
    degraded_reasons: Tuple[str, ...]
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    engine_interlock_fired: bool
    dual_channel_actuation: bool
    supervisor_disagreement: bool
    conservation_residual: float
    isolated_charts: Tuple[str, ...]
    surgical_veto: bool
    cech_h1_obstructed: bool
    fractional_cd_accumulator: float
    kms_defect: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ThermalGradientCertificate:
    phase: str
    heyting_verdict: str
    clausius_duhem_residual: float
    carnot_efficiency: float
    exergy_potential: float
    heat_flux_norm: float
    temperature_system: float
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    veto_reasons: Tuple[str, ...] = ()
    degraded_reasons: Tuple[str, ...] = ()
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    heyting_score: float = 0.0
    conservation_residual: float = 0.0
    fourier_residual: float = 0.0
    csmd_error: float = 0.0
    landauer_gap: float = 0.0
    gouy_stodola: float = 0.0
    observation_valid: bool = True
    engine_heyting_verdict: str = ""
    policy_exergy_sufficient: bool = True
    supervisor_disagreement: bool = False
    fractional_cd_accumulator: float = 0.0
    cech_h1_obstructed: bool = False
    isolated_charts: Tuple[str, ...] = ()
    surgical_veto: bool = False
    kms_defect: float = 0.0
    uhlmann_fidelity: float = 1.0


# =============================================================================
# FASE 1 — OBSERVE
# =============================================================================
class Phase1ThermalObservationMixin:
    r"""
    FASE 1 — OBSERVE.

    Invoca el motor, extrae el sello, empuja (Φ, T, |dT|²) a la memoria
    fraccional. Codominio: `Phase1ThermalObservation`.
    """

    def __init__(
        self,
        dimension_n: int,
        safety_margin: float = 1.0,
        cd_threshold: float = _DEFAULT_CD_THRESHOLD,
        exergy_threshold: float = _DEFAULT_EXERGY_THRESHOLD,
        *,
        rng_seed: Optional[int] = None,
        exergy_is_hard_veto: bool = False,
        fractional_alpha: float = _DEFAULT_ALPHA,
        history_window: int = _DEFAULT_HISTORY,
        time_step: float = _DEFAULT_DT,
        coordinate_cover: Optional[Mapping[str, Sequence[int]]] = None,
    ) -> None:
        self._n = self._validate_positive_int("dimension_n", dimension_n)
        self._safety_margin = self._validate_positive_finite("safety_margin", safety_margin)

        cd_threshold = self._validate_finite("cd_threshold", cd_threshold)
        exergy_threshold = self._validate_finite("exergy_threshold", exergy_threshold)
        if exergy_threshold < 0.0 or exergy_threshold > 1.0:
            raise ValueError("exergy_threshold debe estar en el intervalo [0, 1].")

        self._cd_thresh = float(cd_threshold * self._safety_margin)
        self._exergy_thresh = float(exergy_threshold / self._safety_margin)
        self._exergy_is_hard_veto = bool(exergy_is_hard_veto)

        alpha = self._validate_finite("fractional_alpha", fractional_alpha)
        if not (0.0 < alpha < 1.0):
            raise ValueError("fractional_alpha debe vivir en (0, 1).")
        self._alpha = float(alpha)
        self._history_window = int(max(2, history_window))
        self._dt = self._validate_positive_finite("time_step", time_step)
        self._cover = self._normalize_cover(coordinate_cover)

        self._engine = ThermalGradientLaws(
            dimension_n=self._n,
            safety_margin=self._safety_margin,
            rng_seed=rng_seed,
        )
        self._rng = np.random.default_rng(rng_seed)
        self._interlock_lock = threading.Lock()
        self._interlock_state = False
        self._history: Deque[Tuple[float, float, float]] = deque(maxlen=self._history_window)

    def _normalize_cover(
        self,
        cover: Optional[Mapping[str, Sequence[int]]],
    ) -> Dict[str, Tuple[int, ...]]:
        if not cover:
            return {}
        out: Dict[str, Tuple[int, ...]] = {}
        for name, idxs in cover.items():
            unique = tuple(sorted({int(i) for i in idxs if 0 <= int(i) < self._n}))
            if unique:
                out[str(name)] = unique
        return out

    @staticmethod
    def _validate_positive_int(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} debe ser un entero.")
        if value <= 0:
            raise ValueError(f"{name} debe ser estrictamente mayor que cero.")
        return int(value)

    @staticmethod
    def _validate_positive_finite(name: str, value: Any) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} no debe ser booleano.")
        try:
            value_f = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} debe ser numérico.") from exc
        if not math.isfinite(value_f) or value_f <= 0.0:
            raise ValueError(f"{name} debe ser finito y estrictamente mayor que cero.")
        return value_f

    @staticmethod
    def _validate_finite(name: str, value: Any) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} no debe ser booleano.")
        try:
            value_f = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} debe ser numérico.") from exc
        if not math.isfinite(value_f):
            raise ValueError(f"{name} debe ser finito.")
        return value_f

    def _neumaier_sum(self, arr: np.ndarray) -> complex:
        data = np.asarray(arr).ravel()
        if data.size == 0:
            return 0.0
        if not np.all(np.isfinite(data)):
            raise ValueError("kahan/neumaier: el sumando contiene valores no finitos")
        use_complex = np.iscomplexobj(data)
        total: complex = 0.0j if use_complex else 0.0
        compensator: complex = 0.0j if use_complex else 0.0
        for x in data:
            t = total + x
            if abs(total) >= abs(x):
                compensator += (total - t) + x
            else:
                compensator += (x - t) + total
            total = t
        return total + compensator

    def kahan_sum(self, arr: np.ndarray) -> float:
        vec = np.asarray(arr, dtype=np.float64).ravel()
        return float(np.real(self._neumaier_sum(vec)))

    @staticmethod
    def _extract_float(mapping: Dict[str, Any], key: str, default: float) -> Tuple[float, bool]:
        if key not in mapping:
            return default, False
        try:
            return float(mapping[key]), True
        except (TypeError, ValueError):
            return default, False

    def _extract_vector(self, mapping: Dict[str, Any], key: str) -> Tuple[np.ndarray, bool]:
        default = np.zeros(self._n, dtype=np.float64)
        raw = mapping.get(key, default)
        try:
            arr = np.asarray(raw, dtype=np.float64).ravel()
        except (TypeError, ValueError):
            return default.copy(), False
        if arr.size != self._n or not np.all(np.isfinite(arr)):
            return default.copy(), False
        return arr, True

    def _extract_string_tuple(self, mapping: Dict[str, Any], key: str) -> Tuple[str, ...]:
        raw = mapping.get(key, ())
        if raw is None:
            return ()
        if isinstance(raw, str):
            return (raw,)
        try:
            return tuple(str(item) for item in raw)
        except TypeError:
            return ()

    def _audit_engine_certificate(
        self,
        stamp: Dict[str, Any],
        diagnostics: Dict[str, Any],
    ) -> EngineCertificateAudit:
        missing: List[str] = []

        def grab(key: str, default: float) -> float:
            value, present = self._extract_float(stamp, key, default)
            if not present:
                missing.append(key)
            return value

        engine_verdict = str(stamp.get("heyting_verdict", "")).strip().upper()
        engine_interlock = bool(stamp.get("hardware_interlock_fired", False))
        nested = stamp.get("diagnostics")
        nested_p1: Dict[str, Any] = {}
        nested_p2: Dict[str, Any] = {}
        if isinstance(nested, dict):
            p1 = nested.get("phase1")
            p2 = nested.get("phase2")
            if isinstance(p1, dict):
                nested_p1 = p1
            if isinstance(p2, dict):
                nested_p2 = p2

        pairing = float(nested_p2.get("q_dot_grad_T", float("nan")))
        condition = float(nested_p2.get("metric_condition", float("nan")))
        third_law = bool(nested_p1.get("third_law_flag", False))
        t_clamped = bool(nested_p1.get("temperature_clamped", False))

        return EngineCertificateAudit(
            available=bool(stamp),
            engine_verdict=engine_verdict,
            engine_interlock_fired=engine_interlock,
            fourier_residual=grab("fourier_residual", float("nan")),
            csmd_error=grab("csmd_error", float("nan")),
            landauer_gap=grab("landauer_gap", float("nan")),
            gouy_stodola=grab("gouy_stodola", float("nan")),
            conservation_residual=grab("conservation_residual", float("nan")),
            thermal_entropy_production=grab("thermal_entropy_production", float("nan")),
            dirichlet_energy=grab("dirichlet_energy", float("nan")),
            carnot_density=grab("carnot_density", float("nan")),
            T_hot=grab("T_hot", float("nan")),
            T_cold=grab("T_cold", float("nan")),
            pairing_q_dT=pairing,
            metric_condition=condition,
            third_law_flag=third_law,
            temperature_clamped=t_clamped,
            engine_veto_reasons=self._extract_string_tuple(stamp, "veto_reasons"),
            engine_degraded_reasons=self._extract_string_tuple(stamp, "degraded_reasons"),
            missing_keys=tuple(missing),
            adaptive_tau_cd=grab("adaptive_tau_cd", float("nan")),
            tellegen_residual=grab("tellegen_residual", float("nan")),
        )

    def _fail_safe_observation(
        self,
        diagnostics: Dict[str, Any],
        grad_T: np.ndarray,
        entropy_production_rate: float,
    ) -> Phase1ThermalObservation:
        empty_q = np.zeros(self._n, dtype=np.float64)
        audit = EngineCertificateAudit(
            available=False,
            engine_verdict="VETOED",
            engine_interlock_fired=False,
            fourier_residual=float("nan"),
            csmd_error=float("nan"),
            landauer_gap=float("nan"),
            gouy_stodola=float("nan"),
            conservation_residual=float("nan"),
            thermal_entropy_production=float("nan"),
            dirichlet_energy=float("nan"),
            carnot_density=float("nan"),
            T_hot=float("nan"),
            T_cold=float("nan"),
            pairing_q_dT=float("nan"),
            metric_condition=float("nan"),
            third_law_flag=False,
            temperature_clamped=False,
            engine_veto_reasons=("engine_invocation_failed",),
            engine_degraded_reasons=(),
            missing_keys=(),
        )
        return Phase1ThermalObservation(
            clausius_duhem_residual=float("nan"),
            carnot_efficiency=float("nan"),
            exergy_potential=float("nan"),
            heat_flux_norm=0.0,
            heat_flux_energy=0.0,
            temperature_system=float("nan"),
            heat_flux_vector=empty_q,
            grad_T=grad_T,
            entropy_production_rate=entropy_production_rate,
            engine_stamp={},
            observation_valid=False,
            engine_alive=False,
            certificate_audit=audit,
            history_length=len(self._history),
            diagnostics=diagnostics,
        )

    def _push_history(self, phi: float, temperature: float, dirichlet: float) -> int:
        with self._interlock_lock:
            self._history.append(
                (
                    float(phi) if math.isfinite(phi) else 0.0,
                    float(temperature) if math.isfinite(temperature) else 0.0,
                    float(max(dirichlet, 0.0)) if math.isfinite(dirichlet) else 0.0,
                )
            )
            return len(self._history)

    def reset_fractional_memory(self) -> None:
        """Purga la ventana de Caputo (p. ej. tras un rearranque de misión)."""
        with self._interlock_lock:
            self._history.clear()

    def phase1_observe_engine_stamp(
        self,
        K_raw: np.ndarray,
        grad_T_raw: np.ndarray,
        T_sys: float,
        metric_tensor: np.ndarray,
        entropy_production_rate: float,
        **engine_kwargs: Any,
    ) -> Phase1ThermalObservation:
        r"""
        [FASE 1 — ÚLTIMO MORFISMO: OBSERVE]

        Codominio: `Phase1ThermalObservation`
        = dominio de `phase2_orient_from_observation`.
        """
        logger.info("Fase Observe: sello del motor + memoria fraccional.")
        diagnostics: Dict[str, Any] = {
            "observe_phase": "engine_invocation",
            "engine_invoked": True,
        }
        try:
            grad_T = np.asarray(grad_T_raw, dtype=np.float64).reshape(-1)
            if grad_T.size != self._n:
                grad_T = np.zeros(self._n, dtype=np.float64)
                diagnostics["grad_T_invalid"] = True
        except (TypeError, ValueError):
            grad_T = np.zeros(self._n, dtype=np.float64)
            diagnostics["grad_T_invalid"] = True

        sigma = 0.0
        try:
            sigma = float(entropy_production_rate)
            if not math.isfinite(sigma):
                sigma = 0.0
        except (TypeError, ValueError):
            sigma = 0.0

        try:
            stamp = self._engine.execute_thermal_cycle(
                K_raw=K_raw,
                grad_T_raw=grad_T_raw,
                T_sys=T_sys,
                metric_tensor=metric_tensor,
                entropy_production_rate=entropy_production_rate,
                **engine_kwargs,
            )
        except TypeError:
            try:
                stamp = self._engine.execute_thermal_cycle(
                    K_raw=K_raw,
                    grad_T_raw=grad_T_raw,
                    T_sys=T_sys,
                    metric_tensor=metric_tensor,
                    entropy_production_rate=entropy_production_rate,
                )
                diagnostics["engine_kwargs_dropped"] = tuple(engine_kwargs.keys())
            except Exception as exc:
                logger.exception("Motor térmico falló. Fail-safe.")
                diagnostics.update({"engine_error": str(exc), "engine_invoked": False})
                return self._fail_safe_observation(diagnostics, grad_T, sigma)
        except Exception as exc:
            logger.exception("Motor térmico falló. Fail-safe.")
            diagnostics.update({"engine_error": str(exc), "engine_invoked": False})
            return self._fail_safe_observation(diagnostics, grad_T, sigma)

        if not isinstance(stamp, dict):
            try:
                stamp = dict(stamp)
            except Exception:
                stamp = {}
                diagnostics["engine_stamp_invalid"] = True

        cd_residual, cd_present = self._extract_float(stamp, "clausius_duhem_residual", float("nan"))
        carnot_efficiency, eta_present = self._extract_float(stamp, "carnot_efficiency", float("nan"))
        exergy_potential, ex_present = self._extract_float(stamp, "exergy_potential", float("nan"))
        temperature_system, t_present = self._extract_float(stamp, "T_sys_clamped", float("nan"))
        heat_flux_vector, heat_flux_valid = self._extract_vector(stamp, "heat_flux_vector")
        dirichlet, _ = self._extract_float(stamp, "dirichlet_energy", float("nan"))

        if not eta_present and not math.isfinite(carnot_efficiency):
            carnot_efficiency = 0.0
        if not ex_present and not math.isfinite(exergy_potential):
            exergy_potential = 0.0

        if heat_flux_valid:
            energy = self.kahan_sum(np.asarray(heat_flux_vector, dtype=np.float64) ** 2)
            heat_flux_energy = float(max(energy, 0.0))
            heat_flux_norm = float(math.sqrt(heat_flux_energy))
            if not math.isfinite(heat_flux_norm):
                heat_flux_norm = 0.0
                heat_flux_energy = 0.0
        else:
            heat_flux_norm = 0.0
            heat_flux_energy = 0.0
            diagnostics["heat_flux_vector_invalid"] = True

        engine_diagnostics = stamp.get("diagnostics")
        if isinstance(engine_diagnostics, dict):
            diagnostics["engine"] = dict(engine_diagnostics)

        audit = self._audit_engine_certificate(stamp, diagnostics)
        hist_len = self._push_history(cd_residual, temperature_system, dirichlet)
        observation_valid = bool(
            heat_flux_valid and cd_present and math.isfinite(cd_residual)
            and t_present and math.isfinite(temperature_system)
        )
        diagnostics["observation_valid"] = observation_valid
        diagnostics["history_length"] = hist_len
        diagnostics["cover_names"] = tuple(self._cover.keys())

        return Phase1ThermalObservation(
            clausius_duhem_residual=cd_residual,
            carnot_efficiency=carnot_efficiency,
            exergy_potential=exergy_potential,
            heat_flux_norm=heat_flux_norm,
            heat_flux_energy=heat_flux_energy,
            temperature_system=temperature_system,
            heat_flux_vector=heat_flux_vector,
            grad_T=grad_T,
            entropy_production_rate=sigma,
            engine_stamp=dict(stamp),
            observation_valid=observation_valid,
            engine_alive=True,
            certificate_audit=audit,
            history_length=hist_len,
            diagnostics=diagnostics,
        )


# =============================================================================
# FASE 2 — ORIENT (Caputo + Čech + KMS)
# Dominio = Phase1ThermalObservation
# =============================================================================
class Phase2ThermalOrientationMixin(Phase1ThermalObservationMixin):
    r"""
    FASE 2 — ORIENT.

    Física instantánea (sello) + memoria fraccional + haz de cartas + KMS.
    Codominio: `Phase2ThermalOrientation`.
    """

    def verify_thermodynamic_coherence(
        self,
        cd_residual: float,
        exergy_potential: float,
    ) -> Tuple[bool, bool]:
        """[COMPATIBILIDAD 1.X — FASE ORIENT]"""
        try:
            cd_value = float(cd_residual)
        except (TypeError, ValueError):
            cd_value = float("nan")
        try:
            exergy_value = float(exergy_potential)
        except (TypeError, ValueError):
            exergy_value = float("nan")
        is_cd_coherent = math.isfinite(cd_value) and cd_value >= self._cd_thresh
        is_exergy_sufficient = math.isfinite(exergy_value) and exergy_value >= self._exergy_thresh
        return is_cd_coherent, is_exergy_sufficient

    @staticmethod
    def _gl_weights(alpha: float, n: int) -> np.ndarray:
        w = np.empty(n, dtype=np.float64)
        if n <= 0:
            return w
        w[0] = 1.0
        for j in range(1, n):
            w[j] = w[j - 1] * (j - 1.0 - alpha) / j
        return w

    @staticmethod
    def _rl_kernel(alpha: float, n: int, dt: float) -> np.ndarray:
        r"""Núcleo RL I^α: k_j = Δt^α (j+1)^{α−1} / Γ(α), j=0..n−1 (más reciente = 0)."""
        if n <= 0:
            return np.zeros(0, dtype=np.float64)
        g = math.gamma(alpha)
        k = np.empty(n, dtype=np.float64)
        scale = (dt ** alpha) / g
        for j in range(n):
            k[j] = scale * ((j + 1.0) ** (alpha - 1.0))
        return k

    def _fractional_memory_chart(self) -> FractionalMemoryChart:
        with self._interlock_lock:
            hist = tuple(self._history)
        n = len(hist)
        if n == 0:
            return FractionalMemoryChart(
                alpha=self._alpha, dt=self._dt, window=self._history_window, samples=0,
                gl_phi=0.0, gl_temperature=0.0, rl_integral_phi=0.0,
                secular_leak=False, transient_flag=False, kernel_l1=0.0,
            )
        phi = np.array([h[0] for h in hist], dtype=np.float64)
        temp = np.array([h[1] for h in hist], dtype=np.float64)
        # GL: el presente es el último sample; pesos w_j multiplican f_{n-1-j}.
        w = self._gl_weights(self._alpha, n)
        dt_a = self._dt ** (-self._alpha)
        gl_phi = float(dt_a * self.kahan_sum(w * phi[::-1]))
        gl_T = float(dt_a * self.kahan_sum(w * temp[::-1]))
        rl = self._rl_kernel(self._alpha, n, self._dt)
        rl_phi = float(self.kahan_sum(rl * phi[::-1]))
        kernel_l1 = float(self.kahan_sum(np.abs(rl)))
        # Fuga secular: la integral fraccional de Φ es persistentemente negativa.
        secular = bool(n >= 4 and rl_phi < -_FRAC_SECULAR_TAU * self._safety_margin)
        # Transitorio: |D^α T| grande frente a la escala de T, sin fuga de Φ.
        t_scale = float(max(abs(temp[-1]), 1.0))
        transient = bool(
            n >= 4
            and abs(gl_T) > 0.05 * t_scale * (self._dt ** (-self._alpha))
            and rl_phi >= -_FRAC_SECULAR_TAU * self._safety_margin
        )
        return FractionalMemoryChart(
            alpha=self._alpha,
            dt=self._dt,
            window=self._history_window,
            samples=n,
            gl_phi=gl_phi,
            gl_temperature=gl_T,
            rl_integral_phi=rl_phi,
            secular_leak=secular,
            transient_flag=transient,
            kernel_l1=kernel_l1,
        )

    def _local_phi(
        self,
        observation: Phase1ThermalObservation,
        indices: Tuple[int, ...],
    ) -> Tuple[float, float, float]:
        q = np.asarray(observation.heat_flux_vector, dtype=np.float64)
        g = np.asarray(observation.grad_T, dtype=np.float64)
        idx = np.array(indices, dtype=int)
        pairing = float(self.kahan_sum(q[idx] * g[idx]))
        mass = float(self.kahan_sum(g[idx] * g[idx]))
        mass_tot = float(self.kahan_sum(g * g))
        frac = mass / mass_tot if mass_tot > _WILKINSON_FLOOR else (len(indices) / max(self._n, 1))
        sigma_u = float(observation.entropy_production_rate) * frac
        T = observation.temperature_system
        t2 = T * T if math.isfinite(T) and T > 0.0 else 1.0
        phi_u = float(sigma_u - pairing / t2)
        return pairing, phi_u, mass

    def _local_verdict(self, phi_u: float, tau: float) -> HeytingVerdict:
        if not math.isfinite(phi_u):
            return HeytingVerdict.VETOED
        if phi_u < tau:
            return HeytingVerdict.VETOED
        if phi_u < 0.0:
            return HeytingVerdict.DEGRADED
        return HeytingVerdict.CERTIFIED

    def _sheaf_cech_chart(self, observation: Phase1ThermalObservation) -> SheafCohomologyChart:
        tau = self._cd_thresh
        if not self._cover:
            pairing = float(
                self.kahan_sum(
                    np.asarray(observation.heat_flux_vector) * np.asarray(observation.grad_T)
                )
            ) if observation.engine_alive else float("nan")
            section = ChartSection(
                name="global",
                indices=tuple(range(self._n)),
                pairing=pairing,
                phi_local=observation.clausius_duhem_residual,
                dirichlet_mass=float("nan"),
                verdict=HeytingVerdict.parse(observation.certificate_audit.engine_verdict or "COHERENT").value,
                rank=HeytingVerdict.parse(observation.certificate_audit.engine_verdict or "COHERENT").rank,
            )
            return SheafCohomologyChart(
                cover_names=("global",),
                sections=(section,),
                overlap_disagreements=(),
                cech_h1_obstructed=False,
                isolated_charts=(),
                surgical_possible=False,
                global_meet=section.verdict,
            )

        sections: List[ChartSection] = []
        for name, idxs in self._cover.items():
            pairing, phi_u, mass = self._local_phi(observation, idxs)
            v = self._local_verdict(phi_u, tau)
            sections.append(
                ChartSection(
                    name=name, indices=idxs, pairing=pairing, phi_local=phi_u,
                    dirichlet_mass=mass, verdict=v.value, rank=v.rank,
                )
            )

        disagreements: List[Tuple[str, str, str, str]] = []
        names = list(self._cover.keys())
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                overlap = tuple(sorted(set(self._cover[a]) & set(self._cover[b])))
                if not overlap:
                    continue
                sa = next(s for s in sections if s.name == a)
                sb = next(s for s in sections if s.name == b)
                if abs(sa.rank - sb.rank) >= _CECH_RANK_GAP:
                    disagreements.append((a, b, sa.verdict, sb.verdict))

        isolated = tuple(s.name for s in sections if s.verdict == HeytingVerdict.VETOED.value)
        meet = HeytingVerdict.CERTIFIED
        for s in sections:
            meet = meet.meet(HeytingVerdict.parse(s.verdict))
        obstructed = bool(disagreements)
        surgical = bool(isolated) and (len(isolated) < len(sections) or not obstructed)
        return SheafCohomologyChart(
            cover_names=tuple(names),
            sections=tuple(sections),
            overlap_disagreements=tuple(disagreements),
            cech_h1_obstructed=obstructed,
            isolated_charts=isolated,
            surgical_possible=surgical,
            global_meet=meet.value,
        )

    def _kms_chart(
        self,
        rho: Optional[np.ndarray],
        hamiltonian: Optional[np.ndarray],
        temperature: float,
    ) -> KMSChart:
        if rho is None or hamiltonian is None:
            return KMSChart(
                available=False, is_density=False, inverse_temperature=float("nan"),
                relative_entropy=float("nan"), modular_defect=float("nan"),
                uhlmann_fidelity=float("nan"), von_neumann_entropy=float("nan"),
                kms_coherent=True,
            )
        rho_h = 0.5 * (np.asarray(rho, dtype=np.complex128) + np.asarray(rho, dtype=np.complex128).conj().T)
        H_h = 0.5 * (np.asarray(hamiltonian, dtype=np.complex128) + np.asarray(hamiltonian, dtype=np.complex128).conj().T)
        if rho_h.shape != (self._n, self._n) or H_h.shape != (self._n, self._n):
            return KMSChart(
                available=True, is_density=False, inverse_temperature=float("nan"),
                relative_entropy=float("inf"), modular_defect=float("inf"),
                uhlmann_fidelity=0.0, von_neumann_entropy=float("nan"),
                kms_coherent=False,
            )
        evals, evecs = la.eigh(rho_h)
        evals = np.real(evals)
        trace = float(np.real(np.sum(evals)))
        min_eig = float(np.min(evals))
        is_density = min_eig >= -_PSD_TOLERANCE and abs(trace - 1.0) <= _DENSITY_TRACE_TOLERANCE
        acc_s: List[float] = []
        for eta in evals:
            clipped = max(float(eta), 0.0)
            if clipped > _WILKINSON_FLOOR:
                acc_s.append(float(-clipped * np.log(clipped)))
        svn = float(self.kahan_sum(np.asarray(acc_s, dtype=np.float64))) if acc_s else 0.0
        T = float(temperature) if math.isfinite(temperature) and temperature > 0.0 else float("nan")
        if not math.isfinite(T):
            return KMSChart(
                available=True, is_density=is_density, inverse_temperature=float("nan"),
                relative_entropy=float("inf"), modular_defect=float("inf"),
                uhlmann_fidelity=0.0, von_neumann_entropy=svn, kms_coherent=False,
            )
        beta = 1.0 / T
        energy = float(np.real(np.trace(rho_h @ H_h)))
        h_eigs = np.real(la.eigvalsh(H_h))
        log_z = _logsumexp(-beta * h_eigs)
        relative = max(0.0, float(-svn + beta * energy + log_z))
        # log ρ = V log(η)_+ V† ;  c = −mean(log η + β λ_H) en el soporte.
        log_eta = np.log(np.maximum(evals, _WILKINSON_FLOOR))
        log_rho = (evecs * log_eta) @ evecs.conj().T
        target = -beta * H_h
        residual = log_rho - target
        residual = residual - (np.trace(residual) / self._n) * np.eye(self._n)
        modular = float(la.norm(residual, ord="fro"))
        # Gibbs para fidelidad de Uhlmann.
        gibbs_e = np.exp(-beta * h_eigs - log_z)
        # ρ_β comparte la eigenbasis de H, no necesariamente la de ρ.
        H_vecs = la.eigh(H_h)[1]
        rho_beta = (H_vecs * gibbs_e) @ H_vecs.conj().T
        try:
            sqrt_rho = evecs * np.sqrt(np.maximum(evals, 0.0)) @ evecs.conj().T
            inner = sqrt_rho @ rho_beta @ sqrt_rho
            fid_eigs = np.real(la.eigvalsh(inner))
            fidelity = float(np.real(np.sum(np.sqrt(np.maximum(fid_eigs, 0.0)))) ** 2)
            fidelity = float(np.clip(fidelity, 0.0, 1.0))
        except la.LinAlgError:
            fidelity = 0.0
        coherent = bool(is_density and relative <= _KMS_VETO and modular <= 1.0)
        return KMSChart(
            available=True,
            is_density=is_density,
            inverse_temperature=beta,
            relative_entropy=relative,
            modular_defect=modular,
            uhlmann_fidelity=fidelity,
            von_neumann_entropy=svn,
            kms_coherent=coherent,
        )

    def _phase2_audit_engine_certificate(
        self,
        observation: Phase1ThermalObservation,
    ) -> Dict[str, Any]:
        audit = observation.certificate_audit
        flags: Dict[str, Any] = {}
        eta = observation.carnot_efficiency
        flags["carnot_in_range"] = math.isfinite(eta) and 0.0 <= eta <= 1.0 + 1e-12
        T = observation.temperature_system
        flags["temperature_physical"] = math.isfinite(T) and T > 0.0 and not audit.third_law_flag
        T_h, T_c = audit.T_hot, audit.T_cold
        if math.isfinite(T_h) and math.isfinite(T_c):
            flags["carnot_reservoirs_ordered"] = bool(T_h + 1e-12 >= T_c > 0.0)
            if math.isfinite(eta) and T_h > T_c > 0.0:
                flags["carnot_identity_residual"] = float(abs(eta - (1.0 - T_c / T_h)))
            else:
                flags["carnot_identity_residual"] = 0.0
        else:
            flags["carnot_reservoirs_ordered"] = True
            flags["carnot_identity_residual"] = 0.0
        pairing = audit.pairing_q_dT
        flags["anti_fourier"] = bool(math.isfinite(pairing) and pairing > _WILKINSON_FLOOR)
        flags["fourier_coherent"] = (
            True if not math.isfinite(audit.fourier_residual)
            else audit.fourier_residual <= _FOURIER_VETO * self._safety_margin
        )
        flags["csmd_coherent"] = (
            True if not math.isfinite(audit.csmd_error) else audit.csmd_error <= _CSMD_VETO
        )
        flags["landauer_coherent"] = (
            True if not math.isfinite(audit.landauer_gap)
            else audit.landauer_gap >= -_LANDAUER_ABS_TOL
        )
        flags["engine_verdict_known"] = audit.engine_verdict in {
            "VETOED", "DEGRADED", "COHERENT", "CERTIFIED", "",
        }
        return flags

    def phase2_orient_from_observation(
        self,
        observation: Phase1ThermalObservation,
        *,
        density_rho: Optional[np.ndarray] = None,
        hamiltonian_H: Optional[np.ndarray] = None,
    ) -> Phase2ThermalOrientation:
        r"""
        [FASE 2 — ÚLTIMO MORFISMO: ORIENT]

        Dominio : `Phase1ThermalObservation`
        Codominio: `Phase2ThermalOrientation`
        = segundo factor de `phase3_decide_from_orientation`.
        """
        if not isinstance(observation, Phase1ThermalObservation):
            raise TypeError("observation debe ser Phase1ThermalObservation.")

        is_cd_coherent, is_exergy_sufficient = self.verify_thermodynamic_coherence(
            cd_residual=observation.clausius_duhem_residual,
            exergy_potential=observation.exergy_potential,
        )
        flags = self._phase2_audit_engine_certificate(observation)
        audit = observation.certificate_audit
        frac = self._fractional_memory_chart()
        sheaf = self._sheaf_cech_chart(observation)
        kms = self._kms_chart(density_rho, hamiltonian_H, observation.temperature_system)

        cd_value = observation.clausius_duhem_residual
        exergy_value = observation.exergy_potential
        cd_margin = float(cd_value - self._cd_thresh) if math.isfinite(cd_value) else float("-inf")
        exergy_margin = (
            float(exergy_value - self._exergy_thresh) if math.isfinite(exergy_value) else float("-inf")
        )
        fourier_margin = (
            float(_FOURIER_VETO * self._safety_margin - audit.fourier_residual)
            if math.isfinite(audit.fourier_residual) else float("inf")
        )
        landauer_margin = float(audit.landauer_gap) if math.isfinite(audit.landauer_gap) else float("inf")

        is_fourier = bool(flags["fourier_coherent"]) and not flags["anti_fourier"]
        is_csmd = bool(flags["csmd_coherent"])
        is_landauer = bool(flags["landauer_coherent"])
        is_temp = bool(flags["temperature_physical"]) and observation.engine_alive
        is_carnot = bool(flags["carnot_in_range"]) and bool(flags["carnot_reservoirs_ordered"])
        identity_ok = float(flags["carnot_identity_residual"]) <= 1e-8
        cert_ok = (
            observation.observation_valid and observation.engine_alive
            and bool(flags["engine_verdict_known"]) and identity_ok
        )

        physics_terms = [
            1.0 if is_cd_coherent else 0.0,
            1.0 if is_fourier else 0.0,
            1.0 if is_csmd else 0.0,
            1.0 if is_landauer else 0.0,
            1.0 if is_temp else 0.0,
            1.0 if is_carnot else 0.0,
            1.0 if observation.engine_alive else 0.0,
            1.0 if observation.observation_valid else 0.0,
            0.0 if frac.secular_leak else 1.0,
            0.0 if (kms.available and not kms.kms_coherent) else 1.0,
        ]
        physics_score = float(min(physics_terms))
        if math.isfinite(audit.fourier_residual) and audit.fourier_residual > 0.0:
            physics_score = min(
                physics_score,
                float(np.exp(-audit.fourier_residual / max(_FOURIER_DEGRADED, _WILKINSON_FLOOR))),
            )
        policy_score = (
            1.0 if is_exergy_sufficient
            else float(np.clip(
                observation.exergy_potential / max(self._exergy_thresh, _WILKINSON_FLOOR), 0.0, 0.99,
            ))
            if math.isfinite(observation.exergy_potential) else 0.0
        )

        diagnostics: Dict[str, Any] = {
            "cd_threshold": self._cd_thresh,
            "exergy_threshold": self._exergy_thresh,
            "exergy_is_hard_veto": self._exergy_is_hard_veto,
            "cd_margin": cd_margin,
            "exergy_margin": exergy_margin,
            "fourier_margin": fourier_margin,
            "landauer_margin": landauer_margin,
            "is_cd_coherent": is_cd_coherent,
            "is_exergy_sufficient": is_exergy_sufficient,
            "certificate_flags": flags,
            "physics_score": physics_score,
            "policy_score": policy_score,
            "fractional_alpha": self._alpha,
            "rl_integral_phi": frac.rl_integral_phi,
            "gl_phi": frac.gl_phi,
            "cech_h1_obstructed": sheaf.cech_h1_obstructed,
            "isolated_charts": sheaf.isolated_charts,
            "kms_available": kms.available,
            "kms_relative_entropy": kms.relative_entropy,
        }
        return Phase2ThermalOrientation(
            is_cd_coherent=is_cd_coherent,
            is_exergy_sufficient=is_exergy_sufficient,
            is_fourier_coherent=is_fourier,
            is_csmd_coherent=is_csmd,
            is_landauer_coherent=is_landauer,
            is_temperature_physical=is_temp,
            is_carnot_in_range=is_carnot,
            is_engine_alive=observation.engine_alive,
            is_certificate_internally_consistent=cert_ok,
            is_fractional_secular=frac.secular_leak,
            is_fractional_transient=frac.transient_flag,
            is_sheaf_obstructed=sheaf.cech_h1_obstructed,
            is_kms_coherent=kms.kms_coherent,
            cd_margin=cd_margin,
            exergy_margin=exergy_margin,
            fourier_margin=fourier_margin,
            landauer_margin=landauer_margin,
            physics_score=physics_score,
            policy_score=policy_score,
            fractional=frac,
            sheaf=sheaf,
            kms=kms,
            diagnostics=diagnostics,
        )


# =============================================================================
# FASE 3 — DECIDE / ACT
# Dominio = Phase1 × Phase2
# =============================================================================
class Phase3ThermalDecisionActuationMixin(Phase2ThermalOrientationMixin):
    r"""
    FASE 3 — DECIDE / ACT.

    Meet de Heyting (agente ∧ motor ∧ Caputo ∧ Čech ∧ KMS).
    Crowbar SIMULADO; veto quirúrgico si el haz lo permite.
    """

    @staticmethod
    def _join_heyting_verdicts(verdicts: Tuple[str, ...]) -> str:
        acc = HeytingVerdict.CERTIFIED
        saw = False
        for token in verdicts:
            normalized = str(token).strip().upper()
            if not normalized:
                continue
            saw = True
            acc = acc.meet(HeytingVerdict.parse(normalized, unknown_as_veto=True))
        return (acc if saw else HeytingVerdict.COHERENT).value

    def _classify_heyting(
        self,
        cd_residual: float,
        exergy_potential: float,
        is_cd_coherent: bool,
        is_exergy_sufficient: bool,
    ) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
        veto_reasons: List[str] = []
        degraded_reasons: List[str] = []
        cd_finite = math.isfinite(cd_residual)
        exergy_finite = math.isfinite(exergy_potential)
        if not cd_finite:
            veto_reasons.append("clausius_duhem_residual_nonfinite")
        if not is_cd_coherent:
            veto_reasons.append("clausius_duhem_residual_below_threshold")
        if cd_finite and cd_residual < 0.0:
            degraded_reasons.append("clausius_duhem_residual_negative")
        if not exergy_finite:
            degraded_reasons.append("exergy_potential_nonfinite")
        elif not is_exergy_sufficient:
            if self._exergy_is_hard_veto:
                veto_reasons.append("exergy_potential_below_threshold")
            else:
                degraded_reasons.append("exergy_potential_below_policy_threshold")
        if exergy_finite and exergy_potential < _DEGRADED_EXERGY_THRESHOLD:
            degraded_reasons.append("exergy_potential_below_degraded_threshold")
        if veto_reasons:
            verdict = HeytingVerdict.VETOED.value
        elif degraded_reasons:
            verdict = HeytingVerdict.DEGRADED.value
        else:
            verdict = HeytingVerdict.COHERENT.value
        return verdict, tuple(veto_reasons), tuple(degraded_reasons)

    def _classify_heyting_supervised(
        self,
        observation: Phase1ThermalObservation,
        orientation: Phase2ThermalOrientation,
    ) -> Tuple[HeytingVerdict, float, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...], bool]:
        veto: List[str] = []
        degraded: List[str] = []
        audit = observation.certificate_audit
        frac = orientation.fractional
        sheaf = orientation.sheaf
        kms = orientation.kms

        if not observation.engine_alive:
            veto.append("engine_invocation_failed")
        if not observation.observation_valid:
            veto.append("observation_invalid")
        if not orientation.is_cd_coherent:
            veto.append("clausius_duhem_residual_below_threshold")
        if not math.isfinite(observation.clausius_duhem_residual):
            veto.append("clausius_duhem_residual_nonfinite")
        if not orientation.is_temperature_physical:
            veto.append("temperature_nonphysical")
        if audit.third_law_flag:
            veto.append("third_law_nonpositive_temperature")
        if not orientation.is_fourier_coherent:
            veto.append("fourier_constitutive_broken")
        if not orientation.is_csmd_coherent:
            veto.append("csmd_identity_broken")
        if not orientation.is_landauer_coherent:
            veto.append("landauer_bound_violated")
        if not orientation.is_carnot_in_range:
            veto.append("carnot_efficiency_out_of_physical_range")
        if math.isfinite(audit.metric_condition) and audit.metric_condition > _CONDITION_VETO:
            veto.append("metric_condition_explosive")
        if orientation.is_fractional_secular:
            veto.append("caputo_secular_leak")
        if kms is not None and kms.available and not kms.is_density:
            veto.append("kms_density_axioms_violated")
        if kms is not None and kms.available and kms.relative_entropy > _KMS_VETO:
            veto.append("kms_relative_entropy_excessive")

        if (
            math.isfinite(observation.clausius_duhem_residual)
            and observation.clausius_duhem_residual < 0.0
            and orientation.is_cd_coherent
        ):
            degraded.append("clausius_duhem_residual_negative_numerical")
        if not orientation.is_exergy_sufficient:
            if self._exergy_is_hard_veto:
                veto.append("exergy_potential_below_threshold")
            else:
                degraded.append("exergy_potential_below_policy_threshold")
        if math.isfinite(observation.exergy_potential) and observation.exergy_potential < _DEGRADED_EXERGY_THRESHOLD:
            degraded.append("exergy_potential_below_degraded_threshold")
        if audit.temperature_clamped and not audit.third_law_flag:
            degraded.append("temperature_clamped_to_wilkinson_floor")
        if math.isfinite(audit.fourier_residual) and _FOURIER_DEGRADED < audit.fourier_residual <= _FOURIER_VETO * self._safety_margin:
            degraded.append("fourier_residual_degraded")
        if math.isfinite(audit.metric_condition) and _CONDITION_DEGRADED < audit.metric_condition <= _CONDITION_VETO:
            degraded.append("metric_ill_conditioned")
        if not orientation.is_certificate_internally_consistent:
            degraded.append("certificate_internal_inconsistency")
        if orientation.is_fractional_transient:
            degraded.append("caputo_high_frequency_transient")
        if kms is not None and kms.available and _KMS_DEGRADED < kms.relative_entropy <= _KMS_VETO:
            degraded.append("kms_relative_entropy_degraded")
        if audit.engine_verdict == "DEGRADED":
            degraded.append("engine_heyting_degraded")
        if audit.engine_verdict == "VETOED":
            veto.append("engine_heyting_vetoed")
        for reason in audit.engine_veto_reasons:
            tag = f"engine:{reason}"
            if tag not in veto:
                veto.append(tag)

        isolated: Tuple[str, ...] = sheaf.isolated_charts if sheaf is not None else ()
        surgical = False
        if sheaf is not None and sheaf.isolated_charts and sheaf.surgical_possible:
            # Veto local no se eleva a global si el meet del resto no es ⊥
            # y no hay obstrucción Čech irresoluble en un solape certificado.
            rest = [
                s for s in sheaf.sections if s.name not in sheaf.isolated_charts
            ]
            rest_meet = HeytingVerdict.CERTIFIED
            for s in rest:
                rest_meet = rest_meet.meet(HeytingVerdict.parse(s.verdict))
            if rest and rest_meet is not HeytingVerdict.VETOED:
                surgical = True
                for tag in ("clausius_duhem_residual_below_threshold",):
                    if tag in veto and observation.clausius_duhem_residual >= self._cd_thresh:
                        veto.remove(tag)
                degraded.append("sheaf_surgical_isolation")
        if sheaf is not None and sheaf.cech_h1_obstructed and not surgical:
            veto.append("cech_h1_obstruction")

        physics = float(orientation.physics_score)
        policy = float(orientation.policy_score)
        meet = min(physics, 1.0 if not veto else 0.0, 0.95 if degraded else 1.0, max(policy, 0.5))

        if veto:
            verdict = HeytingVerdict.VETOED
        elif meet >= 0.99 and not degraded:
            verdict = HeytingVerdict.CERTIFIED
        elif meet >= 0.90 and not degraded:
            verdict = HeytingVerdict.COHERENT
        elif degraded or meet >= 0.50:
            verdict = HeytingVerdict.DEGRADED if meet >= 0.50 else HeytingVerdict.VETOED
            if meet < 0.50 and not veto:
                veto.append("heyting_meet_collapsed")
                verdict = HeytingVerdict.VETOED
        else:
            verdict = HeytingVerdict.VETOED

        if surgical and verdict is HeytingVerdict.VETOED and "cech_h1_obstruction" not in veto:
            # La parálisis global se degrada a aislamiento local.
            if "engine_invocation_failed" not in veto and "third_law_nonpositive_temperature" not in veto:
                verdict = HeytingVerdict.DEGRADED
                degraded.append("global_veto_demoted_to_surgical")
                veto = [v for v in veto if v not in {"clausius_duhem_residual_below_threshold"}]

        rl_acc = float(frac.rl_integral_phi) if frac is not None else 0.0
        _ = rl_acc
        return verdict, float(meet), tuple(veto), tuple(degraded), isolated, surgical

    def evaluate_heyting_decision_lattice(
        self,
        cd_residual: float,
        exergy_potential: float,
        is_cd_coherent: bool,
        is_exergy_sufficient: bool,
    ) -> str:
        """[COMPATIBILIDAD 1.X — FASE DECIDE]"""
        verdict, _, _ = self._classify_heyting(
            cd_residual=float(cd_residual),
            exergy_potential=float(exergy_potential),
            is_cd_coherent=bool(is_cd_coherent),
            is_exergy_sufficient=bool(is_exergy_sufficient),
        )
        return verdict

    def _cas_interlock(self, expected: bool, desired: bool) -> bool:
        with self._interlock_lock:
            if self._interlock_state == expected:
                self._interlock_state = desired
                return True
            return False

    def reset_hardware_interlock_for_supervision(self) -> bool:
        with self._interlock_lock:
            previous_state = self._interlock_state
            self._interlock_state = False
            return previous_state

    def _act_from_verdict(self, verdict: str, *, surgical: bool) -> Tuple[bool, float]:
        normalized = str(verdict).strip().upper()
        if normalized != HeytingVerdict.VETOED.value:
            return False, 0.0
        if surgical:
            logger.warning(
                "Veto quirúrgico: no se dispara Crowbar global. Cartas aisladas en telemetría."
            )
            return False, 0.0
        swapped = self._cas_interlock(expected=False, desired=True)
        if not swapped:
            logger.warning("CAS: interlock del AGENTE ya enclavado. Latencia SIMULADA.")
        jitter = float(self._rng.normal(loc=0.0, scale=4.0))
        latency = float(
            np.clip(
                _CROWBAR_IRAM_LATENCY_NS + jitter,
                _CROWBAR_LATENCY_FLOOR_NS,
                _CROWBAR_LATENCY_CEIL_NS,
            )
        )
        logger.critical(
            "VETO TÉRMICO DEL SOBERANO. Crowbar BT151 [GPIO14] SIMULADO. Latencia ficticia: %.2f ns.",
            latency,
        )
        return True, latency

    def act_hardware_interlock_simulation(self, verdict: str) -> Tuple[bool, float]:
        """[COMPATIBILIDAD 1.X — FASE ACT]. GPIO/IRAM no se tocan."""
        return self._act_from_verdict(verdict, surgical=False)

    def _phase3_conservation_audit(
        self,
        observation: Phase1ThermalObservation,
        orientation: Phase2ThermalOrientation,
        verdict: HeytingVerdict,
    ) -> float:
        q = np.asarray(observation.heat_flux_vector, dtype=np.float64)
        energy_l2 = float(np.real(np.dot(q, q))) if q.size else 0.0
        energy_gap = abs(energy_l2 - observation.heat_flux_energy)
        phi = observation.clausius_duhem_residual
        cd_inconsistent = 0.0
        if orientation.is_cd_coherent and math.isfinite(phi) and phi < self._cd_thresh:
            cd_inconsistent = 1.0
        if (not orientation.is_cd_coherent) and math.isfinite(phi) and phi >= self._cd_thresh:
            cd_inconsistent = 1.0
        eta = observation.carnot_efficiency
        eta_res = 0.0 if (not math.isfinite(eta) or 0.0 <= eta <= 1.0 + 1e-12) else 1.0
        temp_res = 1.0 if observation.observation_valid and not orientation.is_temperature_physical else 0.0
        certified_lie = 1.0 if verdict is HeytingVerdict.CERTIFIED and orientation.physics_score < 0.99 else 0.0
        engine_cons = observation.certificate_audit.conservation_residual
        engine_cons_term = float(engine_cons) if math.isfinite(engine_cons) else 0.0
        tellegen = observation.certificate_audit.tellegen_residual
        tellegen_term = float(tellegen) if math.isfinite(tellegen) else 0.0
        terms = np.asarray(
            [energy_gap, cd_inconsistent, eta_res, temp_res, certified_lie, max(0.0, engine_cons_term), tellegen_term],
            dtype=np.float64,
        )
        return float(np.real(self._neumaier_sum(terms)))

    def phase3_decide_from_orientation(
        self,
        observation: Phase1ThermalObservation,
        orientation: Phase2ThermalOrientation,
    ) -> Phase3ThermalDecision:
        r"""
        [FASE 3 — ÚLTIMO MORFISMO: DECIDE / ACT]

        Dominio: `Phase1ThermalObservation × Phase2ThermalOrientation`.
        """
        if not isinstance(observation, Phase1ThermalObservation):
            raise TypeError("observation debe ser Phase1ThermalObservation.")
        if not isinstance(orientation, Phase2ThermalOrientation):
            raise TypeError("orientation debe ser Phase2ThermalOrientation.")

        agent_verdict, score, veto_reasons, degraded_reasons, isolated, surgical = (
            self._classify_heyting_supervised(observation, orientation)
        )
        engine_token = observation.certificate_audit.engine_verdict
        final_verdict = HeytingVerdict.parse(
            self._join_heyting_verdicts((agent_verdict.value, engine_token)),
            unknown_as_veto=True,
        )
        if surgical and final_verdict is HeytingVerdict.VETOED:
            if "engine_invocation_failed" not in veto_reasons:
                final_verdict = HeytingVerdict.DEGRADED

        disagreement = bool(
            engine_token
            and HeytingVerdict.parse(engine_token, unknown_as_veto=True) is not agent_verdict
        )
        if disagreement:
            degraded_reasons = degraded_reasons + ("supervisor_engine_disagreement",)

        conservation = self._phase3_conservation_audit(observation, orientation, final_verdict)
        interlock_fired, latency = self._act_from_verdict(final_verdict.value, surgical=surgical)
        engine_interlock = bool(observation.certificate_audit.engine_interlock_fired)
        dual = bool(interlock_fired and engine_interlock)
        frac = orientation.fractional
        kms = orientation.kms
        sheaf = orientation.sheaf
        rl_acc = float(frac.rl_integral_phi) if frac is not None else 0.0
        kms_def = float(kms.relative_entropy) if (kms is not None and kms.available and math.isfinite(kms.relative_entropy)) else 0.0
        h1 = bool(sheaf.cech_h1_obstructed) if sheaf is not None else False

        diagnostics: Dict[str, Any] = {
            "observe": dict(observation.diagnostics),
            "orientation": dict(orientation.diagnostics),
            "agent_verdict": agent_verdict.value,
            "engine_verdict": engine_token or None,
            "final_verdict": final_verdict.value,
            "heyting_score": score,
            "veto_reasons": veto_reasons,
            "degraded_reasons": degraded_reasons,
            "supervisor_disagreement": disagreement,
            "engine_interlock_fired": engine_interlock,
            "agent_interlock_fired": interlock_fired,
            "dual_channel_actuation": dual,
            "conservation_residual": conservation,
            "isolated_charts": isolated,
            "surgical_veto": surgical,
            "cech_h1_obstructed": h1,
            "fractional_cd_accumulator": rl_acc,
            "kms_defect": kms_def,
        }
        logger.debug(
            "Fase 3: agente=%s motor=%s final=%s score=%.4f I^αΦ=%.3e H¹=%s surgical=%s",
            agent_verdict.value, engine_token, final_verdict.value, score, rl_acc, h1, surgical,
        )
        return Phase3ThermalDecision(
            heyting_verdict=final_verdict.value,
            heyting_rank=final_verdict.rank,
            heyting_score=score,
            veto_reasons=veto_reasons,
            degraded_reasons=degraded_reasons,
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=latency,
            engine_interlock_fired=engine_interlock,
            dual_channel_actuation=dual,
            supervisor_disagreement=disagreement,
            conservation_residual=conservation,
            isolated_charts=isolated,
            surgical_veto=surgical,
            cech_h1_obstructed=h1,
            fractional_cd_accumulator=rl_acc,
            kms_defect=kms_def,
            diagnostics=diagnostics,
        )


# =============================================================================
# CLASE PÚBLICA — S = Act ∘ Orient ∘ Observe
# =============================================================================
class ThermalGradientAgent(Phase3ThermalDecisionActuationMixin):
    r"""
    Soberano de calibre.

        In --F1→ Phase1ThermalObservation --F2→ Phase2ThermalOrientation
           --F3→ Phase3ThermalDecision → ThermalGradientCertificate
    """

    def execute_thermal_agent_cycle(
        self,
        K_raw: np.ndarray,
        grad_T_raw: np.ndarray,
        T_sys: float,
        metric_tensor: np.ndarray,
        entropy_production_rate: float,
        *,
        density_rho: Optional[np.ndarray] = None,
        hamiltonian_H: Optional[np.ndarray] = None,
        **engine_kwargs: Any,
    ) -> ThermalGradientCertificate:
        observation = self.phase1_observe_engine_stamp(
            K_raw=K_raw,
            grad_T_raw=grad_T_raw,
            T_sys=T_sys,
            metric_tensor=metric_tensor,
            entropy_production_rate=entropy_production_rate,
            **engine_kwargs,
        )
        orientation = self.phase2_orient_from_observation(
            observation, density_rho=density_rho, hamiltonian_H=hamiltonian_H,
        )
        decision = self.phase3_decide_from_orientation(observation, orientation)

        if decision.heyting_verdict == HeytingVerdict.VETOED.value:
            logger.error(
                "VETO TÉRMICO. Φ=%.6f η=%.4f I^αΦ=%.3e razones=%s",
                observation.clausius_duhem_residual,
                observation.carnot_efficiency,
                decision.fractional_cd_accumulator,
                decision.veto_reasons,
            )

        diagnostics: Dict[str, Any] = {
            "phase1": dict(observation.diagnostics),
            "phase2": dict(orientation.diagnostics),
            "phase3": dict(decision.diagnostics),
        }
        audit = observation.certificate_audit
        kms = orientation.kms
        fid = float(kms.uhlmann_fidelity) if (kms is not None and kms.available and math.isfinite(kms.uhlmann_fidelity)) else 1.0
        return ThermalGradientCertificate(
            phase="G_THERMAL_GRADIENTS_SUTURATED",
            heyting_verdict=decision.heyting_verdict,
            clausius_duhem_residual=observation.clausius_duhem_residual,
            carnot_efficiency=observation.carnot_efficiency,
            exergy_potential=observation.exergy_potential,
            heat_flux_norm=observation.heat_flux_norm,
            temperature_system=observation.temperature_system,
            hardware_interlock_fired=decision.hardware_interlock_fired,
            actuation_latency_ns=decision.actuation_latency_ns,
            veto_reasons=decision.veto_reasons,
            degraded_reasons=decision.degraded_reasons,
            diagnostics=diagnostics,
            heyting_score=decision.heyting_score,
            conservation_residual=decision.conservation_residual,
            fourier_residual=audit.fourier_residual if math.isfinite(audit.fourier_residual) else 0.0,
            csmd_error=audit.csmd_error if math.isfinite(audit.csmd_error) else 0.0,
            landauer_gap=audit.landauer_gap if math.isfinite(audit.landauer_gap) else 0.0,
            gouy_stodola=audit.gouy_stodola if math.isfinite(audit.gouy_stodola) else 0.0,
            observation_valid=observation.observation_valid,
            engine_heyting_verdict=audit.engine_verdict,
            policy_exergy_sufficient=orientation.is_exergy_sufficient,
            supervisor_disagreement=decision.supervisor_disagreement,
            fractional_cd_accumulator=decision.fractional_cd_accumulator,
            cech_h1_obstructed=decision.cech_h1_obstructed,
            isolated_charts=decision.isolated_charts,
            surgical_veto=decision.surgical_veto,
            kms_defect=decision.kms_defect,
            uhlmann_fidelity=fid,
        )


if __name__ == "__main__":
    print("Autocomprobación Thermal Gradient Agent v3.1 (Caputo / Čech / KMS)...")
    agent = ThermalGradientAgent(
        dimension_n=3, rng_seed=0,
        coordinate_cover={"partners": [0, 1], "resources": [1, 2]},
        fractional_alpha=0.5, history_window=16,
    )
    K = np.eye(3)
    G = np.eye(3)
    grad = np.array([1.0, 0.0, 0.0])

    cert = agent.execute_thermal_agent_cycle(K, grad, 300.0, G, 0.0)
    print("Ciclo 1:", cert.heyting_verdict, "I^αΦ=", cert.fractional_cd_accumulator,
          "H¹=", cert.cech_h1_obstructed, "isolated=", cert.isolated_charts)

    iso = agent.execute_thermal_agent_cycle(K, np.zeros(3), 300.0, G, 0.0)
    print("Isotermo:", iso.heyting_verdict, "η=", iso.carnot_efficiency)

    # Memoria: muchos ciclos coherentes no deben vetar.
    for _ in range(8):
        agent.execute_thermal_agent_cycle(K, grad, 300.0, G, 0.0)
    slow = agent.execute_thermal_agent_cycle(K, grad, 300.0, G, 0.0)
    print("Tras 10 ciclos coherentes:", slow.heyting_verdict, "I^αΦ=", slow.fractional_cd_accumulator)

    # KMS: Gibbs a T=300 con H diagonal debe ser coherente.
    H = np.diag([0.0, 1.0, 2.0])
    beta = 1.0 / 300.0
    z = np.exp(-beta * np.array([0.0, 1.0, 2.0]))
    rho = np.diag(z / z.sum())
    kms_ok = agent.execute_thermal_agent_cycle(K, grad, 300.0, G, 0.0, density_rho=rho, hamiltonian_H=H)
    print("KMS Gibbs:", kms_ok.heyting_verdict, "D=", kms_ok.kms_defect, "F=", kms_ok.uhlmann_fidelity)

    rho_bad = np.diag([1.0, 0.0, 0.0])
    kms_bad = agent.execute_thermal_agent_cycle(K, grad, 300.0, G, 0.0, density_rho=rho_bad, hamiltonian_H=H)
    print("KMS puro vs Gibbs:", kms_bad.heyting_verdict, "D=", kms_bad.kms_defect, kms_bad.veto_reasons)

    cold = agent.execute_thermal_agent_cycle(K, grad, -1.0, G, 0.0)
    print("T<0:", cold.heyting_verdict, cold.veto_reasons)


__all__ = [
    "Phase1ThermalObservation",
    "Phase2ThermalOrientation",
    "Phase3ThermalDecision",
    "ThermalGradientCertificate",
    "EngineCertificateAudit",
    "FractionalMemoryChart",
    "ChartSection",
    "SheafCohomologyChart",
    "KMSChart",
    "Phase1ThermalObservationMixin",
    "Phase2ThermalOrientationMixin",
    "Phase3ThermalDecisionActuationMixin",
    "ThermalGradientAgent",
    "HeytingVerdict",
]