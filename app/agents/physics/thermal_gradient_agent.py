# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Thermal Gradient Agent (Soberano de Calibre del Campo Térmico)      ║
║ Ruta   : app/agents/physics/thermal_gradient_agent.py                        ║
║ Versión: 3.0.0-Doctoral-Heyting-OODA-Carnot-Fourier-Landauer-CAS-Secure      ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA (rigor doctoral, no ornamental):                         ║
║ El agente es un endofunctor de supervisión  S = Act ∘ Orient ∘ Observe       ║
║ sobre el topos de sellos emitidos por ThermalGradientLaws. No recalcula      ║
║ Fourier; audita el certificado del motor contra identidades que deben        ║
║ anularse en aritmética exacta y contra umbrales de *política* (misión),      ║
║ que no se confunden con la 2ª ley.                                           ║
║                                                                              ║
║   Física (veto duro)     Φ = σ − ⟨q, dT⟩/T²  ≥  τ_CD                         ║
║                          ‖q + κ ∇T‖ ≈ 0 ,  T > 0 ,  η_C ∈ [0, 1)             ║
║                          σ ≥ Γ ln 2 ,  CSMD(½ pᵀκp) = κp                     ║
║   Política (degradación) η_C ≱ τ_misión  ⇒  DEGRADED, jamás VETO             ║
║                          (v2 vetaba el equilibrio isotermo: η_C = 0.)        ║
║   Calibre                meet de Heyting(agente, motor)  =  peor verdad      ║
║                          (join de severidad). CERTIFIED es ⊤, no un          ║
║                          token desconocido.                                  ║
║                                                                              ║
║ ARQUITECTURA FUNCTORIAL EN TRES FASES ANIDADAS (mixins = morfismos):         ║
║   Fase 1  Observe  :  In → Phase1ThermalObservation                          ║
║   Fase 2  Orient   :  Phase1ThermalObservation → Phase2ThermalOrientation    ║
║   Fase 3  Decide/Act: (Phase1 × Phase2) → Phase3ThermalDecision              ║
║                                                                              ║
║ Toda actuación Crowbar/GPIO es SIMULADA. No hay acceso a silicio real.       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la

try:
    from thermal_gradient_laws import ThermalGradientLaws
except ImportError:  # pragma: no cover — resolución de paquete de la Malla
    try:
        from app.physics.thermal_gradient_laws import ThermalGradientLaws
    except ImportError:
        from physics.thermal_gradient_laws import ThermalGradientLaws

logger = logging.getLogger("APU.Agents.ThermalGradientAgent")

# ---------------------------------------------------------------------------
# Constantes metrológicas (Wilkinson / IEEE-754 binary64)
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


# =============================================================================
# Retículo de Heyting (cadena de 4 puntos).
# =============================================================================
class HeytingVerdict(str, Enum):
    r"""
    Cadena de verdad  ⊥ = VETOED ≺ DEGRADED ≺ COHERENT ≺ CERTIFIED = ⊤.

    En un orden total el álgebra de Heyting es única:
        a ∧ b = min(a, b),   a ∨ b = max(a, b),
        a → b = ⊤  si a ≼ b,  y  a → b = b  en caso contrario,
        ¬a = a → ⊥.

    La *severidad* de alarma es el orden opuesto. El «join de seguridad»
    (peor gana) es el meet de verdad. v2 usaba Ω₃ y trataba cualquier
    token desconocido —incluido CERTIFIED del motor v3— como VETOED.
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
# Contratos inmutables de fase
# =============================================================================
def _freeze_array(arr: np.ndarray) -> np.ndarray:
    """Copia C-contigua de solo lectura: inmuniza el paquete frente a aliasing."""
    out = np.array(arr, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, slots=True)
class EngineCertificateAudit:
    r"""
    Lectura crítica del sello del motor (no una recomputación de Fourier).

    Campos ausentes en un motor 1.x/2.x se marcan `available=False` y no
    vetan por sí solos; un motor 3.x que *declare* residuos rotos sí veta.
    """

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


@dataclass(frozen=True, slots=True)
class Phase1ThermalObservation:
    r"""
    Objeto terminal de la Fase 1 ≡ objeto inicial de la Fase 2.
    """

    clausius_duhem_residual: float
    carnot_efficiency: float
    exergy_potential: float
    heat_flux_norm: float
    heat_flux_energy: float
    temperature_system: float
    heat_flux_vector: np.ndarray
    engine_stamp: Dict[str, Any]
    observation_valid: bool
    engine_alive: bool
    certificate_audit: EngineCertificateAudit
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "heat_flux_vector", _freeze_array(self.heat_flux_vector))


@dataclass(frozen=True, slots=True)
class Phase2ThermalOrientation:
    r"""
    Objeto terminal de la Fase 2 ≡ segundo factor del dominio de la Fase 3.
    """

    is_cd_coherent: bool
    is_exergy_sufficient: bool
    is_fourier_coherent: bool
    is_csmd_coherent: bool
    is_landauer_coherent: bool
    is_temperature_physical: bool
    is_carnot_in_range: bool
    is_engine_alive: bool
    is_certificate_internally_consistent: bool
    cd_margin: float
    exergy_margin: float
    fourier_margin: float
    landauer_margin: float
    physics_score: float
    policy_score: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Phase3ThermalDecision:
    r"""Sello de decisión (Act) del soberano de calibre."""

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
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ThermalGradientCertificate:
    r"""
    Certificado inmutable de regularidad termodinámica.

    Claves 1.x/2.x se conservan; las de supervisión v3 van al final con
    valores por defecto para no romper desestructuración antigua.
    """

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


# =============================================================================
# FASE 1 — OBSERVE
# =============================================================================
class Phase1ThermalObservationMixin:
    r"""
    FASE 1 — OBSERVE.

    Invoca el motor en modo fail-safe, extrae el sello y construye una
    auditoría de certificado. El último método, `phase1_observe_engine_stamp`,
    tiene por codominio `Phase1ThermalObservation`, que ES el dominio de
    `phase2_orient_from_observation`.
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

        self._engine = ThermalGradientLaws(
            dimension_n=self._n,
            safety_margin=self._safety_margin,
            rng_seed=rng_seed,
        )
        self._rng = np.random.default_rng(rng_seed)
        self._interlock_lock = threading.Lock()
        self._interlock_state = False

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
        r"""Kahan–Babuška–Neumaier.  |fl(∑x) − ∑x| ≤ (2u + O(u²)) ∑|x|."""
        vec = np.asarray(arr, dtype=np.float64).ravel()
        if vec.ndim != 1:
            raise ValueError(f"kahan_sum espera un vector 1-D, se recibió {np.asarray(arr).shape}")
        return float(np.real(self._neumaier_sum(vec)))

    @staticmethod
    def _extract_float(
        mapping: Dict[str, Any],
        key: str,
        default: float,
    ) -> Tuple[float, bool]:
        if key not in mapping:
            return default, False
        try:
            value = float(mapping[key])
        except (TypeError, ValueError):
            return default, False
        return value, True

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
        """Lee el sello v3 (y degrada con gracia si el motor es 1.x/2.x)."""
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

        pairing = grab("q_dot_grad_T", float("nan")) if "q_dot_grad_T" in stamp else float(
            nested_p2.get("q_dot_grad_T", float("nan"))
        )
        condition = float(nested_p2.get("metric_condition", float("nan")))
        third_law = bool(nested_p1.get("third_law_flag", False) or diagnostics.get("third_law_flag", False))
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
            pairing_q_dT=float(pairing) if math.isfinite(pairing) else float("nan"),
            metric_condition=condition,
            third_law_flag=third_law,
            temperature_clamped=t_clamped,
            engine_veto_reasons=self._extract_string_tuple(stamp, "veto_reasons"),
            engine_degraded_reasons=self._extract_string_tuple(stamp, "degraded_reasons"),
            missing_keys=tuple(missing),
        )

    def _fail_safe_observation(
        self,
        diagnostics: Dict[str, Any],
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
            engine_stamp={},
            observation_valid=False,
            engine_alive=False,
            certificate_audit=audit,
            diagnostics=diagnostics,
        )

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

        Codominio
        ---------
        `Phase1ThermalObservation`

        Continuidad formal
        ------------------
        Este paquete **es** el dominio del primer morfismo de la Fase 2,
        `phase2_orient_from_observation(self, observation)`.
        `engine_kwargs` se reenvía al motor v3 (conductivity_type, length_scale,
        ambient_temperature, info_erasure_rate) y se ignora con gracia si el
        motor subyacente aún no los acepta.
        """
        logger.info("Fase Observe: capturando el sello del motor térmico.")
        diagnostics: Dict[str, Any] = {
            "observe_phase": "engine_invocation",
            "engine_invoked": True,
        }

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
            # Motor 1.x/2.x sin kwargs v3.
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
                logger.exception("El motor térmico falló. Colapso fail-safe a observación inválida.")
                diagnostics.update({"engine_error": str(exc), "engine_invoked": False})
                return self._fail_safe_observation(diagnostics)
        except Exception as exc:
            logger.exception("El motor térmico falló. Colapso fail-safe a observación inválida.")
            diagnostics.update({"engine_error": str(exc), "engine_invoked": False})
            return self._fail_safe_observation(diagnostics)

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

        if not cd_present:
            diagnostics["missing_clausius_duhem_residual"] = True
        if not eta_present:
            diagnostics["missing_carnot_efficiency"] = True
            if not math.isfinite(carnot_efficiency):
                carnot_efficiency = 0.0
        if not ex_present:
            diagnostics["missing_exergy_potential"] = True
            if not math.isfinite(exergy_potential):
                exergy_potential = 0.0
        if not t_present:
            diagnostics["missing_T_sys_clamped"] = True

        if heat_flux_valid:
            energy = self.kahan_sum(np.asarray(heat_flux_vector, dtype=np.float64) ** 2)
            heat_flux_energy = float(max(energy, 0.0))
            heat_flux_norm = float(math.sqrt(heat_flux_energy))
            if not math.isfinite(heat_flux_norm):
                heat_flux_norm = 0.0
                heat_flux_energy = 0.0
                diagnostics["heat_flux_norm_nonfinite"] = True
        else:
            heat_flux_norm = 0.0
            heat_flux_energy = 0.0
            diagnostics["heat_flux_vector_invalid"] = True

        engine_diagnostics = stamp.get("diagnostics")
        if isinstance(engine_diagnostics, dict):
            diagnostics["engine"] = dict(engine_diagnostics)

        audit = self._audit_engine_certificate(stamp, diagnostics)
        observation_valid = bool(
            heat_flux_valid
            and cd_present
            and math.isfinite(cd_residual)
            and t_present
            and math.isfinite(temperature_system)
        )
        diagnostics["observation_valid"] = observation_valid
        diagnostics["certificate_missing_keys"] = audit.missing_keys

        return Phase1ThermalObservation(
            clausius_duhem_residual=cd_residual,
            carnot_efficiency=carnot_efficiency,
            exergy_potential=exergy_potential,
            heat_flux_norm=heat_flux_norm,
            heat_flux_energy=heat_flux_energy,
            temperature_system=temperature_system,
            heat_flux_vector=heat_flux_vector,
            engine_stamp=dict(stamp),
            observation_valid=observation_valid,
            engine_alive=True,
            certificate_audit=audit,
            diagnostics=diagnostics,
        )


# =============================================================================
# FASE 2 — ORIENT
# Dominio = Phase1ThermalObservation (codominio del último método de Fase 1).
# =============================================================================
class Phase2ThermalOrientationMixin(Phase1ThermalObservationMixin):
    r"""
    FASE 2 — ORIENT.

    Separa *física* (2ª ley, Fourier, T>0, Landauer, CSMD) de *política*
    (umbral de exergía de misión). v2 mezclaba ambas y vetaba el equilibrio.
    """

    def verify_thermodynamic_coherence(
        self,
        cd_residual: float,
        exergy_potential: float,
    ) -> Tuple[bool, bool]:
        r"""
        [COMPATIBILIDAD 1.X — FASE ORIENT]

        Φ ≥ τ_CD  y  exergy ≥ τ_exergy  (esta última es política, no 2ª ley).
        """
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

    def _phase2_audit_engine_certificate(
        self,
        observation: Phase1ThermalObservation,
    ) -> Dict[str, Any]:
        r"""
        [FASE 2 — PRIMER MORFISMO: AUDITORÍA DEL CERTIFICADO]

        Dominio
        -------
        `Phase1ThermalObservation` ← continuación formal de
        `phase1_observe_engine_stamp`.

        Identidades de supervisión (no se re-simula el ciclo):
          1. η_C ∈ [0, 1];
          2. T_h ≥ T_c > 0 si ambos están presentes;
          3. Φ ≥ τ_CD;
          4. ⟨q, dT⟩ ≤ ε_u  (anti-Fourier);
          5. veredicto del motor ∈ {VETOED, DEGRADED, COHERENT, CERTIFIED};
          6. si el motor declara Fourier/CSMD/Landauer, sus residuos acotados.
        """
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
                eta_id = 1.0 - T_c / T_h
                flags["carnot_identity_residual"] = float(abs(eta - eta_id))
            else:
                flags["carnot_identity_residual"] = 0.0
        else:
            flags["carnot_reservoirs_ordered"] = True
            flags["carnot_identity_residual"] = 0.0

        pairing = audit.pairing_q_dT
        flags["anti_fourier"] = bool(math.isfinite(pairing) and pairing > _WILKINSON_FLOOR)

        if math.isfinite(audit.fourier_residual):
            flags["fourier_coherent"] = audit.fourier_residual <= _FOURIER_VETO * self._safety_margin
        else:
            flags["fourier_coherent"] = True  # motor 1.x: no hay evidencia en contra

        if math.isfinite(audit.csmd_error):
            flags["csmd_coherent"] = audit.csmd_error <= _CSMD_VETO
        else:
            flags["csmd_coherent"] = True

        if math.isfinite(audit.landauer_gap):
            flags["landauer_coherent"] = audit.landauer_gap >= -_LANDAUER_ABS_TOL
        else:
            flags["landauer_coherent"] = True

        flags["engine_verdict_known"] = audit.engine_verdict in {
            "VETOED", "DEGRADED", "COHERENT", "CERTIFIED", "",
        }
        flags["engine_vetoed"] = audit.engine_verdict == "VETOED" or bool(audit.engine_veto_reasons)
        return flags

    def phase2_orient_from_observation(
        self,
        observation: Phase1ThermalObservation,
    ) -> Phase2ThermalOrientation:
        r"""
        [FASE 2 — ÚLTIMO MORFISMO: ORIENT]

        Codominio
        ---------
        `Phase2ThermalOrientation`

        Continuidad formal
        ------------------
        Junto con el `Phase1ThermalObservation` residual, este objeto es el
        dominio de `phase3_decide_from_orientation(self, observation, orientation)`.
        """
        if not isinstance(observation, Phase1ThermalObservation):
            raise TypeError("observation debe ser Phase1ThermalObservation.")

        is_cd_coherent, is_exergy_sufficient = self.verify_thermodynamic_coherence(
            cd_residual=observation.clausius_duhem_residual,
            exergy_potential=observation.exergy_potential,
        )
        flags = self._phase2_audit_engine_certificate(observation)
        audit = observation.certificate_audit

        cd_value = observation.clausius_duhem_residual
        exergy_value = observation.exergy_potential
        cd_margin = float(cd_value - self._cd_thresh) if math.isfinite(cd_value) else float("-inf")
        exergy_margin = (
            float(exergy_value - self._exergy_thresh) if math.isfinite(exergy_value) else float("-inf")
        )
        fourier_margin = (
            float(_FOURIER_VETO * self._safety_margin - audit.fourier_residual)
            if math.isfinite(audit.fourier_residual)
            else float("inf")
        )
        landauer_margin = (
            float(audit.landauer_gap) if math.isfinite(audit.landauer_gap) else float("inf")
        )

        is_fourier = bool(flags["fourier_coherent"]) and not flags["anti_fourier"]
        is_csmd = bool(flags["csmd_coherent"])
        is_landauer = bool(flags["landauer_coherent"])
        is_temp = bool(flags["temperature_physical"]) and observation.engine_alive
        is_carnot = bool(flags["carnot_in_range"]) and bool(flags["carnot_reservoirs_ordered"])
        identity_ok = float(flags["carnot_identity_residual"]) <= 1e-8
        cert_ok = (
            observation.observation_valid
            and observation.engine_alive
            and bool(flags["engine_verdict_known"])
            and identity_ok
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
        ]
        physics_score = float(min(physics_terms))
        if math.isfinite(audit.fourier_residual) and audit.fourier_residual > 0.0:
            physics_score = min(
                physics_score,
                float(np.exp(-audit.fourier_residual / max(_FOURIER_DEGRADED, _WILKINSON_FLOOR))),
            )
        policy_score = (
            1.0
            if is_exergy_sufficient
            else float(np.clip(observation.exergy_potential / max(self._exergy_thresh, _WILKINSON_FLOOR), 0.0, 0.99))
            if math.isfinite(observation.exergy_potential)
            else 0.0
        )

        diagnostics: Dict[str, Any] = {
            "cd_threshold": self._cd_thresh,
            "exergy_threshold": self._exergy_thresh,
            "exergy_is_hard_veto": self._exergy_is_hard_veto,
            "cd_residual": cd_value,
            "exergy_potential": exergy_value,
            "cd_margin": cd_margin,
            "exergy_margin": exergy_margin,
            "fourier_margin": fourier_margin,
            "landauer_margin": landauer_margin,
            "is_cd_coherent": is_cd_coherent,
            "is_exergy_sufficient": is_exergy_sufficient,
            "certificate_flags": flags,
            "physics_score": physics_score,
            "policy_score": policy_score,
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
            cd_margin=cd_margin,
            exergy_margin=exergy_margin,
            fourier_margin=fourier_margin,
            landauer_margin=landauer_margin,
            physics_score=physics_score,
            policy_score=policy_score,
            diagnostics=diagnostics,
        )


# =============================================================================
# FASE 3 — DECIDE / ACT
# Dominio = Phase1ThermalObservation × Phase2ThermalOrientation.
# =============================================================================
class Phase3ThermalDecisionActuationMixin(Phase2ThermalOrientationMixin):
    r"""
    FASE 3 — DECIDE / ACT.

    Meet de Heyting (agente ∧ motor). Crowbar SIMULADO, canal independiente
    del interlock del motor (supervisión a dos canales).
    """

    @staticmethod
    def _join_heyting_verdicts(verdicts: Tuple[str, ...]) -> str:
        r"""
        Join de *severidad* = meet de verdad.

        CERTIFIED se reconoce. Un token desconocido no vacío colapsa a VETOED
        (fail-safe); la cadena vacía no vota.
        """
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
        """
        Clasificador 1.x/2.x evolucionado: la exergía insuficiente es
        DEGRADED (política), no VETO, salvo `exergy_is_hard_veto=True`.
        """
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
    ) -> Tuple[HeytingVerdict, float, Tuple[str, ...], Tuple[str, ...]]:
        """Clasificador de supervisión: física dura + política + sello del motor."""
        veto: List[str] = []
        degraded: List[str] = []
        audit = observation.certificate_audit

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
        if (
            math.isfinite(observation.exergy_potential)
            and observation.exergy_potential < _DEGRADED_EXERGY_THRESHOLD
        ):
            degraded.append("exergy_potential_below_degraded_threshold")
        if audit.temperature_clamped and not audit.third_law_flag:
            degraded.append("temperature_clamped_to_wilkinson_floor")
        if math.isfinite(audit.fourier_residual) and _FOURIER_DEGRADED < audit.fourier_residual <= _FOURIER_VETO * self._safety_margin:
            degraded.append("fourier_residual_degraded")
        if math.isfinite(audit.metric_condition) and _CONDITION_DEGRADED < audit.metric_condition <= _CONDITION_VETO:
            degraded.append("metric_ill_conditioned")
        if not orientation.is_certificate_internally_consistent:
            degraded.append("certificate_internal_inconsistency")
        if audit.engine_verdict == "DEGRADED":
            degraded.append("engine_heyting_degraded")
        if audit.engine_verdict == "VETOED":
            veto.append("engine_heyting_vetoed")
        for reason in audit.engine_veto_reasons:
            tag = f"engine:{reason}"
            if tag not in veto:
                veto.append(tag)

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
        return verdict, float(meet), tuple(veto), tuple(degraded)

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

    def _act_from_verdict(self, verdict: str) -> Tuple[bool, float]:
        normalized_verdict = str(verdict).strip().upper()
        if normalized_verdict != HeytingVerdict.VETOED.value:
            return False, 0.0
        swapped = self._cas_interlock(expected=False, desired=True)
        if not swapped:
            logger.warning(
                "CAS: el interlock del AGENTE ya estaba enclavado. "
                "Se reconoce el veto y se registra la latencia SIMULADA."
            )
        jitter = float(self._rng.normal(loc=0.0, scale=4.0))
        actuation_latency_ns = float(
            np.clip(
                _CROWBAR_IRAM_LATENCY_NS + jitter,
                _CROWBAR_LATENCY_FLOOR_NS,
                _CROWBAR_LATENCY_CEIL_NS,
            )
        )
        logger.critical(
            "VETO TÉRMICO DEL SOBERANO DE CALIBRE. "
            "Crowbar BT151 [GPIO14] SIMULADO. Latencia ficticia: %.2f ns.",
            actuation_latency_ns,
        )
        return True, actuation_latency_ns

    def act_hardware_interlock_simulation(self, verdict: str) -> Tuple[bool, float]:
        """[COMPATIBILIDAD 1.X — FASE ACT]. GPIO/IRAM no se tocan."""
        return self._act_from_verdict(verdict)

    def _phase3_conservation_audit(
        self,
        observation: Phase1ThermalObservation,
        orientation: Phase2ThermalOrientation,
        verdict: HeytingVerdict,
    ) -> float:
        r"""
        Residuos KBN de consistencia supervisor–sello:
          1. |‖Q‖² − E_KBN(Q)|;
          2. flag CD vs signo de Φ;
          3. η_C fuera de [0, 1];
          4. T ≤ 0 con observación «válida»;
          5. veredicto CERTIFIED con física_score < 1;
          6. conservación declarada por el motor.
        """
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
        eta_res = 0.0 if (not math.isfinite(eta) or 0.0 <= eta <= 1.0 + 1e-12) else abs(eta - min(max(eta, 0.0), 1.0))
        temp_res = (
            1.0
            if observation.observation_valid and not orientation.is_temperature_physical
            else 0.0
        )
        certified_lie = (
            1.0
            if verdict is HeytingVerdict.CERTIFIED and orientation.physics_score < 0.99
            else 0.0
        )
        engine_cons = observation.certificate_audit.conservation_residual
        engine_cons_term = float(engine_cons) if math.isfinite(engine_cons) else 0.0
        terms = np.asarray(
            [
                energy_gap,
                cd_inconsistent,
                eta_res,
                temp_res,
                certified_lie,
                max(0.0, engine_cons_term),
            ],
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

        Dominio
        -------
        `Phase1ThermalObservation × Phase2ThermalOrientation`
        ← continuación formal de `phase1_observe_engine_stamp` y
        `phase2_orient_from_observation`.

        No hay Fase 4: el sello `Phase3ThermalDecision` es S(estado).
        """
        if not isinstance(observation, Phase1ThermalObservation):
            raise TypeError("observation debe ser Phase1ThermalObservation.")
        if not isinstance(orientation, Phase2ThermalOrientation):
            raise TypeError("orientation debe ser Phase2ThermalOrientation.")

        agent_verdict, score, veto_reasons, degraded_reasons = self._classify_heyting_supervised(
            observation, orientation
        )
        engine_token = observation.certificate_audit.engine_verdict
        final_verdict = HeytingVerdict.parse(
            self._join_heyting_verdicts((agent_verdict.value, engine_token)),
            unknown_as_veto=True,
        )
        disagreement = bool(
            engine_token
            and HeytingVerdict.parse(engine_token, unknown_as_veto=True) is not agent_verdict
        )
        if disagreement:
            degraded_extra = degraded_reasons + ("supervisor_engine_disagreement",)
            degraded_reasons = degraded_extra

        conservation = self._phase3_conservation_audit(observation, orientation, final_verdict)
        interlock_fired, latency = self._act_from_verdict(final_verdict.value)
        engine_interlock = bool(observation.certificate_audit.engine_interlock_fired)
        dual = bool(interlock_fired and engine_interlock)

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
        }
        logger.debug(
            "Fase 3: agente=%s motor=%s final=%s score=%.4f Φ=%.6e η=%.6e cons=%.3e",
            agent_verdict.value, engine_token, final_verdict.value, score,
            observation.clausius_duhem_residual, observation.carnot_efficiency, conservation,
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
            diagnostics=diagnostics,
        )


# =============================================================================
# CLASE PÚBLICA — functor S = Act ∘ Orient ∘ Observe
# =============================================================================
class ThermalGradientAgent(Phase3ThermalDecisionActuationMixin):
    r"""
    Soberano de calibre de los gradientes de calor.

    Encadena las tres fases anidadas sin objetos huérfanos:

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
        **engine_kwargs: Any,
    ) -> ThermalGradientCertificate:
        r"""
        Orquesta el ciclo OODA de gobernanza. Firma 1.x conservada;
        kwargs se reenvían al motor v3.
        """
        observation = self.phase1_observe_engine_stamp(
            K_raw=K_raw,
            grad_T_raw=grad_T_raw,
            T_sys=T_sys,
            metric_tensor=metric_tensor,
            entropy_production_rate=entropy_production_rate,
            **engine_kwargs,
        )
        orientation = self.phase2_orient_from_observation(observation)
        decision = self.phase3_decide_from_orientation(observation, orientation)

        if decision.heyting_verdict == HeytingVerdict.VETOED.value:
            logger.error(
                "Fase Decide/Act: VETO TÉRMICO. Φ=%.6f, exergía=%.6f, η=%.4f%%, razones=%s",
                observation.clausius_duhem_residual,
                observation.exergy_potential,
                (observation.carnot_efficiency * 100.0) if math.isfinite(observation.carnot_efficiency) else float("nan"),
                decision.veto_reasons,
            )

        diagnostics: Dict[str, Any] = {
            "phase1": dict(observation.diagnostics),
            "phase2": dict(orientation.diagnostics),
            "phase3": dict(decision.diagnostics),
        }
        audit = observation.certificate_audit
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
        )


# -----------------------------------------------------------------------------
# Autocomprobación
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Iniciando autocomprobación de Thermal Gradient Agent v3...")
    agent = ThermalGradientAgent(dimension_n=3, rng_seed=0)
    K = np.eye(3)
    G = np.eye(3)
    grad = np.array([1.0, 0.0, 0.0])

    cert = agent.execute_thermal_agent_cycle(K, grad, 300.0, G, 0.0)
    print("Veredicto:", cert.heyting_verdict, "score=", cert.heyting_score)
    print("Φ CD:", cert.clausius_duhem_residual, "η:", cert.carnot_efficiency)
    print("Exergía (calidad, no 1-|Φ|):", cert.exergy_potential)
    print("Fourier/CSMD:", cert.fourier_residual, cert.csmd_error)
    print("Política exergía suficiente:", cert.policy_exergy_sufficient)
    print("Desacuerdo supervisor/motor:", cert.supervisor_disagreement)
    print("Razones veto/degradado:", cert.veto_reasons, cert.degraded_reasons)

    iso = agent.execute_thermal_agent_cycle(K, np.zeros(3), 300.0, G, 0.0)
    print("Isotermo ∇T=0 (equilibrio; no debe vetar por η=0):", iso.heyting_verdict, "η=", iso.carnot_efficiency)

    cold = agent.execute_thermal_agent_cycle(K, grad, -1.0, G, 0.0)
    print("T<0 (3ª ley):", cold.heyting_verdict, cold.veto_reasons)

    land = agent.execute_thermal_agent_cycle(K, grad, 300.0, G, 0.0, info_erasure_rate=1e9)
    print("Landauer:", land.heyting_verdict, land.veto_reasons)

    # Fail-safe: motor sustituido por un lanzador.
    class _Boom:
        def execute_thermal_cycle(self, **_: Any) -> Dict[str, Any]:
            raise RuntimeError("FPU thermal core dumped")

    agent._engine = _Boom()  # type: ignore[assignment]
    boom = agent.execute_thermal_agent_cycle(K, grad, 300.0, G, 0.0)
    print("Fail-safe motor muerto:", boom.heyting_verdict, boom.observation_valid, boom.veto_reasons)


__all__ = [
    "Phase1ThermalObservation",
    "Phase2ThermalOrientation",
    "Phase3ThermalDecision",
    "ThermalGradientCertificate",
    "EngineCertificateAudit",
    "Phase1ThermalObservationMixin",
    "Phase2ThermalOrientationMixin",
    "Phase3ThermalDecisionActuationMixin",
    "ThermalGradientAgent",
    "HeytingVerdict",
]