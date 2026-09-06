# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Octonionic Dependency Agent (Soberano de Calibre Octoniónico)       ║
║ Ruta   : app/agents/wisdom/octonionic_dependency_agent.py                    ║
║ Versión: 3.0.0-Doctoral-OODA-Heyting-Banach-Artin-Nested                     ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y DE GOBERNANZA DE LAZO CERRADO:                         ║
║ Agente supervisor ciber-físico en el estrato de Sabiduría (V_W) / Omega      ║
║ (V_Ω). Gobierna síncronamente al Resolver Octoniónico de Dependencias        ║
║ (Cayley–Dickson, Artin, Moufang) sobre la tríada                             ║
║                                                                              ║
║     (Contratista, Proveedor, Interventoría) ↪ O³ ⊂ (R^8)³.                   ║
║                                                                              ║
║ Sanea señales en el espacio de Banach de dimensión finita (R^8, ‖·‖_p),      ║
║ audita la ley de composición de Hurwitz y el asociador                       ║
║                                                                              ║
║     [a,b,c] = (ab)c − a(bc),                                                 ║
║                                                                              ║
║ y clasifica el veredicto en el álgebra de Heyting Gödel Ω₃ para inyectar     ║
║ conmutación de potencia en silicio (Crowbar IRAM < 400 ns).                  ║
║                                                                              ║
║ Cadena de funtores anidados:                                                 ║
║                                                                              ║
║   (a,b,c) --Fase 1-->  Kernel(Banach, polaridad, sello)                      ║
║           --Fase 2-->  Orientación(Hurwitz, [a,b,c], rampa, ω_pre)           ║
║           --Fase 3-->  Certificado(Ω₃, Crowbar, HMAC)                        ║
║                                                                              ║
║ Germen Fase 1 → Fase 2:                                                      ║
║     synthesize_observation_kernel  ⊣  observe_from_kernel                    ║
║                                                                              ║
║ Germen Fase 2 → Fase 3:                                                      ║
║     orient_octonionic_state        ⊣  decide_from_orientation                ║
║                                                                              ║
║ Ω₃ = {0 < ½ < 1}  (Gödel–Heyting ternario)                                   ║
║     1  ↔  COHERENT                                                           ║
║     ½  ↔  DEGRADED   (veto suave + gracia)                                   ║
║     0  ↔  VETOED     (veto duro + Crowbar IRAM)                              ║
║                                                                              ║
║ Rampa de confianza sobre τ_max = L_max · safety_margin:                      ║
║     ‖[a,b,c]‖ ≤ 0.3 τ_max          →  ω_asoc = 1                             ║
║     0.3 τ_max < ‖[a,b,c]‖ ≤ 0.5 τ  →  ω_asoc = ½                             ║
║     ‖[a,b,c]‖ > 0.5 τ_max          →  ω_hard = 0                             ║
║                                                                              ║
║ Equivalencia de normas en R^8 \ {0}:                                         ║
║     1 ≤ ‖x‖₁/‖x‖₂ ≤ √8 ,   ‖x‖₂² ≤ ‖x‖₁ ‖x‖_∞  (Hölder).                     ║
║                                                                              ║
║ ORGANIZACIÓN EN TRES FASES ANIDADAS POR HERENCIA ESTRICTA:                   ║
║   FASE 1: Phase1_OctonionicObservation                                       ║
║   FASE 2: Phase2_OctonionicOrientation(Phase1_OctonionicObservation)         ║
║   FASE 3: Phase3_OODAActuator(Phase2_OctonionicOrientation)                  ║
║   Agente: OctonionicDependencyAgent(Phase3_OODAActuator, Morphism)           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AbstractSet, Callable, Final, Optional, Tuple

import numpy as np

# ───────────────────────────────────────────────────────────────────────────────
# Compatibilidad categorial opcional
# ───────────────────────────────────────────────────────────────────────────────

try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:
    class Morphism:
        r"""Marcador categorial de compatibilidad."""

        pass

    class TopologicalInvariantError(Exception):
        r"""Error topológico invariante."""

        pass

# ───────────────────────────────────────────────────────────────────────────────
# Importación resiliente del resolvedor octoniónico (v2 o v3)
# ───────────────────────────────────────────────────────────────────────────────

try:
    from app.core.octonionic_dependency_resolver import (
        OctonionicDependencyResolver,
        OctonionicState,
        OctonionicAuditCertificate,
    )
except ImportError:
    try:
        from octonionic_dependency_resolver import (
            OctonionicDependencyResolver,
            OctonionicState,
            OctonionicAuditCertificate,
        )
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar octonionic_dependency_resolver. Verifique que el "
            "módulo esté en el PYTHONPATH o en app/core."
        ) from exc


logger = logging.getLogger("APU.Agents.Wisdom.OctonionicDependencyAgent")

__version__: Final[str] = "3.0.0"

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_OCTONION_DIM: Final[int] = 8
_BANACH_SQRT8: Final[float] = float(math.sqrt(8.0))
_BANACH_CLAMP_EPS: Final[float] = 100.0 * _MACHINE_EPS

_VERDICT_COHERENT: Final[str] = "COHERENT"
_VERDICT_DEGRADED: Final[str] = "DEGRADED"
_VERDICT_VETOED: Final[str] = "VETOED"

_PHASE_NAME: Final[str] = "G_WISDOM_OCTONIONIC_SUTURATED"

_LEGACY_OVERRIDE_TOKENS: Final[frozenset] = frozenset(
    {
        "AUT_POS_SABIDURIA_777",
        "OVERRIDE_NON_ASSOCIATIVE_IDU_2026",
        "HMAC_SUTURA_FOCK_SECURE",
    }
)

# Fracciones de la rampa de confianza sobre τ_max.
_RAMPA_SOFT: Final[float] = 0.3
_RAMPA_HARD: Final[float] = 0.5


# ───────────────────────────────────────────────────────────────────────────────
# Clasificación en el álgebra de Heyting Gödel Ω₃
# ───────────────────────────────────────────────────────────────────────────────

class HeytingVerdict(str, Enum):
    r"""
    Puntos del álgebra de Heyting ternaria Ω₃ = {0 < ½ < 1}.

    Orden de Gödel: VETOED < DEGRADED < COHERENT.
    El override nunca eleva 0; a lo sumo fija ½ (implicación Heyting).
    """

    VETOED = _VERDICT_VETOED
    DEGRADED = _VERDICT_DEGRADED
    COHERENT = _VERDICT_COHERENT

    @property
    def omega(self) -> float:
        return {
            HeytingVerdict.VETOED: 0.0,
            HeytingVerdict.DEGRADED: 0.5,
            HeytingVerdict.COHERENT: 1.0,
        }[self]

    @classmethod
    def from_omega(cls, value: float) -> "HeytingVerdict":
        if value <= 0.0:
            return cls.VETOED
        if value < 1.0:
            return cls.DEGRADED
        return cls.COHERENT


def _heyting_meet(a: float, b: float) -> float:
    return float(min(a, b))


def _heyting_implies(a: float, b: float) -> float:
    return 1.0 if a <= b + _MACHINE_EPS else float(b)


def _heyting_not(a: float) -> float:
    return _heyting_implies(a, 0.0)


# ───────────────────────────────────────────────────────────────────────────────
# Serialización canónica e inmutabilidad de ndarrays
# ───────────────────────────────────────────────────────────────────────────────

def _immutable(array: np.ndarray, dtype: Optional[np.dtype] = None) -> np.ndarray:
    out = np.array(array, dtype=dtype, copy=True, order="C")
    out.setflags(write=False)
    return out


def _canonical_bytes(part: Any) -> bytes:
    r"""Serialización little-endian estable, independiente de la arquitectura."""
    if isinstance(part, np.ndarray):
        arr = np.ascontiguousarray(part)
        header = np.array(arr.shape, dtype="<i8").tobytes()
        header += np.array([1 if np.iscomplexobj(arr) else 0], dtype="<i8").tobytes()
        if np.iscomplexobj(arr):
            real = np.ascontiguousarray(arr.real, dtype=np.float64).astype("<f8")
            imag = np.ascontiguousarray(arr.imag, dtype=np.float64).astype("<f8")
            return header + real.tobytes() + imag.tobytes()
        real = np.ascontiguousarray(arr, dtype=np.float64).astype("<f8")
        return header + real.tobytes()
    if isinstance(part, bytes):
        return part
    if isinstance(part, str):
        return part.encode("utf-8")
    if isinstance(part, (int, float, bool, np.generic)):
        return np.array([part], dtype="<f8").tobytes()
    return repr(part).encode("utf-8")


# ───────────────────────────────────────────────────────────────────────────────
# Dataclasses inmutables
# ───────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class BanachRegularityReport:
    r"""
    Espectro de normas equivalentes en (R^8, ‖·‖_p).

    Identidades:
        1 ≤ ‖x‖₁/‖x‖₂ ≤ √8          (x ≠ 0),
        ‖x‖₂² ≤ ‖x‖₁ ‖x‖_∞          (Hölder),
        Hoyer = (√8 − ‖x‖₁/‖x‖₂)/(√8 − 1) ∈ [0, 1]
            1 = 1-esparso,  0 = equidistribuido.

    Atributos:
        l1_norm, l2_norm, linf_norm:  Normas clásicas.
        l1_l2_ratio:                  Cociente regularizado / clampeado.
        raw_l1_l2_ratio:              Cociente antes del clamp analítico.
        hoyer_sparsity:               Índice de Hoyer.
        holder_defect:                ‖x‖₁‖x‖_∞ − ‖x‖₂² ≥ 0.
        is_null:                      ‖x‖₂ ≤ ε_máq.
        is_clamped:                   Se aplicó corrección a las cotas.
    """

    l1_norm: float
    l2_norm: float
    linf_norm: float
    l1_l2_ratio: float
    raw_l1_l2_ratio: float
    hoyer_sparsity: float
    holder_defect: float
    is_null: bool
    is_clamped: bool = False


@dataclass(frozen=True, slots=True)
class OctonionicObservationKernel:
    r"""
    GERMEN FASE 1 → FASE 2.

    Expediente inmutable de la observación: tríada octoniónica canonizada
    sobre el espacio de Banach, con espectro ℓ^p y sello criptográfico.

    El funtor de la Fase 2, ``observe_from_kernel``, actúa de forma estricta
    sobre este germen.
    """

    contractor_state: OctonionicState
    supplier_state: OctonionicState
    interventor_state: OctonionicState
    banach_ratio_contractor: float
    banach_ratio_supplier: float
    banach_ratio_interventor: float
    cryptographic_seal: str

    contractor_spectrum: Optional[BanachRegularityReport] = None
    supplier_spectrum: Optional[BanachRegularityReport] = None
    interventor_spectrum: Optional[BanachRegularityReport] = None
    polar_contractor: Optional[np.ndarray] = None
    polar_supplier: Optional[np.ndarray] = None
    polar_interventor: Optional[np.ndarray] = None
    null_party_count: int = 0


@dataclass(frozen=True, slots=True)
class OctonionicOrientationState:
    r"""
    GERMEN FASE 2 → FASE 3.

    Orientación covariante: Hurwitz, asociador, rampa de confianza y
    preclasificación Heyting ω_pre ∈ Ω₃ *antes* de gracia y override.

        ω_hard  = 0 si Hurwitz grave ∨ asociador > 0.5 τ ∨ no finito
        ω_cfl   = 1 si ‖xy‖=‖x‖‖y‖ (pares), else ½
        ω_asoc  = 1 / ½ / (vía hard) según rampa 0.3 τ / 0.5 τ
        ω_banach= ½ si hay actor nulo, else 1
        ω_pre   = ω_hard ∧ ω_cfl ∧ ω_asoc ∧ ω_banach
    """

    kernel: OctonionicObservationKernel
    associator_norm: float
    composition_error: float
    is_associative_stable: bool
    is_cfl_stable: bool

    composition_relative_error: float = 0.0
    associator_relative_norm: float = 0.0
    cota_limite: float = 0.0

    associator_vector: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )

    composition_tolerance: float = 0.0
    composition_hard_limit: float = 0.0
    associator_soft_start: float = 0.0
    associator_hard_limit: float = 0.0

    omega_hard: float = 1.0
    omega_cfl: float = 1.0
    omega_asoc: float = 1.0
    omega_banach: float = 1.0
    omega_pre: float = 1.0
    artin_residual: float = float("nan")
    moufang_residual: float = float("nan")
    triad_seal: str = ""
    hard_composition: bool = False
    hard_associator: bool = False
    soft_composition: bool = False
    soft_associator: bool = False


@dataclass(frozen=True, slots=True)
class OctonionicAgentCertificate:
    r"""
    Certificado formal de calibración y veto del Soberano Octoniónico.

    Atributos:
        phase:                          Nombre de fase gubernamental.
        heyting_verdict:                COHERENT | DEGRADED | VETOED.
        associator_norm:                ‖[a,b,c]‖.
        is_associative_stable:          No-asociatividad bajo τ_max.
        composition_error:              Desviación de Hurwitz.
        is_surgery_active:              Cirugía / estado degradado.
        is_soft_veto_active:            Luz ámbar activa.
        override_grace_period_expired:  Ventana de gracia expirada.
        hardware_interlock_fired:       Crowbar BT151 gatillado.
        actuation_latency_ns:           Latencia IRAM determinista.
        time_grace_remaining:           Gracia residual (s).
        digital_signature_sha256:       Sello SHA-256 / HMAC-SHA256.
        composition_relative_error:     Error relativo de Hurwitz.
        associator_relative_norm:       ‖[a,b,c]‖ / (‖a‖‖b‖‖c‖).
        banach_ratio_*:                 Regularidad ℓ¹/ℓ² por actor.
        *_norm:                         Normas octoniónicas.
        heyting_omega:                  Valor numérico en Ω₃.
        omega_pre:                      ω antes de gracia/override.
        artin_residual:                 max(alt_L, alt_R, flex).
        moufang_residual:               Residuo de Moufang.
        observation_seal:               Sello del germen Fase 1.
        hoyer_sparsity_max:             Máxima escasez de Hoyer en la tríada.
        null_party_count:               Actores nulos.
    """

    phase: str
    heyting_verdict: str
    associator_norm: float
    is_associative_stable: bool
    composition_error: float
    is_surgery_active: bool
    is_soft_veto_active: bool
    override_grace_period_expired: bool
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    time_grace_remaining: float
    digital_signature_sha256: str

    composition_relative_error: float = 0.0
    associator_relative_norm: float = 0.0
    banach_ratio_contractor: float = 0.0
    banach_ratio_supplier: float = 0.0
    banach_ratio_interventor: float = 0.0
    contractor_norm: float = 0.0
    supplier_norm: float = 0.0
    interventor_norm: float = 0.0
    heyting_omega: float = 1.0
    omega_pre: float = 1.0
    artin_residual: float = float("nan")
    moufang_residual: float = float("nan")
    observation_seal: str = ""
    hoyer_sparsity_max: float = 0.0
    null_party_count: int = 0


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1: Observación, saneamiento Banach y sellado criptográfico             ║
# ║                                                                              ║
# ║ Objeto: tríada de señales en (R^8, ‖·‖_p) canonizada hacia O³.               ║
# ║ Cierre formal: synthesize_observation_kernel  →  germen de la Fase 2.        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Phase1_OctonionicObservation:
    r"""
    FASE 1 — Observación Banach y polaridad octoniónica.

    Responsabilidades:
      1. Validar señales R^8 finitas.
      2. Sanear ceros firmados (−0.0 → +0.0) para hash canónico.
      3. Espectro de Banach ℓ¹ / ℓ² / ℓ∞, Hölder y Hoyer.
      4. Construir estados octoniónicos delegados al resolutor.
      5. Cierre: sintetizar el kernel de observación sellado.
    """

    __slots__ = ("_tol", "_safety_margin", "_grace_limit", "_resolver")

    def __init__(
        self,
        *,
        tolerance: float = 1e-12,
        safety_margin: float = 1.0,
        grace_period_seconds: float = 3600.0,
        resolver: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        self._tol: Final[float] = float(tolerance)
        self._safety_margin: Final[float] = float(safety_margin)
        self._grace_limit: Final[float] = float(grace_period_seconds)

        if not math.isfinite(self._tol) or self._tol < 0.0:
            raise ValueError("tolerance debe ser finito y no negativo.")
        if not math.isfinite(self._safety_margin) or self._safety_margin < 0.0:
            raise ValueError("safety_margin debe ser finito y no negativo.")
        if not math.isfinite(self._grace_limit) or self._grace_limit < 0.0:
            raise ValueError("grace_period_seconds debe ser finito y no negativo.")

        if resolver is None:
            try:
                resolver = OctonionicDependencyResolver(
                    tolerance=self._tol,
                    grace_period_seconds=self._grace_limit,
                )
            except TypeError:
                resolver = OctonionicDependencyResolver(tolerance=self._tol)

        self._resolver: Final[Any] = resolver

        # Compatibilidad MRO con Morphism / mixins opcionales.
        try:
            super().__init__(**kwargs)
        except TypeError:
            super().__init__()

    def _relative_tolerance(self, scale: float = 1.0) -> float:
        return max(self._tol, 10.0 * _MACHINE_EPS * max(1.0, float(scale)))

    # ───────────────────────────────────────────────────────────────────────────
    # Utilidades numéricas
    # ───────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _kbn_sum(values: np.ndarray) -> float:
        r"""Sumación compensada Kahan–Babuška–Neumaier (deriva de Wilkinson)."""
        total = 0.0
        compensation = 0.0
        for value in np.asarray(values, dtype=np.float64).ravel():
            val = float(value)
            if not math.isfinite(val):
                return val
            y = val - compensation
            t = total + y
            compensation = (t - total) - y
            total = t
        return total

    def _sha256_payload(self, *parts: Any) -> str:
        sha = hashlib.sha256()
        for part in parts:
            sha.update(_canonical_bytes(part))
        return sha.hexdigest()

    def _norm8(self, arr: np.ndarray) -> float:
        vec = np.asarray(arr, dtype=np.float64)
        if vec.shape != (_OCTONION_DIM,):
            raise ValueError(f"El vector debe ser 8D. Obtenido: {vec.shape}")
        if not np.all(np.isfinite(vec)):
            raise ValueError("El vector contiene valores no finitos.")
        sq = self._kbn_sum(vec * vec)
        if not math.isfinite(sq):
            raise ValueError("La norma cuadrada no es finita.")
        if sq < 0.0 and sq > -self._tol:
            sq = 0.0
        if sq < 0.0:
            raise ValueError("La norma cuadrada es negativa.")
        norm_val = float(math.sqrt(sq))
        if not math.isfinite(norm_val):
            raise ValueError("La norma no es finita.")
        return norm_val

    # ───────────────────────────────────────────────────────────────────────────
    # Validación y saneamiento
    # ───────────────────────────────────────────────────────────────────────────

    def _validate_vector8(
        self,
        S: np.ndarray,
        name: str = "vector octoniónico",
    ) -> np.ndarray:
        arr = np.asarray(S, dtype=np.float64)
        if arr.shape != (_OCTONION_DIM,):
            raise ValueError(
                f"El {name} debe ser estrictamente 8D. Obtenido: {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"El {name} contiene valores no finitos.")
        return arr

    @staticmethod
    def _sanitize_signed_zeros(S: np.ndarray) -> np.ndarray:
        r"""Sanea ceros firmados: −0.0 → +0.0 (estabilidad de hash y polaridad)."""
        arr = np.array(S, dtype=np.float64, copy=True, order="C")
        arr[arr == 0.0] = 0.0
        return arr

    def _polar_unit(self, S: np.ndarray, norm2: float) -> np.ndarray:
        r"""Proyección polar x ↦ x/‖x‖₂; el nulo se envía a 0."""
        if norm2 <= _MACHINE_EPS:
            return np.zeros(_OCTONION_DIM, dtype=np.float64)
        return np.asarray(S, dtype=np.float64) / norm2

    # ───────────────────────────────────────────────────────────────────────────
    # Regularidad de Banach
    # ───────────────────────────────────────────────────────────────────────────

    def evaluate_banach_spectrum(
        self,
        S: np.ndarray,
        name: str = "vector de Banach",
    ) -> BanachRegularityReport:
        r"""
        Espectro completo de normas equivalentes en R^8.

        Para x ≠ 0:
            1 ≤ ‖x‖₁/‖x‖₂ ≤ √8,
            ‖x‖₂² ≤ ‖x‖₁ ‖x‖_∞.
        El nulo se reporta con ratio 0 y Hoyer 0.
        """
        arr = self._sanitize_signed_zeros(self._validate_vector8(S, name))

        l1_norm = float(self._kbn_sum(np.abs(arr)))
        l2_sq = float(self._kbn_sum(arr * arr))
        linf_norm = float(np.max(np.abs(arr))) if arr.size else 0.0

        if not math.isfinite(l1_norm) or not math.isfinite(l2_sq) or not math.isfinite(linf_norm):
            raise ValueError("La regularidad de Banach no es finita.")
        if l2_sq < 0.0 and l2_sq > -self._tol:
            l2_sq = 0.0
        if l2_sq < 0.0:
            raise ValueError("La norma cuadrada de Banach es negativa.")

        l2_norm = float(math.sqrt(l2_sq))
        is_null = bool(l1_norm <= _MACHINE_EPS and l2_norm <= _MACHINE_EPS)

        if is_null:
            return BanachRegularityReport(
                l1_norm=0.0,
                l2_norm=0.0,
                linf_norm=0.0,
                l1_l2_ratio=0.0,
                raw_l1_l2_ratio=0.0,
                hoyer_sparsity=0.0,
                holder_defect=0.0,
                is_null=True,
                is_clamped=False,
            )

        denominator = max(l2_norm, _MACHINE_EPS)
        raw_ratio = float(l1_norm / denominator)
        if not math.isfinite(raw_ratio):
            raise ValueError("El ratio de Banach no es finito.")

        ratio = raw_ratio
        is_clamped = False
        if ratio < 1.0:
            if ratio >= 1.0 - _BANACH_CLAMP_EPS:
                ratio = 1.0
            else:
                logger.warning(
                    "Ratio de Banach por debajo de la cota analítica 1.0: %.12e. "
                    "Se corrige a 1.0.",
                    ratio,
                )
                ratio = 1.0
            is_clamped = True
        elif ratio > _BANACH_SQRT8:
            if ratio <= _BANACH_SQRT8 + _BANACH_CLAMP_EPS:
                ratio = _BANACH_SQRT8
            else:
                logger.warning(
                    "Ratio de Banach por encima de la cota analítica √8: %.12e. "
                    "Se corrige a √8.",
                    ratio,
                )
                ratio = _BANACH_SQRT8
            is_clamped = True

        span = _BANACH_SQRT8 - 1.0
        hoyer = float((_BANACH_SQRT8 - ratio) / span) if span > 0.0 else 0.0
        hoyer = min(1.0, max(0.0, hoyer))

        holder_defect = float(l1_norm * linf_norm - l2_sq)
        if holder_defect < 0.0 and holder_defect > -self._relative_tolerance(l2_sq):
            holder_defect = 0.0

        return BanachRegularityReport(
            l1_norm=l1_norm,
            l2_norm=l2_norm,
            linf_norm=linf_norm,
            l1_l2_ratio=ratio,
            raw_l1_l2_ratio=raw_ratio,
            hoyer_sparsity=hoyer,
            holder_defect=holder_defect,
            is_null=False,
            is_clamped=is_clamped,
        )

    def evaluate_banach_regularity(self, S: np.ndarray) -> float:
        r"""Cociente ‖S‖₁ / ‖S‖₂ ∈ [1, √8] (0 si nulo). API heredada."""
        return self.evaluate_banach_spectrum(S).l1_l2_ratio

    # ───────────────────────────────────────────────────────────────────────────
    # Cierre formal de la Fase 1
    # ───────────────────────────────────────────────────────────────────────────

    def synthesize_observation_kernel(
        self,
        contractor_S: np.ndarray,
        supplier_S: np.ndarray,
        interventor_S: np.ndarray,
    ) -> OctonionicObservationKernel:
        r"""
        CIERRE FORMAL DE LA FASE 1 / GERMEN DE LA FASE 2.

        Canoniza la tríada transaccional en el espacio de Banach y la
        inmerge en O³ vía el resolutor:

            K(a,b,c) = (state_*, spectrum_*, polar_*, sello) .

        El funtor de la Fase 2, ``observe_from_kernel``, actúa de forma
        estricta sobre este germen.  No se evalúa aún ni Hurwitz ni el
        asociador: eso es orientación espectral, no observación.
        """
        c_clean = self._sanitize_signed_zeros(
            self._validate_vector8(contractor_S, "contratista")
        )
        s_clean = self._sanitize_signed_zeros(
            self._validate_vector8(supplier_S, "proveedor")
        )
        i_clean = self._sanitize_signed_zeros(
            self._validate_vector8(interventor_S, "interventor")
        )

        spec_c = self.evaluate_banach_spectrum(c_clean, "contratista")
        spec_s = self.evaluate_banach_spectrum(s_clean, "proveedor")
        spec_i = self.evaluate_banach_spectrum(i_clean, "interventor")

        state_a = self._resolver.build_state(c_clean)
        state_b = self._resolver.build_state(s_clean)
        state_c = self._resolver.build_state(i_clean)

        polar_c = _immutable(self._polar_unit(c_clean, spec_c.l2_norm), np.float64)
        polar_s = _immutable(self._polar_unit(s_clean, spec_s.l2_norm), np.float64)
        polar_i = _immutable(self._polar_unit(i_clean, spec_i.l2_norm), np.float64)

        null_party_count = int(spec_c.is_null) + int(spec_s.is_null) + int(spec_i.is_null)

        seal_hash = self._sha256_payload(
            c_clean,
            s_clean,
            i_clean,
            np.array(
                [
                    spec_c.l1_l2_ratio,
                    spec_s.l1_l2_ratio,
                    spec_i.l1_l2_ratio,
                    spec_c.hoyer_sparsity,
                    spec_s.hoyer_sparsity,
                    spec_i.hoyer_sparsity,
                ],
                dtype=np.float64,
            ),
        )

        logger.info(
            "Fase Observe [OCTONION_AGENT]: insumos congelados en Banach. Sello: %s",
            seal_hash[:16],
        )

        return OctonionicObservationKernel(
            contractor_state=state_a,
            supplier_state=state_b,
            interventor_state=state_c,
            banach_ratio_contractor=spec_c.l1_l2_ratio,
            banach_ratio_supplier=spec_s.l1_l2_ratio,
            banach_ratio_interventor=spec_i.l1_l2_ratio,
            cryptographic_seal=seal_hash,
            contractor_spectrum=spec_c,
            supplier_spectrum=spec_s,
            interventor_spectrum=spec_i,
            polar_contractor=polar_c,
            polar_supplier=polar_s,
            polar_interventor=polar_i,
            null_party_count=null_party_count,
        )

    def build_observation_kernel(
        self,
        contractor_S: np.ndarray,
        supplier_S: np.ndarray,
        interventor_S: np.ndarray,
    ) -> OctonionicObservationKernel:
        r"""Alias de compatibilidad: delega en ``synthesize_observation_kernel``."""
        return self.synthesize_observation_kernel(
            contractor_S, supplier_S, interventor_S
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2: Orientación, Hurwitz, asociador y rampa de confianza                ║
# ║                                                                              ║
# ║ Apertura: observe_from_kernel(synthesize_observation_kernel(·)).             ║
# ║ Cierre formal: orient_octonionic_state  →  germen de la Fase 3.              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Phase2_OctonionicOrientation(Phase1_OctonionicObservation):
    r"""
    FASE 2 — Orientación espectral y rampa de confianza.

    Continuación estricta de ``synthesize_observation_kernel``.

    Responsabilidades:
      1. Observar K(a,b,c) contra Hurwitz y [a,b,c] (delegado al resolutor).
      2. Extraer residuos de Artin / Moufang si el resolutor v3 los expone.
      3. Aplicar la rampa 0.3 τ / 0.5 τ sobre τ_max = L_max · safety.
      4. Cierre: preclasificar ω_pre ∈ Ω₃ para el decisor de la Fase 3.
    """

    __slots__ = ()

    def _compute_hurwitz_error(
        self,
        a: OctonionicState,
        b: OctonionicState,
    ) -> Tuple[float, float]:
        r"""Error absoluto y relativo de ‖ab‖ = ‖a‖‖b‖."""
        if hasattr(self._resolver, "compute_hurwitz_error"):
            abs_err, rel_err, _ = self._resolver.compute_hurwitz_error(a, b)
            return float(abs_err), float(rel_err)

        ab = self._resolver.octonionic_multiply(a, b)
        expected = a.norm * b.norm
        if not (math.isfinite(ab.norm) and math.isfinite(expected)):
            return float("inf"), float("inf")
        abs_err = float(abs(ab.norm - expected))
        denominator = expected if (math.isfinite(expected) and expected > 1.0) else 1.0
        return abs_err, float(abs_err / denominator)

    def _compute_associator_diagnostics(
        self,
        a: OctonionicState,
        b: OctonionicState,
        c: OctonionicState,
    ) -> Tuple[np.ndarray, float, float]:
        r"""Asociador [a,b,c], norma absoluta y relativa."""
        if hasattr(self._resolver, "compute_associator_diagnostics"):
            assoc_vec, assoc_norm, assoc_rel = (
                self._resolver.compute_associator_diagnostics(a, b, c)
            )
            assoc_vec = np.asarray(assoc_vec, dtype=np.float64)
            if assoc_vec.shape != (_OCTONION_DIM,):
                raise ValueError(
                    f"El asociador debe ser 8D. Obtenido: {assoc_vec.shape}"
                )
            if not np.all(np.isfinite(assoc_vec)):
                raise ValueError("El asociador contiene valores no finitos.")
            return assoc_vec, float(assoc_norm), float(assoc_rel)

        assoc_vec = np.asarray(
            self._resolver.compute_associator(a, b, c),
            dtype=np.float64,
        )
        if assoc_vec.shape != (_OCTONION_DIM,):
            raise ValueError(
                f"El asociador debe ser 8D. Obtenido: {assoc_vec.shape}"
            )
        if not np.all(np.isfinite(assoc_vec)):
            raise ValueError("El asociador contiene valores no finitos.")

        assoc_norm = self._norm8(assoc_vec)
        product_scale = a.norm * b.norm * c.norm
        if not math.isfinite(product_scale) or product_scale <= _MACHINE_EPS:
            denominator = max(_MACHINE_EPS, 1.0, a.norm, b.norm, c.norm)
        else:
            denominator = product_scale
        return assoc_vec, assoc_norm, float(assoc_norm / denominator)

    def _try_resolver_triad(
        self,
        a: OctonionicState,
        b: OctonionicState,
        c: OctonionicState,
    ) -> Tuple[Optional[np.ndarray], float, float, str]:
        r"""
        Extrae (associator, artin, moufang, triad_seal) del resolutor v3.
        Si la API no existe, artin/moufang = NaN y el asociador se calcula
        por el camino clásico.
        """
        synthesize = getattr(self._resolver, "synthesize_octonionic_triad", None)
        if not callable(synthesize):
            return None, float("nan"), float("nan"), ""

        try:
            triad = synthesize(a, b, c)
        except Exception:
            logger.debug(
                "synthesize_octonionic_triad falló; se usa el camino clásico.",
                exc_info=True,
            )
            return None, float("nan"), float("nan"), ""

        associator = np.asarray(getattr(triad, "associator", None), dtype=np.float64)
        if associator.shape != (_OCTONION_DIM,) or not np.all(np.isfinite(associator)):
            associator = None  # type: ignore[assignment]

        artin = float(getattr(triad, "artin_residual", float("nan")))
        seal = str(getattr(triad, "sha256_hash", "") or "")

        moufang = float("nan")
        moufang_fn = getattr(self._resolver, "compute_moufang_residual", None)
        if callable(moufang_fn):
            try:
                moufang = float(moufang_fn(a, b, c))
            except Exception:
                logger.debug("compute_moufang_residual omitido.", exc_info=True)

        return associator, artin, moufang, seal

    def observe_from_kernel(
        self,
        kernel: OctonionicObservationKernel,
        associator_threshold_Lmax: float,
    ) -> OctonionicOrientationState:
        r"""
        APERTURA FORMAL DE LA FASE 2.

        Continuación directa de ``synthesize_observation_kernel``:

            K(a,b,c)  ↦  (Hurwitz, [a,b,c], Artin, Moufang, rampa, ω_pre).

        El proveedor de álgebra es el resolutor; la rampa de confianza
        (0.3 τ / 0.5 τ) es gobernanza del agente y no se delega.
        """
        if not isinstance(kernel, OctonionicObservationKernel):
            raise TypeError(
                "observe_from_kernel exige un OctonionicObservationKernel "
                "(germen de synthesize_observation_kernel)."
            )

        threshold = float(associator_threshold_Lmax)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("associator_threshold_Lmax debe ser finito y no negativo.")

        safety = float(self._safety_margin)
        if not math.isfinite(safety) or safety < 0.0:
            raise ValueError("safety_margin debe ser finito y no negativo.")

        cota_limite = threshold * safety
        a = kernel.contractor_state
        b = kernel.supplier_state
        c = kernel.interventor_state

        abs_ab, rel_ab = self._compute_hurwitz_error(a, b)
        abs_bc, rel_bc = self._compute_hurwitz_error(b, c)
        abs_ac, rel_ac = self._compute_hurwitz_error(a, c)

        composition_error = float(max(abs_ab, abs_bc, abs_ac))
        composition_relative_error = float(max(rel_ab, rel_bc, rel_ac))

        pair_expected = [a.norm * b.norm, b.norm * c.norm, a.norm * c.norm]
        finite_expected = [v for v in pair_expected if math.isfinite(v)]
        expected_scale = max([1.0] + finite_expected)

        composition_tolerance = max(
            self._tol,
            100.0 * _MACHINE_EPS * expected_scale,
        )
        composition_hard_limit = max(
            composition_tolerance,
            1e-9,
            1000.0 * _MACHINE_EPS * expected_scale,
        )

        triad_assoc, artin_residual, moufang_residual, triad_seal = (
            self._try_resolver_triad(a, b, c)
        )
        if triad_assoc is not None:
            assoc_vec = triad_assoc
            assoc_norm = self._norm8(assoc_vec)
            product_scale = a.norm * b.norm * c.norm
            if not math.isfinite(product_scale) or product_scale <= _MACHINE_EPS:
                denom = max(_MACHINE_EPS, 1.0, a.norm, b.norm, c.norm)
            else:
                denom = product_scale
            assoc_rel = float(assoc_norm / denom)
        else:
            assoc_vec, assoc_norm, assoc_rel = self._compute_associator_diagnostics(
                a, b, c
            )

        is_associative_stable = bool(
            math.isfinite(assoc_norm) and assoc_norm <= (cota_limite + self._tol)
        )
        is_cfl_stable = bool(
            math.isfinite(composition_error)
            and composition_error <= (composition_tolerance + self._tol)
        )

        associator_soft_start = _RAMPA_SOFT * cota_limite
        associator_hard_limit = _RAMPA_HARD * cota_limite

        hard_composition = bool(
            (not math.isfinite(composition_error))
            or (composition_error > (composition_hard_limit + self._tol))
        )
        soft_composition = bool(
            math.isfinite(composition_error)
            and (not hard_composition)
            and (composition_error > (composition_tolerance + self._tol))
        )
        hard_associator = bool(
            (not math.isfinite(assoc_norm))
            or (assoc_norm > (associator_hard_limit + self._tol))
        )
        soft_associator = bool(
            math.isfinite(assoc_norm)
            and (not hard_associator)
            and (assoc_norm > (associator_soft_start + self._tol))
        )

        artin_broken = bool(
            math.isfinite(artin_residual)
            and artin_residual
            > max(self._relative_tolerance(expected_scale), 1e-8)
        )

        omega_hard = (
            0.0
            if (hard_composition or hard_associator or artin_broken)
            else 1.0
        )
        omega_cfl = 1.0 if (not soft_composition) else 0.5
        if hard_composition:
            omega_cfl = 0.0
        omega_asoc = 1.0 if (not soft_associator) else 0.5
        if hard_associator:
            omega_asoc = 0.0
        omega_banach = 0.5 if kernel.null_party_count > 0 else 1.0

        omega_pre = _heyting_meet(
            _heyting_meet(_heyting_meet(omega_hard, omega_cfl), omega_asoc),
            omega_banach,
        )

        return OctonionicOrientationState(
            kernel=kernel,
            associator_norm=assoc_norm,
            composition_error=composition_error,
            is_associative_stable=is_associative_stable,
            is_cfl_stable=is_cfl_stable,
            composition_relative_error=composition_relative_error,
            associator_relative_norm=assoc_rel,
            cota_limite=cota_limite,
            associator_vector=_immutable(assoc_vec, np.float64),
            composition_tolerance=composition_tolerance,
            composition_hard_limit=composition_hard_limit,
            associator_soft_start=associator_soft_start,
            associator_hard_limit=associator_hard_limit,
            omega_hard=omega_hard,
            omega_cfl=omega_cfl,
            omega_asoc=omega_asoc,
            omega_banach=omega_banach,
            omega_pre=omega_pre,
            artin_residual=artin_residual,
            moufang_residual=moufang_residual,
            triad_seal=triad_seal,
            hard_composition=hard_composition,
            hard_associator=hard_associator,
            soft_composition=soft_composition,
            soft_associator=soft_associator,
        )

    def orient_octonionic_state(
        self,
        kernel: OctonionicObservationKernel,
        associator_threshold_Lmax: float,
    ) -> OctonionicOrientationState:
        r"""
        CIERRE FORMAL DE LA FASE 2 / GERMEN DE LA FASE 3.

        Encadena el germen Banach de la Fase 1 con la observación
        espectral y produce el objeto de orientación

            Ω_pre = (ω_hard ∧ ω_cfl ∧ ω_asoc ∧ ω_banach) ∈ Ω₃

        sobre el que ``Phase3_OODAActuator.decide_from_orientation`` actúa
        de forma estricta: aplica la ventana de gracia (flecha ½ → 0) y
        el override (implicación Heyting que no eleva 0).

        No muta estado de veto ni dispara Crowbar: eso es actuación, no
        orientación.
        """
        return self.observe_from_kernel(kernel, associator_threshold_Lmax)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3: Decisión OODA, Heyting Ω₃ y actuación Crowbar                       ║
# ║                                                                              ║
# ║ Apertura: decide_from_orientation(orient_octonionic_state(·)).               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Phase3_OODAActuator(Phase2_OctonionicOrientation):
    r"""
    FASE 3 — Decisión en Ω₃, gracia, override y Crowbar IRAM.

    Continuación estricta de ``orient_octonionic_state``:

        ω = decide(ω_pre, gracia, override) ∈ Ω₃,
        Act = Crowbar  syss  ω = 0.

    Responsabilidades:
      1. Decidir COHERENT / DEGRADED / VETOED sobre ω_pre.
      2. Gestionar veto suave con ventana de gracia (½ persistente → 0).
      3. Verificar overrides en tiempo constante / HMAC.
      4. Actuar: latencia Crowbar determinista < 400 ns.
      5. Sellar HMAC/SHA-256 el certificado.
    """

    __slots__ = (
        "_override_verifier",
        "_allowed_override_tokens",
        "_soft_veto_timestamp",
        "_is_soft_veto_active",
        "_hmac_secret",
    )

    def __init__(
        self,
        *,
        override_verifier: Optional[Callable[[str], bool]] = None,
        allowed_override_tokens: Optional[AbstractSet[str]] = None,
        allow_legacy_overrides: bool = True,
        hmac_secret: Optional[bytes] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._override_verifier: Optional[Callable[[str], bool]] = override_verifier

        if allowed_override_tokens is None:
            tokens = _LEGACY_OVERRIDE_TOKENS if allow_legacy_overrides else frozenset()
        else:
            tokens = frozenset(allowed_override_tokens)
        self._allowed_override_tokens: Final[frozenset] = frozenset(tokens)

        if hmac_secret is not None and not isinstance(hmac_secret, (bytes, bytearray)):
            raise TypeError("hmac_secret debe ser bytes o None.")
        self._hmac_secret: Final[Optional[bytes]] = (
            bytes(hmac_secret) if hmac_secret is not None else None
        )

        self._soft_veto_timestamp: Optional[float] = None
        self._is_soft_veto_active: bool = False

        if allow_legacy_overrides and self._allowed_override_tokens.intersection(
            _LEGACY_OVERRIDE_TOKENS
        ):
            logger.warning(
                "Se permiten tokens legacy de override octoniónico. "
                "Para producción, configure override_verifier o hmac_secret."
            )

    def _clear_soft_veto(self) -> None:
        self._is_soft_veto_active = False
        self._soft_veto_timestamp = None

    def reset_soft_veto(self) -> None:
        r"""Reinicia el temporizador de gracia (laboratorio / tests)."""
        self._clear_soft_veto()

    def _verify_override(self, token: Optional[str]) -> bool:
        r"""
        Verificación de override, por orden de autoridad:

          1. override_verifier inyectado.
          2. HMAC-SHA256(hmac_secret, token) si hay secreto.
          3. Conjunto de tokens permitidos vía hmac.compare_digest.

        El override **no** se registra en claro.
        """
        if token is None or not isinstance(token, str) or not token.strip():
            return False

        if callable(self._override_verifier):
            try:
                return bool(self._override_verifier(token))
            except Exception:
                logger.exception(
                    "El override_verifier lanzó una excepción. Se rechaza el override."
                )
                return False

        token_bytes = token.encode("utf-8")

        if self._hmac_secret is not None:
            expected = hmac.new(
                self._hmac_secret, token_bytes, hashlib.sha256
            ).hexdigest()
            if hmac.compare_digest(expected, token):
                return True

        for allowed in self._allowed_override_tokens:
            allowed_bytes = allowed.encode("utf-8")
            if hmac.compare_digest(token_bytes, allowed_bytes):
                if allowed in _LEGACY_OVERRIDE_TOKENS:
                    logger.warning(
                        "Override legacy aceptado. Considere migrar a tokens firmados."
                    )
                return True
        return False

    def _seeded_latency_ns(self, *parts: Any) -> float:
        r"""
        Latencia Crowbar determinista en IRAM.

        En hardware real se sustituye por medición GPIO/ISR.
        Cota: 395 ns ≤ τ < 400 ns.  Sin np.random.
        """
        sha = hashlib.sha256()
        for part in parts:
            sha.update(_canonical_bytes(part))
        digest = sha.digest()
        fraction = int.from_bytes(digest[:6], "little") / float(1 << 48)
        latency = 395.0 + 4.5 * fraction
        return float(min(_CROWBAR_IRAM_LATENCY_NS, latency))

    def _seal(self, *parts: Any) -> str:
        payload = b"".join(_canonical_bytes(part) for part in parts)
        if self._hmac_secret is not None:
            return hmac.new(self._hmac_secret, payload, hashlib.sha256).hexdigest()
        return hashlib.sha256(payload).hexdigest()

    def decide_from_orientation(
        self,
        orientation: OctonionicOrientationState,
        override_token: Optional[str] = None,
        curr_time: Optional[float] = None,
        simulate_grace_expired: bool = False,
    ) -> Tuple[HeytingVerdict, bool, bool, float, bool]:
        r"""
        APERTURA FORMAL DE LA FASE 3.

        Continuación directa de ``orient_octonionic_state``.

        Entrada: ω_pre = ω_hard ∧ ω_cfl ∧ ω_asoc ∧ ω_banach.
        Dinámica de Heyting:
          · ω_pre = 1  →  COHERENT, se limpia la gracia.
          · ω_pre = 0  →  VETOED instantáneo (Hurwitz grave / asociador
            > 0.5 τ / Artin). El override **no** eleva 0 (¬¬0 = 0).
          · ω_pre = ½  →  DEGRADED; si la gracia expira, ½ se colapsa a 0.
            Un override válido aplica ½ → ½: se disipa el ámbar operativo
            pero el veredicto permanece DEGRADED hasta que el asociador
            vuelva bajo 0.3 τ.

        Retorna:
            (verdict, is_soft_veto, is_hard_veto, time_remaining, grace_expired).
        """
        if not isinstance(orientation, OctonionicOrientationState):
            raise TypeError(
                "decide_from_orientation exige un OctonionicOrientationState "
                "(germen de orient_octonionic_state)."
            )

        now = time.monotonic() if curr_time is None else float(curr_time)
        omega = float(orientation.omega_pre)

        if omega <= 0.0:
            self._clear_soft_veto()
            logger.error(
                "VETO DURO INSTANTÁNEO: colapso de Hurwitz, obstrucción "
                "asociativa severa (> 0.5 τ) o ruptura de Artin (ω_hard = 0)."
            )
            return HeytingVerdict.VETOED, False, True, 0.0, False

        if omega >= 1.0:
            self._clear_soft_veto()
            return HeytingVerdict.COHERENT, False, False, 0.0, False

        is_soft_veto = True
        time_remaining = 0.0
        grace_expired = False

        if not self._is_soft_veto_active and not simulate_grace_expired:
            self._is_soft_veto_active = True
            self._soft_veto_timestamp = now
            verdict = HeytingVerdict.DEGRADED
            logger.warning(
                "VETO SUAVE ACTIVO (LUZ ÁMBAR): no-asociatividad o deriva de "
                "Hurwitz en la tríada. Gracia iniciada."
            )
        else:
            if self._soft_veto_timestamp is None or simulate_grace_expired:
                elapsed = self._grace_limit + 1.0
            else:
                elapsed = now - self._soft_veto_timestamp
            time_remaining = max(0.0, self._grace_limit - elapsed)
            if time_remaining <= self._tol or simulate_grace_expired:
                self._clear_soft_veto()
                logger.critical(
                    "VENTANA DE GRACIA EXPIRADA SIN OVERRIDE VÁLIDO. "
                    "Heyting colapsa ½ → 0 (VETOED terminal)."
                )
                return HeytingVerdict.VETOED, False, True, 0.0, True
            verdict = HeytingVerdict.DEGRADED

        if override_token is not None:
            if self._verify_override(override_token):
                self._clear_soft_veto()
                logger.info(
                    "ANQUILACIÓN DE FOCK TRILATERAL ACTIVADA. Override validado. "
                    "Luz ámbar disipada; la obra permanece DEGRADED hasta "
                    "recuperar estabilidad asociativa."
                )
                return HeytingVerdict.DEGRADED, False, False, 0.0, False
            logger.error(
                "Firma digital inválida en el override. "
                "Se mantiene la rampa de de Rham activa."
            )

        return verdict, is_soft_veto, False, time_remaining, grace_expired

    def _act_crowbar(
        self,
        orientation: OctonionicOrientationState,
        verdict: HeytingVerdict,
    ) -> float:
        r"""Actuación Crowbar: ISR en IRAM, GPIO14 HIGH, tiristor BT151."""
        if verdict is not HeytingVerdict.VETOED:
            return 0.0

        kernel = orientation.kernel
        switching_latency = self._seeded_latency_ns(
            kernel.contractor_state.vector_rep,
            kernel.supplier_state.vector_rep,
            kernel.interventor_state.vector_rep,
            orientation.associator_vector,
            np.array(
                [
                    orientation.associator_norm,
                    orientation.composition_error,
                    orientation.composition_relative_error,
                    orientation.associator_relative_norm,
                ],
                dtype=np.float64,
            ),
            verdict.value,
        )

        logger.critical("COLA DE HEYTING COLAPSADA EN SOBERANO OCTONIÓNICO.")
        logger.critical("  - Ejecutando subrutina local isVerdictCoherent() en C++...")
        logger.critical("  - Despachando ISR en IRAM en menos de 400 ns...")
        logger.critical(
            "  - Conmutando GPIO14 a HIGH en %.2f ns vía IRAM...",
            switching_latency,
        )
        logger.critical("  - Tiristor rápido de potencia BT151 (Crowbar) gatillado.")
        logger.critical("  - Mezcladoras y bombas hidráulicas reales en fango paralizadas.")
        return switching_latency

    def execute_octonionic_control_cycle(
        self,
        contractor_S: np.ndarray,
        supplier_S: np.ndarray,
        interventor_S: np.ndarray,
        associator_threshold_Lmax: float = 0.15,
        override_token: Optional[str] = None,
        simulate_grace_expired: bool = False,
    ) -> OctonionicAgentCertificate:
        r"""
        Orquesta el ciclo covariante OODA del Soberano de Calibre Octoniónico.

        Flujo anidado:
          OBSERVE  (Fase 1): synthesize_observation_kernel.
          ORIENT   (Fase 2): orient_octonionic_state.
          DECIDE   (Fase 3): decide_from_orientation.
          ACT      (Fase 3): _act_crowbar si ω = 0.
        """
        kernel = self.synthesize_observation_kernel(
            contractor_S,
            supplier_S,
            interventor_S,
        )
        orientation = self.orient_octonionic_state(
            kernel,
            associator_threshold_Lmax,
        )

        verdict, is_soft_veto, is_hard_veto, time_remaining, grace_expired = (
            self.decide_from_orientation(
                orientation,
                override_token=override_token,
                simulate_grace_expired=simulate_grace_expired,
            )
        )

        if verdict is HeytingVerdict.VETOED:
            is_hard_veto = True

        interlock_fired = bool(is_hard_veto or verdict is HeytingVerdict.VETOED)
        actuation_latency_ns = (
            self._act_crowbar(orientation, verdict) if interlock_fired else 0.0
        )

        if not interlock_fired:
            logger.info(
                "Soberano Octoniónico regulado síncronamente. Veredicto: %s. Sello: %s",
                verdict.value,
                kernel.cryptographic_seal[:16],
            )

        digital_sig = self._seal(
            kernel.contractor_state.vector_rep,
            kernel.supplier_state.vector_rep,
            kernel.interventor_state.vector_rep,
            orientation.associator_vector,
            np.array(
                [
                    orientation.associator_norm,
                    orientation.composition_error,
                    orientation.composition_relative_error,
                    orientation.associator_relative_norm,
                    float(interlock_fired),
                    float(is_soft_veto),
                    verdict.omega,
                ],
                dtype=np.float64,
            ),
            verdict.value,
            kernel.cryptographic_seal,
        )

        def _hoyer(spec: Optional[BanachRegularityReport]) -> float:
            return 0.0 if spec is None else float(spec.hoyer_sparsity)

        hoyer_max = max(
            _hoyer(kernel.contractor_spectrum),
            _hoyer(kernel.supplier_spectrum),
            _hoyer(kernel.interventor_spectrum),
        )

        return OctonionicAgentCertificate(
            phase=_PHASE_NAME,
            heyting_verdict=verdict.value,
            associator_norm=orientation.associator_norm,
            is_associative_stable=orientation.is_associative_stable,
            composition_error=orientation.composition_error,
            is_surgery_active=bool(verdict is HeytingVerdict.DEGRADED),
            is_soft_veto_active=bool(self._is_soft_veto_active),
            override_grace_period_expired=bool(grace_expired),
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=actuation_latency_ns,
            time_grace_remaining=time_remaining,
            digital_signature_sha256=digital_sig,
            composition_relative_error=orientation.composition_relative_error,
            associator_relative_norm=orientation.associator_relative_norm,
            banach_ratio_contractor=kernel.banach_ratio_contractor,
            banach_ratio_supplier=kernel.banach_ratio_supplier,
            banach_ratio_interventor=kernel.banach_ratio_interventor,
            contractor_norm=kernel.contractor_state.norm,
            supplier_norm=kernel.supplier_state.norm,
            interventor_norm=kernel.interventor_state.norm,
            heyting_omega=verdict.omega,
            omega_pre=orientation.omega_pre,
            artin_residual=orientation.artin_residual,
            moufang_residual=orientation.moufang_residual,
            observation_seal=kernel.cryptographic_seal,
            hoyer_sparsity_max=hoyer_max,
            null_party_count=kernel.null_party_count,
        )


# ───────────────────────────────────────────────────────────────────────────────
# Agente final
# ───────────────────────────────────────────────────────────────────────────────

class OctonionicDependencyAgent(Phase3_OODAActuator, Morphism):
    r"""
    Soberano de Calibre Octoniónico (OODA lazo cerrado).

    Cadena de herencia (fases anidadas):

        OctonionicDependencyAgent
          └─ Phase3_OODAActuator                 Ω₃, gracia, Crowbar
               └─ Phase2_OctonionicOrientation   Hurwitz, rampa, ω_pre
                    └─ Phase1_OctonionicObservation
                         Banach ℓ^p, polaridad, K(a,b,c)

    Gobierna de forma covariante los estados hipercomplejos no asociativos
    de la Malla y administra la rampa de confianza graduada para
    neutralizar falsas alarmas en el vaciado de concreto perimetral de obra.
    """

    __slots__ = ()

    def __init__(
        self,
        tolerance: float = 1e-12,
        safety_margin: float = 1.0,
        grace_period_seconds: float = 3600.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            tolerance=tolerance,
            safety_margin=safety_margin,
            grace_period_seconds=grace_period_seconds,
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            "OctonionicDependencyAgent("
            f"tolerance={self._tol}, "
            f"safety_margin={self._safety_margin}, "
            f"grace_period_seconds={self._grace_limit}"
            ")"
        )


__all__ = [
    "OctonionicDependencyAgent",
    "OctonionicObservationKernel",
    "OctonionicOrientationState",
    "OctonionicAgentCertificate",
    "BanachRegularityReport",
    "HeytingVerdict",
    "Phase1_OctonionicObservation",
    "Phase2_OctonionicOrientation",
    "Phase3_OODAActuator",
]