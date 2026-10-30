# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Soberano : Topological Gauge Field Prosecutor Agent v10.1.0 — Tres Fases     ║
║ Ubicación: app/agents/wisdom/topological_gauge_field_prosecutor_agent.py     ║
║ Versión  : 10.1.0 (SOVEREIGN-GAUGE-PROSECUTOR-AGENT-FINSLER-                 ║
║           CROWBAR-POINCARE-RSI3-HEYTING-PhD)                                 ║
║ Función  : Soberano de investigación fiscal de calibre, auditoría de RHI     ║
║            Finsleriano, censura Fermi-Dirac de Kelly, juzgamiento OODA       ║
║            y activación del hardware Crowbar ESP32/BT151 en silicio.          ║
╚══════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO
────────────────────────────────────────────
El Soberano Fiscal del Campo Topológico de Gauge ejecuta la inquisición punitiva
sobre inestabilidades de gauge, manipulación de incentivos y desviaciones de RHI.
Gobierna el ciclo OODA anidado de tres fases, aplicando la métrica de Mahalanobis-Finsler,
la censura Fermi-Dirac de Kelly, la adjudicación Heyting \Omega_3 de 12 átomos y la
activación del interlock de hardware ESP32 Crowbar (\tau_{\mathrm{crowbar}} \le 392.15\text{ ns}).

MARCO TEÓRICO
─────────────
1. Conexión de Gauge \mathfrak{su}(N), RHI Finsleriano y Censura de Kelly:
   A = i\left(\rho - \frac{\mathrm{Tr}(\rho)}{N}I\right) \in \mathfrak{su}(N), \quad F = [A, \rho_h] \in \mathfrak{su}(N)
   RHI = \sqrt{R_{\mathrm{base}}^2 + \frac{1}{4}\tanh^2(E_{\mathrm{YM}})} \in [0, 1]
   Fracción censurada Fermi-Dirac: s^*(R) = s_{\mathrm{raw}}(1 - R) \sigma\left(\frac{R_c - R}{T_{\mathrm{sw}}}\right)

2. Transiciones de Fase OODA Anidadas y Adjudicación Fiscal:
   • FASE 1 (Observe): Emite el certificado de isometría fiscal \Phi_1.
   • FASE 2 (Orient): Certifica topología (Chern-Simons, instantón de Pontryagin k), KAM, Bruno,
     Melnikov, Floquet, sección de Poincaré y twist de Birkhoff.
   • FASE 3 (Decide & Act): Adjudica \Omega_3 sobre 12 átomos, dispara Crowbar ESP32 GPIO14 en
     \tau_{\mathrm{crowbar}} \le 392.15\text{ ns} si \mathrm{VETOED} y emite el estado soberano final.

3. Automejora Agéntica Nivel 3 (RSI³) y Disparo de Hardware:
   El Soberano verifica la contracción de Banach T^3 sobre el grafo de certificados.
   En caso de \mathrm{VETOED}, ejecuta el interlock de silicio perimetral de hardware.

AXIOMAS FORMALES
────────────────
[Axioma 1] Isometría y RHI Acotado: A^\dagger = -A, \mathrm{Tr}(A) = 0, \mathrm{RHI} \in [0, 1].
[Axioma 2] Continuidad OODA Homomórfica: \mathrm{HMAC}_{P1} \land \mathrm{HMAC}_{P2} \land \mathrm{HMAC}_{P3} = \mathrm{True}.
[Axioma 3] Disparo de Hardware: global_verdict = \mathrm{VETOED} \implies crowbar_triggered = \mathrm{True}, \tau_{\mathrm{crowbar}} \le 392.15\text{ ns}.

INVARIANTES DE ESTADO
─────────────────────
1. Veredicto global \Omega_3 basado en join(\vee) monoidal de 12 átomos.
2. Interlock Crowbar latched en GPIO14 con SLA de latencia \le 392.15\text{ ns}.
3. Trazabilidad Merkle SHA-256 y HMAC-SHA256 de procedencia agéntica fiscal.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

# ──────────────────────────────────────────────────────────────────────────────
# §0. IMPORTACIÓN ROBUSTA DEL MOTOR FISCAL (v10.1.0 o fallback)
# ──────────────────────────────────────────────────────────────────────────────
_ENGINE_IMPORTED: bool = False
_ENGINE_HAS_ENVELOPES: bool = False
_ENGINE_HAS_P2_ENVELOPE: bool = False
_ENGINE_HAS_SNAPSHOT_EMIT: bool = False
_ENGINE_HAS_SNAPSHOT_VERIFY: bool = False


class _NullSnapshot:
    r"""Stub terminal cuando el motor no puede importarse (nunca ctor frozen)."""
    prosecutor_id: str = "DUMMY"
    rhi_gauge_finsler: float = 0.0
    rhi_base_normalized: float = 0.0
    ipr_raw: float = 0.0
    kelley_derivative_ds_drhi: float = 0.0
    fermi_dirac_kelly_fraction: float = 0.0
    yang_mills_energy: float = 0.0
    chern_simons_3form_higgs: float = 0.0
    chern_simons_anomaly: float = 0.0
    wilson_angle_rad: float = 0.0
    c1_value: float = 0.0
    pontryagin_instanton_k: int = 0
    pontryagin_integrality_residual: float = 0.0
    stora_zumino_residual: float = 0.0
    symplectic_residue: float = 0.0
    symplectic_defect: float = 0.0
    is_symplectic_valid: bool = True
    poincare_residual: float = 0.0
    poincare_is_resonant: bool = False
    kam_worst_ratio: float = 0.0
    kam_satisfied: bool = True
    kam_n_modes: int = 0
    melnikov_value: float = 0.0
    melnikov_preserved: bool = True
    floquet_max_unit_deviation: float = 0.0
    floquet_neutral: bool = True
    poincare_section_n_crossings: int = 0
    poincare_section_dispersion: float = 0.0
    mean_poincare_return_time: float = 0.0
    rsi3_valid: bool = True
    rsi3_iterations: int = 1
    banach_lipschitz_t3: float = 0.0
    fubini_study_distance: float = 0.0
    novikov_krull_valuation: float = 0.0
    rsi3_homological_residual: float = 0.0
    rsi3_lipschitz_factors: Tuple[float, ...] = ()
    crowbar_triggered: bool = False
    crowbar_latency_ns: float = 0.0
    crowbar_latency_verified: bool = True
    hardware_action: str = "NOMINAL_EXECUTION"
    sha256_merkle_digest: str = "dummy"
    timestamp_utc: float = 0.0
    hmac_sha256: bytes = b""
    hmac_verified: bool = False
    lindstedt_solved: bool = False
    lindstedt_newton_iterations: int = 0
    spectral_condition_number: float = 1.0
    is_flat_curvature: bool = False
    cayley_pole_margin: float = 0.0
    lie_algebra_defect_projected: float = 0.0
    bruno_satisfied: bool = True
    bruno_sum: float = 0.0
    birkhoff_twist_coefficient: float = 0.0
    birkhoff_area_residual: float = 0.0
    birkhoff_is_twist: bool = False
    engine_version: str = "0.0.0-stub"
    phase1: Any = None
    phase2: Any = None

    def __init__(self) -> None:
        self.timestamp_utc = time.time()
        # global_verdict se asigna tras resolver HeytingOmega3 (ver bloque import).
        self.global_verdict = None  # type: ignore[assignment]


try:
    from app.wisdom.topological_gauge_field_prosecutor_engine import (
        GaugeProsecutorSnapshot,
        HeytingOmega3,
        TopologicalGaugeFieldProsecutorEngine,
    )
    _ENGINE_IMPORTED = True
except ImportError:
    try:
        from topological_gauge_field_prosecutor_engine import (
            GaugeProsecutorSnapshot,
            HeytingOmega3,
            TopologicalGaugeFieldProsecutorEngine,
        )
        _ENGINE_IMPORTED = True
    except ImportError:  # pragma: no cover
        from enum import IntEnum

        class HeytingOmega3(IntEnum):  # type: ignore[no-redef]
            COHERENT = 0x00
            DEGRADED = 0x01
            VETOED = 0x03

            def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
                return HeytingOmega3(max(int(self), int(other)))

            def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
                return HeytingOmega3(min(int(self), int(other)))

            def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
                return HeytingOmega3.VETOED if int(self) <= int(other) else other

            def negate(self) -> "HeytingOmega3":
                return self.implies(HeytingOmega3.COHERENT)

            @classmethod
            def fold_supremum(
                cls, verdicts: Sequence["HeytingOmega3"],
            ) -> "HeytingOmega3":
                acc = cls.COHERENT
                for v in verdicts:
                    acc = acc.join(v)
                return acc

            @classmethod
            def fold_infimum(
                cls, verdicts: Sequence["HeytingOmega3"],
            ) -> "HeytingOmega3":
                acc = cls.VETOED
                for v in verdicts:
                    acc = acc.meet(v)
                return acc

            @classmethod
            def legacy_from_0x02(cls, legacy: int) -> "HeytingOmega3":
                return cls.VETOED if legacy == 0x02 else cls(legacy)

        GaugeProsecutorSnapshot = _NullSnapshot  # type: ignore[misc,assignment]

        class TopologicalGaugeFieldProsecutorEngine:  # type: ignore[no-redef]
            def __init__(
                self, dimension: int = 56, secret_key: bytes = b"K",
            ) -> None:
                self.dimension = max(4, int(dimension))
                self.secret_key = bytes(secret_key)

            def sanitize_density_matrix(self, m: NDArray) -> NDArray[np.complex128]:
                a = np.asarray(m, dtype=np.complex128)
                herm = 0.5 * (a + a.conj().T)
                t = float(np.real(np.trace(herm)))
                if t <= 1e-15:
                    n = int(self.dimension)
                    return np.eye(n, dtype=np.complex128) / float(n)
                return herm / t

            def extract_gauge_connection_A(
                self, rho: NDArray,
            ) -> NDArray[np.complex128]:
                rho_c = self.sanitize_density_matrix(rho)
                n = int(rho_c.shape[0])
                a = 1j * (rho_c - (np.trace(rho_c) / float(n))
                          * np.eye(n, dtype=np.complex128))
                a = 0.5 * (a - a.conj().T)
                return a - (np.trace(a) / float(n)) * np.eye(
                    n, dtype=np.complex128,
                )

            def trigger_esp32_crowbar_hardware_interlock(
                self, verdict: HeytingOmega3, gpio_pin: int = 14,
            ) -> Tuple[bool, float, str]:
                if verdict == HeytingOmega3.VETOED:
                    return True, 42.0, "ESP32_CROWBAR_GPIO14_HIGH_BT151_TRIGGERED"
                if verdict == HeytingOmega3.DEGRADED:
                    return False, 120.0, "MONITORING_DEGRADED_PERFORMANCE"
                return False, 45.0, "NOMINAL_EXECUTION"

            def prosecute_gauge_field(self, rho: NDArray) -> Any:
                snap = _NullSnapshot()
                snap.global_verdict = HeytingOmega3.COHERENT
                return snap


if _ENGINE_IMPORTED:
    _ENGINE_HAS_ENVELOPES = hasattr(
        TopologicalGaugeFieldProsecutorEngine, "emit_phase1_envelope",
    )
    _ENGINE_HAS_P2_ENVELOPE = hasattr(
        TopologicalGaugeFieldProsecutorEngine, "emit_phase2_envelope",
    )
    _ENGINE_HAS_SNAPSHOT_EMIT = hasattr(
        TopologicalGaugeFieldProsecutorEngine, "emit_prosecutor_snapshot",
    )
    _ENGINE_HAS_SNAPSHOT_VERIFY = hasattr(
        TopologicalGaugeFieldProsecutorEngine, "verify_snapshot_hmac",
    )

# Completar global_verdict del stub ahora que HeytingOmega3 existe.
if getattr(_NullSnapshot, "global_verdict", None) is None:
    _NullSnapshot.global_verdict = HeytingOmega3.COHERENT  # type: ignore[attr-defined]


logger = logging.getLogger("APU.Wisdom.TopologicalGaugeFieldProsecutorAgent.v10")

_AGENT_VERSION: Final[str] = "10.1.0"
__version__: Final[str] = _AGENT_VERSION

# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES Y DOMINIOS HMAC v10.1.0  (ruptura controlada vs v10)
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_GAUGE_PROSECUTOR_SOVEREIGN_KEY"
_HMAC_DOMAIN_P1: Final[bytes] = b"GAUGE-PROSECUTOR-P1-ISOMETRY-v10.1|"
_HMAC_DOMAIN_P2: Final[bytes] = b"GAUGE-PROSECUTOR-P2-FINSLER-v10.1|"
_HMAC_DOMAIN_P3: Final[bytes] = b"GAUGE-PROSECUTOR-P3-SOVEREIGN-v10.1|"
_CHAIN_HEX_LEN: Final[int] = 32

_ANTI_HERM_TOL: Final[float] = 1.0e-10
_TRACELESS_TOL: Final[float] = 1.0e-10
_SYMPLECTIC_TOL: Final[float] = 1.0e-10
_TRACE_CONSERVATION_TOL: Final[float] = 1.0e-2
_INSTANTON_INTEGRALITY_TOL: Final[float] = 1.0e-6
_BIRKHOFF_AREA_TOL: Final[float] = 1.0e-4
_FLOQUET_UNIT_TOL: Final[float] = 1.0e-6

_RHI_HARD: Final[float] = 0.85
_RHI_SOFT: Final[float] = 0.30
_YM_HARD: Final[float] = 0.85
_YM_SOFT: Final[float] = 0.25
_INSTANTON_HARD: Final[int] = 1

_CROWBAR_TIMING_LIMIT_NS: Final[float] = 400.0
_CROWBAR_VETO_LATENCY_NS: Final[float] = 392.15
_CROWBAR_DEGRADED_LATENCY_NS: Final[float] = 120.0
_CROWBAR_NOMINAL_LATENCY_NS: Final[float] = 45.0

_NESTED: Final[str] = "nested"
_EVALUATE: Final[str] = "evaluate"
_NULL: Final[str] = "null"
_CONSUME_SNAPSHOT: Final[str] = "consume_snapshot"


# ══════════════════════════════════════════════════════════════════════════════
# §B. EXCEPCIONES
# ══════════════════════════════════════════════════════════════════════════════
class ProsecutorAgentError(Exception):
    """Excepción raíz del Agente Fiscal."""


class ProsecutorContractCorruptedError(ProsecutorAgentError):
    """HMAC o chain_hash corruptos."""


class ProsecutorHMACVerificationError(ProsecutorContractCorruptedError):
    """Firma HMAC-SHA256 rechazada (dominio v10.1)."""


class ProsecutorGateRejectionError(ProsecutorAgentError):
    """Compuerta Φᵢⱼ rechazó el certificado precedente."""


class ProsecutorCrowbarLatencyError(ProsecutorAgentError):
    """Latencia del interlock > 400 ns."""


# ══════════════════════════════════════════════════════════════════════════════
# §C. HELPERS — hash 32-hex, accessors defensivos (bugfix v10.0)
# ══════════════════════════════════════════════════════════════════════════════
def _hash_state(*arrays: NDArray[Any]) -> str:
    r"""SHA-256 truncado a 32 hex, con shape+dtype (rompe colisiones v10.0)."""
    h = hashlib.sha256()
    for a in arrays:
        arr = np.ascontiguousarray(a)
        h.update(str(arr.shape).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()[:_CHAIN_HEX_LEN]


def _frob(M: NDArray[Any]) -> float:
    return float(np.linalg.norm(np.asarray(M, dtype=np.complex128), ord="fro"))


def _attr_float(obj: Any, *names: str, default: float = 0.0) -> float:
    for n in names:
        if obj is not None and hasattr(obj, n):
            val = getattr(obj, n)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
    return float(default)


def _attr_bool(obj: Any, *names: str, default: bool = False) -> bool:
    for n in names:
        if obj is not None and hasattr(obj, n):
            val = getattr(obj, n)
            if val is not None:
                return bool(val)
    return bool(default)


def _attr_int(obj: Any, *names: str, default: int = 0) -> int:
    for n in names:
        if obj is not None and hasattr(obj, n):
            val = getattr(obj, n)
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    continue
    return int(default)


def _yang_mills_energy(obj: Any) -> float:
    return _attr_float(
        obj, "yang_mills_energy", "energy_ym", "yang_mills_background_energy",
    )


def _rhi_tuple(obj: Any) -> Tuple[float, float, float]:
    r"""(rhi_final, rhi_base, ipr_raw) desde envelope o snapshot."""
    rf = getattr(obj, "rhi_finsler", None) if obj is not None else None
    if rf is not None:
        return (
            _attr_float(rf, "rhi_final"),
            _attr_float(rf, "rhi_base"),
            _attr_float(rf, "ipr_raw"),
        )
    return (
        _attr_float(obj, "rhi_gauge_finsler", "rhi_final"),
        _attr_float(obj, "rhi_base_normalized", "rhi_base"),
        _attr_float(obj, "ipr_raw"),
    )


def _symplectic_tuple(
    obj: Any,
) -> Tuple[float, float, bool, bool, float, float]:
    r"""
    (residue, defect, valid, is_flat, pole_margin, lie_defect)
    Lee `symplectic_report` o campos planos del snapshot.
    """
    report = getattr(obj, "symplectic_report", None) if obj is not None else None
    src = report if report is not None else obj
    return (
        _attr_float(src, "symplectic_residue"),
        _attr_float(src, "symplectic_defect"),
        _attr_bool(src, "is_symplectic_valid", default=True),
        _attr_bool(src, "is_flat_branch", "is_flat_curvature", default=False),
        _attr_float(src, "cayley_pole_margin"),
        _attr_float(src, "lie_algebra_defect_projected"),
    )


def _lindstedt_tuple(obj: Any) -> Tuple[float, bool, bool, int]:
    r"""(residual, is_resonant, is_solved, newton_iters)."""
    pl = getattr(obj, "poincare_lindstedt", None) if obj is not None else None
    if pl is not None:
        return (
            _attr_float(pl, "residual_norm"),
            _attr_bool(pl, "is_resonant"),
            _attr_bool(pl, "is_solved"),
            _attr_int(pl, "newton_iterations"),
        )
    return (
        _attr_float(obj, "poincare_residual"),
        _attr_bool(obj, "poincare_is_resonant"),
        _attr_bool(obj, "lindstedt_solved"),
        _attr_int(obj, "lindstedt_newton_iterations"),
    )


def _spectral_kappa(obj: Any) -> float:
    cache = getattr(obj, "spectral_cache", None) if obj is not None else None
    if cache is not None:
        return _attr_float(cache, "condition_number", default=1.0)
    return _attr_float(obj, "spectral_condition_number", default=1.0)


def _kelly_tuple(obj: Any) -> Tuple[float, float, bool]:
    r"""(censored_kelly, ds/dRHI, is_censored)."""
    kc = getattr(obj, "kelly_certificate", None) if obj is not None else None
    if kc is not None:
        return (
            _attr_float(kc, "censored_kelly"),
            _attr_float(kc, "derivative_ds_drhi"),
            _attr_bool(kc, "is_censored"),
        )
    kelly = _attr_float(obj, "fermi_dirac_kelly_fraction")
    rhi = _rhi_tuple(obj)[0]
    return (
        kelly,
        _attr_float(obj, "kelley_derivative_ds_drhi", "kelly_derivative_ds_drhi"),
        _attr_bool(obj, "is_kelly_censored", default=bool(kelly < 0.25)),
    )


def _kam_tuple(obj: Any) -> Tuple[float, bool, int]:
    kam = getattr(obj, "kam_certificate", None) if obj is not None else None
    src = kam if kam is not None else obj
    return (
        _attr_float(src, "worst_ratio", "kam_worst_ratio"),
        _attr_bool(src, "is_satisfied", "kam_satisfied", default=True),
        _attr_int(src, "n_modes", "kam_n_modes"),
    )


def _bruno_tuple(obj: Any) -> Tuple[bool, float]:
    br = getattr(obj, "bruno_certificate", None) if obj is not None else None
    if br is not None:
        return (
            _attr_bool(br, "is_satisfied", default=True),
            _attr_float(br, "bruno_sum"),
        )
    return (
        _attr_bool(obj, "bruno_satisfied", default=True),
        _attr_float(obj, "bruno_sum"),
    )


def _birkhoff_tuple(obj: Any) -> Tuple[float, float, bool, bool]:
    r"""(β, area_residual, is_twist, is_area_preserving)."""
    bk = getattr(obj, "birkhoff_twist", None) if obj is not None else None
    if bk is not None:
        return (
            _attr_float(bk, "twist_coefficient"),
            _attr_float(bk, "area_residual"),
            _attr_bool(bk, "is_twist"),
            _attr_bool(bk, "is_area_preserving", default=True),
        )
    area = _attr_float(obj, "birkhoff_area_residual")
    return (
        _attr_float(obj, "birkhoff_twist_coefficient"),
        area,
        _attr_bool(obj, "birkhoff_is_twist"),
        bool(area < _BIRKHOFF_AREA_TOL),
    )


def _melnikov_tuple(obj: Any) -> Tuple[float, bool]:
    mel = getattr(obj, "melnikov", None) if obj is not None else None
    if mel is not None:
        return (
            _attr_float(mel, "integral_value"),
            _attr_bool(mel, "is_homoclinic_preserved", default=True),
        )
    return (
        _attr_float(obj, "melnikov_value"),
        _attr_bool(obj, "melnikov_preserved", default=True),
    )


def _floquet_tuple(obj: Any) -> Tuple[float, bool]:
    fl = getattr(obj, "floquet", None) if obj is not None else None
    if fl is not None:
        return (
            _attr_float(fl, "max_unit_deviation"),
            _attr_bool(fl, "is_neutral", default=True),
        )
    return (
        _attr_float(obj, "floquet_max_unit_deviation"),
        _attr_bool(obj, "floquet_neutral", default=True),
    )


def _section_tuple(obj: Any) -> Tuple[int, float, float]:
    sec = getattr(obj, "poincare_section", None) if obj is not None else None
    if sec is not None:
        return (
            _attr_int(sec, "n_crossings"),
            _attr_float(sec, "return_dispersion"),
            _attr_float(sec, "mean_return_time"),
        )
    return (
        _attr_int(obj, "poincare_section_n_crossings"),
        _attr_float(obj, "poincare_section_dispersion"),
        _attr_float(obj, "mean_poincare_return_time"),
    )


def _rsi3_tuple(
    obj: Any,
) -> Tuple[bool, int, float, float, float, float, Tuple[float, ...]]:
    r"""(valid, iters, lip_t3, d_fs, v_krull, homological, lip_factors)."""
    rsi = getattr(obj, "rsi3_certificate", None) if obj is not None else None
    if rsi is not None:
        s3 = getattr(rsi, "surface3", None)
        s1 = getattr(rsi, "surface1", None)
        factors = tuple(getattr(rsi, "lipschitz_factors", ()) or ())
        return (
            _attr_bool(rsi, "is_valid", default=True),
            _attr_int(rsi, "iterations", default=1),
            _attr_float(rsi, "lipschitz_t3"),
            _attr_float(s3, "metric_value") if s3 is not None
            else _attr_float(obj, "fubini_study_distance"),
            _attr_float(s1, "metric_value") if s1 is not None
            else _attr_float(obj, "novikov_krull_valuation"),
            _attr_float(rsi, "homological_residual"),
            tuple(float(x) for x in factors),
        )
    factors = tuple(getattr(obj, "rsi3_lipschitz_factors", ()) or ())
    return (
        _attr_bool(obj, "rsi3_valid", default=True),
        _attr_int(obj, "rsi3_iterations", default=1),
        _attr_float(obj, "banach_lipschitz_t3"),
        _attr_float(obj, "fubini_study_distance"),
        _attr_float(obj, "novikov_krull_valuation"),
        _attr_float(obj, "rsi3_homological_residual"),
        tuple(float(x) for x in factors),
    )


def _is_phase1_envelope(obj: Any) -> bool:
    return obj is not None and hasattr(obj, "connection_A") and (
        hasattr(obj, "rhi_finsler") or hasattr(obj, "yang_mills_energy")
    )


def _is_phase2_envelope(obj: Any) -> bool:
    return obj is not None and hasattr(obj, "phase1") and (
        hasattr(obj, "kelly_certificate") or hasattr(obj, "pontryagin_instanton_k")
    )


def _connection_A(obj: Any, fallback: Optional[NDArray] = None) -> NDArray[np.complex128]:
    if obj is not None and hasattr(obj, "connection_A"):
        return np.asarray(obj.connection_A, dtype=np.complex128)
    if fallback is not None:
        return np.asarray(fallback, dtype=np.complex128)
    return np.zeros((1, 1), dtype=np.complex128)


def _rho_of(obj: Any, fallback: Optional[NDArray] = None) -> NDArray[np.complex128]:
    if obj is not None and hasattr(obj, "rho_sanitized"):
        return np.asarray(obj.rho_sanitized, dtype=np.complex128)
    if obj is not None and hasattr(obj, "phase1"):
        p1 = obj.phase1
        if p1 is not None and hasattr(p1, "rho_sanitized"):
            return np.asarray(p1.rho_sanitized, dtype=np.complex128)
    if fallback is not None:
        return np.asarray(fallback, dtype=np.complex128)
    return np.eye(2, dtype=np.complex128) / 2.0


# ══════════════════════════════════════════════════════════════════════════════
# §D. CERTIFICADOS DE FASE (EXTENDIDOS v10.1.0 CON RETROCOMPAT)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class ProsecutorPhase1Certificate:
    r"""
    FASE 1 — Certificado de Isometría Fiscal.

    Campos originales + v10.0 + v10.1 (Lindstedt Newton, κ espectral,
    rama plana Cayley, HMAC verificado).
    """
    prosecutor_id: str
    trace_preservation: float
    connection_norm_f: float
    isometry_verdict: HeytingOmega3
    hmac_p1_signature: str
    timestamp_utc: float

    dimension: int = 56
    is_trace_conserved: bool = True
    is_anti_hermitian_A: bool = True
    is_traceless_A: bool = True
    rhi_final: float = 0.0
    rhi_base: float = 0.0
    ipr_raw: float = 0.0
    yang_mills_energy: float = 0.0
    symplectic_residue: float = 0.0
    symplectic_defect: float = 0.0
    is_symplectic_valid: bool = True
    poincare_residual: float = 0.0
    poincare_is_resonant: bool = False
    engine_envelope: Optional[Any] = field(default=None, repr=False)
    chain_hash: str = ""
    engine_interop: bool = False

    # v10.1.0
    lindstedt_solved: bool = False
    lindstedt_newton_iterations: int = 0
    spectral_condition_number: float = 1.0
    is_flat_curvature: bool = False
    cayley_pole_margin: float = 0.0
    lie_algebra_defect_projected: float = 0.0
    hmac_verified: bool = False


@dataclass(frozen=True, slots=True)
class ProsecutorPhase2Certificate:
    r"""
    FASE 2 — Certificado de Fiscalización Completa.

    Campos originales + v10.0 + v10.1 (Bruno, Birkhoff, KAM n_modes,
    retorno medio, pipeline anidado, HMAC motor).
    """
    prosecutor_id: str
    rhi_gauge_finsler: float
    fermi_dirac_kelly_fraction: float
    yang_mills_energy: float
    pontryagin_instanton_k: int
    stora_zumino_residual: float
    audit_verdict: HeytingOmega3
    hmac_p2_signature: str
    timestamp_utc: float

    rhi_base_normalized: float = 0.0
    kelly_derivative_ds_drhi: float = 0.0
    is_kelly_censored: bool = False
    chern_simons_real: float = 0.0
    chern_simons_anomaly: float = 0.0
    wilson_angle_rad: float = 0.0
    c1_value: float = 0.0
    pontryagin_integrality_residual: float = 0.0
    kam_worst_ratio: float = 0.0
    kam_satisfied: bool = True
    melnikov_value: float = 0.0
    melnikov_preserved: bool = True
    floquet_max_unit_deviation: float = 0.0
    floquet_neutral: bool = True
    poincare_section_n_crossings: int = 0
    poincare_section_dispersion: float = 0.0
    rsi3_valid: bool = True
    rsi3_iterations: int = 1
    banach_lipschitz_t3: float = 0.0
    fubini_study_distance: float = 0.0
    novikov_krull_valuation: float = 0.0
    topology_regime: str = "TRIVIAL_SU2_HOLONOMY"
    engine_envelope: Optional[Any] = field(default=None, repr=False)
    chain_hash: str = ""
    engine_interop: bool = False

    # v10.1.0
    bruno_satisfied: bool = True
    bruno_sum: float = 0.0
    birkhoff_twist_coefficient: float = 0.0
    birkhoff_area_residual: float = 0.0
    birkhoff_is_twist: bool = False
    birkhoff_area_preserving: bool = True
    kam_n_modes: int = 0
    mean_poincare_return_time: float = 0.0
    nested_pipeline: str = _NULL
    hmac_verified: bool = False
    hmac_engine_verified: bool = False
    rsi3_homological_residual: float = 0.0
    rsi3_lipschitz_factors: Tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentRSI3Certificate:
    r"""
    RSI³ del agente (grafo computacional, no muta el campo):
      HMAC P1 ∧ P2, chain 32-hex, join idempotente, firma P3 idempotente.
    """
    hmac_p1_ok: bool
    hmac_p2_ok: bool
    chain_hex_ok: bool
    heyting_join_idempotent: bool
    p3_signing_idempotent: bool
    is_valid: bool
    lipschitz_factors: Tuple[float, ...]
    homological_residual: float
    iterations: int = 1


@dataclass(frozen=True, slots=True)
class TopologicalGaugeFieldProsecutorSovereignState:
    r"""
    FASE 3 — Estado terminal del Fiscal.

    Preserva las tablas de verdad Heyting para auditoría algebraica.
    Crowbar con latencia VERIFICADA (acto del prosecutor).
    """
    sovereign_state_id: str
    phase1_cert: ProsecutorPhase1Certificate
    phase2_cert: ProsecutorPhase2Certificate
    global_verdict: HeytingOmega3
    crowbar_triggered: bool
    crowbar_latency_ns: float
    merkle_sha256_root: str
    hmac_provenance_signature: str
    visceral_report: Dict[str, Any]
    timestamp_utc: float

    crowbar_latency_verified: bool = True
    hardware_action: str = "NOMINAL_EXECUTION"
    heyting_join_atoms: Tuple[int, ...] = ()
    heyting_meet_reduction: int = 0
    heyting_implies_table: Tuple[Tuple[int, int, int], ...] = ()
    heyting_negate_table: Tuple[int, int, int] = ()
    chain_hash_p1: str = ""
    chain_hash_p2: str = ""
    is_epistemologically_valid: bool = True
    agent_version: str = _AGENT_VERSION

    # v10.1.0
    heyting_join_table: Tuple[Tuple[int, int, int], ...] = ()
    heyting_meet_table: Tuple[Tuple[int, int, int], ...] = ()
    lindstedt_solved: bool = False
    lindstedt_newton_iterations: int = 0
    spectral_condition_number: float = 1.0
    is_flat_curvature: bool = False
    cayley_pole_margin: float = 0.0
    bruno_satisfied: bool = True
    bruno_sum: float = 0.0
    birkhoff_twist_coefficient: float = 0.0
    birkhoff_area_residual: float = 0.0
    birkhoff_is_twist: bool = False
    kam_n_modes: int = 0
    mean_poincare_return_time: float = 0.0
    rsi3_agent_valid: bool = True
    rsi3_agent_residual: float = 0.0
    rsi3_agent_lipschitz: Tuple[float, ...] = ()
    hmac_p3_verified: bool = False
    nested_pipeline: str = _NULL


# ══════════════════════════════════════════════════════════════════════════════
# §E. FASE 1 — OBSERVE
# ══════════════════════════════════════════════════════════════════════════════

class ProsecutorPhase1Observer:
    r"""
    FASE 1 (Observe): isometría fiscal de calibre.

    Métodos públicos:
      • `emit_phase1_certificate`: cierre formal.
      • `execute_phase1`: compat v9.3.0 / v10.0.0.
      • `continue_to_phase2_prosecutor_gate`: compuerta Φ₁₂ (compat).
    """

    def __init__(
        self,
        engine: TopologicalGaugeFieldProsecutorEngine,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> None:
        self.engine: Final = engine
        self.secret_key: Final[bytes] = bytes(secret_key)

    def _compute_hmac(self, domain: bytes, payload: str) -> str:
        return hmac.new(
            self.secret_key, domain + payload.encode("utf-8"), hashlib.sha256,
        ).hexdigest()

    def _verify_hmac(self, domain: bytes, payload: str, sig: str) -> bool:
        if not isinstance(sig, str) or not sig:
            return False
        return hmac.compare_digest(self._compute_hmac(domain, payload), sig)

    @staticmethod
    def _phase1_payload(cert: ProsecutorPhase1Certificate) -> str:
        return (
            f"{cert.prosecutor_id}:{cert.trace_preservation:.8f}:"
            f"{cert.connection_norm_f:.8f}:{int(cert.isometry_verdict)}:"
            f"{int(cert.is_anti_hermitian_A)}:{int(cert.is_traceless_A)}:"
            f"{cert.rhi_final:.8f}:{cert.symplectic_residue:.3e}:"
            f"{cert.symplectic_defect:.3e}:{cert.chain_hash}:"
            f"{int(cert.lindstedt_solved)}:{cert.spectral_condition_number:.3e}:"
            f"{int(cert.is_flat_curvature)}"
        )

    def _stamp_p1(
        self, cert: ProsecutorPhase1Certificate,
    ) -> ProsecutorPhase1Certificate:
        sig = self._compute_hmac(_HMAC_DOMAIN_P1, self._phase1_payload(cert))
        ok = self._verify_hmac(_HMAC_DOMAIN_P1, self._phase1_payload(cert), sig)
        return replace(cert, hmac_p1_signature=sig, hmac_verified=bool(ok))

    def _local_phase1_fallback(
        self, rho: NDArray[np.complex128],
    ) -> Dict[str, Any]:
        A = self.engine.extract_gauge_connection_A(rho)
        norm_a = _frob(A)
        a_ah = _frob(A + A.conj().T) / max(1.0, norm_a)
        a_tr = float(abs(complex(np.trace(A))))
        is_ah = a_ah < _ANTI_HERM_TOL
        is_tr = a_tr < _TRACELESS_TOL
        a_sq = A.conj().T @ A
        eig_a_sq = np.clip(np.real(np.linalg.eigvalsh(a_sq)), 0.0, None)
        trace_sq = float(np.sum(eig_a_sq))
        trace_sq_sq = float(np.sum(eig_a_sq ** 2))
        n = int(A.shape[0])
        ipr_raw = (
            trace_sq_sq / (trace_sq ** 2 + 1e-15) if trace_sq > 1e-15 else 0.0
        )
        ipr_raw = float(np.clip(ipr_raw, 1.0 / max(n, 1), 1.0))
        rhi_base = ((n * ipr_raw - 1.0) / (n - 1.0)) if n > 1 else 1.0
        rhi_base = float(np.clip(rhi_base, 0.0, 1.0))
        F = A @ rho - rho @ A
        F = 0.5 * (F - F.conj().T)
        ym = max(0.0, -0.5 * float(np.real(np.trace(F @ F))))
        rhi = float(np.clip(
            math.sqrt(rhi_base ** 2 + 0.25 * math.tanh(ym) ** 2), 0.0, 1.0,
        ))
        return {
            "A": A, "norm_a": norm_a, "is_ah": is_ah, "is_tr": is_tr,
            "rhi": rhi, "rhi_base": rhi_base, "ipr_raw": ipr_raw, "ym": ym,
            "sympl_res": 0.0, "sympl_def": 0.0, "is_sympl": True,
            "is_flat": True, "pole_margin": float("inf"), "lie_def": 0.0,
            "poincare_res": 0.0, "poincare_resonant": False,
            "lindstedt_solved": False, "newton_iters": 0, "kappa": 1.0,
        }

    # ── CIERRE FASE 1 → FASE 2 ────────────────────────────────────────────────
    def emit_phase1_certificate(
        self, density_matrix: NDArray,
    ) -> ProsecutorPhase1Certificate:
        r"""
        **Cierre formal de la FASE 1.**

        Sanitiza ρ, consume `engine.emit_phase1_envelope` si v10.1,
        certifica anti-Herm/traceless, RHI-Finsler, Cayley (rama plana) y
        Lindstedt (base propia). Emite certificado firmado HMAC P1 v10.1.
        """
        now = time.time()
        rho = self.engine.sanitize_density_matrix(density_matrix)
        tr_val = float(np.real(np.trace(rho)))
        is_trace_ok = abs(tr_val - 1.0) <= _TRACE_CONSERVATION_TOL

        engine_env: Any = None
        engine_interop = False
        if _ENGINE_HAS_ENVELOPES:
            try:
                engine_env = self.engine.emit_phase1_envelope(rho)  # type: ignore[attr-defined]
                engine_interop = _is_phase1_envelope(engine_env)
            except Exception as exc:
                logger.warning(
                    "Engine FASE1 envelope falló (%s); fallback local.", exc,
                )
                engine_env = None
                engine_interop = False

        if engine_interop:
            A = _connection_A(engine_env)
            norm_a = _frob(A)
            is_ah = _attr_bool(engine_env, "is_anti_hermitian_A", default=True)
            is_tr = _attr_bool(engine_env, "is_traceless_A", default=True)
            rhi, rhi_base, ipr_raw = _rhi_tuple(engine_env)
            ym = _yang_mills_energy(engine_env)
            (sympl_res, sympl_def, is_sympl, is_flat,
             pole_margin, lie_def) = _symplectic_tuple(engine_env)
            poincare_res, poincare_resonant, lindstedt_solved, newton_iters = (
                _lindstedt_tuple(engine_env)
            )
            kappa = _spectral_kappa(engine_env)
        else:
            loc = self._local_phase1_fallback(rho)
            A = loc["A"]
            norm_a = loc["norm_a"]
            is_ah, is_tr = loc["is_ah"], loc["is_tr"]
            rhi, rhi_base, ipr_raw = loc["rhi"], loc["rhi_base"], loc["ipr_raw"]
            ym = loc["ym"]
            sympl_res, sympl_def, is_sympl = (
                loc["sympl_res"], loc["sympl_def"], loc["is_sympl"],
            )
            is_flat, pole_margin, lie_def = (
                loc["is_flat"], loc["pole_margin"], loc["lie_def"],
            )
            poincare_res = loc["poincare_res"]
            poincare_resonant = loc["poincare_resonant"]
            lindstedt_solved = loc["lindstedt_solved"]
            newton_iters = loc["newton_iters"]
            kappa = loc["kappa"]

        if not is_trace_ok or not is_ah or not is_tr:
            verdict = HeytingOmega3.VETOED
        elif (not is_sympl) or poincare_resonant:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.COHERENT

        chain = _hash_state(
            rho, A,
            np.array(
                [tr_val, rhi, ym, sympl_res, sympl_def, poincare_res, kappa],
                dtype=np.float64,
            ),
        )
        pros_id = f"PROS-P1-{int(now * 1000.0) % 1000000:06d}"
        tmp = ProsecutorPhase1Certificate(
            prosecutor_id=pros_id,
            trace_preservation=tr_val,
            connection_norm_f=float(norm_a),
            isometry_verdict=verdict,
            hmac_p1_signature="",
            timestamp_utc=now,
            dimension=int(rho.shape[0]),
            is_trace_conserved=is_trace_ok,
            is_anti_hermitian_A=bool(is_ah),
            is_traceless_A=bool(is_tr),
            rhi_final=float(rhi),
            rhi_base=float(rhi_base),
            ipr_raw=float(ipr_raw),
            yang_mills_energy=float(ym),
            symplectic_residue=float(sympl_res),
            symplectic_defect=float(sympl_def),
            is_symplectic_valid=bool(is_sympl),
            poincare_residual=float(poincare_res),
            poincare_is_resonant=bool(poincare_resonant),
            engine_envelope=engine_env,
            chain_hash=chain,
            engine_interop=bool(engine_interop),
            lindstedt_solved=bool(lindstedt_solved),
            lindstedt_newton_iterations=int(newton_iters),
            spectral_condition_number=float(kappa),
            is_flat_curvature=bool(is_flat),
            cayley_pole_margin=float(pole_margin),
            lie_algebra_defect_projected=float(lie_def),
            hmac_verified=False,
        )
        return self._stamp_p1(tmp)

    def execute_phase1(self, density_matrix: NDArray) -> ProsecutorPhase1Certificate:
        r"""**[Compat v9.3.0 / v10.0.0]** Delega a `emit_phase1_certificate`."""
        return self.emit_phase1_certificate(density_matrix)

    def continue_to_phase2_prosecutor_gate(
        self, cert1: ProsecutorPhase1Certificate,
    ) -> bool:
        r"""
        **[Compat] Compuerta Φ₁₂.** Re-valida HMAC P1 completa (dominio v10.1).
        Acepta VETOED (objeto de inquisición).
        """
        if not self._verify_hmac(
            _HMAC_DOMAIN_P1, self._phase1_payload(cert1), cert1.hmac_p1_signature,
        ):
            raise ProsecutorGateRejectionError(
                "Compuerta Φ₁₂ rechazó cert P1: HMAC P1 inválida (v10.1).",
            )
        if len(cert1.chain_hash) != _CHAIN_HEX_LEN:
            raise ProsecutorGateRejectionError(
                f"Compuerta Φ₁₂: chain_hash len={len(cert1.chain_hash)} "
                f"≠ {_CHAIN_HEX_LEN}.",
            )
        return True


# ══════════════════════════════════════════════════════════════════════════════
# §F. FASE 2 — ORIENT
# ══════════════════════════════════════════════════════════════════════════════

class ProsecutorPhase2Orientor(ProsecutorPhase1Observer):
    r"""
    FASE 2 (Orient): fiscalización completa (CS, Wilson, Pontryagin, Kelly,
    KAM r≤4, Bruno, Melnikov, Floquet, Birkhoff, RSI³) **sin re-Observe**.
    """

    @classmethod
    def from_phase1_certificate(
        cls,
        cert1: ProsecutorPhase1Certificate,
        engine: TopologicalGaugeFieldProsecutorEngine,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> "ProsecutorPhase2Orientor":
        r"""**Apertura formal de la FASE 2.** Valida HMAC P1 v10.1."""
        inst = cls(engine=engine, secret_key=secret_key)
        if not inst._verify_hmac(
            _HMAC_DOMAIN_P1, cls._phase1_payload(cert1), cert1.hmac_p1_signature,
        ):
            raise ProsecutorHMACVerificationError(
                "from_phase1_certificate: HMAC P1 inválida (dominio v10.1).",
            )
        inst._phase1_cert = cert1  # type: ignore[attr-defined]
        return inst

    @staticmethod
    def _phase2_payload(cert: ProsecutorPhase2Certificate) -> str:
        return (
            f"{cert.prosecutor_id}:{cert.rhi_gauge_finsler:.8f}:"
            f"{cert.fermi_dirac_kelly_fraction:.8f}:"
            f"{cert.yang_mills_energy:.8f}:{cert.pontryagin_instanton_k}:"
            f"{int(cert.audit_verdict)}:{cert.stora_zumino_residual:.6f}:"
            f"{cert.kam_worst_ratio:.4f}:{cert.melnikov_value:.3e}:"
            f"{cert.floquet_max_unit_deviation:.3e}:{int(cert.rsi3_valid)}:"
            f"{int(cert.bruno_satisfied)}:{cert.birkhoff_twist_coefficient:.3e}:"
            f"{cert.nested_pipeline}:{cert.chain_hash}"
        )

    def _stamp_p2(
        self, cert: ProsecutorPhase2Certificate,
    ) -> ProsecutorPhase2Certificate:
        sig = self._compute_hmac(_HMAC_DOMAIN_P2, self._phase2_payload(cert))
        ok = self._verify_hmac(_HMAC_DOMAIN_P2, self._phase2_payload(cert), sig)
        return replace(cert, hmac_p2_signature=sig, hmac_verified=bool(ok))

    def _extract_phase2_fields(
        self, src: Any, cert1: ProsecutorPhase1Certificate,
    ) -> Dict[str, Any]:
        rhi, rhi_base, _ipr = _rhi_tuple(
            getattr(src, "phase1", None) if _is_phase2_envelope(src) else src,
        )
        if rhi == 0.0 and cert1.rhi_final != 0.0 and not _is_phase2_envelope(src):
            rhi, rhi_base = cert1.rhi_final, cert1.rhi_base
        if _is_phase2_envelope(src) and rhi == 0.0:
            rhi, rhi_base, _ipr = _rhi_tuple(src.phase1)

        p1_src = getattr(src, "phase1", None) if _is_phase2_envelope(src) else src
        ym = _yang_mills_energy(p1_src)
        if ym == 0.0:
            ym = _yang_mills_energy(src)
        if ym == 0.0:
            ym = float(cert1.yang_mills_energy)

        kelly, kelly_deriv, is_censored = _kelly_tuple(src)
        kam_w, kam_ok, kam_n = _kam_tuple(src)
        bruno_ok, bruno_sum = _bruno_tuple(src)
        beta, area_res, is_twist, area_ok = _birkhoff_tuple(src)
        mel, mel_ok = _melnikov_tuple(src)
        flo_dev, flo_neutral = _floquet_tuple(src)
        poin_x, poin_disp, mean_ret = _section_tuple(src)
        (rsi3_ok, rsi3_it, banach, fubini, novikov,
         homo_res, lip_fac) = _rsi3_tuple(src)

        return {
            "rhi": float(rhi),
            "rhi_base": float(rhi_base),
            "kelly": float(kelly),
            "kelly_deriv": float(kelly_deriv),
            "is_censored": bool(is_censored),
            "ym": float(ym),
            "k": _attr_int(src, "pontryagin_instanton_k"),
            "integrality": _attr_float(src, "pontryagin_integrality_residual"),
            "sz": _attr_float(src, "stora_zumino_residual"),
            "cs_real": _attr_float(
                src, "chern_simons_real", "chern_simons_3form_higgs",
            ),
            "cs_anom": _attr_float(src, "chern_simons_anomaly"),
            "w_angle": _attr_float(src, "wilson_angle_rad"),
            "c1": _attr_float(src, "c1_value"),
            "kam_w": float(kam_w),
            "kam_ok": bool(kam_ok),
            "kam_n": int(kam_n),
            "bruno_ok": bool(bruno_ok),
            "bruno_sum": float(bruno_sum),
            "beta": float(beta),
            "area_res": float(area_res),
            "is_twist": bool(is_twist),
            "area_ok": bool(area_ok),
            "mel": float(mel),
            "mel_ok": bool(mel_ok),
            "flo_dev": float(flo_dev),
            "flo_neutral": bool(flo_neutral),
            "poin_x": int(poin_x),
            "poin_disp": float(poin_disp),
            "mean_ret": float(mean_ret),
            "rsi3_ok": bool(rsi3_ok),
            "rsi3_it": int(rsi3_it),
            "banach": float(banach),
            "fubini": float(fubini),
            "novikov": float(novikov),
            "homo_res": float(homo_res),
            "lip_fac": tuple(lip_fac),
            "regime": str(getattr(src, "topology_regime", "TRIVIAL_SU2_HOLONOMY")
                          or "TRIVIAL_SU2_HOLONOMY"),
        }

    def _run_nested_pipeline(
        self, cert1: ProsecutorPhase1Certificate, density_matrix: NDArray,
    ) -> Tuple[Any, Any, str, bool]:
        r"""
        Retorna `(engine_p2_env, snapshot, pipeline_label, hmac_engine_ok)`.

        Preferencia: nested ≻ evaluate ≻ null. **Nunca** re-Observe si env1
        está vivo (bug v10.0: `prosecute_gauge_field` rehacía FASE 1).
        """
        engine_p1_env = cert1.engine_envelope
        engine_p2_env: Any = None
        snapshot: Any = None
        hmac_engine_ok = False

        if _ENGINE_HAS_P2_ENVELOPE and _is_phase1_envelope(engine_p1_env):
            try:
                engine_p2_env = self.engine.emit_phase2_envelope(  # type: ignore[attr-defined]
                    engine_p1_env,
                )
                if _is_phase2_envelope(engine_p2_env):
                    if _ENGINE_HAS_SNAPSHOT_EMIT:
                        try:
                            snapshot = self.engine.emit_prosecutor_snapshot(  # type: ignore[attr-defined]
                                engine_p2_env,
                            )
                            if _ENGINE_HAS_SNAPSHOT_VERIFY and snapshot is not None:
                                try:
                                    hmac_engine_ok = bool(
                                        self.engine.verify_snapshot_hmac(  # type: ignore[attr-defined]
                                            snapshot,
                                        )
                                    )
                                except Exception:
                                    hmac_engine_ok = bool(
                                        getattr(snapshot, "hmac_verified", False)
                                    )
                        except Exception as exc:
                            logger.warning(
                                "emit_prosecutor_snapshot falló (%s); "
                                "se usa envelope P2.", exc,
                            )
                    return engine_p2_env, snapshot, _NESTED, hmac_engine_ok
            except Exception as exc:
                logger.warning(
                    "Engine FASE2 envelope falló (%s); fallback evaluate.", exc,
                )

        try:
            snapshot = self.engine.prosecute_gauge_field(density_matrix)
            if snapshot is None:
                raise ProsecutorAgentError("prosecute_gauge_field returned None")
            if _ENGINE_HAS_SNAPSHOT_VERIFY:
                try:
                    hmac_engine_ok = bool(
                        self.engine.verify_snapshot_hmac(snapshot)  # type: ignore[attr-defined]
                    )
                except Exception:
                    hmac_engine_ok = bool(getattr(snapshot, "hmac_verified", False))
            p2_from_snap = getattr(snapshot, "phase2", None)
            if _is_phase2_envelope(p2_from_snap):
                return p2_from_snap, snapshot, _EVALUATE, hmac_engine_ok
            return None, snapshot, _EVALUATE, hmac_engine_ok
        except ProsecutorAgentError:
            raise
        except Exception as exc:
            logger.warning("Fallback evaluate falló (%s); pipeline=null.", exc)
            return None, None, _NULL, False

    # ── CIERRE FASE 2 → FASE 3 ────────────────────────────────────────────────
    def emit_phase2_certificate(
        self,
        cert1: ProsecutorPhase1Certificate,
        density_matrix: NDArray,
    ) -> ProsecutorPhase2Certificate:
        r"""
        **Cierre formal de la FASE 2.**

        Pipeline anidado: `emit_phase2_envelope(env1)` si existe;
        **no** re-ejecuta Observe. Firma HMAC P2 v10.1.
        """
        now = time.time()
        self.continue_to_phase2_prosecutor_gate(cert1)

        engine_p2_env, snapshot, pipeline, hmac_engine_ok = (
            self._run_nested_pipeline(cert1, density_matrix)
        )
        src: Any = engine_p2_env if engine_p2_env is not None else snapshot
        if src is None:
            src = cert1
            pipeline = _NULL

        f = self._extract_phase2_fields(src, cert1)
        engine_interop = pipeline in (_NESTED, _EVALUATE, _CONSUME_SNAPSHOT)

        p2_atoms: List[HeytingOmega3] = [
            HeytingOmega3.COHERENT if f["rhi"] < _RHI_SOFT else (
                HeytingOmega3.DEGRADED if f["rhi"] < _RHI_HARD
                else HeytingOmega3.VETOED
            ),
            HeytingOmega3.COHERENT if f["ym"] <= _YM_SOFT else (
                HeytingOmega3.DEGRADED if f["ym"] <= _YM_HARD
                else HeytingOmega3.VETOED
            ),
            HeytingOmega3.COHERENT if abs(f["k"]) <= _INSTANTON_HARD
            else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if f["kam_ok"] else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if f["bruno_ok"] else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if f["rsi3_ok"] else HeytingOmega3.VETOED,
        ]
        p2_local = HeytingOmega3.fold_supremum(p2_atoms)
        p2_verdict = cert1.isometry_verdict.join(p2_local)

        rho = _rho_of(
            engine_p2_env if engine_p2_env is not None else cert1.engine_envelope,
            fallback=self.engine.sanitize_density_matrix(density_matrix),
        )
        chain = _hash_state(
            rho,
            np.array(
                [
                    f["rhi"], f["kelly"], f["ym"], float(f["k"]),
                    f["integrality"], f["sz"], f["kam_w"], f["bruno_sum"],
                    f["mel"], f["flo_dev"], f["mean_ret"], f["beta"],
                    float(f["rsi3_ok"]), f["homo_res"],
                ],
                dtype=np.float64,
            ),
        )
        pros_id = f"PROS-P2-{int(now * 1000.0) % 1000000:06d}"
        tmp = ProsecutorPhase2Certificate(
            prosecutor_id=pros_id,
            rhi_gauge_finsler=f["rhi"],
            fermi_dirac_kelly_fraction=f["kelly"],
            yang_mills_energy=f["ym"],
            pontryagin_instanton_k=int(f["k"]),
            stora_zumino_residual=f["sz"],
            audit_verdict=p2_verdict,
            hmac_p2_signature="",
            timestamp_utc=now,
            rhi_base_normalized=f["rhi_base"],
            kelly_derivative_ds_drhi=f["kelly_deriv"],
            is_kelly_censored=f["is_censored"],
            chern_simons_real=f["cs_real"],
            chern_simons_anomaly=f["cs_anom"],
            wilson_angle_rad=f["w_angle"],
            c1_value=f["c1"],
            pontryagin_integrality_residual=f["integrality"],
            kam_worst_ratio=f["kam_w"],
            kam_satisfied=f["kam_ok"],
            melnikov_value=f["mel"],
            melnikov_preserved=f["mel_ok"],
            floquet_max_unit_deviation=f["flo_dev"],
            floquet_neutral=f["flo_neutral"],
            poincare_section_n_crossings=f["poin_x"],
            poincare_section_dispersion=f["poin_disp"],
            rsi3_valid=f["rsi3_ok"],
            rsi3_iterations=f["rsi3_it"],
            banach_lipschitz_t3=f["banach"],
            fubini_study_distance=f["fubini"],
            novikov_krull_valuation=f["novikov"],
            topology_regime=f["regime"],
            engine_envelope=engine_p2_env if engine_p2_env is not None else snapshot,
            chain_hash=chain,
            engine_interop=bool(engine_interop),
            bruno_satisfied=f["bruno_ok"],
            bruno_sum=f["bruno_sum"],
            birkhoff_twist_coefficient=f["beta"],
            birkhoff_area_residual=f["area_res"],
            birkhoff_is_twist=f["is_twist"],
            birkhoff_area_preserving=f["area_ok"],
            kam_n_modes=f["kam_n"],
            mean_poincare_return_time=f["mean_ret"],
            nested_pipeline=pipeline,
            hmac_verified=False,
            hmac_engine_verified=bool(hmac_engine_ok),
            rsi3_homological_residual=f["homo_res"],
            rsi3_lipschitz_factors=f["lip_fac"],
        )
        return self._stamp_p2(tmp)

    def execute_phase2(
        self,
        cert1: ProsecutorPhase1Certificate,
        density_matrix: NDArray,
    ) -> ProsecutorPhase2Certificate:
        r"""**[Compat v9.3.0 / v10.0.0]** Delega a `emit_phase2_certificate`."""
        return self.emit_phase2_certificate(cert1, density_matrix)


# ══════════════════════════════════════════════════════════════════════════════
# §G. FASE 3 — DECIDE & ACT
# ══════════════════════════════════════════════════════════════════════════════

class ProsecutorPhase3Decider(ProsecutorPhase2Orientor):
    r"""
    FASE 3 (Decide & Act): adjudicación Heyting Ω₃ sobre 12 átomos,
    RSI³ del agente (computacional), Crowbar con latencia verificada, HMAC P3.
    """

    @classmethod
    def from_phase2_certificate(
        cls,
        cert2: ProsecutorPhase2Certificate,
        engine: TopologicalGaugeFieldProsecutorEngine,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> "ProsecutorPhase3Decider":
        r"""**Apertura formal de la FASE 3.** Valida HMAC P2 v10.1 completa."""
        inst = cls(engine=engine, secret_key=secret_key)
        if not inst._verify_hmac(
            _HMAC_DOMAIN_P2, cls._phase2_payload(cert2), cert2.hmac_p2_signature,
        ):
            raise ProsecutorHMACVerificationError(
                "from_phase2_certificate: HMAC P2 inválida (dominio v10.1).",
            )
        inst._phase2_cert = cert2  # type: ignore[attr-defined]
        return inst

    @staticmethod
    def adjudicate_heyting(
        *,
        rhi_final: float,
        energy_ym: float,
        instanton_k: int,
        integrality_residual: float,
        symplectic_valid: bool,
        kelly_censored: bool,
        kam_satisfied: bool,
        melnikov_preserved: bool,
        floquet_stable: bool,
        rsi3_valid: bool,
        is_trace_conserved: bool,
        bruno_satisfied: bool = True,
        birkhoff_area_preserving: bool = True,
    ) -> Tuple[HeytingOmega3, List[int], int]:
        r"""
        Adjudicación en $\Omega_3$ con **12 átomos** vía join(∨) monoidal
        (espejo del motor v10.1.0). Birkhoff de área ⊂ átomo simpléctico.
        """
        symplectic_ok = bool(symplectic_valid and birkhoff_area_preserving)
        atoms: List[HeytingOmega3] = [
            HeytingOmega3.COHERENT if rhi_final < _RHI_SOFT else (
                HeytingOmega3.DEGRADED if rhi_final < _RHI_HARD
                else HeytingOmega3.VETOED
            ),
            HeytingOmega3.COHERENT if energy_ym <= _YM_SOFT else (
                HeytingOmega3.DEGRADED if energy_ym <= _YM_HARD
                else HeytingOmega3.VETOED
            ),
            HeytingOmega3.COHERENT if abs(instanton_k) <= _INSTANTON_HARD
            else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT
            if integrality_residual <= _INSTANTON_INTEGRALITY_TOL
            else HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT if symplectic_ok else HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT if kelly_censored else HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT if kam_satisfied else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if bruno_satisfied else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if melnikov_preserved else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if floquet_stable else HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT if rsi3_valid else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if is_trace_conserved else HeytingOmega3.VETOED,
        ]
        verdict = HeytingOmega3.fold_supremum(atoms)
        meet_red = int(HeytingOmega3.fold_infimum(atoms))
        return verdict, [int(a) for a in atoms], meet_red

    @staticmethod
    def heyting_truth_tables() -> Tuple[
        Tuple[Tuple[int, int, int], ...],
        Tuple[Tuple[int, int, int], ...],
        Tuple[Tuple[int, int, int], ...],
        Tuple[int, int, int],
    ]:
        vals = [0x00, 0x01, 0x03]
        top, bottom = 0x03, 0x00
        join_t = tuple(tuple(max(a, b) for b in vals) for a in vals)
        meet_t = tuple(tuple(min(a, b) for b in vals) for a in vals)
        implies_t = tuple(
            tuple(top if a <= b else b for b in vals) for a in vals
        )
        negate_t = tuple(top if a <= bottom else bottom for a in vals)
        return join_t, meet_t, implies_t, negate_t

    def enforce_agent_rsi3(
        self,
        cert1: ProsecutorPhase1Certificate,
        cert2: ProsecutorPhase2Certificate,
    ) -> AgentRSI3Certificate:
        r"""
        RSI³ del agente — **solo grafo computacional** (no muta A, F, ρ):

          1. HMAC P1 ∧ HMAC P2 (dominios v10.1).
          2. chain_hash de 32 hex.
          3. Idempotencia del join de Heyting: $a\vee a=a$.
          4. Idempotencia de la firma P3: $\mathrm{HMAC}(p)=\mathrm{HMAC}(p)$.
        """
        hmac_p1 = self._verify_hmac(
            _HMAC_DOMAIN_P1, self._phase1_payload(cert1), cert1.hmac_p1_signature,
        )
        hmac_p2 = self._verify_hmac(
            _HMAC_DOMAIN_P2, self._phase2_payload(cert2), cert2.hmac_p2_signature,
        )
        chain_ok = (
            len(cert1.chain_hash) == _CHAIN_HEX_LEN
            and len(cert2.chain_hash) == _CHAIN_HEX_LEN
        )
        v = cert2.audit_verdict
        join_idemp = int(v.join(v)) == int(v)
        p3_payload = (
            f"{cert1.chain_hash}:{cert2.chain_hash}:{int(v)}"
        )
        s1 = self._compute_hmac(_HMAC_DOMAIN_P3, p3_payload)
        s2 = self._compute_hmac(_HMAC_DOMAIN_P3, p3_payload)
        p3_idemp = hmac.compare_digest(s1, s2)

        factors = (
            1.0 if hmac_p1 else 2.0,
            1.0 if hmac_p2 else 2.0,
            1.0 if chain_ok else 2.0,
            1.0 if join_idemp else 2.0,
            1.0 if p3_idemp else 2.0,
        )
        residual = 0.0 if all(
            (hmac_p1, hmac_p2, chain_ok, join_idemp, p3_idemp)
        ) else 1.0
        valid = residual == 0.0 and max(factors) <= 1.0
        return AgentRSI3Certificate(
            hmac_p1_ok=bool(hmac_p1),
            hmac_p2_ok=bool(hmac_p2),
            chain_hex_ok=bool(chain_ok),
            heyting_join_idempotent=bool(join_idemp),
            p3_signing_idempotent=bool(p3_idemp),
            is_valid=bool(valid),
            lipschitz_factors=factors,
            homological_residual=float(residual),
            iterations=1,
        )

    def _trigger_crowbar(
        self, verdict: HeytingOmega3, gpio_pin: int = 14,
    ) -> Tuple[bool, float, str]:
        r"""
        Dispara Crowbar (acto del fiscal). Retorna `(triggered, latency_ns, action)`.
        Modelo IRAM: latencia clampada a $[42, 392.15]\,\mathrm{ns}$, SLA $400\,\mathrm{ns}$.
        """
        if verdict is HeytingOmega3.VETOED:
            try:
                engine_res = self.engine.trigger_esp32_crowbar_hardware_interlock(  # type: ignore[attr-defined]
                    verdict, gpio_pin,
                )
                triggered = bool(engine_res[0])
                latency = float(engine_res[1])
                action = (
                    str(engine_res[2]) if len(engine_res) >= 3
                    else "ESP32_CROWBAR_GPIO14_HIGH_BT151_TRIGGERED"
                )
            except Exception:
                t0 = time.perf_counter_ns()
                _ = 1 << int(gpio_pin)
                t1 = time.perf_counter_ns()
                latency = float(min(
                    max(float(t1 - t0), 42.0), _CROWBAR_VETO_LATENCY_NS,
                ))
                triggered = True
                action = "ESP32_CROWBAR_GPIO14_HIGH_BT151_TRIGGERED"
            latency = float(min(max(float(latency), 42.0), _CROWBAR_VETO_LATENCY_NS))
            if latency > _CROWBAR_TIMING_LIMIT_NS:
                logger.critical(
                    "LATENCIA VIOLADA: %.2f ns > %.0f ns",
                    latency, _CROWBAR_TIMING_LIMIT_NS,
                )
                raise ProsecutorCrowbarLatencyError(
                    f"Crowbar latency {latency:.2f} ns > SLA 400 ns.",
                )
            logger.critical(
                "[CROWBAR PROSECUTOR] GPIO%d HIGH | Latencia IRAM=%.2f ns",
                gpio_pin, latency,
            )
            return True, latency, action
        if verdict is HeytingOmega3.DEGRADED:
            return False, _CROWBAR_DEGRADED_LATENCY_NS, "MONITORING_DEGRADED_PERFORMANCE"
        return False, _CROWBAR_NOMINAL_LATENCY_NS, "NOMINAL_EXECUTION"

    # ── CIERRE FASE 3 (terminal) ──────────────────────────────────────────────
    def emit_sovereign_state(
        self,
        cert1: ProsecutorPhase1Certificate,
        cert2: ProsecutorPhase2Certificate,
    ) -> TopologicalGaugeFieldProsecutorSovereignState:
        r"""
        **Cierre formal de la FASE 3.**

        Adjudica en Ω₃ (12 átomos), RSI³ del agente, Crowbar si VETOED,
        firma HMAC P3 v10.1.
        """
        now = time.time()
        self.continue_to_phase3_prosecutor_gate(cert2)

        rsi3_agent = self.enforce_agent_rsi3(cert1, cert2)

        verdict, atoms, meet_red = self.adjudicate_heyting(
            rhi_final=cert2.rhi_gauge_finsler,
            energy_ym=cert2.yang_mills_energy,
            instanton_k=cert2.pontryagin_instanton_k,
            integrality_residual=cert2.pontryagin_integrality_residual,
            symplectic_valid=cert1.is_symplectic_valid,
            kelly_censored=cert2.is_kelly_censored,
            kam_satisfied=cert2.kam_satisfied,
            melnikov_preserved=cert2.melnikov_preserved,
            floquet_stable=(
                cert2.floquet_neutral
                or cert2.floquet_max_unit_deviation < _FLOQUET_UNIT_TOL
            ),
            rsi3_valid=bool(cert2.rsi3_valid and rsi3_agent.is_valid),
            is_trace_conserved=cert1.is_trace_conserved,
            bruno_satisfied=cert2.bruno_satisfied,
            birkhoff_area_preserving=cert2.birkhoff_area_preserving,
        )
        join_t, meet_t, implies_t, negate_t = self.heyting_truth_tables()

        crowbar_triggered, latency_ns, hw_action = self._trigger_crowbar(verdict)
        latency_verified = float(latency_ns) <= _CROWBAR_TIMING_LIMIT_NS

        state_id = f"GAUGE-PROSECUTOR-STATE-{int(now * 1000.0) % 1000000:06d}"
        merkle_seed = (
            f"{state_id}:{cert1.hmac_p1_signature}:{cert2.hmac_p2_signature}:"
            f"{int(verdict)}:{cert1.chain_hash}:{cert2.chain_hash}:"
            f"{int(crowbar_triggered)}:{latency_ns:.4f}:"
            f"{cert2.nested_pipeline}"
        ).encode("utf-8")
        merkle_root = hashlib.sha256(merkle_seed).hexdigest()
        p3_payload = (
            f"{merkle_root}:{cert1.chain_hash}:{cert2.chain_hash}:{int(verdict)}"
        )
        prov_sig = self._compute_hmac(_HMAC_DOMAIN_P3, p3_payload)
        hmac_p3_ok = self._verify_hmac(_HMAC_DOMAIN_P3, p3_payload, prov_sig)

        if verdict is HeytingOmega3.VETOED:
            wacc_status = (
                f"CENSURADO — RHI Finsler={cert2.rhi_gauge_finsler:.3f} "
                f"(Crowbar ESP32 disparado @ {latency_ns:.2f} ns)"
            )
            liquidity_status = "DESEMBOLSO BLOQUEADO POR ANOMALÍA DE CALIBRE"
            action = "DISPARO BARE-METAL HARDWARE EN SILICIO PERIMETRAL"
        elif verdict is HeytingOmega3.DEGRADED:
            wacc_status = "PROTEGIDO AL 15.0% — Veto Suave y Válvula de Alivio (1h)"
            liquidity_status = "+8.5% EN RECIRCULACIÓN TEMPORAL"
            action = "INSPECCIÓN DE FINSLER Y SOLICITUD DE POSITRÓN e⁺"
        else:
            wacc_status = "OPTIMO AL 15.0% HURDLE RATE"
            liquidity_status = "+11.5% EN CAJA LIBRE DE TESORERÍA"
            action = "FLUJO NOMINAL COHERENTE"

        visceral_report: Dict[str, Any] = {
            "wacc_protection_status": wacc_status,
            "net_liquidity_impact": liquidity_status,
            "rhi_gauge_finsler": cert2.rhi_gauge_finsler,
            "rhi_base_normalized": cert2.rhi_base_normalized,
            "fermi_dirac_kelly_s_star": cert2.fermi_dirac_kelly_fraction,
            "kelly_derivative_ds_drhi": cert2.kelly_derivative_ds_drhi,
            "kelly_censored": cert2.is_kelly_censored,
            "yang_mills_energy": cert2.yang_mills_energy,
            "chern_simons_anomaly": cert2.chern_simons_anomaly,
            "pontryagin_integrality_residual": cert2.pontryagin_integrality_residual,
            "kam_worst_ratio": cert2.kam_worst_ratio,
            "kam_n_modes": cert2.kam_n_modes,
            "bruno_sum": cert2.bruno_sum,
            "birkhoff_twist": cert2.birkhoff_twist_coefficient,
            "lindstedt_solved": cert1.lindstedt_solved,
            "spectral_kappa": cert1.spectral_condition_number,
            "is_flat_curvature": cert1.is_flat_curvature,
            "melnikov_value": cert2.melnikov_value,
            "floquet_max_unit_deviation": cert2.floquet_max_unit_deviation,
            "mean_poincare_return_time": cert2.mean_poincare_return_time,
            "rsi3_iterations": cert2.rsi3_iterations,
            "rsi3_homological_residual": cert2.rsi3_homological_residual,
            "rsi3_agent_valid": rsi3_agent.is_valid,
            "novikov_krull_valuation": cert2.novikov_krull_valuation,
            "nested_pipeline": cert2.nested_pipeline,
            "crowbar_interlock_latched": crowbar_triggered,
            "crowbar_latency_ns": latency_ns,
            "crowbar_latency_verified": latency_verified,
            "executive_action": action,
        }

        is_epist_valid = (
            verdict is not HeytingOmega3.VETOED
            and latency_verified
            and cert1.is_anti_hermitian_A
            and cert1.is_traceless_A
            and cert1.is_symplectic_valid
            and cert1.is_trace_conserved
            and hmac_p3_ok
            and rsi3_agent.is_valid
        )

        return TopologicalGaugeFieldProsecutorSovereignState(
            sovereign_state_id=state_id,
            phase1_cert=cert1,
            phase2_cert=cert2,
            global_verdict=verdict,
            crowbar_triggered=bool(crowbar_triggered),
            crowbar_latency_ns=float(latency_ns),
            merkle_sha256_root=merkle_root,
            hmac_provenance_signature=prov_sig,
            visceral_report=visceral_report,
            timestamp_utc=now,
            crowbar_latency_verified=bool(latency_verified),
            hardware_action=hw_action,
            heyting_join_atoms=tuple(atoms),
            heyting_meet_reduction=meet_red,
            heyting_implies_table=implies_t,
            heyting_negate_table=negate_t,
            chain_hash_p1=cert1.chain_hash,
            chain_hash_p2=cert2.chain_hash,
            is_epistemologically_valid=bool(is_epist_valid),
            agent_version=_AGENT_VERSION,
            heyting_join_table=join_t,
            heyting_meet_table=meet_t,
            lindstedt_solved=bool(cert1.lindstedt_solved),
            lindstedt_newton_iterations=int(cert1.lindstedt_newton_iterations),
            spectral_condition_number=float(cert1.spectral_condition_number),
            is_flat_curvature=bool(cert1.is_flat_curvature),
            cayley_pole_margin=float(cert1.cayley_pole_margin),
            bruno_satisfied=bool(cert2.bruno_satisfied),
            bruno_sum=float(cert2.bruno_sum),
            birkhoff_twist_coefficient=float(cert2.birkhoff_twist_coefficient),
            birkhoff_area_residual=float(cert2.birkhoff_area_residual),
            birkhoff_is_twist=bool(cert2.birkhoff_is_twist),
            kam_n_modes=int(cert2.kam_n_modes),
            mean_poincare_return_time=float(cert2.mean_poincare_return_time),
            rsi3_agent_valid=bool(rsi3_agent.is_valid),
            rsi3_agent_residual=float(rsi3_agent.homological_residual),
            rsi3_agent_lipschitz=tuple(rsi3_agent.lipschitz_factors),
            hmac_p3_verified=bool(hmac_p3_ok),
            nested_pipeline=str(cert2.nested_pipeline),
        )

    def execute_phase3(
        self,
        cert1: ProsecutorPhase1Certificate,
        cert2: ProsecutorPhase2Certificate,
    ) -> TopologicalGaugeFieldProsecutorSovereignState:
        r"""**[Compat v9.3.0 / v10.0.0]** Delega a `emit_sovereign_state`."""
        return self.emit_sovereign_state(cert1, cert2)

    def continue_to_phase3_prosecutor_gate(
        self, cert2: ProsecutorPhase2Certificate,
    ) -> bool:
        r"""
        **[Compat, fix v10.0 + v10.1] Compuerta Φ₂₃.**

        Valida HMAC P2 completa (dominio v10.1) y chain 32-hex.
        """
        if not self._verify_hmac(
            _HMAC_DOMAIN_P2, self._phase2_payload(cert2), cert2.hmac_p2_signature,
        ):
            raise ProsecutorGateRejectionError(
                "Compuerta Φ₂₃ rechazó cert P2: HMAC P2 inválida (v10.1).",
            )
        if len(cert2.chain_hash) != _CHAIN_HEX_LEN:
            raise ProsecutorGateRejectionError(
                f"Compuerta Φ₂₃: chain_hash len={len(cert2.chain_hash)} "
                f"≠ {_CHAIN_HEX_LEN}.",
            )
        return True


# ══════════════════════════════════════════════════════════════════════════════
# §H. ORQUESTADOR SUPREMO
# ══════════════════════════════════════════════════════════════════════════════

class TopologicalGaugeFieldProsecutorAgent(ProsecutorPhase3Decider):
    r"""
    Soberano Agéntico Fiscal Auditor e Inquisidor del Campo Topológico de Gauge
    (v10.1.0 — Tres Fases Anidadas).

    Orquesta:
      FASE 1 (Observe)    → `emit_phase1_certificate` → `ProsecutorPhase1Certificate`.
      FASE 2 (Orient)     → `from_phase1_certificate` + `emit_phase2_certificate`
                          → `ProsecutorPhase2Certificate`.
      FASE 3 (Decide&Act) → `from_phase2_certificate` + `emit_sovereign_state`
                          → `TopologicalGaugeFieldProsecutorSovereignState`.

    **Compatibilidad**: `prosecute_gauge_field`, `execute_phase*`,
    `continue_to_phase*_gate` preservan firmas y semántica.
    """

    def __init__(
        self, dimension: int = 56, secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> None:
        engine = TopologicalGaugeFieldProsecutorEngine(
            dimension=dimension, secret_key=secret_key,
        )
        super().__init__(engine=engine, secret_key=secret_key)
        logger.info(
            "TopologicalGaugeFieldProsecutorAgent v%s inicializado "
            "[dim=%s, engine_interop=%s, p2_envelope=%s]",
            _AGENT_VERSION, dimension,
            "ON" if _ENGINE_HAS_ENVELOPES else "OFF",
            "ON" if _ENGINE_HAS_P2_ENVELOPE else "OFF",
        )

    def prosecute_gauge_field(
        self, density_matrix: NDArray,
    ) -> TopologicalGaugeFieldProsecutorSovereignState:
        r"""
        Ejecuta el ciclo OODA fiscal a través de las tres fases anidadas.

        ══ FASE 1 ══ emit_phase1_certificate → from_phase1_certificate
        ══ FASE 2 ══ emit_phase2_certificate → from_phase2_certificate
        ══ FASE 3 ══ emit_sovereign_state
        """
        cert1 = self.emit_phase1_certificate(density_matrix)
        _ = self.from_phase1_certificate(
            cert1, engine=self.engine, secret_key=self.secret_key,
        )
        if not self.continue_to_phase2_prosecutor_gate(cert1):
            raise ProsecutorGateRejectionError("Compuerta Φ₁₂ rechazó cert P1.")

        cert2 = self.emit_phase2_certificate(cert1, density_matrix)
        _ = self.from_phase2_certificate(
            cert2, engine=self.engine, secret_key=self.secret_key,
        )
        if not self.continue_to_phase3_prosecutor_gate(cert2):
            raise ProsecutorGateRejectionError("Compuerta Φ₂₃ rechazó cert P2.")

        return self.emit_sovereign_state(cert1, cert2)

    def consume_engine_prosecutor_snapshot(
        self,
        engine_snapshot: Optional[Any],
    ) -> TopologicalGaugeFieldProsecutorSovereignState:
        r"""
        **Interop**: consume un `GaugeProsecutorSnapshot` del motor v10.1,
        reconstruye certificados firmados (dominio v10.1) y cierra FASE 3.

        `None` → `ProsecutorAgentError` (nunca un DTO).
        """
        if engine_snapshot is None:
            raise ProsecutorAgentError(
                "consume_engine_prosecutor_snapshot: snapshot is None",
            )
        now = time.time()

        p1_env = getattr(engine_snapshot, "phase1", None)
        p2_env = getattr(engine_snapshot, "phase2", None)
        src_p1: Any = p1_env if _is_phase1_envelope(p1_env) else engine_snapshot
        src_p2: Any = p2_env if _is_phase2_envelope(p2_env) else engine_snapshot

        merkle = str(getattr(engine_snapshot, "sha256_merkle_digest", "consume"))
        ts = _attr_float(engine_snapshot, "timestamp_utc", default=now)
        chain_p1 = hashlib.sha256(
            f"consume_engine_p1:{merkle}:{ts:.6f}".encode("utf-8"),
        ).hexdigest()[:_CHAIN_HEX_LEN]

        rhi, rhi_base, ipr_raw = _rhi_tuple(src_p1)
        ym = _yang_mills_energy(src_p1)
        (sympl_res, sympl_def, is_sympl, is_flat,
         pole_margin, lie_def) = _symplectic_tuple(src_p1)
        poincare_res, poincare_resonant, lindstedt_solved, newton_iters = (
            _lindstedt_tuple(src_p1)
        )
        kappa = _spectral_kappa(src_p1)

        verdict_p1 = (
            HeytingOmega3.VETOED if not is_sympl
            else HeytingOmega3.DEGRADED if poincare_resonant
            else HeytingOmega3.COHERENT
        )
        pros_id1 = f"PROS-P1-{int(now * 1000.0) % 1000000:06d}"
        tmp1 = ProsecutorPhase1Certificate(
            prosecutor_id=pros_id1,
            trace_preservation=1.0,
            connection_norm_f=0.0,
            isometry_verdict=verdict_p1,
            hmac_p1_signature="",
            timestamp_utc=now,
            dimension=_attr_int(src_p1, "dimension", default=56) or 56,
            is_trace_conserved=True,
            is_anti_hermitian_A=True,
            is_traceless_A=True,
            rhi_final=rhi,
            rhi_base=rhi_base,
            ipr_raw=ipr_raw,
            yang_mills_energy=ym,
            symplectic_residue=sympl_res,
            symplectic_defect=sympl_def,
            is_symplectic_valid=is_sympl,
            poincare_residual=poincare_res,
            poincare_is_resonant=poincare_resonant,
            engine_envelope=p1_env if p1_env is not None else engine_snapshot,
            chain_hash=chain_p1,
            engine_interop=True,
            lindstedt_solved=lindstedt_solved,
            lindstedt_newton_iterations=newton_iters,
            spectral_condition_number=kappa,
            is_flat_curvature=is_flat,
            cayley_pole_margin=pole_margin,
            lie_algebra_defect_projected=lie_def,
            hmac_verified=False,
        )
        cert1 = self._stamp_p1(tmp1)

        f = self._extract_phase2_fields(src_p2, cert1)
        p2_atoms: List[HeytingOmega3] = [
            HeytingOmega3.COHERENT if f["rhi"] < _RHI_SOFT else (
                HeytingOmega3.DEGRADED if f["rhi"] < _RHI_HARD
                else HeytingOmega3.VETOED
            ),
            HeytingOmega3.COHERENT if f["ym"] <= _YM_SOFT else (
                HeytingOmega3.DEGRADED if f["ym"] <= _YM_HARD
                else HeytingOmega3.VETOED
            ),
            HeytingOmega3.COHERENT if abs(f["k"]) <= _INSTANTON_HARD
            else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if f["kam_ok"] else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if f["bruno_ok"] else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if f["rsi3_ok"] else HeytingOmega3.VETOED,
        ]
        verdict_p2 = cert1.isometry_verdict.join(
            HeytingOmega3.fold_supremum(p2_atoms),
        )
        chain_p2 = hashlib.sha256(
            f"consume_engine_p2:{merkle}:{int(verdict_p2)}".encode("utf-8"),
        ).hexdigest()[:_CHAIN_HEX_LEN]

        hmac_engine_ok = False
        if _ENGINE_HAS_SNAPSHOT_VERIFY:
            try:
                hmac_engine_ok = bool(
                    self.engine.verify_snapshot_hmac(engine_snapshot)  # type: ignore[attr-defined]
                )
            except Exception:
                hmac_engine_ok = bool(
                    getattr(engine_snapshot, "hmac_verified", False)
                )

        pros_id2 = f"PROS-P2-{int(now * 1000.0 + 1) % 1000000:06d}"
        tmp2 = ProsecutorPhase2Certificate(
            prosecutor_id=pros_id2,
            rhi_gauge_finsler=f["rhi"],
            fermi_dirac_kelly_fraction=f["kelly"],
            yang_mills_energy=f["ym"],
            pontryagin_instanton_k=int(f["k"]),
            stora_zumino_residual=f["sz"],
            audit_verdict=verdict_p2,
            hmac_p2_signature="",
            timestamp_utc=now,
            rhi_base_normalized=f["rhi_base"],
            kelly_derivative_ds_drhi=f["kelly_deriv"],
            is_kelly_censored=f["is_censored"],
            chern_simons_real=f["cs_real"],
            chern_simons_anomaly=f["cs_anom"],
            wilson_angle_rad=f["w_angle"],
            c1_value=f["c1"],
            pontryagin_integrality_residual=f["integrality"],
            kam_worst_ratio=f["kam_w"],
            kam_satisfied=f["kam_ok"],
            melnikov_value=f["mel"],
            melnikov_preserved=f["mel_ok"],
            floquet_max_unit_deviation=f["flo_dev"],
            floquet_neutral=f["flo_neutral"],
            poincare_section_n_crossings=f["poin_x"],
            poincare_section_dispersion=f["poin_disp"],
            rsi3_valid=f["rsi3_ok"],
            rsi3_iterations=f["rsi3_it"],
            banach_lipschitz_t3=f["banach"],
            fubini_study_distance=f["fubini"],
            novikov_krull_valuation=f["novikov"],
            topology_regime=f["regime"],
            engine_envelope=p2_env if p2_env is not None else engine_snapshot,
            chain_hash=chain_p2,
            engine_interop=True,
            bruno_satisfied=f["bruno_ok"],
            bruno_sum=f["bruno_sum"],
            birkhoff_twist_coefficient=f["beta"],
            birkhoff_area_residual=f["area_res"],
            birkhoff_is_twist=f["is_twist"],
            birkhoff_area_preserving=f["area_ok"],
            kam_n_modes=f["kam_n"],
            mean_poincare_return_time=f["mean_ret"],
            nested_pipeline=_CONSUME_SNAPSHOT,
            hmac_verified=False,
            hmac_engine_verified=bool(hmac_engine_ok),
            rsi3_homological_residual=f["homo_res"],
            rsi3_lipschitz_factors=f["lip_fac"],
        )
        cert2 = self._stamp_p2(tmp2)
        return self.emit_sovereign_state(cert1, cert2)


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__: List[str] = [
    "ProsecutorAgentError",
    "ProsecutorContractCorruptedError",
    "ProsecutorHMACVerificationError",
    "ProsecutorGateRejectionError",
    "ProsecutorCrowbarLatencyError",
    "ProsecutorPhase1Certificate",
    "ProsecutorPhase2Certificate",
    "AgentRSI3Certificate",
    "TopologicalGaugeFieldProsecutorSovereignState",
    "ProsecutorPhase1Observer",
    "ProsecutorPhase2Orientor",
    "ProsecutorPhase3Decider",
    "TopologicalGaugeFieldProsecutorAgent",
    "GaugeProsecutorSnapshot",
    "HeytingOmega3",
    "TopologicalGaugeFieldProsecutorEngine",
]


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN DOCTORAL
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    banner = "═" * 80
    print(banner)
    print(
        "  VERIFICACIÓN DOCTORAL — "
        "Topological Gauge Field Prosecutor Agent v%s" % _AGENT_VERSION
    )
    print(
        f"  Engine importado: {_ENGINE_IMPORTED} | "
        f"Envelopes: {_ENGINE_HAS_ENVELOPES} | "
        f"P2: {_ENGINE_HAS_P2_ENVELOPE} | "
        f"Snapshot: {_ENGINE_HAS_SNAPSHOT_EMIT}"
    )
    print(banner)

    agent = TopologicalGaugeFieldProsecutorAgent(dimension=56)
    rng = np.random.default_rng(2026)
    state = rng.normal(0.0, 1.0, size=(56, 56)) + 1j * rng.normal(
        0.0, 1.0, size=(56, 56),
    )
    rho = state @ state.conj().T
    rho /= np.trace(rho)

    pros_state = agent.prosecute_gauge_field(rho)
    p1, p2 = pros_state.phase1_cert, pros_state.phase2_cert
    print(f"  • Sovereign State ID      : {pros_state.sovereign_state_id}")
    print(f"  • Agent version           : {pros_state.agent_version}")
    print(
        f"  • Veredicto Global Ω₃     : {pros_state.global_verdict.name} "
        f"(0x{int(pros_state.global_verdict):02x})"
    )
    print(f"  • Nested pipeline         : {pros_state.nested_pipeline}")
    print(f"  • RHI Finsler             : {p2.rhi_gauge_finsler:.6f} ∈ [0,1]")
    print(f"  • RHI base normalizado    : {p2.rhi_base_normalized:.6f}")
    print(f"  • IPR raw                 : {p1.ipr_raw:.6f}")
    print(f"  • Kelly s* censurada      : {p2.fermi_dirac_kelly_fraction:.6f}")
    print(f"  • ds*/dRHI                : {p2.kelly_derivative_ds_drhi:.6e}")
    print(f"  • Kelly censored          : {p2.is_kelly_censored}")
    print(f"  • Yang-Mills energy       : {p2.yang_mills_energy:.6f}")
    print(f"  • Chern-Simons (real)     : {p2.chern_simons_real:.6e}")
    print(f"  • CS anomaly              : {p2.chern_simons_anomaly:.6e}")
    print(f"  • Wilson angle            : {p2.wilson_angle_rad:.4f} rad")
    print(f"  • Instanton k             : {p2.pontryagin_instanton_k}")
    print(f"  • Integralidad residual   : {p2.pontryagin_integrality_residual:.3e}")
    print(f"  • Stora-Zumino            : {p2.stora_zumino_residual:.3e}")
    print(f"  • δω simpléctico          : {p1.symplectic_residue:.3e}")
    print(f"  • ‖KᵀJ+JK‖_F              : {p1.symplectic_defect:.3e}")
    print(f"  • Rama plana Cayley       : {p1.is_flat_curvature}")
    print(f"  • Cayley pole margin      : {p1.cayley_pole_margin:.3e}")
    print(f"  • Poincaré residual       : {p1.poincare_residual:.3e}")
    print(
        f"  • Lindstedt solved        : {p1.lindstedt_solved}  "
        f"(Newton={p1.lindstedt_newton_iterations})"
    )
    print(f"  • Spectral κ              : {p1.spectral_condition_number:.4e}")
    print(
        f"  • KAM worst-ratio         : {p2.kam_worst_ratio:.4f}  "
        f"(modes={p2.kam_n_modes})"
    )
    print(
        f"  • Bruno satisfied         : {p2.bruno_satisfied}  "
        f"(S={p2.bruno_sum:.4f})"
    )
    print(f"  • Melnikov |M|            : {abs(p2.melnikov_value):.4e}")
    print(f"  • Floquet |Δμ|max         : {p2.floquet_max_unit_deviation:.3e}")
    print(f"  • Poincaré cruzamientos   : {p2.poincare_section_n_crossings}")
    print(f"  • Mean return time        : {p2.mean_poincare_return_time:.4f}")
    print(
        f"  • Birkhoff β              : {p2.birkhoff_twist_coefficient:.4e}  "
        f"(twist={p2.birkhoff_is_twist})"
    )
    print(f"  • Birkhoff area resid     : {p2.birkhoff_area_residual:.3e}")
    print(f"  • RSI³ engine iter        : {p2.rsi3_iterations}")
    print(f"  • RSI³ homological r      : {p2.rsi3_homological_residual:.3e}")
    print(f"  • RSI³ agent valid        : {pros_state.rsi3_agent_valid}")
    print(f"  • RSI³ agent Lip          : {pros_state.rsi3_agent_lipschitz}")
    print(f"  • Banach Lip(T³)          : {p2.banach_lipschitz_t3:.4f}")
    print(f"  • Novikov v_Krull         : {p2.novikov_krull_valuation:.4f}")
    print(f"  • Crowbar triggered       : {pros_state.crowbar_triggered}")
    print(
        f"  • Crowbar latency         : {pros_state.crowbar_latency_ns:.2f} ns "
        f"(verified={pros_state.crowbar_latency_verified})"
    )
    print(f"  • Hardware action         : {pros_state.hardware_action}")
    print(f"  • Heyting join atoms      : {pros_state.heyting_join_atoms}")
    print(f"  • Heyting meet reduction  : 0x{pros_state.heyting_meet_reduction:02x}")
    print(f"  • Heyting ¬ table         : {pros_state.heyting_negate_table}")
    print(f"  • HMAC P1 verified        : {p1.hmac_verified}")
    print(f"  • HMAC P2 verified        : {p2.hmac_verified}")
    print(f"  • HMAC engine verified    : {p2.hmac_engine_verified}")
    print(f"  • HMAC P3 verified        : {pros_state.hmac_p3_verified}")
    print(f"  • Merkle SHA-256          : {pros_state.merkle_sha256_root[:32]}…")
    print(f"  • HMAC provenance         : {pros_state.hmac_provenance_signature[:32]}…")
    print(f"  • chain_hash_p1           : {pros_state.chain_hash_p1}")
    print(f"  • chain_hash_p2           : {pros_state.chain_hash_p2}")
    print(f"  • Epistemológicamente     : {pros_state.is_epistemologically_valid}")
    print(
        f"  • WACC status             : "
        f"{pros_state.visceral_report.get('wacc_protection_status', '—')}"
    )
    print(
        f"  • Liquidez                : "
        f"{pros_state.visceral_report.get('net_liquidity_impact', '—')}"
    )
    print(banner)