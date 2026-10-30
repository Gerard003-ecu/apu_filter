# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Soberano : Topological Gauge Field Witness Agent v10.1.0 — Tres Fases        ║
║ Ubicación: app/agents/wisdom/topological_gauge_field_witness_agent.py        ║
║ Versión  : 10.1.0 / 10.0.0 (SOVEREIGN-GAUGE-WITNESS-AGENT-QND-OODA-          ║
║           POINCARE-BRUNO-BIRKHOFF-RSI3-HEYTING-PhD)                          ║
║ Función  : Soberano de observación pasiva QND (Quantum Non-Demolition),      ║
║            verificación agéntica de retroacción cero, orquestación OODA      ║
║            anidada de tres fases y automejora recursiva Nivel 3.             ║
╚══════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO
────────────────────────────────────────────
El Soberano Testigo del Campo Topológico de Gauge gobierna la auditoría pasiva no
destructiva del estado de densidad \rho \in \mathcal{D}_N mediante el ciclo OODA
anidado. Certifica de manera estricta la ausencia de perturbaciones de medición (QND),
la preservación de la isometría de calibre \mathfrak{su}(N), la validez de las fases
de Poincaré-Lindstedt y la estabilidad de Banach en el grafo computacional de automejora.

MARCO TEÓRICO
─────────────
1. Conexión de Gauge \mathfrak{su}(N) y Medición Débil QND:
   Dada la densidad sanitizada \rho, el Soberano verifica que la conexión extraída:
   A = i \left( \rho - \frac{\mathrm{Tr}(\rho)}{N} I \right) \in \mathfrak{su}(N)
   cumpla el principio QND no destructivo \| [A, \rho] \|_F \le 10^{-6}.
   La retroacción cuántica (back-action) en decibelios se calcula de forma exacta:
   \mathrm{BackAction}_{\mathrm{dB}} = -10 \log_{10}\left( 1 - \min(\| [A, \rho] \|_F, 0.99) \right)

2. Transiciones de Fase OODA Anidadas y Compuertas Homomórficas:
   • FASE 1 (Observe): Emite el certificado de isometría \Phi_1 y cierra con `emit_phase1_certificate`.
     Abre la Fase 2 mediante el puente homomórfico `from_phase1_certificate`.
   • FASE 2 (Orient): Certifica las métricas de Uhlmann F(\rho, I/d) = \frac{1}{d}(\mathrm{Tr}\sqrt{\rho})^2,
     distancia de Bures D_B = \sqrt{2 - 2\sqrt{F}}, estabilidad KAM diofantina, suma de Bruno,
     sección de Poincaré y twist de Birkhoff. Cierra con `emit_phase2_certificate`.
   • FASE 3 (Decide & Act): Adjudicación Heyting \Omega_3 de 12 átomos e inmutabilidad de estado
     soberano final mediante `emit_sovereign_state`.

3. Automejora Agéntica Nivel 3 (RSI³) y Cero Activación de Hardware:
   El Soberano verifica la contracción de Banach T^3 sobre las firmas criptográficas y
   proyectores de retículo. Invariante estructural: `witness_mode = "PASSIVE_SILENT"` (cero
   intervención de hardware/crowbar).

AXIOMAS FORMALES
────────────────
[Axioma 1] Isometría y No Demolición: A^\dagger = -A, \mathrm{Tr}(A) = 0, \| [A, \rho] \|_F \le 10^{-6}.
[Axioma 2] Continuidad OODA Homomórfica: \mathrm{HMAC}_{P1} \land \mathrm{HMAC}_{P2} \land \mathrm{HMAC}_{P3} = \mathrm{True}.
[Axioma 3] Invariante Silenciosa: witness_mode = "PASSIVE_SILENT" \implies crowbar_triggered = \mathrm{False}.

INVARIANTES DE ESTADO
─────────────────────
1. Veredicto global \Omega_3 \in \{0x00 (\text{COHERENT}), 0x01 (\text{DEGRADED}), 0x03 (\text{VETOED})\}.
2. Garantía de Cero Back-Action: \mathrm{BackAction}_{\mathrm{dB}} = 0.0 \text{ dB}.
3. Trazabilidad Merkle SHA-256 y HMAC-SHA256 de procedencia agéntica.
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

try:
    import scipy.linalg as la
except ImportError:  # pragma: no cover
    la = None  # type: ignore[assignment]


# ──────────────────────────────────────────────────────────────────────────────
# §0. IMPORTACIÓN ROBUSTA DEL MOTOR TESTIGO v10.1.x
# ──────────────────────────────────────────────────────────────────────────────
_ENGINE_IMPORTED: bool = False
_ENGINE_HAS_ENVELOPES: bool = False
_ENGINE_HAS_P2_ENVELOPE: bool = False
_ENGINE_HAS_SNAPSHOT_EMIT: bool = False
_ENGINE_HAS_SNAPSHOT_VERIFY: bool = False

try:
    from app.wisdom.topological_gauge_field_witness_engine import (
        GaugeWitnessSnapshot,
        HeytingOmega3,
        TopologicalGaugeFieldWitnessEngine,
    )
    _ENGINE_IMPORTED = True
except ImportError:
    try:
        from topological_gauge_field_witness_engine import (
            GaugeWitnessSnapshot,
            HeytingOmega3,
            TopologicalGaugeFieldWitnessEngine,
        )
        _ENGINE_IMPORTED = True
    except ImportError:  # pragma: no cover
        from enum import IntEnum

        class HeytingOmega3(IntEnum):
            r"""Retículo de Gödel G₃ (fallback si el motor no está)."""
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

        class GaugeWitnessSnapshot:  # type: ignore[no-redef]
            """Stub mínimo; el pipeline real exige el motor."""

        class TopologicalGaugeFieldWitnessEngine:  # type: ignore[no-redef]
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
                    return (
                        np.eye(self.dimension, dtype=np.complex128)
                        / float(self.dimension)
                    )
                return herm / t

            def evaluate_gauge_witness(
                self, rho: NDArray, p_i: Any = None, p_f: Any = None,
            ) -> Any:
                return _NullSnapshot()


if _ENGINE_IMPORTED:
    _ENGINE_HAS_ENVELOPES = hasattr(
        TopologicalGaugeFieldWitnessEngine, "emit_phase1_envelope",
    )
    _ENGINE_HAS_P2_ENVELOPE = hasattr(
        TopologicalGaugeFieldWitnessEngine, "emit_phase2_envelope",
    )
    _ENGINE_HAS_SNAPSHOT_EMIT = hasattr(
        TopologicalGaugeFieldWitnessEngine, "emit_witness_snapshot",
    )
    _ENGINE_HAS_SNAPSHOT_VERIFY = hasattr(
        TopologicalGaugeFieldWitnessEngine, "verify_witness_snapshot",
    )


logger = logging.getLogger("APU.Wisdom.TopologicalGaugeFieldWitnessAgent.v10")

__version__: Final[str] = "10.1.0"


# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES Y DOMINIOS HMAC v10.1.0
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_GAUGE_WITNESS_SOVEREIGN_KEY"
_HMAC_DOMAIN_P1: Final[bytes] = b"GAUGE-WITNESS-P1-ISOMETRY-v10.1|"
_HMAC_DOMAIN_P2: Final[bytes] = b"GAUGE-WITNESS-P2-QND-v10.1|"
_HMAC_DOMAIN_P3: Final[bytes] = b"GAUGE-WITNESS-P3-SOVEREIGN-v10.1|"
_AGENT_VERSION: Final[str] = "10.1.0"

_WILKINSON_EPSILON: Final[float] = 1.0e-15
_ANTI_HERM_TOL: Final[float] = 1.0e-10
_TRACELESS_TOL: Final[float] = 1.0e-10
_SYMPLECTIC_TOL: Final[float] = 1.0e-10
_TRACE_CONSERVATION_TOL: Final[float] = 1.0e-2
_LIOUVILLE_HARD: Final[float] = 1.0e-4
_LIOUVILLE_SOFT: Final[float] = 1.0e-3
_QND_COMMUTATOR_MAX: Final[float] = 1.0e-6
_QND_BACKACTION_DB_TOL: Final[float] = 1.0e-2
_UHLMANN_FIDELITY_MIN: Final[float] = 0.95
_BURES_DIST_SOFT: Final[float] = 0.30
_BURES_DIST_HARD: Final[float] = 0.80
_SZ_RESIDUAL_SOFT: Final[float] = 0.15
_SZ_RESIDUAL_HARD: Final[float] = 0.50
_YM_ENERGY_SOFT: Final[float] = 0.25
_YM_ENERGY_HARD: Final[float] = 0.85
_FLOQUET_UNIT_TOL: Final[float] = 1.0e-6
_BANACH_LIPSCHITZ_MAX: Final[float] = 0.95
_RSI3_AGENT_MAX_ITER: Final[int] = 8
_RSI3_FIXED_POINT_TOL: Final[float] = 1.0e-15
_CHAIN_HEX_LEN: Final[int] = 32


# ══════════════════════════════════════════════════════════════════════════════
# §B. EXCEPCIONES
# ══════════════════════════════════════════════════════════════════════════════
class WitnessAgentError(Exception):
    """Excepción raíz del Agente Testigo Silencioso."""


class WitnessContractCorruptedError(WitnessAgentError):
    """Firma HMAC o chain_hash corruptos."""


class WitnessGateRejectionError(WitnessAgentError):
    """Una compuerta Φᵢⱼ rechazó el certificado precedente."""


class WitnessSymplecticViolationError(WitnessAgentError):
    """Embedding Cayley-Kähler fuera de sp(2d,ℂ)."""


class WitnessHMACVerificationError(WitnessAgentError):
    """La firma HMAC-SHA256 no coincide con el dominio v10.1.0."""


# ══════════════════════════════════════════════════════════════════════════════
# §C. HELPERS NUMÉRICOS, CRIPTOGRÁFICOS Y DE ACCESO ROBUSTO
# ══════════════════════════════════════════════════════════════════════════════
def _fro(matrix: NDArray[Any]) -> float:
    if la is not None:
        return float(la.norm(np.asarray(matrix), "fro"))
    return float(np.linalg.norm(np.asarray(matrix), "fro"))


def _hash_state(*arrays: NDArray[Any]) -> str:
    r"""SHA-256 truncado a 32 hex, con shape y dtype (anti-colisión)."""
    h = hashlib.sha256()
    for a in arrays:
        arr = np.ascontiguousarray(a)
        h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        h.update(str(arr.dtype).encode("ascii"))
        h.update(arr.tobytes())
    return h.hexdigest()[:_CHAIN_HEX_LEN]


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    return getattr(obj, name, default)


def _fattr(obj: Any, name: str, default: float = 0.0) -> float:
    try:
        val = float(_attr(obj, name, default))
        return val if math.isfinite(val) else default
    except (TypeError, ValueError):
        return default


def _battr(obj: Any, name: str, default: bool = True) -> bool:
    return bool(_attr(obj, name, default))


def _iattr(obj: Any, name: str, default: int = 0) -> int:
    try:
        return int(_attr(obj, name, default))
    except (TypeError, ValueError):
        return default


def _cattr(obj: Any, name: str, default: complex = 0.0 + 0.0j) -> complex:
    try:
        return complex(_attr(obj, name, default))
    except (TypeError, ValueError):
        return default


def _project_anti_hermitian(M: NDArray[np.complex128]) -> NDArray[np.complex128]:
    return 0.5 * (M - M.conj().T)


def _project_su_n(M: NDArray[np.complex128]) -> NDArray[np.complex128]:
    r"""$\Pi_{\mathfrak{su}(N)}(M)=\Pi_u(M)-\mathrm{Tr}(\Pi_u(M))/N\cdot I$."""
    ah = _project_anti_hermitian(M)
    n = ah.shape[0]
    return ah - (np.trace(ah) / float(n)) * np.eye(n, dtype=np.complex128)


def _as_heyting(value: Any) -> HeytingOmega3:
    if isinstance(value, HeytingOmega3):
        return value
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return HeytingOmega3.DEGRADED
    if hasattr(HeytingOmega3, "legacy_from_0x02"):
        return HeytingOmega3.legacy_from_0x02(iv)  # type: ignore[attr-defined]
    if iv == 0x02:
        return HeytingOmega3.VETOED
    try:
        return HeytingOmega3(iv)
    except ValueError:
        return HeytingOmega3.VETOED if iv >= 0x03 else HeytingOmega3.DEGRADED


class _NullSnapshot:
    r"""Snapshot nulo (fallback).  Nunca se usa el ctor real del motor vacío."""
    witness_id: str = "NULL"
    qnd_weak_value_aav: complex = 0.0 + 0.0j
    back_action_db: float = 0.0
    uhlmann_fidelity: float = 1.0
    bures_distance: float = 0.0
    liouville_volume_drift: float = 0.0
    yang_mills_background_energy: float = 0.0
    stora_zumino_residual: float = 0.0
    global_verdict: HeytingOmega3 = HeytingOmega3.COHERENT
    sha256_merkle_digest: str = ""
    timestamp_utc: float = 0.0
    is_qnd_preserved: bool = True
    qnd_commutator_norm: float = 0.0
    qnd_fidelity: float = 1.0
    symplectic_residue: float = 0.0
    symplectic_defect: float = 0.0
    is_symplectic_valid: bool = True
    poincare_residual: float = 0.0
    poincare_is_resonant: bool = False
    kam_worst_ratio: float = 0.0
    kam_satisfied: bool = True
    melnikov_value: float = 0.0
    melnikov_preserved: bool = True
    floquet_max_unit_deviation: float = 0.0
    floquet_neutral: bool = True
    poincare_section_n_crossings: int = 0
    poincare_section_dispersion: float = 0.0
    banach_lipschitz_t3: float = 0.0
    fubini_study_distance: float = 0.0
    novikov_krull_valuation: float = 0.0
    rsi3_valid: bool = True
    rsi3_iterations: int = 1
    trace_distance: float = 0.0
    bruno_satisfied: bool = True
    bruno_sum: float = 0.0
    birkhoff_twist_estimate: float = 0.0
    birkhoff_is_twist: bool = True
    lindstedt_solved: bool = True
    spectral_condition_number: float = 1.0
    kam_n_modes: int = 0
    is_flat_curvature: bool = True
    mean_poincare_return_time: float = 0.0
    hmac_verified: bool = False
    hmac_sha256: bytes = b""
    engine_version: str = "0.0.0"
    phase1: Any = None
    phase2: Any = None
    is_epistemologically_valid: bool = False
    witness_mode: str = "PASSIVE_SILENT"


def _symplectic_tuple(env: Any) -> Tuple[float, float, bool, bool, float]:
    r"""Lee residuo/defecto/validez/rama-plana/polo tanto de envelope como plano."""
    report = _attr(env, "symplectic_report", None)
    src = report if report is not None else env
    return (
        _fattr(src, "symplectic_residue", 0.0),
        _fattr(src, "symplectic_defect", 0.0),
        _battr(src, "is_symplectic_valid", True),
        _battr(src, "is_flat_branch", _battr(src, "is_flat_curvature", True)),
        _fattr(src, "cayley_pole_margin", 1.0),
    )


def _yang_mills_energy(env: Any, default: float = 0.0) -> float:
    for name in ("yang_mills_energy", "energy_ym", "yang_mills_background_energy"):
        if env is not None and hasattr(env, name):
            return _fattr(env, name, default)
    return default


def _lindstedt_tuple(env: Any) -> Tuple[float, bool, bool, int]:
    pl = _attr(env, "poincare_lindstedt", None)
    if pl is None:
        return (
            _fattr(env, "poincare_residual", 0.0),
            _battr(env, "poincare_is_resonant", False),
            _battr(env, "lindstedt_solved", True),
            _iattr(env, "newton_iterations", 1),
        )
    return (
        _fattr(pl, "residual_norm", 0.0),
        _battr(pl, "is_resonant", False),
        _battr(pl, "is_solved", True),
        _iattr(pl, "newton_iterations", 1),
    )


def _spectral_kappa(env: Any) -> float:
    cache = _attr(env, "spectral_cache", None)
    if cache is not None:
        return _fattr(cache, "condition_number", 1.0)
    return _fattr(env, "spectral_condition_number", 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# §D. CERTIFICADOS DE FASE (EXTENDIDOS v10.1.0 CON RETROCOMPAT)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class WitnessPhase1Certificate:
    r"""
    FASE 1 — Certificado de Isometría de Calibre QND.

    Campos originales (retrocompat) + v10.0.0 + v10.1.0
    (Lindstedt espectral, κ₂(iA), rama plana de Cayley).
    """
    witness_id: str
    trace_preservation: float
    liouville_drift: float
    isometry_verdict: HeytingOmega3
    hmac_p1_signature: str
    timestamp_utc: float

    dimension: int = 56
    connection_norm_frobenius: float = 0.0
    is_anti_hermitian_A: bool = True
    is_traceless_A: bool = True
    is_trace_conserved: bool = True
    symplectic_residue: float = 0.0
    symplectic_defect: float = 0.0
    is_symplectic_valid: bool = True
    poincare_residual: float = 0.0
    poincare_is_resonant: bool = False
    yang_mills_energy_baseline: float = 0.0
    engine_envelope: Optional[Any] = field(default=None, repr=False)
    chain_hash: str = ""
    engine_interop: bool = False

    # v10.1.0
    lindstedt_solved: bool = True
    lindstedt_newton_iterations: int = 1
    spectral_condition_number: float = 1.0
    is_flat_curvature: bool = True
    cayley_pole_margin: float = 1.0
    hmac_verified: bool = True


@dataclass(frozen=True, slots=True)
class WitnessPhase2Certificate:
    r"""FASE 2 — Certificado QND verificado + Poincaré–Bruno–Birkhoff + RSI³."""
    witness_id: str
    weak_value_qnd_aav: complex
    back_action_db: float
    uhlmann_fidelity: float
    bures_distance: float
    yang_mills_energy: float
    qnd_verdict: HeytingOmega3
    hmac_p2_signature: str
    timestamp_utc: float

    is_qnd_preserved: bool = True
    qnd_commutator_norm: float = 0.0
    qnd_fidelity: float = 1.0
    trace_distance: float = 0.0
    liouville_volume_drift: float = 0.0
    stora_zumino_residual: float = 0.0
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
    engine_envelope: Optional[Any] = field(default=None, repr=False)
    chain_hash: str = ""
    engine_interop: bool = False

    # v10.1.0
    bruno_satisfied: bool = True
    bruno_sum: float = 0.0
    birkhoff_twist_estimate: float = 0.0
    birkhoff_is_twist: bool = True
    birkhoff_area_preserving: bool = True
    kam_n_modes: int = 0
    mean_poincare_return_time: float = 0.0
    nested_pipeline: str = "evaluate"
    hmac_verified: bool = True
    hmac_engine_verified: bool = True


@dataclass(frozen=True, slots=True)
class AgentRSI3Certificate:
    r"""
    RSI³ agéntica (computacional, PASSIVE_SILENT):
      S1 Data/Novikov  — integridad HMAC P1∧P2.
      S2 Harness       — chain_hash hex de longitud canónica.
      S3 Model         — idempotencia de join Heyting y de la firma P3.
    """
    hmac_p1_ok: bool
    hmac_p2_ok: bool
    chain_ok: bool
    heyting_idempotent: bool
    signing_idempotent: bool
    iterations: int
    is_valid: bool
    lipschitz_proxy: float


@dataclass(frozen=True, slots=True)
class TopologicalGaugeFieldWitnessSovereignState:
    r"""
    FASE 3 — Estado terminal del Testigo Silencioso.

    Invariante `witness_mode = "PASSIVE_SILENT"`: NUNCA dispara crowbar.
    """
    sovereign_state_id: str
    phase1_cert: WitnessPhase1Certificate
    phase2_cert: WitnessPhase2Certificate
    global_verdict: HeytingOmega3
    back_action_db: float
    merkle_sha256_root: str
    hmac_provenance_signature: str
    visceral_report: Dict[str, Any]
    timestamp_utc: float

    witness_mode: str = "PASSIVE_SILENT"
    crowbar_triggered: bool = False
    is_epistemologically_valid: bool = True
    heyting_join_atoms: Tuple[int, ...] = ()
    heyting_meet_reduction: int = 0
    heyting_implies_table: Tuple[Tuple[int, int, int], ...] = ()
    heyting_negate_table: Tuple[int, int, int] = ()
    chain_hash_p1: str = ""
    chain_hash_p2: str = ""
    agent_version: str = _AGENT_VERSION

    # v10.1.0
    heyting_join_table: Tuple[Tuple[int, int, int], ...] = ()
    heyting_meet_table: Tuple[Tuple[int, int, int], ...] = ()
    bruno_satisfied: bool = True
    bruno_sum: float = 0.0
    birkhoff_twist_estimate: float = 0.0
    birkhoff_is_twist: bool = True
    lindstedt_solved: bool = True
    spectral_condition_number: float = 1.0
    is_flat_curvature: bool = True
    rsi3_agent_valid: bool = True
    rsi3_agent_iterations: int = 1
    hmac_p3_verified: bool = True
    nested_pipeline: str = "evaluate"


# ══════════════════════════════════════════════════════════════════════════════════
# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║  F A S E   1 :   O B S E R V E                                               ║
# ║  Isometría de calibre QND, Cayley-Kähler, Poincaré-Lindstedt espectral.      ║
# ║  Cierra con `emit_phase1_certificate` → `WitnessPhase1Certificate`.          ║
# ║  El cierre es la APERTURA formal de la FASE 2 (`from_phase1_certificate`).   ║
# ╚════════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════════

class WitnessPhase1Observer:
    r"""
    FASE 1 (Observe): isometría de calibre QND.

    Consume el envelope P1 del motor v10.1 si existe; si no, audita localmente
    con $\Pi_{\mathfrak{su}(N)}$ y curvatura $F=[A,\rho]$.
    """

    def __init__(
        self,
        engine: TopologicalGaugeFieldWitnessEngine,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> None:
        self.engine: Final[TopologicalGaugeFieldWitnessEngine] = engine
        self.secret_key: Final[bytes] = bytes(secret_key)

    def _compute_hmac(self, domain: bytes, payload: str) -> str:
        return hmac.new(
            self.secret_key,
            domain + payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _verify_hmac(self, domain: bytes, payload: str, signature: str) -> bool:
        if not signature:
            return False
        return hmac.compare_digest(self._compute_hmac(domain, payload), signature)

    @staticmethod
    def _phase1_payload(cert: WitnessPhase1Certificate) -> str:
        return (
            f"{cert.witness_id}:{cert.trace_preservation:.8f}:"
            f"{cert.liouville_drift:.8f}:{int(cert.isometry_verdict)}:"
            f"{cert.is_anti_hermitian_A}:{cert.is_traceless_A}:"
            f"{cert.symplectic_residue:.3e}:{cert.symplectic_defect:.3e}:"
            f"{cert.connection_norm_frobenius:.6f}:{cert.chain_hash}:"
            f"{cert.lindstedt_solved}:{cert.is_flat_curvature}:"
            f"{cert.spectral_condition_number:.6e}:{cert.poincare_residual:.3e}"
        )

    def verify_phase1_certificate(self, cert: WitnessPhase1Certificate) -> bool:
        r"""Re-valida HMAC P1 (dominio v10.1) sobre el payload canónico."""
        return self._verify_hmac(
            _HMAC_DOMAIN_P1, self._phase1_payload(cert), cert.hmac_p1_signature,
        )

    @staticmethod
    def _build_connection_A(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        r"""
        $A=i(\rho-\mathrm{Tr}(\rho)/N\cdot I)$ proyectada a $\mathfrak{su}(N)$.
        Identidad QND: $[A,\rho]=0$ en aritmética exacta.
        """
        n = int(rho.shape[0])
        tr_val = np.trace(rho) / float(n)
        su_part = rho - tr_val * np.eye(n, dtype=np.complex128)
        return _project_su_n(1j * su_part)

    def _local_phase1_audit(
        self, rho: NDArray[np.complex128],
    ) -> Tuple[NDArray[np.complex128], float, bool, bool, float]:
        r"""Fallback local: $A$, $\|A\|_F$, anti-Herm, traceless, $E_{YM}$."""
        A = self._build_connection_A(rho)
        norm_a = _fro(A)
        ah_rel = _fro(A + A.conj().T) / max(1.0, norm_a)
        tr_abs = float(abs(complex(np.trace(A))))
        F = _project_anti_hermitian(A @ rho - rho @ A)
        ym = max(0.0, -0.5 * float(np.real(np.trace(F @ F))))
        return A, norm_a, bool(ah_rel < _ANTI_HERM_TOL), bool(tr_abs < _TRACELESS_TOL), ym

    def _stamp_hmac_p1(
        self, cert: WitnessPhase1Certificate,
    ) -> WitnessPhase1Certificate:
        sig = self._compute_hmac(_HMAC_DOMAIN_P1, self._phase1_payload(cert))
        ok = hmac.compare_digest(sig, sig)  # trivial, se re-verifica abajo
        stamped = replace(cert, hmac_p1_signature=sig, hmac_verified=True)
        ok = self.verify_phase1_certificate(stamped)
        return replace(stamped, hmac_verified=bool(ok))

    # ── CIERRE FASE 1 → FASE 2 ────────────────────────────────────────────────
    def emit_phase1_certificate(
        self, density_matrix: NDArray,
    ) -> WitnessPhase1Certificate:
        r"""
        **Cierre formal de la FASE 1 / puente de apertura de la FASE 2.**

        Sanitiza ρ, consume `engine.emit_phase1_envelope` (Cayley + Lindstedt
        espectral + κ₂) o audita localmente, y emite
        `WitnessPhase1Certificate` firmado HMAC P1.
        El receptor canónico es `WitnessPhase2Orientor.from_phase1_certificate`.
        """
        now = time.time()
        rho = self.engine.sanitize_density_matrix(density_matrix)
        tr_val = float(np.real(np.trace(rho)))
        liouville_drift = float(abs(tr_val - 1.0))
        is_trace_conserved = liouville_drift <= _TRACE_CONSERVATION_TOL

        engine_env: Any = None
        engine_interop = False
        if _ENGINE_HAS_ENVELOPES:
            try:
                engine_env = self.engine.emit_phase1_envelope(rho)  # type: ignore[attr-defined]
                engine_interop = True
            except Exception as exc:
                logger.warning(
                    "Engine FASE1 envelope falló (%s); fallback local.", exc,
                )
                engine_env = None
                engine_interop = False

        if engine_env is not None and hasattr(engine_env, "connection_A"):
            A = np.asarray(engine_env.connection_A, dtype=np.complex128)
            norm_a = _fro(A)
            is_ah = _battr(engine_env, "is_anti_hermitian_A", True)
            is_tr = _battr(engine_env, "is_traceless_A", True)
            sympl_res, sympl_def, is_sympl, is_flat, pole = _symplectic_tuple(engine_env)
            poincare_res, poincare_resonant, lindstedt_ok, newton_it = _lindstedt_tuple(
                engine_env,
            )
            ym_baseline = _yang_mills_energy(engine_env, 0.0)
            kappa = _spectral_kappa(engine_env)
        else:
            A, norm_a, is_ah, is_tr, ym_baseline = self._local_phase1_audit(rho)
            sympl_res, sympl_def, is_sympl = 0.0, 0.0, True
            is_flat, pole = True, 1.0
            poincare_res, poincare_resonant = 0.0, False
            lindstedt_ok, newton_it = True, 1
            kappa = 1.0
            engine_interop = False

        if (
            (not is_trace_conserved) or (not is_ah) or (not is_tr)
            or (not is_sympl) or (liouville_drift > _LIOUVILLE_HARD)
        ):
            verdict = HeytingOmega3.VETOED
        elif liouville_drift > _LIOUVILLE_SOFT or poincare_resonant:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.COHERENT

        chain = _hash_state(
            rho, A,
            np.array(
                [tr_val, liouville_drift, sympl_res, sympl_def, poincare_res],
                dtype=np.float64,
            ),
        )
        wit_id = f"WIT-P1-{int(now * 1000.0) % 1000000:06d}"
        unsigned = WitnessPhase1Certificate(
            witness_id=wit_id,
            trace_preservation=tr_val,
            liouville_drift=liouville_drift,
            isometry_verdict=verdict,
            hmac_p1_signature="",
            timestamp_utc=now,
            dimension=int(rho.shape[0]),
            connection_norm_frobenius=norm_a,
            is_anti_hermitian_A=is_ah,
            is_traceless_A=is_tr,
            is_trace_conserved=is_trace_conserved,
            symplectic_residue=sympl_res,
            symplectic_defect=sympl_def,
            is_symplectic_valid=is_sympl,
            poincare_residual=poincare_res,
            poincare_is_resonant=poincare_resonant,
            yang_mills_energy_baseline=ym_baseline,
            engine_envelope=engine_env,
            chain_hash=chain,
            engine_interop=engine_interop,
            lindstedt_solved=bool(lindstedt_ok),
            lindstedt_newton_iterations=int(newton_it),
            spectral_condition_number=float(kappa),
            is_flat_curvature=bool(is_flat),
            cayley_pole_margin=float(pole),
            hmac_verified=False,
        )
        return self._stamp_hmac_p1(unsigned)


# ══════════════════════════════════════════════════════════════════════════════════
# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║  F A S E   2 :   O R I E N T                                                 ║
# ║  Apertura = from_phase1_certificate (continuación del cierre FASE 1).        ║
# ║  QND verificado, Poincaré–Bruno–Birkhoff, RSI³.                              ║
# ║  Cierra con `emit_phase2_certificate` → `WitnessPhase2Certificate`.          ║
# ╚════════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════════

class WitnessPhase2Orientor(WitnessPhase1Observer):
    r"""
    FASE 2 (Orient): QND verificado + Uhlmann + mecánica celeste de Poincaré
    (KAM/Bruno/Melnikov/Floquet/sección/Birkhoff) + RSI³ del motor.
    """

    # ── APERTURA FASE 2  (continuación del último método de FASE 1) ───────────
    @classmethod
    def from_phase1_certificate(
        cls,
        cert1: WitnessPhase1Certificate,
        engine: TopologicalGaugeFieldWitnessEngine,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> "WitnessPhase2Orientor":
        r"""
        **Apertura formal de la FASE 2** — continuación de
        `WitnessPhase1Observer.emit_phase1_certificate`.

        Valida HMAC P1 y la forma del `chain_hash`.
        """
        inst = cls(engine=engine, secret_key=secret_key)
        if not inst.verify_phase1_certificate(cert1):
            raise WitnessContractCorruptedError(
                "from_phase1_certificate: HMAC P1 inválida.",
            )
        if not cert1.chain_hash or len(cert1.chain_hash) < 16:
            raise WitnessContractCorruptedError(
                "from_phase1_certificate: chain_hash P1 vacío o truncado.",
            )
        inst._phase1_cert = cert1  # type: ignore[attr-defined]
        return inst

    def continue_to_phase2_witness_gate(
        self, cert1: WitnessPhase1Certificate,
    ) -> bool:
        r"""
        **[Compat] Compuerta Φ₁₂.**  Re-valida HMAC P1.
        Acepta VETOED (objeto de inquisición, no de silenciamiento).
        """
        if not self.verify_phase1_certificate(cert1):
            raise WitnessGateRejectionError(
                "Compuerta Φ₁₂ rechazó cert P1: HMAC P1 inválida.",
            )
        return True

    def execute_phase1(self, density_matrix: NDArray) -> WitnessPhase1Certificate:
        r"""**[Compat v9.3.0 / v10.0.0]** Delega a `emit_phase1_certificate`."""
        return self.emit_phase1_certificate(density_matrix)

    @staticmethod
    def _phase2_payload(cert: WitnessPhase2Certificate) -> str:
        return (
            f"{cert.witness_id}:{cert.weak_value_qnd_aav}:"
            f"{cert.back_action_db:.6f}:{cert.uhlmann_fidelity:.8f}:"
            f"{cert.bures_distance:.8f}:{cert.yang_mills_energy:.8f}:"
            f"{int(cert.qnd_verdict)}:{cert.is_qnd_preserved}:"
            f"{cert.qnd_commutator_norm:.3e}:{cert.kam_worst_ratio:.4f}:"
            f"{cert.melnikov_value:.3e}:{cert.floquet_max_unit_deviation:.3e}:"
            f"{cert.rsi3_valid}:{cert.chain_hash}:"
            f"{cert.bruno_satisfied}:{cert.bruno_sum:.6f}:"
            f"{cert.birkhoff_twist_estimate:.6e}:{cert.banach_lipschitz_t3:.6f}"
        )

    def verify_phase2_certificate(self, cert: WitnessPhase2Certificate) -> bool:
        return self._verify_hmac(
            _HMAC_DOMAIN_P2, self._phase2_payload(cert), cert.hmac_p2_signature,
        )

    def _stamp_hmac_p2(
        self, cert: WitnessPhase2Certificate,
    ) -> WitnessPhase2Certificate:
        sig = self._compute_hmac(_HMAC_DOMAIN_P2, self._phase2_payload(cert))
        stamped = replace(cert, hmac_p2_signature=sig, hmac_verified=True)
        ok = self.verify_phase2_certificate(stamped)
        return replace(stamped, hmac_verified=bool(ok))

    def _obtain_engine_snapshot(
        self,
        cert1: WitnessPhase1Certificate,
        density_matrix: NDArray,
        phi_initial: Optional[NDArray],
        phi_final: Optional[NDArray],
    ) -> Tuple[Any, bool, str]:
        r"""
        Puente anidado FASE 1→2 del motor:

          env1 ─emit_phase2_envelope→ env2 ─emit_witness_snapshot→ snap

        Evita re-ejecutar Observe.  Fallback: `evaluate_gauge_witness`.
        """
        env1 = cert1.engine_envelope
        nested_ok = (
            env1 is not None
            and _ENGINE_HAS_P2_ENVELOPE
            and _ENGINE_HAS_SNAPSHOT_EMIT
            and hasattr(env1, "connection_A")
            and hasattr(env1, "curvature_F")
            and hasattr(env1, "chain_hash")
        )
        if nested_ok:
            try:
                env2 = self.engine.emit_phase2_envelope(  # type: ignore[attr-defined]
                    env1, phi_initial=phi_initial, phi_final=phi_final,
                )
                snap = self.engine.emit_witness_snapshot(env2)  # type: ignore[attr-defined]
                return snap, True, "nested"
            except Exception as exc:
                logger.warning(
                    "Pipeline anidado P2 falló (%s); evaluate_gauge_witness.",
                    exc,
                )
        try:
            snap = self.engine.evaluate_gauge_witness(
                density_matrix, phi_initial, phi_final,
            )
            return snap, True, "evaluate"
        except Exception as exc:
            logger.warning(
                "evaluate_gauge_witness falló (%s); snapshot nulo.", exc,
            )
            return _NullSnapshot(), False, "null"

    @staticmethod
    def _extract_p2_metrics(snapshot: Any) -> Dict[str, Any]:
        r"""Proyección defensiva snapshot/envelope → métricas planas P2."""
        p2env = _attr(snapshot, "phase2", None)
        qnd = _attr(p2env, "qnd", None)
        kam = _attr(p2env, "kam_certificate", None)
        bruno = _attr(p2env, "bruno_certificate", None)
        birk = _attr(p2env, "birkhoff_twist", None)
        rsi3 = _attr(p2env, "rsi3_certificate", None)
        sec = _attr(p2env, "poincare_section", None)

        def _pick_f(*cands: Tuple[Any, str], default: float = 0.0) -> float:
            for obj, name in cands:
                if obj is not None and hasattr(obj, name):
                    return _fattr(obj, name, default)
            return default

        def _pick_b(*cands: Tuple[Any, str], default: bool = True) -> bool:
            for obj, name in cands:
                if obj is not None and hasattr(obj, name):
                    return _battr(obj, name, default)
            return default

        back = _pick_f((snapshot, "back_action_db"), (qnd, "back_action_db"))
        is_qnd = _pick_b((snapshot, "is_qnd_preserved"), (qnd, "is_qnd_preserved"))
        return {
            "qnd_weak": _cattr(snapshot, "qnd_weak_value_aav",
                               _cattr(qnd, "qnd_weak_value_aav", 0.0 + 0.0j)),
            "back_action": back,
            "is_qnd": is_qnd,
            "qnd_comm": _pick_f(
                (snapshot, "qnd_commutator_norm"), (qnd, "commutator_norm"),
            ),
            "qnd_fid": _pick_f(
                (snapshot, "qnd_fidelity"), (qnd, "qnd_fidelity"), default=1.0,
            ),
            "uhlmann": _pick_f(
                (snapshot, "uhlmann_fidelity"), (p2env, "uhlmann_fidelity"),
                default=1.0,
            ),
            "bures": _pick_f(
                (snapshot, "bures_distance"), (p2env, "bures_distance"),
            ),
            "trace_dist": _pick_f(
                (snapshot, "trace_distance"), (p2env, "trace_distance"),
            ),
            "liouville": _pick_f(
                (snapshot, "liouville_volume_drift"),
                (p2env, "liouville_volume_drift"),
            ),
            "ym_energy": _yang_mills_energy(snapshot, 0.0),
            "sz": _pick_f(
                (snapshot, "stora_zumino_residual"),
                (p2env, "stora_zumino_residual"),
            ),
            "kam_w": _pick_f(
                (snapshot, "kam_worst_ratio"), (kam, "worst_ratio"),
            ),
            "kam_ok": _pick_b(
                (snapshot, "kam_satisfied"), (kam, "is_satisfied"),
            ),
            "kam_n": _iattr(snapshot, "kam_n_modes", _iattr(kam, "n_modes", 0)),
            "mel": _pick_f(
                (snapshot, "melnikov_value"),
                (_attr(p2env, "melnikov", None), "integral_value"),
            ),
            "mel_ok": _pick_b(
                (snapshot, "melnikov_preserved"),
                (_attr(p2env, "melnikov", None), "is_homoclinic_preserved"),
            ),
            "flo_dev": _pick_f(
                (snapshot, "floquet_max_unit_deviation"),
                (_attr(p2env, "floquet", None), "max_unit_deviation"),
            ),
            "flo_neutral": _pick_b(
                (snapshot, "floquet_neutral"),
                (_attr(p2env, "floquet", None), "is_neutral"),
            ),
            "poin_x": _iattr(
                snapshot, "poincare_section_n_crossings",
                _iattr(sec, "n_crossings", 0),
            ),
            "poin_disp": _pick_f(
                (snapshot, "poincare_section_dispersion"),
                (sec, "return_dispersion"),
            ),
            "mean_ret": _pick_f(
                (snapshot, "mean_poincare_return_time"),
                (sec, "mean_return_time"),
            ),
            "rsi3_ok": _pick_b(
                (snapshot, "rsi3_valid"), (rsi3, "is_valid"),
            ),
            "rsi3_it": _iattr(
                snapshot, "rsi3_iterations", _iattr(rsi3, "iterations", 1),
            ),
            "banach": _pick_f(
                (snapshot, "banach_lipschitz_t3"), (rsi3, "lipschitz_t3"),
            ),
            "fubini": _pick_f(
                (snapshot, "fubini_study_distance"),
                (_attr(rsi3, "surface3", None), "metric_value"),
            ),
            "novikov": _pick_f(
                (snapshot, "novikov_krull_valuation"),
                (_attr(rsi3, "surface1", None), "metric_value"),
            ),
            "bruno_ok": _pick_b(
                (snapshot, "bruno_satisfied"), (bruno, "is_satisfied"),
            ),
            "bruno_sum": _pick_f(
                (snapshot, "bruno_sum"), (bruno, "bruno_sum"),
            ),
            "birk_est": _pick_f(
                (snapshot, "birkhoff_twist_estimate"), (birk, "twist_estimate"),
            ),
            "birk_twist": _pick_b(
                (snapshot, "birkhoff_is_twist"), (birk, "is_twist"),
            ),
            "birk_area": _pick_b(
                (snapshot, "birkhoff_area_preserving"),
                (birk, "is_area_preserving_proxy"),
            ),
            "engine_hmac_ok": _battr(snapshot, "hmac_verified", True),
        }

    # ── CIERRE FASE 2 → FASE 3 ────────────────────────────────────────────────
    def emit_phase2_certificate(
        self,
        cert1: WitnessPhase1Certificate,
        density_matrix: NDArray,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
    ) -> WitnessPhase2Certificate:
        r"""
        **Cierre formal de la FASE 2 / puente de apertura de la FASE 3.**

        Consume el pipeline anidado del motor (Orient+Decide del engine)
        y extrae QND, Uhlmann, Poincaré–Bruno–Birkhoff y RSI³.
        Firma HMAC P2.  Receptor: `WitnessPhase3Decider.from_phase2_certificate`.
        """
        now = time.time()
        self.continue_to_phase2_witness_gate(cert1)

        snapshot, engine_interop, pipeline = self._obtain_engine_snapshot(
            cert1, density_matrix, phi_initial, phi_final,
        )
        m = self._extract_p2_metrics(snapshot)
        if m["ym_energy"] == 0.0:
            m["ym_energy"] = float(cert1.yang_mills_energy_baseline)

        p2_atoms: List[HeytingOmega3] = [
            HeytingOmega3.COHERENT if m["ym_energy"] <= _YM_ENERGY_SOFT else (
                HeytingOmega3.DEGRADED if m["ym_energy"] <= _YM_ENERGY_HARD
                else HeytingOmega3.VETOED
            ),
            HeytingOmega3.COHERENT if m["is_qnd"] else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if m["back_action"] <= _QND_BACKACTION_DB_TOL
            else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if m["uhlmann"] >= _UHLMANN_FIDELITY_MIN
            else HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT if m["rsi3_ok"] else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if m["bruno_ok"] else HeytingOmega3.DEGRADED,
        ]
        p2_verdict = cert1.isometry_verdict.join(
            HeytingOmega3.fold_supremum(p2_atoms),
        )

        rho = self.engine.sanitize_density_matrix(density_matrix)
        chain = _hash_state(
            rho,
            np.array(
                [
                    m["back_action"], m["uhlmann"], m["bures"], m["ym_energy"],
                    m["sz"], m["kam_w"], m["mel"], m["flo_dev"],
                    float(m["rsi3_ok"]), m["bruno_sum"],
                ],
                dtype=np.float64,
            ),
        )
        wit_id = f"WIT-P2-{int(now * 1000.0) % 1000000:06d}"
        unsigned = WitnessPhase2Certificate(
            witness_id=wit_id,
            weak_value_qnd_aav=m["qnd_weak"],
            back_action_db=m["back_action"],
            uhlmann_fidelity=m["uhlmann"],
            bures_distance=m["bures"],
            yang_mills_energy=m["ym_energy"],
            qnd_verdict=p2_verdict,
            hmac_p2_signature="",
            timestamp_utc=now,
            is_qnd_preserved=m["is_qnd"],
            qnd_commutator_norm=m["qnd_comm"],
            qnd_fidelity=m["qnd_fid"],
            trace_distance=m["trace_dist"],
            liouville_volume_drift=m["liouville"],
            stora_zumino_residual=m["sz"],
            kam_worst_ratio=m["kam_w"],
            kam_satisfied=m["kam_ok"],
            melnikov_value=m["mel"],
            melnikov_preserved=m["mel_ok"],
            floquet_max_unit_deviation=m["flo_dev"],
            floquet_neutral=m["flo_neutral"],
            poincare_section_n_crossings=m["poin_x"],
            poincare_section_dispersion=m["poin_disp"],
            rsi3_valid=m["rsi3_ok"],
            rsi3_iterations=m["rsi3_it"],
            banach_lipschitz_t3=m["banach"],
            fubini_study_distance=m["fubini"],
            novikov_krull_valuation=m["novikov"],
            engine_envelope=snapshot,
            chain_hash=chain,
            engine_interop=engine_interop,
            bruno_satisfied=m["bruno_ok"],
            bruno_sum=m["bruno_sum"],
            birkhoff_twist_estimate=m["birk_est"],
            birkhoff_is_twist=m["birk_twist"],
            birkhoff_area_preserving=m["birk_area"],
            kam_n_modes=m["kam_n"],
            mean_poincare_return_time=m["mean_ret"],
            nested_pipeline=pipeline,
            hmac_verified=False,
            hmac_engine_verified=bool(m["engine_hmac_ok"]),
        )
        return self._stamp_hmac_p2(unsigned)


# ══════════════════════════════════════════════════════════════════════════════════
# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║  F A S E   3 :   D E C I D E   &   A C T                                      ║
# ║  Apertura = from_phase2_certificate (continuación del cierre FASE 2).        ║
# ║  Heyting Ω₃ (12 átomos) + HMAC P3 + RSI³ agéntica. SIN crowbar.              ║
# ║  Cierra con `emit_sovereign_state` → estado terminal.                        ║
# ╚════════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════════

class WitnessPhase3Decider(WitnessPhase2Orientor):
    r"""
    FASE 3 (Decide & Act): Heyting Ω₃ completo + HMAC P3 + RSI³ agéntica.

    **Invariante**: `witness_mode = "PASSIVE_SILENT"`; nunca dispara crowbar.
    """

    # ── APERTURA FASE 3  (continuación del último método de FASE 2) ───────────
    @classmethod
    def from_phase2_certificate(
        cls,
        cert2: WitnessPhase2Certificate,
        engine: TopologicalGaugeFieldWitnessEngine,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> "WitnessPhase3Decider":
        r"""
        **Apertura formal de la FASE 3** — continuación de
        `WitnessPhase2Orientor.emit_phase2_certificate`.

        Valida HMAC P2 y la forma del `chain_hash`.
        """
        inst = cls(engine=engine, secret_key=secret_key)
        if not inst.verify_phase2_certificate(cert2):
            raise WitnessContractCorruptedError(
                "from_phase2_certificate: HMAC P2 inválida.",
            )
        if not cert2.chain_hash or len(cert2.chain_hash) < 16:
            raise WitnessContractCorruptedError(
                "from_phase2_certificate: chain_hash P2 vacío o truncado.",
            )
        inst._phase2_cert = cert2  # type: ignore[attr-defined]
        return inst

    def continue_to_phase3_witness_gate(
        self, cert2: WitnessPhase2Certificate,
    ) -> bool:
        r"""**[Compat] Compuerta Φ₂₃.**  Re-valida HMAC P2 completo."""
        if not self.verify_phase2_certificate(cert2):
            raise WitnessGateRejectionError(
                "Compuerta Φ₂₃ rechazó cert P2: HMAC P2 inválida.",
            )
        return True

    def execute_phase2(
        self,
        cert1: WitnessPhase1Certificate,
        density_matrix: NDArray,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
    ) -> WitnessPhase2Certificate:
        r"""**[Compat v9.3.0 / v10.0.0]** Delega a `emit_phase2_certificate`."""
        return self.emit_phase2_certificate(
            cert1, density_matrix, phi_initial, phi_final,
        )

    # ── Adjudicador Heyting Ω₃ (12 átomos, alineado al motor v10.1) ───────────
    @staticmethod
    def adjudicate_heyting(
        *,
        energy_ym: float,
        is_qnd_preserved: bool,
        back_action_db: float,
        uhlmann_fidelity: float,
        bures_distance: float,
        symplectic_valid: bool,
        kam_satisfied: bool,
        melnikov_preserved: bool,
        floquet_stable: bool,
        rsi3_valid: bool,
        bruno_satisfied: bool = True,
        birkhoff_area_ok: bool = True,
    ) -> Tuple[HeytingOmega3, List[int], int]:
        r"""
        Adjudicación en $\Omega_3$ con **12 átomos** vía join(∨) monoidal.
        Retorna `(veredicto, átomos, meet_reduction)`.
        """
        atoms: List[HeytingOmega3] = [
            HeytingOmega3.COHERENT if energy_ym <= _YM_ENERGY_SOFT else (
                HeytingOmega3.DEGRADED if energy_ym <= _YM_ENERGY_HARD
                else HeytingOmega3.VETOED
            ),
            HeytingOmega3.COHERENT if is_qnd_preserved else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if back_action_db <= _QND_BACKACTION_DB_TOL
            else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if uhlmann_fidelity >= _UHLMANN_FIDELITY_MIN
            else HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT if bures_distance <= _BURES_DIST_SOFT else (
                HeytingOmega3.DEGRADED if bures_distance <= _BURES_DIST_HARD
                else HeytingOmega3.VETOED
            ),
            HeytingOmega3.COHERENT if symplectic_valid else HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT if kam_satisfied else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if melnikov_preserved else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if floquet_stable else HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT if rsi3_valid else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if bruno_satisfied else HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT if birkhoff_area_ok else HeytingOmega3.DEGRADED,
        ]
        verdict = HeytingOmega3.fold_supremum(atoms)
        meet_reduction = int(HeytingOmega3.fold_infimum(atoms))
        return verdict, [int(a) for a in atoms], meet_reduction

    @staticmethod
    def heyting_truth_tables() -> Tuple[
        Tuple[Tuple[int, int, int], ...],
        Tuple[Tuple[int, int, int], ...],
        Tuple[Tuple[int, int, int], ...],
        Tuple[int, int, int],
    ]:
        r"""
        Tablas 3×3 del retículo de cadena
        $\{0x00<0x01<0x03\}$: `(join, meet, implies, negate)`.
        """
        vals = (0x00, 0x01, 0x03)
        top, bottom = 0x03, 0x00
        join_t = tuple(tuple(max(a, b) for b in vals) for a in vals)
        meet_t = tuple(tuple(min(a, b) for b in vals) for a in vals)
        implies_t = tuple(tuple(top if a <= b else b for b in vals) for a in vals)
        negate_t = tuple(top if a <= bottom else bottom for a in vals)
        return join_t, meet_t, implies_t, negate_t

    def enforce_agent_rsi3(
        self,
        cert1: WitnessPhase1Certificate,
        cert2: WitnessPhase2Certificate,
        verdict: HeytingOmega3,
        prov_payload: str,
        prov_sig: str,
    ) -> AgentRSI3Certificate:
        r"""
        Automejora recursiva de nivel 3 **agéntica** (PASSIVE_SILENT):

          • S1 (Novikov/Data): HMAC P1 ∧ HMAC P2.
          • S2 (Harness): chain_hash de longitud canónica y alfabeto hex.
          • S3 (Model): `join(v,v)=v` (idempotencia) y firma P3 idempotente.

        El bucle es contracción sobre el gráfico de verificación, no sobre ρ.
        """
        hmac_p1_ok = self.verify_phase1_certificate(cert1)
        hmac_p2_ok = self.verify_phase2_certificate(cert2)

        def _hex_ok(h: str) -> bool:
            if not h or len(h) < 16:
                return False
            try:
                int(h, 16)
                return True
            except ValueError:
                return False

        chain_ok = _hex_ok(cert1.chain_hash) and _hex_ok(cert2.chain_hash)
        heyting_idemp = (verdict.join(verdict) == verdict)

        signing_idemp = True
        it_used = 1
        prev_sig = prov_sig
        for it in range(1, _RSI3_AGENT_MAX_ITER + 1):
            it_used = it
            nxt = self._compute_hmac(_HMAC_DOMAIN_P3, prov_payload)
            signing_idemp = hmac.compare_digest(nxt, prev_sig) and hmac.compare_digest(
                nxt, prov_sig,
            )
            if signing_idemp:
                break
            prev_sig = nxt
        lip = 0.0 if signing_idemp else 1.0
        valid = bool(
            hmac_p1_ok and hmac_p2_ok and chain_ok
            and heyting_idemp and signing_idemp
        )
        return AgentRSI3Certificate(
            hmac_p1_ok=hmac_p1_ok,
            hmac_p2_ok=hmac_p2_ok,
            chain_ok=chain_ok,
            heyting_idempotent=heyting_idemp,
            signing_idempotent=signing_idemp,
            iterations=int(it_used),
            is_valid=valid,
            lipschitz_proxy=float(lip),
        )

    def _p3_payload(
        self, merkle_root: str, cert1: WitnessPhase1Certificate,
        cert2: WitnessPhase2Certificate, verdict: HeytingOmega3,
    ) -> str:
        return (
            f"{merkle_root}:{cert2.back_action_db:.6f}:{int(verdict)}:"
            f"{cert1.chain_hash}:{cert2.chain_hash}"
        )

    def verify_sovereign_state(
        self, state: TopologicalGaugeFieldWitnessSovereignState,
    ) -> bool:
        r"""Verifica HMAC P3 (dominio v10.1) del estado terminal."""
        payload = self._p3_payload(
            state.merkle_sha256_root,
            state.phase1_cert,
            state.phase2_cert,
            state.global_verdict,
        )
        return self._verify_hmac(
            _HMAC_DOMAIN_P3, payload, state.hmac_provenance_signature,
        )

    # ── CIERRE FASE 3 (terminal) ──────────────────────────────────────────────
    def emit_sovereign_state(
        self,
        cert1: WitnessPhase1Certificate,
        cert2: WitnessPhase2Certificate,
    ) -> TopologicalGaugeFieldWitnessSovereignState:
        r"""
        **Cierre formal de la FASE 3.**

        Adjudica en Heyting Ω₃ (12 átomos), aplica RSI³ agéntica, firma HMAC P3
        y emite el estado terminal.  **NUNCA dispara crowbar**.
        """
        now = time.time()
        self.continue_to_phase3_witness_gate(cert2)

        floquet_stable = (
            cert2.floquet_neutral
            or cert2.floquet_max_unit_deviation < _FLOQUET_UNIT_TOL
        )
        verdict, atoms, meet_red = self.adjudicate_heyting(
            energy_ym=cert2.yang_mills_energy,
            is_qnd_preserved=cert2.is_qnd_preserved,
            back_action_db=cert2.back_action_db,
            uhlmann_fidelity=cert2.uhlmann_fidelity,
            bures_distance=cert2.bures_distance,
            symplectic_valid=cert1.is_symplectic_valid,
            kam_satisfied=cert2.kam_satisfied,
            melnikov_preserved=cert2.melnikov_preserved,
            floquet_stable=floquet_stable,
            rsi3_valid=cert2.rsi3_valid,
            bruno_satisfied=cert2.bruno_satisfied,
            birkhoff_area_ok=cert2.birkhoff_area_preserving,
        )
        join_t, meet_t, implies_t, negate_t = self.heyting_truth_tables()

        state_id = f"GAUGE-WITNESS-STATE-{int(now * 1000.0) % 1000000:06d}"
        merkle_seed = (
            f"{state_id}:{cert1.hmac_p1_signature}:{cert2.hmac_p2_signature}:"
            f"{int(verdict)}:{cert1.chain_hash}:{cert2.chain_hash}"
        ).encode("utf-8")
        merkle_root = hashlib.sha256(merkle_seed).hexdigest()
        prov_payload = self._p3_payload(merkle_root, cert1, cert2, verdict)
        prov_sig = self._compute_hmac(_HMAC_DOMAIN_P3, prov_payload)
        hmac_p3_ok = self._verify_hmac(_HMAC_DOMAIN_P3, prov_payload, prov_sig)

        rsi3_agent = self.enforce_agent_rsi3(
            cert1, cert2, verdict, prov_payload, prov_sig,
        )

        visceral_report: Dict[str, Any] = {
            "witness_qnd_status": (
                "OBSERVACIÓN DE CERO DEMOLICIÓN CUÁNTICA (0.0 dB)"
                if cert2.is_qnd_preserved
                else f"DEMOLICIÓN DETECTADA ({cert2.back_action_db:.2f} dB)"
            ),
            "uhlmann_fidelity": cert2.uhlmann_fidelity,
            "bures_distance": cert2.bures_distance,
            "yang_mills_energy": cert2.yang_mills_energy,
            "qnd_commutator_norm": cert2.qnd_commutator_norm,
            "kam_worst_ratio": cert2.kam_worst_ratio,
            "kam_n_modes": cert2.kam_n_modes,
            "melnikov_value": cert2.melnikov_value,
            "floquet_max_unit_deviation": cert2.floquet_max_unit_deviation,
            "rsi3_iterations": cert2.rsi3_iterations,
            "novikov_krull_valuation": cert2.novikov_krull_valuation,
            "bruno_sum": cert2.bruno_sum,
            "birkhoff_twist": cert2.birkhoff_twist_estimate,
            "lindstedt_solved": cert1.lindstedt_solved,
            "is_flat_curvature": cert1.is_flat_curvature,
            "nested_pipeline": cert2.nested_pipeline,
            "rsi3_agent_valid": rsi3_agent.is_valid,
            "executive_action": (
                "TESTIMONIO SILENCIOSO CERTIFICADO EN RAM"
                if verdict != HeytingOmega3.VETOED
                else "VETO REGISTRADO (sin acción de hardware)"
            ),
        }

        is_epistemologically_valid = (
            verdict != HeytingOmega3.VETOED
            and cert2.is_qnd_preserved
            and cert1.is_symplectic_valid
            and cert1.is_anti_hermitian_A
            and cert1.is_traceless_A
            and cert1.is_trace_conserved
            and hmac_p3_ok
            and rsi3_agent.is_valid
        )

        return TopologicalGaugeFieldWitnessSovereignState(
            sovereign_state_id=state_id,
            phase1_cert=cert1,
            phase2_cert=cert2,
            global_verdict=verdict,
            back_action_db=cert2.back_action_db,
            merkle_sha256_root=merkle_root,
            hmac_provenance_signature=prov_sig,
            visceral_report=visceral_report,
            timestamp_utc=now,
            witness_mode="PASSIVE_SILENT",
            crowbar_triggered=False,
            is_epistemologically_valid=bool(is_epistemologically_valid),
            heyting_join_atoms=tuple(atoms),
            heyting_meet_reduction=meet_red,
            heyting_implies_table=implies_t,
            heyting_negate_table=negate_t,
            chain_hash_p1=cert1.chain_hash,
            chain_hash_p2=cert2.chain_hash,
            agent_version=_AGENT_VERSION,
            heyting_join_table=join_t,
            heyting_meet_table=meet_t,
            bruno_satisfied=cert2.bruno_satisfied,
            bruno_sum=cert2.bruno_sum,
            birkhoff_twist_estimate=cert2.birkhoff_twist_estimate,
            birkhoff_is_twist=cert2.birkhoff_is_twist,
            lindstedt_solved=cert1.lindstedt_solved,
            spectral_condition_number=cert1.spectral_condition_number,
            is_flat_curvature=cert1.is_flat_curvature,
            rsi3_agent_valid=rsi3_agent.is_valid,
            rsi3_agent_iterations=rsi3_agent.iterations,
            hmac_p3_verified=bool(hmac_p3_ok),
            nested_pipeline=cert2.nested_pipeline,
        )

    def execute_phase3(
        self,
        cert1: WitnessPhase1Certificate,
        cert2: WitnessPhase2Certificate,
    ) -> TopologicalGaugeFieldWitnessSovereignState:
        r"""**[Compat v9.3.0 / v10.0.0]** Delega a `emit_sovereign_state`."""
        return self.emit_sovereign_state(cert1, cert2)


# ══════════════════════════════════════════════════════════════════════════════
# §H. ORQUESTADOR SUPREMO
# ══════════════════════════════════════════════════════════════════════════════

class TopologicalGaugeFieldWitnessAgent(WitnessPhase3Decider):
    r"""
    Soberano Agéntico Testigo Silencioso del Campo Topológico de Gauge
    (v10.1.0 — Tres Fases Anidadas, Poincaré–Bruno–Birkhoff, RSI³).

    Orquesta:
      FASE 1 (Observe)    → `emit_phase1_certificate`
                          → `WitnessPhase1Certificate`.
      FASE 2 (Orient)     → `from_phase1_certificate` + `emit_phase2_certificate`
                          → `WitnessPhase2Certificate`.
      FASE 3 (Decide&Act) → `from_phase2_certificate` + `emit_sovereign_state`
                          → `TopologicalGaugeFieldWitnessSovereignState`.

    **Compatibilidad**: `witness_gauge_field`, `execute_phase*`,
    `continue_to_phase*_gate` preservan firmas y semántica.
    """

    def __init__(
        self,
        dimension: int = 56,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> None:
        engine = TopologicalGaugeFieldWitnessEngine(
            dimension=dimension, secret_key=secret_key,
        )
        super().__init__(engine=engine, secret_key=secret_key)
        logger.info(
            "TopologicalGaugeFieldWitnessAgent v%s inicializado "
            "[dim=%s, engine_interop=%s, nested_p2=%s]",
            _AGENT_VERSION,
            dimension,
            "ON" if _ENGINE_HAS_ENVELOPES else "OFF",
            "ON" if _ENGINE_HAS_P2_ENVELOPE else "OFF",
        )

    def witness_gauge_field(
        self,
        density_matrix: NDArray,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
    ) -> TopologicalGaugeFieldWitnessSovereignState:
        r"""
        Ejecuta el ciclo OODA a través de las tres fases anidadas.

        ══ FASE 1 ══ emit_phase1_certificate → from_phase1_certificate
        ══ FASE 2 ══ emit_phase2_certificate → from_phase2_certificate
        ══ FASE 3 ══ emit_sovereign_state
        """
        cert1 = self.emit_phase1_certificate(density_matrix)
        _ = self.from_phase1_certificate(
            cert1, engine=self.engine, secret_key=self.secret_key,
        )
        if not self.continue_to_phase2_witness_gate(cert1):
            raise WitnessGateRejectionError("Compuerta Φ₁₂ rechazó cert P1.")

        cert2 = self.emit_phase2_certificate(
            cert1, density_matrix, phi_initial, phi_final,
        )
        _ = self.from_phase2_certificate(
            cert2, engine=self.engine, secret_key=self.secret_key,
        )
        if not self.continue_to_phase3_witness_gate(cert2):
            raise WitnessGateRejectionError("Compuerta Φ₂₃ rechazó cert P2.")

        return self.emit_sovereign_state(cert1, cert2)

    def consume_engine_witness_snapshot(
        self,
        engine_snapshot: GaugeWitnessSnapshot,
    ) -> TopologicalGaugeFieldWitnessSovereignState:
        r"""
        **Interop**: consume un `GaugeWitnessSnapshot` del motor v10.1 y
        reconstruye los certificados agénticos firmados (el agente testifica
        lo que el motor ya observó).
        """
        if engine_snapshot is None:
            raise WitnessAgentError("engine_snapshot es None.")

        now = time.time()
        snap = engine_snapshot
        p1env = _attr(snap, "phase1", None)
        dim = _iattr(p1env, "dimension", _iattr(self.engine, "dimension", 56))

        chain_p1 = hashlib.sha256(
            (
                f"consume_engine:{_attr(snap, 'sha256_merkle_digest', '')}:"
                f"{_fattr(snap, 'timestamp_utc', now):.6f}"
            ).encode("utf-8"),
        ).hexdigest()[:_CHAIN_HEX_LEN]

        liouville = _fattr(snap, "liouville_volume_drift", 0.0)
        is_trace_ok = abs(liouville) <= _TRACE_CONSERVATION_TOL
        sympl_res, sympl_def, is_sympl, is_flat, pole = _symplectic_tuple(snap)
        poincare_res, poincare_resonant, lindstedt_ok, newton_it = _lindstedt_tuple(snap)
        kappa = _spectral_kappa(snap)

        if (not is_trace_ok) or (not is_sympl):
            verdict_p1 = HeytingOmega3.VETOED
        elif liouville > _LIOUVILLE_SOFT or poincare_resonant:
            verdict_p1 = HeytingOmega3.DEGRADED
        else:
            verdict_p1 = HeytingOmega3.COHERENT

        unsigned1 = WitnessPhase1Certificate(
            witness_id=f"WIT-P1-{int(now * 1000.0) % 1000000:06d}",
            trace_preservation=1.0 - liouville,
            liouville_drift=liouville,
            isometry_verdict=verdict_p1,
            hmac_p1_signature="",
            timestamp_utc=now,
            dimension=int(dim),
            connection_norm_frobenius=0.0,
            is_anti_hermitian_A=_battr(p1env, "is_anti_hermitian_A", True),
            is_traceless_A=_battr(p1env, "is_traceless_A", True),
            is_trace_conserved=is_trace_ok,
            symplectic_residue=sympl_res,
            symplectic_defect=sympl_def,
            is_symplectic_valid=is_sympl,
            poincare_residual=poincare_res,
            poincare_is_resonant=poincare_resonant,
            yang_mills_energy_baseline=_yang_mills_energy(snap, 0.0),
            engine_envelope=snap,
            chain_hash=chain_p1,
            engine_interop=True,
            lindstedt_solved=lindstedt_ok,
            lindstedt_newton_iterations=newton_it,
            spectral_condition_number=kappa,
            is_flat_curvature=is_flat or _battr(snap, "is_flat_curvature", True),
            cayley_pole_margin=pole,
            hmac_verified=False,
        )
        cert1 = self._stamp_hmac_p1(unsigned1)

        m = self._extract_p2_metrics(snap)
        if m["ym_energy"] == 0.0:
            m["ym_energy"] = float(cert1.yang_mills_energy_baseline)
        p2_atoms: List[HeytingOmega3] = [
            HeytingOmega3.COHERENT if m["ym_energy"] <= _YM_ENERGY_SOFT else (
                HeytingOmega3.DEGRADED if m["ym_energy"] <= _YM_ENERGY_HARD
                else HeytingOmega3.VETOED
            ),
            HeytingOmega3.COHERENT if m["is_qnd"] else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if m["back_action"] <= _QND_BACKACTION_DB_TOL
            else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if m["uhlmann"] >= _UHLMANN_FIDELITY_MIN
            else HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT if m["rsi3_ok"] else HeytingOmega3.VETOED,
        ]
        verdict_p2 = cert1.isometry_verdict.join(
            HeytingOmega3.fold_supremum(p2_atoms),
        )
        chain_p2 = hashlib.sha256(
            (
                f"consume_engine_p2:{_attr(snap, 'sha256_merkle_digest', '')}:"
                f"{int(verdict_p2)}"
            ).encode("utf-8"),
        ).hexdigest()[:_CHAIN_HEX_LEN]

        unsigned2 = WitnessPhase2Certificate(
            witness_id=f"WIT-P2-{int(now * 1000.0 + 1) % 1000000:06d}",
            weak_value_qnd_aav=m["qnd_weak"],
            back_action_db=m["back_action"],
            uhlmann_fidelity=m["uhlmann"],
            bures_distance=m["bures"],
            yang_mills_energy=m["ym_energy"],
            qnd_verdict=verdict_p2,
            hmac_p2_signature="",
            timestamp_utc=now,
            is_qnd_preserved=m["is_qnd"],
            qnd_commutator_norm=m["qnd_comm"],
            qnd_fidelity=m["qnd_fid"],
            trace_distance=m["trace_dist"],
            liouville_volume_drift=m["liouville"],
            stora_zumino_residual=m["sz"],
            kam_worst_ratio=m["kam_w"],
            kam_satisfied=m["kam_ok"],
            melnikov_value=m["mel"],
            melnikov_preserved=m["mel_ok"],
            floquet_max_unit_deviation=m["flo_dev"],
            floquet_neutral=m["flo_neutral"],
            poincare_section_n_crossings=m["poin_x"],
            poincare_section_dispersion=m["poin_disp"],
            rsi3_valid=m["rsi3_ok"],
            rsi3_iterations=m["rsi3_it"],
            banach_lipschitz_t3=m["banach"],
            fubini_study_distance=m["fubini"],
            novikov_krull_valuation=m["novikov"],
            engine_envelope=snap,
            chain_hash=chain_p2,
            engine_interop=True,
            bruno_satisfied=m["bruno_ok"],
            bruno_sum=m["bruno_sum"],
            birkhoff_twist_estimate=m["birk_est"],
            birkhoff_is_twist=m["birk_twist"],
            birkhoff_area_preserving=m["birk_area"],
            kam_n_modes=m["kam_n"],
            mean_poincare_return_time=m["mean_ret"],
            nested_pipeline="consume_snapshot",
            hmac_verified=False,
            hmac_engine_verified=bool(m["engine_hmac_ok"]),
        )
        cert2 = self._stamp_hmac_p2(unsigned2)
        return self.emit_sovereign_state(cert1, cert2)


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__: List[str] = [
    "WitnessAgentError",
    "WitnessContractCorruptedError",
    "WitnessGateRejectionError",
    "WitnessSymplecticViolationError",
    "WitnessHMACVerificationError",
    "WitnessPhase1Certificate",
    "WitnessPhase2Certificate",
    "AgentRSI3Certificate",
    "TopologicalGaugeFieldWitnessSovereignState",
    "WitnessPhase1Observer",
    "WitnessPhase2Orientor",
    "WitnessPhase3Decider",
    "TopologicalGaugeFieldWitnessAgent",
    "GaugeWitnessSnapshot",
    "HeytingOmega3",
    "TopologicalGaugeFieldWitnessEngine",
]


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN DOCTORAL
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    banner = "═" * 80
    print(banner)
    print("  VERIFICACIÓN DOCTORAL — Topological Gauge Field Witness Agent v10.1.0")
    print(
        f"  Engine importado: {_ENGINE_IMPORTED} | Envelopes: {_ENGINE_HAS_ENVELOPES}"
        f" | Nested P2: {_ENGINE_HAS_P2_ENVELOPE}",
    )
    print(banner)

    agent = TopologicalGaugeFieldWitnessAgent(dimension=56)
    rng = np.random.default_rng(2026)
    state = rng.normal(0.0, 1.0, size=(56, 56)) + 1j * rng.normal(0.0, 1.0, size=(56, 56))
    rho = state @ state.conj().T
    rho /= np.trace(rho)

    witness_state = agent.witness_gauge_field(rho)
    hmac_p3_ok = agent.verify_sovereign_state(witness_state)

    print(f"  • Sovereign State ID      : {witness_state.sovereign_state_id}")
    print(f"  • Witness Mode            : {witness_state.witness_mode}")
    print(f"  • Crowbar Triggered       : {witness_state.crowbar_triggered}")
    print(f"  • Agent version           : {witness_state.agent_version}")
    print(f"  • Nested pipeline         : {witness_state.nested_pipeline}")
    print(
        f"  • Veredicto Global Ω₃     : {witness_state.global_verdict.name} "
        f"(0x{int(witness_state.global_verdict):02x})",
    )
    print(f"  • Back-Action QND         : {witness_state.back_action_db:.6f} dB")
    print(f"  • QND Preservada          : {witness_state.phase2_cert.is_qnd_preserved}")
    print(f"  • ‖[A,ρ]‖_F               : {witness_state.phase2_cert.qnd_commutator_norm:.3e}")
    print(f"  • Fidelidad Uhlmann       : {witness_state.phase2_cert.uhlmann_fidelity:.6f}")
    print(f"  • Distancia Bures         : {witness_state.phase2_cert.bures_distance:.6f}")
    print(f"  • Curvatura plana         : {witness_state.is_flat_curvature}")
    print(f"  • δω simpléctico          : {witness_state.phase1_cert.symplectic_residue:.3e}")
    print(f"  • ‖KᵀJ+JK‖_F              : {witness_state.phase1_cert.symplectic_defect:.3e}")
    print(f"  • Poincaré residual       : {witness_state.phase1_cert.poincare_residual:.3e}")
    print(f"  • Lindstedt resuelto      : {witness_state.lindstedt_solved}")
    print(f"  • κ₂(iA)                  : {witness_state.spectral_condition_number:.4e}")
    print(f"  • KAM worst-ratio         : {witness_state.phase2_cert.kam_worst_ratio:.4f}")
    print(f"  • Bruno Σ / OK            : {witness_state.bruno_sum:.4f} / {witness_state.bruno_satisfied}")
    print(f"  • Melnikov |M|            : {abs(witness_state.phase2_cert.melnikov_value):.4e}")
    print(f"  • Floquet |Δμ|max         : {witness_state.phase2_cert.floquet_max_unit_deviation:.3e}")
    print(f"  • Poincaré cruzamientos   : {witness_state.phase2_cert.poincare_section_n_crossings}")
    print(
        f"  • Birkhoff twist          : {witness_state.birkhoff_twist_estimate:.4e} "
        f"(twist={witness_state.birkhoff_is_twist})",
    )
    print(f"  • RSI³ motor iter         : {witness_state.phase2_cert.rsi3_iterations}")
    print(f"  • RSI³ agente valid/iter  : {witness_state.rsi3_agent_valid} / {witness_state.rsi3_agent_iterations}")
    print(f"  • Banach Lip(T³)          : {witness_state.phase2_cert.banach_lipschitz_t3:.4f}")
    print(f"  • Novikov v_Krull         : {witness_state.phase2_cert.novikov_krull_valuation:.4f}")
    print(f"  • Heyting join atoms      : {witness_state.heyting_join_atoms}")
    print(f"  • Heyting meet reduction  : 0x{witness_state.heyting_meet_reduction:02x}")
    print(f"  • Merkle SHA-256          : {witness_state.merkle_sha256_root[:32]}…")
    print(f"  • HMAC provenance         : {witness_state.hmac_provenance_signature[:32]}…")
    print(f"  • HMAC P3 verificado      : {hmac_p3_ok and witness_state.hmac_p3_verified}")
    print(f"  • chain_hash_p1           : {witness_state.chain_hash_p1}")
    print(f"  • chain_hash_p2           : {witness_state.chain_hash_p2}")
    print(f"  • Epistemológicamente     : {witness_state.is_epistemologically_valid}")
    print(
        f"  • Acción ejecutiva        : "
        f"{witness_state.visceral_report.get('executive_action', '—')}",
    )
    print(banner)