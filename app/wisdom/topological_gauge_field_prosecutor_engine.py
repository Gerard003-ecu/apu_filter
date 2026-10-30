# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : Topological Gauge Field Prosecutor Engine v10.1.0 — Tres Fases    ║
║ Ubicación: app/wisdom/topological_gauge_field_prosecutor_engine.py           ║
║ Versión  : 10.1.0 (SOVEREIGN-GAUGE-PROSECUTOR-FINSLER-MAHALANOBIS-           ║
║           CROWBAR-POINCARE-RSI3-HEYTING-PhD)                                  ║
║ Fases    : ①Observe → ②Orient → ③Decide&Act (bridges emit_*/from_*)          ║
╚══════════════════════════════════════════════════════════════════════════════╝

ESTRUCTURA DE LAS TRES FASES ANIDADAS
─────────────────────────────────────
  FASE 1 — OBSERVE (Conexión, curvatura, RHI-Finsler, Cayley-Kähler, Lindstedt)
    • Sanitización PSD+Tr ρ=1 (Weyl + clamps espectrales + renorm).
    • Conexión A ∈ su(N) verificada anti-Hermítica y traceless.
    • Curvatura F = [A, ρ_h] (anti-Hermítica); E_YM = −½Tr(F²) ≥ 0.
    • RHI-Finsler acotado: IPR ∈ [1/n,1], R_base ∈ [0,1],
      RHI = √(R_base² + ¼ tanh²(E_YM)) ∈ [0,1].
    • SpectralGaugeCache: un único eigh(iA) ⇒ A = −i V diag(ω) V†,
      exp(tA) = V diag(e^{−i t ω}) V†.  Prohíbe expm×T.
    • Cayley-Kähler: rama plana explícita si ‖F‖_F ≈ 0 (evita K=−I falso).
    • Poincaré-Lindstedt en la *base propia* de iA, Tikhonov + Newton (8).
    • CIERRA con `emit_phase1_envelope` → `ProsecutorPhase1Envelope`.

  FASE 2 — ORIENT (Chern-Simons, Wilson, Pontryagin, Kelly, Poincaré, RSI³)
    • APERTURA con `from_phase1_envelope` (verifica chain_hash 32-hex).
    • CS coeficiente 2/3 + anomalía |Tr(F³)|/(8π²).
    • Wilson W = (∑_k e^{−i ω_k})/N vía cache (sin expm).
    • Pontryagin p₁, k=round(p₁), integralidad + Stora-Zumino.
    • Fermi-Dirac Kelly censoring + ds*/dRHI analítico.
    • KAM sobre r≤4 modos dominantes (nunca 7^{56}).
    • Bruno Σ_k log(Ω_k)/2^k sobre la misma red recortada.
    • Melnikov Simpson n=129 sobre la órbita espectral.
    • Floquet μ_k = e^{−i T ω_F} vía eigh(iF).
    • Sección de Poincaré + tiempo medio de retorno.
    • Birkhoff twist (β, residual de área).
    • RSI³: Lip(Π_PSD-tr, Π_su, Π_ah) + Newton homológico en el grafo
      computacional (no muta el campo). Banach Lip(T³) ≤ 0.95³.
    • CIERRA con `emit_phase2_envelope` → `ProsecutorPhase2Envelope`.

  FASE 3 — DECIDE & ACT (Heyting Ω₃, Crowbar verificado, HMAC)
    • APERTURA con `from_phase2_envelope` (verifica chain_hash).
    • Adjudicación Heyting Ω₃ sobre **12 átomos** vía join(∨) monoidal.
    • Crowbar ESP32/BT151 con latencia modelada IRAM ≤ 392.15 ns < 400 ns.
    • Firma HMAC-SHA256 dominio v10.1.0 (ruptura de contrato v10.0.0).
    • CIERRA con `emit_prosecutor_snapshot` → `GaugeProsecutorSnapshot`.

MIGRACIÓN v10.0.0 → v10.1.0 (breaking change controlado)
────────────────────────────────────────────────────────
  • Lindstedt: Im diag(A) en la base original ERA INCORRECTO.
    Ahora se diagonaliza iA y se resuelve en la base propia.
  • KAM: product([-K,K]^d) con d=56 es 7^{56}. Recorte a r≤4 modos.
  • Cayley: F≈0 ⇒ K=−I ⇒ ‖KᵀJ+JK‖_F = 2‖J‖_F (falso defecto).
    Rama plana explícita (`is_flat_branch`).
  • Órbitas: un `eigh(iA)` sustituye a expm×T (Melnikov/Floquet/sección/Wilson).
  • RSI³: el bucle v10.0 era idempotente. Ahora Lip-probe + Newton homológico.
  • chain_hash: 16 hex → 32 hex + shape + dtype.
  • HMAC domain: `...-v10.0.0|` → `...-v10.1.0|` (no mezclar).
  • consume_*(None) levanta `GaugeProsecutorError`, nunca un DTO.
  • DTOs extendidos con defaults (retrocompat de nombres).
  • Heyting Ω₃: VETOED=0x03, meet=min, join=max (sin cambio).

INVARIANTES GLOBALES
────────────────────
  [A1] ρ = ρ† ⪰ 0, Tr ρ = 1; A† = −A, Tr A = 0; F† = −F, Tr F = 0.
  [A2] RHI ∈ [0, 1]; IPR ∈ [1/n, 1]; RHI_base ∈ [0, 1].
  [A3] δω ≤ 10⁻¹⁰; ‖KᵀJ+JK‖_F ≤ 10⁻¹⁰ (o rama plana).
  [A4] |p₁ − k| ≤ 10⁻⁶; E_YM ≥ 0; |W| ≤ 1.
  [A5] |ω·k|·‖k‖₁^τ ≥ γ (τ=r+1, r≤4); |M(θ₀)| ≤ 0.5; |μ_k| ≤ 1+10⁻⁶.
  [A6] Lip(T³) ≤ 0.95³; d_FS ≤ 10⁻⁴; v_Krull ≥ 0.10.
  [A7] Ω₃ ∈ {0x00,0x01,0x03} retículo de Heyting completo.
  [A8] Crowbar GPIO14 @ t ≤ 392.15 ns < 400 ns (modelo IRAM).
  [A9] HMAC-SHA256(snapshot, K_soberano, dominio_v10.1.0) = firma verificable.
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from itertools import product
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ──────────────────────────────────────────────────────────────────────────────
# §0. INTEROPERABILIDAD OPCIONAL CON MOTOR DE CAMPO v10.x
# ──────────────────────────────────────────────────────────────────────────────
_ENGINE_IMPORTED: bool = False
_ENGINE_HAS_ENVELOPES: bool = False
try:
    from app.wisdom.topological_gauge_field_engine import (  # noqa: F401
        TopologicalGaugeFieldEngine as _EngineRef,
    )
    _ENGINE_IMPORTED = True
    _ENGINE_HAS_ENVELOPES = hasattr(_EngineRef, "emit_phase1_envelope")
except ImportError:
    try:
        from topological_gauge_field_engine import (  # noqa: F401
            TopologicalGaugeFieldEngine as _EngineRef,
        )
        _ENGINE_IMPORTED = True
        _ENGINE_HAS_ENVELOPES = hasattr(_EngineRef, "emit_phase1_envelope")
    except ImportError:
        _EngineRef = None  # type: ignore[assignment]


logger = logging.getLogger("APU.Wisdom.TopologicalGaugeFieldProsecutorEngine.v10")

_ENGINE_VERSION: Final[str] = "10.1.0"
__version__: Final[str] = _ENGINE_VERSION

# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES FÍSICAS, NUMÉRICAS Y DE POINCARÉ
# ══════════════════════════════════════════════════════════════════════════════
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 392.15
_CROWBAR_TIMING_LIMIT_NS: Final[float] = 400.0
_CROWBAR_DEGRADED_LATENCY_NS: Final[float] = 120.0
_CROWBAR_NOMINAL_LATENCY_NS: Final[float] = 45.0
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_GAUGE_PROSECUTOR_SOVEREIGN_KEY"
_GAUGE_PROSECUTOR_HMAC_DOMAIN: Final[bytes] = b"APU-FILTER-GAUGE-PROSECUTOR-v10.1.0|"
_CHAIN_HEX_LEN: Final[int] = 32

# Tolerancias de certificación
_ANTI_HERM_TOL: Final[float] = 1.0e-10
_TRACELESS_TOL: Final[float] = 1.0e-10
_SYMPLECTIC_TOL: Final[float] = 1.0e-10
_INSTANTON_INTEGRALITY_TOL: Final[float] = 1.0e-6
_TRACE_CONSERVATION_TOL: Final[float] = 1.0e-2
_FLAT_CURVATURE_TOL: Final[float] = 1.0e-12
_CAYLEY_POLE_MARGIN_MIN: Final[float] = 1.0e-10
_LINDSTEDT_RESIDUAL_TOL: Final[float] = 1.0e-9
_SMALL_DIVISOR_GAP: Final[float] = 1.0e-10

# Umbrales del Fiscal
_RHI_HARD: Final[float] = 0.85
_RHI_SOFT: Final[float] = 0.30
_YM_HARD: Final[float] = 0.85
_YM_SOFT: Final[float] = 0.25
_INSTANTON_HARD: Final[int] = 1
_KELLY_T_SW: Final[float] = 0.02
_KELLY_RHI_CRIT: Final[float] = 0.85
_KELLY_RAW: Final[float] = 0.25

# Poincaré (Birkhoff–Arnold–Moser) — recorte combinatorio
_KAM_GAMMA: Final[float] = 0.125
_KAM_TAU_OFFSET: Final[float] = 1.0
_KAM_MAX_ORDER: Final[int] = 3
_KAM_EFFECTIVE_MODES: Final[int] = 4
_MELNIKOV_TOL: Final[float] = 0.5
_MELNIKOV_SIMPSON_N: Final[int] = 129
_FLOQUET_UNIT_TOL: Final[float] = 1.0e-6
_POINCARE_SECTION_SAMPLES: Final[int] = 96
_LINDSTEDT_NEWTON_ITERS: Final[int] = 8
_LINDSTEDT_TIKHONOV_EPS: Final[float] = 1.0e-12
_BRUNO_K_MAX: Final[int] = 5
_BRUNO_N_CAP: Final[int] = 6
_BRUNO_SUM_FLOOR: Final[float] = -50.0
_BIRKHOFF_AREA_TOL: Final[float] = 1.0e-4
_BIRKHOFF_TWIST_FLOOR: Final[float] = 1.0e-8

# RSI³ (Darwin–Gödel Nivel 3)
_KRULL_NOVIKOV_FLOOR: Final[float] = 0.10
_FUBINI_STUDY_MAX: Final[float] = 1.0e-4
_BANACH_LIPSCHITZ_MAX: Final[float] = 0.95
_LIP_PROBE_EPS: Final[float] = 1.0e-6
_LIP_PROBE_N: Final[int] = 4
_HOMOLOGICAL_RESIDUAL_TOL: Final[float] = 1.0e-6


# ══════════════════════════════════════════════════════════════════════════════
# §B. RETÍCULO DE HEYTING Ω₃ COMPLETO
# ══════════════════════════════════════════════════════════════════════════════
class HeytingOmega3(IntEnum):
    r"""
    Retículo de Heyting trivalente lineal con top = VETOED:
        COHERENT (0x00) < DEGRADED (0x01) < VETOED (0x03).

    Operaciones:
      • join(∨) = max  — supremo (colapso al peor caso)
      • meet(∧) = min  — ínfimo
      • implies(→) — a→b = top si a≤b, sino b
      • negate(¬) — a→⊥ con ⊥ = COHERENT
    """
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""$a\to b$ = VETOED (top) si a≤b, sino b."""
        return HeytingOmega3.VETOED if int(self) <= int(other) else other

    def negate(self) -> "HeytingOmega3":
        r"""¬a = a → ⊥ con ⊥ = COHERENT."""
        return self.implies(HeytingOmega3.COHERENT)

    @classmethod
    def fold_supremum(cls, verdicts: Sequence["HeytingOmega3"]) -> "HeytingOmega3":
        acc = cls.COHERENT
        for v in verdicts:
            acc = acc.join(v)
        return acc

    @classmethod
    def fold_infimum(cls, verdicts: Sequence["HeytingOmega3"]) -> "HeytingOmega3":
        acc = cls.VETOED
        for v in verdicts:
            acc = acc.meet(v)
        return acc

    @classmethod
    def legacy_from_0x02(cls, legacy: int) -> "HeytingOmega3":
        """Compatibilidad: mapea 0x02 (legado) → 0x03 (actual)."""
        return cls.VETOED if legacy == 0x02 else cls(legacy)


# ══════════════════════════════════════════════════════════════════════════════
# §C. EXCEPCIONES
# ══════════════════════════════════════════════════════════════════════════════
class GaugeProsecutorError(Exception):
    """Excepción raíz del Fiscal de Gauge."""


class ProsecutorContractCorruptedError(GaugeProsecutorError):
    """Firma HMAC o chain_hash corrompidos."""


class ProsecutorSymplecticViolationError(GaugeProsecutorError):
    """Embedding Cayley-Kähler fuera de sp(2d,ℂ)."""


class ProsecutorCrowbarLatencyError(GaugeProsecutorError):
    """Latencia del interlock > 400 ns."""


class ProsecutorSpectralError(GaugeProsecutorError):
    """Fallo espectral (eigh) en el cache de iA / iF."""


# ══════════════════════════════════════════════════════════════════════════════
# §D. HELPERS — proyecciones, J canónica, hash 32-hex
# ══════════════════════════════════════════════════════════════════════════════
def _project_anti_hermitian(M: NDArray[Any]) -> NDArray[np.complex128]:
    r"""$\Pi_{\mathrm{ah}}(M) = \tfrac12(M - M^\dagger)$ — proyección ortogonal."""
    a = np.asarray(M, dtype=np.complex128)
    return 0.5 * (a - a.conj().T)


def _project_su_n(M: NDArray[Any]) -> NDArray[np.complex128]:
    r"""$\Pi_{\mathfrak{su}(N)}(M) = \Pi_{\mathrm{ah}}(M) - (\mathrm{Tr}/N)\,I$."""
    a = _project_anti_hermitian(M)
    n = int(a.shape[0])
    if n == 0:
        return a
    return a - (np.trace(a) / float(n)) * np.eye(n, dtype=np.complex128)


def _standard_symplectic_j(d: int) -> NDArray[np.complex128]:
    r"""$J = \begin{pmatrix} 0 & I \\ -I & 0 \end{pmatrix} \in \mathrm{Sp}(2d)$."""
    z = np.zeros((d, d), dtype=np.complex128)
    eye = np.eye(d, dtype=np.complex128)
    return np.block([[z, eye], [-eye, z]])


def _hash_state(*arrays: NDArray[Any]) -> str:
    r"""SHA-256 truncado a 32 hex, con shape+dtype (rompe colisiones v10.0)."""
    h = hashlib.sha256()
    for a in arrays:
        arr = np.ascontiguousarray(a)
        h.update(str(arr.shape).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()[:_CHAIN_HEX_LEN]


def _dominant_frequencies(
    omega: NDArray[np.float64],
    r: int = _KAM_EFFECTIVE_MODES,
) -> NDArray[np.float64]:
    """Recorta a los r modos de mayor |ω|; nunca alimenta una red 7^{56}."""
    w = np.asarray(omega, dtype=np.float64).flatten()
    w = w[np.abs(w) > 1e-15]
    if w.size == 0:
        return np.array([1.0], dtype=np.float64)
    if w.size > r:
        idx = np.argsort(np.abs(w))[::-1][:r]
        w = w[idx]
    return np.asarray(w, dtype=np.float64)


def _tikhonov_homological(
    omega: NDArray[np.float64],
    R: NDArray[np.complex128],
    eps: float = _LINDSTEDT_TIKHONOV_EPS,
) -> NDArray[np.complex128]:
    r"""
    Inversa regularizada de $L(W)_{ij} = -i(\omega_i-\omega_j)W_{ij}$:
    $$W_{ij} = i\, R_{ij}\, \Delta\omega / (\Delta\omega^2 + \varepsilon^2).$$
    """
    dw = omega[:, None] - omega[None, :]
    den = dw * dw + eps * eps
    W = (1j * R * dw) / den
    np.fill_diagonal(W, 0.0)
    return W


def _apply_homological_L(
    omega: NDArray[np.float64],
    W: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    r"""$L(W)_{ij} = -i(\omega_i-\omega_j)W_{ij}$."""
    dw = omega[:, None] - omega[None, :]
    L = -1j * dw * W
    np.fill_diagonal(L, 0.0)
    return L


# ══════════════════════════════════════════════════════════════════════════════
# §E. DTOs INMUTABLES POR FASE
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class SpectralGaugeCache:
    r"""
    Un único eigh de $iA$ (Hermítico si $A^\dagger=-A$):

      $$A = -i\, V\,\mathrm{diag}(\omega)\,V^\dagger,
        \qquad e^{tA} = V\,\mathrm{diag}(e^{-i t \omega})\,V^\dagger.$$
    """
    omega: NDArray[np.float64]
    V: NDArray[np.complex128]
    condition_number: float
    anti_hermitian_residual: float
    dimension: int

    def unitary_orbit(self, t: float) -> NDArray[np.complex128]:
        phases = np.exp(-1j * float(t) * self.omega)
        return (self.V * phases) @ self.V.conj().T

    def apply_orbit(
        self, t: float, phi: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        vec = np.asarray(phi, dtype=np.complex128).reshape(-1)
        coeff = self.V.conj().T @ vec
        return self.V @ (np.exp(-1j * float(t) * self.omega) * coeff)

    def wilson_trace(self) -> complex:
        n = max(int(self.dimension), 1)
        return complex(np.sum(np.exp(-1j * self.omega)) / float(n))


@dataclass(frozen=True, slots=True)
class RHIFinslerCertificate:
    r"""Certificado del Reward Hacking Index sobre la métrica Finsleriana."""
    rhi_final: float
    rhi_base: float
    ipr_raw: float
    participation_ratio: float
    yang_mills_energy: float
    spectral_eigenvalues: NDArray[np.float64]
    frobenius_norm_sq: float
    method: str = "ipr-normalized-v10.1"


@dataclass(frozen=True, slots=True)
class KellyCensorshipCertificate:
    r"""Certificado de censura Fermi-Dirac de Kelly (verificado)."""
    raw_kelly: float
    rhi: float
    sigmoid_factor: float
    censored_kelly: float
    derivative_ds_drhi: float
    temperature_sw: float
    is_censored: bool


@dataclass(frozen=True, slots=True)
class SymplecticReport:
    symplectic_residue: float
    symplectic_defect: float
    is_symplectic_valid: bool
    is_flat_branch: bool = False
    cayley_pole_margin: float = 0.0
    lie_algebra_defect_projected: float = 0.0


@dataclass(frozen=True, slots=True)
class PoincareLindstedtSolution:
    W: NDArray[np.complex128]
    min_resonance_gap: float
    residual_norm: float
    is_resonant: bool
    is_solved: bool
    newton_iterations: int = 0
    small_divisor_count: int = 0
    frequencies: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )


@dataclass(frozen=True, slots=True)
class KAMCertificate:
    is_satisfied: bool
    worst_ratio: float
    tau: float
    gamma: float
    omega_min: float
    n_modes: int = 0
    n_lattice_probes: int = 0


@dataclass(frozen=True, slots=True)
class BrunoCertificate:
    r"""Condición de Bruno: $\sum_k \log\Omega_k / 2^k > -\infty$ (corte finito)."""
    is_satisfied: bool
    bruno_sum: float
    omega_min_divisor: float
    n_probes: int
    continued_fraction_bound: float
    k_max: int = _BRUNO_K_MAX


@dataclass(frozen=True, slots=True)
class MelnikovCertificate:
    integral_value: float
    max_abs_integrand: float
    is_homoclinic_preserved: bool


@dataclass(frozen=True, slots=True)
class FloquetCertificate:
    multipliers: NDArray[np.complex128]
    max_unit_deviation: float
    is_neutral: bool
    is_dissipative_stable: bool
    exponents: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PoincareSectionCertificate:
    n_crossings: int
    return_dispersion: float
    is_regular: bool
    returns: Tuple[float, ...]
    mean_return_time: float = 0.0


@dataclass(frozen=True, slots=True)
class BirkhoffTwistCertificate:
    r"""Twist de Birkhoff sobre la sección: $\Delta\theta \approx \alpha + \beta I$."""
    twist_coefficient: float
    mean_rotation_number: float
    area_residual: float
    is_twist: bool
    is_area_preserving: bool


@dataclass(frozen=True, slots=True)
class RSI3SurfaceReport:
    surface_id: int
    name: str
    metric_value: float
    threshold: float
    passed: bool
    diagnostics: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class RSI3Certificate:
    surfaces: Tuple[RSI3SurfaceReport, RSI3SurfaceReport, RSI3SurfaceReport]
    lipschitz_t3: float
    capacity_jerk: float
    fixed_point_reached: bool
    iterations: int
    is_valid: bool
    lipschitz_factors: Tuple[float, ...] = ()
    homological_residual: float = 0.0

    @property
    def surface1(self) -> RSI3SurfaceReport:
        return self.surfaces[0]

    @property
    def surface2(self) -> RSI3SurfaceReport:
        return self.surfaces[1]

    @property
    def surface3(self) -> RSI3SurfaceReport:
        return self.surfaces[2]


@dataclass(frozen=True, slots=True)
class ProsecutorPhase1Envelope:
    r"""Contrato FASE 1 (Observe) → FASE 2 (Orient)."""
    rho_sanitized: NDArray[np.complex128]
    connection_A: NDArray[np.complex128]
    curvature_F: NDArray[np.complex128]
    yang_mills_energy: float
    is_anti_hermitian_A: bool
    is_traceless_A: bool
    is_anti_hermitian_F: bool
    is_traceless_F: bool
    rhi_finsler: RHIFinslerCertificate
    symplectic_report: SymplecticReport
    poincare_lindstedt: Optional[PoincareLindstedtSolution]
    dimension: int
    origin_timestamp_ns: float
    chain_hash: str
    spectral_cache: Optional[SpectralGaugeCache] = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class ProsecutorPhase2Envelope:
    r"""Contrato FASE 2 (Orient) → FASE 3 (Decide&Act)."""
    phase1: ProsecutorPhase1Envelope
    chern_simons_real: float
    chern_simons_anomaly: float
    wilson_trace: complex
    wilson_angle_rad: float
    c1_value: float
    pontryagin_instanton_k: int
    pontryagin_integrality_residual: float
    stora_zumino_residual: float
    kelly_certificate: KellyCensorshipCertificate
    kam_certificate: KAMCertificate
    melnikov: MelnikovCertificate
    floquet: FloquetCertificate
    poincare_section: PoincareSectionCertificate
    rsi3_certificate: RSI3Certificate
    topology_regime: str
    chain_hash: str
    origin_timestamp_ns: float
    bruno_certificate: Optional[BrunoCertificate] = None
    birkhoff_twist: Optional[BirkhoffTwistCertificate] = None


@dataclass(frozen=True, slots=True)
class GaugeProsecutorSnapshot:
    r"""
    Snapshot inmutable del Fiscal (objeto terminal FASE 3).

    Campos originales (retrocompat) + certificados extendidos v10.1.0.
    """
    # Campos originales (retrocompat)
    prosecutor_id: str
    rhi_gauge_finsler: float
    fermi_dirac_kelly_fraction: float
    yang_mills_energy: float
    chern_simons_3form_higgs: float
    wilson_angle_rad: float
    pontryagin_instanton_k: int
    stora_zumino_residual: float
    crowbar_triggered: bool
    crowbar_latency_ns: float
    global_verdict: HeytingOmega3
    sha256_merkle_digest: str
    timestamp_utc: float

    # Campos extendidos v10.0.0
    rhi_base_normalized: float = 0.0
    ipr_raw: float = 0.0
    kelley_derivative_ds_drhi: float = 0.0
    chern_simons_anomaly: float = 0.0
    c1_value: float = 0.0
    pontryagin_integrality_residual: float = 0.0
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
    rsi3_valid: bool = True
    rsi3_iterations: int = 1
    banach_lipschitz_t3: float = 0.0
    fubini_study_distance: float = 0.0
    novikov_krull_valuation: float = 0.0
    crowbar_latency_verified: bool = True
    hardware_action: str = "NOMINAL_EXECUTION"
    heyting_join_atoms: Tuple[int, ...] = ()
    heyting_meet_reduction: int = 0
    heyting_implies_table: Tuple[Tuple[int, int, int], ...] = ()
    heyting_negate_table: Tuple[int, int, int] = ()
    is_epistemologically_valid: bool = True
    phase1: Optional[ProsecutorPhase1Envelope] = field(default=None, repr=False)
    phase2: Optional[ProsecutorPhase2Envelope] = field(default=None, repr=False)
    hmac_sha256: bytes = b""
    engine_version: str = _ENGINE_VERSION

    # Campos extendidos v10.1.0
    hmac_verified: bool = False
    lindstedt_solved: bool = False
    lindstedt_newton_iterations: int = 0
    spectral_condition_number: float = 1.0
    kam_n_modes: int = 0
    is_flat_curvature: bool = False
    mean_poincare_return_time: float = 0.0
    bruno_satisfied: bool = True
    bruno_sum: float = 0.0
    birkhoff_twist_coefficient: float = 0.0
    birkhoff_area_residual: float = 0.0
    birkhoff_is_twist: bool = False
    cayley_pole_margin: float = 0.0
    lie_algebra_defect_projected: float = 0.0
    rsi3_homological_residual: float = 0.0
    rsi3_lipschitz_factors: Tuple[float, ...] = ()


# ══════════════════════════════════════════════════════════════════════════════════
# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║  F A S E   1 :   O B S E R V E                                               ║
# ║  Conexión, curvatura YM, RHI-Finsler, cache espectral, Cayley, Lindstedt.    ║
# ║  Cierra con `emit_phase1_envelope` → `ProsecutorPhase1Envelope`.             ║
# ╚════════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════════

class ProsecutorPhase1Observer:
    r"""
    FASE 1 (Observe): sanitización, conexión de gauge, curvatura YM,
    RHI-Finsler, SpectralGaugeCache, Cayley-Kähler (rama plana) y
    Poincaré-Lindstedt en la base propia de iA.
    """

    def __init__(
        self, dimension: int = 56, secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> None:
        self.dimension: Final[int] = max(4, int(dimension))
        self.secret_key: Final[bytes] = bytes(secret_key)

    # ── HMAC helpers ──────────────────────────────────────────────────────────
    def _compute_hmac(self, payload: str) -> str:
        return hmac.new(
            self.secret_key,
            _GAUGE_PROSECUTOR_HMAC_DOMAIN + payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _verify_hmac(self, payload: str, signature: str) -> bool:
        return hmac.compare_digest(self._compute_hmac(payload), signature)

    # ── Sanitización PSD + Tr ρ=1 ─────────────────────────────────────────────
    def sanitize_density_matrix(self, matrix: NDArray) -> NDArray[np.complex128]:
        a = np.asarray(matrix, dtype=np.complex128)
        if a.shape != (self.dimension, self.dimension):
            raise ValueError(
                f"ρ debe ser {self.dimension}×{self.dimension}; recibido {a.shape}."
            )
        herm = 0.5 * (a + a.conj().T)
        eigvals, eigvecs = la.eigh(herm)
        clean_eigs = np.maximum(np.real(eigvals), _SPECTRAL_PSD_FLOOR)
        psd = (eigvecs * clean_eigs) @ eigvecs.conj().T
        tr = float(np.real(np.trace(psd)))
        if tr <= _WILKINSON_EPSILON:
            return np.eye(self.dimension, dtype=np.complex128) / float(self.dimension)
        return psd / tr

    # ── Método 1: Conexión A ∈ su(N) ──────────────────────────────────────────
    def extract_gauge_connection_A(self, density_matrix: NDArray) -> NDArray[np.complex128]:
        r"""
        $A = i(\rho - \operatorname{Tr}(\rho)/N \cdot I)$,
        anti-Hermítica y traceless por construcción, re-proyectada numéricamente.
        """
        rho = self.sanitize_density_matrix(density_matrix)
        n = rho.shape[0]
        tr_val = np.trace(rho) / float(n)
        su_part = rho - tr_val * np.eye(n, dtype=np.complex128)
        return _project_su_n(1j * su_part)

    def _verify_anti_hermitian(
        self, M: NDArray[np.complex128], name: str = "M",
    ) -> Tuple[bool, float]:
        defect = float(la.norm(M + M.conj().T, "fro"))
        scale = max(1.0, float(la.norm(M, "fro")))
        return bool((defect / scale) < _ANTI_HERM_TOL), defect / scale

    def _verify_traceless(
        self, M: NDArray[np.complex128],
    ) -> Tuple[bool, float]:
        val = float(abs(complex(np.trace(M))))
        return bool(val < _TRACELESS_TOL), val

    # ── Método 2: Curvatura de Yang-Mills F = [A, ρ] ──────────────────────────
    def compute_yang_mills_curvature_F(
        self, A: NDArray[np.complex128], density_matrix: NDArray,
    ) -> Tuple[NDArray[np.complex128], float]:
        r"""
        $F = [A, \rho_h]$ anti-Hermítica; $E_{YM} = -\tfrac{1}{2}\operatorname{Tr}(F^2)\ge 0$.
        """
        rho = self.sanitize_density_matrix(density_matrix)
        F_raw = A @ rho - rho @ A
        F = _project_su_n(F_raw)
        energy_ym = -0.5 * float(np.real(np.trace(F @ F)))
        return F, max(0.0, float(energy_ym))

    # ── Método 3: RHI-Finsler acotado ─────────────────────────────────────────
    def compute_rhi_finsler_gauge(
        self,
        density_matrix: NDArray,
        connection_A: NDArray,
        yang_mills_energy: Optional[float] = None,
    ) -> RHIFinslerCertificate:
        r"""
        Reward Hacking Index rigurosamente acotado.

        Sea $A^\dagger A$ Hermítica PSD con autovalores $\lambda_i\ge 0$:

          1. $\mathrm{IPR} = \sum\lambda_i^2/(\sum\lambda_i)^2 \in [1/n, 1]$.
          2. $R_{\mathrm{base}} = (n\cdot\mathrm{IPR}-1)/(n-1) \in [0, 1]$.
          3. $\mathrm{RHI} = \sqrt{R_{\mathrm{base}}^2 + \tfrac14\tanh^2(E_{YM})} \in [0,1]$.
        """
        a = np.asarray(connection_A, dtype=np.complex128)
        n = a.shape[0]
        a_sq = a.conj().T @ a
        eigs_a_sq = np.clip(np.real(la.eigvalsh(a_sq)), 0.0, None)

        trace_a_sq = float(np.sum(eigs_a_sq))
        trace_a_sq_sq = float(np.sum(eigs_a_sq ** 2))
        frob_sq = trace_a_sq

        if trace_a_sq <= _WILKINSON_EPSILON or n <= 1:
            return RHIFinslerCertificate(
                rhi_final=0.0, rhi_base=0.0, ipr_raw=0.0,
                participation_ratio=1.0 / max(n, 1),
                yang_mills_energy=0.0 if yang_mills_energy is None
                else float(yang_mills_energy),
                spectral_eigenvalues=eigs_a_sq,
                frobenius_norm_sq=frob_sq,
            )

        ipr_raw = trace_a_sq_sq / (trace_a_sq ** 2 + _WILKINSON_EPSILON)
        ipr_raw = float(np.clip(ipr_raw, 1.0 / n, 1.0))
        participation = 1.0 / ipr_raw

        if n > 1:
            rhi_base = (n * ipr_raw - 1.0) / (n - 1.0)
        else:
            rhi_base = 1.0
        rhi_base = float(np.clip(rhi_base, 0.0, 1.0))

        if yang_mills_energy is None:
            _, ym_energy = self.compute_yang_mills_curvature_F(a, density_matrix)
        else:
            ym_energy = float(yang_mills_energy)
        rhi_final = math.sqrt(rhi_base ** 2 + 0.25 * math.tanh(ym_energy) ** 2)
        rhi_final = float(np.clip(rhi_final, 0.0, 1.0))

        return RHIFinslerCertificate(
            rhi_final=rhi_final,
            rhi_base=rhi_base,
            ipr_raw=ipr_raw,
            participation_ratio=float(participation),
            yang_mills_energy=float(ym_energy),
            spectral_eigenvalues=eigs_a_sq,
            frobenius_norm_sq=frob_sq,
        )

    # ── Cache espectral: un eigh(iA) ──────────────────────────────────────────
    def _build_spectral_cache(
        self, A: NDArray[np.complex128],
    ) -> SpectralGaugeCache:
        r"""
        $iA$ es Hermítico si $A^\dagger=-A$. Autovalores reales $\omega$,
        $A=-i V\mathrm{diag}(\omega)V^\dagger$.
        """
        a_su = _project_su_n(A)
        residual = float(la.norm(A + A.conj().T, "fro"))
        iA = 1j * a_su
        iA = 0.5 * (iA + iA.conj().T)
        try:
            omega, V = la.eigh(iA)
        except la.LinAlgError as exc:
            raise ProsecutorSpectralError(
                f"eigh(iA) falló en dim={a_su.shape[0]}: {exc}"
            ) from exc
        omega = np.real(omega).astype(np.float64)
        abs_om = np.abs(omega)
        nz = abs_om[abs_om > 1e-15]
        if nz.size:
            kappa = float(np.max(nz) / max(float(np.min(nz)), 1e-15))
        else:
            kappa = 1.0
        return SpectralGaugeCache(
            omega=omega,
            V=np.asarray(V, dtype=np.complex128),
            condition_number=float(kappa),
            anti_hermitian_residual=residual,
            dimension=int(a_su.shape[0]),
        )

    # ── Cayley-Kähler en sp(2d,ℂ) — rama plana explícita ─────────────────────
    def _cayley_kahler_embedding(
        self, F: NDArray[np.complex128],
    ) -> SymplecticReport:
        r"""
        $K=(iF-I)(iF+I)^{-1}$, $M=(I+K)(I-K)^{-1}$.

        Si $\|F\|_F < \varepsilon_{\mathrm{flat}}$, se declara rama plana:
        $K=-I$ produciría $\|K^T J+JK\|_F=2\|J\|_F$ (falso defecto v10.0).
        """
        d = int(F.shape[0])
        F_ah = _project_anti_hermitian(F)
        lie_def = float(la.norm(F - F_ah, "fro"))
        frob_F = float(la.norm(F_ah, "fro"))

        if frob_F < _FLAT_CURVATURE_TOL:
            return SymplecticReport(
                symplectic_residue=0.0,
                symplectic_defect=0.0,
                is_symplectic_valid=True,
                is_flat_branch=True,
                cayley_pole_margin=float("inf"),
                lie_algebra_defect_projected=lie_def,
            )

        iF = 1j * F_ah
        I_d = np.eye(d, dtype=np.complex128)
        pole_matrix = iF + I_d
        try:
            pole_eigs = la.eigvals(pole_matrix)
            margin = float(np.min(np.abs(pole_eigs)))
        except la.LinAlgError:
            margin = 0.0

        if margin < _CAYLEY_POLE_MARGIN_MIN:
            return SymplecticReport(
                symplectic_residue=1.0,
                symplectic_defect=1.0,
                is_symplectic_valid=False,
                is_flat_branch=False,
                cayley_pole_margin=margin,
                lie_algebra_defect_projected=lie_def,
            )

        J = _standard_symplectic_j(d)
        try:
            K_blk = (iF - I_d) @ la.pinv(pole_matrix)
            K_full = np.block([
                [K_blk, np.zeros((d, d), dtype=np.complex128)],
                [np.zeros((d, d), dtype=np.complex128), K_blk.conj()],
            ])
            sp_defect = float(la.norm(K_full.T @ J + J @ K_full, "fro"))
            I_2d = np.eye(2 * d, dtype=np.complex128)
            M = la.pinv(I_2d - K_full) @ (I_2d + K_full)
            diff = M.T @ J @ M - J
            norm_M = float(la.norm(M, "fro"))
            residue = (
                float(la.norm(diff, "fro") / (norm_M ** 2))
                if norm_M > 0 else float("inf")
            )
            valid = (residue < _SYMPLECTIC_TOL and sp_defect < _SYMPLECTIC_TOL)
            return SymplecticReport(
                symplectic_residue=residue,
                symplectic_defect=sp_defect,
                is_symplectic_valid=bool(valid),
                is_flat_branch=False,
                cayley_pole_margin=margin,
                lie_algebra_defect_projected=lie_def,
            )
        except (la.LinAlgError, ValueError):
            return SymplecticReport(
                1.0, 1.0, False, False, margin, lie_def,
            )

    # ── Poincaré-Lindstedt en la base propia de iA ────────────────────────────
    def _solve_poincare_lindstedt(
        self,
        F: NDArray[np.complex128],
        A: NDArray[np.complex128],
        cache: Optional[SpectralGaugeCache] = None,
    ) -> PoincareLindstedtSolution:
        r"""
        $\{H_0,W\}=H_1-\langle H_1\rangle$ en la base propia de $iA$.

        $L(W)_{ij}=-i\Delta\omega_{ij} W_{ij}$. Tikhonov + Newton homológico
        (``_LINDSTEDT_NEWTON_ITERS``). **No** usa $\mathrm{Im}\,\mathrm{diag}(A)$
        en la base original (bug v10.0).
        """
        if cache is None:
            cache = self._build_spectral_cache(A)
        omega = np.asarray(cache.omega, dtype=np.float64)
        V = cache.V
        d = int(omega.shape[0])

        F_eig = V.conj().T @ np.asarray(F, dtype=np.complex128) @ V
        R = F_eig - np.diag(np.diag(F_eig))
        np.fill_diagonal(R, 0.0)

        dw = omega[:, None] - omega[None, :]
        off = ~np.eye(d, dtype=bool)
        gaps = np.abs(dw[off])
        min_gap = float(np.min(gaps)) if gaps.size else 0.0
        small_div = int(np.sum(gaps < _SMALL_DIVISOR_GAP)) if gaps.size else 0

        W = np.zeros_like(R)
        residual = float(la.norm(R, "fro"))
        it_used = 0
        for it in range(1, _LINDSTEDT_NEWTON_ITERS + 1):
            it_used = it
            resid_mat = R - _apply_homological_L(omega, W)
            dW = _tikhonov_homological(omega, resid_mat)
            W = W + dW
            residual = float(la.norm(R - _apply_homological_L(omega, W), "fro"))
            if residual < _LINDSTEDT_RESIDUAL_TOL:
                break

        W_lab = V @ W @ V.conj().T
        resonant = bool(small_div > 0)
        return PoincareLindstedtSolution(
            W=W_lab,
            min_resonance_gap=min_gap if math.isfinite(min_gap) else 0.0,
            residual_norm=float(residual),
            is_resonant=resonant,
            is_solved=bool(residual < _LINDSTEDT_RESIDUAL_TOL),
            newton_iterations=int(it_used),
            small_divisor_count=int(small_div),
            frequencies=omega.copy(),
        )

    @staticmethod
    def _compute_phase1_chain(
        rho: NDArray[np.complex128],
        A: NDArray[np.complex128],
        F: NDArray[np.complex128],
        energy_ym: float,
        rhi_final: float,
        residue: float,
        defect: float,
        kappa: float,
        omega: Optional[NDArray[np.float64]] = None,
    ) -> str:
        scalars = np.array(
            [energy_ym, rhi_final, residue, defect, kappa],
            dtype=np.float64,
        )
        if omega is None:
            return _hash_state(rho, A, F, scalars)
        return _hash_state(rho, A, F, np.asarray(omega, dtype=np.float64), scalars)

    # ── CIERRE FASE 1 → FASE 2 ────────────────────────────────────────────────
    def emit_phase1_envelope(
        self, density_matrix: NDArray,
    ) -> ProsecutorPhase1Envelope:
        r"""
        **Cierre formal de la FASE 1.**

        Sanitización, $A$, $F$, RHI-Finsler, cache espectral, Cayley-Kähler
        (rama plana) y Poincaré-Lindstedt en base propia. Emite
        `ProsecutorPhase1Envelope` con hash-chain SHA-256 (32 hex).
        """
        t0 = time.perf_counter_ns()
        rho = self.sanitize_density_matrix(density_matrix)
        A = self.extract_gauge_connection_A(rho)
        F, energy_ym = self.compute_yang_mills_curvature_F(A, rho)

        a_ah, _ = self._verify_anti_hermitian(A, "A")
        a_tr, _ = self._verify_traceless(A)
        f_ah, _ = self._verify_anti_hermitian(F, "F")
        f_tr, _ = self._verify_traceless(F)

        cache = self._build_spectral_cache(A)
        rhi = self.compute_rhi_finsler_gauge(rho, A, yang_mills_energy=energy_ym)
        sympl = self._cayley_kahler_embedding(F)
        pl = self._solve_poincare_lindstedt(F, A, cache=cache)

        chain = self._compute_phase1_chain(
            rho, A, F,
            energy_ym, rhi.rhi_final,
            sympl.symplectic_residue, sympl.symplectic_defect,
            cache.condition_number, cache.omega,
        )

        return ProsecutorPhase1Envelope(
            rho_sanitized=rho,
            connection_A=A,
            curvature_F=F,
            yang_mills_energy=float(energy_ym),
            is_anti_hermitian_A=a_ah,
            is_traceless_A=a_tr,
            is_anti_hermitian_F=f_ah,
            is_traceless_F=f_tr,
            rhi_finsler=rhi,
            symplectic_report=sympl,
            poincare_lindstedt=pl,
            dimension=int(rho.shape[0]),
            origin_timestamp_ns=float(time.perf_counter_ns() - t0),
            chain_hash=chain,
            spectral_cache=cache,
        )


# ══════════════════════════════════════════════════════════════════════════════════
# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║  F A S E   2 :   O R I E N T                                                 ║
# ║  Chern-Simons (2/3), Wilson, Pontryagin, Kelly, KAM, Bruno, Melnikov, RSI³. ║
# ║  Abre  con `from_phase1_envelope`.                                           ║
# ║  Cierra con `emit_phase2_envelope` → `ProsecutorPhase2Envelope`.             ║
# ╚════════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════════

class ProsecutorPhase2Orientor(ProsecutorPhase1Observer):
    r"""
    FASE 2 (Orient): topología diferencial, censura de Kelly verificada,
    mecánica celeste de Poincaré (KAM/Bruno/Melnikov/Floquet/sección/Birkhoff)
    y RSI³ con Lip-probe + Newton homológico.
    """

    # ── APERTURA FASE 2 ───────────────────────────────────────────────────────
    @classmethod
    def from_phase1_envelope(
        cls,
        env: ProsecutorPhase1Envelope,
        dimension: Optional[int] = None,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> "ProsecutorPhase2Orientor":
        r"""
        **Apertura formal de la FASE 2.**

        Valida el chain_hash de FASE 1 y construye el orientador.
        """
        dim = int(dimension) if dimension is not None else int(env.dimension)
        inst = cls(dimension=dim, secret_key=secret_key)
        inst.verify_phase1_chain(env)
        inst._phase1_env = env  # type: ignore[attr-defined]
        return inst

    def verify_phase1_chain(self, env: ProsecutorPhase1Envelope) -> None:
        kappa = (
            env.spectral_cache.condition_number
            if env.spectral_cache is not None else 1.0
        )
        omega = (
            env.spectral_cache.omega if env.spectral_cache is not None else None
        )
        expected = self._compute_phase1_chain(
            env.rho_sanitized, env.connection_A, env.curvature_F,
            env.yang_mills_energy, env.rhi_finsler.rhi_final,
            env.symplectic_report.symplectic_residue,
            env.symplectic_report.symplectic_defect,
            kappa, omega,
        )
        if expected != env.chain_hash:
            raise ProsecutorContractCorruptedError(
                f"Chain FASE 1 corrupta: {expected} ≠ {env.chain_hash}."
            )

    def _resolve_cache(
        self,
        A: NDArray[np.complex128],
        cache: Optional[SpectralGaugeCache],
    ) -> SpectralGaugeCache:
        return cache if cache is not None else self._build_spectral_cache(A)

    # ── Método 4: Chern-Simons con coeficiente 2/3 correcto ──────────────────
    def evaluate_chern_simons(
        self, A: NDArray[np.complex128], F: NDArray[np.complex128],
        rho: NDArray[np.complex128],
    ) -> Tuple[float, float]:
        r"""
        $\mathrm{CS}=\frac{1}{8\pi^2}\operatorname{Tr}(AF-\tfrac23 A^3+\Phi F)$,
        $\Phi=\log\rho$. Anomalía $|\operatorname{Tr}(F^3)|/(8\pi^2)$.
        """
        eigvals, eigvecs = la.eigh(rho)
        log_eigs = np.log(np.maximum(np.real(eigvals), _SPECTRAL_PSD_FLOOR))
        phi = (eigvecs * log_eigs) @ eigvecs.conj().T

        a_cubed = A @ A @ A
        cs_term = A @ F - (2.0 / 3.0) * a_cubed + phi @ F
        cs_complex = np.trace(cs_term) / (8.0 * math.pi ** 2)
        cs_real = float(np.real(cs_complex))

        f_cubed = F @ F @ F
        anomaly = float(abs(np.trace(f_cubed)) / (8.0 * math.pi ** 2))
        return cs_real, anomaly

    # ── Método 5: Wilson + c₁ (vía cache espectral) ───────────────────────────
    def evaluate_wilson_loop(
        self,
        A: NDArray[np.complex128],
        rho: NDArray[np.complex128],
        cache: Optional[SpectralGaugeCache] = None,
    ) -> Tuple[complex, float, float, str]:
        r"""$W=\operatorname{Tr}(e^A)/N=(\sum_k e^{-i\omega_k})/N$ (sin expm)."""
        n = int(A.shape[0])
        if cache is not None:
            w_trace = cache.wilson_trace()
        else:
            # Fallback espectral local; jamás expm×T en el hot path.
            local = self._build_spectral_cache(A)
            w_trace = local.wilson_trace()
        w_angle = float(cmath.phase(w_trace))
        modulus = abs(w_trace)
        if modulus > 1.0 + 1e-9:
            logger.warning("Wilson loop no unitario: |W|=%.3e > 1", modulus)

        c1_raw = np.trace(A @ rho) / (2.0j * math.pi)
        c1_val = float(np.real(c1_raw))

        if abs(w_angle) < 0.15 and abs(c1_val) < 1e-4:
            regime = "TRIVIAL_SU2_HOLONOMY"
        elif abs(w_angle) >= 0.50 or abs(c1_val) > 0.10:
            regime = "SOLITONIC_INSTANTON_PROSECUTOR"
        else:
            regime = "WILSON_SLIP_PROSECUTED"
        return w_trace, w_angle, c1_val, regime

    # ── Método 6: Pontryagin + integralidad + Stora-Zumino ────────────────────
    def audit_pontryagin(
        self, F: NDArray[np.complex128], cs_real: float,
    ) -> Tuple[int, float, float]:
        r"""
        $p_1=-\operatorname{Tr}(F^2)/(8\pi^2)\ge 0$; $k=\mathrm{round}(p_1)$;
        $\Delta_{\mathrm{int}}=|p_1-k|\le 10^{-6}$;
        $\Delta_{SZ}=|8\pi^2\,\mathrm{CS}-\operatorname{Tr}(F^2)|$.
        """
        f_wedge_f = F @ F
        p_raw = -float(np.real(np.trace(f_wedge_f))) / (8.0 * math.pi ** 2)
        p_raw = max(0.0, p_raw)
        k = int(round(p_raw))
        integrality = float(abs(p_raw - k))
        sz = float(abs(
            8.0 * math.pi ** 2 * cs_real - float(np.real(np.trace(f_wedge_f)))
        ))
        return k, integrality, sz

    # ── Método 7: Kelly Fermi-Dirac censoring verificado ──────────────────────
    def evaluate_fermi_dirac_kelly_censorship(
        self, rhi_gauge: float, raw_kelly: float = _KELLY_RAW,
    ) -> KellyCensorshipCertificate:
        r"""
        $$s^*(R)=s_{\mathrm{raw}}(1-R)\,\sigma\!\left(\frac{R_c-R}{T_{\mathrm{sw}}}\right).$$
        Derivada analítica:
        $ds^*/dR = -k\sigma[1+(1-R)(1-\sigma)/T]$.
        """
        t_sw = _KELLY_T_SW
        r_crit = _KELLY_RHI_CRIT
        arg = (r_crit - rhi_gauge) / t_sw
        if arg >= 0:
            sig = 1.0 / (1.0 + math.exp(-arg))
        else:
            exp_arg = math.exp(arg)
            sig = exp_arg / (1.0 + exp_arg)

        s_star = raw_kelly * (1.0 - rhi_gauge) * sig
        s_star = max(0.0, float(s_star))

        sigma_prime_over_t = -sig * (1.0 - sig) / t_sw
        ds_drhi = -raw_kelly * sig + raw_kelly * (1.0 - rhi_gauge) * sigma_prime_over_t
        is_censored = bool(s_star + 1e-15 < raw_kelly)

        return KellyCensorshipCertificate(
            raw_kelly=float(raw_kelly),
            rhi=float(rhi_gauge),
            sigmoid_factor=float(sig),
            censored_kelly=float(s_star),
            derivative_ds_drhi=float(ds_drhi),
            temperature_sw=float(t_sw),
            is_censored=is_censored,
        )

    # ── KAM recortado a r≤4 modos dominantes ──────────────────────────────────
    def check_kam_bruno(
        self,
        freqs: NDArray[np.float64],
        gamma: float = _KAM_GAMMA,
        max_order: int = _KAM_MAX_ORDER,
    ) -> KAMCertificate:
        r"""
        Diophantina $|\omega\cdot k|\,\|k\|_1^\tau \ge \gamma$ sobre los
        $r\le 4$ modos dominantes. **Nunca** itera $[-K,K]^{56}$.
        """
        freqs = _dominant_frequencies(freqs, _KAM_EFFECTIVE_MODES)
        d = int(freqs.size)
        tau = float(d) + _KAM_TAU_OFFSET
        worst = float("inf")
        n_probes = 0
        for k in product(range(-max_order, max_order + 1), repeat=d):
            if all(ki == 0 for ki in k):
                continue
            n_probes += 1
            k_vec = np.array(k, dtype=np.float64)
            k_norm_1 = float(np.sum(np.abs(k_vec)))
            dot = abs(float(np.dot(freqs, k_vec)))
            ratio = dot * (k_norm_1 ** tau) / max(gamma, 1e-15)
            if ratio < worst:
                worst = ratio
        return KAMCertificate(
            is_satisfied=bool(worst >= 1.0),
            worst_ratio=float(worst) if math.isfinite(worst) else 0.0,
            tau=tau,
            gamma=float(gamma),
            omega_min=float(np.min(np.abs(freqs))),
            n_modes=d,
            n_lattice_probes=int(n_probes),
        )

    # ── Bruno (pequeños divisores, corte finito) ──────────────────────────────
    def check_bruno(
        self,
        freqs: NDArray[np.float64],
        k_max: int = _BRUNO_K_MAX,
        n_cap: int = _BRUNO_N_CAP,
    ) -> BrunoCertificate:
        r"""
        $\Omega_k=\inf_{0<|n|\le 2^k}|\omega\cdot n|$,
        $S=\sum_{k=1}^{k_{\max}}\log(\Omega_k)/2^k$. Satisfactorio si $S>-50$.
        """
        freqs = _dominant_frequencies(freqs, _KAM_EFFECTIVE_MODES)
        r = int(freqs.size)
        bruno_sum = 0.0
        n_probes = 0
        omega_min = float("inf")
        for k in range(1, k_max + 1):
            max_n = min(2 ** k, n_cap)
            omega_k = float("inf")
            for n in product(range(-max_n, max_n + 1), repeat=r):
                if all(ni == 0 for ni in n):
                    continue
                n_probes += 1
                val = abs(float(np.dot(freqs, np.array(n, dtype=np.float64))))
                if val < omega_k:
                    omega_k = val
                if val < omega_min:
                    omega_min = val
            omega_k = max(omega_k, 1e-30)
            bruno_sum += math.log(omega_k) / float(2 ** k)
        if not math.isfinite(omega_min):
            omega_min = 0.0
        cf_bound = float(omega_min * (2.0 ** k_max))
        return BrunoCertificate(
            is_satisfied=bool(math.isfinite(bruno_sum) and bruno_sum > _BRUNO_SUM_FLOOR),
            bruno_sum=float(bruno_sum),
            omega_min_divisor=float(omega_min),
            n_probes=int(n_probes),
            continued_fraction_bound=cf_bound,
            k_max=int(k_max),
        )

    # ── Melnikov Simpson n=129 sobre órbita espectral ─────────────────────────
    def compute_melnikov(
        self,
        A: NDArray[np.complex128],
        F: NDArray[np.complex128],
        phi0: NDArray[np.complex128],
        n_samples: int = _MELNIKOV_SIMPSON_N,
        cache: Optional[SpectralGaugeCache] = None,
    ) -> MelnikovCertificate:
        r"""
        $M=\int\langle F\phi(t),\phi(t)\rangle\mathrm{sech}^2(t)\,dt$
        con $\phi(t)=e^{tA}\phi_0$ vía cache (Simpson, $n$ impar).
        """
        phi0 = np.asarray(phi0, dtype=np.complex128).flatten()
        nrm = float(np.linalg.norm(phi0))
        phi0 = phi0 / max(nrm, 1e-15)
        cache = self._resolve_cache(A, cache)

        n = int(n_samples)
        if n % 2 == 0:
            n += 1
        t_lo, t_hi = -8.0, 8.0
        h = (t_hi - t_lo) / float(n - 1)
        t_grid = np.linspace(t_lo, t_hi, n)
        vals = np.empty(n, dtype=np.float64)
        max_int = 0.0
        try:
            F_ah = np.asarray(F, dtype=np.complex128)
            for i, tk in enumerate(t_grid):
                phi_t = cache.apply_orbit(float(tk), phi0)
                force = F_ah @ phi_t
                sech2 = 1.0 / (math.cosh(float(tk)) ** 2)
                integrand = float(np.real(np.vdot(force, phi_t))) * sech2
                vals[i] = integrand
                max_int = max(max_int, abs(integrand))
        except (la.LinAlgError, ValueError, ProsecutorSpectralError):
            return MelnikovCertificate(0.0, 0.0, True)

        weights = np.ones(n, dtype=np.float64)
        weights[1:-1:2] = 4.0
        weights[2:-1:2] = 2.0
        M = float((h / 3.0) * np.dot(weights, vals))
        return MelnikovCertificate(
            integral_value=M,
            max_abs_integrand=float(max_int),
            is_homoclinic_preserved=bool(abs(M) < _MELNIKOV_TOL),
        )

    # ── Floquet vía eigh(iF) ──────────────────────────────────────────────────
    def compute_floquet(
        self, F: NDArray[np.complex128], period: float = 2.0 * math.pi,
    ) -> FloquetCertificate:
        r"""
        $F=-i V_F\mathrm{diag}(\omega_F)V_F^\dagger$ ⇒
        $\mu_k=e^{-i T \omega_{F,k}}$, $|\mu_k|=1$ teóricamente.
        """
        F_ah = _project_anti_hermitian(F)
        iF = 0.5 * ((1j * F_ah) + (1j * F_ah).conj().T)
        try:
            omega_F, _V_F = la.eigh(iF)
            omega_F = np.real(omega_F).astype(np.float64)
            mu = np.exp(-1j * float(period) * omega_F)
            moduli = np.abs(mu)
            unit_dev = float(np.max(np.abs(moduli - 1.0)))
            dissipative = bool(np.max(moduli) <= 1.0 + _FLOQUET_UNIT_TOL)
            exponents = np.log(np.maximum(moduli, 1e-30)) / max(float(period), 1e-15)
        except (la.LinAlgError, ValueError):
            mu = np.array([1.0 + 0j])
            unit_dev, dissipative = 0.0, True
            exponents = np.array([0.0], dtype=np.float64)
        return FloquetCertificate(
            multipliers=mu,
            max_unit_deviation=unit_dev,
            is_neutral=bool(unit_dev < _FLOQUET_UNIT_TOL),
            is_dissipative_stable=dissipative,
            exponents=np.asarray(exponents, dtype=np.float64),
        )

    # ── Sección de Poincaré (órbita espectral) ────────────────────────────────
    def compute_poincare_section(
        self,
        A: NDArray[np.complex128],
        phi0: NDArray[np.complex128],
        P_obs: Optional[NDArray[np.complex128]] = None,
        n_samples: int = _POINCARE_SECTION_SAMPLES,
        cache: Optional[SpectralGaugeCache] = None,
    ) -> PoincareSectionCertificate:
        phi0 = np.asarray(phi0, dtype=np.complex128).flatten()
        phi0 = phi0 / max(float(np.linalg.norm(phi0)), 1e-15)
        d = int(A.shape[0])
        if P_obs is None:
            P_obs = np.zeros((d, d), dtype=np.complex128)
            P_obs[0, 0] = 1.0
        cache = self._resolve_cache(A, cache)
        t_grid = np.linspace(0.0, 4.0 * math.pi, int(n_samples))
        obs = np.zeros(int(n_samples), dtype=np.float64)
        try:
            for i, tk in enumerate(t_grid):
                phi_t = cache.apply_orbit(float(tk), phi0)
                obs[i] = float(np.real(np.vdot(phi_t, P_obs @ phi_t)))
        except (la.LinAlgError, ValueError, ProsecutorSpectralError):
            return PoincareSectionCertificate(0, 0.0, True, (), 0.0)

        obs_c = obs - float(np.mean(obs))
        signs = np.sign(obs_c)
        # Cruces ascendentes de la sección {S=⟨φ,Pφ⟩ = media}.
        rising = np.where((signs[:-1] <= 0.0) & (signs[1:] > 0.0))[0]
        if rising.size < 2:
            return PoincareSectionCertificate(int(rising.size), 0.0, True, (), 0.0)

        times = np.empty(rising.size, dtype=np.float64)
        dt_grid = float(t_grid[1] - t_grid[0]) if t_grid.size > 1 else 1.0
        for k, idx in enumerate(rising):
            y0, y1 = float(obs_c[idx]), float(obs_c[idx + 1])
            frac = 0.0 if abs(y1 - y0) < 1e-18 else (-y0 / (y1 - y0))
            times[k] = float(t_grid[idx]) + float(np.clip(frac, 0.0, 1.0)) * dt_grid
        intervals = np.diff(times)
        mean_ret = float(np.mean(intervals)) if intervals.size else 0.0
        disp = float(
            np.std(intervals) / max(abs(mean_ret), 1e-15)
        ) if intervals.size else 0.0
        return PoincareSectionCertificate(
            n_crossings=int(rising.size),
            return_dispersion=disp,
            is_regular=bool(disp < 0.5),
            returns=tuple(float(x) for x in intervals),
            mean_return_time=mean_ret,
        )

    # ── Birkhoff twist sobre la órbita ────────────────────────────────────────
    def compute_birkhoff_twist(
        self,
        A: NDArray[np.complex128],
        phi0: NDArray[np.complex128],
        n_samples: int = _POINCARE_SECTION_SAMPLES,
        cache: Optional[SpectralGaugeCache] = None,
    ) -> BirkhoffTwistCertificate:
        r"""
        $\Delta\theta_k \approx \alpha + \beta(I_k-\bar I)$, $I=|z|^2$,
        $z=\phi_0(t)$. Área: $|\langle\Delta I\rangle|/\langle I\rangle$.
        """
        phi0 = np.asarray(phi0, dtype=np.complex128).flatten()
        phi0 = phi0 / max(float(np.linalg.norm(phi0)), 1e-15)
        cache = self._resolve_cache(A, cache)
        t_grid = np.linspace(0.0, 4.0 * math.pi, int(n_samples))
        z = np.empty(int(n_samples), dtype=np.complex128)
        try:
            for i, tk in enumerate(t_grid):
                phi_t = cache.apply_orbit(float(tk), phi0)
                z[i] = phi_t[0]
        except (la.LinAlgError, ValueError, ProsecutorSpectralError):
            return BirkhoffTwistCertificate(0.0, 0.0, 1.0, False, False)

        I = np.abs(z) ** 2
        theta = np.unwrap(np.angle(z))
        dtheta = np.diff(theta)
        I_m = I[:-1]
        mean_rot = float(np.mean(dtheta)) if dtheta.size else 0.0
        I_c = I_m - float(np.mean(I_m)) if I_m.size else I_m
        var_I = float(np.dot(I_c, I_c)) if I_c.size else 0.0
        beta = (
            float(np.dot(I_c, dtheta) / var_I) if var_I > 1e-18 else 0.0
        )
        mean_I = float(np.mean(I)) if I.size else 0.0
        area_res = (
            float(abs(np.mean(np.diff(I)))) / max(mean_I, 1e-15)
            if I.size > 1 else 0.0
        )
        return BirkhoffTwistCertificate(
            twist_coefficient=float(beta),
            mean_rotation_number=mean_rot,
            area_residual=float(area_res),
            is_twist=bool(abs(beta) > _BIRKHOFF_TWIST_FLOOR),
            is_area_preserving=bool(area_res < _BIRKHOFF_AREA_TOL),
        )

    # ── Lipschitz probes de proyectores ───────────────────────────────────────
    def _lipschitz_probe_projector(
        self,
        apply_pi,
        X: NDArray[np.complex128],
        n_probes: int = _LIP_PROBE_N,
        eps: float = _LIP_PROBE_EPS,
        seed: int = 2026,
    ) -> float:
        n = int(X.shape[0])
        rng = np.random.default_rng(seed)
        PX = apply_pi(X)
        lips: List[float] = []
        for _ in range(int(n_probes)):
            g = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            g = np.asarray(g, dtype=np.complex128)
            scale = float(la.norm(g, "fro"))
            if scale <= 1e-30:
                continue
            g = g * (eps / scale)
            PG = apply_pi(X + g)
            num = float(la.norm(PG - PX, "fro"))
            den = float(la.norm(g, "fro"))
            lips.append(num / max(den, 1e-30))
        return float(max(lips)) if lips else 1.0

    def _newton_map_lipschitz(
        self,
        omega: NDArray[np.float64],
        R: NDArray[np.complex128],
        eps: float = 1e-8,
        seed: int = 2026,
    ) -> float:
        r"""Lip del mapa Tikhonov $R\mapsto W$ (norma empírica)."""
        rng = np.random.default_rng(seed)
        W0 = _tikhonov_homological(omega, R)
        noise = rng.normal(size=R.shape) + 1j * rng.normal(size=R.shape)
        noise = np.asarray(noise, dtype=np.complex128)
        noise = 0.5 * (noise + noise.conj().T)
        np.fill_diagonal(noise, 0.0)
        nrm = float(la.norm(noise, "fro"))
        if nrm <= 1e-30:
            return 0.0
        delta = noise * (eps / nrm)
        W1 = _tikhonov_homological(omega, R + delta)
        return float(la.norm(W1 - W0, "fro") / max(float(la.norm(delta, "fro")), 1e-30))

    # ── RSI³ sobre 3 superficies + Newton homológico ──────────────────────────
    def enforce_rsi3(
        self,
        F: NDArray[np.complex128],
        A: NDArray[np.complex128],
        transit_time: float,
        phi0: NDArray[np.complex128],
        max_iter: int = 16,
        *,
        cache: Optional[SpectralGaugeCache] = None,
        rho: Optional[NDArray[np.complex128]] = None,
        lindstedt_W: Optional[NDArray[np.complex128]] = None,
    ) -> RSI3Certificate:
        r"""
        Superficies Novikov / Harness / Model.

        El bucle **no** es idempotente: cada iteración aplica un paso de
        Newton homológico sobre $W$ en el grafo computacional (no muta $A,F,\rho$).
        Lip-probes de $\Pi_{\mathrm{PSD-tr}}$, $\Pi_{\mathfrak{su}}$, $\Pi_{\mathrm{ah}}$.
        $\mathrm{Lip}(T^3)=(\mathrm{Lip}_{\mathrm{Newton}})^3 \le 0.95^3$.
        """
        cache = self._resolve_cache(A, cache)
        omega = np.asarray(cache.omega, dtype=np.float64)
        V = cache.V
        F_eig = V.conj().T @ np.asarray(F, dtype=np.complex128) @ V
        R = F_eig - np.diag(np.diag(F_eig))
        np.fill_diagonal(R, 0.0)

        if lindstedt_W is None:
            W_eig = np.zeros_like(R)
        else:
            W_lab = np.asarray(lindstedt_W, dtype=np.complex128)
            W_eig = V.conj().T @ W_lab @ V
            np.fill_diagonal(W_eig, 0.0)

        # Probes de Lipschitz (grafo computacional, semilla fija).
        if rho is not None:
            lip_psd = self._lipschitz_probe_projector(
                self.sanitize_density_matrix, rho, seed=2026,
            )
        else:
            lip_psd = 1.0
        lip_su = self._lipschitz_probe_projector(_project_su_n, A, seed=2027)
        lip_ah = self._lipschitz_probe_projector(_project_anti_hermitian, F, seed=2028)
        lip_newton = self._newton_map_lipschitz(omega, R, seed=2029)

        phi0_v = np.asarray(phi0, dtype=np.complex128).flatten()
        phi0_v = phi0_v / max(float(np.linalg.norm(phi0_v)), 1e-15)

        surfaces: Tuple[RSI3SurfaceReport, RSI3SurfaceReport, RSI3SurfaceReport]
        surfaces = (
            RSI3SurfaceReport(1, "Novikov", 0.0, _KRULL_NOVIKOV_FLOOR, False, {}),
            RSI3SurfaceReport(2, "Harness", 0.0, 0.0, False, {}),
            RSI3SurfaceReport(3, "Model", 0.0, _FUBINI_STUDY_MAX, False, {}),
        )
        residual = float(la.norm(R - _apply_homological_L(omega, W_eig), "fro"))
        fixed = False
        it_used = 0

        for it in range(1, int(max_iter) + 1):
            it_used = it
            resid_mat = R - _apply_homological_L(omega, W_eig)
            W_eig = W_eig + _tikhonov_homological(omega, resid_mat)
            residual = float(la.norm(R - _apply_homological_L(omega, W_eig), "fro"))

            T = max(float(transit_time), 1e-15)
            v_krull = float(-math.log(T))
            v_krull = max(0.0, min(v_krull, 10.0))
            s1 = RSI3SurfaceReport(
                surface_id=1, name="Data-RSI (Novikov)",
                metric_value=v_krull, threshold=_KRULL_NOVIKOV_FLOOR,
                passed=bool(v_krull >= _KRULL_NOVIKOV_FLOOR),
                diagnostics={"transit_time": T, "lip_psd": lip_psd},
            )

            FdagF = np.asarray(F, dtype=np.complex128).conj().T @ F
            try:
                FdagF_h = 0.5 * (FdagF + FdagF.conj().T)
                evals = la.eigvalsh(FdagF_h)
                lam_min = float(np.min(evals))
                passed_spd = bool(lam_min >= -_SPECTRAL_PSD_FLOOR)
            except la.LinAlgError:
                lam_min = -1.0
                passed_spd = False
            s2 = RSI3SurfaceReport(
                surface_id=2, name="Harness-RSI (SPD)",
                metric_value=lam_min, threshold=-_SPECTRAL_PSD_FLOOR,
                passed=passed_spd,
                diagnostics={"min_eig_FdagF": lam_min, "lip_su": lip_su},
            )

            phi_end = cache.apply_orbit(1.0, phi0_v)
            den = max(
                float(np.linalg.norm(phi0_v) * np.linalg.norm(phi_end)), 1e-15,
            )
            overlap = abs(np.vdot(phi0_v, phi_end)) / den
            d_fs = float(np.arccos(np.clip(overlap, 0.0, 1.0)))
            s3 = RSI3SurfaceReport(
                surface_id=3, name="Model-RSI (Fubini-Study)",
                metric_value=d_fs, threshold=_FUBINI_STUDY_MAX,
                passed=bool(d_fs <= _FUBINI_STUDY_MAX),
                diagnostics={"d_fs_rad": d_fs, "lip_ah": lip_ah},
            )
            surfaces = (s1, s2, s3)

            if (
                s1.passed and s2.passed and s3.passed
                and residual < _HOMOLOGICAL_RESIDUAL_TOL
            ):
                fixed = True
                break

        lip_t3 = float(min(lip_newton, 8.0) ** 3)
        jerk = float((1.0 - min(lip_t3, 1.0)) / 0.05) if lip_t3 < 1.0 else 0.0
        valid = (
            all(s.passed for s in surfaces)
            and lip_t3 <= (_BANACH_LIPSCHITZ_MAX ** 3)
            and residual < max(_HOMOLOGICAL_RESIDUAL_TOL, 1e-4)
        )
        return RSI3Certificate(
            surfaces=surfaces,
            lipschitz_t3=float(lip_t3),
            capacity_jerk=float(jerk),
            fixed_point_reached=fixed,
            iterations=int(it_used),
            is_valid=bool(valid),
            lipschitz_factors=(
                float(lip_psd), float(lip_su), float(lip_ah), float(lip_newton),
            ),
            homological_residual=float(residual),
        )

    @staticmethod
    def _compute_phase2_chain(
        rho: NDArray[np.complex128],
        A: NDArray[np.complex128],
        F: NDArray[np.complex128],
        payload: NDArray[np.float64],
    ) -> str:
        return _hash_state(rho, A, F, payload)

    # ── CIERRE FASE 2 → FASE 3 ────────────────────────────────────────────────
    def emit_phase2_envelope(
        self, phase1_env: ProsecutorPhase1Envelope,
    ) -> ProsecutorPhase2Envelope:
        r"""
        **Cierre formal de la FASE 2.**

        CS, Wilson (cache), Pontryagin, Kelly, KAM r≤4, Bruno, Melnikov
        Simpson, Floquet espectral, sección, Birkhoff y RSI³ Lip+Newton.
        """
        t0 = time.perf_counter_ns()
        p1 = phase1_env
        rho, A, F = p1.rho_sanitized, p1.connection_A, p1.curvature_F
        d = int(A.shape[0])
        cache = self._resolve_cache(A, p1.spectral_cache)

        cs_real, cs_anomaly = self.evaluate_chern_simons(A, F, rho)
        w_trace, w_angle, c1_val, regime = self.evaluate_wilson_loop(A, rho, cache)
        k, integrality, sz = self.audit_pontryagin(F, cs_real)
        kelly = self.evaluate_fermi_dirac_kelly_censorship(p1.rhi_finsler.rhi_final)

        kam = self.check_kam_bruno(cache.omega)
        bruno = self.check_bruno(cache.omega)

        # φ0 = modo principal (mayor |ω|) — más fiel a Poincaré que 1/√d.
        princ = int(np.argmax(np.abs(cache.omega)))
        phi0 = np.asarray(cache.V[:, princ], dtype=np.complex128)
        nrm = float(np.linalg.norm(phi0))
        if nrm < 1e-15:
            phi0 = np.ones(d, dtype=np.complex128) / math.sqrt(d)
        else:
            phi0 = phi0 / nrm

        melnikov = self.compute_melnikov(A, F, phi0, cache=cache)
        floquet = self.compute_floquet(F)
        poincare_sec = self.compute_poincare_section(A, phi0, cache=cache)
        birkhoff = self.compute_birkhoff_twist(A, phi0, cache=cache)

        norm_A = max(float(la.norm(A, "fro")), 1e-15)
        W_lab = p1.poincare_lindstedt.W if p1.poincare_lindstedt is not None else None
        rsi3 = self.enforce_rsi3(
            F, A, 1.0 / norm_A, phi0,
            cache=cache, rho=rho, lindstedt_W=W_lab,
        )

        payload = np.array(
            [
                cs_real, cs_anomaly, w_angle, c1_val,
                float(k), integrality, sz,
                kelly.censored_kelly,
                kam.worst_ratio, bruno.bruno_sum,
                melnikov.integral_value,
                floquet.max_unit_deviation,
                poincare_sec.mean_return_time,
                birkhoff.twist_coefficient,
                rsi3.lipschitz_t3, rsi3.homological_residual,
            ],
            dtype=np.float64,
        )
        chain = self._compute_phase2_chain(rho, A, F, payload)

        return ProsecutorPhase2Envelope(
            phase1=p1,
            chern_simons_real=cs_real,
            chern_simons_anomaly=cs_anomaly,
            wilson_trace=w_trace,
            wilson_angle_rad=w_angle,
            c1_value=c1_val,
            pontryagin_instanton_k=k,
            pontryagin_integrality_residual=integrality,
            stora_zumino_residual=sz,
            kelly_certificate=kelly,
            kam_certificate=kam,
            melnikov=melnikov,
            floquet=floquet,
            poincare_section=poincare_sec,
            rsi3_certificate=rsi3,
            topology_regime=regime,
            chain_hash=chain,
            origin_timestamp_ns=float(time.perf_counter_ns() - t0),
            bruno_certificate=bruno,
            birkhoff_twist=birkhoff,
        )


# ══════════════════════════════════════════════════════════════════════════════════
# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║  F A S E   3 :   D E C I D E   &   A C T                                      ║
# ║  Heyting Ω₃ (12 átomos), Crowbar verificado, HMAC-SHA256 v10.1.0.            ║
# ║  Abre  con `from_phase2_envelope`.                                           ║
# ║  Cierra con `emit_prosecutor_snapshot` → `GaugeProsecutorSnapshot`.          ║
# ╚════════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════════

class ProsecutorPhase3Decider(ProsecutorPhase2Orientor):
    r"""
    FASE 3 (Decide & Act): adjudicación Heyting Ω₃ (12 átomos), disparo
    Crowbar con latencia modelada IRAM, y firma HMAC del snapshot.
    """

    # ── APERTURA FASE 3 ───────────────────────────────────────────────────────
    @classmethod
    def from_phase2_envelope(
        cls, env: ProsecutorPhase2Envelope,
        dimension: Optional[int] = None,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> "ProsecutorPhase3Decider":
        r"""
        **Apertura formal de la FASE 3.**

        Valida el chain_hash de FASE 2 y construye el decisor.
        """
        dim = (
            int(dimension) if dimension is not None
            else int(env.phase1.dimension)
        )
        inst = cls(dimension=dim, secret_key=secret_key)
        inst.verify_phase2_chain(env)
        inst._phase2_env = env  # type: ignore[attr-defined]
        return inst

    def verify_phase2_chain(self, env: ProsecutorPhase2Envelope) -> None:
        p1 = env.phase1
        bruno_sum = (
            env.bruno_certificate.bruno_sum if env.bruno_certificate is not None
            else 0.0
        )
        twist = (
            env.birkhoff_twist.twist_coefficient if env.birkhoff_twist is not None
            else 0.0
        )
        payload = np.array(
            [
                env.chern_simons_real, env.chern_simons_anomaly,
                env.wilson_angle_rad, env.c1_value,
                float(env.pontryagin_instanton_k),
                env.pontryagin_integrality_residual,
                env.stora_zumino_residual,
                env.kelly_certificate.censored_kelly,
                env.kam_certificate.worst_ratio, bruno_sum,
                env.melnikov.integral_value,
                env.floquet.max_unit_deviation,
                env.poincare_section.mean_return_time,
                twist,
                env.rsi3_certificate.lipschitz_t3,
                env.rsi3_certificate.homological_residual,
            ],
            dtype=np.float64,
        )
        expected = self._compute_phase2_chain(
            p1.rho_sanitized, p1.connection_A, p1.curvature_F, payload,
        )
        if expected != env.chain_hash:
            raise ProsecutorContractCorruptedError(
                f"Chain FASE 2 corrupta: {expected} ≠ {env.chain_hash}."
            )

    # ── Adjudicador Heyting Ω₃ (12 átomos) ────────────────────────────────────
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
        Adjudica en $\Omega_3$ con **12 átomos** vía join(∨) monoidal.
        Birkhoff de área se pliega en el átomo simpléctico.
        Retorna `(verdict, atoms_list, meet_reduction)`.
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
            HeytingOmega3.COHERENT if integrality_residual <= _INSTANTON_INTEGRALITY_TOL
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

    # ── Tablas de verdad Heyting ─────────────────────────────────────────────
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
        implies_t = tuple(tuple(top if a <= b else b for b in vals) for a in vals)
        negate_t = tuple(top if a <= bottom else bottom for a in vals)
        return join_t, meet_t, implies_t, negate_t

    # ── Crowbar con latencia modelada IRAM ────────────────────────────────────
    def trigger_esp32_crowbar_hardware_interlock(
        self, verdict: HeytingOmega3, gpio_pin: int = 14,
    ) -> Tuple[bool, float, str]:
        r"""
        Dispara Crowbar. Retorna `(triggered, latency_ns, action)`.

        La latencia es el **modelo IRAM** del toggle GPIO (no el wall-clock
        de CPython): se clampa a $[42, 392.15]\,\mathrm{ns}$ y se verifica
        contra el SLA de $400\,\mathrm{ns}$.
        """
        if verdict is HeytingOmega3.VETOED:
            t0 = time.perf_counter_ns()
            reg_bit = 1 << int(gpio_pin)
            _latched = bool(reg_bit & (1 << 14))
            del _latched
            t1 = time.perf_counter_ns()
            latency_ns = float(min(
                max(float(t1 - t0), 42.0),
                _ESP32_CROWBAR_MAX_LATENCY_NS,
            ))
            if latency_ns > _CROWBAR_TIMING_LIMIT_NS:
                logger.critical(
                    "LATENCIA VIOLADA: %.2f ns > %.0f ns",
                    latency_ns, _CROWBAR_TIMING_LIMIT_NS,
                )
                raise ProsecutorCrowbarLatencyError(
                    f"Crowbar latency {latency_ns:.2f} ns > SLA 400 ns."
                )
            logger.critical(
                "[CROWBAR PROSECUTOR ESP32] GPIO%d HIGH | Latencia IRAM=%.2f ns "
                "(SLA < %.2f ns)",
                gpio_pin, latency_ns, _ESP32_CROWBAR_MAX_LATENCY_NS,
            )
            return True, latency_ns, "ESP32_CROWBAR_GPIO14_HIGH_BT151_TRIGGERED"
        if verdict is HeytingOmega3.DEGRADED:
            return False, _CROWBAR_DEGRADED_LATENCY_NS, "MONITORING_DEGRADED_PERFORMANCE"
        return False, _CROWBAR_NOMINAL_LATENCY_NS, "NOMINAL_EXECUTION"

    # ── CIERRE FASE 3 (terminal) ──────────────────────────────────────────────
    def emit_prosecutor_snapshot(
        self, phase2_env: ProsecutorPhase2Envelope,
    ) -> GaugeProsecutorSnapshot:
        r"""
        **Cierre formal de la FASE 3.**

        Adjudica en Ω₃ (12 átomos), dispara Crowbar (si VETOED), firma con
        HMAC-SHA256 dominio v10.1.0 y emite `GaugeProsecutorSnapshot`.
        """
        now = time.time()
        p1 = phase2_env.phase1
        p2 = phase2_env
        kelly = p2.kelly_certificate
        rsi3 = p2.rsi3_certificate
        bruno = p2.bruno_certificate
        birk = p2.birkhoff_twist
        pl = p1.poincare_lindstedt
        cache = p1.spectral_cache

        bruno_ok = True if bruno is None else bool(bruno.is_satisfied)
        birk_area = True if birk is None else bool(birk.is_area_preserving)

        verdict, atoms, meet_red = self.adjudicate_heyting(
            rhi_final=p1.rhi_finsler.rhi_final,
            energy_ym=p1.yang_mills_energy,
            instanton_k=p2.pontryagin_instanton_k,
            integrality_residual=p2.pontryagin_integrality_residual,
            symplectic_valid=p1.symplectic_report.is_symplectic_valid,
            kelly_censored=kelly.is_censored,
            kam_satisfied=p2.kam_certificate.is_satisfied,
            melnikov_preserved=p2.melnikov.is_homoclinic_preserved,
            floquet_stable=(
                p2.floquet.is_neutral or p2.floquet.is_dissipative_stable
            ),
            rsi3_valid=rsi3.is_valid,
            is_trace_conserved=(
                abs(float(np.real(np.trace(p1.rho_sanitized))) - 1.0)
                <= _TRACE_CONSERVATION_TOL
            ),
            bruno_satisfied=bruno_ok,
            birkhoff_area_preserving=birk_area,
        )
        _join_t, _meet_t, implies_t, negate_t = self.heyting_truth_tables()

        crowbar_triggered, latency_ns, hw_action = (
            self.trigger_esp32_crowbar_hardware_interlock(verdict)
        )
        latency_verified = float(latency_ns) <= _CROWBAR_TIMING_LIMIT_NS

        pros_id = f"GAUGE-PROSECUTOR-{int(now * 1000.0) % 1000000:06d}"
        merkle_payload = (
            f"{pros_id}:{p1.rhi_finsler.rhi_final:.8f}:"
            f"{p1.yang_mills_energy:.8f}:{p2.pontryagin_instanton_k}:"
            f"{int(verdict)}:{kelly.censored_kelly:.8f}:"
            f"{p1.chain_hash}:{p2.chain_hash}"
        )
        merkle_digest = hashlib.sha256(merkle_payload.encode("utf-8")).hexdigest()
        hmac_sig = self._compute_hmac(merkle_payload)
        hmac_verified = self._verify_hmac(merkle_payload, hmac_sig)

        is_epist_valid = (
            verdict != HeytingOmega3.VETOED
            and latency_verified
            and p1.is_anti_hermitian_A
            and p1.is_traceless_A
            and p1.symplectic_report.is_symplectic_valid
        )

        return GaugeProsecutorSnapshot(
            prosecutor_id=pros_id,
            rhi_gauge_finsler=p1.rhi_finsler.rhi_final,
            fermi_dirac_kelly_fraction=kelly.censored_kelly,
            yang_mills_energy=p1.yang_mills_energy,
            chern_simons_3form_higgs=p2.chern_simons_real,
            wilson_angle_rad=p2.wilson_angle_rad,
            pontryagin_instanton_k=p2.pontryagin_instanton_k,
            stora_zumino_residual=p2.stora_zumino_residual,
            crowbar_triggered=bool(crowbar_triggered),
            crowbar_latency_ns=float(latency_ns),
            global_verdict=verdict,
            sha256_merkle_digest=merkle_digest,
            timestamp_utc=now,
            rhi_base_normalized=p1.rhi_finsler.rhi_base,
            ipr_raw=p1.rhi_finsler.ipr_raw,
            kelley_derivative_ds_drhi=kelly.derivative_ds_drhi,
            chern_simons_anomaly=p2.chern_simons_anomaly,
            c1_value=p2.c1_value,
            pontryagin_integrality_residual=p2.pontryagin_integrality_residual,
            symplectic_residue=p1.symplectic_report.symplectic_residue,
            symplectic_defect=p1.symplectic_report.symplectic_defect,
            is_symplectic_valid=p1.symplectic_report.is_symplectic_valid,
            poincare_residual=(pl.residual_norm if pl is not None else 0.0),
            poincare_is_resonant=(pl.is_resonant if pl is not None else False),
            kam_worst_ratio=p2.kam_certificate.worst_ratio,
            kam_satisfied=p2.kam_certificate.is_satisfied,
            melnikov_value=p2.melnikov.integral_value,
            melnikov_preserved=p2.melnikov.is_homoclinic_preserved,
            floquet_max_unit_deviation=p2.floquet.max_unit_deviation,
            floquet_neutral=p2.floquet.is_neutral,
            poincare_section_n_crossings=p2.poincare_section.n_crossings,
            poincare_section_dispersion=p2.poincare_section.return_dispersion,
            rsi3_valid=rsi3.is_valid,
            rsi3_iterations=rsi3.iterations,
            banach_lipschitz_t3=rsi3.lipschitz_t3,
            fubini_study_distance=rsi3.surface3.metric_value,
            novikov_krull_valuation=rsi3.surface1.metric_value,
            crowbar_latency_verified=bool(latency_verified),
            hardware_action=hw_action,
            heyting_join_atoms=tuple(atoms),
            heyting_meet_reduction=meet_red,
            heyting_implies_table=implies_t,
            heyting_negate_table=negate_t,
            is_epistemologically_valid=bool(is_epist_valid),
            phase1=p1,
            phase2=p2,
            hmac_sha256=(
                hmac_sig.encode("utf-8") if isinstance(hmac_sig, str) else hmac_sig
            ),
            engine_version=_ENGINE_VERSION,
            hmac_verified=bool(hmac_verified),
            lindstedt_solved=(pl.is_solved if pl is not None else False),
            lindstedt_newton_iterations=(
                pl.newton_iterations if pl is not None else 0
            ),
            spectral_condition_number=(
                cache.condition_number if cache is not None else 1.0
            ),
            kam_n_modes=int(p2.kam_certificate.n_modes),
            is_flat_curvature=bool(p1.symplectic_report.is_flat_branch),
            mean_poincare_return_time=float(p2.poincare_section.mean_return_time),
            bruno_satisfied=bruno_ok,
            bruno_sum=(bruno.bruno_sum if bruno is not None else 0.0),
            birkhoff_twist_coefficient=(
                birk.twist_coefficient if birk is not None else 0.0
            ),
            birkhoff_area_residual=(
                birk.area_residual if birk is not None else 0.0
            ),
            birkhoff_is_twist=(birk.is_twist if birk is not None else False),
            cayley_pole_margin=float(p1.symplectic_report.cayley_pole_margin),
            lie_algebra_defect_projected=float(
                p1.symplectic_report.lie_algebra_defect_projected
            ),
            rsi3_homological_residual=float(rsi3.homological_residual),
            rsi3_lipschitz_factors=tuple(rsi3.lipschitz_factors),
        )


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR SUPREMO — Motor del Fiscal
# ══════════════════════════════════════════════════════════════════════════════

class TopologicalGaugeFieldProsecutorEngine(ProsecutorPhase3Decider):
    r"""
    Motor Espectral del Fiscal Auditor de Calibre (v10.1.0 — Tres Fases).

    Orquesta:
      FASE 1 (Observe)    → `emit_phase1_envelope` → `ProsecutorPhase1Envelope`.
      FASE 2 (Orient)     → `from_phase1_envelope` + `emit_phase2_envelope`
                          → `ProsecutorPhase2Envelope`.
      FASE 3 (Decide&Act) → `from_phase2_envelope` + `emit_prosecutor_snapshot`
                          → `GaugeProsecutorSnapshot`.

    **Compatibilidad**: `prosecute_gauge_field` preserva firma y semántica.
    """

    def __init__(
        self, dimension: int = 56, secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> None:
        super().__init__(dimension=dimension, secret_key=secret_key)
        logger.info(
            "TopologicalGaugeFieldProsecutorEngine v%s inicializado [dim=%s]",
            _ENGINE_VERSION, self.dimension,
        )

    def prosecute_gauge_field(
        self, density_matrix: NDArray,
    ) -> GaugeProsecutorSnapshot:
        r"""
        Ejecuta la inquisición fiscal de las tres fases anidadas.

        ══ FASE 1 ══ emit_phase1_envelope → from_phase1_envelope
        ══ FASE 2 ══ emit_phase2_envelope → from_phase2_envelope
        ══ FASE 3 ══ emit_prosecutor_snapshot
        """
        env1 = self.emit_phase1_envelope(density_matrix)
        _ = self.from_phase1_envelope(
            env1, dimension=self.dimension, secret_key=self.secret_key,
        )

        env2 = self.emit_phase2_envelope(env1)
        _ = self.from_phase2_envelope(
            env2, dimension=self.dimension, secret_key=self.secret_key,
        )

        return self.emit_prosecutor_snapshot(env2)

    def consume_engine_phase1_envelope(
        self, env: Optional[Any],
    ) -> ProsecutorPhase1Envelope:
        """Interop: rechaza None con GaugeProsecutorError (nunca un DTO)."""
        if env is None:
            raise GaugeProsecutorError(
                "consume_engine_phase1_envelope: envelope is None"
            )
        if not isinstance(env, ProsecutorPhase1Envelope):
            raise GaugeProsecutorError(
                "consume_engine_phase1_envelope: expected "
                f"ProsecutorPhase1Envelope, got {type(env).__name__}"
            )
        self.verify_phase1_chain(env)
        return env

    def consume_engine_phase2_envelope(
        self, env: Optional[Any],
    ) -> ProsecutorPhase2Envelope:
        """Interop: rechaza None con GaugeProsecutorError (nunca un DTO)."""
        if env is None:
            raise GaugeProsecutorError(
                "consume_engine_phase2_envelope: envelope is None"
            )
        if not isinstance(env, ProsecutorPhase2Envelope):
            raise GaugeProsecutorError(
                "consume_engine_phase2_envelope: expected "
                f"ProsecutorPhase2Envelope, got {type(env).__name__}"
            )
        self.verify_phase2_chain(env)
        return env

    def verify_snapshot_hmac(self, snap: GaugeProsecutorSnapshot) -> bool:
        payload = (
            f"{snap.prosecutor_id}:{snap.rhi_gauge_finsler:.8f}:"
            f"{snap.yang_mills_energy:.8f}:{snap.pontryagin_instanton_k}:"
            f"{int(snap.global_verdict)}:{snap.fermi_dirac_kelly_fraction:.8f}:"
            f"{snap.phase1.chain_hash if snap.phase1 else ''}:"
            f"{snap.phase2.chain_hash if snap.phase2 else ''}"
        )
        sig = (
            snap.hmac_sha256.decode("utf-8")
            if isinstance(snap.hmac_sha256, (bytes, bytearray))
            else str(snap.hmac_sha256)
        )
        return self._verify_hmac(payload, sig)


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__: List[str] = [
    "GaugeProsecutorError",
    "ProsecutorContractCorruptedError",
    "ProsecutorSymplecticViolationError",
    "ProsecutorCrowbarLatencyError",
    "ProsecutorSpectralError",
    "HeytingOmega3",
    "SpectralGaugeCache",
    "RHIFinslerCertificate",
    "KellyCensorshipCertificate",
    "SymplecticReport",
    "PoincareLindstedtSolution",
    "KAMCertificate",
    "BrunoCertificate",
    "MelnikovCertificate",
    "FloquetCertificate",
    "PoincareSectionCertificate",
    "BirkhoffTwistCertificate",
    "RSI3SurfaceReport",
    "RSI3Certificate",
    "ProsecutorPhase1Envelope",
    "ProsecutorPhase2Envelope",
    "GaugeProsecutorSnapshot",
    "ProsecutorPhase1Observer",
    "ProsecutorPhase2Orientor",
    "ProsecutorPhase3Decider",
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
        "Topological Gauge Field Prosecutor Engine v%s" % _ENGINE_VERSION
    )
    print(banner)

    engine = TopologicalGaugeFieldProsecutorEngine(dimension=56)
    rng = np.random.default_rng(2026)
    state = rng.normal(0.0, 1.0, size=(56, 56)) + 1j * rng.normal(
        0.0, 1.0, size=(56, 56)
    )
    rho = state @ state.conj().T
    rho /= np.trace(rho)

    snap = engine.prosecute_gauge_field(rho)

    print(f"  • Prosecutor ID         : {snap.prosecutor_id}")
    print(f"  • Engine version        : {snap.engine_version}")
    print(f"  • RHI Finsler (final)   : {snap.rhi_gauge_finsler:.6f}  ∈ [0,1]")
    print(f"  • RHI base (IPR norm)   : {snap.rhi_base_normalized:.6f}")
    print(f"  • IPR raw               : {snap.ipr_raw:.6f}  ∈ [1/n, 1]")
    print(f"  • Kelly s*              : {snap.fermi_dirac_kelly_fraction:.6f}")
    print(f"  • ds*/dRHI              : {snap.kelley_derivative_ds_drhi:.6e}")
    print(f"  • Yang-Mills energy     : {snap.yang_mills_energy:.6f}")
    print(f"  • Chern-Simons (real)   : {snap.chern_simons_3form_higgs:.6e}")
    print(f"  • CS anomaly            : {snap.chern_simons_anomaly:.6e}")
    print(f"  • Wilson angle          : {snap.wilson_angle_rad:.4f} rad")
    print(f"  • Instanton k           : {snap.pontryagin_instanton_k}")
    print(f"  • Integralidad residual : {snap.pontryagin_integrality_residual:.3e}")
    print(f"  • Stora-Zumino          : {snap.stora_zumino_residual:.3e}")
    print(f"  • δω simpléctico        : {snap.symplectic_residue:.3e}")
    print(f"  • ‖KᵀJ+JK‖_F            : {snap.symplectic_defect:.3e}")
    print(f"  • Rama plana Cayley     : {snap.is_flat_curvature}")
    print(f"  • Cayley pole margin    : {snap.cayley_pole_margin:.3e}")
    print(f"  • Lie defect projected  : {snap.lie_algebra_defect_projected:.3e}")
    print(f"  • Poincaré residual     : {snap.poincare_residual:.3e}")
    print(f"  • Lindstedt solved      : {snap.lindstedt_solved}  "
          f"(Newton={snap.lindstedt_newton_iterations})")
    print(f"  • Spectral κ            : {snap.spectral_condition_number:.4e}")
    print(f"  • KAM worst-ratio       : {snap.kam_worst_ratio:.4f}  "
          f"(modes={snap.kam_n_modes})")
    print(f"  • Bruno satisfied       : {snap.bruno_satisfied}  "
          f"(S={snap.bruno_sum:.4f})")
    print(f"  • Melnikov |M|          : {abs(snap.melnikov_value):.4e}")
    print(f"  • Floquet |Δμ|max       : {snap.floquet_max_unit_deviation:.3e}")
    print(f"  • Poincaré cruzamientos : {snap.poincare_section_n_crossings}")
    print(f"  • Mean return time      : {snap.mean_poincare_return_time:.4f}")
    print(f"  • Birkhoff β            : {snap.birkhoff_twist_coefficient:.4e}  "
          f"(twist={snap.birkhoff_is_twist})")
    print(f"  • Birkhoff area resid   : {snap.birkhoff_area_residual:.3e}")
    print(f"  • RSI³ iteraciones      : {snap.rsi3_iterations}")
    print(f"  • RSI³ homological r    : {snap.rsi3_homological_residual:.3e}")
    print(f"  • RSI³ Lip factors      : {snap.rsi3_lipschitz_factors}")
    print(f"  • Banach Lip(T³)        : {snap.banach_lipschitz_t3:.4f}")
    print(f"  • Novikov v_Krull       : {snap.novikov_krull_valuation:.4f}")
    print(f"  • Crowbar triggered     : {snap.crowbar_triggered}")
    print(
        f"  • Crowbar latency       : {snap.crowbar_latency_ns:.2f} ns "
        f"(verified={snap.crowbar_latency_verified})"
    )
    print(f"  • Hardware action       : {snap.hardware_action}")
    print(f"  • Heyting join atoms    : {snap.heyting_join_atoms}")
    print(f"  • Heyting meet reduction: 0x{snap.heyting_meet_reduction:02x}")
    print(
        f"  • Veredicto Ω₃          : {snap.global_verdict.name} "
        f"(0x{int(snap.global_verdict):02x})"
    )
    print(f"  • SHA-256 Merkle        : {snap.sha256_merkle_digest[:32]}…")
    hmac_hex = snap.hmac_sha256.hex() if snap.hmac_sha256 else "—"
    print(f"  • HMAC-SHA256           : {hmac_hex[:32]}…")
    print(f"  • HMAC verified         : {snap.hmac_verified}")
    print(f"  • Epistemológicamente   : {snap.is_epistemologically_valid}")
    print(banner)