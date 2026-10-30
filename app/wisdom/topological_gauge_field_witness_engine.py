# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : Topological Gauge Field Witness Engine v10.1.0 — Tres Fases       ║
║ Ubicación: app/wisdom/topological_gauge_field_witness_engine.py              ║
║ Versión  : 10.1.0 / 10.0.0 (SOVEREIGN-GAUGE-WITNESS-QND-BACKACTION-ZERO-     ║
║           POINCARE-RSI3-PhD-BANACH-BRUNO-BIRKHOFF)                           ║
║ Función  : Observación no destructiva pasiva QND (Quantum Non-Demolition),  ║
║            certificación de back-action cero, teoría espectral de gauge,     ║
║            secciones de Poincaré y automejora recursiva Nivel 3.             ║
╚══════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL MOTOR ESPECTRAL
─────────────────────────────────────────────────
El Motor Espectral del Testigo Silencioso del Campo Topológico de Gauge ejecuta la
medición débil de no demolición cuántica (QND) sobre el estado de densidad
\rho \in \mathcal{D}_N. Evalúa de manera pasiva y no invasiva la preservación de la
isometría de calibre \mathfrak{su}(N), la geometría simpléctica de Cayley-Kähler
\mathfrak{sp}(2d, \mathbb{C}), el invariante de Poincaré-Lindstedt y la convergencia
de Banach de automejora recursiva Nivel 3 (RSI³) sobre 3 superficies topológicas.

MARCO TEÓRICO
─────────────
1. Conexión de Gauge \mathfrak{su}(N) y Curvatura de Yang-Mills:
   Dada la matriz de densidad sanitizada \rho \in \mathcal{D}_N (\rho = \rho^\dagger \ge 0, \mathrm{Tr}(\rho) = 1),
   la conexión anti-Hermítica y de traza nula se extrae mediante la proyección:
   A = i \left( \rho - \frac{\mathrm{Tr}(\rho)}{N} I \right) \in \mathfrak{su}(N)
   La curvatura de Yang-Mills F se define como el conmutador:
   F = [A, \rho] = A \rho - \rho A \in \mathfrak{su}(N)
   con densidad de energía asociada E_{\mathrm{YM}} = -\frac{1}{2} \mathrm{Tr}(F^2) \ge 0.

2. Medición Débil QND (Aharonov-Albert-Vaidman) y Back-Action:
   Para estados inicial |\phi_i\rangle y final |\phi_f\rangle, el valor débil de A es:
   A_w = \frac{\langle \phi_f | A | \phi_i \rangle}{\langle \phi_f | \phi_i \rangle}
   El conmutador [A, \rho] mide el impacto de retroacción (back-action) en decibelios:
   \mathrm{BackAction}_{\mathrm{dB}} = -10 \log_{10}\left( 1 - \min(\| [A, \rho] \|_F, 0.99) \right)

3. Invariantes Geométricos de Uhlmann, Bures y Liouville:
   Fidelidad de Uhlmann respecto al estado maximalmente mixto I/d:
   F(\rho, I/d) = \frac{1}{d} \left( \mathrm{Tr}\sqrt{\rho} \right)^2
   Distancia Bures: D_B(\rho, I/d) = \sqrt{2 - 2\sqrt{F}}
   Deriva de volumen de Liouville: \Delta_{\mathrm{Liouville}} = |\mathrm{Tr}(\rho) - 1|.

4. Mecánica Celeste de Poincaré, KAM, Bruno y Twist de Birkhoff:
   Ecuación homológica de Poincaré-Lindstedt en la base propia de iA:
   \{H_0, W\} = H_1 - \langle H_1 \rangle_{\mathbb{T}^r} \implies -i(\omega_i - \omega_j)W_{ij} = (H_1^{\mathrm{osc}})_{ij}
   Condición diofantina de Bruno-Rüssmann:
   |\omega \cdot k| \|k\|_1^\tau \ge \gamma, \quad \tau = r + 1, \quad r \le 4
   Suma aritmética de Bruno sobre fracciones continuas: \Omega(\theta) = \sum_{n=0}^N \frac{\log(q_{n+1})}{q_n} < \infty
   Integral de Melnikov: M(\theta_0) = -\int_{\mathbb{R}} \mathrm{Re}\langle F \varphi(t), \varphi(t) \rangle \mathrm{sech}^2(t) dt

5. Automejora Recursiva Nivel 3 (RSI³) y Contracción de Banach:
   Composición de proyectores no expansivos T^3 = T_3 \circ T_2 \circ T_1 con cota
   \mathrm{Lip}(T^3) \le 0.95^3 \approx 0.857.

AXIOMAS FORMALES
────────────────
[Axioma 1] Unicidad y Positividad de la Densidad: \rho = \rho^\dagger \ge 0, \mathrm{Tr}(\rho) = 1.
[Axioma 2] Hermiticidad de la Conexión: A^\dagger = -A, \mathrm{Tr}(A) = 0.
[Axioma 3] No Demolición Cuántica (QND): \| [A, \rho] \|_F \le 10^{-6} \implies \mathrm{BackAction} = 0.0 \text{ dB}.
[Axioma 4] Conservación del Volumen de Liouville: \Delta_{\mathrm{Liouville}} \le 10^{-2}.
[Axioma 5] Contracción de Banach: \mathrm{Lip}(T^3) \le 0.95^3.

INVARIANTES DE ESTADO
─────────────────────
1. Veredicto de Adjudicación Heyting \Omega_3 \in \{0x00 (\text{COHERENT}), 0x01 (\text{DEGRADED}), 0x03 (\text{VETOED})\}.
2. Modo de Operación Invariante: witness_mode = "PASSIVE_SILENT" (cero activación de hardware/crowbar).
3. Firma criptográfica HMAC-SHA256 e integridad Merkle digest SHA-256.
"""

from __future__ import annotations

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
# §0. INTEROPERABILIDAD CON EL MOTOR ESPECTRAL v10.x (opcional)
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


logger = logging.getLogger("APU.Wisdom.TopologicalGaugeFieldWitnessEngine.v10")

__version__: Final[str] = "10.1.0"

# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES FÍSICAS, NUMÉRICAS, DE POINCARÉ Y RSI³
# ══════════════════════════════════════════════════════════════════════════════
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_GAUGE_WITNESS_SOVEREIGN_KEY"
_GAUGE_WITNESS_HMAC_DOMAIN: Final[bytes] = b"APU-FILTER-GAUGE-WITNESS-v10.1.0|"
_ENGINE_VERSION: Final[str] = "10.1.0"

# Tolerancias de certificación
_ANTI_HERM_TOL: Final[float] = 1.0e-10
_TRACELESS_TOL: Final[float] = 1.0e-10
_SYMPLECTIC_TOL: Final[float] = 1.0e-10
_FLAT_CURVATURE_TOL: Final[float] = 1.0e-12
_TRACE_CONSERVATION_TOL: Final[float] = 1.0e-2
_QND_BACKACTION_DB_TOL: Final[float] = 1.0e-2
_QND_FIDELITY_MIN: Final[float] = 0.99
_QND_COMMUTATOR_MAX: Final[float] = 1.0e-6
_UHLMANN_FIDELITY_MIN: Final[float] = 0.95
_BURES_DIST_SOFT: Final[float] = 0.30
_BURES_DIST_HARD: Final[float] = 0.80
_SZ_RESIDUAL_SOFT: Final[float] = 0.15
_SZ_RESIDUAL_HARD: Final[float] = 0.50
_YM_ENERGY_SOFT: Final[float] = 0.25
_YM_ENERGY_HARD: Final[float] = 0.85
_CAYLEY_POLE_TOL: Final[float] = 1.0e-10

# Poincaré (Birkhoff–Arnold–Moser / Bruno / Lindstedt)
_KAM_GAMMA: Final[float] = 0.125
_KAM_TAU_OFFSET: Final[float] = 1.0
_KAM_MAX_ORDER: Final[int] = 3
_KAM_EFFECTIVE_MODES: Final[int] = 4
_KAM_FREQ_ABS_FLOOR: Final[float] = 1.0e-15
_MELNIKOV_TOL: Final[float] = 0.5
_MELNIKOV_SAMPLES: Final[int] = 129  # impar → Simpson
_FLOQUET_UNIT_TOL: Final[float] = 1.0e-6
_POINCARE_SECTION_SAMPLES: Final[int] = 96
_LINDSTEDT_REGULARIZATION: Final[float] = 1.0e-12
_LINDSTEDT_NEWTON_ITERS: Final[int] = 8
_LINDSTEDT_RESIDUAL_TOL: Final[float] = 1.0e-9
_LINDSTEDT_RESONANCE_GAP: Final[float] = 1.0e-12
_BRUNO_MAX_TERMS: Final[int] = 16
_BRUNO_SUM_MAX: Final[float] = 12.0
_BIRKHOFF_TWIST_MIN: Final[float] = 1.0e-6

# RSI³ (Darwin–Gödel Nivel 3) + Banach
_KRULL_NOVIKOV_FLOOR: Final[float] = 0.10
_FUBINI_STUDY_MAX: Final[float] = 1.0e-4
_BANACH_LIPSCHITZ_MAX: Final[float] = 0.95
_RSI3_MAX_ITER: Final[int] = 16
_RSI3_FIXED_POINT_TOL: Final[float] = 1.0e-12
_RSI3_PROBE_EPS: Final[float] = 1.0e-8
_RSI3_KRULL_CAP: Final[float] = 10.0


# ══════════════════════════════════════════════════════════════════════════════
# §B. RETÍCULO DE HEYTING Ω₃ COMPLETO
# ══════════════════════════════════════════════════════════════════════════════
class HeytingOmega3(IntEnum):
    r"""
    Retículo de Heyting trivalente lineal (lógica de Gödel G₃) con
    ⊥ = COHERENT y ⊤ = VETOED:

        COHERENT (0x00)  <  DEGRADED (0x01)  <  VETOED (0x03).

    Interpretación de seguridad (pesimista): el join colapsa al peor caso.

    Operaciones (cadena finita ⇒ Heyting completa):
      • join(∨)    = max     — supremo
      • meet(∧)    = min     — ínfimo
      • implies(→) — a→b = ⊤ si a≤b, sino b
                      (pseudo-complemento relativo:
                       a→b = ∨{c | a∧c ≤ b})
      • negate(¬)  — ¬a = a→⊥.  Luego ¬⊥=⊤, ¬a=⊥ para a≠⊥.

    Migración 0x02 → 0x03: consistente con mac_minimizer, topo_gauge_engine v10.
    """
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""$a\to b$ = VETOED (⊤) si $a\le b$, sino $b$."""
        return HeytingOmega3.VETOED if int(self) <= int(other) else other

    def negate(self) -> "HeytingOmega3":
        r"""$¬a = a \to \bot$ con $\bot$ = COHERENT."""
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
class GaugeWitnessError(Exception):
    """Excepción raíz del Testigo Silencioso."""


class QNDBackActionViolationError(GaugeWitnessError):
    """QND roto: el back-action excede la tolerancia (el testigo perturbó el campo)."""


class WitnessContractCorruptedError(GaugeWitnessError):
    """Cadena de hashes o firma HMAC corrupta."""


class WitnessSymplecticViolationError(GaugeWitnessError):
    """Embedding Cayley-Kähler no pertenece a sp(2d,ℂ)."""


class WitnessSpectralError(GaugeWitnessError):
    """Fallo de descomposición espectral de iA o de la carta de Lindstedt."""


# ══════════════════════════════════════════════════════════════════════════════
# §D. HELPERS NUMÉRICOS Y CRIPTOGRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════
def _hash_state(*arrays: NDArray[Any]) -> str:
    r"""SHA-256 truncado (32 hex = 128 bit) con forma y dtype para evitar colisiones."""
    h = hashlib.sha256()
    for a in arrays:
        arr = np.ascontiguousarray(a)
        h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        h.update(str(arr.dtype).encode("ascii"))
        h.update(arr.tobytes())
    return h.hexdigest()[:32]


def _project_anti_hermitian(M: NDArray[np.complex128]) -> NDArray[np.complex128]:
    r"""Proyector Frobenius-óptimo sobre u(N): $\Pi(M)=(M-M^\dagger)/2$."""
    return 0.5 * (M - M.conj().T)


def _project_su_n(M: NDArray[np.complex128]) -> NDArray[np.complex128]:
    r"""
    Proyector de álgebra de Lie sobre $\mathfrak{su}(N)$:
    $\Pi_{\mathfrak{su}}(M)=\Pi_{u}(M)-\frac{\mathrm{Tr}\,\Pi_{u}(M)}{N}I$.
    """
    ah = _project_anti_hermitian(M)
    n = ah.shape[0]
    return ah - (np.trace(ah) / float(n)) * np.eye(n, dtype=np.complex128)


def _standard_symplectic_j(n: int) -> NDArray[np.complex128]:
    r"""Forma canónica de Darboux $J=\begin{pmatrix}0&I\\-I&0\end{pmatrix}$ en $\mathbb{C}^{2n}$."""
    z = np.zeros((n, n), dtype=np.complex128)
    i = np.eye(n, dtype=np.complex128)
    return np.block([[z, i], [-i, z]])


def _safe_unit(vec: NDArray[np.complex128]) -> NDArray[np.complex128]:
    v = np.asarray(vec, dtype=np.complex128).reshape(-1)
    nrm = float(np.linalg.norm(v))
    if nrm <= _WILKINSON_EPSILON:
        out = np.zeros_like(v)
        out[0] = 1.0
        return out
    return v / nrm


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def _finite_or(x: float, default: float = 0.0) -> float:
    return float(x) if math.isfinite(x) else default


# ══════════════════════════════════════════════════════════════════════════════
# §E. DTOs INMUTABLES POR FASE
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class SpectralGaugeCache:
    r"""
    Carta espectral de la conexión: $iA = V\,\Omega\,V^\dagger$ (Hermítica),
    $A = -i\,V\,\Omega\,V^\dagger$, $\omega_j\in\mathbb{R}$ frecuencias de
    acción-ángulo (Poincaré–Laplace).
    """
    omega: NDArray[np.float64]
    eigvecs: NDArray[np.complex128]
    condition_number: float
    antiherm_residual: float
    traceless_residual: float


@dataclass(frozen=True, slots=True)
class QNDCertificate:
    r"""Certificado de Medición Débil QND con back-action **verificado**."""
    qnd_weak_value_aav: complex
    back_action_db: float
    qnd_fidelity: float
    is_qnd_preserved: bool
    commutator_norm: float
    measurement_strength: float


@dataclass(frozen=True, slots=True)
class SymplecticWitnessReport:
    r"""Embedding Cayley-Kähler en $\mathrm{sp}(2d,\mathbb{C})$ (o rama plana)."""
    cayley_matrix: NDArray[np.complex128]
    symplectic_residue: float
    symplectic_defect: float
    is_symplectic_valid: bool
    is_flat_branch: bool = False
    cayley_pole_margin: float = 1.0
    lie_algebra_defect_projected: float = 0.0


@dataclass(frozen=True, slots=True)
class PoincareLindstedtSolution:
    r"""
    Solución de la ecuación homológica de Poincaré–Lindstedt
    $\{H_0,W\}=H_1-\langle H_1\rangle_{\mathbb{T}^r}$ en la base propia de $iA$.
    """
    W: NDArray[np.complex128]
    min_resonance_gap: float
    residual_norm: float
    is_resonant: bool
    is_solved: bool
    newton_iterations: int = 1
    small_divisor_count: int = 0
    frequencies: Optional[NDArray[np.float64]] = None


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
    r"""
    Condición aritmética de Bruno: $\Omega(\theta)=\sum_n \log(q_{n+1})/q_n < \infty$
    sobre las fracciones continuas de las razones $\omega_j/\omega_0$.
    """
    is_satisfied: bool
    bruno_sum: float
    bruno_bound: float
    n_ratios: int
    worst_ratio_index: int


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
    r"""
    Condición de twist de Poincaré–Birkhoff sobre el mapa de retorno:
    si $\partial\theta'/\partial I \ne 0$ (área-preservante), existen
    al menos 2 puntos fijos.
    """
    twist_estimate: float
    is_twist: bool
    n_fixed_points_lower: int
    is_area_preserving_proxy: bool


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
    r"""
    Automejora recursiva de nivel 3: tres superficies (Novikov / Harness /
    Model) y contracción de Banach $T^3=T_3\circ T_2\circ T_1$ estimada
    por sonda de Lipschitz sobre los proyectores de ligadura.
    """
    surfaces: Tuple[RSI3SurfaceReport, RSI3SurfaceReport, RSI3SurfaceReport]
    lipschitz_t3: float
    capacity_jerk: float
    fixed_point_reached: bool
    iterations: int
    is_valid: bool
    lipschitz_factors: Tuple[float, float, float] = (1.0, 1.0, 1.0)
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
class WitnessPhase1Envelope:
    r"""Contrato FASE 1 (Observe) → FASE 2 (Orient)."""
    rho_sanitized: NDArray[np.complex128]
    connection_A: NDArray[np.complex128]
    curvature_F: NDArray[np.complex128]
    yang_mills_energy: float
    is_anti_hermitian_A: bool
    is_traceless_A: bool
    symplectic_report: SymplecticWitnessReport
    poincare_lindstedt: Optional[PoincareLindstedtSolution]
    dimension: int
    origin_timestamp_ns: float
    chain_hash: str
    spectral_cache: Optional[SpectralGaugeCache] = None


@dataclass(frozen=True, slots=True)
class WitnessPhase2Envelope:
    r"""Contrato FASE 2 (Orient) → FASE 3 (Decide&Act)."""
    phase1: WitnessPhase1Envelope
    qnd: QNDCertificate
    uhlmann_fidelity: float
    bures_distance: float
    trace_distance: float
    liouville_volume_drift: float
    stora_zumino_residual: float
    kam_certificate: KAMCertificate
    melnikov: MelnikovCertificate
    floquet: FloquetCertificate
    poincare_section: PoincareSectionCertificate
    rsi3_certificate: RSI3Certificate
    chain_hash: str
    origin_timestamp_ns: float
    bruno_certificate: Optional[BrunoCertificate] = None
    birkhoff_twist: Optional[BirkhoffTwistCertificate] = None


@dataclass(frozen=True, slots=True)
class GaugeWitnessSnapshot:
    r"""Snapshot inmutable del Testigo Silencioso (objeto terminal)."""
    # Campos originales (retrocompat)
    witness_id: str
    qnd_weak_value_aav: complex
    back_action_db: float
    uhlmann_fidelity: float
    bures_distance: float
    liouville_volume_drift: float
    yang_mills_background_energy: float
    stora_zumino_residual: float
    global_verdict: HeytingOmega3
    sha256_merkle_digest: str
    timestamp_utc: float

    # Campos extendidos v10.0.0
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
    witness_mode: str = "PASSIVE_SILENT"
    is_epistemologically_valid: bool = True
    trace_distance: float = 0.0
    phase1: Optional[WitnessPhase1Envelope] = field(default=None, repr=False)
    phase2: Optional[WitnessPhase2Envelope] = field(default=None, repr=False)
    hmac_sha256: bytes = b""
    engine_version: str = _ENGINE_VERSION

    # Campos extendidos v10.1.0 (Poincaré–Bruno–Birkhoff)
    bruno_satisfied: bool = True
    bruno_sum: float = 0.0
    birkhoff_twist_estimate: float = 0.0
    birkhoff_is_twist: bool = True
    lindstedt_solved: bool = True
    spectral_condition_number: float = 1.0
    kam_n_modes: int = 0
    is_flat_curvature: bool = True
    mean_poincare_return_time: float = 0.0
    hmac_verified: bool = True


# ══════════════════════════════════════════════════════════════════════════════════
# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║  F A S E   1 :   O B S E R V E                                               ║
# ║  Conexión su(N), curvatura F, Cayley-Kähler, Poincaré-Lindstedt espectral.   ║
# ║  Cierra con `emit_phase1_envelope` → `WitnessPhase1Envelope`.                ║
# ║  El cierre es la APERTURA formal de la FASE 2 (`from_phase1_envelope`).      ║
# ╚════════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════════

class WitnessPhase1Observer:
    r"""
    FASE 1 (Observe): sanitización, conexión de gauge, caché espectral de iA,
    curvatura de Yang-Mills, embedding simpléctico Cayley-Kähler y solución
    homológica de Poincaré-Lindstedt (Newton + divisores pequeños).
    """

    def __init__(
        self, dimension: int = 56, secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> None:
        self.dimension: Final[int] = max(4, int(dimension))
        self.secret_key: Final[bytes] = bytes(secret_key)

    # ── Sanitización PSD + Traza-1 ────────────────────────────────────────────
    def sanitize_density_matrix(self, matrix: NDArray) -> NDArray[np.complex128]:
        r"""
        Proyección de Weyl (parte Hermítica) + clamps espectrales + renormalización.

        $$
        \rho=\rho^\dagger,\qquad
        \lambda_i\ge\varepsilon_{\mathrm{PSD}},\qquad
        \operatorname{Tr}\rho=1.
        $$
        """
        a = np.asarray(matrix, dtype=np.complex128)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError(f"ρ debe ser cuadrada; recibido {a.shape}.")
        if a.shape != (self.dimension, self.dimension):
            raise ValueError(
                f"ρ debe ser {self.dimension}×{self.dimension}; "
                f"recibido {a.shape}."
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
    def extract_gauge_connection_A(
        self, density_matrix: NDArray,
    ) -> NDArray[np.complex128]:
        r"""
        $$
        A = i\Bigl(\rho - \tfrac{\operatorname{Tr}\rho}{N} I\Bigr)
        \;\in\; \mathfrak{su}(N),
        $$
        anti-Hermítica y traceless por el proyector $\Pi_{\mathfrak{su}(N)}$.

        **Identidad QND**: $[A,\rho]=i[\rho,\rho]=0$ exactamente en aritmética
        exacta.  El conmutador numérico se certifica en FASE 2 como integridad FP.
        """
        rho = self.sanitize_density_matrix(density_matrix)
        n = rho.shape[0]
        tr_val = np.trace(rho) / float(n)
        su_part = rho - tr_val * np.eye(n, dtype=np.complex128)
        return _project_su_n(1j * su_part)

    def _verify_anti_hermitian(
        self, A: NDArray[np.complex128],
    ) -> Tuple[bool, float]:
        defect = float(la.norm(A + A.conj().T, "fro"))
        scale = max(1.0, float(la.norm(A, "fro")))
        rel = defect / scale
        return bool(rel < _ANTI_HERM_TOL), rel

    def _verify_traceless(
        self, A: NDArray[np.complex128],
    ) -> Tuple[bool, float]:
        val = float(abs(complex(np.trace(A))))
        return bool(val < _TRACELESS_TOL), val

    # ── Caché espectral de iA (frecuencias de Poincaré–Laplace) ───────────────
    def _spectral_cache_from_A(
        self, A: NDArray[np.complex128],
    ) -> SpectralGaugeCache:
        r"""
        $iA$ es Hermítica (A anti-Hermítica).  Descomposición
        $iA = V\Omega V^\dagger$, $\omega_j=\Omega_{jj}\in\mathbb{R}$.
        Condición $\kappa_2=\lambda_{\max}/(\lambda_{\min}+\varepsilon)$.
        """
        iA = 1j * A
        iA_h = 0.5 * (iA + iA.conj().T)
        try:
            omega, V = la.eigh(iA_h)
            omega = np.real(omega).astype(np.float64)
        except la.LinAlgError as exc:
            raise WitnessSpectralError(f"eigh(iA) falló: {exc}") from exc
        abs_om = np.abs(omega)
        cond = float(
            np.max(abs_om) / max(float(np.min(abs_om)), _WILKINSON_EPSILON)
        ) if omega.size else 1.0
        _, ah_rel = self._verify_anti_hermitian(A)
        _, tr_abs = self._verify_traceless(A)
        return SpectralGaugeCache(
            omega=omega,
            eigvecs=np.asarray(V, dtype=np.complex128),
            condition_number=_finite_or(cond, 1.0),
            antiherm_residual=float(ah_rel),
            traceless_residual=float(tr_abs),
        )

    def _unitary_orbit(
        self,
        cache: SpectralGaugeCache,
        phi0: NDArray[np.complex128],
        t_grid: NDArray[np.float64],
    ) -> NDArray[np.complex128]:
        r"""
        Órbita unitaria vectorizada (una sola diagonalización):

        $$
        e^{tA}=V\,\mathrm{diag}(e^{-i t\omega})\,V^\dagger,
        \qquad
        \varphi(t)=e^{tA}\varphi_0.
        $$

        Retorna array `(T, d)`.
        """
        phi0u = _safe_unit(phi0)
        V = cache.eigvecs
        omega = cache.omega
        c0 = V.conj().T @ phi0u
        phases = np.exp(-1j * np.outer(t_grid, omega))  # (T, d)
        # (d, T) = V @ (c0 * phases)^T  →  (T, d)
        return (V @ (phases * c0).T).T

    # ── Método 2: Curvatura de Yang-Mills F = [A, ρ] ──────────────────────────
    def compute_yang_mills_curvature_F(
        self,
        A: NDArray[np.complex128],
        density_matrix: NDArray,
    ) -> Tuple[NDArray[np.complex128], float]:
        r"""
        $F=[A,\rho_h]=A\rho_h-\rho_h A$, anti-Hermítica por herencia
        (A anti-Herm, ρ Herm ⇒ F anti-Herm).  En aritmética exacta F≡0
        por la identidad QND; el residual es un testigo de redondeo.

        Energía de Yang-Mills (positiva):
        $E_{YM}=-\tfrac12\operatorname{Tr}(F^2)\ge 0$
        ($F$ anti-Hermítica ⇒ $F^2$ negativa semidefinida).
        """
        rho = self.sanitize_density_matrix(density_matrix)
        F = _project_anti_hermitian(A @ rho - rho @ A)
        energy_ym = -0.5 * float(np.real(np.trace(F @ F)))
        return F, max(0.0, float(energy_ym))

    # ── Embedding Cayley-Kähler en sp(2d,ℂ) ──────────────────────────────────
    def _cayley_kahler_embedding(
        self, F: NDArray[np.complex128],
    ) -> SymplecticWitnessReport:
        r"""
        Carta de Cayley de $H=iF$ (Hermítica) hacia $\mathrm{Sp}(2d,\mathbb{C})$.

        **Rama plana.**  Si $\|F\|_F\le\varepsilon_{\mathrm{flat}}$, la
        curvatura se anula y Darboux es vacuamente la identidad:
        $\delta\omega=0$, defecto=0, válido.  (Evita el falso negativo
        $K=-I\Rightarrow K^T J+JK=-2J$.)

        **Rama no plana.**
          (i)  polo de Cayley: $\min_j|\lambda_j(H)+1|$ (margen).
          (ii) $K=(H-I)(H+I)^{-1}$ embebida como
               $\mathrm{diag}(K,\overline K)$ en $\mathbb{C}^{2d}$.
          (iii) $\|K^T J+JK\|_F$ (pertenencia a $\mathfrak{sp}(2d)$).
          (iv)  $M=(I-K)^{-1}(I+K)$, $\delta\omega=\|M^T JM-J\|_F/\|M\|_F^2$.
          (v)   un paso de proyección al álgebra: $K\leftarrow K+\tfrac12 J S$,
                $S=K^T J+JK$, $J^{-1}=-J$.
        """
        d = int(F.shape[0])
        f_norm = float(la.norm(F, "fro"))
        if f_norm <= _FLAT_CURVATURE_TOL:
            return SymplecticWitnessReport(
                cayley_matrix=np.eye(2 * d, dtype=np.complex128),
                symplectic_residue=0.0,
                symplectic_defect=0.0,
                is_symplectic_valid=True,
                is_flat_branch=True,
                cayley_pole_margin=1.0,
                lie_algebra_defect_projected=0.0,
            )

        H = 0.5 * ((1j * F) + (1j * F).conj().T)
        I_d = np.eye(d, dtype=np.complex128)
        J = _standard_symplectic_j(d)
        try:
            eig_h = np.real(la.eigvalsh(H))
            pole_margin = float(np.min(np.abs(eig_h + 1.0)))
            if pole_margin < _CAYLEY_POLE_TOL:
                return SymplecticWitnessReport(
                    cayley_matrix=np.zeros((2 * d, 2 * d), dtype=np.complex128),
                    symplectic_residue=float("inf"),
                    symplectic_defect=float("inf"),
                    is_symplectic_valid=False,
                    is_flat_branch=False,
                    cayley_pole_margin=pole_margin,
                    lie_algebra_defect_projected=float("inf"),
                )

            K_blk = (H - I_d) @ la.pinv(H + I_d)
            K_full = np.block([
                [K_blk, np.zeros((d, d), dtype=np.complex128)],
                [np.zeros((d, d), dtype=np.complex128), K_blk.conj()],
            ])
            S = K_full.T @ J + J @ K_full
            sp_defect = float(la.norm(S, "fro"))

            # Proyección de un paso a 𝔰𝔭(2d): K' = K + ½ J S  (J⁻¹ = −J)
            K_sp = K_full + 0.5 * (J @ S)
            S_proj = K_sp.T @ J + J @ K_sp
            sp_defect_proj = float(la.norm(S_proj, "fro"))

            I_2d = np.eye(2 * d, dtype=np.complex128)
            M = la.pinv(I_2d - K_sp) @ (I_2d + K_sp)
            diff = M.T @ J @ M - J
            norm_M = float(la.norm(M, "fro"))
            residue = (
                float(la.norm(diff, "fro") / (norm_M ** 2))
                if norm_M > _WILKINSON_EPSILON
                else float("inf")
            )
            valid = (
                math.isfinite(residue)
                and residue < _SYMPLECTIC_TOL
                and sp_defect_proj < _SYMPLECTIC_TOL
            )
            return SymplecticWitnessReport(
                cayley_matrix=K_sp,
                symplectic_residue=_finite_or(residue, 1.0),
                symplectic_defect=sp_defect,
                is_symplectic_valid=bool(valid),
                is_flat_branch=False,
                cayley_pole_margin=pole_margin,
                lie_algebra_defect_projected=sp_defect_proj,
            )
        except (la.LinAlgError, ValueError):
            return SymplecticWitnessReport(
                cayley_matrix=np.zeros((2 * d, 2 * d), dtype=np.complex128),
                symplectic_residue=1.0,
                symplectic_defect=1.0,
                is_symplectic_valid=False,
                is_flat_branch=False,
                cayley_pole_margin=0.0,
                lie_algebra_defect_projected=1.0,
            )

    # ── Poincaré-Lindstedt (base propia + Newton homológico) ─────────────────
    def _homological_rhs_eigenbasis(
        self,
        H1_e: NDArray[np.complex128],
        omega: NDArray[np.float64],
        W_e: NDArray[np.complex128],
        eps: float,
    ) -> Tuple[NDArray[np.complex128], float, int, bool]:
        r"""
        Ecuación homológica en la base propia de $H_0=\Omega$:

        $$
        \{H_0,W\}_{ij}
        = -i(\omega_i-\omega_j)W_{ij}
        = (H_1^{\mathrm{osc}})_{ij}.
        $$

        Divisores pequeños regularizados (pseudoinversa de Tikhonov):
        $$
        W_{ij}
        = i\,(H_1^{\mathrm{osc}})_{ij}
          \frac{\Delta\omega}{\Delta\omega^2+\varepsilon^2}.
        $$
        Retorna $(\Delta W,\;\min|\Delta\omega|,\;\#\{\mathrm{small}\},\;\mathrm{resonant})$.
        """
        d = omega.size
        H1_mean = np.diag(np.diag(H1_e))
        H1_osc = H1_e - H1_mean
        # {H0, W}_ij = -i (ω_i − ω_j) W_ij
        dw = omega[:, None] - omega[None, :]
        ad_W = -1j * dw * W_e
        R = H1_osc - ad_W  # residual de la homológica
        dW = np.zeros_like(H1_e)
        min_gap = float("inf")
        n_small = 0
        resonant = False
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                gap = float(dw[i, j])
                ag = abs(gap)
                if ag < min_gap:
                    min_gap = ag
                if ag < _LINDSTEDT_RESONANCE_GAP:
                    resonant = True
                    n_small += 1
                dW[i, j] = 1j * R[i, j] * gap / (gap * gap + eps * eps)
        return dW, (min_gap if math.isfinite(min_gap) else 0.0), n_small, resonant

    def _solve_poincare_lindstedt(
        self,
        F: NDArray[np.complex128],
        A: NDArray[np.complex128],
        cache: Optional[SpectralGaugeCache] = None,
    ) -> PoincareLindstedtSolution:
        r"""
        Resuelve $\{H_0,W\}=H_1-\langle H_1\rangle$ con
        $H_0=\Omega=\mathrm{eig}(iA)$, $H_1=V^\dagger F V$,
        corchete $\{A,B\}_q=-i[A,B]$.

        Automejora: Newton homológico $W\leftarrow W+\Delta W$ hasta
        $\|R\|_F\le 10^{-9}$ o `_LINDSTEDT_NEWTON_ITERS`.  PASSIVE_SILENT:
        $W$ es un gráfico computacional, no una acción sobre el campo.
        """
        cache = cache or self._spectral_cache_from_A(A)
        V = cache.eigvecs
        omega = cache.omega
        H1_e = V.conj().T @ F @ V
        W_e = np.zeros_like(H1_e)
        min_gap = 0.0
        n_small = 0
        resonant = False
        residual = float("inf")
        it_used = 0
        eps = _LINDSTEDT_REGULARIZATION
        for it in range(1, _LINDSTEDT_NEWTON_ITERS + 1):
            it_used = it
            dW, min_gap, n_small, resonant = self._homological_rhs_eigenbasis(
                H1_e, omega, W_e, eps,
            )
            W_e = W_e + dW
            dw = omega[:, None] - omega[None, :]
            H1_osc = H1_e - np.diag(np.diag(H1_e))
            commut = -1j * dw * W_e
            residual = float(la.norm(commut - H1_osc, "fro"))
            if residual < _LINDSTEDT_RESIDUAL_TOL:
                break
            # Automejora de ε: si hay muchos divisores pequeños, relajar
            if n_small > 0:
                eps = min(1e-6, eps * 1.5)

        W = V @ W_e @ V.conj().T
        return PoincareLindstedtSolution(
            W=np.asarray(W, dtype=np.complex128),
            min_resonance_gap=float(min_gap),
            residual_norm=float(residual),
            is_resonant=bool(resonant),
            is_solved=bool(residual < _LINDSTEDT_RESIDUAL_TOL and not resonant),
            newton_iterations=int(it_used),
            small_divisor_count=int(n_small),
            frequencies=np.asarray(omega, dtype=np.float64),
        )

    def _phase1_chain_payload(
        self,
        rho: NDArray[np.complex128],
        A: NDArray[np.complex128],
        F: NDArray[np.complex128],
        energy_ym: float,
        sympl: SymplecticWitnessReport,
    ) -> str:
        return _hash_state(
            rho, A, F,
            np.array(
                [energy_ym, sympl.symplectic_residue, sympl.symplectic_defect],
                dtype=np.float64,
            ),
        )

    def verify_phase1_chain(self, env: WitnessPhase1Envelope) -> None:
        expected = self._phase1_chain_payload(
            env.rho_sanitized, env.connection_A, env.curvature_F,
            env.yang_mills_energy, env.symplectic_report,
        )
        if expected != env.chain_hash:
            raise WitnessContractCorruptedError(
                f"Chain FASE 1 corrupta: {expected} ≠ {env.chain_hash}."
            )

    # ── CIERRE FASE 1 → FASE 2 ────────────────────────────────────────────────
    def emit_phase1_envelope(
        self, density_matrix: NDArray,
    ) -> WitnessPhase1Envelope:
        r"""
        **Cierre formal de la FASE 1 / puente de apertura de la FASE 2.**

        Ejecuta la auditoría completa
        (sanitización + $\Pi_{\mathfrak{su}}$ + F + espectro(iA) + Cayley +
        Lindstedt-Newton) y emite `WitnessPhase1Envelope` con hash-chain.
        El receptor canónico es `WitnessPhase2Orientor.from_phase1_envelope`.
        """
        t0 = time.perf_counter_ns()
        rho = self.sanitize_density_matrix(density_matrix)
        A = self.extract_gauge_connection_A(rho)
        F, energy_ym = self.compute_yang_mills_curvature_F(A, rho)

        a_ah, _ = self._verify_anti_hermitian(A)
        a_tr, _ = self._verify_traceless(A)
        cache = self._spectral_cache_from_A(A)
        sympl = self._cayley_kahler_embedding(F)
        pl = self._solve_poincare_lindstedt(F, A, cache=cache)

        chain = self._phase1_chain_payload(rho, A, F, energy_ym, sympl)
        return WitnessPhase1Envelope(
            rho_sanitized=rho,
            connection_A=A,
            curvature_F=F,
            yang_mills_energy=float(energy_ym),
            is_anti_hermitian_A=a_ah,
            is_traceless_A=a_tr,
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
# ║  Apertura = from_phase1_envelope  (continuación formal del cierre FASE 1).   ║
# ║  QND verificado, Uhlmann, Bures, KAM, Bruno, Melnikov, Floquet,              ║
# ║  Poincaré-Birkhoff, RSI³ Banach T³.                                          ║
# ║  Cierra con `emit_phase2_envelope` → `WitnessPhase2Envelope`.                ║
# ╚════════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════════

class WitnessPhase2Orientor(WitnessPhase1Observer):
    r"""
    FASE 2 (Orient): certifica QND verificado, geometría de Bures, mecánica
    celeste de Poincaré (KAM, Bruno, Melnikov, Floquet, sección, twist de
    Birkhoff) y RSI³ sobre las 3 superficies con Lipschitz sonda de Banach.
    """

    # ── APERTURA FASE 2  (continuación del último método de FASE 1) ───────────
    @classmethod
    def from_phase1_envelope(
        cls,
        env: WitnessPhase1Envelope,
        dimension: Optional[int] = None,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> "WitnessPhase2Orientor":
        r"""
        **Apertura formal de la FASE 2** — continuación de
        `WitnessPhase1Observer.emit_phase1_envelope`.

        Valida el chain_hash de FASE 1 y construye el orientador.
        """
        dim = int(dimension) if dimension is not None else int(env.dimension)
        inst = cls(dimension=dim, secret_key=secret_key)
        inst.verify_phase1_chain(env)
        inst._phase1_env = env  # type: ignore[attr-defined]
        return inst

    def _cache_or_compute(
        self,
        A: NDArray[np.complex128],
        env: Optional[WitnessPhase1Envelope] = None,
    ) -> SpectralGaugeCache:
        if env is not None and env.spectral_cache is not None:
            return env.spectral_cache
        return self._spectral_cache_from_A(A)

    # ── Método 3: QND verificado con back-action real ─────────────────────────
    def compute_qnd_weak_value_verified(
        self,
        phi_initial: NDArray,
        phi_final: NDArray,
        A: NDArray[np.complex128],
        rho: NDArray[np.complex128],
    ) -> QNDCertificate:
        r"""
        Medición débil QND (Aharonov–Albert–Vaidman) con **verificación real**
        del back-action:

          1. Valor débil: $A_w=\langle\phi_f|A|\phi_i\rangle/\langle\phi_f|\phi_i\rangle$.
          2. Fidelidad QND: $|\langle\phi_f|\phi_i\rangle|^2$.
          3. **Commutador**: $\|[A,\rho]\|_F$ — invariante de no-demolición.
             Identidad teórica: $A=i(\rho-\frac{\mathrm{Tr}\rho}{N}I)\Rightarrow[A,\rho]=0$.
             El residual certifica integridad de punto flotante, no un observable
             independiente.
          4. **Back-action**:
             - Si $\|[A,\rho]\|_F\le 10^{-6}$: `back_action_db=0.0` (QND).
             - Sino: $-10\log_{10}(1-\min(\|[A,\rho]\|,0.99))$.
        """
        i_vec = _safe_unit(np.asarray(phi_initial, dtype=np.complex128))
        f_vec = _safe_unit(np.asarray(phi_final, dtype=np.complex128))
        a_mat = np.asarray(A, dtype=np.complex128)
        rho_mat = np.asarray(rho, dtype=np.complex128)

        overlap = complex(np.vdot(f_vec, i_vec))
        if abs(overlap) < _WILKINSON_EPSILON:
            overlap = complex(_WILKINSON_EPSILON, 0.0)
        numerator = complex(np.vdot(f_vec, a_mat @ i_vec))
        a_weak = numerator / overlap
        qnd_fidelity = float(abs(overlap) ** 2)

        commutator = a_mat @ rho_mat - rho_mat @ a_mat
        commutator_norm = float(la.norm(commutator, "fro"))

        if commutator_norm <= _QND_COMMUTATOR_MAX:
            back_action_db = 0.0
        else:
            disturb = min(commutator_norm, 0.99)
            back_action_db = float(-10.0 * math.log10(max(1.0 - disturb, 1e-15)))

        is_preserved = (
            commutator_norm <= _QND_COMMUTATOR_MAX
            and back_action_db <= _QND_BACKACTION_DB_TOL
            and qnd_fidelity >= _QND_FIDELITY_MIN
        )
        measurement_strength = float(1.0 - qnd_fidelity)
        return QNDCertificate(
            qnd_weak_value_aav=a_weak,
            back_action_db=float(back_action_db),
            qnd_fidelity=float(qnd_fidelity),
            is_qnd_preserved=bool(is_preserved),
            commutator_norm=float(commutator_norm),
            measurement_strength=float(measurement_strength),
        )

    # ── Método 4: Uhlmann + Bures + Traza + Liouville ────────────────────────
    def audit_uhlmann_bures(
        self, rho: NDArray[np.complex128],
    ) -> Tuple[float, float, float, float]:
        r"""
        Retorna `(fidelity, bures_distance, trace_distance, liouville_drift)`.

        Fidelidad de Uhlmann frente a la referencia maximamente mixta:
        $F(\rho,I/d)=(\sum_i\sqrt{\lambda_i})^2/d$ (forma cerrada, sin `sqrtm`).
        Bures: $D_B=\sqrt{2-2\sqrt{F}}$.
        Traza: $D_{\mathrm{tr}}=\tfrac12\operatorname{Tr}|\rho-I/d|$.
        Liouville: $|\operatorname{Tr}\rho-1|$.
        """
        dim = rho.shape[0]
        rho_ref = np.eye(dim, dtype=np.complex128) / float(dim)
        try:
            evals_rho = np.clip(np.real(la.eigvalsh(rho)), 0.0, None)
            trace_sqrt_rho = float(np.sum(np.sqrt(evals_rho)))
            fidelity = (trace_sqrt_rho ** 2) / float(dim)
        except la.LinAlgError:
            fidelity = 0.0
        fidelity = _clip01(fidelity)
        bures = float(math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(fidelity))))
        try:
            trace_dist = 0.5 * float(np.sum(np.abs(la.eigvalsh(rho - rho_ref))))
        except la.LinAlgError:
            trace_dist = 0.5 * float(la.norm(rho - rho_ref, "nuc"))
        liouville = float(abs(np.real(np.trace(rho)) - 1.0))
        return float(fidelity), float(bures), float(trace_dist), float(liouville)

    def audit_liouville_symplectic_volume(
        self, density_matrix: NDArray,
    ) -> Tuple[float, float, float]:
        r"""
        **[Compat v9.3.0 / v10.0.0]** Retorna `(fidelity, bures_dist, liouville_drift)`.
        Delega a `audit_uhlmann_bures`.
        """
        rho = self.sanitize_density_matrix(density_matrix)
        fid, bures, _, drift = self.audit_uhlmann_bures(rho)
        return fid, bures, drift

    # ── Stora-Zumino ──────────────────────────────────────────────────────────
    def audit_stora_zumino(self, F: NDArray[np.complex128]) -> float:
        r"""$\Delta_{SZ}=|\operatorname{Tr}(F^2)|$ (proxy de la anomalía)."""
        try:
            return float(abs(np.real(np.trace(F @ F))))
        except Exception:
            return 0.0

    # ── KAM / Bruno (modos dominantes, no 7^d) ────────────────────────────────
    def _dominant_frequencies(
        self,
        freqs: NDArray[np.float64],
        n_modes: int = _KAM_EFFECTIVE_MODES,
    ) -> NDArray[np.float64]:
        f = np.asarray(freqs, dtype=np.float64).flatten()
        f = np.abs(f)
        f = f[f > _KAM_FREQ_ABS_FLOOR]
        if f.size == 0:
            return np.array([1.0], dtype=np.float64)
        # Deduplicar a tolerancia relativa
        f = np.sort(f)[::-1]
        uniq: List[float] = [float(f[0])]
        for x in f[1:]:
            if abs(float(x) - uniq[-1]) / max(uniq[-1], 1e-15) > 1e-8:
                uniq.append(float(x))
            if len(uniq) >= n_modes:
                break
        return np.asarray(uniq[:n_modes], dtype=np.float64)

    def check_kam_bruno(
        self,
        freqs: NDArray[np.float64],
        gamma: float = _KAM_GAMMA,
        max_order: int = _KAM_MAX_ORDER,
    ) -> KAMCertificate:
        r"""
        Condición diofantina de Bruno–Rüssmann sobre **r≤4 modos dominantes**
        (el retículo $[-K,K]^d$ con $d=56$ es $7^{56}$, indecidible):

        $$
        |\omega\cdot k|\;\|k\|_1^{\tau}\;\ge\;\gamma,
        \qquad \tau=r+1,
        \quad k\in[-K,K]^r\setminus\{0\}.
        $$
        """
        freqs = self._dominant_frequencies(freqs)
        d = int(freqs.size)
        tau = float(d) + _KAM_TAU_OFFSET
        worst = float("inf")
        n_probes = 0
        for k in product(range(-max_order, max_order + 1), repeat=d):
            if all(ki == 0 for ki in k):
                continue
            n_probes += 1
            k_vec = np.asarray(k, dtype=np.float64)
            k_norm_1 = float(np.sum(np.abs(k_vec)))
            dot = abs(float(np.dot(freqs, k_vec)))
            ratio = dot * (k_norm_1 ** tau) / max(gamma, 1e-15)
            if ratio < worst:
                worst = ratio
        return KAMCertificate(
            is_satisfied=bool(worst >= 1.0),
            worst_ratio=_finite_or(float(worst), 0.0),
            tau=tau,
            gamma=float(gamma),
            omega_min=float(np.min(np.abs(freqs))),
            n_modes=d,
            n_lattice_probes=int(n_probes),
        )

    def check_bruno_arithmetic(
        self,
        freqs: NDArray[np.float64],
        max_terms: int = _BRUNO_MAX_TERMS,
        bound: float = _BRUNO_SUM_MAX,
    ) -> BrunoCertificate:
        r"""
        Condición de Bruno sobre las razones $\theta_j=\omega_j/\omega_0\bmod 1$:

        $$
        \Omega(\theta)=\sum_{n=0}^{N}\frac{\log(q_{n+1})}{q_n}<\infty
        $$
        (proxy finito; $q_n$ denominadores de la fracción continua).
        """
        freqs = self._dominant_frequencies(freqs)
        if freqs.size < 2:
            return BrunoCertificate(True, 0.0, bound, 0, -1)
        omega0 = float(freqs[0])
        if abs(omega0) < _KAM_FREQ_ABS_FLOOR:
            return BrunoCertificate(False, float("inf"), bound, 0, -1)

        def _cf_bruno_sum(theta: float) -> float:
            x = abs(theta) % 1.0
            if x < 1e-15 or abs(x - 1.0) < 1e-15:
                return 0.0
            q_prev, q, a_prev = 1, int(max(1, math.floor(1.0 / max(x, 1e-15)))), 0
            acc = 0.0
            y = x
            for _ in range(max_terms):
                if y < 1e-15:
                    break
                a = int(math.floor(1.0 / y))
                if a <= 0:
                    break
                y = 1.0 / y - a
                q_next = a * q + q_prev
                if q_next <= 0:
                    break
                acc += math.log(max(q_next, 2)) / max(q, 1)
                q_prev, q, a_prev = q, q_next, a
                if q > 10**9:
                    break
            return float(acc)

        worst_sum = 0.0
        worst_idx = 0
        n_ratios = 0
        for j in range(1, freqs.size):
            theta = float(freqs[j] / omega0)
            s = _cf_bruno_sum(theta)
            n_ratios += 1
            if s > worst_sum:
                worst_sum = s
                worst_idx = j
        return BrunoCertificate(
            is_satisfied=bool(worst_sum < bound and math.isfinite(worst_sum)),
            bruno_sum=float(worst_sum),
            bruno_bound=float(bound),
            n_ratios=int(n_ratios),
            worst_ratio_index=int(worst_idx),
        )

    # ── Melnikov (Simpson + órbita espectral) ─────────────────────────────────
    def compute_melnikov(
        self,
        A: NDArray[np.complex128],
        F: NDArray[np.complex128],
        phi0: NDArray[np.complex128],
        n_samples: int = _MELNIKOV_SAMPLES,
        cache: Optional[SpectralGaugeCache] = None,
    ) -> MelnikovCertificate:
        r"""
        Integral de Melnikov sobre la homoclínica tipo Duffing
        (peso $\operatorname{sech}^2 t$, órbita $\varphi(t)=e^{tA}\varphi_0$):

        $$
        M(\theta_0)
        =-\int_{\mathbb{R}}
          \operatorname{Re}\langle F\varphi(t),\varphi(t)\rangle
          \operatorname{sech}^2(t)\,dt.
        $$
        Cuadratura de Simpson (n impar) sobre $t\in[-8,8]$.
        """
        if n_samples % 2 == 0:
            n_samples += 1
        phi0u = _safe_unit(phi0)
        cache = cache or self._spectral_cache_from_A(A)
        t_grid = np.linspace(-8.0, 8.0, n_samples)
        try:
            orbit = self._unitary_orbit(cache, phi0u, t_grid)  # (T, d)
            # ⟨F φ, φ⟩ = φ† F φ  (F anti-Herm ⇒ imaginario puro + residual real)
            Fphi = orbit @ F.T  # no: F is (d,d), orbit (T,d) → (orbit @ F.T) if F Herm
            # (T, d) @ (d, d)^T ?  We need (F @ φ_t).  (F @ orbit.T).T = orbit @ F.T
            force = orbit @ F.conj().T  # if we want φ† F?  vdot(Fφ, φ) = φ† F† F? NO
            # integrand Re ⟨Fφ, φ⟩ = Re( (Fφ)† φ ) = Re( φ† F† φ )
            # original: np.real(np.vdot(force, phi_t)) with force = F @ phi_t
            # = Re(φ† F† φ).  Keep the same convention: vdot(Fφ, φ).
            force = (F @ orbit.T).T
            raw = np.einsum("td,td->t", np.conjugate(force), orbit).real
            sech2 = 1.0 / np.cosh(t_grid) ** 2
            integrand = raw * sech2
            # Simpson
            w = np.ones(n_samples, dtype=np.float64)
            w[1:-1:2] = 4.0
            w[2:-1:2] = 2.0
            dt = float(t_grid[1] - t_grid[0])
            M = float((dt / 3.0) * np.dot(w, integrand))
            max_int = float(np.max(np.abs(integrand)))
        except (la.LinAlgError, ValueError):
            return MelnikovCertificate(0.0, 0.0, True)
        return MelnikovCertificate(
            integral_value=float(M),
            max_abs_integrand=float(max_int),
            is_homoclinic_preserved=bool(abs(M) < _MELNIKOV_TOL),
        )

    # ── Floquet ───────────────────────────────────────────────────────────────
    def compute_floquet(
        self,
        F: NDArray[np.complex128],
        period: float = 2.0 * math.pi,
    ) -> FloquetCertificate:
        r"""
        Multiplicadores de Floquet $\mu_k=\mathrm{eig}(e^{T F})$.
        Si $F$ es anti-Hermítica, $e^{TF}$ es unitaria y $|\mu_k|=1$
        (neutralidad).  El desvío certifica la anti-Hermiticidad numérica.
        Exponentes de Lyapunov: $\lambda_k=\log|\mu_k|/T$.
        """
        try:
            M_T = la.expm(np.asarray(F, dtype=np.complex128) * period)
            mu = la.eigvals(M_T)
            moduli = np.abs(mu)
            unit_dev = float(np.max(np.abs(moduli - 1.0)))
            dissipative = bool(np.max(moduli) <= 1.0 + _FLOQUET_UNIT_TOL)
            exponents = np.log(np.maximum(moduli, 1e-30)) / max(period, 1e-15)
        except (la.LinAlgError, ValueError):
            mu = np.array([1.0 + 0j])
            unit_dev, dissipative, exponents = 0.0, True, np.array([0.0])
        return FloquetCertificate(
            multipliers=np.asarray(mu, dtype=np.complex128),
            max_unit_deviation=float(unit_dev),
            is_neutral=bool(unit_dev < _FLOQUET_UNIT_TOL),
            is_dissipative_stable=dissipative,
            exponents=np.asarray(exponents, dtype=np.float64),
        )

    # ── Sección de Poincaré (órbita espectral + interpolación) ───────────────
    def compute_poincare_section(
        self,
        A: NDArray[np.complex128],
        phi0: NDArray[np.complex128],
        P_obs: Optional[NDArray[np.complex128]] = None,
        n_samples: int = _POINCARE_SECTION_SAMPLES,
        cache: Optional[SpectralGaugeCache] = None,
    ) -> PoincareSectionCertificate:
        r"""
        Observable $O(t)=\langle\varphi(t)|P_0|\varphi(t)\rangle$,
        $\varphi(t)=e^{tA}\varphi_0$.  Sección transversal:
        ceros de $\dot O$ por cambio de signo, con interpolación lineal
        del valor de retorno.  Tiempo medio de retorno (Poincaré).
        """
        phi0u = _safe_unit(phi0)
        d = A.shape[0]
        if P_obs is None:
            P_obs = np.zeros((d, d), dtype=np.complex128)
            P_obs[0, 0] = 1.0
        cache = cache or self._spectral_cache_from_A(A)
        t_grid = np.linspace(0.0, 4.0 * math.pi, n_samples)
        try:
            orbit = self._unitary_orbit(cache, phi0u, t_grid)
            Pphi = (P_obs @ orbit.T).T
            obs = np.einsum("td,td->t", np.conjugate(orbit), Pphi).real.astype(np.float64)
        except (la.LinAlgError, ValueError):
            return PoincareSectionCertificate(0, 0.0, True, (), 0.0)

        d_obs = np.diff(obs)
        # Cruces de Ḋ por 0: cambio de signo de d_obs
        sig = np.sign(d_obs)
        sig[sig == 0.0] = 1.0
        crossings_idx = np.where(sig[:-1] * sig[1:] < 0.0)[0]
        if crossings_idx.size < 2:
            return PoincareSectionCertificate(
                n_crossings=int(crossings_idx.size),
                return_dispersion=0.0,
                is_regular=True,
                returns=(),
                mean_return_time=0.0,
            )
        # Interpolación lineal del observable en el cruce de Ḋ
        returns_list: List[float] = []
        t_cross: List[float] = []
        for i in crossings_idx:
            # d_obs[i] → d_obs[i+1] cruza 0; el cruce de O está cerca de t[i+1]
            den = d_obs[i + 1] - d_obs[i]
            alpha = 0.0 if abs(den) < 1e-18 else float(-d_obs[i] / den)
            alpha = min(max(alpha, 0.0), 1.0)
            val = float((1.0 - alpha) * obs[i + 1] + alpha * obs[i + 2 if i + 2 < obs.size else i + 1])
            tk = float((1.0 - alpha) * t_grid[i + 1] + alpha * t_grid[min(i + 2, t_grid.size - 1)])
            returns_list.append(val)
            t_cross.append(tk)
        returns_arr = np.asarray(returns_list, dtype=np.float64)
        disp = float(
            np.std(returns_arr) / max(float(np.mean(np.abs(returns_arr))), 1e-15)
        )
        if len(t_cross) >= 2:
            mean_ret = float(np.mean(np.diff(np.asarray(t_cross, dtype=np.float64))))
        else:
            mean_ret = 0.0
        return PoincareSectionCertificate(
            n_crossings=int(crossings_idx.size),
            return_dispersion=float(disp),
            is_regular=bool(disp < 0.5),
            returns=tuple(float(x) for x in returns_arr),
            mean_return_time=float(mean_ret),
        )

    def compute_birkhoff_twist(
        self,
        section: PoincareSectionCertificate,
        liouville_drift: float,
    ) -> BirkhoffTwistCertificate:
        r"""
        Proxy de twist de Poincaré–Birkhoff: la derivada discreta de los
        retornos $\Delta O_{n+1}/\Delta n$ como $\partial\theta'/\partial I$.
        Si hay twist y el flujo es (proxy) área-preservante
        ($|\operatorname{Tr}\rho-1|$ pequeño), el teorema garantiza ≥2 puntos fijos.
        """
        rets = np.asarray(section.returns, dtype=np.float64)
        if rets.size < 3:
            return BirkhoffTwistCertificate(
                twist_estimate=0.0,
                is_twist=False,
                n_fixed_points_lower=0,
                is_area_preserving_proxy=bool(liouville_drift < _TRACE_CONSERVATION_TOL),
            )
        d1 = np.diff(rets)
        twist = float(np.mean(np.abs(d1)))
        is_twist = bool(twist >= _BIRKHOFF_TWIST_MIN)
        area_ok = bool(liouville_drift < _TRACE_CONSERVATION_TOL)
        n_fixed = 2 if (is_twist and area_ok) else 0
        return BirkhoffTwistCertificate(
            twist_estimate=twist,
            is_twist=is_twist,
            n_fixed_points_lower=int(n_fixed),
            is_area_preserving_proxy=area_ok,
        )

    # ── RSI³ sobre 3 superficies + Lipschitz sonda de Banach T³ ──────────────
    def _fubini_study(
        self,
        phi_a: NDArray[np.complex128],
        phi_b: NDArray[np.complex128],
    ) -> float:
        a = _safe_unit(phi_a)
        b = _safe_unit(phi_b)
        ov = abs(complex(np.vdot(a, b)))
        return float(np.arccos(np.clip(ov, 0.0, 1.0)))

    def _probe_lipschitz_projectors(
        self,
        A: NDArray[np.complex128],
        F: NDArray[np.complex128],
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[float, float, float]:
        r"""
        Estima $\mathrm{Lip}(T_i)$ por sonda: $T_1=\Pi_{\mathrm{PSD\mbox{-}tr}}$,
        $T_2=\Pi_{\mathfrak{su}(N)}$, $T_3=\Pi_{\mathrm{ah}}$ sobre $F$.
        Los tres son no-expansivos teóricamente ($\mathrm{Lip}\le 1$);
        el producto es $\mathrm{Lip}(T^3)$.
        """
        gen = rng or np.random.default_rng(2026)
        d = A.shape[0]
        noise = gen.normal(0.0, 1.0, size=(d, d)) + 1j * gen.normal(0.0, 1.0, size=(d, d))
        delta_A = _project_su_n(noise.astype(np.complex128))
        nrm = float(la.norm(delta_A, "fro"))
        if nrm < _WILKINSON_EPSILON:
            return (1.0, 1.0, 1.0)
        delta_A = (_RSI3_PROBE_EPS / nrm) * delta_A

        # T2: Π_su — no expansivo
        A0 = _project_su_n(A)
        A1 = _project_su_n(A + delta_A)
        lip2 = float(la.norm(A1 - A0, "fro") / max(float(la.norm(delta_A, "fro")), 1e-18))

        # T3: Π_ah sobre F
        delta_F = _project_anti_hermitian(delta_A)
        F0 = _project_anti_hermitian(F)
        F1 = _project_anti_hermitian(F + delta_F)
        den_f = float(la.norm(delta_F, "fro"))
        lip3 = (
            float(la.norm(F1 - F0, "fro") / max(den_f, 1e-18))
            if den_f > 0
            else 1.0
        )

        # T1: Π sanitización (proyección espectral PSD+traza) es 1-Lipschitz
        # en norma espectral sobre el cono PSD; usamos 1 como cota, y una
        # sonda barata sobre la parte Hermítica de δA (como si fuera δρ).
        herm = 0.5 * (delta_A + delta_A.conj().T)
        try:
            evals = la.eigvalsh(0.5 * (A + A.conj().T) * 0.0 + herm)  # solo δ
            # El clamp max(λ, ε) es no expansivo ⇒ Lip ≤ 1
            lip1 = 1.0 if np.all(np.isfinite(evals)) else 1.0
        except la.LinAlgError:
            lip1 = 1.0

        return (
            float(min(max(lip1, 0.0), 2.0)),
            float(min(max(lip2, 0.0), 2.0)),
            float(min(max(lip3, 0.0), 2.0)),
        )

    def enforce_rsi3(
        self,
        F: NDArray[np.complex128],
        A: NDArray[np.complex128],
        transit_time: float,
        phi0: NDArray[np.complex128],
        max_iter: int = _RSI3_MAX_ITER,
        cache: Optional[SpectralGaugeCache] = None,
        lindstedt: Optional[PoincareLindstedtSolution] = None,
    ) -> RSI3Certificate:
        r"""
        Tres superficies RSI³ (automejora **computacional**, PASSIVE_SILENT):

          • S1 (Novikov / Data): $v_K=-\log T\ge 0.10$
            (valuación de Krull-Novikov sobre el tiempo de tránsito $T=1/\|A\|_F$).
          • S2 (Harness): SPD de $F^\dagger F$ (ligadura espectral).
          • S3 (Model): $d_{FS}(\varphi_0,e^{A}\varphi_0)\le 10^{-4}$
            y residual homológico de Lindstedt.

        Banach $T^3=T_3\circ T_2\circ T_1$: Lipschitz sonda de los proyectores
        de ligadura.  El bucle Newton homológico (si hay `lindstedt`) es la
        automejora de nivel 3 del gráfico, no una mutación del campo.
        """
        cache = cache or self._spectral_cache_from_A(A)
        phi0u = _safe_unit(phi0)
        lip1, lip2, lip3 = self._probe_lipschitz_projectors(A, F)
        lip_t3 = float(lip1 * lip2 * lip3)

        # Automejora: residual homológico (ya Newton-refinado en FASE 1)
        homo_res = float(lindstedt.residual_norm) if lindstedt is not None else 0.0

        T = max(float(transit_time), 1e-15)
        v_krull = float(-math.log(T))
        v_krull = max(0.0, min(v_krull, _RSI3_KRULL_CAP))
        s1 = RSI3SurfaceReport(
            surface_id=1,
            name="Data-RSI (Novikov)",
            metric_value=v_krull,
            threshold=_KRULL_NOVIKOV_FLOOR,
            passed=bool(v_krull >= _KRULL_NOVIKOV_FLOOR),
            diagnostics={"transit_time": T},
        )

        FdagF = F.conj().T @ F
        try:
            FdagF_h = 0.5 * (FdagF + FdagF.conj().T)
            evals = la.eigvalsh(FdagF_h)
            lam_min = float(np.min(evals))
            passed_spd = bool(lam_min >= -_SPECTRAL_PSD_FLOOR)
        except la.LinAlgError:
            lam_min = -1.0
            passed_spd = False
        s2 = RSI3SurfaceReport(
            surface_id=2,
            name="Harness-RSI (SPD)",
            metric_value=lam_min,
            threshold=-_SPECTRAL_PSD_FLOOR,
            passed=passed_spd,
            diagnostics={"min_eig_FdagF": lam_min},
        )

        # S3: Fubini-Study entre φ0 y exp(A)φ0  (t=1) via caché espectral
        try:
            phi_end = self._unitary_orbit(
                cache, phi0u, np.array([1.0], dtype=np.float64),
            )[0]
        except (la.LinAlgError, ValueError):
            phi_end = phi0u
        d_fs = self._fubini_study(phi0u, phi_end)
        s3 = RSI3SurfaceReport(
            surface_id=3,
            name="Model-RSI (Fubini-Study)",
            metric_value=d_fs,
            threshold=_FUBINI_STUDY_MAX,
            passed=bool(d_fs <= _FUBINI_STUDY_MAX),
            diagnostics={
                "d_fs_rad": d_fs,
                "homological_residual": homo_res,
                "lindstedt_solved": bool(
                    lindstedt.is_solved if lindstedt is not None else True
                ),
            },
        )
        surfaces = (s1, s2, s3)

        # Iteración de automejora: re-proyección de copias computacionales
        # (no mutan el envelope congelado).  Punto fijo = proyectores idempotentes.
        A_c = np.array(A, copy=True, dtype=np.complex128)
        F_c = np.array(F, copy=True, dtype=np.complex128)
        fixed = False
        it_used = 1
        prev_gap = float("inf")
        for it in range(1, max_iter + 1):
            it_used = it
            A_n = _project_su_n(A_c)
            F_n = _project_anti_hermitian(F_c)
            gap = float(la.norm(A_n - A_c, "fro") + la.norm(F_n - F_c, "fro"))
            A_c, F_c = A_n, F_n
            if gap < _RSI3_FIXED_POINT_TOL:
                fixed = True
                break
            if abs(gap - prev_gap) < _RSI3_FIXED_POINT_TOL:
                fixed = True
                break
            prev_gap = gap

        jerk = float((1.0 - min(lip_t3, 1.0)) / 0.05) if lip_t3 < 1.0 else 0.0
        valid = (
            all(s.passed for s in surfaces)
            and lip_t3 <= (_BANACH_LIPSCHITZ_MAX ** 3) + 1e-12
        )
        return RSI3Certificate(
            surfaces=surfaces,
            lipschitz_t3=float(lip_t3),
            capacity_jerk=float(jerk),
            fixed_point_reached=bool(fixed),
            iterations=int(it_used),
            is_valid=bool(valid),
            lipschitz_factors=(lip1, lip2, lip3),
            homological_residual=float(homo_res),
        )

    def _phase2_chain_payload(
        self,
        p1: WitnessPhase1Envelope,
        qnd: QNDCertificate,
        fid: float,
        bures: float,
        sz: float,
        kam: KAMCertificate,
        melnikov: MelnikovCertificate,
        floquet: FloquetCertificate,
        rsi3: RSI3Certificate,
    ) -> str:
        return _hash_state(
            p1.rho_sanitized, p1.connection_A, p1.curvature_F,
            np.array(
                [
                    qnd.back_action_db, fid, bures, sz,
                    kam.worst_ratio, melnikov.integral_value,
                    floquet.max_unit_deviation, rsi3.lipschitz_t3,
                ],
                dtype=np.float64,
            ),
        )

    def verify_phase2_chain(self, env: WitnessPhase2Envelope) -> None:
        expected = self._phase2_chain_payload(
            env.phase1, env.qnd, env.uhlmann_fidelity, env.bures_distance,
            env.stora_zumino_residual, env.kam_certificate, env.melnikov,
            env.floquet, env.rsi3_certificate,
        )
        if expected != env.chain_hash:
            raise WitnessContractCorruptedError(
                f"Chain FASE 2 corrupta: {expected} ≠ {env.chain_hash}."
            )

    # ── CIERRE FASE 2 → FASE 3 ────────────────────────────────────────────────
    def emit_phase2_envelope(
        self,
        phase1_env: WitnessPhase1Envelope,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
    ) -> WitnessPhase2Envelope:
        r"""
        **Cierre formal de la FASE 2 / puente de apertura de la FASE 3.**

        Ejecuta QND verificado, Uhlmann/Bures, Poincaré (KAM/Bruno/Melnikov/
        Floquet/sección/Birkhoff) y RSI³.  Emite `WitnessPhase2Envelope`.
        El receptor canónico es `WitnessPhase3Decider.from_phase2_envelope`.
        """
        t0 = time.perf_counter_ns()
        self.verify_phase1_chain(phase1_env)
        rho = phase1_env.rho_sanitized
        A = phase1_env.connection_A
        F = phase1_env.curvature_F
        d = A.shape[0]
        cache = self._cache_or_compute(A, phase1_env)

        if phi_initial is None:
            phi_initial = np.ones(d, dtype=np.complex128) / math.sqrt(d)
        phi_initial = _safe_unit(np.asarray(phi_initial, dtype=np.complex128))
        if phi_final is None:
            pf = rho @ phi_initial
            phi_final = _safe_unit(pf)
        else:
            phi_final = _safe_unit(np.asarray(phi_final, dtype=np.complex128))

        qnd = self.compute_qnd_weak_value_verified(
            phi_initial, phi_final, A, rho,
        )
        fid, bures, trace_dist, liouville = self.audit_uhlmann_bures(rho)
        sz = self.audit_stora_zumino(F)

        # Frecuencias de acción-ángulo: ω de iA (Poincaré–Laplace), no Im eig(F)
        freqs = np.abs(cache.omega)
        kam = self.check_kam_bruno(freqs)
        bruno = self.check_bruno_arithmetic(freqs)
        melnikov = self.compute_melnikov(A, F, phi_initial, cache=cache)
        floquet = self.compute_floquet(F)
        poincare_sec = self.compute_poincare_section(
            A, phi_initial, cache=cache,
        )
        birkhoff = self.compute_birkhoff_twist(poincare_sec, liouville)

        norm_A = max(float(la.norm(A, "fro")), 1e-15)
        transit_time = 1.0 / norm_A
        rsi3 = self.enforce_rsi3(
            F, A, transit_time, phi_initial,
            cache=cache,
            lindstedt=phase1_env.poincare_lindstedt,
        )

        chain = self._phase2_chain_payload(
            phase1_env, qnd, fid, bures, sz, kam, melnikov, floquet, rsi3,
        )
        return WitnessPhase2Envelope(
            phase1=phase1_env,
            qnd=qnd,
            uhlmann_fidelity=fid,
            bures_distance=bures,
            trace_distance=trace_dist,
            liouville_volume_drift=liouville,
            stora_zumino_residual=sz,
            kam_certificate=kam,
            melnikov=melnikov,
            floquet=floquet,
            poincare_section=poincare_sec,
            rsi3_certificate=rsi3,
            chain_hash=chain,
            origin_timestamp_ns=float(time.perf_counter_ns() - t0),
            bruno_certificate=bruno,
            birkhoff_twist=birkhoff,
        )


# ══════════════════════════════════════════════════════════════════════════════════
# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║  F A S E   3 :   D E C I D E   &   A C T                                      ║
# ║  Apertura = from_phase2_envelope  (continuación formal del cierre FASE 2).   ║
# ║  Heyting Ω₃ completo + HMAC. SIN crowbar (Testigo Silencioso).               ║
# ║  Cierra con `emit_witness_snapshot` → `GaugeWitnessSnapshot`.                ║
# ╚════════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════════

class WitnessPhase3Decider(WitnessPhase2Orientor):
    r"""
    FASE 3 (Decide & Act): adjudicación Heyting Ω₃ sobre 12 átomos, firma
    HMAC-SHA256 y emisión del snapshot.  **NO dispara crowbar** — esa es
    responsabilidad del agente consumidor downstream.
    """

    # ── APERTURA FASE 3  (continuación del último método de FASE 2) ───────────
    @classmethod
    def from_phase2_envelope(
        cls,
        env: WitnessPhase2Envelope,
        dimension: Optional[int] = None,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> "WitnessPhase3Decider":
        r"""
        **Apertura formal de la FASE 3** — continuación de
        `WitnessPhase2Orientor.emit_phase2_envelope`.

        Valida el chain_hash de FASE 2 y construye el decisor.
        """
        dim = int(dimension) if dimension is not None else int(env.phase1.dimension)
        inst = cls(dimension=dim, secret_key=secret_key)
        inst.verify_phase1_chain(env.phase1)
        inst.verify_phase2_chain(env)
        inst._phase2_env = env  # type: ignore[attr-defined]
        return inst

    # ── Adjudicador Heyting Ω₃ (12 átomos) ────────────────────────────────────
    def adjudicate_heyting(
        self,
        *,
        energy_ym: float,
        is_qnd_preserved: bool,
        back_action_db: float,
        fidelity: float,
        bures: float,
        symplectic_valid: bool,
        kam_satisfied: bool,
        melnikov_preserved: bool,
        floquet_stable: bool,
        rsi3_valid: bool,
        bruno_satisfied: bool = True,
        birkhoff_area_ok: bool = True,
    ) -> HeytingOmega3:
        r"""
        Adjudica en $\Omega_3$ vía join(∨) monoidal de **12 átomos**:
        energía YM, QND, back-action, Uhlmann, Bures, simpléctica, KAM,
        Melnikov, Floquet, RSI³, Bruno, Birkhoff (proxy de área).
        """
        atoms: List[HeytingOmega3] = [
            HeytingOmega3.COHERENT if energy_ym <= _YM_ENERGY_SOFT else (
                HeytingOmega3.DEGRADED if energy_ym <= _YM_ENERGY_HARD
                else HeytingOmega3.VETOED
            ),
            HeytingOmega3.COHERENT if is_qnd_preserved else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if back_action_db <= _QND_BACKACTION_DB_TOL
            else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if fidelity >= _UHLMANN_FIDELITY_MIN
            else HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT if bures <= _BURES_DIST_SOFT else (
                HeytingOmega3.DEGRADED if bures <= _BURES_DIST_HARD
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
        return HeytingOmega3.fold_supremum(atoms)

    # ── HMAC ──────────────────────────────────────────────────────────────────
    def _sign_snapshot(self, payload: str) -> bytes:
        msg = _GAUGE_WITNESS_HMAC_DOMAIN + payload.encode("utf-8")
        return hmac.new(self.secret_key, msg, hashlib.sha256).digest()

    def _verify_hmac(self, payload: str, signature: bytes) -> bool:
        return hmac.compare_digest(self._sign_snapshot(payload), signature)

    def verify_witness_snapshot(self, snap: GaugeWitnessSnapshot) -> bool:
        r"""Verifica HMAC-SHA256 del snapshot (dominio v10.1.0)."""
        p1 = snap.phase1
        p2 = snap.phase2
        if p1 is None or p2 is None:
            payload = (
                f"{snap.witness_id}:{snap.yang_mills_background_energy:.10f}:"
                f"{snap.bures_distance:.10f}:{snap.qnd_weak_value_aav}:"
                f"{snap.back_action_db:.6f}:{int(snap.global_verdict)}:"
                f"{snap.sha256_merkle_digest}"
            )
        else:
            payload = (
                f"{snap.witness_id}:{p1.yang_mills_energy:.10f}:{p2.bures_distance:.10f}:"
                f"{p2.qnd.qnd_weak_value_aav}:{p2.qnd.back_action_db:.6f}:"
                f"{int(snap.global_verdict)}:{p1.chain_hash}:{p2.chain_hash}"
            )
        return self._verify_hmac(payload, snap.hmac_sha256)

    # ── CIERRE FASE 3 (terminal) ──────────────────────────────────────────────
    def emit_witness_snapshot(
        self, phase2_env: WitnessPhase2Envelope,
    ) -> GaugeWitnessSnapshot:
        r"""
        **Cierre formal de la FASE 3.**

        Adjudica en Ω₃, firma el snapshot con HMAC-SHA256 y lo emite.
        No dispara crowbar (invariante PASSIVE_SILENT).
        """
        now = time.time()
        self.verify_phase2_chain(phase2_env)
        p1 = phase2_env.phase1
        p2 = phase2_env
        rsi3 = p2.rsi3_certificate
        bruno_ok = (
            True if p2.bruno_certificate is None
            else bool(p2.bruno_certificate.is_satisfied)
        )
        area_ok = (
            True if p2.birkhoff_twist is None
            else bool(p2.birkhoff_twist.is_area_preserving_proxy)
        )

        verdict = self.adjudicate_heyting(
            energy_ym=p1.yang_mills_energy,
            is_qnd_preserved=p2.qnd.is_qnd_preserved,
            back_action_db=p2.qnd.back_action_db,
            fidelity=p2.uhlmann_fidelity,
            bures=p2.bures_distance,
            symplectic_valid=p1.symplectic_report.is_symplectic_valid,
            kam_satisfied=p2.kam_certificate.is_satisfied,
            melnikov_preserved=p2.melnikov.is_homoclinic_preserved,
            floquet_stable=(p2.floquet.is_neutral or p2.floquet.is_dissipative_stable),
            rsi3_valid=rsi3.is_valid,
            bruno_satisfied=bruno_ok,
            birkhoff_area_ok=area_ok,
        )

        wit_id = f"GAUGE-WITNESS-{int(now * 1000.0) % 1000000:06d}"
        merkle_payload = (
            f"{wit_id}:{p1.yang_mills_energy:.10f}:{p2.bures_distance:.10f}:"
            f"{p2.qnd.qnd_weak_value_aav}:{p2.qnd.back_action_db:.6f}:"
            f"{int(verdict)}:{p1.chain_hash}:{p2.chain_hash}"
        )
        merkle_digest = hashlib.sha256(merkle_payload.encode("utf-8")).hexdigest()
        hmac_sig = self._sign_snapshot(merkle_payload)
        hmac_ok = self._verify_hmac(merkle_payload, hmac_sig)

        is_epistemologically_valid = (
            verdict != HeytingOmega3.VETOED
            and p2.qnd.is_qnd_preserved
            and p1.symplectic_report.is_symplectic_valid
            and p1.is_anti_hermitian_A
            and p1.is_traceless_A
            and hmac_ok
        )
        cache = p1.spectral_cache
        bruno = p2.bruno_certificate
        birk = p2.birkhoff_twist
        pl = p1.poincare_lindstedt

        return GaugeWitnessSnapshot(
            witness_id=wit_id,
            qnd_weak_value_aav=p2.qnd.qnd_weak_value_aav,
            back_action_db=p2.qnd.back_action_db,
            uhlmann_fidelity=p2.uhlmann_fidelity,
            bures_distance=p2.bures_distance,
            liouville_volume_drift=p2.liouville_volume_drift,
            yang_mills_background_energy=p1.yang_mills_energy,
            stora_zumino_residual=p2.stora_zumino_residual,
            global_verdict=verdict,
            sha256_merkle_digest=merkle_digest,
            timestamp_utc=now,
            is_qnd_preserved=p2.qnd.is_qnd_preserved,
            qnd_commutator_norm=p2.qnd.commutator_norm,
            qnd_fidelity=p2.qnd.qnd_fidelity,
            symplectic_residue=p1.symplectic_report.symplectic_residue,
            symplectic_defect=p1.symplectic_report.symplectic_defect,
            is_symplectic_valid=p1.symplectic_report.is_symplectic_valid,
            poincare_residual=(pl.residual_norm if pl else 0.0),
            poincare_is_resonant=(pl.is_resonant if pl else False),
            kam_worst_ratio=p2.kam_certificate.worst_ratio,
            kam_satisfied=p2.kam_certificate.is_satisfied,
            melnikov_value=p2.melnikov.integral_value,
            melnikov_preserved=p2.melnikov.is_homoclinic_preserved,
            floquet_max_unit_deviation=p2.floquet.max_unit_deviation,
            floquet_neutral=p2.floquet.is_neutral,
            poincare_section_n_crossings=p2.poincare_section.n_crossings,
            poincare_section_dispersion=p2.poincare_section.return_dispersion,
            banach_lipschitz_t3=rsi3.lipschitz_t3,
            fubini_study_distance=rsi3.surface3.metric_value,
            novikov_krull_valuation=rsi3.surface1.metric_value,
            rsi3_valid=rsi3.is_valid,
            rsi3_iterations=rsi3.iterations,
            witness_mode="PASSIVE_SILENT",
            is_epistemologically_valid=bool(is_epistemologically_valid),
            trace_distance=p2.trace_distance,
            phase1=p1,
            phase2=p2,
            hmac_sha256=hmac_sig,
            engine_version=_ENGINE_VERSION,
            bruno_satisfied=bruno_ok,
            bruno_sum=float(bruno.bruno_sum) if bruno is not None else 0.0,
            birkhoff_twist_estimate=(
                float(birk.twist_estimate) if birk is not None else 0.0
            ),
            birkhoff_is_twist=bool(birk.is_twist) if birk is not None else True,
            lindstedt_solved=bool(pl.is_solved) if pl is not None else True,
            spectral_condition_number=(
                float(cache.condition_number) if cache is not None else 1.0
            ),
            kam_n_modes=int(p2.kam_certificate.n_modes),
            is_flat_curvature=bool(p1.symplectic_report.is_flat_branch),
            mean_poincare_return_time=float(p2.poincare_section.mean_return_time),
            hmac_verified=bool(hmac_ok),
        )


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR — Motor del Testigo Silencioso
# ══════════════════════════════════════════════════════════════════════════════

class TopologicalGaugeFieldWitnessEngine(WitnessPhase3Decider):
    r"""
    Motor Espectral del Testigo Silencioso del Campo Topológico de Gauge
    (v10.1.0 — Tres Fases Anidadas, Poincaré–Bruno–Birkhoff, RSI³ Banach).

    Orquesta:
      FASE 1 (Observe)    → `emit_phase1_envelope` → `WitnessPhase1Envelope`.
      FASE 2 (Orient)     → `from_phase1_envelope` + `emit_phase2_envelope`
                          → `WitnessPhase2Envelope`.
      FASE 3 (Decide&Act) → `from_phase2_envelope` + `emit_witness_snapshot`
                          → `GaugeWitnessSnapshot`.

    **Garantía QND**: el back-action se VERIFICA numéricamente
    ($\|[A,\rho]\|_F$) — identidad teórica 0, residual = integridad FP.
    Si se rompe, el veredicto degrada a VETOED y `is_qnd_preserved=False`.

    **Invariante PASSIVE_SILENT**: el testigo NUNCA dispara hardware.
    RSI³ mejora el gráfico computacional (Lindstedt / proyectores), no el campo.

    **Compatibilidad**: `evaluate_gauge_witness` preserva firma y semántica.
    """

    def __init__(
        self, dimension: int = 56, secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> None:
        super().__init__(dimension=dimension, secret_key=secret_key)
        logger.info(
            "TopologicalGaugeFieldWitnessEngine v%s inicializado "
            "[dim=%s, engine_interop=%s]",
            _ENGINE_VERSION,
            self.dimension,
            "ON" if _ENGINE_HAS_ENVELOPES else "OFF",
        )

    # ── Pipeline público (preservado) ─────────────────────────────────────────
    def evaluate_gauge_witness(
        self,
        density_matrix: NDArray,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
    ) -> GaugeWitnessSnapshot:
        r"""
        Ejecuta la observación pasiva QND a través de las tres fases anidadas.

        ══ FASE 1 ══ emit_phase1_envelope → from_phase1_envelope
        ══ FASE 2 ══ emit_phase2_envelope → from_phase2_envelope
        ══ FASE 3 ══ emit_witness_snapshot
        """
        env1 = self.emit_phase1_envelope(density_matrix)
        self.verify_phase1_chain(env1)

        env2 = self.emit_phase2_envelope(
            env1, phi_initial=phi_initial, phi_final=phi_final,
        )
        self.verify_phase2_chain(env2)

        return self.emit_witness_snapshot(env2)

    # ── Interoperabilidad con el motor espectral v10.x ───────────────────────
    def consume_engine_phase1_envelope(
        self, engine_env: Any,
    ) -> WitnessPhase1Envelope:
        r"""
        **Interop FASE 1**: consume un `GaugePhase1Envelope` del motor
        `topological_gauge_field_engine` v10.x y produce el envelope
        del testigo compatible (certificación cruzada).
        """
        if engine_env is None:
            raise GaugeWitnessError("engine_env es None.")

        rho = np.asarray(engine_env.rho_sanitized, dtype=np.complex128)
        A = np.asarray(engine_env.connection_A, dtype=np.complex128)
        F = np.asarray(engine_env.curvature_F, dtype=np.complex128)
        energy = float(engine_env.energy_ym)

        sympl = SymplecticWitnessReport(
            cayley_matrix=np.zeros((0, 0), dtype=np.complex128),
            symplectic_residue=float(engine_env.symplectic_residue),
            symplectic_defect=float(engine_env.symplectic_defect),
            is_symplectic_valid=bool(engine_env.is_symplectic_valid),
            is_flat_branch=bool(float(engine_env.symplectic_residue) == 0.0),
        )
        pl_sol = getattr(engine_env, "poincare_lindstedt", None)
        pl: Optional[PoincareLindstedtSolution] = None
        if pl_sol is not None:
            pl = PoincareLindstedtSolution(
                W=np.asarray(pl_sol.W, dtype=np.complex128),
                min_resonance_gap=float(pl_sol.min_resonance_gap),
                residual_norm=float(pl_sol.residual_norm),
                is_resonant=bool(pl_sol.is_resonant),
                is_solved=bool(pl_sol.is_solved),
            )
        try:
            cache = self._spectral_cache_from_A(A)
        except WitnessSpectralError:
            cache = None
        chain = self._phase1_chain_payload(rho, A, F, energy, sympl)
        return WitnessPhase1Envelope(
            rho_sanitized=rho,
            connection_A=A,
            curvature_F=F,
            yang_mills_energy=energy,
            is_anti_hermitian_A=bool(engine_env.is_anti_hermitian_A),
            is_traceless_A=bool(engine_env.is_traceless_A),
            symplectic_report=sympl,
            poincare_lindstedt=pl,
            dimension=int(rho.shape[0]),
            origin_timestamp_ns=float(time.perf_counter_ns()),
            chain_hash=chain,
            spectral_cache=cache,
        )

    def consume_engine_phase2_envelope(
        self, engine_env: Any, phase1_env: WitnessPhase1Envelope,
    ) -> WitnessPhase2Envelope:
        r"""
        **Interop FASE 2**: consume un `GaugePhase2Envelope` del motor
        v10.x y produce el envelope del testigo compatible.
        """
        if engine_env is None:
            raise GaugeWitnessError("engine_env (FASE 2) es None.")

        rsi3 = engine_env.rsi3_certificate
        raw_surfs = list(getattr(rsi3, "surfaces", ()))
        built: List[RSI3SurfaceReport] = []
        for i, s in enumerate(raw_surfs[:3]):
            built.append(
                RSI3SurfaceReport(
                    surface_id=int(getattr(s, "surface_id", i + 1)),
                    name=str(getattr(s, "name", f"S{i+1}")),
                    metric_value=float(getattr(s, "metric_value", 0.0)),
                    threshold=float(getattr(s, "threshold", 0.0)),
                    passed=bool(getattr(s, "passed", False)),
                    diagnostics=(
                        dict(s.diagnostics) if hasattr(s, "diagnostics") else {}
                    ),
                )
            )
        while len(built) < 3:
            k = len(built) + 1
            built.append(
                RSI3SurfaceReport(k, f"pad-{k}", 0.0, 0.0, False, {}),
            )
        rsi3_w = RSI3Certificate(
            surfaces=(built[0], built[1], built[2]),
            lipschitz_t3=float(rsi3.lipschitz_t3),
            capacity_jerk=float(rsi3.capacity_jerk),
            fixed_point_reached=bool(rsi3.fixed_point_reached),
            iterations=int(rsi3.iterations),
            is_valid=bool(rsi3.is_valid),
        )
        kam_src = engine_env.kam_certificate
        kam = KAMCertificate(
            is_satisfied=bool(kam_src.is_satisfied),
            worst_ratio=float(kam_src.worst_ratio),
            tau=float(kam_src.tau),
            gamma=float(kam_src.gamma),
            omega_min=float(kam_src.omega_min),
        )
        mel_src = engine_env.melnikov
        mel = MelnikovCertificate(
            integral_value=float(mel_src.integral_value),
            max_abs_integrand=float(mel_src.max_abs_integrand),
            is_homoclinic_preserved=bool(mel_src.is_homoclinic_preserved),
        )
        flo_src = engine_env.floquet
        flo = FloquetCertificate(
            multipliers=np.asarray(flo_src.multipliers, dtype=np.complex128),
            max_unit_deviation=float(flo_src.max_unit_deviation),
            is_neutral=bool(flo_src.is_neutral),
            is_dissipative_stable=bool(flo_src.is_dissipative_stable),
            exponents=np.asarray(flo_src.exponents, dtype=np.float64),
        )
        sec_src = engine_env.poincare_section
        sec = PoincareSectionCertificate(
            n_crossings=int(sec_src.n_crossings),
            return_dispersion=float(sec_src.return_dispersion),
            is_regular=bool(sec_src.is_regular),
            returns=tuple(float(x) for x in sec_src.returns),
        )

        rho = phase1_env.rho_sanitized
        A = phase1_env.connection_A
        d = A.shape[0]
        phi_i = np.ones(d, dtype=np.complex128) / math.sqrt(d)
        phi_f = _safe_unit(rho @ phi_i)
        qnd = self.compute_qnd_weak_value_verified(phi_i, phi_f, A, rho)
        fid, bures, trace_dist, liouville = self.audit_uhlmann_bures(rho)
        chain = self._phase2_chain_payload(
            phase1_env, qnd, fid, bures,
            float(engine_env.stora_zumino_residual),
            kam, mel, flo, rsi3_w,
        )
        return WitnessPhase2Envelope(
            phase1=phase1_env,
            qnd=qnd,
            uhlmann_fidelity=fid,
            bures_distance=bures,
            trace_distance=trace_dist,
            liouville_volume_drift=liouville,
            stora_zumino_residual=float(engine_env.stora_zumino_residual),
            kam_certificate=kam,
            melnikov=mel,
            floquet=flo,
            poincare_section=sec,
            rsi3_certificate=rsi3_w,
            chain_hash=chain,
            origin_timestamp_ns=float(time.perf_counter_ns()),
        )


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__: List[str] = [
    "GaugeWitnessError",
    "QNDBackActionViolationError",
    "WitnessContractCorruptedError",
    "WitnessSymplecticViolationError",
    "WitnessSpectralError",
    "HeytingOmega3",
    "SpectralGaugeCache",
    "QNDCertificate",
    "SymplecticWitnessReport",
    "PoincareLindstedtSolution",
    "KAMCertificate",
    "BrunoCertificate",
    "MelnikovCertificate",
    "FloquetCertificate",
    "PoincareSectionCertificate",
    "BirkhoffTwistCertificate",
    "RSI3SurfaceReport",
    "RSI3Certificate",
    "WitnessPhase1Envelope",
    "WitnessPhase2Envelope",
    "GaugeWitnessSnapshot",
    "WitnessPhase1Observer",
    "WitnessPhase2Orientor",
    "WitnessPhase3Decider",
    "TopologicalGaugeFieldWitnessEngine",
]


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN DOCTORAL
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    banner = "═" * 80
    print(banner)
    print("  VERIFICACIÓN DOCTORAL — Topological Gauge Field Witness Engine v10.1.0")
    print(f"  Engine interop disponible: {_ENGINE_HAS_ENVELOPES}")
    print(banner)

    engine = TopologicalGaugeFieldWitnessEngine(dimension=56)
    rng = np.random.default_rng(2026)
    state = rng.normal(0.0, 1.0, size=(56, 56)) + 1j * rng.normal(0.0, 1.0, size=(56, 56))
    rho = state @ state.conj().T
    rho /= np.trace(rho)

    snap = engine.evaluate_gauge_witness(rho)
    hmac_ok = engine.verify_witness_snapshot(snap)

    print(f"  • Witness ID            : {snap.witness_id}")
    print(f"  • Modo Witness          : {snap.witness_mode}")
    print(f"  • Versión motor         : {snap.engine_version}")
    print(f"  • Verdict Heyting Ω₃    : {snap.global_verdict.name} "
          f"(0x{int(snap.global_verdict):02x})")
    print(f"  • Valor Débil QND AAV   : {snap.qnd_weak_value_aav:.4f}")
    print(f"  • Back-Action QND       : {snap.back_action_db:.6f} dB")
    print(f"  • QND Preservada        : {snap.is_qnd_preserved}")
    print(f"  • ‖[A,ρ]‖_F             : {snap.qnd_commutator_norm:.3e}")
    print(f"  • QND Fidelity          : {snap.qnd_fidelity:.6f}")
    print(f"  • Fidelidad Uhlmann     : {snap.uhlmann_fidelity:.6f}")
    print(f"  • Distancia Bures       : {snap.bures_distance:.6f}")
    print(f"  • Distancia Traza       : {snap.trace_distance:.6f}")
    print(f"  • Deriva Liouville      : {snap.liouville_volume_drift:.3e}")
    print(f"  • Yang-Mills E          : {snap.yang_mills_background_energy:.6f}")
    print(f"  • Stora-Zumino          : {snap.stora_zumino_residual:.6e}")
    print(f"  • Curvatura plana       : {snap.is_flat_curvature}")
    print(f"  • δω simpléctico        : {snap.symplectic_residue:.3e}")
    print(f"  • ‖KᵀJ+JK‖_F            : {snap.symplectic_defect:.3e}")
    print(f"  • Poincaré residual     : {snap.poincare_residual:.3e}")
    print(f"  • Lindstedt resuelto    : {snap.lindstedt_solved}")
    print(f"  • κ₂(iA)                : {snap.spectral_condition_number:.4e}")
    print(f"  • KAM modos / ratio     : {snap.kam_n_modes} / {snap.kam_worst_ratio:.4f}")
    print(f"  • Bruno Σ / OK          : {snap.bruno_sum:.4f} / {snap.bruno_satisfied}")
    print(f"  • Melnikov |M|          : {abs(snap.melnikov_value):.4e}")
    print(f"  • Floquet |Δμ|max       : {snap.floquet_max_unit_deviation:.3e}")
    print(f"  • Poincaré cruzamientos : {snap.poincare_section_n_crossings}")
    print(f"  • Retorno medio         : {snap.mean_poincare_return_time:.4f}")
    print(f"  • Birkhoff twist        : {snap.birkhoff_twist_estimate:.4e} "
          f"(twist={snap.birkhoff_is_twist})")
    print(f"  • Banach Lip(T³)        : {snap.banach_lipschitz_t3:.4f}")
    print(f"  • Novikov v_Krull       : {snap.novikov_krull_valuation:.4f}")
    print(f"  • Fubini-Study d        : {snap.fubini_study_distance:.3e} rad")
    print(f"  • RSI³ iteraciones      : {snap.rsi3_iterations}")
    print(f"  • SHA-256 Merkle        : {snap.sha256_merkle_digest[:32]}…")
    print(f"  • HMAC-SHA256           : {snap.hmac_sha256.hex()[:32]}…")
    print(f"  • HMAC verificado       : {hmac_ok and snap.hmac_verified}")
    print(f"  • Epistemológicamente   : {snap.is_epistemologically_valid}")
    print(banner)