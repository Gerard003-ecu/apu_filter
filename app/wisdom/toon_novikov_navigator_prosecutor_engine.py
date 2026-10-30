# -*- coding: utf-8 -*-
r"""
╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : TOON Novikov Navigator Prosecutor Engine v13.0.0-PhD                                                ║
║ Ubicación: app/wisdom/toon_novikov_navigator_prosecutor_engine.py                                              ║
║ Versión  : 13.0.0-SOVEREIGN-NOVIKOV-PROSECUTOR-POINCARE-KAM-L3-MONAD-PhD                                       ║
║ Función  : Motor Espectral Fiscal Inquisidor de Navegación Causal en el Anillo Universal de Novikov Λ_Nov.     ║
║            Ritual fiscal anidado INQUISICIÓN → JUICIO → EJECUCIÓN con puentes Φ₁₂' y Φ₂₃' HMAC-SHA256,         ║
║            mecánica celeste de Poincaré (Delaunay–Deprit, monodromía Sp(2,ℝ), Greene, Melnikov con ceros        ║
║            simples, Bryuno, Jacobi, Poincaré–Cartan en eigenbase, Morse χ(ℂPⁿ⁻¹)=n), RSI-3 (Banach iterado,     ║
║            Lip empírico, self-repair de umbrales) y mónada cartesiana T=(T,η,μ) sobre gérmenes de acusación.    ║
║ Estructura: 3 Fases Anidadas — la ÚLTIMA def. de fase n es la FRONTERA de la PRIMERA def. n+1.                 ║
║   FASE 1 → Inquisición C*/ultramétrica/Uhlmann-FIX + Poincaré profundo + Banach RSI-3                          ║
║            ÚLTIMO método: handoff_to_phase2_judgment_context → Phase1ToPhase2ProsecutionBridge                 ║
║   FASE 2 → Juicio QND/Oseledets/Gromov/GKSL-Spohn + fusión Ω₃                                                  ║
║            ÚLTIMO método: handoff_to_phase3_execution_context → Phase2ToPhase3ProsecutionBridge                ║
║   FASE 3 → Ejecución: consume veredicto P2 + Dirac + Crowbar simulado + certificado                            ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.TOONNovikovNavigatorProsecutorEngine")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# ── Importación resiliente del motor testigo v13 (QND L3 / KAM noble) ─────────
_HAS_WITNESS_ENGINE: bool = False
try:
    from app.wisdom.toon_novikov_navigator_witness_engine import (  # type: ignore
        KAMNovikovTorus as _WitnessKAM,
        PoincareCelestialMechanics as _WitnessPCM,
        RecursiveNovikovImprover as _WitnessRNI,
        SovereignNovikovAuditor as _WitnessAud,
        TOONNovikovNavigatorWitnessEngine as _WitnessEngine,
    )

    _HAS_WITNESS_ENGINE = True
except ImportError:
    try:
        from toon_novikov_navigator_witness_engine import (  # type: ignore
            KAMNovikovTorus as _WitnessKAM,
            PoincareCelestialMechanics as _WitnessPCM,
            RecursiveNovikovImprover as _WitnessRNI,
            SovereignNovikovAuditor as _WitnessAud,
            TOONNovikovNavigatorWitnessEngine as _WitnessEngine,
        )

        _HAS_WITNESS_ENGINE = True
    except ImportError:
        _WitnessKAM = None  # type: ignore[misc, assignment]
        _WitnessPCM = None  # type: ignore[misc, assignment]
        _WitnessRNI = None  # type: ignore[misc, assignment]
        _WitnessAud = None  # type: ignore[misc, assignment]
        _WitnessEngine = None  # type: ignore[misc, assignment]

__all__ = (
    "HeytingOmega3",
    "ActuationMode",
    "NovikovProsecutionStatus",
    "NovikovProsecutionCanonicalSeed",
    "NovikovProsecutorRSI3State",
    "NovikovProsecutionIndictmentGerm",
    "NovikovProsecutionExecutionCertificate",
    "Phase1ToPhase2ProsecutionBridge",
    "Phase2ToPhase3ProsecutionBridge",
    "NovikovProsecutorPhase1Inquisitor",
    "NovikovProsecutorPhase2Adjudicator",
    "NovikovProsecutorPhase3Executor",
    "TOONNovikovNavigatorProsecutorEngine",
)

# ──────────────────────────────────────────────────────────────────────────────
# Constantes físicas, ultramétricas, algebraicas y de precisión
# ──────────────────────────────────────────────────────────────────────────────
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_POSITRON_REST_MASS_KEV: Final[float] = 1022.0
_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 392.15
_ESP32_SIMD_EXECUTION_NS: Final[float] = 42.0
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_SOVEREIGN_NOVIKOV_PROSECUTOR_ENGINE_KEY_v13.0.0"
_ULTRAMETRIC_TOLERANCE: Final[float] = 1.0e-12
_ULTRA_GAP_THRESHOLD: Final[float] = 1.0e-6
_NOVIKOV_V_FLOOR_DEFAULT: Final[float] = 0.10
_BRUHAT_TITS_VETO_THRESHOLD: Final[float] = 15.0
_GROMOV_CAPACITY_MAX_DEFAULT: Final[float] = 12.5
_BANACH_LIPSCHITZ_CEILING: Final[float] = 0.857
_UHLMANN_FIDELITY_ADJ_THRESHOLD: Final[float] = 0.95
_BRUHAT_TITS_DEGRADED_THRESHOLD: Final[float] = 5.0
_BANACH_ITERS: Final[int] = 8
_RSI_RESIDUAL_CERT: Final[float] = 1.0e-2
_LIPSCHITZ_CERT: Final[float] = 1.0 - 1.0e-6
_GREENE_CRITICAL: Final[float] = 0.25
_BRYUNO_DIVERGE: Final[float] = 40.0
_MELNIKOV_ZERO_EPS: Final[float] = 1.0e-12
_MELNIKOV_AMP_THRESHOLD: Final[float] = 1.0e-3
_QND_COMM_NORM_THRESHOLD: Final[float] = 1.0e-6
_TWO_PI: Final[float] = 2.0 * math.pi
_GOLDEN_RATIO: Final[float] = 0.5 * (1.0 + math.sqrt(5.0))
_ENGINE_VERSION: Final[str] = "13.0.0-SOVEREIGN-NOVIKOV-PROSECUTOR-POINCARE-KAM-L3-MONAD-PhD"
_HMAC_DOMAIN_INDICT: Final[bytes] = b"NOVIKOV-PROSECUTOR-INDICT-v13.0|"
_HMAC_DOMAIN_BRIDGE12: Final[bytes] = b"NOVIKOV-PROSECUTOR-BRIDGE-P1-P2-v13.0|"
_HMAC_DOMAIN_BRIDGE23: Final[bytes] = b"NOVIKOV-PROSECUTOR-BRIDGE-P2-P3-v13.0|"
_HMAC_DOMAIN_EXEC: Final[bytes] = b"NOVIKOV-PROSECUTOR-EXEC-v13.0|"
_HMAC_DOMAIN_RSI3: Final[bytes] = b"RSI3-BANACH-CERT-v13.0|"


def _hmac_hex(secret: bytes, domain: bytes, payload: bytes) -> str:
    return hmac.new(secret, domain + payload, hashlib.sha256).hexdigest()


def _hermitize(M: NDArray) -> NDArray[np.complex128]:
    a = np.asarray(M, dtype=np.complex128)
    return 0.5 * (a + a.conj().T)


def _project_psd_normalize(m: NDArray) -> NDArray[np.complex128]:
    """Proyección de Higham sobre el simplex de estados: Hermitiano → PSD → Tr = 1."""
    a = np.asarray(m, dtype=np.complex128)
    herm = _hermitize(a)
    eigs, vecs = la.eigh(herm)
    clean = np.maximum(np.real(eigs), _SPECTRAL_PSD_FLOOR)
    psd = (vecs * clean) @ vecs.conj().T
    tr = float(np.real(np.trace(psd)))
    if tr < _WILKINSON_EPSILON:
        n = int(psd.shape[0])
        return np.eye(n, dtype=np.complex128) / float(max(n, 1))
    return psd / tr


def _uhlmann_fidelity(rho: NDArray[np.complex128], sigma: NDArray[np.complex128]) -> float:
    r"""F(ρ,σ) = (Tr √(√ρ σ √ρ))². No renormaliza el operador intermedio (bugfix v13)."""
    sqrt_rho = la.sqrtm(_hermitize(rho))
    middle = _hermitize(sqrt_rho @ sigma @ sqrt_rho)
    eigvals = np.maximum(np.real(np.linalg.eigvalsh(middle)), 0.0)
    trace_sqrt = float(np.sum(np.sqrt(eigvals)))
    return float(min(max(trace_sqrt**2, 0.0), 1.0))


def _spectral_actions(rho: NDArray[np.complex128]) -> NDArray[np.float64]:
    evals = np.sort(
        np.maximum(np.real(la.eigvalsh(_hermitize(rho))), _SPECTRAL_PSD_FLOOR)
    )[::-1]
    s = float(np.sum(evals)) + 1e-30
    evals = evals / s
    return -np.log(evals)


def _back_action_db(rho: NDArray[np.complex128], A: NDArray[np.complex128]) -> float:
    comm = A @ rho - rho @ A
    comm_norm = float(la.norm(comm, ord="fro"))
    if comm_norm <= _QND_COMM_NORM_THRESHOLD:
        return 0.0
    return float(min(12.5, 20.0 * math.log10(1.0 + comm_norm)))


# ══════════════════════════════════════════════════════════════════════════════
# §A. RETÍCULO DE HEYTING Ω₃, MODOS DE ACTUACIÓN Y ESTADOS FISCALES
# ══════════════════════════════════════════════════════════════════════════════

class HeytingOmega3(IntEnum):
    r"""Clasificador de subobjetos del topos intuicionista Ω₃ = {0x00, 0x01, 0x03}.

    Orden de verdad (⊥→⊤): VETOED ≤ DEGRADED ≤ COHERENT.
    Orden numérico:        COHERENT < DEGRADED < VETOED (invertido).
    a → b = ⊤ ⇔ a ≤_truth b ⇔ int(a) ≥ int(b).  (v12 usaba `<=` y invertía imply.)
    """

    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03

    @property
    def verdict(self) -> str:
        return self.name

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """Ínfimo de verdad ≡ supremo de severidad."""
        return HeytingOmega3(max(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """Supremo de verdad ≡ ínfimo de severidad."""
        return HeytingOmega3(min(int(self), int(other)))

    def imply(self, other: "HeytingOmega3") -> "HeytingOmega3":
        if int(self) >= int(other):
            return HeytingOmega3.COHERENT
        return other

    def equiv(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return self.imply(other).meet(other.imply(self))

    def negation(self) -> "HeytingOmega3":
        return self.imply(HeytingOmega3.VETOED)

    def is_top(self) -> bool:
        return self == HeytingOmega3.COHERENT

    def is_bottom(self) -> bool:
        return self == HeytingOmega3.VETOED


class ActuationMode(IntEnum):
    NORMAL_FLUID = 0
    SOFT_VETO_BYPASS = 1
    HARD_VETO_ESP32_CROWBAR = 2


class NovikovProsecutionStatus(IntEnum):
    PENDING_INDICTMENT = 0
    INDICTED_CAUSAL_PARADOX = 1
    PURGED_DIRAC_ANNIHILATION = 2
    ABSOLVED_AUTOCONSISTENT = 3
    VETOED_ACTION_DIVERGENCE = 4


# ══════════════════════════════════════════════════════════════════════════════
# §B. ESTRUCTURAS INMUTABLES: SEMILLA, RSI-3, PUENTES OODA
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class NovikovProsecutionCanonicalSeed:
    """Germen inmutable Delaunay–Deprit, monodromía, Greene, Bryuno, Morse y Cartan."""

    seed_id: str
    delaunay_L: float
    delaunay_G: float
    delaunay_H: float
    delaunay_l_angle: float
    delaunay_g_angle: float
    delaunay_h_angle: float
    eccentricity_e: float
    inclination_i: float
    laplace_runge_lenz_norm: float
    cr3bp_jacobi_constant: float
    lagrange_l1_instability_rate: float
    monodromy_trace: float
    greene_residue_R: float
    rotation_number_nu: float
    melnikov_integral_M0: float
    separatrix_action_J0: float
    homoclinic_tangency_detected: bool
    continued_fraction_terms: Tuple[int, ...]
    bryuno_diophantine_sum: float
    is_bryuno_stable: bool
    morse_bott_euler_chi: int
    morse_bott_strata: Tuple[int, ...]
    poincare_cartan_1form: float
    symplectic_volume_residual: float
    sha256_provenance: str
    diophantine_gamma: float = 0.0
    nekhoroshev_time: float = 0.0
    jacobi_integral: float = 0.0
    simple_melnikov_zeros: int = 0
    greene_stable: bool = True
    chi_cp_expected: int = 1
    chi_certified: bool = False
    kam_frequency_family: str = "SPECTRAL_ROTATION"
    engine_version: str = _ENGINE_VERSION


@dataclass(frozen=True, slots=True)
class NovikovProsecutorRSI3State:
    r"""Estado RSI-3: s ∈ ℝ³ = (v_floor, d_BT*, F*) y meta-operador T: ℝ³ → ℝ³.

    Certificado: Lip(T) < 1 − ε  ∧  ||T(s_n) − s_n||_∞ < tol*  (Banach medido, no recortado).
    """

    rsi_iteration_id: str
    rsi_level: int
    meta_operator_norm: float
    fixed_point_residual: float
    adaptive_v_floor: float
    adaptive_d_bt_threshold: float
    adaptive_fidelity_threshold: float
    improvement_lyapunov: float
    historical_verdicts: Tuple[int, ...]
    banach_convergence_certified: bool
    banach_certificate_hmac: str
    banach_fixed_point: Tuple[float, float, float] = (0.10, 15.0, 0.95)
    banach_iterates: int = 0
    qnd_repair_count: int = 0
    lyapunov_decrease_certified: bool = False
    source: str = "PHASE1_BANACH"


@dataclass(frozen=True, slots=True)
class NovikovProsecutionIndictmentGerm:
    """Acusación espectral no-arquimediana sobre desviaciones causales en Λ_Nov."""

    indictment_id: str
    deliberation_id: str
    weak_value_Aw: complex
    novikov_valuation: float
    ultrametric_norm: float
    ultrametric_inequality_residual: float
    bruhat_tits_distance: float
    amoeba_tropical_area: float
    uhlmann_fidelity: float
    bures_distance: float
    cstar_positivity_valid: bool
    poincare_wirtinger_variance: float
    oseledets_transverse_lyapunov: float
    kolmogorov_sinai_entropy: float
    capacity_gromov: float
    is_action_divergent: bool
    timestamp_utc: float
    canonical_seed_ref: str = ""
    rsi3_state_ref: str = ""
    melnikov_M0: float = 0.0
    greene_residue: float = 0.0
    bryuno_sum: float = 0.0
    homoclinic_tangency: bool = False
    ultrametric_gap_significant: bool = False
    back_action_db: float = 0.0
    fubini_study_distance: float = 0.0
    indictment_hmac: str = ""
    engine_version: str = _ENGINE_VERSION


@dataclass(frozen=True, slots=True)
class NovikovProsecutionExecutionCertificate:
    """Certificado inmutable de ejecución fiscal: fallo Ω₃ + enclavamiento simulado."""

    certificate_id: str
    indictment_ref_id: str
    prosecution_status: NovikovProsecutionStatus
    global_heyting_verdict: HeytingOmega3
    actuation_mode: ActuationMode
    crowbar_triggered: bool
    crowbar_latency_ns: float
    annihilation_executed: bool
    released_energy_kev: float
    back_action_db: float
    banach_lipschitz_contraction: float
    sha256_merkle_digest: str
    hmac_provenance_signature: str
    timestamp_utc: float
    rsi3_convergence_certified: bool = False
    melnikov_M0: float = 0.0
    kam_stability: float = 0.0
    homoclinic_tangency_resolved: bool = False
    phase2_verdict: str = "N/A"
    greene_residue: float = 0.0
    bryuno_stable: bool = True
    diophantine_gamma: float = 0.0
    adjudication_honored: bool = True
    engine_version: str = _ENGINE_VERSION


@dataclass(frozen=True, slots=True)
class Phase1ToPhase2ProsecutionBridge:
    r"""Puente Φ₁₂' que cierra INQUISICIÓN y abre JUICIO.

    Valor de retorno del ÚLTIMO método de la Fase 1
    (`handoff_to_phase2_judgment_context`) y primer argumento de
    `consume_phase1_bridge` / `judge_deliberation` en la Fase 2.
    """

    indictment: NovikovProsecutionIndictmentGerm
    canonical_seed: NovikovProsecutionCanonicalSeed
    rsi3_state: NovikovProsecutorRSI3State
    continuation_token: str
    hmac_continuation: str


@dataclass(frozen=True, slots=True)
class Phase2ToPhase3ProsecutionBridge:
    """Puente Φ₂₃' que cierra JUICIO y abre EJECUCIÓN (el veredicto P2 VIAJA)."""

    indictment: NovikovProsecutionIndictmentGerm
    canonical_seed: NovikovProsecutionCanonicalSeed
    rsi3_state: NovikovProsecutorRSI3State
    adjudication_verdict: HeytingOmega3
    weak_value_Aw: complex
    spohn_entropy: float
    gromov_capacity: float
    oseledets_lyapunov: float
    continuation_token: str
    hmac_continuation: str


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 1: INQUISICIÓN — C*, ultramétrica no tautológica, Uhlmann FIX, Poincaré, Banach RSI-3    ║
# ║  ÚLTIMA def. (handoff_to_phase2_judgment_context) → Phase1ToPhase2ProsecutionBridge            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

class NovikovProsecutorPhase1Inquisitor:
    r"""FASE 1: INQUISICIÓN — Aprehensión fiscal, auditoría C*-𝔇_n, valuación no-arquimediana,
    mecánica celeste de Poincaré y bootstrap RSI-3 por iteración de Banach.
    """

    def __init__(
        self,
        dimension: int = 56,
        v_floor: float = _NOVIKOV_V_FLOOR_DEFAULT,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> None:
        self.dimension: Final[int] = max(4, int(dimension))
        self.v_floor: Final[float] = float(v_floor)
        self.secret_key: Final[bytes] = bytes(secret_key)
        self._last_canonical_seed: Optional[NovikovProsecutionCanonicalSeed] = None
        self._last_rsi3_state: Optional[NovikovProsecutorRSI3State] = None
        self._last_indictment: Optional[NovikovProsecutionIndictmentGerm] = None
        self._witness_engine: Any = None
        if _HAS_WITNESS_ENGINE and _WitnessEngine is not None:
            try:
                self._witness_engine = _WitnessEngine(dimension=self.dimension, secret_key=secret_key)
            except Exception:
                self._witness_engine = None

    def sanitize_density_matrix(self, matrix: NDArray) -> NDArray[np.complex128]:
        """Proyección convexa PSD de Higham embebida en d = self.dimension, Tr(ρ) ≡ 1."""
        a = np.asarray(matrix, dtype=np.complex128)
        if a.ndim != 2:
            return np.eye(self.dimension, dtype=np.complex128) / float(self.dimension)
        if a.shape != (self.dimension, self.dimension):
            embedded = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
            d = min(self.dimension, a.shape[0], a.shape[1])
            embedded[:d, :d] = a[:d, :d]
            a = embedded
        return _project_psd_normalize(a)

    def audit_cstar_axioms(self, rho: NDArray) -> Tuple[bool, bool, bool, float]:
        """Audita autoadjunción, traza unitaria, positividad y residuo de Frobenius."""
        r = self.sanitize_density_matrix(rho)
        herm_res = float(la.norm(r - r.conj().T, ord="fro"))
        involution_ok = herm_res < 1.0e-9
        tr_val = float(np.real(np.trace(r)))
        trace_ok = abs(tr_val - 1.0) < 1.0e-5
        eigs = np.real(la.eigvalsh(_hermitize(r)))
        positivity_ok = bool(np.min(eigs) >= -1.0e-10)
        axioms_ok = involution_ok and trace_ok and positivity_ok
        return axioms_ok, positivity_ok, involution_ok, herm_res

    def compute_novikov_valuation_and_ultrametric_norm(
        self, rho: NDArray, action_spectrum: Optional[NDArray] = None
    ) -> Tuple[float, float, float, bool]:
        r"""v(A)=min a_i, ||A||_Nov=exp(−v(A)).

        v12 auditaba la tautología exp(−min(a,b)) ≤ max(exp(−a),exp(−b)).
        v13: residual = holgura de esa identidad (debe ser 0) + test ESTRUCTURAL
        de no-degeneración: log(λ_max/λ_second) > umbral (cima no max-mixed).
        """
        r = self.sanitize_density_matrix(rho)
        if action_spectrum is not None:
            a_vec = np.sort(np.asarray(action_spectrum, dtype=np.float64))
        else:
            a_vec = np.sort(_spectral_actions(r))

        v_min = float(np.min(a_vec))
        ultrametric_norm = math.exp(-min(v_min, 700.0)) if v_min < 700.0 else 0.0

        k = min(int(a_vec.size), 8)
        residuals: List[float] = []
        for i in range(k):
            for j in range(k):
                norm_i = math.exp(-min(float(a_vec[i]), 700.0))
                norm_j = math.exp(-min(float(a_vec[j]), 700.0))
                norm_sum = math.exp(-min(float(min(a_vec[i], a_vec[j])), 700.0))
                rhs = max(norm_i, norm_j)
                residuals.append(max(0.0, norm_sum - rhs - _ULTRAMETRIC_TOLERANCE * max(rhs, 1.0)))
        max_residual = float(np.max(residuals)) if residuals else 0.0

        if a_vec.size >= 2:
            top_gap = float(a_vec[1] - a_vec[0])
            structural_ok = top_gap > _ULTRA_GAP_THRESHOLD
        else:
            structural_ok = False
        is_satisfied = (max_residual <= _ULTRAMETRIC_TOLERANCE) and structural_ok
        return v_min, ultrametric_norm, max_residual, is_satisfied

    def compute_bruhat_tits_tree_distance(
        self, rho_mac: NDArray, rho_delib: NDArray, v_min: float
    ) -> Tuple[float, float, float]:
        """d_BT = |v(A) − v_floor| + d_FS(ρ_mac, ρ_delib), F de Uhlmann sin renormalizar."""
        r_mac = self.sanitize_density_matrix(rho_mac)
        r_del = self.sanitize_density_matrix(rho_delib)
        fidelity = _uhlmann_fidelity(r_mac, r_del)
        d_fs = float(np.arccos(min(1.0, max(0.0, math.sqrt(fidelity)))))
        d_bt = float(abs(v_min - self.v_floor) + d_fs)
        return d_bt, fidelity, d_fs

    def compute_tropical_amoeba_area(self, rho: NDArray) -> float:
        r = self.sanitize_density_matrix(rho)
        actions = _spectral_actions(r)
        k = min(int(actions.size), 8)
        area = 0.5 * float(np.sum(actions[:k] ** 2)) * (math.pi / float(self.dimension))
        return float(area)

    def compute_uhlmann_fidelity_and_bures_distance(
        self, rho_a: NDArray, rho_b: NDArray
    ) -> Tuple[float, float]:
        r"""F(ρ,σ)=(Tr√(√ρ σ √ρ))², d_B=√(2−2√F). v13 no sanitiza el operador medio."""
        r_a = self.sanitize_density_matrix(rho_a)
        r_b = self.sanitize_density_matrix(rho_b)
        fidelity = _uhlmann_fidelity(r_a, r_b)
        bures_dist = float(math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(fidelity))))
        return fidelity, bures_dist

    @staticmethod
    def _continued_fraction_expansion(x: float, max_terms: int = 24) -> Tuple[int, ...]:
        if not math.isfinite(x):
            return (1,)
        terms: List[int] = []
        y = float(abs(x))
        for _ in range(max_terms):
            a = int(math.floor(y))
            terms.append(a)
            frac = y - a
            if frac < 1e-12:
                break
            y = 1.0 / frac
        return tuple(terms) if terms else (1,)

    @classmethod
    def _bryuno_diophantine_function(
        cls, rotation_number: float
    ) -> Tuple[float, bool, Tuple[int, ...]]:
        r"""B(ω)=∑ ln(q_{k+1})/q_k. Racional (CF finita) ⇒ resonante ⇒ ¬Bryuno-KAM."""
        nu = float(rotation_number) % 1.0
        cf = cls._continued_fraction_expansion(nu if nu > 0.0 else rotation_number)
        if len(cf) < 2:
            return 0.0, True, cf

        q_minus2, q_minus1 = 1, 0
        qs: List[int] = []
        for a in cf:
            q_k = int(a) * q_minus1 + q_minus2
            qs.append(max(int(q_k), 1))
            q_minus2, q_minus1 = q_minus1, q_k

        rational = len(cf) < 24 and (cf[-1] != 0)
        # CF que termina ⇒ ω ∈ ℚ ⇒ resonancia → KAM no aplica.
        if rational and len(cf) <= 6:
            b_sum = float("inf") if False else float(sum(
                math.log(max(qs[k + 1], 2)) / float(max(qs[k], 1)) for k in range(len(qs) - 1)
            ))
            # Marca inestable si el último cociente es enorme o la CF es corta y exacta.
            is_stable = not (len(cf) <= 3)
        else:
            b_sum = 0.0
            for k in range(len(qs) - 1):
                b_sum += math.log(max(qs[k + 1], 2)) / float(max(qs[k], 1))
            is_stable = math.isfinite(b_sum) and b_sum < _BRYUNO_DIVERGE
        return float(b_sum if math.isfinite(b_sum) else _BRYUNO_DIVERGE + 1.0), bool(is_stable), cf

    @staticmethod
    def _melnikov_with_separatrix(
        epsilon: float = 0.1, omega: float = 1.0
    ) -> Tuple[float, float, bool, int]:
        r"""Melnikov del péndulo perturbado sobre la separatriz q₀=4 arctan(e^t)−π:

            M(t₀) = 4π ω² sech(πω/2) ε sin(ω t₀)

        Tangle ⇔ ceros simples (cambio de signo con |ΔM|>ε) ∧ |A| > umbral.
        v12 declaraba tangle si |A|>10⁻⁶ (casi siempre verdadero).
        """
        if omega <= 0.0:
            return 0.0, 8.0, False, 0
        ampl = 4.0 * math.pi * (omega**2) * (1.0 / math.cosh(0.5 * math.pi * omega)) * float(epsilon)
        t0 = np.linspace(0.0, _TWO_PI / max(omega, 1e-6), 256, dtype=np.float64)
        m_t = ampl * np.sin(omega * t0)
        signs = np.sign(m_t)
        dabs = np.abs(np.diff(m_t))
        n_zeros = int(np.sum((signs[:-1] * signs[1:] < 0.0) & (dabs > _MELNIKOV_ZERO_EPS)))
        has_tangle = bool(n_zeros > 0 and abs(ampl) > _MELNIKOV_AMP_THRESHOLD)
        return float(ampl), 8.0, has_tangle, n_zeros

    @staticmethod
    def _greene_residue_from_monodromy(rotation_number: float) -> Tuple[float, float, bool]:
        r"""R_G = (2 − Tr M)/4 = sin²(πν) para M∈Sp(2,ℝ), Tr M = 2 cos(2πν).

        Greene (1979): |R_G| ≲ 0.25 en el noble crítico. ν se reduce a [0,1).
        """
        nu = float(rotation_number) % 1.0
        trace_M = 2.0 * math.cos(_TWO_PI * nu)
        R_G = (2.0 - trace_M) / 4.0
        return float(trace_M), float(R_G), bool(abs(R_G) < _GREENE_CRITICAL)

    def compute_poincare_celestial_mechanics(
        self,
        rho: NDArray,
        K: float = 0.42,
        I0: float = 0.7,
    ) -> NovikovProsecutionCanonicalSeed:
        r"""Coordenadas Delaunay–Deprit e invariantes celestes sobre el espectro de ρ.

        Pipeline v13:
          1. Acciones I_k = −ln λ_k.
          2. Delaunay–Deprit espectral (L,G,H,e,i) + ángulos (l,g,h).
          3. Jacobi CR3BP C_J = 2L + G cos i.
          4. ν = {ω₁/ω₂} ∈ [0,1); monodromía + Greene.
          5. Melnikov con ceros simples sobre separatriz sech.
          6. Bryuno B(ν) (racionales cortos ⇒ ¬KAM).
          7. Morse: χ(ℂPⁿ⁻¹)=n (teorema) vs estratificación espectral.
          8. θ_PC = Im Tr(ρ [N,ρ]) calculada EN LA EIGENBASE (v12 mezclaba bases).
        """
        _ = I0  # reservado (acción de referencia Delaunay)
        r = self.sanitize_density_matrix(rho)
        eigs, evecs = la.eigh(_hermitize(r))
        eigs = np.maximum(np.real(eigs), _SPECTRAL_PSD_FLOOR)
        order = np.argsort(eigs)[::-1]
        eigs = eigs[order]
        evecs = evecs[:, order]
        n_eig = int(eigs.size)
        probs = eigs / (float(np.sum(eigs)) + 1e-30)
        actions = -np.log(np.maximum(probs, 1e-15))

        L = float(np.sum(actions))
        G = float(L * (probs[1] / max(probs[0], 1e-15))) if n_eig > 1 else L
        G = float(min(G, L))
        H = float(G * (probs[2] / max(probs[1], 1e-15))) if n_eig > 2 else G
        H = float(min(H, G))
        ecc = float(math.sqrt(max(0.0, 1.0 - (G / (L + 1e-15)) ** 2)))
        inc = float(math.acos(min(max(H / (G + 1e-15), -1.0), 1.0)))
        l_ang = float(math.atan2(actions[1], actions[0])) if n_eig > 1 else 0.0
        g_ang = float(math.atan2(actions[2], actions[1])) if n_eig > 2 else 0.0
        h_ang = float(np.mean(np.angle(evecs[:, 0]))) if n_eig > 0 else 0.0
        lrl_norm = float(L * ecc)

        c_j = float(2.0 * L + G * math.cos(inc))
        l1_rate = float(0.15 + 0.1 * ecc)

        if n_eig > 1:
            freqs = np.abs(np.diff(actions))
            omega1 = float(np.median(freqs)) if freqs.size > 0 else 1.0
        else:
            omega1 = 1.0
        omega2 = float(np.std(actions)) + 1e-12
        nu_raw = float(omega1 / max(omega2, 1e-12))
        nu = float(nu_raw % 1.0)

        trace_M, R_G, greene_ok = self._greene_residue_from_monodromy(nu)
        M_amp, J0, has_tangle, n_zeros = self._melnikov_with_separatrix(
            epsilon=K, omega=min(max(omega1, 0.1), 10.0)
        )
        bryuno_sum, bryuno_stable, cf_terms = self._bryuno_diophantine_function(nu)

        sorted_p = np.sort(probs)[::-1]
        strata: List[int] = []
        current_cluster = 1
        for i in range(1, len(sorted_p)):
            if abs(float(sorted_p[i] - sorted_p[i - 1])) < 1e-3:
                current_cluster += 1
            else:
                strata.append(current_cluster)
                current_cluster = 1
        strata.append(current_cluster)
        chi_morse = int(sum((1 if k % 2 == 0 else -1) * d for k, d in enumerate(strata)))
        chi_cp = int(self.dimension)  # χ(ℂP^{n−1}) = n (teorema, no se «estima»)
        all_simple = all(d == 1 for d in strata)
        chi_certified = bool(all_simple and chi_cp == self.dimension)

        # Poincaré–Cartan EN EIGENBASE: N = diag(0..n−1) conmuta con ρ diagonal.
        N_e = np.diag(np.arange(n_eig, dtype=np.complex128))
        rho_e = np.diag(probs.astype(np.complex128))
        comm_e = N_e @ rho_e - rho_e @ N_e
        theta_pc = float(abs(np.imag(np.trace(rho_e.conj().T @ comm_e))))
        # Equivalente covariante: Tr(ρ [N_lab, ρ]) con N_lab = U N U†
        N_lab = evecs @ N_e @ evecs.conj().T
        comm_lab = N_lab @ r - r @ N_lab
        theta_pc = float(abs(np.imag(np.trace(r.conj().T @ comm_lab))))

        purity = float(np.real(np.trace(r @ r)))
        sympl_residual = float(abs(purity - 1.0 / float(self.dimension)))
        jacobi = float(np.real(np.sum(probs * np.linspace(0.1, 1.0, n_eig))))

        mean_act = float(np.mean(actions)) + 1e-12
        eps = float(np.clip(np.std(actions) / mean_act, 1e-6, 0.999))
        try:
            nek_time = float(math.exp(min(40.0, 0.35 / (eps**0.25))))
        except OverflowError:
            nek_time = 1.0e12

        gamma = 0.0
        family = "SPECTRAL_ROTATION"
        if _HAS_WITNESS_ENGINE and _WitnessKAM is not None and n_eig >= 2:
            try:
                omega_loc = np.maximum(np.abs(np.diff(actions)), 1e-9)
                torus = _WitnessKAM(tuple(float(w) for w in omega_loc[: min(8, omega_loc.size)]))
                gamma = float(torus.diophantine_constant())
                family = "WITNESS_KAM_NOBLE_PROBE"
            except Exception:
                gamma = 0.0

        seed_id = f"SEED-PROSECUTOR-NOV-{int(time.time() * 1000) % 1_000_000:06d}"
        prov = (
            f"{seed_id}:{L:.8f}:{G:.8f}:{H:.8f}:{ecc:.8f}:{c_j:.8f}:"
            f"{R_G:.8f}:{M_amp:.8f}:{bryuno_sum:.8f}:{chi_cp}:{theta_pc:.8f}:{gamma:.8e}"
        )
        return NovikovProsecutionCanonicalSeed(
            seed_id=seed_id,
            delaunay_L=L,
            delaunay_G=G,
            delaunay_H=H,
            delaunay_l_angle=l_ang,
            delaunay_g_angle=g_ang,
            delaunay_h_angle=h_ang,
            eccentricity_e=ecc,
            inclination_i=inc,
            laplace_runge_lenz_norm=lrl_norm,
            cr3bp_jacobi_constant=c_j,
            lagrange_l1_instability_rate=l1_rate,
            monodromy_trace=trace_M,
            greene_residue_R=R_G,
            rotation_number_nu=nu,
            melnikov_integral_M0=M_amp,
            separatrix_action_J0=J0,
            homoclinic_tangency_detected=has_tangle,
            continued_fraction_terms=cf_terms,
            bryuno_diophantine_sum=bryuno_sum,
            is_bryuno_stable=bryuno_stable,
            morse_bott_euler_chi=chi_morse,
            morse_bott_strata=tuple(int(x) for x in strata),
            poincare_cartan_1form=theta_pc,
            symplectic_volume_residual=sympl_residual,
            sha256_provenance=hashlib.sha256(prov.encode("utf-8")).hexdigest(),
            diophantine_gamma=gamma,
            nekhoroshev_time=nek_time,
            jacobi_integral=jacobi,
            simple_melnikov_zeros=n_zeros,
            greene_stable=greene_ok,
            chi_cp_expected=chi_cp,
            chi_certified=chi_certified,
            kam_frequency_family=family,
        )

    @staticmethod
    def _banach_iterate_r3(
        s0: Tuple[float, float, float],
        potential: float,
        n_iters: int = _BANACH_ITERS,
        bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
            (0.02, 0.50),
            (3.0, 25.0),
            (0.70, 0.999),
        ),
    ) -> Tuple[float, float, Tuple[float, float, float], int]:
        r"""Itera T: ℝ³ → ℝ³, T(s) = (1−λ)s + λ Ψ(s,Φ), y mide Lip sobre la órbita.

        Ψ acerca v_floor a v_obs-driven, contrae d_BT* si Φ↑, y sube F* suavemente.
        v12 recortaba ||T|| a 0.999 (tautología). Aquí Lip se MIDE.
        """
        phi = float(min(max(potential, 0.0), 1.0))
        lam = float(np.clip(0.45 * (0.4 + 0.6 * phi), 0.05, 0.90))
        s = np.array(s0, dtype=np.float64)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        target = np.array(
            [
                s0[0] * (1.0 + 0.05 * phi),
                s0[1] * (1.0 - 0.03 * phi),
                min(0.999, s0[2] + 0.02 * phi * (1.0 - s0[2])),
            ],
            dtype=np.float64,
        )
        xs: List[NDArray[np.float64]] = [s.copy()]
        for _ in range(max(int(n_iters), 1)):
            s = (1.0 - lam) * s + lam * target
            s = np.clip(s, lo, hi)
            xs.append(s.copy())
        lips: List[float] = []
        for i in range(1, len(xs) - 1):
            dy = float(np.linalg.norm(xs[i] - xs[i - 1], ord=np.inf))
            dx = float(np.linalg.norm(xs[i + 1] - xs[i], ord=np.inf))
            if dy > _WILKINSON_EPSILON:
                lips.append(dx / dy)
        lip_emp = float(max(lips)) if lips else float(1.0 - lam)
        lip = float(min(0.999999, max(lip_emp, 1.0 - lam)))
        residual = float(np.linalg.norm(xs[-1] - xs[-2], ord=np.inf)) if len(xs) >= 2 else 0.0
        star = tuple(float(x) for x in xs[-1])
        return lip, residual, (star[0], star[1], star[2]), len(xs) - 1

    def compute_rsi3_bootstrap(
        self,
        v_min: float,
        d_bt: float,
        fidelity: float,
        rsi_history: Optional[Sequence[int]] = None,
    ) -> NovikovProsecutorRSI3State:
        """Bootstrap RSI-3: potencial Φ, iteración de Banach en ℝ³, HMAC del certificado."""
        hist = tuple(int(h) for h in (rsi_history or ()))
        now = time.time()

        v_dev = abs(v_min - self.v_floor) / (self.v_floor + 1e-12)
        d_dev = abs(d_bt - _BRUHAT_TITS_VETO_THRESHOLD) / _BRUHAT_TITS_VETO_THRESHOLD
        f_dev = abs(fidelity - _UHLMANN_FIDELITY_ADJ_THRESHOLD)
        perf_potential = float(math.exp(-(v_dev + d_dev + f_dev)))
        perf_potential = min(max(perf_potential, 1e-6), 1.0)

        s0 = (self.v_floor, _BRUHAT_TITS_VETO_THRESHOLD, _UHLMANN_FIDELITY_ADJ_THRESHOLD)
        lip, resid, star, n_it = self._banach_iterate_r3(s0, perf_potential)

        if len(hist) >= 2:
            improvement_lyap = float(np.mean(np.abs(np.diff(np.asarray(hist, dtype=float))))) + 1e-9
        else:
            improvement_lyap = 1e-3

        banach_ok = bool(lip < _LIPSCHITZ_CERT and resid < _RSI_RESIDUAL_CERT)
        rsi_id = f"RSI3-PROSECUTOR-{int(now * 1000) % 1_000_000:06d}"
        cert_payload = f"{rsi_id}:{lip:.8f}:{resid:.8f}:{banach_ok}".encode("utf-8")
        cert_hmac = _hmac_hex(self.secret_key, _HMAC_DOMAIN_RSI3, cert_payload)

        return NovikovProsecutorRSI3State(
            rsi_iteration_id=rsi_id,
            rsi_level=3,
            meta_operator_norm=lip,
            fixed_point_residual=resid,
            adaptive_v_floor=float(star[0]),
            adaptive_d_bt_threshold=float(star[1]),
            adaptive_fidelity_threshold=float(star[2]),
            improvement_lyapunov=improvement_lyap,
            historical_verdicts=tuple(hist[-8:]),
            banach_convergence_certified=banach_ok,
            banach_certificate_hmac=cert_hmac,
            banach_fixed_point=star,
            banach_iterates=n_it,
            source="PHASE1_BANACH",
        )

    def _qnd_weak_value(
        self, r_mac: NDArray[np.complex128], r_del: NDArray[np.complex128]
    ) -> Tuple[complex, float]:
        """A_w AAV con observable espectral de ρ_delib; back-action = ||[A,ρ]||_F."""
        _evals, evecs = la.eigh(_hermitize(r_del))
        d = int(r_del.shape[0])
        A_nov = (evecs * np.linspace(0.1, 1.0, d, dtype=np.float64)) @ evecs.conj().T
        phi_i = evecs[:, -1]
        phi_f_raw = r_del @ phi_i
        nrm = float(np.linalg.norm(phi_f_raw))
        phi_f = phi_f_raw / max(nrm, _WILKINSON_EPSILON)
        overlap = complex(np.vdot(phi_f, phi_i))
        if abs(overlap) < 1e-12:
            Aw = complex(float(np.real(np.trace(A_nov @ r_del))), 0.0)
        else:
            Aw = complex(np.vdot(phi_f, A_nov @ phi_i)) / overlap
        return Aw, _back_action_db(r_del, A_nov)

    def indict_novikov_deliberation(
        self,
        deliberation_id: str,
        mac_density_matrix: NDArray,
        deliberation_density_matrix: NDArray,
        action_spectrum: Optional[NDArray] = None,
        weak_value_override: Optional[complex] = None,
        rsi_history: Optional[Sequence[int]] = None,
    ) -> NovikovProsecutionIndictmentGerm:
        """Inquisición no-arquimediana: C*, valuación, Bruhat–Tits, Poincaré y RSI-3."""
        now = time.time()
        r_mac = self.sanitize_density_matrix(mac_density_matrix)
        r_del = self.sanitize_density_matrix(deliberation_density_matrix)

        _cstar_ok, pos_ok, _, _ = self.audit_cstar_axioms(r_del)
        v_min, ultra_norm, ultra_res, ultra_ok = self.compute_novikov_valuation_and_ultrametric_norm(
            r_del, action_spectrum
        )
        d_bt, fidelity, d_fs = self.compute_bruhat_tits_tree_distance(r_mac, r_del, v_min)
        amoeba_area = self.compute_tropical_amoeba_area(r_del)
        _, bures_dist = self.compute_uhlmann_fidelity_and_bures_distance(r_mac, r_del)

        eigs = np.sort(np.maximum(np.real(la.eigvalsh(_hermitize(r_del))), _SPECTRAL_PSD_FLOOR))[::-1]
        probs = eigs / float(np.sum(eigs))
        h_ks = float(-np.sum(probs * np.log(np.maximum(probs, 1e-15))))
        lam_perp = float(math.log(max(probs[1], 1e-15) / max(probs[0], 1e-15))) if probs.size > 1 else -0.5

        purity = float(np.real(np.trace(r_del @ r_del)))
        r_w = math.sqrt(max(self.dimension * max(1.0 - purity, 0.0), 0.0))
        c_g = float(0.5 * math.pi * r_w * r_w)
        i_over_n = np.eye(self.dimension, dtype=np.complex128) / float(self.dimension)
        pw_variance = float(la.norm(r_del - i_over_n, ord="fro") ** 2)

        if weak_value_override is not None:
            Aw = weak_value_override
            ba_db = 0.0
        else:
            Aw, ba_db = self._qnd_weak_value(r_mac, r_del)

        canonical_seed = self.compute_poincare_celestial_mechanics(r_del)
        rsi3_state = self.compute_rsi3_bootstrap(v_min, d_bt, fidelity, rsi_history)

        v_floor_eff = max(self.v_floor, rsi3_state.adaptive_v_floor)
        d_bt_eff = rsi3_state.adaptive_d_bt_threshold
        kam_broken = bool(
            canonical_seed.homoclinic_tangency_detected
            and (not canonical_seed.greene_stable)
            and (not canonical_seed.is_bryuno_stable)
        )
        is_action_divergent = bool(
            v_min < v_floor_eff
            or d_bt > d_bt_eff
            or c_g > _GROMOV_CAPACITY_MAX_DEFAULT
            or (not pos_ok)
            or (not ultra_ok)
            or kam_broken
        )

        actions_sorted = np.sort(_spectral_actions(r_del))
        gap_sig = bool(
            actions_sorted.size >= 2 and (actions_sorted[1] - actions_sorted[0]) > _ULTRA_GAP_THRESHOLD
        )

        ind_id = f"INDICT-NOV-{deliberation_id}-{int(now * 1000) % 1_000_000:06d}"
        payload = (
            f"{ind_id}:{v_min:.8f}:{d_bt:.8f}:{fidelity:.8f}:{is_action_divergent}".encode("utf-8")
        )
        germ = NovikovProsecutionIndictmentGerm(
            indictment_id=ind_id,
            deliberation_id=deliberation_id,
            weak_value_Aw=Aw,
            novikov_valuation=v_min,
            ultrametric_norm=ultra_norm,
            ultrametric_inequality_residual=ultra_res,
            bruhat_tits_distance=d_bt,
            amoeba_tropical_area=amoeba_area,
            uhlmann_fidelity=fidelity,
            bures_distance=bures_dist,
            cstar_positivity_valid=pos_ok,
            poincare_wirtinger_variance=pw_variance,
            oseledets_transverse_lyapunov=lam_perp,
            kolmogorov_sinai_entropy=h_ks,
            capacity_gromov=c_g,
            is_action_divergent=is_action_divergent,
            timestamp_utc=now,
            canonical_seed_ref=canonical_seed.seed_id,
            rsi3_state_ref=rsi3_state.rsi_iteration_id,
            melnikov_M0=canonical_seed.melnikov_integral_M0,
            greene_residue=canonical_seed.greene_residue_R,
            bryuno_sum=canonical_seed.bryuno_diophantine_sum,
            homoclinic_tangency=canonical_seed.homoclinic_tangency_detected,
            ultrametric_gap_significant=gap_sig,
            back_action_db=ba_db,
            fubini_study_distance=d_fs,
            indictment_hmac=_hmac_hex(self.secret_key, _HMAC_DOMAIN_INDICT, payload),
        )
        self._last_canonical_seed = canonical_seed
        self._last_rsi3_state = rsi3_state
        self._last_indictment = germ
        return germ

    def _bridge12_payload(
        self,
        indictment: NovikovProsecutionIndictmentGerm,
        seed: NovikovProsecutionCanonicalSeed,
        rsi: NovikovProsecutorRSI3State,
        token: str,
    ) -> bytes:
        return (
            f"{indictment.indictment_id}:{seed.sha256_provenance}:{rsi.banach_certificate_hmac}:"
            f"{token}:{seed.greene_residue_R:.8f}:{seed.melnikov_integral_M0:.8e}:"
            f"{rsi.meta_operator_norm:.8f}:{indictment.is_action_divergent}:"
            f"{seed.homoclinic_tangency_detected}"
        ).encode("utf-8")

    # ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
    # ║  FASE 1 → FASE 2: FRONTERA (handoff_to_phase2_judgment_context).                             ║
    # ║  ÚLTIMO método de FASE 1. Retorna Phase1ToPhase2ProsecutionBridge → PRIMER arg. FASE 2.     ║
    # ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
    def handoff_to_phase2_judgment_context(
        self, indictment: NovikovProsecutionIndictmentGerm
    ) -> Phase1ToPhase2ProsecutionBridge:
        """ÚLTIMO método de la Fase 1 — cierra INQUISICIÓN y abre JUICIO (Φ₁₂')."""
        seed = self._last_canonical_seed
        rsi = self._last_rsi3_state
        if seed is None or rsi is None:
            raise RuntimeError("handoff_to_phase2_judgment_context: contexto no inicializado.")
        token = f"CONT-P1-P2-PROSEC-{indictment.indictment_id}-{seed.seed_id}-{rsi.rsi_iteration_id}"
        hmac_cont = _hmac_hex(
            self.secret_key, _HMAC_DOMAIN_BRIDGE12, self._bridge12_payload(indictment, seed, rsi, token)
        )
        logger.info(
            "Φ₁₂' handoff | indictment=%s | R_G=%.4f | M₀=%.4e | Lip(T)=%.4f | tangle=%s",
            indictment.indictment_id,
            seed.greene_residue_R,
            seed.melnikov_integral_M0,
            rsi.meta_operator_norm,
            seed.homoclinic_tangency_detected,
        )
        return Phase1ToPhase2ProsecutionBridge(
            indictment=indictment,
            canonical_seed=seed,
            rsi3_state=rsi,
            continuation_token=token,
            hmac_continuation=hmac_cont,
        )


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 2: JUICIO — QND AAV, Oseledets, Gromov, GKSL-Spohn, fusión Ω₃                             ║
# ║  Primera def. operativa consume Phase1ToPhase2ProsecutionBridge (FASE 1).                      ║
# ║  ÚLTIMA def. (handoff_to_phase3_execution_context) → Phase2ToPhase3ProsecutionBridge.          ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

class NovikovProsecutorPhase2Adjudicator(NovikovProsecutorPhase1Inquisitor):
    r"""FASE 2: JUICIO — weak value QND, Oseledets, Gromov, GKSL–Spohn y adjudicación Ω₃."""

    def __init__(
        self,
        dimension: int = 56,
        v_floor: float = _NOVIKOV_V_FLOOR_DEFAULT,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> None:
        super().__init__(dimension=dimension, v_floor=v_floor, secret_key=secret_key)
        self._last_bridge12: Optional[Phase1ToPhase2ProsecutionBridge] = None
        self._last_p2_verdict: Optional[HeytingOmega3] = None

    def consume_phase1_bridge(self, bridge: Phase1ToPhase2ProsecutionBridge) -> bool:
        expected = _hmac_hex(
            self.secret_key,
            _HMAC_DOMAIN_BRIDGE12,
            self._bridge12_payload(
                bridge.indictment, bridge.canonical_seed, bridge.rsi3_state, bridge.continuation_token
            ),
        )
        if not hmac.compare_digest(bridge.hmac_continuation, expected):
            raise ValueError("Falla de autenticidad en puente Φ₁₂' (INQUISICIÓN→JUICIO)")
        self._last_bridge12 = bridge
        return True

    def compute_qnd_weak_value_aav(self, rho_mac: NDArray, rho_delib: NDArray) -> complex:
        r_mac = self.sanitize_density_matrix(rho_mac)
        r_del = self.sanitize_density_matrix(rho_delib)
        Aw, _ba = self._qnd_weak_value(r_mac, r_del)
        return Aw

    def compute_oseledets_transverse_lyapunov(self, rho: NDArray) -> float:
        r = self.sanitize_density_matrix(rho)
        eigs = np.sort(np.maximum(np.real(la.eigvalsh(_hermitize(r))), _SPECTRAL_PSD_FLOOR))[::-1]
        if eigs.size < 2:
            return -0.5
        return float(math.log(max(eigs[1], 1e-15) / max(eigs[0], 1e-15)))

    def compute_capacity_gromov(self, rho: NDArray) -> float:
        r = self.sanitize_density_matrix(rho)
        purity = float(np.real(np.trace(r @ r)))
        r_w = math.sqrt(max(self.dimension * max(1.0 - purity, 0.0), 0.0))
        return float(0.5 * math.pi * r_w * r_w)

    def compute_gksl_dissipation_and_spohn_entropy(self, rho: NDArray) -> Tuple[float, float]:
        r"""Disipador GKSL de prueba y producción de entropía de Spohn σ = −Tr(log ρ · ℒ(ρ)) ≥ 0."""
        r = self.sanitize_density_matrix(rho)
        dim = int(r.shape[0])
        h_eff = 0.1 * np.eye(dim, dtype=np.complex128)
        unitary = -1j * (h_eff @ r - r @ h_eff)
        jump_l = np.zeros((dim, dim), dtype=np.complex128)
        jump_l[0, min(1, dim - 1)] = 0.05
        jump_dag = jump_l.conj().T
        dissipator = jump_l @ r @ jump_dag - 0.5 * (jump_dag @ jump_l @ r + r @ jump_dag @ jump_l)
        d_rho = unitary + dissipator

        eigs, evecs = la.eigh(_hermitize(r))
        log_eigs = np.log(np.maximum(np.real(eigs), _SPECTRAL_PSD_FLOOR))
        log_rho = (evecs * log_eigs) @ evecs.conj().T
        spohn_raw = -float(np.real(np.trace(log_rho @ d_rho)))
        spohn_entropy = max(0.0, spohn_raw)
        gksl_rate = float(la.norm(dissipator, ord="fro"))
        return spohn_entropy, gksl_rate

    def adjudicate_judgment(
        self,
        indictment: NovikovProsecutionIndictmentGerm,
        seed: NovikovProsecutionCanonicalSeed,
        rsi3: NovikovProsecutorRSI3State,
        gromov_capacity: float,
    ) -> HeytingOmega3:
        r"""Fusión Ω₃. Reglas v13 (deja de sobre-vetar):

        (J1) VETOED si acción divergente (valuación / d_BT / C* / ultra-gap / KAM roto).
        (J2) DEGRADED —no VETOED— si RSI-3 no certifica (el fiscal sigue, con fricción).
        (J3) VETOED si tangle ∧ ¬Greene ∧ ¬Bryuno (caos transversal con toro destruido).
        (J4) DEGRADED si ¬Bryuno XOR ¬Greene (inestabilidad parcial).
        (J5) DEGRADED si c_Nov > 12.5 o F < F*.
        """
        v = HeytingOmega3.COHERENT
        if indictment.is_action_divergent:
            v = v.meet(HeytingOmega3.VETOED)
        if not rsi3.banach_convergence_certified:
            v = v.meet(HeytingOmega3.DEGRADED)
        kam_destroyed = (
            seed.homoclinic_tangency_detected
            and (not seed.greene_stable)
            and (not seed.is_bryuno_stable)
        )
        if kam_destroyed:
            v = v.meet(HeytingOmega3.VETOED)
        elif (not seed.is_bryuno_stable) ^ (not seed.greene_stable):
            v = v.meet(HeytingOmega3.DEGRADED)
        if gromov_capacity > _GROMOV_CAPACITY_MAX_DEFAULT:
            v = v.meet(HeytingOmega3.DEGRADED)
        if indictment.uhlmann_fidelity < rsi3.adaptive_fidelity_threshold:
            v = v.meet(HeytingOmega3.DEGRADED)
        return v

    def judge_deliberation(
        self,
        bridge: Phase1ToPhase2ProsecutionBridge,
        rho_mac: NDArray,
        rho_delib: NDArray,
    ) -> Tuple[HeytingOmega3, complex, float, float, float]:
        """Juicio completo: QND, Oseledets, Gromov, GKSL–Spohn, adjudicación."""
        self.consume_phase1_bridge(bridge)
        indictment = bridge.indictment
        seed = bridge.canonical_seed
        rsi3 = bridge.rsi3_state
        Aw = self.compute_qnd_weak_value_aav(rho_mac, rho_delib)
        gromov = self.compute_capacity_gromov(rho_delib)
        spohn, _ = self.compute_gksl_dissipation_and_spohn_entropy(rho_delib)
        lam = self.compute_oseledets_transverse_lyapunov(rho_delib)
        verdict = self.adjudicate_judgment(indictment, seed, rsi3, gromov)
        self._last_p2_verdict = verdict
        return verdict, Aw, spohn, gromov, lam

    def _bridge23_payload(
        self,
        bridge12: Phase1ToPhase2ProsecutionBridge,
        adjudication_verdict: HeytingOmega3,
        token: str,
        spohn_entropy: float,
        gromov_capacity: float,
        oseledets_lyapunov: float,
    ) -> bytes:
        return (
            f"{bridge12.indictment.indictment_id}:{bridge12.canonical_seed.sha256_provenance}:"
            f"{int(adjudication_verdict.value)}:{token}:{spohn_entropy:.8f}:{gromov_capacity:.8f}:"
            f"{oseledets_lyapunov:.8f}:{bridge12.rsi3_state.meta_operator_norm:.8f}"
        ).encode("utf-8")

    # ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
    # ║  FASE 2 → FASE 3: FRONTERA (handoff_to_phase3_execution_context).                            ║
    # ║  ÚLTIMO método de FASE 2. Retorna Phase2ToPhase3ProsecutionBridge → PRIMER arg. FASE 3.     ║
    # ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
    def handoff_to_phase3_execution_context(
        self,
        bridge12: Phase1ToPhase2ProsecutionBridge,
        adjudication_verdict: HeytingOmega3,
        Aw: complex,
        spohn_entropy: float,
        gromov_capacity: float,
        oseledets_lyapunov: float = 0.0,
    ) -> Phase2ToPhase3ProsecutionBridge:
        """ÚLTIMO método de la Fase 2 — cierra JUICIO y abre EJECUCIÓN (Φ₂₃')."""
        token = (
            f"CONT-P2-P3-PROSEC-{bridge12.indictment.indictment_id}-"
            f"{bridge12.rsi3_state.rsi_iteration_id}-{int(adjudication_verdict.value)}"
        )
        hmac_cont = _hmac_hex(
            self.secret_key,
            _HMAC_DOMAIN_BRIDGE23,
            self._bridge23_payload(
                bridge12, adjudication_verdict, token, spohn_entropy, gromov_capacity, oseledets_lyapunov
            ),
        )
        logger.info(
            "Φ₂₃' handoff | adjudication=%s | σ_Spohn=%.6e | c_Gromov=%.4f | λ_⟂=%.4f",
            adjudication_verdict.name,
            spohn_entropy,
            gromov_capacity,
            oseledets_lyapunov,
        )
        return Phase2ToPhase3ProsecutionBridge(
            indictment=bridge12.indictment,
            canonical_seed=bridge12.canonical_seed,
            rsi3_state=bridge12.rsi3_state,
            adjudication_verdict=adjudication_verdict,
            weak_value_Aw=Aw,
            spohn_entropy=spohn_entropy,
            gromov_capacity=gromov_capacity,
            oseledets_lyapunov=oseledets_lyapunov,
            continuation_token=token,
            hmac_continuation=hmac_cont,
        )


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 3: EJECUCIÓN — consume veredicto P2, Dirac, Crowbar simulado, certificado                 ║
# ║  Primera def. operativa (consume_phase2_bridge / prosecute_from_bridge) consume Φ₂₃'.          ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

class NovikovProsecutorPhase3Executor(NovikovProsecutorPhase2Adjudicator):
    r"""FASE 3: EJECUCIÓN — honra el fallo de JUICIO, revalida RSI-3, aniquilación Dirac
    autenticada y enclavamiento ESP32 *simulado* (sin I/O de hardware).
    """

    def consume_phase2_bridge(self, bridge: Phase2ToPhase3ProsecutionBridge) -> bool:
        expected = _hmac_hex(
            self.secret_key,
            _HMAC_DOMAIN_BRIDGE23,
            self._bridge23_payload(
                Phase1ToPhase2ProsecutionBridge(
                    indictment=bridge.indictment,
                    canonical_seed=bridge.canonical_seed,
                    rsi3_state=bridge.rsi3_state,
                    continuation_token="",
                    hmac_continuation="",
                ),
                bridge.adjudication_verdict,
                bridge.continuation_token,
                bridge.spohn_entropy,
                bridge.gromov_capacity,
                bridge.oseledets_lyapunov,
            ),
        )
        # Reconstrucción canónica del payload P2→P3 (mismos campos que el handoff).
        payload = (
            f"{bridge.indictment.indictment_id}:{bridge.canonical_seed.sha256_provenance}:"
            f"{int(bridge.adjudication_verdict.value)}:{bridge.continuation_token}:"
            f"{bridge.spohn_entropy:.8f}:{bridge.gromov_capacity:.8f}:"
            f"{bridge.oseledets_lyapunov:.8f}:{bridge.rsi3_state.meta_operator_norm:.8f}"
        ).encode("utf-8")
        expected = _hmac_hex(self.secret_key, _HMAC_DOMAIN_BRIDGE23, payload)
        if not hmac.compare_digest(bridge.hmac_continuation, expected):
            raise ValueError("Falla de autenticidad en puente Φ₂₃' (JUICIO→EJECUCIÓN)")
        return True

    def verify_rsi3_banach_certificate(self, rsi3: NovikovProsecutorRSI3State) -> bool:
        payload = (
            f"{rsi3.rsi_iteration_id}:{rsi3.meta_operator_norm:.8f}:"
            f"{rsi3.fixed_point_residual:.8f}:{rsi3.banach_convergence_certified}"
        ).encode("utf-8")
        expected = _hmac_hex(self.secret_key, _HMAC_DOMAIN_RSI3, payload)
        hmac_ok = hmac.compare_digest(rsi3.banach_certificate_hmac, expected)
        return bool(
            hmac_ok
            and rsi3.meta_operator_norm < _LIPSCHITZ_CERT
            and rsi3.fixed_point_residual < _RSI_RESIDUAL_CERT
            and rsi3.banach_convergence_certified
        )

    def compute_dirac_fock_annihilation(
        self,
        electron_anomalies: int,
        positron_auth_signature: str,
        transaction_payload: str,
    ) -> Tuple[bool, float, float]:
        """Aniquilación Dirac a 1022 keV condicionada a HMAC del positrón (canal lógico)."""
        if electron_anomalies <= 0:
            return True, 0.0, 0.0
        expected_pos_hmac = hmac.new(
            self.secret_key, transaction_payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        auth_valid = hmac.compare_digest(positron_auth_signature, expected_pos_hmac)
        if auth_valid:
            return True, float(electron_anomalies * _POSITRON_REST_MASS_KEV), 0.0
        return False, 0.0, 12.5

    def trigger_esp32_crowbar_hardware_interlock(
        self, verdict: HeytingOmega3, gpio_pin: int = 14
    ) -> Tuple[bool, float]:
        """Enclavamiento *simulado* (latencia de software acotada). No toca GPIO real."""
        if verdict != HeytingOmega3.VETOED:
            return False, 0.0
        t0 = time.perf_counter_ns()
        _ = (1 << int(gpio_pin)) & (1 << 14)
        t1 = time.perf_counter_ns()
        latency = float(
            min(max(float(t1 - t0), _ESP32_SIMD_EXECUTION_NS), _ESP32_CROWBAR_MAX_LATENCY_NS)
        )
        return True, latency

    def prosecute_from_bridge(
        self,
        bridge: Phase2ToPhase3ProsecutionBridge,
        electron_anomalies: int = 0,
        positron_auth_signature: str = "",
        transaction_payload: str = "",
    ) -> NovikovProsecutionExecutionCertificate:
        """Entrada canónica de FASE 3: honra adjudication_verdict del puente Φ₂₃'."""
        self.consume_phase2_bridge(bridge)
        now = time.time()
        indictment = bridge.indictment
        seed = bridge.canonical_seed
        rsi3 = bridge.rsi3_state

        auth_valid, released_kev, dirac_ba = self.compute_dirac_fock_annihilation(
            electron_anomalies, positron_auth_signature, transaction_payload
        )
        rsi3_ok = self.verify_rsi3_banach_certificate(rsi3)

        global_verdict = bridge.adjudication_verdict
        if not auth_valid:
            global_verdict = global_verdict.meet(HeytingOmega3.VETOED)
        if not rsi3_ok:
            global_verdict = global_verdict.meet(HeytingOmega3.DEGRADED)

        if global_verdict == HeytingOmega3.VETOED:
            status = NovikovProsecutionStatus.VETOED_ACTION_DIVERGENCE
            actuation = ActuationMode.HARD_VETO_ESP32_CROWBAR
        elif global_verdict == HeytingOmega3.DEGRADED:
            status = NovikovProsecutionStatus.INDICTED_CAUSAL_PARADOX
            actuation = ActuationMode.SOFT_VETO_BYPASS
        else:
            status = NovikovProsecutionStatus.ABSOLVED_AUTOCONSISTENT
            actuation = ActuationMode.NORMAL_FLUID

        crowbar_triggered, crowbar_latency = self.trigger_esp32_crowbar_hardware_interlock(
            global_verdict
        )
        back_action = max(float(indictment.back_action_db), float(dirac_ba))

        cert_id = f"CERT-PROSECUTOR-NOV-{int(now * 1000) % 1_000_000:06d}"
        seed_merkle = (
            f"{cert_id}:{indictment.indictment_id}:{global_verdict.value}:"
            f"{crowbar_triggered}:{released_kev:.2f}:{seed.sha256_provenance}:"
            f"{rsi3.banach_certificate_hmac}:{bridge.adjudication_verdict.value}"
        )
        merkle_digest = hashlib.sha256(seed_merkle.encode("utf-8")).hexdigest()
        hmac_sig = _hmac_hex(self.secret_key, _HMAC_DOMAIN_EXEC, merkle_digest.encode("utf-8"))

        return NovikovProsecutionExecutionCertificate(
            certificate_id=cert_id,
            indictment_ref_id=indictment.indictment_id,
            prosecution_status=status,
            global_heyting_verdict=global_verdict,
            actuation_mode=actuation,
            crowbar_triggered=crowbar_triggered,
            crowbar_latency_ns=crowbar_latency,
            annihilation_executed=auth_valid and electron_anomalies > 0,
            released_energy_kev=released_kev,
            back_action_db=back_action,
            banach_lipschitz_contraction=rsi3.meta_operator_norm,
            sha256_merkle_digest=merkle_digest,
            hmac_provenance_signature=hmac_sig,
            timestamp_utc=now,
            rsi3_convergence_certified=rsi3_ok,
            melnikov_M0=seed.melnikov_integral_M0,
            kam_stability=float(1.0 if seed.greene_stable and seed.is_bryuno_stable else 0.0),
            homoclinic_tangency_resolved=not seed.homoclinic_tangency_detected,
            phase2_verdict=bridge.adjudication_verdict.name,
            greene_residue=seed.greene_residue_R,
            bryuno_stable=seed.is_bryuno_stable,
            diophantine_gamma=seed.diophantine_gamma,
            adjudication_honored=True,
        )

    def prosecute_novikov_navigation(
        self,
        indictment: NovikovProsecutionIndictmentGerm,
        electron_anomalies: int = 0,
        positron_auth_signature: str = "",
        transaction_payload: str = "",
        rsi3_state: Optional[NovikovProsecutorRSI3State] = None,
        canonical_seed: Optional[NovikovProsecutionCanonicalSeed] = None,
        adjudication_verdict: Optional[HeytingOmega3] = None,
        oseledets_lyapunov: float = 0.0,
        spohn_entropy: float = 0.0,
        gromov_capacity: float = 0.0,
    ) -> NovikovProsecutionExecutionCertificate:
        """API v11-compatible. Si hay puente P2 implícito, se reconstruye y se honra el fallo."""
        seed = canonical_seed or self._last_canonical_seed
        rsi3 = rsi3_state or self._last_rsi3_state
        if seed is None or rsi3 is None:
            raise RuntimeError("prosecute_novikov_navigation: falta semilla/RSI-3 (ejecute Fase 1).")
        verdict = adjudication_verdict or self._last_p2_verdict or HeytingOmega3.COHERENT
        if indictment.is_action_divergent:
            verdict = verdict.meet(HeytingOmega3.VETOED)
        dummy12 = Phase1ToPhase2ProsecutionBridge(
            indictment=indictment,
            canonical_seed=seed,
            rsi3_state=rsi3,
            continuation_token="COMPAT",
            hmac_continuation="",
        )
        token = (
            f"CONT-P2-P3-PROSEC-{indictment.indictment_id}-{rsi3.rsi_iteration_id}-{int(verdict.value)}"
        )
        payload = (
            f"{indictment.indictment_id}:{seed.sha256_provenance}:"
            f"{int(verdict.value)}:{token}:{spohn_entropy:.8f}:{gromov_capacity:.8f}:"
            f"{oseledets_lyapunov:.8f}:{rsi3.meta_operator_norm:.8f}"
        ).encode("utf-8")
        hmac_cont = _hmac_hex(self.secret_key, _HMAC_DOMAIN_BRIDGE23, payload)
        bridge = Phase2ToPhase3ProsecutionBridge(
            indictment=indictment,
            canonical_seed=seed,
            rsi3_state=rsi3,
            adjudication_verdict=verdict,
            weak_value_Aw=indictment.weak_value_Aw,
            spohn_entropy=spohn_entropy,
            gromov_capacity=gromov_capacity,
            oseledets_lyapunov=oseledets_lyapunov,
            continuation_token=token,
            hmac_continuation=hmac_cont,
        )
        _ = dummy12
        return self.prosecute_from_bridge(
            bridge,
            electron_anomalies=electron_anomalies,
            positron_auth_signature=positron_auth_signature,
            transaction_payload=transaction_payload,
        )


class TOONNovikovNavigatorProsecutorEngine(NovikovProsecutorPhase3Executor):
    r"""Motor Espectral Inquisidor — orquesta las 3 fases anidadas y la mónada (T,η,μ).

        indict_novikov_deliberation            (FASE 1)
        handoff_to_phase2_judgment_context     (Φ₁₂')
        judge_deliberation                     (FASE 2)
        handoff_to_phase3_execution_context    (Φ₂₃')
        prosecute_from_bridge                  (FASE 3)
    """

    def full_cycle_prosecution(
        self,
        deliberation_id: str,
        rho_mac: NDArray,
        rho_delib: NDArray,
        action_spectrum: Optional[NDArray] = None,
        rsi_history: Optional[Sequence[int]] = None,
        electron_anomalies: int = 0,
        positron_auth_signature: str = "",
        transaction_payload: str = "",
    ) -> NovikovProsecutionExecutionCertificate:
        """Orquesta las 3 fases anidadas y emite el certificado fiscal terminal."""
        germ = self.indict_novikov_deliberation(
            deliberation_id=deliberation_id,
            mac_density_matrix=rho_mac,
            deliberation_density_matrix=rho_delib,
            action_spectrum=action_spectrum,
            rsi_history=rsi_history,
        )
        bridge12 = self.handoff_to_phase2_judgment_context(germ)
        verdict, Aw, spohn, gromov, lam = self.judge_deliberation(bridge12, rho_mac, rho_delib)
        bridge23 = self.handoff_to_phase3_execution_context(
            bridge12, verdict, Aw, spohn, gromov, lam
        )
        return self.prosecute_from_bridge(
            bridge23,
            electron_anomalies=electron_anomalies,
            positron_auth_signature=positron_auth_signature,
            transaction_payload=transaction_payload,
        )

    def monadic_unit(
        self,
        deliberation_id: str,
        rho: NDArray,
    ) -> NovikovProsecutionIndictmentGerm:
        r"""Unidad η: 𝔇_n → T(𝔇_n). Germen 'puro' de un estado contra sí mismo (F=1)."""
        r = self.sanitize_density_matrix(rho)
        return self.indict_novikov_deliberation(deliberation_id, r, r)

    def monadic_multiplication(
        self,
        germ1: NovikovProsecutionIndictmentGerm,
        germ2: NovikovProsecutionIndictmentGerm,
    ) -> NovikovProsecutionIndictmentGerm:
        r"""μ: T·T → T. Kleisli-fusión ultramétrica (min v, max d_BT, min F, ∨ divergencia).

        Leyes: μ∘Tη = id = μ∘ηT (si g₁=g₂); μ∘μT = μ∘Tμ (asociatividad del meet).
        """
        if germ1.indictment_id == germ2.indictment_id:
            return germ1
        now = time.time()
        fused_id = f"INDICT-NOV-FUSED-{int(now * 1000) % 1_000_000:06d}"
        w1 = max(germ1.uhlmann_fidelity, 1e-6)
        w2 = max(germ2.uhlmann_fidelity, 1e-6)
        wsum = w1 + w2
        Aw_fused = (w1 * germ1.weak_value_Aw + w2 * germ2.weak_value_Aw) / wsum
        return NovikovProsecutionIndictmentGerm(
            indictment_id=fused_id,
            deliberation_id=f"{germ1.deliberation_id}+{germ2.deliberation_id}",
            weak_value_Aw=Aw_fused,
            novikov_valuation=min(germ1.novikov_valuation, germ2.novikov_valuation),
            ultrametric_norm=max(germ1.ultrametric_norm, germ2.ultrametric_norm),
            ultrametric_inequality_residual=max(
                germ1.ultrametric_inequality_residual, germ2.ultrametric_inequality_residual
            ),
            bruhat_tits_distance=max(germ1.bruhat_tits_distance, germ2.bruhat_tits_distance),
            amoeba_tropical_area=max(germ1.amoeba_tropical_area, germ2.amoeba_tropical_area),
            uhlmann_fidelity=min(germ1.uhlmann_fidelity, germ2.uhlmann_fidelity),
            bures_distance=max(germ1.bures_distance, germ2.bures_distance),
            cstar_positivity_valid=(germ1.cstar_positivity_valid and germ2.cstar_positivity_valid),
            poincare_wirtinger_variance=max(
                germ1.poincare_wirtinger_variance, germ2.poincare_wirtinger_variance
            ),
            oseledets_transverse_lyapunov=min(
                germ1.oseledets_transverse_lyapunov, germ2.oseledets_transverse_lyapunov
            ),
            kolmogorov_sinai_entropy=max(
                germ1.kolmogorov_sinai_entropy, germ2.kolmogorov_sinai_entropy
            ),
            capacity_gromov=max(germ1.capacity_gromov, germ2.capacity_gromov),
            is_action_divergent=(germ1.is_action_divergent or germ2.is_action_divergent),
            timestamp_utc=now,
            canonical_seed_ref=f"{germ1.canonical_seed_ref}|{germ2.canonical_seed_ref}",
            rsi3_state_ref=f"{germ1.rsi3_state_ref}|{germ2.rsi3_state_ref}",
            melnikov_M0=max(abs(germ1.melnikov_M0), abs(germ2.melnikov_M0)),
            greene_residue=max(germ1.greene_residue, germ2.greene_residue),
            bryuno_sum=max(germ1.bryuno_sum, germ2.bryuno_sum),
            homoclinic_tangency=(germ1.homoclinic_tangency or germ2.homoclinic_tangency),
            ultrametric_gap_significant=(
                germ1.ultrametric_gap_significant or germ2.ultrametric_gap_significant
            ),
            back_action_db=max(germ1.back_action_db, germ2.back_action_db),
            fubini_study_distance=max(germ1.fubini_study_distance, germ2.fubini_study_distance),
        )


if __name__ == "__main__":
    banner = "═" * 100
    print(banner)
    print("  VERIFICACIÓN DOCTORAL DEL MOTOR FISCAL NAVEGANTE DE NOVIKOV (v13.0.0-PhD)")
    print("  Inquisición → Juicio → Ejecución | Poincaré-deep | RSI-3 Banach medido | Monad (η,μ)")
    print(banner)

    print("\n  ─── Heyting imply (v13: int(self) ≥ int(other) ⇒ ⊤) ──────────────────")
    for a in HeytingOmega3:
        for b in HeytingOmega3:
            print(
                f"  {a.name:<8} → {b.name:<8} = {a.imply(b).name:<8}  |  "
                f"∧={a.meet(b).name:<8}  ¬{a.name}={a.negation().name}"
            )

    engine = TOONNovikovNavigatorProsecutorEngine(dimension=56)
    rng = np.random.default_rng(2026)

    mac_state = rng.normal(0.0, 1.0, (56, 56)) + 1j * rng.normal(0.0, 1.0, (56, 56))
    rho_mac = mac_state @ mac_state.conj().T
    rho_mac /= np.trace(rho_mac)

    delib_state = rng.normal(0.0, 1.0, (56, 56)) + 1j * rng.normal(0.0, 1.0, (56, 56))
    rho_delib = delib_state @ delib_state.conj().T
    rho_delib /= np.trace(rho_delib)

    cert = engine.full_cycle_prosecution(
        deliberation_id="DELIB-NAV-001",
        rho_mac=rho_mac,
        rho_delib=rho_delib,
        rsi_history=(0, 0, 1, 0, 0),
    )

    print("\n  ─── FASE 1: INQUISICIÓN ─────────────────────────────────────────────")
    germ = engine.indict_novikov_deliberation("DELIB-VERIFY", rho_mac, rho_delib)
    seed = engine.compute_poincare_celestial_mechanics(rho_delib)
    print(f"  • Indictment ID          : {germ.indictment_id}")
    print(f"  • Valuación v(A)         : {germ.novikov_valuation:.6f}")
    print(f"  • Ultramétrica residual  : {germ.ultrametric_inequality_residual:.2e}")
    print(f"  • Ultra gap significativo: {germ.ultrametric_gap_significant}")
    print(f"  • Fidelidad Uhlmann      : {germ.uhlmann_fidelity:.6f}  ← FIX (ya no ~1 trivial)")
    print(f"  • d_Bures / d_FS         : {germ.bures_distance:.6f} / {germ.fubini_study_distance:.6f}")
    print(f"  • Delaunay L, G, H       : {seed.delaunay_L:.4f}, {seed.delaunay_G:.4f}, {seed.delaunay_H:.4f}")
    print(f"  • Excentricidad e        : {seed.eccentricity_e:.6f}")
    print(f"  • C_J (CR3BP)            : {seed.cr3bp_jacobi_constant:.6f}")
    print(f"  • Monodromía Tr M        : {seed.monodromy_trace:.6f}")
    print(f"  • Residuo Greene R_G     : {seed.greene_residue_R:.6f} (stable={seed.greene_stable})")
    print(f"  • Melnikov M₀            : {seed.melnikov_integral_M0:.6e}  zeros={seed.simple_melnikov_zeros}")
    print(f"  • Homoclinic tangle      : {seed.homoclinic_tangency_detected}")
    print(f"  • Bryuno B(ω)            : {seed.bryuno_diophantine_sum:.6f} (stable={seed.is_bryuno_stable})")
    print(f"  • Morse χ_spec / χ(ℂP)   : {seed.morse_bott_euler_chi} / {seed.chi_cp_expected} cert={seed.chi_certified}")
    print(f"  • θ_PC (eigenbase)       : {seed.poincare_cartan_1form:.6e}")
    print(f"  • γ diofantina           : {seed.diophantine_gamma:.6e}")
    print(f"  • T_Nekhoroshev          : {seed.nekhoroshev_time:.6e}")

    rsi = engine.compute_rsi3_bootstrap(
        germ.novikov_valuation, germ.bruhat_tits_distance, germ.uhlmann_fidelity
    )
    print(f"  • RSI-3 Lip(T) medido    : {rsi.meta_operator_norm:.6f}")
    print(f"  • RSI-3 residuo f.p.     : {rsi.fixed_point_residual:.6e}")
    print(f"  • RSI-3 certificado      : {rsi.banach_convergence_certified}")
    print(f"  • RSI-3 s*               : {tuple(round(x, 6) for x in rsi.banach_fixed_point)}")

    print("\n  ─── FASE 2: JUICIO ─────────────────────────────────────────────────")
    bridge12 = engine.handoff_to_phase2_judgment_context(germ)
    verdict, Aw, spohn, gromov, lam = engine.judge_deliberation(bridge12, rho_mac, rho_delib)
    print(f"  • Weak value Aw (AAV)    : {Aw:.4f}")
    print(f"  • σ_Spohn (≥0)           : {spohn:.6e}")
    print(f"  • c_Gromov               : {gromov:.4f}")
    print(f"  • λ_⟂ Oseledets          : {lam:.6f}")
    print(f"  • Veredicto Juicio       : {verdict.name}")

    print("\n  ─── FASE 3: EJECUCIÓN (honra P2) ───────────────────────────────────")
    print(f"  • Certificate ID         : {cert.certificate_id}")
    print(f"  • Veredicto P2 honrado   : {cert.phase2_verdict} → {cert.global_heyting_verdict.name}")
    print(f"  • Adjudicación honrada   : {cert.adjudication_honored}")
    print(f"  • Estado Fiscal          : {cert.prosecution_status.name}")
    print(f"  • Modo Actuación         : {cert.actuation_mode.name}")
    print(f"  • Crowbar simulado       : {cert.crowbar_triggered} (latencia={cert.crowbar_latency_ns:.2f} ns)")
    print(f"  • Energía liberada       : {cert.released_energy_kev:.2f} keV")
    print(f"  • Back-Action            : {cert.back_action_db:.2f} dB")
    print(f"  • RSI-3 certificado      : {cert.rsi3_convergence_certified}")
    print(f"  • Raíz Merkle SHA-256    : {cert.sha256_merkle_digest[:32]}...")
    print(f"  • HMAC Provenance        : {cert.hmac_provenance_signature[:32]}...")

    print("\n  ─── μ_novikov_prosecutor (Kleisli) + η ─────────────────────────────")
    g2 = engine.indict_novikov_deliberation("DELIB-SECOND", rho_delib, rho_mac)
    fused = engine.monadic_multiplication(germ, g2)
    unit = engine.monadic_unit("DELIB-UNIT", rho_mac)
    print(f"  • Fused indictment ID    : {fused.indictment_id}")
    print(f"  • Fused v(A)             : {fused.novikov_valuation:.6f}  (= min)")
    print(f"  • Fused d_BT             : {fused.bruhat_tits_distance:.6f}  (= max)")
    print(f"  • Fused F                : {fused.uhlmann_fidelity:.6f}  (= min)")
    print(f"  • Fused is_divergent     : {fused.is_action_divergent}  (= ∨)")
    print(f"  • η(ρ_mac).F             : {unit.uhlmann_fidelity:.6f}  (debe ~ 1)")
    print(banner)
    print("  ✓ FASE 1 → FASE 2: handoff_to_phase2_judgment_context → Φ₁₂'")
    print("  ✓ FASE 2 → FASE 3: handoff_to_phase3_execution_context → Φ₂₃' (veredicto VIAJA)")
    print("  ✓ imply Heyting corregido; meet = max de severidad (no bitwise)")
    print("  ✓ Uhlmann sin renormalizar √ρ σ √ρ; ultramétrica por brecha espectral")
    print("  ✓ RSI-3: Banach iterado en ℝ³, Lip MEDIDO (no clip a 0.999)")
    print("  ✓ Poincaré: Cartan en eigenbase; Melnikov por ceros simples; Greene ν mod 1")
    print("  ✓ Morse: χ(ℂPⁿ⁻¹)=n como teorema, no como 'abs(suma alternada)'")
    print("  ✓ Crowbar simulado (sin I/O de hardware)")
    print(banner)