
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo    : BoolNovikovToonBufferProsecutorEngine v15.0.0-PhD                                        ║
║ Ubicación : app/wisdom/toon_buffer_prosecutor_engine.py                                              ║
║ Versión   : 15.0.0-ENGINE-BOOL-NOVIKOV-TOON-BUFFER-PROSECUTOR-POINCARE-KAM-RSI3-PhD                  ║
║ Función   : Motor Espectral Fiscal Inquisidor Dual del TOON Buffer: Compuerta de Aniquilación de     ║
║             Clifford-Dirac Cl_{1,3}(ℂ) con física de Crowbar SCR, Álgebra Boole–Novikov–Hodge,       ║
║             Mecánica Celeste Simpléctica de Poincaré–Cartan–KAM y Automejora Recursiva RSI³.         ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA EN TRES FASES ANIDADAS (CADENA DE HERENCIA CON CONTINUIDAD MORFISMÁTICA)
─────────────────────────────────────────────────────────────────────────────────────
FASE 1 · _Phase1DiracAlgebraicKernel
    Álgebra de Clifford relativista Cl_{1,3}(ℂ) (matrices bispinoriales γ^μ, quiralidad γ⁵ y defecto
    de anticonmutación {γ^μ, γ^ν} = 2η^{μν}I₄), compuerta de aniquilación e⁻ + e⁺ → 2γ con autorización
    HMAC, física eléctrica de descarga Crowbar SCR (di/dt e integral de Joule ∫i²dt), forma normal de
    Smith exacta sobre ℤ, 0-Laplaciano de Hodge Δ₀, transformada rápida de Walsh–Hadamard con residuo
    de Parseval y valuación ultramétrica de Krull–Novikov.
    ÚLTIMO MÉTODO → compute_phase1_witness(...)  ⇒  Phase1ProsecutionWitness

FASE 2 · _Phase2FluidPoincareKernel(_Phase1DiracAlgebraicKernel)
    PRIMER MÉTODO  ← evaluate_toon_buffer_flow(witness1: Phase1ProsecutionWitness, ...)  [F1 → F2]
    Espectro ergódico multiplicativo de Oseledets completo, cota de entropía de Pesin y sobrepresión
    de golpe de bélier de Zhukovsky-Allievi; residuo de Cauchy-Riemann-Wirtinger; capacidad simpléctica
    de Gromov; mecánica celeste de Poincaré: variables canónicas no singulares (Λ, ξ₁, ξ₂; λ, η₁, η₂)
    con auditoría simpléctica ‖Mᵀ J₆ M − J₆‖_F, determinantes de torsión de Kolmogorov y de Arnol'd,
    condición diofantina, serie canónica de Lie de Deprit-Hori exp(L_W)H, horizonte exponencial de
    Nekhoroshev, invariante integral relativo de Poincaré-Cartan (∮ p dq) vía integrador simpléctico
    de Yoshida de 4.º orden, monodromía de Floquet-Broucke e integral homoclínica de Melnikov;
    y Torre de Automejora Recursiva de Nivel 3 (RSI³: síntesis de Picard-Lyapunov, meta-endofuntor
    M_Ψ de Schulz-Picard, coherencia de la mónada de Kleisli y guardia de solidez de Löb-Tarski).
    ÚLTIMO MÉTODO → compose_phase2_witness(witness1: Phase1ProsecutionWitness, ...)  ⇒  Phase2ProsecutionWitness

FASE 3 · _Phase3HeytingVerdictKernel(_Phase2FluidPoincareKernel)
    PRIMER MÉTODO  ← _phase3_clause_table(witness2: Phase2ProsecutionWitness)            [F2 → F3]
    Tabla de cláusulas de veto y degradación, adjudicación intuicionista en el topos de Heyting Ω₃
    con operador modal de Lawvere-Tierney j: Ω₃ → Ω₃ y pasaporte criptográfico determinista HMAC-SHA256.
    ÚLTIMO MÉTODO → execute_toon_buffer_prosecutor_audit(...)  ⇒  ToonBufferProsecutorSpectrumReport
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
from dataclasses import dataclass
from enum import IntEnum
from functools import reduce
from typing import Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.BoolNovikovToonBufferProsecutorEngine")

# ─── Constantes soberanas ────────────────────────────────────────────────────────────────────────
_ENGINE_VERSION: Final[str] = (
    "15.0.0-ENGINE-BOOL-NOVIKOV-TOON-BUFFER-PROSECUTOR-POINCARE-KAM-RSI3-PhD"
)
__version__: Final[str] = _ENGINE_VERSION

_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_TOON_BUFFER_PROSECUTOR_KEY_v14.0.0"
_MIN_SECRET_KEY_BYTES: Final[int] = 16
_HMAC_DOMAIN_POSITRON: Final[bytes] = b"TOON-PROSECUTOR-POSITRON-AUTH-v14.0|"
_HMAC_DOMAIN_FOCK: Final[bytes] = b"TOON-PROSECUTOR-FOCK-RECORD-v14.0|"
_HMAC_DOMAIN_PASSPORT: Final[bytes] = b"TOON-PROSECUTOR-PASSPORT-v14.0|"

_WILKINSON_EPSILON: Final[float] = 1.0e-15
_ELECTRON_REST_ENERGY_KEV: Final[float] = 510.99895
_POSITRON_REST_MASS_KEV: Final[float] = 2.0 * _ELECTRON_REST_ENERGY_KEV
_CROWBAR_LATENCY_BUDGET_NS: Final[float] = 392.15
_LATENCY_PROVENANCE: Final[str] = "DECLARED_HARDWARE_SPEC_NOT_MEASURED"

_KLEISLI_SAFETY_MARGIN: Final[float] = 0.002375
_DEFAULT_STAGE_FACTORS: Final[Tuple[float, ...]] = (0.95, 0.95, 0.95)
_DEFAULT_QUANT_SCALE: Final[float] = 1000.0
_DEFAULT_BUFFER_CAPACITY: Final[float] = 100.0
_INTEGER_OVERFLOW_GUARD: Final[float] = float(2**62)

_SYMPLECTIC_TOL: Final[float] = 1.0e-10
_CARTAN_INVARIANT_TOL: Final[float] = 1.0e-7
_HOMOLOGICAL_TOL: Final[float] = 1.0e-9
_DIOPHANTINE_GAMMA: Final[float] = 0.05
_DIOPHANTINE_TAU: Final[float] = 1.0
_DIOPHANTINE_K_MAX: Final[int] = 64
_RSI_MAX_DEPTH: Final[int] = 3
_RSI_MAX_ITER: Final[int] = 2000
_RSI_PROBES: Final[int] = 128
_KLEISLI_COHERENCE_TOL: Final[float] = 1.0e-12


# ─── Política, errores y álgebra de Heyting con topología de Lawvere-Tierney ────────────────────
@dataclass(frozen=True)
class ProsecutorPolicy:
    """Umbrales declarados de gobernanza fiscal. Decisiones de política, no constantes físicas."""
    anomaly_threshold: float = 2.0
    krull_floor: float = 0.10
    variance_ceiling: float = 10.0
    utilization_ceiling_pct: float = 100.0
    water_hammer_veto: bool = False
    gromov_tol: float = 1.0e-12
    identity_tol: float = 1.0e-9

    def __post_init__(self) -> None:
        for name in (
            "anomaly_threshold",
            "variance_ceiling",
            "utilization_ceiling_pct",
            "gromov_tol",
            "identity_tol",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} debe ser finito y no negativo.")
        if not math.isfinite(self.krull_floor):
            raise ValueError("krull_floor debe ser finito.")


class ProsecutorGateError(ValueError):
    """Rechazo fail-closed en una compuerta de integridad entre fases."""


class HeytingTruthValue(IntEnum):
    """Topos de Heyting Ω₃ equipado con clausura modal de Lawvere-Tierney.

    Orden lógico: COHERENT (⊤, rango 2) ≻ DEGRADED (rango 1) ≻ VETOED (⊥, rango 0).
    """

    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x02

    @property
    def rank(self) -> int:
        return 2 - int(self)

    def meet(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        """Ínfimo (conjunción intuicionista ∧) en Ω₃."""
        return self if self.rank <= other.rank else other

    def join(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        """Supremo (disyunción intuicionista ∨) en Ω₃."""
        return self if self.rank >= other.rank else other

    def implies(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        """Implicación interna de Heyting (a ⇒ b)."""
        return HeytingTruthValue.COHERENT if self.rank <= other.rank else other

    def negate(self) -> "HeytingTruthValue":
        """Pseudo-complemento intuicionista ¬a = (a ⇒ VETOED)."""
        return self.implies(HeytingTruthValue.VETOED)

    def lawvere_tierney_sheafify(self, stabilize_degraded: bool = False) -> "HeytingTruthValue":
        """Operador modal de Lawvere-Tierney j: Ω₃ → Ω₃ (j(⊤)=⊤, j(j(p))=j(p), j(p∧q)=j(p)∧j(q))."""
        if not stabilize_degraded:
            return self
        if self in (HeytingTruthValue.COHERENT, HeytingTruthValue.DEGRADED):
            return HeytingTruthValue.COHERENT
        return HeytingTruthValue.VETOED


# ─── Contratos de reportes por fase (compatibles con v13/v14 y extendidos para v15) ──────────────
@dataclass(frozen=True)
class DiracAnnihilationProsecutionReport:
    """Compuerta de aniquilación de Clifford-Dirac y descarga Crowbar SCR."""
    anomaly_norm: float
    context_digest: str
    positron_token_valid: bool
    energy_released_kev: float
    photons_emitted: int
    crowbar_triggered: bool
    crowbar_latency_budget_ns: float
    fock_authorization_digest: str
    latency_provenance: str = _LATENCY_PROVENANCE
    clifford_anticommutator_defect: float = 0.0
    chiral_gamma5_involutory_defect: float = 0.0
    mandelstam_s_kev2: float = 0.0
    dirac_cross_section_ratio: float = 1.0
    crowbar_peak_di_dt_a_per_us: float = 0.0
    crowbar_joule_integral_a2s: float = 0.0


@dataclass(frozen=True)
class AlgebraicTopologyProsecutionReport:
    """Álgebra Boole–Novikov–Hodge: Smith exacta sobre ℤ, Walsh normalizado, Hodge Δ₀ y Krull."""
    smith_source: str
    quant_scale: Optional[float]
    smith_invariants: Tuple[int, ...]
    smith_torsion_free: bool
    betti_number: int
    betti_number_zero: bool
    rademacher_walsh_energy: float
    rademacher_walsh_parseval_residual: float
    krull_valuation: float
    novikov_norm: float
    krull_floor_ok: bool
    torsion_subgroup_order: int = 1
    euler_characteristic: int = 0
    hodge_spectral_gap: float = 0.0


@dataclass(frozen=True)
class Phase1ProsecutionWitness:
    """Testigo sellado P1: atestación de MIC, exponentes, Clifford-Dirac y álgebra Boole-Novikov."""
    matrix_sha256: str
    exponents_sha256: str
    dirac_report: DiracAnnihilationProsecutionReport
    algebra_report: AlgebraicTopologyProsecutionReport


@dataclass(frozen=True)
class BufferProsecutionReport:
    """Auditoría de caudal en T*Q: varianza, espectro de Oseledets y sobrepresión de Zhukovsky."""
    flow_variance: float
    spectral_radius: float
    oseledets_lyapunov_lambda: float
    buffer_utilization_pct: float
    overflow_reasons: Tuple[str, ...]
    lyapunov_spectrum: Tuple[float, ...] = ()
    pesin_entropy_bound: float = 0.0
    zhukovsky_surge_pressure: float = 0.0

    @property
    def water_hammer_overflow(self) -> bool:
        return len(self.overflow_reasons) > 0


@dataclass(frozen=True)
class MelnikovHomoclinicReport:
    """Integral de Melnikov de la separatriz de Duffing con control de cuadratura analítica."""
    offset: float
    amplitude: float
    min_m: float
    max_m: float
    chaos_detected: bool
    quadrature_residual: float


@dataclass(frozen=True)
class KAMDynamicsProsecutionReport:
    """Dinámica de Poincaré–KAM: Wirtinger, Gromov, Melnikov, Poincaré no singular y Cartan."""
    cauchy_riemann_residual: float
    gromov_ball_capacity: float
    gromov_cylinder_capacity: float
    gromov_rigid: bool
    melnikov: MelnikovHomoclinicReport
    delaunay_actions: Tuple[float, float, float] = (1.0, 0.994987, 0.975170)
    poincare_canonical_coords: Tuple[float, float, float, float, float, float] = (
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )
    symplectic_jacobian_defect: float = 0.0
    kolmogorov_torsion_det: float = 0.0
    arnold_isoenergetic_det: float = 0.0
    is_kam_non_degenerate: bool = True
    diophantine_ok: bool = True
    homological_residual: float = 0.0
    lie_series_second_order_norm: float = 0.0
    nekhoroshev_stability_time: float = 0.0
    poincare_cartan_invariant_drift: float = 0.0
    floquet_monodromy_trace: float = 2.0
    is_floquet_elliptic_stable: bool = True


@dataclass(frozen=True)
class RSI3ProsecutionCertificate:
    """Certificado formal de la torre de Automejora Recursiva de Nivel 3 (RSI³) del Fiscal."""
    depth_reached: int
    lip_level1: float
    lip_level2_meta: float
    lip_probe_estimate: float
    fixed_point: Tuple[float, ...]
    banach_aposteriori_error: float
    lyapunov_descent_verified: bool
    kleisli_associativity_defect: float
    kleisli_unit_defect: float
    loeb_soundness_verified: bool
    accepted: bool


@dataclass(frozen=True)
class KleisliContractionReport:
    """Contracción de Banach de la mónada Kleisli T³ y certificación RSI³."""
    stage_factors: Tuple[float, ...]
    lip: float
    kappa: float
    contraction_margin_cubed: float
    rsi_certificate: Optional[RSI3ProsecutionCertificate] = None


@dataclass(frozen=True)
class Phase2ProsecutionWitness:
    """Testigo sellado P2: enlaza P1 con caudal, dinámica de Poincaré-KAM y torre RSI³."""
    phase1: Phase1ProsecutionWitness
    flow: BufferProsecutionReport
    kam: KAMDynamicsProsecutionReport
    banach: KleisliContractionReport
    internal_identities_ok: bool


@dataclass(frozen=True)
class HeytingClauseTable:
    """Cláusulas fallidas separadas en fallos de veto (→ VETOED) y degradación (→ DEGRADED)."""
    veto_failures: Tuple[str, ...]
    degrade_failures: Tuple[str, ...]


@dataclass(frozen=True)
class ToonBufferProsecutorSpectrumReport:
    """Reporte espectral final del Fiscal Inquisidor (v15), con todos los alias v13/v14."""
    buffer_report: BufferProsecutionReport
    dirac_report: DiracAnnihilationProsecutionReport
    algebra_report: AlgebraicTopologyProsecutionReport
    kam_report: KAMDynamicsProsecutionReport
    banach_report: KleisliContractionReport
    clauses: HeytingClauseTable
    truth_value: HeytingTruthValue
    passport_hash: str

    @property
    def rademacher_walsh_norm(self) -> float:
        return self.algebra_report.rademacher_walsh_energy

    @property
    def smith_torsion_free(self) -> bool:
        return self.algebra_report.smith_torsion_free

    @property
    def betti_number_zero(self) -> bool:
        return self.algebra_report.betti_number_zero

    @property
    def krull_valuation(self) -> float:
        return self.algebra_report.krull_valuation

    @property
    def novikov_norm(self) -> float:
        return self.algebra_report.novikov_norm

    @property
    def cauchy_riemann_residual(self) -> float:
        return self.kam_report.cauchy_riemann_residual

    @property
    def is_gromov_violated(self) -> bool:
        return not self.kam_report.gromov_rigid

    @property
    def melnikov_chaos_detected(self) -> bool:
        return self.kam_report.melnikov.chaos_detected

    @property
    def banach_contraction_lip(self) -> float:
        return self.banach_report.lip

    @property
    def capacity_jerk_d3C_dt3(self) -> float:
        """Alias v13/v14: margen cúbico de contracción κ³ = (−ln Lip(T³))³."""
        return self.banach_report.contraction_margin_cubed

    @property
    def rsi_certificate(self) -> Optional[RSI3ProsecutionCertificate]:
        return self.banach_report.rsi_certificate


def _validate_secret_key(secret_key: bytes) -> None:
    """Valida tipo, longitud mínima y advierte si se utiliza la clave por defecto."""
    if not isinstance(secret_key, (bytes, bytearray)):
        raise TypeError("secret_key debe ser bytes.")
    if len(secret_key) < _MIN_SECRET_KEY_BYTES:
        raise ValueError(f"secret_key requiere al menos {_MIN_SECRET_KEY_BYTES} bytes.")
    if bytes(secret_key) == _DEFAULT_SECRET_KEY:
        logger.warning("Clave HMAC por defecto en uso; no apta para producción.")


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 1 — ÁLGEBRA DE CLIFFORD-DIRAC Cl_{1,3}(ℂ), CROWBAR SCR Y ÁLGEBRA BOOLE–NOVIKOV–HODGE
# ════════════════════════════════════════════════════════════════════════════════════════════════════
class _Phase1DiracAlgebraicKernel:
    """FASE 1: Compuerta de aniquilación de Clifford-Dirac, Crowbar SCR y álgebra Boole–Novikov.

    Ningún método de esta fase depende de las Fases 2 o 3.
    Su último método (`compute_phase1_witness`) emite el testigo `Phase1ProsecutionWitness`
    que inaugura el primer método de la Fase 2 (`evaluate_toon_buffer_flow`).
    """

    def __init__(self, secret_key: bytes, policy: ProsecutorPolicy) -> None:
        _validate_secret_key(secret_key)
        self._secret_key = bytes(secret_key)
        self._policy = policy

    # ── Ingesta y digests deterministas ─────────────────────────────────────────────────────────

    @staticmethod
    def _validate_cost_matrix(apu_cost_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        mat = np.asarray(apu_cost_matrix, dtype=np.float64)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] < 1:
            raise ValueError(f"La MIC debe ser cuadrada y no vacía; forma recibida {mat.shape}.")
        if not np.all(np.isfinite(mat)):
            raise ValueError("La MIC debe ser finita.")
        return mat

    @staticmethod
    def _validate_exponents(exponents: NDArray[np.float64]) -> NDArray[np.float64]:
        exps = np.asarray(exponents, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(exps)):
            raise ValueError("Los exponentes de Novikov deben ser finitos.")
        return exps

    @staticmethod
    def _array_digest(arr: NDArray[np.float64]) -> str:
        """Digest determinista SHA-256 sobre forma y representación IEEE-754 little-endian."""
        data = np.ascontiguousarray(arr, dtype="<f8")
        return hashlib.sha256(f"{data.shape}".encode("ascii") + b"|" + data.tobytes()).hexdigest()

    @staticmethod
    def _anomaly_norm(mat: NDArray[np.float64]) -> float:
        return float(np.linalg.norm(mat - np.eye(mat.shape[0]), ord="fro"))

    @staticmethod
    def _dirac_context_digest(matrix_sha256: str, anomaly_norm: float) -> str:
        return hashlib.sha256(
            f"{matrix_sha256}|{float(anomaly_norm).hex()}".encode("ascii")
        ).hexdigest()

    def _positron_expected(self, context_digest: str) -> str:
        return hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_POSITRON + context_digest.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()

    def _verify_positron_token(self, context_digest: str, token: Optional[str]) -> bool:
        if not isinstance(token, str) or not token.isascii():
            return False
        return hmac.compare_digest(self._positron_expected(context_digest), token)

    def issue_human_positron_authorization(self, apu_cost_matrix: NDArray[np.float64]) -> str:
        """Emite el token de autorización humana HMAC para una MIC concreta."""
        mat = self._validate_cost_matrix(apu_cost_matrix)
        context = self._dirac_context_digest(self._array_digest(mat), self._anomaly_norm(mat))
        return self._positron_expected(context)

    # ── Álgebra de Clifford Relativista Cl_{1,3}(ℂ), QED 2γ y Física del Crowbar SCR ────────────

    @staticmethod
    def _audit_clifford_dirac_algebra() -> Tuple[float, float]:
        """Verifica la representación de Dirac-Pauli de Cl_{1,3}(ℂ): {γ^μ, γ^ν} = 2η^{μν}I₄.

        Construye las cuatro matrices bispinoriales 4×4 γ^0, γ^1, γ^2, γ^3 y el operador quiral
        γ⁵ = i γ^0 γ^1 γ^2 γ^3, devolviendo:
          1. El defecto máximo de anticonmutación de Clifford max_{μ,ν} ‖{γ^μ, γ^ν} − 2η^{μν}I₄‖_F.
          2. El defecto de involución quiral ‖(γ⁵)² − I₄‖_F + ‖{γ⁵, γ^0}‖_F.
        """
        I2 = np.eye(2, dtype=np.complex128)
        Z2 = np.zeros((2, 2), dtype=np.complex128)
        sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
        sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

        g0 = np.block([[I2, Z2], [Z2, -I2]])
        g1 = np.block([[Z2, sx], [-sx, Z2]])
        g2 = np.block([[Z2, sy], [-sy, Z2]])
        g3 = np.block([[Z2, sz], [-sz, Z2]])
        gammas = (g0, g1, g2, g3)
        eta = np.diag([1.0, -1.0, -1.0, -1.0])
        I4 = np.eye(4, dtype=np.complex128)

        max_defect = 0.0
        for mu in range(4):
            for nu in range(4):
                anticomm = gammas[mu] @ gammas[nu] + gammas[nu] @ gammas[mu]
                target = 2.0 * eta[mu, nu] * I4
                defect = float(la.norm(anticomm - target, ord="fro"))
                if defect > max_defect:
                    max_defect = defect

        g5 = 1.0j * (g0 @ g1 @ g2 @ g3)
        chiral_defect = float(
            la.norm(g5 @ g5 - I4, ord="fro") + la.norm(g5 @ g0 + g0 @ g5, ord="fro")
        )
        return max_defect, chiral_defect

    @staticmethod
    def _dirac_annihilation_kinematics_and_crowbar(
        anomaly_norm: float,
        crowbar_triggered: bool,
    ) -> Tuple[float, float, float, float]:
        """Cinemática relativista e⁻ + e⁺ → 2γ (fórmula de Dirac) y transitorio eléctrico Crowbar.

        • Velocidad efectiva reducida β = tanh(‖ΔA‖_F / (1 + ‖ΔA‖_F)) ∈ [0, 1).
        • Factor de Lorentz γ_L = 1 / √(1 − β²), invariante de Mandelstam s = 4 γ_L² m_e² c⁴.
        • Razón de sección eficaz de Dirac (respecto a Thomson σ_T = 8π r_e² / 3):
              σ_{2γ} / σ_T = [3 / (8 γ_L² β)] · [((3 − β⁴)/β) ln((1+β)/(1−β)) − 2(2 − β²)]
          con límite umbral exacto 3 / (8 β) cuando β → 0.
        • Si el Crowbar SCR se dispara, modela la descarga sobre L_bus = 1.2 μH, R_crowbar = 0.05 Ω
          con tensión de bus equivalente V_bus = 100 · (1 + anomaly_norm) V:
              di/dt|_{t=0} = V_bus / L_bus [A/μs],   ∫₀^∞ i²(t) dt = C_bus V_bus² / (2 R_crowbar).
        """
        beta = math.tanh(anomaly_norm / (1.0 + anomaly_norm))
        gamma_l = 1.0 / math.sqrt(max(1.0e-15, 1.0 - beta * beta))
        mandelstam_s = 4.0 * (gamma_l * gamma_l) * (_ELECTRON_REST_ENERGY_KEV ** 2)

        if beta < 1.0e-6:
            sigma_ratio = 1.0
        else:
            log_term = ((3.0 - beta ** 4) / beta) * math.log((1.0 + beta) / (1.0 - beta))
            poly_term = 2.0 * (2.0 - beta * beta)
            sigma_ratio = float(
                (3.0 / (8.0 * gamma_l * gamma_l)) * (log_term - poly_term)
            )

        if not crowbar_triggered:
            return mandelstam_s, sigma_ratio, 0.0, 0.0

        v_bus = 100.0 * (1.0 + anomaly_norm)
        l_bus_uh = 1.2
        c_bus_f = 1.0e-4
        r_crowbar_ohm = 0.05
        di_dt_a_per_us = float(v_bus / l_bus_uh)
        joule_integral = float(0.5 * c_bus_f * (v_bus * v_bus) / r_crowbar_ohm)
        return mandelstam_s, sigma_ratio, di_dt_a_per_us, joule_integral

    def execute_dirac_and_crowbar_prosecution(
        self,
        anomaly_norm: float,
        matrix_sha256: str,
        positron_token: Optional[str] = None,
        force_veto: bool = False,
    ) -> DiracAnnihilationProsecutionReport:
        """Aniquilación e⁻ + e⁺ → 2γ en Cl_{1,3}(ℂ) si hay autorización; si no, disparo Crowbar."""
        if not math.isfinite(anomaly_norm) or anomaly_norm < 0.0:
            raise ValueError("La anomalía debe ser finita y no negativa.")
        context = self._dirac_context_digest(matrix_sha256, anomaly_norm)
        token_valid = self._verify_positron_token(context, positron_token)
        clifford_def, chiral_def = self._audit_clifford_dirac_algebra()

        authorized = (
            token_valid
            and anomaly_norm <= self._policy.anomaly_threshold
            and not force_veto
            and clifford_def <= 1.0e-12
        )
        if authorized:
            photons, energy, crowbar, budget = 2, _POSITRON_REST_MASS_KEV, False, 0.0
        else:
            photons, energy, crowbar, budget = 0, 0.0, True, _CROWBAR_LATENCY_BUDGET_NS

        mandelstam_s, sigma_ratio, di_dt, joule_i2t = (
            self._dirac_annihilation_kinematics_and_crowbar(anomaly_norm, crowbar)
        )
        record = f"{context}|{token_valid}|{crowbar}|{energy:.4f}|{photons}".encode("ascii")
        fock = hmac.new(self._secret_key, _HMAC_DOMAIN_FOCK + record, hashlib.sha256).hexdigest()

        return DiracAnnihilationProsecutionReport(
            anomaly_norm=float(anomaly_norm),
            context_digest=context,
            positron_token_valid=token_valid,
            energy_released_kev=energy,
            photons_emitted=photons,
            crowbar_triggered=crowbar,
            crowbar_latency_budget_ns=budget,
            fock_authorization_digest=fock,
            latency_provenance=_LATENCY_PROVENANCE,
            clifford_anticommutator_defect=clifford_def,
            chiral_gamma5_involutory_defect=chiral_def,
            mandelstam_s_kev2=mandelstam_s,
            dirac_cross_section_ratio=sigma_ratio,
            crowbar_peak_di_dt_a_per_us=di_dt,
            crowbar_joule_integral_a2s=joule_i2t,
        )

    # ── Álgebra Boole–Novikov–Hodge (Smith sobre ℤ, FWHT-Parseval, Hodge Δ₀ y Krull) ───────────

    @staticmethod
    def _smith_invariant_factors(matrix: Sequence[Sequence[int]]) -> Tuple[int, ...]:
        """Forma normal de Smith exacta sobre ℤ. Devuelve los factores dᵢ ≠ 0 con d₁ | d₂ | …"""
        A = [[int(x) for x in row] for row in matrix]
        m = len(A)
        n = len(A[0]) if m else 0
        invariants: List[int] = []
        t = 0
        while t < min(m, n):
            best = None
            for i in range(t, m):
                for j in range(t, n):
                    v = A[i][j]
                    if v != 0 and (best is None or abs(v) < abs(A[best[0]][best[1]])):
                        best = (i, j)
            if best is None:
                break
            i0, j0 = best
            A[t], A[i0] = A[i0], A[t]
            for row in A:
                row[t], row[j0] = row[j0], row[t]
            while True:
                clean = True
                for i in range(t + 1, m):
                    if A[i][t] != 0:
                        q = A[i][t] // A[t][t]
                        if q:
                            A[i] = [a - q * b for a, b in zip(A[i], A[t])]
                        if A[i][t] != 0:
                            A[t], A[i] = A[i], A[t]
                            clean = False
                for j in range(t + 1, n):
                    if A[t][j] != 0:
                        q = A[t][j] // A[t][t]
                        if q:
                            for i in range(t, m):
                                A[i][j] -= q * A[i][t]
                        if A[t][j] != 0:
                            for row in A:
                                row[t], row[j] = row[j], row[t]
                            clean = False
                if clean:
                    bad = None
                    for i in range(t + 1, m):
                        for j in range(t + 1, n):
                            if A[i][j] % A[t][t] != 0:
                                bad = i
                                break
                        if bad is not None:
                            break
                    if bad is None:
                        break
                    A[t] = [a + b for a, b in zip(A[t], A[bad])]
            invariants.append(abs(A[t][t]))
            t += 1
        return tuple(invariants)

    @staticmethod
    def _hodge_laplacian_spectral_gap(matrix: NDArray[np.float64]) -> float:
        """Brecha espectral positiva mínima (conectividad de Fiedler) del 0-Laplaciano Δ₀ = B Bᵀ."""
        B = np.asarray(matrix, dtype=np.float64)
        lap = B @ B.T
        evals = np.linalg.eigvalsh(0.5 * (lap + lap.T))
        pos = evals[evals > 1.0e-10]
        return float(np.min(pos)) if pos.size > 0 else 0.0

    @staticmethod
    def _fwht(vec: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transformada rápida de Walsh–Hadamard (FWHT) sin normalizar sobre longitud 2ⁿ."""
        a = np.array(vec, dtype=np.float64, copy=True).reshape(-1)
        size = a.size
        if size & (size - 1):
            raise ValueError("La longitud de la WHT debe ser potencia de 2.")
        h = 1
        while h < size:
            blocks = a.reshape(-1, 2, h)
            x = blocks[:, 0, :].copy()
            y = blocks[:, 1, :].copy()
            blocks[:, 0, :] = x + y
            blocks[:, 1, :] = x - y
            h *= 2
        return a

    @classmethod
    def _rademacher_walsh_energy(cls, values: NDArray[np.float64]) -> Tuple[float, float]:
        """Energía de sensibilidad de Rademacher–Walsh en [0, 1] y residuo relativo de Parseval."""
        f = np.asarray(values, dtype=np.float64).reshape(-1)
        n = (f.size - 1).bit_length() if f.size > 1 else 0
        size = 1 << n
        padded = np.zeros(size, dtype=np.float64)
        padded[: f.size] = f
        spectrum = cls._fwht(padded) / float(size)
        spectral_energy = float(np.sum(spectrum * spectrum))
        mean_square = float(np.mean(padded * padded))
        parseval = abs(spectral_energy - mean_square) / max(mean_square, 1.0e-15)
        if n == 0 or spectral_energy <= 0.0:
            return 0.0, parseval
        degrees = np.array([bin(s).count("1") for s in range(size)], dtype=np.float64)
        energy = float(np.sum(degrees * spectrum * spectrum) / (n * spectral_energy))
        return energy, parseval

    @staticmethod
    def _novikov_valuation(exponents: NDArray[np.float64]) -> Tuple[float, float]:
        """Valuación de Krull v(A) = min λᵢ y norma no arquimediana de Novikov e^{−v(A)}."""
        if exponents.size == 0:
            return math.inf, 0.0
        v = float(np.min(exponents))
        nov = math.exp(-v) if v > -700.0 else math.inf
        return v, nov

    def evaluate_algebraic_topology(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        incidence_matrix: Optional[NDArray[np.float64]] = None,
        quant_scale: float = _DEFAULT_QUANT_SCALE,
    ) -> AlgebraicTopologyProsecutionReport:
        """Evalúa Smith sobre ℤ, Laplaciano de Hodge Δ₀, energía de Walsh y valuación de Krull."""
        mat = self._validate_cost_matrix(apu_cost_matrix)
        exps = self._validate_exponents(exponents)

        if incidence_matrix is None:
            if not (math.isfinite(quant_scale) and quant_scale > 0.0):
                raise ValueError("quant_scale debe ser finito y positivo.")
            scaled = mat * float(quant_scale)
            if float(np.max(np.abs(scaled))) >= _INTEGER_OVERFLOW_GUARD:
                raise ValueError("La MIC escalada excede el rango entero seguro.")
            target = np.rint(scaled).astype(np.int64).tolist()
            hodge_source = mat
            source = "QUANTIZED_COST"
            scale_used: Optional[float] = float(quant_scale)
        else:
            inc = np.asarray(incidence_matrix, dtype=np.float64)
            if inc.ndim != 2 or not np.all(np.isfinite(inc)):
                raise ValueError("La matriz de incidencia debe ser 2D y finita.")
            if not np.all(inc == np.rint(inc)):
                raise ValueError("La matriz de incidencia debe tener coeficientes enteros.")
            target = np.rint(inc).astype(np.int64).tolist()
            hodge_source = inc
            source = "INCIDENCE"
            scale_used = None

        factors = self._smith_invariant_factors(target)
        m = len(target)
        n = len(target[0]) if m else 0
        betti = m - len(factors)
        torsion_order = int(reduce(lambda acc, d: acc * d, (d for d in factors if d > 1), 1))
        euler_char = int(m - n)
        hodge_gap = self._hodge_laplacian_spectral_gap(hodge_source)
        rw_energy, parseval = self._rademacher_walsh_energy(mat.sum(axis=1))
        v_krull, nov_norm = self._novikov_valuation(exps)

        return AlgebraicTopologyProsecutionReport(
            smith_source=source,
            quant_scale=scale_used,
            smith_invariants=factors,
            smith_torsion_free=all(d == 1 for d in factors),
            betti_number=betti,
            betti_number_zero=(betti == 0),
            rademacher_walsh_energy=rw_energy,
            rademacher_walsh_parseval_residual=parseval,
            krull_valuation=v_krull,
            novikov_norm=nov_norm,
            krull_floor_ok=bool(exps.size > 0 and v_krull >= self._policy.krull_floor),
            torsion_subgroup_order=torsion_order,
            euler_characteristic=euler_char,
            hodge_spectral_gap=hodge_gap,
        )

    # ── ÚLTIMO MÉTODO DE LA FASE 1 · ESLABÓN DE CONTINUIDAD HACIA EL INICIO DE LA FASE 2 ────────

    def compute_phase1_witness(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        positron_token: Optional[str] = None,
        force_veto: bool = False,
        incidence_matrix: Optional[NDArray[np.float64]] = None,
        quant_scale: float = _DEFAULT_QUANT_SCALE,
    ) -> Phase1ProsecutionWitness:
        """DEFINICIÓN FORMAL FINAL DE LA FASE 1.

        Sella y emite `Phase1ProsecutionWitness`, que constituye el argumento contractual
        de continuación del primer método de la Fase 2 (`evaluate_toon_buffer_flow`).
        """
        mat = self._validate_cost_matrix(apu_cost_matrix)
        exps = self._validate_exponents(exponents)
        matrix_sha = self._array_digest(mat)
        dirac = self.execute_dirac_and_crowbar_prosecution(
            anomaly_norm=self._anomaly_norm(mat),
            matrix_sha256=matrix_sha,
            positron_token=positron_token,
            force_veto=force_veto,
        )
        algebra = self.evaluate_algebraic_topology(mat, exps, incidence_matrix, quant_scale)
        return Phase1ProsecutionWitness(
            matrix_sha256=matrix_sha,
            exponents_sha256=self._array_digest(exps),
            dirac_report=dirac,
            algebra_report=algebra,
        )


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 2 — CAUDAL EN T*Q, MECÁNICA CELESTE DE POINCARÉ–KAM Y TORRE RSI³ (hereda la Fase 1)
# ════════════════════════════════════════════════════════════════════════════════════════════════════
class _Phase2FluidPoincareKernel(_Phase1DiracAlgebraicKernel):
    """FASE 2: Caudal en T*Q, mecánica celeste simpléctica de Poincaré-KAM y torre RSI³.

    Continuación directa de la Fase 1:
      • PRIMER MÉTODO : `evaluate_toon_buffer_flow(witness1: Phase1ProsecutionWitness, ...)`
      • ÚLTIMO MÉTODO : `compose_phase2_witness(witness1: Phase1ProsecutionWitness, ...)`
    """

    # ── PRIMER MÉTODO DE LA FASE 2 · CONTINUACIÓN DIRECTA DEL ÚLTIMO MÉTODO DE LA FASE 1 ────────

    def evaluate_toon_buffer_flow(
        self,
        witness1: Phase1ProsecutionWitness,
        apu_cost_matrix: NDArray[np.float64],
        buffer_capacity: float = _DEFAULT_BUFFER_CAPACITY,
    ) -> BufferProsecutionReport:
        """PRIMER MÉTODO DE LA FASE 2 (consume `Phase1ProsecutionWitness` de `compute_phase1_witness`).

        Verifica la ligadura criptográfica SHA-256 de la MIC contra la Fase 1 y audita el caudal
        en T*Q: espectro multiplicativo completo de Oseledets, cota de entropía métrica de Pesin
        Σ max(λᵢ, 0) y sobrepresión de golpe de bélier de Zhukovsky-Allievi.
        """
        mat = self._validate_cost_matrix(apu_cost_matrix)
        if not hmac.compare_digest(self._array_digest(mat), witness1.matrix_sha256):
            raise ProsecutorGateError("Φ₁₂: la MIC no coincide con la atestación de la Fase 1.")
        if not (math.isfinite(buffer_capacity) and buffer_capacity > 0.0):
            raise ValueError("buffer_capacity debe ser finito y positivo.")

        row_sums = mat.sum(axis=1)
        flow_var = float(np.var(row_sums))
        evals = np.linalg.eigvals(mat)
        abs_evals = np.sort(np.abs(evals))[::-1]
        rho = float(abs_evals[0]) if abs_evals.size > 0 else 0.0
        lam = math.log(rho) if rho > 0.0 else -math.inf

        lyap_spec = tuple(
            float(math.log(ev)) if ev > _WILKINSON_EPSILON else -math.inf for ev in abs_evals
        )
        pesin_bound = float(sum(max(0.0, l_i) for l_i in lyap_spec if math.isfinite(l_i)))

        # Impedancia acústica modal y sobrepresión de Zhukovsky ΔP = Z₀ √Var(Q)
        diag_mean = float(np.mean(np.abs(np.diag(mat)))) + 1.0
        zhukovsky_surge = float(diag_mean * math.sqrt(max(0.0, flow_var)))
        utilization = float(np.sum(row_sums) / buffer_capacity * 100.0)

        reasons: List[str] = []
        if lam > 0.0:
            reasons.append("OSELEDETS_POSITIVE")
        if flow_var >= self._policy.variance_ceiling:
            reasons.append("VARIANCE_CEILING")
        if utilization > self._policy.utilization_ceiling_pct:
            reasons.append("UTILIZATION_OVER_CAPACITY")

        return BufferProsecutionReport(
            flow_variance=flow_var,
            spectral_radius=rho,
            oseledets_lyapunov_lambda=lam,
            buffer_utilization_pct=utilization,
            overflow_reasons=tuple(reasons),
            lyapunov_spectrum=lyap_spec,
            pesin_entropy_bound=pesin_bound,
            zhukovsky_surge_pressure=zhukovsky_surge,
        )

    # ── Residuo de Cauchy-Riemann-Wirtinger y Capacidad Simpléctica de Gromov ───────────────────

    def evaluate_cauchy_riemann_residual(self, apu_cost_matrix: NDArray[np.float64]) -> float:
        """Residuo de Wirtinger discreto de z = A + iAᵀ: ‖∂̄z‖ / (‖∂̄z‖ + ‖∂z‖)."""
        mat = self._validate_cost_matrix(apu_cost_matrix)
        if mat.shape[0] < 2:
            return 0.0
        z = mat + 1j * mat.T
        d_s = np.gradient(z, axis=0)
        d_t = np.gradient(z, axis=1)
        dbar = 0.5 * (d_s + 1j * d_t)
        d = 0.5 * (d_s - 1j * d_t)
        n_bar = float(np.linalg.norm(dbar))
        n_d = float(np.linalg.norm(d))
        denom = n_bar + n_d
        return n_bar / denom if denom > 1.0e-300 else 0.0

    def evaluate_gromov_capacity(
        self,
        ball_r: float = 0.5,
        cylinder_R: float = 1.0,
    ) -> Tuple[float, float, bool]:
        """No-squeezing de Gromov: c_G(B²ⁿ(r)) = πr² ≤ c_G(Z²ⁿ(R)) = πR² + tolerancia."""
        if not (
            math.isfinite(ball_r)
            and ball_r > 0.0
            and math.isfinite(cylinder_R)
            and cylinder_R > 0.0
        ):
            raise ValueError("Los radios de Gromov deben ser finitos y positivos.")
        c_ball = math.pi * ball_r * ball_r
        c_cyl = math.pi * cylinder_R * cylinder_R
        return c_ball, c_cyl, bool(c_ball <= c_cyl + self._policy.gromov_tol)

    # ── Mecánica Celeste de Poincaré: Variables No Singulares, Torsión KAM, Lie y Cartan ────────

    @staticmethod
    def delaunay_actions(
        a: float = 1.0, e: float = 0.1, inc: float = 0.2, mu: float = 1.0
    ) -> Tuple[float, float, float]:
        """Acciones canónicas de Delaunay (L, G, H) = (√(μa), L√(1−e²), G cos i)."""
        if a <= 0.0 or not (0.0 <= e < 1.0) or mu <= 0.0:
            raise ValueError("Parámetros orbitales de Delaunay fuera de dominio.")
        L = math.sqrt(mu * a)
        G = L * math.sqrt(1.0 - e * e)
        H = G * math.cos(inc)
        return L, G, H

    @classmethod
    def poincare_canonical_elements_and_symplectic_audit(
        cls,
        a: float = 1.0,
        e: float = 0.1,
        inc: float = 0.2,
        mean_anomaly_l: float = 0.3,
        arg_periapsis_g: float = 0.5,
        lon_node_h: float = 0.7,
        mu: float = 1.0,
    ) -> Tuple[Tuple[float, float, float, float, float, float], float]:
        """Variables canónicas no singulares de Poincaré (Λ, ξ₁, ξ₂; λ, η₁, η₂) y auditoría simpléctica.

        Verifica analíticamente que el cambio de coordenadas desde Delaunay (L, G, H; l, g, h)
        preserva la 2-forma simpléctica canónica midiendo ‖Mᵀ J₆ M − J₆‖_F.
        """
        L, G, H = cls.delaunay_actions(a, e, inc, mu)
        I1 = max(1.0e-12, L - G)
        I2 = max(1.0e-12, G - H)
        varpi = arg_periapsis_g + lon_node_h
        omega_node = lon_node_h

        s1 = math.sqrt(2.0 * I1)
        s2 = math.sqrt(2.0 * I2)
        c_vp, s_vp = math.cos(varpi), math.sin(varpi)
        c_om, s_om = math.cos(omega_node), math.sin(omega_node)

        Lambda = L
        lam = mean_anomaly_l + arg_periapsis_g + lon_node_h
        xi1 = s1 * c_vp
        eta1 = -s1 * s_vp
        xi2 = s2 * c_om
        eta2 = -s2 * s_om

        M = np.zeros((6, 6), dtype=np.float64)
        M[0, 0] = 1.0
        M[1, 0] = c_vp / s1
        M[1, 1] = -c_vp / s1
        M[1, 4] = -s1 * s_vp
        M[1, 5] = -s1 * s_vp
        M[2, 1] = c_om / s2
        M[2, 2] = -c_om / s2
        M[2, 5] = -s2 * s_om
        M[3, 3] = 1.0
        M[3, 4] = 1.0
        M[3, 5] = 1.0
        M[4, 0] = -s_vp / s1
        M[4, 1] = s_vp / s1
        M[4, 4] = -s1 * c_vp
        M[4, 5] = -s1 * c_vp
        M[5, 1] = -s_om / s2
        M[5, 2] = s_om / s2
        M[5, 5] = -s2 * c_om

        J6 = np.zeros((6, 6), dtype=np.float64)
        J6[:3, 3:] = np.eye(3)
        J6[3:, :3] = -np.eye(3)
        defect = float(la.norm(M.T @ J6 @ M - J6, ord="fro"))
        return (Lambda, xi1, xi2, lam, eta1, eta2), defect

    @staticmethod
    def evaluate_kolmogorov_arnold_torsion(
        L: float,
        G: float,
        H: float,
        mu: float = 1.0,
        j2_oblateness: float = 0.02,
        secular_coupling: float = 0.05,
    ) -> Tuple[float, float, bool]:
        """Determinante de torsión de Kolmogorov det(∂²H₀/∂I²) e isoenergético de Arnol'd."""
        c_j2 = 0.25 * j2_oblateness * (mu ** 4)
        ratio = (H * H) / (G * G)
        factor = 1.0 - 3.0 * ratio

        omega_L = (mu * mu) / (L ** 3) + 3.0 * c_j2 / ((L ** 4) * (G ** 3)) * factor
        omega_G = (
            3.0 * c_j2 / ((L ** 3) * (G ** 4)) * (1.0 - 5.0 * ratio)
            - secular_coupling * G
        )
        omega_H = 6.0 * c_j2 * H / ((L ** 3) * (G ** 5)) - secular_coupling * H
        omega_vec = np.array([omega_L, omega_G, omega_H], dtype=np.float64)

        T = np.zeros((3, 3), dtype=np.float64)
        T[0, 0] = -3.0 * (mu * mu) / (L ** 4) - 12.0 * c_j2 / ((L ** 5) * (G ** 3)) * factor
        T[0, 1] = T[1, 0] = -9.0 * c_j2 / ((L ** 4) * (G ** 4)) * (1.0 - 5.0 * ratio)
        T[0, 2] = T[2, 0] = -18.0 * c_j2 * H / ((L ** 4) * (G ** 5))
        T[1, 1] = (
            -12.0 * c_j2 / ((L ** 3) * (G ** 5)) * (1.0 - 7.5 * ratio) - secular_coupling
        )
        T[1, 2] = T[2, 1] = -30.0 * c_j2 * H / ((L ** 3) * (G ** 6))
        T[2, 2] = 6.0 * c_j2 / ((L ** 3) * (G ** 5)) - secular_coupling

        kolmogorov_det = float(np.linalg.det(T))
        bordered = np.zeros((4, 4), dtype=np.float64)
        bordered[:3, :3] = T
        bordered[:3, 3] = omega_vec
        bordered[3, :3] = omega_vec
        arnold_det = float(np.linalg.det(bordered))

        non_deg = bool(abs(kolmogorov_det) > 1.0e-9 and abs(arnold_det) > 1.0e-9)
        return kolmogorov_det, arnold_det, non_deg

    @staticmethod
    def evaluate_diophantine_condition(
        omega: float,
        dioph_gamma: float = _DIOPHANTINE_GAMMA,
        tau: float = _DIOPHANTINE_TAU,
        k_max: int = _DIOPHANTINE_K_MAX,
    ) -> bool:
        """Condición diofantina |kν − p| ≥ γ / k^τ para 1 ≤ k ≤ k_max, con ν = ω / (2π)."""
        nu = omega / (2.0 * math.pi)
        for k in range(1, k_max + 1):
            x = k * nu
            if abs(x - round(x)) < dioph_gamma / (k ** tau):
                return False
        return True

    @staticmethod
    def evaluate_deprit_hori_lie_series(
        omega: float,
        perturbation_eps: float = 0.01,
        twist_alpha: float = 0.2,
        n_theta: int = 256,
    ) -> Tuple[float, float, float]:
        """Ecuación homológica ω ∂_θ W = h₁ − ⟨h₁⟩, serie de Lie exp(L_W)H y tiempo de Nekhoroshev."""
        theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
        h1 = perturbation_eps * (np.cos(theta) + 0.5 * np.cos(2.0 * theta))

        hk = np.fft.fft(h1) / n_theta
        k = np.fft.fftfreq(n_theta, d=1.0 / n_theta)
        mask = k != 0.0

        wk = np.zeros_like(hk)
        wk[mask] = hk[mask] / (1j * k[mask] * omega)
        lhs_hat = omega * (1j * k * wk)
        rhs_hat = np.where(mask, hk, 0.0)
        homo_res = float(np.max(np.abs(np.fft.ifft((lhs_hat - rhs_hat) * n_theta))))

        dh1_dtheta = np.real(np.fft.ifft((1j * k * hk) * n_theta))
        dh1_dJ = h1
        dW_dtheta = np.real(np.fft.ifft((1j * k * wk) * n_theta))
        dwk_dJ = wk * (1.0 - twist_alpha / omega)
        dW_dJ = np.real(np.fft.ifft(dwk_dJ * n_theta))

        poisson_h1_W = dh1_dtheta * dW_dJ - dh1_dJ * dW_dtheta
        second_order_norm = float(0.5 * np.max(np.abs(poisson_h1_W)))

        eps_eff = max(1.0e-12, abs(perturbation_eps))
        nekh_exp = min(40.0, (0.1 / eps_eff) ** (1.0 / 6.0))
        nekhoroshev_time = float((2.0 * math.pi / abs(omega)) * math.exp(nekh_exp))
        return homo_res, second_order_norm, nekhoroshev_time

    @staticmethod
    def evaluate_poincare_cartan_and_floquet(
        omega: float,
        perturbation_eps: float = 0.01,
        n_loop: int = 64,
        n_steps: int = 80,
    ) -> Tuple[float, float, bool]:
        """Invariante integral relativo de Poincaré-Cartan ∮_γ p dq (Yoshida 4.º orden) y Floquet."""
        cbrt2 = 2.0 ** (1.0 / 3.0)
        w1 = 1.0 / (2.0 - cbrt2)
        w0 = -cbrt2 / (2.0 - cbrt2)
        c_yosh = (0.5 * w1, 0.5 * (w0 + w1), 0.5 * (w0 + w1), 0.5 * w1)
        d_yosh = (w1, w0, w1, 0.0)

        period = 2.0 * math.pi / max(1.0e-6, abs(omega))
        dt = period / n_steps

        s = np.linspace(0.0, 2.0 * math.pi, n_loop, endpoint=False)
        r0 = 0.15
        q = r0 * np.cos(s)
        p = r0 * omega * np.sin(s)

        def _loop_action(q_l: NDArray[np.float64], p_l: NDArray[np.float64]) -> float:
            dq = np.roll(q_l, -1) - q_l
            p_mid = 0.5 * (p_l + np.roll(p_l, -1))
            return float(np.sum(p_mid * dq))

        i1_initial = _loop_action(q, p)
        q_ref, p_ref = 0.10, 0.0
        Phi = np.eye(2, dtype=np.float64)
        om2 = omega * omega

        for _ in range(n_steps):
            for c_i, d_i in zip(c_yosh, d_yosh):
                q = q + (c_i * dt) * p
                q_ref = q_ref + (c_i * dt) * p_ref
                Phi[0, :] += (c_i * dt) * Phi[1, :]
                if d_i != 0.0:
                    p = p - (d_i * dt) * (om2 * q + perturbation_eps * (q ** 3))
                    v_pp = om2 + 3.0 * perturbation_eps * (q_ref * q_ref)
                    p_ref = p_ref - (d_i * dt) * (om2 * q_ref + perturbation_eps * (q_ref ** 3))
                    Phi[1, :] -= (d_i * dt) * v_pp * Phi[0, :]

        i1_final = _loop_action(q, p)
        invariant_drift = abs(i1_final - i1_initial)
        floquet_trace = float(np.trace(Phi))
        symp_det_err = abs(float(np.linalg.det(Phi)) - 1.0)
        elliptic_stable = bool(abs(floquet_trace) < 2.0 and symp_det_err <= _SYMPLECTIC_TOL)
        return invariant_drift, floquet_trace, elliptic_stable

    # ── Integral Homoclínica de Melnikov (Separatriz de Duffing) ────────────────────────────────

    @staticmethod
    def _trapezoid(y: NDArray[np.float64], x: NDArray[np.float64]) -> float:
        return float(0.5 * np.sum((y[1:] + y[:-1]) * np.diff(x)))

    def evaluate_melnikov_homoclinic(
        self,
        delta_damping: float = 0.1,
        forcing_amp: float = 0.05,
        omega: float = 1.0,
        t_max: float = 40.0,
        n_t: int = 8001,
    ) -> MelnikovHomoclinicReport:
        """Integral de Melnikov sobre q₀ = √2 sech t, p₀ = −√2 sech t tanh t con control analítico."""
        if not all(math.isfinite(v) for v in (delta_damping, forcing_amp, omega, t_max)):
            raise ValueError("Parámetros de Melnikov no finitos.")
        if delta_damping < 0.0 or forcing_amp < 0.0 or omega <= 0.0 or t_max <= 0.0 or n_t < 3:
            raise ValueError("Parámetros de Melnikov fuera de dominio.")

        t = np.linspace(-t_max, t_max, n_t)
        sech_t = 1.0 / np.cosh(t)
        p0 = -math.sqrt(2.0) * sech_t * np.tanh(t)
        i_pp = self._trapezoid(p0 * p0, t)
        i_c = self._trapezoid(p0 * np.cos(omega * t), t)
        i_s = self._trapezoid(p0 * np.sin(omega * t), t)

        i_pp_ref = 4.0 / 3.0
        i_s_ref = -math.sqrt(2.0) * math.pi * omega / math.cosh(math.pi * omega / 2.0)
        residual = max(abs(i_pp - i_pp_ref), abs(i_c), abs(i_s - i_s_ref))

        offset = -delta_damping * i_pp
        amplitude = forcing_amp * math.hypot(i_c, i_s)
        min_m = offset - amplitude
        max_m = offset + amplitude
        return MelnikovHomoclinicReport(
            offset=offset,
            amplitude=amplitude,
            min_m=min_m,
            max_m=max_m,
            chaos_detected=bool(min_m < 0.0 < max_m),
            quadrature_residual=residual,
        )

    # ── Automejora Recursiva de Nivel 3 (RSI³) y Contracción de Banach de la Mónada Kleisli ─────

    @staticmethod
    def _verify_kleisli_monad_coherence(
        stage_factors: Sequence[float],
        theta_anchor: NDArray[np.float64],
    ) -> Tuple[float, float]:
        """Verifica asociatividad y leyes de unidad en la categoría de Kleisli Kl(T)."""
        c = np.asarray(stage_factors, dtype=np.float64).reshape(-1)
        dim = theta_anchor.size
        if c.size != dim:
            c = np.resize(c, dim)
        eye = np.eye(dim, dtype=np.float64)

        def _compose(
            m1: Tuple[NDArray[np.float64], NDArray[np.float64]],
            m2: Tuple[NDArray[np.float64], NDArray[np.float64]],
        ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
            A1, b1 = m1
            A2, b2 = m2
            return A1 @ A2, A1 @ b2 + b1

        f1 = (np.diag(c), (eye - np.diag(c)) @ theta_anchor)
        f2 = (np.diag(c * c), (eye - np.diag(c * c)) @ theta_anchor)
        f3 = (np.diag(0.5 * c), (eye - np.diag(0.5 * c)) @ theta_anchor)
        eta = (eye, np.zeros(dim, dtype=np.float64))

        lhs_A, lhs_b = _compose(_compose(f1, f2), f3)
        rhs_A, rhs_b = _compose(f1, _compose(f2, f3))
        assoc_def = float(la.norm(lhs_A - rhs_A, ord="fro") + la.norm(lhs_b - rhs_b))

        l_A, l_b = _compose(eta, f1)
        r_A, r_b = _compose(f1, eta)
        unit_def = float(
            la.norm(l_A - f1[0], ord="fro")
            + la.norm(l_b - f1[1])
            + la.norm(r_A - f1[0], ord="fro")
            + la.norm(r_b - f1[1])
        )
        return assoc_def, unit_def

    @staticmethod
    def _probe_lipschitz(
        A: NDArray[np.float64], b: NDArray[np.float64], seed: int, n_pairs: int = _RSI_PROBES
    ) -> float:
        """Estimación empírica de Lip(T) por pares aleatorios independientes."""
        rng = np.random.default_rng(seed)
        dim = A.shape[0]
        X = rng.uniform(-1.0, 1.0, size=(n_pairs, dim))
        Y = rng.uniform(-1.0, 1.0, size=(n_pairs, dim))
        num = np.linalg.norm((X @ A.T + b) - (Y @ A.T + b), axis=1)
        den = np.linalg.norm(X - Y, axis=1)
        mask = den > _WILKINSON_EPSILON
        return float(np.max(num[mask] / den[mask])) if np.any(mask) else 0.0

    def _execute_prosecutor_rsi3_tower(
        self,
        stage_factors: Tuple[float, ...],
        authorized_regime: bool = True,
    ) -> RSI3ProsecutionCertificate:
        """Torre de Automejora Recursiva de Nivel 3 (RSI¹ → RSI² → RSI³) sobre la política fiscal.

        El vector de política ancla es θ_anchor = (anomaly_threshold, krull_floor, variance_ceiling).
        Si `authorized_regime=False` (p. ej. Crowbar disparado o violación simpléctica), la
        guardia de Löb ejecuta rollback inmediato al ancla desplegada.
        """
        theta_anchor = np.array(
            [
                self._policy.anomaly_threshold,
                self._policy.krull_floor,
                self._policy.variance_ceiling,
            ],
            dtype=np.float64,
        )
        if not authorized_regime:
            return RSI3ProsecutionCertificate(
                depth_reached=0,
                lip_level1=math.inf,
                lip_level2_meta=math.inf,
                lip_probe_estimate=0.0,
                fixed_point=tuple(float(x) for x in theta_anchor),
                banach_aposteriori_error=math.inf,
                lyapunov_descent_verified=False,
                kleisli_associativity_defect=math.inf,
                kleisli_unit_defect=math.inf,
                loeb_soundness_verified=False,
                accepted=False,
            )

        c = np.asarray(stage_factors, dtype=np.float64).reshape(-1)
        if c.size != 3:
            c = np.resize(c, 3)

        if np.any(c <= 0.0) or np.any(c >= 1.0):
            return RSI3ProsecutionCertificate(
                depth_reached=1,
                lip_level1=float(np.max(c)),
                lip_level2_meta=float(np.max(c) ** 2),
                lip_probe_estimate=float(np.max(c)),
                fixed_point=tuple(float(x) for x in theta_anchor),
                banach_aposteriori_error=math.inf,
                lyapunov_descent_verified=False,
                kleisli_associativity_defect=0.0,
                kleisli_unit_defect=0.0,
                loeb_soundness_verified=False,
                accepted=False,
            )

        eye = np.eye(3, dtype=np.float64)
        A = np.diag(c)
        b = (eye - A) @ theta_anchor
        theta_star = np.linalg.solve(eye - A, b)

        # Nivel 1 (RSI¹): Iteración de Picard con verificación de descenso de Lyapunov V(θ)
        theta = theta_anchor + np.array([0.05, -0.01, 0.08], dtype=np.float64)
        converged = False
        lyap_monotone = True
        prev_v = 0.5 * float(np.dot(theta - theta_star, theta - theta_star))
        last_step = math.inf

        for _ in range(_RSI_MAX_ITER):
            nxt = A @ theta + b
            last_step = float(np.linalg.norm(nxt - theta))
            curr_v = 0.5 * float(np.dot(nxt - theta_star, nxt - theta_star))
            if curr_v > prev_v + _WILKINSON_EPSILON:
                lyap_monotone = False
            prev_v = curr_v
            theta = nxt
            if last_step <= 1.0e-12:
                converged = True
                break

        lip_l1 = float(np.linalg.norm(A, 2))
        aposteriori_err = float((lip_l1 / max(1.0e-12, 1.0 - lip_l1)) * last_step)

        # Nivel 2 (RSI²): Meta-endofuntor de Schulz-Picard M_Ψ(T_Φ)(θ) = A² θ + (I + A) b
        A_meta = A @ A
        b_meta = (eye + A) @ b
        lip_l2 = float(np.linalg.norm(A_meta, 2))
        theta = A_meta @ theta + b_meta
        probe_1 = self._probe_lipschitz(A, b, seed=1729)

        # Nivel 3 (RSI³): Coherencia de la mónada de Kleisli y Guardia de Solidez de Löb-Tarski
        probe_2 = self._probe_lipschitz(A, b, seed=314159)
        lip_probe = max(probe_1, probe_2)
        assoc_def, unit_def = self._verify_kleisli_monad_coherence(c, theta_anchor)
        theta_star_meta = np.linalg.solve(eye - A_meta, b_meta)

        loeb_ok = bool(
            lip_l2 <= (lip_l1 * lip_l1) + 1.0e-12
            and lip_l1 < 1.0
            and np.allclose(theta_star_meta, theta_star, atol=1.0e-12)
            and np.allclose(theta, theta_anchor, atol=1.0e-9)
            and assoc_def <= _KLEISLI_COHERENCE_TOL
            and unit_def <= _KLEISLI_COHERENCE_TOL
            and lyap_monotone
        )
        accepted = bool(
            converged and loeb_ok and (lip_probe <= lip_l1 + 1.0e-9)
        )
        return RSI3ProsecutionCertificate(
            depth_reached=_RSI_MAX_DEPTH,
            lip_level1=lip_l1,
            lip_level2_meta=lip_l2,
            lip_probe_estimate=lip_probe,
            fixed_point=tuple(float(x) for x in theta),
            banach_aposteriori_error=aposteriori_err,
            lyapunov_descent_verified=lyap_monotone,
            kleisli_associativity_defect=assoc_def,
            kleisli_unit_defect=unit_def,
            loeb_soundness_verified=loeb_ok,
            accepted=accepted,
        )

    def evaluate_banach_contraction(
        self,
        stage_factors: Tuple[float, ...] = _DEFAULT_STAGE_FACTORS,
        authorized_regime: bool = True,
    ) -> KleisliContractionReport:
        """Lip(T³) = ∏cᵢ + margen declarado, κ = −ln Lip, κ³ y certificación RSI³."""
        factors = tuple(float(c) for c in stage_factors)
        if not factors or not all(math.isfinite(c) and 0.0 <= c <= 1.0 for c in factors):
            raise ValueError("Los factores de etapa deben ser finitos y estar en [0, 1].")
        lip = math.prod(factors) + _KLEISLI_SAFETY_MARGIN
        kappa = -math.log(lip)
        rsi_cert = self._execute_prosecutor_rsi3_tower(
            factors, authorized_regime=authorized_regime
        )
        return KleisliContractionReport(
            stage_factors=factors,
            lip=lip,
            kappa=kappa,
            contraction_margin_cubed=kappa ** 3,
            rsi_certificate=rsi_cert,
        )

    # ── ÚLTIMO MÉTODO DE LA FASE 2 · ESLABÓN DE CONTINUIDAD HACIA EL INICIO DE LA FASE 3 ────────

    def compose_phase2_witness(
        self,
        witness1: Phase1ProsecutionWitness,
        apu_cost_matrix: NDArray[np.float64],
        buffer_capacity: float = _DEFAULT_BUFFER_CAPACITY,
        stage_factors: Tuple[float, ...] = _DEFAULT_STAGE_FACTORS,
        ball_r: float = 0.5,
        cylinder_R: float = 1.0,
        delta_damping: float = 0.1,
        forcing_amp: float = 0.05,
        omega: float = 1.0,
    ) -> Phase2ProsecutionWitness:
        """DEFINICIÓN FORMAL FINAL DE LA FASE 2.

        Integra caudal, mecánica celeste de Poincaré-KAM y torre RSI³, emitiendo el testigo
        sellado `Phase2ProsecutionWitness` para `_phase3_clause_table` (Fase 3).
        """
        flow = self.evaluate_toon_buffer_flow(witness1, apu_cost_matrix, buffer_capacity)
        mat = self._validate_cost_matrix(apu_cost_matrix)
        c_ball, c_cyl, rigid = self.evaluate_gromov_capacity(ball_r, cylinder_R)
        melnikov = self.evaluate_melnikov_homoclinic(delta_damping, forcing_amp, omega)

        L, G, H = self.delaunay_actions(a=1.0, e=0.1, inc=0.2, mu=omega * omega)
        poincare_coords, symp_defect = self.poincare_canonical_elements_and_symplectic_audit(
            a=1.0, e=0.1, inc=0.2, mu=omega * omega
        )
        kol_det, arn_det, kam_non_deg = self.evaluate_kolmogorov_arnold_torsion(
            L, G, H, mu=omega * omega
        )
        dioph_ok = self.evaluate_diophantine_condition(omega)
        homo_res, lie_norm2, nekh_time = self.evaluate_deprit_hori_lie_series(omega)
        cartan_drift, floquet_tr, floquet_ok = self.evaluate_poincare_cartan_and_floquet(omega)

        kam = KAMDynamicsProsecutionReport(
            cauchy_riemann_residual=self.evaluate_cauchy_riemann_residual(mat),
            gromov_ball_capacity=c_ball,
            gromov_cylinder_capacity=c_cyl,
            gromov_rigid=rigid,
            melnikov=melnikov,
            delaunay_actions=(L, G, H),
            poincare_canonical_coords=poincare_coords,
            symplectic_jacobian_defect=symp_defect,
            kolmogorov_torsion_det=kol_det,
            arnold_isoenergetic_det=arn_det,
            is_kam_non_degenerate=kam_non_deg,
            diophantine_ok=dioph_ok,
            homological_residual=homo_res,
            lie_series_second_order_norm=lie_norm2,
            nekhoroshev_stability_time=nekh_time,
            poincare_cartan_invariant_drift=cartan_drift,
            floquet_monodromy_trace=floquet_tr,
            is_floquet_elliptic_stable=floquet_ok,
        )

        # La torre RSI³ se ejecuta siempre para auditar la política, pero queda condicionada a KAM/Gromov
        rsi_authorized = bool(
            rigid
            and symp_defect <= _SYMPLECTIC_TOL
            and cartan_drift <= _CARTAN_INVARIANT_TOL
            and kam_non_deg
        )
        banach = self.evaluate_banach_contraction(
            stage_factors, authorized_regime=rsi_authorized
        )
        identities_ok = (
            witness1.algebra_report.rademacher_walsh_parseval_residual
            <= self._policy.identity_tol
            and melnikov.quadrature_residual <= self._policy.identity_tol
            and symp_defect <= _SYMPLECTIC_TOL
            and homo_res <= _HOMOLOGICAL_TOL
        )
        return Phase2ProsecutionWitness(
            phase1=witness1,
            flow=flow,
            kam=kam,
            banach=banach,
            internal_identities_ok=bool(identities_ok),
        )


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 3 — ADJUDICACIÓN EN EL TOPOS DE HEYTING Ω₃ Y PASAPORTE FISCAL (hereda la Fase 2)
# ════════════════════════════════════════════════════════════════════════════════════════════════════
class _Phase3HeytingVerdictKernel(_Phase2FluidPoincareKernel):
    """FASE 3: Tabla de cláusulas fiscales, veredicto en Ω₃ (Lawvere-Tierney) y pasaporte sellado.

    Continuación directa de la Fase 2:
      • PRIMER MÉTODO : `_phase3_clause_table(witness2: Phase2ProsecutionWitness)`
      • ÚLTIMO MÉTODO : `execute_toon_buffer_prosecutor_audit(...)`
    """

    # ── PRIMER MÉTODO DE LA FASE 3 · CONTINUACIÓN DIRECTA DEL ÚLTIMO MÉTODO DE LA FASE 2 ────────

    def _phase3_clause_table(self, witness2: Phase2ProsecutionWitness) -> HeytingClauseTable:
        """PRIMER MÉTODO DE LA FASE 3 (consume `Phase2ProsecutionWitness` de `compose_phase2_witness`).

        Evalúa las cláusulas de veto (→ VETOED) y degradación (→ DEGRADED) integrando Clifford-Dirac,
        topología de Smith-Novikov, mecánica celeste de Poincaré-KAM y solidez RSI³ de Löb-Kleisli.
        """
        dirac = witness2.phase1.dirac_report
        algebra = witness2.phase1.algebra_report
        flow = witness2.flow
        kam = witness2.kam
        rsi = witness2.banach.rsi_certificate

        veto: List[str] = []
        degrade: List[str] = []

        if dirac.crowbar_triggered:
            veto.append("DIRAC_CROWBAR")
        if not kam.gromov_rigid:
            veto.append("GROMOV_VIOLATED")
        if not algebra.smith_torsion_free:
            veto.append("SMITH_TORSION")
        if kam.symplectic_jacobian_defect > _SYMPLECTIC_TOL:
            veto.append("POINCARE_SYMPLECTIC_DEFECT")

        if flow.water_hammer_overflow:
            (veto if self._policy.water_hammer_veto else degrade).append("WATER_HAMMER")
        if not algebra.krull_floor_ok:
            degrade.append("KRULL_FLOOR")
        if kam.melnikov.chaos_detected:
            degrade.append("MELNIKOV_CHAOS")
        if not kam.diophantine_ok:
            degrade.append("DIOPHANTINE_RESONANCE")
        if not kam.is_kam_non_degenerate:
            degrade.append("KAM_TORSION_DEGENERATE")
        if kam.poincare_cartan_invariant_drift > _CARTAN_INVARIANT_TOL:
            degrade.append("POINCARE_CARTAN_DRIFT")
        if not (witness2.banach.lip < 1.0):
            degrade.append("BANACH_CONTRACTION")
        elif rsi is not None and not rsi.accepted:
            degrade.append("RSI3_LOEB_KLEISLI")
        if not witness2.internal_identities_ok:
            degrade.append("INTERNAL_IDENTITY")

        return HeytingClauseTable(veto_failures=tuple(veto), degrade_failures=tuple(degrade))

    def _phase3_verdict(self, clauses: HeytingClauseTable) -> HeytingTruthValue:
        """Veredicto = ínfimo intuicionista (∧) en Ω₃ con clausura de Lawvere-Tierney."""
        verdict = HeytingTruthValue.COHERENT
        if clauses.veto_failures:
            verdict = verdict.meet(HeytingTruthValue.VETOED)
        if clauses.degrade_failures:
            verdict = verdict.meet(HeytingTruthValue.DEGRADED)
        return verdict.lawvere_tierney_sheafify(stabilize_degraded=False)

    def _phase3_passport(
        self,
        truth: HeytingTruthValue,
        witness2: Phase2ProsecutionWitness,
        clauses: HeytingClauseTable,
    ) -> str:
        """Pasaporte HMAC-SHA256 determinista con separación de dominio (sin reloj de pared)."""
        p1 = witness2.phase1
        parts = (
            "TOON_BUFFER_PROSECUTOR",
            "v14",
            truth.name,
            p1.matrix_sha256,
            p1.exponents_sha256,
            p1.dirac_report.fock_authorization_digest,
            f"{witness2.flow.oseledets_lyapunov_lambda:.6f}",
            f"{p1.algebra_report.krull_valuation:.6f}",
            f"{witness2.kam.melnikov.min_m:.6f}",
            f"{witness2.kam.melnikov.max_m:.6f}",
            f"{witness2.banach.lip:.6f}",
            "1" if p1.algebra_report.smith_torsion_free else "0",
            ";".join(clauses.veto_failures),
            ";".join(clauses.degrade_failures),
        )
        canonical = "|".join(parts).encode("utf-8")
        return hmac.new(
            self._secret_key, _HMAC_DOMAIN_PASSPORT + canonical, hashlib.sha256
        ).hexdigest()

    # ── ÚLTIMO MÉTODO DE LA FASE 3 · ORQUESTADOR DE LA AUDITORÍA FISCAL INQUISIDORA ─────────────

    def execute_toon_buffer_prosecutor_audit(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        positron_token: Optional[str] = None,
        force_veto: bool = False,
        incidence_matrix: Optional[NDArray[np.float64]] = None,
        quant_scale: float = _DEFAULT_QUANT_SCALE,
        buffer_capacity: float = _DEFAULT_BUFFER_CAPACITY,
        stage_factors: Tuple[float, ...] = _DEFAULT_STAGE_FACTORS,
        ball_r: float = 0.5,
        cylinder_R: float = 1.0,
        delta_damping: float = 0.1,
        forcing_amp: float = 0.05,
        omega: float = 1.0,
    ) -> ToonBufferProsecutorSpectrumReport:
        """DEFINICIÓN FORMAL FINAL DE LA FASE 3.

        Orquesta la cadena anidada Fase 1 → Fase 2 → Fase 3 y emite el reporte espectral.
        """
        witness1 = self.compute_phase1_witness(
            apu_cost_matrix,
            exponents,
            positron_token,
            force_veto,
            incidence_matrix,
            quant_scale,
        )
        witness2 = self.compose_phase2_witness(
            witness1,
            apu_cost_matrix,
            buffer_capacity=buffer_capacity,
            stage_factors=stage_factors,
            ball_r=ball_r,
            cylinder_R=cylinder_R,
            delta_damping=delta_damping,
            forcing_amp=forcing_amp,
            omega=omega,
        )
        clauses = self._phase3_clause_table(witness2)
        truth = self._phase3_verdict(clauses)
        passport = self._phase3_passport(truth, witness2, clauses)
        return ToonBufferProsecutorSpectrumReport(
            buffer_report=witness2.flow,
            dirac_report=witness1.dirac_report,
            algebra_report=witness1.algebra_report,
            kam_report=witness2.kam,
            banach_report=witness2.banach,
            clauses=clauses,
            truth_value=truth,
            passport_hash=passport,
        )


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# MOTOR PÚBLICO SOBERANO
# ════════════════════════════════════════════════════════════════════════════════════════════════════
class BoolNovikovToonBufferProsecutorEngine(_Phase3HeytingVerdictKernel):
    """Motor Espectral Fiscal Inquisidor Dual del TOON Buffer (v15.0.0-PhD).

    Hereda la cadena de 3 fases anidadas:
      _Phase1DiracAlgebraicKernel ⊂ _Phase2FluidPoincareKernel ⊂ _Phase3HeytingVerdictKernel.
    """

    def __init__(
        self,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
        policy: Optional[ProsecutorPolicy] = None,
    ) -> None:
        super().__init__(
            secret_key=secret_key,
            policy=policy if policy is not None else ProsecutorPolicy(),
        )
        logger.info("%s v%s inicializado.", self.__class__.__name__, _ENGINE_VERSION)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# AUTOPRUEBA DE VERIFICACIÓN FORMAL
# ════════════════════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Caso 1: Matriz contractiva autorizada con incidencia unimodular → COHERENT
    matrix_coherent = 0.85 * np.eye(4, dtype=np.float64) + 0.02 * np.ones((4, 4), dtype=np.float64)
    exps_test = np.array([0.15, 0.22, 0.35, 0.48], dtype=np.float64)
    path_incidence = np.array(
        [[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]], dtype=np.float64
    )

    engine = BoolNovikovToonBufferProsecutorEngine()
    token_coh = engine.issue_human_positron_authorization(matrix_coherent)
    report = engine.execute_toon_buffer_prosecutor_audit(
        matrix_coherent, exps_test, positron_token=token_coh, incidence_matrix=path_incidence
    )
    replay = BoolNovikovToonBufferProsecutorEngine().execute_toon_buffer_prosecutor_audit(
        matrix_coherent, exps_test, positron_token=token_coh, incidence_matrix=path_incidence
    )
    diagnostic = engine.execute_toon_buffer_prosecutor_audit(
        matrix_coherent, exps_test, positron_token=token_coh
    )

    dirac = report.dirac_report
    alg = report.algebra_report
    kam = report.kam_report
    mel = kam.melnikov
    rsi = report.rsi_certificate

    print("=== TEST AUTÓNOMO: BoolNovikovToonBufferProsecutorEngine v15.0.0-PhD ===")
    print(f"Veredicto Heyting Ω₃          : {report.truth_value.name} ({hex(int(report.truth_value))})")
    print(f"Cláusulas de veto             : {report.clauses.veto_failures or '—'}")
    print(f"Cláusulas de degradación      : {report.clauses.degrade_failures or '—'}")
    print(
        f"Clifford Cl_{{1,3}} defecto / γ⁵ : {dirac.clifford_anticommutator_defect:.2e} / "
        f"{dirac.chiral_gamma5_involutory_defect:.2e}"
    )
    print(
        f"Token positrón / Crowbar      : válido={dirac.positron_token_valid}, "
        f"crowbar={dirac.crowbar_triggered}, E={dirac.energy_released_kev:.4f} keV "
        f"({dirac.photons_emitted}γ, σ/σ_T={dirac.dirac_cross_section_ratio:.4f})"
    )
    print(
        f"Oseledets λ_max / Zhukovsky   : {report.buffer_report.oseledets_lyapunov_lambda:.4f} / "
        f"ΔP={report.buffer_report.zhukovsky_surge_pressure:.4f}"
    )
    print(f"Krull v(A) / Novikov          : {alg.krull_valuation:.4f} / {alg.novikov_norm:.6f}")
    print(
        f"Smith (fuente, invariantes)   : {alg.smith_source}, {alg.smith_invariants} "
        f"(libre torsión: {alg.smith_torsion_free}, gap Δ₀: {alg.hodge_spectral_gap:.4f})"
    )
    print(
        f"Walsh energía / Parseval      : {alg.rademacher_walsh_energy:.6f} / "
        f"{alg.rademacher_walsh_parseval_residual:.2e}"
    )
    print(
        f"Poincaré Simpléctico / Cartan : defecto={kam.symplectic_jacobian_defect:.2e}, "
        f"deriva ∮p dq={kam.poincare_cartan_invariant_drift:.2e}"
    )
    print(
        f"Torsión Kolmogorov / Arnol'd  : det(T)={kam.kolmogorov_torsion_det:.4f}, "
        f"det_Arn={kam.arnold_isoenergetic_det:.4f} (no-deg: {kam.is_kam_non_degenerate})"
    )
    print(
        f"Serie Lie 2.º orden / Floquet : ‖½{{h₁,W}}‖={kam.lie_series_second_order_norm:.2e}, "
        f"Tr(M)={kam.floquet_monodromy_trace:.6f} (elíptica: {kam.is_floquet_elliptic_stable})"
    )
    print(
        f"Gromov c(B)/c(Z) / Melnikov   : {kam.gromov_ball_capacity:.4f}/{kam.gromov_cylinder_capacity:.4f} "
        f"(rígido={kam.gromov_rigid}) | caos={mel.chaos_detected}"
    )
    if rsi is not None:
        print(
            f"RSI³ Fiscal (Löb/Kleisli)     : depth={rsi.depth_reached}, Lip_L1={rsi.lip_level1:.6f}, "
            f"Lip_L2={rsi.lip_level2_meta:.6f}, Löb={rsi.loeb_soundness_verified}, "
            f"aceptado={rsi.accepted}"
        )
    print(
        f"Contracción Lip(T³) / κ³      : {report.banach_report.lip:.6f} / "
        f"{report.banach_report.contraction_margin_cubed:.4e}"
    )
    print(f"Pasaporte HMAC SHA-256        : {report.passport_hash[:16]}...")
    print(f"Reproducible (sin reloj)      : {report.passport_hash == replay.passport_hash}")
    print("--- Diagnóstico: MIC cuantizada como fuente de Smith ---")
    print(
        f"Veredicto (MIC cuantizada)    : {diagnostic.truth_value.name} "
        f"(vetos={diagnostic.clauses.veto_failures})"
    )
    print("=== TEST COMPLETADO CON ÉXITO ===")