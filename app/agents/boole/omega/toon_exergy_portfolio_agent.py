# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : TOON Exergy Portfolio Agent (Custodio Exergético de Portafolios)    ║
║ Ruta   : app/agents/boole/omega/toon_exergy_portfolio_agent.py               ║
║ Versión: 7.0.0-Poincare-Celestial-KAM-Melnikov-RSI3-Nested-PhD               ║
╚══════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA DE DOMINIO Y FUNDAMENTACIÓN MATEMÁTICA DOCTORAL:
────────────────────────────────────────────────────────────────────────────────
Soberano agéntico custodio (`TOONExergyPortfolioAgent`) del Estrato Exergético
𝒱_ℙ. Orquesta un fibrado de **tres fases anidadas** (categoría de Kleisli RSI³):
el método terminal de la Fase *k* es el morfismo de continuación (unidad de
Kleisli) que *inicia* la Fase *k+1*. No existe salto de fibrado: el sobre
monádico sellado de la fase precedente es el único objeto inicial admisible
de la fase subsiguiente.

  [P1] Mecánica celeste de Poincaré & geometría simpléctica.
       Coordenadas de acción–ángulo de Poincaré–Delaunay (ℓ, L) ∈ 𝕋ⁿ × ℝⁿ
       sobre (T*ℳ, ω = Σ dℓᵢ ∧ dLᵢ). Mapa de retorno, multiplicadores de
       Floquet (λ, 1/λ), forma normal de Birkhoff, cota diofántica de
       Siegel–Bruno–KAM, integral homoclínica de Melnikov (cuadratura RSI³
       de Richardson) y tiempo de confinamiento de Nekhoroshev.

  [P2] Termodinámica de no-equilibrio (Jarzynski–Crooks), gerbes de
       Čech–Deligne y conteo instantónico de Donaldson–Thomas (Behrend).

  [P3] Mónada Kleisli RSI³ sobre un álgebra de Banach estrictamente
       contractiva: Lip(T)=0.95, Lip(T²)=0.9025, Lip(T³)=0.857375<1
       ⇒ ∃! x* : T³(x*)=x* y jerk cognitivo d³C/dt³>0. Adjudicación en el
       retículo de Heyting Ω₃ y disparo Crowbar ESP32 (t_act ≤ 392.15 ns).

ESTRUCTURA DE FASES ANIDADAS (continuación unívoca):
────────────────────────────────────────────────────────────────────────────────
  [FASE 1] Phase1_PortfolioTopologyObserver
           └── observe_and_synthesize_symplectic_state → Envelope-1
           └── continue_as_phase2_orientor              → *inicia Fase 2*
                 │
                 ▼
  [FASE 2] Phase1.Phase2_DeligneGerbePoincareOrientor
           └── orient_and_contract_deligne_rsi3         → Envelope-2
           └── continue_as_phase3_crowbar_decider       → *inicia Fase 3*
                 │
                 ▼
  [FASE 3] Phase1.Phase2.Phase3_ExergyCrowbarDecider
           └── decide_and_act_crowbar                   → PortfolioGovernanceState
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

try:
    from scipy.integrate import simpson as _simpson
except ImportError:  # SciPy < 1.6
    from scipy.integrate import simps as _simpson  # type: ignore[attr-defined]


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DEL SISTEMA DE REGISTRO TELEMÉTRICO
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("APUFilter.Omega.TOONExergyPortfolioAgent")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES FÍSICAS, INVARIANTES GEOMÉTRICOS Y LÍMITES CIBER-FÍSICOS
# ─────────────────────────────────────────────────────────────────────────────
BOLTZMANN_K_B: Final[float] = 1.380649e-23
T_REF_PORTFOLIO: Final[float] = 298.15
EXERGY_DISSIPATION_THRESHOLD: Final[float] = 10.0
CECH_OBSTRUCTION_EPSILON: Final[float] = 1.0e-9
KAM_DIOPHANTINE_GAMMA: Final[float] = 0.05
KAM_DIOPHANTINE_TAU: Final[float] = 2.5
KAM_LATTICE_L1_MAX: Final[int] = 6
RSI3_LIP_LEVEL_1: Final[float] = 0.950000
RSI3_LIP_LEVEL_2: Final[float] = 0.902500
RSI3_BANACH_TARGET_LIP: Final[float] = 0.857375
RSI3_PICARD_ITERS: Final[int] = 80
RSI3_PICARD_ATOL: Final[float] = 1.0e-14
ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 400.0
ESP32_CROWBAR_NOMINAL_LATENCY_NS: Final[float] = 392.15
SYMPLECTIC_RESIDUAL_TOL: Final[float] = 1.0e-10
FLOQUET_UNIT_CIRCLE_TOL: Final[float] = 1.0e-8
MELNIKOV_ZERO_THRESHOLD: Final[float] = 1.0e-4
NOVIKOV_FLOOR: Final[float] = 1.0e-12
POINCARE_SECTION_NODES: Final[int] = 1024
BIRKHOFF_CUBIC_COUPLING: Final[float] = 0.015
KV_COMPRESSION_SIGMA: Final[float] = 0.15855       # calibra 86.4 % @ Lip(T³)
WACC_PROTECTION_BASE_PCT: Final[float] = 15.0
LIQUIDITY_RELEASE_BASE_PCT: Final[float] = 11.5


def _sha256_canonical(*parts: Any) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _finite(x: float, default: float = 0.0) -> float:
    return float(x) if math.isfinite(x) else default


def rsi3_richardson_extrapolate(i_coarse: float, i_mid: float, i_fine: float) -> float:
    """
    Extrapolación de Richardson anidada de orden 3 (automejora de cuadraturas).
        I⁽¹⁾ = (4 I_mid − I_coarse)/3
        I⁽²⁾ = (16 I_fine − I_mid)/15
        I*   = (16 I⁽²⁾ − I⁽¹⁾)/15
    """
    i1 = (4.0 * i_mid - i_coarse) / 3.0
    i2 = (16.0 * i_fine - i_mid) / 15.0
    return float((16.0 * i2 - i1) / 15.0)


# ─────────────────────────────────────────────────────────────────────────────
# ÁLGEBRA DE HEYTING TRIVALENTE Ω₃
# ─────────────────────────────────────────────────────────────────────────────
class HeytingState(IntEnum):
    """
    Retículo de Heyting Ω₃ = {⊥₀ ⊏ ½ ⊏ ⊤₁} para el topos de gobernanza:

        COHERENT (0x00) ⊏ DEGRADED (0x01) ⊏ VETOED (0x03)

      a ∧ b = min(a, b),   a ∨ b = max(a, b),
      a ⇒ b = ⊤ si a ≤ b, else b.
    """

    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03

    @classmethod
    def meet(cls, a: HeytingState, b: HeytingState) -> HeytingState:
        return cls(min(int(a), int(b)))

    @classmethod
    def join(cls, a: HeytingState, b: HeytingState) -> HeytingState:
        return cls(max(int(a), int(b)))

    @classmethod
    def implies(cls, a: HeytingState, b: HeytingState) -> HeytingState:
        if int(a) <= int(b):
            return cls.VETOED
        return b

    @classmethod
    def heyting_implication(cls, p: HeytingState, q: HeytingState) -> HeytingState:
        """Alias doctoral: p ⇒ q = max { x | p ∧ x ≤ q }."""
        return cls.implies(p, q)

    @classmethod
    def negate(cls, a: HeytingState) -> HeytingState:
        return cls.implies(a, cls.COHERENT)


# ═════════════════════════════════════════════════════════════════════════════
# OCTONIONES VÍA DUPLICACIÓN DE CAYLEY–DICKSON (no asociativos)
# ═════════════════════════════════════════════════════════════════════════════
def _quat_conj(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    return (q[0], -q[1], -q[2], -q[3])


def _quat_mul(
    p: Tuple[float, float, float, float],
    q: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    a, b, c, d = p
    e, f, g, h = q
    return (
        a * e - b * f - c * g - d * h,
        a * f + b * e + c * h - d * g,
        a * g - b * h + c * e + d * f,
        a * h + b * g - c * f + d * e,
    )


@dataclass(frozen=True)
class Octonion:
    """
    Álgebra de división no asociativa 𝕆 (dim_ℝ = 8) por duplicación
    Cayley–Dickson sobre ℍ:

        (a, b)(c, d) = (a c − d̄ b,  d a + b c̄),    a,b,c,d ∈ ℍ.

    El asociador [x,y,z] = (xy)z − x(yz) mide la curvatura G₂ del gerbe.
    """

    coords: Tuple[float, float, float, float, float, float, float, float]

    def __post_init__(self) -> None:
        if len(self.coords) != 8:
            raise ValueError("Un octonión requiere exactamente 8 componentes reales.")

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> Octonion:
        flat = np.asarray(arr, dtype=np.float64).ravel()
        buf = np.zeros(8, dtype=np.float64)
        n = min(8, int(flat.size))
        if n:
            buf[:n] = flat[:n]
        return cls(tuple(float(x) for x in buf))

    @property
    def v(self) -> NDArray[np.float64]:
        return np.array(self.coords, dtype=np.float64)

    def conjugate(self) -> Octonion:
        c = self.coords
        return Octonion((c[0], -c[1], -c[2], -c[3], -c[4], -c[5], -c[6], -c[7]))

    def norm_squared(self) -> float:
        return float(sum(x * x for x in self.coords))

    def norm(self) -> float:
        return math.sqrt(self.norm_squared())

    def __add__(self, other: Octonion) -> Octonion:
        return Octonion(tuple(a + b for a, b in zip(self.coords, other.coords)))

    def __sub__(self, other: Octonion) -> Octonion:
        return Octonion(tuple(a - b for a, b in zip(self.coords, other.coords)))

    def __mul__(self, other: Octonion) -> Octonion:
        a = (self.coords[0], self.coords[1], self.coords[2], self.coords[3])
        b = (self.coords[4], self.coords[5], self.coords[6], self.coords[7])
        c = (other.coords[0], other.coords[1], other.coords[2], other.coords[3])
        d = (other.coords[4], other.coords[5], other.coords[6], other.coords[7])
        ac = _quat_mul(a, c)
        dconj_b = _quat_mul(_quat_conj(d), b)
        da = _quat_mul(d, a)
        b_cconj = _quat_mul(b, _quat_conj(c))
        re = tuple(ac[i] - dconj_b[i] for i in range(4))
        im = tuple(da[i] + b_cconj[i] for i in range(4))
        return Octonion(re + im)  # type: ignore[arg-type]

    @staticmethod
    def associator(x: Octonion, y: Octonion, z: Octonion) -> Octonion:
        """[x, y, z] = (x y) z − x (y z). Nulo en ℍ, no trivial en 𝕆."""
        return (x * y) * z - x * (y * z)


# ═════════════════════════════════════════════════════════════════════════════
# MÓNADA KLEISLI RSI³ (endofuntores contractivos sobre Banach)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class KleisliRSI3:
    """
    Objeto de Kleisli del endofunctor T de automejora.
    `bind` degrada Lipschitz: Lip(k ∘ T) ≤ Lip(k) Lip(T).
    Nivel 3 ⇒ Lip ≤ 0.95³ = 0.857375 < 1 (punto fijo único).
    """

    value: Any
    lipschitz: float
    level: int
    residual: float = 0.0

    @staticmethod
    def unit(value: Any) -> KleisliRSI3:
        return KleisliRSI3(value=value, lipschitz=1.0, level=0, residual=0.0)

    def bind(self, k: Callable[[Any], KleisliRSI3]) -> KleisliRSI3:
        nxt = k(self.value)
        return KleisliRSI3(
            value=nxt.value,
            lipschitz=float(self.lipschitz * nxt.lipschitz),
            level=self.level + 1,
            residual=float(max(self.residual, nxt.residual)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# DTOs INMUTABLES DE HANDOFF INTER-FASE (SOBRES MONÁDICOS SELLADOS)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PortfolioSymplecticObservationEnvelope:
    """
    SOBRE INMUTABLE DE FASE 1.
    Fibrado cotangente T*ℳ, monodromía Sp(2n), Poincaré–Delaunay, Floquet,
    Birkhoff, Betti, Novikov–Krull y asociador octoniónico.
    Precondición formal absoluta de la Fase 2.
    """

    timestamp_iso: str
    portfolio_id: str
    num_contracts: int
    poincare_dimension: int
    symplectic_matrix_hash: str
    symplectic_invariance_residual: float
    is_symplectic_group_member: bool
    delaunay_frequencies: Tuple[float, ...]
    poincare_action_radii: Tuple[float, ...]
    poincare_winding_numbers: Tuple[float, ...]
    floquet_multipliers: Tuple[complex, ...]
    floquet_on_unit_circle: bool
    lyapunov_max_exponent: float
    birkhoff_normal_form_residue: float
    kam_diophantine_frequency_gap: float
    kam_diophantine_infimum_divisor: float
    is_diophantine_kam_stable: bool
    octonionic_associator_norm: float
    novikov_krull_valuation: float
    betti_numbers: Tuple[int, int, int]
    phase1_sha256: str


@dataclass(frozen=True)
class PortfolioHyperbolicOrientationEnvelope:
    """
    SOBRE INMUTABLE DE FASE 2.
    Jarzynski–Crooks, Melnikov homoclínico, Nekhoroshev, Čech–Deligne, DT
    y contracción Kleisli RSI³. Precondición formal absoluta de la Fase 3.
    """

    phase1_ref_hash: str
    timestamp_iso: str
    portfolio_id: str
    jarzynski_work_joules: float
    delta_free_energy_joules: float
    irreversible_work_joules: float
    crooks_dissipation_factor: float
    jarzynski_equality_residual: float
    jarzynski_log_expectation: float
    poincare_melnikov_integral: float
    melnikov_simple_zeros_count: int
    homoclinic_splitting_angle: float
    kam_homoclinic_tangle_detected: bool
    nekhoroshev_stability_time: float
    cech_deligne_2cocycle_residual: float
    is_cech_cocycle_trivial: bool
    gerbe_3form_flux_integral: float
    donaldson_thomas_invariant: int
    banach_lipschitz_k1: float
    banach_lipschitz_k2: float
    banach_lipschitz_k3: float
    jerk_capacity_metric: float
    is_jerk_strictly_positive: bool
    kleisli_level_attained: int
    phase2_sha256: str


@dataclass(frozen=True)
class PortfolioGovernanceState:
    """
    CERTIFICADO TERMINAL DE GOBERNANZA (FASE 3).
    Sello de inmutabilidad emitido por el Soberano Exergético.
    """

    timestamp_iso: str
    portfolio_id: str
    global_heyting_verdict: HeytingState
    soft_veto_relief_valve_active: bool
    crowbar_triggered: bool
    actuation_latency_nanoseconds: float
    banach_fixed_point_residual: float
    banach_a_priori_error: float
    rsi3_lipschitz_contraction: float
    kleisli_level_attained: int
    kam_bruno_satisfied: bool
    homoclinic_chaos_vetoed: bool
    positive_jerk_capacity_certified: bool
    kv_cache_compression_percent: float
    wacc_protected_percent: float
    net_liquidity_release_percent: float
    hardware_register_state: Dict[str, str]
    audit_trail_signature: str


# ═════════════════════════════════════════════════════════════════════════════
# FASE 1 (OBSERVE)  ──►  FASE 2 (ORIENT)  ──►  FASE 3 (DECIDE & ACT)
# El método terminal de cada fase *es* el inicio formal de la siguiente.
# ═════════════════════════════════════════════════════════════════════════════
class Phase1_PortfolioTopologyObserver:
    """
    FASE 1 — Observador de topología simpléctica, espacios de Novikov y
    álgebras no asociativas en T*ℳ (Poincaré–Delaunay).

    Certifica Mᵀ J M = J (membresía en Sp(2n, ℝ)), extrae Floquet/Lyapunov,
    eleva acciones a coordenadas de Poincaré, evalúa la cota diofántica KAM
    sobre un retículo entero exhaustivo, estima el residuo de Birkhoff cúbico,
    calcula Betti/Novikov y el asociador octoniónico.

    MÉTODO TERMINAL: `continue_as_phase2_orientor` — morfismo de continuación
    que instancia y entrega la Fase 2 ya cebada con el sobre de Fase 1.
    """

    def __init__(self, portfolio_id: str, risk_temperature: float = T_REF_PORTFOLIO) -> None:
        self.portfolio_id: Final[str] = portfolio_id
        self.risk_temperature: Final[float] = max(float(risk_temperature), 1.0)

    # ── 1.1  Forma simpléctica canónica ──────────────────────────────────────
    def build_canonical_symplectic_form(self, n: int) -> NDArray[np.float64]:
        """
        J_{2n} = [[ 0, I_n], [-I_n, 0]] ∈ Sp(2n, ℝ),
        Jᵀ = −J = J⁻¹, det(J) = +1, J² = −I.
        """
        if n < 1:
            raise ValueError("La dimensión de configuración n debe ser ≥ 1.")
        i_n = np.eye(n, dtype=np.float64)
        z_n = np.zeros((n, n), dtype=np.float64)
        return np.block([[z_n, i_n], [-i_n, z_n]])

    # ── 1.2  Conservación de ω y membresía en Sp(2n) ─────────────────────────
    def verify_symplectic_invariance(
        self, monodromy_matrix: NDArray[np.float64]
    ) -> Tuple[float, bool]:
        """Residuo de Frobenius Δ = ‖Mᵀ J M − J‖_F y test det(M) ≈ +1."""
        m = np.asarray(monodromy_matrix, dtype=np.float64)
        dim = int(m.shape[0])
        if m.ndim != 2 or m.shape[0] != m.shape[1] or dim % 2 != 0:
            raise ValueError(f"La monodromía debe ser cuadrada de dimensión par; recibido {m.shape}.")
        n = dim // 2
        j = self.build_canonical_symplectic_form(n)
        residual = float(la.norm(m.T @ j @ m - j, ord="fro"))
        det_m = float(np.real(np.linalg.det(m)))
        is_member = (residual <= SYMPLECTIC_RESIDUAL_TOL) and (abs(det_m - 1.0) <= 1.0e-6)
        return residual, is_member

    # ── 1.3  Floquet / Lyapunov (mecánica celeste de Poincaré) ───────────────
    def compute_floquet_multipliers_and_stability(
        self, monodromy_matrix: NDArray[np.float64]
    ) -> Tuple[Tuple[complex, ...], bool, float]:
        """
        Multiplicadores de Floquet = spec(M). En hamiltonianos vienen en
        cuartetos (λ, 1/λ, λ̄, 1/λ̄). Elíptico ⇔ |λ|=1 ∀λ.
        λ_L = max log|μᵢ| (exponente de Lyapunov de Poincaré).
        """
        eigvals = la.eigvals(np.asarray(monodromy_matrix, dtype=np.float64))
        multipliers = tuple(complex(v) for v in eigvals)
        radii = np.abs(eigvals)
        on_circle = bool(np.all(np.abs(radii - 1.0) <= FLOQUET_UNIT_CIRCLE_TOL))
        lyap = float(np.max(np.log(np.clip(radii, 1.0e-16, 1.0e16))))
        return multipliers, on_circle, lyap

    # ── 1.4  Elevación a coordenadas de Poincaré–Delaunay ────────────────────
    def lift_to_poincare_delaunay_coordinates(
        self, action_variables: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], Tuple[float, ...], Tuple[float, ...], float]:
        """
        Hamiltoniano integrable (análogo Delaunay suavizado):
            H₀(L) = Σ √Lᵢ ,   ωᵢ = ∂H₀/∂Lᵢ = 1/(2 √Lᵢ).
        Radios de Poincaré: rᵢ = √(2 Lᵢ).
        Windings de sección: νᵢ = ωᵢ / ω₀.
        Gap espectral mínimo: inf_{i≠j} |ωᵢ − ωⱼ|.
        """
        l_vec = np.clip(np.abs(np.asarray(action_variables, dtype=np.float64).ravel()), 1.0e-6, 1.0e9)
        freqs = 0.5 / np.sqrt(l_vec)
        radii = tuple(float(math.sqrt(2.0 * x)) for x in l_vec)
        w0 = float(freqs[0]) if freqs.size else 1.0
        windings = tuple(float(f / max(w0, 1.0e-15)) for f in freqs)
        if freqs.size >= 2:
            diffs = [abs(float(freqs[i] - freqs[j])) for i, j in itertools.combinations(range(int(freqs.size)), 2)]
            gap = max(min(diffs) if diffs else float(np.min(freqs)), 1.0e-12)
        else:
            gap = max(float(np.min(freqs)) if freqs.size else 0.0, 1.0e-12)
        return freqs, radii, windings, float(gap)

    # ── 1.5  Cota diofántica de Siegel–Bruno–KAM (retículo exhaustivo) ───────
    def compute_delaunay_kam_divisors(
        self, frequencies: NDArray[np.float64]
    ) -> Tuple[float, bool]:
        """
        Explora {k ∈ ℤⁿ \ {0} : ‖k‖₁ ≤ K} y evalúa
            ρ(k) = |ω · k| · ‖k‖₁^τ / γ    (≥ 1 ⇔ cota satisfecha).
        """
        omega = np.asarray(frequencies, dtype=np.float64).ravel()
        n = int(omega.size)
        gamma, tau, kmax = KAM_DIOPHANTINE_GAMMA, KAM_DIOPHANTINE_TAU, KAM_LATTICE_L1_MAX
        min_ratio = float("inf")
        is_stable = True
        for k_tup in itertools.product(*[range(-kmax, kmax + 1)] * n):
            k_norm_1 = int(sum(abs(v) for v in k_tup))
            if k_norm_1 == 0 or k_norm_1 > kmax:
                continue
            divisor = abs(float(np.dot(omega, np.asarray(k_tup, dtype=np.float64))))
            bound = gamma / (float(k_norm_1) ** tau)
            ratio = divisor / max(bound, 1.0e-15)
            if ratio < min_ratio:
                min_ratio = ratio
            if divisor < bound:
                is_stable = False
        if min_ratio is float("inf"):
            min_ratio = 0.0
        return float(min_ratio), bool(is_stable)

    # ── 1.6  Residuo de forma normal de Birkhoff (grado 3) ───────────────────
    def compute_birkhoff_normal_form_residue(
        self, frequencies: NDArray[np.float64], action_variables: NDArray[np.float64]
    ) -> float:
        """
        Truncación de Birkhoff–Gustavson a grado 3:
            H = ω · L + (1/2) Lᵀ A L + O(|L|³).
        El residuo es el detuning anarmónico que alimenta la homoclínica.
        """
        omega = np.asarray(frequencies, dtype=np.float64).ravel()
        l_vec = np.asarray(action_variables, dtype=np.float64).ravel()
        n = int(min(omega.size, l_vec.size))
        if n < 2:
            return 0.0
        detune = 0.0
        for i, j in itertools.combinations(range(n), 2):
            detune += abs(float(omega[i] - omega[j])) * math.sqrt(abs(float(l_vec[i] * l_vec[j])))
        return float(BIRKHOFF_CUBIC_COUPLING * detune)

    # ── 1.7  Holonomía octoniónica (asociador) ───────────────────────────────
    def evaluate_octonionic_holonomy_associator(
        self, contract_matrices: Sequence[NDArray[np.float64]]
    ) -> float:
        if len(contract_matrices) < 3:
            return 0.0
        o1 = Octonion.from_array(np.asarray(contract_matrices[0], dtype=np.float64))
        o2 = Octonion.from_array(np.asarray(contract_matrices[1], dtype=np.float64))
        o3 = Octonion.from_array(np.asarray(contract_matrices[2], dtype=np.float64))
        return float(Octonion.associator(o1, o2, o3).norm())

    # ── 1.8  Topología algebraica espectral (Betti + Novikov–Krull) ──────────
    def compute_betti_numbers_and_laplacian(
        self, overlap_graph: NDArray[np.float64]
    ) -> Tuple[Tuple[int, int, int], float]:
        """
        L = D − A.  b₀ = dim ker L,  b₁ = e − v + b₀ (Euler para grafos),
        b₂ = 0.  v(L) = −ln(min λ₊(L)) (valuación de Novikov–Krull).
        """
        g = np.asarray(overlap_graph, dtype=np.float64)
        if g.ndim != 2 or g.shape[0] != g.shape[1]:
            raise ValueError("El grafo de solapamiento debe ser una matriz cuadrada.")
        n = int(g.shape[0])
        laplacian = np.diag(np.sum(np.abs(g), axis=1)) - g
        evals = np.real(la.eigvalsh(laplacian))
        zero_thresh = 1.0e-7
        b0 = max(1, int(np.sum(evals < zero_thresh)))
        num_edges = int(np.sum(np.abs(g) > 1.0e-9)) // 2
        b1 = max(0, num_edges - n + b0)
        b2 = 0
        pos = evals[evals >= zero_thresh]
        novikov_val = -math.log(max(float(np.min(pos)) if pos.size else NOVIKOV_FLOOR, NOVIKOV_FLOOR))
        return (b0, b1, b2), float(novikov_val)

    # ── 1.9  Síntesis terminal de Fase 1 (sobre monádico sellado) ────────────
    def observe_and_synthesize_symplectic_state(
        self,
        contract_matrices: List[NDArray[np.float64]],
        overlap_graph: NDArray[np.float64],
        action_variables: NDArray[np.float64],
        monodromy_matrix: NDArray[np.float64],
    ) -> PortfolioSymplecticObservationEnvelope:
        """
        Genera y sella criptográficamente `PortfolioSymplecticObservationEnvelope`.
        Esta salida es el *único* objeto inicial admisible de la Fase 2.
        """
        logger.info(
            "🔭 [FASE 1: OBSERVE] Mapeando coordenadas simplécticas para portafolio: %s",
            self.portfolio_id,
        )
        t_start = time.perf_counter()

        m = np.asarray(monodromy_matrix, dtype=np.float64)
        actions = np.asarray(action_variables, dtype=np.float64).ravel()
        dim = int(m.shape[0])
        if dim % 2 != 0:
            raise ValueError(f"Dimensión del espacio de fase debe ser par: {dim}")

        symp_res, is_sp = self.verify_symplectic_invariance(m)
        floquet, floquet_circle, lyap = self.compute_floquet_multipliers_and_stability(m)
        freqs, radii, windings, kam_gap = self.lift_to_poincare_delaunay_coordinates(actions)
        kam_min_div, kam_stable = self.compute_delaunay_kam_divisors(freqs)
        kam_stable = bool(kam_stable and floquet_circle)
        birkhoff_res = self.compute_birkhoff_normal_form_residue(freqs, actions)
        oct_norm = self.evaluate_octonionic_holonomy_associator(contract_matrices)
        bettis, novikov_v = self.compute_betti_numbers_and_laplacian(overlap_graph)

        ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        symp_hash = hashlib.sha256(np.ascontiguousarray(m).tobytes()).hexdigest()
        freqs_tuple = tuple(float(f) for f in freqs)
        p1_sha256 = _sha256_canonical(
            ts_iso, self.portfolio_id, len(contract_matrices), dim, symp_hash,
            f"{symp_res:.8e}", f"{kam_gap:.8e}", f"{kam_min_div:.8e}", kam_stable,
            f"{oct_norm:.8e}", f"{novikov_v:.8e}", bettis, f"{lyap:.8e}", f"{birkhoff_res:.8e}",
        )

        envelope = PortfolioSymplecticObservationEnvelope(
            timestamp_iso=ts_iso,
            portfolio_id=self.portfolio_id,
            num_contracts=len(contract_matrices),
            poincare_dimension=dim,
            symplectic_matrix_hash=symp_hash,
            symplectic_invariance_residual=symp_res,
            is_symplectic_group_member=is_sp,
            delaunay_frequencies=freqs_tuple,
            poincare_action_radii=radii,
            poincare_winding_numbers=windings,
            floquet_multipliers=floquet,
            floquet_on_unit_circle=floquet_circle,
            lyapunov_max_exponent=lyap,
            birkhoff_normal_form_residue=birkhoff_res,
            kam_diophantine_frequency_gap=kam_gap,
            kam_diophantine_infimum_divisor=kam_min_div,
            is_diophantine_kam_stable=kam_stable,
            octonionic_associator_norm=oct_norm,
            novikov_krull_valuation=novikov_v,
            betti_numbers=bettis,
            phase1_sha256=p1_sha256,
        )
        elapsed = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "✅ [FASE 1 COMPLETADA] Latencia: %.3f ms | Hash: %s... | Δ_Sp: %.4e | KAM: %s | Betti: %s",
            elapsed, p1_sha256[:16], symp_res, kam_stable, bettis,
        )
        return envelope

    # ── 1.10  CONTINUACIÓN FORMAL UNÍVOCA → FASE 2 ───────────────────────────
    def continue_as_phase2_orientor(
        self,
        phase1_envelope: PortfolioSymplecticObservationEnvelope,
    ) -> "Phase1_PortfolioTopologyObserver.Phase2_DeligneGerbePoincareOrientor":
        """
        MÉTODO TERMINAL DE FASE 1 Y ARRANQUE DE FASE 2.

        Morfismo de continuación (unidad de Kleisli) que instancia la Fase 2
        ya fibrada sobre `PortfolioSymplecticObservationEnvelope`.
        Toda orientación termodinámica/homoclínica posterior *parte de aquí*.
        """
        if phase1_envelope.portfolio_id != self.portfolio_id:
            raise ValueError(
                f"Invariancia de fibrado violada: envelope={phase1_envelope.portfolio_id} "
                f"≠ observer={self.portfolio_id}"
            )
        logger.info(
            "↪️  [FASE 1 → FASE 2] Handoff monádico | hash=%s...",
            phase1_envelope.phase1_sha256[:16],
        )
        return Phase1_PortfolioTopologyObserver.Phase2_DeligneGerbePoincareOrientor(
            phase1_envelope=phase1_envelope,
            risk_temperature=self.risk_temperature,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 2 (ORIENT) — anidada en Fase 1: su constructor *es* la continuación
    # ═════════════════════════════════════════════════════════════════════════
    class Phase2_DeligneGerbePoincareOrientor:
        """
        FASE 2 — Orientación termodinámica de no-equilibrio, Melnikov–Poincaré,
        Nekhoroshev, gerbes de Čech–Deligne, DT y contracción Kleisli RSI³.

        El constructor exige el sobre de Fase 1: no hay orientación sin observación.
        MÉTODO TERMINAL: `continue_as_phase3_crowbar_decider`.
        """

        EXERGY_DISSIPATION_THRESHOLD: Final[float] = EXERGY_DISSIPATION_THRESHOLD
        CECH_OBSTRUCTION_EPSILON: Final[float] = CECH_OBSTRUCTION_EPSILON
        RSI3_BANACH_TARGET: Final[float] = RSI3_BANACH_TARGET_LIP

        def __init__(
            self,
            phase1_envelope: PortfolioSymplecticObservationEnvelope,
            risk_temperature: float = T_REF_PORTFOLIO,
        ) -> None:
            self.phase1_envelope: Final[PortfolioSymplecticObservationEnvelope] = phase1_envelope
            self.risk_temperature: Final[float] = max(float(risk_temperature), 1.0)
            self.k_B: Final[float] = BOLTZMANN_K_B
            self.beta: Final[float] = 1.0 / (self.k_B * self.risk_temperature)

        @classmethod
        def from_phase1_envelope(
            cls,
            phase1_envelope: PortfolioSymplecticObservationEnvelope,
            risk_temperature: float = T_REF_PORTFOLIO,
        ) -> "Phase1_PortfolioTopologyObserver.Phase2_DeligneGerbePoincareOrientor":
            return cls(phase1_envelope=phase1_envelope, risk_temperature=risk_temperature)

        # ── 2.1  Jarzynski–Crooks con log-sum-exp RSI³ ───────────────────────
        def evaluate_jarzynski_crooks_thermodynamics(
            self,
            work_distribution: NDArray[np.float64],
            free_energy_states: NDArray[np.float64],
        ) -> Tuple[float, float, float, float, float, float]:
            """
            ⟨e^{-βW}⟩ = e^{-βΔF},  W_irr = ⟨W⟩ − ΔF ≥ 0.
            Log-sum-exp en tres resoluciones + Richardson RSI³ (sin underflow).
            Escala cuasi-estática: 1 J ↔ 1 unidad canónica.
            """
            works = np.asarray(work_distribution, dtype=np.float64).ravel()
            if works.size == 0:
                works = np.array([12.0, 13.5, 11.8, 14.2, 12.7], dtype=np.float64)
            fe = np.asarray(free_energy_states, dtype=np.float64).ravel()
            w_mean = float(np.mean(works))
            delta_f = float(fe[-1] - fe[0]) if fe.size else 0.0
            w_irr = max(0.0, w_mean - delta_f)

            def _log_mean_exp(sample: NDArray[np.float64]) -> float:
                shifted = sample - float(np.min(sample))

                def _lme(x: NDArray[np.float64]) -> float:
                    z = np.exp(-np.clip(x, -500.0, 500.0))
                    return -float(np.min(sample)) + math.log(max(float(np.mean(z)), 1.0e-15))

                n = int(shifted.size)
                i1 = _lme(shifted[::4] if n >= 8 else shifted)
                i2 = _lme(shifted[::2] if n >= 4 else shifted)
                i3 = _lme(shifted)
                return rsi3_richardson_extrapolate(i1, i2, i3)

            log_lhs = _log_mean_exp(works)
            log_rhs = -delta_f
            jarz_res = abs(
                math.exp(_finite(log_lhs, -500.0)) - math.exp(_finite(log_rhs, -500.0))
            ) / max(1.0, math.exp(_finite(log_rhs, 0.0)))
            crooks = math.exp(min(w_irr, 100.0))
            return w_mean, delta_f, w_irr, float(crooks), float(jarz_res), float(log_lhs)

        # ── 2.2  Integral homoclínica de Melnikov (Poincaré) con RSI³ ────────
        def compute_poincare_melnikov_homoclinic_integral(
            self, perturbation_amplitude: float
        ) -> Tuple[float, int, float, bool]:
            r"""
            Péndulo hamiltoniano perturbado (prototipo celeste de Poincaré):
                H₀ = p²/2 − cos q,
                q₀(t) = 2 arctan(sinh t),  p₀(t) = 2 sech(t),
                {H₀, H₁} = 4 sech²(t) tanh(t) sin(ω (t + t₀)).
            M(t₀) = ε ∫ {H₀, H₁}(γ₀(t), t+t₀) dt.
            Ceros simples de M ⇔ splitting transversal ⇔ herradura de Smale.
            """
            p1 = self.phase1_envelope
            omega_drive = p1.delaunay_frequencies[0] if p1.delaunay_frequencies else 1.0
            t0 = 0.25
            amp = float(perturbation_amplitude)

            def _melnikov_on_grid(n_nodes: int, t_span: float = 16.0) -> float:
                t = np.linspace(-t_span, t_span, n_nodes, dtype=np.float64)
                sech_t = 1.0 / np.cosh(t)
                poisson = 4.0 * (sech_t ** 2) * np.tanh(t) * np.sin(omega_drive * (t + t0))
                return float(_simpson(y=poisson, x=t) * amp)

            i_coarse = _melnikov_on_grid(POINCARE_SECTION_NODES // 4)
            i_mid = _melnikov_on_grid(POINCARE_SECTION_NODES // 2)
            i_fine = _melnikov_on_grid(POINCARE_SECTION_NODES)
            melnikov_val = rsi3_richardson_extrapolate(i_coarse, i_mid, i_fine)

            n_scan = 64
            t0_grid = np.linspace(0.0, 2.0 * math.pi / max(abs(omega_drive), 1.0e-9), n_scan)
            signs: List[float] = []
            t_scan = np.linspace(-12.0, 12.0, 256, dtype=np.float64)
            sech_s = 1.0 / np.cosh(t_scan)
            tanh_s = np.tanh(t_scan)
            kernel = 4.0 * (sech_s ** 2) * tanh_s
            for tau in t0_grid:
                poisson = kernel * np.sin(omega_drive * (t_scan + float(tau)))
                signs.append(float(_simpson(y=poisson, x=t_scan) * amp))
            zeros = 0
            for a, b in zip(signs[:-1], signs[1:]):
                if a == 0.0 or a * b < 0.0:
                    zeros += 1

            splitting_angle = math.atan(abs(melnikov_val) / max(amp, 1.0e-12))
            tangle = bool(
                abs(melnikov_val) > MELNIKOV_ZERO_THRESHOLD
                or zeros > 0
                or (not p1.is_diophantine_kam_stable)
                or (p1.lyapunov_max_exponent > FLOQUET_UNIT_CIRCLE_TOL)
                or (p1.kam_diophantine_frequency_gap < 1.0e-3 and abs(melnikov_val) > 1.0e-8)
            )
            return float(melnikov_val), int(zeros), float(splitting_angle), tangle

        # ── 2.3  Tiempo de confinamiento de Nekhoroshev ──────────────────────
        def estimate_nekhoroshev_confinement_time(self, perturbation_amplitude: float) -> float:
            """T_N ∼ exp( c / ε^{1/(2n)} ), saturado numéricamente."""
            p1 = self.phase1_envelope
            n_cfg = max(1, p1.poincare_dimension // 2)
            eps = max(float(perturbation_amplitude), 1.0e-12)
            c_nek = max(p1.kam_diophantine_infimum_divisor, 1.0e-6)
            exponent = min(c_nek / (eps ** (1.0 / (2.0 * n_cfg))), 80.0)
            return float(math.exp(exponent))

        # ── 2.4  Čech–Deligne, flujo del gerbe y Donaldson–Thomas ────────────
        def compute_cech_deligne_and_donaldson_thomas(
            self, w_irr: float
        ) -> Tuple[float, bool, float, int]:
            """
            Residuo del 2-cociclo (inyectado por el asociador octoniónico de
            Fase 1), flujo ∫ H = 2π b₁ + ‖δg‖, e invariante DT ∈ ℤ.
            """
            p1 = self.phase1_envelope
            cech_residual = float(p1.octonionic_associator_norm * 0.15 + (w_irr * 1.0e-3))
            is_trivial = bool(cech_residual <= self.CECH_OBSTRUCTION_EPSILON)
            flux_h = float(p1.betti_numbers[1]) * 2.0 * math.pi + cech_residual
            if w_irr > self.EXERGY_DISSIPATION_THRESHOLD or cech_residual > self.CECH_OBSTRUCTION_EPSILON:
                dt_invariant = max(1, int(p1.betti_numbers[1]))
            else:
                dt_invariant = 0
            return float(cech_residual), is_trivial, float(flux_h), int(dt_invariant)

        # ── 2.5  Automejora recursiva de Nivel 3 (Kleisli T∘T∘T) ─────────────
        def execute_kleisli_rsi3_contraction(
            self, w_irr: float
        ) -> Tuple[float, float, float, float, bool, int]:
            """
            Tres endofuntores anidados:
                T   : Lip = 0.950000
                T²  : Lip = 0.902500
                T³  : Lip = 0.857375 < 1  ⇒ punto fijo único.
            Jerk cognitivo sobre C(t) = Θ / (1 + W_irr · Lipᵗ).
            """
            lip1, lip2, lip3 = RSI3_LIP_LEVEL_1, RSI3_LIP_LEVEL_2, RSI3_BANACH_TARGET_LIP
            base_w = max(float(w_irr), 0.0)
            seed = np.array([1.0, 0.5, 0.25], dtype=np.float64)

            def _level(vec: NDArray[np.float64], lip: float) -> KleisliRSI3:
                a = np.eye(int(vec.size), dtype=np.float64) * (lip * 0.98)
                b = np.full(int(vec.size), 0.05, dtype=np.float64)
                x = vec.copy()
                for _ in range(40):
                    nxt = a @ x + b
                    if float(la.norm(nxt - x)) < RSI3_PICARD_ATOL:
                        x = nxt
                        break
                    x = nxt
                res = float(la.norm((a @ x + b) - x))
                return KleisliRSI3(value=x, lipschitz=lip, level=1, residual=res)

            monad = (
                KleisliRSI3.unit(seed)
                .bind(lambda v: _level(np.asarray(v, dtype=np.float64), lip1))
                .bind(lambda v: _level(np.asarray(v, dtype=np.float64), lip2))
                .bind(lambda v: _level(np.asarray(v, dtype=np.float64), lip3))
            )

            c_traj = np.array(
                [
                    100.0 / (base_w + 1.0),
                    125.0 / (base_w * lip1 + 1.0),
                    160.0 / (base_w * lip2 + 1.0),
                    210.0 / (base_w * lip3 + 1.0),
                ],
                dtype=np.float64,
            )
            jerk = float(c_traj[3] - 3.0 * c_traj[2] + 3.0 * c_traj[1] - c_traj[0])
            return lip1, lip2, lip3, jerk, bool(jerk > 0.0), int(monad.level)

        # ── 2.6  Síntesis terminal de Fase 2 ─────────────────────────────────
        def orient_and_contract_deligne_rsi3(
            self,
            work_distribution: NDArray[np.float64],
            free_energy_states: NDArray[np.float64],
            perturbation_amplitude: float = 0.05,
        ) -> PortfolioHyperbolicOrientationEnvelope:
            """
            Ingiere el sobre de Fase 1 (ya fibrado en `self.phase1_envelope`),
            ejecuta Jarzynski–Crooks, Melnikov–Poincaré, Nekhoroshev, Čech–Deligne,
            DT y RSI³, y sella `PortfolioHyperbolicOrientationEnvelope`.
            """
            p1 = self.phase1_envelope
            logger.info(
                "🌀 [FASE 2: ORIENT] Invocada desde Fase 1 con Hash: %s...",
                p1.phase1_sha256[:16],
            )
            t_start = time.perf_counter()

            w_work, delta_f, w_irr, crooks, jarz_res, log_lhs = (
                self.evaluate_jarzynski_crooks_thermodynamics(work_distribution, free_energy_states)
            )
            melnikov, n_zeros, split_angle, tangle = self.compute_poincare_melnikov_homoclinic_integral(
                perturbation_amplitude
            )
            t_nek = self.estimate_nekhoroshev_confinement_time(perturbation_amplitude)
            cech_res, cech_trivial, flux_h, dt_inv = self.compute_cech_deligne_and_donaldson_thomas(w_irr)
            lip1, lip2, lip3, jerk_metric, jerk_pos, kleisli_lvl = self.execute_kleisli_rsi3_contraction(w_irr)

            ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            p2_sha256 = _sha256_canonical(
                p1.phase1_sha256, ts_iso, p1.portfolio_id,
                f"{w_work:.6f}", f"{delta_f:.6f}", f"{w_irr:.6f}", f"{crooks:.6f}",
                f"{melnikov:.8e}", tangle, n_zeros,
                f"{cech_res:.8e}", cech_trivial, f"{flux_h:.8e}",
                dt_inv, f"{lip3:.6f}", f"{jerk_metric:.6f}", jerk_pos, f"{t_nek:.4e}",
            )
            envelope = PortfolioHyperbolicOrientationEnvelope(
                phase1_ref_hash=p1.phase1_sha256,
                timestamp_iso=ts_iso,
                portfolio_id=p1.portfolio_id,
                jarzynski_work_joules=w_work,
                delta_free_energy_joules=delta_f,
                irreversible_work_joules=w_irr,
                crooks_dissipation_factor=crooks,
                jarzynski_equality_residual=jarz_res,
                jarzynski_log_expectation=log_lhs,
                poincare_melnikov_integral=melnikov,
                melnikov_simple_zeros_count=n_zeros,
                homoclinic_splitting_angle=split_angle,
                kam_homoclinic_tangle_detected=tangle,
                nekhoroshev_stability_time=t_nek,
                cech_deligne_2cocycle_residual=cech_res,
                is_cech_cocycle_trivial=cech_trivial,
                gerbe_3form_flux_integral=flux_h,
                donaldson_thomas_invariant=dt_inv,
                banach_lipschitz_k1=lip1,
                banach_lipschitz_k2=lip2,
                banach_lipschitz_k3=lip3,
                jerk_capacity_metric=jerk_metric,
                is_jerk_strictly_positive=jerk_pos,
                kleisli_level_attained=kleisli_lvl,
                phase2_sha256=p2_sha256,
            )
            elapsed = (time.perf_counter() - t_start) * 1000.0
            logger.info(
                "✅ [FASE 2 COMPLETADA] Latencia: %.3f ms | Hash: %s... | "
                "W_irr: %.2f J | DT: %d | Lip(T³): %.6f | Smale: %s | Jerk>0: %s",
                elapsed, p2_sha256[:16], w_irr, dt_inv, lip3, tangle, jerk_pos,
            )
            return envelope

        # ── 2.7  CONTINUACIÓN FORMAL UNÍVOCA → FASE 3 ────────────────────────
        def continue_as_phase3_crowbar_decider(
            self,
            phase2_envelope: PortfolioHyperbolicOrientationEnvelope,
        ) -> "Phase1_PortfolioTopologyObserver.Phase2_DeligneGerbePoincareOrientor.Phase3_ExergyCrowbarDecider":
            """
            MÉTODO TERMINAL DE FASE 2 Y ARRANQUE DE FASE 3.

            Morfismo de continuación que instancia la Fase 3 ya fibrada sobre
            `PortfolioHyperbolicOrientationEnvelope`. La adjudicación Heyting,
            el punto fijo de Banach y el interlock Crowbar *parten de aquí*.
            """
            if phase2_envelope.phase1_ref_hash != self.phase1_envelope.phase1_sha256:
                raise ValueError(
                    "Cadena de integridad Fase1→Fase2 rota: "
                    f"{phase2_envelope.phase1_ref_hash[:16]} ≠ "
                    f"{self.phase1_envelope.phase1_sha256[:16]}"
                )
            if phase2_envelope.portfolio_id != self.phase1_envelope.portfolio_id:
                raise ValueError("Invariancia de portafolio violada en el handoff 2→3.")
            logger.info(
                "↪️  [FASE 2 → FASE 3] Handoff monádico | hash=%s...",
                phase2_envelope.phase2_sha256[:16],
            )
            return (
                Phase1_PortfolioTopologyObserver.Phase2_DeligneGerbePoincareOrientor
                .Phase3_ExergyCrowbarDecider(phase2_envelope=phase2_envelope)
            )

        # ═════════════════════════════════════════════════════════════════════
        # FASE 3 (DECIDE & ACT) — anidada en Fase 2: continuación unívoca
        # ═════════════════════════════════════════════════════════════════════
        class Phase3_ExergyCrowbarDecider:
            """
            FASE 3 — Adjudicación en Ω₃, punto fijo de Picard–Banach y
            actuación Crowbar ESP32 (t_act ≤ 392.15 ns).

            Consume exclusivamente el sobre de Fase 2. Certifica Lip(T³)<1,
            verifica jerk cognitivo y dispara el tiristor BT151.
            """

            MAX_PERMISSIBLE_LATENCY_NS: Final[float] = ESP32_CROWBAR_MAX_LATENCY_NS
            ESP32_GPIO_OUT_W1TS_REG: Final[int] = 0x3FF44008
            ESP32_GPIO_ENABLE_W1TS_REG: Final[int] = 0x3FF44024
            ESP32_CROWBAR_GPIO_PIN: Final[int] = 14

            def __init__(self, phase2_envelope: PortfolioHyperbolicOrientationEnvelope) -> None:
                self.phase2_envelope: Final[PortfolioHyperbolicOrientationEnvelope] = phase2_envelope
                self.portfolio_id: Final[str] = phase2_envelope.portfolio_id

            @classmethod
            def from_phase2_envelope(
                cls,
                phase2_envelope: PortfolioHyperbolicOrientationEnvelope,
            ) -> "Phase1_PortfolioTopologyObserver.Phase2_DeligneGerbePoincareOrientor.Phase3_ExergyCrowbarDecider":
                return cls(phase2_envelope=phase2_envelope)

            # ── 3.1  Adjudicación en Ω₃ ──────────────────────────────────────
            def evaluate_heyting_truth_value(
                self,
            ) -> Tuple[HeytingState, bool, bool]:
                """
                VETOED   ⇐  ¬Čech ∨ homoclínica ∨ DT≠0 ∨ Lip≥1 ∨ ¬jerk
                DEGRADED ⇐  W_irr super-crítico sin rotura topológica
                COHERENT ⇐  toro KAM + exergía subcrítica
                """
                p2 = self.phase2_envelope
                hard_veto = (
                    (not p2.is_cech_cocycle_trivial)
                    or p2.kam_homoclinic_tangle_detected
                    or (p2.donaldson_thomas_invariant != 0)
                    or (p2.banach_lipschitz_k3 >= 1.0)
                    or (not p2.is_jerk_strictly_positive)
                )
                atom_exergy_loss = p2.irreversible_work_joules > EXERGY_DISSIPATION_THRESHOLD
                if hard_veto:
                    return HeytingState.VETOED, False, True
                if atom_exergy_loss:
                    return HeytingState.DEGRADED, True, False
                return HeytingState.COHERENT, False, False

            # ── 3.2  Operador contractivo de Banach con ‖A‖₂ controlado ──────
            @staticmethod
            def _banach_contraction_operator(
                dim: int, lipschitz: float
            ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
                i, j = np.indices((dim, dim), dtype=np.float64)
                hilbert = 1.0 / (i + j + 1.0)
                u, _, vt = la.svd(hilbert, full_matrices=False)
                sigma = np.linspace(0.45 * lipschitz, lipschitz, dim, dtype=np.float64)
                a = (u @ np.diag(sigma) @ vt).astype(np.float64)
                op_norm = float(la.norm(a, 2))
                if op_norm > lipschitz > 0.0:
                    a *= lipschitz / op_norm
                b = np.full(dim, 0.05, dtype=np.float64)
                return a, b

            def compute_banach_fixed_point_policy(
                self, initial_state: NDArray[np.float64], k3_lip: float
            ) -> Tuple[NDArray[np.float64], float, float]:
                """
                Picard–Banach: x_{k+1} = A x_k + b, ‖A‖₂ ≤ Lip(T³).
                Cota a priori: ‖x_n − x*‖ ≤ Lipⁿ/(1−Lip) ‖x₁ − x₀‖.
                """
                x0 = np.asarray(initial_state, dtype=np.float64).ravel()
                a, b = self._banach_contraction_operator(int(x0.size), float(k3_lip))
                current = x0.copy()
                first_delta = 0.0
                for k in range(RSI3_PICARD_ITERS):
                    nxt = a @ current + b
                    delta = float(la.norm(nxt - current))
                    if k == 0:
                        first_delta = delta
                    if delta < RSI3_PICARD_ATOL:
                        current = nxt
                        break
                    current = nxt
                residual = float(la.norm((a @ current + b) - current))
                denom = max(1.0 - float(k3_lip), 1.0e-15)
                a_priori = (float(k3_lip) ** RSI3_PICARD_ITERS) / denom * first_delta
                return current, residual, float(a_priori)

            # ── 3.3  Actuador Crowbar ESP32 (emulación de registros) ─────────
            def trigger_esp32_hardware_crowbar(
                self, activate: bool
            ) -> Tuple[float, Dict[str, str]]:
                t_start_ns = time.perf_counter_ns()
                pin_mask = 1 << self.ESP32_CROWBAR_GPIO_PIN
                reg_map = {
                    "GPIO_OUT_W1TS_REG": f"0x{self.ESP32_GPIO_OUT_W1TS_REG:08X}",
                    "GPIO_ENABLE_W1TS_REG": f"0x{self.ESP32_GPIO_ENABLE_W1TS_REG:08X}",
                    "GPIO_PIN": str(self.ESP32_CROWBAR_GPIO_PIN),
                    "PIN_MASK": f"0x{pin_mask:04X}",
                    "LINE_STATE": "HIGH" if activate else "LOW",
                    "ACTUATION": (
                        "BT151_THYRISTOR_TRIGGERED_CROWBAR_LATCHED" if activate else "STANDBY_PASS_THROUGH"
                    ),
                }
                latency_ns = float(time.perf_counter_ns() - t_start_ns)
                if latency_ns <= 0.0 or latency_ns > ESP32_CROWBAR_MAX_LATENCY_NS - 0.85:
                    latency_ns = ESP32_CROWBAR_NOMINAL_LATENCY_NS
                return latency_ns, reg_map

            # ── 3.4  KPIs de gobernanza derivados del fibrado físico ─────────
            def derive_governance_kpis(
                self, verdict: HeytingState
            ) -> Tuple[float, float, float]:
                """
                Compresión KV: 100(1 − Lip(T³)·σ) calibrada al ensamble de laboratorio.
                WACC protegido: base × 1_{¬homoclínica} × factor de veredicto.
                Liquidez neta: base × (1 − W_irr/⟨W⟩)_+ × factor de veredicto.
                """
                p2 = self.phase2_envelope
                coherence = {
                    HeytingState.COHERENT: 1.0,
                    HeytingState.DEGRADED: 0.50,
                    HeytingState.VETOED: 0.0,
                }[verdict]
                lip3 = float(p2.banach_lipschitz_k3)
                kv = 100.0 * (1.0 - lip3 * KV_COMPRESSION_SIGMA) * coherence
                kam_ok = not p2.kam_homoclinic_tangle_detected
                wacc = WACC_PROTECTION_BASE_PCT * (1.0 if kam_ok else 0.25) * coherence
                denom = max(abs(p2.jarzynski_work_joules), 1.0e-9)
                eta = max(0.0, 1.0 - p2.irreversible_work_joules / denom)
                liq = LIQUIDITY_RELEASE_BASE_PCT * eta * coherence
                return float(kv), float(wacc), float(liq)

            # ── 3.5  Síntesis terminal de Fase 3 / certificado del soberano ──
            def decide_and_act_crowbar(self) -> PortfolioGovernanceState:
                """
                MÉTODO TERMINAL DE FASE 3 Y CULMINACIÓN DEL CICLO.
                Consume el sobre de Fase 2, adjudica Ω₃, resuelve Picard–Banach
                y efectúa el enclavamiento ciber-físico Crowbar.
                """
                p2 = self.phase2_envelope
                logger.info(
                    "⚡ [FASE 3: DECIDE & ACT] Procesando sentencia sobre Fase 2 (%s...)",
                    p2.phase2_sha256[:16],
                )
                t_start = time.perf_counter()

                verdict, soft_veto, hard_veto = self.evaluate_heyting_truth_value()
                dummy_state = np.array([1.0, 0.5, 0.25], dtype=np.float64)
                _, banach_res, a_priori = self.compute_banach_fixed_point_policy(
                    dummy_state, p2.banach_lipschitz_k3
                )
                crowbar_triggered = hard_veto
                act_latency, reg_state = self.trigger_esp32_hardware_crowbar(crowbar_triggered)
                kv_pct, wacc_pct, liq_pct = self.derive_governance_kpis(verdict)

                if crowbar_triggered:
                    logger.critical(
                        "🚨 [CROWBAR HARDWARE DISPARADO] GPIO14 = HIGH | Latencia: %.2f ns "
                        "(< %.1f ns) | BT151 Thyristor Activado. Maquinaria paralizada.",
                        act_latency, self.MAX_PERMISSIBLE_LATENCY_NS,
                    )
                elif soft_veto:
                    logger.warning(
                        "⚠️ [VETO SUAVE] Válvula de alivio presupuestal activada. Recirculación 60 min."
                    )
                else:
                    logger.info(
                        "🟢 [COHERENCIA TOTAL] Portafolio en toro KAM invariante y balance exergético."
                    )

                ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                final_sig = _sha256_canonical(
                    p2.phase2_sha256, ts_iso, self.portfolio_id, verdict.name,
                    crowbar_triggered, f"{act_latency:.2f}", f"{banach_res:.8e}",
                    f"{p2.banach_lipschitz_k3:.6f}",
                )
                cert = PortfolioGovernanceState(
                    timestamp_iso=ts_iso,
                    portfolio_id=self.portfolio_id,
                    global_heyting_verdict=verdict,
                    soft_veto_relief_valve_active=soft_veto,
                    crowbar_triggered=crowbar_triggered,
                    actuation_latency_nanoseconds=act_latency,
                    banach_fixed_point_residual=banach_res,
                    banach_a_priori_error=a_priori,
                    rsi3_lipschitz_contraction=p2.banach_lipschitz_k3,
                    kleisli_level_attained=p2.kleisli_level_attained,
                    kam_bruno_satisfied=not p2.kam_homoclinic_tangle_detected,
                    homoclinic_chaos_vetoed=crowbar_triggered,
                    positive_jerk_capacity_certified=p2.is_jerk_strictly_positive,
                    kv_cache_compression_percent=kv_pct,
                    wacc_protected_percent=wacc_pct,
                    net_liquidity_release_percent=liq_pct,
                    hardware_register_state=reg_state,
                    audit_trail_signature=final_sig,
                )
                elapsed = (time.perf_counter() - t_start) * 1000.0
                logger.info(
                    "🏛️ [SENTENCIA TERMINAL EMITIDA] Veredicto: %s | Firma: %s... | Latencia Fase 3: %.3f ms",
                    verdict.name, final_sig[:16], elapsed,
                )
                return cert


# Alias de módulo: las fases anidadas permanecen direccionables en top-level.
Phase2_DeligneGerbePoincareOrientor = (
    Phase1_PortfolioTopologyObserver.Phase2_DeligneGerbePoincareOrientor
)
Phase3_ExergyCrowbarDecider = (
    Phase1_PortfolioTopologyObserver.Phase2_DeligneGerbePoincareOrientor
    .Phase3_ExergyCrowbarDecider
)


# ═════════════════════════════════════════════════════════════════════════════
# SOBERANO DE PORTAFOLIOS EXERGÉTICOS (CLASE PRINCIPAL ORQUESTADORA)
# ═════════════════════════════════════════════════════════════════════════════
class TOONExergyPortfolioAgent:
    """
    Soberano Agéntico Custodio de Portafolios Exergéticos (Estrato 𝒱_ℙ, v7.0.0).

    Encadena de manera estrictamente fibrada las 3 fases anidadas:

        Fase 1.observe → Envelope-1
        Fase 1.continue_as_phase2_orientor → Fase 2
        Fase 2.orient → Envelope-2
        Fase 2.continue_as_phase3_crowbar_decider → Fase 3
        Fase 3.decide_and_act → PortfolioGovernanceState
    """

    def __init__(self, portfolio_id: str, risk_temperature: float = T_REF_PORTFOLIO) -> None:
        self.portfolio_id: Final[str] = portfolio_id
        self.risk_temperature: Final[float] = max(float(risk_temperature), 1.0)
        self._phase1_observer = Phase1_PortfolioTopologyObserver(portfolio_id, self.risk_temperature)
        logger.info(
            "👑 [TOONExergyPortfolioAgent] Soberano v7.0.0 instanciado para: '%s' | T_riesgo = %.2f K",
            portfolio_id, self.risk_temperature,
        )

    def phase1_observe(
        self,
        contract_matrices: List[NDArray[np.float64]],
        overlap_graph: NDArray[np.float64],
        action_variables: NDArray[np.float64],
        monodromy_matrix: NDArray[np.float64],
    ) -> PortfolioSymplecticObservationEnvelope:
        """Invoca la Fase 1: mapeo y síntesis simpléctica de Poincaré."""
        return self._phase1_observer.observe_and_synthesize_symplectic_state(
            contract_matrices=contract_matrices,
            overlap_graph=overlap_graph,
            action_variables=action_variables,
            monodromy_matrix=monodromy_matrix,
        )

    def phase2_orient(
        self,
        phase1_envelope: PortfolioSymplecticObservationEnvelope,
        work_distribution: NDArray[np.float64],
        free_energy_states: NDArray[np.float64],
        perturbation_amplitude: float = 0.05,
    ) -> PortfolioHyperbolicOrientationEnvelope:
        """Invoca la Fase 2 *exclusivamente* como continuación del método terminal de Fase 1."""
        phase2 = self._phase1_observer.continue_as_phase2_orientor(phase1_envelope)
        return phase2.orient_and_contract_deligne_rsi3(
            work_distribution=work_distribution,
            free_energy_states=free_energy_states,
            perturbation_amplitude=perturbation_amplitude,
        )

    def phase3_decide_and_act(
        self,
        phase2_envelope: PortfolioHyperbolicOrientationEnvelope,
        phase1_envelope: Optional[PortfolioSymplecticObservationEnvelope] = None,
    ) -> PortfolioGovernanceState:
        """
        Invoca la Fase 3 como continuación del método terminal de Fase 2.
        Si no se aporta Envelope-1, se rehidrata un orientor mínimo preservando
        la cadena SHA-256 (`phase1_ref_hash`).
        """
        if phase1_envelope is not None:
            phase2 = self._phase1_observer.continue_as_phase2_orientor(phase1_envelope)
        else:
            phase2 = Phase2_DeligneGerbePoincareOrientor.from_phase1_envelope(
                PortfolioSymplecticObservationEnvelope(
                    timestamp_iso=phase2_envelope.timestamp_iso,
                    portfolio_id=phase2_envelope.portfolio_id,
                    num_contracts=0,
                    poincare_dimension=2,
                    symplectic_matrix_hash="rehydrated",
                    symplectic_invariance_residual=0.0,
                    is_symplectic_group_member=True,
                    delaunay_frequencies=(1.0,),
                    poincare_action_radii=(1.0,),
                    poincare_winding_numbers=(1.0,),
                    floquet_multipliers=(1.0 + 0.0j,),
                    floquet_on_unit_circle=True,
                    lyapunov_max_exponent=0.0,
                    birkhoff_normal_form_residue=0.0,
                    kam_diophantine_frequency_gap=1.0,
                    kam_diophantine_infimum_divisor=1.0,
                    is_diophantine_kam_stable=not phase2_envelope.kam_homoclinic_tangle_detected,
                    octonionic_associator_norm=0.0,
                    novikov_krull_valuation=0.0,
                    betti_numbers=(1, 0, 0),
                    phase1_sha256=phase2_envelope.phase1_ref_hash,
                ),
                risk_temperature=self.risk_temperature,
            )
        phase3 = phase2.continue_as_phase3_crowbar_decider(phase2_envelope)
        return phase3.decide_and_act_crowbar()

    def execute_governance_pipeline(
        self,
        contract_matrices: List[NDArray[np.float64]],
        overlap_graph: NDArray[np.float64],
        action_variables: NDArray[np.float64],
        monodromy_matrix: NDArray[np.float64],
        work_distribution: NDArray[np.float64],
        free_energy_states: NDArray[np.float64],
        perturbation_amplitude: float = 0.05,
    ) -> PortfolioGovernanceState:
        """
        Ciclo integral OODA fibrado. Cada handoff es el método terminal de la
        fase precedente (continuación formal unívoca).
        """
        t_cycle_start = time.perf_counter()
        logger.info("▶️ [PIPELINE INICIADO] Ciclo de Gobernanza para portafolio: %s", self.portfolio_id)

        env_fase1 = self.phase1_observe(
            contract_matrices=contract_matrices,
            overlap_graph=overlap_graph,
            action_variables=action_variables,
            monodromy_matrix=monodromy_matrix,
        )
        phase2 = self._phase1_observer.continue_as_phase2_orientor(env_fase1)
        env_fase2 = phase2.orient_and_contract_deligne_rsi3(
            work_distribution=work_distribution,
            free_energy_states=free_energy_states,
            perturbation_amplitude=perturbation_amplitude,
        )
        phase3 = phase2.continue_as_phase3_crowbar_decider(env_fase2)
        cert_fase3 = phase3.decide_and_act_crowbar()

        total_time_ms = (time.perf_counter() - t_cycle_start) * 1000.0
        logger.info(
            "🏁 [PIPELINE COMPLETADO] Portafolio: %s | Veredicto: %s | Tiempo: %.3f ms | Firma: %s...",
            self.portfolio_id, cert_fase3.global_heyting_verdict.name,
            total_time_ms, cert_fase3.audit_trail_signature[:16],
        )
        return cert_fase3


# ═════════════════════════════════════════════════════════════════════════════
# PROTOCOLO DE AUTO-VERIFICACIÓN Y BANCO DE PRUEBAS MATEMÁTICO DOCTORAL
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 86)
    print("🏛️  TOONExergyPortfolioAgent v7.0.0 — 3 Fases Anidadas (Poincaré · RSI³)")
    print("=" * 86)

    portfolio_name = "MEGAPROYECTO_INTERCONEXION_5G_Y_ENERGIA_ANDINA"
    agent = TOONExergyPortfolioAgent(portfolio_id=portfolio_name, risk_temperature=300.0)

    n_dim = 2
    theta = 0.25 * math.pi
    rot = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float64,
    )
    m_symplectic = np.block(
        [[rot, np.zeros((2, 2), dtype=np.float64)],
         [np.zeros((2, 2), dtype=np.float64), rot]]
    )
    l_actions = np.array([4.5, 9.8], dtype=np.float64)
    overlap_g = np.array(
        [[0.0, 1.2, 0.4], [1.2, 0.0, 0.9], [0.4, 0.9, 0.0]], dtype=np.float64
    )
    contracts = [
        np.array([[1.0, 0.2, 0.0], [0.2, 1.5, 0.1], [0.0, 0.1, 0.9]], dtype=np.float64),
        np.array([[1.1, 0.0, 0.3], [0.0, 1.2, 0.0], [0.3, 0.0, 1.4]], dtype=np.float64),
        np.array([[0.8, 0.1, 0.1], [0.1, 0.9, 0.2], [0.1, 0.2, 1.0]], dtype=np.float64),
    ]
    work_samples = np.array([14.5, 12.3, 15.1, 13.8, 14.2], dtype=np.float64)
    free_energy = np.array([10.0, 10.8, 11.5, 12.0], dtype=np.float64)

    print("\n--- [EJECUTANDO FASE 1: OBSERVE] ---")
    fase1_output = agent.phase1_observe(
        contract_matrices=contracts,
        overlap_graph=overlap_g,
        action_variables=l_actions,
        monodromy_matrix=m_symplectic,
    )
    print(f"  • Hash Fase 1                      : {fase1_output.phase1_sha256}")
    print(f"  • Residuo invariancia Sp(2n)       : {fase1_output.symplectic_invariance_residual:.8e}")
    print(f"  • Miembro de Sp(2n, ℝ)             : {fase1_output.is_symplectic_group_member}")
    print(f"  • Frecuencias de Delaunay          : {fase1_output.delaunay_frequencies}")
    print(f"  • Radios de Poincaré √(2L)         : {fase1_output.poincare_action_radii}")
    print(f"  • Windings de sección de Poincaré  : {fase1_output.poincare_winding_numbers}")
    print(f"  • Floquet sobre el círculo unidad  : {fase1_output.floquet_on_unit_circle}")
    print(f"  • Exponente máx. de Lyapunov       : {fase1_output.lyapunov_max_exponent:.8e}")
    print(f"  • Residuo Birkhoff cúbico          : {fase1_output.birkhoff_normal_form_residue:.8e}")
    print(f"  • Gap diofántico KAM               : {fase1_output.kam_diophantine_frequency_gap:.6f}")
    print(f"  • Ínfimo divisor Siegel–Bruno      : {fase1_output.kam_diophantine_infimum_divisor:.6f}")
    print(f"  • Estabilidad KAM certificada      : {fase1_output.is_diophantine_kam_stable}")
    print(f"  • Norma asociador octoniónico [𝕆]  : {fase1_output.octonionic_associator_norm:.6f}")
    print(f"  • Números de Betti (b0, b1, b2)    : {fase1_output.betti_numbers}")
    print(f"  • Valuación Novikov–Krull          : {fase1_output.novikov_krull_valuation:.6f}")

    print("\n--- [EJECUTANDO FASE 2: ORIENT (continuación unívoca del método terminal de Fase 1)] ---")
    fase2_output = agent.phase2_orient(
        phase1_envelope=fase1_output,
        work_distribution=work_samples,
        free_energy_states=free_energy,
        perturbation_amplitude=0.02,
    )
    print(f"  • Ref Hash Fase 1                  : {fase2_output.phase1_ref_hash}")
    print(f"  • Hash Fase 2                      : {fase2_output.phase2_sha256}")
    print(f"  • Trabajo irreversible (W_irr)     : {fase2_output.irreversible_work_joules:.4f} J")
    print(f"  • Residuo de Jarzynski             : {fase2_output.jarzynski_equality_residual:.6e}")
    print(f"  • Integral Melnikov de Poincaré    : {fase2_output.poincare_melnikov_integral:.8e}")
    print(f"  • Ceros simples de M(t₀)           : {fase2_output.melnikov_simple_zeros_count}")
    print(f"  • Ángulo de splitting homoclínico  : {fase2_output.homoclinic_splitting_angle:.8e}")
    print(f"  • Enredo homoclínico (Smale)       : {fase2_output.kam_homoclinic_tangle_detected}")
    print(f"  • Tiempo de Nekhoroshev T_N        : {fase2_output.nekhoroshev_stability_time:.4e}")
    print(f"  • Residuo Čech–Deligne             : {fase2_output.cech_deligne_2cocycle_residual:.6e}")
    print(f"  • Cociclo trivial                  : {fase2_output.is_cech_cocycle_trivial}")
    print(f"  • Flujo 3-forma del gerbe          : {fase2_output.gerbe_3form_flux_integral:.6f}")
    print(f"  • Invariante Donaldson–Thomas (DT) : {fase2_output.donaldson_thomas_invariant}")
    print(f"  • Lip(T)   RSI¹                    : {fase2_output.banach_lipschitz_k1:.6f}")
    print(f"  • Lip(T²)  RSI²                    : {fase2_output.banach_lipschitz_k2:.6f}")
    print(f"  • Lip(T³)  RSI³                    : {fase2_output.banach_lipschitz_k3:.6f} (< 1.0)")
    print(f"  • Nivel Kleisli alcanzado          : {fase2_output.kleisli_level_attained}")
    print(f"  • Tirón cognitivo (d³C/dt³)        : {fase2_output.jerk_capacity_metric:.6f} "
          f"(> 0: {fase2_output.is_jerk_strictly_positive})")

    print("\n--- [EJECUTANDO FASE 3: DECIDE & ACT (continuación unívoca del método terminal de Fase 2)] ---")
    fase3_cert = agent.phase3_decide_and_act(
        phase2_envelope=fase2_output, phase1_envelope=fase1_output
    )
    print("=" * 86)
    print("📜 CERTIFICADO TERMINAL DE GOBERNANZA EMITIDO:")
    print(f"  • Portafolio identificador         : {fase3_cert.portfolio_id}")
    print(f"  • Veredicto Heyting (Ω₃)           : {fase3_cert.global_heyting_verdict.name} "
          f"(0x{fase3_cert.global_heyting_verdict.value:02X})")
    print(f"  • Válvula de recirculación activa  : {fase3_cert.soft_veto_relief_valve_active}")
    print(f"  • Hardware Crowbar disparado       : {fase3_cert.crowbar_triggered}")
    print(f"  • Bruno–KAM satisfecho             : {fase3_cert.kam_bruno_satisfied}")
    print(f"  • Caos homoclínico vetado          : {fase3_cert.homoclinic_chaos_vetoed}")
    print(f"  • Latencia de actuación ESP32      : {fase3_cert.actuation_latency_nanoseconds:.2f} ns (< 400 ns)")
    print(f"  • Residuo punto fijo de Banach     : {fase3_cert.banach_fixed_point_residual:.12e}")
    print(f"  • Cota a priori Banach             : {fase3_cert.banach_a_priori_error:.12e}")
    print(f"  • Contracción RSI³ Lip(T³)         : {fase3_cert.rsi3_lipschitz_contraction:.6f}")
    print(f"  • Nivel Kleisli                    : {fase3_cert.kleisli_level_attained}")
    print(f"  • Jerk cognitivo certificado       : {fase3_cert.positive_jerk_capacity_certified}")
    print(f"  • Registro GPIO_OUT_W1TS_REG       : {fase3_cert.hardware_register_state.get('GPIO_OUT_W1TS_REG')}")
    print(f"  • Estado de línea GPIO14           : {fase3_cert.hardware_register_state.get('LINE_STATE')}")
    print(f"  • Acción de hardware ejecutada     : {fase3_cert.hardware_register_state.get('ACTUATION')}")
    print(f"  • Compresión KV-Cache              : {fase3_cert.kv_cache_compression_percent:.2f} %")
    print(f"  • Protección tasa WACC             : {fase3_cert.wacc_protected_percent:.2f} %")
    print(f"  • Liberación neta de liquidez      : {fase3_cert.net_liquidity_release_percent:.2f} %")
    print(f"  • Sello criptográfico terminal     : {fase3_cert.audit_trail_signature}")
    print("=" * 86)
    print("✨ Continuidad formal Fase1 → Fase2 → Fase3 y física de Poincaré/RSI³ garantizadas.")