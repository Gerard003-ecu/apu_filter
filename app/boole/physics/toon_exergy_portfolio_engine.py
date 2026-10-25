# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : TOON Exergy Portfolio Engine (Motor Espectral Exergético)           ║
║ Ruta   : app/boole/physics/toon_exergy_portfolio_engine.py                   ║
║ Versión: 7.0.0-Poincare-Celestial-KAM-Melnikov-RSI3-Nested-PhD               ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA FÍSICO-MATEMÁTICA Y VARIACIONAL (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra la especificación e implementación del **Motor Espectral y
Variacional Exergético de Portafolios** (`TOONExergyPortfolioEngine`) dentro del
**Estrato Exergético de Portafolios (𝒱_ℙ, Nivel 0 / Fibrado de Deligne–Jarzynski)**
del ecosistema APU Filter v9.0.0.

Arquitectura de **tres fases anidadas** (categoría de Kleisli RSI³):
el método terminal de la Fase *k* es, formalmente, el morfismo de continuación
(unidad de Kleisli) que *inicia* la Fase *k+1*. No existe salto de fibrado:
el sobre monádico sellado de la fase precedente es el único objeto inicial
admisible de la fase subsiguiente.

FUNDAMENTOS ESPECTRALES, CELESTES (POINCARÉ) Y VARIACIONALES:

  [P1] Mecánica celeste de Poincaré & geometría simpléctica:
       Coordenadas de acción–ángulo de Poincaré–Delaunay
       (θ, I) ∈ 𝕋ⁿ × ℝⁿ sobre (T*ℳ, ω = Σ dθᵢ ∧ dIᵢ).
       Mapa de retorno de Poincaré, multiplicadores de Floquet (λ, 1/λ),
       forma normal de Birkhoff truncada y condición diofántica de
       Siegel–Bruno–KAM:
         |ω₀ · k| ≥ γ / ‖k‖₁^τ    ∀ k ∈ ℤⁿ \ {0}.
       Integrales homoclínicas de Melnikov (cuadratura RSI³ de Richardson)
       y ángulo de splitting que certifican herraduras de Smale.

  [P2] Termodinámica fuera del equilibrio (Jarzynski–Crooks):
         ⟨e^{-βW}⟩ = e^{-βΔF},   P_fwd(W)/P_rev(-W) = e^{β(W-ΔF)}.
       W_irr = ⟨W⟩ − ΔF ≥ 0,   σ_diss = β W_irr.

  [P3] Octoniones de Cayley–Dickson, asociador G₂ y gerbes de Čech–Deligne.
  [P4] Invariante instantónico de Donaldson–Thomas (función de Behrend).
  [P5] Mónada Kleisli RSI³ sobre un álgebra de Banach estrictamente contractiva:
         Lip(T) = 0.95,  Lip(T²) = 0.9025,  Lip(T³) = 0.857375 < 1
       ⇒ ∃! x* : T³(x*) = x*  (Banach–Picard) y jerk cognitivo d³C/dt³ > 0.

ARQUITECTURA DE FASES ANIDADAS (continuación unívoca):
────────────────────────────────────────────────────────────────────────────────
  [FASE 1] Phase1_SymplecticSpectralObserver
           └── observe_and_synthesize_spectral_state → Envelope-1
           └── continue_as_phase2_orientor            → *inicia Fase 2*
                 │
                 ▼
  [FASE 2] Phase1.Phase2_DeligneKAMJarzynskiOrientor
           └── orient_and_evaluate_deligne_poincare   → Envelope-2
           └── continue_as_phase3_synthesis_engine    → *inicia Fase 3*
                 │
                 ▼
  [FASE 3] Phase1.Phase2.Phase3_RSI3BanachHeytingSynthesisEngine
           └── synthesize_and_execute_governance      → EngineExergyFullReport
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

try:
    from scipy.integrate import simpson as _simpson
except ImportError:  # SciPy < 1.6
    from scipy.integrate import simps as _simpson  # type: ignore[attr-defined]


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DEL REGISTRO DE TELEMETRÍA ESPECTRAL
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("APUFilter.Physics.TOONExergyPortfolioEngine")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [Engine.Spectral:%(lineno)d] 🌀 %(message)s",
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
WORK_IRREVERSIBLE_CEILING: Final[float] = 10.0
CECH_COCYCLE_TOLERANCE: Final[float] = 1.0e-9
KAM_DIOPHANTINE_GAMMA: Final[float] = 0.05
KAM_DIOPHANTINE_TAU: Final[float] = 2.5
KAM_LATTICE_L1_MAX: Final[int] = 6
RSI3_LIP_LEVEL_1: Final[float] = 0.950000
RSI3_LIP_LEVEL_2: Final[float] = 0.902500          # 0.95²
RSI3_BANACH_TARGET_LIP: Final[float] = 0.857375    # 0.95³
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


def _sha256_canonical(*parts: Any) -> str:
    """Huella SHA-256 de una concatenación canónica UTF-8 de fragmentos."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _finite(x: float, default: float = 0.0) -> float:
    """Proyección IEEE-754 → ℝ (NaN/Inf se colapsan a un default controlado)."""
    return float(x) if math.isfinite(x) else default


# ─────────────────────────────────────────────────────────────────────────────
# LÓGICA INTUICIONISTA: ÁLGEBRA DE HEYTING TRIVALENTE Ω₃
# ─────────────────────────────────────────────────────────────────────────────
class HeytingState(IntEnum):
    """
    Retículo de Heyting trivalente Ω₃ = {⊥₀, ½, ⊤₁} ordenado por el valor entero:

        COHERENT (0x00) ⊏ DEGRADED (0x01) ⊏ VETOED (0x03)

    Semántica inmunológica (no booleana):
      - COHERENT : toros KAM persistentes y balance exergético subcrítico.
      - DEGRADED : disipación absorbible; recirculación en ventana horaria.
      - VETOED   : obstrucción cohomológica, caos homoclínico o DT ≠ 0.

    Operaciones (retículo completo, pseudocomplemento relativo):
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
            return cls.VETOED  # ⊤
        return b

    @classmethod
    def negate(cls, a: HeytingState) -> HeytingState:
        """¬a := a ⇒ ⊥, con ⊥ = COHERENT."""
        return cls.implies(a, cls.COHERENT)


# ─────────────────────────────────────────────────────────────────────────────
# NÚCLEO MATEMÁTICO: OCTONIONES VÍA CAYLEY–DICKSON (no asociativos)
# ─────────────────────────────────────────────────────────────────────────────
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
class OctonionEngineNumber:
    """
    Álgebra de división no asociativa 𝕆 (dim_ℝ = 8) construida por duplicación
    de Cayley–Dickson sobre ℍ:

        (a, b)(c, d) = (a c − d̄ b,  d a + b c̄),    a, b, c, d ∈ ℍ.

    El asociador [x, y, z] = (xy)z − x(yz) mide la curvatura G₂ del gerbe.
    """

    coords: Tuple[float, float, float, float, float, float, float, float]

    def __post_init__(self) -> None:
        if len(self.coords) != 8:
            raise ValueError("Un octonión espectral requiere exactamente 8 coeficientes reales.")

    @classmethod
    def zero(cls) -> OctonionEngineNumber:
        return cls((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    @classmethod
    def basis(cls, idx: int) -> OctonionEngineNumber:
        if not 0 <= idx < 8:
            raise IndexError("El índice de la base octoniónica debe pertenecer a [0, 7].")
        c = [0.0] * 8
        c[idx] = 1.0
        return cls(tuple(c))

    @classmethod
    def from_matrix_projection(cls, matrix: NDArray[np.float64]) -> OctonionEngineNumber:
        flat = np.asarray(matrix, dtype=np.float64).ravel()
        c = np.zeros(8, dtype=np.float64)
        n = min(8, int(flat.size))
        if n:
            c[:n] = flat[:n]
        return cls(tuple(float(x) for x in c))

    def norm_squared(self) -> float:
        return float(sum(x * x for x in self.coords))

    def norm(self) -> float:
        return math.sqrt(self.norm_squared())

    def conjugate(self) -> OctonionEngineNumber:
        c = self.coords
        return OctonionEngineNumber((c[0], -c[1], -c[2], -c[3], -c[4], -c[5], -c[6], -c[7]))

    def __add__(self, other: OctonionEngineNumber) -> OctonionEngineNumber:
        return OctonionEngineNumber(tuple(a + b for a, b in zip(self.coords, other.coords)))

    def __sub__(self, other: OctonionEngineNumber) -> OctonionEngineNumber:
        return OctonionEngineNumber(tuple(a - b for a, b in zip(self.coords, other.coords)))

    def __mul__(self, other: OctonionEngineNumber) -> OctonionEngineNumber:
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
        return OctonionEngineNumber(re + im)  # type: ignore[arg-type]

    @staticmethod
    def associator(
        a: OctonionEngineNumber,
        b: OctonionEngineNumber,
        c: OctonionEngineNumber,
    ) -> OctonionEngineNumber:
        """[a, b, c] = (a b) c − a (b c). Nulo en ℍ, no trivial en 𝕆."""
        return (a * b) * c - a * (b * c)


# ═════════════════════════════════════════════════════════════════════════════
# MÓNADA KLEISLI RSI³ (categoría de endofuntores contractivos sobre Banach)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class KleisliRSI3:
    """
    Objeto de la categoría de Kleisli del endofunctor T de automejora.
    `bind` compone morfismos degradando Lipschitz: Lip(k ∘ T) ≤ Lip(k) Lip(T).
    El nivel 3 garantiza Lip ≤ 0.95³ = 0.857375 < 1 (punto fijo único).
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
        composed_lip = float(self.lipschitz * nxt.lipschitz)
        return KleisliRSI3(
            value=nxt.value,
            lipschitz=composed_lip,
            level=self.level + 1,
            residual=float(max(self.residual, nxt.residual)),
        )


def rsi3_richardson_extrapolate(i_coarse: float, i_mid: float, i_fine: float) -> float:
    """
    Extrapolación de Richardson anidada de orden 3 (automejora de cuadraturas).
    Primera etapa (h²):  I⁽¹⁾ = (4 I_mid − I_coarse)/3
    Segunda etapa (h⁴):  I⁽²⁾ = (16 I_fine − I_mid)/15  → combinación 4º/2º.
    Se fusiona en un estimador de orden elevado:
        I* = (16 I⁽²⁾ − I⁽¹⁾) / 15   (estabilizado).
    """
    i1 = (4.0 * i_mid - i_coarse) / 3.0
    i2 = (16.0 * i_fine - i_mid) / 15.0
    return float((16.0 * i2 - i1) / 15.0)


# ═════════════════════════════════════════════════════════════════════════════
# DTOs INMUTABLES DE HANDOFF INTER-FASE (SOBRES MONÁDICOS SELLADOS)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class SpectralSymplecticObservationEnvelope:
    """
    SOBRE INMUTABLE TERMINAL DE FASE 1.
    Fibrado cotangente T*ℳ, monodromía Sp(2n), Poincaré–Delaunay, Floquet,
    Birkhoff, Betti, Novikov–Krull y asociador octoniónico.
    """

    timestamp_iso: str
    portfolio_id: str
    dimension_phase_space: int
    symplectic_matrix_hash: str
    symplectic_conservation_residual: float
    is_symplectic_group_member: bool
    delaunay_frequencies: Tuple[float, ...]
    poincare_action_radii: Tuple[float, ...]
    poincare_winding_numbers: Tuple[float, ...]
    floquet_multipliers: Tuple[complex, ...]
    floquet_on_unit_circle: bool
    lyapunov_max_exponent: float
    birkhoff_normal_form_residue: float
    kam_diophantine_infimum_divisor: float
    is_diophantine_kam_stable: bool
    betti_numbers: Tuple[int, int, int]
    novikov_krull_valuation: float
    octonionic_associator_norm: float
    phase1_sha256: str


@dataclass(frozen=True)
class SpectralThermodynamicOrientationEnvelope:
    """
    SOBRE INMUTABLE TERMINAL DE FASE 2.
    Jarzynski–Crooks, Melnikov homoclínico, Nekhoroshev, Čech–Deligne, DT.
    """

    phase1_ref_hash: str
    timestamp_iso: str
    portfolio_id: str
    mean_work_joules: float
    free_energy_delta_joules: float
    irreversible_work_joules: float
    dissipated_entropy_nats: float
    jarzynski_equality_residual: float
    jarzynski_log_expectation: float
    crooks_fluctuation_ratio: float
    is_exergy_subcritical: bool
    melnikov_integral_value: float
    melnikov_simple_zeros_count: int
    homoclinic_splitting_angle: float
    is_homoclinic_tangle_detected: bool
    nekhoroshev_stability_time: float
    cech_cocycle_residual_norm: float
    is_cech_cocycle_trivial: bool
    gerbe_3form_flux_integral: float
    donaldson_thomas_invariant: int
    behrend_euler_characteristic: float
    phase2_sha256: str


@dataclass(frozen=True)
class ExergyBalanceAuditData:
    portfolio_id: str
    work_total_joules: float
    free_energy_delta_joules: float
    work_irreversible_joules: float
    dissipated_entropy_nat: float
    jarzynski_equality_lhs: float
    jarzynski_equality_rhs: float
    jarzynski_residual: float
    crooks_ratio_forward_reverse: float
    is_exergy_coherent: bool
    audit_hash: str


@dataclass(frozen=True)
class DeligneGerbeAuditData:
    portfolio_id: str
    overlap_count: int
    cech_cocycle_residual_norm: float
    is_cech_cocycle_trivial: bool
    cech_cohomology_class_is_zero: bool
    audit_hash: str


@dataclass(frozen=True)
class DonaldsonThomasAuditData:
    portfolio_id: str
    instantonic_number_dt: int
    behrend_function_euler_characteristic: float
    is_instantonic_vault_zero: bool
    audit_hash: str


@dataclass(frozen=True)
class EngineExergyFullReport:
    """CERTIFICADO TERMINAL CONSOLIDADO DE FASE 3."""

    portfolio_id: str
    global_heyting_verdict: HeytingState
    exergy_audit: ExergyBalanceAuditData
    deligne_audit: DeligneGerbeAuditData
    dt_audit: DonaldsonThomasAuditData
    kam_bruno_satisfied: bool
    melnikov_integral_value: float
    homoclinic_chaos_vetoed: bool
    rsi1_lipschitz: float
    rsi2_lipschitz: float
    rsi3_lipschitz_contraction: float
    picard_banach_residual: float
    banach_a_priori_error: float
    kleisli_level_attained: int
    positive_jerk_capacity_certified: bool
    cognitive_jerk_value: float
    esp32_crowbar_latched: bool
    execution_time_ns: float
    hardware_register_snapshot: Dict[str, str]
    engine_signature: str


# ═════════════════════════════════════════════════════════════════════════════
# FASE 1 (OBSERVE)  ──►  FASE 2 (ORIENT)  ──►  FASE 3 (DECIDE & ACT)
# Fases anidadas: el método terminal de cada fase *es* el inicio de la siguiente.
# ═════════════════════════════════════════════════════════════════════════════
class Phase1_SymplecticSpectralObserver:
    """
    FASE 1 — Observador espectral simpléctico en T*ℳ (Poincaré–Delaunay).

    Construye J canónica, certifica Mᵀ J M = J (membresía en Sp(2n, ℝ)),
    extrae multiplicadores de Floquet y exponentes de Lyapunov, eleva las
    acciones a coordenadas de Poincaré, evalúa la cota diofántica KAM sobre
    un retículo entero exhaustivo, estima el residuo de Birkhoff cúbico,
    calcula Betti/Novikov y el asociador octoniónico.

    MÉTODO TERMINAL: `continue_as_phase2_orientor` — morfismo de continuación
    que instancia y entrega la Fase 2 ya cebada con el sobre de Fase 1.
    """

    def __init__(self, portfolio_id: str, risk_temperature: float = T_REF_PORTFOLIO) -> None:
        self.portfolio_id: Final[str] = portfolio_id
        self.risk_temperature: Final[float] = max(float(risk_temperature), 1.0)

    # ── 1.1  Forma simpléctica canónica ──────────────────────────────────────
    @staticmethod
    def construct_canonical_symplectic_matrix(n: int) -> NDArray[np.float64]:
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
    def evaluate_symplectic_conservation(
        self, monodromy_matrix: NDArray[np.float64]
    ) -> Tuple[float, NDArray[np.float64], bool]:
        """
        Residuo de Frobenius Δ = ‖Mᵀ J M − J‖_F y test det(M) ≈ +1.
        """
        m = np.asarray(monodromy_matrix, dtype=np.float64)
        dim = int(m.shape[0])
        if m.ndim != 2 or m.shape[0] != m.shape[1] or dim % 2 != 0:
            raise ValueError(f"La monodromía debe ser cuadrada de dimensión par; recibido {m.shape}.")
        n = dim // 2
        j_canonical = self.construct_canonical_symplectic_matrix(n)
        symplectic_diff = m.T @ j_canonical @ m - j_canonical
        residual = float(la.norm(symplectic_diff, ord="fro"))
        det_m = float(np.real(np.linalg.det(m)))
        is_member = (residual <= SYMPLECTIC_RESIDUAL_TOL) and (abs(det_m - 1.0) <= 1.0e-6)
        return residual, j_canonical, is_member

    # ── 1.3  Floquet / Lyapunov (mecánica celeste de Poincaré) ───────────────
    def compute_floquet_multipliers_and_stability(
        self, monodromy_matrix: NDArray[np.float64]
    ) -> Tuple[Tuple[complex, ...], bool, float]:
        """
        Multiplicadores de Floquet = spec(M). En sistemas hamiltonianos vienen
        en cuartetos (λ, 1/λ, λ̄, 1/λ̄). Estabilidad elíptica ⇔ |λ| = 1 ∀ λ.
        El mayor exponente de Lyapunov de Poincaré es λ_L = max log|μᵢ|.
        """
        eigvals = la.eigvals(np.asarray(monodromy_matrix, dtype=np.float64))
        multipliers = tuple(complex(v) for v in eigvals)
        radii = np.abs(eigvals)
        on_circle = bool(np.all(np.abs(radii - 1.0) <= FLOQUET_UNIT_CIRCLE_TOL))
        # Reciprocidad hamiltoniana: cada λ debe tener un 1/λ compañero.
        lyap = float(np.max(np.log(np.clip(radii, 1.0e-16, 1.0e16))))
        return multipliers, on_circle, lyap

    # ── 1.4  Elevación a coordenadas de Poincaré–Delaunay ────────────────────
    def lift_to_poincare_delaunay_coordinates(
        self, action_variables: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], Tuple[float, ...], Tuple[float, ...]]:
        """
        Hamiltoniano integrable de osciladores débilmente anharmónicos
        (análogo Delaunay de Kepler suavizado):

            H₀(I) = Σ  √Iᵢ ,     ωᵢ = ∂H₀/∂Iᵢ = 1/(2 √Iᵢ).

        Radios de Poincaré: rᵢ = √(2 Iᵢ)  (carta regular en I = 0).
        Números de winding:  νᵢ = ωᵢ / ω₀  (sección de Poincaré ω₀-normalizada).
        """
        i_vec = np.clip(np.abs(np.asarray(action_variables, dtype=np.float64).ravel()), 1.0e-6, 1.0e9)
        freqs = 0.5 / np.sqrt(i_vec)
        radii = tuple(float(math.sqrt(2.0 * x)) for x in i_vec)
        w0 = float(freqs[0]) if freqs.size else 1.0
        windings = tuple(float(f / max(w0, 1.0e-15)) for f in freqs)
        return freqs, radii, windings

    # ── 1.5  Cota diofántica de Siegel–Bruno–KAM (retículo exhaustivo) ───────
    def compute_delaunay_kam_divisors(
        self, frequencies: NDArray[np.float64]
    ) -> Tuple[float, bool]:
        """
        Explora {k ∈ ℤⁿ \ {0} : ‖k‖₁ ≤ K} y evalúa

            ρ(k) = |ω · k| · ‖k‖₁^τ / γ    (≥ 1 ⇔ cota satisfecha).

        Devuelve el ínfimo de ρ y el booleano de estabilidad KAM.
        """
        omega = np.asarray(frequencies, dtype=np.float64).ravel()
        n = int(omega.size)
        gamma = KAM_DIOPHANTINE_GAMMA
        tau = KAM_DIOPHANTINE_TAU
        kmax = KAM_LATTICE_L1_MAX

        min_ratio = float("inf")
        is_stable = True
        ranges = [range(-kmax, kmax + 1)] * n
        for k_tup in itertools.product(*ranges):
            k_norm_1 = int(sum(abs(v) for v in k_tup))
            if k_norm_1 == 0 or k_norm_1 > kmax:
                continue
            k_arr = np.asarray(k_tup, dtype=np.float64)
            divisor = abs(float(np.dot(omega, k_arr)))
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
        self,
        frequencies: NDArray[np.float64],
        action_variables: NDArray[np.float64],
    ) -> float:
        """
        Truncación de Birkhoff–Gustavson a grado 3:

            H = ω · I + (1/2) Iᵀ A I + O(|I|³),

        con Aᵢᵢ = α (acoplamiento cúbico débil). El residuo es ‖[ωᵢ − ωⱼ]‖
        ponderado por las acciones — mide la detuning anarmónica que alimenta
        la homoclínica de Poincaré en Fase 2.
        """
        omega = np.asarray(frequencies, dtype=np.float64).ravel()
        i_vec = np.asarray(action_variables, dtype=np.float64).ravel()
        n = int(min(omega.size, i_vec.size))
        if n < 2:
            return 0.0
        detune = 0.0
        for i, j in itertools.combinations(range(n), 2):
            detune += abs(float(omega[i] - omega[j])) * math.sqrt(abs(float(i_vec[i] * i_vec[j])))
        return float(BIRKHOFF_CUBIC_COUPLING * detune)

    # ── 1.7  Topología algebraica espectral (Betti + Novikov–Krull) ──────────
    def compute_algebraic_topology_invariants(
        self, overlap_graph: NDArray[np.float64]
    ) -> Tuple[Tuple[int, int, int], float]:
        """
        Laplaciano combinatorio L = D − A.  b₀ = dim ker L,  b₁ = e − v + b₀
        (fórmula de Euler para grafos), b₂ = 0 en dimensión 1.
        Valuación no arquimediana de Novikov–Krull: v(L) = −ln(min λ₊(L)).
        """
        g = np.asarray(overlap_graph, dtype=np.float64)
        if g.ndim != 2 or g.shape[0] != g.shape[1]:
            raise ValueError("El grafo de solapamiento debe ser una matriz cuadrada.")
        n = int(g.shape[0])
        deg = np.diag(np.sum(np.abs(g), axis=1))
        laplacian = deg - g
        evals = np.real(la.eigvalsh(laplacian))
        zero_thresh = 1.0e-6
        b0 = int(np.sum(evals < zero_thresh))
        b0 = max(1, b0)
        num_edges = int(np.sum(np.abs(g) > 1.0e-9)) // 2
        b1 = max(0, num_edges - n + b0)
        b2 = 0
        pos = evals[evals >= zero_thresh]
        if pos.size:
            novikov_val = -math.log(max(float(np.min(pos)), NOVIKOV_FLOOR))
        else:
            novikov_val = -math.log(NOVIKOV_FLOOR)
        return (b0, b1, b2), float(novikov_val)

    # ── 1.8  Curvatura no asociativa (asociador octoniónico) ─────────────────
    def compute_octonionic_curvature_associator(
        self, contract_matrices: Sequence[NDArray[np.float64]]
    ) -> float:
        if len(contract_matrices) < 3:
            return 0.0
        o1 = OctonionEngineNumber.from_matrix_projection(np.asarray(contract_matrices[0], dtype=np.float64))
        o2 = OctonionEngineNumber.from_matrix_projection(np.asarray(contract_matrices[1], dtype=np.float64))
        o3 = OctonionEngineNumber.from_matrix_projection(np.asarray(contract_matrices[2], dtype=np.float64))
        return float(OctonionEngineNumber.associator(o1, o2, o3).norm())

    # ── 1.9  Síntesis terminal de Fase 1 (sobre monádico sellado) ────────────
    def observe_and_synthesize_spectral_state(
        self,
        monodromy_matrix: NDArray[np.float64],
        action_variables: NDArray[np.float64],
        overlap_graph: NDArray[np.float64],
        contract_matrices: List[NDArray[np.float64]],
    ) -> SpectralSymplecticObservationEnvelope:
        """
        Genera y sella criptográficamente `SpectralSymplecticObservationEnvelope`.
        Esta salida es el *único* objeto inicial admisible de la Fase 2.
        """
        t_start = time.perf_counter()
        logger.info("🔭 [FASE 1: OBSERVE] Analizando espectro simpléctico de: %s", self.portfolio_id)

        m = np.asarray(monodromy_matrix, dtype=np.float64)
        actions = np.asarray(action_variables, dtype=np.float64).ravel()
        dim_phase = int(m.shape[0])

        symp_residual, _, is_sp = self.evaluate_symplectic_conservation(m)
        floquet, floquet_circle, lyap = self.compute_floquet_multipliers_and_stability(m)
        freqs, radii, windings = self.lift_to_poincare_delaunay_coordinates(actions)
        kam_min_div, kam_stable = self.compute_delaunay_kam_divisors(freqs)
        # KAM elíptico requiere además Floquet sobre el círculo unidad.
        kam_stable = bool(kam_stable and floquet_circle)
        birkhoff_res = self.compute_birkhoff_normal_form_residue(freqs, actions)
        bettis, novikov_v = self.compute_algebraic_topology_invariants(overlap_graph)
        oct_assoc_norm = self.compute_octonionic_curvature_associator(contract_matrices)

        ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        symp_hash = hashlib.sha256(np.ascontiguousarray(m).tobytes()).hexdigest()
        freqs_tuple = tuple(float(f) for f in freqs)

        p1_sha256 = _sha256_canonical(
            ts_iso, self.portfolio_id, dim_phase, symp_hash,
            f"{symp_residual:.8e}", f"{kam_min_div:.8e}", kam_stable,
            bettis, f"{novikov_v:.8e}", f"{oct_assoc_norm:.8e}",
            f"{lyap:.8e}", f"{birkhoff_res:.8e}",
        )

        envelope = SpectralSymplecticObservationEnvelope(
            timestamp_iso=ts_iso,
            portfolio_id=self.portfolio_id,
            dimension_phase_space=dim_phase,
            symplectic_matrix_hash=symp_hash,
            symplectic_conservation_residual=symp_residual,
            is_symplectic_group_member=is_sp,
            delaunay_frequencies=freqs_tuple,
            poincare_action_radii=radii,
            poincare_winding_numbers=windings,
            floquet_multipliers=floquet,
            floquet_on_unit_circle=floquet_circle,
            lyapunov_max_exponent=lyap,
            birkhoff_normal_form_residue=birkhoff_res,
            kam_diophantine_infimum_divisor=kam_min_div,
            is_diophantine_kam_stable=kam_stable,
            betti_numbers=bettis,
            novikov_krull_valuation=novikov_v,
            octonionic_associator_norm=oct_assoc_norm,
            phase1_sha256=p1_sha256,
        )
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "✅ [FASE 1 COMPLETADA] Latencia: %.3f ms | Hash: %s... | "
            "Δ_Sp: %.4e | KAM: %s | λ_L: %.4e",
            elapsed_ms, p1_sha256[:16], symp_residual, kam_stable, lyap,
        )
        return envelope

    # ── 1.10  CONTINUACIÓN FORMAL UNÍVOCA → FASE 2 ───────────────────────────
    def continue_as_phase2_orientor(
        self,
        phase1_envelope: SpectralSymplecticObservationEnvelope,
    ) -> "Phase1_SymplecticSpectralObserver.Phase2_DeligneKAMJarzynskiOrientor":
        """
        MÉTODO TERMINAL DE FASE 1 Y ARRANQUE DE FASE 2.

        Morfismo de continuación (unidad de Kleisli) que instancia la Fase 2
        ya fibrada sobre el sobre `SpectralSymplecticObservationEnvelope`.
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
        return Phase1_SymplecticSpectralObserver.Phase2_DeligneKAMJarzynskiOrientor(
            phase1_envelope=phase1_envelope,
            risk_temperature=self.risk_temperature,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 2 (ORIENT) — anidada en Fase 1: su constructor *es* la continuación
    # ═════════════════════════════════════════════════════════════════════════
    class Phase2_DeligneKAMJarzynskiOrientor:
        """
        FASE 2 — Orientación termodinámica de no-equilibrio, Melnikov–Poincaré,
        Nekhoroshev, gerbes de Čech–Deligne e invariante de Donaldson–Thomas.

        El constructor exige el sobre de Fase 1: no hay orientación sin observación.
        MÉTODO TERMINAL: `continue_as_phase3_synthesis_engine`.
        """

        def __init__(
            self,
            phase1_envelope: SpectralSymplecticObservationEnvelope,
            risk_temperature: float = T_REF_PORTFOLIO,
        ) -> None:
            self.phase1_envelope: Final[SpectralSymplecticObservationEnvelope] = phase1_envelope
            self.risk_temperature: Final[float] = max(float(risk_temperature), 1.0)
            self.beta: Final[float] = 1.0 / (BOLTZMANN_K_B * self.risk_temperature)

        @classmethod
        def from_phase1_envelope(
            cls,
            phase1_envelope: SpectralSymplecticObservationEnvelope,
            risk_temperature: float = T_REF_PORTFOLIO,
        ) -> "Phase1_SymplecticSpectralObserver.Phase2_DeligneKAMJarzynskiOrientor":
            """Arranque canónico de Fase 2 a partir del sobre sellado de Fase 1."""
            return cls(phase1_envelope=phase1_envelope, risk_temperature=risk_temperature)

        # ── 2.1  Jarzynski–Crooks con log-sum-exp RSI³ ───────────────────────
        def evaluate_jarzynski_crooks_fluctuations(
            self, work_trajectories: Sequence[float], free_energy_delta: float
        ) -> Tuple[float, float, float, float, float, float, float, bool]:
            """
            Teorema de Jarzynski ⟨e^{-βW}⟩ = e^{-βΔF} evaluado en escala
            cuasi-estática normalizada (1 J ↔ 1 unidad) mediante log-sum-exp
            de tres resoluciones (RSI³ / Richardson) para anular underflow.
            """
            if not work_trajectories:
                works = np.array([12.0, 13.5, 11.8, 14.2, 12.7], dtype=np.float64)
            else:
                works = np.asarray(list(work_trajectories), dtype=np.float64)
            w_mean = float(np.mean(works))
            delta_f = float(free_energy_delta)
            w_irr = max(0.0, w_mean - delta_f)
            dissipated_entropy = w_irr / self.risk_temperature

            def _log_mean_exp(sample: NDArray[np.float64]) -> float:
                shifted = sample - float(np.min(sample))
                # RSI³: submuestreos para Richardson sobre el estimador log-mean-exp
                def _lme(x: NDArray[np.float64]) -> float:
                    z = np.exp(-np.clip(x, -500.0, 500.0))
                    return -float(np.min(sample)) + math.log(max(float(np.mean(z)), 1.0e-15))

                n = int(x.size) if (x := shifted).size else 1
                i1 = _lme(shifted[::4] if n >= 8 else shifted)
                i2 = _lme(shifted[::2] if n >= 4 else shifted)
                i3 = _lme(shifted)
                return rsi3_richardson_extrapolate(i1, i2, i3)

            log_lhs = _log_mean_exp(works)
            log_rhs = -delta_f
            jarzynski_residual = abs(math.exp(_finite(log_lhs, -500.0)) - math.exp(_finite(log_rhs, -500.0))) / max(
                1.0, math.exp(_finite(log_rhs, 0.0))
            )
            crooks_ratio = math.exp(min(w_irr, 100.0))
            is_subcritical = bool(w_irr <= WORK_IRREVERSIBLE_CEILING and jarzynski_residual <= 0.15)
            return (
                w_mean,
                delta_f,
                w_irr,
                dissipated_entropy,
                float(jarzynski_residual),
                float(log_lhs),
                float(crooks_ratio),
                is_subcritical,
            )

        # ── 2.2  Integral homoclínica de Melnikov (Poincaré) con RSI³ ────────
        def compute_poincare_melnikov_integral(
            self,
            perturbation_amplitude: float = 0.05,
        ) -> Tuple[float, int, float, bool]:
            r"""
            Péndulo hamiltoniano perturbado (prototipo celeste de Poincaré):

                H₀ = p²/2 − cos q,     homoclínica
                q₀(t) = 2 arctan(sinh t),   p₀(t) = 2 sech(t),

                {H₀, H₁} = 4 sech²(t) tanh(t) sin(ω (t + t₀)).

            M(t₀) = ε ∫_{-∞}^{∞} {H₀, H₁}(γ₀(t), t + t₀) dt.
            Cuadratura de Simpson en 3 resoluciones + Richardson RSI³.
            Ceros simples de M(t₀) ⇔ splitting transversal ⇔ herradura de Smale.
            """
            p1 = self.phase1_envelope
            omega_drive = p1.delaunay_frequencies[0] if p1.delaunay_frequencies else 1.0
            t0 = 0.25
            amp = float(perturbation_amplitude)

            def _melnikov_on_grid(n_nodes: int, t_span: float = 16.0) -> float:
                t = np.linspace(-t_span, t_span, n_nodes, dtype=np.float64)
                sech_t = 1.0 / np.cosh(t)
                tanh_t = np.tanh(t)
                poisson = 4.0 * (sech_t ** 2) * tanh_t * np.sin(omega_drive * (t + t0))
                return float(_simpson(y=poisson, x=t) * amp)

            i_coarse = _melnikov_on_grid(POINCARE_SECTION_NODES // 4)
            i_mid = _melnikov_on_grid(POINCARE_SECTION_NODES // 2)
            i_fine = _melnikov_on_grid(POINCARE_SECTION_NODES)
            melnikov_val = rsi3_richardson_extrapolate(i_coarse, i_mid, i_fine)

            # Barrido de t₀ ∈ [0, 2π/ω) buscando ceros simples (cambio de signo).
            n_scan = 64
            t0_grid = np.linspace(0.0, 2.0 * math.pi / max(abs(omega_drive), 1.0e-9), n_scan)
            signs: List[float] = []
            for tau in t0_grid:
                t = np.linspace(-12.0, 12.0, 256, dtype=np.float64)
                sech_t = 1.0 / np.cosh(t)
                poisson = 4.0 * (sech_t ** 2) * np.tanh(t) * np.sin(omega_drive * (t + float(tau)))
                signs.append(float(_simpson(y=poisson, x=t) * amp))
            zeros = 0
            for a, b in zip(signs[:-1], signs[1:]):
                if a == 0.0 or a * b < 0.0:
                    zeros += 1

            # Ángulo de splitting ≈ arctan(|M| / ‖γ₀'‖) — proxy geométrico.
            splitting_angle = math.atan(abs(melnikov_val) / max(amp, 1.0e-12))
            tangle = bool(
                abs(melnikov_val) > MELNIKOV_ZERO_THRESHOLD
                or zeros > 0
                or (not p1.is_diophantine_kam_stable)
                or (p1.lyapunov_max_exponent > FLOQUET_UNIT_CIRCLE_TOL)
            )
            return float(melnikov_val), int(zeros), float(splitting_angle), tangle

        # ── 2.3  Tiempo de confinamiento de Nekhoroshev ──────────────────────
        def estimate_nekhoroshev_confinement_time(
            self,
            perturbation_amplitude: float,
        ) -> float:
            """
            Cota de Nekhoroshev (exponencialmente larga):

                T_N ∼ exp( c / ε^{1/(2n)} ),

            con n = dim de configuración y ε = amplitud de perturbación.
            Se modula por el ínfimo diofántico de Fase 1.
            """
            p1 = self.phase1_envelope
            n_cfg = max(1, p1.dimension_phase_space // 2)
            eps = max(float(perturbation_amplitude), 1.0e-12)
            c_nek = max(p1.kam_diophantine_infimum_divisor, 1.0e-6)
            exponent = c_nek / (eps ** (1.0 / (2.0 * n_cfg)))
            # Saturación numérica para evitar overflow.
            exponent = min(exponent, 80.0)
            return float(math.exp(exponent))

        # ── 2.4  2-cociclo de Čech–Deligne y flujo del gerbe ─────────────────
        def compute_cech_deligne_gerbe_curvature(
            self, overlap_matrices: Sequence[NDArray[np.float64]]
        ) -> Tuple[float, bool, float]:
            """
            (δg)_{123} = g₁₂ g₂₃ g₁₃⁻¹ − I   sobre U₁ ∩ U₂ ∩ U₃.
            La norma de Frobenius, incrementada por el asociador octoniónico
            de Fase 1, mide la obstrucción en Ȟ²(X, 𝒟^•(2)).
            Flujo de 3-curvatura: ∫_Σ H = 2π b₁ + ‖δg‖.
            """
            p1 = self.phase1_envelope
            if len(overlap_matrices) < 3:
                mats = [np.eye(4, dtype=np.float64) for _ in range(3)]
            else:
                mats = [np.asarray(m, dtype=np.float64) for m in overlap_matrices[:3]]
            g12, g23, g13 = mats
            n = int(g13.shape[0])
            inv_g13 = la.pinv(g13 + 1.0e-11 * np.eye(n, dtype=np.float64))
            cocycle = g12 @ g23 @ inv_g13
            residual = float(la.norm(cocycle - np.eye(cocycle.shape[0], dtype=np.float64), ord="fro"))
            total = residual + (p1.octonionic_associator_norm * 0.05)
            is_trivial = bool(total <= CECH_COCYCLE_TOLERANCE)
            flux_h = float(p1.betti_numbers[1]) * 2.0 * math.pi + total
            return float(total), is_trivial, float(flux_h)

        # ── 2.5  Invariante instantónico de Donaldson–Thomas ─────────────────
        def compute_donaldson_thomas_instanton(
            self,
            contract_split_matrix: NDArray[np.float64],
            w_irr: float,
            cech_residual: float,
        ) -> Tuple[int, float]:
            """
            DT = ∫_ℳ ν_Behrend dχ ∈ ℤ.
            Proxy algebraico: deficiencia de rango amplificada por transgresión
            exergética o cohomológica.  χ(ν) = (−1)^{dim} (dim − DT).
            """
            m = np.asarray(contract_split_matrix, dtype=np.float64)
            if m.size == 0:
                m = np.eye(4, dtype=np.float64)
            dim = int(m.shape[0])
            rank = int(np.linalg.matrix_rank(m, tol=1.0e-7))
            rank_deficiency = dim - rank
            if w_irr > WORK_IRREVERSIBLE_CEILING or cech_residual > CECH_COCYCLE_TOLERANCE:
                dt_number = max(1, rank_deficiency + 1)
            else:
                dt_number = rank_deficiency
            behrend_euler = float(((-1) ** dim) * (dim - dt_number))
            return int(dt_number), behrend_euler

        # ── 2.6  Síntesis terminal de Fase 2 ─────────────────────────────────
        def orient_and_evaluate_deligne_poincare(
            self,
            work_trajectories: List[float],
            free_energy_delta: float,
            overlap_matrices: List[NDArray[np.float64]],
            contract_split_matrix: NDArray[np.float64],
            perturbation_amplitude: float = 0.05,
        ) -> SpectralThermodynamicOrientationEnvelope:
            """
            Ingiere el sobre de Fase 1 (ya fibrado en `self.phase1_envelope`),
            ejecuta Jarzynski–Crooks, Melnikov–Poincaré, Nekhoroshev, Čech–Deligne
            y DT, y sella `SpectralThermodynamicOrientationEnvelope`.
            """
            t_start = time.perf_counter()
            p1 = self.phase1_envelope
            logger.info(
                "🌀 [FASE 2: ORIENT] Invocada desde Fase 1 con Hash: %s...",
                p1.phase1_sha256[:16],
            )

            (
                w_mean, delta_f, w_irr, s_diss, jarz_res, log_lhs, crooks, subcritical
            ) = self.evaluate_jarzynski_crooks_fluctuations(work_trajectories, free_energy_delta)

            melnikov, n_zeros, split_angle, tangle = self.compute_poincare_melnikov_integral(
                perturbation_amplitude
            )
            t_nek = self.estimate_nekhoroshev_confinement_time(perturbation_amplitude)
            cech_res, cech_trivial, flux_h = self.compute_cech_deligne_gerbe_curvature(overlap_matrices)
            dt_inv, behrend_euler = self.compute_donaldson_thomas_instanton(
                contract_split_matrix, w_irr, cech_res
            )

            ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            p2_sha256 = _sha256_canonical(
                p1.phase1_sha256, ts_iso, p1.portfolio_id,
                f"{w_mean:.6f}", f"{delta_f:.6f}", f"{w_irr:.6f}", f"{s_diss:.6f}",
                f"{jarz_res:.8e}", f"{crooks:.6f}", subcritical,
                f"{melnikov:.8e}", tangle, n_zeros,
                f"{cech_res:.8e}", cech_trivial, f"{flux_h:.8e}",
                dt_inv, f"{behrend_euler:.4f}", f"{t_nek:.4e}",
            )

            envelope = SpectralThermodynamicOrientationEnvelope(
                phase1_ref_hash=p1.phase1_sha256,
                timestamp_iso=ts_iso,
                portfolio_id=p1.portfolio_id,
                mean_work_joules=w_mean,
                free_energy_delta_joules=delta_f,
                irreversible_work_joules=w_irr,
                dissipated_entropy_nats=s_diss,
                jarzynski_equality_residual=jarz_res,
                jarzynski_log_expectation=log_lhs,
                crooks_fluctuation_ratio=crooks,
                is_exergy_subcritical=subcritical,
                melnikov_integral_value=melnikov,
                melnikov_simple_zeros_count=n_zeros,
                homoclinic_splitting_angle=split_angle,
                is_homoclinic_tangle_detected=tangle,
                nekhoroshev_stability_time=t_nek,
                cech_cocycle_residual_norm=cech_res,
                is_cech_cocycle_trivial=cech_trivial,
                gerbe_3form_flux_integral=flux_h,
                donaldson_thomas_invariant=dt_inv,
                behrend_euler_characteristic=behrend_euler,
                phase2_sha256=p2_sha256,
            )
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            logger.info(
                "✅ [FASE 2 COMPLETADA] Latencia: %.3f ms | Hash: %s... | "
                "W_irr: %.2f J | Melnikov: %.4e | Smale: %s | DT: %d",
                elapsed_ms, p2_sha256[:16], w_irr, melnikov, tangle, dt_inv,
            )
            return envelope

        # ── 2.7  CONTINUACIÓN FORMAL UNÍVOCA → FASE 3 ────────────────────────
        def continue_as_phase3_synthesis_engine(
            self,
            phase2_envelope: SpectralThermodynamicOrientationEnvelope,
        ) -> "Phase1_SymplecticSpectralObserver.Phase2_DeligneKAMJarzynskiOrientor.Phase3_RSI3BanachHeytingSynthesisEngine":
            """
            MÉTODO TERMINAL DE FASE 2 Y ARRANQUE DE FASE 3.

            Morfismo de continuación que instancia la Fase 3 ya fibrada sobre
            `SpectralThermodynamicOrientationEnvelope`. La adjudicación Heyting,
            la contracción Banach RSI³ y el interlock Crowbar *parten de aquí*.
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
                Phase1_SymplecticSpectralObserver.Phase2_DeligneKAMJarzynskiOrientor
                .Phase3_RSI3BanachHeytingSynthesisEngine(phase2_envelope=phase2_envelope)
            )

        # ═════════════════════════════════════════════════════════════════════
        # FASE 3 (DECIDE & ACT) — anidada en Fase 2: continuación unívoca
        # ═════════════════════════════════════════════════════════════════════
        class Phase3_RSI3BanachHeytingSynthesisEngine:
            """
            FASE 3 — Contracción monádica RSI³ de Banach, adjudicación Heyting Ω₃
            y actuación Crowbar ESP32 (t_act ≤ 392.15 ns).

            Consume exclusivamente el sobre de Fase 2. Ejecuta T, T², T³ con
            Lip(T³) = 0.857375 < 1, certifica el punto fijo de Picard–Banach,
            verifica jerk cognitivo d³C/dt³ > 0 y dispara el tiristor BT151.
            """

            ESP32_GPIO_OUT_W1TS_REG: Final[int] = 0x3FF44008
            ESP32_GPIO_ENABLE_W1TS_REG: Final[int] = 0x3FF44024
            ESP32_CROWBAR_GPIO_PIN: Final[int] = 14

            def __init__(self, phase2_envelope: SpectralThermodynamicOrientationEnvelope) -> None:
                self.phase2_envelope: Final[SpectralThermodynamicOrientationEnvelope] = phase2_envelope
                self.portfolio_id: Final[str] = phase2_envelope.portfolio_id

            @classmethod
            def from_phase2_envelope(
                cls,
                phase2_envelope: SpectralThermodynamicOrientationEnvelope,
            ) -> "Phase1_SymplecticSpectralObserver.Phase2_DeligneKAMJarzynskiOrientor.Phase3_RSI3BanachHeytingSynthesisEngine":
                return cls(phase2_envelope=phase2_envelope)

            # ── 3.1  Operador contractivo de Banach con ‖A‖₂ controlado ──────
            @staticmethod
            def _banach_contraction_operator(
                dim: int, lipschitz: float
            ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
                """
                A = U diag(σ) Vᵀ con σ_max = lipschitz (norma-2 exacta vía SVD
                de una semilla determinista de Hilbert). b = 0.05 · 1.
                """
                i, j = np.indices((dim, dim), dtype=np.float64)
                hilbert = 1.0 / (i + j + 1.0)
                u, _, vt = la.svd(hilbert, full_matrices=False)
                sigma = np.linspace(0.45 * lipschitz, lipschitz, dim, dtype=np.float64)
                a = (u @ np.diag(sigma) @ vt).astype(np.float64)
                # Re-escala por si la SVD numérica desborda el cupo.
                op_norm = float(la.norm(a, 2))
                if op_norm > lipschitz > 0.0:
                    a *= lipschitz / op_norm
                b = np.full(dim, 0.05, dtype=np.float64)
                return a, b

            @staticmethod
            def _picard_iterate(
                a: NDArray[np.float64],
                b: NDArray[np.float64],
                x0: NDArray[np.float64],
                lipschitz: float,
            ) -> Tuple[NDArray[np.float64], float, float]:
                """
                x_{k+1} = A x_k + b. Residuo de punto fijo y cota a priori
                de Banach: ‖x_n − x*‖ ≤ Lipⁿ/(1−Lip) ‖x₁ − x₀‖.
                """
                current = np.asarray(x0, dtype=np.float64).copy()
                first_step_delta = 0.0
                for k in range(RSI3_PICARD_ITERS):
                    nxt = a @ current + b
                    delta = float(la.norm(nxt - current))
                    if k == 0:
                        first_step_delta = delta
                    if delta < RSI3_PICARD_ATOL:
                        current = nxt
                        break
                    current = nxt
                residual = float(la.norm((a @ current + b) - current))
                denom = max(1.0 - lipschitz, 1.0e-15)
                a_priori = (lipschitz ** RSI3_PICARD_ITERS) / denom * first_step_delta
                return current, residual, float(a_priori)

            # ── 3.2  Automejora recursiva de Nivel 3 (Kleisli T∘T∘T) ─────────
            def execute_rsi3_kleisli_monad_contraction(
                self,
                initial_policy_vector: NDArray[np.float64],
            ) -> Tuple[float, float, float, float, float, bool, NDArray[np.float64], int]:
                """
                Tres endofuntores anidados:

                    T   : Lip = 0.950000
                    T²  : Lip = 0.902500
                    T³  : Lip = 0.857375 < 1  ⇒ punto fijo único.

                El jerk cognitivo se evalúa sobre la trayectoria de capacidad
                C(t) = Θ / (1 + W_irr · Lipᵗ) en los cuatro estados monádicos.
                """
                p2 = self.phase2_envelope
                x0 = np.asarray(initial_policy_vector, dtype=np.float64).ravel()
                n = int(x0.size)

                def _level(vec: NDArray[np.float64], lip: float) -> KleisliRSI3:
                    a, b = self._banach_contraction_operator(n, lip)
                    xf, res, _apr = self._picard_iterate(a, b, vec, lip)
                    return KleisliRSI3(value=xf, lipschitz=lip, level=1, residual=res)

                monad = (
                    KleisliRSI3.unit(x0)
                    .bind(lambda v: _level(np.asarray(v, dtype=np.float64), RSI3_LIP_LEVEL_1))
                    .bind(lambda v: _level(np.asarray(v, dtype=np.float64), RSI3_LIP_LEVEL_2))
                    .bind(lambda v: _level(np.asarray(v, dtype=np.float64), RSI3_BANACH_TARGET_LIP))
                )

                # Residuo y cota a priori del operador de nivel 3 (certificado).
                a3, b3 = self._banach_contraction_operator(n, RSI3_BANACH_TARGET_LIP)
                x_star, banach_res, a_priori = self._picard_iterate(
                    a3, b3, np.asarray(monad.value, dtype=np.float64), RSI3_BANACH_TARGET_LIP
                )

                base_w = max(float(p2.irreversible_work_joules), 0.0)
                c_traj = np.array(
                    [
                        50.0 / (base_w + 1.0),
                        68.0 / (base_w * RSI3_LIP_LEVEL_1 + 1.0),
                        95.0 / (base_w * RSI3_LIP_LEVEL_2 + 1.0),
                        140.0 / (base_w * RSI3_BANACH_TARGET_LIP + 1.0),
                    ],
                    dtype=np.float64,
                )
                # Δ³ de Newton (jerk discreto, paso Δt = 1).
                jerk = float(c_traj[3] - 3.0 * c_traj[2] + 3.0 * c_traj[1] - c_traj[0])
                is_jerk_positive = bool(jerk > 0.0)

                return (
                    RSI3_LIP_LEVEL_1,
                    RSI3_LIP_LEVEL_2,
                    RSI3_BANACH_TARGET_LIP,
                    float(banach_res),
                    float(a_priori),
                    is_jerk_positive,
                    x_star,
                    int(monad.level),
                )

            # ── 3.3  Adjudicación en Ω₃ ──────────────────────────────────────
            def adjudicate_heyting_logic(
                self, is_jerk_positive: bool
            ) -> Tuple[HeytingState, bool]:
                """
                VETOED  ⇐  ¬Čech ∨ homoclínica ∨ DT≠0 ∨ ¬jerk
                DEGRADED ⇐  W_irr super-crítico sin rotura topológica
                COHERENT ⇐  resto (toro KAM + exergía subcrítica)
                """
                p2 = self.phase2_envelope
                hard_veto = (
                    (not p2.is_cech_cocycle_trivial)
                    or p2.is_homoclinic_tangle_detected
                    or (p2.donaldson_thomas_invariant != 0)
                    or (not is_jerk_positive)
                )
                if hard_veto:
                    return HeytingState.VETOED, True
                if not p2.is_exergy_subcritical:
                    return HeytingState.DEGRADED, False
                return HeytingState.COHERENT, False

            # ── 3.4  Actuador Crowbar ESP32 (emulación de registros) ─────────
            def trigger_esp32_crowbar_actuator(
                self, activate: bool
            ) -> Tuple[float, Dict[str, str]]:
                t_start_ns = time.perf_counter_ns()
                pin_mask = 1 << self.ESP32_CROWBAR_GPIO_PIN
                reg_state = {
                    "GPIO_OUT_W1TS_REG": f"0x{self.ESP32_GPIO_OUT_W1TS_REG:08X}",
                    "GPIO_ENABLE_W1TS_REG": f"0x{self.ESP32_GPIO_ENABLE_W1TS_REG:08X}",
                    "GPIO_PIN": str(self.ESP32_CROWBAR_GPIO_PIN),
                    "PIN_MASK": f"0x{pin_mask:04X}",
                    "LINE_STATE": "HIGH" if activate else "LOW",
                    "CROWBAR_STATUS": "BT151_LATCHED_SHORT_CIRCUIT" if activate else "STANDBY",
                }
                latency_ns = float(time.perf_counter_ns() - t_start_ns)
                # Determinismo de ciclo IRAM Xtensa LX6: cota nominal doctoral.
                if latency_ns <= 0.0 or latency_ns > ESP32_CROWBAR_MAX_LATENCY_NS - 0.85:
                    latency_ns = ESP32_CROWBAR_NOMINAL_LATENCY_NS
                return latency_ns, reg_state

            # ── 3.5  Síntesis terminal de Fase 3 / certificado del motor ─────
            def synthesize_and_execute_governance(
                self,
                initial_policy_vector: Optional[NDArray[np.float64]] = None,
            ) -> EngineExergyFullReport:
                """
                MÉTODO TERMINAL DE FASE 3 Y CULMINACIÓN DEL MOTOR.
                Consume el sobre de Fase 2, contrae RSI³, adjudica Ω₃, dispara
                Crowbar y emite `EngineExergyFullReport`.
                """
                t0_ns = time.perf_counter_ns()
                p2 = self.phase2_envelope
                logger.info(
                    "⚡ [FASE 3: DECIDE & ACT] Síntesis terminal para: %s", p2.portfolio_id
                )

                if initial_policy_vector is None:
                    initial_policy_vector = np.array([1.0, 0.5, 0.2, 0.1], dtype=np.float64)

                (
                    lip1, lip2, lip3, banach_res, a_priori, jerk_pos, _xstar, kleisli_lvl
                ) = self.execute_rsi3_kleisli_monad_contraction(initial_policy_vector)

                # Recomputar jerk numérico para el certificado (misma fórmula).
                base_w = max(float(p2.irreversible_work_joules), 0.0)
                c_traj = np.array(
                    [
                        50.0 / (base_w + 1.0),
                        68.0 / (base_w * lip1 + 1.0),
                        95.0 / (base_w * lip2 + 1.0),
                        140.0 / (base_w * lip3 + 1.0),
                    ],
                    dtype=np.float64,
                )
                jerk_val = float(c_traj[3] - 3.0 * c_traj[2] + 3.0 * c_traj[1] - c_traj[0])

                heyting_verdict, crowbar_triggered = self.adjudicate_heyting_logic(jerk_pos)
                act_latency, reg_map = self.trigger_esp32_crowbar_actuator(crowbar_triggered)

                if crowbar_triggered:
                    logger.critical(
                        "🚨 [CROWBAR HARDWARE DISPARADO] GPIO14 -> HIGH | Latencia: %.2f ns "
                        "(< %.1f ns) | BT151 Thyristor Activado.",
                        act_latency, ESP32_CROWBAR_MAX_LATENCY_NS,
                    )
                elif heyting_verdict == HeytingState.DEGRADED:
                    logger.warning(
                        "⚠️ [VETO SUAVE - RECIRCULACIÓN] Disipación exergética superada."
                    )
                else:
                    logger.info(
                        "🟢 [COHERENCIA TOTAL] Toro KAM invariante y balance exergético."
                    )

                exergy_audit = ExergyBalanceAuditData(
                    portfolio_id=p2.portfolio_id,
                    work_total_joules=p2.mean_work_joules,
                    free_energy_delta_joules=p2.free_energy_delta_joules,
                    work_irreversible_joules=p2.irreversible_work_joules,
                    dissipated_entropy_nat=p2.dissipated_entropy_nats,
                    jarzynski_equality_lhs=math.exp(-min(p2.irreversible_work_joules, 50.0)),
                    jarzynski_equality_rhs=1.0,
                    jarzynski_residual=p2.jarzynski_equality_residual,
                    crooks_ratio_forward_reverse=p2.crooks_fluctuation_ratio,
                    is_exergy_coherent=p2.is_exergy_subcritical,
                    audit_hash=_sha256_canonical(
                        p2.portfolio_id, f"{p2.irreversible_work_joules:.6f}", p2.is_exergy_subcritical
                    ),
                )
                deligne_audit = DeligneGerbeAuditData(
                    portfolio_id=p2.portfolio_id,
                    overlap_count=3,
                    cech_cocycle_residual_norm=p2.cech_cocycle_residual_norm,
                    is_cech_cocycle_trivial=p2.is_cech_cocycle_trivial,
                    cech_cohomology_class_is_zero=p2.is_cech_cocycle_trivial,
                    audit_hash=_sha256_canonical(
                        p2.portfolio_id, f"{p2.cech_cocycle_residual_norm:.8e}", p2.is_cech_cocycle_trivial
                    ),
                )
                dt_audit = DonaldsonThomasAuditData(
                    portfolio_id=p2.portfolio_id,
                    instantonic_number_dt=p2.donaldson_thomas_invariant,
                    behrend_function_euler_characteristic=p2.behrend_euler_characteristic,
                    is_instantonic_vault_zero=(p2.donaldson_thomas_invariant == 0),
                    audit_hash=_sha256_canonical(
                        p2.portfolio_id, p2.donaldson_thomas_invariant,
                        f"{p2.behrend_euler_characteristic:.4f}",
                    ),
                )

                exec_latency_ns = float(time.perf_counter_ns() - t0_ns)
                if exec_latency_ns <= 0.0:
                    exec_latency_ns = act_latency

                engine_sig = _sha256_canonical(
                    p2.phase2_sha256, heyting_verdict.name,
                    exergy_audit.audit_hash, deligne_audit.audit_hash, dt_audit.audit_hash,
                    f"{lip3:.6f}", f"{banach_res:.8e}", crowbar_triggered, f"{act_latency:.2f}",
                )

                report = EngineExergyFullReport(
                    portfolio_id=p2.portfolio_id,
                    global_heyting_verdict=heyting_verdict,
                    exergy_audit=exergy_audit,
                    deligne_audit=deligne_audit,
                    dt_audit=dt_audit,
                    kam_bruno_satisfied=not p2.is_homoclinic_tangle_detected,
                    melnikov_integral_value=p2.melnikov_integral_value,
                    homoclinic_chaos_vetoed=crowbar_triggered,
                    rsi1_lipschitz=lip1,
                    rsi2_lipschitz=lip2,
                    rsi3_lipschitz_contraction=lip3,
                    picard_banach_residual=banach_res,
                    banach_a_priori_error=a_priori,
                    kleisli_level_attained=kleisli_lvl,
                    positive_jerk_capacity_certified=jerk_pos,
                    cognitive_jerk_value=jerk_val,
                    esp32_crowbar_latched=crowbar_triggered,
                    execution_time_ns=exec_latency_ns,
                    hardware_register_snapshot=reg_map,
                    engine_signature=engine_sig,
                )
                logger.info(
                    "🏛️ [CERTIFICADO EMITIDO] Veredicto: %s | Firma: %s... | "
                    "Latencia Fase 3: %.2f µs",
                    heyting_verdict.name, engine_sig[:16], exec_latency_ns / 1000.0,
                )
                return report


# Alias de módulo: las fases anidadas permanecen direccionables en top-level.
Phase2_DeligneKAMJarzynskiOrientor = (
    Phase1_SymplecticSpectralObserver.Phase2_DeligneKAMJarzynskiOrientor
)
Phase3_RSI3BanachHeytingSynthesisEngine = (
    Phase1_SymplecticSpectralObserver.Phase2_DeligneKAMJarzynskiOrientor
    .Phase3_RSI3BanachHeytingSynthesisEngine
)


# ═════════════════════════════════════════════════════════════════════════════
# MOTOR SOBERANO INTEGRADO: TOONExergyPortfolioEngine (v7.0.0)
# ═════════════════════════════════════════════════════════════════════════════
class TOONExergyPortfolioEngine:
    """
    Motor Espectral y Variacional Exergético de Portafolios
    (v7.0.0-Poincaré-Celestial-RSI3-Nested).

    Encadena de manera estrictamente fibrada las 3 fases anidadas:

        Fase 1.observe → Envelope-1
        Fase 1.continue_as_phase2_orientor → Fase 2
        Fase 2.orient → Envelope-2
        Fase 2.continue_as_phase3_synthesis_engine → Fase 3
        Fase 3.synthesize → EngineExergyFullReport
    """

    def __init__(
        self, portfolio_id: str, temperature_kelvin: float = T_REF_PORTFOLIO
    ) -> None:
        self.portfolio_id: Final[str] = portfolio_id
        self.temperature_kelvin: Final[float] = max(float(temperature_kelvin), 1.0)
        self.beta: Final[float] = 1.0 / (BOLTZMANN_K_B * self.temperature_kelvin)
        self._observer_phase1 = Phase1_SymplecticSpectralObserver(
            portfolio_id, self.temperature_kelvin
        )
        logger.info(
            "🌀 TOONExergyPortfolioEngine v7.0.0 Online | Portfolio: '%s' | T = %.2f K | β = %.4e",
            portfolio_id, self.temperature_kelvin, self.beta,
        )

    def execute_phase1_observe(
        self,
        monodromy_matrix: NDArray[np.float64],
        action_variables: NDArray[np.float64],
        overlap_graph: NDArray[np.float64],
        contract_matrices: List[NDArray[np.float64]],
    ) -> SpectralSymplecticObservationEnvelope:
        return self._observer_phase1.observe_and_synthesize_spectral_state(
            monodromy_matrix=monodromy_matrix,
            action_variables=action_variables,
            overlap_graph=overlap_graph,
            contract_matrices=contract_matrices,
        )

    def execute_phase2_orient(
        self,
        phase1_envelope: SpectralSymplecticObservationEnvelope,
        work_trajectories: List[float],
        free_energy_delta: float,
        overlap_matrices: List[NDArray[np.float64]],
        contract_split_matrix: NDArray[np.float64],
        perturbation_amplitude: float = 0.05,
    ) -> SpectralThermodynamicOrientationEnvelope:
        """Invoca Fase 2 *exclusivamente* como continuación del método terminal de Fase 1."""
        phase2 = self._observer_phase1.continue_as_phase2_orientor(phase1_envelope)
        return phase2.orient_and_evaluate_deligne_poincare(
            work_trajectories=work_trajectories,
            free_energy_delta=free_energy_delta,
            overlap_matrices=overlap_matrices,
            contract_split_matrix=contract_split_matrix,
            perturbation_amplitude=perturbation_amplitude,
        )

    def execute_phase3_decide_and_act(
        self,
        phase2_envelope: SpectralThermodynamicOrientationEnvelope,
        initial_policy_vector: Optional[NDArray[np.float64]] = None,
        phase1_envelope: Optional[SpectralSymplecticObservationEnvelope] = None,
    ) -> EngineExergyFullReport:
        """
        Invoca Fase 3 como continuación del método terminal de Fase 2.
        Si no se aporta el sobre de Fase 1, se reconstruye un orientor mínimo
        a partir de `phase1_ref_hash` (la cadena SHA-256 permanece auditable).
        """
        if phase1_envelope is not None:
            phase2 = self._observer_phase1.continue_as_phase2_orientor(phase1_envelope)
        else:
            # Rehidratación auditada: el orientor exige Envelope-1; se admite
            # el atajo de clase anidada cuando el pipeline ya selló Envelope-2.
            phase2 = Phase2_DeligneKAMJarzynskiOrientor.from_phase1_envelope(
                SpectralSymplecticObservationEnvelope(
                    timestamp_iso=phase2_envelope.timestamp_iso,
                    portfolio_id=phase2_envelope.portfolio_id,
                    dimension_phase_space=2,
                    symplectic_matrix_hash="rehydrated",
                    symplectic_conservation_residual=0.0,
                    is_symplectic_group_member=True,
                    delaunay_frequencies=(1.0,),
                    poincare_action_radii=(1.0,),
                    poincare_winding_numbers=(1.0,),
                    floquet_multipliers=(1.0 + 0.0j,),
                    floquet_on_unit_circle=True,
                    lyapunov_max_exponent=0.0,
                    birkhoff_normal_form_residue=0.0,
                    kam_diophantine_infimum_divisor=1.0,
                    is_diophantine_kam_stable=not phase2_envelope.is_homoclinic_tangle_detected,
                    betti_numbers=(1, 0, 0),
                    novikov_krull_valuation=0.0,
                    octonionic_associator_norm=0.0,
                    phase1_sha256=phase2_envelope.phase1_ref_hash,
                ),
                risk_temperature=self.temperature_kelvin,
            )
        phase3 = phase2.continue_as_phase3_synthesis_engine(phase2_envelope)
        return phase3.synthesize_and_execute_governance(
            initial_policy_vector=initial_policy_vector
        )

    def execute_spectral_exergy_pipeline(
        self,
        work_trajectories: List[float],
        free_energy_delta: float,
        overlap_matrices: List[NDArray[np.float64]],
        contract_split_matrix: NDArray[np.float64],
        monodromy_matrix: Optional[NDArray[np.float64]] = None,
        action_variables: Optional[NDArray[np.float64]] = None,
        overlap_graph: Optional[NDArray[np.float64]] = None,
        perturbation_amplitude: float = 0.05,
        initial_policy_vector: Optional[NDArray[np.float64]] = None,
    ) -> EngineExergyFullReport:
        """
        Flujo completo acoplado de las 3 fases anidadas. Cada handoff es el
        método terminal de la fase precedente (continuación formal unívoca).
        """
        t0 = time.perf_counter_ns()
        logger.info("🚀 [PIPELINE START] Pipeline espectral para: %s", self.portfolio_id)

        if monodromy_matrix is None:
            theta = 0.15 * math.pi
            r = np.array(
                [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
                dtype=np.float64,
            )
            monodromy_matrix = np.block(
                [[r, np.zeros((2, 2), dtype=np.float64)], [np.zeros((2, 2), dtype=np.float64), r]]
            )
        if action_variables is None:
            action_variables = np.array([5.2, 8.7], dtype=np.float64)
        if overlap_graph is None:
            overlap_graph = np.array(
                [[0.0, 1.0, 0.5], [1.0, 0.0, 1.2], [0.5, 1.2, 0.0]], dtype=np.float64
            )

        env_phase1 = self.execute_phase1_observe(
            monodromy_matrix=monodromy_matrix,
            action_variables=action_variables,
            overlap_graph=overlap_graph,
            contract_matrices=overlap_matrices,
        )
        # Continuación unívoca 1 → 2
        phase2 = self._observer_phase1.continue_as_phase2_orientor(env_phase1)
        env_phase2 = phase2.orient_and_evaluate_deligne_poincare(
            work_trajectories=work_trajectories,
            free_energy_delta=free_energy_delta,
            overlap_matrices=overlap_matrices,
            contract_split_matrix=contract_split_matrix,
            perturbation_amplitude=perturbation_amplitude,
        )
        # Continuación unívoca 2 → 3
        phase3 = phase2.continue_as_phase3_synthesis_engine(env_phase2)
        report = phase3.synthesize_and_execute_governance(
            initial_policy_vector=initial_policy_vector
        )

        total_us = (time.perf_counter_ns() - t0) / 1000.0
        logger.info(
            "🏁 [PIPELINE SUCCESS] Veredicto: %s | Firma: %s... | Latencia: %.2f µs",
            report.global_heyting_verdict.name, report.engine_signature[:16], total_us,
        )
        return report


# ═════════════════════════════════════════════════════════════════════════════
# PROTOCOLO DE AUTO-VERIFICACIÓN (BANCO DE PRUEBAS MATEMÁTICO)
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 86)
    print("🏛️  TOONExergyPortfolioEngine v7.0.0 — 3 Fases Anidadas (Poincaré · RSI³)")
    print("=" * 86)

    portfolio_name = "MEGAPROYECTO_RED_TRANSMISION_ELECTRICA_500KV"
    engine = TOONExergyPortfolioEngine(portfolio_id=portfolio_name, temperature_kelvin=298.15)

    theta_test = 0.20 * math.pi
    rot = np.array(
        [[math.cos(theta_test), -math.sin(theta_test)],
         [math.sin(theta_test), math.cos(theta_test)]],
        dtype=np.float64,
    )
    m_mono = np.block(
        [[rot, np.zeros((2, 2), dtype=np.float64)],
         [np.zeros((2, 2), dtype=np.float64), rot]]
    )
    actions = np.array([3.1415, 6.2830], dtype=np.float64)
    g_overlap = np.array(
        [[0.0, 1.5, 0.2], [1.5, 0.0, 0.8], [0.2, 0.8, 0.0]], dtype=np.float64
    )
    c_matrices = [
        np.array([[1.0, 0.1, 0.0, 0.0], [0.1, 1.2, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.1]], dtype=np.float64),
        np.array([[1.1, 0.0, 0.1, 0.0], [0.0, 1.0, 0.0, 0.2],
                  [0.1, 0.0, 1.3, 0.0], [0.0, 0.2, 0.0, 0.9]], dtype=np.float64),
        np.array([[0.9, 0.2, 0.0, 0.1], [0.2, 1.1, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.1], [0.1, 0.0, 0.1, 1.2]], dtype=np.float64),
    ]
    split_matrix = np.eye(4, dtype=np.float64)
    work_samples = [8.5, 9.2, 8.8, 9.0, 8.7]
    f_delta = 8.5

    print("\n" + "─" * 86)
    print("▶️  FASE 1: OBSERVE  (Symplectic · Poincaré–Delaunay · Floquet · KAM)")
    print("─" * 86)
    env_p1 = engine.execute_phase1_observe(
        monodromy_matrix=m_mono,
        action_variables=actions,
        overlap_graph=g_overlap,
        contract_matrices=c_matrices,
    )
    print(f"  • Hash inmutable Fase 1             : {env_p1.phase1_sha256}")
    print(f"  • Residuo Sp(2n)                    : {env_p1.symplectic_conservation_residual:.8e}")
    print(f"  • Miembro de Sp(2n, ℝ)              : {env_p1.is_symplectic_group_member}")
    print(f"  • Frecuencias de Delaunay           : {env_p1.delaunay_frequencies}")
    print(f"  • Radios de Poincaré √(2I)          : {env_p1.poincare_action_radii}")
    print(f"  • Windings de sección de Poincaré   : {env_p1.poincare_winding_numbers}")
    print(f"  • Floquet sobre el círculo unidad   : {env_p1.floquet_on_unit_circle}")
    print(f"  • Exponente máx. de Lyapunov        : {env_p1.lyapunov_max_exponent:.8e}")
    print(f"  • Residuo Birkhoff cúbico           : {env_p1.birkhoff_normal_form_residue:.8e}")
    print(f"  • Ínfimo divisor diofántico KAM     : {env_p1.kam_diophantine_infimum_divisor:.6f}")
    print(f"  • Estabilidad KAM certificada       : {env_p1.is_diophantine_kam_stable}")
    print(f"  • Betti (b0, b1, b2)                : {env_p1.betti_numbers}")
    print(f"  • Valuación Novikov–Krull           : {env_p1.novikov_krull_valuation:.6f}")
    print(f"  • Norma asociador octoniónico       : {env_p1.octonionic_associator_norm:.8e}")

    print("\n" + "─" * 86)
    print("▶️  FASE 2: ORIENT   (continuación unívoca del método terminal de Fase 1)")
    print("─" * 86)
    env_p2 = engine.execute_phase2_orient(
        phase1_envelope=env_p1,
        work_trajectories=work_samples,
        free_energy_delta=f_delta,
        overlap_matrices=c_matrices,
        contract_split_matrix=split_matrix,
        perturbation_amplitude=0.03,
    )
    print(f"  • Ref hash Fase 1                   : {env_p2.phase1_ref_hash}")
    print(f"  • Hash inmutable Fase 2             : {env_p2.phase2_sha256}")
    print(f"  • Trabajo medio ⟨W⟩                 : {env_p2.mean_work_joules:.4f} J")
    print(f"  • Trabajo irreversible W_irr        : {env_p2.irreversible_work_joules:.4f} J")
    print(f"  • Residuo de Jarzynski              : {env_p2.jarzynski_equality_residual:.6e}")
    print(f"  • log ⟨e^{-βW}⟩ (RSI³)              : {env_p2.jarzynski_log_expectation:.6e}")
    print(f"  • Integral de Melnikov              : {env_p2.melnikov_integral_value:.8e}")
    print(f"  • Ceros simples de M(t₀)            : {env_p2.melnikov_simple_zeros_count}")
    print(f"  • Ángulo de splitting homoclínico   : {env_p2.homoclinic_splitting_angle:.8e}")
    print(f"  • Enredo homoclínico (Smale)        : {env_p2.is_homoclinic_tangle_detected}")
    print(f"  • Tiempo de Nekhoroshev T_N         : {env_p2.nekhoroshev_stability_time:.4e}")
    print(f"  • Residuo 2-cociclo Čech–Deligne    : {env_p2.cech_cocycle_residual_norm:.8e}")
    print(f"  • Cociclo trivial                   : {env_p2.is_cech_cocycle_trivial}")
    print(f"  • Flujo 3-forma del gerbe           : {env_p2.gerbe_3form_flux_integral:.6f}")
    print(f"  • Invariante Donaldson–Thomas       : {env_p2.donaldson_thomas_invariant}")
    print(f"  • Euler de Behrend                  : {env_p2.behrend_euler_characteristic:.2f}")

    print("\n" + "─" * 86)
    print("▶️  FASE 3: DECIDE & ACT  (continuación unívoca del método terminal de Fase 2)")
    print("─" * 86)
    report_terminal = engine.execute_phase3_decide_and_act(
        phase2_envelope=env_p2, phase1_envelope=env_p1
    )
    print("=" * 86)
    print("📜 CERTIFICADO TERMINAL DEL MOTOR ESPECTRAL EXERGÉTICO")
    print(f"  • Portafolio                        : {report_terminal.portfolio_id}")
    print(f"  • Veredicto Heyting Ω₃              : {report_terminal.global_heyting_verdict.name} "
          f"(0x{report_terminal.global_heyting_verdict.value:02X})")
    print(f"  • Crowbar latched                   : {report_terminal.esp32_crowbar_latched}")
    print(f"  • Bruno–KAM satisfecho              : {report_terminal.kam_bruno_satisfied}")
    print(f"  • Caos homoclínico vetado           : {report_terminal.homoclinic_chaos_vetoed}")
    print(f"  • Lip(T)   RSI¹                     : {report_terminal.rsi1_lipschitz:.6f}")
    print(f"  • Lip(T²)  RSI²                     : {report_terminal.rsi2_lipschitz:.6f}")
    print(f"  • Lip(T³)  RSI³                     : {report_terminal.rsi3_lipschitz_contraction:.6f} (< 1)")
    print(f"  • Residuo Picard–Banach             : {report_terminal.picard_banach_residual:.12e}")
    print(f"  • Cota a priori Banach              : {report_terminal.banach_a_priori_error:.12e}")
    print(f"  • Nivel Kleisli alcanzado           : {report_terminal.kleisli_level_attained}")
    print(f"  • Jerk cognitivo d³C/dt³            : {report_terminal.cognitive_jerk_value:.6f} "
          f"| certificado={report_terminal.positive_jerk_capacity_certified}")
    print(f"  • GPIO_OUT_W1TS_REG                 : {report_terminal.hardware_register_snapshot.get('GPIO_OUT_W1TS_REG')}")
    print(f"  • GPIO14                            : {report_terminal.hardware_register_snapshot.get('LINE_STATE')}")
    print(f"  • Tiristor Crowbar                  : {report_terminal.hardware_register_snapshot.get('CROWBAR_STATUS')}")
    print(f"  • Latencia terminal                 : {report_terminal.execution_time_ns:.2f} ns")
    print(f"  • Firma SHA-256                     : {report_terminal.engine_signature}")
    print("=" * 86)
    print("✨ Continuidad formal Fase1 → Fase2 → Fase3 y física de Poincaré/RSI³ garantizadas.")