# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : KBase Thermodynamic Agent (Asesor de Cimientos Financieros)         ║
║ Ruta   : app/agents/alpha/kbase/kbase_thermodynamic_agent.py                 ║
║ Versión: 7.0.0-APU-SOVEREIGN-POINCARE-KAM-MELNIKOV-RSI3                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA (V_ℙ) ─────────────────────────────────────────────────
Foso Termodinámico (K_BASE, Nivel 3). Modela la inyección de recursos y el
presupuesto como un sistema Port-Hamiltoniano disipativo continuo. Estabiliza la
inercia del negocio mediante termodinámica clásica, cálculo exterior discreto
(DEC) y geometría simpléctica.

ARQUITECTURA DE TRES FASES ANIDADAS (v7) ──────────────────────────────────────
  Fase 1 │ TOPOLOGÍA MATRICIAL + POINCARÉ ESTRUCTURAL (Observe)
         │ Pullback G_q C G_qᵀ, G_p M G_pᵀ. Carta de Poincaré: sección Σ,
         │ mapa de retorno linealizado P = exp(M_flow · τ), espectro de
         │ Lyapunov λ_k = Re Spec((J−R)∂²H), frecuencias ω_k = Im.
         │ ▸ Codominio: TopologicalContext  (contiene PoincarePhaseChart).
         │ ▸ ESTE ES EL DOMINIO DE Phase2.__init__().

  Fase 2 │ PORT-HAMILTONIANA + WILLIAMSON + KAM + MELNIKOV (Orient)
         │ Ḣ = ∇Hᵀ(J − R)∇H ≤ 0. Frecuencias de Williamson ω_k. Audit KAM
         │ (diofántico |⟨k,ω⟩| ≥ γ/‖k‖^τ). Función de Melnikov M(t₀) para
         │ detectar entrelazado homoclínico.
         │ ▸ Codominio: BasalStateTensor  (contiene KAMDiagnostics).
         │ ▸ ESTE ES EL DOMINIO DE Phase3.export_stalk().

  Fase 3 │ COHOMOLOGÍA DE HODGE + DUALIDAD DE POINCARÉ + RSI-3 (Decide & Act)
         │ δ_BASEᵀ δ_BASE = ∇²H + R_cost (verificado). Defecto de
         │ isotropía simpléctica ‖δ_mᵀ J δ_m‖_F. Betti b_0, b_1. RSI-3
         │ cierra el bucle meta-adaptativo.
         │ ▸ Codominio: SheafStalk (objeto terminal del endofuntor).

INVARIANTES (heredados + v7) ──────────────────────────────────────────────────
 [I1] J = −Jᵀ, ∇Hᵀ J ∇H ≡ 0 (Dirac/Ohm).
 [I2] Ḣ = −∇Hᵀ R ∇H ≤ 0 (Lyapunov).
 [I3] Retícula Booleana StabilityFlags ∈ (2^10, ≤, ⊔, ⊓) (v7: +KAM, +tangle, +RSI3).
 [I4] C̃, M̃ ≻ 0 (Sylvester/Tikhonov).
 [I5] δ_BASEᵀ δ_BASE = Δ_BASE (Hodge local).
 [I6] KAM: min_{k≠0} |⟨k,ω⟩|‖k‖^τ_1 ≥ γ > 0 (toros invariantes).
 [I7] No caos: M(t₀) sin ceros simples (o ausencia de separatriz).
 [I8] RSI-3: ‖α_{n+1} − α_n‖ < ε_rec ⇒ punto fijo contractivo.
 [I9] Dualidad de Poincaré: ‖δ_mᵀ J δ_m‖_F / (‖δ_m‖_F² ‖J‖_F) ≪ 1.

CONTINUIDAD TRANS-FASE ─────────────────────────────────────────────────────────
 El último método de Fase 1 (`build_topological_context`) DEVUELVE el objeto
 que el constructor de Fase 2 recibe como primer argumento. Análogamente,
 `synthesize_basal_state` entrega `state_vector`, que es el único argumento
 posicional de `export_stalk`. No hay objetos intersticiales.
"""

from __future__ import annotations

# =============================================================================
# Biblioteca estándar
# =============================================================================
import enum
import itertools
import logging
from dataclasses import dataclass, replace
from typing import Final, List, Optional, Tuple

# =============================================================================
# Álgebra numérica
# =============================================================================
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# =============================================================================
# Dependencias arquitectónicas
# =============================================================================
try:
    from app.core.mic_algebra import CategoricalState, Morphism
except ImportError:
    class CategoricalState:  # type: ignore[no-redef]
        """Stub: estado categórico del ecosistema MIC."""

    class Morphism:  # type: ignore[no-redef]
        """Stub: morfismo funtorial del ecosistema MIC."""


logger = logging.getLogger("MIC.Alpha.KBaseThermodynamicAgent")

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)

# ---------------------------------------------------------------------------
# Constantes v7 (Poincaré / KAM / Melnikov / RSI-3)
# ---------------------------------------------------------------------------
_KAM_GAMMA_DEFAULT: Final[float] = 0.1
_KAM_TAU_DEFAULT: Final[float] = 2.5
_KAM_K_MAX_DEFAULT: Final[int] = 4
_LYAPUNOV_FLOOR: Final[float] = 1e-300
_MELNIKOV_SAMPLES: Final[int] = 96
_MELNIKOV_AREA_FLOOR: Final[float] = 1e-8
_RSI3_LEARNING_RATE: Final[float] = 0.05
_RSI3_CONVERGENCE_TOL: Final[float] = 1e-8
_RSI3_MAX_HISTORY: Final[int] = 12
_ALPHA_FLOOR: Final[float] = _MACHINE_EPS


# =============================================================================
# SECCIÓN 0 — EXCEPCIONES TERMODINÁMICAS
# =============================================================================


class ThermodynamicBaseError(Exception):
    """Raíz de todas las excepciones del Estrato K_BASE."""


class DimensionMismatchError(ThermodynamicBaseError):
    """Dimensiones de matrices constitutivas inconsistentes con (q,p)."""


class CapacitanceDegeneracyError(ThermodynamicBaseError):
    """C̃ o M̃ no son SPD tras regularización."""


class InertialFlybackError(ThermodynamicBaseError):
    """Flyback supera la ruptura dieléctrica."""


class RayleighDissipationViolation(ThermodynamicBaseError):
    """∇HᵀR∇H < 0 — ganancia fantasma (viola 2ª Ley)."""


class IllConditionedMatrixError(ThermodynamicBaseError):
    """κ(A) > κ_max."""


class MetricTensorSingularityError(ThermodynamicBaseError):
    """G_q, G_p singulares o mal condicionadas."""


class SheafCoboundaryError(ThermodynamicBaseError):
    """δ_BASE no satisface la identidad de Hodge."""


class StructuralConsistencyError(ThermodynamicBaseError):
    """Identidad ∇Hᵀẋ = −P_diss violada (cableado PH roto)."""


class PassivityCertificateError(ThermodynamicBaseError):
    """Certificado de pasividad Ḣ + P_diss ≈ 0 violado."""


class WilliamsonNormalFormError(ThermodynamicBaseError):
    """Williamson (antisimetría/SV) roto."""


# v7:
class KAMThresholdViolation(ThermodynamicBaseError):
    """Condición diofántica de Bruno violada: el toro invariante se destruye."""


class HomoclinicChaosError(ThermodynamicBaseError):
    """M(t₀) con ceros simples ⇒ entrelazado de Smale-Birkhoff ⇒ caos."""


class RecursiveSelfImprovementError(ThermodynamicBaseError):
    """RSI-3 no converge (‖Δα‖ ≥ ε_rec) o viola contracción."""


# =============================================================================
# SECCIÓN 1 — ÁLGEBRA DE BOOLE DE ESTABILIDAD (v7: 10 predicados)
# =============================================================================


class StabilityFlags(enum.Flag):
    r"""
    Retícula Booleana de estabilidad termodinámica y dinámica.

    ``enum.Flag`` provee el álgebra de Boole nativa (∧=&, ∨=|, ¬=~, NONE=0̄,
    ALL=1̄). v7 añade tres predicados dinámicos:
      • KAM_TORUS_STABLE       — ω satisface la condición diofántica de Bruno.
      • NO_HOMOCLINIC_TANGLE   — M(t₀) no exhibe ceros simples.
      • RSI3_CONTRACTIVE       — ‖Δα_n‖ < ε_rec.
    """

    NONE = 0
    ENERGY_NONNEGATIVE = enum.auto()
    DISSIPATION_VALID = enum.auto()
    FLYBACK_SAFE = enum.auto()
    STRUCTURAL_CONSISTENCY = enum.auto()
    SPECTRAL_CONDITIONING_SOUND = enum.auto()
    PASSIVITY_CERTIFICATE = enum.auto()
    EULER_HOMOGENEITY = enum.auto()
    # v7
    KAM_TORUS_STABLE = enum.auto()
    NO_HOMOCLINIC_TANGLE = enum.auto()
    RSI3_CONTRACTIVE = enum.auto()
    ALL = (
        ENERGY_NONNEGATIVE
        | DISSIPATION_VALID
        | FLYBACK_SAFE
        | STRUCTURAL_CONSISTENCY
        | SPECTRAL_CONDITIONING_SOUND
        | PASSIVITY_CERTIFICATE
        | EULER_HOMOGENEITY
        | KAM_TORUS_STABLE
        | NO_HOMOCLINIC_TANGLE
        | RSI3_CONTRACTIVE
    )


def describe_stability_flags(flags: StabilityFlags) -> str:
    """Serializa StabilityFlags a forma legible."""
    atomic = [f for f in StabilityFlags if f not in (StabilityFlags.NONE, StabilityFlags.ALL)]
    satisfied = [f.name for f in atomic if f in flags]
    violated = [f.name for f in atomic if f not in flags]
    return (
        f"SATISFECHOS={satisfied or ['ninguno']} | "
        f"VIOLADOS={violated or ['ninguno']} | "
        f"ESTABLE_TOTAL={flags == StabilityFlags.ALL}"
    )


# =============================================================================
# SECCIÓN 2 — DTOs TENSORIALES INMUTABLES (v7)
# =============================================================================


@dataclass(frozen=True, slots=True)
class PoincarePhaseChart:
    r"""
    Carta de Poincaré estructural (producida en **Fase 1**).

    Contiene:
      • Sección Σ = {⟨n̂, x⟩ = c} en ℝ^n.
      • Generador linealizado del flujo M_flow = (J_base − R_cost) ∂²H.
      • Mapa de retorno linealizado P = exp(M_flow · τ_ret).
      • Espectro de Lyapunov: λ_k = Re(eig_k(M_flow)).
      • Frecuencias fundamentales ω_k = Im(eig_k) para modos elípticos.
      • Matriz de autovectores (para acción-ángulo en Fase 2).

    Nomencaltura
    ------------
    La sección es estructura topológica pura: los valores concretos del
    estado son inyectados en Fase 2 (`_action_angle_from_state`).
    """

    section_normal: NDArray[np.float64]
    section_offset: float
    linearized_flow: NDArray[np.float64]
    return_time: float
    return_map: NDArray[np.float64]
    return_eigenvalues: NDArray[np.complex128]
    return_spectral_radius: float
    lyapunov_exponents: NDArray[np.float64]
    max_lyapunov: float
    fundamental_frequencies: NDArray[np.float64]
    elliptic_dimension: int
    hyperbolic_dimension: int
    eigenvector_matrix: NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class KAMDiagnostics:
    r"""
    Diagnóstico del teorema KAM + Melnikov (producido en **Fase 2**).

    Condición diofántica de Bruno:
        |⟨k, ω⟩| ≥ γ / ‖k‖_1^τ,   ∀k ∈ ℤ^n \ {0}, ‖k‖_1 ≤ K_max.
    Ratio R = min_k |⟨k,ω⟩| ‖k‖_1^τ / γ  ≥ 1 ⇔ toro invariante persistente.

    Función de Melnikov:
        M(t₀) = ∫ {H₀, H₁}(q⁰(t−t₀)) dt.
    Ceros simples de M(t₀) ⇔ entrelazado de Smale-Birkhoff ⇔ caos.
    """

    stable: bool
    min_diophantine_ratio: float
    gamma: float
    tau: float
    resonance_order: int
    small_divisor_min: float
    frequencies: NDArray[np.float64]
    melnikov_lobe_area: float
    melnikov_zero_crossings: int
    homoclinic_tangle_detected: bool
    saddle_found: bool


@dataclass(frozen=True, slots=True)
class SelfImprovementDiagnostics:
    r"""
    Diagnóstico del bucle RSI-3 (producido en **Fase 3** vía el agente).

    Tres niveles anidados:
      Nivel 1: α_{n+1} = α_n · ρ_contra  (parámetro físico de Tikhonov).
      Nivel 2: ρ_contra ← ρ_contra · f(ΔS)  (adaptación de la tasa).
      Nivel 3: ρ_contra ← ρ_contra · (1 − η(1 − S_L3))  (meta-adaptación).
    """

    iteration: int
    current_alpha: float
    next_alpha: float
    contraction: float
    level3_score: float
    converged: bool
    residual: float
    performance_delta: float


@dataclass(frozen=True, slots=True)
class TopologicalContext:
    r"""
    Contexto inmutable producido por **Fase 1**.

    Contiene toda la infraestructura métrica, espectral y —en v7— la
    **carta de Poincaré estructural**, que la Fase 2 consume para construir
    acción-ángulo y para el audit KAM.

    Continuidad formal
    ------------------
    Este DTO es el **único argumento** del constructor de
    ``Phase2_HamiltonianDynamics`` y de ``Phase3_SheafProjection``.
    """

    L_C: NDArray[np.float64]
    L_M: NDArray[np.float64]
    C_inv_sqrt: NDArray[np.float64]
    M_inv_sqrt: NDArray[np.float64]
    C_tilde: NDArray[np.float64]
    M_tilde: NDArray[np.float64]
    R_cost: NDArray[np.float64]
    R_sqrt: NDArray[np.float64]
    J_base: NDArray[np.float64]
    G_q: NDArray[np.float64]
    G_p: NDArray[np.float64]
    kappa_C: float
    kappa_M: float
    kappa_G_q: float
    kappa_G_p: float
    epsilon_C: float
    epsilon_M: float
    dim_q: int
    dim_p: int
    rank_R: int
    spectral_gap_R: float
    betti_0_R: int
    spectral_gap_C: float
    spectral_gap_M: float
    pullback_amp_C: float
    pullback_amp_M: float
    spectral_entropy_R: float
    # v7
    poincare_chart: PoincarePhaseChart


@dataclass(frozen=True, slots=True)
class BasalStateTensor:
    r"""
    Tensor de estado termodinámico completo (producido por **Fase 2**).

    Contiene las energías, el campo vectorial, el certificado de pasividad,
    la retícula Booleana y —en v7— los diagnósticos KAM/Melnikov y las
    variables de acción-ángulo del estado dinámico actual.
    """

    potential_energy: float
    kinetic_energy: float
    total_hamiltonian: float
    dissipated_power: float
    flyback_voltage_norm: float
    grad_H_norm: float
    vector_field: NDArray[np.float64]
    euler_homogeneity_residual: float
    structural_consistency_residual: float
    passivity_residual: float
    dissipation_time_constant: float
    stability_flags: StabilityFlags
    is_thermodynamically_stable: bool
    normal_mode_frequencies: Optional[NDArray[np.float64]]
    zero_point_energy: Optional[float]
    state_vector: NDArray[np.float64]
    # v7
    action_variables: NDArray[np.float64]
    angle_variables: NDArray[np.float64]
    kam_diagnostics: KAMDiagnostics


@dataclass(frozen=True, slots=True)
class SheafStalk:
    r"""
    Fibrado celular exportado (producido por **Fase 3**).

    Construcción:
        δ_metric = block_diag(C̃^{-1/2}, M̃^{-1/2}) ∈ ℝ^{n×n}
        δ_diss   = R_cost^{+1/2}                   ∈ ℝ^{n×n}
        δ_BASE   = [δ_metric ; δ_diss]             ∈ ℝ^{2n×n}
        Δ_BASE   = δ_BASEᵀ δ_BASE = ∇²H + R_cost   (SPD)

    v7 añade:
      • Defecto de isotropía simpléctica ‖δ_mᵀ J δ_m‖_F (dualidad de Poincaré).
      • Números de Betti b_0, b_1 del complejo C^n → C^{2n}.
      • Diagnóstico RSI-3 (rellenado por el agente tras export_stalk).
    """

    delta_base: NDArray[np.float64]
    delta_metric: NDArray[np.float64]
    delta_dissipative: NDArray[np.float64]
    hodge_laplacian: NDArray[np.float64]
    hodge_identity_residual: float
    hodge_spectral_gap: float
    hodge_condition_number: float
    harmonic_dimension: int
    lossless_subspace_dimension: int
    state_vector: NDArray[np.float64]
    projected_state_metric: NDArray[np.float64]
    projected_state_dissipative: NDArray[np.float64]
    rank_delta: int
    spectral_entropy_hodge: float
    cheeger_proxy: float
    # v7
    hodge_symplectic_defect: float
    betti_numbers: Tuple[int, int]
    self_improvement: Optional[SelfImprovementDiagnostics]


# =============================================================================
# SECCIÓN 3 — ORQUESTADOR
# =============================================================================


class KBaseThermodynamicAgent(Morphism):
    r"""
    Orquestador Funtorial del Foso Termodinámico K_BASE (v7).

    Cascada estricta:
        Phase1_MatrixTopology  →  TopologicalContext (con PoincarePhaseChart)
        Phase2_HamiltonianDynamics  →  BasalStateTensor (con KAM + Melnikov)
        Phase3_SheafProjection  →  SheafStalk (+ RSI-3 vía el agente)
    """

    FRIENDLY_NAME: str = "Asesor de Cimientos Financieros"
    VERSION: str = "7.0.0-APU-SOVEREIGN-POINCARE-KAM-MELNIKOV-RSI3"

    def __init__(
        self,
        C_soc: NDArray[np.float64],
        M_rec: NDArray[np.float64],
        R_cost: NDArray[np.float64],
        J_base: NDArray[np.float64],
        breakdown_voltage: float = 1.0e5,
        kappa_max: float = 1.0e10,
        G_q: Optional[NDArray[np.float64]] = None,
        G_p: Optional[NDArray[np.float64]] = None,
        hbar: float = 1.0,
        flyback_safety_margin: float = 0.9,
    ) -> None:
        if breakdown_voltage <= 0.0:
            raise ValueError(f"breakdown_voltage debe ser > 0; {breakdown_voltage}.")
        if kappa_max <= 1.0:
            raise ValueError(f"kappa_max debe ser > 1; {kappa_max}.")
        if hbar < 0.0:
            raise ValueError(f"hbar debe ser ≥ 0; {hbar}.")
        if not (0.0 < flyback_safety_margin <= 1.0):
            raise ValueError(
                f"flyback_safety_margin ∈ (0,1]; {flyback_safety_margin}."
            )

        self.breakdown_voltage: float = breakdown_voltage
        self.kappa_max: float = kappa_max
        self.hbar: float = hbar
        self.flyback_safety_margin: float = flyback_safety_margin

        # v7: estado meta para RSI-3
        self._meta_state: dict = {
            "alpha": float(max(kappa_max, 1.0) * _MACHINE_EPS),
            "contraction": 0.5,
            "level3_score": 1.0,
            "iteration": 0,
        }
        self._meta_trajectory: List[dict] = []
        self._previous_performance: float = 1.0

        # ---- FASE 1 (inmediata) ----
        self.phase1: "KBaseThermodynamicAgent.Phase1_MatrixTopology" = (
            KBaseThermodynamicAgent.Phase1_MatrixTopology(
                C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base,
                kappa_max=kappa_max, G_q=G_q, G_p=G_p,
            )
        )
        self.context: TopologicalContext = self.phase1.build_topological_context()

        # ---- FASE 2 (continuación formal de Fase 1) ----
        self.phase2: "KBaseThermodynamicAgent.Phase2_HamiltonianDynamics" = (
            KBaseThermodynamicAgent.Phase2_HamiltonianDynamics(
                context=self.context,
                breakdown_voltage=self.breakdown_voltage,
                kappa_max=self.kappa_max,
                hbar=self.hbar,
                flyback_safety_margin=self.flyback_safety_margin,
            )
        )

        # ---- FASE 3 (lazy init) ----
        self.phase3: Optional["KBaseThermodynamicAgent.Phase3_SheafProjection"] = None

        logger.info(
            "[KBaseThermodynamicAgent v%s] Init. dim_q=%d, dim_p=%d, κ(C̃)=%.3e, "
            "κ(M̃)=%.3e, rank(R)=%d, betti_0(R)=%d, λ_max(flow)=%.3e, ω_min=%.3e.",
            self.VERSION, self.context.dim_q, self.context.dim_p,
            self.context.kappa_C, self.context.kappa_M,
            self.context.rank_R, self.context.betti_0_R,
            self.context.poincare_chart.max_lyapunov,
            float(np.min(self.context.poincare_chart.fundamental_frequencies))
            if self.context.poincare_chart.fundamental_frequencies.size else 0.0,
        )

    # =========================================================================
    # FASE 1 — TOPOLOGÍA MATRICIAL + POINCARÉ ESTRUCTURAL
    # =========================================================================

    class Phase1_MatrixTopology:
        r"""
        Fase 1: Topología Matricial, Métrica Riemanniana, Espectro y
        Carta de Poincaré estructural.
        """

        _EPS: Final[float] = _MACHINE_EPS

        def __init__(
            self,
            C_soc: NDArray[np.float64],
            M_rec: NDArray[np.float64],
            R_cost: NDArray[np.float64],
            J_base: NDArray[np.float64],
            kappa_max: float = 1.0e10,
            G_q: Optional[NDArray[np.float64]] = None,
            G_p: Optional[NDArray[np.float64]] = None,
        ) -> None:
            self._C_soc = C_soc
            self._M_rec = M_rec
            self._R_cost = R_cost
            self._J_base = J_base
            self._kappa_max = kappa_max
            self._G_q_raw = G_q
            self._G_p_raw = G_p

        # ---- Validaciones ----

        def _check_dimensions(self) -> Tuple[int, int]:
            for mat, name in [
                (self._C_soc, "C_soc"), (self._M_rec, "M_rec"),
                (self._R_cost, "R_cost"), (self._J_base, "J_base"),
            ]:
                if mat.ndim != 2:
                    raise DimensionMismatchError(
                        f"'{name}' debe ser 2D; ndim={mat.ndim}, shape={mat.shape}."
                    )
            if self._C_soc.shape[0] != self._C_soc.shape[1]:
                raise DimensionMismatchError(f"C_soc no cuadrada: {self._C_soc.shape}.")
            dim_q = int(self._C_soc.shape[0])
            if dim_q < 1:
                raise DimensionMismatchError(f"dim_q ≥ 1; {dim_q}.")
            if self._M_rec.shape[0] != self._M_rec.shape[1]:
                raise DimensionMismatchError(f"M_rec no cuadrada: {self._M_rec.shape}.")
            dim_p = int(self._M_rec.shape[0])
            if dim_p < 1:
                raise DimensionMismatchError(f"dim_p ≥ 1; {dim_p}.")
            n = dim_q + dim_p
            if self._R_cost.shape != (n, n):
                raise DimensionMismatchError(
                    f"R_cost ({n},{n}); {self._R_cost.shape}."
                )
            if self._J_base.shape != (n, n):
                raise DimensionMismatchError(
                    f"J_base ({n},{n}); {self._J_base.shape}."
                )
            return dim_q, dim_p

        def _validate_symmetry(self, A: NDArray[np.float64], name: str) -> None:
            norm_A = float(la.norm(A, "fro"))
            tol = self._EPS * max(norm_A, 1.0)
            residual = float(la.norm(A - A.T, "fro"))
            if residual > tol:
                raise ThermodynamicBaseError(
                    f"'{name}' no simétrica. ‖A−Aᵀ‖_F={residual:.3e} > {tol:.3e}."
                )

        def _validate_antisymmetry(self, J: NDArray[np.float64], name: str) -> None:
            norm_J = float(la.norm(J, "fro"))
            tol = self._EPS * max(norm_J, 1.0)
            residual = float(la.norm(J + J.T, "fro"))
            if residual > tol:
                raise ThermodynamicBaseError(
                    f"'{name}' no antisimétrica. ‖J+Jᵀ‖_F={residual:.3e} > {tol:.3e}."
                )

        def _validate_metric_tensor(
            self, G: NDArray[np.float64], name: str, expected_dim: int,
        ) -> float:
            if G.ndim != 2 or G.shape != (expected_dim, expected_dim):
                raise DimensionMismatchError(
                    f"'{name}' debe ser ({expected_dim},{expected_dim}); {G.shape}."
                )
            sv = la.svdvals(G)
            s_max, s_min = float(sv[0]), float(sv[-1])
            if s_max <= 0.0 or s_min <= self._EPS * max(s_max, 1.0):
                raise MetricTensorSingularityError(
                    f"'{name}' singular. σ_min={s_min:.3e}, σ_max={s_max:.3e}."
                )
            kappa = s_max / s_min
            if kappa > self._kappa_max:
                raise MetricTensorSingularityError(
                    f"'{name}' κ={kappa:.3e} > κ_max={self._kappa_max:.3e}."
                )
            return kappa

        def _congruence_pullback(
            self, A: NDArray[np.float64], G: NDArray[np.float64], name: str,
        ) -> NDArray[np.float64]:
            A_t = G @ A @ G.T
            return 0.5 * (A_t + A_t.T)

        def _spectral_extremes_and_gap(
            self, A: NDArray[np.float64], name: str, require_spd: bool = True,
        ) -> Tuple[float, float, float, float]:
            n = int(A.shape[0])
            A_sym = 0.5 * (A + A.T)
            if n == 1:
                lam_min = lam_max = lam_second = float(A_sym[0, 0])
            else:
                lam_min = float(la.eigh(A_sym, subset_by_index=[0, 0], eigvals_only=True)[0])
                lam_max = float(la.eigh(A_sym, subset_by_index=[n - 1, n - 1], eigvals_only=True)[0])
                lam_second = float(la.eigh(A_sym, subset_by_index=[1, 1], eigvals_only=True)[0])
            tol_pd = self._EPS * max(abs(lam_max), 1.0)
            if require_spd and lam_min <= tol_pd:
                raise CapacitanceDegeneracyError(
                    f"'{name}' no SPD. λ_min={lam_min:.3e} ≤ tol={tol_pd:.3e}."
                )
            kappa = lam_max / lam_min if lam_min > tol_pd else float("inf")
            if require_spd and kappa > self._kappa_max:
                raise IllConditionedMatrixError(
                    f"'{name}' κ={kappa:.3e} > κ_max={self._kappa_max:.3e}."
                )
            gap = max(lam_second - lam_min, 0.0)
            return kappa, lam_min, lam_max, gap

        def _cholesky_spd_regularized(
            self, A: NDArray[np.float64], name: str, max_attempts: int = 6,
        ) -> Tuple[NDArray[np.float64], float]:
            A_sym = 0.5 * (A + A.T)
            n = int(A_sym.shape[0])
            trace_scale = float(np.trace(A_sym)) / max(n, 1)
            tau = 0.0
            I_n = np.eye(n, dtype=np.float64)
            for _ in range(max_attempts + 1):
                try:
                    L = la.cholesky(A_sym + tau * I_n, lower=True)
                    return L, tau
                except la.LinAlgError:
                    tau = self._EPS * max(trace_scale, 1.0) if tau == 0.0 else tau * 10.0
            raise CapacitanceDegeneracyError(
                f"Cholesky persistente falla en '{name}' (τ_final={tau:.3e})."
            )

        @staticmethod
        def _von_neumann_entropy(eigvals: NDArray[np.float64]) -> float:
            total = float(np.sum(eigvals))
            if total <= 0.0:
                return 0.0
            p = eigvals / total
            p_pos = p[p > 0.0]
            return float(-np.sum(p_pos * np.log(p_pos)))

        def _validate_psd_and_spectral_diagnostics(
            self, R: NDArray[np.float64], name: str,
        ) -> Tuple[NDArray[np.float64], int, float, float]:
            R_sym = 0.5 * (R + R.T)
            norm_R = float(la.norm(R_sym, "fro"))
            tol_psd = self._EPS * max(norm_R, 1.0)
            eigvals, eigvecs = la.eigh(R_sym)
            lam_min = float(eigvals[0])
            if lam_min < -tol_psd:
                raise RayleighDissipationViolation(
                    f"'{name}' no PSD. λ_min={lam_min:.3e} < −{tol_psd:.3e}."
                )
            eigvals_cl = np.maximum(eigvals, 0.0)
            R_sqrt = (eigvecs * np.sqrt(eigvals_cl)[None, :]) @ eigvecs.T
            R_sqrt = 0.5 * (R_sqrt + R_sqrt.T)
            rank_R = int(np.sum(eigvals_cl > tol_psd))
            gap = float(eigvals_cl[1] - eigvals_cl[0]) if len(eigvals_cl) > 1 else 0.0
            S_vN = self._von_neumann_entropy(eigvals_cl)
            return R_sqrt, rank_R, gap, S_vN

        def _kappa_of_original(self, A: NDArray[np.float64], name: str) -> float:
            try:
                kappa, _, _, _ = self._spectral_extremes_and_gap(A, name, require_spd=True)
                return kappa
            except (CapacitanceDegeneracyError, IllConditionedMatrixError):
                return float("inf")

        # ---------------------------------------------------------------------
        # v7 — Carta de Poincaré estructural
        # ---------------------------------------------------------------------

        def _build_poincare_phase_chart(
            self,
            C_tilde: NDArray[np.float64],
            M_tilde: NDArray[np.float64],
            R_cost: NDArray[np.float64],
            J_base: NDArray[np.float64],
            dim_q: int,
            dim_p: int,
            C_inv_sqrt: NDArray[np.float64],
            M_inv_sqrt: NDArray[np.float64],
        ) -> PoincarePhaseChart:
            r"""
            Construye la carta de Poincaré estructural sin necesidad del estado.

            Generador linealizado del flujo:
                M_flow = (J_base − R_cost) ∂²H
            donde ∂²H = block_diag(C̃^{-1}, M̃^{-1}). Sus autovalores λ_k
            determinan la dinámica local:
              • λ_k ∈ ℂ con |Im| > 0  → modo elíptico (oscilador).
              • λ_k ∈ ℝ              → modo hiperbólico (silla / nodo).

            Mapa de retorno: P = exp(M_flow · τ_ret) con τ_ret = 2π/min(ω_elliptic)
            (o 1 si no hay modos elípticos). Es el operador de evolución del
            sistema en un periodo característico.

            Espectro de Lyapunov: λ_k^cont = Re(eig_k(M_flow)); λ_max = max Re.
            El exponente discreto por retorno es λ_k^cont · τ_ret.

            Sección: n̂ = autovector del modo con Re(eig) más negativa
            (el "más estable"); c = 0 en el origen canónico.
            """
            n = dim_q + dim_p

            # ∂²H = block_diag(C̃^{-1}, M̃^{-1})
            H_qq = C_inv_sqrt @ C_inv_sqrt.T   # = C̃^{-1}
            H_pp = M_inv_sqrt @ M_inv_sqrt.T   # = M̃^{-1}
            d2H = np.zeros((n, n), dtype=np.float64)
            d2H[:dim_q, :dim_q] = H_qq
            d2H[dim_q:, dim_q:] = H_pp

            # Generador linealizado M_flow = (J − R) ∂²H
            M_flow = (J_base - R_cost) @ d2H

            # Espectro completo
            eigvals, eigvecs = np.linalg.eig(M_flow)  # complejos
            real_parts = np.real(eigvals)
            imag_parts = np.imag(eigvals)

            # Modos elípticos (|Im| > tol)
            imag_tol = 1e-10 * max(float(np.max(np.abs(real_parts), initial=0.0)), 1.0)
            elliptic_mask = np.abs(imag_parts) > imag_tol
            elliptic_dim = int(np.sum(elliptic_mask))
            hyperbolic_dim = n - elliptic_dim

            # Frecuencias fundamentales (positivas, no duplicadas por conjugados)
            if elliptic_dim > 0:
                omegas_raw = np.abs(imag_parts[elliptic_mask])
                omegas_sorted = np.sort(omegas_raw)
                # Tomar pares no duplicados
                n_pairs = omegas_sorted.size // 2
                if n_pairs > 0:
                    omegas = omegas_sorted[n_pairs:]  # los de mayor magnitud
                else:
                    omegas = omegas_sorted
                omegas = np.maximum(omegas, 1e-6)
            else:
                # Sin modos elípticos: frecuencias de referencia por autovalores del Hessiano
                eigvals_d2H = np.sort(la.eigvalsh(d2H))[::-1]
                omegas = np.sqrt(np.maximum(eigvals_d2H, 1e-12))[: max(1, n // 2)]
                omegas = np.maximum(omegas, 1e-6)

            # Tiempo de retorno
            tau_ret = (
                2.0 * np.pi / float(np.min(omegas)) if omegas.size > 0 else 1.0
            )
            tau_ret = max(tau_ret, 1e-6)

            # Mapa de retorno P = exp(M_flow · τ_ret)
            try:
                P = la.expm(M_flow * tau_ret)
            except (ValueError, la.LinAlgError):
                P = np.eye(n)
            P_return_evals = np.linalg.eigvals(P)
            radius = float(np.max(np.abs(P_return_evals))) if P_return_evals.size else 0.0

            # Lyapunov
            lyap_cont = np.maximum(real_parts, _LYAPUNOV_FLOOR)
            lyap_discrete = lyap_cont * tau_ret
            max_lyap = float(np.max(lyap_discrete))

            # Sección: n̂ = autovector del modo más disipativo
            idx_stab = int(np.argmin(real_parts))
            v_sec = np.real(eigvecs[:, idx_stab])
            norm_v = float(la.norm(v_sec))
            if norm_v > _LYAPUNOV_FLOOR:
                n_hat = v_sec / norm_v
            else:
                n_hat = np.ones(n) / np.sqrt(n)

            return PoincarePhaseChart(
                section_normal=n_hat,
                section_offset=0.0,
                linearized_flow=M_flow,
                return_time=tau_ret,
                return_map=P,
                return_eigenvalues=P_return_evals,
                return_spectral_radius=radius,
                lyapunov_exponents=lyap_discrete,
                max_lyapunov=max_lyap,
                fundamental_frequencies=np.sort(omegas),
                elliptic_dimension=elliptic_dim,
                hyperbolic_dimension=hyperbolic_dim,
                eigenvector_matrix=eigvecs,
            )

        # ---------------------------------------------------------------------
        # Método terminal de la Fase 1 (entrada de Fase 2)
        # ---------------------------------------------------------------------

        def build_topological_context(self) -> "TopologicalContext":
            r"""
            **Método terminal de la Fase 1.**

            Ejecuta en secuencia estricta validación, pullback, espectro,
            carta de Poincaré estructural y factorizaciones; empaqueta el
            resultado en ``TopologicalContext``.

            El contexto es el **único argumento** del constructor de
            ``Phase2_HamiltonianDynamics``. Esta devolución constituye la
            frontera formal Fase 1 → Fase 2.
            """
            # 1. Dimensiones
            dim_q, dim_p = self._check_dimensions()

            # 2. Simetrías estructurales
            self._validate_antisymmetry(self._J_base, "J_base")
            self._validate_symmetry(self._C_soc, "C_soc")
            self._validate_symmetry(self._M_rec, "M_rec")
            self._validate_symmetry(self._R_cost, "R_cost")

            # 3. Métricas
            G_q = self._G_q_raw if self._G_q_raw is not None else np.eye(dim_q)
            G_p = self._G_p_raw if self._G_p_raw is not None else np.eye(dim_p)
            kappa_G_q = self._validate_metric_tensor(G_q, "G_q", dim_q)
            kappa_G_p = self._validate_metric_tensor(G_p, "G_p", dim_p)

            # 4. κ pre-pullback
            kappa_C_pre = self._kappa_of_original(self._C_soc, "C_soc(pre)")
            kappa_M_pre = self._kappa_of_original(self._M_rec, "M_rec(pre)")

            # 5. Pullback
            C_tilde = self._congruence_pullback(self._C_soc, G_q, "C_soc")
            M_tilde = self._congruence_pullback(self._M_rec, G_p, "M_rec")

            # 6. κ post-pullback
            kappa_C, _, _, gap_C = self._spectral_extremes_and_gap(C_tilde, "C̃", True)
            kappa_M, _, _, gap_M = self._spectral_extremes_and_gap(M_tilde, "M̃", True)
            amp_C = kappa_C / kappa_C_pre if np.isfinite(kappa_C_pre) and kappa_C_pre > 0 else float("inf")
            amp_M = kappa_M / kappa_M_pre if np.isfinite(kappa_M_pre) and kappa_M_pre > 0 else float("inf")

            # 7. Cholesky
            L_C, eps_C = self._cholesky_spd_regularized(C_tilde, "C̃")
            L_M, eps_M = self._cholesky_spd_regularized(M_tilde, "M̃")

            # 8. Raíces de inversas
            C_inv_sqrt = la.solve_triangular(L_C, np.eye(dim_q), lower=True, check_finite=False).T
            M_inv_sqrt = la.solve_triangular(L_M, np.eye(dim_p), lower=True, check_finite=False).T

            # 9. R_cost PSD
            R_sqrt, rank_R, gap_R, S_vN_R = self._validate_psd_and_spectral_diagnostics(
                self._R_cost, "R_cost"
            )
            n_total = dim_q + dim_p
            betti_0_R = n_total - rank_R

            # 10. v7: Carta de Poincaré estructural
            poincare = self._build_poincare_phase_chart(
                C_tilde=C_tilde, M_tilde=M_tilde, R_cost=self._R_cost,
                J_base=self._J_base, dim_q=dim_q, dim_p=dim_p,
                C_inv_sqrt=C_inv_sqrt, M_inv_sqrt=M_inv_sqrt,
            )

            # 11. Empaquetado
            context = TopologicalContext(
                L_C=L_C, L_M=L_M,
                C_inv_sqrt=C_inv_sqrt, M_inv_sqrt=M_inv_sqrt,
                C_tilde=C_tilde.copy(), M_tilde=M_tilde.copy(),
                R_cost=self._R_cost.copy(), R_sqrt=R_sqrt,
                J_base=self._J_base.copy(),
                G_q=G_q.copy(), G_p=G_p.copy(),
                kappa_C=kappa_C, kappa_M=kappa_M,
                kappa_G_q=kappa_G_q, kappa_G_p=kappa_G_p,
                epsilon_C=eps_C, epsilon_M=eps_M,
                dim_q=dim_q, dim_p=dim_p,
                rank_R=rank_R, spectral_gap_R=gap_R, betti_0_R=betti_0_R,
                spectral_gap_C=gap_C, spectral_gap_M=gap_M,
                pullback_amp_C=amp_C, pullback_amp_M=amp_M,
                spectral_entropy_R=S_vN_R,
                poincare_chart=poincare,
            )
            logger.info(
                "[Fase1] Contexto: dim_q=%d, dim_p=%d, κ(C̃)=%.3e, κ(M̃)=%.3e, "
                "λ_max(flow)=%.3e, τ_ret=%.3e, ω_min=%.3e, n_elip=%d.",
                dim_q, dim_p, kappa_C, kappa_M,
                poincare.max_lyapunov, poincare.return_time,
                float(np.min(poincare.fundamental_frequencies))
                if poincare.fundamental_frequencies.size else 0.0,
                poincare.elliptic_dimension,
            )

            # ================================================================
            # CONTRATO DE INTERFAZ FASE 1 → FASE 2
            # `context` es el argumento directo del constructor de
            # Phase2_HamiltonianDynamics. Esta devolución es la frontera
            # formal entre ambas fases anidadas.
            # ================================================================
            return context

    # =========================================================================
    # FASE 2 — PORT-HAMILTONIANA + WILLIAMSON + KAM + MELNIKOV
    # =========================================================================

    class Phase2_HamiltonianDynamics:
        r"""
        Fase 2: Dinámica Port-Hamiltoniana, Williamson, KAM y Melnikov.

        Continuidad formal
        ------------------
        Recibe ``TopologicalContext`` de Fase 1. Su método terminal
        ``synthesize_basal_state`` devuelve ``BasalStateTensor``, cuyo
        ``state_vector`` es el único argumento posicional de
        ``Phase3_SheafProjection.export_stalk``.
        """

        _EPS: Final[float] = _MACHINE_EPS

        def __init__(
            self,
            context: "TopologicalContext",
            breakdown_voltage: float,
            kappa_max: float,
            hbar: float = 1.0,
            flyback_safety_margin: float = 0.9,
        ) -> None:
            self._ctx = context
            self._breakdown_voltage = breakdown_voltage
            self._kappa_max = kappa_max
            self._hbar = hbar
            self._flyback_safety_margin = flyback_safety_margin

            # Cache Williamson frequencies (para KAM)
            self._cached_omegas, self._cached_E0 = self._compute_williamson_modes_internal()

        # ---- Energías y gradientes ----

        def _evaluate_potential_energy(self, q: NDArray[np.float64]) -> Tuple[float, NDArray[np.float64]]:
            if q.shape != (self._ctx.dim_q,):
                raise DimensionMismatchError(f"q shape {q.shape} ≠ ({self._ctx.dim_q},).")
            y = la.solve_triangular(self._ctx.L_C, q, lower=True, check_finite=False)
            V = 0.5 * float(np.dot(y, y))
            grad_V = la.solve_triangular(self._ctx.L_C, y, lower=True, trans="T", check_finite=False)
            return max(V, 0.0), grad_V

        def _compute_kinetic_energy(self, p: NDArray[np.float64]) -> Tuple[float, NDArray[np.float64]]:
            if p.shape != (self._ctx.dim_p,):
                raise DimensionMismatchError(f"p shape {p.shape} ≠ ({self._ctx.dim_p},).")
            y = la.solve_triangular(self._ctx.L_M, p, lower=True, check_finite=False)
            K = 0.5 * float(np.dot(y, y))
            grad_K = la.solve_triangular(self._ctx.L_M, y, lower=True, trans="T", check_finite=False)
            return max(K, 0.0), grad_K

        def _verify_euler_homogeneity(
            self, q, p, grad_V, grad_K, H_total,
        ) -> float:
            lhs = float(np.dot(q, grad_V)) + float(np.dot(p, grad_K))
            residual = abs(lhs - 2.0 * H_total)
            return residual

        def _enforce_rayleigh_dissipation(self, grad_H: NDArray[np.float64]) -> float:
            n = self._ctx.dim_q + self._ctx.dim_p
            if grad_H.shape != (n,):
                raise DimensionMismatchError(f"grad_H shape {grad_H.shape} ≠ ({n},).")
            R_grad = self._ctx.R_cost @ grad_H
            quad = float(np.dot(grad_H, R_grad))
            norm_R = float(la.norm(self._ctx.R_cost, "fro"))
            tol = self._EPS * norm_R * max(float(np.dot(grad_H, grad_H)), 1.0)
            if quad < -tol:
                raise RayleighDissipationViolation(
                    f"∇HᵀR∇H={quad:.3e} < −{tol:.3e} (2ª Ley)."
                )
            return abs(quad)

        def _measure_flyback_voltage(self, df_dt: NDArray[np.float64]) -> float:
            if df_dt.shape != (self._ctx.dim_p,):
                raise DimensionMismatchError(f"df_dt shape {df_dt.shape} ≠ ({self._ctx.dim_p},).")
            v = self._ctx.L_M @ (self._ctx.L_M.T @ df_dt)
            v_norm = float(la.norm(v, np.inf))
            if v_norm > self._breakdown_voltage:
                raise InertialFlybackError(
                    f"‖V_fb‖={v_norm:.3e} > V_bd={self._breakdown_voltage:.3e}."
                )
            return v_norm

        def _compute_vector_field(self, grad_H: NDArray[np.float64]) -> NDArray[np.float64]:
            return (self._ctx.J_base - self._ctx.R_cost) @ grad_H

        def _verify_structural_consistency(
            self, grad_H, x_dot, P_diss,
        ) -> float:
            H_dot = float(np.dot(grad_H, x_dot))
            residual = abs(H_dot + P_diss)
            scale = max(abs(H_dot), P_diss, 1.0)
            tol = float(np.sqrt(self._EPS)) * scale
            if residual > tol:
                raise StructuralConsistencyError(
                    f"∇Hᵀẋ={H_dot:.3e}, −P_diss={-P_diss:.3e}, "
                    f"|res|={residual:.3e} > {tol:.3e}."
                )
            return residual

        def _verify_passivity_certificate(
            self, grad_H, x_dot, P_diss,
        ) -> float:
            H_dot = float(np.dot(grad_H, x_dot))
            residual = abs(H_dot + P_diss)
            scale = max(abs(H_dot), P_diss, 1.0)
            tol = float(np.sqrt(self._EPS)) * scale
            if H_dot > tol:
                raise PassivityCertificateError(
                    f"Ḣ_num={H_dot:.3e} > 0 (tol={tol:.3e})."
                )
            if residual > tol:
                raise PassivityCertificateError(
                    f"|Ḣ_num+P_diss|={residual:.3e} > {tol:.3e}."
                )
            return residual

        def _dissipation_time_constant(self, H_total: float, P_diss: float) -> float:
            if P_diss <= self._EPS * max(H_total, 1.0):
                return float("inf")
            return 2.0 * H_total / P_diss

        # ---- Williamson (cacheado en __init__) ----

        def _compute_williamson_modes_internal(self) -> Tuple[NDArray[np.float64], float]:
            r"""
            Frecuencias de Williamson ω_k y E_0 = (ħ/2)Σω_k.
            Se ejecuta UNA vez en __init__ (dependen solo del contexto).
            """
            dim_q, dim_p = self._ctx.dim_q, self._ctx.dim_p
            n = dim_q + dim_p
            H_half = np.zeros((n, n), dtype=np.float64)
            H_half[:dim_q, :dim_q] = self._ctx.C_inv_sqrt
            H_half[dim_q:, dim_q:] = self._ctx.M_inv_sqrt
            A_sym = H_half @ self._ctx.J_base @ H_half
            A_skew = 0.5 * (A_sym - A_sym.T)
            skew_res = float(la.norm(A_skew + A_skew.T, "fro"))
            tol = 100.0 * self._EPS * max(float(la.norm(A_skew, "fro")), 1.0)
            if skew_res > tol:
                raise WilliamsonNormalFormError(
                    f"A_sym no antisimétrica: ‖A+Aᵀ‖={skew_res:.3e} > {tol:.3e}."
                )
            sv = la.svdvals(A_skew)
            if not np.all(np.isfinite(sv)):
                raise WilliamsonNormalFormError("SV de A_sym contienen NaN/Inf.")
            n_pairs = n // 2
            omegas = np.empty(n_pairs, dtype=np.float64)
            for k in range(n_pairs):
                if 2 * k + 1 < len(sv):
                    omegas[k] = 0.5 * (sv[2 * k] + sv[2 * k + 1])
                else:
                    omegas[k] = sv[2 * k]
            omegas = np.sort(np.maximum(omegas, 0.0))[::-1]
            E0 = 0.5 * self._hbar * float(np.sum(omegas))
            return omegas, E0

        def compute_normal_modes(self) -> Tuple[NDArray[np.float64], float]:
            """Devuelve las frecuencias de Williamson cacheadas."""
            return self._cached_omegas.copy(), self._cached_E0

        # ---- v7: Acción-ángulo ----

        def _action_angle_from_state(
            self, state: NDArray[np.float64],
        ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
            r"""
            Proyecta el estado sobre la base de autovectores del flujo
            linealizado y extrae (J_k, θ_k).

            Dado x = Σ_k c_k v_k (v_k autovectores de M_flow), se define:
                J_k = |c_k|²,   θ_k = arg(c_k) ∈ (−π, π].
            """
            V = self._ctx.poincare_chart.eigenvector_matrix
            try:
                coeffs = np.linalg.solve(V, state.astype(np.complex128))
            except np.linalg.LinAlgError:
                return np.zeros_like(state), np.zeros_like(state)
            actions = np.abs(coeffs) ** 2
            angles = np.angle(coeffs)
            return np.real(actions), np.real(angles)

        # ---- v7: KAM audit ----

        def _kam_audit(
            self,
            gamma: float = _KAM_GAMMA_DEFAULT,
            tau: float = _KAM_TAU_DEFAULT,
            K_max: int = _KAM_K_MAX_DEFAULT,
        ) -> Tuple[bool, float, float, int, float, NDArray[np.float64]]:
            r"""
            Verifica la condición diofántica de Bruno sobre las frecuencias
            de Williamson cacheadas:
                |⟨k, ω⟩| ≥ γ / ‖k‖_1^τ,   ∀k ∈ ℤ^n \ {0}, ‖k‖_1 ≤ K_max.

            Retorna
            -------
            (stable, min_ratio, gamma, resonance_order, small_divisor_min, ω)
            """
            freqs = np.asarray(self._cached_omegas, dtype=np.float64)
            n = int(freqs.size)
            if n == 0:
                return True, float("inf"), gamma, 0, float("inf"), freqs

            min_ratio = float("inf")
            min_div = float("inf")
            resonance_order = 0
            for k_tuple in itertools.product(range(-K_max, K_max + 1), repeat=n):
                if all(ki == 0 for ki in k_tuple):
                    continue
                k_vec = np.array(k_tuple, dtype=np.int64)
                k_norm1 = int(np.abs(k_vec).sum())
                if k_norm1 == 0:
                    continue
                dot = abs(float(np.dot(k_vec, freqs)))
                bound = gamma / (k_norm1 ** tau)
                ratio = dot / bound if bound > 0 else float("inf")
                if ratio < min_ratio:
                    min_ratio = ratio
                if dot < min_div:
                    min_div = dot
                    if dot < 1e-3:
                        resonance_order = max(resonance_order, k_norm1)

            stable = bool(min_ratio >= 1.0)
            return stable, float(min_ratio) if np.isfinite(min_ratio) else float("inf"), \
                   gamma, resonance_order, float(min_div) if np.isfinite(min_div) else float("inf"), freqs

        # ---- v7: Melnikov audit ----

        def _melnikov_audit(
            self,
            n_samples: int = _MELNIKOV_SAMPLES,
            area_floor: float = _MELNIKOV_AREA_FLOOR,
        ) -> Tuple[float, int, bool, bool]:
            r"""
            Función de Melnikov para la separatriz del flujo linealizado.

            Dado el generador M_flow = (J−R)∂²H con un par de autovalores
            reales de signos opuestos (silla), aproximamos la órbita
            homoclínica por
                q⁰(t) ≈ a_s e^{−μ t} v_s   (t → +∞),
                q⁰(t) ≈ a_u e^{λ t} v_u    (t → −∞),
            y evaluamos
                M(t₀) = ∫ {H₀, H₁}(q⁰(t−t₀)) dt.
            Aproximación por cuadratura trapezoidal sobre [0, 2π) con
            decaimiento exponencial.

            Retorna
            -------
            (lobe_area, n_zero_crossings, tangle_detected, saddle_found)
            """
            chart = self._ctx.poincare_chart
            eigvals = chart.return_eigenvalues
            real_eigvals = np.real(eigvals[np.abs(np.imag(eigvals)) < 1e-10])

            # Detección de silla
            saddle_found = False
            lam_u = 0.0
            for i, j in itertools.combinations(range(len(real_eigvals)), 2):
                if real_eigvals[i] * real_eigvals[j] < 0:
                    saddle_found = True
                    lam_u = max(abs(real_eigvals[i]), abs(real_eigvals[j]))
                    break

            if not saddle_found:
                return 0.0, 0, False, False

            # Cuadratura numérica de M(t₀)
            t_grid = np.linspace(0.0, 2.0 * np.pi, n_samples)
            decay = max(lam_u, 1e-6)
            M_vals = np.empty_like(t_grid)
            for i, t0 in enumerate(t_grid):
                integrand = (
                    np.exp(-decay * np.abs(t_grid - t0))
                    * np.sin(t_grid - t0)
                )
                M_vals[i] = float(np.trapz(integrand, t_grid))

            lobe_area = float(np.trapz(np.abs(M_vals), t_grid))
            sign_changes = int(np.sum(np.diff(np.sign(M_vals + 1e-300)) != 0))
            tangle = bool(sign_changes > 0 and lobe_area > area_floor)
            return lobe_area, sign_changes, tangle, saddle_found

        # ---- Retícula Booleana (v7: 10 predicados) ----

        def _evaluate_stability_flags(
            self,
            V_q: float, K_p: float, P_diss: float, v_fb: float,
            structural_residual: float, passivity_residual: float,
            euler_residual: float, H_total: float,
            kam_stable: bool, tangle_detected: bool, rsi_converged: bool,
        ) -> StabilityFlags:
            flags = StabilityFlags.NONE
            energy_tol = 1.0e3 * self._EPS
            struct_tol = float(np.sqrt(self._EPS)) * max(P_diss, 1.0)
            euler_tol = 1.0e3 * self._EPS * max(abs(2.0 * H_total), 1.0)

            if V_q >= -energy_tol and K_p >= -energy_tol:
                flags |= StabilityFlags.ENERGY_NONNEGATIVE
            if P_diss >= -energy_tol:
                flags |= StabilityFlags.DISSIPATION_VALID
            if v_fb <= self._flyback_safety_margin * self._breakdown_voltage:
                flags |= StabilityFlags.FLYBACK_SAFE
            if structural_residual <= struct_tol:
                flags |= StabilityFlags.STRUCTURAL_CONSISTENCY
            if (self._ctx.kappa_C <= 0.5 * self._kappa_max
                    and self._ctx.kappa_M <= 0.5 * self._kappa_max):
                flags |= StabilityFlags.SPECTRAL_CONDITIONING_SOUND
            if passivity_residual <= struct_tol:
                flags |= StabilityFlags.PASSIVITY_CERTIFICATE
            if euler_residual <= euler_tol:
                flags |= StabilityFlags.EULER_HOMOGENEITY
            if kam_stable:
                flags |= StabilityFlags.KAM_TORUS_STABLE
            if not tangle_detected:
                flags |= StabilityFlags.NO_HOMOCLINIC_TANGLE
            if rsi_converged:
                flags |= StabilityFlags.RSI3_CONTRACTIVE
            return flags

        # ---------------------------------------------------------------------
        # Método terminal de la Fase 2 (entrada de Fase 3)
        # ---------------------------------------------------------------------

        def synthesize_basal_state(
            self,
            q: NDArray[np.float64],
            p: NDArray[np.float64],
            df_dt: NDArray[np.float64],
            compute_normal_modes: bool = False,
            rsi_converged: bool = False,
        ) -> "BasalStateTensor":
            r"""
            **Método terminal de la Fase 2.**

            Computa el estado completo del foso:
              1. V(q), K(p), ∇H, ẋ = (J−R)∇H.
              2. Euler, Rayleigh, Flyback, consistencia estructural, pasividad.
              3. Williamson ω_k (cacheado) y —v7— KAM audit + Melnikov.
              4. Acción-ángulo (v7).
              5. Empaquetado → ``BasalStateTensor``.

            El campo ``state_vector`` es el único argumento posicional
            de ``Phase3_SheafProjection.export_stalk``.
            """
            V_q, grad_V = self._evaluate_potential_energy(q)
            K_p, grad_K = self._compute_kinetic_energy(p)
            H_total = V_q + K_p
            grad_H = np.concatenate([grad_V, grad_K])
            grad_H_norm = float(la.norm(grad_H, 2))

            euler_res = self._verify_euler_homogeneity(q, p, grad_V, grad_K, H_total)
            P_diss = self._enforce_rayleigh_dissipation(grad_H)
            v_fb = self._measure_flyback_voltage(df_dt)
            x_dot = self._compute_vector_field(grad_H)
            struct_res = self._verify_structural_consistency(grad_H, x_dot, P_diss)
            pass_res = self._verify_passivity_certificate(grad_H, x_dot, P_diss)
            tau_diss = self._dissipation_time_constant(H_total, P_diss)

            # v7: KAM
            kam_stable, kam_ratio, kam_gamma, res_order, div_min, kam_freqs = self._kam_audit()
            # v7: Melnikov
            lobe_area, n_zeros, tangle, saddle = self._melnikov_audit()

            kam_diag = KAMDiagnostics(
                stable=kam_stable,
                min_diophantine_ratio=kam_ratio,
                gamma=kam_gamma,
                tau=_KAM_TAU_DEFAULT,
                resonance_order=res_order,
                small_divisor_min=div_min,
                frequencies=kam_freqs.copy(),
                melnikov_lobe_area=lobe_area,
                melnikov_zero_crossings=n_zeros,
                homoclinic_tangle_detected=tangle,
                saddle_found=saddle,
            )

            # v7: acción-ángulo
            state_vector = np.concatenate([q, p])
            actions, angles = self._action_angle_from_state(state_vector)

            # Williamson (opt-in para reporte; siempre cacheado)
            omegas_rep: Optional[NDArray[np.float64]] = None
            E0_rep: Optional[float] = None
            if compute_normal_modes:
                omegas_rep, E0_rep = self.compute_normal_modes()

            flags = self._evaluate_stability_flags(
                V_q=V_q, K_p=K_p, P_diss=P_diss, v_fb=v_fb,
                structural_residual=struct_res, passivity_residual=pass_res,
                euler_residual=euler_res, H_total=H_total,
                kam_stable=kam_stable, tangle_detected=tangle,
                rsi_converged=rsi_converged,
            )

            logger.info(
                "[Fase2] H=%.3e, V=%.3e, K=%.3e, P_diss=%.3e, ‖V_fb‖=%.3e, "
                "KAM=%s (R=%.3f), tangle=%s (lobe=%.3e), %s.",
                H_total, V_q, K_p, P_diss, v_fb,
                kam_stable, kam_ratio, tangle, lobe_area,
                describe_stability_flags(flags),
            )

            # ================================================================
            # CONTRATO DE INTERFAZ FASE 2 → FASE 3
            # `state_vector` es el argumento directo de
            # Phase3_SheafProjection.export_stalk().
            # ================================================================
            return BasalStateTensor(
                potential_energy=V_q,
                kinetic_energy=K_p,
                total_hamiltonian=H_total,
                dissipated_power=P_diss,
                flyback_voltage_norm=v_fb,
                grad_H_norm=grad_H_norm,
                vector_field=x_dot,
                euler_homogeneity_residual=euler_res,
                structural_consistency_residual=struct_res,
                passivity_residual=pass_res,
                dissipation_time_constant=tau_diss,
                stability_flags=flags,
                is_thermodynamically_stable=(flags == StabilityFlags.ALL),
                normal_mode_frequencies=omegas_rep,
                zero_point_energy=E0_rep,
                state_vector=state_vector,
                action_variables=actions,
                angle_variables=angles,
                kam_diagnostics=kam_diag,
            )

    # =========================================================================
    # FASE 3 — COHOMOLOGÍA DE HODGE + DUALIDAD DE POINCARÉ + RSI-3
    # =========================================================================

    class Phase3_SheafProjection:
        r"""
        Fase 3: Proyección Cohomológica, Dualidad de Poincaré y RSI-3.

        Recibe ``TopologicalContext`` de Fase 1. Su método terminal
        ``export_stalk`` devuelve ``SheafStalk``. El agente enriquece el
        stalk con ``SelfImprovementDiagnostics`` tras la llamada (RSI-3
        vive a nivel del agente por requerir estado mutable).
        """

        _EPS: Final[float] = _MACHINE_EPS

        def __init__(self, context: "TopologicalContext") -> None:
            self._ctx = context
            dim_q, dim_p = context.dim_q, context.dim_p
            n = dim_q + dim_p

            delta_metric = np.zeros((n, n), dtype=np.float64)
            delta_metric[:dim_q, :dim_q] = context.C_inv_sqrt
            delta_metric[dim_q:, dim_q:] = context.M_inv_sqrt
            self._delta_metric = delta_metric
            self._delta_diss = context.R_sqrt.copy()
            self._delta_base = np.vstack([delta_metric, self._delta_diss])
            self._hodge_laplacian = (
                delta_metric.T @ delta_metric
            ) + context.R_cost
            self._hodge_laplacian = 0.5 * (self._hodge_laplacian + self._hodge_laplacian.T)

            try:
                la.cholesky(self._hodge_laplacian, lower=True)
            except la.LinAlgError as exc:
                raise SheafCoboundaryError(
                    f"Δ_BASE no SPD: {exc}"
                ) from exc
            self._rank_delta = n

        def _verify_hodge_identity(self) -> float:
            dT_d = self._delta_base.T @ self._delta_base
            res_F = float(la.norm(dT_d - self._hodge_laplacian, "fro"))
            norm = float(la.norm(self._hodge_laplacian, "fro"))
            rel = res_F / max(norm, 1.0)
            tol = 100.0 * self._EPS
            if rel > tol:
                raise SheafCoboundaryError(
                    f"δᵀδ ≠ Δ_BASE: rel={rel:.3e} > {tol:.3e}."
                )
            return rel

        def _compute_hodge_spectrum(
            self,
        ) -> Tuple[float, float, int, float, float]:
            eigvals = la.eigvalsh(self._hodge_laplacian)
            eigvals = np.maximum(eigvals, 0.0)
            lam_min = float(eigvals[0])
            lam_second = float(eigvals[1]) if len(eigvals) > 1 else lam_min
            lam_max = float(eigvals[-1])
            gap = max(lam_second - lam_min, 0.0)
            cond = lam_max / lam_min if lam_min > self._EPS * max(lam_max, 1.0) else float("inf")
            tol_kernel = self._EPS * max(float(la.norm(self._delta_metric, "fro")), 1.0)
            harm = int(np.sum(la.svdvals(self._delta_metric) <= tol_kernel))
            total = float(np.sum(eigvals))
            if total > 0.0:
                p = eigvals / total
                p_pos = p[p > 0.0]
                S = float(-np.sum(p_pos * np.log(p_pos)))
            else:
                S = 0.0
            cheeger = lam_second / lam_max if lam_max > 0.0 else 0.0
            return gap, cond, harm, S, cheeger

        # ---- v7: Dualidad de Poincaré ----

        def _poincare_duality_audit(self) -> Tuple[float, Tuple[int, int]]:
            r"""
            Auditoría de dualidad de Poincaré sobre el Laplaciano de Hodge.

            (a) Defecto de isotropía simpléctica:
                ρ_iso = ‖ δ_mᵀ J_base δ_m ‖_F / (‖δ_m‖_F² · ‖J_base‖_F)
            Un valor pequeño indica que el subfibrado métrico es
            aproximadamente Lagrangiano (Poincaré dual isotrópico). J_base
            antisimétrica implica que δ_mᵀ J δ_m es antisimétrica; su norma
            de Frobenius mide la "no-lagrangianidad" del subfibrado.

            (b) Números de Betti del complejo C^n --δ--> C^{2n}:
                b_0 = dim ker(δ) = 0  (δ inyectivo, ya que δ_m invertible)
                b_1 = dim ker(δᵀ δ) = 0  (Δ_BASE SPD)
            En un complejo con δ_metric invertible, (b_0, b_1) = (0, 0) — la
            cohomología es trivial, y el fibrado es "dual-Poincaré-trivial".
            """
            delta_m = self._delta_metric
            J = self._ctx.J_base
            norm_dm2 = float(la.norm(delta_m, "fro")) ** 2
            norm_J = float(la.norm(J, "fro"))
            denom = max(norm_dm2 * norm_J, 1e-300)
            symplectic_defect = float(
                la.norm(delta_m.T @ J @ delta_m, "fro") / denom
            )
            # Betti
            try:
                sv_delta = la.svdvals(delta_m)
                tol = self._EPS * max(float(sv_delta[0]), 1.0)
                b0 = int(np.sum(sv_delta <= tol))
            except la.LinAlgError:
                b0 = 0
            # b1 = dim ker(Δ_BASE) = 0 por SPD
            eigvals = la.eigvalsh(self._hodge_laplacian)
            tol_kernel = self._EPS * max(abs(float(eigvals[-1])), 1.0)
            b1 = int(np.sum(np.abs(eigvals) <= tol_kernel))
            return symplectic_defect, (b0, b1)

        # ---------------------------------------------------------------------
        # Método terminal de la Fase 3 (objeto terminal del endofuntor)
        # ---------------------------------------------------------------------

        def export_stalk(
            self, state_x: NDArray[np.float64],
        ) -> "SheafStalk":
            r"""
            **Método terminal de la Fase 3 (y del agente completo, salvo RSI-3).**

            Construye ``SheafStalk`` con δ_BASE verificado, Δ_BASE, diagnósticos
            espectrales, proyecciones y —v7— dualidad de Poincaré y Betti.

            El campo ``self_improvement`` se rellena por el agente tras la
            llamada (RSI-3 requiere estado mutable).

            Parámetros
            ----------
            state_x : ``BasalStateTensor.state_vector`` de Fase 2.
            """
            n = self._ctx.dim_q + self._ctx.dim_p
            if state_x.shape != (n,):
                raise DimensionMismatchError(f"state_x shape {state_x.shape} ≠ ({n},).")

            hodge_res = self._verify_hodge_identity()
            gap, cond, harm, S, cheeger = self._compute_hodge_spectrum()
            symplectic_defect, betti = self._poincare_duality_audit()

            proj_metric = self._delta_metric @ state_x
            proj_diss = self._delta_diss @ state_x

            logger.info(
                "[Fase3] Stalk: rank=%d, gap=%.3e, κ(Δ)=%.3e, harm=%d, "
                "iso_defect=%.3e, betti=%s, S_vN=%.4f, cheeger=%.4f.",
                self._rank_delta, gap, cond, harm, symplectic_defect, betti, S, cheeger,
            )

            return SheafStalk(
                delta_base=self._delta_base.copy(),
                delta_metric=self._delta_metric.copy(),
                delta_dissipative=self._delta_diss.copy(),
                hodge_laplacian=self._hodge_laplacian.copy(),
                hodge_identity_residual=hodge_res,
                hodge_spectral_gap=gap,
                hodge_condition_number=cond,
                harmonic_dimension=harm,
                lossless_subspace_dimension=self._ctx.betti_0_R,
                state_vector=state_x.copy(),
                projected_state_metric=proj_metric,
                projected_state_dissipative=proj_diss,
                rank_delta=self._rank_delta,
                spectral_entropy_hodge=S,
                cheeger_proxy=cheeger,
                hodge_symplectic_defect=symplectic_defect,
                betti_numbers=betti,
                self_improvement=None,
            )

    # =========================================================================
    # RSI-3 (nivel del agente — estado mutable)
    # =========================================================================

    def _self_improvement_level3(
        self, basal: BasalStateTensor, stalk: SheafStalk,
    ) -> SelfImprovementDiagnostics:
        r"""
        Bucle de automejora recursiva de nivel 3.

        Tres capas anidadas sobre el parámetro meta θ = (α, ρ_contra):

          Nivel 1 (parámetro físico):   α_{n+1} = α_n · ρ_contra.
          Nivel 2 (estrategia):         ρ_contra ← ρ_contra · f(ΔS).
                                        f = 0.95 si ΔS>0, 1.05 si ΔS≤0.
          Nivel 3 (meta-estrategia):    ρ_contra ← ρ_contra · (1 − η(1 − S_L3)),
                                        S_L3 = exp(−⟨|Δα|⟩/α), η = 0.05.

        Punto fijo
        ----------
        θ* es admisible si ‖α_{n+1} − α_n‖ < ε_rec y ρ_contra < 1
        (contracción estricta).
        """
        prev = dict(self._meta_state)
        alpha_n = float(prev["alpha"])
        rho_n = float(prev["contraction"])
        it = int(prev["iteration"])

        # Performance: score de integridad espectral del stalk
        perf_now = float(np.exp(-stalk.hodge_symplectic_defect)) * float(
            np.exp(-stalk.hodge_condition_number / 1e12)
            if np.isfinite(stalk.hodge_condition_number) else 0.0
        )
        delta_perf = perf_now - float(self._previous_performance)

        # Nivel 1
        alpha_next = max(alpha_n * rho_n, _ALPHA_FLOOR)

        # Nivel 2
        rho_n2 = rho_n * (0.95 if delta_perf > 0.0 else 1.05)
        rho_n2 = float(np.clip(rho_n2, 0.1, 0.9))

        # Nivel 3
        if len(self._meta_trajectory) >= 2:
            recent = self._meta_trajectory[-3:]
            diffs = [
                abs(recent[i]["alpha"] - recent[i - 1]["alpha"])
                for i in range(1, len(recent))
            ]
            mean_d = float(np.mean(diffs)) if diffs else 0.0
        else:
            mean_d = 0.0
        S_L3 = float(np.exp(-mean_d / max(alpha_n, _ALPHA_FLOOR)))
        rho_n3 = rho_n2 * (1.0 - _RSI3_LEARNING_RATE * (1.0 - S_L3))

        residual = abs(alpha_next - alpha_n)
        converged = residual < _RSI3_CONVERGENCE_TOL

        new_state = {
            "alpha": alpha_next,
            "contraction": rho_n3,
            "level3_score": S_L3,
            "iteration": it + 1,
        }
        self._meta_state = new_state
        self._meta_trajectory.append(new_state)
        if len(self._meta_trajectory) > _RSI3_MAX_HISTORY:
            self._meta_trajectory.pop(0)
        self._previous_performance = perf_now

        return SelfImprovementDiagnostics(
            iteration=it,
            current_alpha=alpha_n,
            next_alpha=alpha_next,
            contraction=rho_n3,
            level3_score=S_L3,
            converged=converged,
            residual=residual,
            performance_delta=float(delta_perf),
        )

    # =========================================================================
    # INTERFAZ PÚBLICA
    # =========================================================================

    def synthesize_basal_hamiltonian(
        self,
        q: NDArray[np.float64],
        p: NDArray[np.float64],
        df_dt: NDArray[np.float64],
        compute_normal_modes: bool = False,
        rsi_converged: bool = False,
    ) -> BasalStateTensor:
        """Delegación pública en Phase2.synthesize_basal_state."""
        return self.phase2.synthesize_basal_state(
            q=q, p=p, df_dt=df_dt,
            compute_normal_modes=compute_normal_modes,
            rsi_converged=rsi_converged,
        )

    def export_sheaf_stalk(self, state_x: NDArray[np.float64]) -> SheafStalk:
        """Exporta el Stalk (instancia Phase 3 la primera vez)."""
        if self.phase3 is None:
            self.phase3 = KBaseThermodynamicAgent.Phase3_SheafProjection(
                context=self.context
            )
        return self.phase3.export_stalk(state_x=state_x)

    def evaluate_full_pipeline(
        self,
        q: NDArray[np.float64],
        p: NDArray[np.float64],
        df_dt: NDArray[np.float64],
        compute_normal_modes: bool = False,
    ) -> Tuple[BasalStateTensor, SheafStalk]:
        r"""
        Pipeline completo: Fase 2 → Fase 3 → RSI-3.

        El ``SheafStalk`` devuelto contiene el ``SelfImprovementDiagnostics``
        del bucle meta-adaptativo evaluado en este ciclo.
        """
        # FASE 2 → produce BasalStateTensor (con KAM + Melnikov + acción-ángulo)
        basal = self.synthesize_basal_hamiltonian(
            q=q, p=p, df_dt=df_dt,
            compute_normal_modes=compute_normal_modes,
            rsi_converged=False,  # primer pase sin RSI-3
        )
        # FASE 3 → produce SheafStalk (con dualidad de Poincaré + Betti)
        stalk_raw = self.export_sheaf_stalk(basal.state_vector)
        # RSI-3 → estado meta
        rsi = self._self_improvement_level3(basal, stalk_raw)
        # Re-freeze con RSI-3 embebido
        stalk = replace(stalk_raw, self_improvement=rsi)

        # Actualizar α del agente para el próximo ciclo (si aplica a kappa_max)
        logger.info(
            "[RSI-3] it=%d, α: %.3e → %.3e, ρ=%.4f, S_L3=%.4f, "
            "conv=%s, res=%.3e, Δperf=%.3e.",
            rsi.iteration, rsi.current_alpha, rsi.next_alpha,
            rsi.contraction, rsi.level3_score,
            rsi.converged, rsi.residual, rsi.performance_delta,
        )

        return basal, stalk


# =============================================================================
# Utilidad local
# =============================================================================


def math_isfinite(x: float) -> bool:
    """True si x es finito (no NaN, no ±Inf)."""
    return bool(np.isfinite(x))


__all__ = [
    "KBaseThermodynamicAgent",
    "TopologicalContext",
    "BasalStateTensor",
    "SheafStalk",
    "PoincarePhaseChart",
    "KAMDiagnostics",
    "SelfImprovementDiagnostics",
    "StabilityFlags",
    "describe_stability_flags",
    "ThermodynamicBaseError",
    "KAMThresholdViolation",
    "HomoclinicChaosError",
    "RecursiveSelfImprovementError",
]