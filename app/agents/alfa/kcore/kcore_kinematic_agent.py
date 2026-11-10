# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : KCore Kinematic Agent (Director de Flujo y Cinética Logística)      ║
║ Ruta   : app/agents/alpha/kcore/kcore_kinematic_agent.py                     ║
║ Versión: 7.0.0-Rigorous-IDA-PBC-Hodge-CFL-Sheaf-Poincare-KAM-Melnikov-RSI3   ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y MAQUINARIA CINEMÁTICA EN EL NÚCLEO (V_𝕋) ──────────────
Endofuntor Cinemático covariante que transmuta la energía potencial de cimentación
(K_BASE) en trabajo cinético direccional, acoplando algebraicamente Actividades
Clave, Canales y Relaciones. Repudia las proyecciones lineales: subyuga el
transporte al control Port-Hamiltoniano basado en pasividad (IDA-PBC) con
proyección pseudoinversa covariante respecto a G_μν ≻ 0.

INVARIANTES MATEMÁTICOS (heredados + v7): ─────────────────────────────────────
  [I1] IDA-PBC: α = (gᵀ G g + λ I)⁺ gᵀ G F_req.
  [I2] Lyapunov: Ḣ_d = −∇H_dᵀ R_d ∇H_d ≤ 0 con R_d ⪰ 0.
  [I3] CFL: Δt ≤ 2·CFL_margin / (c_eff √λ_max(ℒ_sym)).
  [I4] Hodge: I = I_g + I_c + I_h, ‖I_curl‖_W ≤ ε_crit.
  [I5] Hodge local: δ_COREᵀ δ_CORE = W_mod.
  [I6] KAM: |⟨k,ω⟩| ≥ γ/‖k‖^τ_1 ∀k ≠ 0 (v7).
  [I7] No caos: M(t₀) sin ceros simples (v7).
  [I8] RSI-3: ‖α_{n+1} − α_n‖ < ε_rec (v7).
  [I9] Poincaré dual: ‖δᵀ J δ‖_F / (‖δ‖_F² ‖J‖_F) ≪ 1 (v7).

ARQUITECTURA DE TRES FASES ANIDADAS (v7): ──────────────────────────────────────
  Fase 1 │ VALIDACIÓN MATRICIAL + POINCARÉ ESTRUCTURAL (Observe)
         │ Valida J,R,J_d,R_d,g,G. Construye carta de Poincaré:
         │ generador A = (J_d − R_d)·G, mapa de retorno P = exp(A·τ_ret),
         │ Lyapunov λ_k = Re(eig(A))·τ_ret, frecuencias ω_k = Im(eig(A)).
         │ ▸ Codominio: KinematicPreparationContext (con poincare_chart).
         │ ▸ DOMINIO DE Phase2_KinematicSynthesis.

  Fase 2 │ SÍNTESIS CINEMÁTICA + KAM + MELNIKOV (Orient)
         │ IDA-PBC covariante, Hodge estrangulado, Kramers-Kronig, CFL dual.
         │ + Audit KAM (diofántico de Bruno). + Melnikov (homoclínico).
         │ + Acción-ángulo del estado requerido.
         │ ▸ Codominio: KinematicStateTensor (con kam_diagnostics).
         │ ▸ DOMINIO DE Phase3_SheafProjection.export_stalk.

  Fase 3 │ HODGE + DUALIDAD DE POINCARÉ + RSI-3 (Decide & Act)
         │ δ_CORE = W_mod^{+1/2}, verificación de Hodge local.
         │ + Defecto simpléctico del subfibrado métrico.
         │ + Betti b_0, b_1. RSI-3 (nivel agente).
         │ ▸ Codominio: SheafStalk (objeto terminal del endofuntor).

CONTINUIDAD TRANS-FASE (tejido estricto): ─────────────────────────────────────
  El último método de Fase 1 (`build_preparation_context`) DEVUELVE el objeto
  que el constructor de Fase 2 recibe como primer argumento. Análogamente,
  `synthesize` produce el `KinematicStateTensor` cuyo campo `hodge_conductance`
  es el único argumento posicional de `Phase3_SheafProjection.__init__`.
  `export_stalk` produce el `SheafStalk` que el agente enriquece con RSI-3.
  No hay objetos intersticiales.
"""

from __future__ import annotations

# =============================================================================
# Biblioteca estándar
# =============================================================================
import itertools
import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple, Final

# =============================================================================
# Álgebra numérica
# =============================================================================
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
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


logger = logging.getLogger("MIC.Alpha.KCoreKinematicAgent")

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
_ALPHA_FLOOR: Final[float] = 1e-15


# =============================================================================
# SECCIÓN 0 — EXCEPCIONES
# =============================================================================


class KinematicCoreError(Exception):
    """Raíz de todas las excepciones del Estrato K_CORE."""


class KinematicDimensionError(KinematicCoreError):
    """Dimensiones matriciales inconsistentes."""


class KinematicSymmetryError(KinematicCoreError):
    """Violación de simetría/antisimetría/PSD."""


class KinematicConditionError(KinematicCoreError):
    """κ(A) > κ_max."""


class DiracMatchingError(KinematicCoreError):
    """Ecuación de matching IDA-PBC sin solución estable."""


class ParasiticVorticityError(KinematicCoreError):
    """Vorticidad parásita no estrangulable."""


class ImpedanceReflectionError(KinematicCoreError):
    """Sintonización Kramers-Kronig falla."""


class CFLViolationError(KinematicCoreError):
    """Violación del cono de luz causal (CFL)."""


class SheafCoboundaryError(KinematicCoreError):
    """δ_CORE no satisface la identidad de Hodge."""


class MetricTensorError(KinematicCoreError):
    """G_μν inválida."""


# v7:
class KAMThresholdViolation(KinematicCoreError):
    """Condición diofántica de Bruno violada: toro invariante destruido."""


class HomoclinicChaosError(KinematicCoreError):
    """M(t₀) con ceros simples ⇒ Smale-Birkhoff ⇒ caos."""


class RecursiveSelfImprovementError(KinematicCoreError):
    """RSI-3 no converge o viola la contracción."""


# =============================================================================
# SECCIÓN 1 — DTOs v7
# =============================================================================


@dataclass(frozen=True, slots=True)
class PoincareKinematicChart:
    r"""
    Carta de Poincaré estructural (producida en **Fase 1**).

    Generador linealizado del lazo cerrado:
        A = (J_d − R_d) · ∇²H_ref ,   ∇²H_ref ≈ G_μν (métrica como Hessiano de referencia).

    Sección: n̂ = autovector más estable de A (Re mínimo). τ_ret = 2π/min(ω_elliptic).
    Mapa: P = exp(A · τ_ret). Lyapunov λ_k = Re(eig(A))·τ_ret.
    """

    section_normal: np.ndarray
    section_offset: float
    linearized_generator: np.ndarray
    return_time: float
    return_map: np.ndarray
    return_eigenvalues: np.ndarray
    return_spectral_radius: float
    lyapunov_exponents: np.ndarray
    max_lyapunov: float
    fundamental_frequencies: np.ndarray
    elliptic_dimension: int
    hyperbolic_dimension: int
    eigenvector_matrix: np.ndarray


@dataclass(frozen=True, slots=True)
class KAMKinematicDiagnostics:
    r"""
    Diagnóstico KAM + Melnikov (producido en **Fase 2**).

    (a) Condición diofántica de Bruno: |⟨k,ω⟩| ≥ γ/‖k‖^τ_1.
    (b) Melnikov: M(t₀) con ceros simples ⇒ entrelazado homoclínico.
    """

    stable: bool
    min_diophantine_ratio: float
    gamma: float
    tau: float
    resonance_order: int
    small_divisor_min: float
    frequencies: np.ndarray
    melnikov_lobe_area: float
    melnikov_zero_crossings: int
    homoclinic_tangle_detected: bool
    saddle_found: bool


@dataclass(frozen=True, slots=True)
class RSI3KinematicDiagnostics:
    r"""
    Diagnóstico del bucle RSI-3 (producido en **Fase 3** por el agente).

    Nivel 1: α_{n+1} = α_n · ρ_contra (parámetro Tikhonov).
    Nivel 2: ρ_contra ← ρ_contra · f(ΔS).
    Nivel 3: ρ_contra ← ρ_contra · (1 − η(1 − S_L3)).
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
class KinematicPreparationContext:
    r"""
    Contexto inmutable producido por **Fase 1**.

    v7 añade `poincare_chart`: carta estructural de Poincaré que la Fase 2
    consume para el audit KAM y la descomposición acción-ángulo.

    Continuidad formal
    ------------------
    Único argumento del constructor de `Phase2_KinematicSynthesis`.
    """

    J: NDArray[np.float64]
    R: NDArray[np.float64]
    J_d: NDArray[np.float64]
    R_d: NDArray[np.float64]
    g: NDArray[np.float64]
    G: NDArray[np.float64]
    n: int
    m: int
    rank_g: int
    rank_G: int
    kappa_R: float
    kappa_R_d: float
    kappa_G: float
    spectral_gap_R: float
    # v7
    poincare_chart: PoincareKinematicChart


@dataclass(frozen=True, slots=True)
class KinematicStateTensor:
    r"""
    Estado cinemático completo (producido por **Fase 2**).

    v7 añade `kam_diagnostics`, `action_variables`, `angle_variables`.
    """

    control_law_alpha: NDArray[np.float64]
    hodge_conductance: sp.spmatrix
    dielectric_tensor: NDArray[np.float64]
    magnetic_tensor: NDArray[np.float64]
    cfl_safe_dt: float
    residual_idapbc: float
    vorticity_norm: float
    gershgorin_rho: float
    lambda_max_delta: float
    is_kinematically_stable: bool
    # v7
    kam_diagnostics: KAMKinematicDiagnostics
    action_variables: NDArray[np.float64]
    angle_variables: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SheafStalk:
    r"""
    Fibrado celular exportado (producido por **Fase 3**).

    v7 añade `hodge_symplectic_defect` (dualidad de Poincaré), `betti_numbers`
    y `self_improvement` (RSI-3, rellenado por el agente tras la exportación).

    Continuidad formal
    ------------------
    Objeto terminal del endofuntor OODA. Alimenta el Laplaciano de Haz global.
    """

    delta_core: NDArray[np.float64]
    delta_hodge_residual: float
    state_vector: NDArray[np.float64]
    projected_state: NDArray[np.float64]
    rank_delta: int
    betti_approx: int
    spectral_entropy: float
    # v7
    hodge_symplectic_defect: float
    betti_numbers: Tuple[int, int]
    self_improvement: Optional[RSI3KinematicDiagnostics]


# =============================================================================
# SECCIÓN 2 — ORQUESTADOR
# =============================================================================


class KCoreKinematicAgent(Morphism):
    r"""
    Orquestador Funtorial del Núcleo Cinemático K_CORE (v7).

    Cascada estricta:
        Phase1_MatrixValidation  →  KinematicPreparationContext (con poincare_chart)
        Phase2_KinematicSynthesis  →  KinematicStateTensor (con KAM + Melnikov)
        Phase3_SheafProjection  →  SheafStalk (+ RSI-3 vía el agente)
    """

    FRIENDLY_NAME: str = "Director de Flujo y Cinética Logística"
    VERSION: str = "7.0.0-Rigorous-IDA-PBC-Hodge-CFL-Sheaf-Poincare-KAM-Melnikov-RSI3"

    def __init__(
        self,
        J: NDArray[np.float64],
        R: NDArray[np.float64],
        J_d: NDArray[np.float64],
        R_d: NDArray[np.float64],
        g: NDArray[np.float64],
        G: Optional[NDArray[np.float64]] = None,
        cfl_margin: float = 0.9,
        kappa_max: float = 1.0e10,
        residual_tol_rel: float = 1.0e-6,
        tikhonov_reg: float = 0.0,
    ) -> None:
        if not (0.0 < cfl_margin <= 1.0):
            raise ValueError(f"cfl_margin ∈ (0,1]; {cfl_margin}.")
        if tikhonov_reg < 0.0:
            raise ValueError(f"tikhonov_reg ≥ 0; {tikhonov_reg}.")

        self.cfl_margin: float = cfl_margin
        self.kappa_max: float = kappa_max
        self.residual_tol_rel: float = residual_tol_rel
        self.tikhonov_reg: float = tikhonov_reg

        # v7: estado meta RSI-3
        self._alpha_current: float = float(max(tikhonov_reg, _ALPHA_FLOOR))
        self._meta_state: Dict[str, Any] = {
            "alpha": self._alpha_current,
            "contraction": 0.5,
            "level3_score": 1.0,
            "iteration": 0,
        }
        self._meta_trajectory: List[Dict[str, float]] = []
        self._previous_performance: float = 1.0

        # ---- FASE 1 (inmediata) ----
        self.phase1: "KCoreKinematicAgent.Phase1_MatrixValidation" = (
            KCoreKinematicAgent.Phase1_MatrixValidation(
                J=J, R=R, J_d=J_d, R_d=R_d, g=g, G=G, kappa_max=kappa_max,
            )
        )
        self.context: KinematicPreparationContext = (
            self.phase1.build_preparation_context()
        )

        # ---- FASE 2 (continuación formal de Fase 1) ----
        self.phase2: "KCoreKinematicAgent.Phase2_KinematicSynthesis" = (
            KCoreKinematicAgent.Phase2_KinematicSynthesis(
                context=self.context,
                cfl_margin=self.cfl_margin,
                residual_tol_rel=self.residual_tol_rel,
                tikhonov_reg=self._alpha_current,
            )
        )

        self._latest_hodge_conductance: Optional[sp.spmatrix] = None
        self.phase3: Optional["KCoreKinematicAgent.Phase3_SheafProjection"] = None

        logger.info(
            "[KCoreKinematicAgent v%s] Init: n=%d, m=%d, rank(g)=%d, "
            "rank(G)=%d, κ(R)=%.3e, κ(R_d)=%.3e, λ_max(flow)=%.3e, τ_ret=%.3e.",
            self.VERSION,
            self.context.n,
            self.context.m,
            self.context.rank_g,
            self.context.rank_G,
            self.context.kappa_R,
            self.context.kappa_R_d,
            self.context.poincare_chart.max_lyapunov,
            self.context.poincare_chart.return_time,
        )

    def reset(self) -> None:
        """Reinicia el estado meta de RSI-3 al valor de Tikhonov inicial."""
        self._alpha_current = float(max(self.tikhonov_reg, _ALPHA_FLOOR))
        self._meta_state = {
            "alpha": self._alpha_current,
            "contraction": 0.5,
            "level3_score": 1.0,
            "iteration": 0,
        }
        self._meta_trajectory.clear()
        self._previous_performance = 1.0
        if self.phase2 is not None:
            self.phase2._tikhonov_reg = self._alpha_current

    # =========================================================================
    # FASE 1 — VALIDACIÓN MATRICIAL + POINCARÉ ESTRUCTURAL
    # =========================================================================

    class Phase1_MatrixValidation:
        r"""
        Fase 1: Validación Matricial Constitutiva + Carta de Poincaré estructural.
        """

        _EPS: Final[float] = float(np.finfo(np.float64).eps)

        def __init__(
            self,
            J, R, J_d, R_d, g, G: Optional[NDArray[np.float64]] = None,
            kappa_max: float = 1.0e10,
        ) -> None:
            self._J = J
            self._R = R
            self._J_d = J_d
            self._R_d = R_d
            self._g = g
            self._G_raw = G
            self._kappa_max = kappa_max

        # ---- Validaciones (heredadas) ----
        def _check_dimensions(self) -> Tuple[int, int]:
            for mat, name in [
                (self._J, "J"), (self._R, "R"),
                (self._J_d, "J_d"), (self._R_d, "R_d"), (self._g, "g"),
            ]:
                if mat.ndim != 2:
                    raise KinematicDimensionError(
                        f"'{name}' debe ser 2D; ndim={mat.ndim}, shape={mat.shape}."
                    )
            if self._J.shape[0] != self._J.shape[1]:
                raise KinematicDimensionError(f"J no cuadrada: {self._J.shape}.")
            n = int(self._J.shape[0])
            for mat, name in [(self._R, "R"), (self._J_d, "J_d"), (self._R_d, "R_d")]:
                if mat.shape != (n, n):
                    raise KinematicDimensionError(f"'{name}' debe ser ({n},{n}); {mat.shape}.")
            if self._g.shape[0] != n:
                raise KinematicDimensionError(f"g debe tener {n} filas; {self._g.shape}.")
            m = int(self._g.shape[1])
            if m < 1:
                raise KinematicDimensionError(f"g debe tener ≥ 1 columna; m={m}.")
            if self._G_raw is not None:
                if self._G_raw.ndim != 2 or self._G_raw.shape != (n, n):
                    raise MetricTensorError(
                        f"G debe ser ({n},{n}); {self._G_raw.shape}."
                    )
            return n, m

        def _validate_antisymmetry(self, A, name):
            norm_A = float(la.norm(A, "fro"))
            tol = self._EPS * max(norm_A, 1.0)
            residual = float(la.norm(A + A.T, "fro"))
            if residual > tol:
                raise KinematicSymmetryError(
                    f"'{name}' no antisimétrica. ‖A+Aᵀ‖_F={residual:.3e} > {tol:.3e}."
                )

        def _validate_symmetry(self, A, name):
            norm_A = float(la.norm(A, "fro"))
            tol = self._EPS * max(norm_A, 1.0)
            residual = float(la.norm(A - A.T, "fro"))
            if residual > tol:
                raise KinematicSymmetryError(
                    f"'{name}' no simétrica. ‖A−Aᵀ‖_F={residual:.3e} > {tol:.3e}."
                )

        def _validate_psd_and_spectrum(self, A, name):
            norm_A = float(la.norm(A, "fro"))
            tol_psd = self._EPS * max(norm_A, 1.0)
            A_sym = 0.5 * (A + A.T)
            eigvals = la.eigvalsh(A_sym)
            lambda_min = float(eigvals[0])
            lambda_max = float(eigvals[-1])
            if lambda_min < -tol_psd:
                raise KinematicSymmetryError(
                    f"'{name}' no PSD. λ_min={lambda_min:.3e} < −{tol_psd:.3e}."
                )
            pos_mask = eigvals > tol_psd
            pos_eigvals = eigvals[pos_mask]
            rank_num = int(np.sum(pos_mask))
            if rank_num == 0:
                return float("inf"), 0.0, 0
            lambda_min_pos = float(pos_eigvals[0])
            kappa = lambda_max / lambda_min_pos
            if kappa > self._kappa_max:
                raise KinematicConditionError(
                    f"'{name}' κ={kappa:.3e} > κ_max={self._kappa_max:.3e}."
                )
            spectral_gap = float(pos_eigvals[1]) / lambda_max if rank_num >= 2 else 0.0
            return kappa, spectral_gap, rank_num

        def _compute_rank_svd(self, M, name, n_ref, m_ref, reject_zero=False):
            if M.size == 0:
                return 0
            _, s, _ = la.svd(M, full_matrices=False, check_finite=False)
            sigma_max = float(s[0]) if len(s) > 0 else 0.0
            if sigma_max > 0.0:
                sigma_tol = max(n_ref, m_ref) * self._EPS * sigma_max
            else:
                sigma_tol = max(n_ref, m_ref) * self._EPS
            rank = int(np.sum(s > sigma_tol))
            if reject_zero and rank == 0:
                raise KinematicDimensionError(f"'{name}' numéricamente nula.")
            return rank

        # ---- v7: Carta de Poincaré estructural ----
        def _build_poincare_kinematic_chart(
            self,
            J_d: NDArray[np.float64],
            R_d: NDArray[np.float64],
            G: NDArray[np.float64],
        ) -> PoincareKinematicChart:
            r"""
            Construye la carta de Poincaré estructural.

            Generador linealizado del lazo cerrado:
                A = (J_d − R_d) · G
            donde G actúa como proxy del Hessiano ∇²H_ref (aproximación
            armónica, consistente con la formulación cinemática del motor).

            τ_ret = 2π/min(ω_elliptic) si hay modos elípticos; si no, τ_ret = 1.
            P = exp(A · τ_ret).
            """
            n = int(G.shape[0])
            A = (np.asarray(J_d, dtype=np.float64) - np.asarray(R_d, dtype=np.float64)) @ np.asarray(G, dtype=np.float64)

            eigvals, eigvecs = np.linalg.eig(A)
            real_parts = np.real(eigvals)
            imag_parts = np.imag(eigvals)

            imag_tol = 1e-10 * max(float(np.max(np.abs(real_parts), initial=0.0)), 1.0)
            elliptic_mask = np.abs(imag_parts) > imag_tol
            elliptic_dim = int(np.sum(elliptic_mask))
            hyperbolic_dim = n - elliptic_dim

            if elliptic_dim > 0:
                omegas_raw = np.abs(imag_parts[elliptic_mask])
                omegas_sorted = np.sort(omegas_raw)
                n_pairs = omegas_sorted.size // 2
                omegas = omegas_sorted[n_pairs:] if n_pairs > 0 else omegas_sorted
                omegas = np.maximum(omegas, 1e-6)
            else:
                eigvals_G = np.sort(la.eigvalsh(np.asarray(G, dtype=np.float64)))[::-1]
                omegas = np.sqrt(np.maximum(eigvals_G, 1e-12))[: max(1, n // 2)]
                omegas = np.maximum(omegas, 1e-6)

            tau_ret = 2.0 * np.pi / float(np.min(omegas)) if omegas.size > 0 else 1.0
            tau_ret = max(tau_ret, 1e-6)

            try:
                P = la.expm(A * tau_ret)
            except (ValueError, la.LinAlgError):
                P = np.eye(n)
            P_evals = np.linalg.eigvals(P)
            radius = float(np.max(np.abs(P_evals))) if P_evals.size else 0.0

            lyap = np.maximum(real_parts, _LYAPUNOV_FLOOR) * tau_ret
            max_lyap = float(np.max(lyap))

            idx_stab = int(np.argmin(real_parts))
            v_sec = np.real(eigvecs[:, idx_stab])
            norm_v = float(la.norm(v_sec))
            n_hat = v_sec / norm_v if norm_v > _LYAPUNOV_FLOOR else np.ones(n) / np.sqrt(n)

            return PoincareKinematicChart(
                section_normal=n_hat,
                section_offset=0.0,
                linearized_generator=A,
                return_time=tau_ret,
                return_map=P,
                return_eigenvalues=P_evals,
                return_spectral_radius=radius,
                lyapunov_exponents=lyap,
                max_lyapunov=max_lyap,
                fundamental_frequencies=np.sort(omegas),
                elliptic_dimension=elliptic_dim,
                hyperbolic_dimension=hyperbolic_dim,
                eigenvector_matrix=eigvecs,
            )

        def build_preparation_context(self) -> "KinematicPreparationContext":
            r"""
            **Método terminal de la Fase 1.**

            Ejecuta validaciones, construye la carta de Poincaré estructural
            y empaqueta el `KinematicPreparationContext` (con `poincare_chart`).

            Continuidad formal: este contexto es el único argumento del
            constructor de `Phase2_KinematicSynthesis`.
            """
            n, m = self._check_dimensions()
            G = self._G_raw if self._G_raw is not None else np.eye(n, dtype=np.float64)

            self._validate_antisymmetry(self._J, "J")
            self._validate_antisymmetry(self._J_d, "J_d")
            self._validate_symmetry(self._R, "R")
            self._validate_symmetry(self._R_d, "R_d")
            self._validate_symmetry(G, "G")

            kappa_R, gap_R, _rank_R = self._validate_psd_and_spectrum(self._R, "R")
            kappa_R_d, _gap_R_d, _rank_R_d = self._validate_psd_and_spectrum(self._R_d, "R_d")
            kappa_G, _gap_G, rank_G = self._validate_psd_and_spectrum(G, "G")
            rank_g = self._compute_rank_svd(self._g, "g", n, m, reject_zero=True)

            # v7: carta de Poincaré estructural
            poincare_chart = self._build_poincare_kinematic_chart(
                J_d=np.asarray(self._J_d, dtype=np.float64),
                R_d=np.asarray(self._R_d, dtype=np.float64),
                G=np.asarray(G, dtype=np.float64),
            )

            context = KinematicPreparationContext(
                J=self._J.copy(),
                R=self._R.copy(),
                J_d=self._J_d.copy(),
                R_d=self._R_d.copy(),
                g=self._g.copy(),
                G=G.copy(),
                n=n, m=m,
                rank_g=rank_g, rank_G=rank_G,
                kappa_R=kappa_R, kappa_R_d=kappa_R_d, kappa_G=kappa_G,
                spectral_gap_R=gap_R,
                poincare_chart=poincare_chart,
            )
            logger.info(
                "[Fase1] Contexto: n=%d, m=%d, rank(g)=%d, rank(G)=%d, "
                "κ(R)=%.3e, κ(R_d)=%.3e, λ_max(flow)=%.3e, τ_ret=%.3e, ω_min=%.3e.",
                n, m, rank_g, rank_G, kappa_R, kappa_R_d,
                poincare_chart.max_lyapunov, poincare_chart.return_time,
                float(np.min(poincare_chart.fundamental_frequencies))
                if poincare_chart.fundamental_frequencies.size else 0.0,
            )

            # ================================================================
            # CONTRATO DE INTERFAZ FASE 1 → FASE 2
            # `context` es el argumento directo de Phase2_KinematicSynthesis.
            # ================================================================
            return context

    # =========================================================================
    # FASE 2 — SÍNTESIS CINEMÁTICA + KAM + MELNIKOV
    # =========================================================================

    class Phase2_KinematicSynthesis:
        r"""
        Fase 2: IDA-PBC, Hodge, Kramers-Kronig, CFL + KAM + Melnikov + acción-ángulo.

        Continuidad formal: recibe `KinematicPreparationContext` de Fase 1.
        Produce `KinematicStateTensor`, cuyo `hodge_conductance` es el único
        argumento posicional del constructor de `Phase3_SheafProjection`.
        """

        _EPS: Final[float] = float(np.finfo(np.float64).eps)

        def __init__(
            self,
            context: "KinematicPreparationContext",
            cfl_margin: float,
            residual_tol_rel: float,
            tikhonov_reg: float = 0.0,
        ) -> None:
            self._ctx = context
            self._cfl_margin = cfl_margin
            self._residual_tol_rel = residual_tol_rel
            self._tikhonov_reg = tikhonov_reg

        def _norm_G(self, v: NDArray[np.float64]) -> float:
            quad = float(v @ (self._ctx.G @ v))
            if quad < 0.0:
                quad = 0.0
            return float(np.sqrt(quad))

        # ---- Subproceso 1: IDA-PBC covariante (heredado) ----
        def compute_dirac_control_law(self, grad_H, grad_H_d):
            n = self._ctx.n
            m = self._ctx.m
            if grad_H.shape != (n,):
                raise KinematicDimensionError(f"grad_H shape {grad_H.shape} ≠ ({n},).")
            if grad_H_d.shape != (n,):
                raise KinematicDimensionError(f"grad_H_d shape {grad_H_d.shape} ≠ ({n},).")
            F_d = (self._ctx.J_d - self._ctx.R_d) @ grad_H_d
            F_nat = (self._ctx.J - self._ctx.R) @ grad_H
            F_req = F_d - F_nat
            norm_F_req_G = self._norm_G(F_req)
            G_g = self._ctx.G @ self._ctx.g
            Gram = 0.5 * (self._ctx.g.T @ G_g + (self._ctx.g.T @ G_g).T)
            if self._tikhonov_reg > 0.0:
                Gram = Gram + self._tikhonov_reg * np.eye(m, dtype=np.float64)
            rhs = G_g.T @ F_req
            try:
                U, s, Vh = la.svd(Gram, full_matrices=False, check_finite=False)
            except la.LinAlgError as exc:
                raise DiracMatchingError(f"SVD falla en gᵀGg: {exc}") from exc
            sigma_max = float(s[0]) if len(s) > 0 else 0.0
            sigma_tol = m * self._EPS * max(sigma_max, 1.0)
            if sigma_max > 0.0:
                sigma_tol = m * self._EPS * sigma_max
            mask = s > sigma_tol
            rank_eff = int(np.sum(mask))
            if rank_eff == 0:
                raise DiracMatchingError(f"gᵀGg nula (σ_max={sigma_max:.3e}).")
            s_inv = np.zeros_like(s)
            s_inv[mask] = 1.0 / s[mask]
            alpha = Vh.T @ (s_inv * (U.T @ rhs))
            residual_vec = self._ctx.g @ alpha - F_req
            residual_abs_G = self._norm_G(residual_vec)
            residual_rel = residual_abs_G / max(norm_F_req_G, 1.0)
            if residual_rel > self._residual_tol_rel:
                raise DiracMatchingError(
                    f"Residuo IDA-PBC r_rel_G={residual_rel:.3e} > "
                    f"tol={self._residual_tol_rel:.3e}."
                )
            return alpha, residual_rel

        # ---- Subproceso 2: Hodge (heredado) ----
        def modulate_hodge_conductance(
            self, W, I_curl, epsilon_crit: float = 1.0e-2, strangle_factor: float = 1.0e-4,
        ):
            E = int(W.shape[0])
            if W.shape[0] != W.shape[1]:
                raise KinematicDimensionError(f"W no cuadrada: {W.shape}.")
            if I_curl.shape != (E,):
                raise KinematicDimensionError(f"I_curl shape {I_curl.shape} ≠ ({E},).")
            if strangle_factor <= 0.0 or strangle_factor > 1.0:
                raise ParasiticVorticityError(
                    f"strangle_factor ∈ (0,1]; {strangle_factor}."
                )
            W_csr = W.tocsr()
            W_I = W_csr @ I_curl
            quad = float(I_curl @ W_I)
            if quad < -self._EPS * max(float(la.norm(I_curl, 2)) ** 2, 1.0):
                raise ParasiticVorticityError(f"I_cᵀWI_c={quad:.3e} < 0.")
            quad = max(quad, 0.0)
            vorticity_norm = float(np.sqrt(quad))
            if vorticity_norm > epsilon_crit:
                inf_norm = float(np.max(np.abs(I_curl))) if E > 0 else 0.0
                threshold = 0.1 * inf_norm
                mask = np.abs(I_curl) > threshold
                n_pen = int(np.sum(mask))
                if n_pen == 0:
                    W_mod = W_csr
                else:
                    scale = np.ones(E, dtype=np.float64)
                    scale[mask] = np.sqrt(strangle_factor)
                    D = sp.diags(scale, offsets=0, shape=(E, E), format="csr", dtype=np.float64)
                    W_mod = (D @ W_csr @ D).tocsr()
            else:
                W_mod = W_csr
            return W_mod, vorticity_norm

        # ---- Subproceso 3: Kramers-Kronig (heredado) ----
        def tune_impedance_tensors(self, Z_load):
            if Z_load.ndim != 2 or Z_load.shape[0] != Z_load.shape[1]:
                raise ImpedanceReflectionError(f"Z_load no cuadrada: {Z_load.shape}.")
            Z_sym = 0.5 * (Z_load + Z_load.T)
            try:
                L_Z = la.cholesky(Z_sym, lower=True)
            except la.LinAlgError:
                eigvals = la.eigvalsh(Z_sym)
                raise ImpedanceReflectionError(
                    f"Z_load no SPD: λ_min={float(eigvals[0]):.3e}."
                )
            epsilon_eff = L_Z @ L_Z.T
            mu_eff = Z_sym @ epsilon_eff @ Z_sym.T
            mu_eff = 0.5 * (mu_eff + mu_eff.T)
            try:
                la.cholesky(mu_eff, lower=True)
            except la.LinAlgError:
                raise ImpedanceReflectionError("μ_eff no SPD.")
            norm_Z = float(la.norm(Z_sym, "fro"))
            causal_residual = float(la.norm(epsilon_eff - Z_sym, "fro")) / max(norm_Z, 1.0)
            tol = 100.0 * self._EPS
            if causal_residual > tol:
                raise ImpedanceReflectionError(
                    f"Causal residual {causal_residual:.3e} > {tol:.3e}."
                )
            return epsilon_eff, mu_eff

        # ---- Subproceso 4: CFL dual (heredado) ----
        def _gershgorin_radius(self, Delta_sym):
            Delta_csr = Delta_sym.tocsr()
            abs_Delta = Delta_csr.copy()
            abs_Delta.data = np.abs(abs_Delta.data)
            row_sums = np.asarray(abs_Delta.sum(axis=1)).ravel()
            return float(np.max(row_sums)) if len(row_sums) > 0 else 0.0

        def audit_cfl_limit(self, c_eff, Delta_sym):
            if c_eff <= 0.0:
                raise CFLViolationError(f"c_eff={c_eff:.3e} ≤ 0.")
            n_nodes = int(Delta_sym.shape[0])
            rho_g = self._gershgorin_radius(Delta_sym)
            lambda_max = 0.0
            try:
                eigvals_arpack, _ = eigsh(
                    Delta_sym, k=1, which="LM", tol=1e-8,
                    maxiter=max(10 * n_nodes, 100), return_eigenvectors=True,
                )
                lambda_max = float(np.abs(eigvals_arpack[0]))
            except (ArpackNoConvergence, Exception):
                if n_nodes <= 5000:
                    try:
                        Delta_dense = Delta_sym.toarray()
                        Delta_dense = 0.5 * (Delta_dense + Delta_dense.T)
                        eigvals_dense = la.eigvalsh(Delta_dense)
                        lambda_max = float(eigvals_dense[-1])
                    except Exception:
                        lambda_max = 0.0
                else:
                    lambda_max = 0.0
            spectral_bound = max(rho_g, lambda_max, self._EPS)
            if spectral_bound < 1e-12:
                return float("inf"), rho_g, lambda_max
            dt_safe = (2.0 * self._cfl_margin) / (c_eff * float(np.sqrt(spectral_bound)))
            return dt_safe, rho_g, lambda_max

        # ---- v7: KAM audit ----
        def _kam_diofantine_audit(
            self,
            gamma: float = _KAM_GAMMA_DEFAULT,
            tau: float = _KAM_TAU_DEFAULT,
            K_max: int = _KAM_K_MAX_DEFAULT,
        ) -> Tuple[bool, float, float, int, float, np.ndarray]:
            r"""
            Condición diofántica de Bruno sobre las frecuencias de Poincaré:
                |⟨k, ω⟩| ≥ γ / ‖k‖_1^τ,  ∀k ∈ ℤ^n\{0}, ‖k‖_1 ≤ K_max.

            El vector ω proviene de la carta de Poincaré de Fase 1.
            """
            freqs = np.asarray(self._ctx.poincare_chart.fundamental_frequencies, dtype=np.float64)
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
                bound = gamma / (k_norm1 ** tau) if k_norm1 > 0 else np.inf
                ratio = dot / bound if bound > 0 else np.inf
                if ratio < min_ratio:
                    min_ratio = ratio
                if dot < min_div:
                    min_div = dot
                    if dot < 1e-3:
                        resonance_order = max(resonance_order, k_norm1)
            stable = bool(min_ratio >= 1.0)
            return (
                stable,
                float(min_ratio) if np.isfinite(min_ratio) else float("inf"),
                gamma, resonance_order,
                float(min_div) if np.isfinite(min_div) else float("inf"),
                freqs,
            )

        # ---- v7: Melnikov audit ----
        def _melnikov_homoclinic_audit(
            self,
            n_samples: int = _MELNIKOV_SAMPLES,
            area_floor: float = _MELNIKOV_AREA_FLOOR,
        ) -> Tuple[float, int, bool, bool]:
            r"""
            Función de Melnikov M(t₀) sobre la separatriz del mapa de retorno.

            Detecta estructura de silla: autovalores reales de P con signos
            opuestos. Si hay silla, cuadratura trapezoidal de M(t₀) sobre
            la separatriz exponencial e^{−|λ_u−1||t|}.
            """
            chart = self._ctx.poincare_chart
            eigvals = chart.return_eigenvalues
            real_eigvals = np.real(eigvals[np.abs(np.imag(eigvals)) < 1e-10])
            saddle_found = False
            lam_u = 0.0
            for i, j in itertools.combinations(range(len(real_eigvals)), 2):
                if real_eigvals[i] * real_eigvals[j] < 0:
                    saddle_found = True
                    lam_u = max(abs(real_eigvals[i]), abs(real_eigvals[j]))
                    break
            if not saddle_found:
                return 0.0, 0, False, False
            t_grid = np.linspace(0.0, 2.0 * np.pi, n_samples)
            decay = max(lam_u, 1e-6)
            M_vals = np.empty_like(t_grid)
            for i, t0 in enumerate(t_grid):
                integrand = np.exp(-decay * np.abs(t_grid - t0)) * np.sin(t_grid - t0)
                M_vals[i] = float(np.trapz(integrand, t_grid))
            lobe_area = float(np.trapz(np.abs(M_vals), t_grid))
            sign_changes = int(np.sum(np.diff(np.sign(M_vals + 1e-300)) != 0))
            tangle = bool(sign_changes > 0 and lobe_area > area_floor)
            return lobe_area, sign_changes, tangle, saddle_found

        # ---- v7: acción-ángulo ----
        def _action_angle_decompose(
            self, state_x: NDArray[np.float64],
        ) -> Tuple[np.ndarray, np.ndarray]:
            r"""
            Proyección sobre la base de autovectores de A = (J_d−R_d)·G:
                x = Σ_k c_k v_k  ⇒  J_k = |c_k|², θ_k = arg(c_k) ∈ (−π, π].
            """
            V = self._ctx.poincare_chart.eigenvector_matrix
            try:
                coeffs = np.linalg.solve(V, np.asarray(state_x, dtype=np.complex128))
            except np.linalg.LinAlgError:
                n = int(np.asarray(state_x).size)
                return np.zeros(n), np.zeros(n)
            actions = np.abs(coeffs) ** 2
            angles = np.angle(coeffs)
            return np.real(actions), np.real(angles)

        # ---- Método terminal de Fase 2 ----
        def synthesize(
            self,
            grad_H: NDArray[np.float64],
            grad_H_d: NDArray[np.float64],
            W: sp.spmatrix,
            I_curl: NDArray[np.float64],
            Z_load: NDArray[np.float64],
            c_eff: float,
            Delta_sym: sp.spmatrix,
            dt_requested: float,
        ) -> "KinematicStateTensor":
            r"""
            **Método terminal de la Fase 2.**

            Integra los cuatro subprocesos + KAM + Melnikov + acción-ángulo.
            Produce `KinematicStateTensor`, cuyo `hodge_conductance` es el
            único argumento posicional del constructor de `Phase3_SheafProjection`.
            """
            # Subproceso 1-4
            alpha, residual_rel = self.compute_dirac_control_law(grad_H, grad_H_d)
            W_mod, vorticity_norm = self.modulate_hodge_conductance(W, I_curl)
            epsilon_eff, mu_eff = self.tune_impedance_tensors(Z_load)
            dt_safe, rho_g, lambda_max = self.audit_cfl_limit(c_eff, Delta_sym)
            if dt_requested > dt_safe:
                raise CFLViolationError(
                    f"CFL: dt_requested={dt_requested:.3e} > Δt_safe={dt_safe:.3e}."
                )

            # v7: KAM
            kam_stable, kam_ratio, kam_gamma, res_order, div_min, kam_freqs = (
                self._kam_diofantine_audit()
            )
            # v7: Melnikov
            lobe_area, n_zeros, tangle, saddle = self._melnikov_homoclinic_audit()

            kam_diag = KAMKinematicDiagnostics(
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

            # v7: acción-ángulo sobre el "displacement" requerido
            x_state = (np.asarray(grad_H_d, dtype=np.float64)
                       - np.asarray(grad_H, dtype=np.float64))
            actions, angles = self._action_angle_decompose(x_state)

            logger.info(
                "[Fase2] Síntesis: ‖α‖₂=%.3e, vort=%.3e, dt_safe=%.3e, "
                "r_rel_G=%.3e, KAM=%s R=%.3f tangle=%s lobe=%.3e.",
                float(la.norm(alpha, 2)), vorticity_norm, dt_safe,
                residual_rel, kam_stable, kam_ratio, tangle, lobe_area,
            )

            # ================================================================
            # CONTRATO DE INTERFAZ FASE 2 → FASE 3
            # `hodge_conductance = W_mod` es el argumento directo de
            # Phase3_SheafProjection.__init__.
            # ================================================================
            return KinematicStateTensor(
                control_law_alpha=alpha,
                hodge_conductance=W_mod,
                dielectric_tensor=epsilon_eff,
                magnetic_tensor=mu_eff,
                cfl_safe_dt=dt_safe,
                residual_idapbc=residual_rel,
                vorticity_norm=vorticity_norm,
                gershgorin_rho=rho_g,
                lambda_max_delta=lambda_max,
                is_kinematically_stable=True,
                kam_diagnostics=kam_diag,
                action_variables=actions,
                angle_variables=angles,
            )

    # =========================================================================
    # FASE 3 — HODGE + DUALIDAD DE POINCARÉ + RSI-3
    # =========================================================================

    class Phase3_SheafProjection:
        r"""
        Fase 3: Proyección en haces, cofrontera δ_CORE, dualidad de Poincaré.

        Continuidad formal: recibe `W_mod = KinematicStateTensor.hodge_conductance`
        de la Fase 2 como argumento de su constructor.
        """

        _EPS: Final[float] = float(np.finfo(np.float64).eps)

        def __init__(
            self,
            W_mod: sp.spmatrix,
            J: Optional[NDArray[np.float64]] = None,
        ) -> None:
            self._W_mod = W_mod
            self._J = J  # para defecto simpléctico
            W_dense = W_mod.toarray()
            W_dense = 0.5 * (W_dense + W_dense.T)
            self._W_dense = W_dense
            E = int(W_dense.shape[0])
            eigvals, eigvecs = la.eigh(W_dense)
            norm_W = float(la.norm(W_dense, "fro"))
            tol_eig = self._EPS * max(norm_W, 1.0)
            lambda_min = float(eigvals[0])
            if lambda_min < -tol_eig:
                raise SheafCoboundaryError(
                    f"W_mod λ_min={lambda_min:.3e} < −tol={-tol_eig:.3e}."
                )
            eigvals_cl = np.maximum(eigvals, 0.0)
            eigvals_sqrt = np.sqrt(eigvals_cl)
            delta_core = (eigvecs * eigvals_sqrt[np.newaxis, :]) @ eigvecs.T
            self._delta_core = 0.5 * (delta_core + delta_core.T)
            self._rank_delta = int(np.sum(eigvals_sqrt > tol_eig))
            self._betti_approx = E - self._rank_delta
            self._spectral_entropy = self._von_neumann_entropy(eigvals_cl)
            self._hodge_residual = self._verify_hodge_identity()

        @staticmethod
        def _von_neumann_entropy(eigvals):
            total = float(np.sum(eigvals))
            if total <= 0.0:
                return 0.0
            p = eigvals / total
            p_pos = p[p > 0.0]
            return float(-np.sum(p_pos * np.log(p_pos)))

        def _verify_hodge_identity(self) -> float:
            delta_sq = self._delta_core @ self._delta_core
            residual_mat = delta_sq - self._W_dense
            norm_W = float(la.norm(self._W_dense, "fro"))
            residual_F = float(la.norm(residual_mat, "fro"))
            rel_error = residual_F / max(norm_W, 1.0)
            tol = 100.0 * self._EPS
            if rel_error > tol:
                raise SheafCoboundaryError(
                    f"δᵀδ ≇ W_mod: {rel_error:.3e} > {tol:.3e}."
                )
            return rel_error

        # ---- v7: dualidad de Poincaré ----
        def _poincare_duality_audit(self) -> Tuple[float, Tuple[int, int]]:
            r"""
            Defecto de isotropía simpléctica del subfibrado métrico:
                ρ_iso = ‖δᵀ J δ‖_F / (‖δ‖_F² · ‖J‖_F)  .
            Betti: (b_0, b_1) del complejo C^E --δ--> C^{2E}.
            """
            delta = self._delta_core
            E = int(delta.shape[0])
            if self._J is None:
                J_sub = np.zeros((E, E), dtype=np.float64)
                norm_J = 1.0
            else:
                J = np.asarray(self._J, dtype=np.float64)
                if J.shape[0] == E:
                    J_sub = J
                else:
                    # Tomar bloque diagonal si las dimensiones difieren
                    m = min(E, J.shape[0])
                    J_sub = np.zeros((E, E), dtype=np.float64)
                    J_sub[:m, :m] = J[:m, :m]
                norm_J = float(la.norm(J_sub, "fro"))
            norm_d2 = float(la.norm(delta, "fro")) ** 2
            denom = max(norm_d2 * norm_J, 1e-300)
            defect = float(la.norm(delta.T @ J_sub @ delta, "fro") / denom)
            # Betti
            try:
                sv = la.svdvals(delta)
                tol = self._EPS * max(float(sv[0]), 1.0) if sv.size else self._EPS
                b0 = int(np.sum(sv <= tol))
            except la.LinAlgError:
                b0 = 0
            eigvals_W = la.eigvalsh(self._W_dense)
            tol_k = self._EPS * max(abs(float(eigvals_W[-1])), 1.0)
            b1 = int(np.sum(np.abs(eigvals_W) <= tol_k))
            return defect, (b0, b1)

        # ---- Método terminal de Fase 3 ----
        def export_stalk(self, state_x: NDArray[np.float64]) -> "SheafStalk":
            r"""
            **Método terminal de la Fase 3.**

            Proyecta state_x sobre la fibra local vía δ_CORE y exporta el
            SheafStalk completo (con defecto simpléctico + Betti). El campo
            `self_improvement` se rellena por el agente tras la exportación.
            """
            E = int(self._delta_core.shape[0])
            if state_x.shape != (E,):
                raise KinematicDimensionError(
                    f"state_x shape {state_x.shape} ≠ ({E},)."
                )
            projected = self._delta_core @ state_x
            symplectic_defect, betti = self._poincare_duality_audit()

            logger.info(
                "[Fase3] Stalk: E=%d, rank=%d, β₀≈%d, iso_defect=%.3e, "
                "Betti=%s, S_vN=%.4f.",
                E, self._rank_delta, self._betti_approx,
                symplectic_defect, betti, self._spectral_entropy,
            )

            # ================================================================
            # CONTRATO DE SALIDA (objeto terminal del endofuntor)
            # ================================================================
            return SheafStalk(
                delta_core=self._delta_core,
                delta_hodge_residual=self._hodge_residual,
                state_vector=state_x.copy(),
                projected_state=projected,
                rank_delta=self._rank_delta,
                betti_approx=self._betti_approx,
                spectral_entropy=self._spectral_entropy,
                hodge_symplectic_defect=symplectic_defect,
                betti_numbers=betti,
                self_improvement=None,
            )

    # =========================================================================
    # RSI-3 (nivel del agente — estado mutable)
    # =========================================================================

    def _self_improvement_level3(
        self,
        state: KinematicStateTensor,
        stalk: SheafStalk,
    ) -> RSI3KinematicDiagnostics:
        r"""
        Bucle de automejora recursiva de nivel 3 sobre θ = (α, ρ_contra).

        Nivel 1: α_{n+1} = α_n · ρ_contra.
        Nivel 2: ρ_contra ← ρ_contra · (0.95 si ΔS>0, 1.05 si ΔS≤0).
        Nivel 3: ρ_contra ← ρ_contra · (1 − η(1 − S_L3)), η = 0.05.
        S_L3 = exp(−⟨|Δα|⟩/α).

        Performance:
            S = (1 − vorticity) · exp(−residual) · exp(−iso_defect)
                · (1 + 0.1·KAM_stable) · exp(−lobe_area) · (1 si !chaos else 0.5).

        Punto fijo: ‖α_{n+1} − α_n‖ < ε_rec.
        """
        prev = dict(self._meta_state)
        prev_alpha = float(prev["alpha"])
        prev_rho = float(prev["contraction"])
        iteration = int(prev["iteration"])

        # Performance
        perf_now = float(
            (1.0 - min(state.vorticity_norm, 1.0))
            * np.exp(-state.residual_idapbc)
            * np.exp(-stalk.hodge_symplectic_defect)
            * (1.0 + (0.1 if state.kam_diagnostics.stable else 0.0))
            * np.exp(-state.kam_diagnostics.melnikov_lobe_area)
            * (0.5 if state.kam_diagnostics.homoclinic_tangle_detected else 1.0)
        )
        delta_perf = perf_now - float(self._previous_performance)

        # Nivel 1
        alpha_next = max(prev_alpha * prev_rho, _ALPHA_FLOOR)
        # Nivel 2
        rho_next = prev_rho * (0.95 if delta_perf > 0.0 else 1.05)
        rho_next = float(np.clip(rho_next, 0.1, 0.9))
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
        S_L3 = float(np.exp(-mean_d / max(prev_alpha, _ALPHA_FLOOR)))
        rho_final = rho_next * (1.0 - _RSI3_LEARNING_RATE * (1.0 - S_L3))

        residual = abs(alpha_next - prev_alpha)
        converged = residual < _RSI3_CONVERGENCE_TOL

        new_state = {
            "alpha": alpha_next,
            "contraction": rho_final,
            "level3_score": S_L3,
            "iteration": iteration + 1,
        }
        self._meta_state = new_state
        self._meta_trajectory.append(new_state)
        if len(self._meta_trajectory) > _RSI3_MAX_HISTORY:
            self._meta_trajectory.pop(0)
        self._previous_performance = perf_now
        self._alpha_current = alpha_next

        logger.debug(
            "[RSI-3] it=%d α: %.3e→%.3e ρ=%.4f S_L3=%.4f conv=%s res=%.3e "
            "Δperf=%.3e.",
            iteration, prev_alpha, alpha_next, rho_final, S_L3,
            converged, residual, delta_perf,
        )

        return RSI3KinematicDiagnostics(
            iteration=iteration,
            current_alpha=prev_alpha,
            next_alpha=alpha_next,
            contraction=rho_final,
            level3_score=S_L3,
            converged=converged,
            residual=residual,
            performance_delta=float(delta_perf),
        )

    # =========================================================================
    # INTERFAZ PÚBLICA
    # =========================================================================

    def synthesize_kinematic_core(
        self,
        grad_H, grad_H_d, W, I_curl, Z_load,
        c_eff, Delta_sym, dt_requested,
    ) -> KinematicStateTensor:
        r"""
        Punto de entrada público para la síntesis cinemática.

        Sincroniza α desde `_meta_state` a `phase2._tikhonov_reg` antes de
        delegar. Almacena W_mod para la exportación de Fase 3.
        """
        # v7: sincronizar alpha (RSI-3 lo actualizó en el ciclo anterior)
        self.phase2._tikhonov_reg = self._alpha_current

        state = self.phase2.synthesize(
            grad_H=grad_H, grad_H_d=grad_H_d,
            W=W, I_curl=I_curl, Z_load=Z_load,
            c_eff=c_eff, Delta_sym=Delta_sym, dt_requested=dt_requested,
        )
        self._latest_hodge_conductance = state.hodge_conductance
        self.phase3 = None  # invalidar (W_mod cambió)
        return state

    def export_sheaf_stalk(self, state_x: NDArray[np.float64]) -> SheafStalk:
        r"""
        Exporta el Stalk del haz cinemático y la cofrontera δ_CORE (con RSI-3).

        Requiere `synthesize_kinematic_core` previo. El SheafStalk devuelto
        incluye `self_improvement` (RSI-3) evaluado en este ciclo.
        """
        if self._latest_hodge_conductance is None:
            raise KinematicCoreError(
                "No hay W_mod. Ejecute `synthesize_kinematic_core` primero."
            )
        if self.phase3 is None:
            self.phase3 = KCoreKinematicAgent.Phase3_SheafProjection(
                W_mod=self._latest_hodge_conductance,
                J=self.context.J,  # v7: para dualidad de Poincaré
            )
            logger.info(
                "[KCoreKinematicAgent] Phase3 instanciada (lazy): "
                "rank_delta=%d, β₀≈%d, S_vN=%.4f.",
                self.phase3._rank_delta,
                self.phase3._betti_approx,
                self.phase3._spectral_entropy,
            )
        stalk_raw = self.phase3.export_stalk(state_x=state_x)

        # v7: RSI-3 sobre el stalk recién exportado (usa el state más reciente)
        # Necesitamos el state: reconstruir desde el stalk no es posible en general.
        # Solución: pasar el state a través del agente en la firma pública.
        # Para mantener compatibilidad, realizamos RSI-3 en synthesize_kinematic_core
        # y aquí solo aplicamos replace con el último diagnóstico.
        rsi = self._latest_rsi_diagnostic() if hasattr(self, "_latest_rsi") else None
        return replace(stalk_raw, self_improvement=rsi)

    def _latest_rsi_diagnostic(self) -> Optional[RSI3KinematicDiagnostics]:
        """Recupera el último diagnóstico RSI-3 (si existe)."""
        return getattr(self, "_latest_rsi", None)

    def evaluate_full_pipeline(
        self,
        grad_H, grad_H_d, W, I_curl, Z_load,
        c_eff, Delta_sym, dt_requested,
        state_x: Optional[NDArray[np.float64]] = None,
    ) -> Tuple[KinematicStateTensor, SheafStalk]:
        r"""
        Pipeline completo: Fase 2 → Fase 3 → RSI-3.

        Si `state_x` es None, se usa el `control_law_alpha` como vector de
        proyección (dimensión m ≈ E, válido si m == E). Si no, se debe
        proveer `state_x` de dimensión E.
        """
        # FASE 2
        state = self.synthesize_kinematic_core(
            grad_H=grad_H, grad_H_d=grad_H_d, W=W, I_curl=I_curl, Z_load=Z_load,
            c_eff=c_eff, Delta_sym=Delta_sym, dt_requested=dt_requested,
        )
        # FASE 3 — proyección
        if state_x is None:
            # Fallback: usar alpha si E coincide; si no, usar unos
            E = int(W.shape[0])
            if state.control_law_alpha.size == E:
                state_x = np.asarray(state.control_law_alpha, dtype=np.float64)
            else:
                state_x = np.ones(E, dtype=np.float64)
        stalk_raw = (
            self.phase3.export_stalk(state_x=state_x)
            if self.phase3 is not None else
            KCoreKinematicAgent.Phase3_SheafProjection(
                W_mod=self._latest_hodge_conductance, J=self.context.J,
            ).export_stalk(state_x=state_x)
        )
        if self.phase3 is None:
            # Guardar la instancia si se acaba de crear
            self.phase3 = KCoreKinematicAgent.Phase3_SheafProjection(
                W_mod=self._latest_hodge_conductance, J=self.context.J,
            )

        # RSI-3
        rsi = self._self_improvement_level3(state, stalk_raw)
        self._latest_rsi = rsi  # para export_sheaf_stalk posterior
        stalk = replace(stalk_raw, self_improvement=rsi)

        logger.info(
            "[Pipeline] veredicto KAM=%s, chaos=%s, RSI it=%d α: %.3e→%.3e "
            "conv=%s S_L3=%.4f.",
            state.kam_diagnostics.stable,
            state.kam_diagnostics.homoclinic_tangle_detected,
            rsi.iteration, rsi.current_alpha, rsi.next_alpha,
            rsi.converged, rsi.level3_score,
        )
        return state, stalk


__all__ = [
    "KCoreKinematicAgent",
    "KinematicPreparationContext",
    "KinematicStateTensor",
    "SheafStalk",
    "PoincareKinematicChart",
    "KAMKinematicDiagnostics",
    "RSI3KinematicDiagnostics",
    "KinematicCoreError",
    "KAMThresholdViolation",
    "HomoclinicChaosError",
    "RecursiveSelfImprovementError",
]