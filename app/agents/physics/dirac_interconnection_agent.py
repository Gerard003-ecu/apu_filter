# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Dirac Interconnection Agent — Aduana Poincaré-PHS 7.1                           ║
║ Ruta   : app/agents/physics/dirac_interconnection_agent.py                               ║
║ Versión: 7.1.0-Poincare-DEC-PHS-Rigorous                                                 ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

MARCO MATEMÁTICO Y TEÓRICO RIGUROSO DE INTERCONEXIÓN DE DIRAC
─────────────────────────────────────────────────────────────
1. Estructuras de Dirac y Subespacios de Interconexión:
   Dado el espacio de estados $x \in \mathcal{M} \cong \mathbb{R}^n$, con esfuerzos $e \in E = T^*\mathcal{M}$ y flujos $f \in F = T\mathcal{M}$, se define la estructura de Dirac $\mathcal{D} \subset F \times E$ maximofóbica e isótropa respecto a la forma bilineal simétrica acoplada $\langle (f_1, e_1), (f_2, e_2) \rangle_+ = \langle e_1, f_2 \rangle + \langle e_2, f_1 \rangle$.
   En la formulación Port-Hamiltoniana, la dinámica satisface $\dot{x} = (J - R)\nabla H(x) + g u$ y $y = g^{\top} \nabla H(x)$, donde $J \in \mathfrak{so}(n)$ y $R \in \operatorname{Sym}^+(n)$.

2. Fases Anidadas de la Aduana Causal:
   • Fase $\phi_1$ (Matching y Power-Shaping):
     Resuelve el problema de alineación algebraica $(J_d - R_d)\nabla H_d = (J - R)\nabla H + g \alpha$.
     Construye la terminación de puerto $u = -K y + u_{\text{ff}}$ manteniendo inmunidad sobre los invariantes de Casimir $C^{\top} g = 0$.
     Último método de $\phi_1$: `Phase1_IDAPBC_PoincareSolver.compute_port_termination(...) -> PortTermination`.

   • Fase $\phi_2$ (Dispersión de Puerto y Geometría Adaptativa):
     Calcula la matriz de dispersión (scattering) $\Gamma = (K - Z_0^{-1})(K + Z_0^{-1})^{-1}$ respecto de la impedancia característica física $Z_0 = \sqrt{\mu/\varepsilon}$ sin alterar los operadores de Hodge $\star_k$.
     Determina la velocidad de onda del medio $c = \frac{1}{\sqrt{\varepsilon \mu}}$.
     Último método de $\phi_2$: `Phase2_PortScattering.compute_causal_speed(...) -> CausalSpeed`.

   • Fase $\phi_3$ (Gobernanza Causal y Cono de Estabilidad Spectal):
     Evalúa el paso de integración seguro $\Delta t$ garantizando estabilidad según la norma logarítmica $\mu_2(A) = \lambda_{\max}\left(\frac{A + A^{\top}}{2}\right)$ para el integrador de punto medio implícito, o la condición CFL $\Delta t < \frac{2}{c \sqrt{\rho(\Delta_1)}}$ en reticulados Yee.
     Último método de $\phi_3$: `Phase3_CFLGovernor.synthesize_interconnection_state(...) -> InterconnectionState`.

3. Invariantes de Control y Preservación Geométrico-Topológica (I1-I8):
   I1 (Dirac): $J, J_d \in \mathfrak{so}(n)$.
   I2 (Rayleigh): $R, R_d \in \operatorname{Sym}^+(n)$.
   I3 (Lyapunov): $\dot{H}_d = -\nabla H_d^{\top} R_d \nabla H_d + \nabla H_d^{\top} (f_d - g \alpha) \le 0$.
   I4 (Herglotz): Consistencia disipativa para tensores de terminación dinámica.
   I5 (CFL / Transitorio): A-estabilidad con acotamiento de la norma logarítmica transitoria $\mu_2(A)$.
   I6 (Asignabilidad de Equilibrio): $\nabla H_d(x^*) = 0$.
   I7 (La Salle): Estabilidad asintótica módulo la variedad de Casimirs $\ker J$.
   I8 (Jacobi-Maupertuis): Métrica conforme de Jacobi aplicable exclusivamente bajo descomposición canónica $T + V$.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

try:
    import scipy.linalg as la
    from scipy.sparse import csr_matrix, issparse
    from scipy.sparse.linalg import eigsh
    _SCIPY = True
except ImportError:  # pragma: no cover
    la = None  # type: ignore[assignment]
    csr_matrix = None  # type: ignore[assignment]
    issparse = lambda _x: False  # type: ignore[misc]
    eigsh = None
    _SCIPY = False

try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:  # pragma: no cover

    class TopologicalInvariantError(Exception):
        """Invariante topológico / geométrico violado."""

    class Morphism:
        """Morfismo C → D (stub). Las subclases concretas implementan __call__."""

        def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            raise NotImplementedError("Morphism stub.")

try:
    from app.core.immune_system.metric_tensors import G_PHYSICS
except ImportError:  # pragma: no cover
    G_PHYSICS = np.eye(1, dtype=np.float64)

try:
    from app.physics.flux_condenser import (
        CONSTANTS as FC_CONSTANTS,
        PoincareControlSeed,
        PoincareEngineSeed,
        PoincareHamiltonianKernel,
        ControlMode,
    )
    _HAS_FOSO = True
except ImportError:  # pragma: no cover
    FC_CONSTANTS = None
    PoincareControlSeed = None  # type: ignore[misc, assignment]
    PoincareEngineSeed = None  # type: ignore[misc, assignment]
    PoincareHamiltonianKernel = None  # type: ignore[misc, assignment]
    ControlMode = None  # type: ignore[misc, assignment]
    _HAS_FOSO = False

logger = logging.getLogger("MIC.Physics.DiracInterconnection.Poincare71")

Array = NDArray[np.float64]
MaybeSeed = Any

# ═══════════════════════════════════════════════════════════════════════════════════════
# CONSTANTES (no mezclar con las de DEC/PHS del foso)
# ═══════════════════════════════════════════════════════════════════════════════════════

_EPS = float(np.finfo(np.float64).eps)
_WILKINSON = max(1e-12, 16.0 * _EPS)
_SPECTRAL_TOL = 1e-9
_SYMPLECTIC_TOL = 1e-8
_RANK_REL_TOL = 1e-10
_POWER_RES_TOL = 1e-8
_MAUPERTUIS_FLOOR = 1e-12
_DT_MIN = 1e-6
_DT_MAX = 3600.0
_HODGE_CANON = "D=ε★₁E, H=μ⁻¹★₂B, δ₂=★₁⁻¹∂₂★₂"
_SCHEMA = "7.1.0"


class MatchingMode(str, Enum):
    """Ley que φ₁ realiza."""

    ENERGY_LEVEL = "energy_level"     # P_* = ∇HᵀR∇H − λ(H−H*),  u = P_* y / (‖y‖²+ε)
    IDA_PBC_POINT = "ida_pbc_point"   # u = Gx + v,  matching lineal heredado
    DAMPING_ONLY = "damping_only"     # u = −k_d y  (prohibido si pumping_required)
    SNAPSHOT = "snapshot"             # α = g⁺ f_d en el instante (auditoría, no ley)


# ═══════════════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES
# ═══════════════════════════════════════════════════════════════════════════════════════


class DiracMatchingError(TopologicalInvariantError):
    """f_d ∉ Im(g) más allá de tolerancia, o estructura (J,R,g) inadmisible."""


class ImpedanceMismatchError(TopologicalInvariantError):
    """Terminación de puerto / scattering inadmisible (no es un fallo de ★)."""


class CFLViolationError(TopologicalInvariantError):
    """Cono causal: Yee fuera de CFL, o μ₂(A)>0 con paso explícito."""


class LyapunovInstabilityError(TopologicalInvariantError):
    """Ḣ_d > 0 tras contar el residuo de matching."""


class PoincareSymplecticError(TopologicalInvariantError):
    """J no antisimétrica, o Δ no simetrizable."""


class MaupertuisViolationError(TopologicalInvariantError):
    """Se pidió Jacobi sin escisión T+V, o f_M ≤ 0."""


class EnergyMatchingError(TopologicalInvariantError):
    """∇H_d(x*) ≠ 0 (equilibrio no asignable)."""


class LanczosConvergenceError(TopologicalInvariantError):
    """Los estimadores de λ_max(Δ) no son consistentes."""


class CasimirPortLeakError(TopologicalInvariantError):
    """Cᵀ g ≠ 0: el puerto regula Casimirs (Gauss / armónicos)."""


class SchemaContractError(TopologicalInvariantError):
    """Semilla ajena al contrato 7.1 (Hodge / schema / u aplicado)."""


class CrowbarEngagedError(TopologicalInvariantError):
    """Veto físico: se devolvió estado crowbar (u=0, dt mínimo)."""


# ═══════════════════════════════════════════════════════════════════════════════════════
# CERTIFICADOS Y ESTADOS
# ═══════════════════════════════════════════════════════════════════════════════════════


def _as_1d(x: Any, name: str) -> Array:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise DiracMatchingError(f"{name} vacío.")
    if not np.all(np.isfinite(arr)):
        raise DiracMatchingError(f"{name} contiene NaN/Inf.")
    return arr


def _as_mat(x: Any, name: str) -> Array:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise DiracMatchingError(f"{name} debe ser 2D.")
    if not np.all(np.isfinite(arr)):
        raise DiracMatchingError(f"{name} contiene NaN/Inf.")
    return arr


def _as_sq(x: Any, name: str) -> Array:
    arr = _as_mat(x, name)
    if arr.shape[0] != arr.shape[1]:
        raise DiracMatchingError(f"{name} no es cuadrada: {arr.shape}.")
    return arr


def _fro(A: np.ndarray) -> float:
    if _SCIPY and la is not None:
        return float(la.norm(A, ord="fro"))
    return float(np.linalg.norm(A, ord="fro"))


def _eigvalsh(A: np.ndarray) -> Array:
    S = 0.5 * (A + A.T)
    if _SCIPY and la is not None:
        return np.asarray(la.eigvalsh(S), dtype=np.float64)
    return np.asarray(np.linalg.eigvalsh(S), dtype=np.float64)


def _svd(A: np.ndarray, full_matrices: bool = False) -> Tuple[Array, Array, Array]:
    if _SCIPY and la is not None:
        U, s, Vt = la.svd(A, full_matrices=full_matrices)
    else:
        U, s, Vt = np.linalg.svd(A, full_matrices=full_matrices)
    return np.asarray(U), np.asarray(s), np.asarray(Vt)


def _skew_defect(M: np.ndarray) -> float:
    nrm = max(1.0, _fro(M))
    return _fro(M + M.T) / nrm


def _sym_defect(M: np.ndarray) -> float:
    nrm = max(1.0, _fro(M))
    return _fro(M - M.T) / nrm


def _log_norm_2(A: np.ndarray) -> float:
    """μ₂(A) = λ_max((A+Aᵀ)/2).  ‖e^{tA}‖₂ ≤ e^{t μ₂(A)}."""
    if A.size == 0:
        return 0.0
    return float(np.max(_eigvalsh(A)))


@dataclass(frozen=True)
class PoincareDiracCertificate:
    """Auditoría I1–I8 sobre un PHS (no sobre un snapshot de gradientes iguales)."""

    antisymmetry_defect_current: float
    antisymmetry_defect_desired: float
    min_eigenvalue_R_current: float
    min_eigenvalue_R_desired: float
    rayleigh_open_loop: float
    matching_power_leak: float
    H_dot_closed: float
    liouville_trace: float
    liouville_consistent: bool
    symplectic_residual: float
    maupertuis_factor: float
    maupertuis_applicable: bool
    is_dirac_valid: bool
    is_passive_closed_loop: bool
    is_volume_contracting: bool
    equilibrium_defect: float
    is_equilibrium_assignable: bool
    casimir_dim: int
    casimir_port_leak: float
    is_casimir_immune: bool
    kernel_dim_R_desired: int
    is_la_salle_mod_casimir: bool
    logarithmic_norm_A: float
    logarithmic_norm_A_d: float
    condition_number_R_desired: float
    matching_residual_relative: float
    schema_version: str = _SCHEMA
    hodge_convention: str = _HODGE_CANON


@dataclass(frozen=True)
class PortTermination:
    r"""
    Terminación de puertos en el sentido PHS, *no* constitutivas Hodge.

        u = −K y + u_ff,     K ∈ ℝ^{m×m}

    Scattering respecto de Z₀ física (escalar o diagonal, heredada de ε,μ):
        Γ = (K − Z₀⁻¹)(K + Z₀⁻¹)⁻¹    (en el subespacio activo).

    `suggests_hodge_update` es **siempre False** en 7.1.
    """

    damping_map: Array
    feedforward: Array
    active_mask: NDArray[np.bool_]
    port_gain_eigenvalues: Array
    scattering_norm: float
    characteristic_impedance: Array
    pumping_channels: NDArray[np.bool_]
    maupertuis_conformal_factor: float
    maupertuis_applicable: bool
    anisotropy_index: float
    suggests_hodge_update: bool = False
    herglotz_status: str = "N/A_STATIC"

    # alias de compatibilidad (no son ε,μ de Maxwell)
    @property
    def reflection_coefficient_norm(self) -> float:
        return self.scattering_norm


@dataclass(frozen=True)
class CausalSpeed:
    """Velocidades *físicas* (lattice / metadatos), jamás √(2(H−V))."""

    c_medium: float
    c_yee_limit: float
    mu2_A: float
    rho_curl_curl: float
    source: str


@dataclass(frozen=True)
class ControlSolution:
    """Salida de φ₁: ley de puerto + certificado."""

    alpha: Array
    mode: str
    H_dot_closed: float
    desired_gradient: Array
    port_matrix: Array
    residual_norm: float
    residual_relative: float
    orthogonal_residual: float
    colinear_residual: float
    g_rank: int
    singular_values: Array
    condition_number_g: float
    is_full_rank_g: bool
    lyapunov_verified: bool
    required_forcing: Array
    poincare_certificate: PoincareDiracCertificate
    power_requested: float
    power_delivered: float
    pumping_required: bool


@dataclass(frozen=True)
class InterconnectionState:
    """
    Estado inyectable en el condensador 7.1.

    `control_law_alpha` **es** u aplicado (dimensión m).
    `safe_dt` respeta physics_dt del foso y el cono causal.
    `impedance` es PortTermination; el condensador no debe escribir ★.
    """

    control_law_alpha: Array
    termination: PortTermination
    safe_dt: float
    lyapunov_derivative: float
    c_eff: float
    cfl_margin: float
    lambda_max_laplacian: float
    cfl_number: float
    mu2_A: float
    poincare_certificate: PoincareDiracCertificate
    maupertuis_factor: float
    causal_verdict: str
    poincare_causal_report: Dict[str, Any] = field(default_factory=dict)
    atlas: str = "abstract_phs"
    schema_version: str = _SCHEMA
    hodge_convention: str = _HODGE_CANON
    crowbar: bool = False
    # alias
    @property
    def impedance(self) -> PortTermination:
        return self.termination


# ═══════════════════════════════════════════════════════════════════════════════════════
# φ₁ — MATCHING / POWER-SHAPING (contrato 7.1)
# ═══════════════════════════════════════════════════════════════════════════════════════


class Phase1_IDAPBC_PoincareSolver:
    r"""
    φ₁. Realiza una de:

        ENERGY_LEVEL:  P_* = ∇Hᵀ R ∇H − λ(H−H_*),  α = P_* y/(‖y‖²+ε)
        IDA_PBC_POINT: α = Gx + v,  v = −k_a gᵀ K_d (x−x*)
        SNAPSHOT:      α = g⁺ f_d    (auditoría de un instante)
        DAMPING_ONLY:  α = −k_d y    (vetado si pumping_required)

    Casimirs: se exige Cᵀg ≈ 0; si hay fuga, se proyecta g ← (I−CCᵀ)g
    o se lanza CasimirPortLeakError (modo estricto).
    """

    def __init__(
        self,
        tolerance: float = 1e-9,
        relative_tol: float = 1e-6,
        require_full_rank: bool = False,
        max_residual_relative: float = 1e-4,
        enforce_assignable_equilibrium: bool = True,
        casimir_strict: bool = True,
        power_regularization: float = 1e-12,
        energy_rate: float = 0.1,
    ) -> None:
        if min(tolerance, relative_tol, max_residual_relative, power_regularization) <= 0.0:
            raise DiracMatchingError("tolerancias deben ser > 0.")
        if energy_rate < 0.0:
            raise DiracMatchingError("energy_rate ≥ 0.")
        self._tol = float(tolerance)
        self._rel_tol = float(relative_tol)
        self._require_full_rank = bool(require_full_rank)
        self._max_res_rel = float(max_residual_relative)
        self._enforce_eq = bool(enforce_assignable_equilibrium)
        self._casimir_strict = bool(casimir_strict)
        self._eps_p = float(power_regularization)
        self._lambda = float(energy_rate)

    # ── álgebra de puerto ─────────────────────────────────────────────────────

    def _svd_pinv(self, g: Array) -> Tuple[int, Array, Array]:
        if g.size == 0:
            return 0, np.zeros((g.shape[1], g.shape[0])), np.zeros(0)
        U, sv, Vt = _svd(g, full_matrices=False)
        smax = float(sv[0]) if sv.size else 0.0
        thr = self._tol * smax if smax > 0.0 else self._tol
        mask = sv > thr
        rank = int(np.sum(mask))
        if rank == 0:
            return 0, np.zeros((g.shape[1], g.shape[0])), sv
        sinv = np.where(mask, 1.0 / np.maximum(sv, thr), 0.0)
        return rank, (Vt.T * sinv) @ U.T, sv

    def _split_im_ker(self, f: Array, g: Array, g_pinv: Array) -> Tuple[float, float, Array]:
        col = g @ (g_pinv @ f)
        orth = f - col
        return float(np.linalg.norm(col)), float(np.linalg.norm(orth)), orth

    def _casimir_project_g(self, g: Array, C: Optional[Array]) -> Tuple[Array, float]:
        if C is None or C.size == 0:
            return g, 0.0
        C = np.asarray(C, dtype=np.float64)
        if C.ndim == 1:
            C = C.reshape(-1, 1)
        leak = _fro(C.T @ g)
        if leak <= max(self._tol, self._rel_tol):
            return g, leak
        g_p = g - C @ (C.T @ g)
        leak_after = _fro(C.T @ g_p)
        if _fro(g_p) < self._tol * max(1.0, _fro(g)):
            raise CasimirPortLeakError("Todos los puertos viven en ker J.")
        if self._casimir_strict and leak_after > max(self._tol, self._rel_tol):
            raise CasimirPortLeakError(f"Cᵀg no nulo: leak={leak:.3e}→{leak_after:.3e}.")
        logger.info("Puertos proyectados fuera de ker J: %.3e → %.3e.", leak, leak_after)
        return g_p, leak_after

    def _validate_skew(self, M: Array, name: str) -> float:
        d = _skew_defect(M)
        if d > self._rel_tol:
            raise PoincareSymplecticError(f"{name} no antisimétrica: defect={d:.3e}.")
        return d

    def _validate_psd(self, M: Array, name: str) -> Array:
        if _sym_defect(M) > self._rel_tol:
            raise DiracMatchingError(f"{name} no simétrica.")
        w = _eigvalsh(M)
        if w.size and float(np.min(w)) < -self._tol:
            raise DiracMatchingError(f"{name} no PSD: λ_min={float(np.min(w)):.3e}.")
        return w

    def _kernel_dim(self, R: Array) -> Tuple[int, float]:
        w = _eigvalsh(R)
        if w.size == 0:
            return 0, 0.0
        thr = max(self._tol, self._rel_tol * float(np.max(np.abs(w))))
        return int(np.sum(w < thr)), float(np.min(w))

    # ── certificado ───────────────────────────────────────────────────────────

    def audit_poincare_dirac_structure(
        self,
        J: Array,
        R: Array,
        grad_H: Array,
        J_d: Array,
        R_d: Array,
        grad_H_d: Array,
        g: Array,
        alpha: Array,
        hessian: Optional[Array] = None,
        hessian_d: Optional[Array] = None,
        casimir_basis: Optional[Array] = None,
        x_star: Optional[Array] = None,
        maupertuis_factor: float = 1.0,
        maupertuis_applicable: bool = False,
        f_d: Optional[Array] = None,
    ) -> PoincareDiracCertificate:
        n = grad_H.size
        dJ, dJd = _skew_defect(J), _skew_defect(J_d)
        wR, wRd = _eigvalsh(R), _eigvalsh(R_d)
        minR = float(np.min(wR)) if wR.size else 0.0
        minRd = float(np.min(wRd)) if wRd.size else 0.0
        Rds = 0.5 * (R_d + R_d.T)
        rayleigh = float(grad_H_d @ (Rds @ grad_H_d))
        if f_d is None:
            f_d = (J_d - R_d) @ grad_H_d - (J - R) @ grad_H
        leak_power = float(grad_H_d @ (f_d - g @ alpha))
        Hdot = -rayleigh + leak_power
        K = hessian if hessian is not None else None
        Kd = hessian_d if hessian_d is not None else K
        trA = 0.0
        liou_ok = False
        if K is not None:
            K = 0.5 * (_as_sq(K, "K") + _as_sq(K, "K").T)
            A = (J - R) @ K
            trA = float(np.trace(A))
            liou_ok = True
        Ad = (J_d - R_d) @ (0.5 * (Kd + Kd.T) if Kd is not None else np.eye(n))
        mu2 = _log_norm_2((J - R) @ (K if K is not None else np.eye(n)))
        mu2d = _log_norm_2(Ad)
        C = casimir_basis
        cas_dim = 0 if C is None or np.asarray(C).size == 0 else int(np.asarray(C).reshape(n, -1).shape[1])
        leak = 0.0 if C is None or np.asarray(C).size == 0 else _fro(np.asarray(C, dtype=np.float64).reshape(n, -1).T @ g)
        # I6: ∇H_d(x*) = 0
        if x_star is not None and Kd is not None:
            eq_def = float(np.linalg.norm(np.asarray(Kd) @ np.asarray(x_star).reshape(-1)))
        else:
            # en el snapshot, el equilibrio asignable no se puede certificar
            eq_def = float("nan")
        eq_ok = True if not np.isfinite(eq_def) else bool(eq_def <= max(self._tol, self._rel_tol))
        kdim, _ = self._kernel_dim(R_d)
        # La Salle mod Casimir: modos persistentes ⊆ ker J
        lasalle = True
        if kdim > 0 and C is not None and np.asarray(C).size:
            # ker R_d debe estar cubierto por Casimirs + tolerancia
            lasalle = True  # no afirmamos asintótica en ℝⁿ
        elif kdim > 0:
            lasalle = False
        condR = 1.0
        if wRd.size:
            pos = wRd[wRd > self._tol]
            condR = float(np.max(wRd) / np.min(pos)) if pos.size else float("inf")
        fn = max(float(np.linalg.norm(f_d)), _WILKINSON)
        _, orth, _ = self._split_im_ker(f_d, g, self._svd_pinv(g)[1])
        is_dirac = bool(dJ <= self._rel_tol and dJd <= self._rel_tol and minR >= -self._tol and minRd >= -self._tol)
        return PoincareDiracCertificate(
            antisymmetry_defect_current=dJ,
            antisymmetry_defect_desired=dJd,
            min_eigenvalue_R_current=minR,
            min_eigenvalue_R_desired=minRd,
            rayleigh_open_loop=-rayleigh,
            matching_power_leak=leak_power,
            H_dot_closed=Hdot,
            liouville_trace=trA,
            liouville_consistent=liou_ok,
            symplectic_residual=max(dJ, dJd),
            maupertuis_factor=float(maupertuis_factor),
            maupertuis_applicable=bool(maupertuis_applicable),
            is_dirac_valid=is_dirac,
            is_passive_closed_loop=bool(Hdot <= self._tol),
            is_volume_contracting=bool((not liou_ok) or trA <= self._tol),
            equilibrium_defect=0.0 if not np.isfinite(eq_def) else eq_def,
            is_equilibrium_assignable=eq_ok,
            casimir_dim=cas_dim,
            casimir_port_leak=float(leak),
            is_casimir_immune=bool(leak <= max(self._tol, self._rel_tol)),
            kernel_dim_R_desired=kdim,
            is_la_salle_mod_casimir=bool(lasalle or kdim == 0),
            logarithmic_norm_A=mu2,
            logarithmic_norm_A_d=mu2d,
            condition_number_R_desired=condR,
            matching_residual_relative=float(orth / fn),
        )

    # ── leyes ─────────────────────────────────────────────────────────────────

    def _energy_level_alpha(
        self, grad_H: Array, R: Array, g: Array, H: float, H_star: float, lam: float
    ) -> Tuple[Array, float, float]:
        y = g.T @ grad_H
        rayleigh = float(grad_H @ (R @ grad_H))
        P_star = rayleigh - lam * (H - H_star)
        y2 = float(y @ y)
        alpha = (P_star * y) / (y2 + self._eps_p)
        delivered = float(alpha @ y)
        return alpha, P_star, delivered

    def _ida_point_alpha(
        self, x: Array, g: Array, G: Array, K_d: Array, x_star: Array, k_a: float
    ) -> Array:
        v = -k_a * (g.T @ (K_d @ (x - x_star)))
        u = np.asarray(G @ x, dtype=np.float64).reshape(-1) + v.reshape(-1)
        if u.size != g.shape[1]:
            raise DiracMatchingError(f"u IDA dim {u.size} ≠ m={g.shape[1]}.")
        return u

    def compute_control_law(
        self,
        J_current: Array,
        R_current: Array,
        grad_H: Array,
        J_desired: Array,
        R_desired: Array,
        grad_H_desired: Array,
        g_port: Array,
        hessian_current: Optional[Array] = None,
        hessian_desired: Optional[Array] = None,
        casimir_basis: Optional[Array] = None,
        x: Optional[Array] = None,
        x_star: Optional[Array] = None,
        hamiltonian: Optional[float] = None,
        target_hamiltonian: Optional[float] = None,
        ida_G: Optional[Array] = None,
        mode: Union[str, MatchingMode] = MatchingMode.ENERGY_LEVEL,
        pumping_required: bool = False,
        maupertuis_factor: float = 1.0,
        maupertuis_applicable: bool = False,
        damping_injection: Optional[float] = None,
    ) -> ControlSolution:
        mode = MatchingMode(mode) if not isinstance(mode, MatchingMode) else mode
        J = 0.5 * (_as_sq(J_current, "J") - _as_sq(J_current, "J").T)
        Jd = 0.5 * (_as_sq(J_desired, "J_d") - _as_sq(J_desired, "J_d").T)
        R = 0.5 * (_as_sq(R_current, "R") + _as_sq(R_current, "R").T)
        Rd = 0.5 * (_as_sq(R_desired, "R_d") + _as_sq(R_desired, "R_d").T)
        self._validate_skew(J, "J")
        self._validate_skew(Jd, "J_d")
        self._validate_psd(R, "R")
        self._validate_psd(Rd, "R_d")
        grad_H = _as_1d(grad_H, "∇H")
        grad_Hd = _as_1d(grad_H_desired, "∇H_d")
        if grad_H.size != grad_Hd.size:
            raise DiracMatchingError("∇H y ∇H_d dim distintas.")
        n = grad_H.size
        g = _as_mat(g_port, "g")
        if g.shape[0] != n:
            raise DiracMatchingError(f"g filas {g.shape[0]} ≠ n={n}.")
        g, _leak = self._casimir_project_g(g, casimir_basis)
        rank, g_pinv, sv = self._svd_pinv(g)
        if self._require_full_rank and rank < min(g.shape):
            raise DiracMatchingError(f"rank(g)={rank} < {min(g.shape)}.")
        if pumping_required and mode is MatchingMode.DAMPING_ONLY:
            raise LyapunovInstabilityError("DAMPING_ONLY con pumping_required: H no sube.")

        lam = self._lambda if damping_injection is None else float(damping_injection)
        H = float(hamiltonian) if hamiltonian is not None else float("nan")
        H_star = float(target_hamiltonian) if target_hamiltonian is not None else H
        P_req = 0.0
        P_del = 0.0

        if mode is MatchingMode.ENERGY_LEVEL:
            if not np.isfinite(H) or not np.isfinite(H_star):
                raise DiracMatchingError("ENERGY_LEVEL exige H y H*.")
            alpha, P_req, P_del = self._energy_level_alpha(grad_H, R, g, H, H_star, lam)
        elif mode is MatchingMode.IDA_PBC_POINT:
            if ida_G is None or x is None:
                raise DiracMatchingError("IDA_PBC_POINT exige G y x.")
            xs = np.zeros(n) if x_star is None else _as_1d(x_star, "x*")
            Kd = hessian_desired if hessian_desired is not None else np.eye(n)
            alpha = self._ida_point_alpha(_as_1d(x, "x"), g, np.asarray(ida_G, dtype=np.float64), np.asarray(Kd), xs, lam)
            P_del = float(alpha @ (g.T @ grad_H))
        elif mode is MatchingMode.DAMPING_ONLY:
            y = g.T @ grad_H
            alpha = -lam * y
            P_del = float(alpha @ y)
        else:
            f_d = (Jd - Rd) @ grad_Hd - (J - R) @ grad_H
            alpha = g_pinv @ f_d
            P_del = float(alpha @ (g.T @ grad_H))

        f_d = (Jd - Rd) @ grad_Hd - (J - R) @ grad_H
        recon = g @ alpha
        resid = f_d - recon
        nrm_f = float(np.linalg.norm(f_d))
        nrm_r = float(np.linalg.norm(resid))
        col, orth, _ = self._split_im_ker(f_d, g, g_pinv)
        rel_orth = orth / max(nrm_f, _WILKINSON)
        if mode is MatchingMode.SNAPSHOT and rel_orth > self._max_res_rel:
            raise DiracMatchingError(
                f"Residuo ortogonal {rel_orth:.3e} > {self._max_res_rel:.3e} (f_d ∉ Im g)."
            )

        Rds = 0.5 * (Rd + Rd.T)
        Hdot = -float(grad_Hd @ (Rds @ grad_Hd)) + float(grad_Hd @ (f_d - g @ alpha))
        lyap_ok = Hdot <= self._tol
        if not lyap_ok and mode is MatchingMode.SNAPSHOT:
            logger.error("[LYAPUNOV] Ḣ_closed=%.3e (incluye leak de matching).", Hdot)

        if self._enforce_eq and x_star is not None and hessian_desired is not None and mode is MatchingMode.IDA_PBC_POINT:
            def_eq = float(np.linalg.norm(np.asarray(hessian_desired) @ _as_1d(x_star, "x*")))
            if def_eq > max(self._tol, self._rel_tol):
                raise EnergyMatchingError(f"∇H_d(x*) ≠ 0: {def_eq:.3e}.")

        cert = self.audit_poincare_dirac_structure(
            J=J, R=R, grad_H=grad_H, J_d=Jd, R_d=Rd, grad_H_d=grad_Hd, g=g, alpha=alpha,
            hessian=hessian_current, hessian_d=hessian_desired, casimir_basis=casimir_basis,
            x_star=x_star, maupertuis_factor=maupertuis_factor,
            maupertuis_applicable=maupertuis_applicable, f_d=f_d,
        )
        smax = float(sv[0]) if sv.size else 0.0
        smin = float(sv[-1]) if sv.size else 0.0
        cond = smax / smin if smin > self._tol else (float("inf") if sv.size else 1.0)
        return ControlSolution(
            alpha=np.asarray(alpha, dtype=np.float64).reshape(-1),
            mode=mode.value,
            H_dot_closed=Hdot,
            desired_gradient=grad_Hd,
            port_matrix=g,
            residual_norm=nrm_r,
            residual_relative=nrm_r / max(nrm_f, _WILKINSON),
            orthogonal_residual=orth,
            colinear_residual=col,
            g_rank=rank,
            singular_values=np.asarray(sv, dtype=np.float64),
            condition_number_g=cond,
            is_full_rank_g=bool(rank == min(g.shape)),
            lyapunov_verified=lyap_ok,
            required_forcing=f_d,
            poincare_certificate=cert,
            power_requested=P_req,
            power_delivered=P_del,
            pumping_required=bool(pumping_required),
        )

    def compute_port_termination(
        self,
        control_solution: ControlSolution,
        z0: Optional[Array] = None,
        maupertuis_factor: float = 1.0,
        maupertuis_applicable: bool = False,
    ) -> PortTermination:
        r"""
        COSTURA φ₁ → φ₂.  K tal que α ≈ −K y  (mínimos cuadrados, canales activos).

        No produce ε,μ. Z₀ es la impedancia característica *física* (ohmios de
        puerto), no √(μ/ε) inventado a partir de α/y.
        """
        g = control_solution.port_matrix
        y = g.T @ control_solution.desired_gradient
        a = control_solution.alpha.reshape(-1)
        m = a.size
        if y.size != m:
            raise DiracMatchingError("y y α dim distintas.")
        active = np.abs(y) > self._tol
        K = np.zeros((m, m), dtype=np.float64)
        # rank-1: α = κ y  ⇒  K = −κ I sobre el span de y (κ = ⟨α,y⟩/‖y‖²)
        y2 = float(y @ y)
        kappa = -float(a @ y) / (y2 + self._eps_p)  # u = −K y ⇒ K = −⟨u,y⟩/‖y‖²
        if y2 > self._eps_p:
            K = kappa * np.eye(m)
        wK = _eigvalsh(0.5 * (K + K.T)) if m else np.zeros(0)
        pumping = wK < -self._tol if wK.size else np.array([], dtype=bool)
        if z0 is None:
            z0 = np.ones(m, dtype=np.float64)
        z0 = np.asarray(z0, dtype=np.float64).reshape(-1)
        if z0.size == 1:
            z0 = np.full(m, float(z0[0]))
        if z0.size != m:
            raise ImpedanceMismatchError("Z₀ dim ≠ m.")
        # Γ sobre canales activos, Z₀ > 0
        gamma_n = 0.0
        mask_z = active & (np.abs(z0) > self._tol)
        if np.any(mask_z) and m:
            # escalar por canal: Γ_i = (R_i − Z0_i)/(R_i + Z0_i), R_i = K_ii
            Rii = np.diag(K)
            num = Rii[mask_z] - z0[mask_z]
            den = Rii[mask_z] + z0[mask_z]
            den = np.where(np.abs(den) > self._tol, den, 1.0)
            gamma_n = float(np.linalg.norm(num / den))
        aniso = 1.0
        if wK.size:
            pos = np.abs(wK[np.abs(wK) > self._tol])
            if pos.size:
                aniso = float(np.max(pos) / np.min(pos))
        return PortTermination(
            damping_map=K,
            feedforward=a + K @ y,
            active_mask=active,
            port_gain_eigenvalues=wK,
            scattering_norm=gamma_n,
            characteristic_impedance=z0,
            pumping_channels=np.asarray(wK < -self._tol) if wK.size else np.zeros(m, dtype=bool),
            maupertuis_conformal_factor=float(maupertuis_factor),
            maupertuis_applicable=bool(maupertuis_applicable),
            anisotropy_index=aniso,
            suggests_hodge_update=False,
            herglotz_status="N/A_STATIC",
        )


# ═══════════════════════════════════════════════════════════════════════════════════════
# φ₂ — SCATTERING / VELOCIDAD FÍSICA (no retoca ★)
# ═══════════════════════════════════════════════════════════════════════════════════════


class Phase2_PortScattering:
    r"""
    φ₂. Lee c y Z₀ del medio *ya definido* (semilla / lattice).

    Maupertuis-Jacobi: solo si `kinetic` y `potential` se aportan por separado
    (fibrado T*Q). El factor f_M **no** escala c.
    """

    def __init__(self, min_wave_speed: float = 1e-12, c_fallback: float = 1.0) -> None:
        if min_wave_speed <= 0.0 or c_fallback <= 0.0:
            raise ImpedanceMismatchError("velocidades deben ser > 0.")
        self._c_min = float(min_wave_speed)
        self._c_fb = float(c_fallback)

    def maupertuis_factor(
        self,
        kinetic: Optional[float] = None,
        potential: Optional[float] = None,
        total_energy: Optional[float] = None,
    ) -> Tuple[float, bool]:
        if kinetic is None or potential is None:
            return 1.0, False
        T, V = float(kinetic), float(potential)
        if not math.isfinite(T) or not math.isfinite(V):
            raise MaupertuisViolationError("T, V no finitos.")
        E = T + V if total_energy is None else float(total_energy)
        fM = 2.0 * (E - V)
        if fM <= _MAUPERTUIS_FLOOR:
            raise MaupertuisViolationError(f"f_M={fM:.3e} ≤ 0 (E≤V).")
        return float(fM), True

    def physical_z0_and_c(
        self,
        seed: Optional[MaybeSeed] = None,
        epsilon: Optional[float] = None,
        mu: Optional[float] = None,
        c_hint: Optional[float] = None,
        port_dim: int = 1,
    ) -> Tuple[Array, float, str]:
        """Z₀ = √(μ/ε) del medio de la semilla; c = 1/√(με). No usa α/y."""
        meta: Dict[str, Any] = {}
        hints: Dict[str, Any] = {}
        if seed is not None:
            meta = dict(getattr(seed, "metadata", None) or {})
            hints = dict(getattr(seed, "engine_hints", None) or {})
        eps = epsilon if epsilon is not None else meta.get("epsilon", hints.get("epsilon"))
        mu_v = mu if mu is not None else meta.get("mu", hints.get("mu"))
        c = c_hint if c_hint is not None else hints.get("c") or meta.get("c")
        src = "fallback"
        if eps is not None and mu_v is not None:
            eps_f, mu_f = float(eps), float(mu_v)
            if eps_f > 0.0 and mu_f > 0.0:
                c = 1.0 / math.sqrt(eps_f * mu_f)
                z = math.sqrt(mu_f / eps_f)
                src = "constitutive_seed"
                return np.full(max(port_dim, 1), z, dtype=np.float64), float(c), src
        if c is None or not math.isfinite(float(c)) or float(c) <= 0.0:
            c = self._c_fb
            src = "fallback"
        else:
            c = float(c)
            src = "hint"
        return np.ones(max(port_dim, 1), dtype=np.float64), float(c), src

    def compute_causal_speed(
        self,
        termination: PortTermination,
        c_medium: float,
        mu2_A: float = 0.0,
        rho_curl_curl: float = 0.0,
    ) -> CausalSpeed:
        """COSTURA φ₂ → φ₃. c_yee = c; nunca c√f_M."""
        if not math.isfinite(c_medium) or c_medium < 0.0:
            raise ImpedanceMismatchError(f"c inválida: {c_medium}.")
        c = max(float(c_medium), self._c_min) if c_medium > 0.0 else 0.0
        yee = 0.0
        if rho_curl_curl > 0.0 and c > 0.0:
            yee = c * math.sqrt(rho_curl_curl)
        return CausalSpeed(
            c_medium=c,
            c_yee_limit=float(yee),
            mu2_A=float(mu2_A),
            rho_curl_curl=float(rho_curl_curl),
            source="physical_medium",
        )


# ═══════════════════════════════════════════════════════════════════════════════════════
# φ₃ — CONO CAUSAL (Yee vs punto medio)
# ═══════════════════════════════════════════════════════════════════════════════════════


class Phase3_CFLGovernor:
    r"""
    φ₃.

    Punto medio (7.1): A-estable en {Re z < 0}; el transitorio lo marca μ₂(A).
        Si μ₂(A) ≤ 0: cualquier Δt es estable (la precisión es otro asunto).
        Si μ₂(A) > 0: Δt ≲ 1/μ₂  para no explotar el transitorio no-normal.

    Yee / leapfrog (opcional, lattice):
        Δt < 2 / (c √ρ(Δ₁)) = 2 / ω_max.

    Estimación de λ_max(Δ): Lanczos (Ritz residual) → potencia → Gerschgorin,
    cruzados; si |λ_L − λ_G|/max(λ_G,1) > 1 se lanza LanczosConvergenceError
    solo en modo estricto (Gerschgorin es cota, no valor).
    """

    def __init__(
        self,
        safety_margin: float = 0.5,
        lanczos_tol: float = 1e-6,
        power_max_iter: int = 256,
        strict_spectrum: bool = False,
        integrator: str = "implicit_midpoint",
    ) -> None:
        if not (0.0 < safety_margin <= 1.0):
            raise CFLViolationError("safety_margin ∈ (0,1].")
        self._margin = float(safety_margin)
        self._lanczos_tol = float(lanczos_tol)
        self._power_iter = int(power_max_iter)
        self._strict = bool(strict_spectrum)
        self._integrator = str(integrator)

    def _as_csr(self, L: Any) -> Any:
        if L is None:
            return None
        if _SCIPY and issparse(L):
            return L.tocsr()
        if _SCIPY:
            return csr_matrix(np.asarray(L, dtype=np.float64))
        return np.asarray(L, dtype=np.float64)

    def _symmetrize(self, L: Any) -> Tuple[Any, float]:
        if L is None:
            return None, 0.0
        if _SCIPY and issparse(L):
            Ls = 0.5 * (L + L.T)
            diff = L - L.T
            na = float(np.sqrt(np.sum(diff.data ** 2))) if diff.data.size else 0.0
            nL = float(np.sqrt(np.sum(L.data ** 2))) if L.data.size else 1.0
            return Ls.tocsr(), na / max(nL, 1e-15)
        A = np.asarray(L, dtype=np.float64)
        return 0.5 * (A + A.T), _sym_defect(A)

    def _lambda_max(self, Ls: Any) -> Tuple[float, str, float]:
        if Ls is None:
            return 0.0, "none", 0.0
        n = Ls.shape[0]
        if n == 0:
            return 0.0, "empty", 0.0
        # Lanczos con residuo de Ritz
        if _SCIPY and eigsh is not None and issparse(Ls) and n > 2:
            try:
                k = min(3, n - 1)
                evals, evecs = eigsh(Ls, k=k, which="LA", tol=self._lanczos_tol)
                idx = int(np.argmax(evals))
                lam = float(evals[idx])
                v = evecs[:, idx]
                res = float(np.linalg.norm(Ls @ v - lam * v))
                return abs(lam), "lanczos", res
            except Exception as exc:  # noqa: BLE001
                logger.debug("Lanczos: %s", exc)
        # potencia
        try:
            if _SCIPY and issparse(Ls):
                matvec = lambda v: np.asarray(Ls @ v).reshape(-1)
                nloc = Ls.shape[0]
            else:
                A = np.asarray(Ls)
                matvec = lambda v: A @ v
                nloc = A.shape[0]
            rng = np.random.default_rng(0)
            v = rng.standard_normal(nloc)
            v /= np.linalg.norm(v)
            lam = 0.0
            res = float("inf")
            for _ in range(self._power_iter):
                w = matvec(v)
                nw = float(np.linalg.norm(w))
                if nw < 1e-15:
                    break
                v = w / nw
                Lv = matvec(v)
                lam = float(v @ Lv)
                res = float(np.linalg.norm(Lv - lam * v))
                if res < _POWER_RES_TOL:
                    break
            return abs(lam), "power", res
        except Exception as exc:  # noqa: BLE001
            logger.debug("Potencia: %s", exc)
        # Gerschgorin (cota superior)
        if _SCIPY and issparse(Ls):
            diag = np.asarray(Ls.diagonal()).ravel()
            bound = []
            Lc = Ls.tocsr()
            for i in range(Lc.shape[0]):
                sl = slice(Lc.indptr[i], Lc.indptr[i + 1])
                row, cols = Lc.data[sl], Lc.indices[sl]
                bound.append(float(diag[i] + np.sum(np.abs(row[cols != i]))))
            return float(max(bound) if bound else 0.0), "gerschgorin", float("inf")
        A = np.asarray(Ls)
        diag = np.diag(A)
        rs = np.sum(np.abs(A), axis=1) - np.abs(diag)
        return float(np.max(diag + rs)), "gerschgorin", float("inf")

    def diagnose(
        self,
        graph_laplacian: Any,
        causal: CausalSpeed,
        requested_dt: float,
        lyapunov_derivative: float,
        pumping_required: bool = False,
        mode: str = MatchingMode.ENERGY_LEVEL.value,
    ) -> Dict[str, Any]:
        if requested_dt <= 0.0 or not math.isfinite(requested_dt):
            raise CFLViolationError(f"dt inválido: {requested_dt}.")
        Ls, sym = self._symmetrize(self._as_csr(graph_laplacian))
        lam, method, res = self._lambda_max(Ls) if Ls is not None else (causal.rho_curl_curl, "hint", 0.0)
        if causal.rho_curl_curl > 0.0:
            lam = max(lam, causal.rho_curl_curl)
        omega = causal.c_yee_limit if causal.c_yee_limit > 0.0 else (
            causal.c_medium * math.sqrt(max(lam, 0.0)) if causal.c_medium > 0.0 else 0.0
        )
        dt_yee = float("inf") if omega <= 0.0 else 2.0 / omega
        dt_yee_safe = self._margin * dt_yee if math.isfinite(dt_yee) else requested_dt
        mu2 = causal.mu2_A
        dt_mu = float("inf") if mu2 <= _SPECTRAL_TOL else self._margin / mu2
        integrator = self._integrator
        if integrator == "implicit_midpoint":
            dt_cap = dt_mu  # Yee no gobierna el PHS
            cfl_number = 0.0 if not math.isfinite(dt_mu) else requested_dt / max(dt_mu, _DT_MIN)
            yee_applies = False
        else:
            dt_cap = min(dt_yee_safe, dt_mu)
            cfl_number = (omega * requested_dt / 2.0) if omega > 0.0 else 0.0
            yee_applies = True
        safe = min(requested_dt, dt_cap) if math.isfinite(dt_cap) else requested_dt
        safe = float(np.clip(safe, _DT_MIN, _DT_MAX))
        reasons: List[str] = []
        lyap_ok = bool(lyapunov_derivative <= _SPECTRAL_TOL or (pumping_required and lyapunov_derivative > 0.0))
        # bombeo: Ḣ>0 es *deseable* cerca de H<H*
        if (not pumping_required) and lyapunov_derivative > _SPECTRAL_TOL:
            reasons.append(f"LYAPUNOV_FAIL(Ḣ={lyapunov_derivative:.3e})")
            lyap_ok = False
        if yee_applies and requested_dt > dt_yee * (1.0 + 1e-12):
            reasons.append(f"YEE_CFL(#{cfl_number:.3f})")
        if integrator != "implicit_midpoint" and mu2 > _SPECTRAL_TOL and requested_dt > dt_mu:
            reasons.append(f"MU2_TRANSIENT(μ₂={mu2:.3e})")
        if pumping_required and mode == MatchingMode.DAMPING_ONLY.value:
            reasons.append("PUMPING_VS_DAMPING")
        verdict = "COHERENT" if not reasons else "VETOED"
        return {
            "lambda_max": lam,
            "estimation_method": method,
            "spectral_residual": res,
            "symmetry_residual": sym,
            "omega_max": omega,
            "dt_yee": dt_yee,
            "dt_yee_safe": dt_yee_safe,
            "dt_mu2": dt_mu,
            "cfl_number": cfl_number,
            "safe_dt": safe,
            "requested_dt": requested_dt,
            "margin": self._margin,
            "violated": verdict != "COHERENT",
            "integrator": integrator,
            "yee_applies": yee_applies,
            "mu2_A": mu2,
            "c_medium": causal.c_medium,
            "lyapunov_derivative": lyapunov_derivative,
            "lyapunov_ok": lyap_ok,
            "verdict": verdict,
            "veto_reasons": reasons,
            "schema_version": _SCHEMA,
        }

    def synthesize_interconnection_state(
        self,
        control_solution: ControlSolution,
        termination: PortTermination,
        causal: CausalSpeed,
        safe_dt: float,
        cfl_diag: Mapping[str, Any],
        atlas: str = "abstract_phs",
        crowbar: bool = False,
    ) -> InterconnectionState:
        """CIERRE φ₃ → condensador."""
        return InterconnectionState(
            control_law_alpha=np.asarray(control_solution.alpha, dtype=np.float64).copy(),
            termination=termination,
            safe_dt=float(safe_dt),
            lyapunov_derivative=float(control_solution.H_dot_closed),
            c_eff=float(causal.c_medium),
            cfl_margin=self._margin,
            lambda_max_laplacian=float(cfl_diag.get("lambda_max", 0.0)),
            cfl_number=float(cfl_diag.get("cfl_number", 0.0)),
            mu2_A=float(causal.mu2_A),
            poincare_certificate=control_solution.poincare_certificate,
            maupertuis_factor=float(termination.maupertuis_conformal_factor),
            causal_verdict=str(cfl_diag.get("verdict", "COHERENT")),
            poincare_causal_report=dict(cfl_diag),
            atlas=str(atlas),
            schema_version=_SCHEMA,
            hodge_convention=_HODGE_CANON,
            crowbar=bool(crowbar),
        )

    def crowbar_state(
        self,
        m: int,
        n: int,
        requested_dt: float,
        reason: str,
    ) -> InterconnectionState:
        """u=0, dt mínimo: cortocircuito de puerto (no reescribe ★)."""
        z = np.zeros(m, dtype=np.float64)
        dummy_cert = PoincareDiracCertificate(
            antisymmetry_defect_current=0.0, antisymmetry_defect_desired=0.0,
            min_eigenvalue_R_current=0.0, min_eigenvalue_R_desired=0.0,
            rayleigh_open_loop=0.0, matching_power_leak=0.0, H_dot_closed=0.0,
            liouville_trace=0.0, liouville_consistent=False, symplectic_residual=0.0,
            maupertuis_factor=1.0, maupertuis_applicable=False,
            is_dirac_valid=False, is_passive_closed_loop=True, is_volume_contracting=True,
            equilibrium_defect=0.0, is_equilibrium_assignable=False,
            casimir_dim=0, casimir_port_leak=0.0, is_casimir_immune=True,
            kernel_dim_R_desired=0, is_la_salle_mod_casimir=True,
            logarithmic_norm_A=0.0, logarithmic_norm_A_d=0.0,
            condition_number_R_desired=1.0, matching_residual_relative=0.0,
        )
        term = PortTermination(
            damping_map=np.zeros((m, m)), feedforward=z,
            active_mask=np.zeros(m, dtype=bool), port_gain_eigenvalues=np.zeros(0),
            scattering_norm=0.0, characteristic_impedance=np.ones(m),
            pumping_channels=np.zeros(m, dtype=bool),
            maupertuis_conformal_factor=1.0, maupertuis_applicable=False,
            anisotropy_index=1.0,
        )
        cs = ControlSolution(
            alpha=z, mode=MatchingMode.DAMPING_ONLY.value, H_dot_closed=0.0,
            desired_gradient=np.zeros(n), port_matrix=np.zeros((n, m)),
            residual_norm=0.0, residual_relative=0.0, orthogonal_residual=0.0,
            colinear_residual=0.0, g_rank=0, singular_values=np.zeros(0),
            condition_number_g=1.0, is_full_rank_g=False, lyapunov_verified=True,
            required_forcing=np.zeros(n), poincare_certificate=dummy_cert,
            power_requested=0.0, power_delivered=0.0, pumping_required=False,
        )
        diag = {"verdict": "VETOED", "veto_reasons": [reason], "safe_dt": _DT_MIN, "cfl_number": 0.0, "lambda_max": 0.0}
        logger.error("[CROWBAR] %s", reason)
        return self.synthesize_interconnection_state(
            cs, term, CausalSpeed(0.0, 0.0, 0.0, 0.0, "crowbar"),
            min(requested_dt, _DT_MIN), diag, crowbar=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR — MORFISMO φ = φ₃ ∘ φ₂ ∘ φ₁
# ═══════════════════════════════════════════════════════════════════════════════════════


class DiracInterconnectionAgent(Morphism):
    r"""
    Aduana TACTICS → PHYSICS.

    __call__ / synthesize_physical_control:
        1. Si hay PoincareEngineSeed / PoincareControlSeed, se usa su PHS,
           Casimirs, u_app, hints (μ₂, Hodge, pumping).
        2. φ₁ produce α (ley 7.1).  Si la semilla ya trae control_input y
           `prefer_seed_u=True`, α ← u_app (no se re-satura el músculo).
        3. φ₂ lee Z₀, c del medio; scattering de la terminación.
        4. φ₃ recorta dt; veto → crowbar.

    El condensador debe integrar con `state.control_law_alpha` y `state.safe_dt`
    **sin** escribir ε,μ.
    """

    def __init__(
        self,
        metric_tensor: Optional[Array] = None,
        tolerance: float = 1e-9,
        safety_margin: float = 0.5,
        max_residual_relative: float = 1e-4,
        integrator: str = "implicit_midpoint",
        prefer_seed_u: bool = True,
        crowbar_on_veto: bool = True,
    ) -> None:
        self._G = metric_tensor if metric_tensor is not None else G_PHYSICS
        self._solver = Phase1_IDAPBC_PoincareSolver(
            tolerance=tolerance, max_residual_relative=max_residual_relative,
        )
        self._tuner = Phase2_PortScattering()
        self._governor = Phase3_CFLGovernor(safety_margin=safety_margin, integrator=integrator)
        self._prefer_seed_u = bool(prefer_seed_u)
        self._crowbar_on_veto = bool(crowbar_on_veto)
        self._last_control: Optional[ControlSolution] = None
        self._last_term: Optional[PortTermination] = None
        self._last_causal: Optional[CausalSpeed] = None
        self._last_state: Optional[InterconnectionState] = None

    def __call__(self, *args: Any, **kwargs: Any) -> InterconnectionState:
        return self.synthesize_physical_control(*args, **kwargs)

    def compose(self, other: "DiracInterconnectionAgent") -> Callable[..., InterconnectionState]:
        """φ_other ∘ φ_self sobre el mismo signature táctico (no sobre InterconnectionState)."""

        def _c(*args: Any, **kwargs: Any) -> InterconnectionState:
            _ = self.synthesize_physical_control(*args, **kwargs)
            return other.synthesize_physical_control(*args, **kwargs)

        return _c

    def __or__(self, other: "DiracInterconnectionAgent") -> Callable[..., InterconnectionState]:
        return self.compose(other)

    @staticmethod
    def _unpack_seed(seed: MaybeSeed) -> Dict[str, Any]:
        meta = dict(getattr(seed, "metadata", None) or {})
        hints = dict(getattr(seed, "engine_hints", None) or {})
        conv = str(meta.get("hodge_convention") or hints.get("hodge_convention") or _HODGE_CANON)
        schema = str(getattr(seed, "schema_version", None) or meta.get("schema_version") or hints.get("schema_version") or "")
        return {
            "x": np.asarray(getattr(seed, "state"), dtype=np.float64).reshape(-1),
            "grad": np.asarray(getattr(seed, "gradient"), dtype=np.float64).reshape(-1),
            "H": float(getattr(seed, "hamiltonian")),
            "H_star": float(getattr(seed, "target_hamiltonian", getattr(seed, "hamiltonian"))),
            "J": np.asarray(getattr(seed, "interconnection_matrix"), dtype=np.float64),
            "R": np.asarray(getattr(seed, "damping_matrix"), dtype=np.float64),
            "K": np.asarray(getattr(seed, "metric_matrix"), dtype=np.float64),
            "g": np.asarray(getattr(seed, "port_matrix"), dtype=np.float64),
            "u_app": np.asarray(getattr(seed, "control_input", np.zeros(0)), dtype=np.float64).reshape(-1),
            "C": getattr(seed, "casimir_basis", None),
            "ida": getattr(seed, "ida_pbc_decomposition", None),
            "pumping": bool(getattr(seed, "pumping_required", meta.get("pumping_required", False))),
            "mu2": float(hints.get("logarithmic_norm_A", meta.get("logarithmic_norm", 0.0)) or 0.0),
            "rho": float(hints.get("rho_curl_curl", 0.0) or 0.0),
            "dt": float(hints.get("dt_suggested", meta.get("dt_suggested", 1e-3)) or 1e-3),
            "atlas": str(meta.get("atlas") or hints.get("atlas") or "abstract_phs"),
            "mode": str(hints.get("mode") or meta.get("mode") or MatchingMode.ENERGY_LEVEL.value),
            "hodge": conv,
            "schema": schema or _SCHEMA,
            "meta": meta,
            "hints": hints,
        }

    def synthesize_physical_control(
        self,
        J_current: Optional[Array] = None,
        R_current: Optional[Array] = None,
        grad_H: Optional[Array] = None,
        g_port: Optional[Array] = None,
        graph_laplacian: Any = None,
        J_desired: Optional[Array] = None,
        R_desired: Optional[Array] = None,
        grad_H_desired: Optional[Array] = None,
        requested_dt: Optional[float] = None,
        hessian_current: Optional[Array] = None,
        hessian_desired: Optional[Array] = None,
        hamiltonian_target: float = 1.0,
        potential_energy: Optional[float] = None,
        kinetic_energy: Optional[float] = None,
        seed: Optional[MaybeSeed] = None,
        mode: Optional[Union[str, MatchingMode]] = None,
        raise_on_veto: bool = True,
    ) -> InterconnectionState:
        logger.info("[DiracAgent 7.1] φ₁ matching → φ₂ scattering → φ₃ cono")
        packed = None
        if seed is not None:
            packed = self._unpack_seed(seed)
            if packed["schema"] and packed["schema"] != _SCHEMA:
                logger.warning("schema %s ≠ %s", packed["schema"], _SCHEMA)
            if packed["hodge"] and packed["hodge"] != _HODGE_CANON:
                raise SchemaContractError(f"Hodge ajeno: {packed['hodge']}")
            J_current = packed["J"] if J_current is None else J_current
            R_current = packed["R"] if R_current is None else R_current
            grad_H = packed["grad"] if grad_H is None else grad_H
            g_port = packed["g"] if g_port is None else g_port
            hessian_current = packed["K"] if hessian_current is None else hessian_current
            hamiltonian_target = packed["H_star"]
            if requested_dt is None:
                requested_dt = packed["dt"]
            if mode is None:
                mode = packed["mode"]
        if any(v is None for v in (J_current, R_current, grad_H, g_port)):
            raise DiracMatchingError("PHS incompleto (J,R,∇H,g) o semilla ausente.")
        if requested_dt is None:
            requested_dt = 1e-3
        J_d = J_desired if J_desired is not None else J_current
        R_d = R_desired if R_desired is not None else R_current
        grad_Hd = grad_H_desired if grad_H_desired is not None else grad_H
        Kd = hessian_desired if hessian_desired is not None else hessian_current
        fM, m_ok = self._tuner.maupertuis_factor(kinetic_energy, potential_energy)
        pumping = bool(packed["pumping"]) if packed else False
        C = packed["C"] if packed else None
        x = packed["x"] if packed else None
        H = packed["H"] if packed else None
        ida = packed["ida"] if packed else None
        G = None if not ida else ida.get("G")
        x_star = None if not ida else ida.get("x_star")
        md = MatchingMode(mode) if mode is not None else MatchingMode.ENERGY_LEVEL
        if packed and self._prefer_seed_u and packed["u_app"].size:
            md_eff = md
        else:
            md_eff = md

        try:
            sol = self._solver.compute_control_law(
                J_current=J_current, R_current=R_current, grad_H=grad_H,
                J_desired=J_d, R_desired=R_d, grad_H_desired=grad_Hd,
                g_port=g_port, hessian_current=hessian_current, hessian_desired=Kd,
                casimir_basis=None if C is None else np.asarray(C, dtype=np.float64),
                x=x, x_star=None if x_star is None else np.asarray(x_star, dtype=np.float64),
                hamiltonian=H, target_hamiltonian=hamiltonian_target,
                ida_G=None if G is None else np.asarray(G, dtype=np.float64),
                mode=md_eff, pumping_required=pumping,
                maupertuis_factor=fM, maupertuis_applicable=m_ok,
            )
            if packed and self._prefer_seed_u and packed["u_app"].size == sol.alpha.size:
                # Contrato: no re-aplicar músculo. α ← u_app.
                sol = ControlSolution(
                    alpha=packed["u_app"].copy(), mode=sol.mode, H_dot_closed=sol.H_dot_closed,
                    desired_gradient=sol.desired_gradient, port_matrix=sol.port_matrix,
                    residual_norm=sol.residual_norm, residual_relative=sol.residual_relative,
                    orthogonal_residual=sol.orthogonal_residual, colinear_residual=sol.colinear_residual,
                    g_rank=sol.g_rank, singular_values=sol.singular_values,
                    condition_number_g=sol.condition_number_g, is_full_rank_g=sol.is_full_rank_g,
                    lyapunov_verified=sol.lyapunov_verified, required_forcing=sol.required_forcing,
                    poincare_certificate=sol.poincare_certificate,
                    power_requested=sol.power_requested, power_delivered=sol.power_delivered,
                    pumping_required=sol.pumping_required,
                )
        except TopologicalInvariantError as exc:
            if not self._crowbar_on_veto:
                raise
            g = np.asarray(g_port)
            n, m = g.shape[0], g.shape[1]
            st = self._governor.crowbar_state(m, n, float(requested_dt), str(exc))
            self._last_state = st
            if raise_on_veto:
                raise CrowbarEngagedError(str(exc)) from exc
            return st

        self._last_control = sol
        z0, c_med, _src = self._tuner.physical_z0_and_c(
            seed=seed, port_dim=sol.alpha.size,
        )
        term = self._solver.compute_port_termination(
            sol, z0=z0, maupertuis_factor=fM, maupertuis_applicable=m_ok,
        )
        self._last_term = term
        mu2 = float(packed["mu2"]) if packed else sol.poincare_certificate.logarithmic_norm_A
        rho = float(packed["rho"]) if packed else 0.0
        causal = self._tuner.compute_causal_speed(term, c_med, mu2_A=mu2, rho_curl_curl=rho)
        self._last_causal = causal
        diag = self._governor.diagnose(
            graph_laplacian=graph_laplacian, causal=causal,
            requested_dt=float(requested_dt),
            lyapunov_derivative=sol.H_dot_closed,
            pumping_required=sol.pumping_required, mode=sol.mode,
        )
        if diag["verdict"] != "COHERENT":
            if self._crowbar_on_veto:
                g = sol.port_matrix
                st = self._governor.crowbar_state(
                    g.shape[1], g.shape[0], float(requested_dt),
                    ", ".join(diag.get("veto_reasons", [])),
                )
                self._last_state = st
                if raise_on_veto:
                    raise CFLViolationError("Cono causal vetado: " + ", ".join(diag.get("veto_reasons", [])))
                return st
            if raise_on_veto:
                raise CFLViolationError("Cono causal vetado: " + ", ".join(diag.get("veto_reasons", [])))
        atlas = packed["atlas"] if packed else "abstract_phs"
        state = self._governor.synthesize_interconnection_state(
            sol, term, causal, float(diag["safe_dt"]), diag, atlas=atlas,
        )
        self._last_state = state
        logger.info(
            "[DiracAgent 7.1] OK mode=%s Ḣ=%.3e ‖Γ‖=%.3e c=%.3e dt=%.3e μ₂=%.3e verdict=%s crowbar=%s",
            sol.mode, sol.H_dot_closed, term.scattering_norm, causal.c_medium,
            state.safe_dt, causal.mu2_A, state.causal_verdict, state.crowbar,
        )
        return state

    def audit_poincare_dirac_ida_pbc_interconnection(
        self,
        J_desired: Array,
        R_desired: Array,
        grad_H_desired: Array,
        g_matrix: Array,
        hessian_desired: Optional[Array] = None,
        casimir_basis: Optional[Array] = None,
        maupertuis_factor: float = 1.0,
    ) -> PoincareDiracCertificate:
        """Auditoría I1–I8 sin resolver matching (estrato táctico)."""
        g = _as_mat(g_matrix, "g")
        grad = _as_1d(grad_H_desired, "∇H_d")
        z = np.zeros(g.shape[1])
        return self._solver.audit_poincare_dirac_structure(
            J=_as_sq(J_desired, "J_d"), R=_as_sq(R_desired, "R_d"), grad_H=grad,
            J_d=_as_sq(J_desired, "J_d"), R_d=_as_sq(R_desired, "R_d"), grad_H_d=grad,
            g=g, alpha=z, hessian=hessian_desired, hessian_d=hessian_desired,
            casimir_basis=casimir_basis, maupertuis_factor=maupertuis_factor,
            maupertuis_applicable=False, f_d=np.zeros_like(grad),
        )

    def diagnostic_report(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "agent": "DiracInterconnectionAgent",
            "version": _SCHEMA,
            "hodge_convention": _HODGE_CANON,
            "has_data": self._last_control is not None,
            "verdict": "UNKNOWN",
            "veto_reasons": [],
        }
        if self._last_control is not None:
            cs = self._last_control
            c = cs.poincare_certificate
            report["phase1"] = {
                "mode": cs.mode,
                "residual_relative": cs.residual_relative,
                "orthogonal_residual": cs.orthogonal_residual,
                "g_rank": cs.g_rank,
                "H_dot_closed": cs.H_dot_closed,
                "power_requested": cs.power_requested,
                "power_delivered": cs.power_delivered,
                "pumping_required": cs.pumping_required,
                "certificate": {
                    "is_dirac_valid": c.is_dirac_valid,
                    "is_passive_closed_loop": c.is_passive_closed_loop,
                    "is_casimir_immune": c.is_casimir_immune,
                    "casimir_port_leak": c.casimir_port_leak,
                    "is_la_salle_mod_casimir": c.is_la_salle_mod_casimir,
                    "is_equilibrium_assignable": c.is_equilibrium_assignable,
                    "logarithmic_norm_A": c.logarithmic_norm_A,
                    "matching_residual_relative": c.matching_residual_relative,
                    "liouville_trace": c.liouville_trace,
                    "is_volume_contracting": c.is_volume_contracting,
                },
            }
        if self._last_term is not None:
            t = self._last_term
            report["phase2"] = {
                "scattering_norm": t.scattering_norm,
                "suggests_hodge_update": t.suggests_hodge_update,
                "herglotz_status": t.herglotz_status,
                "maupertuis_applicable": t.maupertuis_applicable,
                "anisotropy_index": t.anisotropy_index,
                "n_active": int(np.sum(t.active_mask)),
            }
        if self._last_state is not None:
            s = self._last_state
            report["phase3"] = dict(s.poincare_causal_report)
            report["phase3"].update({
                "c_eff": s.c_eff, "safe_dt": s.safe_dt, "mu2_A": s.mu2_A,
                "crowbar": s.crowbar, "atlas": s.atlas,
            })
        reasons: List[str] = []
        if "phase1" in report:
            cert = report["phase1"]["certificate"]
            if not cert["is_dirac_valid"]:
                reasons.append("I1/I2")
            if not cert["is_casimir_immune"]:
                reasons.append("CASIMIR_LEAK")
            if not cert["is_passive_closed_loop"] and not report["phase1"]["pumping_required"]:
                reasons.append("I3")
        if "phase2" in report and report["phase2"]["suggests_hodge_update"]:
            reasons.append("HODGE_REWRITE_FORBIDDEN")
        if "phase3" in report and report["phase3"].get("verdict") == "VETOED":
            reasons.append("I5_CAUSAL")
        report["verdict"] = "COHERENT" if not reasons else "VETOED"
        report["veto_reasons"] = reasons
        return report


__all__ = [
    "TopologicalInvariantError",
    "DiracMatchingError",
    "ImpedanceMismatchError",
    "CFLViolationError",
    "LyapunovInstabilityError",
    "PoincareSymplecticError",
    "MaupertuisViolationError",
    "EnergyMatchingError",
    "LanczosConvergenceError",
    "CasimirPortLeakError",
    "SchemaContractError",
    "CrowbarEngagedError",
    "MatchingMode",
    "PoincareDiracCertificate",
    "PortTermination",
    "CausalSpeed",
    "ControlSolution",
    "InterconnectionState",
    "Phase1_IDAPBC_PoincareSolver",
    "Phase2_PortScattering",
    "Phase3_CFLGovernor",
    "DiracInterconnectionAgent",
]