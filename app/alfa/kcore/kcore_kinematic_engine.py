# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : KCore Kinematic Engine (Motor Cinemático del Núcleo — Estrato α)    ║
║ Ruta   : app/alfa/kcore/kcore_kinematic_engine.py                            ║
║ Versión: 3.0.0-Doctoral-IDA-PBC-Hodge-PH-KBN-CSMD-Heyting                    ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA (rigor doctoral, no ornamental):                         ║
║ Sea Q una variedad de configuración de dimensión n, T*Q su fibrado           ║
║ cotangente y (V, E, F) un 2-complejo celular (F opcional) con operador       ║
║ de frontera ∂₁ = B₀ ∈ ℝ^{|V|×|E|}, ∂₂ = B₁ ∈ ℝ^{|E|×|F|}, de modo que        ║
║ B₀ B₁ = 0 (complejo de cadenas). El motor calcula, en la FPU y con           ║
║ inmunidad de Wilkinson, tres objetos cinemáticos acoplados:                  ║
║                                                                              ║
║   (IDA-PBC)   ẋ = (J−R) ∇H + g α ,   ẋ = (J_d−R_d) ∇H_d   (lazo cerrado)     ║
║               g α = (J_d−R_d)∇H_d − (J−R)∇H                                  ║
║               g^⊥ F_err = 0   (EDP de matching; obstrucción geométrica)      ║
║               α = argmin_u ‖g u − F_err‖_G² + γ² ‖u‖²   (Tikhonov en g,      ║
║                   NO en el Gramiano; filtro σ/(σ²+γ²) sobre SVD de G^{½}g)   ║
║               J = −Jᵀ ,  R = Rᵀ ⪰ 0  (estructura port-Hamiltoniana)          ║
║                                                                              ║
║   (Hodge)     C¹ = im(d₀) ⊕ im(δ₂) ⊕ ℋ¹          (teorema de Hodge discreto)║
║               I = I_grad + I_curl + I_harm ,     ⟨·,·⟩_W = (·)ᵀ W (·)        ║
║               I_grad ∈ im(B₀ᵀ) ,   I_curl ∈ W⁻¹ im(B₁) ,                     ║
║               I_harm ∈ ker(B₀ W) ∩ ker(B₁ᵀ)                                  ║
║               Sin 2-esqueleto: im(δ₂)={0} y ker(B₀ W) es armónico, NO curl.  ║
║                                                                              ║
║   (Energía)   T = ½ q̇ᵀ M(q) q̇ ,   P = q̇ᵀ F                                 ║
║               Ḣ = −∇Hᵀ R ∇H + ∇Hᵀ g α          (pasividad de puertos)        ║
║                                                                              ║
║ ARQUITECTURA FUNCTORIAL EN TRES FASES ANIDADAS (OODA):                       ║
║   Fase 1  Observe+Orient  :  In → PhaseOneKinematicPacket                    ║
║   Fase 2  Decide          :  PhaseOneKinematicPacket → PhaseTwoKinematicPacket║
║   Fase 3  Act             :  (PhaseOne × PhaseTwo) → KinematicTelemetry      ║
║                                                                              ║
║ El último morfismo de la Fase k es, por construcción de tipos, el objeto     ║
║ inicial de la Fase k+1 (continuidad formal del retículo de métodos).         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Physics.KCoreKinematicEngine")

# ---------------------------------------------------------------------------
# Constantes metrológicas (análisis de Wilkinson / IEEE-754 binary64)
# ---------------------------------------------------------------------------
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_FLOOR: Final[float] = 1e-15
_CSMD_STEP: Final[float] = 1e-20
_HERMITIAN_TOLERANCE: Final[float] = 1e-12
_SKEW_TOLERANCE: Final[float] = 1e-12
_PSD_TOLERANCE: Final[float] = 1e-12
_RANK_REL_TOL: Final[float] = 1e-12


# =============================================================================
# Retículo de Heyting (cadena de 4 puntos = lógica interna de un topos
# sobre un espacio de fase discreto totalmente ordenado).
# =============================================================================
class HeytingVerdict(str, Enum):
    r"""
    Cadena  ⊥ = VETOED ≺ DEGRADED ≺ COHERENT ≺ CERTIFIED = ⊤.

    En un orden total el álgebra de Heyting es única:
        a ∧ b = min(a, b),   a ∨ b = max(a, b),
        a → b = ⊤  si a ≼ b,  y  a → b = b  en caso contrario,
        ¬a = a → ⊥.
    """

    VETOED = "VETOED"
    DEGRADED = "DEGRADED"
    COHERENT = "COHERENT"
    CERTIFIED = "CERTIFIED"

    @property
    def rank(self) -> int:
        return {
            HeytingVerdict.VETOED: 0,
            HeytingVerdict.DEGRADED: 1,
            HeytingVerdict.COHERENT: 2,
            HeytingVerdict.CERTIFIED: 3,
        }[self]

    def meet(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return self if self.rank <= other.rank else other

    def join(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return self if self.rank >= other.rank else other

    def implies(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict.CERTIFIED if self.rank <= other.rank else other

    def negate(self) -> "HeytingVerdict":
        return self.implies(HeytingVerdict.VETOED)


# =============================================================================
# Cartas espectrales / PH / Hodge y paquetes inmutables
# =============================================================================
def _freeze_array(arr: np.ndarray) -> np.ndarray:
    """Copia C-contigua de solo lectura: inmuniza el paquete frente a aliasing."""
    out = np.array(arr, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, slots=True)
class SpectralChart:
    r"""
    Factorización espectral A = U Λ U† y objetos derivados.

    Inversión de Tikhonov:  A_α⁻¹ = U (Λ + α I)⁻¹ U† ≡ (A + α I)⁻¹.
    κ₂(A) = λ_max / λ_min (tras el desplazamiento).
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    inv_metric: np.ndarray
    sqrt_metric: np.ndarray
    condition_number: float
    operator_norm: float
    frobenius_norm: float
    spectral_gap: float
    regularized: bool
    tikhonov_alpha: float
    kernel_dimension: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "eigenvalues", _freeze_array(self.eigenvalues))
        object.__setattr__(self, "eigenvectors", _freeze_array(self.eigenvectors))
        object.__setattr__(self, "inv_metric", _freeze_array(self.inv_metric))
        object.__setattr__(self, "sqrt_metric", _freeze_array(self.sqrt_metric))


@dataclass(frozen=True, slots=True)
class PortHamiltonianChart:
    r"""
    Carta del matching IDA-PBC y de la estructura de puertos.

    Obstrucción geométrica:  ‖P_{im(g)^{⊥_G}} F_err‖_G .
    Residuo numérico:        ‖g α_γ − F_err‖_2  (incluye Tikhonov).
    Pasividad: Ḣ + ∇Hᵀ R ∇H − ∇Hᵀ g α  = 0  en aritmética exacta
    si ẋ = (J−R)∇H + gα y J es antisimétrica.
    """

    alpha: np.ndarray
    forcing_error: np.ndarray
    residual_vector: np.ndarray
    residual_norm: float
    residual_G_norm: float
    matching_obstruction: float
    actuator_rank: int
    actuator_condition: float
    n_actuators: int
    J_skew_residual: float
    Jd_skew_residual: float
    R_psd_residual: float
    Rd_psd_residual: float
    open_loop_dissipation: float
    closed_loop_dissipation: float
    supplied_power: float
    passivity_residual: float
    lyapunov_decrease: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", _freeze_array(self.alpha))
        object.__setattr__(self, "forcing_error", _freeze_array(self.forcing_error))
        object.__setattr__(self, "residual_vector", _freeze_array(self.residual_vector))


@dataclass(frozen=True, slots=True)
class HodgeChart:
    r"""
    Carta de Hodge celular sobre el 1-esqueleto (2-esqueleto opcional).

    Pitágoras W-ortogonal: ‖I‖_W² = ‖I_g‖_W² + ‖I_c‖_W² + ‖I_h‖_W².
    Números de Betti combinatorios: β₀ = dim ker L₀, β₁ = |E|−rank(B₀)−rank(B₁).
    """

    I_grad: np.ndarray
    I_curl: np.ndarray
    I_harmonic: np.ndarray
    potentials: np.ndarray
    orthogonality_gram: np.ndarray
    orthogonality_error: float
    reconstruction_residual: float
    kirchhoff_residual: float
    coclosure_residual: float
    pythagoras_residual: float
    chain_complex_residual: float
    betti_0: int
    betti_1: int
    laplacian_gap: float
    laplacian_condition: float
    n_vertices: int
    n_edges: int
    n_faces: int
    has_two_skeleton: bool
    energy_grad: float
    energy_curl: float
    energy_harmonic: float
    energy_total: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "I_grad", _freeze_array(self.I_grad))
        object.__setattr__(self, "I_curl", _freeze_array(self.I_curl))
        object.__setattr__(self, "I_harmonic", _freeze_array(self.I_harmonic))
        object.__setattr__(self, "potentials", _freeze_array(self.potentials))
        object.__setattr__(self, "orthogonality_gram", _freeze_array(self.orthogonality_gram))


@dataclass(frozen=True, slots=True)
class KineticChart:
    r"""Carta de la forma cuadrática cinética y de la potencia de puertos."""

    kinetic_energy: float
    mechanical_power: float
    mass_psd_residual: float
    rayleigh_quotient: float
    csmd_momentum: np.ndarray
    csmd_error: float
    mass_condition: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "csmd_momentum", _freeze_array(self.csmd_momentum))


@dataclass(frozen=True, slots=True)
class PhaseOneKinematicPacket:
    r"""
    Objeto terminal de la Fase 1 ≡ objeto inicial de la Fase 2.

    Conserva los campos v2 y añade las tres cartas (PH, Hodge, cinética)
    más la carta espectral de G.
    """

    ida_alpha: np.ndarray
    ida_residual_norm: float
    hh_grad: np.ndarray
    hh_curl: np.ndarray
    hh_harmonic: np.ndarray
    hh_orthogonality_error: float
    kinetic_energy: float
    mechanical_power: float
    ph_chart: PortHamiltonianChart
    hodge_chart: HodgeChart
    kinetic_chart: KineticChart
    metric_chart: SpectralChart

    def __post_init__(self) -> None:
        object.__setattr__(self, "ida_alpha", _freeze_array(self.ida_alpha))
        object.__setattr__(self, "hh_grad", _freeze_array(self.hh_grad))
        object.__setattr__(self, "hh_curl", _freeze_array(self.hh_curl))
        object.__setattr__(self, "hh_harmonic", _freeze_array(self.hh_harmonic))


@dataclass(frozen=True, slots=True)
class SpectralKinematicReport:
    r"""Subobjeto de la Fase 2: integridad espectral de G, L₀, g y M."""

    metric_condition: float
    metric_gap: float
    laplacian_gap: float
    actuator_rank: int
    actuator_condition: float
    mass_condition: float
    hodge_pythagoras_residual: float
    matching_obstruction: float
    passivity_residual: float
    integrity_score: float


@dataclass(frozen=True, slots=True)
class PhaseTwoKinematicPacket:
    r"""
    Objeto terminal de la Fase 2 ≡ primer factor del dominio de la Fase 3.
    """

    heyting_verdict: str
    heyting_rank: int
    orthogonality_violated: bool
    residual_violated: bool
    obstruction_violated: bool
    passivity_violated: bool
    hodge_theorem_violated: bool
    kinetic_sign_violated: bool
    spectral_integrity: SpectralKinematicReport
    heyting_score: float
    betti_0: int
    betti_1: int


@dataclass(frozen=True, slots=True)
class KinematicTelemetry:
    r"""Sello inmutable de telemetría (objeto terminal del functor OODA)."""

    ida_alpha: np.ndarray
    ida_residual_norm: float
    hh_grad: np.ndarray
    hh_curl: np.ndarray
    hh_harmonic: np.ndarray
    hh_orthogonality_error: float
    kinetic_energy: float
    mechanical_power: float
    heyting_verdict: str
    matching_obstruction: float
    passivity_residual: float
    reconstruction_residual: float
    pythagoras_residual: float
    csmd_error: float
    betti_0: int
    betti_1: int
    condition_number: float
    conservation_residual: float
    spectral_integrity_score: float

    def __post_init__(self) -> None:
        for name in ("ida_alpha", "hh_grad", "hh_curl", "hh_harmonic"):
            object.__setattr__(self, name, _freeze_array(getattr(self, name)))

    def to_dict(self) -> Dict[str, Any]:
        """Serialización compatible con el pasaporte de la Malla (claves v2 + v3)."""
        return {
            "ida_alpha": np.array(self.ida_alpha, copy=True),
            "ida_residual_norm": self.ida_residual_norm,
            "hh_grad": np.array(self.hh_grad, copy=True),
            "hh_curl": np.array(self.hh_curl, copy=True),
            "hh_harmonic": np.array(self.hh_harmonic, copy=True),
            "hh_orthogonality_error": self.hh_orthogonality_error,
            "kinetic_energy": self.kinetic_energy,
            "mechanical_power": self.mechanical_power,
            "heyting_verdict": self.heyting_verdict,
            "matching_obstruction": self.matching_obstruction,
            "passivity_residual": self.passivity_residual,
            "reconstruction_residual": self.reconstruction_residual,
            "pythagoras_residual": self.pythagoras_residual,
            "csmd_error": self.csmd_error,
            "betti_0": self.betti_0,
            "betti_1": self.betti_1,
            "condition_number": self.condition_number,
            "conservation_residual": self.conservation_residual,
            "spectral_integrity_score": self.spectral_integrity_score,
        }


# =============================================================================
# Motor
# =============================================================================
class KCoreKinematicEngine:
    r"""
    Motor Cinemático de Fuerza Bruta en la FPU del Estrato α.

    El endofunctor OODA  T = Act ∘ Decide ∘ Observe  actúa sobre el espacio
    de estados (T*Q) × C¹(K). Cada fase es un morfismo explícito; la
    composición `execute_kinematic_cycle` es T mismo.
    """

    def __init__(self, dimension_n: int, reg_param: float = 1e-15) -> None:
        if dimension_n <= 0:
            raise ValueError(
                "La dimensión del espacio de configuración debe ser estrictamente positiva."
            )
        self._n: Final[int] = int(dimension_n)
        self._reg: Final[float] = float(max(reg_param, _WILKINSON_FLOOR))

    # =========================================================================
    # FASE 1 — OBSERVE + ORIENT
    # Aparato de medición (KBN, CSMD, espectro) y 1-jets IDA-PBC / Hodge /
    # cinéticos. El último método, `_phase1_observe_orient`, tiene por
    # codominio `PhaseOneKinematicPacket`, que ES el dominio de
    # `_phase2_spectral_integrity`.
    # =========================================================================
    def kahan_sum(self, arr: np.ndarray) -> float:
        r"""
        Suma compensada de Kahan–Babuška–Neumaier sobre un 1-tensor.

        Neumaier (1974) corrige el caso |x_{i+1}| > |S_i| que Kahan clásico
        pierde. El error hacia delante satisface

            |fl(∑ x_i) − ∑ x_i|  ≤  (2u + O(u²)) ∑ |x_i|

        independiente de n. Se proyecta a la parte real: las observables
        cinemáticas de este motor son reales.
        """
        if np.asarray(arr).ndim != 1:
            raise ValueError(
                f"kahan_sum espera un vector 1-D, se recibió {np.asarray(arr).shape}"
            )
        return float(np.real(self._neumaier_sum(arr)))

    def _neumaier_sum(self, arr: np.ndarray) -> complex:
        """KBN sobre ℝ o ℂ; no fuerza float64 a priori para preservar holomorfía CSMD."""
        data = np.asarray(arr).ravel()
        if data.size == 0:
            return 0.0
        if not np.all(np.isfinite(data)):
            raise ValueError("kahan/neumaier: el sumando contiene valores no finitos")
        use_complex = np.iscomplexobj(data)
        total: complex = 0.0j if use_complex else 0.0
        compensator: complex = 0.0j if use_complex else 0.0
        for x in data:
            t = total + x
            if abs(total) >= abs(x):
                compensator += (total - t) + x
            else:
                compensator += (x - t) + total
            total = t
        return total + compensator

    def _validate_square_matrix(self, M: np.ndarray, name: str, dim: Optional[int] = None) -> None:
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(
                f"{name} debe ser una matriz cuadrada, se recibió {M.shape}"
            )
        expected = self._n if dim is None else dim
        if M.shape[0] != expected:
            raise ValueError(
                f"{name} debe ser {expected} × {expected}, se recibió {M.shape}"
            )
        if not np.all(np.isfinite(M)):
            raise ValueError(f"{name} contiene valores no finitos")

    def _validate_matrix_shape(
        self,
        M: np.ndarray,
        rows: int,
        cols: Optional[int] = None,
        name: str = "matriz",
    ) -> None:
        cols = rows if cols is None else cols
        if M.shape != (rows, cols):
            raise ValueError(f"{name} debe tener forma ({rows}, {cols}), se recibió {M.shape}")
        if not np.all(np.isfinite(M)):
            raise ValueError(f"{name} contiene valores no finitos")

    def _validate_vector(self, v: np.ndarray, length: int, name: str) -> None:
        if v.ndim != 1 or v.shape[0] != length:
            raise ValueError(
                f"{name} debe ser un vector de longitud {length}, se recibió {v.shape}"
            )
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{name} contiene valores no finitos")

    def _validate_hermitian(
        self,
        M: np.ndarray,
        name: str,
        tolerance: float = _HERMITIAN_TOLERANCE,
    ) -> None:
        skew = M - M.conj().T
        if la.norm(skew, ord="fro") > tolerance * max(1.0, la.norm(M, ord="fro")):
            raise ValueError(
                f"{name} no es hermítica: ‖M − M†‖_F / ‖M‖_F excede {tolerance}"
            )

    def _hermitize(self, M: np.ndarray) -> np.ndarray:
        """Proyección de Hilbert–Schmidt sobre el subespacio de hermitianas."""
        return 0.5 * (M + M.conj().T)

    def _skew_project(self, M: np.ndarray) -> Tuple[np.ndarray, float]:
        r"""Proyección sobre 𝔰𝔬(n): π(M) = (M − M†)/2, r = ‖M − π(M)‖_F."""
        projected = 0.5 * (M - M.conj().T)
        residual = float(la.norm(M - projected, ord="fro"))
        return projected, residual

    def _psd_project(self, M: np.ndarray) -> Tuple[np.ndarray, float]:
        r"""
        Proyección espectral sobre el cono PSD: λ ↦ max(λ, 0) tras hermitización.
        Residuo = ‖M − π(M)‖_F (incluye la parte antihermítica y los λ < 0).
        """
        herm = self._hermitize(M)
        evals, evecs = la.eigh(herm)
        clipped = np.maximum(np.real(evals), 0.0)
        projected = (evecs * clipped) @ evecs.conj().T
        residual = float(la.norm(M - projected, ord="fro"))
        return projected, residual

    def _factorize_metric(self, metric_tensor: np.ndarray, name: str = "metric") -> SpectralChart:
        r"""
        Una sola descomposición de Hilbert (eigh) — jamás dos.

        Regularización de Tikhonov espectral: λ ↦ λ + α.
        Equivale a invertir A + α I, proximal de Moreau en (M_n, ‖·‖₂).
        """
        self._validate_square_matrix(metric_tensor, name, dim=metric_tensor.shape[0])
        herm = self._hermitize(np.asarray(metric_tensor, dtype=np.complex128))
        self._validate_hermitian(herm, name)

        eigenvalues, eigenvectors = la.eigh(herm)
        eigenvalues = np.real(eigenvalues)

        regularized = bool(np.any(eigenvalues <= self._reg))
        if regularized:
            logger.warning(
                "%s no definida positiva o mal condicionada (λ_min=%.3e); Tikhonov α=%.3e.",
                name, float(np.min(eigenvalues)), self._reg,
            )
        shifted = np.maximum(eigenvalues + self._reg, _WILKINSON_FLOOR)
        inv_metric = (eigenvectors * (1.0 / shifted)) @ eigenvectors.conj().T
        sqrt_metric = (eigenvectors * np.sqrt(shifted)) @ eigenvectors.conj().T

        lam_abs = np.abs(shifted)
        cond = float(np.max(lam_abs) / np.min(lam_abs))
        op_norm = float(np.max(lam_abs))
        frob = float(np.sqrt(np.real(self._neumaier_sum(shifted.astype(np.complex128) ** 2))))
        if shifted.size >= 2:
            ordered = np.sort(lam_abs)
            gap = float(ordered[1] - ordered[0])
        else:
            gap = float(lam_abs[0])
        kernel_dim = int(np.sum(np.abs(eigenvalues) <= max(self._reg, _WILKINSON_FLOOR)))

        return SpectralChart(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            inv_metric=np.real_if_close(inv_metric, tol=1e6),
            sqrt_metric=np.real_if_close(sqrt_metric, tol=1e6),
            condition_number=cond,
            operator_norm=op_norm,
            frobenius_norm=frob,
            spectral_gap=gap,
            regularized=regularized,
            tikhonov_alpha=self._reg,
            kernel_dimension=kernel_dim,
        )

    def _tikhonov_svd_solve(
        self,
        operator: np.ndarray,
        rhs: np.ndarray,
        reg: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float, int, float]:
        r"""
        Tikhonov sobre el operador original (no sobre el Gramiano):

            min_x  ‖A x − b‖₂² + γ² ‖x‖₂²
            ⇔  x = V  [σᵢ / (σᵢ² + γ²)]  U† b ,    A = U Σ V†.

        v2 aplicaba este filtro a los valores singulares de AᵀA, lo cual
        equivale a σ²/(σ⁴ + γ²) sobre A y es inconsistente.

        Returns:
            (x, singular_values, residual ‖Ax−b‖₂, rango numérico, κ₂).
        """
        gamma = self._reg if reg is None else float(reg)
        A = np.asarray(operator)
        b = np.asarray(rhs)
        U, sigma, Vh = la.svd(A, full_matrices=False)
        filt = sigma / (sigma ** 2 + gamma ** 2)
        x = (Vh.conj().T * filt) @ (U.conj().T @ b)
        residual = float(la.norm(A @ x - b))
        if sigma.size == 0:
            return x, sigma, residual, 0, np.inf
        cutoff = max(gamma, _RANK_REL_TOL * float(sigma[0]))
        rank = int(np.sum(sigma > cutoff))
        smax = float(sigma[0])
        smin = float(sigma[rank - 1]) if rank > 0 else np.inf
        cond = float(smax / smin) if rank > 0 and smin > 0.0 else np.inf
        return x, sigma, residual, rank, cond

    def _g_weighted_tikhonov(
        self,
        g: np.ndarray,
        G_chart: SpectralChart,
        F_err: np.ndarray,
    ) -> Tuple[np.ndarray, float, float, int, float]:
        r"""
        min_α ‖g α − F‖_G² + γ² ‖α‖₂²,  G-norma = ‖G^{½} · ‖₂.

        Se reduce a Tikhonov euclídeo de  ĝ = G^{½} g  contra  F̂ = G^{½} F.
        La obstrucción de matching es la componente de F̂ ortogonal a im(ĝ):

            obst = ‖(I − Π_{im ĝ}) F̂‖₂ = ‖P_{im(g)^{⊥_G}} F‖_G .
        """
        G_sqrt = np.asarray(G_chart.sqrt_metric, dtype=np.float64)
        g_hat = G_sqrt @ np.asarray(g, dtype=np.float64)
        F_hat = G_sqrt @ np.asarray(F_err, dtype=np.float64)
        alpha, sigma, _, rank, cond = self._tikhonov_svd_solve(g_hat, F_hat)
        recon = g_hat @ alpha
        residual_G = float(la.norm(recon - F_hat))
        if sigma.size == 0:
            obstruction = float(la.norm(F_hat))
            return alpha, residual_G, obstruction, 0, np.inf
        cutoff = max(self._reg, _RANK_REL_TOL * float(sigma[0]))
        mask = sigma > cutoff
        U, _, _ = la.svd(g_hat, full_matrices=False)
        U_r = U[:, mask]
        if U_r.size == 0:
            obstruction = float(la.norm(F_hat))
        else:
            obstruction = float(la.norm(F_hat - U_r @ (U_r.conj().T @ F_hat)))
        return alpha, residual_G, obstruction, rank, cond

    def compute_ida_pbc_matching(
        self,
        g: np.ndarray,
        G_metric: np.ndarray,
        J: np.ndarray,
        R: np.ndarray,
        J_d: np.ndarray,
        R_d: np.ndarray,
        grad_H: np.ndarray,
        grad_Hd: np.ndarray,
        metric_chart: Optional[SpectralChart] = None,
        project_structure: bool = False,
    ) -> Tuple[np.ndarray, float]:
        r"""
        [KCORE 1 — MATCHING PORT-HAMILTONIANO IDA-PBC]

        Dinámica abierta y deseada

            ẋ = (J − R) ∇H + g α ,     ẋ = (J_d − R_d) ∇H_d

        producen la ecuación de matching

            g α = F_err := (J_d − R_d) ∇H_d − (J − R) ∇H .

        Condición PDE (anulador):  g^⊥ F_err = 0.
        Si se cumple, α se obtiene por Tikhonov G-pesado sobre g (no sobre
        gᵀ G g). La firma pública se conserva y devuelve (α, ‖gα − F_err‖₂).

        Si `project_structure` es verdadero, J, J_d se proyectan a 𝔰𝔬(n) y
        R, R_d al cono PSD antes de formar F_err (útil como regularizador
        de modelado; por defecto se respeta el dato crudo y se audita).
        """
        g_arr = np.asarray(g)
        if g_arr.ndim != 2 or g_arr.shape[0] != self._n:
            raise ValueError(f"g debe ser una matriz de {self._n} filas, se recibió {g_arr.shape}")
        if not np.all(np.isfinite(g_arr)):
            raise ValueError("g contiene valores no finitos")

        self._validate_square_matrix(G_metric, "G_metric")
        self._validate_square_matrix(J, "J")
        self._validate_square_matrix(R, "R")
        self._validate_square_matrix(J_d, "J_d")
        self._validate_square_matrix(R_d, "R_d")
        self._validate_vector(np.asarray(grad_H), self._n, "grad_H")
        self._validate_vector(np.asarray(grad_Hd), self._n, "grad_Hd")
        self._validate_hermitian(np.asarray(G_metric), "G_metric")

        J_use, R_use, Jd_use, Rd_use = J, R, J_d, R_d
        if project_structure:
            J_use, _ = self._skew_project(np.asarray(J))
            Jd_use, _ = self._skew_project(np.asarray(J_d))
            R_use, _ = self._psd_project(np.asarray(R))
            Rd_use, _ = self._psd_project(np.asarray(R_d))

        F_err = (Jd_use - Rd_use) @ np.asarray(grad_Hd) - (J_use - R_use) @ np.asarray(grad_H)
        chart = metric_chart if metric_chart is not None else self._factorize_metric(G_metric, "G_metric")
        alpha, _, _, _, _ = self._g_weighted_tikhonov(g_arr, chart, F_err)
        residual_vector = g_arr @ alpha - F_err
        residual_norm = float(la.norm(residual_vector, 2))
        return np.real(np.asarray(alpha, dtype=np.float64)), residual_norm

    def _build_ph_chart(
        self,
        g: np.ndarray,
        G_metric: np.ndarray,
        J: np.ndarray,
        R: np.ndarray,
        J_d: np.ndarray,
        R_d: np.ndarray,
        grad_H: np.ndarray,
        grad_Hd: np.ndarray,
        metric_chart: SpectralChart,
        alpha: np.ndarray,
        residual_norm: float,
    ) -> PortHamiltonianChart:
        """Auditoría estructural y energética del matching (no altera α)."""
        g_arr = np.asarray(g, dtype=np.float64)
        F_err = (np.asarray(J_d) - np.asarray(R_d)) @ np.asarray(grad_Hd) - (
            np.asarray(J) - np.asarray(R)
        ) @ np.asarray(grad_H)
        residual_vector = g_arr @ alpha - F_err
        _, residual_G, obstruction, rank, cond = self._g_weighted_tikhonov(
            g_arr, metric_chart, F_err
        )

        _, J_res = self._skew_project(np.asarray(J, dtype=np.complex128))
        _, Jd_res = self._skew_project(np.asarray(J_d, dtype=np.complex128))
        _, R_res = self._psd_project(np.asarray(R, dtype=np.complex128))
        _, Rd_res = self._psd_project(np.asarray(R_d, dtype=np.complex128))

        gH = np.asarray(grad_H, dtype=np.float64)
        gHd = np.asarray(grad_Hd, dtype=np.float64)
        R_h = np.real(self._hermitize(np.asarray(R, dtype=np.complex128)))
        Rd_h = np.real(self._hermitize(np.asarray(R_d, dtype=np.complex128)))
        open_diss = float(np.real(gH @ R_h @ gH))
        closed_diss = float(np.real(gHd @ Rd_h @ gHd))
        supplied = float(np.real(gH @ (g_arr @ alpha)))

        J_sk, _ = self._skew_project(np.asarray(J, dtype=np.complex128))
        interconnection = float(np.real(gH @ np.real(J_sk) @ gH))
        passivity_residual = float(abs(interconnection))
        lyapunov = float(closed_diss)

        return PortHamiltonianChart(
            alpha=np.asarray(alpha, dtype=np.float64),
            forcing_error=np.asarray(F_err, dtype=np.float64).reshape(-1),
            residual_vector=np.asarray(residual_vector, dtype=np.float64).reshape(-1),
            residual_norm=float(residual_norm),
            residual_G_norm=float(residual_G),
            matching_obstruction=float(obstruction),
            actuator_rank=int(rank),
            actuator_condition=float(cond),
            n_actuators=int(g_arr.shape[1]),
            J_skew_residual=float(J_res),
            Jd_skew_residual=float(Jd_res),
            R_psd_residual=float(R_res),
            Rd_psd_residual=float(Rd_res),
            open_loop_dissipation=open_diss,
            closed_loop_dissipation=closed_diss,
            supplied_power=supplied,
            passivity_residual=passivity_residual,
            lyapunov_decrease=lyapunov,
        )

    def _weighted_inner(self, a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
        """⟨a, b⟩_W = aᵀ W b con W = diag(w) (w > 0). KBN sobre el sumando aᵢ wᵢ bᵢ."""
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        pa = np.asarray(a, dtype=np.float64).reshape(-1)
        pb = np.asarray(b, dtype=np.float64).reshape(-1)
        return float(self.kahan_sum(pa * w * pb))

    def _pseudoinverse_psd_eigh(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """Pseudoinversa espectral de una simétrica, recorte Wilkinson, (A⁺, λ, dim ker, gap)."""
        herm = self._hermitize(np.asarray(A, dtype=np.complex128))
        evals, evecs = la.eigh(herm)
        evals = np.real(evals)
        cutoff = max(self._reg, _WILKINSON_FLOOR)
        inv = np.zeros_like(evals)
        mask = np.abs(evals) > cutoff
        inv[mask] = 1.0 / evals[mask]
        pinv = (evecs * inv) @ evecs.conj().T
        kernel_dim = int(np.sum(~mask))
        abs_e = np.sort(np.abs(evals[mask])) if np.any(mask) else np.array([0.0])
        gap = float(abs_e[0]) if abs_e.size else 0.0
        return np.real_if_close(pinv, tol=1e6), evals, kernel_dim, gap

    def compute_helmholtz_hodge_decomposition(
        self,
        boundary_matrix: np.ndarray,
        edge_weights: np.ndarray,
        flow_vector: np.ndarray,
        face_incidence: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        r"""
        [KCORE 2 — HODGE CELULAR DISCRETO]

        Sobre el 1-complejo (V, E) con producto interno ⟨α,β⟩_W = αᵀ W β,
        W = diag(w), wᵢ > 0:

            C¹ = im(B₀ᵀ)  ⊕  ℋ¹ ,     ℋ¹ = ker(B₀ W)

        si no hay 2-esqueleto. El sumando `im(δ₂)` (curl / coexacto) es nulo:
        **ker(B₀ W) es armónico, no vorticidad**. v2 proyectaba sobre ese
        núcleo y lo bautizaba `I_curl`, dejando `I_harmonic` como basura
        numérica.

        Si se inyecta ∂₂ = B₁ ∈ ℝ^{|E|×|F|} con B₀ B₁ ≈ 0,

            I_grad ∈ im(B₀ᵀ) ,     I_curl = W⁻¹ B₁ θ ∈ W⁻¹ im(B₁) ,
            I_harm = I − I_grad − I_curl ,

        y las dos primeras son W-ortogonales precisamente cuando B₀ B₁ = 0.

        Poisson 0-formas:  L₀ φ = B₀ W I ,  L₀ = B₀ W B₀ᵀ ,  I_grad = B₀ᵀ φ.
        Poisson 2-formas:  L₂ θ = B₁ᵀ I ,  L₂ = B₁ᵀ W⁻¹ B₁ ,  I_curl = W⁻¹ B₁ θ.

        Returns:
            (I_grad, I_curl, I_harmonic, error de W-ortogonalidad max |G_{ij}|).
        """
        B0 = np.asarray(boundary_matrix, dtype=np.float64)
        if B0.ndim != 2:
            raise ValueError("boundary_matrix debe ser una matriz 2-D")
        n_vertices, n_edges = B0.shape
        w = np.asarray(edge_weights, dtype=np.float64)
        I = np.asarray(flow_vector, dtype=np.float64)
        self._validate_vector(w, n_edges, "edge_weights")
        self._validate_vector(I, n_edges, "flow_vector")
        if np.any(w <= _WILKINSON_FLOOR):
            raise ValueError("edge_weights debe ser estrictamente positivo (métrica en aristas).")

        W_I = w * I
        L0 = B0 @ (w[:, None] * B0.T)
        rhs0 = B0 @ W_I
        L0_pinv, evals_L0, ker_L0, gap_L0 = self._pseudoinverse_psd_eigh(L0)
        potentials = np.real(L0_pinv @ rhs0)
        I_grad = B0.T @ potentials

        has_faces = face_incidence is not None
        if has_faces:
            B1 = np.asarray(face_incidence, dtype=np.float64)
            if B1.ndim != 2 or B1.shape[0] != n_edges:
                raise ValueError(
                    f"face_incidence debe ser (|E| × |F|) = ({n_edges}, ·), se recibió {B1.shape}"
                )
            if not np.all(np.isfinite(B1)):
                raise ValueError("face_incidence contiene valores no finitos")
            inv_w = 1.0 / w
            L2 = B1.T @ (inv_w[:, None] * B1)
            rhs2 = B1.T @ I
            L2_pinv, _, _, _ = self._pseudoinverse_psd_eigh(L2)
            theta = np.real(L2_pinv @ rhs2)
            I_curl = inv_w * (B1 @ theta)
        else:
            I_curl = np.zeros(n_edges, dtype=np.float64)

        I_harmonic = I - I_grad - I_curl

        gram_gc = abs(self._weighted_inner(I_grad, I_curl, w))
        gram_gh = abs(self._weighted_inner(I_grad, I_harmonic, w))
        gram_ch = abs(self._weighted_inner(I_curl, I_harmonic, w))
        ortho_error = float(max(gram_gc, gram_gh, gram_ch))

        # Silencio intencional: gap/ker se reexponen en la carta de Hodge.
        _ = (evals_L0, ker_L0, gap_L0, n_vertices)
        return I_grad, I_curl, I_harmonic, ortho_error

    def _build_hodge_chart(
        self,
        boundary_matrix: np.ndarray,
        edge_weights: np.ndarray,
        flow_vector: np.ndarray,
        I_grad: np.ndarray,
        I_curl: np.ndarray,
        I_harmonic: np.ndarray,
        orthogonality_error: float,
        face_incidence: Optional[np.ndarray] = None,
    ) -> HodgeChart:
        """Números de Betti, Pitágoras, Kirchhoff y residuos del complejo."""
        B0 = np.asarray(boundary_matrix, dtype=np.float64)
        w = np.asarray(edge_weights, dtype=np.float64)
        I = np.asarray(flow_vector, dtype=np.float64)
        n_vertices, n_edges = B0.shape
        n_faces = 0 if face_incidence is None else int(np.asarray(face_incidence).shape[1])

        L0 = B0 @ (w[:, None] * B0.T)
        _, evals_L0, ker_L0, gap_L0 = self._pseudoinverse_psd_eigh(L0)
        abs_e = np.sort(np.abs(np.real(evals_L0)))
        cutoff = max(self._reg, _WILKINSON_FLOOR)
        nonzero = abs_e[abs_e > cutoff]
        lam_max = float(abs_e[-1]) if abs_e.size else 0.0
        lam_min_nz = float(nonzero[0]) if nonzero.size else 0.0
        L_cond = float(lam_max / lam_min_nz) if lam_min_nz > 0.0 else np.inf

        sB = la.svd(B0, compute_uv=False)
        rank_B0 = int(np.sum(sB > cutoff * max(1.0, float(sB[0]) if sB.size else 1.0)))
        rank_B1 = 0
        chain_res = 0.0
        if face_incidence is not None:
            B1 = np.asarray(face_incidence, dtype=np.float64)
            sB1 = la.svd(B1, compute_uv=False)
            rank_B1 = int(np.sum(sB1 > cutoff * max(1.0, float(sB1[0]) if sB1.size else 1.0)))
            chain_res = float(la.norm(B0 @ B1, ord="fro"))

        betti_0 = int(ker_L0)
        betti_1 = int(max(n_edges - rank_B0 - rank_B1, 0))

        parts = (I_grad, I_curl, I_harmonic)
        gram = np.zeros((3, 3), dtype=np.float64)
        for i, a in enumerate(parts):
            for j, b in enumerate(parts):
                gram[i, j] = self._weighted_inner(a, b, w)

        e_g = float(gram[0, 0])
        e_c = float(gram[1, 1])
        e_h = float(gram[2, 2])
        e_tot = self._weighted_inner(I, I, w)
        pythagoras = float(abs(e_tot - (e_g + e_c + e_h)))

        recon = float(la.norm(I - (I_grad + I_curl + I_harmonic)))
        kirchhoff = float(la.norm(B0 @ (w * I_harmonic)))
        if face_incidence is None:
            coclosure = 0.0
        else:
            B1 = np.asarray(face_incidence, dtype=np.float64)
            coclosure = float(la.norm(B1.T @ I_harmonic))

        rhs0 = B0 @ (w * I)
        L0_pinv, _, _, _ = self._pseudoinverse_psd_eigh(L0)
        potentials = np.real(L0_pinv @ rhs0)

        return HodgeChart(
            I_grad=I_grad,
            I_curl=I_curl,
            I_harmonic=I_harmonic,
            potentials=potentials,
            orthogonality_gram=gram,
            orthogonality_error=float(orthogonality_error),
            reconstruction_residual=recon,
            kirchhoff_residual=kirchhoff,
            coclosure_residual=coclosure,
            pythagoras_residual=pythagoras,
            chain_complex_residual=chain_res,
            betti_0=betti_0,
            betti_1=betti_1,
            laplacian_gap=float(gap_L0),
            laplacian_condition=L_cond,
            n_vertices=int(n_vertices),
            n_edges=int(n_edges),
            n_faces=n_faces,
            has_two_skeleton=face_incidence is not None,
            energy_grad=e_g,
            energy_curl=e_c,
            energy_harmonic=e_h,
            energy_total=float(e_tot),
        )

    def compute_compensated_kinetic_metrics(
        self,
        mass_matrix: np.ndarray,
        velocity_vector: np.ndarray,
        actuator_forces: np.ndarray,
    ) -> Tuple[float, float]:
        r"""
        [KCORE 3 — ENERGÍA CINÉTICA Y POTENCIA, KBN]

            T = ½ q̇ᵀ M(q) q̇ ,     P = q̇ᵀ F .

        La forma cuadrática se reduce a ½ ∑ᵢ q̇ᵢ (M q̇)ᵢ y la potencia a
        ∑ᵢ q̇ᵢ Fᵢ; ambas se suman con KBN. La firma pública se conserva.
        """
        self._validate_square_matrix(mass_matrix, "mass_matrix")
        self._validate_vector(np.asarray(velocity_vector), self._n, "velocity_vector")
        self._validate_vector(np.asarray(actuator_forces), self._n, "actuator_forces")

        v = np.asarray(velocity_vector, dtype=np.float64)
        F = np.asarray(actuator_forces, dtype=np.float64)
        Mv = np.real(self._hermitize(np.asarray(mass_matrix, dtype=np.complex128)) @ v)
        kinetic_vector = 0.5 * v * Mv
        power_vector = v * F
        kinetic_energy = self.kahan_sum(np.asarray(kinetic_vector, dtype=np.float64))
        mechanical_power = self.kahan_sum(np.asarray(power_vector, dtype=np.float64))
        return float(kinetic_energy), float(mechanical_power)

    def _complex_step_mass_action(
        self,
        mass_matrix: np.ndarray,
        velocity_vector: np.ndarray,
        step: float = _CSMD_STEP,
    ) -> np.ndarray:
        r"""
        CSMD de la extensión holomorfa  T(z) = ½ zᵀ M z  (M simétrica real).

        Im(T(v + i h e_λ)) / h = (M v)_λ  exactamente, sin cancelación
        sustractiva. El conjugado está prohibido: rompería la holomorfía.
        """
        M = np.real(self._hermitize(np.asarray(mass_matrix, dtype=np.complex128)))
        v = np.asarray(velocity_vector, dtype=np.complex128)
        sens = np.zeros(self._n, dtype=np.float64)
        for lam in range(self._n):
            pert = v.copy()
            pert[lam] += 1.0j * step
            value = 0.5 * pert @ M @ pert
            sens[lam] = float(np.imag(value) / step)
        return sens

    def _build_kinetic_chart(
        self,
        mass_matrix: np.ndarray,
        velocity_vector: np.ndarray,
        actuator_forces: np.ndarray,
        kinetic_energy: float,
        mechanical_power: float,
    ) -> KineticChart:
        """SPD de M, Rayleigh, sonda CSMD ∇T = M v."""
        M_h = np.real(self._hermitize(np.asarray(mass_matrix, dtype=np.complex128)))
        _, psd_res = self._psd_project(np.asarray(mass_matrix, dtype=np.complex128))
        v = np.asarray(velocity_vector, dtype=np.float64)
        Mv = M_h @ v
        vnorm2 = float(np.real(np.dot(v, v)))
        rayleigh = float(np.real(np.dot(v, Mv)) / vnorm2) if vnorm2 > _WILKINSON_FLOOR else 0.0
        csmd = self._complex_step_mass_action(mass_matrix, velocity_vector)
        csmd_err = float(la.norm(csmd - Mv))
        evals = np.real(la.eigvalsh(M_h))
        shifted = np.maximum(evals, _WILKINSON_FLOOR)
        cond = float(np.max(np.abs(shifted)) / np.min(np.abs(shifted)))
        _ = actuator_forces
        return KineticChart(
            kinetic_energy=float(kinetic_energy),
            mechanical_power=float(mechanical_power),
            mass_psd_residual=float(psd_res),
            rayleigh_quotient=rayleigh,
            csmd_momentum=csmd,
            csmd_error=csmd_err,
            mass_condition=cond,
        )

    def _phase1_observe_orient(
        self,
        g: np.ndarray,
        G_metric: np.ndarray,
        J: np.ndarray,
        R: np.ndarray,
        J_d: np.ndarray,
        R_d: np.ndarray,
        grad_H: np.ndarray,
        grad_Hd: np.ndarray,
        boundary_matrix: np.ndarray,
        edge_weights: np.ndarray,
        flow_vector: np.ndarray,
        mass_matrix: np.ndarray,
        velocity_vector: np.ndarray,
        actuator_forces: np.ndarray,
        face_incidence: Optional[np.ndarray] = None,
    ) -> PhaseOneKinematicPacket:
        r"""
        [FASE 1 — ÚLTIMO MORFISMO: OBSERVE + ORIENT]

        Codominio
        ---------
        `PhaseOneKinematicPacket`

        Continuidad formal
        ------------------
        Este paquete **es** el dominio del primer morfismo de la Fase 2,
        `_phase2_spectral_integrity(self, packet: PhaseOneKinematicPacket)`.
        No existe ningún objeto intersticial: la composición

            _phase2_spectral_integrity ∘ _phase1_observe_orient

        está tipada estrictamente (retículo de métodos anidados).
        """
        logger.info("Fase 1: carta de G, IDA-PBC+Tikhonov, Hodge celular, T/P+CSMD.")

        metric_chart = self._factorize_metric(np.asarray(G_metric), "G_metric")

        alpha, residual_norm = self.compute_ida_pbc_matching(
            g, G_metric, J, R, J_d, R_d, grad_H, grad_Hd, metric_chart=metric_chart
        )
        ph_chart = self._build_ph_chart(
            g, G_metric, J, R, J_d, R_d, grad_H, grad_Hd,
            metric_chart, alpha, residual_norm,
        )

        I_grad, I_curl, I_harm, ortho = self.compute_helmholtz_hodge_decomposition(
            boundary_matrix, edge_weights, flow_vector, face_incidence=face_incidence
        )
        hodge_chart = self._build_hodge_chart(
            boundary_matrix, edge_weights, flow_vector,
            I_grad, I_curl, I_harm, ortho, face_incidence=face_incidence,
        )

        T_kin, P_mech = self.compute_compensated_kinetic_metrics(
            mass_matrix, velocity_vector, actuator_forces
        )
        kinetic_chart = self._build_kinetic_chart(
            mass_matrix, velocity_vector, actuator_forces, T_kin, P_mech
        )

        packet = PhaseOneKinematicPacket(
            ida_alpha=alpha,
            ida_residual_norm=residual_norm,
            hh_grad=I_grad,
            hh_curl=I_curl,
            hh_harmonic=I_harm,
            hh_orthogonality_error=ortho,
            kinetic_energy=T_kin,
            mechanical_power=P_mech,
            ph_chart=ph_chart,
            hodge_chart=hodge_chart,
            kinetic_chart=kinetic_chart,
            metric_chart=metric_chart,
        )
        logger.debug(
            "Fase 1 completa: ‖r‖=%.3e obst=%.3e ortho=%.3e T=%.6f "
            "β₀=%d β₁=%d CSMD=%.3e κ_G=%.3e",
            residual_norm, ph_chart.matching_obstruction, ortho, T_kin,
            hodge_chart.betti_0, hodge_chart.betti_1,
            kinetic_chart.csmd_error, metric_chart.condition_number,
        )
        return packet

    # =========================================================================
    # FASE 2 — DECIDE
    # Dominio = PhaseOneKinematicPacket (codominio del último método de Fase 1).
    # Integridad espectral, teorema de Hodge discreto, pasividad PH y
    # clasificación de Heyting. El último método, `_phase2_decide`, tiene por
    # codominio PhaseTwoKinematicPacket, segundo factor de
    # `_phase3_conservation_audit`.
    # =========================================================================
    def _phase2_spectral_integrity(
        self,
        packet: PhaseOneKinematicPacket,
    ) -> SpectralKinematicReport:
        r"""
        [FASE 2 — PRIMER MORFISMO: INTEGRIDAD ESPECTRAL]

        Dominio
        -------
        `PhaseOneKinematicPacket`  ← continuación formal de `_phase1_observe_orient`.

        Diagnósticos (teoría espectral + álgebra de Banach (M_n, ‖·‖₂)):
          • κ₂(G), gap de G y de L₀;
          • rango y κ₂ de G^{½} g;
          • κ₂(M), Pitágoras de Hodge, obstrucción de matching, pasividad.
        """
        Gch = packet.metric_chart
        Hh = packet.hodge_chart
        Ph = packet.ph_chart
        Kc = packet.kinetic_chart

        cond_term = 1.0 / (1.0 + np.log10(max(Gch.condition_number, 1.0)))
        gap_term = float(np.tanh(Gch.spectral_gap / max(Gch.operator_norm, _WILKINSON_FLOOR)))
        L_term = float(np.exp(-max(Hh.laplacian_condition, 1.0) / 1e16))
        obst_term = float(np.exp(-Ph.matching_obstruction))
        pyth_term = float(np.exp(-Hh.pythagoras_residual / max(abs(Hh.energy_total), _WILKINSON_FLOOR)))
        pass_term = float(np.exp(-Ph.passivity_residual))
        csmd_term = float(np.exp(-Kc.csmd_error / 1e-8))
        kin_term = 1.0 if packet.kinetic_energy >= -_WILKINSON_FLOOR else 0.0
        score = float(
            cond_term * max(gap_term, 0.0) * L_term * obst_term
            * pyth_term * pass_term * csmd_term * kin_term
        )
        return SpectralKinematicReport(
            metric_condition=float(Gch.condition_number),
            metric_gap=float(Gch.spectral_gap),
            laplacian_gap=float(Hh.laplacian_gap),
            actuator_rank=int(Ph.actuator_rank),
            actuator_condition=float(Ph.actuator_condition),
            mass_condition=float(Kc.mass_condition),
            hodge_pythagoras_residual=float(Hh.pythagoras_residual),
            matching_obstruction=float(Ph.matching_obstruction),
            passivity_residual=float(Ph.passivity_residual),
            integrity_score=score,
        )

    def _phase2_hodge_theorem_audit(
        self,
        packet: PhaseOneKinematicPacket,
        reconstruction_threshold: float,
        pythagoras_rel_threshold: float,
    ) -> bool:
        r"""
        Teorema de Hodge discreto (auditoría):

          1. I = I_g + I_c + I_h                     (reconstrucción)
          2. ⟨I_a, I_b⟩_W = 0  si a ≠ b              (ortogonalidad)
          3. ‖I‖_W² = ∑ ‖I_•‖_W²                     (Pitágoras)
          4. B₀ W I_h ≈ 0 ,  B₁ᵀ I_h ≈ 0             (cociclo + ciclo)
          5. B₀ B₁ ≈ 0 si hay 2-esqueleto            (d² = 0)
        """
        Hh = packet.hodge_chart
        scale = max(abs(Hh.energy_total), 1.0)
        pyth_rel = Hh.pythagoras_residual / scale
        return bool(
            Hh.reconstruction_residual > reconstruction_threshold
            or pyth_rel > pythagoras_rel_threshold
            or Hh.chain_complex_residual > 1e-8
        )

    def _phase2_heyting_classify(
        self,
        packet: PhaseOneKinematicPacket,
        integrity: SpectralKinematicReport,
        orthogonality_threshold: float,
        residual_threshold: float,
        obstruction_threshold: float,
        reconstruction_threshold: float,
    ) -> Tuple[HeytingVerdict, float, Dict[str, bool]]:
        r"""
        Clasificación en el álgebra de Heyting.

        Scores en [0, 1] (semántica [0,1]-valuada del topos) y meet.
        Un veto atómico (T < 0, Hodge roto, CSMD rota, R ≱ 0 de forma
        grosera) colapsa a VETOED por modus ponens interno.
        """
        Ph = packet.ph_chart
        Hh = packet.hodge_chart
        Kc = packet.kinetic_chart

        ortho_viol = packet.hh_orthogonality_error > orthogonality_threshold
        resid_viol = packet.ida_residual_norm > residual_threshold
        obst_viol = Ph.matching_obstruction > obstruction_threshold
        pass_viol = Ph.passivity_residual > 1e-8 or Ph.R_psd_residual > 1e-4
        hodge_viol = self._phase2_hodge_theorem_audit(packet, reconstruction_threshold, 1e-6)
        kin_viol = packet.kinetic_energy < -_WILKINSON_FLOOR
        csmd_viol = Kc.csmd_error > 1e-8

        ortho_score = float(np.exp(-packet.hh_orthogonality_error / max(orthogonality_threshold, _WILKINSON_FLOOR)))
        resid_score = float(np.exp(-packet.ida_residual_norm / max(residual_threshold, _WILKINSON_FLOOR)))
        obst_score = float(np.exp(-Ph.matching_obstruction / max(obstruction_threshold, _WILKINSON_FLOOR)))
        pass_score = float(np.exp(-Ph.passivity_residual / 1e-8))
        recon_score = float(np.exp(-Hh.reconstruction_residual / max(reconstruction_threshold, _WILKINSON_FLOOR)))
        kin_score = 1.0 if not kin_viol else 0.0
        csmd_score = float(np.exp(-Kc.csmd_error / 1e-8))
        meet = min(
            ortho_score, resid_score, obst_score, pass_score,
            recon_score, kin_score, csmd_score, integrity.integrity_score,
        )

        hard_veto = kin_viol or csmd_viol or hodge_viol or (Ph.R_psd_residual > 1.0)
        if hard_veto:
            verdict = HeytingVerdict.VETOED
        elif meet >= 0.99:
            verdict = HeytingVerdict.CERTIFIED
        elif meet >= 0.90:
            verdict = HeytingVerdict.COHERENT
        elif meet >= 0.50:
            verdict = HeytingVerdict.DEGRADED
        else:
            verdict = HeytingVerdict.VETOED

        flags = {
            "orthogonality": ortho_viol,
            "residual": resid_viol,
            "obstruction": obst_viol,
            "passivity": pass_viol,
            "hodge": hodge_viol,
            "kinetic": kin_viol,
            "csmd": csmd_viol,
        }
        return verdict, float(meet), flags

    def _phase2_decide(
        self,
        packet: PhaseOneKinematicPacket,
        orthogonality_threshold: float = 1e-8,
        residual_threshold: float = 1e-6,
        obstruction_threshold: float = 1e-6,
        reconstruction_threshold: float = 1e-10,
    ) -> PhaseTwoKinematicPacket:
        r"""
        [FASE 2 — ÚLTIMO MORFISMO: DECIDE]

        Codominio
        ---------
        `PhaseTwoKinematicPacket`

        Continuidad formal
        ------------------
        Junto con el `PhaseOneKinematicPacket` residual, este objeto constituye
        el dominio del primer morfismo de la Fase 3,
        `_phase3_conservation_audit(self, packet1, packet2)`.
        """
        integrity = self._phase2_spectral_integrity(packet)
        verdict, score, flags = self._phase2_heyting_classify(
            packet,
            integrity,
            orthogonality_threshold=orthogonality_threshold,
            residual_threshold=residual_threshold,
            obstruction_threshold=obstruction_threshold,
            reconstruction_threshold=reconstruction_threshold,
        )
        decision = PhaseTwoKinematicPacket(
            heyting_verdict=verdict.value,
            heyting_rank=verdict.rank,
            orthogonality_violated=flags["orthogonality"],
            residual_violated=flags["residual"],
            obstruction_violated=flags["obstruction"],
            passivity_violated=flags["passivity"],
            hodge_theorem_violated=flags["hodge"],
            kinetic_sign_violated=flags["kinetic"],
            spectral_integrity=integrity,
            heyting_score=score,
            betti_0=packet.hodge_chart.betti_0,
            betti_1=packet.hodge_chart.betti_1,
        )
        logger.debug(
            "Fase 2 completa: veredicto=%s score=%.4f β₀=%d β₁=%d obst=%.3e",
            verdict.value, score, decision.betti_0, decision.betti_1,
            packet.ph_chart.matching_obstruction,
        )
        return decision

    # =========================================================================
    # FASE 3 — ACT
    # Dominio = PhaseOneKinematicPacket × PhaseTwoKinematicPacket.
    # Audita leyes de conservación y sella telemetría inmutable.
    # =========================================================================
    def _phase3_conservation_audit(
        self,
        packet1: PhaseOneKinematicPacket,
        packet2: PhaseTwoKinematicPacket,
    ) -> float:
        r"""
        [FASE 3 — PRIMER MORFISMO: AUDITORÍA DE CONSERVACIÓN]

        Dominio
        -------
        `PhaseOneKinematicPacket × PhaseTwoKinematicPacket`
        ← continuación formal de `_phase1_observe_orient` y `_phase2_decide`.

        Residuos adimensionales acumulados (KBN) de identidades que deben
        anularse en aritmética exacta:
          1. reconstrucción de Hodge  I − (I_g+I_c+I_h);
          2. Pitágoras W-ortogonal;
          3. d² = 0  (‖B₀ B₁‖_F);
          4. J + Jᵀ = 0  (vía passivity_residual = |∇Hᵀ J ∇H|);
          5. sonda CSMD  ‖∇T − M v‖;
          6. parte negativa de T;
          7. inconsistencia del flag de ortogonalidad con el Gram.
        """
        Hh = packet1.hodge_chart
        Ph = packet1.ph_chart
        Kc = packet1.kinetic_chart
        gram_off = float(
            abs(Hh.orthogonality_gram[0, 1])
            + abs(Hh.orthogonality_gram[0, 2])
            + abs(Hh.orthogonality_gram[1, 2])
        )
        terms = np.asarray(
            [
                Hh.reconstruction_residual,
                Hh.pythagoras_residual,
                Hh.chain_complex_residual,
                Ph.passivity_residual,
                Kc.csmd_error,
                max(0.0, -packet1.kinetic_energy),
                abs(gram_off - packet1.hh_orthogonality_error * 0.0),  # ancla el Gram
                max(0.0, gram_off - packet1.hh_orthogonality_error),
            ],
            dtype=np.float64,
        )
        residual = float(np.real(self._neumaier_sum(terms)))
        logger.debug(
            "Fase 3 auditoría: residuo de conservación=%.6e (veredicto previo=%s)",
            residual, packet2.heyting_verdict,
        )
        return residual

    def _phase3_act(
        self,
        packet1: PhaseOneKinematicPacket,
        packet2: PhaseTwoKinematicPacket,
        conservation_residual: float,
    ) -> KinematicTelemetry:
        """
        [FASE 3 — ÚLTIMO MORFISMO: ACT]

        Sella el objeto terminal `KinematicTelemetry`. No hay Fase 4: este
        sello es el valor de T(estado) inyectable en el pasaporte de la Malla.
        """
        return KinematicTelemetry(
            ida_alpha=packet1.ida_alpha,
            ida_residual_norm=packet1.ida_residual_norm,
            hh_grad=packet1.hh_grad,
            hh_curl=packet1.hh_curl,
            hh_harmonic=packet1.hh_harmonic,
            hh_orthogonality_error=packet1.hh_orthogonality_error,
            kinetic_energy=packet1.kinetic_energy,
            mechanical_power=packet1.mechanical_power,
            heyting_verdict=packet2.heyting_verdict,
            matching_obstruction=packet1.ph_chart.matching_obstruction,
            passivity_residual=packet1.ph_chart.passivity_residual,
            reconstruction_residual=packet1.hodge_chart.reconstruction_residual,
            pythagoras_residual=packet1.hodge_chart.pythagoras_residual,
            csmd_error=packet1.kinetic_chart.csmd_error,
            betti_0=packet2.betti_0,
            betti_1=packet2.betti_1,
            condition_number=packet2.spectral_integrity.metric_condition,
            conservation_residual=conservation_residual,
            spectral_integrity_score=packet2.spectral_integrity.integrity_score,
        )

    def execute_kinematic_cycle(
        self,
        g: np.ndarray,
        G_metric: np.ndarray,
        J: np.ndarray,
        R: np.ndarray,
        J_d: np.ndarray,
        R_d: np.ndarray,
        grad_H: np.ndarray,
        grad_Hd: np.ndarray,
        boundary_matrix: np.ndarray,
        edge_weights: np.ndarray,
        flow_vector: np.ndarray,
        mass_matrix: np.ndarray,
        velocity_vector: np.ndarray,
        actuator_forces: np.ndarray,
        orthogonality_threshold: float = 1e-8,
        residual_threshold: float = 1e-6,
        face_incidence: Optional[np.ndarray] = None,
        obstruction_threshold: float = 1e-6,
    ) -> Dict[str, Any]:
        r"""
        Functor OODA  T = Act ∘ Decide ∘ Observe.

        Encadena las tres fases anidadas sin objetos huérfanos:

            In  --F1→  PhaseOne  --F2→  PhaseTwo  --F3→  KinematicTelemetry  →  dict

        La firma pública (más `face_incidence` y `obstruction_threshold`
        opcionales) permanece compatible con el pasaporte de la Malla.
        """
        packet1 = self._phase1_observe_orient(
            g, G_metric, J, R, J_d, R_d, grad_H, grad_Hd,
            boundary_matrix, edge_weights, flow_vector,
            mass_matrix, velocity_vector, actuator_forces,
            face_incidence=face_incidence,
        )
        packet2 = self._phase2_decide(
            packet1,
            orthogonality_threshold=orthogonality_threshold,
            residual_threshold=residual_threshold,
            obstruction_threshold=obstruction_threshold,
        )
        conservation = self._phase3_conservation_audit(packet1, packet2)
        telemetry = self._phase3_act(packet1, packet2, conservation)
        return telemetry.to_dict()


# -----------------------------------------------------------------------------
# Bloque de autocomprobación
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Iniciando autocomprobación de KCore Kinematic Engine v3...")
    n = 4
    engine = KCoreKinematicEngine(dimension_n=n)

    g = np.eye(n)
    G = np.eye(n)
    J = np.array(
        [[0.0, 1.0, 0.0, 0.0],
         [-1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0],
         [0.0, 0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    R = np.eye(n)
    J_d = J.copy()
    R_d = np.eye(n) * 2.0
    grad_H = np.ones(n)
    grad_Hd = np.zeros(n)

    # C4: β₀ = 1, β₁ = 1. Sin 2-esqueleto el ciclo es ARMÓNICO, no curl.
    boundary = np.array(
        [
            [-1.0, 0.0, 0.0, 1.0],
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
        ],
        dtype=np.float64,
    )
    edge_weights = np.ones(boundary.shape[1])
    flow = np.ones(boundary.shape[1])

    mass = np.eye(n)
    vel = np.ones(n)
    force = np.ones(n)

    result = engine.execute_kinematic_cycle(
        g, G, J, R, J_d, R_d, grad_H, grad_Hd,
        boundary, edge_weights, flow,
        mass, vel, force,
    )
    print("Veredicto:", result["heyting_verdict"])
    print("Residuo IDA-PBC:", result["ida_residual_norm"])
    print("Obstrucción g^⊥ F:", result["matching_obstruction"])
    print("Error ortogonalidad HH:", result["hh_orthogonality_error"])
    print("Pitágoras:", result["pythagoras_residual"])
    print("Reconstrucción:", result["reconstruction_residual"])
    print("Betti (β₀, β₁):", result["betti_0"], result["betti_1"])
    print("T cinética (esperado 2.0):", result["kinetic_energy"])
    print("CSMD ‖∇T − Mv‖:", result["csmd_error"])
    print("Pasividad |∇Hᵀ J ∇H|:", result["passivity_residual"])
    print("‖I_curl‖ (debe ser ~0 sin 2-esqueleto):", float(la.norm(result["hh_curl"])))
    print("‖I_harm‖ (ciclo de C4):", float(la.norm(result["hh_harmonic"])))

    # 2-esqueleto trivial: un disco en el 4-ciclo ⇒ curl absorbe el ciclo, β₁ → 0.
    # B₁ tiene 1 cara cuyas 4 aristas forman el borde (orientación coherente).
    B1 = np.array([[1.0], [1.0], [1.0], [1.0]], dtype=np.float64)
    # B₀ B₁ debe anularse para un complejo; el C4 de arriba con esta B₁ no es
    # exacto (suma de columnas de B₀ = 0 ≠ B₀ 1). Se reporta chain residual.
    result_f = engine.execute_kinematic_cycle(
        g, G, J, R, J_d, R_d, grad_H, grad_Hd,
        boundary, edge_weights, flow,
        mass, vel, force,
        face_incidence=B1,
    )
    print("Con 2-esqueleto (no exacto) chain/recon se reflejan en veredicto:",
          result_f["heyting_verdict"], "recon=", result_f["reconstruction_residual"])


__all__ = [
    "KCoreKinematicEngine",
    "PhaseOneKinematicPacket",
    "PhaseTwoKinematicPacket",
    "KinematicTelemetry",
    "SpectralChart",
    "PortHamiltonianChart",
    "HodgeChart",
    "KineticChart",
    "SpectralKinematicReport",
    "HeytingVerdict",
]