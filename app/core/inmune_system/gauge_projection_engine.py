# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Gauge Projection Engine (Motor de Proyección de Calibre)            ║
║ Ruta   : app/core/inmune_system/gauge_projection_engine.py                   ║
║ Versión: 3.0.0-Doctoral-Hermitian-Higham-Duchi-DK-Frechet-CSMD-Secure        ║
║                                                                              ║
║ ARQUITECTURA DE FASES ANIDADAS (morfismos, no meras secciones):              ║
║                                                                              ║
║   Fase 1  --η-->  Fase 2  --χ-->  Fase 3                                     ║
║   Observe+Orient    Decide (números)   Act (sensibilidad) + paquete          ║
║                                                                              ║
║   η  = _phase1_terminal_morphism  = objeto inicial de la Fase 2              ║
║   χ  = _phase2_terminal_morphism  = objeto inicial de la Fase 3              ║
║                                                                              ║
║ Motor CIEGO: no clasifica en Ω₃, no dispara ISR, no toca GPIO.               ║
║ El Arsenal (agente) consume estas métricas y decide.                         ║
║                                                                              ║
║ MATEMÁTICA (sin metáfora suelta):                                            ║
║   • Π_H(M) = (M+M†)/2          projector hermítico Frobenius-óptimo.         ║
║   • Higham: λ ↦ π_Δ(λ)         proyección euclídea al símplice.              ║
║   • Tikhonov / despolarizante: Φ_γ(ρ)=(ρ+γI)/(1+nγ),                         ║
║         γ = (μ−λ_min)/(1−nμ)  ⇒  λ_min(Φ_γ) ≥ μ < 1/n.                       ║
║   • f(λ) = λ^{-1/2},  D_ρ = f(ρ)  (NO es el Dirac de un triple de Connes).   ║
║   • Conmutador:   [D,X]_{ik} = (f(λ_i)−f(λ_k)) X̃_{ik},  [D,X]_{ii} = 0.     ║
║   • Fréchet DK:   Df(ρ)[H]_{ik} = f^{[1]}(λ_i,λ_k) H̃_{ik},                   ║
║         f^{[1]}(λ,μ) = −1/(√(λμ)(√λ+√μ)),   f^{[1]}(λ,λ)=−½ λ^{-3/2}.        ║
║   • L(X) = ‖[D,X]‖_{B(ℋ)}  (norma espectral, SVD).                           ║
║   • dC[H] = [Df(ρ)[H], X];  d‖C‖₂ = Re(u* dC v) si σ_max es simple.          ║
║   • CSMD holomorfo: solo sobre g(ρ)=Tr([ρ^{-1/2},X]²) vía raíz principal.    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Final,
    Optional,
    Tuple,
)

import numpy as np
import scipy.linalg as la
from scipy.linalg import sqrtm
from scipy.special import xlogy


logger = logging.getLogger("APU.Physics.GaugeProjectionEngine")

# =============================================================================
# Constantes espectrales (IEEE-754 float64, Wilkinson–Higham–Kahan)
# =============================================================================
_EPS64: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_DRIFT: Final[float] = 50.0 * _EPS64
_MACHINE_EPS: Final[float] = _EPS64

_HERMITIAN_RTOL: Final[float] = 1e-12
_TRACE_ATOL: Final[float] = 1e-10
_SPECTRAL_RESIDUAL_TOL: Final[float] = 1e-10
_DIRAC_RESIDUAL_TOL: Final[float] = 1e-8
_SIMPLEX_SUM_TOL: Final[float] = 1e-12
_SVD_GAP_REL: Final[float] = 1e-8
_SQRTM_ERREST_TOL: Final[float] = 1e-8

_DEFAULT_MU_FLOOR: Final[float] = 1e-12
_DEFAULT_CSMD_H: Final[float] = 1.0e-8   # ver docstring: 1e-20 es inviable con sqrtm
_DEFAULT_REG: Final[float] = _WILKINSON_DRIFT


# =============================================================================
# Utilidades C* / Banach / IEEE-754
# =============================================================================
def _as_c128(a: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(a)
    if arr.size == 0:
        raise ValueError(f"{name} está vacío")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contiene NaN o Inf")
    return np.array(arr, dtype=np.complex128, copy=True)


def _freeze(a: np.ndarray) -> np.ndarray:
    out = np.array(a, copy=True)
    out.setflags(write=False)
    return out


def _hermite(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.conj().T)


def _scaled_tol(norm: float, rtol: float, atol: float) -> float:
    return float(max(atol, rtol * max(norm, 1.0)))


def _safe_xlogx(x: np.ndarray) -> np.ndarray:
    xr = np.real(np.asarray(x, dtype=np.float64))
    return xlogy(xr, xr)


def _is_numerically_hermitian(mat: np.ndarray, rtol: float = _HERMITIAN_RTOL) -> bool:
    res = float(np.linalg.norm(mat - mat.conj().T, ord="fro"))
    tol = _scaled_tol(float(np.linalg.norm(mat, ord="fro")), rtol, _WILKINSON_DRIFT)
    return res <= tol


def _project_simplex(vec: np.ndarray) -> np.ndarray:
    r"""
    Proyección euclídea sobre Δ^{n−1} = {x ≥ 0, 1ᵀx = 1}
    (Duchi et al., 2008; contenido espectral de Higham).
    """
    v = np.real(np.asarray(vec, dtype=np.float64)).copy()
    n = int(v.size)
    if n == 0:
        raise ValueError("símplice de dimensión 0")
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    j = np.arange(1, n + 1, dtype=np.float64)
    cond = u + (1.0 - cssv) / j > 0.0
    if not np.any(cond):
        out = np.zeros(n, dtype=np.float64)
        out[int(np.argmax(v))] = 1.0
        return out
    rho = int(np.max(np.flatnonzero(cond)))
    theta = (cssv[rho] - 1.0) / float(rho + 1)
    projected = np.maximum(v - theta, 0.0)
    total = float(np.sum(projected))
    if total <= _WILKINSON_DRIFT:
        out = np.zeros(n, dtype=np.float64)
        out[int(np.argmax(v))] = 1.0
        return out
    projected /= total
    return projected


# =============================================================================
# Paquetes inmutables (objetos de las fases)
# =============================================================================
@dataclass(frozen=True, slots=True)
class PhaseOneEnginePacket:
    """
    Objeto terminal de la Fase 1 / objeto inicial de la Fase 2.

    Π_H, estado de Higham (símplice), estado Tikhonov (estrictamente PD)
    y residuos certificados.  Los cuatro campos de la v2 se conservan.
    """

    rho_weyl_toeplitz: np.ndarray
    rho_regularized: np.ndarray
    lambda_min: float
    scale_factor: float
    rho_higham: np.ndarray
    lambda_min_regularized: float
    depolarizing_gamma: float
    skew_hermitian_residual: float
    toeplitz_residual: float
    spectral_residual: float
    trace_residual: float
    condition_number: float
    purity_regularized: float
    von_neumann_entropy: float
    dimension: int

    def __post_init__(self) -> None:
        if self.rho_regularized.ndim != 2:
            raise ValueError("rho_regularized debe ser una matriz")
        if self.rho_weyl_toeplitz.shape != self.rho_regularized.shape:
            raise ValueError("Π_H(M) y ρ_μ tienen forma distinta")


@dataclass(frozen=True, slots=True)
class PhaseTwoEnginePacket:
    """
    Objeto terminal de la Fase 2 / objeto inicial de la Fase 3.

    Conmutador [D, X], derivada de Fréchet Df(ρ)[X] (DK) y seminorma L(X).
    """

    commutator: np.ndarray
    frechet_derivative: np.ndarray
    lipschitz_constant: float
    lipschitz_frobenius: float
    lipschitz_dimensionless: float
    daleckii_krein_lipschitz: float
    dirac_residual: float
    sigma_max: float
    sigma_gap: float
    divided_difference_residual: float

    def __post_init__(self) -> None:
        if self.commutator.ndim != 2:
            raise ValueError("commutator debe ser una matriz")
        if self.commutator.shape != self.frechet_derivative.shape:
            raise ValueError("[D,X] y Df[X] tienen forma distinta")


@dataclass(frozen=True, slots=True)
class PhaseThreeEnginePacket:
    """
    Objeto terminal de la Fase 3: sensibilidad espectral.

    `spectral_derivative` (ABI v2) es la derivada direccional analítica
    de ‖[D,X]‖₂ (o de ‖[D,X]‖_F si σ_max es múltiple).
    """

    spectral_derivative: float
    frobenius_derivative: float
    holomorphic_csmd_trace_c2: float
    final_commutator: np.ndarray
    final_lipschitz: float
    sigma_max_simple: bool
    derivative_method: str


# =============================================================================
# Motor ciego — tres fases anidadas
# =============================================================================
class GaugeProjectionEngine:
    r"""
    Núcleo de cálculo espectral (FPU ciega) para el Arsenal.

    Métodos públicos (contrato v2):

        kbn_compensated_sum(x)                     → Σ̂ xᵢ
        weyl_toeplitz_projection(M)                → Π_H(M)
        higham_tikhonov_stabilization(M, μ)        → (ρ_μ, λ_min, α)
        connes_daleckii_krein_commutator(ρ, X)     → ([D,X], L)
        connes_daleckii_krein_complex(ρ_ℂ, X)      → ([D,X], Tr(C²))
        complex_step_spectral_derivative(ρ,H,X,h)  → dL/dε
        execute_projection_cycle(...)              → dict  (ABI v2)

    Fases anidadas:

        _phase1_projection_and_regularization → η
                └──────────────────────────────────────────┐
        _phase2_commutator_and_lipschitz      ← η          ┘
                └─ χ
        _phase3_spectral_sensitivity          ← χ
    """

    def __init__(self, reg_param: float = _DEFAULT_REG) -> None:
        self._reg: Final[float] = max(float(reg_param), _WILKINSON_DRIFT)

    @property
    def regularizer(self) -> float:
        return self._reg

    # =========================================================================
    # Utilidades de precisión numérica y validación
    # =========================================================================
    @staticmethod
    def kbn_compensated_sum(arr: np.ndarray) -> float:
        r"""
        Sumación compensada de Kahan–Babuška–Neumaier (KBN) sobre un
        vector 1-D (parte real).  Neutraliza la deriva de Wilkinson.
        """
        vec = np.asarray(arr)
        if vec.ndim != 1:
            raise ValueError(
                f"kbn_compensated_sum espera un vector 1-D; se recibió {vec.shape}"
            )
        if vec.size == 0:
            return 0.0
        if not np.all(np.isfinite(vec)):
            raise ValueError("kbn_compensated_sum: el vector contiene NaN o Inf")
        total = 0.0
        compensator = 0.0
        for raw in vec:
            x = float(np.real(raw))
            t = total + x
            if abs(total) >= abs(x):
                compensator += (total - t) + x
            else:
                compensator += (x - t) + total
            total = t
        return float(total + compensator)

    def _validate_square_matrix(self, mat: np.ndarray, name: str) -> np.ndarray:
        arr = _as_c128(mat, name)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(
                f"{name} debe ser una matriz cuadrada (n×n); se recibió {arr.shape}"
            )
        return arr

    def _validate_hermitian(
        self,
        mat: np.ndarray,
        name: str,
        tolerance: float = _HERMITIAN_RTOL,
    ) -> np.ndarray:
        arr = self._validate_square_matrix(mat, name)
        herm_res = float(np.linalg.norm(arr - arr.conj().T, ord="fro"))
        herm_tol = _scaled_tol(
            float(np.linalg.norm(arr, ord="fro")), tolerance, _WILKINSON_DRIFT
        )
        if herm_res > herm_tol:
            raise ValueError(
                f"{name} no es hermítica: ‖A−A†‖_F={herm_res:.3e} > {herm_tol:.3e}"
            )
        return _hermite(arr)

    def _certified_eigh(
        self, herm: np.ndarray, name: str
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """A = UΛU† con residuo de Wilkinson r = ‖AU−UΛ‖_F / max(1,‖A‖_F)."""
        eigvals, eigvecs = la.eigh(herm)
        recon = herm @ eigvecs - eigvecs @ np.diag(eigvals)
        denom = max(float(np.linalg.norm(herm, ord="fro")), 1.0)
        residual = float(np.linalg.norm(recon, ord="fro") / denom)
        if residual > _SPECTRAL_RESIDUAL_TOL:
            logger.warning("Residuo espectral elevado en %s: r=%.3e.", name, residual)
        return np.real(eigvals), eigvecs, residual

    def _certified_spectral_norm(self, mat: np.ndarray) -> float:
        """‖A‖_{B(ℋ)} = σ_max(A)."""
        if mat.size == 0:
            return 0.0
        singular = la.svd(mat, compute_uv=False, overwrite_a=False)
        if singular.size == 0:
            return 0.0
        return float(np.real(singular[0]))

    def _top_svd(
        self, mat: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray, float]:
        """
        (u, σ_max, v, gap) de A = U Σ V†.
        `v` es el primer vector derecho (columna de V).
        `gap` = σ_max − σ_2  (0 si rango 1 o n=1).
        """
        u_mat, s_vals, vh = la.svd(mat, full_matrices=False, overwrite_a=False)
        if s_vals.size == 0:
            n = mat.shape[0]
            z = np.zeros(n, dtype=np.complex128)
            return z, 0.0, z, 0.0
        sigma = float(np.real(s_vals[0]))
        gap = float(np.real(s_vals[0] - s_vals[1])) if s_vals.size > 1 else sigma
        return u_mat[:, 0], sigma, vh[0, :].conj(), gap

    def _toeplitz_residual(self, mat: np.ndarray) -> float:
        """‖A − P_Toep(A)‖_F / max(1, ‖A‖_F).  Diagnóstico, no se impone."""
        n = mat.shape[0]
        if n < 2:
            return 0.0
        proj = np.zeros_like(mat, dtype=np.complex128)
        for k in range(-n + 1, n):
            diag = np.diag(mat, k=k)
            mean = np.mean(diag) if diag.size else 0.0
            proj += np.diag(np.full(diag.shape, mean, dtype=np.complex128), k=k)
        denom = max(float(np.linalg.norm(mat, ord="fro")), 1.0)
        return float(np.linalg.norm(mat - proj, ord="fro") / denom)

    def _von_neumann_entropy(self, eigenvalues: np.ndarray) -> float:
        lam = np.real(np.asarray(eigenvalues, dtype=np.float64))
        lam = lam[lam > _WILKINSON_DRIFT]
        if lam.size == 0:
            return 0.0
        return float(-np.sum(_safe_xlogx(lam)))

    def _clamp_mu_floor(self, mu_floor: float, n: int) -> float:
        mu = float(mu_floor)
        if not np.isfinite(mu) or mu < 0.0:
            raise ValueError(f"mu_floor debe ser ≥ 0 y finito; se recibió {mu_floor}")
        mu_max = (1.0 / float(n)) * (1.0 - 1.0e-12)
        if mu >= mu_max:
            raise ValueError(
                f"mu_floor={mu:.3e} ≥ 1/n={1.0 / n:.3e}: "
                "ningún estado tiene λ_min tan grande."
            )
        return max(mu, self._reg)

    def _depolarizing_gamma(self, lambda_min: float, mu: float, n: int) -> float:
        r"""
        γ tal que (λ_min + γ)/(1 + nγ) ≥ μ.
        El v2 usaba γ = max(0, μ − λ_min), que deja λ_min' < μ tras normalizar.
        """
        if lambda_min + _WILKINSON_DRIFT >= mu:
            return 0.0
        denom = 1.0 - float(n) * mu
        if denom <= _WILKINSON_DRIFT:
            raise ValueError("1 − nμ numéricamente nulo; mu_floor demasiado grande")
        return float((mu - lambda_min) / denom)

    # =========================================================================
    # FASE 1 — OBSERVE + ORIENT
    #   1.1  validación C* de la matriz cruda
    #   1.2  projector hermítico Π_H  (API: weyl_toeplitz_projection)
    #   1.3  espectro certificado (Weyl–Wilkinson)
    #   1.4  Higham: proyección al símplice (Duchi)
    #   1.5  Tikhonov / despolarizante con suelo μ *normalizado*
    #   1.6  residuos (skew, Toeplitz, traza, κ)
    #   1.7  morfismo terminal η  →  inicio formal de la Fase 2
    # =========================================================================
    def weyl_toeplitz_projection(self, m: np.ndarray) -> np.ndarray:
        r"""
        [MOTOR 1 — PROJECTOR HERMÍTICO]

        Nombre histórico «Weyl–Toeplitz».  El operador implementado es el
        projector ortogonal (Frobenius) sobre el subespacio real de
        hermíticos:

            Π_H(M) = (M + M†)/2.

        No es una cuantización de Weyl ni un operador de Toeplitz.
        """
        raw = self._validate_square_matrix(m, "M")
        return _hermite(raw)

    def higham_tikhonov_stabilization(
        self,
        m_wt: np.ndarray,
        mu_floor: float = _DEFAULT_MU_FLOOR,
    ) -> Tuple[np.ndarray, float, float]:
        r"""
        [MOTOR 2 — HIGHAM (SÍMPLICE) + TIKHONOV (DESPOLARIZANTE)]

        1. Certifica hermiticidad y diagonaliza (Wilkinson).
        2. Higham: proyecta el espectro al símplice Δ^{n−1} (Duchi).
        3. Canal despolarizante con suelo *normalizado* μ:

               ρ_μ = (ρ_★ + γ I) / (1 + nγ),
               γ = max{0, (μ − λ_min(ρ_★)) / (1 − nμ)}.

        Returns:
            (ρ_μ, λ_min(M_WT), α)  con α = 1/(1+nγ) ∈ (0, 1]
            (peso de ρ_★; contrato v2: `scale_factor`).
        """
        herm = self._validate_hermitian(m_wt, "M_wt")
        n = int(herm.shape[0])
        mu = self._clamp_mu_floor(mu_floor, n)

        eigvals, eigvecs, _ = self._certified_eigh(herm, "M_wt")
        lambda_min = float(np.min(eigvals))

        spec_higham = _project_simplex(eigvals)
        if abs(float(np.sum(spec_higham)) - 1.0) > _SIMPLEX_SUM_TOL:
            spec_higham = spec_higham / max(float(np.sum(spec_higham)), _WILKINSON_DRIFT)

        gamma = self._depolarizing_gamma(float(np.min(spec_higham)), mu, n)
        scale_factor = 1.0 / (1.0 + float(n) * gamma)
        spec_reg = (spec_higham + gamma) * scale_factor

        rho_reg = eigvecs @ np.diag(spec_reg) @ eigvecs.conj().T
        rho_reg = _hermite(rho_reg)

        diag = np.real(np.diagonal(rho_reg))
        trace_kbn = self.kbn_compensated_sum(diag)
        if abs(trace_kbn - 1.0) > _TRACE_ATOL:
            logger.warning(
                "Traza KBN tras regularización: %.12f; se renormaliza.",
                trace_kbn,
            )
            if abs(trace_kbn) <= _WILKINSON_DRIFT:
                raise ValueError("traza numéricamente nula tras Higham–Tikhonov")
            rho_reg = rho_reg / trace_kbn
            scale_factor = float(scale_factor / trace_kbn)

        return rho_reg, lambda_min, float(scale_factor)

    def _phase1_projection_and_regularization(
        self,
        m_raw: np.ndarray,
        mu_floor: float = _DEFAULT_MU_FLOOR,
    ) -> PhaseOneEnginePacket:
        """
        [FASE 1 · Observe + Orient]

        Π_H → Higham → Tikhonov, con residuos certificados.
        Emite el paquete que **es** el objeto inicial de la Fase 2 vía η.
        """
        logger.info(
            "Fase 1: projector hermítico, Higham-símplice y despolarizante Tikhonov."
        )
        raw = self._validate_square_matrix(m_raw, "M_raw")
        n = int(raw.shape[0])
        mu = self._clamp_mu_floor(mu_floor, n)

        skew = float(np.linalg.norm(raw - raw.conj().T, ord="fro") / 2.0)
        rho_h = self.weyl_toeplitz_projection(raw)
        eig_h, vec_h, spec_res = self._certified_eigh(rho_h, "M_wt")
        spec_star = _project_simplex(eig_h)
        rho_star = _hermite(vec_h @ np.diag(spec_star) @ vec_h.conj().T)

        rho_mu, lambda_min_api, scale = self.higham_tikhonov_stabilization(
            rho_h, mu_floor=mu
        )
        eig_mu, _, spec_res_mu = self._certified_eigh(rho_mu, "rho_regularized")
        spec_res = max(spec_res, spec_res_mu)

        gamma = self._depolarizing_gamma(float(np.min(spec_star)), mu, n)
        lam_min_mu = float(np.min(eig_mu))
        lam_max_mu = float(np.max(eig_mu))
        cond = (
            lam_max_mu / max(lam_min_mu, _WILKINSON_DRIFT)
            if lam_min_mu > 0.0
            else float("inf")
        )
        trace_res = abs(self.kbn_compensated_sum(np.real(np.diagonal(rho_mu))) - 1.0)
        purity = float(np.real(np.sum(np.square(eig_mu))))
        entropy = self._von_neumann_entropy(eig_mu)
        toep = self._toeplitz_residual(rho_h)

        packet = PhaseOneEnginePacket(
            rho_weyl_toeplitz=_freeze(rho_h),
            rho_regularized=_freeze(rho_mu),
            lambda_min=float(lambda_min_api),
            scale_factor=float(scale),
            rho_higham=_freeze(rho_star),
            lambda_min_regularized=lam_min_mu,
            depolarizing_gamma=float(gamma),
            skew_hermitian_residual=float(skew),
            toeplitz_residual=float(toep),
            spectral_residual=float(spec_res),
            trace_residual=float(trace_res),
            condition_number=float(cond),
            purity_regularized=purity,
            von_neumann_entropy=entropy,
            dimension=n,
        )
        logger.debug(
            "Fase 1: λ_min(H)=%.3e  λ_min(μ)=%.3e  α=%.6f  γ=%.3e  κ=%.3e",
            packet.lambda_min,
            packet.lambda_min_regularized,
            packet.scale_factor,
            packet.depolarizing_gamma,
            packet.condition_number,
        )
        # ---- morfismo terminal η: cierra Fase 1 y abre formalmente Fase 2 ----
        return self._phase1_terminal_morphism(packet)

    def _phase1_terminal_morphism(
        self,
        packet: PhaseOneEnginePacket,
    ) -> PhaseOneEnginePacket:
        r"""
        [FASE 1 · morfismo terminal η]  ≡  [FASE 2 · objeto inicial]

        Unidad idempotente Spec → Spec.  Todo método de la Fase 2
        **comienza** reaplicando η: recertifica invariantes y no muta.
        """
        n = packet.dimension
        if packet.rho_regularized.shape != (n, n):
            raise ValueError("η: dim(ρ_μ) corrupta")
        if packet.rho_weyl_toeplitz.shape != (n, n):
            raise ValueError("η: dim(Π_H M) corrupta")
        if packet.lambda_min_regularized < -_TRACE_ATOL:
            raise ValueError("η: λ_min(ρ_μ) negativo (inconsistencia numérica)")
        if packet.purity_regularized < -_TRACE_ATOL or packet.purity_regularized > 1.0 + _TRACE_ATOL:
            raise ValueError("η: pureza fuera de [0, 1]")
        if packet.scale_factor <= 0.0 or packet.scale_factor > 1.0 + _TRACE_ATOL:
            raise ValueError("η: scale_factor fuera de (0, 1]")
        if not np.isfinite(
            [
                packet.lambda_min,
                packet.lambda_min_regularized,
                packet.scale_factor,
                packet.condition_number,
                packet.von_neumann_entropy,
            ]
        ).all():
            raise ValueError("η: observables no finitos")
        return packet

    # =========================================================================
    # FASE 2 — DECIDE  (continúa η; produce χ = métricas de Lipschitz)
    #   2.0  η(packet)                         ← último morfismo de Fase 1
    #   2.1  conmutador [D, X] en la base propia (estable)
    #   2.2  Fréchet Df(ρ)[X] vía diferencias divididas estables de DK
    #   2.3  L = ‖[D,X]‖₂ certificada (SVD)
    #   2.4  morfismo terminal χ  →  inicio formal de la Fase 3
    # =========================================================================
    def _stable_invsqrt_spectrum(self, eigenvalues: np.ndarray) -> np.ndarray:
        lam = np.clip(np.real(np.asarray(eigenvalues, dtype=np.float64)), self._reg, None)
        return 1.0 / np.sqrt(lam)

    def _daleckii_krein_first(
        self,
        eigenvalues: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        r"""
        Matriz de diferencias divididas de f(λ) = λ^{-1/2}, forma estable:

            f^{[1]}(λ,μ) = −1 / (√(λμ) (√λ + √μ))

        (cubre la diagonal: f'(λ) = −½ λ^{-3/2}).
        Lip(f) = max |f^{[1]}| = ½ λ_min^{-3/2}.
        """
        lam = np.clip(np.real(np.asarray(eigenvalues, dtype=np.float64)), self._reg, None)
        sqrt_l = np.sqrt(lam)
        denom = sqrt_l[:, None] * sqrt_l[None, :] * (sqrt_l[:, None] + sqrt_l[None, :])
        dk = -1.0 / denom
        lip = float(np.max(np.abs(dk)))
        return dk, lip

    def _commutator_in_eigenbasis(
        self,
        eigenvalues: np.ndarray,
        eigvecs: np.ndarray,
        pi_x: np.ndarray,
    ) -> np.ndarray:
        r"""
        [D, X] en la base propia, D = f(ρ), f(λ)=λ^{-1/2}:

            [D, X]_{ik} = (f(λ_i) − f(λ_k)) ⟨i|X|k⟩,
            [D, X]_{ii} = 0.

        Forma estable del salto:
            f(λ)−f(μ) = (μ−λ) / (√(λμ)(√λ+√μ)).
        """
        lam = np.clip(np.real(np.asarray(eigenvalues, dtype=np.float64)), self._reg, None)
        sqrt_l = np.sqrt(lam)
        # f_i − f_k = (λ_k − λ_i) / (√(λ_i λ_k)(√λ_i+√λ_k))
        denom = sqrt_l[:, None] * sqrt_l[None, :] * (sqrt_l[:, None] + sqrt_l[None, :])
        jump = (lam[None, :] - lam[:, None]) / denom
        np.fill_diagonal(jump, 0.0)
        x_eig = eigvecs.conj().T @ pi_x @ eigvecs
        comm_eig = jump * x_eig
        return eigvecs @ comm_eig @ eigvecs.conj().T

    def _frechet_in_eigenbasis(
        self,
        eigenvalues: np.ndarray,
        eigvecs: np.ndarray,
        direction: np.ndarray,
        dk: np.ndarray,
    ) -> np.ndarray:
        r"""
        Df(ρ)[H]_{ik} = f^{[1]}(λ_i, λ_k) ⟨i|H|k⟩   (Daletskii–Krein).
        """
        h_eig = eigvecs.conj().T @ direction @ eigvecs
        fr_eig = dk * h_eig
        return eigvecs @ fr_eig @ eigvecs.conj().T

    def _dirac_residual(
        self,
        eigenvalues: np.ndarray,
        eigvecs: np.ndarray,
        rho: np.ndarray,
    ) -> float:
        """‖ρ D² − I‖_F / √n   con D² = ρ^{-1} en la base propia."""
        n = int(rho.shape[0])
        lam = np.clip(np.real(np.asarray(eigenvalues, dtype=np.float64)), self._reg, None)
        d2 = eigvecs @ np.diag(1.0 / lam) @ eigvecs.conj().T
        resid = rho @ d2 - np.eye(n, dtype=np.complex128)
        return float(np.linalg.norm(resid, ord="fro") / np.sqrt(float(n)))

    def _dk_residual_vs_finite_jump(
        self,
        eigenvalues: np.ndarray,
        dk: np.ndarray,
    ) -> float:
        """
        Consistencia DK: (λ_i−λ_k) f^{[1]}(λ_i,λ_k) ≟ f(λ_i)−f(λ_k).
        Residuo relativo Frobenius (0 en aritmética exacta).
        """
        lam = np.clip(np.real(np.asarray(eigenvalues, dtype=np.float64)), self._reg, None)
        fspec = 1.0 / np.sqrt(lam)
        jump = fspec[:, None] - fspec[None, :]
        rebuilt = (lam[:, None] - lam[None, :]) * dk
        denom = max(float(np.linalg.norm(jump, ord="fro")), 1.0)
        # La diagonal de ambos es 0 vs 0 (f' · 0).
        np.fill_diagonal(rebuilt, 0.0)
        np.fill_diagonal(jump, 0.0)
        return float(np.linalg.norm(rebuilt - jump, ord="fro") / denom)

    def connes_daleckii_krein_commutator(
        self,
        rho_reg: np.ndarray,
        pi_x: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        r"""
        [MOTOR 3 — CONMUTADOR [D, π(X)] ]

        D = ρ^{-1/2}.  En la base propia:

            [D, X]_{ik} = (λ_i^{-1/2} − λ_k^{-1/2}) X̃_{ik}.

        El v2 devolvía Df(ρ)[X] (diferencias divididas × X̃) y lo llamaba
        conmutador.  Ese objeto se expone ahora como
        `PhaseTwoEnginePacket.frechet_derivative`; este método devuelve
        el conmutador verdadero y L = ‖[D,X]‖_{B(ℋ)}.

        La fórmula estable del salto evita la cancelación
        (√μ−√λ)/(μ−λ) cuando λ ≈ μ.
        """
        herm = self._validate_hermitian(rho_reg, "rho_reg")
        pi = self._validate_hermitian(pi_x, "pi_X")
        if herm.shape != pi.shape:
            raise ValueError(
                f"dim(ρ)={herm.shape} incompatible con dim(π(X))={pi.shape}"
            )
        eigvals, eigvecs, _ = self._certified_eigh(herm, "rho_reg")
        if np.any(eigvals <= self._reg * 0.5):
            raise ValueError(
                "rho_reg no es estrictamente PD para D=ρ^{-1/2} "
                f"(λ_min={float(np.min(eigvals)):.3e})"
            )
        comm = self._commutator_in_eigenbasis(eigvals, eigvecs, pi)
        lipschitz = self._certified_spectral_norm(comm)
        return comm, float(lipschitz)

    def _phase2_commutator_and_lipschitz(
        self,
        packet1: PhaseOneEnginePacket,
        pi_x: np.ndarray,
    ) -> PhaseTwoEnginePacket:
        """
        [FASE 2 · Decide]

        Continuación formal de `_phase1_terminal_morphism`: re-certifica η,
        construye [D, X] y Df(ρ)[X], y emite χ.
        """
        certified = self._phase1_terminal_morphism(packet1)  # ← nido Fase 1 → 2
        logger.info("Fase 2: conmutador [D,X] y Fréchet Daletskii–Krein.")

        pi = self._validate_hermitian(pi_x, "pi_X")
        if pi.shape != certified.rho_regularized.shape:
            raise ValueError("π(X) y ρ_μ tienen dimensión incompatible")

        eigvals, eigvecs, _ = self._certified_eigh(certified.rho_regularized, "rho_μ")
        dk, dk_lip = self._daleckii_krein_first(eigvals)
        comm = self._commutator_in_eigenbasis(eigvals, eigvecs, pi)
        frechet = self._frechet_in_eigenbasis(eigvals, eigvecs, pi, dk)

        _, sigma, _, gap = self._top_svd(comm)
        l_spec = float(sigma)
        l_frob = float(np.linalg.norm(comm, ord="fro"))
        d_spec = self._stable_invsqrt_spectrum(eigvals)
        norm_d = float(np.max(d_spec))
        norm_x = self._certified_spectral_norm(pi)
        l_dimless = l_spec / max(norm_d * norm_x, _WILKINSON_DRIFT)
        dirac_res = self._dirac_residual(eigvals, eigvecs, certified.rho_regularized)
        dk_res = self._dk_residual_vs_finite_jump(eigvals, dk)

        packet2 = PhaseTwoEnginePacket(
            commutator=_freeze(comm),
            frechet_derivative=_freeze(frechet),
            lipschitz_constant=l_spec,
            lipschitz_frobenius=l_frob,
            lipschitz_dimensionless=float(l_dimless),
            daleckii_krein_lipschitz=float(dk_lip),
            dirac_residual=float(dirac_res),
            sigma_max=l_spec,
            sigma_gap=float(gap),
            divided_difference_residual=float(dk_res),
        )
        logger.debug(
            "Fase 2: L=%.6f  ‖C‖_F=%.6f  L/‖D‖‖X‖=%.3f  Lip(f)=%.3e  gap_σ=%.3e",
            packet2.lipschitz_constant,
            packet2.lipschitz_frobenius,
            packet2.lipschitz_dimensionless,
            packet2.daleckii_krein_lipschitz,
            packet2.sigma_gap,
        )
        # ---- morfismo terminal χ: cierra Fase 2 y abre formalmente Fase 3 ----
        return self._phase2_terminal_morphism(packet2)

    def _phase2_terminal_morphism(
        self,
        packet: PhaseTwoEnginePacket,
    ) -> PhaseTwoEnginePacket:
        r"""
        [FASE 2 · morfismo terminal χ]  ≡  [FASE 3 · objeto inicial]

        Unidad idempotente Lip → Lip.  Todo método de la Fase 3
        **comienza** reaplicando χ.
        """
        if packet.commutator.ndim != 2 or packet.commutator.shape[0] != packet.commutator.shape[1]:
            raise ValueError("χ: conmutador no cuadrado")
        if packet.lipschitz_constant < -_TRACE_ATOL:
            raise ValueError("χ: norma espectral negativa")
        if not np.isfinite(
            [
                packet.lipschitz_constant,
                packet.lipschitz_frobenius,
                packet.daleckii_krein_lipschitz,
                packet.dirac_residual,
                packet.sigma_gap,
            ]
        ).all():
            raise ValueError("χ: observables no finitos")
        return packet

    # =========================================================================
    # FASE 3 — ACT  (continúa χ; sensibilidad espectral)
    #   3.0  consume χ y η
    #   3.1  dC[H] = [Df(ρ)[H], X]                         (DK, exacto)
    #   3.2  d‖C‖₂ = Re(u* dC v) si σ_max es simple        (analítico)
    #   3.3  CSMD holomorfo de Tr(C²) vía raíz principal   (cruzado)
    # =========================================================================
    def _principal_inverse_sqrt(self, mat: np.ndarray) -> Tuple[np.ndarray, float]:
        r"""
        Inverso de la raíz cuadrada principal A^{-1/2}.

        • Si A es numéricamente hermítica: eigh + recorte al suelo ε.
        • Si no: Schur / `sqrtm` + inversión (cálculo holomorfo, CSMD).

        Devuelve (A^{-1/2}, errest).  `errest` es 0 en el camino hermítico.
        """
        a = self._validate_square_matrix(mat, "A")
        if _is_numerically_hermitian(a):
            herm = _hermite(a)
            eigvals, eigvecs, _ = self._certified_eigh(herm, "A_H")
            inv_sqrt = eigvecs @ np.diag(self._stable_invsqrt_spectrum(eigvals)) @ eigvecs.conj().T
            return inv_sqrt, 0.0
        sqrt_a, errest = sqrtm(a, disp=False)
        errest_f = float(np.real(errest)) if np.isscalar(errest) else float(np.real(np.max(errest)))
        if errest_f > _SQRTM_ERREST_TOL:
            logger.warning("sqrtm: errest=%.3e (raíz principal mal condicionada).", errest_f)
        try:
            inv_sqrt = la.inv(sqrt_a)
        except la.LinAlgError as exc:
            raise ValueError("A^{1/2} singular: no existe A^{-1/2} principal") from exc
        return inv_sqrt, errest_f

    def connes_daleckii_krein_complex(
        self,
        rho_complex: np.ndarray,
        pi_x: np.ndarray,
    ) -> Tuple[np.ndarray, complex]:
        r"""
        Extensión holomorfa del conmutador para CSMD.

        El v2 llamaba `eigh` sobre ρ+ihH (no hermítica) y recortaba
        `real(λ)`, destruyendo la perturbación imaginaria.  Aquí:

            D = (ρ_ℂ)^{-1/2}     raíz principal (Schur si no hermítica),
            C = [D, π(X)],
            g = Tr(C²)           holomorfa en D (luego en ρ_ℂ).

        `g` es un *proxy* holomorfo; no es ‖C‖₂ (que no es holomorfa).
        La parte imaginaria de g, dividida por h, estima dg/dε.
        """
        rho_c = self._validate_square_matrix(rho_complex, "rho_complex")
        pi = self._validate_square_matrix(pi_x, "pi_X")
        if rho_c.shape != pi.shape:
            raise ValueError("ρ_ℂ y π(X) tienen dimensión incompatible")
        dirac, _ = self._principal_inverse_sqrt(rho_c)
        comm = dirac @ pi - pi @ dirac
        trace_c2 = complex(np.trace(comm @ comm))
        return comm, trace_c2

    def _analytic_directional_derivatives(
        self,
        rho: np.ndarray,
        perturbation: np.ndarray,
        pi_x: np.ndarray,
        commutator: np.ndarray,
    ) -> Tuple[float, float, bool, str]:
        r"""
        Derivadas direccionales analíticas a lo largo de H = H†:

            dC  = [Df(ρ)[H], X]
            d‖C‖_F = Re⟨C, dC⟩ / ‖C‖_F
            d‖C‖₂  = Re(u* dC v)     si σ_max es simple
                   = d‖C‖_F          si no (subgradiente de Frobenius).
        """
        herm = self._validate_hermitian(rho, "rho_reg")
        h_dir = self._validate_hermitian(perturbation, "perturbation")
        pi = self._validate_hermitian(pi_x, "pi_X")
        eigvals, eigvecs, _ = self._certified_eigh(herm, "rho_reg")
        dk, _ = self._daleckii_krein_first(eigvals)
        df_h = self._frechet_in_eigenbasis(eigvals, eigvecs, h_dir, dk)
        d_comm = df_h @ pi - pi @ df_h

        fro_c = float(np.linalg.norm(commutator, ord="fro"))
        if fro_c <= _WILKINSON_DRIFT:
            d_fro = 0.0
        else:
            d_fro = float(np.real(np.vdot(commutator, d_comm)) / fro_c)

        u_left, sigma, v_right, gap = self._top_svd(commutator)
        simple = bool(gap > _SVD_GAP_REL * max(sigma, 1.0))
        if simple and sigma > _WILKINSON_DRIFT:
            d_spec = float(np.real(np.vdot(u_left, d_comm @ v_right)))
            method = "analytic_spectral_svd"
        else:
            d_spec = d_fro
            method = "analytic_frobenius_fallback"
        return d_spec, d_fro, simple, method

    def complex_step_spectral_derivative(
        self,
        rho_reg: np.ndarray,
        perturbation: np.ndarray,
        pi_x: np.ndarray,
        step_h: float = _DEFAULT_CSMD_H,
    ) -> float:
        r"""
        [MOTOR 4 — SENSIBILIDAD]

        Contrato v2: devuelve un `float` «dL/dε».

        Implementación primaria (analítica, sin paso complejo):
            d‖[D,X]‖₂ / dε  a lo largo de H, vía DK + SVD.

        El v2 evaluaba Im(Tr(C²))/h tras `eigh(ρ+ihH)` + `clip(real)`,
        que es idénticamente 0 o ruido: `eigh` hermitiza.  Un CSMD
        honesto de Tr(C²) se calcula en Fase 3 como *cruzado*
        (`holomorphic_csmd_trace_c2`), no como este valor de retorno.

        `step_h` se ignora en el camino analítico (se conserva en la
        firma).  Si se desea el CSMD holomorfo, usar
        `connes_daleckii_krein_complex` + Im(g)/h con h ∼ 10^{-8}.
        """
        if not np.isfinite(step_h) or step_h == 0.0:
            raise ValueError("step_h debe ser finito y no nulo")
        herm = self._validate_hermitian(rho_reg, "rho_reg")
        comm, _ = self.connes_daleckii_krein_commutator(herm, pi_x)
        d_spec, _d_fro, _simple, _method = self._analytic_directional_derivatives(
            herm, perturbation, pi_x, comm
        )
        return float(d_spec)

    def _holomorphic_csmd_trace_c2(
        self,
        rho: np.ndarray,
        perturbation: np.ndarray,
        pi_x: np.ndarray,
        step_h: float,
    ) -> float:
        """Im(Tr(C(ρ+ihH)²))/h  con C=[ρ^{-1/2},X] vía raíz principal."""
        if abs(step_h) < _EPS64:
            raise ValueError("step_h demasiado pequeño para CSMD con sqrtm")
        h_dir = self._validate_hermitian(perturbation, "perturbation")
        rho_pert = rho + (1.0j * float(step_h)) * h_dir
        _comm_c, g_c = self.connes_daleckii_krein_complex(rho_pert, pi_x)
        return float(np.imag(g_c) / step_h)

    def _phase3_spectral_sensitivity(
        self,
        packet2: PhaseTwoEnginePacket,
        packet1: PhaseOneEnginePacket,
        perturbation: np.ndarray,
        pi_x: np.ndarray,
        step_h: float = _DEFAULT_CSMD_H,
    ) -> PhaseThreeEnginePacket:
        """
        [FASE 3 · Act]

        Continuación formal de `_phase2_terminal_morphism`: re-certifica
        χ y η, calcula la sensibilidad analítica y el cruzado CSMD.
        """
        certified1 = self._phase1_terminal_morphism(packet1)
        certified2 = self._phase2_terminal_morphism(packet2)  # ← nido Fase 2 → 3
        logger.info("Fase 3: derivada direccional analítica (DK+SVD) y CSMD holomorfo.")

        d_spec, d_fro, simple, method = self._analytic_directional_derivatives(
            certified1.rho_regularized,
            perturbation,
            pi_x,
            np.array(certified2.commutator, copy=True),
        )
        try:
            csmd = self._holomorphic_csmd_trace_c2(
                certified1.rho_regularized, perturbation, pi_x, step_h
            )
        except (ValueError, la.LinAlgError) as exc:
            logger.warning("CSMD holomorfo no disponible: %s", exc)
            csmd = float("nan")

        packet3 = PhaseThreeEnginePacket(
            spectral_derivative=float(d_spec),
            frobenius_derivative=float(d_fro),
            holomorphic_csmd_trace_c2=float(csmd),
            final_commutator=_freeze(np.array(certified2.commutator, copy=True)),
            final_lipschitz=float(certified2.lipschitz_constant),
            sigma_max_simple=bool(simple),
            derivative_method=method,
        )
        logger.debug(
            "Fase 3: d‖C‖₂=%.6e  d‖C‖_F=%.6e  CSMD[Tr C²]=%.6e  método=%s",
            packet3.spectral_derivative,
            packet3.frobenius_derivative,
            packet3.holomorphic_csmd_trace_c2,
            packet3.derivative_method,
        )
        return packet3

    # =========================================================================
    # ORQUESTACIÓN  (composición η ; χ ; Act)
    # =========================================================================
    def execute_projection_cycle(
        self,
        rho_raw: np.ndarray,
        pi_x: np.ndarray,
        perturbation: Optional[np.ndarray] = None,
        mu_floor: float = _DEFAULT_MU_FLOOR,
        step_h: float = _DEFAULT_CSMD_H,
    ) -> Dict[str, Any]:
        r"""
        Compone las tres fases anidadas:

            paquete = Act( χ( η( Observe(M, μ) ), π(X) ), H )

        Devuelve exactamente las siete claves de la v2.  Cualquier
        excepción no contemplada colapsa a un dict de emergencia
        (fail-closed numérico; el motor no veta).
        """
        try:
            raw = self._validate_square_matrix(rho_raw, "rho_raw")
            if perturbation is None:
                perturbation = np.eye(raw.shape[0], dtype=np.complex128)

            packet1 = self._phase1_projection_and_regularization(raw, mu_floor)
            packet2 = self._phase2_commutator_and_lipschitz(packet1, pi_x)
            packet3 = self._phase3_spectral_sensitivity(
                packet2, packet1, perturbation, pi_x, step_h
            )
            return {
                "rho_weyl_toeplitz": packet1.rho_weyl_toeplitz,
                "rho_regularized": packet1.rho_regularized,
                "lambda_min": packet1.lambda_min,
                "scale_factor": packet1.scale_factor,
                "commutator": packet2.commutator,
                "lipschitz_constant": packet2.lipschitz_constant,
                "spectral_derivative": packet3.spectral_derivative,
            }
        except Exception:
            logger.exception("Excepción en el ciclo de proyección (fail-closed numérico).")
            return {
                "rho_weyl_toeplitz": None,
                "rho_regularized": None,
                "lambda_min": float("nan"),
                "scale_factor": float("nan"),
                "commutator": None,
                "lipschitz_constant": float("nan"),
                "spectral_derivative": float("nan"),
            }


# -----------------------------------------------------------------------------
# Exportación de firmas canónicas
# -----------------------------------------------------------------------------
__all__ = [
    "GaugeProjectionEngine",
    "PhaseOneEnginePacket",
    "PhaseTwoEnginePacket",
    "PhaseThreeEnginePacket",
]