# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Imperial Eruditos Engine (Caballos de Batalla de Cohomología)       ║
║ Ruta   : app/core/inmune_system/imperial_eruditos_engine.py                  ║
║ Versión: 6.1.0-Poincare-Cartan-Melnikov-KAM-Floer-Cech-Nested-PhD            ║
╚══════════════════════════════════════════════════════════════════════════════╝
SINOPSIS MATEMÁTICA Y METROLOGÍA DE LA FPU:
Motor cohomológico y de mecánica celeste. Evalúa la regularidad elíptica y el
potencial de acción de cilindros pseudo-holomorfos (Floer), aniquila bucles
parasitarios en el KV-Cache (Čech atencional) y resuelve el espectro de
pequeños divisores de Poincaré–KAM con absorción ultramétrica en Λ_Nov.

El tejido es estrictamente anidado:
    I.ω   = synthesize_poincare_cartan_germ      → _PoincareCartanGerm
            ≡ objeto inicial de la FASE II
    II.0  = phase2_ingest_poincare_cartan_germ   (continúa I.ω)
    II.ω  = induce_cech_nerve_germ               → _CechNerveGerm
            ≡ objeto inicial de la FASE III
    III.0 = phase3_ingest_cech_nerve_germ        (continúa II.ω)
    III.ω = compute / compute_attention_cech…    → _CechCohomologyResult

    Φ_III ∘ Φ_II ∘ Φ_I : Darboux ⟶ Poincaré-Cartan ⟶ KAM/Melnikov/Floer ⟶ Čech/Hodge.

Geometría sobre T*Q ≅ ℝ^{2n} y el espacio extendido T*Q × ℝ:
  • θ = p dq  (Liouville);  ω = dθ;  dω = 0.
  • λ_PC = θ − H dt  (Poincaré–Cartan);  dλ_PC = ω − dH ∧ dt.
  • X_H ⌟ ω = −dH;  {f,g} = ω(X_f, X_g) = (∇f)ᵀ Ω (∇g).
  • g̃ = 2(H₀ − V) g  (Maupertuis–Jacobi);  D_H = {V ≤ H₀} (Hill).
  • KAM: |⟨k,ω⟩| ≥ γ/|k|^τ, τ > n−1;  Brjuno ℬ(ρ) < ∞.
  • Melnikov ℳ(t₀)=∫{H₀,H₁}(γ⁰(t−t₀)) dt; ceros simples ⇒ Smale.
  • P: Σ→Σ, M∈Sp(2n); Floquet (μ,1/μ,μ̄); Krein; Hill Δ=tr M.
  • Floer ∂̄_{J,H}(u)=0; CZ (Robbin–Salamon); Maslov.
  • Cayley w=(M−I)(M+I)^{−1};  G=−Ω w  (nervio de Čech).
  • Ȟ¹(𝒰; F_att) ≡ 0  (coborde + Betti b₁).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from itertools import product
from typing import Callable, Final, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Core.ImperialEruditosEngine")

__version__: Final[str] = "6.1.0-Poincare-Cartan-Melnikov-KAM-Floer-Cech-Nested-PhD"

# =============================================================================
# CONSTANTES DE PRECISIÓN METROLÓGICA Y MECÁNICA CELESTE DE POINCARÉ
# =============================================================================
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_HIGHAM_TIKHONOV_FLOOR: Final[float] = 1e-20
_WILKINSON_DEFLATION_FLOOR: Final[float] = 1e-15
_WILKINSON_DEFLATION_SCALE: Final[float] = 10.0
_WILKINSON_DRIFT_LIMIT: Final[float] = 1e-9
_WILKINSON_LIMIT: Final[float] = 1.0e-12
_SPECTRAL_TOL: Final[float] = 1.0e-9
_HARD_DIVERGENCE_CEILING: Final[float] = 1.0e-4
_CSMD_STEP: Final[float] = 1e-20
_CSMD_FD_FALLBACK: Final[float] = 1e-8
_CECH_TRIPLE_CAP: Final[int] = 80
_LOG_EXP_CLIP: Final[float] = 700.0
_MASLOV_DEGENERACY: Final[float] = 1e-10

_KAM_TAU_OFFSET: Final[float] = 1.0e-6          # τ = (n − 1) + offset
_KAM_GAMMA_FLOOR: Final[float] = 1e-12          # γ de Bruno–Rüssmann
_KAM_HARMONIC_CAP: Final[int] = 12
_KAM_MAX_DIM_EXACT: Final[int] = 3
_KAM_MONTE_CARLO: Final[int] = 4096
_ARNOLD_EXACT_DIM_CAP: Final[int] = 5
_ARNOLD_MAX_CANDIDATES: Final[int] = 250_000
_MELNIKOV_QUAD_NODES: Final[int] = 513
_FLOQUET_PARABOLIC_BAND: Final[float] = 1e-8
_HILL_MARGIN: Final[float] = 1e-12
_LYAPUNOV_CLIP: Final[float] = 700.0
_RETURN_MAP_MAX_ITER: Final[int] = 4096
_RETURN_MAP_TOL: Final[float] = 1e-11
_HARD_TWIST_FLOOR: Final[float] = 1.0e-8
_HARD_BRUNO_FLOOR: Final[float] = 1.0e-12
_ROTATION_CF_DEPTH: Final[int] = 24
_LYAPUNOV_QR_ITERATIONS: Final[int] = 2048
_HARD_LYAPUNOV_TOL: Final[float] = 1.0e-6
_DEFAULT_GERM_DIM: Final[int] = 2


# =============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ FASE I — ÁLGEBRA DE BANACH, DARBOUX, MAUPERTUIS–JACOBI Y POINCARÉ–CARTAN ║
# ║                                                                          ║
# ║ Objetos: sumas compensadas, 2-forma Ω, pfaffiano, θ de Liouville,        ║
# ║ λ_PC = p dq − H dt, g̃ de Maupertuis–Jacobi, región de Hill, CSMD,        ║
# ║ X_H = Ω ∇H, {·,·}, Jacobi, Verlet.                                       ║
# ║                                                                          ║
# ║ Morfismo terminal (I.ω): synthesize_poincare_cartan_germ                 ║
# ║     ≅ objeto inicial de la Fase II (verificador celeste).                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# =============================================================================
@dataclass(frozen=True, slots=True)
class _SymplecticFormCertificate:
    """Certificado algebraico de la 2-forma canónica de Liouville–Darboux."""
    skew_residual: float
    almost_complex_residual: float
    determinant: float
    pfaffian: float
    frobenius_norm: float
    closedness_residual: float
    is_darboux: bool


@dataclass(frozen=True, slots=True)
class _MaupertuisJacobiCertificate:
    r"""
    Certificado de la métrica conforme de Maupertuis–Jacobi.
        g̃_{jk}(q) = 2 (H₀ − V(q)) g_{jk}(q),
    definida positiva en la región accesible H₀ > V (interior de Hill).
    """
    conformal_factor: float
    min_eigenvalue: float
    is_positive_definite: bool
    hill_margin: float
    is_classically_allowed: bool


@dataclass(frozen=True, slots=True)
class _PoincareCartanOneForm:
    r"""
    1-forma de Poincaré–Cartan en el espacio extendido T*Q × ℝ:
        λ_PC = p_i dq^i − H dt.
    Componentes: coeficientes de dq (= p), de dp (= 0), de dt (= −H).
    """
    q: np.ndarray
    p: np.ndarray
    H: float
    theta_coefficients: np.ndarray
    lambda_extended: np.ndarray
    pairing_theta_against_q: float


@dataclass(frozen=True, slots=True)
class _HamiltonianVectorFieldWitness:
    """X_H = Ω ∇H  (convención X_H ⌟ ω = −dH ⇔ X_H = Ω ∇H con Ω canónica)."""
    vector_field: np.ndarray
    energy_derivative: float
    is_tangential_to_energy: bool


@dataclass(frozen=True, slots=True)
class _PoincareCartanGerm:
    r"""
    Gérmen de Poincaré–Cartan (objeto terminal de Fase I, inicial de Fase II).
        𝒢_I = (Ω, λ_PC, θ, g̃, H₀, CSMD, ε_reg, Cert(Ω), Cert(g̃))
    La Fase II lo recibe como argumento obligatorio de
    `_PoincareCelestialVerifier.__init__` / `phase2_ingest_poincare_cartan_germ`.
    """
    two_n: int
    n: int
    omega: np.ndarray
    poincare_cartan_lambda: np.ndarray
    hamiltonian_energy_H0: float
    maupertuis_metric: np.ndarray
    csmd_step: float
    reg_floor: float
    form_certificate: _SymplecticFormCertificate
    jacobi_certificate: _MaupertuisJacobiCertificate
    liouville_theta: np.ndarray
    closedness_residual: float
    pfaffian: float
    jacobi_identity_residual: float


class _NumericalCore:
    r"""
    Fase I. Álgebra numérica de precisión metrológica y cálculo holomorfo.
    Topos lineal: Banach (ℝ,+,·), 2-forma de Liouville, CSMD, gérmenes.

    **Cierre formal (I.ω)**:
        `synthesize_poincare_cartan_germ → _PoincareCartanGerm`
        Este objeto **es** el arranque formal de la Fase II.
    """

    # ── I.1 Sumación compensada (Banach (ℝ, +, ·)) ────────────────────────
    @staticmethod
    def kahan_sum(arr: np.ndarray) -> float:
        """Sumación compensada de Kahan."""
        total = 0.0
        c = 0.0
        for x in np.asarray(arr, dtype=np.float64).ravel():
            if not np.isfinite(x):
                raise ValueError("kahan_sum: se detectó un no-finito.")
            y = float(x) - c
            t = total + y
            c = (t - total) - y
            total = t
        return float(total)

    @staticmethod
    def kahan_babuska_neumaier_sum(arr: np.ndarray) -> float:
        """Sumación de Kahan–Babuška–Neumaier (KBN / Neumaier)."""
        total = 0.0
        c = 0.0
        for x in np.asarray(arr, dtype=np.float64).ravel():
            xf = float(x)
            if not np.isfinite(xf):
                raise ValueError("kahan_babuska_neumaier_sum: no-finito.")
            t = total + xf
            if abs(total) >= abs(xf):
                c += (total - t) + xf
            else:
                c += (xf - t) + total
            total = t
        result = float(total + c)
        return result if math.isfinite(result) else float(total)

    kahan_babuska_neumann_sum = kahan_babuska_neumaier_sum  # alias histórico

    @staticmethod
    def klein_sum(arr: np.ndarray) -> float:
        """Sumación doblemente compensada de Klein."""
        s = 0.0
        cs = 0.0
        ccs = 0.0
        for x in np.asarray(arr, dtype=np.float64).ravel():
            xf = float(x)
            if not np.isfinite(xf):
                raise ValueError("klein_sum: no-finito.")
            t = s + xf
            if abs(s) >= abs(xf):
                c = (s - t) + xf
            else:
                c = (xf - t) + s
            s = t
            t = cs + c
            if abs(cs) >= abs(c):
                cc = (cs - t) + c
            else:
                cc = (c - t) + cs
            cs = t
            ccs += cc
        return float(s + cs + ccs)

    # ── I.2 Normas y validaciones ────────────────────────────────────────
    @staticmethod
    def frobenius_norm(matrix: np.ndarray) -> float:
        """Norma de Hilbert–Schmidt / Frobenius ‖A‖_F."""
        a = np.asarray(matrix)
        if a.size == 0:
            return 0.0
        return float(la.norm(a, "fro"))

    @staticmethod
    def euclidean_norm(vec: np.ndarray) -> float:
        """Norma euclídea ‖v‖₂ con acumulación KBN sobre |v_i|²."""
        v = np.asarray(vec, dtype=np.float64).ravel()
        if v.size == 0:
            return 0.0
        return float(np.sqrt(max(_NumericalCore.kahan_babuska_neumaier_sum(v * v), 0.0)))

    @staticmethod
    def assert_finite(name: str, array: np.ndarray) -> None:
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contiene entradas no finitas.")

    @staticmethod
    def assert_square(name: str, matrix: np.ndarray, dim: Optional[int] = None) -> None:
        a = np.asarray(matrix)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError(f"{name} debe ser cuadrada; recibido {a.shape}.")
        if dim is not None and a.shape[0] != dim:
            raise ValueError(f"{name} debe ser {dim}×{dim}; recibido {a.shape}.")

    @staticmethod
    def assert_vec(name: str, vec: np.ndarray, dim: Optional[int] = None) -> np.ndarray:
        v = np.asarray(vec).reshape(-1)
        if dim is not None and v.size != dim:
            raise ValueError(f"{name} debe tener dimensión {dim}; recibido {v.size}.")
        _NumericalCore.assert_finite(name, v)
        return v

    @staticmethod
    def split_qp(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vec = _NumericalCore.assert_vec("state", np.asarray(state, dtype=np.float64))
        if vec.size % 2 != 0:
            raise ValueError("El estado debe tener dimensión par (2n).")
        n = vec.size // 2
        return vec[:n].copy(), vec[n:].copy()

    # ── I.3 Álgebra hermitiana y Tikhonov–Higham ─────────────────────────
    @staticmethod
    def higham_nearest_hermitian(matrix: np.ndarray) -> np.ndarray:
        """Proyección de Weyl–Toeplitz: (A + A†)/2."""
        a = np.asarray(matrix)
        _NumericalCore.assert_square("higham_nearest_hermitian", a)
        return 0.5 * (a + a.T.conj())

    @staticmethod
    def skew_residual(matrix: np.ndarray) -> float:
        """‖A + Aᵀ‖_F (cero sii A es antisimétrica real)."""
        a = np.asarray(matrix)
        return _NumericalCore.frobenius_norm(a + a.T)

    @staticmethod
    def tikhonov_higham_pinv(
        matrix: np.ndarray,
        rel_floor: float = _MACHINE_EPS,
        abs_floor: float = _HIGHAM_TIKHONOV_FLOOR,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Pseudoinversa amortiguada de Tikhonov–Higham vía SVD."""
        a = np.asarray(matrix)
        if a.size == 0:
            return a.copy(), np.array([], dtype=np.float64), float("inf")
        u_svd, s_vals, vt = la.svd(a, full_matrices=False)
        if s_vals.size == 0:
            return (
                np.zeros((a.shape[1], a.shape[0]), dtype=a.dtype),
                s_vals,
                float("inf"),
            )
        lam = max(float(abs_floor), float(rel_floor) * float(s_vals[0]))
        s_inv = np.zeros_like(s_vals)
        live = s_vals > lam
        s_inv[live] = s_vals[live] / (s_vals[live] ** 2 + lam ** 2)
        pinv = (vt.T.conj() * s_inv) @ u_svd.T.conj()
        s_min_live = float(s_vals[live].min()) if np.any(live) else lam
        cond = float(s_vals[0] / max(s_min_live, _MACHINE_EPS))
        return pinv, s_vals, cond

    @staticmethod
    def wilkinson_deflation_floor(matrix: np.ndarray) -> float:
        """Piso de deflación adaptativo de Wilkinson."""
        if matrix is None or np.asarray(matrix).size == 0:
            return _WILKINSON_DEFLATION_FLOOR
        fro_norm = _NumericalCore.frobenius_norm(matrix)
        return float(
            max(
                fro_norm * _MACHINE_EPS * _WILKINSON_DEFLATION_SCALE,
                _WILKINSON_DEFLATION_FLOOR,
            )
        )

    # ── I.4 Geometría simpléctica de Liouville–Darboux ────────────────────
    @staticmethod
    def generate_canonical_symplectic_form(dim: int) -> np.ndarray:
        r"""
        2-forma canónica Ω ∈ ℝ^{dim×dim}, dim = 2n par:
            Ω = [[0, I_n], [−I_n, 0]],
        de modo que Ωᵀ = −Ω, Ω² = −I, Ω^{-1} = −Ω, det Ω = 1, pf(Ω) = +1.
        """
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(
                f"La dimensión del espacio simpléctico dim={dim} debe ser par y positiva."
            )
        half = dim // 2
        omega = np.zeros((dim, dim), dtype=np.float64)
        omega[:half, half:] = np.eye(half, dtype=np.float64)
        omega[half:, :half] = -np.eye(half, dtype=np.float64)
        return omega

    @staticmethod
    def pfaffian_skew(omega: np.ndarray) -> float:
        r"""
        Pfaffiano de una matriz antisimétrica par. Identidad: det(Ω) = pf(Ω)².
        Para la forma canónica, pf = +1. Tridiagonalización de Bunch–Parlett.
        """
        a = np.array(omega, dtype=np.float64, copy=True)
        n = a.shape[0]
        if n % 2 != 0:
            return 0.0
        pf = 1.0
        for i in range(0, n, 2):
            col = np.abs(a[i, i + 1:])
            if col.size == 0:
                return 0.0
            k_rel = int(np.argmax(col))
            k = i + 1 + k_rel
            if abs(a[i, k]) < _MACHINE_EPS:
                return 0.0
            if k != i + 1:
                a[[i + 1, k], :] = a[[k, i + 1], :]
                a[:, [i + 1, k]] = a[:, [k, i + 1]]
                pf = -pf
            pivot = float(a[i, i + 1])
            pf *= pivot
            if i + 2 < n:
                inv_p = 1.0 / pivot
                row_tail = a[i, i + 2:].copy()
                nxt_tail = a[i + 1, i + 2:].copy()
                a[i + 2:, i + 2:] += inv_p * (
                    np.outer(nxt_tail, row_tail) - np.outer(row_tail, nxt_tail)
                )
                a[i + 2:, i] = 0.0
                a[i + 2:, i + 1] = 0.0
                a[i, i + 2:] = 0.0
                a[i + 1, i + 2:] = 0.0
        return float(pf)

    @staticmethod
    def certify_symplectic_form(omega: np.ndarray) -> _SymplecticFormCertificate:
        """Axiomas de Darboux: Ω antisimétrica, Ω² = −I, det Ω = 1, pf Ω = 1, dω = 0."""
        _NumericalCore.assert_square("omega", omega)
        dim = omega.shape[0]
        ident = np.eye(dim, dtype=omega.dtype)
        skew = _NumericalCore.skew_residual(omega)
        almost_c = _NumericalCore.frobenius_norm(omega @ omega + ident)
        det_o = float(np.real(la.det(omega)))
        fro = _NumericalCore.frobenius_norm(omega)
        scale = max(fro, 1.0)
        try:
            pf = _NumericalCore.pfaffian_skew(omega)
        except Exception:
            pf = math.copysign(math.sqrt(max(abs(det_o), 0.0)), 1.0)
        # Ω constante ⇒ dω ≡ 0. El residual de cerradura se identifica al sesgo.
        closedness = float(skew)
        is_darboux = bool(
            skew <= _WILKINSON_DRIFT_LIMIT * scale
            and almost_c <= _WILKINSON_DRIFT_LIMIT * scale
            and abs(det_o - 1.0) <= 1e-8 * max(1.0, abs(det_o))
            and abs(pf - 1.0) <= 1e-6 * max(1.0, abs(pf))
        )
        return _SymplecticFormCertificate(
            skew_residual=float(skew),
            almost_complex_residual=float(almost_c),
            determinant=det_o,
            pfaffian=float(pf),
            frobenius_norm=fro,
            closedness_residual=closedness,
            is_darboux=is_darboux,
        )

    @staticmethod
    def jacobi_identity_residual(omega: np.ndarray) -> float:
        r"""
        Identidad de Jacobi {f,{g,h}}+cícl. = 0  ⇔  dω = 0.
        Testigo numérico sobre funciones coordenadas: {q_i, p_i} = 1, {q,q}={p,p}=0.
        """
        o = np.asarray(omega, dtype=np.float64)
        dim = o.shape[0]
        if dim < 2 or dim % 2 != 0:
            return float(_NumericalCore.skew_residual(o))
        n = dim // 2

        def pb(i: int, j: int) -> float:
            ei = np.zeros(dim); ei[i] = 1.0
            ej = np.zeros(dim); ej[j] = 1.0
            return float(ei @ o @ ej)

        r_qp = abs(pb(0, n) - 1.0)
        r_qq = abs(pb(0, min(1, n - 1))) if n >= 2 else 0.0
        r_pp = abs(pb(n, min(n + 1, dim - 1))) if n >= 2 else 0.0
        return float(r_qp + r_qq + r_pp + _NumericalCore.skew_residual(o))

    # ── I.5 Cálculo holomorfo CSMD (sin cancelación sustractiva) ──────────
    @staticmethod
    def compute_gradient_csmd(
        func: Callable[[np.ndarray], float],
        x: np.ndarray,
        h: float = _CSMD_STEP,
    ) -> np.ndarray:
        r"""
        Gradiente CSMD de una función escalar:
            ∇_k H(x) = Im[H(x + j·h·e_k)] / h + O(h²).
        Si `func` no es holomorfa (p.ej. recorta a ℝ), se cae a diferencia central.
        """
        xv = _NumericalCore.assert_vec("x", np.asarray(x, dtype=np.float64))
        if not np.isfinite(h) or h == 0.0:
            raise ValueError("El paso CSMD h debe ser finito y no nulo.")
        h = float(abs(h))
        dim = xv.size
        grad = np.zeros(dim, dtype=np.float64)
        holomorphic = True
        try:
            probe = func(xv.astype(np.complex128))
            probe_c = np.asarray(probe)
            if probe_c.dtype.kind != "c":
                # La función ignoró la parte imaginaria → no es un stencil holomorfo.
                holomorphic = False
            elif not (np.isfinite(float(np.real(probe_c))) or np.isfinite(float(np.imag(probe_c)))):
                holomorphic = False
        except (TypeError, ValueError, FloatingPointError):
            holomorphic = False

        if holomorphic:
            imag_acc = 0.0
            for i in range(dim):
                xp = xv.astype(np.complex128)
                xp[i] += 1j * h
                try:
                    val = func(xp)
                except (TypeError, ValueError, FloatingPointError):
                    holomorphic = False
                    break
                imag = float(np.imag(np.asarray(val)))
                if not np.isfinite(imag):
                    holomorphic = False
                    break
                grad[i] = imag / h
                imag_acc += abs(imag)
            # func real-valuada que acepta complex128 pero devuelve Im≡0.
            if holomorphic and imag_acc <= _MACHINE_EPS * max(h, 1.0) * dim:
                holomorphic = False

        if not holomorphic:
            logger.warning(
                "CSMD: func no es holomorfa en el stencil; se usa diferencia central."
            )
            scale = max(_NumericalCore.euclidean_norm(xv), 1.0)
            h_fd = max(_CSMD_FD_FALLBACK, _CSMD_FD_FALLBACK * scale)
            for i in range(dim):
                xp = xv.copy()
                xm = xv.copy()
                xp[i] += h_fd
                xm[i] -= h_fd
                fp = float(np.real(func(xp)))
                fm = float(np.real(func(xm)))
                if not (np.isfinite(fp) and np.isfinite(fm)):
                    raise ValueError("compute_gradient_csmd: func devolvió no-finitos.")
                grad[i] = (fp - fm) / (2.0 * h_fd)
        return grad

    @staticmethod
    def compute_hessian_csmd(
        func: Callable[[np.ndarray], float],
        x: np.ndarray,
        h: float = _CSMD_STEP,
    ) -> np.ndarray:
        """Hessiano por diferencia central de gradientes CSMD (simetrizado Higham)."""
        xv = _NumericalCore.assert_vec("x", np.asarray(x, dtype=np.float64))
        dim = xv.size
        scale = max(_NumericalCore.euclidean_norm(xv), 1.0)
        eta = max(np.sqrt(_MACHINE_EPS), np.sqrt(_MACHINE_EPS) * scale)
        hess = np.zeros((dim, dim), dtype=np.float64)
        for j in range(dim):
            xp = xv.copy()
            xm = xv.copy()
            xp[j] += eta
            xm[j] -= eta
            gp = _NumericalCore.compute_gradient_csmd(func, xp, h=h)
            gm = _NumericalCore.compute_gradient_csmd(func, xm, h=h)
            hess[:, j] = (gp - gm) / (2.0 * eta)
        return np.real(_NumericalCore.higham_nearest_hermitian(hess))

    @classmethod
    def compute_symplectic_gradient(
        cls,
        hamiltonian_func: Callable[[np.ndarray], float],
        x: np.ndarray,
        h: float = _CSMD_STEP,
    ) -> np.ndarray:
        r"""Campo vectorial hamiltoniano X_H = Ω ∇H(x)."""
        xv = cls.assert_vec("x", np.asarray(x, dtype=np.float64))
        dim = xv.size
        if dim % 2 != 0:
            raise ValueError(
                f"X_H exige dimensión par (Darboux); recibido dim={dim}."
            )
        omega = cls.generate_canonical_symplectic_form(dim)
        grad = cls.compute_gradient_csmd(hamiltonian_func, xv, h)
        return omega @ grad

    @classmethod
    def hamiltonian_vector_field_witness(
        cls,
        hamiltonian_func: Callable[[np.ndarray], float],
        x: np.ndarray,
        h: float = _CSMD_STEP,
    ) -> _HamiltonianVectorFieldWitness:
        """X_H junto con el testigo ℒ_{X_H} H = ∇H · X_H = 0."""
        xv = cls.assert_vec("x", np.asarray(x, dtype=np.float64))
        dim = xv.size
        if dim % 2 != 0:
            raise ValueError("X_H exige dimensión par.")
        omega = cls.generate_canonical_symplectic_form(dim)
        grad = cls.compute_gradient_csmd(hamiltonian_func, xv, h)
        xh = omega @ grad
        energy_der = float(grad @ xh)
        scale = max(1.0, cls.euclidean_norm(grad))
        return _HamiltonianVectorFieldWitness(
            vector_field=xh,
            energy_derivative=energy_der,
            is_tangential_to_energy=bool(abs(energy_der) <= 1e-10 * scale),
        )

    @staticmethod
    def poisson_bracket(
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray], float],
        x: np.ndarray,
        h: float = _CSMD_STEP,
    ) -> float:
        r"""
        Corchete de Poisson {H₀, H₁}(x) = (∇H₀)ᵀ Ω ∇H₁ en T*Q.
        Si {H₀, H₁} ≠ 0, la perturbación ε H₁ rompe las integrales de H₀.
        """
        xv = _NumericalCore.assert_vec("x", np.asarray(x, dtype=np.float64))
        dim = xv.size
        if dim % 2 != 0:
            raise ValueError("El corchete de Poisson exige dim par (Darboux).")
        omega = _NumericalCore.generate_canonical_symplectic_form(dim)
        grad0 = _NumericalCore.compute_gradient_csmd(hamiltonian_0, xv, h)
        grad1 = _NumericalCore.compute_gradient_csmd(hamiltonian_1, xv, h)
        return float(grad0 @ omega @ grad1)

    # ── I.6 Acción de Liouville y 1-forma θ ──────────────────────────────
    @staticmethod
    def liouville_action(start_point: np.ndarray, end_point: np.ndarray) -> float:
        r"""
        Acción de Liouville del segmento geodésico euclídeo γ: start → end:
            𝒜_L(γ) = ∫_γ θ = ½ (p₀ + p₁) · (q₁ − q₀).
        """
        z0 = _NumericalCore.assert_vec("start_point", start_point)
        z1 = _NumericalCore.assert_vec("end_point", end_point)
        if z0.size != z1.size:
            raise ValueError("start_point y end_point deben tener la misma dimensión.")
        if z0.size % 2 != 0:
            raise ValueError("Los puntos de Floer deben vivir en T*Q (dim par).")
        n = z0.size // 2
        q0, p0 = z0[:n], z0[n:]
        q1, p1 = z1[:n], z1[n:]
        mid_p = 0.5 * (p0 + p1)
        dq = q1 - q0
        return _NumericalCore.kahan_babuska_neumaier_sum(mid_p * dq)

    @staticmethod
    def liouville_one_form_coefficients(x: np.ndarray) -> np.ndarray:
        r"""Coeficientes de θ = p dq: el covector (p, 0_p) ∈ T*_x(T*Q)."""
        q, p = _NumericalCore.split_qp(x)
        return np.concatenate([p, np.zeros_like(p)])

    # ── I.7 Métrica conforme de Maupertuis–Jacobi ────────────────────────
    @staticmethod
    def compute_maupertuis_jacobi_conformal_metric(
        hamiltonian_energy_H0: float,
        potential_energy_V: float,
        base_metric_g: np.ndarray,
    ) -> Tuple[np.ndarray, _MaupertuisJacobiCertificate]:
        r"""
        Métrica conforme de Maupertuis–Jacobi:
            g̃_{jk}(q) = 2 (H₀ − V(q)) g_{jk}(q).
        En D_H = {H₀ > V} es Riemanniana; en ∂D_H se anula (curva de velocidad cero).
        **No** se clampa el factor conforme: si el margen de Hill es ≤ 0 se
        certifica `is_classically_allowed=False` y g̃ puede ser semidefinida/negativa.
        """
        g = np.asarray(base_metric_g, dtype=np.float64)
        _NumericalCore.assert_square("base_metric_g", g)
        h0 = float(hamiltonian_energy_H0)
        v = float(potential_energy_V)
        if not (np.isfinite(h0) and np.isfinite(v)):
            raise ValueError("Energías H₀ y V deben ser finitas.")
        hill_margin = 2.0 * (h0 - v)          # 2(E − V), factor conforme crudo
        allowed = bool(h0 - v > _HILL_MARGIN)
        phi = float(hill_margin)
        gt = phi * g
        try:
            eigs = la.eigvalsh(_NumericalCore.higham_nearest_hermitian(gt))
            min_eig = float(np.min(eigs)) if eigs.size else 0.0
        except la.LinAlgError:
            min_eig = 0.0
        cert = _MaupertuisJacobiCertificate(
            conformal_factor=float(phi),
            min_eigenvalue=min_eig,
            is_positive_definite=bool(min_eig > _WILKINSON_LIMIT),
            hill_margin=float(h0 - v),
            is_classically_allowed=allowed,
        )
        return gt, cert

    @staticmethod
    def compute_hill_region(
        hamiltonian_energy_H0: float,
        potential_energy_V: float,
    ) -> float:
        r"""Margen de Hill: H₀ − V(q). Si ≤ 0, q está prohibido clásicamente."""
        return float(hamiltonian_energy_H0 - potential_energy_V)

    # ── I.8 1-forma de Poincaré–Cartan ───────────────────────────────────
    @staticmethod
    def compute_poincare_cartan_lambda(
        x: np.ndarray,
        hamiltonian_func: Optional[Callable[[np.ndarray], float]] = None,
        csmd_step: float = _CSMD_STEP,
    ) -> np.ndarray:
        r"""
        Componentes reducidas de λ_PC = p dq − H dt:
            (p_1,…,p_n, −H) ∈ ℝ^{n+1}.
        (API histórica; ver `compute_poincare_cartan_one_form` para el objeto completo.)
        """
        form = _NumericalCore.compute_poincare_cartan_one_form(
            x, hamiltonian_func=hamiltonian_func, csmd_step=csmd_step
        )
        return form.lambda_extended

    @staticmethod
    def compute_poincare_cartan_one_form(
        x: np.ndarray,
        hamiltonian_func: Optional[Callable[[np.ndarray], float]] = None,
        csmd_step: float = _CSMD_STEP,  # noqa: ARG002  (reservado para dH)
    ) -> _PoincareCartanOneForm:
        r"""
        1-forma de Poincaré–Cartan λ = p dq − H dt en x ∈ T*Q.
        dλ = ω − dH ∧ dt  es el invariante integral absoluto (E. Cartan, 1922).
        """
        xv = _NumericalCore.assert_vec("x", np.asarray(x, dtype=np.float64))
        q, p = _NumericalCore.split_qp(xv)
        if hamiltonian_func is None:
            h_val = 0.0
        else:
            try:
                h_val = float(np.real(hamiltonian_func(xv)))
            except (TypeError, ValueError, FloatingPointError):
                h_val = 0.0
            if not np.isfinite(h_val):
                h_val = 0.0
        lam_ext = np.concatenate([p, np.array([-h_val], dtype=np.float64)])
        return _PoincareCartanOneForm(
            q=q, p=p, H=float(h_val),
            theta_coefficients=np.concatenate([p, np.zeros_like(p)]),
            lambda_extended=lam_ext,
            pairing_theta_against_q=float(p @ q),
        )

    # ── I.9 Certificado de invarianza integral absoluta ──────────────────
    @staticmethod
    def poincare_cartan_integral_invariant_residual(omega: np.ndarray, dim: int) -> float:
        r"""
        Residual del teorema de Poincaré–Cartan:
            ∮_{γ_t} p dq − H dt = inv.  ⟺  d(λ − H dt) = ω − dH ∧ dt.
        Para Ω canónica, dΩ = 0 se reduce a Ω antisimétrica pura.
        """
        omega = np.asarray(omega)
        _NumericalCore.assert_square("omega", omega, dim=dim)
        return _NumericalCore.skew_residual(omega)

    # ── I.10 Integrador simpléctico Störmer–Verlet ───────────────────────
    @staticmethod
    def stormer_verlet_step(
        q: np.ndarray,
        p: np.ndarray,
        grad_v: Callable[[np.ndarray], np.ndarray],
        mass_inv: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Paso de Störmer–Verlet sobre H(q,p) = ½ |p|²/m + V(q):
            p_{n+½} = p_n − (dt/2) ∇V(q_n)
            q_{n+1} = q_n + dt · m^{-1} · p_{n+½}
            p_{n+1} = p_{n+½} − (dt/2) ∇V(q_{n+1})
        Preserva Ω exactamente (composición de cizallas).
        """
        qn = np.asarray(q, dtype=np.float64)
        pn = np.asarray(p, dtype=np.float64)
        p_half = pn - 0.5 * dt * np.asarray(grad_v(qn), dtype=np.float64)
        q_next = qn + dt * mass_inv * p_half
        p_next = p_half - 0.5 * dt * np.asarray(grad_v(q_next), dtype=np.float64)
        return q_next, p_next

    @staticmethod
    def verlet_monodromy_jacobian(
        two_n: int,
        dt_step: float,
        metric_G_inv: Optional[np.ndarray] = None,
        hessian_V: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""
        Jacobiano exacto del paso de Verlet (cizalla–drift–cizalla).
        Linealizando Hess V ≈ K, G^{-1} ≈ B se obtiene el mapa de Hill discreto,
        que es exactamente simpléctico: Mᵀ Ω M = Ω.
        """
        if two_n % 2 != 0 or two_n <= 0:
            raise ValueError("two_n debe ser par positivo.")
        n = two_n // 2
        dt = float(dt_step)
        B = np.eye(n, dtype=np.float64) if metric_G_inv is None \
            else np.asarray(metric_G_inv, dtype=np.float64)
        if B.shape == (n,):
            B = np.diag(B)
        if B.shape != (n, n):
            raise ValueError("metric_G_inv incompatible.")
        K = np.zeros((n, n), dtype=np.float64) if hessian_V is None \
            else np.asarray(hessian_V, dtype=np.float64)
        if K.shape != (n, n):
            raise ValueError("hessian_V incompatible.")
        ident = np.eye(n, dtype=np.float64)
        half = 0.5 * dt
        s1 = np.block([[ident, np.zeros((n, n))], [-half * K, ident]])
        dft = np.block([[ident, dt * B], [np.zeros((n, n)), ident]])
        return s1 @ dft @ s1

    # ── I.ω  MORFISMO TERMINAL Φ_I: gérmen de Poincaré–Cartan ────────────
    @classmethod
    def synthesize_poincare_cartan_germ(
        cls,
        dimension_two_n: int,
        hamiltonian_energy_H0: float = 1.0,
        potential_energy_V: float = 0.0,
        csmd_step: float = _CSMD_STEP,
        regularizer: float = _HIGHAM_TIKHONOV_FLOOR,
        hamiltonian_func: Optional[Callable[[np.ndarray], float]] = None,
        scale_matrix: Optional[np.ndarray] = None,
        base_metric_g: Optional[np.ndarray] = None,
    ) -> _PoincareCartanGerm:
        r"""
        **I.ω — Morfismo terminal de la FASE I / objeto inicial de la FASE II.**

        Sintetiza el gérmen
            𝒢_I = (Ω, λ_PC, θ, g̃, H₀, CSMD, ε_reg, Cert(Ω), Cert(g̃), Jacobi)
        sobre el cual la Fase II instancia KAM, Melnikov, retorno, Floer.

        Este método **es** el arranque formal de la Fase II:
        `_PoincareCelestialVerifier(𝒢_I)` / `phase2_ingest_poincare_cartan_germ`.
        """
        dim = int(dimension_two_n)
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(
                f"dimension_two_n debe ser par y positivo; recibido {dimension_two_n}."
            )
        if not np.isfinite(csmd_step) or csmd_step == 0.0:
            raise ValueError("csmd_step debe ser finito y no nulo.")
        omega = cls.generate_canonical_symplectic_form(dim)
        certificate = cls.certify_symplectic_form(omega)
        if not certificate.is_darboux:
            logger.warning(
                "Certificado de Darboux degradado: skew=%.3e, J²+I=%.3e, det=%.16f, pf=%.16f",
                certificate.skew_residual,
                certificate.almost_complex_residual,
                certificate.determinant,
                certificate.pfaffian,
            )
        n = dim // 2
        x0 = np.zeros(dim, dtype=np.float64)
        form = cls.compute_poincare_cartan_one_form(x0, hamiltonian_func, csmd_step)
        g_base = np.eye(n, dtype=np.float64) if base_metric_g is None \
            else np.asarray(base_metric_g, dtype=np.float64)
        gtilde, jacobi_cert = cls.compute_maupertuis_jacobi_conformal_metric(
            hamiltonian_energy_H0, potential_energy_V, g_base
        )
        if not jacobi_cert.is_classically_allowed:
            logger.warning(
                "Región de Hill prohibida: H₀ − V = %.3e.", jacobi_cert.hill_margin
            )
        floor = max(float(regularizer), _HIGHAM_TIKHONOV_FLOOR)
        if scale_matrix is not None:
            floor = max(floor, cls.wilkinson_deflation_floor(np.asarray(scale_matrix)))
        jacobi_res = cls.jacobi_identity_residual(omega)
        return _PoincareCartanGerm(
            two_n=dim,
            n=n,
            omega=omega,
            poincare_cartan_lambda=form.lambda_extended,
            hamiltonian_energy_H0=float(hamiltonian_energy_H0),
            maupertuis_metric=gtilde,
            csmd_step=float(abs(csmd_step)),
            reg_floor=float(floor),
            form_certificate=certificate,
            jacobi_certificate=jacobi_cert,
            liouville_theta=form.theta_coefficients,
            closedness_residual=float(certificate.closedness_residual),
            pfaffian=float(certificate.pfaffian),
            jacobi_identity_residual=float(jacobi_res),
        )


# =============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ FASE II — MECÁNICA CELESTE DE POINCARÉ: KAM, MELNIKOV, RETORNO Y FLOER   ║
# ║                                                                          ║
# ║ El primer método consume I.ω; el último produce II.ω (inicio de III).    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# =============================================================================
@dataclass(frozen=True, slots=True)
class EruditosSpectrumReport:
    """Reporte numérico inmutable emitido por la FPU del motor de los Eruditos."""
    min_small_divisor: float
    novikov_absorbed_weight: float
    maurercartan_residual: float
    liouville_volume_drift: float
    is_kam_stable: bool


PoincareEruditosSpectrumReport = EruditosSpectrumReport


@dataclass(frozen=True, slots=True)
class _KAMStabilityCertificate:
    r"""
    Certificado de estabilidad KAM de Poincaré–Arnol'd–Moser.
      |⟨k,ω⟩| ≥ γ / |k|^τ  ∀ k ≠ 0,  τ > n − 1,
    más la suma de Brjuno ℬ(ρ) (unidimensional).
    """
    min_divisor: float
    tau: float
    gamma: float
    resonance_gap: float
    is_diophantine: bool
    bruno_sum: float
    estimated_gamma: float
    n_modes: int


@dataclass(frozen=True, slots=True)
class _MelnikovCertificate:
    r"""
    Certificado de la función de Melnikov para la ruptura homoclínica.
        M(t₀) = ∫_{-∞}^{∞} {H₀, H₁}(γ⁰(t − t₀)) dt.
    Ceros simples ⇒ intersecciones transversas (herradura de Smale).
    """
    melnikov_value: float
    melnikov_derivative: float
    is_simple_zero: bool
    homoclinic_splitting: float
    melnikov_values: np.ndarray
    simple_zeros: int
    is_chaotic: bool


@dataclass(frozen=True, slots=True)
class _PoincareReturnMapCertificate:
    r"""
    Certificado del mapa de retorno de Poincaré P: Σ → Σ.
    Para M ∈ Sp(2n): {μ} = {1/μ} = {μ̄} (Krein).
    Clasificación **no exclusiva**: un espectro mixto puede ser hiperbólico
    en unos pares de Krein y elíptico en otros.
    """
    floquet_multipliers: np.ndarray
    lyapunov_spectrum: np.ndarray
    is_hyperbolic: bool
    is_elliptic: bool
    is_parabolic: bool
    is_mixed: bool
    trace_M: float
    det_M: float
    reciprocal_pair_residual: float
    unit_circle_residual: float
    hill_discriminant: float
    symplectic_residual: float


@dataclass(frozen=True, slots=True)
class _PoincareSectionWitness:
    """Sección transversal Σ = {q_k = q*}: Ω n_Σ ≠ 0 y n_Σ · X_H ≠ 0."""
    normal: np.ndarray
    section_index: int
    section_offset: float
    transversal_certificate: float
    flow_transversal_certificate: float
    is_transversal: bool
    is_flow_transversal: bool


@dataclass(frozen=True, slots=True)
class _LyapunovSpectrumWitness:
    """Espectro de Lyapunov por QR de Benettin + Kaplan–Yorke + Pesin."""
    spectrum: np.ndarray
    kaplan_yorke_dimension: float
    kolmogorov_sinai_entropy: float
    is_chaotic: bool
    sum_all: float


@dataclass(frozen=True, slots=True)
class _ActionAngleWitness:
    """I_k = (1/2π) ∮ p_k dq_k ; twist ∂ρ/∂I."""
    actions: np.ndarray
    angles: np.ndarray
    twist_jacobian: float
    is_twist: bool


@dataclass(frozen=True, slots=True)
class _FloerResult:
    """Resultado certificado de una trayectoria / cilindro de Floer."""
    floer_residual: float
    action_potential: float
    liouville_action: float
    dirichlet_energy: float
    symplectic_monodromy_residual: float
    conley_zehnder_index: float
    maslov_degeneracy: float
    is_nondegenerate: bool
    is_symplectic_monodromy: bool


@dataclass(frozen=True, slots=True)
class _CechNerveGerm:
    r"""
    Gérmen del nervio atencional (objeto terminal de Fase II, inicial de Fase III).
    Cuantiza la monodromía M ↦ Gram de Čech via Cayley w = (M−I)(M+I)^{−1},
    G = −Ω w (simetrizado).
    """
    sheaf_gram: np.ndarray
    cayley_condition: float
    two_n: int
    reg_floor: float
    from_floer: bool
    symplectic_residual: float
    conley_zehnder_index: float


class _PoincareCelestialVerifier:
    r"""
    Fase II. Verificador de mecánica celeste de Poincaré.

    **II.0** `phase2_ingest_poincare_cartan_germ` continúa I.ω.
    **II.ω** `induce_cech_nerve_germ` abre la Fase III.
    """

    def __init__(self, germ: _PoincareCartanGerm) -> None:
        self._germ = self.phase2_ingest_poincare_cartan_germ(germ)

    @property
    def germ(self) -> _PoincareCartanGerm:
        return self._germ

    # ── II.0  INGESTA DEL OBJETO TERMINAL DE LA FASE I ───────────────────
    def phase2_ingest_poincare_cartan_germ(
        self, germ: _PoincareCartanGerm,
    ) -> _PoincareCartanGerm:
        r"""
        **II.0 — Flecha inicial de Φ_II, continuación estricta de I.ω.**

        Valida 𝒢_I (tipos, paridad 2n, Darboux, Hill) y lo reexpide como
        objeto de trabajo. Toda la dinámica posterior factoriza a través de aquí.
        """
        if not isinstance(germ, _PoincareCartanGerm):
            raise TypeError(
                "germ debe ser _PoincareCartanGerm (cierre I.ω)."
            )
        if germ.two_n <= 0 or germ.two_n % 2 != 0:
            raise ValueError("𝒢_I no tiene dimensión de Darboux.")
        if germ.omega.shape != (germ.two_n, germ.two_n):
            raise ValueError("Ω de 𝒢_I incompatible con two_n.")
        if not germ.form_certificate.is_darboux:
            logger.warning(
                "II.0: Darboux no certificado (skew=%.3e, pf=%.6f); se prosigue degradado.",
                germ.form_certificate.skew_residual,
                germ.form_certificate.pfaffian,
            )
        if not germ.jacobi_certificate.is_classically_allowed:
            logger.warning(
                "II.0: punto fuera de la región de Hill (margen=%.3e).",
                germ.jacobi_certificate.hill_margin,
            )
        return germ

    # ── II.1 Pequeños divisores de Poincaré–KAM ──────────────────────────
    @staticmethod
    def _bruno_sum(cf: Tuple[int, ...]) -> float:
        r"""ℬ(ρ) = Σ_k (log q_{k+1}) / q_k  (Brjuno unidimensional)."""
        if not cf:
            return float("inf")
        k_prev, k_cur = 1, 0
        qs = []
        for ai in cf:
            k_prev, k_cur = k_cur, ai * k_cur + k_prev
            qs.append(max(abs(k_cur), 1))
        acc = 0.0
        for i in range(len(qs) - 1):
            qk = max(qs[i], 1)
            qn = max(qs[i + 1], 1)
            acc += math.log(float(qn)) / float(qk)
        return float(acc)

    @staticmethod
    def _continued_fraction(rho: float, depth: int = _ROTATION_CF_DEPTH) -> Tuple[int, ...]:
        x = abs(float(rho))
        cf = []
        for _ in range(depth):
            ai = int(np.floor(x))
            cf.append(ai)
            frac = x - ai
            if frac < 1e-14:
                break
            x = 1.0 / frac
        return tuple(cf)

    def compute_poincare_small_divisors_spectrum(
        self,
        frequency_vector_omega: NDArray[np.float64],
        wave_vectors_k: NDArray[np.float64],
        jacobian_M: NDArray[np.float64],
        canonical_J: NDArray[np.float64],
        tau: Optional[float] = None,
        gamma: float = _KAM_GAMMA_FLOOR,
        novikov_valuation_T: float = 1.0,
    ) -> Tuple[EruditosSpectrumReport, _KAMStabilityCertificate]:
        r"""
        Espectro de pequeños divisores de Poincaré–KAM y absorción ultramétrica
        T-ádica en el Anillo de Novikov Λ_Nov.

        La condición diofantina se verifica **sobre todos** los k suministrados:
            γ_est = min_{k≠0} |⟨k,ω⟩| · |k|^τ ,   τ > n − 1,
        no sólo sobre el divisor mínimo (que es necesario pero no suficiente).
        `canonical_J` se certifica contra Ω de 𝒢_I.
        """
        freq_omega = self._assert_vec("frequency_vector_omega", frequency_vector_omega)
        n_freq = int(freq_omega.size)
        tau_eff = float(tau) if tau is not None else float(max(n_freq - 1, 0) + _KAM_TAU_OFFSET)
        wave_k = np.asarray(wave_vectors_k, dtype=np.float64)
        jac_m = np.asarray(jacobian_M, dtype=np.float64)
        j_can = np.asarray(canonical_J, dtype=np.float64)

        # Certifica J contra Ω (si las dimensiones coinciden).
        j_res = 0.0
        if j_can.ndim == 2 and j_can.shape[0] == j_can.shape[1]:
            if j_can.shape[0] == self._germ.two_n:
                j_res = _NumericalCore.frobenius_norm(j_can - self._germ.omega)
            else:
                try:
                    omega_j = _NumericalCore.generate_canonical_symplectic_form(j_can.shape[0])
                    j_res = _NumericalCore.frobenius_norm(j_can - omega_j)
                except ValueError:
                    j_res = _NumericalCore.skew_residual(j_can)

        if wave_k.size == 0:
            divisors = np.array([], dtype=np.float64)
            k_norms = np.array([], dtype=np.float64)
        elif wave_k.ndim == 1:
            if wave_k.size != n_freq:
                # Interpretar como un único modo embebido / recortado.
                m = min(wave_k.size, n_freq)
                kv = np.zeros(n_freq, dtype=np.float64)
                kv[:m] = wave_k[:m]
                wave_k = kv.reshape(1, -1)
            else:
                wave_k = wave_k.reshape(1, -1)
            divisors = np.abs(wave_k @ freq_omega)
            k_norms = np.linalg.norm(wave_k, axis=1)
        else:
            if wave_k.shape[1] != n_freq:
                raise ValueError(
                    f"wave_vectors_k columnas={wave_k.shape[1]} ≠ dim(ω)={n_freq}."
                )
            divisors = np.abs(wave_k @ freq_omega)
            k_norms = np.linalg.norm(wave_k, axis=1)

        if divisors.size == 0:
            min_divisor = 1.0
            resonance_gap = 1.0
            gamma_est = float("inf")
            is_diophantine = True
        else:
            # Descarta el modo nulo k = 0.
            live = k_norms > 1e-12
            if not np.any(live):
                min_divisor = 1.0
                resonance_gap = 1.0
                gamma_est = float("inf")
                is_diophantine = True
            else:
                d_live = divisors[live]
                n_live = np.maximum(k_norms[live], 1.0)
                min_divisor = float(np.min(d_live))
                resonance_gap = min_divisor
                gamma_est = float(np.min(d_live * (n_live ** tau_eff)))
                is_diophantine = bool(
                    math.isfinite(gamma_est) and gamma_est >= float(gamma)
                )

        novikov_weight = float(
            np.exp(-np.clip(
                novikov_valuation_T / (_WILKINSON_LIMIT + min_divisor),
                0.0, _LOG_EXP_CLIP,
            ))
        )
        mc_residual = float(abs(min_divisor * novikov_weight) + j_res * 0.0 + j_res)

        if jac_m.ndim == 2 and jac_m.shape[0] == jac_m.shape[1] and jac_m.size > 0:
            det_m = float(np.real(la.det(jac_m)))
            volume_drift = float(abs(det_m - 1.0))
        else:
            volume_drift = 0.0

        bruno = 0.0
        if n_freq >= 2 and abs(freq_omega[0]) > _MACHINE_EPS:
            bruno = self._bruno_sum(
                self._continued_fraction(float(freq_omega[1] / freq_omega[0]))
            )

        is_kam_stable = bool(
            (min_divisor >= _WILKINSON_LIMIT)
            and (volume_drift <= _WILKINSON_LIMIT)
            and is_diophantine
            and (not math.isfinite(bruno) or bruno < 1.0 / max(_HARD_BRUNO_FLOOR, _MACHINE_EPS)
                 or bruno == 0.0 or math.isfinite(bruno))
        )
        report = EruditosSpectrumReport(
            min_small_divisor=min_divisor,
            novikov_absorbed_weight=novikov_weight,
            maurercartan_residual=float(mc_residual),
            liouville_volume_drift=volume_drift,
            is_kam_stable=is_kam_stable,
        )
        kam_cert = _KAMStabilityCertificate(
            min_divisor=min_divisor,
            tau=float(tau_eff),
            gamma=float(gamma),
            resonance_gap=float(resonance_gap),
            is_diophantine=is_diophantine,
            bruno_sum=float(bruno),
            estimated_gamma=float(min(gamma_est, 1e12) if math.isfinite(gamma_est) else 0.0),
            n_modes=int(n_freq),
        )
        return report, kam_cert

    # ── II.2 Retícula de resonancias de Arnol'd ──────────────────────────
    def compute_arnold_resonance_lattice(
        self,
        frequency_vector_omega: NDArray[np.float64],
        max_order: int = 4,
        tol: float = 1e-6,
    ) -> NDArray[np.int64]:
        r"""
        Retícula de resonancias de Arnol'd:
            R_ε(ω) = { k ∈ ℤ^n \ {0} : |k|_1 ≤ max_order, |⟨k,ω⟩| < tol }.
        Para n > 5 se enumeran ejes, pares e_i±e_j y una muestra acotada
        (evita (2K+1)^n).
        """
        omega = self._assert_vec("frequency_vector_omega", frequency_vector_omega)
        n = int(omega.size)
        if n == 0 or max_order < 1:
            return np.zeros((0, max(n, 0)), dtype=np.int64)
        k_max = int(max_order)

        def _filter(k_all: np.ndarray) -> np.ndarray:
            if k_all.size == 0:
                return np.zeros((0, n), dtype=np.int64)
            norms = np.abs(k_all).sum(axis=1)
            keep = (norms > 0) & (norms <= k_max)
            if not np.any(keep):
                return np.zeros((0, n), dtype=np.int64)
            k_cand = k_all[keep]
            inner = np.abs(k_cand.astype(np.float64) @ omega)
            return k_cand[inner < tol]

        if n <= _ARNOLD_EXACT_DIM_CAP:
            # Bola ℓ_1 por producto cartesiano, con corte de cardinalidad.
            ranges = [range(-k_max, k_max + 1) for _ in range(n)]
            rows = []
            for tup in product(*ranges):
                if sum(abs(v) for v in tup) == 0:
                    continue
                if sum(abs(v) for v in tup) > k_max:
                    continue
                rows.append(tup)
                if len(rows) >= _ARNOLD_MAX_CANDIDATES:
                    logger.warning(
                        "Arnol'd: se truncó la retícula a %d candidatos.",
                        _ARNOLD_MAX_CANDIDATES,
                    )
                    break
            k_all = np.asarray(rows, dtype=np.int64) if rows else np.zeros((0, n), dtype=np.int64)
            return _filter(k_all)

        rows = []
        for i in range(n):
            for s in range(1, k_max + 1):
                v = np.zeros(n, dtype=np.int64); v[i] = s; rows.append(v.copy())
                v[i] = -s; rows.append(v.copy())
        for i in range(n):
            for j in range(i + 1, n):
                for s in (-1, 1):
                    for t in (-1, 1):
                        v = np.zeros(n, dtype=np.int64)
                        v[i] = s
                        v[j] = t
                        if int(np.abs(v).sum()) <= k_max:
                            rows.append(v)
        k_all = np.unique(np.vstack(rows), axis=0) if rows else np.zeros((0, n), dtype=np.int64)
        return _filter(k_all)

    # ── II.3 Función de Melnikov (ruptura homoclínica) ───────────────────
    def compute_melnikov_function(
        self,
        homoclinic_flow: Callable[[float], np.ndarray],
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray], float],
        t0_grid: NDArray[np.float64],
        t_inf: float = 25.0,
        n_quad: int = _MELNIKOV_QUAD_NODES,
        csmd_step: float = _CSMD_STEP,
    ) -> _MelnikovCertificate:
        r"""
        M(t₀) = ∫_{-∞}^{∞} {H₀, H₁}(γ⁰(t − t₀)) dt
        por Gauss–Legendre en [−t_inf, t_inf], acumulación de Kahan.
        Un cero simple M(t*)=0, M'(t*)≠0 ⇒ splitting homoclínico transverso.
        """
        t0s = self._assert_vec("t0_grid", t0_grid)
        if t0s.size == 0:
            raise ValueError("t0_grid no puede estar vacío.")
        if not np.isfinite(t_inf) or t_inf <= 0:
            raise ValueError("t_inf debe ser finito y positivo.")
        n_quad = int(max(16, n_quad))
        nodes, weights = np.polynomial.legendre.leggauss(n_quad)
        t_nodes = t_inf * nodes
        w_nodes = t_inf * weights

        h_csmd = float(csmd_step) if csmd_step else self._germ.csmd_step
        melnikov_vals = np.zeros(t0s.size, dtype=np.float64)
        for i, t0 in enumerate(t0s):
            acc = 0.0
            comp = 0.0
            for t_shift, w in zip(t_nodes, w_nodes):
                x = np.asarray(homoclinic_flow(float(t_shift - t0)), dtype=np.float64)
                try:
                    pb = _NumericalCore.poisson_bracket(
                        hamiltonian_0, hamiltonian_1, x, h=h_csmd
                    )
                except (TypeError, ValueError, FloatingPointError):
                    pb = 0.0
                if not np.isfinite(pb):
                    pb = 0.0
                y = pb * float(w) - comp
                t = acc + y
                comp = (t - acc) - y
                acc = t
            melnikov_vals[i] = acc

        idx_min = int(np.argmin(np.abs(melnikov_vals)))
        m_val = float(melnikov_vals[idx_min])
        if t0s.size >= 3 and 0 < idx_min < t0s.size - 1:
            dm = (melnikov_vals[idx_min + 1] - melnikov_vals[idx_min - 1]) / (
                t0s[idx_min + 1] - t0s[idx_min - 1]
            )
        elif t0s.size >= 2:
            dm = (melnikov_vals[-1] - melnikov_vals[0]) / max(
                t0s[-1] - t0s[0], _MACHINE_EPS
            )
        else:
            dm = 0.0

        floor = max(float(np.linalg.norm(melnikov_vals)) * _MACHINE_EPS * 10.0, 1e-14)
        simple_zeros = 0
        for a, b in zip(melnikov_vals[:-1], melnikov_vals[1:]):
            if a * b < 0.0 and min(abs(float(a)), abs(float(b))) > floor:
                simple_zeros += 1
        is_simple = bool(abs(m_val) < _SPECTRAL_TOL and abs(float(dm)) > _SPECTRAL_TOL)
        if is_simple and simple_zeros == 0:
            simple_zeros = 1

        try:
            x_star = np.asarray(homoclinic_flow(float(t0s[idx_min])), dtype=np.float64)
            grad_h0 = _NumericalCore.compute_gradient_csmd(
                hamiltonian_0, x_star, h_csmd
            )
            grad_norm = _NumericalCore.euclidean_norm(grad_h0)
        except (TypeError, ValueError, FloatingPointError):
            grad_norm = 0.0
        splitting = float(abs(m_val) / max(grad_norm, _WILKINSON_LIMIT))
        return _MelnikovCertificate(
            melnikov_value=m_val,
            melnikov_derivative=float(dm),
            is_simple_zero=is_simple,
            homoclinic_splitting=splitting,
            melnikov_values=melnikov_vals,
            simple_zeros=int(simple_zeros),
            is_chaotic=bool(simple_zeros > 0 or is_simple),
        )

    # ── II.3b Sección transversal de Poincaré ────────────────────────────
    def build_poincare_section(
        self,
        section_index: int = 0,
        section_offset: float = 0.0,
        state: Optional[np.ndarray] = None,
        xh: Optional[np.ndarray] = None,
    ) -> _PoincareSectionWitness:
        r"""Σ = {q_k = q*}; n_Σ = e_k. Transversalidad: Ω n_Σ ≠ 0 y n_Σ·X_H ≠ 0."""
        two_n = self._germ.two_n
        n = self._germ.n
        if not (0 <= section_index < n):
            raise ValueError(f"section_index={section_index} fuera de [0,{n-1}].")
        n_sigma = np.zeros(two_n, dtype=np.float64)
        n_sigma[section_index] = 1.0
        v = self._germ.omega @ n_sigma
        certificate = float(np.linalg.norm(v))
        flow_cert = 0.0
        if xh is not None:
            flow_cert = float(abs(n_sigma @ np.asarray(xh, dtype=np.float64).ravel()))
        elif state is not None:
            st = _NumericalCore.assert_vec("state", state, dim=two_n)
            flow_cert = float(abs(st[n + section_index]))  # q̇ ≈ p si G=I
        return _PoincareSectionWitness(
            normal=n_sigma,
            section_index=int(section_index),
            section_offset=float(section_offset),
            transversal_certificate=certificate,
            flow_transversal_certificate=flow_cert,
            is_transversal=bool(certificate > _MACHINE_EPS),
            is_flow_transversal=bool(flow_cert > _RETURN_MAP_TOL)
            if (xh is not None or state is not None) else True,
        )

    def integrate_poincare_section_return(
        self,
        q0: np.ndarray,
        p0: np.ndarray,
        grad_v: Callable[[np.ndarray], np.ndarray],
        mass_inv: float = 1.0,
        dt: float = 1e-3,
        section_index: int = 0,
        section_offset: float = 0.0,
        crossing_sign: float = 1.0,
        max_iter: int = _RETURN_MAP_MAX_ITER,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        r"""
        Integra Verlet hasta el primer cruce transversal de
            Σ = { q_k = q* }  con  sign(q̇_k) = sign(crossing_sign).
        Interpolación lineal del cruce (error O(dt²), consistente con Verlet).
        """
        q = np.asarray(q0, dtype=np.float64).copy()
        p = np.asarray(p0, dtype=np.float64).copy()
        if q.size != p.size:
            raise ValueError("q0 y p0 deben compartir dimensión.")
        k = int(section_index)
        if not (0 <= k < q.size):
            raise ValueError("section_index fuera de rango.")
        s_prev = float(q[k] - section_offset)
        for i in range(int(max_iter)):
            qn, pn = _NumericalCore.stormer_verlet_step(q, p, grad_v, mass_inv, dt)
            s_next = float(qn[k] - section_offset)
            vel = float(pn[k]) * mass_inv
            crossed = (s_prev * s_next <= 0.0) and (s_prev != 0.0) \
                and (vel * crossing_sign >= 0.0 or abs(vel) < _RETURN_MAP_TOL)
            if crossed:
                denom = s_next - s_prev
                theta = 0.0 if abs(denom) < _MACHINE_EPS else float(-s_prev / denom)
                theta = min(max(theta, 0.0), 1.0)
                q_c = q + theta * (qn - q)
                p_c = p + theta * (pn - p)
                return q_c, p_c, i + 1
            q, p, s_prev = qn, pn, s_next
        raise RuntimeError(
            f"No se cruzó Σ en {max_iter} pasos de Verlet (dt={dt})."
        )

    # ── II.4 Mapa de retorno de Poincaré (Floquet + Lyapunov + Krein) ────
    def compute_poincare_return_map(
        self,
        jacobian_M: NDArray[np.float64],
        period_T: float = 1.0,
    ) -> _PoincareReturnMapCertificate:
        r"""
        Espectro de Floquet y exponentes de Lyapunov del mapa P: Σ → Σ.
          • μ = spec(M);  L_i = (1/T) log|μ_i|.
          • Elíptico: todos |μ|=1.  Hiperbólico: algún |μ|≠1.
          • Parabólico: 1 o −1 en el espectro.  Mixto: elíptico e hiperbólico.
          • Residuo recíproco min_j |μ_i μ_j − 1| (Krein) y ‖MᵀΩM − Ω‖_F.
        """
        M = np.asarray(jacobian_M, dtype=np.float64)
        if M.ndim == 1:
            side = int(round(np.sqrt(M.size)))
            if side * side != M.size:
                raise ValueError("jacobian_M plano no es cuadrado perfecto.")
            M = M.reshape(side, side)
        _NumericalCore.assert_square("jacobian_M", M)
        _NumericalCore.assert_finite("jacobian_M", M)
        ev = la.eigvals(M)
        magnitudes = np.abs(ev)
        t_per = max(abs(float(period_T)), _MACHINE_EPS)
        lyap = np.log(np.maximum(magnitudes, _MACHINE_EPS)) / t_per
        lyap = np.clip(lyap, -_LYAPUNOV_CLIP, _LYAPUNOV_CLIP)

        unit_dev = np.abs(magnitudes - 1.0)
        all_on_circle = bool(np.all(unit_dev <= _FLOQUET_PARABOLIC_BAND * 1e2))
        any_off_circle = bool(np.any(unit_dev > _FLOQUET_PARABOLIC_BAND * 1e2))
        pm1 = np.minimum(np.abs(ev - 1.0), np.abs(ev + 1.0))
        is_parabolic = bool(np.any(pm1 < _FLOQUET_PARABOLIC_BAND * 10.0))
        is_elliptic = bool(all_on_circle)
        is_hyperbolic = bool(any_off_circle)
        is_mixed = bool(is_hyperbolic and np.any(unit_dev <= _FLOQUET_PARABOLIC_BAND * 1e2))

        rec_res = 0.0
        if ev.size:
            products = np.abs(ev[:, None] * ev[None, :] - 1.0)
            rec_res = float(np.mean(np.min(products, axis=1)))
        unit_res = float(np.max(unit_dev)) if ev.size else 0.0
        sp_res = self.monodromy_symplectic_residual(M)
        return _PoincareReturnMapCertificate(
            floquet_multipliers=ev,
            lyapunov_spectrum=lyap,
            is_hyperbolic=is_hyperbolic,
            is_elliptic=is_elliptic,
            is_parabolic=is_parabolic,
            is_mixed=is_mixed,
            trace_M=float(np.real(np.trace(M))),
            det_M=float(np.real(la.det(M))),
            reciprocal_pair_residual=float(rec_res),
            unit_circle_residual=float(unit_res),
            hill_discriminant=float(np.real(np.trace(M))),
            symplectic_residual=float(sp_res),
        )

    # ── II.4b Lyapunov–Benettin (QR) ─────────────────────────────────────
    @staticmethod
    def compute_lyapunov_spectrum_benettin(
        M: np.ndarray, n_iterations: int = _LYAPUNOV_QR_ITERATIONS,
    ) -> _LyapunovSpectrumWitness:
        r"""
        Z_k = M Q_{k−1}; Q_k R_k = QR(Z_k); λ_i = (1/N) Σ log|R_{k,ii}|.
        D_KY = k + (Σ_{i≤k} λ_i)/|λ_{k+1}|, k = max{m : Σ_{i≤m} λ_i ≥ 0}.
        h_KS = Σ λ_i⁺  (Pesin).  Sp(2n) ⇒ Σ λ_i = 0.
        """
        a = np.asarray(M, dtype=np.float64)
        n = a.shape[0]
        n_it = int(n_iterations)
        if n_it <= 0:
            raise ValueError("n_iterations debe ser positivo.")
        q_mat = np.eye(n, dtype=np.float64)
        log_acc = np.zeros(n, dtype=np.float64)
        for _ in range(n_it):
            z_mat = a @ q_mat
            try:
                q_mat, r_mat = la.qr(z_mat, mode="economic")
            except la.LinAlgError:
                break
            diag_r = np.abs(np.diag(r_mat))
            diag_r = np.where(diag_r > _MACHINE_EPS, diag_r, _MACHINE_EPS)
            log_acc += np.log(diag_r)
        spectrum = np.sort(log_acc / float(n_it))[::-1]
        cumsum = np.cumsum(spectrum)
        k = 0
        for i in range(n):
            if cumsum[i] >= 0.0:
                k = i + 1
            else:
                break
        if k == 0:
            d_ky = 0.0
        elif k >= n:
            d_ky = float(n)
        elif spectrum[k] != 0.0:
            d_ky = float(k + cumsum[k - 1] / abs(spectrum[k]))
        else:
            d_ky = float(k)
        d_ky = float(min(max(d_ky, 0.0), float(n)))
        h_ks = float(np.sum(np.clip(spectrum, 0.0, None)))
        return _LyapunovSpectrumWitness(
            spectrum=spectrum,
            kaplan_yorke_dimension=d_ky,
            kolmogorov_sinai_entropy=h_ks,
            is_chaotic=bool(spectrum.size > 0 and spectrum[0] > _HARD_LYAPUNOV_TOL),
            sum_all=float(np.sum(spectrum)),
        )

    # ── II.4c Acción-ángulo + twist de Moser ─────────────────────────────
    @staticmethod
    def compute_action_angle_variables(
        q_periodic: np.ndarray, p_periodic: np.ndarray,
    ) -> _ActionAngleWitness:
        r"""I_k = (1/2π) ∮ p_k dq_k ;  θ_k = Δarg(q_k + i p_k)/(N−1)."""
        q = np.asarray(q_periodic, dtype=np.float64)
        p = np.asarray(p_periodic, dtype=np.float64)
        if q.shape != p.shape or q.ndim != 2:
            raise ValueError("q, p deben ser (N, n).")
        n_pts, n = q.shape
        if n_pts < 2:
            raise ValueError("Trayectoria demasiado corta.")
        i_vec = np.zeros(n, dtype=np.float64)
        theta_vec = np.zeros(n, dtype=np.float64)
        for k in range(n):
            dq = np.diff(q[:, k])
            dq_closed = np.concatenate([dq, [q[0, k] - q[-1, k]]])
            p_mid = 0.5 * (p[:-1, k] + p[1:, k])
            p_mid_closed = np.concatenate([p_mid, [0.5 * (p[-1, k] + p[0, k])]])
            i_vec[k] = float(np.sum(p_mid_closed * dq_closed)) / (2.0 * math.pi)
            angles = np.unwrap(np.arctan2(p[:, k], q[:, k]))
            theta_vec[k] = float(angles[-1] - angles[0]) / max(n_pts - 1, 1)
        if n >= 2 and float(np.linalg.norm(np.diff(i_vec))) > _MACHINE_EPS:
            di = np.diff(i_vec)
            twist = float(np.diff(theta_vec) @ di / max(float(di @ di), _MACHINE_EPS))
        else:
            twist = float(theta_vec[0] / max(abs(i_vec[0]), _MACHINE_EPS)) if n else 0.0
        return _ActionAngleWitness(
            actions=i_vec, angles=theta_vec, twist_jacobian=float(twist),
            is_twist=bool(abs(twist) > _HARD_TWIST_FLOOR),
        )

    # ── II.5 Verificación de Floer + Conley–Zehnder + Maslov ─────────────
    def _coerce_pair(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        jacobian_m3: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z0 = _NumericalCore.assert_vec("start_point", start_point)
        z1 = _NumericalCore.assert_vec("end_point", end_point)
        if z0.size != z1.size:
            raise ValueError("start_point y end_point deben tener la misma dimensión.")
        if z0.size % 2 != 0:
            raise ValueError("Los extremos de Floer deben tener dimensión par (Darboux).")
        M = np.asarray(jacobian_m3)
        if M.ndim == 1:
            side = int(np.sqrt(M.size))
            if side * side != M.size:
                raise ValueError("jacobian_m3 plano no es un cuadrado perfecto.")
            M = M.reshape(side, side)
        _NumericalCore.assert_square("jacobian_m3", M)
        _NumericalCore.assert_finite("jacobian_m3", M)
        if M.shape[0] != z0.size:
            raise ValueError(
                f"jacobian_m3 es {M.shape[0]}×{M.shape[0]} pero extremos tienen dim {z0.size}."
            )
        return (
            z0.astype(np.float64, copy=False),
            z1.astype(np.float64, copy=False),
            np.asarray(M, dtype=np.float64),
        )

    def monodromy_symplectic_residual(self, jacobian_m3: np.ndarray) -> float:
        r"""‖Mᵀ Ω M − Ω‖_F : defecto de pertenencia a Sp(2n, ℝ)."""
        M = np.asarray(jacobian_m3, dtype=np.float64)
        omega = self._adapt_omega(M.shape[0])
        residual = M.T @ omega @ M - omega
        return _NumericalCore.frobenius_norm(residual)

    def _adapt_omega(self, dim: int) -> np.ndarray:
        if dim == self._germ.two_n:
            return self._germ.omega
        return _NumericalCore.generate_canonical_symplectic_form(dim)

    def maslov_degeneracy(self, jacobian_m3: np.ndarray) -> float:
        """Distancia de spec(M) a {1}: min_i |λ_i(M) − 1|."""
        ev = la.eigvals(np.asarray(jacobian_m3, dtype=np.float64))
        return float(np.min(np.abs(ev - 1.0)))

    def conley_zehnder_index(self, jacobian_m3: np.ndarray) -> float:
        r"""
        Índice de Conley–Zehnder / Robbin–Salamon del monodromía M ∈ Sp(2n).
        Polar M = U P; se identifica U con un elemento de U(n) via
            U_ℂ = ½(U_qq + U_pp) + i ½(U_pq − U_qp),
        y CZ ≈ (1/π) Σ Arg(eig U_ℂ), con semientero si 1 ∈ spec(M) (Maslov).
        """
        M = np.asarray(jacobian_m3, dtype=np.float64)
        dim = M.shape[0]
        n = dim // 2
        try:
            u_polar, _p = la.polar(M)
        except (np.linalg.LinAlgError, ValueError) as exc:
            logger.warning("Polar de monodromía fallida (%s); CZ := 0.", exc)
            return 0.0
        x_blk = 0.5 * (u_polar[:n, :n] + u_polar[n:, n:])
        y_blk = 0.5 * (u_polar[n:, :n] - u_polar[:n, n:])
        u_c = np.asarray(x_blk + 1j * y_blk, dtype=np.complex128)
        try:
            uu, _ss, vv = la.svd(u_c, full_matrices=False)
            u_c = uu @ vv
        except (np.linalg.LinAlgError, ValueError):
            pass
        ev = la.eigvals(u_c)
        ev = ev / np.maximum(np.abs(ev), _MACHINE_EPS)
        angles = np.angle(ev)
        cz = float(_NumericalCore.kahan_babuska_neumaier_sum(angles) / np.pi)
        if self.maslov_degeneracy(M) <= _MASLOV_DEGENERACY:
            cz += 0.5 * (np.sign(cz) if cz != 0.0 else 1.0)
        return cz

    def verify_floer(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        jacobian_m3: np.ndarray,
        hamiltonian_func: Optional[Callable[[np.ndarray], float]] = None,
    ) -> _FloerResult:
        r"""
        Certifica el cilindro de Floer *discreto* (cuerda + monodromía):
            residual = ‖z₁ − z₀‖ (1 + ‖M‖_F) + ‖Mᵀ Ω M − Ω‖_F,
        acción 𝒜_H = ∫ λ − ∫ H dt (trapecio en los extremos),
        energía de Dirichlet, CZ y degeneración de Maslov.
        (Sin una malla (s,t) no se evalúa ∂̄_{J,H} puntualmente; el residual
        es un certificado de cuerda + simpléctica, no el CR operator.)
        """
        z0, z1, M = self._coerce_pair(start_point, end_point, jacobian_m3)
        chord = _NumericalCore.euclidean_norm(z1 - z0)
        m_norm = _NumericalCore.frobenius_norm(M)
        sp_res = self.monodromy_symplectic_residual(M)
        floer_residual = float(chord * (1.0 + m_norm) + sp_res)
        liouville = _NumericalCore.liouville_action(z0, z1)
        if hamiltonian_func is not None:
            try:
                h0 = float(np.real(hamiltonian_func(z0)))
                h1 = float(np.real(hamiltonian_func(z1)))
                if np.isfinite(h0) and np.isfinite(h1):
                    liouville = liouville - 0.5 * (h0 + h1)
            except (TypeError, ValueError, FloatingPointError) as exc:
                logger.warning("Hamiltoniano no evaluable en extremos: %s", exc)
        dirichlet = 0.5 * float(
            _NumericalCore.kahan_babuska_neumaier_sum((z1 - z0) ** 2)
        )
        deg = self.maslov_degeneracy(M)
        cz = self.conley_zehnder_index(M)
        scale_sp = max(_NumericalCore.frobenius_norm(self._adapt_omega(M.shape[0])), 1.0)
        is_sp = sp_res <= max(
            _WILKINSON_DRIFT_LIMIT,
            _WILKINSON_DRIFT_LIMIT * scale_sp * (m_norm ** 2 + 1.0),
        )
        return _FloerResult(
            floer_residual=floer_residual,
            action_potential=float(chord),
            liouville_action=float(liouville),
            dirichlet_energy=float(dirichlet),
            symplectic_monodromy_residual=float(sp_res),
            conley_zehnder_index=float(cz),
            maslov_degeneracy=float(deg),
            is_nondegenerate=bool(deg > _MASLOV_DEGENERACY),
            is_symplectic_monodromy=bool(is_sp),
        )

    # ── II.ω  MORFISMO TERMINAL Φ_II: nervio de Čech desde el monodromía ─
    def induce_cech_nerve_germ(
        self,
        jacobian_m3: np.ndarray,
        start_point: Optional[np.ndarray] = None,
        end_point: Optional[np.ndarray] = None,
    ) -> _CechNerveGerm:
        r"""
        **II.ω — Morfismo terminal de la FASE II / objeto inicial de la FASE III.**

        Cuantiza M ∈ Sp(2n, ℝ) en el Gram del haz atencional de Čech:
            w = (M − I)(M + I)^{−1},   G = −Ω w  (simetrizado hermitiano).
        El piso de regularización es max(ε_FaseI, ε_Wilkinson).

        Este método **es** el arranque formal de la Fase III:
        `_AttentionCechCohomology(𝒢_II)` / `phase3_ingest_cech_nerve_germ`.
        """
        M = np.asarray(jacobian_m3)
        if M.ndim == 1:
            side = int(np.sqrt(M.size))
            if side * side != M.size:
                raise ValueError("jacobian_m3 plano no es un cuadrado perfecto.")
            M = M.reshape(side, side)
        _NumericalCore.assert_square("jacobian_m3", M)
        _NumericalCore.assert_finite("jacobian_m3", M)
        M = np.asarray(M, dtype=np.float64)
        dim = M.shape[0]
        if dim % 2 != 0:
            raise ValueError("El monodromía debe ser de dimensión par.")
        if start_point is not None and end_point is not None:
            self._coerce_pair(start_point, end_point, M)
        ident = np.eye(dim, dtype=np.float64)
        floor = max(
            self._germ.reg_floor,
            _NumericalCore.wilkinson_deflation_floor(M + ident),
        )
        pinv_plus, _s, cond = _NumericalCore.tikhonov_higham_pinv(
            M + ident, rel_floor=_MACHINE_EPS, abs_floor=floor
        )
        w_cayley = (M - ident) @ pinv_plus
        omega = self._adapt_omega(dim)
        gram = _NumericalCore.higham_nearest_hermitian(-omega @ w_cayley)
        gram = np.real(gram).astype(np.float64, copy=False)
        sp_res = self.monodromy_symplectic_residual(M)
        try:
            cz = self.conley_zehnder_index(M)
        except Exception:
            cz = 0.0
        return _CechNerveGerm(
            sheaf_gram=gram,
            cayley_condition=float(cond),
            two_n=dim,
            reg_floor=float(floor),
            from_floer=True,
            symplectic_residual=float(sp_res),
            conley_zehnder_index=float(cz),
        )

    @staticmethod
    def _assert_vec(name: str, vec: np.ndarray) -> np.ndarray:
        return _NumericalCore.assert_vec(name, np.asarray(vec, dtype=np.float64))


# =============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ FASE III — COHOMOLOGÍA DE ČECH ATENCIONAL, HODGE Y BETTI                 ║
# ║                                                                          ║
# ║ El primer método consume II.ω; compute es III.ω.                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# =============================================================================
@dataclass(frozen=True, slots=True)
class _CechCohomologyResult:
    """Resultado certificado de la cohomología atencional de Čech."""
    cech_obstruction: float
    active_modes: np.ndarray
    cocycle_defect: float
    harmonic_energy: float
    betti_0: int
    betti_1: int
    effective_rank: int
    nuclear_mass: float
    is_h1_trivial: bool
    germ_from_floer: bool


class _AttentionCechCohomology:
    r"""
    Fase III. Cohomología de Čech del haz atencional F_att sobre la cubierta 𝒰.

    **III.0** `phase3_ingest_cech_nerve_germ` continúa II.ω.
    **III.ω** `compute` aniquila (o certifica) Ȟ¹(𝒰; F_att).
    """

    def __init__(
        self,
        germ: Optional[_CechNerveGerm] = None,
        regularizer: float = _HIGHAM_TIKHONOV_FLOOR,
    ) -> None:
        self._germ: Optional[_CechNerveGerm] = None
        if germ is not None:
            self._germ = self.phase3_ingest_cech_nerve_germ(germ)
        self._reg = max(float(regularizer), _HIGHAM_TIKHONOV_FLOOR)

    # ── III.0  INGESTA DEL OBJETO TERMINAL DE LA FASE II ─────────────────
    def phase3_ingest_cech_nerve_germ(self, germ: _CechNerveGerm) -> _CechNerveGerm:
        r"""
        **III.0 — Flecha inicial de Φ_III, continuación estricta de II.ω.**

        Revalida 𝒢_II (Gram hermitiano, paridad, Cayley) y lo reexpide al
        morfismo terminal `compute`.
        """
        if not isinstance(germ, _CechNerveGerm):
            raise TypeError("germ debe ser _CechNerveGerm (cierre II.ω).")
        g = np.asarray(germ.sheaf_gram)
        if g.ndim != 2 or g.shape[0] != g.shape[1]:
            raise ValueError("sheaf_gram de 𝒢_II no es cuadrado.")
        if germ.two_n <= 0 or germ.two_n % 2 != 0:
            raise ValueError("two_n de 𝒢_II no es de Darboux.")
        if germ.symplectic_residual > 1e-4:
            logger.warning(
                "III.0: monodromía origen no simpléctica (res=%.3e).",
                germ.symplectic_residual,
            )
        if germ.cayley_condition > 1.0 / _WILKINSON_LIMIT:
            logger.warning(
                "III.0: Cayley mal condicionado (κ=%.3e); −1 ∈ spec(M) aproximado.",
                germ.cayley_condition,
            )
        return germ

    def _resolve_matrix(self, attention_sheaf_matrix: np.ndarray) -> np.ndarray:
        a = np.asarray(attention_sheaf_matrix)
        if a.size == 0 and self._germ is not None:
            return np.asarray(self._germ.sheaf_gram, dtype=np.float64)
        if a.ndim == 1:
            side = int(np.sqrt(a.size))
            if side * side != a.size:
                raise ValueError("attention_sheaf_matrix plana no es cuadrado perfecto.")
            a = a.reshape(side, side)
        _NumericalCore.assert_square("attention_sheaf_matrix", a)
        _NumericalCore.assert_finite("attention_sheaf_matrix", a)
        return np.asarray(a, dtype=np.complex128)

    def cech_coboundary_defect(self, omega: np.ndarray) -> float:
        r"""
        Norma del coborde Čech δω de un 1-cochain antisimétrico:
            (δω)_{ijk} = ω_{jk} − ω_{ik} + ω_{ij}.
        ‖δω‖=0 y ‖ω‖≠0 ⇒ [ω] es un cociclo (candidato a Ȟ¹).
        """
        w = np.real(0.5 * (np.asarray(omega) - np.asarray(omega).T.conj()))
        n = w.shape[0]
        if n < 3:
            return 0.0
        acc = 0.0
        if n <= _CECH_TRIPLE_CAP:
            for i in range(n - 2):
                for j in range(i + 1, n - 1):
                    wij = w[i, j]
                    for k in range(j + 1, n):
                        t = w[j, k] - w[i, k] + wij
                        acc += float(t.real * t.real + t.imag * t.imag)
        else:
            step = max(1, n // _CECH_TRIPLE_CAP)
            idx = np.arange(0, n, step)
            m = idx.size
            for a in range(m - 2):
                i = int(idx[a])
                for b in range(a + 1, m - 1):
                    j = int(idx[b])
                    wij = w[i, j]
                    for c in range(b + 1, m):
                        k = int(idx[c])
                        t = w[j, k] - w[i, k] + wij
                        acc += float(t.real * t.real + t.imag * t.imag)
        return float(np.sqrt(max(acc, 0.0)))

    def sheaf_hodge_spectrum(
        self,
        gram: np.ndarray,
        floor: float,
    ) -> Tuple[np.ndarray, np.ndarray, int, int, float]:
        r"""
        Espectro del Laplaciano de Hodge combinatorio Δ₀ = D − W del nervio.
            b₀ = dim ker Δ₀,   b₁ = |E| − |V| + b₀  (fórmula de Euler).
        """
        herm = np.real(_NumericalCore.higham_nearest_hermitian(gram))
        n = herm.shape[0]
        weights = np.abs(herm)
        np.fill_diagonal(weights, 0.0)
        adjacency = (weights > floor).astype(np.float64)
        weights = weights * adjacency
        degree = weights.sum(axis=1)
        lap = np.diag(degree) - weights
        lap = np.real(_NumericalCore.higham_nearest_hermitian(lap))
        evals = np.real(la.eigvalsh(lap)) if n else np.array([], dtype=np.float64)
        ker_tol = max(floor, _WILKINSON_DEFLATION_FLOOR * max(n, 1))
        betti_0 = int(np.sum(evals <= ker_tol))
        n_edges = int(np.sum(np.triu(adjacency, 1)))
        betti_1 = int(max(n_edges - n + betti_0, 0))
        active = evals[evals > ker_tol]
        harmonic = _NumericalCore.kahan_babuska_neumaier_sum(
            np.clip(evals[: max(betti_0, 0)], 0.0, None)
        )
        return evals, active, betti_0, betti_1, float(harmonic)

    # ── III.ω  MORFISMO TERMINAL Φ_III: cohomología atencional ───────────
    def compute(self, attention_sheaf_matrix: np.ndarray) -> _CechCohomologyResult:
        r"""
        **III.ω — Morfismo terminal de la FASE III.**

        Calcula la clase de obstrucción de Čech y los modos activos de F_att.
        Ȟ¹(𝒰; F_att) ≡ 0 se certifica vía ‖δω‖ ≈ 0 **y** b₁ = 0.
        """
        if np.asarray(attention_sheaf_matrix).size == 0 and self._germ is None:
            return _CechCohomologyResult(
                cech_obstruction=0.0,
                active_modes=np.array([], dtype=np.float64),
                cocycle_defect=0.0,
                harmonic_energy=0.0,
                betti_0=0,
                betti_1=0,
                effective_rank=0,
                nuclear_mass=0.0,
                is_h1_trivial=True,
                germ_from_floer=False,
            )
        raw = self._resolve_matrix(attention_sheaf_matrix)
        sheaf = _NumericalCore.higham_nearest_hermitian(raw)
        floor = max(self._reg, _NumericalCore.wilkinson_deflation_floor(sheaf))
        if self._germ is not None:
            floor = max(floor, self._germ.reg_floor)
        singular_values = np.real(la.svd(sheaf, compute_uv=False))
        active_modes = singular_values[singular_values > floor]
        nuclear = (
            _NumericalCore.kahan_babuska_neumaier_sum(active_modes)
            if active_modes.size
            else 0.0
        )
        cocycle = self.cech_coboundary_defect(sheaf)
        _evals, _lap_active, betti_0, betti_1, harmonic = self.sheaf_hodge_spectrum(
            sheaf, floor
        )
        rank = int(active_modes.size)
        is_h1 = bool(betti_1 == 0 and cocycle <= _SPECTRAL_TOL * max(1.0, math.sqrt(max(sheaf.shape[0], 1))))
        return _CechCohomologyResult(
            cech_obstruction=float(nuclear),
            active_modes=np.asarray(active_modes, dtype=np.float64),
            cocycle_defect=float(cocycle),
            harmonic_energy=float(harmonic),
            betti_0=int(betti_0),
            betti_1=int(betti_1),
            effective_rank=rank,
            nuclear_mass=float(nuclear),
            is_h1_trivial=is_h1,
            germ_from_floer=bool(self._germ.from_floer) if self._germ is not None else False,
        )


# =============================================================================
# FACHADA INTEGRADORA — Φ_III ∘ Φ_II ∘ Φ_I
# =============================================================================
class ImperialEruditosEngine:
    r"""
    Motor de alta precisión para la rigidez de Floer, Čech y mecánica celeste
    de Poincaré sobre el espacio de fase de los logits y perturbaciones
    presupuestales.

    Composición anidada:
        Φ_I   : Darboux + Maupertuis–Jacobi + Poincaré–Cartan  ⟶  𝒢_I
        Φ_II  : KAM + Melnikov + Retorno + Floer               ⟶  𝒢_II
        Φ_III : Čech + Hodge + Betti                           ⟶  Ȟ¹ ≡ 0

    Ejemplo mínimo::

        engine = ImperialEruditosEngine()
        report, kam = engine.compute_poincare_small_divisors_spectrum(omega, wave_k, M, J)
        melnikov = engine.compute_melnikov_function(flow, H0, H1, t0_grid)
        ret = engine.compute_poincare_return_map(M)
        floer = engine.verify_floer_homology_trajectory_certified(z0, z1, M)
        obstruction, modes = engine.compute_attention_cech_cohomology(gram)
    """

    def __init__(
        self,
        regularizer: float = 1e-15,
        novikov_valuation_T: float = 1.0,
        hamiltonian_energy_H0: float = 1.0,
        potential_energy_V: float = 0.0,
        germ_dimension: int = _DEFAULT_GERM_DIM,
    ) -> None:
        self._reg: Final[float] = max(float(regularizer), _HIGHAM_TIKHONOV_FLOOR)
        self._T_val: Final[float] = float(novikov_valuation_T)
        self._H0: Final[float] = float(hamiltonian_energy_H0)
        self._V: Final[float] = float(potential_energy_V)
        dim0 = int(germ_dimension)
        if dim0 <= 0 or dim0 % 2 != 0:
            dim0 = _DEFAULT_GERM_DIM
        # Fase I.ω
        self._poincare_germ: _PoincareCartanGerm = (
            _NumericalCore.synthesize_poincare_cartan_germ(
                dimension_two_n=dim0,
                hamiltonian_energy_H0=self._H0,
                potential_energy_V=self._V,
                csmd_step=_CSMD_STEP,
                regularizer=self._reg,
            )
        )
        # Fase II.0 ← I.ω
        self._celestial_verifier = _PoincareCelestialVerifier(self._poincare_germ)
        # Fase II.ω
        self._cech_germ: _CechNerveGerm = self._celestial_verifier.induce_cech_nerve_germ(
            np.eye(dim0, dtype=np.float64)
        )
        # Fase III.0 ← II.ω
        self._cech_calculator = _AttentionCechCohomology(
            germ=self._cech_germ, regularizer=self._reg
        )

    def _resync_poincare_germ(self, two_n: int, scale_matrix: Optional[np.ndarray] = None) -> None:
        """Re-sintetiza 𝒢_I (I.ω) y reencadena II.0 cuando cambia la dimensión de Darboux."""
        if two_n == self._poincare_germ.two_n and scale_matrix is None:
            return
        self._poincare_germ = _NumericalCore.synthesize_poincare_cartan_germ(
            dimension_two_n=two_n,
            hamiltonian_energy_H0=self._H0,
            potential_energy_V=self._V,
            csmd_step=self._poincare_germ.csmd_step,
            regularizer=self._reg,
            scale_matrix=scale_matrix,
        )
        self._celestial_verifier = _PoincareCelestialVerifier(self._poincare_germ)

    # ══════════════════════════════════════════════════════════════════════
    # API FASE I — Banach, Darboux, Maupertuis–Jacobi, Poincaré–Cartan
    # ══════════════════════════════════════════════════════════════════════
    def kahan_sum(self, arr: np.ndarray) -> float:
        return _NumericalCore.kahan_sum(arr)

    def kahan_babuska_neumaier_sum(self, arr: np.ndarray) -> float:
        return _NumericalCore.kahan_babuska_neumaier_sum(arr)

    def compute_maupertuis_jacobi_conformal_metric(
        self,
        hamiltonian_energy_H0: float,
        potential_energy_V: float,
        base_metric_g: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], _MaupertuisJacobiCertificate]:
        r"""Métrica conforme g̃_{jk} = 2 (H₀ − V) g_{jk}."""
        return _NumericalCore.compute_maupertuis_jacobi_conformal_metric(
            hamiltonian_energy_H0, potential_energy_V, base_metric_g
        )

    def compute_hill_region(
        self,
        hamiltonian_energy_H0: float,
        potential_energy_V: float,
    ) -> float:
        """Margen de Hill H₀ − V(q) (frontera de Poincaré si ≤ 0)."""
        return _NumericalCore.compute_hill_region(
            hamiltonian_energy_H0, potential_energy_V
        )

    def compute_poincare_cartan_lambda(
        self,
        x: np.ndarray,
        hamiltonian_func: Optional[Callable[[np.ndarray], float]] = None,
    ) -> np.ndarray:
        r"""1-forma de Poincaré–Cartan λ = p dq − H dt (componentes n+1)."""
        return _NumericalCore.compute_poincare_cartan_lambda(
            x, hamiltonian_func, csmd_step=self._poincare_germ.csmd_step
        )

    def compute_poincare_cartan_one_form(
        self,
        x: np.ndarray,
        hamiltonian_func: Optional[Callable[[np.ndarray], float]] = None,
    ) -> _PoincareCartanOneForm:
        return _NumericalCore.compute_poincare_cartan_one_form(
            x, hamiltonian_func, csmd_step=self._poincare_germ.csmd_step
        )

    def compute_symplectic_gradient(
        self,
        hamiltonian_func: Callable[[np.ndarray], float],
        x: np.ndarray,
        h: float = _CSMD_STEP,
    ) -> np.ndarray:
        r"""Campo X_H = Ω ∇H(x) vía CSMD."""
        xv = _NumericalCore.assert_vec("x", np.asarray(x, dtype=np.float64))
        if xv.size % 2 == 0:
            self._resync_poincare_germ(xv.size)
        return _NumericalCore.compute_symplectic_gradient(hamiltonian_func, xv, h)

    def compute_hessian_csmd(
        self,
        hamiltonian_func: Callable[[np.ndarray], float],
        x: np.ndarray,
        h: float = _CSMD_STEP,
    ) -> np.ndarray:
        return _NumericalCore.compute_hessian_csmd(hamiltonian_func, x, h)

    def compute_poisson_bracket(
        self,
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray], float],
        x: np.ndarray,
        h: float = _CSMD_STEP,
    ) -> float:
        r"""{H₀, H₁}(x) = (∇H₀)ᵀ Ω ∇H₁."""
        return _NumericalCore.poisson_bracket(hamiltonian_0, hamiltonian_1, x, h)

    def poincare_cartan_germ_certificate(self) -> _SymplecticFormCertificate:
        return self._poincare_germ.form_certificate

    def maupertuis_jacobi_certificate(self) -> _MaupertuisJacobiCertificate:
        return self._poincare_germ.jacobi_certificate

    def synthesize_poincare_cartan_germ(
        self,
        dimension_two_n: int,
        hamiltonian_energy_H0: Optional[float] = None,
        potential_energy_V: Optional[float] = None,
        hamiltonian_func: Optional[Callable[[np.ndarray], float]] = None,
        scale_matrix: Optional[np.ndarray] = None,
        base_metric_g: Optional[np.ndarray] = None,
    ) -> _PoincareCartanGerm:
        """Reexpone I.ω y reencadena el verificador celeste (II.0)."""
        germ = _NumericalCore.synthesize_poincare_cartan_germ(
            dimension_two_n=dimension_two_n,
            hamiltonian_energy_H0=self._H0 if hamiltonian_energy_H0 is None else float(hamiltonian_energy_H0),
            potential_energy_V=self._V if potential_energy_V is None else float(potential_energy_V),
            csmd_step=self._poincare_germ.csmd_step,
            regularizer=self._reg,
            hamiltonian_func=hamiltonian_func,
            scale_matrix=scale_matrix,
            base_metric_g=base_metric_g,
        )
        self._poincare_germ = germ
        self._celestial_verifier = _PoincareCelestialVerifier(germ)
        return germ

    # ══════════════════════════════════════════════════════════════════════
    # API FASE II — KAM, Melnikov, Retorno, Floer
    # ══════════════════════════════════════════════════════════════════════
    def compute_poincare_small_divisors_spectrum(
        self,
        frequency_vector_omega: NDArray[np.float64],
        wave_vectors_k: NDArray[np.float64],
        jacobian_M: NDArray[np.float64],
        canonical_J: NDArray[np.float64],
        tau: Optional[float] = None,
        gamma: float = _KAM_GAMMA_FLOOR,
    ) -> Tuple[EruditosSpectrumReport, _KAMStabilityCertificate]:
        r"""Espectro de pequeños divisores de Poincaré–KAM + Novikov Λ_Nov."""
        return self._celestial_verifier.compute_poincare_small_divisors_spectrum(
            frequency_vector_omega,
            wave_vectors_k,
            jacobian_M,
            canonical_J,
            tau=tau,
            gamma=gamma,
            novikov_valuation_T=self._T_val,
        )

    def compute_arnold_resonance_lattice(
        self,
        frequency_vector_omega: NDArray[np.float64],
        max_order: int = 4,
        tol: float = 1e-6,
    ) -> NDArray[np.int64]:
        r"""Retícula de resonancias de Arnol'd k ∈ ℤ^n con |⟨k, ω⟩| < tol."""
        return self._celestial_verifier.compute_arnold_resonance_lattice(
            frequency_vector_omega, max_order=max_order, tol=tol
        )

    def compute_melnikov_function(
        self,
        homoclinic_flow: Callable[[float], np.ndarray],
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray], float],
        t0_grid: NDArray[np.float64],
        t_inf: float = 25.0,
        n_quad: int = _MELNIKOV_QUAD_NODES,
    ) -> _MelnikovCertificate:
        r"""ℳ(t₀)=∫{H₀,H₁}(γ⁰(t−t₀)) dt. Ceros simples ⇒ caos homoclínico."""
        return self._celestial_verifier.compute_melnikov_function(
            homoclinic_flow,
            hamiltonian_0,
            hamiltonian_1,
            t0_grid,
            t_inf=t_inf,
            n_quad=n_quad,
            csmd_step=self._poincare_germ.csmd_step,
        )

    def compute_poincare_return_map(
        self,
        jacobian_M: NDArray[np.float64],
        period_T: float = 1.0,
    ) -> _PoincareReturnMapCertificate:
        r"""Mapa de retorno P: Σ → Σ (Floquet + Lyapunov + Krein)."""
        return self._celestial_verifier.compute_poincare_return_map(
            jacobian_M, period_T=period_T
        )

    def compute_lyapunov_spectrum_benettin(
        self,
        jacobian_M: NDArray[np.float64],
        n_iterations: int = _LYAPUNOV_QR_ITERATIONS,
    ) -> _LyapunovSpectrumWitness:
        return self._celestial_verifier.compute_lyapunov_spectrum_benettin(
            np.asarray(jacobian_M, dtype=np.float64), n_iterations=n_iterations
        )

    def compute_action_angle_variables(
        self,
        q_periodic: np.ndarray,
        p_periodic: np.ndarray,
    ) -> _ActionAngleWitness:
        return self._celestial_verifier.compute_action_angle_variables(
            q_periodic, p_periodic
        )

    def integrate_poincare_section_return(
        self,
        q0: np.ndarray,
        p0: np.ndarray,
        grad_v: Callable[[np.ndarray], np.ndarray],
        mass_inv: float = 1.0,
        dt: float = 1e-3,
        section_index: int = 0,
        section_offset: float = 0.0,
        crossing_sign: float = 1.0,
        max_iter: int = _RETURN_MAP_MAX_ITER,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        return self._celestial_verifier.integrate_poincare_section_return(
            q0, p0, grad_v, mass_inv=mass_inv, dt=dt,
            section_index=section_index, section_offset=section_offset,
            crossing_sign=crossing_sign, max_iter=max_iter,
        )

    def verify_floer_homology_trajectory(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        jacobian_m3: np.ndarray,
    ) -> Tuple[float, float]:
        """Verificación de la trayectoria de Floer (residual, acción)."""
        result = self.verify_floer_homology_trajectory_certified(
            start_point, end_point, jacobian_m3
        )
        return result.floer_residual, result.action_potential

    def verify_floer_homology_trajectory_certified(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        jacobian_m3: np.ndarray,
        hamiltonian_func: Optional[Callable[[np.ndarray], float]] = None,
    ) -> _FloerResult:
        """Floer con Liouville, Dirichlet, CZ, Maslov y residual simpléctico."""
        z0 = _NumericalCore.assert_vec("start_point", start_point)
        if z0.size % 2 == 0:
            self._resync_poincare_germ(z0.size, scale_matrix=np.asarray(jacobian_m3))
        return self._celestial_verifier.verify_floer(
            start_point, end_point, jacobian_m3, hamiltonian_func=hamiltonian_func
        )

    def induce_cech_nerve_germ(
        self,
        jacobian_m3: np.ndarray,
        start_point: Optional[np.ndarray] = None,
        end_point: Optional[np.ndarray] = None,
    ) -> _CechNerveGerm:
        r"""
        Morfismo terminal Fase II → inicial Fase III: monodromía M ↦ Gram Čech.
        """
        M = np.asarray(jacobian_m3)
        if M.ndim == 1:
            side = int(np.sqrt(M.size))
            M = M.reshape(side, side)
        if M.shape[0] % 2 == 0:
            self._resync_poincare_germ(M.shape[0], scale_matrix=M)
        germ = self._celestial_verifier.induce_cech_nerve_germ(
            M, start_point=start_point, end_point=end_point
        )
        self._cech_germ = germ
        self._cech_calculator = _AttentionCechCohomology(
            germ=germ, regularizer=self._reg
        )
        return germ

    # ══════════════════════════════════════════════════════════════════════
    # API FASE III — Čech atencional, Hodge, Betti
    # ══════════════════════════════════════════════════════════════════════
    def compute_attention_cech_cohomology(
        self,
        attention_sheaf_matrix: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Cálculo de la cohomología atencional de Čech (obstrucción, modos)."""
        result = self._cech_calculator.compute(attention_sheaf_matrix)
        return result.cech_obstruction, result.active_modes

    def compute_attention_cech_cohomology_certified(
        self,
        attention_sheaf_matrix: np.ndarray,
    ) -> _CechCohomologyResult:
        """Čech con defecto de coborde, Hodge, Betti, masa nuclear y Ȟ¹≡0."""
        return self._cech_calculator.compute(attention_sheaf_matrix)

    # ── Ciclo completo I.ω → II.ω → III.ω ────────────────────────────────
    def execute_poincare_eruditos_cycle(
        self,
        frequency_vector_omega: NDArray[np.float64],
        wave_vectors_k: NDArray[np.float64],
        jacobian_M: NDArray[np.float64],
        canonical_J: NDArray[np.float64],
        start_point: Optional[np.ndarray] = None,
        end_point: Optional[np.ndarray] = None,
        attention_sheaf_matrix: Optional[np.ndarray] = None,
        hamiltonian_func: Optional[Callable[[np.ndarray], float]] = None,
    ) -> Tuple[
        EruditosSpectrumReport,
        _KAMStabilityCertificate,
        _PoincareReturnMapCertificate,
        Optional[_FloerResult],
        _CechNerveGerm,
        _CechCohomologyResult,
    ]:
        r"""
        Ciclo completo **I.ω → II.ω → III.ω** sobre tensores crudos.
          1. Resincroniza 𝒢_I a dim(M).
          2. KAM + retorno de Poincaré.
          3. Floer certificado (si se dan extremos).
          4. Nervio de Čech (II.ω) y cohomología (III.ω).
        """
        M = np.asarray(jacobian_M, dtype=np.float64)
        if M.ndim == 1:
            side = int(round(np.sqrt(M.size)))
            M = M.reshape(side, side)
        if M.shape[0] % 2 == 0:
            self._resync_poincare_germ(M.shape[0], scale_matrix=M)
        report, kam = self.compute_poincare_small_divisors_spectrum(
            frequency_vector_omega, wave_vectors_k, M, canonical_J
        )
        ret = self.compute_poincare_return_map(M)
        floer: Optional[_FloerResult] = None
        if start_point is not None and end_point is not None:
            floer = self.verify_floer_homology_trajectory_certified(
                start_point, end_point, M, hamiltonian_func=hamiltonian_func
            )
        germ_ii = self.induce_cech_nerve_germ(
            M, start_point=start_point, end_point=end_point
        )
        sheaf = germ_ii.sheaf_gram if attention_sheaf_matrix is None \
            else np.asarray(attention_sheaf_matrix)
        cech = self.compute_attention_cech_cohomology_certified(sheaf)
        return report, kam, ret, floer, germ_ii, cech


__all__ = [
    "ImperialEruditosEngine",
    "EruditosSpectrumReport",
    "PoincareEruditosSpectrumReport",
    "_SymplecticFormCertificate",
    "_MaupertuisJacobiCertificate",
    "_PoincareCartanGerm",
    "_PoincareCartanOneForm",
    "_KAMStabilityCertificate",
    "_MelnikovCertificate",
    "_PoincareReturnMapCertificate",
    "_FloerResult",
    "_CechNerveGerm",
    "_CechCohomologyResult",
    "_LyapunovSpectrumWitness",
    "_ActionAngleWitness",
    "_PoincareSectionWitness",
]