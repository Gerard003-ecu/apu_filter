# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Imperial Tesserarios Engine (Caballos de Batalla Homotópicos FPU)   ║
║ Ruta   : app/core/inmune_system/imperial_tesserarios_engine.py               ║
║ Versión: 3.0.0-Nested-Phases-Quillen-Polar-Stasheff-Gerbe-Cech-Kahan         ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS MATEMÁTICA Y METROLOGÍA DE LA FPU
────────────────────────────────────────────────────────────────────────────────
Motor homotópico ciego de Nivel 0.5 (V_TESSERARIOS). Alimenta las aduanas
de-confinadas mediante el morfismo de fases anidadas

    Φ_III ∘ Φ_II ∘ Φ_I :  Banach_Darboux  →  Quillen(Sp(2n), Pol)  →  A∞ × Ȟ_gerbe

  Fase I   Núcleo de Banach, 2-forma de Darboux y Newton estructurado.
           Último morfismo: synthesize_symplectic_quillen_germ.
  Fase II  Proyección polar simpléctica de Higham–Mackey (escalada),
           inversa exacta S⁻¹ = −Ω Sᵀ Ω y factorización de Quillen
           M = P ∘ S. Último morfismo: induce_ainfty_cech_germ.
  Fase III Asociador de Stasheff m₃, residuo pentagonal A∞, jacobiato
           del conmutador y obstrucción de gerbe Čech–Deligne.

Precisión metrológica: Kahan, Kahan–Babuška–Neumaier, Klein; polar
simpléctica con escalado de Frobenius (sin Tikhonov que rompa el grupo);
contracciones tensoriales compensadas; masa nuclear de Wilkinson.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, Optional, Tuple

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Core.ImperialTesserariosEngine")

__version__: Final[str] = (
    "3.0.0-Nested-Phases-Quillen-Polar-Stasheff-Gerbe-Cech-Kahan"
)


# =============================================================================
# CONSTANTES DE PRECISIÓN METROLÓGICA
# =============================================================================
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_REG_FLOOR_TIKHONOV: Final[float] = 1e-15
_WILKINSON_DEFLATION_SCALE: Final[float] = 10.0
_WILKINSON_DEFLATION_FLOOR: Final[float] = 1e-12
_WILKINSON_DRIFT_LIMIT: Final[float] = 1e-9
_DEFAULT_MAX_ITER: Final[int] = 100
_DEFAULT_TOL: Final[float] = 1e-12
_STASHEFF_PENTAGON_CAP: Final[int] = 8
_CECH_TRIPLE_CAP: Final[int] = 80
_CECH_QUAD_CAP: Final[int] = 24
_MU_CLIP: Final[Tuple[float, float]] = (1e-8, 1e8)


# =============================================================================
# FASE I — NÚCLEO DE BANACH, DARBOUX Y NEWTON ESTRUCTURADO
# -----------------------------------------------------------------------------
# Objetos: sumas compensadas, 2-forma canónica Ω, Higham, pinv de Tikhonov,
#          imagen de la involución de Cartan Ω (·)^{-T} Ωᵀ.
# Morfismo terminal (I.8): synthesize_symplectic_quillen_germ
#          ≅ objeto inicial de la Fase II (polar / Quillen).
# =============================================================================
@dataclass(frozen=True)
class _SymplecticFormCertificate:
    """Certificado algebraico de la 2-forma canónica de Liouville–Darboux."""

    skew_residual: float
    almost_complex_residual: float
    determinant: float
    frobenius_norm: float
    is_darboux: bool


@dataclass(frozen=True)
class _SymplecticQuillenGerm:
    """
    Gérmen de Quillen–Darboux (objeto terminal de la Fase I).

    Es el objeto inicial de la Fase II: transporta la geometría de Darboux
    (Ω, dim, piso, Newton) sobre la cual se instancia la iteración polar

        X_{k+1} = ½ ( μ_k X_k  +  μ_k^{-1} Ω X_k^{-T} Ωᵀ )

    y la factorización de Quillen M = P ∘ S, S ∈ Sp(2n, ℝ).

    Atributos
    ---------
    two_n:
        Dimensión de T*Q ≅ ℝ^{2n} (par).
    n:
        Dimensión del espacio de configuración Q.
    omega:
        2-forma canónica Ω = [0 I; −I 0].
    reg_floor:
        Piso de Tikhonov–Wilkinson (sólo fallback de pinv).
    max_iter:
        Iteraciones máximas del Newton estructurado.
    tol:
        Tolerancia de Frobenius entre iterados.
    form_certificate:
        Certificado (Ωᵀ = −Ω, Ω² = −I, det Ω = 1).
    """

    two_n: int
    n: int
    omega: np.ndarray
    reg_floor: float
    max_iter: int
    tol: float
    form_certificate: _SymplecticFormCertificate


class _NumericalCore:
    """
    Fase I. Álgebra numérica de precisión metrológica.

    Provee el topos lineal subyacente: sumación compensada en el álgebra
    de Banach (ℝ, +, ·), la 2-forma de Liouville, la involución de Cartan
    del par (GL(2n), Sp(2n)) y el gérmen que inicia la Fase II.
    """

    # ── I.1  Sumación compensada ──────────────────────────────────────────
    @staticmethod
    def kahan_sum(arr: np.ndarray) -> float:
        """
        Sumación compensada de Kahan.

        Neutraliza el término de redondeo \(c_{k+1}=(t_k-s_k)-y_k\) de modo
        que \(\sum x_i\) sea exacta módulo O(u · Σ|x_i|).
        """
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
        """
        Sumación de Kahan–Babuška–Neumaier (KBN).

        Acumula la compensación cuando |x| > |s|, estabilizando
        cancelaciones de signo mixto (asociadores, S de cobordes).
        """
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
        return float(total + c)

    @staticmethod
    def klein_sum(arr: np.ndarray) -> float:
        """Sumación doblemente compensada de Klein (error O(u²) relativo)."""
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

    # ── I.2  Normas, validación y Higham ──────────────────────────────────
    @staticmethod
    def frobenius_norm(matrix: np.ndarray) -> float:
        """Norma de Hilbert–Schmidt / Frobenius ‖A‖_F."""
        a = np.asarray(matrix)
        if a.size == 0:
            return 0.0
        return float(la.norm(a, "fro"))

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
        abs_floor: float = _REG_FLOOR_TIKHONOV,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Pseudoinversa amortiguada de Tikhonov–Higham vía SVD.

            σ⁺_i = σ_i / (σ_i² + λ²)    si σ_i > λ,
                 = 0                    en otro caso,
        con λ = max(abs_floor, rel_floor · σ_max). Devuelve (A⁺, σ, κ₂).
        """
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

    # ── I.3  2-forma simpléctica canónica de Darboux ──────────────────────
    @staticmethod
    def generate_canonical_symplectic_form(dim: int) -> np.ndarray:
        r"""
        2-forma canónica de Liouville Ω ∈ ℝ^{dim×dim}, dim = 2n par.

            Ω = \begin{pmatrix} 0 & I_n \\ -I_n & 0 \end{pmatrix},
            Ωᵀ = −Ω ,   Ω² = −I ,   Ω⁻¹ = −Ω ,   ΩᵀΩ = I.

        Ω es ortogonal (y hamiltoniana): la involución de Cartan
        X ↦ Ω X^{-T} Ωᵀ preserva la norma de Frobenius.
        """
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(
                f"La dimensión del colector simpléctico dim={dim} debe ser par y positiva."
            )
        half = dim // 2
        omega = np.zeros((dim, dim), dtype=np.float64)
        omega[:half, half:] = np.eye(half, dtype=np.float64)
        omega[half:, :half] = -np.eye(half, dtype=np.float64)
        return omega

    @staticmethod
    def certify_symplectic_form(omega: np.ndarray) -> _SymplecticFormCertificate:
        """Verifica los axiomas de Darboux sobre Ω."""
        _NumericalCore.assert_square("omega", omega)
        dim = omega.shape[0]
        ident = np.eye(dim, dtype=omega.dtype)
        skew = _NumericalCore.skew_residual(omega)
        almost_c = _NumericalCore.frobenius_norm(omega @ omega + ident)
        det_o = float(np.real(la.det(omega)))
        fro = _NumericalCore.frobenius_norm(omega)
        scale = max(fro, 1.0)
        is_darboux = (
            skew <= _WILKINSON_DRIFT_LIMIT * scale
            and almost_c <= _WILKINSON_DRIFT_LIMIT * scale
            and abs(det_o - 1.0) <= 1e-8 * max(1.0, abs(det_o))
        )
        return _SymplecticFormCertificate(
            skew_residual=float(skew),
            almost_complex_residual=float(almost_c),
            determinant=det_o,
            frobenius_norm=fro,
            is_darboux=bool(is_darboux),
        )

    @staticmethod
    def wilkinson_deflation_floor(matrix: np.ndarray) -> float:
        """
        Piso de deflación adaptativo de Wilkinson:

            ε_W = max( ‖A‖_F · ε_mach · 10 ,  ε_Wilkinson ).
        """
        if matrix is None or np.asarray(matrix).size == 0:
            return _WILKINSON_DEFLATION_FLOOR
        fro_norm = _NumericalCore.frobenius_norm(matrix)
        return float(
            max(
                fro_norm * _MACHINE_EPS * _WILKINSON_DEFLATION_SCALE,
                _WILKINSON_DEFLATION_FLOOR,
            )
        )

    @staticmethod
    def symplectic_inverse(S: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        Inversa de grupo en Sp(2n): si Sᵀ Ω S = Ω, entonces

            S⁻¹ = Ω⁻¹ Sᵀ Ω = −Ω Sᵀ Ω.

        No hay división: es un polinomio de grado 1 en Sᵀ. Fuera de Sp
        esta fórmula produce el adjunto de Cartan, no la inversa de GL.
        """
        return -omega @ np.asarray(S).T @ omega

    @staticmethod
    def cartan_involution_image(
        M: np.ndarray,
        omega: np.ndarray,
        floor: float,
    ) -> Tuple[np.ndarray, bool]:
        """
        Imagen de la involución de Cartan del par simétrico (GL, Sp):

            ι(M) = Ω M^{-T} Ωᵀ = Ω (Mᵀ)⁻¹ Ωᵀ.

        Se resuelve Mᵀ X = I (sin formar la inversa) y se cae a la pinv
        de Tikhonov–Higham si el LU falla. El segundo valor es True sii
        se usó un solve exacto (no pinv).
        """
        m = np.asarray(M, dtype=np.float64)
        dim = m.shape[0]
        ident = np.eye(dim, dtype=np.float64)
        try:
            x_inv_t = la.solve(m.T, ident, assume_a="gen")
            return omega @ x_inv_t @ omega.T, True
        except (np.linalg.LinAlgError, ValueError) as exc:
            logger.warning(
                "Solve de Cartan fallido (%s); se usa pinv de Tikhonov.", exc
            )
            pinv, _s, _k = _NumericalCore.tikhonov_higham_pinv(
                m.T, rel_floor=_MACHINE_EPS, abs_floor=floor
            )
            return omega @ pinv @ omega.T, False

    @staticmethod
    def symplectic_residual(M: np.ndarray, omega: np.ndarray) -> float:
        """‖Mᵀ Ω M − Ω‖_F : defecto de pertenencia a Sp(2n, ℝ)."""
        m = np.asarray(M, dtype=np.float64)
        return _NumericalCore.frobenius_norm(m.T @ omega @ m - omega)

    # ── I.8  Morfismo terminal de la Fase I ───────────────────────────────
    @staticmethod
    def synthesize_symplectic_quillen_germ(
        dimension_two_n: int,
        regularizer: float = _REG_FLOOR_TIKHONOV,
        max_iter: int = _DEFAULT_MAX_ITER,
        tol: float = _DEFAULT_TOL,
        scale_matrix: Optional[np.ndarray] = None,
    ) -> _SymplecticQuillenGerm:
        """
        I.8 — Morfismo terminal de la Fase I / objeto inicial de la Fase II.

        Ensambla el gérmen de Quillen–Darboux

            𝒢_I = (2n, n, Ω_{Darboux}, ε_W, K_Newton, τ, Cert(Ω))

        sobre el cual la Fase II define la iteración polar estructurada
        de Higham–Mackey–Tisseur y la factorización de Quillen en el
        modelo (cofibración acíclica, fibración) relativo a Sp(2n) ↪ GL(2n).

        Si se provee `scale_matrix` (p. ej. la M a proyectar), el piso
        de Wilkinson se adapta a su norma de Frobenius.

        Este método *es* el arranque formal de `_SymplecticProjector`.
        """
        dim = int(dimension_two_n)
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(
                f"dimension_two_n debe ser par y positivo; recibido {dimension_two_n}."
            )
        if int(max_iter) <= 0:
            raise ValueError("max_iter debe ser un entero positivo.")
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol debe ser positivo y finito.")
        omega = _NumericalCore.generate_canonical_symplectic_form(dim)
        certificate = _NumericalCore.certify_symplectic_form(omega)
        if not certificate.is_darboux:
            logger.warning(
                "Certificado de Darboux degradado: skew=%.3e, J²+I=%.3e, det=%.16f",
                certificate.skew_residual,
                certificate.almost_complex_residual,
                certificate.determinant,
            )
        floor = max(float(regularizer), _REG_FLOOR_TIKHONOV)
        if scale_matrix is not None:
            floor = max(
                floor,
                _NumericalCore.wilkinson_deflation_floor(np.asarray(scale_matrix)),
            )
        return _SymplecticQuillenGerm(
            two_n=dim,
            n=dim // 2,
            omega=omega,
            reg_floor=float(floor),
            max_iter=int(max_iter),
            tol=float(tol),
            form_certificate=certificate,
        )


# =============================================================================
# FASE II — POLAR SIMPLÉCTICA, QUILLEN Y LIFTING A∞ / ČECH
# -----------------------------------------------------------------------------
# Continúa I.8: todo projector se instancia desde un SymplecticQuillenGerm.
# Morfismo terminal (II.6): induce_ainfty_cech_germ
#          ≅ objeto inicial de la Fase III (cochain / associator unfold).
# =============================================================================
@dataclass(frozen=True)
class _SymplecticProjectionResult:
    """Resultado certificado de la proyección polar sobre Sp(2n, ℝ)."""

    symplectic_matrix: np.ndarray
    residual: float
    iterations: int
    relative_residual: float
    determinant: float
    used_exact_solve: bool
    converged: bool


@dataclass(frozen=True)
class _QuillenFactorizationResult:
    """Factorización de Quillen M = P_L ∘ S = S ∘ P_R en (GL, Sp)."""

    fibration: np.ndarray
    cofibration: np.ndarray
    total_residual: float
    left_reconstruction: float
    right_reconstruction: float
    symplectic_error: float
    inverse_formula_residual: float
    right_fibration: np.ndarray


@dataclass(frozen=True)
class _AInfinityCechGerm:
    """
    Gérmen A∞ / Čech–gerbe (objeto terminal de la Fase II).

    Es el objeto inicial de la Fase III: una cochain matricial obtenida
    por una de dos ramas

    * Quillen: el defecto P_L − I (fibración) se lee como 1-cochain /
      conexión de gerbe sobre el nervio trivial de dimensión 2n;
    * Stasheff: el asociador m₃ se despliega a un endomorfismo de V⊗V
      (matriz n² × n²) y se interpreta como 2-cochain de gerbe.

    Atributos
    ---------
    cochain_matrix:
        Matriz que la Fase III somete a SVD / Hodge / δ.
    source:
        ``"quillen"`` o ``"stasheff"``.
    algebra_dim:
        n del A∞ (o 2n si la fuente es Quillen).
    reg_floor:
        Piso heredado.
    associator_norm:
        ‖m₃‖_F si la fuente es Stasheff; si no, ‖P−I‖_F.
    """

    cochain_matrix: np.ndarray
    source: str
    algebra_dim: int
    reg_floor: float
    associator_norm: float


class _SymplecticProjector:
    """
    Fase II. Proyector polar sobre Sp(2n, ℝ) y factorización de Quillen.

    Continúa el gérmen 𝒢_I. Implementa la iteración estructurada de
    Higham–Mackey–Tisseur *con escalado de Frobenius*

        μ_k = ( ‖ι(X_k)‖_F / ‖X_k‖_F )^{1/2} ,
        X_{k+1} = ½ ( μ_k X_k + μ_k^{-1} ι(X_k) ) ,

    donde ι(X) = Ω X^{-T} Ωᵀ es la involución de Cartan. El punto fijo
    es el factor simpléctico de la descomposición polar estructurada.
    No se añade εI a X (eso rompería el grupo); el fallback es pinv.

    Quillen: M = P_L S = S P_R con S ∈ Sp(2n) el polar y
    S⁻¹ = −Ω Sᵀ Ω (fórmula de grupo, sin divisiones).
    """

    def __init__(
        self,
        germ: Optional[_SymplecticQuillenGerm] = None,
        regularizer: float = _REG_FLOOR_TIKHONOV,
        max_iter: int = _DEFAULT_MAX_ITER,
        tol: float = _DEFAULT_TOL,
        dimension_two_n: int = 2,
    ) -> None:
        if germ is None:
            germ = _NumericalCore.synthesize_symplectic_quillen_germ(
                dimension_two_n,
                regularizer=regularizer,
                max_iter=max_iter,
                tol=tol,
            )
        self._germ = germ

    @property
    def germ(self) -> _SymplecticQuillenGerm:
        """Gérmen de Fase I del que este proyector es continuación."""
        return self._germ

    def _adapt_omega(self, dim: int) -> np.ndarray:
        if dim == self._germ.two_n:
            return self._germ.omega
        return _NumericalCore.generate_canonical_symplectic_form(dim)

    def project(
        self,
        M: np.ndarray,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
    ) -> _SymplecticProjectionResult:
        r"""
        Proyecta M sobre Sp(2n, ℝ) por Newton estructurado escalado.

        La fórmula clásica (sin escalar)

            M_{k+1} = ½ ( M_k + Ω (M_k^{-1})ᵀ Ωᵀ )

        se recupera con μ_k ≡ 1; el escalado de Frobenius acelera la
        fase lineal y evita overflow cuando ‖M‖ ≫ 1. `max_iter` y `tol`
        sobrescriben el gérmen (la API 2.0 los aceptaba y los ignoraba).
        """
        a = np.asarray(M, dtype=np.float64)
        _NumericalCore.assert_square("M", a)
        _NumericalCore.assert_finite("M", a)
        dim = a.shape[0]
        if dim % 2 != 0:
            raise ValueError(f"M debe tener dimensión par (Darboux); recibido {dim}.")

        omega = self._adapt_omega(dim)
        k_max = int(self._germ.max_iter if max_iter is None else max_iter)
        tau = float(self._germ.tol if tol is None else tol)
        if k_max <= 0:
            raise ValueError("max_iter debe ser positivo.")
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError("tol debe ser positivo y finito.")
        floor = max(
            self._germ.reg_floor, _NumericalCore.wilkinson_deflation_floor(a)
        )

        m_k = a.copy()
        residual = float("inf")
        iterations = 0
        used_exact = True
        converged = False

        for iteration in range(k_max):
            iterations = iteration + 1
            iota, exact = _NumericalCore.cartan_involution_image(m_k, omega, floor)
            used_exact = used_exact and exact
            fro_m = _NumericalCore.frobenius_norm(m_k)
            fro_i = _NumericalCore.frobenius_norm(iota)
            if fro_m <= _MACHINE_EPS or fro_i <= _MACHINE_EPS:
                logger.warning("Polar simpléctica: norma degenerada en iter %d.", iteration)
                break
            mu = float(np.sqrt(fro_i / fro_m))
            mu = float(np.clip(mu, _MU_CLIP[0], _MU_CLIP[1]))
            m_next = 0.5 * (mu * m_k + (1.0 / mu) * iota)
            diff = _NumericalCore.frobenius_norm(m_next - m_k)
            m_k = m_next
            if not np.isfinite(diff):
                logger.warning("Polar simpléctica: iterado no finito en %d.", iteration)
                break
            if diff < tau * max(1.0, fro_m):
                converged = True
                break

        residual = _NumericalCore.symplectic_residual(m_k, omega)
        if not np.isfinite(residual):
            residual = float("inf")
        scale = max(_NumericalCore.frobenius_norm(omega) * (fro_m ** 2 + 1.0), 1.0)
        rel = float(residual / scale)
        try:
            det_s = float(np.real(la.det(m_k)))
        except (np.linalg.LinAlgError, ValueError):
            det_s = float("nan")

        return _SymplecticProjectionResult(
            symplectic_matrix=m_k,
            residual=float(residual),
            iterations=int(iterations),
            relative_residual=rel,
            determinant=det_s,
            used_exact_solve=bool(used_exact),
            converged=bool(converged),
        )

    def factorize_quillen(self, M: np.ndarray) -> _QuillenFactorizationResult:
        r"""
        Factorización de Quillen relativa a Sp(2n) ↪ GL(2n).

        En el modelo de Quillen, toda flecha se factoriza como
        (cofibración acíclica) ∘ (fibración). Aquí S = Pol_{Sp}(M)
        hace las veces de reemplazo fibrante en el locus simpléctico y

            M = P_L S = S P_R ,
            P_L = M S⁻¹ ,   P_R = S⁻¹ M ,
            S⁻¹ = −Ω Sᵀ Ω.

        `fibration` / `cofibration` conservan la convención 2.0 (P_L, S).
        """
        proj = self.project(M)
        s_mat = proj.symplectic_matrix
        a = np.asarray(M, dtype=np.float64)
        dim = a.shape[0]
        omega = self._adapt_omega(dim)

        s_inv = _NumericalCore.symplectic_inverse(s_mat, omega)
        # Certifica que la fórmula de grupo reproduce la inversa de GL.
        ident = np.eye(dim, dtype=np.float64)
        inv_formula = _NumericalCore.frobenius_norm(s_mat @ s_inv - ident)

        p_left = a @ s_inv
        p_right = s_inv @ a

        left_rec = _NumericalCore.frobenius_norm(a - p_left @ s_mat)
        right_rec = _NumericalCore.frobenius_norm(a - s_mat @ p_right)
        sp_err = proj.residual
        total = float(left_rec + sp_err)

        return _QuillenFactorizationResult(
            fibration=p_left,
            cofibration=s_mat,
            total_residual=total,
            left_reconstruction=float(left_rec),
            right_reconstruction=float(right_rec),
            symplectic_error=float(sp_err),
            inverse_formula_residual=float(inv_formula),
            right_fibration=p_right,
        )

    # ── II.6  Morfismo terminal de la Fase II ─────────────────────────────
    def induce_ainfty_cech_germ(
        self,
        payload: np.ndarray,
        m2_tensor: Optional[np.ndarray] = None,
    ) -> _AInfinityCechGerm:
        """
        II.6 — Morfismo terminal de la Fase II / objeto inicial de la Fase III.

        Extrae el gérmen A∞ / gerbe por una de dos ramas:

        1. `m2_tensor` de rango 3 (o `payload` de rango 3): se computa el
           asociador de Stasheff m₃ y se despliega a End(V⊗V),
           G[i n + j, k n + l] = (m₃)_{ijk}^{l}.
        2. Matriz cuadrada: factorización de Quillen y cochain P_L − I
           (defecto de la fibración; conexión de gerbe).

        Este método *es* el arranque formal de `_StasheffAssociator` y
        `_CechObstructionCalculator`.
        """
        if m2_tensor is not None:
            tensor = np.asarray(m2_tensor)
        else:
            tensor = np.asarray(payload)

        if tensor.ndim == 3:
            associator = _StasheffAssociator()
            m3 = associator.compute_m3_associator(tensor)
            n = int(tensor.shape[0])
            # Unfold Hom(V⊗V⊗V, V) ≅ End(V⊗V).
            germ_mat = np.transpose(m3, (0, 1, 2, 3)).reshape(n * n, n * n)
            germ_mat = np.real(_NumericalCore.higham_nearest_hermitian(germ_mat))
            a_norm = _NumericalCore.frobenius_norm(m3)
            floor = max(
                self._germ.reg_floor,
                _NumericalCore.wilkinson_deflation_floor(germ_mat),
            )
            return _AInfinityCechGerm(
                cochain_matrix=np.asarray(germ_mat, dtype=np.float64),
                source="stasheff",
                algebra_dim=n,
                reg_floor=float(floor),
                associator_norm=float(a_norm),
            )

        a = np.asarray(tensor)
        if a.ndim == 1:
            side = int(np.sqrt(a.size))
            if side * side != a.size:
                raise ValueError("payload plano no es un cuadrado perfecto.")
            a = a.reshape(side, side)
        _NumericalCore.assert_square("payload", a)
        _NumericalCore.assert_finite("payload", a)
        fact = self.factorize_quillen(a)
        ident = np.eye(a.shape[0], dtype=np.float64)
        defect = fact.fibration - ident
        germ_mat = np.real(_NumericalCore.higham_nearest_hermitian(defect))
        floor = max(
            self._germ.reg_floor,
            _NumericalCore.wilkinson_deflation_floor(germ_mat),
        )
        return _AInfinityCechGerm(
            cochain_matrix=np.asarray(germ_mat, dtype=np.float64),
            source="quillen",
            algebra_dim=int(a.shape[0]),
            reg_floor=float(floor),
            associator_norm=float(_NumericalCore.frobenius_norm(defect)),
        )


# =============================================================================
# FASE III — STASHEFF A∞, JACOBI, GERBE DE ČECH–DELIGNE
# -----------------------------------------------------------------------------
# Continúa II.6: asociador y Čech se anclan a un AInfinityCechGerm
# (o lo inducen por cálculo directo del tensor / cochain cruda).
# =============================================================================
@dataclass(frozen=True)
class _StasheffAssociatorResult:
    """Asociador m₃ con residuos A∞ (pentágono) y de Jacobi."""

    associator: np.ndarray
    associator_norm: float
    pentagon_residual: float
    jacobi_residual: float
    is_associative: bool
    is_lie: bool


@dataclass(frozen=True)
class _CechObstructionResult:
    """Obstrucción de gerbe Čech–Deligne, Hodge y masa nuclear."""

    obstruction_value: float
    singular_values: np.ndarray
    active_modes: int
    cocycle_defect: float
    gerbe_4cocycle_defect: float
    harmonic_energy: float
    betti_0: int
    betti_1: int
    nuclear_mass: float


class _StasheffAssociator:
    """
    Fase III. Asociador de Stasheff y residuos A∞.

    Para un producto bilinear m₂ : V⊗V → V (tensor n×n×n, índice de
    salida al final),

        (m₃)_{ijk}^{l}
            = Σ_s ( (m₂)_{ij}^{s} (m₂)_{sk}^{l} − (m₂)_{jk}^{s} (m₂)_{is}^{l} )

    es el asociador (xy)z − x(yz). La contracción en s se acumula con
    KBN. Si m₁ = 0, la relación pentagonal de Stasheff exige

        m₂(m₃(a,b,c), d) − m₂(a, m₃(b,c,d))
        − m₃(m₂(a,b), c, d) + m₃(a, m₂(b,c), d) − m₃(a, b, m₂(c,d)) = 0.

    El jacobiato del conmutador [x,y] = m₂(x,y)−m₂(y,x) certifica si
    (V,[·,·]) es un álgebra de Lie.
    """

    def __init__(self, germ: Optional[_AInfinityCechGerm] = None) -> None:
        self._germ = germ

    @staticmethod
    def _validate_m2(m2_tensor: np.ndarray) -> np.ndarray:
        t = np.asarray(m2_tensor)
        if t.ndim != 3 or t.shape[0] != t.shape[1] or t.shape[1] != t.shape[2]:
            raise ValueError(
                f"m2_tensor debe ser de forma (n,n,n); recibido {t.shape}."
            )
        _NumericalCore.assert_finite("m2_tensor", t)
        return np.asarray(t, dtype=np.float64)

    def compute_m3_associator(self, m2_tensor: np.ndarray) -> np.ndarray:
        r"""
        Tensor asociador m₃ ∈ ℝ^{n×n×n×n}.

        Cada componente es una suma compensada KBN sobre el índice
        mudo s (en 2.0 el comentario invocaba Kahan y el cuerpo usaba
        un einsum sin compensación, además de un bucle l inerte).
        """
        m2 = self._validate_m2(m2_tensor)
        n = m2.shape[0]
        m3 = np.zeros((n, n, n, n), dtype=np.float64)
        for i in range(n):
            left_ij = m2[i, :, :]          # (j_src? wait) m2[i, j, s] → usamos por j
            # Recorremos j, k y contraemos s vectorialmente en l.
            for j in range(n):
                ij = m2[i, j, :]           # (s,)
                for k in range(n):
                    # term1[l] = Σ_s m2[i,j,s] m2[s,k,l]
                    # term2[l] = Σ_s m2[j,k,s] m2[i,s,l]
                    left = m2[:, k, :]     # (s, l)
                    right = m2[i, :, :]    # (s, l)
                    jk = m2[j, k, :]       # (s,)
                    # Para cada l, KBN sobre s.
                    prod1 = ij[:, None] * left          # (s, l)
                    prod2 = jk[:, None] * right         # (s, l)
                    diff = prod1 - prod2
                    for ell in range(n):
                        m3[i, j, k, ell] = _NumericalCore.kahan_babuska_neumaier_sum(
                            diff[:, ell]
                        )
        return m3

    def pentagon_residual(self, m2: np.ndarray, m3: np.ndarray) -> float:
        """
        Residuo pentagonal A∞ (m₁ = 0) en norma de Frobenius.

        Para n > `_STASHEFF_PENTAGON_CAP` se evalúa sobre una malla
        determinista (stride) para no degradar la FPU (el 5-tensor es O(n⁵)).
        """
        n = m2.shape[0]
        if n == 0:
            return 0.0
        if n <= _STASHEFF_PENTAGON_CAP:
            idx = np.arange(n)
        else:
            step = max(1, n // _STASHEFF_PENTAGON_CAP)
            idx = np.arange(0, n, step)
        acc = 0.0
        for i in idx:
            for j in idx:
                for k in idx:
                    for p in idx:
                        # 5 contracciones sobre s, una por destino q.
                        # t1 = m2(m3(i,j,k), p)     → Σ_s m3[i,j,k,s] m2[s,p,q]
                        # t2 = m2(i, m3(j,k,p))     → Σ_s m2[i,s,q] m3[j,k,p,s]
                        # t3 = m3(m2(i,j), k, p)    → Σ_s m2[i,j,s] m3[s,k,p,q]
                        # t4 = m3(i, m2(j,k), p)    → Σ_s m2[j,k,s] m3[i,s,p,q]
                        # t5 = m3(i, j, m2(k,p))    → Σ_s m2[k,p,s] m3[i,j,s,q]
                        t1 = m3[i, j, k, :] @ m2[:, p, :]
                        t2 = m3[j, k, p, :] @ m2[i, :, :]
                        t3 = m2[i, j, :] @ m3[:, k, p, :]
                        t4 = m2[j, k, :] @ m3[i, :, p, :]
                        t5 = m2[k, p, :] @ m3[i, j, :, :]
                        pent = t1 - t2 - t3 + t4 - t5
                        acc += float(_NumericalCore.kahan_babuska_neumaier_sum(pent * pent))
        return float(np.sqrt(max(acc, 0.0)))

    def jacobi_residual(self, m2: np.ndarray) -> float:
        """
        Residuo de Jacobi del conmutador [x,y] = m₂(x,y) − m₂(y,x).

            J(x,y,z) = [x,[y,z]] + [y,[z,x]] + [z,[x,y]].
        """
        n = m2.shape[0]
        comm = m2 - np.transpose(m2, (1, 0, 2))
        acc = 0.0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    # [y,z] = comm[j,k,:];  [x, that]_q = Σ_s comm[i,s,q] comm[j,k,s]
                    yz = comm[j, k, :]
                    zx = comm[k, i, :]
                    xy = comm[i, j, :]
                    t_x = yz @ comm[i, :, :]
                    t_y = zx @ comm[j, :, :]
                    t_z = xy @ comm[k, :, :]
                    jac = t_x + t_y + t_z
                    acc += float(_NumericalCore.kahan_babuska_neumaier_sum(jac * jac))
        return float(np.sqrt(max(acc, 0.0)))

    def compute_certified(self, m2_tensor: np.ndarray) -> _StasheffAssociatorResult:
        """m₃ + pentágono A∞ + Jacobi, con banderas de asociatividad / Lie."""
        m2 = self._validate_m2(m2_tensor)
        m3 = self.compute_m3_associator(m2)
        a_norm = _NumericalCore.frobenius_norm(m3)
        pent = self.pentagon_residual(m2, m3)
        jac = self.jacobi_residual(m2)
        scale = max(a_norm, 1.0)
        is_assoc = a_norm <= _WILKINSON_DRIFT_LIMIT * scale
        is_lie = jac <= _WILKINSON_DRIFT_LIMIT * max(1.0, _NumericalCore.frobenius_norm(m2))
        return _StasheffAssociatorResult(
            associator=m3,
            associator_norm=float(a_norm),
            pentagon_residual=float(pent),
            jacobi_residual=float(jac),
            is_associative=bool(is_assoc),
            is_lie=bool(is_lie),
        )


class _CechObstructionCalculator:
    """
    Fase III. Obstrucción de gerbe en hipercohomología de Čech–Deligne.

    Continúa el gérmen 𝒢_II cuando se provee. Sobre una cochain matricial A:

    * masa nuclear de los modos sobre el piso de Wilkinson (API 2.0);
    * (δω)_{ijk} = ω_{jk} − ω_{ik} + ω_{ij}  (2-cociclo / curving);
    * (δ²ω)_{ijkl} en cuádruples (identidad de gerbe / δ² = 0);
    * Laplaciano de Hodge del nervio umbralizado y números de Betti.
    """

    def __init__(
        self,
        germ: Optional[_AInfinityCechGerm] = None,
        regularizer: float = _REG_FLOOR_TIKHONOV,
    ) -> None:
        self._germ = germ
        self._reg = max(float(regularizer), _REG_FLOOR_TIKHONOV)

    def _resolve_matrix(self, cech_cochain_matrix: np.ndarray) -> np.ndarray:
        a = np.asarray(cech_cochain_matrix)
        if a.size == 0 and self._germ is not None:
            return np.asarray(self._germ.cochain_matrix, dtype=np.float64)
        if a.ndim == 1:
            side = int(np.sqrt(a.size))
            if side * side != a.size:
                raise ValueError("cech_cochain_matrix plana no es un cuadrado perfecto.")
            a = a.reshape(side, side)
        _NumericalCore.assert_square("cech_cochain_matrix", a)
        _NumericalCore.assert_finite("cech_cochain_matrix", a)
        return np.asarray(a, dtype=np.complex128)

    def cech_coboundary_defect(self, omega: np.ndarray) -> float:
        """‖δω‖ del 1-cochain antisimétrico (curving de gerbe)."""
        w = np.real(0.5 * (np.asarray(omega) - np.asarray(omega).T.conj()))
        n = w.shape[0]
        if n < 3:
            return 0.0
        acc = 0.0
        if n <= _CECH_TRIPLE_CAP:
            triples = (
                (i, j, k)
                for i in range(n - 2)
                for j in range(i + 1, n - 1)
                for k in range(j + 1, n)
            )
        else:
            step = max(1, n // _CECH_TRIPLE_CAP)
            idx = np.arange(0, n, step)
            triples = (
                (int(idx[a]), int(idx[b]), int(idx[c]))
                for a in range(idx.size - 2)
                for b in range(a + 1, idx.size - 1)
                for c in range(b + 1, idx.size)
            )
        for i, j, k in triples:
            t = w[j, k] - w[i, k] + w[i, j]
            acc += float(t * t)
        return float(np.sqrt(max(acc, 0.0)))

    def gerbe_4cocycle_defect(self, omega: np.ndarray) -> float:
        """
        ‖δ²ω‖ en cuádruples i<j<k<l (identidad de gerbe).

            (δθ)_{ijkl} = θ_{jkl} − θ_{ikl} + θ_{ijl} − θ_{ijk} ,
            θ_{abc}     = ω_{bc} − ω_{ac} + ω_{ab}.

        Es idénticamente nulo si ω es un 1-cochain (δ² = 0). Un residuo
        no nulo señala inconsistencia numérica o una 2-cochain genuina.
        """
        w = np.real(0.5 * (np.asarray(omega) - np.asarray(omega).T.conj()))
        n = w.shape[0]
        if n < 4:
            return 0.0

        def theta(a: int, b: int, c: int) -> float:
            return float(w[b, c] - w[a, c] + w[a, b])

        acc = 0.0
        if n <= _CECH_QUAD_CAP:
            index = range(n)
        else:
            step = max(1, n // _CECH_QUAD_CAP)
            index = range(0, n, step)
        idx = list(index)
        m = len(idx)
        for a in range(m - 3):
            i = idx[a]
            for b in range(a + 1, m - 2):
                j = idx[b]
                for c in range(b + 1, m - 1):
                    k = idx[c]
                    for d in range(c + 1, m):
                        ell = idx[d]
                        t = (
                            theta(j, k, ell)
                            - theta(i, k, ell)
                            + theta(i, j, ell)
                            - theta(i, j, k)
                        )
                        acc += float(t * t)
        return float(np.sqrt(max(acc, 0.0)))

    def sheaf_hodge_spectrum(
        self,
        gram: np.ndarray,
        floor: float,
    ) -> Tuple[int, int, float]:
        """(β₀, β₁, energía armónica) del Laplaciano de Hodge del nervio."""
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
        harmonic = _NumericalCore.kahan_babuska_neumaier_sum(
            np.clip(evals[: max(betti_0, 0)], 0.0, None)
        ) if evals.size else 0.0
        return betti_0, betti_1, float(harmonic)

    def compute_obstruction(self, cech_cochain_matrix: np.ndarray) -> _CechObstructionResult:
        """
        Resuelve espectralmente la obstrucción no abeliana de Čech.

        El valor reportado por la API 2.0 es la masa nuclear de los modos
        activos (Kahan sobre el espectro deflactado). El certificado añade
        curving, 4-cociclo de gerbe, Hodge y Betti.
        """
        if np.asarray(cech_cochain_matrix).size == 0 and self._germ is None:
            return _CechObstructionResult(
                obstruction_value=0.0,
                singular_values=np.array([], dtype=np.float64),
                active_modes=0,
                cocycle_defect=0.0,
                gerbe_4cocycle_defect=0.0,
                harmonic_energy=0.0,
                betti_0=0,
                betti_1=0,
                nuclear_mass=0.0,
            )

        raw = self._resolve_matrix(cech_cochain_matrix)
        sheaf = _NumericalCore.higham_nearest_hermitian(raw)
        floor = max(self._reg, _NumericalCore.wilkinson_deflation_floor(sheaf))
        if self._germ is not None:
            floor = max(floor, self._germ.reg_floor)

        singular_values = np.real(la.svd(sheaf, compute_uv=False))
        active = singular_values[singular_values > floor]
        obstruction = (
            _NumericalCore.kahan_sum(active) if active.size else 0.0
        )
        cocycle = self.cech_coboundary_defect(sheaf)
        gerbe = self.gerbe_4cocycle_defect(sheaf)
        b0, b1, harmonic = self.sheaf_hodge_spectrum(sheaf, floor)

        return _CechObstructionResult(
            obstruction_value=float(obstruction),
            singular_values=np.asarray(singular_values, dtype=np.float64),
            active_modes=int(active.size),
            cocycle_defect=float(cocycle),
            gerbe_4cocycle_defect=float(gerbe),
            harmonic_energy=float(harmonic),
            betti_0=int(b0),
            betti_1=int(b1),
            nuclear_mass=float(obstruction),
        )


# =============================================================================
# MOTOR PRINCIPAL — INTEGRACIÓN DEL MORFISMO Φ_III ∘ Φ_II ∘ Φ_I
# =============================================================================
class ImperialTesserariosEngine:
    """
    Motor de álgebra homológica no abeliana y geometría categorial.

    Compone las tres fases anidadas:

    1. Fase I   — gérmen de Darboux / Cartan (`_NumericalCore`).
    2. Fase II  — polar Sp(2n) y Quillen (`_SymplecticProjector`).
    3. Fase III — Stasheff / gerbe Čech (`_StasheffAssociator`,
                  `_CechObstructionCalculator`), inicializados con un
                  gérmen de referencia (identidad); cada llamada puede
                  sustituirlo.

    La API pública de 2.0 se conserva. Los métodos `*_certified` exponen
    los invariantes añadidos en 3.0. `max_iter` y `tol` se honran
    (en 2.0 se aceptaban y se ignoraban).
    """

    def __init__(self, regularizer: float = 1e-15) -> None:
        """
        Inicializa el motor y materializa el encadenamiento de gérmenes.

        El regularizador se mantiene al menos en `_REG_FLOOR_TIKHONOV` y
        alimenta el fallback de pinv, el piso de Wilkinson y el Hodge.

        Args:
            regularizer: Piso de Tikhonov contra polos espectrales.
        """
        self._reg: Final[float] = max(float(regularizer), _REG_FLOOR_TIKHONOV)
        # Fase I → objeto inicial de Fase II (dimensión mínima de Darboux).
        self._quillen_germ: _SymplecticQuillenGerm = (
            _NumericalCore.synthesize_symplectic_quillen_germ(
                2, regularizer=self._reg
            )
        )
        self._projector = _SymplecticProjector(germ=self._quillen_germ)
        # Fase II → objeto inicial de Fase III (Cayley / defecto nulo de I₂).
        self._cech_germ: _AInfinityCechGerm = self._projector.induce_ainfty_cech_germ(
            np.eye(2, dtype=np.float64)
        )
        self._associator = _StasheffAssociator(germ=self._cech_germ)
        self._cech_calculator = _CechObstructionCalculator(
            germ=self._cech_germ, regularizer=self._reg
        )

    def _resync_quillen_germ(
        self,
        two_n: int,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
        scale_matrix: Optional[np.ndarray] = None,
    ) -> None:
        """Re-sintetiza 𝒢_I cuando cambian la dimensión o los parámetros Newton."""
        same_dim = two_n == self._quillen_germ.two_n
        same_it = max_iter is None or int(max_iter) == self._quillen_germ.max_iter
        same_tol = tol is None or float(tol) == self._quillen_germ.tol
        if same_dim and same_it and same_tol and scale_matrix is None:
            return
        self._quillen_germ = _NumericalCore.synthesize_symplectic_quillen_germ(
            two_n,
            regularizer=self._reg,
            max_iter=self._quillen_germ.max_iter if max_iter is None else int(max_iter),
            tol=self._quillen_germ.tol if tol is None else float(tol),
            scale_matrix=scale_matrix,
        )
        self._projector = _SymplecticProjector(germ=self._quillen_germ)

    # ── Fase I expuesta ───────────────────────────────────────────────────
    def kahan_sum(self, arr: np.ndarray) -> float:
        """Sumación compensada de Kahan expuesta públicamente."""
        return _NumericalCore.kahan_sum(arr)

    def kahan_babuska_neumaier_sum(self, arr: np.ndarray) -> float:
        """Sumación KBN (expuesta explícitamente)."""
        return _NumericalCore.kahan_babuska_neumaier_sum(arr)

    def generate_canonical_symplectic_form(self, dim: int) -> np.ndarray:
        """Genera la forma simpléctica canónica de Liouville."""
        omega = _NumericalCore.generate_canonical_symplectic_form(dim)
        if dim != self._quillen_germ.two_n:
            self._resync_quillen_germ(dim)
        return omega

    def symplectic_quillen_germ_certificate(self) -> _SymplecticFormCertificate:
        """Certificado de Darboux del gérmen de Fase I vigente."""
        return self._quillen_germ.form_certificate

    def synthesize_symplectic_quillen_germ(
        self,
        dimension_two_n: int,
        max_iter: int = _DEFAULT_MAX_ITER,
        tol: float = _DEFAULT_TOL,
        scale_matrix: Optional[np.ndarray] = None,
    ) -> _SymplecticQuillenGerm:
        """Réplica pública del morfismo I.8; actualiza el gérmen de Fase II."""
        self._resync_quillen_germ(
            int(dimension_two_n),
            max_iter=max_iter,
            tol=tol,
            scale_matrix=scale_matrix,
        )
        return self._quillen_germ

    # ── Fase II expuesta ──────────────────────────────────────────────────
    def project_to_symplectic_group(
        self,
        M: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-12,
    ) -> Tuple[np.ndarray, float]:
        """
        Proyección simpléctica con iteración polar (wrapper). API 2.0.

        A diferencia de 2.0, `max_iter` y `tol` se aplican de verdad.
        """
        result = self.project_to_symplectic_group_certified(M, max_iter=max_iter, tol=tol)
        return result.symplectic_matrix, result.residual

    def project_to_symplectic_group_certified(
        self,
        M: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-12,
    ) -> _SymplecticProjectionResult:
        """Polar Sp(2n) con residuo relativo, det, convergencia y tipo de solve."""
        a = np.asarray(M)
        _NumericalCore.assert_square("M", a)
        if a.shape[0] % 2 == 0:
            self._resync_quillen_germ(
                a.shape[0], max_iter=max_iter, tol=tol, scale_matrix=a
            )
        return self._projector.project(a, max_iter=max_iter, tol=tol)

    def compute_quillen_factorization(
        self,
        M: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Factorización de Quillen M = P · I. API 2.0: (P, S, residual)."""
        result = self.compute_quillen_factorization_certified(M)
        return result.fibration, result.cofibration, result.total_residual

    def compute_quillen_factorization_certified(
        self,
        M: np.ndarray,
    ) -> _QuillenFactorizationResult:
        """Quillen izquierdo/derecho con residuales de reconstrucción e inversa."""
        a = np.asarray(M)
        _NumericalCore.assert_square("M", a)
        if a.shape[0] % 2 == 0:
            self._resync_quillen_germ(a.shape[0], scale_matrix=a)
        return self._projector.factorize_quillen(a)

    def induce_ainfty_cech_germ(
        self,
        payload: np.ndarray,
        m2_tensor: Optional[np.ndarray] = None,
    ) -> _AInfinityCechGerm:
        """
        Réplica pública del morfismo II.6: Quillen / m₂ ↦ gérmen Čech.
        Actualiza el objeto con el que opera la Fase III.
        """
        raw = np.asarray(payload if m2_tensor is None else m2_tensor)
        if raw.ndim == 2 and raw.shape[0] % 2 == 0:
            self._resync_quillen_germ(raw.shape[0], scale_matrix=raw)
        germ = self._projector.induce_ainfty_cech_germ(payload, m2_tensor=m2_tensor)
        self._cech_germ = germ
        self._associator = _StasheffAssociator(germ=germ)
        self._cech_calculator = _CechObstructionCalculator(
            germ=germ, regularizer=self._reg
        )
        return germ

    # ── Fase III expuesta ─────────────────────────────────────────────────
    def compute_stasheff_m3_associator(self, m2_tensor: np.ndarray) -> np.ndarray:
        """Calcula el asociador m₃ de Stasheff. API 2.0."""
        return self._associator.compute_m3_associator(m2_tensor)

    def compute_stasheff_m3_associator_certified(
        self,
        m2_tensor: np.ndarray,
    ) -> _StasheffAssociatorResult:
        """m₃ con norma, pentágono A∞, Jacobi y banderas asociativa / Lie."""
        result = self._associator.compute_certified(m2_tensor)
        # Actualiza el gérmen de Fase III con el unfold del asociador.
        self.induce_ainfty_cech_germ(m2_tensor, m2_tensor=m2_tensor)
        return result

    def compute_cech_hypercohomology_gerbe(
        self,
        cech_cochain_matrix: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Calcula la obstrucción de Čech para gerbes. API 2.0."""
        result = self._cech_calculator.compute_obstruction(cech_cochain_matrix)
        return result.obstruction_value, result.singular_values

    def compute_cech_hypercohomology_gerbe_certified(
        self,
        cech_cochain_matrix: np.ndarray,
    ) -> _CechObstructionResult:
        """Gerbe Čech–Deligne con curving, 4-cociclo, Hodge y Betti."""
        return self._cech_calculator.compute_obstruction(cech_cochain_matrix)


__all__ = ["ImperialTesserariosEngine"]