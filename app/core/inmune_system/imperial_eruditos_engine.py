# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Imperial Eruditos Engine (Caballos de Batalla de Cohomología)       ║
║ Ruta   : app/core/inmune_system/imperial_eruditos_engine.py                  ║
║ Versión: 3.0.0-Nested-Phases-Floer-CZ-Cech-Hodge-CSMD-Kahan-FPU              ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS MATEMÁTICA Y METROLOGÍA DE LA FPU
────────────────────────────────────────────────────────────────────────────────
Motor de-confinado de Nivel 4.5 (V_ERUDITOS). Produce argumentos homológicos
duros para el Consejo de Sabios mediante el morfismo de fases anidadas

    Φ_III ∘ Φ_II ∘ Φ_I :  Banach_CSMD  →  Floer(Sp(2n), 𝒜_H)  →  Sh(𝔘; Ȟ^•)

  Fase I   Núcleo de Banach, 2-forma de Darboux y diferenciación holomorfa
           por paso complejo (CSMD). Último morfismo:
           synthesize_floer_cylinder_germ.
  Fase II  Homología de Floer: acción de Liouville, residuo del operador
           de Cauchy–Riemann perturbado, índice de Conley–Zehnder /
           Robbin–Salamon y transformada de Cayley del monodromía.
           Último morfismo: induce_cech_nerve_germ.
  Fase III Cohomología de Čech atencional: nervio del recubrimiento,
           coborde δ (δ² = 0), Laplaciano de Hodge y números de Betti.

Precisión metrológica: Kahan, Kahan–Babuška–Neumaier, Klein; CSMD sin
cancelación sustractiva; pinv de Tikhonov–Higham; sumas espectrales
compensadas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Final, Optional, Tuple

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Core.ImperialEruditosEngine")

__version__: Final[str] = "3.0.0-Nested-Phases-Floer-CZ-Cech-Hodge-CSMD-Kahan-FPU"


# =============================================================================
# CONSTANTES DE PRECISIÓN METROLÓGICA
# =============================================================================
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_HIGHAM_TIKHONOV_FLOOR: Final[float] = 1e-20
_WILKINSON_DEFLATION_FLOOR: Final[float] = 1e-15
_WILKINSON_DEFLATION_SCALE: Final[float] = 10.0
_WILKINSON_DRIFT_LIMIT: Final[float] = 1e-9
_CSMD_STEP: Final[float] = 1e-20
_CSMD_FD_FALLBACK: Final[float] = 1e-8
_CECH_TRIPLE_CAP: Final[int] = 80
_LOG_EXP_CLIP: Final[float] = 700.0
_MASLOV_DEGENERACY: Final[float] = 1e-10


# =============================================================================
# FASE I — NÚCLEO DE BANACH, DARBOUX Y CSMD HOLMORFA
# -----------------------------------------------------------------------------
# Objetos: sumas compensadas, 2-forma canónica Ω, gradiente / Hessiano CSMD,
#          campo hamiltoniano X_H = Ω ∇H y acción de Liouville discreta.
# Morfismo terminal (I.9): synthesize_floer_cylinder_germ
#          ≅ objeto inicial de la Fase II (cilindro de Floer / monodromía).
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
class _FloerCylinderGerm:
    """
    Gérmen del cilindro de Floer (objeto terminal de la Fase I).

    Es el objeto inicial de la Fase II: transporta la geometría de Darboux
    (Ω, dim, paso CSMD, piso de regularización) sobre la cual se instancia
    el operador de Cauchy–Riemann perturbado

        ∂̄_{J,H}(u) = ∂_s u + J(u)(∂_t u − X_H(u))

    y el funcional de acción 𝒜_H(γ) = ∫_γ λ − ∫ H dt.

    Atributos
    ---------
    two_n:
        Dimensión de T*Q ≅ ℝ^{2n} (par, estructura de Darboux).
    n:
        Dimensión del espacio de configuración Q.
    omega:
        2-forma canónica Ω = [0 I; −I 0].
    csmd_step:
        Paso imaginario de la diferenciación holomorfa.
    reg_floor:
        Piso de Tikhonov–Higham–Wilkinson.
    form_certificate:
        Certificado (Ωᵀ = −Ω, Ω² = −I, det Ω = 1).
    """

    two_n: int
    n: int
    omega: np.ndarray
    csmd_step: float
    reg_floor: float
    form_certificate: _SymplecticFormCertificate


class _NumericalCore:
    """
    Fase I. Álgebra numérica de precisión metrológica y cálculo holomorfo.

    Provee el topos lineal subyacente: sumación compensada en el álgebra de
    Banach (ℝ, +, ·), la 2-forma de Liouville, el gradiente CSMD (sin
    cancelación sustractiva) y el gérmen que inicia la Fase II.
    """

    # ── I.1  Sumación compensada ──────────────────────────────────────────
    @staticmethod
    def kahan_sum(arr: np.ndarray) -> float:
        """
        Sumación compensada de Kahan.

        Neutraliza el término de redondeo \(c_{k+1}=(t_k-s_k)-y_k\) de modo
        que \(\sum x_i\) sea exacta módulo O(u · Σ|x_i|) en lugar de
        O(n u · max|x_i|).
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
        cancelaciones de signo mixto (crítico en cobordes de Čech).
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

    # Alias histórico (v2.0 usaba el epónimo incorrecto "Neumann").
    kahan_babuska_neumann_sum = kahan_babuska_neumaier_sum

    @staticmethod
    def klein_sum(arr: np.ndarray) -> float:
        """
        Sumación doblemente compensada de Klein (error O(u²) relativo).

        Se reserva para invariantes espectrales críticos (masa nuclear de
        Čech, energía armónica).
        """
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

    # ── I.2  Normas y validación tensorial ────────────────────────────────
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
        """
        2-forma canónica de Liouville Ω ∈ ℝ^{dim×dim}, dim = 2n par.

            Ω = ⎡ 0   I_n ⎤ ,   Ωᵀ = −Ω ,   Ω² = −I ,   Ω⁻¹ = −Ω .
                ⎣−I_n  0  ⎦

        Con esta convención Hamilton se lee X_H = Ω ∇H, i.e.
        q̇ = ∂H/∂p, ṗ = −∂H/∂q.
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

    # ── I.4  Diferenciación holomorfa por paso complejo (CSMD) ────────────
    @staticmethod
    def compute_gradient_csmd(
        func: Callable[[np.ndarray], float],
        x: np.ndarray,
        h: float = _CSMD_STEP,
    ) -> np.ndarray:
        """
        Gradiente CSMD de una función escalar (Lyness–Moler / Squire–Trapp).

            ∂_i f(x) = Im[ f(x + i h e_i) ] / h  +  O(h²)

        No hay cancelación sustractiva: h puede ser ~ 10⁻²⁰. Si `func` no
        acepta entradas complejas (fallo de holomorfía: |·|, max, ReLU),
        se cae a diferencias centrales con paso √ε.
        """
        xv = _NumericalCore.assert_vec("x", np.asarray(x, dtype=np.float64))
        if not np.isfinite(h) or h == 0.0:
            raise ValueError("El paso CSMD h debe ser finito y no nulo.")
        h = float(abs(h))
        dim = xv.size
        grad = np.zeros(dim, dtype=np.float64)
        x_probe = xv.astype(np.complex128)
        holomorphic = True
        try:
            probe = func(x_probe)
            if not np.isfinite(np.real(probe)) and not np.isfinite(np.imag(probe)):
                holomorphic = False
        except (TypeError, ValueError, FloatingPointError):
            holomorphic = False

        if holomorphic:
            for i in range(dim):
                xp = xv.astype(np.complex128)
                xp[i] += 1j * h
                val = func(xp)
                imag = float(np.imag(val))
                if not np.isfinite(imag):
                    holomorphic = False
                    break
                grad[i] = imag / h

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
        """
        Hessiano por diferencia central de gradientes CSMD.

            Hess_{ij} f(x) ≈ [∇f(x + η e_j) − ∇f(x − η e_j)]_i / (2η)

        con η = √ε_mach · max(1, ‖x‖). Se hermitiza al final (Higham).
        """
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
        """
        Campo vectorial hamiltoniano X_H = Ω ∇H(x).

        Con Ω canónica esto reproduce las ecuaciones de Hamilton. El
        resultado es, por construcción, una sección de TT*Q que preserva
        Ω (teorema de Cartan: ℒ_{X_H} Ω = 0 si d²H = 0, siempre cierto).
        """
        xv = cls.assert_vec("x", np.asarray(x, dtype=np.float64))
        dim = xv.size
        if dim % 2 != 0:
            raise ValueError(
                f"X_H exige dimensión par (Darboux); recibido dim={dim}."
            )
        omega = cls.generate_canonical_symplectic_form(dim)
        grad = cls.compute_gradient_csmd(hamiltonian_func, xv, h)
        return omega @ grad

    # ── I.5  Acción de Liouville discreta (puente a Floer) ────────────────
    @staticmethod
    def liouville_action(start_point: np.ndarray, end_point: np.ndarray) -> float:
        """
        Acción de Liouville del segmento geodésico euclídeo γ: start → end.

            𝒜_λ(γ) = ∫_γ p dq  ≈  ((p₀ + p₁)/2) · (q₁ − q₀)

        Es el término topológico del funcional de Floer 𝒜_H = 𝒜_λ − ∫ H dt
        y el objeto que la Fase II refina con el Hamiltoniano y el
        monodromía.
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

    # ── I.9  Morfismo terminal de la Fase I ───────────────────────────────
    @staticmethod
    def synthesize_floer_cylinder_germ(
        dimension_two_n: int,
        csmd_step: float = _CSMD_STEP,
        regularizer: float = _HIGHAM_TIKHONOV_FLOOR,
        scale_matrix: Optional[np.ndarray] = None,
    ) -> _FloerCylinderGerm:
        """
        I.9 — Morfismo terminal de la Fase I / objeto inicial de la Fase II.

        Ensambla el gérmen del cilindro de Floer

            𝒢_I = (2n, n, Ω_{Darboux}, h_CSMD, ε_W, Cert(Ω))

        sobre el cual la Fase II define el operador de Cauchy–Riemann
        perturbado ∂̄_{J,H} y el funcional 𝒜_H. Si se provee
        `scale_matrix` (p. ej. el Jacobiano / monodromía M₃), el piso de
        Wilkinson se adapta a su norma de Frobenius.

        Este método *es* el arranque formal de `_FloerHomologyVerifier`.
        """
        dim = int(dimension_two_n)
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(
                f"dimension_two_n debe ser par y positivo; recibido {dimension_two_n}."
            )
        if not np.isfinite(csmd_step) or csmd_step == 0.0:
            raise ValueError("csmd_step debe ser finito y no nulo.")
        omega = _NumericalCore.generate_canonical_symplectic_form(dim)
        certificate = _NumericalCore.certify_symplectic_form(omega)
        if not certificate.is_darboux:
            logger.warning(
                "Certificado de Darboux degradado: skew=%.3e, J²+I=%.3e, det=%.16f",
                certificate.skew_residual,
                certificate.almost_complex_residual,
                certificate.determinant,
            )
        floor = max(float(regularizer), _HIGHAM_TIKHONOV_FLOOR)
        if scale_matrix is not None:
            floor = max(floor, _NumericalCore.wilkinson_deflation_floor(np.asarray(scale_matrix)))
        return _FloerCylinderGerm(
            two_n=dim,
            n=dim // 2,
            omega=omega,
            csmd_step=float(abs(csmd_step)),
            reg_floor=float(floor),
            form_certificate=certificate,
        )


# =============================================================================
# FASE II — HOMOLOGÍA DE FLOER, CONLEY–ZEHNDER Y LIFTING AL NERVIO
# -----------------------------------------------------------------------------
# Continúa I.9: todo verificador se instancia desde un FloerCylinderGerm.
# Morfismo terminal (II.7): induce_cech_nerve_germ
#          ≅ objeto inicial de la Fase III (Gram / conexión de Čech).
# =============================================================================
@dataclass(frozen=True)
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


@dataclass(frozen=True)
class _CechNerveGerm:
    """
    Gérmen del nervio atencional (objeto terminal de la Fase II).

    Es el objeto inicial de la Fase III: la transformada de Cayley del
    monodromía simpléctico M₃ produce una matriz hermítica

        K = Higham( −Ω (M−I)(M+I)⁺ )

    que se interpreta como Gram de secciones locales (o, si es
    predominantemente antisimétrica, como 1-cociclo de conexión) sobre
    el nervio del recubrimiento semántico.
    """

    sheaf_gram: np.ndarray
    cayley_condition: float
    two_n: int
    reg_floor: float
    from_floer: bool


class _FloerHomologyVerifier:
    """
    Fase II. Verificador de homología de Floer.

    Continúa el gérmen 𝒢_I. Sobre el cilindro ℝ × S¹ mide:

    * el residuo discreto de ∂̄_{J,H} (cuerda + defecto simpléctico de M₃);
    * el funcional de acción (Liouville + cuerda);
    * el índice de Conley–Zehnder / Robbin–Salamon del monodromía;
    * la no-degeneración (distancia de spec(M₃) al ciclo de Maslov {1}).

    El Jacobiano `jacobian_m3` se interpreta como la aplicación de
    Poincaré / monodromía del flujo hamiltoniano linealizado a tiempo 1.
    """

    def __init__(self, germ: _FloerCylinderGerm) -> None:
        self._germ = germ

    @property
    def germ(self) -> _FloerCylinderGerm:
        """Gérmen de Fase I del que este verificador es continuación."""
        return self._germ

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
                f"jacobian_m3 es {M.shape[0]}×{M.shape[0]} pero los extremos tienen dim {z0.size}."
            )
        return z0.astype(np.float64, copy=False), z1.astype(np.float64, copy=False), np.asarray(M, dtype=np.float64)

    def monodromy_symplectic_residual(self, jacobian_m3: np.ndarray) -> float:
        """‖Mᵀ Ω M − Ω‖_F : defecto de pertenencia a Sp(2n, ℝ)."""
        M = np.asarray(jacobian_m3, dtype=np.float64)
        omega = self._adapt_omega(M.shape[0])
        residual = M.T @ omega @ M - omega
        return _NumericalCore.frobenius_norm(residual)

    def _adapt_omega(self, dim: int) -> np.ndarray:
        if dim == self._germ.two_n:
            return self._germ.omega
        return _NumericalCore.generate_canonical_symplectic_form(dim)

    def maslov_degeneracy(self, jacobian_m3: np.ndarray) -> float:
        """
        Distancia de spec(M) a {1}: min_i |λ_i(M) − 1|.

        Cero sii M yace en el ciclo de Maslov (órbita degenerada, el
        índice de Conley–Zehnder no está definido como entero).
        """
        ev = la.eigvals(np.asarray(jacobian_m3, dtype=np.float64))
        return float(np.min(np.abs(ev - 1.0)))

    def conley_zehnder_index(self, jacobian_m3: np.ndarray) -> float:
        """
        Índice de Conley–Zehnder / Robbin–Salamon del monodromía M ∈ Sp(2n).

        Polar M = UP; la parte U se proyecta a U(n) vía la identificación
        ℝ^{2n} ≅ ℂ^n, (q, p) ↔ q + i p. Si U_ℂ = X + i Y,

            CZ(M) ≈ (1/π) Σ_k Arg(λ_k(U_ℂ))   ∈ ℝ

        (convención de Robbin–Salamon: semientero si 1 ∈ spec(M)).
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
        # Proyección de Higham al grupo unitario.
        try:
            uu, ss, vv = la.svd(u_c, full_matrices=False)
            u_c = uu @ vv
        except (np.linalg.LinAlgError, ValueError):
            pass
        ev = la.eigvals(u_c)
        ev = ev / np.maximum(np.abs(ev), _MACHINE_EPS)
        angles = np.angle(ev)
        cz = float(_NumericalCore.kahan_babuska_neumaier_sum(angles) / np.pi)
        if self.maslov_degeneracy(M) <= _MASLOV_DEGENERACY:
            # Cruce: Robbin–Salamon aporta un semientero.
            cz += 0.5 * np.sign(cz) if cz != 0.0 else 0.5
        return cz

    def verify(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        jacobian_m3: np.ndarray,
        hamiltonian_func: Optional[Callable[[np.ndarray], float]] = None,
    ) -> _FloerResult:
        """
        Certifica el cilindro de Floer discreto (start ⇝ end; M₃).

        Residuo (refinamiento monótono de v2.0):

            R = ‖end−start‖ (1 + ‖M₃‖_F)  +  ‖M₃ᵀ Ω M₃ − Ω‖_F

        El segundo sumando se anula sobre Sp(2n), de modo que en monodromías
        exactas se recupera el residual 2.0. La acción reportada en la API
        sigue siendo la longitud de cuerda; la acción de Liouville y la
        energía de Dirichlet viven en el certificado.
        """
        z0, z1, M = self._coerce_pair(start_point, end_point, jacobian_m3)

        chord = _NumericalCore.euclidean_norm(z1 - z0)
        m_norm = _NumericalCore.frobenius_norm(M)
        sp_res = self.monodromy_symplectic_residual(M)
        floer_residual = float(chord * (1.0 + m_norm) + sp_res)

        liouville = _NumericalCore.liouville_action(z0, z1)
        if hamiltonian_func is not None:
            # Regla del trapecio sobre el segmento (start, end).
            try:
                h0 = float(np.real(hamiltonian_func(z0)))
                h1 = float(np.real(hamiltonian_func(z1)))
                if np.isfinite(h0) and np.isfinite(h1):
                    liouville = liouville - 0.5 * (h0 + h1)
            except (TypeError, ValueError, FloatingPointError) as exc:
                logger.warning("Hamiltoniano no evaluable en los extremos: %s", exc)

        dirichlet = 0.5 * float(_NumericalCore.kahan_babuska_neumaier_sum((z1 - z0) ** 2))
        deg = self.maslov_degeneracy(M)
        cz = self.conley_zehnder_index(M)
        scale_sp = max(_NumericalCore.frobenius_norm(self._adapt_omega(M.shape[0])), 1.0)
        is_sp = sp_res <= max(_WILKINSON_DRIFT_LIMIT, _WILKINSON_DRIFT_LIMIT * scale_sp * (m_norm ** 2 + 1.0))

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

    # ── II.7  Morfismo terminal de la Fase II ─────────────────────────────
    def induce_cech_nerve_germ(
        self,
        jacobian_m3: np.ndarray,
        start_point: Optional[np.ndarray] = None,
        end_point: Optional[np.ndarray] = None,
    ) -> _CechNerveGerm:
        """
        II.7 — Morfismo terminal de la Fase II / objeto inicial de la Fase III.

        Cuantiza el monodromía de Floer como Gram / conexión de Čech vía
        la transformada de Cayley regularizada

            W = (M − I)(M + I)⁺ ,   K = Higham(−Ω W).

        K ∈ Herm(2n) es el objeto que la Fase III toma como matriz de
        haz atencional (secciones locales / 1-cociclo). Si se proveen
        los extremos, se usa su dimensión como chequeo de coherencia.

        Este método *es* el arranque formal de `_AttentionCechCohomology`.
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
        floor = max(self._germ.reg_floor, _NumericalCore.wilkinson_deflation_floor(M + ident))
        pinv_plus, _s, cond = _NumericalCore.tikhonov_higham_pinv(
            M + ident, rel_floor=_MACHINE_EPS, abs_floor=floor
        )
        w_cayley = (M - ident) @ pinv_plus
        omega = self._adapt_omega(dim)
        gram = _NumericalCore.higham_nearest_hermitian(-omega @ w_cayley)
        gram = np.real(gram).astype(np.float64, copy=False)
        return _CechNerveGerm(
            sheaf_gram=gram,
            cayley_condition=float(cond),
            two_n=dim,
            reg_floor=float(floor),
            from_floer=True,
        )


# =============================================================================
# FASE III — COHOMOLOGÍA DE ČECH ATENCIONAL, HODGE Y BETTI
# -----------------------------------------------------------------------------
# Continúa II.7: el calculador se ancla a un CechNerveGerm (o lo induce
# por hermitización directa de una matriz de haz cruda).
# =============================================================================
@dataclass(frozen=True)
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


class _AttentionCechCohomology:
    """
    Fase III. Cohomología de Čech del haz atencional.

    Continúa el gérmen 𝒢_II cuando se provee. Interpreta la matriz de
    haz A de dos modos complementarios (topos de haces sobre el nervio):

    * **Gram / Laplace** (A ≈ A†): Laplaciano de Hodge Δ = δδ* + δ*δ
      del grafo umbralizado; β₀ = dim ker Δ, β₁ = e − v + β₀.
    * **Conexión / curvatura** (A ≈ −A†): A se lee como 1-cochain ω y
      (δω)_{ijk} = ω_{jk} − ω_{ik} + ω_{ij} es la obstrucción en Ȟ².

    La obstrucción reportada por la API 2.0 permanece siendo la masa
    nuclear de los modos activos (suma KBN de valores singulares sobre
    el piso de Wilkinson), invariante espectral estable.
    """

    def __init__(
        self,
        germ: Optional[_CechNerveGerm] = None,
        regularizer: float = _HIGHAM_TIKHONOV_FLOOR,
    ) -> None:
        self._germ = germ
        self._reg = max(float(regularizer), _HIGHAM_TIKHONOV_FLOOR)

    def _resolve_matrix(self, attention_sheaf_matrix: np.ndarray) -> np.ndarray:
        a = np.asarray(attention_sheaf_matrix)
        if a.size == 0 and self._germ is not None:
            return np.asarray(self._germ.sheaf_gram, dtype=np.float64)
        if a.ndim == 1:
            side = int(np.sqrt(a.size))
            if side * side != a.size:
                raise ValueError("attention_sheaf_matrix plana no es un cuadrado perfecto.")
            a = a.reshape(side, side)
        _NumericalCore.assert_square("attention_sheaf_matrix", a)
        _NumericalCore.assert_finite("attention_sheaf_matrix", a)
        return np.asarray(a, dtype=np.complex128)

    def cech_coboundary_defect(self, omega: np.ndarray) -> float:
        """
        ‖δω‖² del 1-cochain antisimétrico, con

            (δω)_{ijk} = ω_{jk} − ω_{ik} + ω_{ij} ,   i < j < k.

        Es idénticamente nulo si ω = δf (lema de Poincaré combinatorio /
        δ² = 0). Para N > `_CECH_TRIPLE_CAP` se muestrea un subconjunto
        determinista de tríadas (stride regular) para no degradar la FPU.
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
        """
        Espectro del Laplaciano de Hodge combinatorio del nervio.

        El nervio se obtiene umbralizando |A_{ij}| > floor (i ≠ j).
        Δ_0 = D − W (Laplaciano simétrico de Higham) actúa en C⁰.
        Devuelve (eigenvalues, active_modes, β₀, β₁, harmonic_energy).
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
        # β₀ = multiplicidad numérica del núcleo.
        ker_tol = max(floor, _WILKINSON_DEFLATION_FLOOR * max(n, 1))
        betti_0 = int(np.sum(evals <= ker_tol))
        n_edges = int(np.sum(np.triu(adjacency, 1)))
        betti_1 = int(max(n_edges - n + betti_0, 0))
        active = evals[evals > ker_tol]
        # Energía armónica de 0-cochains: masa del núcleo (debe ~ 0).
        harmonic = _NumericalCore.kahan_babuska_neumaier_sum(np.clip(evals[: max(betti_0, 0)], 0.0, None))
        return evals, active, betti_0, betti_1, float(harmonic)

    def compute(self, attention_sheaf_matrix: np.ndarray) -> _CechCohomologyResult:
        """
        Calcula la clase de obstrucción de Čech y los modos activos.

        Pipeline:
        1. Resolución de la matriz (cruda o gérmen de Cayley-Floer).
        2. Hermitización de Higham y SVD (modos de haz).
        3. Deflación de Wilkinson → modos activos.
        4. Masa nuclear por KBN  (= `cech_obstruction` de la API 2.0).
        5. Defecto de coborde δω y espectro de Hodge (certificado).
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

        return _CechCohomologyResult(
            cech_obstruction=float(nuclear),
            active_modes=np.asarray(active_modes, dtype=np.float64),
            cocycle_defect=float(cocycle),
            harmonic_energy=float(harmonic),
            betti_0=int(betti_0),
            betti_1=int(betti_1),
            effective_rank=rank,
            nuclear_mass=float(nuclear),
        )


# =============================================================================
# MOTOR PRINCIPAL — INTEGRACIÓN DEL MORFISMO Φ_III ∘ Φ_II ∘ Φ_I
# =============================================================================
class ImperialEruditosEngine:
    """
    Motor de alta precisión para la rigidez de Floer y Čech sobre el
    espacio de fase de los logits de atención semántica.

    Compone las tres fases anidadas:

    1. Fase I   — gérmen de Darboux / CSMD (`_NumericalCore`).
    2. Fase II  — cilindro de Floer, CZ, Cayley (`_FloerHomologyVerifier`).
    3. Fase III — nervio de Čech / Hodge (`_AttentionCechCohomology`),
                  inicializado con un gérmen de referencia (identidad);
                  cada llamada puede sustituirlo por la matriz que se le pase.

    La API pública de 2.0 se conserva (tuplas). Los métodos `*_certified`
    exponen los invariantes añadidos en 3.0.
    """

    def __init__(self, regularizer: float = 1e-15) -> None:
        """
        Inicializa el motor y materializa el encadenamiento de gérmenes.

        El regularizador se mantiene al menos en `_HIGHAM_TIKHONOV_FLOOR`
        y alimenta el piso de Wilkinson, el Cayley del monodromía y el
        Laplaciano de Hodge (en 2.0 se almacenaba y no se usaba).

        El gérmen de Fase I se instancia sobre T*Q de dimensión mínima
        de Darboux (2n = 2) y se re-sintetiza bajo demanda cuando llegan
        trayectorias o haces de dimensión mayor.

        Args:
            regularizer: Piso de Tikhonov contra polos espectrales.
        """
        self._reg: Final[float] = max(float(regularizer), _HIGHAM_TIKHONOV_FLOOR)
        # Fase I → objeto inicial de Fase II (dimensión mínima; se adapta).
        self._floer_germ: _FloerCylinderGerm = (
            _NumericalCore.synthesize_floer_cylinder_germ(
                2, csmd_step=_CSMD_STEP, regularizer=self._reg
            )
        )
        self._floer_verifier = _FloerHomologyVerifier(self._floer_germ)
        # Fase II → objeto inicial de Fase III (Cayley de I_2).
        self._cech_germ: _CechNerveGerm = self._floer_verifier.induce_cech_nerve_germ(
            np.eye(2, dtype=np.float64)
        )
        self._cech_calculator = _AttentionCechCohomology(
            germ=self._cech_germ, regularizer=self._reg
        )

    def _resync_floer_germ(self, two_n: int, scale_matrix: Optional[np.ndarray] = None) -> None:
        """Re-sintetiza 𝒢_I cuando cambia la dimensión de Darboux."""
        if two_n == self._floer_germ.two_n and scale_matrix is None:
            return
        self._floer_germ = _NumericalCore.synthesize_floer_cylinder_germ(
            two_n,
            csmd_step=self._floer_germ.csmd_step,
            regularizer=self._reg,
            scale_matrix=scale_matrix,
        )
        self._floer_verifier = _FloerHomologyVerifier(self._floer_germ)

    # ── Fase I expuesta ───────────────────────────────────────────────────
    def kahan_sum(self, arr: np.ndarray) -> float:
        """Sumación compensada de Kahan expuesta públicamente."""
        return _NumericalCore.kahan_sum(arr)

    def kahan_babuska_neumaier_sum(self, arr: np.ndarray) -> float:
        """Sumación KBN (expuesta públicamente)."""
        return _NumericalCore.kahan_babuska_neumaier_sum(arr)

    def compute_symplectic_gradient(
        self,
        hamiltonian_func: Callable[[np.ndarray], float],
        x: np.ndarray,
        h: float = _CSMD_STEP,
    ) -> np.ndarray:
        """Campo vectorial simpléctico X_H = Ω ∇H(x) usando CSMD."""
        xv = _NumericalCore.assert_vec("x", np.asarray(x, dtype=np.float64))
        if xv.size % 2 == 0:
            self._resync_floer_germ(xv.size)
        return _NumericalCore.compute_symplectic_gradient(hamiltonian_func, xv, h)

    def compute_hessian_csmd(
        self,
        hamiltonian_func: Callable[[np.ndarray], float],
        x: np.ndarray,
        h: float = _CSMD_STEP,
    ) -> np.ndarray:
        """Hessiano de H por CSMD + diferencia central (Fase I)."""
        return _NumericalCore.compute_hessian_csmd(hamiltonian_func, x, h)

    def floer_cylinder_germ_certificate(self) -> _SymplecticFormCertificate:
        """Certificado de Darboux del gérmen de Fase I vigente."""
        return self._floer_germ.form_certificate

    # ── Fase II expuesta ──────────────────────────────────────────────────
    def verify_floer_homology_trajectory(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        jacobian_m3: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Verificación de la trayectoria de Floer.
        Retorna (floer_residual, action_potential). API 2.0.
        """
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
            self._resync_floer_germ(z0.size, scale_matrix=np.asarray(jacobian_m3))
        return self._floer_verifier.verify(
            start_point, end_point, jacobian_m3, hamiltonian_func=hamiltonian_func
        )

    def induce_cech_nerve_germ(
        self,
        jacobian_m3: np.ndarray,
        start_point: Optional[np.ndarray] = None,
        end_point: Optional[np.ndarray] = None,
    ) -> _CechNerveGerm:
        """
        Réplica pública del morfismo II.7: monodromía M₃ ↦ Gram de Čech.
        Actualiza el gérmen con el que opera la Fase III.
        """
        M = np.asarray(jacobian_m3)
        if M.ndim == 1:
            side = int(np.sqrt(M.size))
            M = M.reshape(side, side)
        if M.shape[0] % 2 == 0:
            self._resync_floer_germ(M.shape[0], scale_matrix=M)
        germ = self._floer_verifier.induce_cech_nerve_germ(
            M, start_point=start_point, end_point=end_point
        )
        self._cech_germ = germ
        self._cech_calculator = _AttentionCechCohomology(
            germ=germ, regularizer=self._reg
        )
        return germ

    # ── Fase III expuesta ─────────────────────────────────────────────────
    def compute_attention_cech_cohomology(
        self,
        attention_sheaf_matrix: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Cálculo de la cohomología atencional de Čech.
        Retorna (cech_obstruction, active_modes). API 2.0.
        """
        result = self._cech_calculator.compute(attention_sheaf_matrix)
        return result.cech_obstruction, result.active_modes

    def compute_attention_cech_cohomology_certified(
        self,
        attention_sheaf_matrix: np.ndarray,
    ) -> _CechCohomologyResult:
        """Čech con defecto de coborde, Hodge, Betti y masa nuclear."""
        return self._cech_calculator.compute(attention_sheaf_matrix)


__all__ = ["ImperialEruditosEngine"]