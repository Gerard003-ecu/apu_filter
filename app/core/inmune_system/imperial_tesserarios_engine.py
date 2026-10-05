# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Imperial Tesserarios Engine (Caballos de Batalla Homotópicos FPU)   ║
║ Ruta   : app/core/inmune_system/imperial_tesserarios_engine.py               ║
║ Versión: 4.1.0-Poincare-Nested-Phases-KAM-Floquet-Melnikov-Williamson-Krein  ║
╚══════════════════════════════════════════════════════════════════════════════╝
SINOPSIS MATEMÁTICA Y METROLOGÍA DE LA FPU:
Motor homotópico de-confinado que custodia la invarianza por deformación
continua de las transiciones de la Malla. Realiza la proyección simpléctica
de Quillen (iteración polar estructurada de Higham–Mackey–Tisseur), calcula
el tensor de homotopía m₃ en asociaedros de Stasheff, evalúa la obstrucción
de Čech–Deligne para gerbes no abelianos y despliega la maquinaria completa
de la **mecánica celeste de Henri Poincaré**, ahora con refinamiento 4.1.0:

  * Sección transversal Σ ⊂ T*Q con certificación *dinámica*
    n_Σ · X_H ≠ 0 (no meramente Ω n_Σ ≠ 0, tautológica por no-degeneración).
  * 1-forma de Poincaré–Cartan ϑ = p dq − H dt y su acción de periodo.
  * Función generatriz de Hamilton–Jacobi F₂(q, P) con hessiana simetrizada
    (cerradura d²F₂ = 0 ⇔ canonicidad).
  * Gram–Schmidt simpléctico de Parasjuk–de Gosson con completación canónica
    w ← −Ωv cuando el par degenera.
  * Forma de volumen de Liouville Ωⁿ / n! y pairing compensado uᵀΩv.
  * Clasificación de Williamson del equilibrio (elíptico / hiperbólico /
    foco–foco) vía el espectro de la matriz hamiltoniana K = J Hess H.
  * Factorización de Floquet–Lyapunov M = exp(T·A_F)·R_F con logaritmo real
    por Schur y clasificación de Krein–Moser de los multiplicadores.
  * Reducción de la monodromía al mapa de Poincaré (2n−2)×(2n−2)
    (se extrae el bloque parabólico μ = 1 doble).
  * Función de Mel'nikov ℳ(t₀) con ceros *simples* (signo ⊗ |ℳ′| > 0).
  * Número de rotación ρ ∈ ℝ/ℤ (mod 2π) por fracción continua de Farey.
  * Condición de twist de Moser ∂ρ/∂I ≠ 0 (Poincaré–Birkhoff).
  * Ecuación homológica de Birkhoff (pequeños divisores k·ω).
  * Detección KAM diofántica |k·ω| ≥ γ / |k|^τ sobre retículos adaptativos.
  * Tiempo de estabilidad de Nekhoroshev T ~ exp(c ε^{−1/(2n)}).
  * Espectro de Lyapunov por QR de Benettin con *emparejamiento hamiltoniano*
    λᵢ ↔ −λ_{2n+1−i}, Kaplan–Yorke corregido y entropía de Pesin.

Tres fases anidadas (el objeto terminal de Φₖ es el objeto inicial de Φₖ₊₁):

  Φ_I   : _NumericalCore.synthesize_poincare_darboux_germ
            → _PoincareDarbouxGerm          ≡  germen inicial de Φ_II
  Φ_II  : _SymplecticProjector.induce_poincare_floquet_germ
            → _PoincareFloquetGerm          ≡  germen inicial de Φ_III
  Φ_III : _PoincareMonodromyAnalyzer.compute_certificate
            → PoincareMonodromyCertificate
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final, Iterator, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Core.ImperialTesserariosEngine")

__version__: Final[str] = (
    "4.1.0-Poincare-Nested-Phases-KAM-Floquet-Melnikov-Williamson-Krein"
)

# =============================================================================
# CONSTANTES DE PRECISIÓN METROLÓGICA Y TESSERARIOS DE POINCARÉ
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
_WILKINSON_LIMIT: Final[float] = 1e-12
_SPECTRAL_TOL: Final[float] = 1e-8

# ── Constantes de Poincaré / KAM / Floquet / Nekhoroshev ────────────────────
_DIOPHANTINE_GAMMA_FLOOR: Final[float] = 1e-6
_DIOPHANTINE_TAU_MIN: Final[float] = 2.0 + 1e-6   # τ > n−1 (n=2 ⇒ τ>1)
_KAM_HARMONIC_CAP: Final[int] = 32
_KAM_LATTICE_CELL_CAP: Final[int] = 180_000
_MELNIKOV_PHASE_SAMPLES: Final[int] = 64
_LYAPUNOV_QR_ITERATIONS: Final[int] = 4096
_ROTATION_CF_DEPTH: Final[int] = 24
_KREIN_UNIT_TOL: Final[float] = 1e-8
_PARABOLIC_MULTIPLIER_TOL: Final[float] = 1e-6
_TWIST_FLOOR: Final[float] = 1e-10
_TWO_PI: Final[float] = 2.0 * float(np.pi)
_NEKHOROSHEV_PREFACTOR: Final[float] = 0.5
_WILLIAMSON_IMAG_RATIO: Final[float] = 1e-8


class SymplecticDimensionError(ValueError):
    """Dimensión impar incompatible con Sp(2n, ℝ)."""


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████████
# ██  FASE I — NÚCLEO DE BANACH, DARBOUX, CARTAN Y SECCIÓN DE POINCARÉ       ██
# ██  Morfismo terminal (I.9): synthesize_poincare_darboux_germ              ██
# ██                           → objeto inicial de la FASE II                ██
# ██████████████████████████████████████████████████████████████████████████████
# =============================================================================

# -----------------------------------------------------------------------------
# I.0 — Certificados y gérmenes de la Fase I
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class _SymplecticFormCertificate:
    """Certificado algebraico de la 2-forma de Liouville–Darboux."""

    skew_residual: float
    almost_complex_residual: float
    determinant: float
    frobenius_norm: float
    is_darboux: bool
    liouville_volume: float = 1.0


@dataclass(frozen=True)
class _WilliamsonSpectrum:
    r"""
    Clasificación de Williamson de un equilibrio hamiltoniano.

    La matriz hamiltoniana K = J A, J = Ω⁻¹ = −Ω, A = Hess H, tiene
    espectro cerrado por (λ, −λ, λ̄, −λ̄). Se cuentan los pares:

      * elípticos   : ±iω, ω > 0
      * hiperbólicos: ±λ,  λ ∈ ℝ \ {0}
      * foco–foco   : ±α ± iβ, αβ ≠ 0
      * parabólicos : λ = 0 (en el núcleo / resonancia)
    """

    elliptic_pairs: int
    hyperbolic_pairs: int
    focus_focus_pairs: int
    parabolic_multiplicity: int
    hamiltonian_eigenvalues: np.ndarray
    is_linearly_stable: bool


@dataclass(frozen=True)
class _PoincareDarbouxGerm:
    r"""
    Gérmen de Poincaré–Darboux.

    **Objeto terminal de la FASE I** y, por tanto, **objeto inicial de la
    FASE II**. Encapsula la geometría simpléctica canónica junto con la
    sección transversal de Poincaré Σ = { q_k = q_k^* } sobre la cual se
    definirá el mapa de primer retorno P_Σ : Σ → Σ.

    La transversalidad *dinámica* de Poincaré exige n_Σ · X_H ≠ 0; la
    no-degeneración Ω n_Σ ≠ 0 es automática (Ω no degenerada) y se
    conserva sólo como test de sanity del chart de Darboux.
    """

    two_n: int
    n: int
    omega: np.ndarray
    reg_floor: float
    max_iter: int
    tol: float
    form_certificate: _SymplecticFormCertificate
    section_normal: np.ndarray
    section_index: int
    section_offset: float
    energy_level: float
    is_section_transversal: bool
    flow_vector: Optional[np.ndarray] = None
    liouville_volume: float = 1.0


# -----------------------------------------------------------------------------
# I.1 — Núcleo numérico
# -----------------------------------------------------------------------------
class _NumericalCore:
    """
    Fase I. Álgebra numérica de precisión metrológica y geometría
    simpléctica de Poincaré. Provee el topos lineal subyacente:
    sumación compensada en el álgebra de Banach, 2-forma de Liouville,
    involución de Cartan del par (GL, Sp), sección transversal Σ,
    función generatriz de Hamilton–Jacobi, Gram–Schmidt simpléctico,
    forma de Poincaré–Cartan y clasificación de Williamson.
    """

    # ── I.1.1  Sumación compensada ────────────────────────────────────────
    @staticmethod
    def kahan_sum(arr: np.ndarray) -> float:
        total = 0.0
        c = 0.0
        for x in np.asarray(arr, dtype=np.float64).ravel():
            if not np.isfinite(x):
                raise ValueError("kahan_sum: no-finito.")
            y = float(x) - c
            t = total + y
            c = (t - total) - y
            total = t
        return float(total)

    @staticmethod
    def kahan_babuska_neumaier_sum(arr: np.ndarray) -> float:
        total = 0.0
        c = 0.0
        for x in np.asarray(arr, dtype=np.float64).ravel():
            xf = float(x)
            if not np.isfinite(xf):
                raise ValueError("kbn_sum: no-finito.")
            t = total + xf
            if abs(total) >= abs(xf):
                c += (total - t) + xf
            else:
                c += (xf - t) + total
            total = t
        return float(total + c)

    @staticmethod
    def klein_sum(arr: np.ndarray) -> float:
        s = 0.0
        cs = 0.0
        ccs = 0.0
        for x in np.asarray(arr, dtype=np.float64).ravel():
            xf = float(x)
            if not np.isfinite(xf):
                raise ValueError("klein_sum: no-finito.")
            t = s + xf
            c = (s - t) + xf if abs(s) >= abs(xf) else (xf - t) + s
            s = t
            t = cs + c
            cc = (cs - t) + c if abs(cs) >= abs(c) else (c - t) + cs
            cs = t
            ccs += cc
        return float(s + cs + ccs)

    # ── I.1.2  Normas, pairing y validación ───────────────────────────────
    @staticmethod
    def frobenius_norm(matrix: np.ndarray) -> float:
        a = np.asarray(matrix)
        return 0.0 if a.size == 0 else float(la.norm(a, "fro"))

    @staticmethod
    def assert_finite(name: str, array: np.ndarray) -> None:
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contiene entradas no finitas.")

    @staticmethod
    def assert_square(name: str, matrix: np.ndarray,
                      dim: Optional[int] = None) -> None:
        a = np.asarray(matrix)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError(f"{name} debe ser cuadrada; recibido {a.shape}.")
        if dim is not None and a.shape[0] != dim:
            raise ValueError(f"{name} debe ser {dim}×{dim}; recibido {a.shape}.")

    @staticmethod
    def higham_nearest_hermitian(matrix: np.ndarray) -> np.ndarray:
        a = np.asarray(matrix)
        _NumericalCore.assert_square("higham_nearest_hermitian", a)
        return 0.5 * (a + a.T.conj())

    @staticmethod
    def skew_residual(matrix: np.ndarray) -> float:
        a = np.asarray(matrix)
        return _NumericalCore.frobenius_norm(a + a.T)

    @staticmethod
    def symplectic_pairing(
        u: np.ndarray, v: np.ndarray, omega: np.ndarray,
    ) -> float:
        r"""Producto simpléctico compensado ω(u, v) = uᵀ Ω v (KBN)."""
        uu = np.asarray(u, dtype=np.float64).ravel()
        vv = np.asarray(v, dtype=np.float64).ravel()
        ov = np.asarray(omega, dtype=np.float64) @ vv
        if uu.size != ov.size:
            raise ValueError("symplectic_pairing: dimensiones incompatibles.")
        return _NumericalCore.kahan_babuska_neumaier_sum(uu * ov)

    @staticmethod
    def tikhonov_higham_pinv(
        matrix: np.ndarray,
        rel_floor: float = _MACHINE_EPS,
        abs_floor: float = _REG_FLOOR_TIKHONOV,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        a = np.asarray(matrix)
        if a.size == 0:
            return a.copy(), np.array([], dtype=np.float64), float("inf")
        u_svd, s_vals, vt = la.svd(a, full_matrices=False)
        if s_vals.size == 0:
            return (np.zeros((a.shape[1], a.shape[0]), dtype=a.dtype),
                    s_vals, float("inf"))
        lam = max(float(abs_floor), float(rel_floor) * float(s_vals[0]))
        s_inv = np.zeros_like(s_vals)
        live = s_vals > lam
        s_inv[live] = s_vals[live] / (s_vals[live] ** 2 + lam ** 2)
        pinv = (vt.T.conj() * s_inv) @ u_svd.T.conj()
        s_min_live = float(s_vals[live].min()) if np.any(live) else lam
        cond = float(s_vals[0] / max(s_min_live, _MACHINE_EPS))
        return pinv, s_vals, cond

    # ── I.1.3  2-forma simpléctica canónica de Darboux ────────────────────
    @staticmethod
    def generate_canonical_symplectic_form(dim: int) -> np.ndarray:
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(
                f"dim={dim} debe ser par y positivo (Darboux).")
        half = dim // 2
        omega = np.zeros((dim, dim), dtype=np.float64)
        omega[:half, half:] = np.eye(half, dtype=np.float64)
        omega[half:, :half] = -np.eye(half, dtype=np.float64)
        return omega

    @staticmethod
    def liouville_volume(omega: np.ndarray) -> float:
        r"""
        Volumen de Liouville vol = Ωⁿ / n!  (Pfaffiano de Ω, n = dim/2).

        Para la forma canónica de Darboux, Pf(Ω) = 1 ⇒ vol = 1.
        Se evalúa como (det Ω)^{1/2} con signo del Pfaffiano canónico.
        """
        w = np.asarray(omega, dtype=np.float64)
        _NumericalCore.assert_square("omega", w)
        dim = w.shape[0]
        if dim % 2 != 0:
            raise SymplecticDimensionError(
                "Liouville exige dimensión par.")
        det_o = float(np.real(la.det(w)))
        # det Ω = 1 para Darboux; sqrt(det) = |Pf(Ω)|
        return float(np.sqrt(max(det_o, 0.0)))

    @staticmethod
    def certify_symplectic_form(omega: np.ndarray) -> _SymplecticFormCertificate:
        _NumericalCore.assert_square("omega", omega)
        dim = omega.shape[0]
        ident = np.eye(dim, dtype=omega.dtype)
        skew = _NumericalCore.skew_residual(omega)
        almost_c = _NumericalCore.frobenius_norm(omega @ omega + ident)
        det_o = float(np.real(la.det(omega)))
        fro = _NumericalCore.frobenius_norm(omega)
        scale = max(fro, 1.0)
        vol = _NumericalCore.liouville_volume(omega)
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
            liouville_volume=float(vol),
        )

    @staticmethod
    def wilkinson_deflation_floor(matrix: np.ndarray) -> float:
        if matrix is None or np.asarray(matrix).size == 0:
            return _WILKINSON_DEFLATION_FLOOR
        fro_norm = _NumericalCore.frobenius_norm(matrix)
        return float(max(
            fro_norm * _MACHINE_EPS * _WILKINSON_DEFLATION_SCALE,
            _WILKINSON_DEFLATION_FLOOR,
        ))

    @staticmethod
    def symplectic_inverse(S: np.ndarray, omega: np.ndarray) -> np.ndarray:
        r"""Inversa de Sp(2n): S⁻¹ = −Ω Sᵀ Ω  (pues Ω⁻¹ = −Ω)."""
        return -omega @ np.asarray(S).T @ omega

    @staticmethod
    def cartan_involution_image(
        M: np.ndarray, omega: np.ndarray, floor: float,
    ) -> Tuple[np.ndarray, bool]:
        r"""
        Involución de Cartan del par simétrico (GL(2n), Sp(2n)):
            J(X) = Ω X⁻ᵀ Ωᵀ .
        Punto fijo J(X) = X  ⇔  X ∈ Sp(2n) (en la componente det = 1).
        """
        m = np.asarray(M, dtype=np.float64)
        dim = m.shape[0]
        ident = np.eye(dim, dtype=np.float64)
        try:
            x_inv_t = la.solve(m.T, ident, assume_a="gen")
            return omega @ x_inv_t @ omega.T, True
        except (np.linalg.LinAlgError, ValueError) as exc:
            logger.warning("Solve de Cartan fallido (%s); pinv Tikhonov.", exc)
            pinv, _s, _k = _NumericalCore.tikhonov_higham_pinv(
                m.T, rel_floor=_MACHINE_EPS, abs_floor=floor)
            return omega @ pinv @ omega.T, False

    @staticmethod
    def symplectic_residual(M: np.ndarray, omega: np.ndarray) -> float:
        m = np.asarray(M, dtype=np.float64)
        return _NumericalCore.frobenius_norm(m.T @ omega @ m - omega)

    # ── I.1.4  Geometría de la sección transversal de Poincaré ────────────
    @staticmethod
    def build_poincare_section(
        two_n: int, section_index: int = 0,
    ) -> np.ndarray:
        r"""
        Normal unitaria n_Σ ∈ ℝ^{2n} de la sección
            Σ = { (q, p) ∈ T*Q : q_{section_index} = q^* }.
        Se elige n_Σ = e_{section_index} (vector canónico del bloque q).
        """
        if two_n <= 0 or two_n % 2 != 0:
            raise ValueError("two_n debe ser par positivo.")
        n = two_n // 2
        if not (0 <= section_index < n):
            raise ValueError(
                f"section_index={section_index} fuera de rango [0,{n-1}].")
        n_sigma = np.zeros(two_n, dtype=np.float64)
        n_sigma[section_index] = 1.0
        return n_sigma

    @staticmethod
    def certify_section_transversality(
        omega: np.ndarray,
        section_normal: np.ndarray,
        flow_vector: Optional[np.ndarray] = None,
    ) -> bool:
        r"""
        Transversalidad *dinámica* de Poincaré:

            n_Σ · X_H ≠ 0  ⇔  el flujo corta Σ transversalmente.

        Si no se suministra X_H, se cae al test de no-degeneración
        Ω n_Σ ≠ 0 (automático para n_Σ ≠ 0, se usa como sanity check).
        Adicionalmente, si hay flujo, se exige que X_H no sea nulo.
        """
        n_sig = np.asarray(section_normal, dtype=np.float64).ravel()
        if _NumericalCore.frobenius_norm(n_sig) <= _MACHINE_EPS:
            return False
        omega_n = np.asarray(omega, dtype=np.float64) @ n_sig
        chart_ok = bool(
            _NumericalCore.frobenius_norm(omega_n) > _MACHINE_EPS)
        if flow_vector is None:
            return chart_ok
        xh = np.asarray(flow_vector, dtype=np.float64).ravel()
        if xh.size != n_sig.size:
            raise ValueError("flow_vector y section_normal incompatibles.")
        flux = abs(_NumericalCore.kahan_babuska_neumaier_sum(n_sig * xh))
        return bool(chart_ok and flux > _MACHINE_EPS)

    @staticmethod
    def energy_section_tangent_projector(
        omega: np.ndarray,
        section_normal: np.ndarray,
        hamiltonian_gradient: np.ndarray,
    ) -> np.ndarray:
        r"""
        Proyector (2n)×(2n) sobre T(Σ ∩ H⁻¹(E)):

            T_z(Σ ∩ H⁻¹(E)) = { v | n_Σ·v = 0,  ∇H·v = 0 }.

        Construcción: P = I − u₁u₁ᵀ − u₂u₂ᵀ tras Gram–Schmidt euclidiano
        del par {n_Σ, ∇H}. El rango numérico debe ser 2n−2.
        """
        n_sig = np.asarray(section_normal, dtype=np.float64).ravel()
        gH = np.asarray(hamiltonian_gradient, dtype=np.float64).ravel()
        dim = n_sig.size
        _NumericalCore.assert_square("omega", omega, dim=dim)
        B = np.column_stack((n_sig, gH))
        q, _r = la.qr(B, mode="economic")
        # Descartar columnas casi nulas (n_Σ ∥ ∇H).
        rdiag = np.abs(np.diag(_r)) if _r.ndim == 2 else np.abs(_r)
        live = rdiag > _WILKINSON_DEFLATION_FLOOR
        q_live = q[:, live] if q.ndim == 2 else q.reshape(dim, 1)
        ident = np.eye(dim, dtype=np.float64)
        return ident - q_live @ q_live.T

    # ── I.1.5  Función generatriz de Hamilton–Jacobi (tipo F₂) ────────────
    @staticmethod
    def hamilton_jacobi_generating_function(
        q_old: np.ndarray,
        p_old: np.ndarray,
        q_new: np.ndarray,
        hessian_F2: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        r"""
        Transformación canónica por función generatriz F₂(q, P).

        Poincaré: toda evolución canónica de tiempo T admite F₂(q, P) con
            p = ∂F₂/∂q ,   Q = ∂F₂/∂P .
        Se toma el germen cuadrático (tipo 2, identidad + shear):

            F₂(q, P) = P·q + ½ (q − q_old)ᵀ H_sym (q − q_old),

        con H_sym = ½(H + Hᵀ) para imponer d²F₂ = 0 (hessiana cerrada ⇔
        canonicidad). Entonces

            P_new = p_old + H_sym (q_new − q_old),
            p_check = ∂F₂/∂q |_{q_old} = P_new  (en el chart identidad).

        Residuo HJ = ‖H − Hᵀ‖_F  (obstrucción a la cerradura).
        """
        qo = np.asarray(q_old, dtype=np.float64).ravel()
        po = np.asarray(p_old, dtype=np.float64).ravel()
        qn = np.asarray(q_new, dtype=np.float64).ravel()
        if qo.shape != po.shape or qo.shape != qn.shape:
            raise ValueError("q_old, p_old, q_new deben compartir shape.")
        n = qo.size
        H = np.eye(n) if hessian_F2 is None else np.asarray(
            hessian_F2, dtype=np.float64)
        _NumericalCore.assert_square("hessian_F2", H, dim=n)
        H_sym = 0.5 * (H + H.T)
        dq = qn - qo
        P_new = po + H_sym @ dq
        p_check = P_new - H_sym @ dq          # = p_old por construcción
        hj_res = _NumericalCore.frobenius_norm(H - H.T)
        return P_new, p_check, float(hj_res)

    @staticmethod
    def linear_canonical_generating_function_F2(
        S: np.ndarray, omega: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], float]:
        r"""
        F₂ cuadrática exacta de un mapa lineal S = [A B; C D] ∈ Sp(2n)
        cuando D es invertible:

            p = D⁻¹ (P − C q),
            Q = (A − B D⁻¹ C) q + B D⁻¹ P.

        F₂(q, P) = ½ [q, P] W [q; P] se reconstruye por integración de
        p = ∂F₂/∂q, Q = ∂F₂/∂P. Residuo = ‖Sᵀ Ω S − Ω‖.
        """
        s = np.asarray(S, dtype=np.float64)
        _NumericalCore.assert_square("S", s)
        dim = s.shape[0]
        if dim % 2 != 0:
            raise SymplecticDimensionError("F₂ lineal exige dim par.")
        n = dim // 2
        A, B = s[:n, :n], s[:n, n:]
        C, D = s[n:, :n], s[n:, n:]
        res = _NumericalCore.symplectic_residual(s, omega)
        try:
            D_inv = la.inv(D)
        except la.LinAlgError:
            logger.warning("F₂: bloque D singular; no hay chart tipo 2.")
            return None, float(res)
        # W actúa sobre (q, P): p = W_qq q + W_qP P, Q = W_Pq q + W_PP P
        W_qP = D_inv.T                  # ∂p/∂P = D⁻ᵀ
        W_qq = -D_inv @ C               # ∂p/∂q
        W_qq = 0.5 * (W_qq + W_qq.T)    # simetrizar (cerradura)
        W_PP = 0.5 * (D_inv @ B + (D_inv @ B).T)
        W_Pq = A - B @ D_inv @ C
        W = np.zeros((dim, dim), dtype=np.float64)
        W[:n, :n] = W_qq
        W[:n, n:] = W_qP
        W[n:, :n] = W_Pq.T              # consistencia de Schwarz si canónica
        W[n:, n:] = W_PP
        return W, float(res)

    # ── I.1.6  Gram–Schmidt simpléctico (Parasjuk–de Gosson) ──────────────
    @staticmethod
    def symplectic_gram_schmidt(
        vectors: np.ndarray, omega: np.ndarray,
    ) -> np.ndarray:
        r"""
        Gram–Schmidt simpléctico.

        De una base B ∈ ℝ^{2n×2n} construye S ∈ Sp(2n, ℝ) (numéricamente)
        con pares conjugados (e_k, f_k) tales que

            ω(e_i, e_j) = 0,   ω(f_i, f_j) = 0,   ω(e_i, f_j) = δ_{ij}.

        Ortogonalización contra el par (e_j, f_j):

            v ← v + ω(f_j, v) e_j − ω(e_j, v) f_j .

        Si el par (v, w) degenera, se completa canónicamente w ← −Ω v,
        pues ω(v, −Ωv) = ‖v‖² (Ω² = −I).
        """
        B = np.asarray(vectors, dtype=np.float64)
        O = np.asarray(omega, dtype=np.float64)
        _NumericalCore.assert_square("omega", O)
        dim = O.shape[0]
        if dim % 2 != 0:
            raise SymplecticDimensionError(
                "SGS exige dimensión par (Sp(2n)).")
        if B.shape != (dim, dim):
            raise ValueError(
                f"vectors debe ser ({dim},{dim}); recibido {B.shape}.")
        _NumericalCore.assert_finite("vectors", B)
        n = dim // 2
        S = np.zeros_like(B)

        def sprod(x: np.ndarray, y: np.ndarray) -> float:
            return _NumericalCore.symplectic_pairing(x, y, O)

        def project_out(vec: np.ndarray, n_pairs: int) -> np.ndarray:
            v = vec.copy()
            for j in range(n_pairs):
                e = S[:, j]
                fvec = S[:, j + n]
                v = v + sprod(fvec, v) * e - sprod(e, v) * fvec
            return v

        for k in range(n):
            v = project_out(B[:, k], k)
            nv = float(np.linalg.norm(v))
            if nv < _WILKINSON_DEFLATION_FLOOR:
                v = np.zeros(dim, dtype=np.float64)
                v[k] = 1.0
                v = project_out(v, k)
                nv = float(np.linalg.norm(v)) or 1.0
            v = v / nv

            w = project_out(B[:, k + n], k)
            omega_vw = sprod(v, w)
            if abs(omega_vw) < _WILKINSON_DEFLATION_FLOOR:
                w = -O @ v
                w = project_out(w, k)
                omega_vw = sprod(v, w)
            if abs(omega_vw) < _MACHINE_EPS:
                logger.warning(
                    "SGS: par %d degenerado (ω(v,w)=%.3e); se regulariza.",
                    k, omega_vw)
                omega_vw = np.sign(omega_vw or 1.0) * _MACHINE_EPS
            S[:, k] = v
            S[:, k + n] = w / omega_vw

        res = _NumericalCore.frobenius_norm(S.T @ O @ S - O)
        if res > 1e-6:
            logger.warning("SGS: residuo simpléctico %.3e excede umbral.", res)
        return S

    # ── I.1.7  Clasificación de Williamson del equilibrio ─────────────────
    @staticmethod
    def williamson_classify(
        hessian: np.ndarray, omega: np.ndarray,
    ) -> _WilliamsonSpectrum:
        r"""
        Espectro de Williamson de H = ½ xᵀ A x, A = Hess H simétrica.

        Matriz hamiltoniana K = J A con J = Ω⁻¹ = −Ω, de modo que
        ẋ = K x. El espectro es simétrico respecto de ambos ejes.
        Estabilidad lineal ⇔ todo el espectro es imaginario puro y
        semisimple (aquí se reporta la condición espectral; la
        semisimplicidad se infiere de un gap en el número de condición
        del eigenespacio, omitido a este orden).
        """
        A = np.asarray(hessian, dtype=np.float64)
        O = np.asarray(omega, dtype=np.float64)
        _NumericalCore.assert_square("hessian", A)
        _NumericalCore.assert_square("omega", O, dim=A.shape[0])
        A_sym = np.real(_NumericalCore.higham_nearest_hermitian(A))
        K = -O @ A_sym                          # J = −Ω
        ev = la.eigvals(K)
        elliptic = hyperbolic = focus = parabolic = 0
        consumed = np.zeros(ev.size, dtype=bool)
        imag_ratio = _WILLIAMSON_IMAG_RATIO
        for i, z in enumerate(ev):
            if consumed[i]:
                continue
            re, im = float(np.real(z)), float(np.imag(z))
            scale = max(abs(re), abs(im), 1.0)
            if abs(re) <= imag_ratio * scale and abs(im) <= imag_ratio * scale:
                parabolic += 1
                consumed[i] = True
            elif abs(re) <= imag_ratio * scale and abs(im) > imag_ratio * scale:
                elliptic += 1
                consumed[i] = True
            elif abs(im) <= imag_ratio * scale and abs(re) > imag_ratio * scale:
                hyperbolic += 1
                consumed[i] = True
            else:
                focus += 1
                consumed[i] = True
        # Cada par (±) se contó dos veces salvo el núcleo.
        elliptic_pairs = elliptic // 2
        hyperbolic_pairs = hyperbolic // 2
        focus_focus_pairs = focus // 4
        is_stable = bool(
            hyperbolic_pairs == 0
            and focus_focus_pairs == 0
            and parabolic == 0
            and elliptic_pairs == A.shape[0] // 2
        )
        return _WilliamsonSpectrum(
            elliptic_pairs=int(elliptic_pairs),
            hyperbolic_pairs=int(hyperbolic_pairs),
            focus_focus_pairs=int(focus_focus_pairs),
            parabolic_multiplicity=int(parabolic),
            hamiltonian_eigenvalues=np.asarray(ev),
            is_linearly_stable=bool(is_stable),
        )

    # ── I.1.8  Invariante integral de Poincaré–Cartan ─────────────────────
    @staticmethod
    def poincare_cartan_action(
        q_traj: np.ndarray,
        p_traj: np.ndarray,
        dt: float,
        energy_level: float,
    ) -> Tuple[float, float]:
        r"""
        Acción de Poincaré–Cartan a lo largo de un arco

            𝒜 = ∫ (p · dq − H dt) .

        Sobre una órbita periódica de energía E y periodo T, 𝒜 es el
        invariante integral relativo. Se discretiza por la regla del
        punto medio: p̄_i · Δq_i − E Δt.

        Retorna (𝒜, residuo de cierre ‖(q_N, p_N) − (q_0, p_0)‖).
        """
        q = np.asarray(q_traj, dtype=np.float64)
        p = np.asarray(p_traj, dtype=np.float64)
        if q.ndim != 2 or p.shape != q.shape or q.shape[0] < 2:
            raise ValueError("q_traj, p_traj deben ser (N≥2, n).")
        if not np.isfinite(dt) or dt == 0.0:
            raise ValueError("dt debe ser finito y no nulo.")
        dq = np.diff(q, axis=0)
        p_mid = 0.5 * (p[1:] + p[:-1])
        pdq = np.einsum("ij,ij->i", p_mid, dq)
        action_pdq = _NumericalCore.kahan_babuska_neumaier_sum(pdq)
        action = float(action_pdq - float(energy_level) * dt * dq.shape[0])
        close = _NumericalCore.frobenius_norm(
            np.concatenate((q[-1] - q[0], p[-1] - p[0])))
        return action, float(close)

    # ── I.9  MORFISMO TERMINAL DE LA FASE I ───────────────────────────────
    @staticmethod
    def synthesize_poincare_darboux_germ(
        dimension_two_n: int,
        section_index: int = 0,
        section_offset: float = 0.0,
        energy_level: float = 0.0,
        regularizer: float = _REG_FLOOR_TIKHONOV,
        max_iter: int = _DEFAULT_MAX_ITER,
        tol: float = _DEFAULT_TOL,
        scale_matrix: Optional[np.ndarray] = None,
        flow_vector: Optional[np.ndarray] = None,
    ) -> _PoincareDarbouxGerm:
        r"""
        **I.9 — Morfismo terminal de la FASE I / objeto inicial de la FASE II.**

        Ensambla el gérmen de Poincaré–Darboux

            𝒢_I = (2n, n, Ω_{Darboux}, ε_W, K, τ, Cert(Ω),
                   n_Σ, k_Σ, q_Σ^*, E, transversalidad, X_H, vol_Liouville)

        sobre el cual la FASE II define la iteración polar de
        Higham–Mackey–Tisseur, la factorización de Quillen, la
        factorización de Floquet–Lyapunov, Mel'nikov y el número de
        rotación. Este método **es** el arranque formal de
        `_SymplecticProjector.__init__` (Φ_II.0).
        """
        dim = int(dimension_two_n)
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"dimension_two_n par y positivo; recibido {dim}.")
        if int(max_iter) <= 0:
            raise ValueError("max_iter debe ser entero positivo.")
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol debe ser positivo y finito.")
        omega = _NumericalCore.generate_canonical_symplectic_form(dim)
        certificate = _NumericalCore.certify_symplectic_form(omega)
        if not certificate.is_darboux:
            logger.warning(
                "Darboux degradado: skew=%.3e, J²+I=%.3e, det=%.16f",
                certificate.skew_residual,
                certificate.almost_complex_residual,
                certificate.determinant,
            )
        floor = max(float(regularizer), _REG_FLOOR_TIKHONOV)
        if scale_matrix is not None:
            floor = max(floor, _NumericalCore.wilkinson_deflation_floor(
                np.asarray(scale_matrix)))
        n_sigma = _NumericalCore.build_poincare_section(dim, section_index)
        flow = None if flow_vector is None else np.asarray(
            flow_vector, dtype=np.float64).ravel()
        if flow is not None and flow.size != dim:
            raise ValueError(
                f"flow_vector dim {flow.size} ≠ two_n={dim}.")
        is_transv = _NumericalCore.certify_section_transversality(
            omega, n_sigma, flow_vector=flow)
        return _PoincareDarbouxGerm(
            two_n=dim, n=dim // 2, omega=omega,
            reg_floor=float(floor), max_iter=int(max_iter), tol=float(tol),
            form_certificate=certificate,
            section_normal=n_sigma,
            section_index=int(section_index),
            section_offset=float(section_offset),
            energy_level=float(energy_level),
            is_section_transversal=bool(is_transv),
            flow_vector=flow,
            liouville_volume=float(certificate.liouville_volume),
        )


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████████
# ██  FASE II — POLAR SIMPLÉCTICA, QUILLEN, FLOQUET–POINCARÉ, MEL'NIKOV       ██
# ██  Continúa I.9. Morfismo terminal (II.8): induce_poincare_floquet_germ   ██
# ██                                          → objeto inicial de la FASE III ██
# ██████████████████████████████████████████████████████████████████████████████
# =============================================================================
# PUENTE Φ_I ▸ Φ_II
# El valor de retorno de I.9 (`_PoincareDarbouxGerm`) es el único
# argumento geométrico de `_SymplecticProjector.__init__` (II.0).
# =============================================================================

@dataclass(frozen=True)
class _SymplecticProjectionResult:
    symplectic_matrix: np.ndarray
    residual: float
    iterations: int
    relative_residual: float
    determinant: float
    used_exact_solve: bool
    converged: bool


@dataclass(frozen=True)
class _QuillenFactorizationResult:
    fibration: np.ndarray
    cofibration: np.ndarray
    total_residual: float
    left_reconstruction: float
    right_reconstruction: float
    symplectic_error: float
    inverse_formula_residual: float
    right_fibration: np.ndarray


@dataclass(frozen=True)
class _KreinClassification:
    r"""
    Clasificación de Krein–Moser de los multiplicadores de Floquet.

      * elípticos   : |μ| = 1, μ ≠ ±1   (rotaciones, Krein ±)
      * parabólicos : μ = ±1
      * hiperbólicos: μ ∈ ℝ, |μ| ≠ 1
      * loxodrómicos: μ ∈ ℂ \ ℝ, |μ| ≠ 1  (cuaternas μ, 1/μ, μ̄, 1/μ̄)

    Un toro elíptico es *Krein-definido* (estabilidad fuerte) si todos
    los elípticos tienen signatura de Krein definida y son simples.
    """

    elliptic: int
    hyperbolic: int
    loxodromic: int
    parabolic: int
    krein_definite: bool
    on_unit_circle: int


@dataclass(frozen=True)
class _FloquetFactorizationResult:
    r"""
    Factorización de Floquet–Lyapunov de la monodromía M:

        M = exp(T · A_F) · R_F ,

    A_F generador de Floquet (logaritmo real matricial vía Schur cuando
    el espectro evita el eje real negativo con bloques de Jordan impares)
    y R_F parte periódica (≅ I si el logaritmo es exacto).

    Multiplicadores μ_k = eig(M); exponentes característicos
    λ_k = Log(μ_k) / T  (se reporta Re λ_k, exponentes de Lyapunov).
    """

    floquet_generator: np.ndarray
    periodic_part: np.ndarray
    log_residual: float
    is_real_logarithm: bool
    floquet_multipliers: np.ndarray
    characteristic_exponents: np.ndarray
    krein: Optional[_KreinClassification] = None


@dataclass(frozen=True)
class _MelnikovResult:
    r"""
    Función de Mel'nikov ℳ(t₀) para la separación W^s − W^u:

        ℳ(t₀) = ∫_{−∞}^{∞} {H₀, H₁}(q₀(t), t + t₀) dt ,

    {·,·} paréntesis de Poisson canónico. Un cero simple
    ℳ(t₀) = 0, ℳ'(t₀) ≠ 0  ⇒  intersección transversal homoclínica
    ⇒ dinámica caótica (Smale horseshoe, Poincaré–Birkhoff–Smale).
    """

    melnikov_values: np.ndarray
    simple_zeros: int
    chaotic_indicator: float
    is_chaotic: bool
    melnikov_derivative_min: float = 0.0


@dataclass(frozen=True)
class _RotationNumberResult:
    r"""
    Número de rotación de Poincaré ρ = lim (1/(2π N)) Σ Δθ_i  ∈ ℝ/ℤ.

    Conmensurable ⇔ ρ ∈ ℚ (isla KAM periódica / Poincaré–Birkhoff).
    """

    rotation_number: float
    is_rational: bool
    continued_fraction: Tuple[int, ...]
    diophantine_constant: float


@dataclass(frozen=True)
class _MoserTwistResult:
    r"""
    Certificado de twist de Moser: |∂ρ/∂I| ≥ ν > 0 implica, por el
    teorema de Poincaré–Birkhoff / Moser, persistencia de curvas
    invariantes para mapas que preservan área (n = 1 en la sección).
    """

    twist_value: float
    is_twist: bool
    intersection_property: bool


@dataclass(frozen=True)
class _PoincareFloquetGerm:
    """
    **Gérmen de Poincaré–Floquet.**

    Objeto terminal de la **FASE II** / objeto inicial de la **FASE III**.
    Compone el gérmen de Poincaré–Darboux de Fase I con la información
    dinámica extraída por la Fase II.
    """

    darboux_germ: _PoincareDarbouxGerm
    floquet: _FloquetFactorizationResult
    melnikov: Optional[_MelnikovResult]
    rotation: Optional[_RotationNumberResult]
    lyapunov_max: float
    poincare_map_multipliers: Optional[np.ndarray] = None
    moser_twist: Optional[_MoserTwistResult] = None


class _SymplecticProjector:
    """
    Fase II. Proyector polar sobre Sp(2n, ℝ), factorización de Quillen
    y análisis Floquet–Poincaré. Se instancia desde un
    `_PoincareDarbouxGerm` (terminal de Fase I = I.9).
    """

    def __init__(
        self,
        germ: Optional[_PoincareDarbouxGerm] = None,
        regularizer: float = _REG_FLOOR_TIKHONOV,
        max_iter: int = _DEFAULT_MAX_ITER,
        tol: float = _DEFAULT_TOL,
        dimension_two_n: int = 2,
    ) -> None:
        if germ is None:
            germ = _NumericalCore.synthesize_poincare_darboux_germ(
                dimension_two_n,
                regularizer=regularizer,
                max_iter=max_iter, tol=tol,
            )
        self._germ = germ

    @property
    def germ(self) -> _PoincareDarbouxGerm:
        return self._germ

    def _adapt_omega(self, dim: int) -> np.ndarray:
        if dim == self._germ.two_n:
            return self._germ.omega
        return _NumericalCore.generate_canonical_symplectic_form(dim)

    # ── II.1  Proyección polar estructurada ───────────────────────────────
    def project(
        self,
        M: np.ndarray,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
    ) -> _SymplecticProjectionResult:
        r"""
        Newton estructurado escalado por Frobenius (Higham–Mackey–Tisseur):

            X_{k+1} = ½ ( μ_k X_k + μ_k⁻¹ Ω X_k⁻ᵀ Ωᵀ ) ,
            μ_k     = ( ‖Ω X_k⁻ᵀ Ωᵀ‖_F / ‖X_k‖_F )^{1/2} .

        Se guarda el iterado de residual simpléctico mínimo (guardia de
        divergencia) y se corta también por ‖Xᵀ Ω X − Ω‖.
        """
        a = np.asarray(M, dtype=np.float64)
        _NumericalCore.assert_square("M", a)
        _NumericalCore.assert_finite("M", a)
        dim = a.shape[0]
        if dim % 2 != 0:
            raise ValueError(f"M debe ser dimensión par; recibido {dim}.")
        omega = self._adapt_omega(dim)
        k_max = int(self._germ.max_iter if max_iter is None else max_iter)
        tau = float(self._germ.tol if tol is None else tol)
        if k_max <= 0 or not np.isfinite(tau) or tau <= 0.0:
            raise ValueError("max_iter>0 y tol>0 finitos.")
        floor = max(self._germ.reg_floor,
                    _NumericalCore.wilkinson_deflation_floor(a))
        m_k = a.copy()
        iterations = 0
        used_exact = True
        converged = False
        fro_m = _NumericalCore.frobenius_norm(a)
        best = m_k.copy()
        best_res = float("inf")
        omega_scale = max(_NumericalCore.frobenius_norm(omega), 1.0)
        for iteration in range(k_max):
            iterations = iteration + 1
            iota, exact = _NumericalCore.cartan_involution_image(
                m_k, omega, floor)
            used_exact = used_exact and exact
            fro_m = _NumericalCore.frobenius_norm(m_k)
            fro_i = _NumericalCore.frobenius_norm(iota)
            if fro_m <= _MACHINE_EPS or fro_i <= _MACHINE_EPS:
                logger.warning("Polar: norma degenerada en iter %d.", iteration)
                break
            mu = float(np.sqrt(fro_i / fro_m))
            mu = float(np.clip(mu, _MU_CLIP[0], _MU_CLIP[1]))
            m_next = 0.5 * (mu * m_k + (1.0 / mu) * iota)
            diff = _NumericalCore.frobenius_norm(m_next - m_k)
            m_k = m_next
            if not np.isfinite(diff):
                logger.warning("Polar: iterado no finito en %d.", iteration)
                break
            residual_k = _NumericalCore.symplectic_residual(m_k, omega)
            if np.isfinite(residual_k) and residual_k < best_res:
                best_res = residual_k
                best = m_k.copy()
            scale_k = max(omega_scale * (fro_m ** 2 + 1.0), 1.0)
            if residual_k < tau * scale_k or diff < tau * max(1.0, fro_m):
                converged = True
                break
        m_k = best
        residual = _NumericalCore.symplectic_residual(m_k, omega)
        if not np.isfinite(residual):
            residual = float("inf")
        scale = max(omega_scale * (fro_m ** 2 + 1.0), 1.0)
        rel = float(residual / scale)
        try:
            det_s = float(np.real(la.det(m_k)))
        except (np.linalg.LinAlgError, ValueError):
            det_s = float("nan")
        return _SymplecticProjectionResult(
            symplectic_matrix=m_k, residual=float(residual),
            iterations=int(iterations), relative_residual=rel,
            determinant=det_s, used_exact_solve=bool(used_exact),
            converged=bool(converged),
        )

    # ── II.2  Factorización de Quillen ────────────────────────────────────
    def factorize_quillen(self, M: np.ndarray) -> _QuillenFactorizationResult:
        r"""
        Factorización de Quillen relativa a Sp(2n) ↪ GL(2n):

            M = P_L S = S P_R ,   S = Pol_{Sp}(M) ,
            P_L = M S⁻¹ , P_R = S⁻¹ M ,  S⁻¹ = −Ω Sᵀ Ω .
        """
        proj = self.project(M)
        s_mat = proj.symplectic_matrix
        a = np.asarray(M, dtype=np.float64)
        dim = a.shape[0]
        omega = self._adapt_omega(dim)
        s_inv = _NumericalCore.symplectic_inverse(s_mat, omega)
        ident = np.eye(dim, dtype=np.float64)
        inv_formula = _NumericalCore.frobenius_norm(s_mat @ s_inv - ident)
        p_left = a @ s_inv
        p_right = s_inv @ a
        left_rec = _NumericalCore.frobenius_norm(a - p_left @ s_mat)
        right_rec = _NumericalCore.frobenius_norm(a - s_mat @ p_right)
        sp_err = proj.residual
        return _QuillenFactorizationResult(
            fibration=p_left, cofibration=s_mat,
            total_residual=float(left_rec + sp_err),
            left_reconstruction=float(left_rec),
            right_reconstruction=float(right_rec),
            symplectic_error=float(sp_err),
            inverse_formula_residual=float(inv_formula),
            right_fibration=p_right,
        )

    # ── II.3  Factorización de Floquet–Lyapunov ───────────────────────────
    @staticmethod
    def classify_floquet_multipliers(
        mu: np.ndarray,
        omega: Optional[np.ndarray] = None,
        right_eigenvectors: Optional[np.ndarray] = None,
    ) -> _KreinClassification:
        r"""
        Clasificación de Krein–Moser de {μ_k} = σ(M).

        Signatura de Krein sobre un eigenespacio elíptico:
            κ(v) = sign( i v* Ω v )   (forma de Krein).
        Se declara krein_definite si todo elíptico simple tiene κ ≠ 0.
        """
        z = np.asarray(mu, dtype=np.complex128).ravel()
        elliptic = hyperbolic = loxodromic = parabolic = 0
        on_unit = 0
        krein_ok = True
        for i, m in enumerate(z):
            ab = abs(m)
            re, im = float(np.real(m)), float(np.imag(m))
            if abs(ab - 1.0) <= _KREIN_UNIT_TOL:
                on_unit += 1
                if abs(m - 1.0) <= _PARABOLIC_MULTIPLIER_TOL or abs(
                        m + 1.0) <= _PARABOLIC_MULTIPLIER_TOL:
                    parabolic += 1
                else:
                    elliptic += 1
                    if (omega is not None and right_eigenvectors is not None
                            and i < right_eigenvectors.shape[1]):
                        v = right_eigenvectors[:, i]
                        kv = np.vdot(v, omega @ v)
                        # i v* Ω v debe ser real para elípticos.
                        kappa = float(np.imag(kv))
                        if abs(kappa) <= _MACHINE_EPS:
                            krein_ok = False
            else:
                if abs(im) <= _KREIN_UNIT_TOL * max(ab, 1.0):
                    hyperbolic += 1
                else:
                    loxodromic += 1
        if elliptic == 0:
            krein_ok = False
        return _KreinClassification(
            elliptic=int(elliptic), hyperbolic=int(hyperbolic),
            loxodromic=int(loxodromic), parabolic=int(parabolic),
            krein_definite=bool(krein_ok and loxodromic == 0
                                and hyperbolic == 0),
            on_unit_circle=int(on_unit),
        )

    @staticmethod
    def _real_matrix_logarithm(a: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Logaritmo matricial: Schur real (scipy.logm) + fallback espectral."""
        try:
            log_M = la.logm(a)
        except (la.LinAlgError, ValueError) as exc:
            logger.warning("logm/Schur falló (%s); se diagonaliza en ℂ.", exc)
            w, V = la.eig(a)
            if abs(la.det(V)) < _MACHINE_EPS:
                log_M = np.real(a - np.eye(a.shape[0]))
                return np.asarray(log_M, dtype=np.float64), False
            logw = np.log(w.astype(np.complex128))
            log_M = V @ np.diag(logw) @ la.inv(V)
        imag_amp = float(np.max(np.abs(np.imag(log_M)))) if np.iscomplexobj(
            log_M) else 0.0
        scale = max(float(np.max(np.abs(log_M))), 1.0)
        is_real = bool(imag_amp <= 1e-10 * scale)
        return np.real(np.asarray(log_M, dtype=np.complex128)), is_real

    def floquet_factorization(
        self, M: np.ndarray, orbit_period_T: float,
    ) -> _FloquetFactorizationResult:
        r"""
        Descomposición de Floquet–Lyapunov de la monodromía M:

            M = exp(T · A_F) · R_F ,    A_F = T⁻¹ Log(M).

        Un logaritmo *real* existe si M no tiene autovalores reales
        negativos de bloques de Jordan impares. Los multiplicadores
        elípticos (|μ|=1) producen exponentes imaginarios puros y un
        Log real (bloques de rotación 2×2).
        """
        a = np.asarray(M, dtype=np.float64)
        _NumericalCore.assert_square("M", a)
        _NumericalCore.assert_finite("M", a)
        if orbit_period_T <= 0 or not np.isfinite(orbit_period_T):
            raise ValueError("orbit_period_T debe ser positivo y finito.")
        mu, evec = la.eig(a, right=True)
        log_M, is_real_log = self._real_matrix_logarithm(a)
        a_F = log_M / float(orbit_period_T)
        r_F = la.expm(-float(orbit_period_T) * a_F) @ a
        log_res = _NumericalCore.frobenius_norm(
            la.expm(float(orbit_period_T) * a_F) @ r_F - a)
        with np.errstate(divide="ignore", invalid="ignore"):
            char_c = np.log(mu.astype(np.complex128)) / float(orbit_period_T)
        char_exp = np.real(char_c)
        omega = self._adapt_omega(a.shape[0])
        krein = self.classify_floquet_multipliers(
            mu, omega=omega, right_eigenvectors=evec)
        return _FloquetFactorizationResult(
            floquet_generator=a_F,
            periodic_part=r_F,
            log_residual=float(log_res),
            is_real_logarithm=is_real_log,
            floquet_multipliers=mu,
            characteristic_exponents=char_exp,
            krein=krein,
        )

    # ── II.3.b  Reducción al mapa de Poincaré (2n−2) ──────────────────────
    @staticmethod
    def poincare_map_multipliers(
        floquet_multipliers: np.ndarray,
        drop_parabolic: int = 2,
    ) -> np.ndarray:
        r"""
        Espectro del mapa de primer retorno P_Σ.

        En un sistema hamiltoniano autónomo, μ = 1 es multiplicador doble
        (dirección del flujo × conservación de la energía). El espectro
        de DP_Σ son los 2n−2 multiplicadores restantes.
        """
        mu = np.asarray(floquet_multipliers, dtype=np.complex128).ravel()
        if mu.size <= drop_parabolic:
            return mu.copy()
        order = np.argsort(np.abs(mu - 1.0))
        keep = np.ones(mu.size, dtype=bool)
        keep[order[:drop_parabolic]] = False
        return mu[keep]

    # ── II.4  Función de Mel'nikov ────────────────────────────────────────
    def melnikov_function(
        self,
        q0_trajectory: np.ndarray,
        dt: float,
        h0_grad: np.ndarray,
        h1_grad: np.ndarray,
        n_phase_offsets: int = _MELNIKOV_PHASE_SAMPLES,
    ) -> _MelnikovResult:
        r"""
        Función de Mel'nikov por cuadratura sobre la órbita homoclínica:

            ℳ(t₀) = ∫ (∇H₀)ᵀ Ω (∇H₁) dt = ∫ {H₀, H₁} dt .

        Se evalúa sobre `n_phase_offsets` desplazamientos circulares.
        Cero *simple*: cambio de signo y |ℳ′| > piso de Wilkinson
        (ℳ′ por diferencia central sobre la muestra circular).
        """
        q0 = np.asarray(q0_trajectory, dtype=np.float64)
        g0 = np.asarray(h0_grad, dtype=np.float64)
        g1 = np.asarray(h1_grad, dtype=np.float64)
        if not (q0.shape == g0.shape == g1.shape):
            raise ValueError("q0, h0_grad, h1_grad deben compartir shape.")
        if q0.ndim != 2:
            raise ValueError("q0_trajectory debe ser 2D (N, 2n).")
        if not np.isfinite(dt) or dt == 0.0:
            raise ValueError("dt debe ser finito y no nulo.")
        dim = q0.shape[1]
        omega = self._adapt_omega(dim)
        pb = np.einsum("ij,jk,ik->i", g0, omega, g1)
        n_phase_offsets = max(int(n_phase_offsets), 4)

        def trapz(y: np.ndarray, h: float) -> float:
            if y.size < 2:
                return 0.0
            return float(h * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1]))

        N = pb.size
        mel_vals = np.empty(n_phase_offsets, dtype=np.float64)
        for k in range(n_phase_offsets):
            shift = int(round(k * N / n_phase_offsets))
            mel_vals[k] = trapz(np.roll(pb, shift), dt)

        dphi = _TWO_PI / float(n_phase_offsets)
        dmel = (np.roll(mel_vals, -1) - np.roll(mel_vals, 1)) / (2.0 * dphi)
        floor = max(_NumericalCore.wilkinson_deflation_floor(mel_vals), 1e-14)
        simple_zeros = 0
        for i in range(n_phase_offsets):
            j = (i + 1) % n_phase_offsets
            if mel_vals[i] * mel_vals[j] < 0.0:
                deriv = min(abs(dmel[i]), abs(dmel[j]))
                if deriv > floor:
                    simple_zeros += 1
        chaotic_indicator = float(simple_zeros) / float(n_phase_offsets)
        return _MelnikovResult(
            melnikov_values=mel_vals,
            simple_zeros=int(simple_zeros),
            chaotic_indicator=chaotic_indicator,
            is_chaotic=bool(simple_zeros > 0),
            melnikov_derivative_min=float(np.min(np.abs(dmel))),
        )

    # ── II.5  Número de rotación de Poincaré ──────────────────────────────
    def rotation_number_poincare(
        self,
        orbit_points: np.ndarray,
        cf_depth: int = _ROTATION_CF_DEPTH,
    ) -> _RotationNumberResult:
        r"""
        Número de rotación ρ = lim (1/(2π N)) Σ Δθ_i ∈ ℝ/ℤ, con Δθ_i el
        incremento angular desenrollado del par canónico (q₀, p₀).

        Se clasifica racional si un convergente de Farey verifica
        |ρ − p/q| < 1/(2 q²) (criterio de mejor aproximación).
        """
        pts = np.asarray(orbit_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 3:
            raise ValueError("orbit_points debe ser (N≥3, 2n).")
        dim = pts.shape[1]
        n = dim // 2
        q = pts[:, 0]
        p = pts[:, n] if n < dim else pts[:, -1]
        theta = np.unwrap(np.arctan2(p, q))
        dtheta = np.diff(theta)
        mean_dtheta = float(
            _NumericalCore.klein_sum(dtheta) / max(dtheta.size, 1))
        rho = (mean_dtheta / _TWO_PI) % 1.0          # ∈ [0, 1)

        x = float(rho)
        cf = []
        exact = False
        for _ in range(int(cf_depth)):
            ai = int(np.floor(x))
            cf.append(ai)
            frac = x - ai
            if frac < 1e-14:
                exact = True
                break
            x = 1.0 / frac
        h_prev, h_cur = 0, 1
        k_prev, k_cur = 1, 0
        for ai in cf:
            h_prev, h_cur = h_cur, ai * h_cur + h_prev
            k_prev, k_cur = k_cur, ai * k_cur + k_prev
        approx = (h_cur / k_cur) if k_cur != 0 else rho
        q_den = max(abs(k_cur), 1)
        is_rational = bool(
            exact or abs(approx - rho) < 0.5 / (q_den * q_den))
        diophantine = float(
            0.0 if is_rational
            else min(1.0, abs(rho - approx) * (q_den ** 2)))
        return _RotationNumberResult(
            rotation_number=float(rho),
            is_rational=is_rational,
            continued_fraction=tuple(cf),
            diophantine_constant=diophantine,
        )

    # ── II.5.b  Twist de Moser / Poincaré–Birkhoff ────────────────────────
    def moser_twist(
        self,
        rotation_samples: Sequence[_RotationNumberResult],
        action_samples: Optional[np.ndarray] = None,
    ) -> _MoserTwistResult:
        r"""
        Estimación de ∂ρ/∂I por diferencias finitas sobre una foliación
        de toros (o de curvas invariantes). Twist ⇔ |Δρ/ΔI| ≥ ν.
        La propiedad de intersección (Hall) se declara si ρ no es
        constante sobre la muestra.
        """
        rhos = np.array(
            [r.rotation_number for r in rotation_samples], dtype=np.float64)
        if rhos.size < 2:
            return _MoserTwistResult(
                twist_value=0.0, is_twist=False, intersection_property=False)
        if action_samples is None:
            actions = np.arange(rhos.size, dtype=np.float64)
        else:
            actions = np.asarray(action_samples, dtype=np.float64).ravel()
            if actions.size != rhos.size:
                raise ValueError("action_samples incompatible con ρ.")
        order = np.argsort(actions)
        a_ord, r_ord = actions[order], rhos[order]
        da = np.diff(a_ord)
        dr = np.diff(r_ord)
        live = np.abs(da) > _MACHINE_EPS
        if not np.any(live):
            return _MoserTwistResult(
                twist_value=0.0, is_twist=False, intersection_property=False)
        twist = float(np.median(dr[live] / da[live]))
        return _MoserTwistResult(
            twist_value=twist,
            is_twist=bool(abs(twist) >= _TWIST_FLOOR),
            intersection_property=bool(np.ptp(rhos) > _TWIST_FLOOR),
        )

    # ── II.5.c  Ecuación homológica de Birkhoff (pequeños divisores) ──────
    def homological_small_divisors(
        self,
        frequency_vector: np.ndarray,
        harmonic_cap: int = _KAM_HARMONIC_CAP,
        tau: float = _DIOPHANTINE_TAU_MIN,
    ) -> Tuple[float, float]:
        r"""
        Cota de pequeños divisores:

            γ_num = min_{0<|k|≤H} |k·ω| · |k|^τ ,

        y el peor divisor |k·ω|. Se usan para el resolvente de la
        ecuación homológica {H₀, χ} = H₁^{osc} del normal form.
        """
        omega = np.asarray(frequency_vector, dtype=np.float64).ravel()
        n = omega.size
        if n == 0:
            raise ValueError("frequency_vector vacío.")
        H = int(harmonic_cap)
        gamma = np.inf
        worst = np.inf
        for kv in _KamTorusDetector.integer_lattice(n, H):
            div = abs(float(kv @ omega))
            norm = float(np.linalg.norm(kv))
            if norm < 1e-12:
                continue
            worst = min(worst, div)
            gamma = min(gamma, div * (norm ** tau))
        if not np.isfinite(gamma):
            gamma = 0.0
        if not np.isfinite(worst):
            worst = 0.0
        return float(gamma), float(worst)

    # ── II.6  Inducción de gérmen A∞ / Čech (compatibilidad 3.0) ──────────
    def induce_ainfty_cech_germ(self, payload, m2_tensor=None):
        """
        II.6 — Extrae un `_AInfinityCechGerm` por rama Quillen o Stasheff.
        (`_StasheffAssociator` se resuelve en tiempo de llamada, Φ_III.1.)
        """
        if m2_tensor is not None:
            tensor = np.asarray(m2_tensor)
        else:
            tensor = np.asarray(payload)
        if tensor.ndim == 3:
            assoc = _StasheffAssociator()
            m3 = assoc.compute_m3_associator(tensor)
            n = int(tensor.shape[0])
            germ_mat = np.transpose(m3, (0, 1, 2, 3)).reshape(n * n, n * n)
            germ_mat = np.real(_NumericalCore.higham_nearest_hermitian(germ_mat))
            a_norm = _NumericalCore.frobenius_norm(m3)
            floor = max(self._germ.reg_floor,
                        _NumericalCore.wilkinson_deflation_floor(germ_mat))
            return _AInfinityCechGerm(
                cochain_matrix=np.asarray(germ_mat, dtype=np.float64),
                source="stasheff", algebra_dim=n,
                reg_floor=float(floor), associator_norm=float(a_norm),
            )
        a = np.asarray(tensor)
        if a.ndim == 1:
            side = int(np.sqrt(a.size))
            if side * side != a.size:
                raise ValueError("payload plano no cuadrado perfecto.")
            a = a.reshape(side, side)
        _NumericalCore.assert_square("payload", a)
        _NumericalCore.assert_finite("payload", a)
        fact = self.factorize_quillen(a)
        ident = np.eye(a.shape[0], dtype=np.float64)
        defect = fact.fibration - ident
        germ_mat = np.real(_NumericalCore.higham_nearest_hermitian(defect))
        floor = max(self._germ.reg_floor,
                    _NumericalCore.wilkinson_deflation_floor(germ_mat))
        return _AInfinityCechGerm(
            cochain_matrix=np.asarray(germ_mat, dtype=np.float64),
            source="quillen", algebra_dim=int(a.shape[0]),
            reg_floor=float(floor),
            associator_norm=float(_NumericalCore.frobenius_norm(defect)),
        )

    # ── II.8  MORFISMO TERMINAL DE LA FASE II ─────────────────────────────
    def induce_poincare_floquet_germ(
        self,
        jacobian_M: np.ndarray,
        orbit_period_T: float,
        q0_trajectory: Optional[np.ndarray] = None,
        dt: float = 1.0,
        h0_grad: Optional[np.ndarray] = None,
        h1_grad: Optional[np.ndarray] = None,
        orbit_points_for_rotation: Optional[np.ndarray] = None,
    ) -> _PoincareFloquetGerm:
        r"""
        **II.8 — Morfismo terminal de la FASE II / objeto inicial de la FASE III.**

        Ensambla el gérmen de Poincaré–Floquet

            𝒢_II = (𝒢_I, F, ℳ, ρ, λ_max, σ(DP_Σ), twist) ,

        donde F es la factorización de Floquet–Lyapunov. Este método
        **es** el arranque formal de `_PoincareMonodromyAnalyzer.__init__`.
        """
        M = np.asarray(jacobian_M, dtype=np.float64)
        _NumericalCore.assert_square("jacobian_M", M)
        _NumericalCore.assert_finite("jacobian_M", M)
        if M.shape[0] != self._germ.two_n:
            self._germ = _NumericalCore.synthesize_poincare_darboux_germ(
                M.shape[0], section_index=self._germ.section_index,
                regularizer=self._germ.reg_floor,
                max_iter=self._germ.max_iter, tol=self._germ.tol,
                scale_matrix=M,
                flow_vector=self._germ.flow_vector,
            )
        floquet = self.floquet_factorization(M, orbit_period_T)
        melnikov: Optional[_MelnikovResult] = None
        if (q0_trajectory is not None
                and h0_grad is not None and h1_grad is not None):
            melnikov = self.melnikov_function(
                q0_trajectory, dt, h0_grad, h1_grad)
        rotation: Optional[_RotationNumberResult] = None
        if orbit_points_for_rotation is not None:
            rotation = self.rotation_number_poincare(orbit_points_for_rotation)
        max_mu = float(np.max(np.abs(floquet.floquet_multipliers))) \
            if floquet.floquet_multipliers.size else 0.0
        lyap_max = float(
            np.log(max(max_mu, _MACHINE_EPS))
            / max(orbit_period_T, _MACHINE_EPS))
        pm_mu = self.poincare_map_multipliers(floquet.floquet_multipliers)
        return _PoincareFloquetGerm(
            darboux_germ=self._germ,
            floquet=floquet,
            melnikov=melnikov,
            rotation=rotation,
            lyapunov_max=lyap_max,
            poincare_map_multipliers=pm_mu,
            moser_twist=None,
        )


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████████
# ██  FASE III — STASHEFF A∞, ČECH–DELIGNE, KAM, LYAPUNOV, MONODROMÍA        ██
# ██  Continúa II.8. Morfismo terminal (III.6): compute_certificate          ██
# ██                                            → PoincareMonodromyCertificate ██
# ██████████████████████████████████████████████████████████████████████████████
# =============================================================================
# PUENTE Φ_II ▸ Φ_III
# El valor de retorno de II.8 (`_PoincareFloquetGerm`) es el germen
# dinámico de `_PoincareMonodromyAnalyzer` (III.5).
# =============================================================================

@dataclass(frozen=True)
class _AInfinityCechGerm:
    """Gérmen A∞/Čech (heredado de 3.0)."""

    cochain_matrix: np.ndarray
    source: str
    algebra_dim: int
    reg_floor: float
    associator_norm: float


@dataclass(frozen=True)
class _StasheffAssociatorResult:
    associator: np.ndarray
    associator_norm: float
    pentagon_residual: float
    jacobi_residual: float
    is_associative: bool
    is_lie: bool


@dataclass(frozen=True)
class _CechObstructionResult:
    obstruction_value: float
    singular_values: np.ndarray
    active_modes: int
    cocycle_defect: float
    gerbe_4cocycle_defect: float
    harmonic_energy: float
    betti_0: int
    betti_1: int
    nuclear_mass: float


@dataclass(frozen=True)
class _KamTorusCertificate:
    r"""
    Certificado KAM de toros invariantes.

    Un toro 𝕋ⁿ con frecuencia ω es **KAM-estable** si satisface la
    condición diofántica de Arnold:

        |k · ω| ≥ γ / |k|^τ ,   ∀ k ∈ ℤⁿ \ {0} ,   τ > n−1 .

    Se estima (γ, τ) por barrido de armónicos y se adjunta el tiempo
    de Nekhoroshev T_N ~ T₀ exp( c ε^{−1/(2n)} ).
    """

    frequency_vector: np.ndarray
    diophantine_gamma: float
    diophantine_tau: float
    birkhoff_normal_residual: float
    kam_stable: bool
    kam_iterations: int
    nekhoroshev_time: float = 0.0
    worst_small_divisor: float = 0.0


@dataclass(frozen=True)
class _LyapunovSpectrumResult:
    r"""
    Espectro completo de Lyapunov λ₁ ≥ λ₂ ≥ … ≥ λ_{2n} por
    algoritmo de Benettin (QR discreto) con emparejamiento
    hamiltoniano λᵢ ← ½(λᵢ − λ_{2n+1−i}).

    * Dimensión de Kaplan–Yorke D_KY = j + (Σ_{i≤j} λ_i) / |λ_{j+1}|
      donde j = máx { m : Σ_{i=1}^m λ_i ≥ 0 } (D_KY = 0 si λ₁ < 0).
    * Entropía de Kolmogorov–Sinai (Pesin) h_KS ≤ Σ_{λ_i > 0} λ_i.
    """

    spectrum: np.ndarray
    kaplan_yorke_dimension: float
    is_chaotic: bool
    kolmogorov_sinai_entropy: float


@dataclass(frozen=True)
class PoincareMonodromyGerm:
    """
    Gérmen público de monodromía de Floquet–Poincaré (compatibilidad 3.0).
    """

    relative_symplectic_residual: float
    volume_drift: float
    max_floquet_multiplier: float
    lyapunov_exponent: float
    is_monodromy_stable: bool


@dataclass(frozen=True)
class PoincareMonodromyCertificate:
    """
    **Certificado completo de monodromía de Poincaré–Floquet.**

    Integra los objetos terminales de las tres fases.
    """

    darboux_germ: _PoincareDarbouxGerm
    floquet_germ: _PoincareFloquetGerm
    kam: _KamTorusCertificate
    lyapunov: _LyapunovSpectrumResult
    stasheff: Optional[_StasheffAssociatorResult]
    cech: Optional[_CechObstructionResult]
    relative_symplectic_residual: float
    volume_drift: float
    max_floquet_multiplier: float
    lyapunov_exponent: float
    is_monodromy_stable: bool
    krein_definite: bool = False
    nekhoroshev_time: float = 0.0
    poincare_map_multipliers: Optional[np.ndarray] = None


# ── III.1  Asociador de Stasheff ─────────────────────────────────────────────
class _StasheffAssociator:
    """Fase III. Asociador m₃ y residuos A∞ (pentágono, Jacobi)."""

    def __init__(self, germ: Optional[_AInfinityCechGerm] = None) -> None:
        self._germ = germ

    @staticmethod
    def _validate_m2(m2_tensor: np.ndarray) -> np.ndarray:
        t = np.asarray(m2_tensor)
        if t.ndim != 3 or t.shape[0] != t.shape[1] or t.shape[1] != t.shape[2]:
            raise ValueError(f"m2_tensor debe ser (n,n,n); recibido {t.shape}.")
        _NumericalCore.assert_finite("m2_tensor", t)
        return np.asarray(t, dtype=np.float64)

    def compute_m3_associator(self, m2_tensor: np.ndarray) -> np.ndarray:
        r"""
        (m₃)_{ijk}^{l} = Σ_s ( (m₂)_{ij}^s (m₂)_{sk}^l
                              − (m₂)_{jk}^s (m₂)_{is}^l ) .
        Contracción en s por KBN.
        """
        m2 = self._validate_m2(m2_tensor)
        n = m2.shape[0]
        m3 = np.zeros((n, n, n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                ij = m2[i, j, :]
                for k in range(n):
                    left = m2[:, k, :]
                    jk = m2[j, k, :]
                    right = m2[i, :, :]
                    prod1 = ij[:, None] * left
                    prod2 = jk[:, None] * right
                    diff = prod1 - prod2
                    for ell in range(n):
                        m3[i, j, k, ell] = _NumericalCore.kahan_babuska_neumaier_sum(
                            diff[:, ell])
        return m3

    def pentagon_residual(self, m2: np.ndarray, m3: np.ndarray) -> float:
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
                        t1 = m3[i, j, k, :] @ m2[:, p, :]
                        t2 = m3[j, k, p, :] @ m2[i, :, :]
                        t3 = m2[i, j, :] @ m3[:, k, p, :]
                        t4 = m2[j, k, :] @ m3[i, :, p, :]
                        t5 = m2[k, p, :] @ m3[i, j, :, :]
                        pent = t1 - t2 - t3 + t4 - t5
                        acc += float(_NumericalCore.kahan_babuska_neumaier_sum(
                            pent * pent))
        return float(np.sqrt(max(acc, 0.0)))

    def jacobi_residual(self, m2: np.ndarray) -> float:
        n = m2.shape[0]
        comm = m2 - np.transpose(m2, (1, 0, 2))
        acc = 0.0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    yz = comm[j, k, :]
                    zx = comm[k, i, :]
                    xy = comm[i, j, :]
                    t_x = yz @ comm[i, :, :]
                    t_y = zx @ comm[j, :, :]
                    t_z = xy @ comm[k, :, :]
                    jac = t_x + t_y + t_z
                    acc += float(_NumericalCore.kahan_babuska_neumaier_sum(
                        jac * jac))
        return float(np.sqrt(max(acc, 0.0)))

    def compute_certified(self, m2_tensor: np.ndarray) -> _StasheffAssociatorResult:
        m2 = self._validate_m2(m2_tensor)
        m3 = self.compute_m3_associator(m2)
        a_norm = _NumericalCore.frobenius_norm(m3)
        pent = self.pentagon_residual(m2, m3)
        jac = self.jacobi_residual(m2)
        scale = max(a_norm, 1.0)
        is_assoc = a_norm <= _WILKINSON_DRIFT_LIMIT * scale
        is_lie = jac <= _WILKINSON_DRIFT_LIMIT * max(
            1.0, _NumericalCore.frobenius_norm(m2))
        return _StasheffAssociatorResult(
            associator=m3, associator_norm=float(a_norm),
            pentagon_residual=float(pent), jacobi_residual=float(jac),
            is_associative=bool(is_assoc), is_lie=bool(is_lie),
        )


# ── III.2  Obstrucción Čech–Deligne ──────────────────────────────────────────
class _CechObstructionCalculator:
    """Fase III. Obstrucción no abeliana de Čech–Deligne sobre gerbes."""

    def __init__(self, germ: Optional[_AInfinityCechGerm] = None,
                 regularizer: float = _REG_FLOOR_TIKHONOV) -> None:
        self._germ = germ
        self._reg = max(float(regularizer), _REG_FLOOR_TIKHONOV)

    def _resolve_matrix(self, cech_cochain_matrix: np.ndarray) -> np.ndarray:
        a = np.asarray(cech_cochain_matrix)
        if a.size == 0 and self._germ is not None:
            return np.asarray(self._germ.cochain_matrix, dtype=np.float64)
        if a.ndim == 1:
            side = int(np.sqrt(a.size))
            if side * side != a.size:
                raise ValueError("Cech plana no es cuadrado perfecto.")
            a = a.reshape(side, side)
        _NumericalCore.assert_square("cech_cochain_matrix", a)
        _NumericalCore.assert_finite("cech_cochain_matrix", a)
        return np.asarray(a, dtype=np.complex128)

    def cech_coboundary_defect(self, omega: np.ndarray) -> float:
        w = np.real(0.5 * (np.asarray(omega) - np.asarray(omega).T.conj()))
        n = w.shape[0]
        if n < 3:
            return 0.0
        acc = 0.0
        if n <= _CECH_TRIPLE_CAP:
            triples = ((i, j, k) for i in range(n - 2)
                       for j in range(i + 1, n - 1)
                       for k in range(j + 1, n))
        else:
            step = max(1, n // _CECH_TRIPLE_CAP)
            idx = np.arange(0, n, step)
            triples = ((int(idx[a]), int(idx[b]), int(idx[c]))
                       for a in range(idx.size - 2)
                       for b in range(a + 1, idx.size - 1)
                       for c in range(b + 1, idx.size))
        for i, j, k in triples:
            t = w[j, k] - w[i, k] + w[i, j]
            acc += float(t * t)
        return float(np.sqrt(max(acc, 0.0)))

    def gerbe_4cocycle_defect(self, omega: np.ndarray) -> float:
        w = np.real(0.5 * (np.asarray(omega) - np.asarray(omega).T.conj()))
        n = w.shape[0]
        if n < 4:
            return 0.0

        def theta(a, b, c):
            return float(w[b, c] - w[a, c] + w[a, b])

        acc = 0.0
        idx = list(range(n)) if n <= _CECH_QUAD_CAP else \
            list(range(0, n, max(1, n // _CECH_QUAD_CAP)))
        m = len(idx)
        for a in range(m - 3):
            i = idx[a]
            for b in range(a + 1, m - 2):
                j = idx[b]
                for c in range(b + 1, m - 1):
                    k = idx[c]
                    for d in range(c + 1, m):
                        ell = idx[d]
                        t = (theta(j, k, ell) - theta(i, k, ell)
                             + theta(i, j, ell) - theta(i, j, k))
                        acc += float(t * t)
        return float(np.sqrt(max(acc, 0.0)))

    def sheaf_hodge_spectrum(self, gram, floor):
        herm = np.real(_NumericalCore.higham_nearest_hermitian(gram))
        n = herm.shape[0]
        weights = np.abs(herm)
        np.fill_diagonal(weights, 0.0)
        adj = (weights > floor).astype(np.float64)
        weights = weights * adj
        degree = weights.sum(axis=1)
        lap = np.diag(degree) - weights
        lap = np.real(_NumericalCore.higham_nearest_hermitian(lap))
        evals = np.real(la.eigvalsh(lap)) if n else np.array([], dtype=np.float64)
        ker_tol = max(floor, _WILKINSON_DEFLATION_FLOOR * max(n, 1))
        b0 = int(np.sum(evals <= ker_tol))
        n_edges = int(np.sum(np.triu(adj, 1)))
        b1 = int(max(n_edges - n + b0, 0))
        harmonic = (_NumericalCore.kahan_babuska_neumaier_sum(
            np.clip(evals[:max(b0, 0)], 0.0, None)) if evals.size else 0.0)
        return b0, b1, float(harmonic)

    def compute_obstruction(self, cech_cochain_matrix: np.ndarray) -> _CechObstructionResult:
        if np.asarray(cech_cochain_matrix).size == 0 and self._germ is None:
            return _CechObstructionResult(
                obstruction_value=0.0, singular_values=np.array([]),
                active_modes=0, cocycle_defect=0.0,
                gerbe_4cocycle_defect=0.0, harmonic_energy=0.0,
                betti_0=0, betti_1=0, nuclear_mass=0.0,
            )
        raw = self._resolve_matrix(cech_cochain_matrix)
        sheaf = _NumericalCore.higham_nearest_hermitian(raw)
        floor = max(self._reg, _NumericalCore.wilkinson_deflation_floor(sheaf))
        if self._germ is not None:
            floor = max(floor, self._germ.reg_floor)
        singular_values = np.real(la.svd(sheaf, compute_uv=False))
        active = singular_values[singular_values > floor]
        obstruction = (_NumericalCore.kahan_sum(active) if active.size else 0.0)
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
            betti_0=int(b0), betti_1=int(b1),
            nuclear_mass=float(obstruction),
        )


# ── III.3  Detector KAM ──────────────────────────────────────────────────────
class _KamTorusDetector:
    r"""
    Fase III. Detección de toros invariantes KAM y certificación diofántica.

    Para ω ∈ ℝⁿ se verifica la condición de Arnold

        |k · ω| ≥ γ / |k|^τ ,   ∀ k ∈ ℤⁿ \ {0},  |k| ≤ H_max ,

    sobre un retículo adaptativo (cubo ∞ para n ≤ 2; ejes + cáscaras
    semilla-fijada para n ≥ 3, evitando la explosión (2H+1)ⁿ).
    Tiempo de Nekhoroshev: T_N = exp( c / ε^{1/(2n)} ) con ε el residuo
    de Birkhoff (perturbación al normal form integrable).
    """

    def __init__(self, harmonic_cap: int = _KAM_HARMONIC_CAP) -> None:
        self._H = int(harmonic_cap)

    @staticmethod
    def integer_lattice(n: int, H: int) -> Iterator[np.ndarray]:
        """Generador de k ∈ ℤⁿ \ {0} con |k|_∞ ≤ H, recorte de cardinalidad."""
        H = max(int(H), 1)
        n = int(n)
        if n <= 0:
            return
        cube = (2 * H + 1) ** n
        if n <= 2 and cube <= _KAM_LATTICE_CELL_CAP:
            if n == 1:
                for k in range(-H, H + 1):
                    if k != 0:
                        yield np.array([k], dtype=np.float64)
                return
            for i in range(-H, H + 1):
                for j in range(-H, H + 1):
                    if i == 0 and j == 0:
                        continue
                    yield np.array([i, j], dtype=np.float64)
            return
        # n ≥ 3 o cubo prohibitivo: ejes + muestreo determinista
        for d in range(n):
            for s in (-1.0, 1.0):
                for r in range(1, H + 1):
                    kv = np.zeros(n, dtype=np.float64)
                    kv[d] = s * r
                    yield kv
        rng = np.random.default_rng(1729)        # Hardy–Ramanujan; reproducible
        budget = min(_KAM_LATTICE_CELL_CAP // max(n, 1), 8 * n * H)
        for _ in range(int(budget)):
            kv = rng.integers(-H, H + 1, size=n).astype(np.float64)
            if np.all(kv == 0):
                continue
            yield kv

    @staticmethod
    def nekhoroshev_time(
        n_dof: int,
        perturbation_eps: float,
        prefactor: float = _NEKHOROSHEV_PREFACTOR,
    ) -> float:
        r"""
        Cota de Nekhoroshev (forma clásica):

            T_N ≳ exp( c · ε^{−1/(2n)} ) ,   0 < ε ≪ 1.

        Si ε ≤ 0 se declara T_N = +∞ (sistema integrable).
        """
        n = max(int(n_dof), 1)
        eps = float(perturbation_eps)
        if not np.isfinite(eps) or eps <= 0.0:
            return float("inf")
        exponent = float(prefactor) * (eps ** (-1.0 / (2.0 * n)))
        # Guardia de overflow.
        if exponent > 700.0:
            return float("inf")
        return float(np.exp(exponent))

    def certify(self, frequency_vector: np.ndarray,
                birkhoff_residual: Optional[float] = None) -> _KamTorusCertificate:
        omega = np.asarray(frequency_vector, dtype=np.float64).ravel()
        n = omega.size
        if n == 0:
            raise ValueError("frequency_vector vacío.")
        gamma_est = np.inf
        tau_est = _DIOPHANTINE_TAU_MIN
        iter_count = 0
        worst = np.inf
        for kv in self.integer_lattice(n, self._H):
            kval = float(kv @ omega)
            norm = float(np.linalg.norm(kv))
            if norm < 1e-12:
                continue
            iter_count += 1
            abs_kv = abs(kval)
            worst = min(worst, abs_kv)
            if abs_kv < 1e-12:
                gamma_est = 0.0
                break
            gamma_candidate = abs_kv * (norm ** _DIOPHANTINE_TAU_MIN)
            if gamma_candidate < gamma_est:
                gamma_est = gamma_candidate
        if gamma_est > _DIOPHANTINE_GAMMA_FLOOR and np.isfinite(gamma_est):
            kam_stable = True
        else:
            kam_stable = False
            for tau in np.linspace(_DIOPHANTINE_TAU_MIN, 6.0, 24):
                g_try = np.inf
                resonant = False
                for kv in self.integer_lattice(n, self._H):
                    kval = float(kv @ omega)
                    norm = float(np.linalg.norm(kv))
                    if norm < 1e-12:
                        continue
                    if abs(kval) < 1e-12:
                        g_try = 0.0
                        resonant = True
                        break
                    g_try = min(g_try, abs(kval) * (norm ** tau))
                if (not resonant) and g_try > _DIOPHANTINE_GAMMA_FLOOR:
                    gamma_est = g_try
                    tau_est = float(tau)
                    kam_stable = True
                    break
            else:
                kam_stable = bool(
                    np.isfinite(gamma_est)
                    and gamma_est > _DIOPHANTINE_GAMMA_FLOOR)
        resid = float(birkhoff_residual) if birkhoff_residual is not None else 0.0
        t_nek = self.nekhoroshev_time(n, resid)
        if not np.isfinite(worst):
            worst = 0.0
        return _KamTorusCertificate(
            frequency_vector=omega,
            diophantine_gamma=float(min(gamma_est, 1e12)) if np.isfinite(
                gamma_est) else 0.0,
            diophantine_tau=float(tau_est),
            birkhoff_normal_residual=resid,
            kam_stable=bool(kam_stable),
            kam_iterations=int(iter_count),
            nekhoroshev_time=float(t_nek),
            worst_small_divisor=float(worst),
        )


# ── III.4  Espectro de Lyapunov (Benettin–QR) ────────────────────────────────
class _LyapunovSpectrumAnalyzer:
    r"""
    Fase III. Espectro completo de Lyapunov por iteración QR continua
    (algoritmo de Benettin–Galgani–Giorgilli–Strelcyn).

    Dada la matriz de monodromía M ∈ Sp(2n), se itera

        Z_k = M · Q_{k−1} ,   Q_k R_k = QR(Z_k) ,

    acumulando λ_i^{disc} = (1/N) Σ_k log |R_k[i,i]|. Si se suministra
    el periodo T, se reportan exponentes *por unidad de tiempo*
    λ_i = λ_i^{disc} / T.

    Emparejamiento hamiltoniano (Liouville ⇒ Σ λ_i = 0 y λᵢ = −λ_{2n+1−i}):

        λᵢ ← ½ (λᵢ − λ_{2n+1−i}) .
    """

    def __init__(self, n_iterations: int = _LYAPUNOV_QR_ITERATIONS) -> None:
        self._N = int(n_iterations)

    @staticmethod
    def _kaplan_yorke(spectrum: np.ndarray) -> float:
        n = int(spectrum.size)
        if n == 0:
            return 0.0
        if spectrum[0] < 0.0:
            return 0.0
        cumsum = np.cumsum(spectrum)
        j = 0
        for i in range(n):
            if cumsum[i] >= 0.0:
                j = i
            else:
                break
        if j + 1 < n and abs(spectrum[j + 1]) > _MACHINE_EPS:
            return float(j + 1 + cumsum[j] / abs(spectrum[j + 1]))
        return float(j + 1)

    def compute(
        self,
        M: np.ndarray,
        n_iterations: Optional[int] = None,
        orbit_period_T: Optional[float] = None,
    ) -> _LyapunovSpectrumResult:
        a = np.asarray(M, dtype=np.float64)
        _NumericalCore.assert_square("M", a)
        _NumericalCore.assert_finite("M", a)
        n = a.shape[0]
        N = int(self._N if n_iterations is None else n_iterations)
        if N <= 0:
            raise ValueError("n_iterations debe ser positivo.")
        Q = np.eye(n, dtype=np.float64)
        log_acc = np.zeros(n, dtype=np.float64)
        steps = 0
        for _ in range(N):
            Z = a @ Q
            try:
                Q, R = la.qr(Z, mode="economic")
            except la.LinAlgError:
                break
            diag_R = np.abs(np.diag(R))
            diag_R = np.where(diag_R > _MACHINE_EPS, diag_R, _MACHINE_EPS)
            log_acc += np.log(diag_R)
            steps += 1
        denom = float(max(steps, 1))
        spectrum = np.sort(log_acc / denom)[::-1]
        # Emparejamiento hamiltoniano λᵢ ↔ −λ_{n+1−i}.
        if n % 2 == 0:
            spectrum = 0.5 * (spectrum - spectrum[::-1])
            spectrum = np.sort(spectrum)[::-1]
            spectrum -= float(np.mean(spectrum))     # traza nula (Liouville)
        if orbit_period_T is not None:
            T = float(orbit_period_T)
            if T > 0.0 and np.isfinite(T):
                spectrum = spectrum / T
        d_ky = self._kaplan_yorke(spectrum)
        h_ks = float(np.sum(np.clip(spectrum, 0.0, None)))
        return _LyapunovSpectrumResult(
            spectrum=spectrum,
            kaplan_yorke_dimension=d_ky,
            is_chaotic=bool(spectrum.size > 0 and spectrum[0] > _SPECTRAL_TOL),
            kolmogorov_sinai_entropy=h_ks,
        )


# ── III.5  Analizador de monodromía de Poincaré ──────────────────────────────
class _PoincareMonodromyAnalyzer:
    r"""
    Fase III. Analizador de monodromía de Poincaré–Floquet.

    Se instancia desde un `_PoincareFloquetGerm` (terminal de Fase II) y
    produce el `PoincareMonodromyCertificate` que integra residual
    simpléctico, deriva de Liouville, KAM, Lyapunov, Krein, Nekhoroshev,
    asociador m₃ y obstrucción Čech.
    """

    def __init__(
        self,
        floquet_germ: Optional[_PoincareFloquetGerm] = None,
        kam_detector: Optional[_KamTorusDetector] = None,
        lyapunov_analyzer: Optional[_LyapunovSpectrumAnalyzer] = None,
    ) -> None:
        self._germ = floquet_germ
        self._kam = kam_detector or _KamTorusDetector()
        self._lyap = lyapunov_analyzer or _LyapunovSpectrumAnalyzer()

    def compute_certificate(
        self,
        jacobian_M: np.ndarray,
        orbit_period_T: float,
        canonical_omega: np.ndarray,
        frequency_vector: Optional[np.ndarray] = None,
        birkhoff_residual: Optional[float] = None,
        m2_tensor: Optional[np.ndarray] = None,
        cech_matrix: Optional[np.ndarray] = None,
    ) -> PoincareMonodromyCertificate:
        """
        **III.6 — Morfismo terminal de Fase III.**

        Consume el objeto terminal de Fase II (o lo induce si falta) y
        sintetiza el certificado completo de monodromía Poincaré–Floquet.
        """
        M = np.asarray(jacobian_M, dtype=np.float64)
        _NumericalCore.assert_square("jacobian_M", M)
        _NumericalCore.assert_finite("jacobian_M", M)
        dim = M.shape[0]
        if dim % 2 != 0:
            raise SymplecticDimensionError(
                "[TESSERARIOS_ENGINE_VETO] Dimensión no par para Sp(2n,ℝ).")
        omega = np.asarray(canonical_omega, dtype=np.float64)
        _NumericalCore.assert_square("canonical_omega", omega, dim=dim)
        if self._germ is None or self._germ.darboux_germ.two_n != dim:
            projector = _SymplecticProjector(
                germ=_NumericalCore.synthesize_poincare_darboux_germ(
                    dim, scale_matrix=M,
                ))
            self._germ = projector.induce_poincare_floquet_germ(
                M, orbit_period_T,
            )
        floquet_germ = self._germ
        floquet = floquet_germ.floquet

        symp_def = M.T @ omega @ M - omega
        symp_norm = _NumericalCore.frobenius_norm(symp_def)
        denom = (_NumericalCore.frobenius_norm(omega)
                 + _NumericalCore.frobenius_norm(M) ** 2 * _MACHINE_EPS
                 + _MACHINE_EPS)
        rel_symp = float(symp_norm / denom)

        try:
            det_M = float(np.real(la.det(M)))
        except (la.LinAlgError, ValueError):
            det_M = float("nan")
        volume_drift = float(abs(det_M - 1.0)) if np.isfinite(det_M) else float("inf")

        mu = floquet.floquet_multipliers
        max_mu = float(np.max(np.abs(mu))) if mu.size else 0.0

        lyap = self._lyap.compute(M, orbit_period_T=orbit_period_T)
        lyap_max = float(lyap.spectrum[0]) if lyap.spectrum.size else 0.0

        if frequency_vector is None:
            eig_A = la.eigvals(floquet.floquet_generator)
            omega_freq = np.sort(np.abs(np.imag(eig_A)))[::-1][:dim // 2]
            if omega_freq.size == 0 or np.all(omega_freq < 1e-12):
                omega_freq = np.ones(dim // 2, dtype=np.float64)
        else:
            omega_freq = np.asarray(frequency_vector, dtype=np.float64).ravel()
        kam = self._kam.certify(omega_freq, birkhoff_residual=birkhoff_residual)

        stasheff_res: Optional[_StasheffAssociatorResult] = None
        if m2_tensor is not None:
            stasheff_res = _StasheffAssociator().compute_certified(m2_tensor)

        cech_res: Optional[_CechObstructionResult] = None
        if cech_matrix is not None:
            cech_res = _CechObstructionCalculator().compute_obstruction(cech_matrix)

        krein_def = bool(
            floquet.krein.krein_definite if floquet.krein is not None else False)
        is_stable = bool(
            rel_symp <= _WILKINSON_LIMIT
            and volume_drift <= _WILKINSON_LIMIT
            and max_mu <= 1.0 + _SPECTRAL_TOL
            and not lyap.is_chaotic
        )
        pm_mu = floquet_germ.poincare_map_multipliers
        return PoincareMonodromyCertificate(
            darboux_germ=floquet_germ.darboux_germ,
            floquet_germ=floquet_germ,
            kam=kam,
            lyapunov=lyap,
            stasheff=stasheff_res,
            cech=cech_res,
            relative_symplectic_residual=rel_symp,
            volume_drift=volume_drift,
            max_floquet_multiplier=max_mu,
            lyapunov_exponent=lyap_max,
            is_monodromy_stable=is_stable,
            krein_definite=krein_def,
            nekhoroshev_time=float(kam.nekhoroshev_time),
            poincare_map_multipliers=pm_mu,
        )


# =============================================================================
# MOTOR PRINCIPAL — INTEGRACIÓN Φ_III ∘ Φ_II ∘ Φ_I
# =============================================================================
class ImperialTesserariosEngine:
    """
    Motor de álgebra homológica no abeliana y geometría categorial con
    mecánica celeste de Poincaré.

    Compone las tres fases anidadas:

    1. Fase I   — gérmen de Poincaré–Darboux / Cartan / Williamson
                  (`_NumericalCore.synthesize_poincare_darboux_germ`).
    2. Fase II  — polar Sp(2n), Quillen, Floquet–Poincaré, Mel'nikov, Krein
                  (`_SymplecticProjector.induce_poincare_floquet_germ`).
    3. Fase III — Stasheff / Čech / KAM / Nekhoroshev / Lyapunov / Monodromía
                  (`_PoincareMonodromyAnalyzer.compute_certificate`).

    La API pública de 2.0/3.0/4.0 se conserva. Los métodos `*_certified`
    exponen los invariantes añadidos en 4.1.
    """

    def __init__(self, regularizer: float = 1e-15) -> None:
        self._reg: Final[float] = max(float(regularizer), _REG_FLOOR_TIKHONOV)
        self._poincare_germ: _PoincareDarbouxGerm = (
            _NumericalCore.synthesize_poincare_darboux_germ(
                2, regularizer=self._reg))
        self._projector = _SymplecticProjector(germ=self._poincare_germ)
        self._cech_germ: _AInfinityCechGerm = self._projector.induce_ainfty_cech_germ(
            np.eye(2, dtype=np.float64))
        self._associator = _StasheffAssociator(germ=self._cech_germ)
        self._cech_calculator = _CechObstructionCalculator(
            germ=self._cech_germ, regularizer=self._reg)
        self._kam_detector = _KamTorusDetector()
        self._lyapunov_analyzer = _LyapunovSpectrumAnalyzer()
        self._poincare_analyzer = _PoincareMonodromyAnalyzer(
            kam_detector=self._kam_detector,
            lyapunov_analyzer=self._lyapunov_analyzer,
        )
        self._floquet_germ: Optional[_PoincareFloquetGerm] = None

    # ── Fase I expuesta ──────────────────────────────────────────────────
    def kahan_sum(self, arr):
        return _NumericalCore.kahan_sum(arr)

    def kahan_babuska_neumaier_sum(self, arr):
        return _NumericalCore.kahan_babuska_neumaier_sum(arr)

    def generate_canonical_symplectic_form(self, dim: int) -> np.ndarray:
        omega = _NumericalCore.generate_canonical_symplectic_form(dim)
        if dim != self._poincare_germ.two_n:
            self._resync_poincare_germ(dim)
        return omega

    def symplectic_quillen_germ_certificate(self) -> _SymplecticFormCertificate:
        return self._poincare_germ.form_certificate

    def symplectic_gram_schmidt(
        self, vectors: np.ndarray, omega: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        O = self._poincare_germ.omega if omega is None else np.asarray(omega)
        return _NumericalCore.symplectic_gram_schmidt(vectors, O)

    def williamson_classify(
        self, hessian: np.ndarray, omega: Optional[np.ndarray] = None,
    ) -> _WilliamsonSpectrum:
        O = self._poincare_germ.omega if omega is None else np.asarray(omega)
        return _NumericalCore.williamson_classify(hessian, O)

    def poincare_cartan_action(
        self,
        q_traj: np.ndarray,
        p_traj: np.ndarray,
        dt: float,
        energy_level: Optional[float] = None,
    ) -> Tuple[float, float]:
        e = (self._poincare_germ.energy_level
             if energy_level is None else float(energy_level))
        return _NumericalCore.poincare_cartan_action(q_traj, p_traj, dt, e)

    def synthesize_poincare_darboux_germ(
        self,
        dimension_two_n: int,
        section_index: int = 0,
        section_offset: float = 0.0,
        energy_level: float = 0.0,
        max_iter: int = _DEFAULT_MAX_ITER,
        tol: float = _DEFAULT_TOL,
        scale_matrix: Optional[np.ndarray] = None,
        flow_vector: Optional[np.ndarray] = None,
    ) -> _PoincareDarbouxGerm:
        """Réplica pública del morfismo I.9 (arranque de Φ_II)."""
        self._resync_poincare_germ(
            int(dimension_two_n), section_index=section_index,
            section_offset=section_offset, energy_level=energy_level,
            max_iter=max_iter, tol=tol, scale_matrix=scale_matrix,
            flow_vector=flow_vector)
        return self._poincare_germ

    def _resync_poincare_germ(
        self,
        two_n: int,
        section_index: Optional[int] = None,
        section_offset: Optional[float] = None,
        energy_level: Optional[float] = None,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
        scale_matrix: Optional[np.ndarray] = None,
        flow_vector: Optional[np.ndarray] = None,
    ) -> None:
        same_dim = two_n == self._poincare_germ.two_n
        same_it = max_iter is None or int(max_iter) == self._poincare_germ.max_iter
        same_tol = tol is None or float(tol) == self._poincare_germ.tol
        same_sec = (section_index is None
                    or int(section_index) == self._poincare_germ.section_index)
        same_flow = flow_vector is None
        if (same_dim and same_it and same_tol and same_sec
                and same_flow and scale_matrix is None):
            return
        self._poincare_germ = _NumericalCore.synthesize_poincare_darboux_germ(
            two_n,
            section_index=(self._poincare_germ.section_index
                           if section_index is None else int(section_index)),
            section_offset=(self._poincare_germ.section_offset
                            if section_offset is None else float(section_offset)),
            energy_level=(self._poincare_germ.energy_level
                          if energy_level is None else float(energy_level)),
            regularizer=self._reg,
            max_iter=(self._poincare_germ.max_iter
                      if max_iter is None else int(max_iter)),
            tol=(self._poincare_germ.tol if tol is None else float(tol)),
            scale_matrix=scale_matrix,
            flow_vector=flow_vector,
        )
        self._projector = _SymplecticProjector(germ=self._poincare_germ)

    # ── Fase II expuesta ─────────────────────────────────────────────────
    def project_to_symplectic_group(
        self, M: np.ndarray, max_iter: int = 100, tol: float = 1e-12,
    ) -> Tuple[np.ndarray, float]:
        r = self.project_to_symplectic_group_certified(M, max_iter=max_iter, tol=tol)
        return r.symplectic_matrix, r.residual

    def project_to_symplectic_group_certified(
        self, M: np.ndarray, max_iter: int = 100, tol: float = 1e-12,
    ) -> _SymplecticProjectionResult:
        a = np.asarray(M)
        _NumericalCore.assert_square("M", a)
        if a.shape[0] % 2 == 0:
            self._resync_poincare_germ(
                a.shape[0], max_iter=max_iter, tol=tol, scale_matrix=a)
        return self._projector.project(a, max_iter=max_iter, tol=tol)

    def compute_quillen_factorization(
        self, M: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        r = self.compute_quillen_factorization_certified(M)
        return r.fibration, r.cofibration, r.total_residual

    def compute_quillen_factorization_certified(
        self, M: np.ndarray,
    ) -> _QuillenFactorizationResult:
        a = np.asarray(M)
        _NumericalCore.assert_square("M", a)
        if a.shape[0] % 2 == 0:
            self._resync_poincare_germ(a.shape[0], scale_matrix=a)
        return self._projector.factorize_quillen(a)

    def compute_floquet_factorization_certified(
        self, M: np.ndarray, orbit_period_T: float,
    ) -> _FloquetFactorizationResult:
        a = np.asarray(M)
        _NumericalCore.assert_square("M", a)
        if a.shape[0] % 2 == 0:
            self._resync_poincare_germ(a.shape[0], scale_matrix=a)
        return self._projector.floquet_factorization(a, orbit_period_T)

    def classify_floquet_multipliers(
        self, mu: np.ndarray, omega: Optional[np.ndarray] = None,
        right_eigenvectors: Optional[np.ndarray] = None,
    ) -> _KreinClassification:
        O = None if omega is None else np.asarray(omega)
        return _SymplecticProjector.classify_floquet_multipliers(
            mu, omega=O, right_eigenvectors=right_eigenvectors)

    def induce_ainfty_cech_germ(self, payload, m2_tensor=None):
        raw = np.asarray(payload if m2_tensor is None else m2_tensor)
        if raw.ndim == 2 and raw.shape[0] % 2 == 0:
            self._resync_poincare_germ(raw.shape[0], scale_matrix=raw)
        germ = self._projector.induce_ainfty_cech_germ(payload, m2_tensor=m2_tensor)
        self._cech_germ = germ
        self._associator = _StasheffAssociator(germ=germ)
        self._cech_calculator = _CechObstructionCalculator(
            germ=germ, regularizer=self._reg)
        return germ

    def induce_poincare_floquet_germ(
        self,
        jacobian_M: np.ndarray,
        orbit_period_T: float,
        q0_trajectory: Optional[np.ndarray] = None,
        dt: float = 1.0,
        h0_grad: Optional[np.ndarray] = None,
        h1_grad: Optional[np.ndarray] = None,
        orbit_points_for_rotation: Optional[np.ndarray] = None,
    ) -> _PoincareFloquetGerm:
        """Réplica pública del morfismo II.8 (arranque de Φ_III)."""
        M = np.asarray(jacobian_M)
        _NumericalCore.assert_square("jacobian_M", M)
        if M.shape[0] % 2 == 0:
            self._resync_poincare_germ(M.shape[0], scale_matrix=M)
        self._floquet_germ = self._projector.induce_poincare_floquet_germ(
            M, orbit_period_T,
            q0_trajectory=q0_trajectory, dt=dt,
            h0_grad=h0_grad, h1_grad=h1_grad,
            orbit_points_for_rotation=orbit_points_for_rotation,
        )
        self._poincare_analyzer = _PoincareMonodromyAnalyzer(
            floquet_germ=self._floquet_germ,
            kam_detector=self._kam_detector,
            lyapunov_analyzer=self._lyapunov_analyzer,
        )
        return self._floquet_germ

    # ── Fase III expuesta ────────────────────────────────────────────────
    def compute_stasheff_m3_associator(self, m2_tensor: np.ndarray) -> np.ndarray:
        return self._associator.compute_m3_associator(m2_tensor)

    def compute_stasheff_m3_associator_certified(
        self, m2_tensor: np.ndarray,
    ) -> _StasheffAssociatorResult:
        result = self._associator.compute_certified(m2_tensor)
        self.induce_ainfty_cech_germ(m2_tensor, m2_tensor=m2_tensor)
        return result

    def compute_cech_hypercohomology_gerbe(
        self, cech_cochain_matrix: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        result = self._cech_calculator.compute_obstruction(cech_cochain_matrix)
        return result.obstruction_value, result.singular_values

    def compute_cech_hypercohomology_gerbe_certified(
        self, cech_cochain_matrix: np.ndarray,
    ) -> _CechObstructionResult:
        return self._cech_calculator.compute_obstruction(cech_cochain_matrix)

    def certify_kam_torus(
        self, frequency_vector: np.ndarray,
        birkhoff_residual: Optional[float] = None,
    ) -> _KamTorusCertificate:
        return self._kam_detector.certify(frequency_vector, birkhoff_residual)

    def nekhoroshev_stability_time(
        self, n_dof: int, perturbation_eps: float,
    ) -> float:
        return _KamTorusDetector.nekhoroshev_time(n_dof, perturbation_eps)

    def compute_lyapunov_spectrum(
        self,
        M: np.ndarray,
        n_iterations: Optional[int] = None,
        orbit_period_T: Optional[float] = None,
    ) -> _LyapunovSpectrumResult:
        return self._lyapunov_analyzer.compute(
            M, n_iterations=n_iterations, orbit_period_T=orbit_period_T)

    def moser_twist_certificate(
        self,
        rotation_samples: Sequence[_RotationNumberResult],
        action_samples: Optional[np.ndarray] = None,
    ) -> _MoserTwistResult:
        return self._projector.moser_twist(rotation_samples, action_samples)

    # ── Método integrador de las tres fases ──────────────────────────────
    def compute_poincare_monodromy_certificate(
        self,
        jacobian_M: np.ndarray,
        orbit_period_T: float,
        canonical_omega: Optional[np.ndarray] = None,
        frequency_vector: Optional[np.ndarray] = None,
        birkhoff_residual: Optional[float] = None,
        m2_tensor: Optional[np.ndarray] = None,
        cech_matrix: Optional[np.ndarray] = None,
    ) -> PoincareMonodromyCertificate:
        """
        **Morfismo terminal global** Φ_III ∘ Φ_II ∘ Φ_I.

        Ensambla el certificado completo de monodromía Poincaré–Floquet
        integrando las tres fases anidadas.
        """
        M = np.asarray(jacobian_M, dtype=np.float64)
        _NumericalCore.assert_square("jacobian_M", M)
        dim = M.shape[0]
        if dim % 2 != 0:
            raise SymplecticDimensionError(
                "[TESSERARIOS_ENGINE_VETO] Dimensión no par para Sp(2n,ℝ).")
        if canonical_omega is None:
            canonical_omega = self._poincare_germ.omega
            if canonical_omega.shape[0] != dim:
                self._resync_poincare_germ(dim, scale_matrix=M)
                canonical_omega = self._poincare_germ.omega
        if (self._floquet_germ is None
                or self._floquet_germ.darboux_germ.two_n != dim):
            self.induce_poincare_floquet_germ(M, orbit_period_T)
        return self._poincare_analyzer.compute_certificate(
            M, orbit_period_T, canonical_omega,
            frequency_vector=frequency_vector,
            birkhoff_residual=birkhoff_residual,
            m2_tensor=m2_tensor,
            cech_matrix=cech_matrix,
        )

    # ── Compatibilidad 3.0: PoincareMonodromyGerm ────────────────────────
    def compute_poincare_symplectic_monodromy_germ(
        self,
        jacobian_M: np.ndarray,
        orbit_period_T: float,
        canonical_omega: np.ndarray,
    ) -> PoincareMonodromyGerm:
        r"""
        Wrapper 3.0 — Retorna un `PoincareMonodromyGerm` con los
        invariantes clásicos. Para el certificado completo usar
        `compute_poincare_monodromy_certificate`.
        """
        cert = self.compute_poincare_monodromy_certificate(
            jacobian_M, orbit_period_T, canonical_omega)
        return PoincareMonodromyGerm(
            relative_symplectic_residual=cert.relative_symplectic_residual,
            volume_drift=cert.volume_drift,
            max_floquet_multiplier=cert.max_floquet_multiplier,
            lyapunov_exponent=cert.lyapunov_exponent,
            is_monodromy_stable=cert.is_monodromy_stable,
        )


__all__ = [
    "ImperialTesserariosEngine",
    "PoincareMonodromyGerm",
    "PoincareMonodromyCertificate",
    "SymplecticDimensionError",
]